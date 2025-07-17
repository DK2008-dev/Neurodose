#!/usr/bin/env python3
"""
Training script for EEG pain classification CNN.

This script trains a deep learning model on preprocessed EEG data
for ternary pain classification.
"""

import argparse
import logging
from pathlib import Path
import numpy as np
import yaml
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score

import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.models.cnn import create_model, ModelTrainer
from src.utils.helpers import (
    setup_logging, load_data, save_data, split_data, 
    create_data_loader, evaluate_model, plot_confusion_matrix,
    plot_training_history, get_class_weights, set_seed, print_model_summary
)


class EarlyStopping:
    """Early stopping utility."""
    
    def __init__(self, patience=7, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False
    
    def save_checkpoint(self, model):
        self.best_weights = model.state_dict().copy()


def train_model(model, train_loader, val_loader, config, device):
    """Train the model with the given configuration."""
    logger = logging.getLogger(__name__)
    
    # Training configuration
    train_config = config['training']
    
    # Initialize trainer
    trainer = ModelTrainer(
        model=model,
        device=device,
        learning_rate=train_config['learning_rate'],
        weight_decay=train_config['weight_decay']
    )
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=train_config['early_stopping']['patience'],
        min_delta=train_config['early_stopping']['min_delta'],
        restore_best_weights=train_config['early_stopping']['restore_best_weights']
    )
    
    # Learning rate scheduler
    scheduler = None
    if train_config['scheduler']['type'] == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            trainer.optimizer,
            step_size=train_config['scheduler']['step_size'],
            gamma=train_config['scheduler']['gamma']
        )
    elif train_config['scheduler']['type'] == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            trainer.optimizer,
            T_max=train_config['epochs']
        )
    elif train_config['scheduler']['type'] == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            trainer.optimizer,
            mode='min',
            patience=10,
            factor=0.5
        )
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    
    # Training loop
    logger.info(f"Starting training for {train_config['epochs']} epochs...")
    
    for epoch in range(train_config['epochs']):
        # Train epoch
        train_loss = trainer.train_epoch(train_loader)
        
        # Validation
        val_loss, val_acc = trainer.evaluate(val_loader)
        
        # Calculate training accuracy
        trainer.model.eval()
        train_correct = 0
        train_total = 0
        with torch.no_grad():
            for data, target in train_loader:
                data, target = data.to(device), target.to(device)
                output = trainer.model(data)
                pred = output.argmax(dim=1, keepdim=True)
                train_correct += pred.eq(target.view_as(pred)).sum().item()
                train_total += target.size(0)
        
        train_acc = 100. * train_correct / train_total
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        # Update learning rate
        if scheduler:
            if train_config['scheduler']['type'] == 'plateau':
                scheduler.step(val_loss)
            else:
                scheduler.step()
        
        # Logging
        current_lr = trainer.optimizer.param_groups[0]['lr']
        logger.info(f'Epoch {epoch+1}/{train_config["epochs"]} - '
                   f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - '
                   f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}% - '
                   f'LR: {current_lr:.6f}')
        
        # Early stopping
        if early_stopping(val_loss, trainer.model):
            logger.info(f'Early stopping at epoch {epoch+1}')
            break
    
    return history


def main():
    parser = argparse.ArgumentParser(description='Train EEG pain classification model')
    parser.add_argument('--config', type=str, default='config/cnn_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--data_file', type=str, default='data/processed/pain_epochs.npz',
                       help='Path to preprocessed data file')
    parser.add_argument('--output_dir', type=str, default='models',
                       help='Output directory for trained models')
    parser.add_argument('--experiment_name', type=str, default='pain_cnn',
                       help='Experiment name for output files')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug logging')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup logging
    log_level = 'DEBUG' if args.debug else config['logging']['level']
    setup_logging(level=log_level)
    logger = logging.getLogger(__name__)
    
    # Set random seed
    set_seed(config['training']['random_state'])
    
    logger.info("Starting EEG pain classification training...")
    
    # Load preprocessed data
    logger.info(f"Loading data from {args.data_file}")
    data = load_data(args.data_file)
    X, y = data['X'], data['y']
    
    logger.info(f"Loaded data shape: {X.shape}")
    logger.info(f"Label distribution: {np.bincount(y)}")
    
    # Split data
    train_config = config['training']
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        X, y,
        test_size=train_config['test_size'],
        val_size=train_config['val_size'],
        random_state=train_config['random_state'],
        stratify=train_config['stratify']
    )
    
    # Create data loaders
    train_loader = create_data_loader(
        X_train, y_train,
        batch_size=train_config['batch_size'],
        shuffle=True
    )
    val_loader = create_data_loader(
        X_val, y_val,
        batch_size=train_config['batch_size'],
        shuffle=False
    )
    test_loader = create_data_loader(
        X_test, y_test,
        batch_size=train_config['batch_size'],
        shuffle=False
    )
    
    # Create model
    model_config = config['model']
    model_type = model_config['type']
    
    model_params = {
        'n_channels': X.shape[1],
        'n_samples': X.shape[2],
        'n_classes': model_config['n_classes']
    }
    model_params.update(model_config.get(model_type, {}))
    
    model = create_model(model_type, **model_params)
    
    # Print model summary
    print_model_summary(model, (X.shape[1], X.shape[2]))
    
    # Setup device
    device = train_config['device']
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    logger.info(f"Using device: {device}")
    model = model.to(device)
    
    # Train model
    history = train_model(model, train_loader, val_loader, config, device)
    
    # Evaluate on test set
    logger.info("Evaluating on test set...")
    trainer = ModelTrainer(model, device)
    test_loss, test_acc = trainer.evaluate(test_loader)
    
    # Get predictions for detailed evaluation
    model.eval()
    y_pred = []
    y_true = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            y_pred.extend(pred.cpu().numpy())
            y_true.extend(target.cpu().numpy())
    
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    
    # Detailed evaluation
    eval_results = evaluate_model(y_true, y_pred, config['evaluation']['class_names'])
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model_path = output_dir / f"{args.experiment_name}_model.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': model_params,
        'model_type': model_type,
        'config': config,
        'test_accuracy': test_acc,
        'history': history
    }, model_path)
    
    # Save results
    results = {
        'test_accuracy': test_acc,
        'test_loss': test_loss,
        'evaluation': eval_results,
        'history': history,
        'config': config
    }
    
    results_path = output_dir / f"{args.experiment_name}_results.json"
    save_data(results, results_path)
    
    # Save data splits for reproducibility
    splits_path = output_dir / f"{args.experiment_name}_splits.npz"
    save_data({
        'X_test': X_test,
        'y_test': y_test,
        'y_pred': y_pred
    }, splits_path)
    
    # Create visualizations
    figures_dir = Path('figures')
    figures_dir.mkdir(exist_ok=True)
    
    # Plot confusion matrix
    if config['evaluation']['plots']['confusion_matrix']:
        cm_path = figures_dir / f"{args.experiment_name}_confusion_matrix.png"
        plot_confusion_matrix(
            eval_results['confusion_matrix'],
            eval_results['class_names'],
            title=f'{args.experiment_name} - Confusion Matrix',
            save_path=cm_path
        )
    
    # Plot training history
    if config['evaluation']['plots']['training_history']:
        history_path = figures_dir / f"{args.experiment_name}_training_history.png"
        plot_training_history(history, save_path=history_path)
    
    # Final summary
    logger.info("\n" + "="*60)
    logger.info("TRAINING COMPLETED")
    logger.info("="*60)
    logger.info(f"Model: {model_type}")
    logger.info(f"Test Accuracy: {test_acc:.2f}%")
    logger.info(f"Test Loss: {test_loss:.4f}")
    logger.info(f"Model saved to: {model_path}")
    logger.info(f"Results saved to: {results_path}")
    logger.info("="*60)


if __name__ == "__main__":
    main()
