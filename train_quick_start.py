#!/usr/bin/env python3
"""
Quick start training script for EEG pain classification.

This script trains CNN models on your preprocessed 281-window dataset
with cross-validation across the 5 participants.
"""

import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add src to path
import sys
sys.path.append(str(Path(__file__).parent / 'src'))

from models.cnn import EEGNet, ShallowConvNet, DeepConvNet

class QuickTrainer:
    """Quick trainer for EEG pain classification."""
    
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=0.001)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=10, factor=0.5, verbose=True
        )
        
    def train_epoch(self, train_loader):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(train_loader)
    
    def evaluate(self, val_loader):
        """Evaluate model on validation set."""
        self.model.eval()
        total_loss = 0
        correct = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                total_loss += loss.item()
                
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        accuracy = 100. * correct / len(val_loader.dataset)
        avg_loss = total_loss / len(val_loader)
        
        return avg_loss, accuracy, all_preds, all_targets

def load_preprocessed_data():
    """Load all preprocessed participant data."""
    data_dir = Path('data/processed/basic_windows')
    
    all_windows = []
    all_labels = []
    all_participants = []
    
    participants = ['vp01', 'vp02', 'vp03', 'vp04', 'vp05']
    
    logger.info("Loading preprocessed data...")
    
    for participant in participants:
        file_path = data_dir / f'{participant}_windows.pkl'
        
        if not file_path.exists():
            logger.warning(f"File not found: {file_path}")
            continue
            
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        windows = data['windows']  # Shape: (n_windows, n_channels, n_samples)
        labels = data['labels']    # Shape: (n_windows, 3) - one-hot encoded
        
        # Convert one-hot to class indices
        label_indices = np.argmax(labels, axis=1)
        
        all_windows.append(windows)
        all_labels.append(label_indices)
        all_participants.extend([participant] * len(windows))
        
        logger.info(f"{participant}: {len(windows)} windows, labels: {np.bincount(label_indices)}")
    
    # Concatenate all data
    X = np.concatenate(all_windows, axis=0)
    y = np.concatenate(all_labels, axis=0)
    participants = np.array(all_participants)
    
    logger.info(f"Total dataset: {X.shape[0]} windows, {X.shape[1]} channels, {X.shape[2]} samples")
    logger.info(f"Label distribution: {np.bincount(y)}")
    
    return X, y, participants

def create_model(model_type='eegnet', n_channels=68, n_samples=2000, n_classes=3):
    """Create model instance."""
    if model_type == 'eegnet':
        return EEGNet(n_channels=n_channels, n_samples=n_samples, n_classes=n_classes)
    elif model_type == 'shallow':
        return ShallowConvNet(n_channels=n_channels, n_samples=n_samples, n_classes=n_classes)
    elif model_type == 'deep':
        return DeepConvNet(n_channels=n_channels, n_samples=n_samples, n_classes=n_classes)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def cross_validate_model(X, y, participants, model_type='eegnet', epochs=50, batch_size=32):
    """Perform leave-one-participant-out cross-validation."""
    
    logger.info(f"Starting cross-validation with {model_type}")
    
    # Setup cross-validation
    logo = LeaveOneGroupOut()
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(logo.split(X, y, participants)):
        val_participant = participants[val_idx][0]  # Get participant name
        logger.info(f"Fold {fold + 1}: Training without {val_participant}")
        
        # Split data
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Normalize data (fit on train, transform both)
        scaler = StandardScaler()
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_val_flat = X_val.reshape(X_val.shape[0], -1)
        
        X_train_norm = scaler.fit_transform(X_train_flat).reshape(X_train.shape)
        X_val_norm = scaler.transform(X_val_flat).reshape(X_val.shape)
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train_norm)
        y_train_tensor = torch.LongTensor(y_train)
        X_val_tensor = torch.FloatTensor(X_val_norm)
        y_val_tensor = torch.LongTensor(y_val)
        
        # Create data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Create and train model
        model = create_model(model_type, n_channels=X.shape[1], n_samples=X.shape[2])
        trainer = QuickTrainer(model)
        
        best_val_acc = 0
        patience_counter = 0
        patience = 15
        
        for epoch in range(epochs):
            # Train
            train_loss = trainer.train_epoch(train_loader)
            
            # Validate
            val_loss, val_acc, val_preds, val_targets = trainer.evaluate(val_loader)
            
            # Learning rate scheduling
            trainer.scheduler.step(val_loss)
            
            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                # Save best model state
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
        
        # Load best model and get final results
        model.load_state_dict(best_model_state)
        final_loss, final_acc, final_preds, final_targets = trainer.evaluate(val_loader)
        
        # Store results
        fold_results.append({
            'participant': val_participant,
            'accuracy': final_acc,
            'predictions': final_preds,
            'targets': final_targets,
            'classification_report': classification_report(final_targets, final_preds, output_dict=True)
        })
        
        logger.info(f"Fold {fold + 1} ({val_participant}): Final Accuracy = {final_acc:.2f}%")
    
    return fold_results

def analyze_results(fold_results):
    """Analyze cross-validation results."""
    
    # Calculate overall metrics
    accuracies = [fold['accuracy'] for fold in fold_results]
    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    
    logger.info("\n" + "="*50)
    logger.info("CROSS-VALIDATION RESULTS")
    logger.info("="*50)
    
    for i, fold in enumerate(fold_results):
        logger.info(f"Fold {i+1} ({fold['participant']}): {fold['accuracy']:.2f}%")
    
    logger.info(f"\nOverall Accuracy: {mean_acc:.2f} ± {std_acc:.2f}%")
    
    # Compare with literature benchmark
    target_acc = 87.94  # MDPI Biology 2025 target
    logger.info(f"Literature Target: {target_acc:.2f}%")
    
    if mean_acc >= target_acc:
        logger.info("🎉 TARGET ACHIEVED! Accuracy exceeds literature benchmark!")
    else:
        gap = target_acc - mean_acc
        logger.info(f"Gap to target: {gap:.2f}% - Room for improvement")
    
    # Aggregate confusion matrix
    all_targets = []
    all_preds = []
    
    for fold in fold_results:
        all_targets.extend(fold['targets'])
        all_preds.extend(fold['predictions'])
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(all_targets, all_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Low', 'Moderate', 'High'],
                yticklabels=['Low', 'Moderate', 'High'])
    plt.title('Confusion Matrix - Cross-Validation Results')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix_cv.png', dpi=300)
    plt.show()
    
    # Classification report
    print("\nOverall Classification Report:")
    print(classification_report(all_targets, all_preds, 
                              target_names=['Low', 'Moderate', 'High']))
    
    return mean_acc, std_acc

def main():
    """Main training function."""
    
    logger.info("EEG Pain Classification - Quick Start Training")
    logger.info("="*50)
    
    # Load data
    X, y, participants = load_preprocessed_data()
    
    # Test different models
    models_to_test = ['eegnet', 'shallow', 'deep']
    results = {}
    
    for model_type in models_to_test:
        logger.info(f"\n🚀 Training {model_type.upper()} model...")
        
        try:
            fold_results = cross_validate_model(X, y, participants, model_type=model_type, epochs=100)
            mean_acc, std_acc = analyze_results(fold_results)
            results[model_type] = {'mean_acc': mean_acc, 'std_acc': std_acc}
            
        except Exception as e:
            logger.error(f"Error training {model_type}: {e}")
            continue
    
    # Compare models
    logger.info("\n" + "="*50)
    logger.info("MODEL COMPARISON")
    logger.info("="*50)
    
    for model_type, result in results.items():
        logger.info(f"{model_type.upper()}: {result['mean_acc']:.2f} ± {result['std_acc']:.2f}%")
    
    # Find best model
    if results:
        best_model = max(results, key=lambda x: results[x]['mean_acc'])
        best_acc = results[best_model]['mean_acc']
        logger.info(f"\n🏆 Best Model: {best_model.upper()} ({best_acc:.2f}%)")
        
        # Save results
        with open('training_results.pkl', 'wb') as f:
            pickle.dump(results, f)
        
        logger.info("Results saved to 'training_results.pkl'")

if __name__ == '__main__':
    main()
