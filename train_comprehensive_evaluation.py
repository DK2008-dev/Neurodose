#!/usr/bin/env python3
"""
Enhanced training script with comprehensive evaluation metrics for EEG pain classification.

This script implements:
- Leave-One-Participant-Out Cross-Validation (LOPOCV)
- Comprehensive metrics: Accuracy, F1 (macro/micro/weighted), AUC-ROC, Precision, Recall
- Proper train/validation/test splits within each fold
- Statistical significance testing
- Detailed per-class and overall performance analysis
"""

import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import LeaveOneGroupOut, train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
    roc_curve, auc, precision_recall_curve, average_precision_score
)
from sklearn.preprocessing import StandardScaler, label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from scipy import stats
import pandas as pd
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add src to path
import sys
sys.path.append(str(Path(__file__).parent / 'src'))

from models.cnn import EEGNet, ShallowConvNet, DeepConvNet

class ComprehensiveTrainer:
    """Enhanced trainer with comprehensive evaluation metrics."""
    
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=10, factor=0.5
        )
        
    def train_epoch(self, train_loader):
        """Train for one epoch with progress bar."""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        # Create progress bar for training
        pbar = tqdm(train_loader, desc='Training', leave=False)
        
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Calculate batch accuracy
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            # Update progress bar with loss and accuracy
            current_acc = 100. * correct / total
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{current_acc:.2f}%'
            })
            
        return total_loss / len(train_loader), 100. * correct / total
    
    def evaluate(self, val_loader, return_probabilities=True, desc="Evaluating"):
        """Comprehensive evaluation with multiple metrics and progress bar."""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_targets = []
        all_probs = []
        
        # Create progress bar for evaluation
        pbar = tqdm(val_loader, desc=desc, leave=False)
        
        with torch.no_grad():
            for data, target in pbar:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                total_loss += loss.item()
                
                # Get predictions and probabilities
                probs = torch.softmax(output, dim=1)
                pred = output.argmax(dim=1)
                
                pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
                
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        # Convert to numpy arrays
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        all_probs = np.array(all_probs)
        
        # Calculate comprehensive metrics
        metrics = self.calculate_metrics(all_targets, all_preds, all_probs)
        
        avg_loss = total_loss / len(val_loader)
        
        if return_probabilities:
            return avg_loss, metrics, all_preds, all_targets, all_probs
        else:
            return avg_loss, metrics, all_preds, all_targets
    
    def calculate_metrics(self, y_true, y_pred, y_probs):
        """Calculate comprehensive evaluation metrics."""
        
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        
        # F1 scores (macro, micro, weighted)
        f1_macro = f1_score(y_true, y_pred, average='macro')
        f1_micro = f1_score(y_true, y_pred, average='micro')
        f1_weighted = f1_score(y_true, y_pred, average='weighted')
        
        # Precision and Recall
        precision_macro = precision_score(y_true, y_pred, average='macro')
        precision_micro = precision_score(y_true, y_pred, average='micro')
        precision_weighted = precision_score(y_true, y_pred, average='weighted')
        
        recall_macro = recall_score(y_true, y_pred, average='macro')
        recall_micro = recall_score(y_true, y_pred, average='micro')
        recall_weighted = recall_score(y_true, y_pred, average='weighted')
        
        # Per-class F1 scores
        f1_per_class = f1_score(y_true, y_pred, average=None)
        
        # AUC-ROC calculation
        n_classes = len(np.unique(y_true))
        if n_classes > 2:
            # Multi-class AUC (one-vs-rest)
            y_true_bin = label_binarize(y_true, classes=range(n_classes))
            auc_macro = roc_auc_score(y_true_bin, y_probs, average='macro', multi_class='ovr')
            auc_weighted = roc_auc_score(y_true_bin, y_probs, average='weighted', multi_class='ovr')
        else:
            # Binary classification
            auc_macro = roc_auc_score(y_true, y_probs[:, 1])
            auc_weighted = auc_macro
        
        metrics = {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_micro': f1_micro,
            'f1_weighted': f1_weighted,
            'f1_per_class': f1_per_class,
            'precision_macro': precision_macro,
            'precision_micro': precision_micro,
            'precision_weighted': precision_weighted,
            'recall_macro': recall_macro,
            'recall_micro': recall_micro,
            'recall_weighted': recall_weighted,
            'auc_macro': auc_macro,
            'auc_weighted': auc_weighted
        }
        
        return metrics

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
        labels = data['ternary_labels']    # Shape: (n_windows,) - class indices
        
        # Labels are already class indices (0, 1, 2), no conversion needed
        label_indices = labels
        
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

def cross_validate_model_comprehensive(X, y, participants, model_type='eegnet', epochs=50, batch_size=32):
    """
    Perform Leave-One-Participant-Out Cross-Validation with comprehensive evaluation.
    
    Strategy:
    - LOPOCV: Use 4 participants for training, 1 for testing
    - Within training set: 80% train, 20% validation for early stopping
    - Final evaluation on held-out test participant
    """
    
    logger.info(f"Starting comprehensive cross-validation with {model_type}")
    logger.info("="*80)
    logger.info("VALIDATION STRATEGY:")
    logger.info("- Leave-One-Participant-Out Cross-Validation (LOPOCV)")
    logger.info("- Training participants: 80% train / 20% validation")
    logger.info("- Test participant: 100% held-out test set")
    logger.info("- Metrics: Accuracy, F1 (macro/micro/weighted), AUC-ROC, Precision, Recall")
    logger.info("="*80)
    
    # Setup cross-validation
    logo = LeaveOneGroupOut()
    fold_results = []
    
    # Create main progress bar for folds
    fold_pbar = tqdm(enumerate(logo.split(X, y, participants)), 
                     total=len(participants), 
                     desc=f"🧠 {model_name.upper()} Cross-Validation", 
                     position=1)
    
    for fold, (train_val_idx, test_idx) in fold_pbar:
        test_participant = participants[test_idx][0]  # Get participant name
        logger.info(f"\nFold {fold + 1}/5: Testing on {test_participant}")
        
        # Split data: test set is one participant, train_val is the rest
        X_train_val, X_test = X[train_val_idx], X[test_idx]
        y_train_val, y_test = y[train_val_idx], y[test_idx]
        participants_train_val = participants[train_val_idx]
        
        # Further split train_val into train and validation (80/20)
        train_idx, val_idx = train_test_split(
            range(len(X_train_val)), 
            test_size=0.2, 
            stratify=y_train_val, 
            random_state=42
        )
        
        X_train, X_val = X_train_val[train_idx], X_train_val[val_idx]
        y_train, y_val = y_train_val[train_idx], y_train_val[val_idx]
        
        logger.info(f"  Train: {len(X_train)} samples, Val: {len(X_val)} samples, Test: {len(X_test)} samples")
        logger.info(f"  Train labels: {np.bincount(y_train)}")
        logger.info(f"  Val labels: {np.bincount(y_val)}")
        logger.info(f"  Test labels: {np.bincount(y_test)}")
        
        # Normalize data (fit on train, transform train/val/test)
        scaler = StandardScaler()
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_val_flat = X_val.reshape(X_val.shape[0], -1)
        X_test_flat = X_test.reshape(X_test.shape[0], -1)
        
        X_train_norm = scaler.fit_transform(X_train_flat).reshape(X_train.shape)
        X_val_norm = scaler.transform(X_val_flat).reshape(X_val.shape)
        X_test_norm = scaler.transform(X_test_flat).reshape(X_test.shape)
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train_norm)
        y_train_tensor = torch.LongTensor(y_train)
        X_val_tensor = torch.FloatTensor(X_val_norm)
        y_val_tensor = torch.LongTensor(y_val)
        X_test_tensor = torch.FloatTensor(X_test_norm)
        y_test_tensor = torch.LongTensor(y_test)
        
        # Create data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        # Create and train model
        model = create_model(model_type, n_channels=X.shape[1], n_samples=X.shape[2])
        trainer = ComprehensiveTrainer(model)
        
        best_val_f1 = 0
        patience_counter = 0
        patience = 15
        
        # Training loop with validation-based early stopping
        epoch_pbar = tqdm(range(epochs), desc=f"Training {model_name}", position=0)
        
        for epoch in epoch_pbar:
            # Train
            train_loss, train_acc = trainer.train_epoch(train_loader)
            
            # Validate
            val_loss, val_metrics, _, _ = trainer.evaluate(val_loader, return_probabilities=False, desc="Validation")
            
            # Learning rate scheduling
            trainer.scheduler.step(val_loss)
            
            # Early stopping based on F1 score
            current_f1 = val_metrics['f1_weighted']
            if current_f1 > best_val_f1:
                best_val_f1 = current_f1
                patience_counter = 0
                # Save best model state
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
            
            # Update epoch progress bar with current metrics
            epoch_pbar.set_postfix({
                'Train_Acc': f'{train_acc:.1f}%',
                'Val_Acc': f'{val_metrics["accuracy"]*100:.1f}%',
                'Val_F1': f'{val_metrics["f1_weighted"]:.3f}',
                'Best_F1': f'{best_val_f1:.3f}',
                'Patience': f'{patience_counter}/{patience}'
            })
            
            if epoch % 10 == 0 or epoch < 5:  # Show more details early and every 10 epochs
                logger.info(f"  Epoch {epoch}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.1f}%, Val Loss: {val_loss:.4f}")
                logger.info(f"    Val Metrics - Acc: {val_metrics['accuracy']*100:.1f}%, F1: {val_metrics['f1_weighted']:.3f}, AUC: {val_metrics['auc_weighted']:.3f}")
            
            if patience_counter >= patience:
                logger.info(f"  Early stopping at epoch {epoch}")
                break
        
        # Load best model and evaluate on test set
        model.load_state_dict(best_model_state)
        test_loss, test_metrics, test_preds, test_targets, test_probs = trainer.evaluate(test_loader, desc=f"Testing {test_participant}")
        
        # Store comprehensive results
        fold_result = {
            'fold': fold + 1,
            'test_participant': test_participant,
            'test_metrics': test_metrics,
            'test_predictions': test_preds,
            'test_targets': test_targets,
            'test_probabilities': test_probs,
            'classification_report': classification_report(test_targets, test_preds, 
                                                         target_names=['Low', 'Moderate', 'High'], 
                                                         output_dict=True)
        }
        
        fold_results.append(fold_result)
        
        # Log fold results
        logger.info(f"  Fold {fold + 1} Results ({test_participant}):")
        logger.info(f"    Accuracy: {test_metrics['accuracy']:.3f}")
        logger.info(f"    F1 Macro: {test_metrics['f1_macro']:.3f}")
        logger.info(f"    F1 Weighted: {test_metrics['f1_weighted']:.3f}")
        logger.info(f"    AUC Weighted: {test_metrics['auc_weighted']:.3f}")
        logger.info(f"    Per-class F1: {test_metrics['f1_per_class']}")
    
    return fold_results

def analyze_comprehensive_results(fold_results):
    """Comprehensive analysis of cross-validation results."""
    
    logger.info("\n" + "="*80)
    logger.info("COMPREHENSIVE CROSS-VALIDATION RESULTS")
    logger.info("="*80)
    
    # Extract metrics across all folds
    metrics_summary = {
        'accuracy': [],
        'f1_macro': [],
        'f1_weighted': [],
        'auc_weighted': [],
        'precision_macro': [],
        'recall_macro': []
    }
    
    # Per-fold detailed results
    logger.info("\nPER-FOLD RESULTS:")
    logger.info("-" * 80)
    for fold in fold_results:
        metrics = fold['test_metrics']
        participant = fold['test_participant']
        
        logger.info(f"Fold {fold['fold']} ({participant}):")
        logger.info(f"  Accuracy:    {metrics['accuracy']:.3f}")
        logger.info(f"  F1 Macro:    {metrics['f1_macro']:.3f}")
        logger.info(f"  F1 Weighted: {metrics['f1_weighted']:.3f}")
        logger.info(f"  AUC:         {metrics['auc_weighted']:.3f}")
        logger.info(f"  Precision:   {metrics['precision_macro']:.3f}")
        logger.info(f"  Recall:      {metrics['recall_macro']:.3f}")
        
        # Store for summary statistics
        for key in metrics_summary:
            metrics_summary[key].append(metrics[key])
    
    # Calculate summary statistics
    logger.info("\nSUMMARY STATISTICS:")
    logger.info("-" * 80)
    
    results_df = pd.DataFrame(metrics_summary)
    
    for metric in metrics_summary:
        values = np.array(metrics_summary[metric])
        mean_val = np.mean(values)
        std_val = np.std(values)
        min_val = np.min(values)
        max_val = np.max(values)
        
        logger.info(f"{metric.upper():15}: {mean_val:.3f} ± {std_val:.3f} (range: {min_val:.3f} - {max_val:.3f})")
    
    # Literature comparison
    logger.info("\nLITERATURE COMPARISON:")
    logger.info("-" * 80)
    target_acc = 87.94  # MDPI Biology 2025 target
    achieved_acc = np.mean(metrics_summary['accuracy']) * 100
    achieved_f1 = np.mean(metrics_summary['f1_weighted']) * 100
    
    logger.info(f"Literature Target (CNN):     {target_acc:.2f}%")
    logger.info(f"Our Accuracy:               {achieved_acc:.2f}%")
    logger.info(f"Our F1 Weighted:            {achieved_f1:.2f}%")
    
    if achieved_acc >= target_acc:
        logger.info("🎉 TARGET ACHIEVED! Accuracy exceeds literature benchmark!")
    else:
        gap = target_acc - achieved_acc
        logger.info(f"Gap to target: {gap:.2f}% - Room for improvement")
    
    # Statistical significance testing (one-sample t-test against chance level)
    chance_level = 1/3  # 33.33% for 3-class problem
    t_stat, p_value = stats.ttest_1samp(metrics_summary['accuracy'], chance_level)
    logger.info(f"\nStatistical Significance (vs. chance level {chance_level:.1%}):")
    logger.info(f"  t-statistic: {t_stat:.3f}")
    logger.info(f"  p-value: {p_value:.6f}")
    logger.info(f"  Significant: {'Yes' if p_value < 0.05 else 'No'}")
    
    # Create comprehensive plots
    create_comprehensive_plots(fold_results, metrics_summary)
    
    # Save detailed results
    save_detailed_results(fold_results, metrics_summary)
    
    return results_df

def create_comprehensive_plots(fold_results, metrics_summary):
    """Create comprehensive visualization plots."""
    
    # 1. Metrics comparison across folds
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Cross-Validation Performance Metrics', fontsize=16)
    
    metrics_to_plot = ['accuracy', 'f1_macro', 'f1_weighted', 'auc_weighted', 'precision_macro', 'recall_macro']
    
    for i, metric in enumerate(metrics_to_plot):
        ax = axes[i//3, i%3]
        values = metrics_summary[metric]
        
        ax.bar(range(1, 6), values, alpha=0.7)
        ax.set_title(f'{metric.replace("_", " ").title()}')
        ax.set_xlabel('Fold')
        ax.set_ylabel('Score')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        
        # Add mean line
        mean_val = np.mean(values)
        ax.axhline(y=mean_val, color='red', linestyle='--', 
                  label=f'Mean: {mean_val:.3f}')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig('cv_metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Aggregate confusion matrix
    all_targets = []
    all_preds = []
    
    for fold in fold_results:
        all_targets.extend(fold['test_targets'])
        all_preds.extend(fold['test_predictions'])
    
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(all_targets, all_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Low', 'Moderate', 'High'],
                yticklabels=['Low', 'Moderate', 'High'])
    plt.title('Aggregate Confusion Matrix - Cross-Validation Results')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Add performance statistics
    accuracy = accuracy_score(all_targets, all_preds)
    f1_weighted = f1_score(all_targets, all_preds, average='weighted')
    plt.figtext(0.02, 0.02, f'Overall Accuracy: {accuracy:.3f}, F1 Weighted: {f1_weighted:.3f}', 
                fontsize=12, bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('aggregate_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

def save_detailed_results(fold_results, metrics_summary):
    """Save detailed results to files."""
    
    # Save comprehensive results
    results_to_save = {
        'fold_results': fold_results,
        'metrics_summary': metrics_summary,
        'overall_stats': {
            'mean_accuracy': np.mean(metrics_summary['accuracy']),
            'std_accuracy': np.std(metrics_summary['accuracy']),
            'mean_f1_weighted': np.mean(metrics_summary['f1_weighted']),
            'std_f1_weighted': np.std(metrics_summary['f1_weighted']),
            'mean_auc': np.mean(metrics_summary['auc_weighted']),
            'std_auc': np.std(metrics_summary['auc_weighted'])
        }
    }
    
    with open('comprehensive_training_results.pkl', 'wb') as f:
        pickle.dump(results_to_save, f)
    
    # Save CSV summary
    df = pd.DataFrame(metrics_summary)
    df.insert(0, 'fold', range(1, 6))
    df.insert(1, 'participant', [fold['test_participant'] for fold in fold_results])
    df.to_csv('cv_results_summary.csv', index=False)
    
    logger.info("Detailed results saved to:")
    logger.info("  - comprehensive_training_results.pkl")
    logger.info("  - cv_results_summary.csv")
    logger.info("  - cv_metrics_comparison.png")
    logger.info("  - aggregate_confusion_matrix.png")

def main():
    """Main training function with comprehensive evaluation."""
    
    logger.info("EEG Pain Classification - Comprehensive Training & Evaluation")
    logger.info("="*80)
    
    # Add explanatory message about accuracy expectations
    logger.info("\n📊 ACCURACY EXPECTATIONS:")
    logger.info("  • Random Baseline (3-class): ~33.3%")
    logger.info("  • Early Epochs (1-10): ~35-60% (normal learning phase)")
    logger.info("  • Target Performance: >87.94% (literature benchmark)")
    logger.info("  • Training typically improves over 50-100 epochs")
    logger.info("  • Progress bars show real-time accuracy improvements")
    logger.info("="*80)
    
    # Load data
    X, y, participants = load_preprocessed_data()
    
    # Test different models
    models_to_test = ['eegnet', 'shallow', 'deep']
    all_results = {}
    
    for model_type in models_to_test:
        logger.info(f"\n🚀 Training {model_type.upper()} model with comprehensive evaluation...")
        
        try:
            fold_results = cross_validate_model_comprehensive(
                X, y, participants, 
                model_type=model_type, 
                epochs=100, 
                batch_size=32
            )
            
            results_df = analyze_comprehensive_results(fold_results)
            all_results[model_type] = fold_results
            
        except Exception as e:
            logger.error(f"Error training {model_type}: {e}")
            continue
    
    # Final model comparison
    logger.info("\n" + "="*80)
    logger.info("FINAL MODEL COMPARISON")
    logger.info("="*80)
    
    comparison_results = {}
    for model_type, fold_results in all_results.items():
        metrics_summary = {
            'accuracy': [fold['test_metrics']['accuracy'] for fold in fold_results],
            'f1_weighted': [fold['test_metrics']['f1_weighted'] for fold in fold_results],
            'auc_weighted': [fold['test_metrics']['auc_weighted'] for fold in fold_results]
        }
        
        comparison_results[model_type] = {
            'accuracy_mean': np.mean(metrics_summary['accuracy']),
            'accuracy_std': np.std(metrics_summary['accuracy']),
            'f1_mean': np.mean(metrics_summary['f1_weighted']),
            'f1_std': np.std(metrics_summary['f1_weighted']),
            'auc_mean': np.mean(metrics_summary['auc_weighted']),
            'auc_std': np.std(metrics_summary['auc_weighted'])
        }
        
        logger.info(f"{model_type.upper()}:")
        logger.info(f"  Accuracy:  {comparison_results[model_type]['accuracy_mean']:.3f} ± {comparison_results[model_type]['accuracy_std']:.3f}")
        logger.info(f"  F1 Score:  {comparison_results[model_type]['f1_mean']:.3f} ± {comparison_results[model_type]['f1_std']:.3f}")
        logger.info(f"  AUC Score: {comparison_results[model_type]['auc_mean']:.3f} ± {comparison_results[model_type]['auc_std']:.3f}")
    
    # Find best model
    if comparison_results:
        best_model = max(comparison_results, key=lambda x: comparison_results[x]['f1_mean'])
        best_f1 = comparison_results[best_model]['f1_mean']
        best_acc = comparison_results[best_model]['accuracy_mean']
        logger.info(f"\n🏆 Best Model: {best_model.upper()}")
        logger.info(f"   F1 Score: {best_f1:.3f}, Accuracy: {best_acc:.3f}")

if __name__ == '__main__':
    main()
