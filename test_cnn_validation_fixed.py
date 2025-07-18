#!/usr/bin/env python3
"""
CNN Validation Test - EEG Pain Classification
Test CNN architectures against XGBoost baseline using LOPOCV.
"""

import os
import sys
import numpy as np
import pickle
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Add project root to path
sys.path.append(str(Path(__file__).parent))
from src.models.cnn import EEGNet, ShallowConvNet, DeepConvNet

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('cnn_validation.log'),
            logging.StreamHandler()
        ]
    )

def load_processed_data(data_dir: str = "data/processed/full_dataset") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load processed EEG data from pickle files"""
    print(f"[DATA] Loading processed data from: {data_dir}")
    
    if not os.path.exists(data_dir):
        print(f"[ERROR] Processed data not found at {data_dir}")
        return None, None, None
    
    files = [f for f in os.listdir(data_dir) if f.endswith('_windows.pkl')]
    if not files:
        print(f"[ERROR] No processed window files found in {data_dir}")
        return None, None, None
    
    all_windows = []
    all_labels = []
    all_participants = []
    
    print(f"[DATA] Found {len(files)} participant files")
    
    for i, file in enumerate(sorted(files)):
        participant_id = file.replace('_windows.pkl', '')
        print(f"   [LOADING] {participant_id} ({i+1}/{len(files)})...")
        
        try:
            with open(os.path.join(data_dir, file), 'rb') as f:
                data = pickle.load(f)
            
            windows = data['windows']  # Shape: (n_windows, n_channels, n_samples)
            labels = data['labels']    # Shape: (n_windows,)
            
            print(f"      [SUCCESS] {windows.shape[0]} windows, labels range {np.min(labels):.1f}-{np.max(labels):.1f}")
            
            all_windows.append(windows)
            all_labels.append(labels)
            all_participants.extend([participant_id] * len(labels))
            
        except Exception as e:
            print(f"      [ERROR] Failed to load {file}: {e}")
            continue
    
    if not all_windows:
        print("[ERROR] No valid data loaded")
        return None, None, None
    
    X = np.concatenate(all_windows, axis=0)
    y = np.concatenate(all_labels, axis=0)
    participants = np.array(all_participants)
    
    print(f"[SUCCESS] Dataset successfully loaded!")
    print(f"   [TOTAL] {X.shape[0]} windows from {len(files)} participants")
    print(f"   [SHAPE] Windows: {X.shape}, Labels: {y.shape}")
    print(f"   [LABELS] Range: {np.min(y):.1f} - {np.max(y):.1f}")
    
    return X, y, participants

def create_ternary_labels(pain_ratings: np.ndarray) -> np.ndarray:
    """Convert continuous pain ratings to ternary labels using percentiles"""
    # Use 33rd and 66th percentiles for balanced classes
    p33 = np.percentile(pain_ratings, 33.3)
    p66 = np.percentile(pain_ratings, 66.7)
    
    labels = np.zeros_like(pain_ratings, dtype=int)
    labels[pain_ratings <= p33] = 0  # Low pain
    labels[(pain_ratings > p33) & (pain_ratings <= p66)] = 1  # Moderate pain
    labels[pain_ratings > p66] = 2  # High pain
    
    print(f"[LABELS] Thresholds: Low <={p33:.1f}, Moderate {p33:.1f}-{p66:.1f}, High >{p66:.1f}")
    
    unique, counts = np.unique(labels, return_counts=True)
    for label, count in zip(unique, counts):
        label_name = ['Low', 'Moderate', 'High'][label]
        print(f"   [CLASS] {label_name}: {count} samples ({count/len(labels)*100:.1f}%)")
    
    return labels

def create_model(architecture: str, n_channels: int, n_samples: int, n_classes: int) -> nn.Module:
    """Create CNN model based on architecture name"""
    if architecture.lower() == 'eegnet':
        return EEGNet(n_channels=n_channels, n_samples=n_samples, n_classes=n_classes)
    elif architecture.lower() == 'shallowconvnet':
        return ShallowConvNet(n_channels=n_channels, n_samples=n_samples, n_classes=n_classes)
    elif architecture.lower() == 'deepconvnet':
        return DeepConvNet(n_channels=n_channels, n_samples=n_samples, n_classes=n_classes)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")

def train_model(model: nn.Module, train_loader: DataLoader, n_epochs: int = 50, 
                learning_rate: float = 0.001, device: str = 'cpu') -> nn.Module:
    """Train CNN model"""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    print(f"   [TRAIN] Training samples: {len(train_loader.dataset)}")
    print(f"   [TRAIN] Batch size: {train_loader.batch_size}, Epochs: {n_epochs}")
    
    model.train()
    for epoch in range(n_epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
        
        if (epoch + 1) % 10 == 0:
            accuracy = 100 * correct / total
            avg_loss = total_loss / len(train_loader)
            print(f"      [EPOCH] {epoch+1}/{n_epochs}: Loss={avg_loss:.4f}, Accuracy={accuracy:.1f}%")
    
    print("[SUCCESS] Training completed!")
    return model

def evaluate_model(model: nn.Module, test_loader: DataLoader, device: str = 'cpu') -> Tuple[float, np.ndarray, np.ndarray]:
    """Evaluate trained model"""
    model = model.to(device)
    model.eval()
    
    print(f"   [EVAL] Test samples: {len(test_loader.dataset)}")
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            _, predicted = torch.max(output.data, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    accuracy = accuracy_score(all_targets, all_predictions)
    
    print(f"[SUCCESS] Evaluation completed! Accuracy: {accuracy:.1%}")
    return accuracy, np.array(all_predictions), np.array(all_targets)

def main():
    print("[CNN] CNN VALIDATION TEST - Deep Learning vs. Traditional ML")
    print("=" * 65)
    print("[GOAL] Test if CNNs can exceed 51.1% XGBoost baseline")
    print("[METHOD] Leave-One-Participant-Out Cross-Validation")
    print("[MODELS] EEGNet, ShallowConvNet, DeepConvNet")
    print()
    
    start_time = time.time()
    
    # Load data
    print("[STEP 1] Loading processed EEG data...")
    X, y, participants = load_processed_data()
    
    if X is None:
        print("[ERROR] Failed to load data. Exiting.")
        return
    
    # Convert to ternary labels
    print("\n[STEP 2] Converting to ternary classification labels...")
    y_ternary = create_ternary_labels(y)
    
    # Setup cross-validation
    print("\n[STEP 3] Setting up Leave-One-Participant-Out Cross-Validation...")
    unique_participants = np.unique(participants)
    print(f"[CV] Total folds to process: {len(unique_participants)}")
    
    # Test architectures
    architectures = ['eegnet', 'shallowconvnet', 'deepconvnet']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[DEVICE] Using device: {device}")
    
    results = {}
    
    for arch in architectures:
        print(f"\n[ARCHITECTURE] Testing {arch.upper()}...")
        print("=" * 50)
        
        fold_accuracies = []
        fold_predictions = []
        fold_targets = []
        
        # Process participants in batches for memory efficiency
        batch_size = 5  # Process 5 participants at a time
        splits = [unique_participants[i:i+batch_size] for i in range(0, len(unique_participants), batch_size)]
        
        print(f"[CV] Processing {len(splits)} batches...")
        
        for batch_idx, participant_batch in enumerate(splits):
            print(f"\n[BATCH] {batch_idx+1}/{len(splits)}: {list(participant_batch)}")
            
            for fold_idx, test_participant in enumerate(participant_batch):
                print(f"   [FOLD] Testing on participant: {test_participant}")
                
                # Create train/test split
                train_idx = participants != test_participant
                test_idx = participants == test_participant
                n_test_samples = np.sum(test_idx)
                
                if n_test_samples == 0:
                    print(f"   [SKIP] No test samples for {test_participant}")
                    continue
                
                print(f"   [DATA] Train samples: {len(train_idx):,}, Test samples: {n_test_samples:,}")
                
                # Create datasets
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y_ternary[train_idx], y_ternary[test_idx]
                
                # Convert to PyTorch tensors
                X_train_tensor = torch.FloatTensor(X_train)
                y_train_tensor = torch.LongTensor(y_train)
                X_test_tensor = torch.FloatTensor(X_test)
                y_test_tensor = torch.LongTensor(y_test)
                
                train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
                test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
                
                train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
                test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
                
                try:
                    fold_start = time.time()
                    
                    # Create and train model
                    model = create_model(arch, X.shape[1], X.shape[2], 3)
                    model = train_model(model, train_loader, n_epochs=50, device=device)
                    
                    # Evaluate model
                    accuracy, predictions, targets = evaluate_model(model, test_loader, device=device)
                    
                    fold_time = time.time() - fold_start
                    fold_accuracies.append(accuracy)
                    fold_predictions.extend(predictions)
                    fold_targets.extend(targets)
                    
                    print(f"   [RESULT] {accuracy:.1%} accuracy in {fold_time:.1f}s")
                    
                    # Running average
                    if len(fold_accuracies) > 1:
                        current_avg = np.mean(fold_accuracies)
                        print(f"   [AVERAGE] Running average: {current_avg:.1%} ({len(fold_accuracies)} folds)")
                    
                except Exception as e:
                    print(f"   [ERROR] FOLD ERROR: {e}")
                    continue
        
        # Calculate final results for this architecture
        if fold_accuracies:
            mean_acc = np.mean(fold_accuracies)
            std_acc = np.std(fold_accuracies)
            
            print(f"\n[FINAL] {arch.upper()} FINAL RESULTS:")
            print(f"   [ACCURACY] Mean Accuracy: {mean_acc:.1%} ± {std_acc:.1%}")
            print(f"   [RANGE] Range: {np.min(fold_accuracies):.1%} - {np.max(fold_accuracies):.1%}")
            print(f"   [FOLDS] Completed folds: {len(fold_accuracies)}")
            
            # Confusion matrix
            if fold_targets:
                cm = confusion_matrix(fold_targets, fold_predictions)
                print(f"   [CONFUSION] Confusion Matrix:")
                print(f"     {cm}")
            
            results[arch] = {
                'mean_accuracy': mean_acc,
                'std_accuracy': std_acc,
                'fold_accuracies': fold_accuracies,
                'predictions': fold_predictions,
                'targets': fold_targets,
                'confusion_matrix': cm.tolist() if fold_targets else None
            }
        else:
            print(f"\n[ERROR] No successful folds for {arch.upper()}")
            results[arch] = None
    
    # Final comparison
    total_time = time.time() - start_time
    print(f"\n[COMPLETE] CNN VALIDATION COMPLETE")
    print("=" * 65)
    print(f"[TIME] Total execution time: {total_time/3600:.2f} hours")
    print(f"[BASELINE] XGBoost baseline to exceed: 51.1% ± 8.4%")
    print()
    
    best_arch = None
    best_acc = 0
    
    for arch, result in results.items():
        if result is not None:
            mean_acc = result['mean_accuracy']
            std_acc = result['std_accuracy']
            
            print(f"[RESULT] {arch.upper()}: {mean_acc:.1%} ± {std_acc:.1%}")
            
            if mean_acc > best_acc:
                best_acc = mean_acc
                best_arch = arch
            
            # Compare to baseline
            if mean_acc > 0.511:  # 51.1%
                print(f"   [SUCCESS] Exceeds XGBoost baseline!")
            else:
                print(f"   [BELOW] Below XGBoost baseline ({mean_acc:.1%} vs 51.1%)")
        else:
            print(f"[RESULT] {arch.upper()}: Failed")
    
    if best_arch:
        print(f"\n[BEST] Best architecture: {best_arch.upper()} ({best_acc:.1%})")
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = f"data/processed/cnn_validation_results_{timestamp}.pkl"
    
    try:
        os.makedirs(os.path.dirname(results_file), exist_ok=True)
        with open(results_file, 'wb') as f:
            pickle.dump({
                'results': results,
                'baseline_accuracy': 0.511,
                'total_time_hours': total_time/3600,
                'best_architecture': best_arch,
                'best_accuracy': best_acc,
                'timestamp': timestamp
            }, f)
        print(f"[SAVED] Results saved to: {results_file}")
    except Exception as e:
        print(f"[ERROR] Failed to save results: {e}")

if __name__ == "__main__":
    main()
