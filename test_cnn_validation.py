#!/usr/bin/env python3
"""
CNN Validation Test - Next Priority After Performance Gap Analysis

This script tests whether CNN models can achieve better performance than 
the 51.1% XGBoost baseline on raw EEG data, validating our hypothesis 
that deep learning is needed to capture temporal patterns.

Based on findings:
- Traditional ML: 51.1% ± 8.4% (essentially random baseline)
- Literature claims: 87-91% (likely due to data augmentation + different CV)
- Our validation: LOPOCV = clinically realistic, no participant data leakage
"""

import os
import sys
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import time
from typing import Tuple, List, Dict

# Add src to path
sys.path.append('src')
from models.cnn import create_model

def load_processed_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load all processed EEG windows and create participant groups."""
    
    print("🔄 Starting data loading process...")
    
    data_dir = "data/processed/full_dataset"
    if not os.path.exists(data_dir):
        print(f"❌ Processed data not found at {data_dir}")
        print("💡 Run automated processing first:")
        print("   python simple_automated_processing.py")
        return None, None, None
    
    # Find all processed files
    print("🔍 Scanning for processed files...")
    files = [f for f in os.listdir(data_dir) if f.endswith('_windows.pkl')]
    if not files:
        print(f"❌ No processed window files found in {data_dir}")
        return None, None, None
    
    print(f"📁 Found {len(files)} processed participants")
    print("⏳ Loading participant data...")
    
    all_windows = []
    all_labels = []
    all_participants = []
    
    for i, file in enumerate(sorted(files)):
        participant_id = file.split('_')[0]  # Extract vpXX from filename
        file_path = os.path.join(data_dir, file)
        
        print(f"   📊 Loading {participant_id} ({i+1}/{len(files)})...")
        
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            
            windows = data['windows']  # Shape: (n_windows, n_channels, n_samples)
            labels = data['labels']    # Pain intensity labels
            
            print(f"      ✅ {windows.shape[0]} windows, labels range {np.min(labels):.1f}-{np.max(labels):.1f}")
            
            all_windows.append(windows)
            all_labels.extend(labels)
            all_participants.extend([participant_id] * len(labels))
            
        except Exception as e:
            print(f"      ⚠️  Error loading {file}: {e}")
            continue
    
    if not all_windows:
        print("❌ No valid data loaded")
        return None, None, None
    
    print("🔄 Combining data from all participants...")
    # Combine all data
    X = np.vstack(all_windows)  # Shape: (total_windows, channels, samples)
    y = np.array(all_labels)
    participants = np.array(all_participants)
    
    print(f"✅ Dataset successfully loaded!")
    print(f"   🎯 Total: {X.shape[0]} windows from {len(files)} participants")
    print(f"   📐 EEG shape: {X.shape}")
    print(f"   📊 Labels range: {np.min(y):.1f} - {np.max(y):.1f}")
    
    return X, y, participants

def create_ternary_labels(pain_ratings: np.ndarray) -> np.ndarray:
    """Convert continuous pain ratings to ternary classes using percentiles."""
    print("🔄 Creating ternary labels from pain ratings...")
    
    # Use same approach as successful XGBoost test
    p33 = np.percentile(pain_ratings, 33.33)
    p66 = np.percentile(pain_ratings, 66.67)
    
    labels = np.zeros(len(pain_ratings), dtype=int)
    labels[pain_ratings > p33] = 1  # Moderate pain
    labels[pain_ratings > p66] = 2  # High pain
    
    print(f"📊 Label thresholds: Low ≤{p33:.1f}, Moderate {p33:.1f}-{p66:.1f}, High >{p66:.1f}")
    unique, counts = np.unique(labels, return_counts=True)
    for label, count in zip(unique, counts):
        pct = count / len(labels) * 100
        pain_type = ['Low', 'Moderate', 'High'][label]
        print(f"   - {pain_type}: {count} samples ({pct:.1f}%)")
    
    return labels

def create_data_loaders(X: np.ndarray, y: np.ndarray, 
                       train_idx: np.ndarray, test_idx: np.ndarray,
                       batch_size: int = 32) -> Tuple[DataLoader, DataLoader]:
    """Create PyTorch data loaders for train and test sets."""
    
    # Convert to tensors
    X_train = torch.FloatTensor(X[train_idx])
    y_train = torch.LongTensor(y[train_idx])
    X_test = torch.FloatTensor(X[test_idx])
    y_test = torch.LongTensor(y[test_idx])
    
    # Create datasets
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

def train_model(model: nn.Module, train_loader: DataLoader, 
                n_epochs: int = 50, learning_rate: float = 0.001,
                participant_name: str = "Unknown") -> nn.Module:
    """Train the CNN model with detailed progress logging."""
    
    print(f"🚀 Starting training for {participant_name}...")
    print(f"   📊 Training samples: {len(train_loader.dataset)}")
    print(f"   🔧 Epochs: {n_epochs}, Learning rate: {learning_rate}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   💻 Device: {device}")
    
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    model.train()
    print("⏳ Training progress:")
    
    for epoch in range(n_epochs):
        total_loss = 0
        correct_predictions = 0
        total_samples = 0
        
        for batch_idx, (batch_X, batch_y) in enumerate(train_loader):
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Calculate batch accuracy
            _, predicted = torch.max(outputs.data, 1)
            total_samples += batch_y.size(0)
            correct_predictions += (predicted == batch_y).sum().item()
        
        # Calculate epoch metrics
        avg_loss = total_loss / len(train_loader)
        accuracy = 100 * correct_predictions / total_samples
        
        # Print progress every 10 epochs or for first/last epochs
        if epoch % 10 == 0 or epoch == n_epochs - 1:
            print(f"   Epoch {epoch+1:2d}/{n_epochs}: Loss={avg_loss:.4f}, Accuracy={accuracy:.1f}%")
    
    print("✅ Training completed!")
    return model

def evaluate_model(model: nn.Module, test_loader: DataLoader, 
                  participant_name: str = "Unknown") -> Tuple[float, np.ndarray, np.ndarray]:
    """Evaluate the trained model with progress logging."""
    
    print(f"🔍 Evaluating model on {participant_name}...")
    print(f"   📊 Test samples: {len(test_loader.dataset)}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    all_predictions = []
    all_targets = []
    
    print("⏳ Running inference...")
    
    with torch.no_grad():
        for batch_idx, (batch_X, batch_y) in enumerate(test_loader):
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            predictions = torch.argmax(outputs, dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(batch_y.numpy())
            
            # Show progress for large test sets
            if len(test_loader) > 10 and (batch_idx + 1) % max(1, len(test_loader) // 5) == 0:
                progress = (batch_idx + 1) / len(test_loader) * 100
                print(f"   Progress: {progress:.0f}%")
    
    accuracy = accuracy_score(all_targets, all_predictions)
    print(f"✅ Evaluation completed! Accuracy: {accuracy:.1%}")
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
    print("STEP 1: DATA LOADING")
    print("-" * 30)
    X, y, participants = load_processed_data()
    if X is None:
        return
    
    # Create ternary labels (same as successful tests)
    print("\nSTEP 2: LABEL CREATION")
    print("-" * 30)
    y_ternary = create_ternary_labels(y)
    
    # Setup Leave-One-Participant-Out Cross-Validation
    print("\nSTEP 3: CROSS-VALIDATION SETUP")
    print("-" * 30)
    logo = LeaveOneGroupOut()
    participant_encoder = LabelEncoder()
    participant_codes = participant_encoder.fit_transform(participants)
    
    unique_participants = np.unique(participants)
    print(f"🔄 LOPOCV configured with {len(unique_participants)} participants")
    print(f"📊 Total folds to process: {len(unique_participants)}")
    
    # Test multiple CNN architectures
    architectures = ['eegnet', 'shallow_conv_net', 'deep_conv_net']
    results = {}
    
    print(f"\nSTEP 4: MODEL TRAINING & EVALUATION")
    print("-" * 30)
    print(f"🏗️  Testing {len(architectures)} CNN architectures")
    
    for arch_idx, arch in enumerate(architectures):
        print(f"\n{'='*50}")
        print(f"🏗️  ARCHITECTURE {arch_idx+1}/{len(architectures)}: {arch.upper()}")
        print(f"{'='*50}")
        
        fold_accuracies = []
        fold_participants = []
        
        splits = list(logo.split(X, y_ternary, participant_codes))
        print(f"📊 Processing {len(splits)} participants...")
        
        for fold, (train_idx, test_idx) in enumerate(splits):
            test_participant = participants[test_idx[0]]
            n_test_samples = len(test_idx)
            
            print(f"\n📋 FOLD {fold+1}/{len(splits)}: Testing on {test_participant}")
            print(f"   📊 Train samples: {len(train_idx):,}, Test samples: {n_test_samples:,}")
            
            fold_start = time.time()
            
            # Skip if insufficient data
            if n_test_samples < 5:
                print(f"      ⚠️  Skipping - insufficient test samples ({n_test_samples})")
                continue
            
            try:
                # Create model
                print(f"   🏗️  Creating {arch} model...")
                model = create_model(arch, n_channels=X.shape[1], n_samples=X.shape[2], n_classes=3)
                
                # Create data loaders
                print(f"   📦 Creating data loaders...")
                train_loader, test_loader = create_data_loaders(X, y_ternary, train_idx, test_idx)
                
                # Train model
                model = train_model(model, train_loader, n_epochs=30, participant_name=test_participant)
                
                # Evaluate model
                accuracy, predictions, targets = evaluate_model(model, test_loader, participant_name=test_participant)
                
                fold_accuracies.append(accuracy)
                fold_participants.append(test_participant)
                
                fold_time = time.time() - fold_start
                print(f"   🎯 FOLD RESULT: {accuracy:.1%} accuracy in {fold_time:.1f}s")
                
                # Show running average
                current_avg = np.mean(fold_accuracies)
                print(f"   📊 Running average: {current_avg:.1%} ({len(fold_accuracies)} folds)")
                
            except Exception as e:
                print(f"   ❌ FOLD ERROR: {e}")
                continue
        
        # Calculate overall performance
        if fold_accuracies:
            mean_acc = np.mean(fold_accuracies)
            std_acc = np.std(fold_accuracies)
            
            results[arch] = {
                'mean_accuracy': mean_acc,
                'std_accuracy': std_acc,
                'fold_accuracies': fold_accuracies,
                'participants': fold_participants
            }
            
            print(f"\n🎯 {arch.upper()} FINAL RESULTS:")
            print(f"   📊 Mean Accuracy: {mean_acc:.1%} ± {std_acc:.1%}")
            print(f"   📈 Range: {np.min(fold_accuracies):.1%} - {np.max(fold_accuracies):.1%}")
            print(f"   ✅ Valid folds: {len(fold_accuracies)}")
        else:
            print(f"\n❌ {arch.upper()}: No valid results - all folds failed")
    
    # Final comparison
    total_time = time.time() - start_time
    print(f"\n{'='*65}")
    print("🏆 FINAL RESULTS COMPARISON")
    print(f"{'='*65}")
    print(f"⏱️  Total execution time: {total_time/60:.1f} minutes")
    print()
    
    print("📊 TRADITIONAL ML BASELINES:")
    print(f"   🤖 XGBoost (Binary):    51.1% ± 8.4%  [LOPOCV, 49 participants]")
    print(f"   🌲 Random Forest:       35.2% ± 5.3%  [Ternary classification]")
    print(f"   🎲 Random Baseline:     33.3%         [Ternary classification]")
    print()
    
    print("🧠 CNN RESULTS (This Test):")
    if not results:
        print("   ❌ No valid CNN results obtained")
    else:
        for arch, result in results.items():
            mean_acc = result['mean_accuracy']
            std_acc = result['std_accuracy']
            n_folds = len(result['fold_accuracies'])
            print(f"   🏗️  {arch.upper():15}: {mean_acc:.1%} ± {std_acc:.1%} ({n_folds} folds)")
    
    print()
    
    # Performance analysis
    best_cnn = max(results.items(), key=lambda x: x[1]['mean_accuracy']) if results else None
    
    if best_cnn:
        best_arch, best_result = best_cnn
        best_acc = best_result['mean_accuracy']
        
        print("🔍 PERFORMANCE ANALYSIS:")
        if best_acc > 0.511:  # Better than XGBoost
            improvement = (best_acc - 0.511) * 100
            print(f"   ✅ CNN BREAKTHROUGH! {best_arch.upper()} beats XGBoost by {improvement:.1f}%")
            print(f"      🧠 Deep learning successfully captures temporal patterns!")
        elif best_acc > 0.40:  # Better than random but not XGBoost
            print(f"   📈 {best_arch.upper()} shows promise ({best_acc:.1%}) but needs optimization")
            print(f"      💡 Consider: data augmentation, longer training, architecture tuning")
        else:
            print(f"   ⚠️  CNNs also struggle ({best_acc:.1%}) - validates dataset difficulty")
            print(f"      🎯 Pain classification may be fundamentally challenging")
        
        print()
        print("💡 NEXT STEPS:")
        if best_acc > 0.40:
            print("   1. Implement MDPI study's data augmentation (5x dataset expansion)")
            print("   2. Try wavelet features instead of raw EEG")
            print("   3. Test longer training epochs with early stopping")
            print("   4. Experiment with different architectures")
        else:
            print("   1. Validate data quality and preprocessing")
            print("   2. Consider participant-specific models")
            print("   3. Explore other modalities (fMRI, physiological)")
            print("   4. Focus on binary classification first")
    else:
        print("❌ No valid CNN results - check data and model implementation")
    
    print(f"\n{'='*65}")
    print("🎯 CONCLUSION")
    print(f"{'='*65}")
    print("Your analysis correctly identified the performance gap!")
    print("LOPOCV provides realistic clinical deployment expectations.")
    print(f"🏁 Test completed in {total_time/60:.1f} minutes")
    
    # Save results for future reference
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_file = f"data/processed/cnn_validation_results_{timestamp}.pkl"
    
    print(f"\n💾 Saving results to: {results_file}")
    
    final_results = {
        'timestamp': timestamp,
        'execution_time_minutes': total_time/60,
        'dataset_info': {
            'total_windows': X.shape[0] if X is not None else 0,
            'participants': len(np.unique(participants)) if participants is not None else 0,
            'eeg_shape': X.shape if X is not None else None
        },
        'cnn_results': results,
        'baselines': {
            'xgboost_binary': {'accuracy': 0.511, 'std': 0.084, 'method': 'LOPOCV'},
            'random_forest_ternary': {'accuracy': 0.352, 'std': 0.053, 'method': 'LOPOCV'},
            'random_baseline_ternary': {'accuracy': 0.333, 'method': 'theoretical'}
        },
        'best_cnn': best_cnn[0] if best_cnn else None,
        'best_accuracy': best_cnn[1]['mean_accuracy'] if best_cnn else None,
        'conclusion': 'CNN breakthrough' if (best_cnn and best_cnn[1]['mean_accuracy'] > 0.511) else 'Traditional ML competitive'
    }
    
    with open(results_file, 'wb') as f:
        pickle.dump(final_results, f)
    
    print(f"✅ Results saved successfully!")
    print(f"📄 Access results later with: pickle.load(open('{results_file}', 'rb'))")

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
