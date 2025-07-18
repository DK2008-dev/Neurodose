#!/usr/bin/env python3
"""
Simplified Binary Pain Classifier - Debug Version
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import scipy.signal

def create_binary_labels(pain_ratings, participant_id, strategy='strict'):
    """Create binary labels from pain ratings"""
    if strategy == 'strict':
        low_threshold = np.percentile(pain_ratings, 33)
        high_threshold = np.percentile(pain_ratings, 67)
        
        labels = np.full(len(pain_ratings), np.nan)
        labels[pain_ratings <= low_threshold] = 0
        labels[pain_ratings >= high_threshold] = 1
        
        print(f"  {participant_id}: Strict labeling - Low ≤{low_threshold:.1f}, High ≥{high_threshold:.1f}")
    else:
        threshold = np.percentile(pain_ratings, 67)
        labels = np.where(pain_ratings >= threshold, 1, 0)
        print(f"  {participant_id}: Broad labeling - High ≥{threshold:.1f}")
    
    return labels

def extract_basic_features(epoch, channel_indices, channel_names, sampling_rate=500):
    """Extract basic spectral features"""
    features = {}
    
    frequency_bands = {
        'delta': (1, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 45)
    }
    
    for i, ch_idx in enumerate(channel_indices):
        try:
            channel_name = ['Cz', 'CPz', 'C3', 'C4', 'Fz', 'Pz'][i]
            signal = epoch[ch_idx, :]
            
            # Basic spectral features
            freqs, psd = scipy.signal.welch(signal, fs=sampling_rate, nperseg=256)
            
            for band_name, (low_freq, high_freq) in frequency_bands.items():
                band_mask = (freqs >= low_freq) & (freqs <= high_freq)
                if np.any(band_mask):
                    band_power = np.trapz(psd[band_mask], freqs[band_mask])
                    features[f'{channel_name}_{band_name}_power'] = np.log10(band_power + 1e-10)
            
            # Basic time-domain features
            features[f'{channel_name}_rms'] = np.sqrt(np.mean(signal ** 2))
            features[f'{channel_name}_variance'] = np.var(signal)
            
        except Exception as e:
            print(f"      Error extracting features for channel {i}: {e}")
            continue
    
    return features

def main():
    print("Simplified Binary Pain Classifier")
    print("="*50)
    
    # Load data
    data_dir = "data/processed/basic_windows"
    data_path = Path(data_dir)
    participant_files = list(data_path.glob("vp*_windows.pkl"))
    
    print(f"Found {len(participant_files)} participant files")
    
    # Target channels
    channels_of_interest = ['Cz', 'CPz', 'C3', 'C4', 'Fz', 'Pz']
    
    all_features = []
    all_labels = []
    all_groups = []
    
    for file_path in sorted(participant_files):
        participant_id = file_path.stem.replace('_windows', '')
        print(f"Processing {participant_id}...")
        
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            
            windows = data['windows']
            ternary_labels = data['ternary_labels']
            channel_names = data['channel_names']
            
            # Create synthetic pain ratings
            pain_ratings = ternary_labels * 30 + 20
            
            # Create binary labels
            binary_labels = create_binary_labels(pain_ratings, participant_id)
            
            # Keep only valid samples
            valid_mask = ~np.isnan(binary_labels)
            if not np.any(valid_mask):
                print(f"  No valid samples for {participant_id}")
                continue
            
            valid_windows = windows[valid_mask]
            valid_labels = binary_labels[valid_mask].astype(int)
            
            print(f"  Valid samples: {len(valid_labels)} ({np.sum(valid_labels == 0)} low, {np.sum(valid_labels == 1)} high)")
            
            # Find channel indices
            channel_indices = []
            for ch_name in channels_of_interest:
                if ch_name in channel_names:
                    channel_indices.append(channel_names.index(ch_name))
                else:
                    print(f"  Warning: {ch_name} not found")
            
            if len(channel_indices) < 3:
                print(f"  Too few channels found for {participant_id}")
                continue
            
            # Extract features
            participant_features = []
            for i, epoch in enumerate(valid_windows):
                try:
                    features = extract_basic_features(epoch, channel_indices, channel_names)
                    if features:
                        participant_features.append(features)
                except Exception as e:
                    print(f"    Error in epoch {i}: {e}")
                    continue
            
            if not participant_features:
                print(f"  No features extracted for {participant_id}")
                continue
            
            # Convert to matrix
            feature_names = list(participant_features[0].keys())
            feature_matrix = np.array([
                [features.get(name, 0.0) for name in feature_names]
                for features in participant_features
            ])
            
            print(f"  Features extracted: {feature_matrix.shape}")
            
            # Store data
            all_features.append(feature_matrix)
            all_labels.extend(valid_labels[:len(participant_features)])
            all_groups.extend([participant_id] * len(participant_features))
            
        except Exception as e:
            print(f"  Error processing {participant_id}: {e}")
            continue
    
    if not all_features:
        print("No valid data found!")
        return
    
    # Combine data
    X = np.vstack(all_features)
    y = np.array(all_labels)
    groups = np.array(all_groups)
    
    print(f"\nDataset summary:")
    print(f"  Samples: {X.shape[0]}")
    print(f"  Features: {X.shape[1]}")
    print(f"  Participants: {len(np.unique(groups))}")
    print(f"  Class distribution: {np.bincount(y)}")
    
    # Train with LOPOCV
    print(f"\nTraining with LOPOCV...")
    
    logo = LeaveOneGroupOut()
    accuracies = []
    f1_scores = []
    auc_scores = []
    
    for train_idx, test_idx in logo.split(X, y, groups):
        test_participant = groups[test_idx][0]
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Check class distribution
        train_classes = np.bincount(y_train, minlength=2)
        test_classes = np.bincount(y_test, minlength=2)
        
        if train_classes[0] == 0 or train_classes[1] == 0 or test_classes[0] == 0 or test_classes[1] == 0:
            print(f"  Skipping {test_participant}: insufficient class diversity")
            continue
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        # Predict
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        # Metrics
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')
        auc = roc_auc_score(y_test, y_pred_proba)
        
        accuracies.append(acc)
        f1_scores.append(f1)
        auc_scores.append(auc)
        
        print(f"  {test_participant}: Acc={acc:.3f}, F1={f1:.3f}, AUC={auc:.3f}")
    
    # Summary
    print(f"\nResults Summary:")
    print(f"  Accuracy: {np.mean(accuracies):.3f} ± {np.std(accuracies):.3f}")
    print(f"  F1-Score: {np.mean(f1_scores):.3f} ± {np.std(f1_scores):.3f}")
    print(f"  AUC:      {np.mean(auc_scores):.3f} ± {np.std(auc_scores):.3f}")
    print(f"  N Folds:  {len(accuracies)}")
    
    # Success check
    best_acc = np.mean(accuracies)
    best_auc = np.mean(auc_scores)
    
    print(f"\nSuccess Criteria:")
    print(f"  ≥65% Accuracy: {'✅' if best_acc >= 0.65 else '❌'} ({best_acc:.1%})")
    print(f"  >70% AUC:      {'✅' if best_auc > 0.70 else '❌'} ({best_auc:.1%})")

if __name__ == "__main__":
    main()
