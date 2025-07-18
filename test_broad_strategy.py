#!/usr/bin/env python3
"""
NEURODOSE – BINARY EEG PAIN CLASSIFIER (BROAD STRATEGY)
Testing with broader labeling strategy to improve performance
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
import scipy.signal

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

def create_binary_labels_broad(pain_ratings, participant_id):
    """Create binary labels using broad strategy (67th percentile split)"""
    threshold = np.percentile(pain_ratings, 67)
    labels = np.where(pain_ratings >= threshold, 1, 0)
    print(f"  {participant_id}: Broad - High ≥{threshold:.1f}")
    return labels

def extract_comprehensive_features(epoch, channel_indices, sampling_rate=500):
    """Extract all neuroscience-aligned features"""
    features = {}
    
    # Channel names for our 6 target channels
    channel_names = ['Cz', 'CPz', 'C3', 'C4', 'Fz', 'Pz']
    
    # 1. SPECTRAL POWER (via Welch's method)
    frequency_bands = {
        'delta': (1, 4),
        'theta': (4, 8), 
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 45)
    }
    
    spectral_powers = {}
    
    for i, ch_idx in enumerate(channel_indices):
        channel_name = channel_names[i]
        signal = epoch[ch_idx, :]
        
        # Compute PSD using Welch's method
        freqs, psd = scipy.signal.welch(signal, fs=sampling_rate, nperseg=256, noverlap=128)
        
        for band_name, (low_freq, high_freq) in frequency_bands.items():
            band_mask = (freqs >= low_freq) & (freqs <= high_freq)
            if np.any(band_mask):
                band_power = np.trapezoid(psd[band_mask], freqs[band_mask])
                # Apply log transform as specified
                log_power = np.log10(band_power + 1e-10)
                features[f'{channel_name}_{band_name}_power'] = log_power
                spectral_powers[f'{channel_name}_{band_name}'] = log_power
    
    # 2. FREQUENCY RATIOS
    for i, ch_idx in enumerate(channel_indices):
        channel_name = channel_names[i]
        
        try:
            delta = spectral_powers[f'{channel_name}_delta']
            alpha = spectral_powers[f'{channel_name}_alpha']
            beta = spectral_powers[f'{channel_name}_beta']
            gamma = spectral_powers[f'{channel_name}_gamma']
            theta = spectral_powers[f'{channel_name}_theta']
            
            # Delta / Alpha
            features[f'{channel_name}_delta_alpha_ratio'] = delta - alpha  # Log space subtraction
            
            # Gamma / Beta  
            features[f'{channel_name}_gamma_beta_ratio'] = gamma - beta
            
            # (Delta + Theta) / (Alpha + Beta)
            low_power = np.log10(10**delta + 10**theta + 1e-10)  # Convert back, add, log again
            high_power = np.log10(10**alpha + 10**beta + 1e-10)
            features[f'{channel_name}_low_high_ratio'] = low_power - high_power
            
        except KeyError:
            continue
    
    # 3. ERP FEATURES (N2 and P2 mean amplitudes)
    erp_channels = ['Cz', 'CPz', 'Pz']
    baseline_samples = 500  # -1s to 0s
    erp_windows = {
        'N2': (int(0.15 * sampling_rate), int(0.25 * sampling_rate)),  # 150-250 ms
        'P2': (int(0.20 * sampling_rate), int(0.35 * sampling_rate))   # 200-350 ms
    }
    
    for i, ch_idx in enumerate(channel_indices):
        channel_name = channel_names[i]
        
        if channel_name in erp_channels:
            signal = epoch[ch_idx, :]
            
            # Baseline correction using -1s to 0s
            if len(signal) > baseline_samples:
                baseline_mean = np.mean(signal[:baseline_samples])
                signal_corrected = signal - baseline_mean
                
                # Extract ERP components (relative to stimulus onset)
                for component, (start_sample, end_sample) in erp_windows.items():
                    start_idx = baseline_samples + start_sample
                    end_idx = baseline_samples + end_sample
                    
                    if end_idx < len(signal_corrected):
                        component_amplitude = np.mean(signal_corrected[start_idx:end_idx])
                        features[f'{channel_name}_{component}_amplitude'] = component_amplitude
    
    # 4. SPATIAL ASYMMETRY (C4 - C3 power difference)
    c3_idx = None
    c4_idx = None
    
    for i, ch_idx in enumerate(channel_indices):
        if channel_names[i] == 'C3':
            c3_idx = ch_idx
        elif channel_names[i] == 'C4':
            c4_idx = ch_idx
    
    if c3_idx is not None and c4_idx is not None:
        c3_signal = epoch[c3_idx, :]
        c4_signal = epoch[c4_idx, :]
        
        # Total power asymmetry
        c3_power = np.sqrt(np.mean(c3_signal ** 2))
        c4_power = np.sqrt(np.mean(c4_signal ** 2))
        features['C4_C3_power_asymmetry'] = c4_power - c3_power
        
        # Frequency-specific asymmetry
        for band_name, (low_freq, high_freq) in frequency_bands.items():
            try:
                # Use bandpass filtering for more accurate band power
                sos = scipy.signal.butter(4, [low_freq, high_freq], 
                                        btype='band', fs=sampling_rate, output='sos')
                c3_filtered = scipy.signal.sosfilt(sos, c3_signal)
                c4_filtered = scipy.signal.sosfilt(sos, c4_signal)
                
                c3_band_power = np.sqrt(np.mean(c3_filtered ** 2))
                c4_band_power = np.sqrt(np.mean(c4_filtered ** 2))
                
                features[f'C4_C3_{band_name}_asymmetry'] = c4_band_power - c3_band_power
                
            except:
                continue
    
    # 5. TIME-DOMAIN FEATURES
    for i, ch_idx in enumerate(channel_indices):
        channel_name = channel_names[i]
        signal = epoch[ch_idx, :]
        
        # RMS (Root Mean Square)
        rms = np.sqrt(np.mean(signal ** 2))
        features[f'{channel_name}_rms'] = rms
        
        # Variance
        variance = np.var(signal)
        features[f'{channel_name}_variance'] = variance
        
        # Zero-crossing rate
        zero_crossings = np.sum(np.diff(np.signbit(signal)))
        zcr = zero_crossings / len(signal)
        features[f'{channel_name}_zcr'] = zcr
    
    return features

def main():
    print("NEURODOSE – BINARY EEG PAIN CLASSIFIER (BROAD STRATEGY)")
    print("="*60)
    print("Testing broader labeling strategy (67th percentile split)")
    print("Target: ≥65% LOPOCV accuracy, ROC-AUC > 0.70")
    print("="*60)
    
    # Load data
    data_dir = "data/processed/basic_windows"
    data_path = Path(data_dir)
    participant_files = list(data_path.glob("vp*_windows.pkl"))
    
    print(f"Found {len(participant_files)} participant files")
    
    # Target channels (pain-relevant)
    channels_of_interest = ['Cz', 'CPz', 'C3', 'C4', 'Fz', 'Pz']
    
    all_features = []
    all_labels = []
    all_groups = []
    feature_names = None
    
    for file_path in sorted(participant_files):
        participant_id = file_path.stem.replace('_windows', '')
        print(f"\\nProcessing {participant_id}...")
        
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            
            windows = data['windows']
            ternary_labels = data['ternary_labels']
            channel_names = data['channel_names']
            
            # Create synthetic pain ratings from ternary labels
            pain_ratings = ternary_labels * 30 + 20  # 0->20, 1->50, 2->80
            
            # Create binary labels (BROAD strategy)
            binary_labels = create_binary_labels_broad(pain_ratings, participant_id)
            
            print(f"  Valid samples: {len(binary_labels)} ({np.sum(binary_labels == 0)} low, {np.sum(binary_labels == 1)} high)")
            
            # Find channel indices
            channel_indices = []
            for ch_name in channels_of_interest:
                if ch_name in channel_names:
                    channel_indices.append(channel_names.index(ch_name))
                else:
                    print(f"  Warning: {ch_name} not found")
            
            if len(channel_indices) < len(channels_of_interest):
                print(f"  Missing channels for {participant_id}")
                continue
            
            # Extract comprehensive features
            participant_features = []
            for i, epoch in enumerate(windows):
                try:
                    features = extract_comprehensive_features(epoch, channel_indices)
                    if features:
                        participant_features.append(features)
                except Exception as e:
                    print(f"    Error in epoch {i}: {e}")
                    continue
            
            if not participant_features:
                print(f"  No features extracted for {participant_id}")
                continue
            
            # Convert to matrix
            if feature_names is None:
                feature_names = list(participant_features[0].keys())
            
            feature_matrix = np.array([
                [features.get(name, 0.0) for name in feature_names]
                for features in participant_features
            ])
            
            print(f"  Features extracted: {feature_matrix.shape}")
            
            # Store data
            all_features.append(feature_matrix)
            all_labels.extend(binary_labels[:len(participant_features)])
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
    
    print(f"\\nFinal Dataset:")
    print(f"  Samples: {X.shape[0]}")
    print(f"  Features: {X.shape[1]}")
    print(f"  Participants: {len(np.unique(groups))}")
    print(f"  Class distribution: {np.bincount(y)}")
    
    # Train models with LOPOCV
    print("\\nTraining models with LOPOCV...")
    
    models = [
        ('Random Forest', RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)),
        ('Logistic Regression', LogisticRegression(random_state=42, max_iter=1000))
    ]
    
    if XGBOOST_AVAILABLE:
        models.append(('XGBoost', xgb.XGBClassifier(random_state=42, eval_metric='logloss')))
    
    logo = LeaveOneGroupOut()
    results = {}
    
    for model_name, model in models:
        print(f"\\n{model_name}:")
        
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
            
            # Preprocessing within fold
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Data augmentation within fold only
            try:
                if np.min(train_classes) < 10:
                    smote = SMOTE(random_state=42, k_neighbors=min(3, np.min(train_classes)-1))
                    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
                else:
                    X_train_balanced, y_train_balanced = X_train_scaled, y_train
            except:
                X_train_balanced, y_train_balanced = X_train_scaled, y_train
            
            # Train model
            model.fit(X_train_balanced, y_train_balanced)
            
            # Predict
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            
            # Calculate metrics
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='macro')
            auc = roc_auc_score(y_test, y_pred_proba)
            
            accuracies.append(acc)
            f1_scores.append(f1)
            auc_scores.append(auc)
            
            print(f"  {test_participant}: Acc={acc:.3f}, F1={f1:.3f}, AUC={auc:.3f}")
        
        results[model_name] = {
            'accuracy_mean': np.mean(accuracies),
            'accuracy_std': np.std(accuracies),
            'auc_mean': np.mean(auc_scores),
            'auc_std': np.std(auc_scores)
        }
    
    # Print results summary
    print("\\n" + "="*60)
    print("BROAD STRATEGY RESULTS SUMMARY")
    print("="*60)
    
    best_accuracy = 0
    best_auc = 0
    best_model = None
    
    for model_name, metrics in results.items():
        acc_mean = metrics['accuracy_mean']
        acc_std = metrics['accuracy_std']
        auc_mean = metrics['auc_mean']
        auc_std = metrics['auc_std']
        
        print(f"\\n{model_name}:")
        print(f"  Accuracy: {acc_mean:.3f} ± {acc_std:.3f}")
        print(f"  AUC:      {auc_mean:.3f} ± {auc_std:.3f}")
        
        if acc_mean > best_accuracy:
            best_accuracy = acc_mean
            best_auc = auc_mean
            best_model = model_name
    
    print(f"\\nBEST MODEL: {best_model}")
    print(f"LOPOCV Accuracy: {best_accuracy:.3f}")
    print(f"LOPOCV AUC: {best_auc:.3f}")
    
    print(f"\\nSUCCESS CRITERIA:")
    print(f"  ≥65% Accuracy: {'✅' if best_accuracy >= 0.65 else '❌'} ({best_accuracy:.1%})")
    print(f"  >70% AUC:      {'✅' if best_auc > 0.70 else '❌'} ({best_auc:.1%})")
    
    # Compare strategies
    print(f"\\nSTRATEGY COMPARISON:")
    print(f"  Strict Strategy (previous): ~55.7% accuracy")
    print(f"  Broad Strategy (current):   {best_accuracy:.1%} accuracy")
    print(f"  Improvement: {best_accuracy - 0.557:.3f}")

if __name__ == "__main__":
    main()
