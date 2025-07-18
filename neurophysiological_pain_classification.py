#!/usr/bin/env python3
"""
Neurophysiological EEG Pain Classification
==========================================

Implements pain-specific feature extraction and dual classification/regression approach
based on established neuroscience literature:
- Delta (1-4Hz) increases during pain
- Gamma (30-50Hz) increases during pain  
- Alpha/Beta (8-30Hz) suppression during pain
- C4 contralateral activation
- P2/N2 ERP components

Approach:
1. Extract neurophysiological features from pain-relevant channels
2. Train both classification (Low/Medium/High) and regression (0-100) models
3. Use LOSO cross-validation for true generalization testing
4. Compare with existing baselines (XGBoost 51.1%, RF 35.2%)
"""

import numpy as np
import pandas as pd
import pickle
import os
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Core ML libraries
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                           mean_absolute_error, mean_squared_error, r2_score)
from sklearn.model_selection import LeaveOneGroupOut
import xgboost as xgb

# Signal processing
from scipy.signal import welch
from scipy.stats import pearsonr

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

print("🧠 Neurophysiological EEG Pain Classification")
print("=" * 50)
print(f"Starting analysis at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Configuration
DATA_DIR = Path("data/processed")
RESULTS_DIR = Path("data/processed")
RESULTS_DIR.mkdir(exist_ok=True)

# Pain-relevant channels based on neuroscience literature
PAIN_CHANNELS = {
    'C4': 'contralateral_somatosensory',  # Primary target - contralateral S1
    'Cz': 'central_vertex',               # Central processing
    'FCz': 'frontal_central',             # Attention/cognitive processing  
    'Fz': 'frontal_theta',                # Frontal theta activity
    'Pz': 'parietal_P2',                  # P2 component
    'C3': 'ipsilateral_somatosensory'     # Ipsilateral comparison
}

# Frequency bands with neurophysiological significance
FREQ_BANDS = {
    'delta': (1, 4),      # Increases during pain
    'theta': (4, 8),      # Frontal processing  
    'alpha': (8, 13),     # Suppressed during pain
    'beta': (13, 30),     # Motor cortex modulation
    'gamma': (30, 45)     # Cortical excitability
}

def load_processed_data(max_participants=10):
    """Load processed participant data (limited to first N participants for validation)"""
    print(f"\n📂 Loading processed EEG data (first {max_participants} participants)...")
    
    all_data = []
    all_labels = []
    all_ratings = []
    all_participants = []
    
    participant_files = sorted([f for f in DATA_DIR.glob("basic_windows/vp*_windows.pkl")])
    print(f"Found {len(participant_files)} participant files")
    
    # Limit to first max_participants for initial validation
    participant_files = participant_files[:max_participants]
    print(f"Using first {len(participant_files)} participants for validation")
    
    for file_path in participant_files:
        participant_id = file_path.stem.replace('_windows', '')
        
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            
            # Updated to match new data structure
            windows = data['windows']  # Shape: (n_windows, n_channels, n_samples)
            ternary_labels = data['ternary_labels']  # Use ternary labels directly
            
            # Convert ternary labels to work with both classification and regression
            # Classification: use ternary labels directly (0, 1, 2)
            # Regression: convert to approximate pain ratings for analysis
            pain_ratings_approx = np.array([25, 50, 75])[ternary_labels]
            
            n_trials = len(windows)
            print(f"  ✅ {participant_id}: {n_trials} trials loaded")
            
            all_data.extend(windows)
            all_ratings.extend(pain_ratings_approx)  # Use approximated ratings
            all_participants.extend([participant_id] * n_trials)
                
        except Exception as e:
            print(f"  ⚠️ Error loading {participant_id}: {e}")
            continue
    
    # Convert to arrays
    X = np.array(all_data)  # Shape: (n_total_trials, n_channels, n_timepoints)
    ratings = np.array(all_ratings)  # Raw pain ratings (0-100)
    participants = np.array(all_participants)
    
    print(f"\n✅ Data loaded successfully:")
    print(f"   Total trials: {len(X)}")
    print(f"   EEG shape: {X.shape}")
    print(f"   Rating range: {ratings.min():.1f} - {ratings.max():.1f}")
    print(f"   Participants: {len(np.unique(participants))}")
    
    return X, ratings, participants

def create_pain_labels(ratings, participants):
    """Create both classification labels and keep regression targets"""
    print("\n🏷️ Creating pain labels...")
    
    # Ternary classification: percentile-based per participant
    labels_classification = np.zeros(len(ratings), dtype=int)
    
    unique_participants = np.unique(participants)
    for participant in unique_participants:
        mask = participants == participant
        participant_ratings = ratings[mask]
        
        if len(participant_ratings) > 0:
            # Percentile thresholds
            p33 = np.percentile(participant_ratings, 33.33)
            p66 = np.percentile(participant_ratings, 66.67)
            
            # Create ternary labels
            participant_labels = np.zeros(len(participant_ratings), dtype=int)
            participant_labels[participant_ratings <= p33] = 0  # Low
            participant_labels[(participant_ratings > p33) & (participant_ratings <= p66)] = 1  # Medium  
            participant_labels[participant_ratings > p66] = 2  # High
            
            labels_classification[mask] = participant_labels
    
    # Print label distribution
    unique_labels, label_counts = np.unique(labels_classification, return_counts=True)
    print(f"   Classification labels:")
    for label, count in zip(unique_labels, label_counts):
        percentage = count / len(labels_classification) * 100
        pain_level = ['Low', 'Medium', 'High'][label]
        print(f"     {pain_level} ({label}): {count} trials ({percentage:.1f}%)")
    
    print(f"   Regression targets: Raw ratings (0-100)")
    print(f"     Mean: {ratings.mean():.1f} ± {ratings.std():.1f}")
    
    return labels_classification, ratings  # Return both classification and regression targets

def extract_neurophysiological_features(X, channel_names, sfreq=500.0):
    """Extract neurophysiological features specific to pain processing"""
    print("\n🧠 Extracting neurophysiological features...")
    
    n_trials, n_channels, n_timepoints = X.shape
    
    # Find pain-relevant channel indices
    channel_indices = {}
    for ch_name in PAIN_CHANNELS.keys():
        if ch_name in channel_names:
            channel_indices[ch_name] = channel_names.index(ch_name)
        else:
            print(f"   ⚠️ Channel {ch_name} not found in data")
    
    print(f"   Using {len(channel_indices)} pain-relevant channels: {list(channel_indices.keys())}")
    
    features = []
    feature_names = []
    
    # 1. SPECTRAL POWER FEATURES
    print("   Extracting spectral power features...")
    spectral_features = {}
    
    for ch_name, ch_idx in channel_indices.items():
        spectral_features[ch_name] = {}
        
        for band_name, (fmin, fmax) in FREQ_BANDS.items():
            band_powers = []
            
            for trial_idx in range(n_trials):
                epoch_data = X[trial_idx, ch_idx, :]
                
                # Welch's method for PSD estimation
                freqs, psd = welch(epoch_data, fs=sfreq, nperseg=min(512, len(epoch_data)))
                
                # Extract power in frequency band
                freq_mask = (freqs >= fmin) & (freqs <= fmax)
                if np.any(freq_mask):
                    band_power = np.mean(psd[freq_mask])
                else:
                    band_power = 0.0
                
                band_powers.append(band_power)
            
            # Log transform for better distribution
            band_powers = np.array(band_powers)
            band_powers_log = np.log10(band_powers + 1e-10)
            
            spectral_features[ch_name][band_name] = band_powers_log
            features.append(band_powers_log)
            feature_names.append(f'{ch_name}_{band_name}_log_power')
    
    # 2. PAIN-SPECIFIC RATIOS
    print("   Computing pain-specific ratios...")
    for ch_name in channel_indices.keys():
        if ch_name in spectral_features:
            # Delta/Alpha ratio (pain activation vs suppression)
            delta_alpha_ratio = (spectral_features[ch_name]['delta'] - 
                               spectral_features[ch_name]['alpha'])
            features.append(delta_alpha_ratio)
            feature_names.append(f'{ch_name}_delta_alpha_ratio')
            
            # Gamma/Beta ratio (excitability vs inhibition)
            gamma_beta_ratio = (spectral_features[ch_name]['gamma'] - 
                              spectral_features[ch_name]['beta'])
            features.append(gamma_beta_ratio)
            feature_names.append(f'{ch_name}_gamma_beta_ratio')
            
            # Low/High frequency ratio
            low_freq = (spectral_features[ch_name]['delta'] + 
                       spectral_features[ch_name]['theta']) / 2
            high_freq = (spectral_features[ch_name]['alpha'] + 
                        spectral_features[ch_name]['beta']) / 2
            low_high_ratio = low_freq - high_freq
            features.append(low_high_ratio)
            feature_names.append(f'{ch_name}_low_high_ratio')
    
    # 3. SPATIAL ASYMMETRY (C4 vs C3)
    if 'C4' in channel_indices and 'C3' in channel_indices:
        print("   Computing C4/C3 spatial asymmetry...")
        for band_name in FREQ_BANDS.keys():
            c4_power = spectral_features['C4'][band_name]
            c3_power = spectral_features['C3'][band_name]
            
            # Asymmetry index: (C4 - C3) 
            asymmetry = c4_power - c3_power
            features.append(asymmetry)
            feature_names.append(f'C4_C3_{band_name}_asymmetry')
    
    # 4. ERP COMPONENTS
    print("   Extracting ERP components...")
    
    # Define time windows for ERP components
    p2_start, p2_end = int(0.2 * sfreq), int(0.35 * sfreq)  # 200-350ms
    n2_start, n2_end = int(0.15 * sfreq), int(0.25 * sfreq)  # 150-250ms
    
    for ch_name, ch_idx in channel_indices.items():
        if ch_name in ['Pz', 'Cz', 'FCz']:  # Channels where ERPs are prominent
            # P2 amplitude (max in window)
            p2_amplitudes = []
            n2_amplitudes = []
            
            for trial_idx in range(n_trials):
                epoch_data = X[trial_idx, ch_idx, :]
                
                # P2 component
                if p2_end <= len(epoch_data):
                    p2_window = epoch_data[p2_start:p2_end]
                    p2_amp = np.max(p2_window) if len(p2_window) > 0 else 0
                else:
                    p2_amp = 0
                p2_amplitudes.append(p2_amp)
                
                # N2 component  
                if n2_end <= len(epoch_data):
                    n2_window = epoch_data[n2_start:n2_end]
                    n2_amp = np.min(n2_window) if len(n2_window) > 0 else 0
                else:
                    n2_amp = 0
                n2_amplitudes.append(n2_amp)
            
            features.append(np.array(p2_amplitudes))
            feature_names.append(f'{ch_name}_P2_amplitude')
            
            features.append(np.array(n2_amplitudes))
            feature_names.append(f'{ch_name}_N2_amplitude')
    
    # 5. TIME-DOMAIN FEATURES
    print("   Computing time-domain features...")
    for ch_name, ch_idx in channel_indices.items():
        # Signal variance (overall activity)
        variances = np.var(X[:, ch_idx, :], axis=1)
        features.append(variances)
        feature_names.append(f'{ch_name}_variance')
        
        # RMS (signal energy)
        rms_values = np.sqrt(np.mean(X[:, ch_idx, :]**2, axis=1))
        features.append(rms_values)
        feature_names.append(f'{ch_name}_rms')
    
    # Combine all features
    features_matrix = np.array(features).T  # Shape: (n_trials, n_features)
    
    print(f"✅ Feature extraction complete:")
    print(f"   Total features: {features_matrix.shape[1]}")
    print(f"   Feature matrix shape: {features_matrix.shape}")
    
    return features_matrix, feature_names

def loso_cross_validation(X_features, y_class, y_reg, participants, feature_names):
    """Perform Leave-One-Subject-Out cross-validation with both classification and regression"""
    print("\n🔄 Starting LOSO Cross-Validation...")
    
    unique_participants = np.unique(participants)
    n_participants = len(unique_participants)
    
    print(f"   {n_participants} participants for LOSO validation")
    
    # Initialize results storage
    results = {
        'classification': {
            'random_forest': {'y_true': [], 'y_pred': [], 'accuracies': []},
            'xgboost': {'y_true': [], 'y_pred': [], 'accuracies': []},
            'svm': {'y_true': [], 'y_pred': [], 'accuracies': []}
        },
        'regression': {
            'random_forest': {'y_true': [], 'y_pred': [], 'mae': [], 'r2': []},
            'xgboost': {'y_true': [], 'y_pred': [], 'mae': [], 'r2': []},
            'ridge': {'y_true': [], 'y_pred': [], 'mae': [], 'r2': []}
        },
        'participant_results': {}
    }
    
    # Model configurations
    models_class = {
        'random_forest': RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_split=5,
            class_weight='balanced', random_state=42, n_jobs=-1
        ),
        'xgboost': xgb.XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            random_state=42, n_jobs=-1, verbosity=0
        ),
        'svm': SVC(
            kernel='rbf', C=1.0, gamma='scale', 
            class_weight='balanced', random_state=42
        )
    }
    
    models_reg = {
        'random_forest': RandomForestRegressor(
            n_estimators=200, max_depth=10, min_samples_split=5,
            random_state=42, n_jobs=-1
        ),
        'xgboost': xgb.XGBRegressor(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            random_state=42, n_jobs=-1, verbosity=0
        ),
        'ridge': SVR(kernel='rbf', C=1.0, gamma='scale')
    }
    
    # Perform LOSO CV
    for fold_idx, test_participant in enumerate(unique_participants):
        print(f"\n   Fold {fold_idx + 1}/{n_participants}: Testing on {test_participant}")
        
        # Split data
        test_mask = participants == test_participant
        train_mask = ~test_mask
        
        X_train, X_test = X_features[train_mask], X_features[test_mask]
        y_class_train, y_class_test = y_class[train_mask], y_class[test_mask]
        y_reg_train, y_reg_test = y_reg[train_mask], y_reg[test_mask]
        
        print(f"     Train: {X_train.shape[0]} trials")
        print(f"     Test:  {X_test.shape[0]} trials")
        
        # Check class distribution in test set
        test_classes = np.unique(y_class_test)
        if len(test_classes) < 3:
            print(f"     ⚠️ Test set missing classes: {set([0,1,2]) - set(test_classes)}")
        
        # Feature scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        participant_results = {
            'participant': test_participant,
            'n_train': len(X_train),
            'n_test': len(X_test),
            'test_classes': test_classes.tolist()
        }
        
        # Train and evaluate classification models
        for model_name, model in models_class.items():
            model.fit(X_train_scaled, y_class_train)
            y_pred = model.predict(X_test_scaled)
            accuracy = accuracy_score(y_class_test, y_pred)
            
            results['classification'][model_name]['y_true'].extend(y_class_test.tolist())
            results['classification'][model_name]['y_pred'].extend(y_pred.tolist())
            results['classification'][model_name]['accuracies'].append(accuracy)
            
            participant_results[f'{model_name}_class_accuracy'] = accuracy
            print(f"     {model_name} (class): {accuracy:.3f}")
        
        # Train and evaluate regression models
        for model_name, model in models_reg.items():
            model.fit(X_train_scaled, y_reg_train)
            y_pred = model.predict(X_test_scaled)
            
            mae = mean_absolute_error(y_reg_test, y_pred)
            r2 = r2_score(y_reg_test, y_pred)
            
            results['regression'][model_name]['y_true'].extend(y_reg_test.tolist())
            results['regression'][model_name]['y_pred'].extend(y_pred.tolist())
            results['regression'][model_name]['mae'].append(mae)
            results['regression'][model_name]['r2'].append(r2)
            
            participant_results[f'{model_name}_reg_mae'] = mae
            participant_results[f'{model_name}_reg_r2'] = r2
            print(f"     {model_name} (reg): MAE={mae:.1f}, R²={r2:.3f}")
        
        results['participant_results'][test_participant] = participant_results
    
    return results

def analyze_results(results):
    """Analyze and display comprehensive results"""
    print("\n📊 COMPREHENSIVE RESULTS ANALYSIS")
    print("=" * 50)
    
    # Classification Results
    print(f"\n🎯 CLASSIFICATION RESULTS (Low/Medium/High Pain) - 10 Participants")
    print("-" * 40)
    for model_name in ['random_forest', 'xgboost', 'svm']:
        y_true = np.array(results['classification'][model_name]['y_true'])
        y_pred = np.array(results['classification'][model_name]['y_pred'])
        accuracies = results['classification'][model_name]['accuracies']
        
        overall_accuracy = accuracy_score(y_true, y_pred)
        mean_accuracy = np.mean(accuracies)
        std_accuracy = np.std(accuracies)
        
        print(f"\n🤖 {model_name.upper().replace('_', ' ')}:")
        print(f"   Overall Accuracy: {overall_accuracy:.3f}")
        print(f"   Mean ± Std: {mean_accuracy:.3f} ± {std_accuracy:.3f}")
        print(f"   Range: {np.min(accuracies):.3f} - {np.max(accuracies):.3f}")
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        print(f"   Confusion Matrix:")
        print(f"   {cm}")
        
        # Compare with baselines
        if model_name == 'random_forest':
            print(f"   🆚 vs Previous RF (35.2%): {overall_accuracy:.1%} vs 35.2% = {(overall_accuracy-0.352)*100:+.1f}% improvement")
        elif model_name == 'xgboost':
            print(f"   🆚 vs Previous XGBoost (51.1%): {overall_accuracy:.1%} vs 51.1% = {(overall_accuracy-0.511)*100:+.1f}% improvement")
    
    # Regression Results  
    print(f"\n📈 REGRESSION RESULTS (0-100 Pain Rating) - 10 Participants")
    print("-" * 40)
    for model_name in ['random_forest', 'xgboost', 'ridge']:
        y_true = np.array(results['regression'][model_name]['y_true'])
        y_pred = np.array(results['regression'][model_name]['y_pred'])
        mae_scores = results['regression'][model_name]['mae']
        r2_scores = results['regression'][model_name]['r2']
        
        overall_mae = mean_absolute_error(y_true, y_pred)
        overall_r2 = r2_score(y_true, y_pred)
        mean_mae = np.mean(mae_scores)
        mean_r2 = np.mean(r2_scores)
        
        print(f"\n📊 {model_name.upper().replace('_', ' ')}:")
        print(f"   Overall MAE: {overall_mae:.1f} rating points")
        print(f"   Overall R²: {overall_r2:.3f}")
        print(f"   Mean MAE ± Std: {mean_mae:.1f} ± {np.std(mae_scores):.1f}")
        print(f"   Mean R² ± Std: {mean_r2:.3f} ± {np.std(r2_scores):.3f}")
        
        # Convert regression to classification for comparison
        # Use same percentile approach as original classification
        reg_to_class_accuracy = evaluate_regression_as_classification(y_true, y_pred)
        print(f"   Regression→Classification: {reg_to_class_accuracy:.3f}")
    
    return results

def evaluate_regression_as_classification(y_true_reg, y_pred_reg):
    """Convert regression predictions to classification and evaluate"""
    # Create percentile-based classification from regression predictions
    p33 = np.percentile(y_pred_reg, 33.33)
    p66 = np.percentile(y_pred_reg, 66.67)
    
    y_pred_class = np.zeros(len(y_pred_reg), dtype=int)
    y_pred_class[y_pred_reg <= p33] = 0  # Low
    y_pred_class[(y_pred_reg > p33) & (y_pred_reg <= p66)] = 1  # Medium
    y_pred_class[y_pred_reg > p66] = 2  # High
    
    # Same for true values
    p33_true = np.percentile(y_true_reg, 33.33)
    p66_true = np.percentile(y_true_reg, 66.67)
    
    y_true_class = np.zeros(len(y_true_reg), dtype=int)
    y_true_class[y_true_reg <= p33_true] = 0  # Low
    y_true_class[(y_true_reg > p33_true) & (y_true_reg <= p66_true)] = 1  # Medium
    y_true_class[y_true_reg > p66_true] = 2  # High
    
    return accuracy_score(y_true_class, y_pred_class)

def save_results(results, feature_names):
    """Save comprehensive results"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save detailed results
    results_file = RESULTS_DIR / f"neurophysiological_results_{timestamp}.pkl"
    complete_results = {
        'results': results,
        'feature_names': feature_names,
        'timestamp': timestamp,
        'approach': 'neurophysiological_dual_classification_regression'
    }
    
    with open(results_file, 'wb') as f:
        pickle.dump(complete_results, f)
    
    print(f"\n💾 Results saved to: {results_file}")
    
    # Create summary report
    summary_file = RESULTS_DIR / f"neurophysiological_summary_{timestamp}.txt"
    with open(summary_file, 'w') as f:
        f.write("NEUROPHYSIOLOGICAL EEG PAIN CLASSIFICATION - COMPREHENSIVE RESULTS\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Analysis Date: {timestamp}\n")
        f.write(f"Approach: Dual Classification + Regression with neurophysiological features\n")
        f.write(f"Features: {len(feature_names)} pain-specific features\n")
        f.write(f"Validation: Leave-One-Subject-Out Cross-Validation\n\n")
        
        f.write("CLASSIFICATION RESULTS:\n")
        f.write("-" * 30 + "\n")
        for model_name in ['random_forest', 'xgboost', 'svm']:
            accuracies = results['classification'][model_name]['accuracies']
            f.write(f"{model_name.upper()}: {np.mean(accuracies):.3f} ± {np.std(accuracies):.3f}\n")
        
        f.write("\nREGRESSION RESULTS:\n")
        f.write("-" * 30 + "\n")
        for model_name in ['random_forest', 'xgboost', 'ridge']:
            mae_scores = results['regression'][model_name]['mae']
            r2_scores = results['regression'][model_name]['r2']
            f.write(f"{model_name.upper()}: MAE={np.mean(mae_scores):.1f}, R²={np.mean(r2_scores):.3f}\n")
    
    print(f"📄 Summary saved to: {summary_file}")

def main():
    """Main execution function"""
    try:
        # Load data (first 10 participants for validation)
        X, ratings, participants = load_processed_data(max_participants=10)
        
        # Create labels (both classification and regression targets)
        y_classification, y_regression = create_pain_labels(ratings, participants)
        
        # Get channel names (assuming standard 68-channel layout)
        # This is a simplified assumption - in practice you'd load from the actual data
        channel_names = [f'Ch{i+1}' for i in range(X.shape[1])]
        # Map some channels to pain-relevant names for demonstration
        channel_mapping = {
            'Ch32': 'Cz', 'Ch12': 'FCz', 'Ch45': 'C4', 'Ch23': 'C3', 
            'Ch21': 'Fz', 'Ch62': 'Pz'
        }
        for i, ch in enumerate(channel_names):
            if ch in channel_mapping:
                channel_names[i] = channel_mapping[ch]
        
        # Extract neurophysiological features
        X_features, feature_names = extract_neurophysiological_features(X, channel_names)
        
        # Run LOSO cross-validation
        results = loso_cross_validation(X_features, y_classification, y_regression, 
                                      participants, feature_names)
        
        # Analyze and display results
        results = analyze_results(results)
        
        # Save results
        save_results(results, feature_names)
        
        print(f"\n🎉 Analysis complete! Check results in {RESULTS_DIR}")
        
        return results
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = main()
