#!/usr/bin/env python3
"""
Comprehensive Random Forest baseline with advanced EEG feature engineering.

This script implements sophisticated EEG-specific features that should achieve 70%+ accuracy:
- Spectral features (PSD in multiple frequency bands)
- Statistical features (moments, entropy, complexity)
- Spatial features (electrode regions, asymmetry)
- Temporal features (autocorrelation, stability)
- Pain-specific features (known pain-relevant channels and bands)
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from scipy import signal, stats
from scipy.stats import entropy
import logging
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdvancedEEGFeatureExtractor:
    """Extract comprehensive EEG features for pain classification."""
    
    def __init__(self, sfreq=500):
        self.sfreq = sfreq
        
        # Define EEG frequency bands (Hz)
        self.bands = {
            'delta': (1, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 45)
        }
        
        # Pain-relevant channel groups (indices for 68-channel cap)
        # Based on literature: central, parietal, frontal regions
        self.channel_groups = {
            'central': [26, 27, 28, 29, 30, 31, 32],  # C3, C1, Cz, C2, C4, CP1, CP2
            'parietal': [33, 34, 35, 36, 37, 38, 39], # CP3, CPz, CP4, P1, Pz, P2, POz
            'frontal': [1, 2, 3, 4, 5, 6, 7, 8, 9],   # Fp1, Fp2, F7, F3, Fz, F4, F8, FC1, FC2
            'temporal': [10, 11, 16, 17, 22, 23],      # T7, T8, TP7, TP8, etc.
            'pain_specific': [29, 34, 4, 8, 9]         # Cz, CPz, Fz, FC1, FC2 (known pain channels)
        }
    
    def extract_spectral_features(self, data):
        """Extract power spectral density features."""
        features = []
        
        # For each channel
        for ch in range(data.shape[0]):
            channel_data = data[ch, :]
            
            # Compute PSD using Welch's method
            freqs, psd = signal.welch(channel_data, fs=self.sfreq, nperseg=min(512, len(channel_data)//4))
            
            # Extract power in each frequency band
            for band_name, (low, high) in self.bands.items():
                band_mask = (freqs >= low) & (freqs <= high)
                band_power = np.mean(psd[band_mask])
                features.append(band_power)
            
            # Total power
            total_power = np.mean(psd)
            features.append(total_power)
            
            # Relative band powers
            for band_name, (low, high) in self.bands.items():
                band_mask = (freqs >= low) & (freqs <= high)
                relative_power = np.mean(psd[band_mask]) / total_power if total_power > 0 else 0
                features.append(relative_power)
            
            # Spectral edge frequency (95% of power)
            cumsum_psd = np.cumsum(psd)
            sef95_idx = np.where(cumsum_psd >= 0.95 * cumsum_psd[-1])[0]
            sef95 = freqs[sef95_idx[0]] if len(sef95_idx) > 0 else 0
            features.append(sef95)
            
            # Peak frequency
            peak_freq = freqs[np.argmax(psd)]
            features.append(peak_freq)
        
        return np.array(features)
    
    def extract_statistical_features(self, data):
        """Extract statistical features from time-domain signals."""
        features = []
        
        for ch in range(data.shape[0]):
            channel_data = data[ch, :]
            
            # Basic statistics
            features.extend([
                np.mean(channel_data),
                np.std(channel_data),
                np.var(channel_data),
                stats.skew(channel_data),
                stats.kurtosis(channel_data),
                np.median(channel_data),
                np.min(channel_data),
                np.max(channel_data),
                np.ptp(channel_data),  # peak-to-peak
            ])
            
            # Percentiles
            percentiles = [10, 25, 75, 90]
            for p in percentiles:
                features.append(np.percentile(channel_data, p))
            
            # Root mean square
            rms = np.sqrt(np.mean(channel_data**2))
            features.append(rms)
            
            # Zero crossing rate
            zero_crossings = np.where(np.diff(np.signbit(channel_data)))[0]
            zcr = len(zero_crossings) / len(channel_data)
            features.append(zcr)
            
            # Sample entropy (complexity measure)
            try:
                # Simplified entropy calculation
                hist, _ = np.histogram(channel_data, bins=50)
                hist = hist[hist > 0]  # Remove zero bins
                sample_entropy = entropy(hist)
                features.append(sample_entropy)
            except:
                features.append(0)
            
            # Hjorth parameters (activity, mobility, complexity)
            activity = np.var(channel_data)
            first_diff = np.diff(channel_data)
            second_diff = np.diff(first_diff)
            
            mobility = np.sqrt(np.var(first_diff) / activity) if activity > 0 else 0
            complexity = np.sqrt(np.var(second_diff) / np.var(first_diff)) / mobility if mobility > 0 else 0
            
            features.extend([activity, mobility, complexity])
        
        return np.array(features)
    
    def extract_spatial_features(self, data):
        """Extract spatial features based on electrode groups."""
        features = []
        
        # Features for each channel group
        for group_name, channels in self.channel_groups.items():
            if max(channels) < data.shape[0]:  # Check if channels exist
                group_data = data[channels, :]
                
                # Average activity in group
                group_mean = np.mean(group_data, axis=0)
                features.extend([
                    np.mean(group_mean),
                    np.std(group_mean),
                    np.var(group_mean)
                ])
                
                # Inter-channel correlation within group
                if len(channels) > 1:
                    corr_matrix = np.corrcoef(group_data)
                    # Average correlation (excluding diagonal)
                    mask = ~np.eye(corr_matrix.shape[0], dtype=bool)
                    avg_corr = np.mean(corr_matrix[mask])
                    features.append(avg_corr)
                else:
                    features.append(0)
        
        # Hemispheric asymmetry (if we have left-right electrode pairs)
        # Simplified: just use a few known pairs
        left_channels = [26, 33, 10]   # C3, CP3, T7 (approximate)
        right_channels = [30, 37, 11]  # C4, CP4, T8 (approximate)
        
        for left_ch, right_ch in zip(left_channels, right_channels):
            if left_ch < data.shape[0] and right_ch < data.shape[0]:
                left_power = np.var(data[left_ch, :])
                right_power = np.var(data[right_ch, :])
                asymmetry = (right_power - left_power) / (right_power + left_power) if (right_power + left_power) > 0 else 0
                features.append(asymmetry)
        
        return np.array(features)
    
    def extract_temporal_features(self, data):
        """Extract temporal features."""
        features = []
        
        # Focus on pain-specific channels for temporal analysis
        pain_channels = self.channel_groups['pain_specific']
        
        for ch in pain_channels:
            if ch < data.shape[0]:
                channel_data = data[ch, :]
                
                # Autocorrelation at different lags
                autocorr_lags = [1, 5, 10, 25, 50]  # Different time lags
                for lag in autocorr_lags:
                    if lag < len(channel_data):
                        autocorr = np.corrcoef(channel_data[:-lag], channel_data[lag:])[0, 1]
                        if np.isfinite(autocorr):
                            features.append(autocorr)
                        else:
                            features.append(0)
                    else:
                        features.append(0)
                
                # Signal stability (inverse of coefficient of variation)
                mean_val = np.mean(channel_data)
                std_val = np.std(channel_data)
                stability = mean_val / std_val if std_val > 0 else 0
                features.append(stability)
                
                # Line length (total variation)
                line_length = np.sum(np.abs(np.diff(channel_data)))
                features.append(line_length)
        
        return np.array(features)
    
    def extract_all_features(self, data):
        """Extract all feature types and concatenate."""
        logger.debug(f"Extracting features from data shape: {data.shape}")
        
        # Extract different feature types
        spectral_features = self.extract_spectral_features(data)
        statistical_features = self.extract_statistical_features(data)
        spatial_features = self.extract_spatial_features(data)
        temporal_features = self.extract_temporal_features(data)
        
        # Concatenate all features
        all_features = np.concatenate([
            spectral_features,
            statistical_features,
            spatial_features,
            temporal_features
        ])
        
        logger.debug(f"Total features extracted: {len(all_features)}")
        logger.debug(f"  Spectral: {len(spectral_features)}")
        logger.debug(f"  Statistical: {len(statistical_features)}")
        logger.debug(f"  Spatial: {len(spatial_features)}")
        logger.debug(f"  Temporal: {len(temporal_features)}")
        
        return all_features

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
        
        if file_path.exists():
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            
            windows = data['windows']  # Shape: (n_windows, n_channels, n_samples)
            labels = data['ternary_labels']  # Shape: (n_windows,)
            
            all_windows.append(windows)
            all_labels.append(labels)
            all_participants.extend([participant] * len(windows))
            
            logger.info(f"{participant}: {len(windows)} windows, labels: {np.bincount(labels)}")
        else:
            logger.warning(f"File not found: {file_path}")
    
    # Concatenate all data
    X = np.concatenate(all_windows, axis=0)
    y = np.concatenate(all_labels, axis=0)
    participants = np.array(all_participants)
    
    logger.info(f"Total dataset: {len(X)} windows, {X.shape[1]} channels, {X.shape[2]} samples")
    logger.info(f"Label distribution: {np.bincount(y)}")
    
    return X, y, participants

def comprehensive_rf_evaluation():
    """Comprehensive Random Forest evaluation with advanced features."""
    
    logger.info("🧠 Comprehensive Random Forest Baseline with Advanced EEG Features")
    logger.info("="*80)
    
    # Load data
    X, y, participants = load_preprocessed_data()
    
    # Initialize feature extractor
    feature_extractor = AdvancedEEGFeatureExtractor(sfreq=500)
    
    # Extract features for all windows
    logger.info("Extracting advanced EEG features...")
    X_features = []
    
    for i in tqdm(range(len(X)), desc="Feature extraction"):
        features = feature_extractor.extract_all_features(X[i])
        X_features.append(features)
    
    X_features = np.array(X_features)
    logger.info(f"Feature matrix shape: {X_features.shape}")
    
    # Check for invalid features
    finite_mask = np.isfinite(X_features).all(axis=1)
    if not finite_mask.all():
        logger.warning(f"Removing {(~finite_mask).sum()} samples with invalid features")
        X_features = X_features[finite_mask]
        y = y[finite_mask]
        participants = participants[finite_mask]
    
    # Standardize features
    scaler = StandardScaler()
    X_features_scaled = scaler.fit_transform(X_features)
    
    logger.info(f"Final dataset: {len(X_features_scaled)} samples, {X_features_scaled.shape[1]} features")
    
    # Cross-validation evaluation
    logger.info("\n📊 COMPREHENSIVE EVALUATION")
    logger.info("-"*50)
    
    # 1. Leave-One-Participant-Out Cross-Validation
    logger.info("\n1. Leave-One-Participant-Out Cross-Validation:")
    logo = LeaveOneGroupOut()
    rf_lopocv = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    lopocv_scores = []
    participant_results = {}
    
    for fold, (train_idx, test_idx) in enumerate(logo.split(X_features_scaled, y, participants)):
        test_participant = participants[test_idx][0]
        
        X_train, X_test = X_features_scaled[train_idx], X_features_scaled[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        rf_lopocv.fit(X_train, y_train)
        y_pred = rf_lopocv.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        lopocv_scores.append(accuracy)
        participant_results[test_participant] = {
            'accuracy': accuracy,
            'n_samples': len(y_test),
            'true_labels': y_test,
            'predictions': y_pred
        }
        
        logger.info(f"  {test_participant}: {accuracy:.3f} ({len(y_test)} samples)")
    
    logger.info(f"\nLOPOCV Results: {np.mean(lopocv_scores):.3f} ± {np.std(lopocv_scores):.3f}")
    logger.info(f"Range: {np.min(lopocv_scores):.3f} - {np.max(lopocv_scores):.3f}")
    
    # 2. Standard 5-fold cross-validation (for comparison)
    logger.info("\n2. Standard 5-Fold Cross-Validation:")
    rf_standard = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    cv_scores = cross_val_score(rf_standard, X_features_scaled, y, cv=5, scoring='accuracy')
    logger.info(f"5-Fold CV: {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}")
    
    # 3. Feature importance analysis
    logger.info("\n3. Feature Importance Analysis:")
    rf_full = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    rf_full.fit(X_features_scaled, y)
    
    feature_importance = rf_full.feature_importances_
    top_features_idx = np.argsort(feature_importance)[-20:]  # Top 20 features
    
    logger.info("Top 20 most important features:")
    for i, idx in enumerate(reversed(top_features_idx)):
        logger.info(f"  Feature {idx}: {feature_importance[idx]:.4f}")
    
    # 4. Detailed classification report
    logger.info("\n4. Detailed Performance Analysis:")
    
    # Train on all data for final evaluation
    y_pred_full = rf_full.predict(X_features_scaled)
    
    logger.info("\nClassification Report (Full Dataset):")
    class_names = ['Low Pain', 'Moderate Pain', 'High Pain']
    print(classification_report(y, y_pred_full, target_names=class_names))
    
    # Confusion matrix
    cm = confusion_matrix(y, y_pred_full)
    logger.info(f"\nConfusion Matrix:")
    logger.info(f"True\\Pred   Low   Mod  High")
    for i, row in enumerate(cm):
        logger.info(f"{class_names[i][:4]:>8}: {row[0]:4d}  {row[1]:4d}  {row[2]:4d}")
    
    # 5. Performance by participant
    logger.info("\n5. Performance by Participant:")
    for participant, results in participant_results.items():
        acc = results['accuracy']
        n_samples = results['n_samples']
        logger.info(f"  {participant}: {acc:.3f} ({n_samples} samples)")
        
        # Per-class accuracy for this participant
        y_true = results['true_labels']
        y_pred = results['predictions']
        for class_idx in range(3):
            class_mask = y_true == class_idx
            if class_mask.sum() > 0:
                class_acc = (y_pred[class_mask] == class_idx).mean()
                logger.info(f"    {class_names[class_idx]}: {class_acc:.3f}")
    
    # Save results
    results = {
        'lopocv_scores': lopocv_scores,
        'lopocv_mean': np.mean(lopocv_scores),
        'lopocv_std': np.std(lopocv_scores),
        'cv_scores': cv_scores,
        'cv_mean': np.mean(cv_scores),
        'cv_std': np.std(cv_scores),
        'feature_importance': feature_importance,
        'participant_results': participant_results,
        'n_features': X_features_scaled.shape[1]
    }
    
    output_file = Path('data/processed/comprehensive_rf_results.pkl')
    with open(output_file, 'wb') as f:
        pickle.dump(results, f)
    
    logger.info(f"\nResults saved to: {output_file}")
    
    # Final assessment
    logger.info("\n" + "="*80)
    logger.info("FINAL ASSESSMENT")
    logger.info("="*80)
    
    lopocv_mean = np.mean(lopocv_scores)
    if lopocv_mean >= 0.70:
        logger.info(f"✅ EXCELLENT: {lopocv_mean:.1%} accuracy achieved!")
        logger.info("   Data quality is sufficient for CNN training.")
    elif lopocv_mean >= 0.60:
        logger.info(f"✅ GOOD: {lopocv_mean:.1%} accuracy achieved.")
        logger.info("   Data shows clear signal, CNN training recommended.")
    elif lopocv_mean >= 0.45:
        logger.info(f"⚠️  MODERATE: {lopocv_mean:.1%} accuracy achieved.")
        logger.info("   Some signal present, CNN may improve performance.")
    else:
        logger.info(f"❌ LOW: {lopocv_mean:.1%} accuracy achieved.")
        logger.info("   Data quality concerns - review preprocessing.")
    
    logger.info(f"\nFeatures extracted: {X_features_scaled.shape[1]}")
    logger.info(f"Random baseline: 33.3%")
    logger.info(f"Current performance: {lopocv_mean:.1%}")
    logger.info(f"Improvement over random: {(lopocv_mean - 0.333) / 0.333 * 100:.1f}%")
    
    return results

if __name__ == '__main__':
    results = comprehensive_rf_evaluation()
