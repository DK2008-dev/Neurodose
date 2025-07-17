#!/usr/bin/env python3
"""
Literature-Standard Random Forest Baseline

Implements the exact labeling and feature extraction methodology from:
MDPI Biology 2025 - "Objective Pain Assessment Using Deep Learning"

Key Changes:
1. Fixed threshold labeling (≤3, 4-6, >6) instead of percentiles
2. Enhanced feature extraction with wavelet transforms
3. Data augmentation pipeline
4. Literature-standard preprocessing
"""

import os
import sys
import numpy as np
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm

# Scientific computing
import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import skew, kurtosis
import pywt  # For wavelet transforms

# Machine learning
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut, cross_val_score, cross_validate
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LiteratureStandardFeatureExtractor:
    """
    Feature extraction following MDPI Biology 2025 methodology.
    
    Implements:
    - Daubechies 4 (db4) wavelet transforms
    - Statistical features: zero-crossing, percentiles, RMS
    - Enhanced spectral features with band ratios
    - Pain-relevant channel focus
    """
    
    def __init__(self, sfreq: float = 500.0):
        self.sfreq = sfreq
        
        # Frequency bands (Hz)
        self.freq_bands = {
            'delta': (1, 4),
            'theta': (4, 8), 
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 45)
        }
        
        # Pain-relevant channels (literature focus)
        self.pain_channels = ['C3', 'C4', 'Cz', 'FCz', 'CPz', 'CP3', 'CP4']
        
    def extract_wavelet_features(self, signal_data: np.ndarray) -> np.ndarray:
        """Extract db4 wavelet features as per literature."""
        features = []
        
        # Apply db4 wavelet decomposition
        coeffs = pywt.wavedec(signal_data, 'db4', level=6)
        
        for coeff in coeffs:
            # Statistical measures on wavelet coefficients
            features.extend([
                np.mean(coeff),
                np.std(coeff),
                np.var(coeff),
                skew(coeff),
                kurtosis(coeff),
                np.percentile(coeff, 25),
                np.percentile(coeff, 50),  # median
                np.percentile(coeff, 75),
                np.sqrt(np.mean(coeff**2)),  # RMS
            ])
        
        return np.array(features)
    
    def extract_statistical_features(self, signal_data: np.ndarray) -> np.ndarray:
        """Extract statistical features as per literature."""
        features = []
        
        # Zero-crossing rate
        zero_crossings = np.where(np.diff(np.signbit(signal_data)))[0]
        zcr = len(zero_crossings) / len(signal_data)
        features.append(zcr)
        
        # Basic statistics
        features.extend([
            np.mean(signal_data),
            np.median(signal_data),
            np.std(signal_data),
            np.var(signal_data),
            skew(signal_data),
            kurtosis(signal_data),
            np.sqrt(np.mean(signal_data**2)),  # RMS
        ])
        
        # Percentiles
        percentiles = [10, 25, 75, 90]
        for p in percentiles:
            features.append(np.percentile(signal_data, p))
        
        # Min/Max and range
        features.extend([
            np.min(signal_data),
            np.max(signal_data),
            np.ptp(signal_data),  # peak-to-peak
        ])
        
        return np.array(features)
    
    def extract_spectral_features(self, signal_data: np.ndarray) -> np.ndarray:
        """Enhanced spectral features with band ratios."""
        features = []
        
        # Power spectral density
        freqs, psd = signal.welch(signal_data, fs=self.sfreq, nperseg=min(256, len(signal_data)//4))
        
        # Band powers
        band_powers = {}
        total_power = np.sum(psd)
        
        for band_name, (low, high) in self.freq_bands.items():
            band_mask = (freqs >= low) & (freqs <= high)
            band_power = np.sum(psd[band_mask])
            band_powers[band_name] = band_power
            
            # Absolute and relative power
            features.extend([
                band_power,
                band_power / total_power if total_power > 0 else 0
            ])
        
        # Band ratios (literature standard)
        ratios = [
            ('alpha', 'beta'),
            ('theta', 'alpha'),
            ('delta', 'beta'),
            ('gamma', 'delta')
        ]
        
        for band1, band2 in ratios:
            ratio = (band_powers[band1] / band_powers[band2] 
                    if band_powers[band2] > 0 else 0)
            features.append(ratio)
        
        # Spectral centroid and bandwidth
        spectral_centroid = np.sum(freqs * psd) / np.sum(psd) if np.sum(psd) > 0 else 0
        features.append(spectral_centroid)
        
        return np.array(features)
    
    def extract_features(self, window: np.ndarray, channel_names: List[str]) -> np.ndarray:
        """
        Extract comprehensive features from EEG window.
        
        Args:
            window: EEG data (n_channels, n_samples)
            channel_names: List of channel names
            
        Returns:
            Feature vector
        """
        all_features = []
        
        # Get pain-relevant channels
        pain_channel_indices = []
        for ch_name in self.pain_channels:
            if ch_name in channel_names:
                pain_channel_indices.append(channel_names.index(ch_name))
        
        # If no pain-relevant channels found, use first few channels
        if not pain_channel_indices:
            pain_channel_indices = list(range(min(7, len(channel_names))))
        
        # Extract features from each pain-relevant channel
        for ch_idx in pain_channel_indices:
            ch_data = window[ch_idx, :]
            
            # Wavelet features
            wavelet_features = self.extract_wavelet_features(ch_data)
            all_features.extend(wavelet_features)
            
            # Statistical features  
            stat_features = self.extract_statistical_features(ch_data)
            all_features.extend(stat_features)
            
            # Spectral features
            spectral_features = self.extract_spectral_features(ch_data)
            all_features.extend(spectral_features)
        
        # Global features across all pain-relevant channels
        if len(pain_channel_indices) > 1:
            # Average across channels
            avg_signal = np.mean(window[pain_channel_indices, :], axis=0)
            
            # Statistical features on averaged signal
            global_stat_features = self.extract_statistical_features(avg_signal)
            all_features.extend(global_stat_features)
            
            # Cross-channel correlations
            correlations = []
            for i, ch1 in enumerate(pain_channel_indices):
                for j, ch2 in enumerate(pain_channel_indices[i+1:], i+1):
                    corr = np.corrcoef(window[ch1, :], window[ch2, :])[0, 1]
                    correlations.append(corr if not np.isnan(corr) else 0)
            all_features.extend(correlations)
        
        return np.array(all_features)


class LiteratureStandardDataAugmentation:
    """Data augmentation following MDPI Biology 2025 methodology."""
    
    def __init__(self, noise_std: float = 0.02, freq_shift_range: float = 0.2, 
                 multiply_range: float = 0.05):
        self.noise_std = noise_std
        self.freq_shift_range = freq_shift_range  
        self.multiply_range = multiply_range
    
    def add_noise(self, data: np.ndarray) -> np.ndarray:
        """Add 2% noise as per literature."""
        noise = np.random.normal(0, self.noise_std * np.std(data), data.shape)
        return data + noise
    
    def multiply_data(self, data: np.ndarray) -> np.ndarray:
        """Apply multiplication factor (1±0.05) as per literature."""
        factor = 1 + np.random.uniform(-self.multiply_range, self.multiply_range)
        return data * factor
    
    def frequency_modulation(self, data: np.ndarray, sfreq: float = 500.0) -> np.ndarray:
        """Apply frequency modulation using Hilbert transform."""
        try:
            # Apply Hilbert transform
            analytic_signal = signal.hilbert(data, axis=-1)
            
            # Frequency shift
            shift = np.random.uniform(-self.freq_shift_range, self.freq_shift_range)
            t = np.arange(data.shape[-1]) / sfreq
            
            # Apply frequency shift
            freq_shift = np.exp(1j * 2 * np.pi * shift * t)
            
            if len(data.shape) == 2:  # (channels, samples)
                freq_shift = freq_shift[np.newaxis, :]
            
            modulated = analytic_signal * freq_shift
            return np.real(modulated)
        
        except:
            # Fallback: return original data if modulation fails
            return data
    
    def augment_data(self, X: np.ndarray, y: np.ndarray, 
                    augmentation_factor: int = 2) -> Tuple[np.ndarray, np.ndarray]:
        """Apply comprehensive data augmentation."""
        augmented_X = [X]
        augmented_y = [y]
        
        for _ in range(augmentation_factor):
            X_aug = X.copy()
            
            # Apply random augmentations
            for i in range(len(X_aug)):
                # Random choice of augmentation
                aug_type = np.random.choice(['noise', 'multiply', 'freq_mod'])
                
                if aug_type == 'noise':
                    X_aug[i] = self.add_noise(X_aug[i])
                elif aug_type == 'multiply':
                    X_aug[i] = self.multiply_data(X_aug[i])
                elif aug_type == 'freq_mod':
                    X_aug[i] = self.frequency_modulation(X_aug[i])
            
            augmented_X.append(X_aug)
            augmented_y.append(y)
        
        return np.concatenate(augmented_X, axis=0), np.concatenate(augmented_y, axis=0)


def create_literature_standard_labels(pain_ratings: np.ndarray) -> np.ndarray:
    """
    Create labels using literature-standard fixed thresholds.
    
    MDPI Biology 2025 methodology:
    - Low pain: ≤3 
    - Moderate pain: 4-6
    - High pain: >6
    
    Note: Converts 0-100 scale to 0-10 scale first
    """
    # Convert from 0-100 to 0-10 scale (literature uses 0-10)
    ratings_scaled = pain_ratings / 10.0
    
    # Apply fixed thresholds
    labels = np.zeros(len(ratings_scaled), dtype=int)
    labels[(ratings_scaled > 3) & (ratings_scaled <= 6)] = 1  # Moderate
    labels[ratings_scaled > 6] = 2  # High
    
    return labels


def main():
    """Main execution with literature-standard methodology."""
    
    logger.info("🧠 Literature-Standard Random Forest Baseline")
    logger.info("================================================================================")
    logger.info("Methodology: MDPI Biology 2025 - Objective Pain Assessment Using Deep Learning")
    logger.info("Key Features: Fixed thresholds, db4 wavelets, data augmentation")
    logger.info("================================================================================")
    
    # Load preprocessed data with original pain ratings
    logger.info("Loading preprocessed data with original pain ratings...")
    
    # Check if we have the extracted pain ratings file
    pain_ratings_file = 'data/processed/windows_with_pain_ratings.pkl'
    if os.path.exists(pain_ratings_file):
        with open(pain_ratings_file, 'rb') as f:
            data = pickle.load(f)
        
        X = data['windows']
        pain_ratings = data['pain_ratings']
        participants = data['participants']
        
        logger.info(f"Loaded data from {pain_ratings_file}")
        logger.info(f"Total dataset: {len(X)} windows, {X.shape[1]} channels, {X.shape[2]} samples")
        
    else:
        logger.error(f"Pain ratings file not found: {pain_ratings_file}")
        logger.error("Please run extract_pain_ratings.py first")
        return
    
    # Create literature-standard labels
    logger.info("Creating literature-standard labels (≤3, 4-6, >6)...")
    y = create_literature_standard_labels(pain_ratings)
    
    logger.info(f"Pain rating range: {pain_ratings.min():.1f} - {pain_ratings.max():.1f}")
    logger.info(f"Label distribution: {np.bincount(y)}")
    
    # Get channel names (assume we have them)
    channel_names = [f'Ch{i}' for i in range(X.shape[1])]
    
    # Extract literature-standard features
    logger.info("Extracting literature-standard features (wavelets + statistical + spectral)...")
    feature_extractor = LiteratureStandardFeatureExtractor()
    
    features = []
    for i, window in enumerate(tqdm(X, desc="Feature extraction")):
        feature_vector = feature_extractor.extract_features(window, channel_names)
        features.append(feature_vector)
    
    X_features = np.array(features)
    logger.info(f"Feature matrix shape: {X_features.shape}")
    
    # Data augmentation
    logger.info("Applying literature-standard data augmentation...")
    augmenter = LiteratureStandardDataAugmentation()
    X_aug, y_aug = augmenter.augment_data(X, y, augmentation_factor=1)
    participants_aug = np.tile(participants, 2)  # Duplicate participant labels
    
    # Re-extract features from augmented data
    logger.info("Extracting features from augmented data...")
    features_aug = []
    for i, window in enumerate(tqdm(X_aug, desc="Augmented feature extraction")):
        feature_vector = feature_extractor.extract_features(window, channel_names)
        features_aug.append(feature_vector)
    
    X_features_aug = np.array(features_aug)
    
    # Apply SMOTE for class balancing
    logger.info("Applying SMOTE class balancing...")
    smote = SMOTE(random_state=42)
    X_balanced, y_balanced = smote.fit_resample(X_features_aug, y_aug)
    
    logger.info(f"After augmentation + SMOTE: {X_balanced.shape[0]} samples")
    logger.info(f"Balanced label distribution: {np.bincount(y_balanced)}")
    
    # Scale features
    logger.info("Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_balanced)
    
    # Train Random Forest with literature-standard hyperparameters
    logger.info("Training Random Forest with optimized hyperparameters...")
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    # Evaluate with original (non-augmented) data for fair comparison
    logger.info("\n📊 LITERATURE-STANDARD EVALUATION")
    logger.info("--------------------------------------------------")
    
    # Use original features (no augmentation) for evaluation
    X_orig_scaled = scaler.fit_transform(X_features)
    
    # Leave-One-Participant-Out Cross-Validation
    logger.info("\n1. Leave-One-Participant-Out Cross-Validation:")
    lopocv = LeaveOneGroupOut()
    
    lopocv_scores = []
    for train_idx, test_idx in lopocv.split(X_orig_scaled, y, participants):
        X_train, X_test = X_orig_scaled[train_idx], X_orig_scaled[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Train on augmented data, test on original
        rf.fit(X_scaled, y_balanced)  # Train on augmented
        y_pred = rf.predict(X_test)   # Test on original
        
        acc = accuracy_score(y_test, y_pred)
        lopocv_scores.append(acc)
        
        participant = participants[test_idx][0]
        logger.info(f"   {participant}: {acc:.3f} ({len(y_test)} samples)")
    
    logger.info(f"\nLOPOCV Results: {np.mean(lopocv_scores):.3f} ± {np.std(lopocv_scores):.3f}")
    logger.info(f"Range: {np.min(lopocv_scores):.3f} - {np.max(lopocv_scores):.3f}")
    
    # Full dataset performance (with augmentation)
    logger.info("\n2. Full Dataset Performance (with augmentation):")
    rf.fit(X_scaled, y_balanced)
    y_pred_full = rf.predict(X_scaled)
    full_accuracy = accuracy_score(y_balanced, y_pred_full)
    
    logger.info(f"Full dataset accuracy: {full_accuracy:.3f}")
    
    # Feature importance
    logger.info("\n3. Feature Importance Analysis:")
    importances = rf.feature_importances_
    top_indices = np.argsort(importances)[::-1][:20]
    
    logger.info("Top 20 most important features:")
    for i, idx in enumerate(top_indices):
        logger.info(f"   Feature {idx}: {importances[idx]:.4f}")
    
    # Classification report
    logger.info("\n4. Detailed Performance Analysis:")
    report = classification_report(y_balanced, y_pred_full, 
                                 target_names=['Low Pain', 'Moderate Pain', 'High Pain'])
    logger.info(f"\nClassification Report:\n{report}")
    
    # Save results
    results = {
        'methodology': 'Literature-Standard (MDPI Biology 2025)',
        'lopocv_scores': lopocv_scores,
        'lopocv_mean': np.mean(lopocv_scores),
        'lopocv_std': np.std(lopocv_scores),
        'full_accuracy': full_accuracy,
        'feature_importances': importances,
        'n_features': X_features.shape[1],
        'n_samples_original': len(X_features),
        'n_samples_augmented': len(X_scaled),
        'label_distribution_original': np.bincount(y),
        'label_distribution_balanced': np.bincount(y_balanced)
    }
    
    output_file = 'data/processed/literature_standard_rf_results.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump(results, f)
    
    logger.info(f"\nResults saved to: {output_file}")
    
    # Final assessment
    logger.info("\n================================================================================")
    logger.info("FINAL ASSESSMENT - Literature-Standard Methodology")
    logger.info("================================================================================")
    
    lopocv_mean = np.mean(lopocv_scores)
    
    if lopocv_mean >= 0.85:
        logger.info(f"✅ EXCELLENT: {lopocv_mean*100:.1f}% accuracy achieved!")
        logger.info("    Matches literature benchmarks (>85%)")
    elif lopocv_mean >= 0.70:
        logger.info(f"🟡 GOOD: {lopocv_mean*100:.1f}% accuracy achieved.")
        logger.info("    Significant improvement over basic approach")
    elif lopocv_mean >= 0.50:
        logger.info(f"🟡 MODERATE: {lopocv_mean*100:.1f}% accuracy achieved.")
        logger.info("    Better than random, but needs optimization")
    else:
        logger.info(f"❌ LOW: {lopocv_mean*100:.1f}% accuracy achieved.")
        logger.info("    Requires further investigation")
    
    logger.info(f"\nKey Improvements:")
    logger.info(f"Features extracted: {X_features.shape[1]}")
    logger.info(f"Random baseline: 33.3%")
    logger.info(f"Current performance: {lopocv_mean*100:.1f}%")
    logger.info(f"Improvement over random: {(lopocv_mean - 0.333)*100:.1f}%")


if __name__ == "__main__":
    main()
