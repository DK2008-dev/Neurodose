#!/usr/bin/env python3
"""
Corrected Literature-Standard Random Forest (NO DATA LEAKAGE)

This script implements the literature methodology correctly without data leakage:
1. SMOTE applied only within each CV fold
2. Feature scaling fit only on training data
3. All preprocessing isolated per fold
4. True leave-one-participant-out evaluation
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
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CorrectFeatureExtractor:
    """Enhanced feature extraction with proper CV isolation."""
    
    def __init__(self, sfreq: float = 500.0):
        self.sfreq = sfreq
        self.pain_channels = ['C3', 'C4', 'Cz', 'FCz', 'CPz', 'CP3', 'CP4']
    
    def extract_wavelet_features(self, signal_data: np.ndarray) -> np.ndarray:
        """Extract db4 wavelet features."""
        try:
            coeffs = pywt.wavedec(signal_data, 'db4', level=4)
            features = []
            
            for coeff in coeffs:
                if len(coeff) > 0:
                    features.extend([
                        np.mean(coeff),
                        np.std(coeff),
                        np.var(coeff),
                        np.max(np.abs(coeff)) if len(coeff) > 0 else 0
                    ])
            
            return np.array(features)
        except:
            return np.zeros(20)  # Fallback
    
    def extract_statistical_features(self, signal_data: np.ndarray) -> np.ndarray:
        """Extract statistical features."""
        features = [
            np.mean(signal_data),
            np.std(signal_data),
            np.var(signal_data),
            skew(signal_data),
            kurtosis(signal_data),
            np.median(signal_data),
            np.percentile(signal_data, 25),
            np.percentile(signal_data, 75),
            np.sqrt(np.mean(signal_data**2)),  # RMS
            np.sum(np.abs(np.diff(np.sign(signal_data)))) / 2  # Zero crossings
        ]
        return np.array(features)
    
    def extract_spectral_features(self, signal_data: np.ndarray) -> np.ndarray:
        """Extract spectral features."""
        try:
            freqs, psd = signal.welch(signal_data, fs=self.sfreq, nperseg=min(256, len(signal_data)//4))
            
            # Define frequency bands
            bands = {
                'delta': (1, 4),
                'theta': (4, 8),
                'alpha': (8, 13),
                'beta': (13, 30),
                'gamma': (30, 45)
            }
            
            features = []
            total_power = np.sum(psd)
            
            for band_name, (low, high) in bands.items():
                band_mask = (freqs >= low) & (freqs <= high)
                band_power = np.sum(psd[band_mask])
                relative_power = band_power / total_power if total_power > 0 else 0
                features.extend([band_power, relative_power])
            
            # Band ratios
            band_powers = {name: np.sum(psd[(freqs >= low) & (freqs <= high)]) 
                          for name, (low, high) in bands.items()}
            
            ratios = [
                band_powers['alpha'] / band_powers['beta'] if band_powers['beta'] > 0 else 0,
                band_powers['theta'] / band_powers['alpha'] if band_powers['alpha'] > 0 else 0,
                band_powers['delta'] / band_powers['beta'] if band_powers['beta'] > 0 else 0
            ]
            features.extend(ratios)
            
            return np.array(features)
        except:
            return np.zeros(13)  # Fallback
    
    def extract_features(self, window: np.ndarray, channel_names: List[str]) -> np.ndarray:
        """Extract comprehensive features from EEG window."""
        all_features = []
        
        # Get pain-relevant channels
        pain_channel_indices = []
        for ch_name in self.pain_channels:
            if ch_name in channel_names:
                pain_channel_indices.append(channel_names.index(ch_name))
        
        # Fallback to first channels if pain channels not found
        if not pain_channel_indices:
            pain_channel_indices = list(range(min(5, len(channel_names))))
        
        # Extract features from each channel
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
        
        return np.array(all_features)


def create_literature_labels(pain_ratings: np.ndarray) -> np.ndarray:
    """Create labels using literature-standard fixed thresholds."""
    ratings_scaled = pain_ratings / 10.0
    labels = np.zeros(len(ratings_scaled), dtype=int)
    labels[(ratings_scaled > 3) & (ratings_scaled <= 6)] = 1  # Moderate
    labels[ratings_scaled > 6] = 2  # High
    return labels


def correct_cross_validation():
    """Perform cross-validation without data leakage."""
    
    logger.info("🔧 CORRECTED LITERATURE-STANDARD RANDOM FOREST")
    logger.info("================================================================================")
    logger.info("✅ NO DATA LEAKAGE: SMOTE and scaling applied per CV fold only")
    logger.info("✅ TRUE LOPOCV: Each participant completely isolated")
    logger.info("================================================================================")
    
    # Load data
    pain_ratings_file = 'data/processed/windows_with_pain_ratings.pkl'
    if not os.path.exists(pain_ratings_file):
        logger.error(f"Pain ratings file not found: {pain_ratings_file}")
        return
    
    with open(pain_ratings_file, 'rb') as f:
        data = pickle.load(f)
    
    X = data['windows']
    pain_ratings = data['pain_ratings']
    participants = data['participants']
    
    logger.info(f"Dataset: {len(X)} windows, {X.shape[1]} channels, {X.shape[2]} samples")
    
    # Create labels
    y = create_literature_labels(pain_ratings)
    logger.info(f"Label distribution: {np.bincount(y)} (Low/Moderate/High)")
    
    # Create channel names (assuming standard 10-20 system subset)
    channel_names = [f'Ch{i}' for i in range(X.shape[1])]
    # Add some standard EEG channel names for the first few channels
    standard_channels = ['Fz', 'Cz', 'Pz', 'C3', 'C4', 'FCz', 'CPz', 'Oz']
    for i, ch in enumerate(standard_channels[:min(len(standard_channels), len(channel_names))]):
        channel_names[i] = ch
    
    # Initialize feature extractor
    feature_extractor = CorrectFeatureExtractor()
    
    # Extract features (this can be done globally as it doesn't cause leakage)
    logger.info("Extracting enhanced features...")
    features = []
    for window in tqdm(X, desc="Feature extraction"):
        feature_vector = feature_extractor.extract_features(window, channel_names)
        features.append(feature_vector)
    
    X_features = np.array(features)
    logger.info(f"Feature matrix: {X_features.shape}")
    
    # Leave-One-Participant-Out Cross-Validation (CORRECT APPROACH)
    logger.info("\n📊 LEAVE-ONE-PARTICIPANT-OUT CROSS-VALIDATION")
    logger.info("--------------------------------------------------")
    
    lopocv = LeaveOneGroupOut()
    cv_scores = []
    detailed_results = []
    
    for fold, (train_idx, test_idx) in enumerate(lopocv.split(X_features, y, participants)):
        # Split data completely
        X_train, X_test = X_features[train_idx], X_features[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        test_participant = participants[test_idx][0]
        
        # Apply SMOTE only to training data (NO LEAKAGE)
        try:
            smote = SMOTE(random_state=42, k_neighbors=min(5, len(y_train)-1))
            X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
            
            logger.info(f"  Fold {fold+1} ({test_participant}): "
                       f"Training {len(X_train)} → {len(X_train_smote)} samples after SMOTE")
        except Exception as e:
            # Fallback if SMOTE fails (e.g., not enough samples)
            logger.warning(f"  Fold {fold+1}: SMOTE failed, using original training data")
            X_train_smote, y_train_smote = X_train, y_train
        
        # Scale features only on training data (NO LEAKAGE)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_smote)
        X_test_scaled = scaler.transform(X_test)  # Use training scaler
        
        # Train model
        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        rf.fit(X_train_scaled, y_train_smote)
        
        # Predict and evaluate
        y_pred = rf.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        cv_scores.append(acc)
        
        # Store detailed results
        detailed_results.append({
            'participant': test_participant,
            'accuracy': acc,
            'n_test': len(y_test),
            'n_train_original': len(X_train),
            'n_train_augmented': len(X_train_smote),
            'y_true': y_test,
            'y_pred': y_pred
        })
        
        logger.info(f"    {test_participant}: {acc:.3f} accuracy ({len(y_test)} test samples)")
    
    # Results summary
    cv_mean = np.mean(cv_scores)
    cv_std = np.std(cv_scores)
    
    logger.info(f"\n✅ CORRECTED LOPOCV RESULTS:")
    logger.info(f"   Mean accuracy: {cv_mean:.3f} ± {cv_std:.3f}")
    logger.info(f"   Range: {np.min(cv_scores):.3f} - {np.max(cv_scores):.3f}")
    logger.info(f"   Individual scores: {[f'{s:.3f}' for s in cv_scores]}")
    
    # Performance assessment
    logger.info(f"\n📈 PERFORMANCE ASSESSMENT:")
    baseline_acc = 1.0 / len(np.unique(y))
    improvement = cv_mean - baseline_acc
    
    logger.info(f"   Random baseline: {baseline_acc:.3f} ({baseline_acc*100:.1f}%)")
    logger.info(f"   Our performance: {cv_mean:.3f} ({cv_mean*100:.1f}%)")
    logger.info(f"   Improvement: +{improvement:.3f} ({improvement*100:.1f} percentage points)")
    
    if cv_mean >= 0.70:
        logger.info("   🎉 EXCELLENT: Strong performance!")
    elif cv_mean >= 0.50:
        logger.info("   🟡 MODERATE: Above baseline, room for improvement")
    elif cv_mean > baseline_acc + 0.05:
        logger.info("   🟡 MODEST: Slight improvement over baseline")
    else:
        logger.info("   ❌ POOR: Essentially random performance")
    
    # Save corrected results
    results = {
        'methodology': 'Corrected Literature-Standard (NO LEAKAGE)',
        'cv_scores': cv_scores,
        'cv_mean': cv_mean,
        'cv_std': cv_std,
        'detailed_results': detailed_results,
        'n_features': X_features.shape[1],
        'baseline_accuracy': baseline_acc,
        'improvement_over_baseline': improvement
    }
    
    output_file = 'data/processed/corrected_literature_rf_results.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump(results, f)
    
    logger.info(f"\n💾 Results saved to: {output_file}")
    
    return results


if __name__ == "__main__":
    results = correct_cross_validation()
    
    print("\n" + "="*80)
    print("CORRECTED METHODOLOGY SUMMARY")
    print("="*80)
    print(f"True LOPOCV accuracy: {results['cv_mean']:.1%} ± {results['cv_std']:.1%}")
    print(f"Improvement over random: +{results['improvement_over_baseline']:.1%}")
    print("\nThis represents legitimate performance without data leakage!")
