#!/usr/bin/env python3
"""
Advanced EEG Pain Classifier with Comprehensive Feature Engineering
Implements wavelet transforms, connectivity measures, hyperparameter optimization,
and ensemble methods for maximum performance on 10-participant dataset.
"""

import os
import sys
import time
import pickle
import logging
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import pearsonr
import pywt
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

def setup_logging():
    """Setup logging for advanced classifier."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

class AdvancedFeatureExtractor:
    """Advanced feature extraction with wavelets, connectivity, and ensemble methods."""
    
    def __init__(self):
        self.pain_channels = ['Cz', 'CPz', 'C3', 'C4', 'Fz', 'Pz']
        self.freq_bands = {
            'delta': (1, 4),
            'theta': (4, 8), 
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 45)
        }
        
    def extract_wavelet_features(self, data, wavelet='db4', levels=5):
        """
        Extract Daubechies 4 wavelet features (literature standard).
        
        Args:
            data: EEG data (channels × samples)
            wavelet: Wavelet type (default: db4)
            levels: Decomposition levels
            
        Returns:
            Dictionary of wavelet features
        """
        features = {}
        
        for ch_idx, ch_name in enumerate(self.pain_channels):
            if ch_idx >= data.shape[0]:
                continue
                
            signal_data = data[ch_idx, :]
            
            # Wavelet decomposition
            coeffs = pywt.wavedec(signal_data, wavelet, level=levels)
            
            # Extract statistical features from each level
            for level, coeff in enumerate(coeffs):
                prefix = f"{ch_name}_wavelet_L{level}"
                
                # Statistical measures (literature standard)
                features[f"{prefix}_mean"] = np.mean(coeff)
                features[f"{prefix}_std"] = np.std(coeff)
                features[f"{prefix}_var"] = np.var(coeff)
                features[f"{prefix}_rms"] = np.sqrt(np.mean(coeff**2))
                features[f"{prefix}_energy"] = np.sum(coeff**2)
                features[f"{prefix}_entropy"] = self._calculate_entropy(coeff)
                
                # Percentiles
                features[f"{prefix}_p25"] = np.percentile(coeff, 25)
                features[f"{prefix}_p50"] = np.percentile(coeff, 50)
                features[f"{prefix}_p75"] = np.percentile(coeff, 75)
                
                # Zero-crossing rate
                features[f"{prefix}_zcr"] = self._zero_crossing_rate(coeff)
        
        return features
    
    def extract_connectivity_features(self, data, sfreq=500):
        """
        Extract connectivity measures between pain-relevant channels.
        
        Args:
            data: EEG data (channels × samples)
            sfreq: Sampling frequency
            
        Returns:
            Dictionary of connectivity features
        """
        features = {}
        
        # Get channel indices
        channel_indices = []
        for ch_name in self.pain_channels:
            # Assuming standard 10-20 layout mapping
            ch_map = {'Cz': 31, 'CPz': 47, 'C3': 23, 'C4': 39, 'Fz': 1, 'Pz': 61}
            if ch_name in ch_map and ch_map[ch_name] < data.shape[0]:
                channel_indices.append(ch_map[ch_name])
        
        if len(channel_indices) < 2:
            return features
        
        # Extract data for pain channels
        pain_data = data[channel_indices, :]
        
        # 1. Coherence-based connectivity
        for i, ch1 in enumerate(self.pain_channels[:len(channel_indices)]):
            for j, ch2 in enumerate(self.pain_channels[:len(channel_indices)]):
                if i < j:  # Avoid duplicate pairs
                    # Compute coherence across frequency bands
                    freqs, coherence = signal.coherence(
                        pain_data[i, :], pain_data[j, :], 
                        fs=sfreq, nperseg=sfreq//4
                    )
                    
                    for band_name, (low_freq, high_freq) in self.freq_bands.items():
                        band_mask = (freqs >= low_freq) & (freqs <= high_freq)
                        if np.any(band_mask):
                            coh_value = np.mean(coherence[band_mask])
                            features[f"coherence_{ch1}_{ch2}_{band_name}"] = coh_value
        
        # 2. Phase-locking value (PLV)
        for band_name, (low_freq, high_freq) in self.freq_bands.items():
            # Filter data for frequency band
            sos = signal.butter(4, [low_freq, high_freq], btype='band', fs=sfreq, output='sos')
            
            for i, ch1 in enumerate(self.pain_channels[:len(channel_indices)]):
                for j, ch2 in enumerate(self.pain_channels[:len(channel_indices)]):
                    if i < j:
                        # Filter signals
                        sig1_filt = signal.sosfilt(sos, pain_data[i, :])
                        sig2_filt = signal.sosfilt(sos, pain_data[j, :])
                        
                        # Compute instantaneous phases
                        phase1 = np.angle(signal.hilbert(sig1_filt))
                        phase2 = np.angle(signal.hilbert(sig2_filt))
                        
                        # Phase-locking value
                        phase_diff = phase1 - phase2
                        plv = np.abs(np.mean(np.exp(1j * phase_diff)))
                        features[f"plv_{ch1}_{ch2}_{band_name}"] = plv
        
        # 3. Cross-correlation features
        for i, ch1 in enumerate(self.pain_channels[:len(channel_indices)]):
            for j, ch2 in enumerate(self.pain_channels[:len(channel_indices)]):
                if i < j:
                    # Pearson correlation
                    corr, _ = pearsonr(pain_data[i, :], pain_data[j, :])
                    features[f"correlation_{ch1}_{ch2}"] = corr
        
        return features
    
    def extract_spectral_features(self, data, sfreq=500):
        """Extract comprehensive spectral features."""
        features = {}
        
        for ch_idx, ch_name in enumerate(self.pain_channels):
            if ch_idx >= data.shape[0]:
                continue
                
            # Get channel data (map to actual EEG channels)
            ch_map = {'Cz': 31, 'CPz': 47, 'C3': 23, 'C4': 39, 'Fz': 1, 'Pz': 61}
            actual_ch_idx = ch_map.get(ch_name, ch_idx)
            
            if actual_ch_idx >= data.shape[0]:
                continue
                
            signal_data = data[actual_ch_idx, :]
            
            # Power spectral density
            freqs, psd = signal.welch(signal_data, fs=sfreq, nperseg=sfreq//2)
            
            # Extract power in frequency bands
            total_power = np.sum(psd)
            for band_name, (low_freq, high_freq) in self.freq_bands.items():
                band_mask = (freqs >= low_freq) & (freqs <= high_freq)
                band_power = np.sum(psd[band_mask])
                
                # Absolute and relative power
                features[f"{ch_name}_{band_name}_power"] = np.log(band_power + 1e-10)
                features[f"{ch_name}_{band_name}_rel_power"] = band_power / (total_power + 1e-10)
            
            # Spectral ratios (literature standard)
            delta_power = np.sum(psd[(freqs >= 1) & (freqs <= 4)])
            theta_power = np.sum(psd[(freqs >= 4) & (freqs <= 8)])
            alpha_power = np.sum(psd[(freqs >= 8) & (freqs <= 13)])
            beta_power = np.sum(psd[(freqs >= 13) & (freqs <= 30)])
            
            features[f"{ch_name}_delta_alpha_ratio"] = delta_power / (alpha_power + 1e-10)
            features[f"{ch_name}_theta_beta_ratio"] = theta_power / (beta_power + 1e-10)
            features[f"{ch_name}_low_high_ratio"] = (delta_power + theta_power) / (alpha_power + beta_power + 1e-10)
        
        return features
    
    def extract_temporal_features(self, data):
        """Extract time-domain features."""
        features = {}
        
        for ch_idx, ch_name in enumerate(self.pain_channels):
            # Map to actual channels
            ch_map = {'Cz': 31, 'CPz': 47, 'C3': 23, 'C4': 39, 'Fz': 1, 'Pz': 61}
            actual_ch_idx = ch_map.get(ch_name, ch_idx)
            
            if actual_ch_idx >= data.shape[0]:
                continue
                
            signal_data = data[actual_ch_idx, :]
            
            # Statistical measures
            features[f"{ch_name}_mean"] = np.mean(signal_data)
            features[f"{ch_name}_std"] = np.std(signal_data)
            features[f"{ch_name}_var"] = np.var(signal_data)
            features[f"{ch_name}_rms"] = np.sqrt(np.mean(signal_data**2))
            features[f"{ch_name}_skewness"] = self._calculate_skewness(signal_data)
            features[f"{ch_name}_kurtosis"] = self._calculate_kurtosis(signal_data)
            features[f"{ch_name}_zcr"] = self._zero_crossing_rate(signal_data)
        
        return features
    
    def extract_all_features(self, data, sfreq=500):
        """Extract all feature types and combine them."""
        all_features = {}
        
        logging.info("Extracting spectral features...")
        spectral_features = self.extract_spectral_features(data, sfreq)
        all_features.update(spectral_features)
        
        logging.info("Extracting wavelet features...")
        wavelet_features = self.extract_wavelet_features(data)
        all_features.update(wavelet_features)
        
        logging.info("Extracting connectivity features...")
        connectivity_features = self.extract_connectivity_features(data, sfreq)
        all_features.update(connectivity_features)
        
        logging.info("Extracting temporal features...")
        temporal_features = self.extract_temporal_features(data)
        all_features.update(temporal_features)
        
        return all_features
    
    def _calculate_entropy(self, signal_data):
        """Calculate Shannon entropy."""
        # Discretize signal
        hist, _ = np.histogram(signal_data, bins=50, density=True)
        hist = hist[hist > 0]  # Remove zero bins
        return -np.sum(hist * np.log2(hist))
    
    def _zero_crossing_rate(self, signal_data):
        """Calculate zero-crossing rate."""
        return np.sum(np.diff(np.sign(signal_data)) != 0) / len(signal_data)
    
    def _calculate_skewness(self, signal_data):
        """Calculate skewness."""
        mean = np.mean(signal_data)
        std = np.std(signal_data)
        return np.mean(((signal_data - mean) / std) ** 3)
    
    def _calculate_kurtosis(self, signal_data):
        """Calculate kurtosis."""
        mean = np.mean(signal_data)
        std = np.std(signal_data)
        return np.mean(((signal_data - mean) / std) ** 4) - 3

class AdvancedClassifierEnsemble:
    """Ensemble classifier with hyperparameter optimization."""
    
    def __init__(self):
        self.models = {}
        self.best_params = {}
        self.feature_scaler = StandardScaler()
        
    def create_models(self):
        """Create model dictionary with hyperparameter grids."""
        self.models = {
            'random_forest': {
                'model': RandomForestClassifier(random_state=42),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2']
                }
            },
            'xgboost': {
                'model': xgb.XGBClassifier(random_state=42),
                'params': {
                    'n_estimators': [100, 200],
                    'max_depth': [3, 6, 9],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'subsample': [0.8, 0.9, 1.0],
                    'colsample_bytree': [0.8, 0.9, 1.0]
                }
            },
            'svm': {
                'model': SVC(random_state=42, probability=True),
                'params': {
                    'C': [0.1, 1, 10, 100],
                    'gamma': ['scale', 'auto', 0.001, 0.01],
                    'kernel': ['rbf', 'polynomial']
                }
            },
            'logistic': {
                'model': LogisticRegression(random_state=42, max_iter=1000),
                'params': {
                    'C': [0.01, 0.1, 1, 10, 100],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear', 'saga']
                }
            }
        }
    
    def optimize_hyperparameters(self, X, y, cv=5):
        """Perform grid search hyperparameter optimization."""
        self.create_models()
        
        logging.info("Starting hyperparameter optimization...")
        
        for name, model_info in self.models.items():
            logging.info(f"Optimizing {name}...")
            
            # Grid search with cross-validation
            grid_search = GridSearchCV(
                model_info['model'],
                model_info['params'],
                cv=cv,
                scoring='accuracy',
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X, y)
            
            # Store best parameters and model
            self.best_params[name] = grid_search.best_params_
            self.models[name]['best_model'] = grid_search.best_estimator_
            
            logging.info(f"{name} best score: {grid_search.best_score_:.4f}")
            logging.info(f"{name} best params: {grid_search.best_params_}")
    
    def create_ensemble(self):
        """Create voting ensemble from optimized models."""
        if not self.best_params:
            raise ValueError("Must run hyperparameter optimization first!")
        
        # Create ensemble with best models
        estimators = []
        for name, model_info in self.models.items():
            if 'best_model' in model_info:
                estimators.append((name, model_info['best_model']))
        
        # Voting classifier (soft voting for probability-based decisions)
        ensemble = VotingClassifier(
            estimators=estimators,
            voting='soft'
        )
        
        return ensemble

def load_processed_data(data_dir="data/processed/basic_windows"):
    """Load processed data from existing 5 participants (can extend to more)."""
    data_path = Path(data_dir)
    
    all_data = []
    all_labels = []
    participants = []
    
    # Load data from each participant
    for pkl_file in data_path.glob("vp*_windows.pkl"):
        participant = pkl_file.stem.replace("_windows", "")
        
        try:
            with open(pkl_file, 'rb') as f:
                windows_data = pickle.load(f)
            
            windows = windows_data['windows']
            labels = windows_data['ternary_labels']  # Use correct key name
            
            all_data.extend(windows)
            all_labels.extend(labels)
            participants.extend([participant] * len(windows))
            
            logging.info(f"Loaded {participant}: {len(windows)} windows")
            
        except Exception as e:
            logging.error(f"Failed to load {participant}: {e}")
    
    return np.array(all_data), np.array(all_labels), np.array(participants)

def main():
    """Main advanced classifier pipeline."""
    setup_logging()
    
    print("="*80)
    print("ADVANCED EEG PAIN CLASSIFIER")
    print("Wavelets + Connectivity + Hyperparameter Optimization + Ensemble")
    print("="*80)
    
    # Check if 5-participant data exists (using existing data)
    data_dir = Path("data/processed/basic_windows")
    if not data_dir.exists():
        logging.error("5-participant data not found. Please run basic preprocessing first.")
        return
    
    # Load data
    logging.info("Loading 5-participant dataset (basic_windows)...")
    X_raw, y, participants = load_processed_data(data_dir)
    
    if len(X_raw) == 0:
        logging.error("No data loaded. Please check data directory.")
        return
    
    logging.info(f"Loaded {len(X_raw)} windows from {len(np.unique(participants))} participants")
    
    # Initialize feature extractor
    feature_extractor = AdvancedFeatureExtractor()
    
    # Extract features for all windows
    logging.info("Extracting advanced features (this may take several minutes)...")
    start_time = time.time()
    
    all_features = []
    for i, window in enumerate(X_raw):
        if i % 50 == 0:
            logging.info(f"Processing window {i+1}/{len(X_raw)}")
        
        features = feature_extractor.extract_all_features(window)
        all_features.append(features)
    
    # Convert to DataFrame
    feature_df = pd.DataFrame(all_features)
    feature_df = feature_df.fillna(0)  # Handle any NaN values
    
    extraction_time = time.time() - start_time
    logging.info(f"Feature extraction completed in {extraction_time/60:.1f} minutes")
    logging.info(f"Extracted {feature_df.shape[1]} features")
    
    # Prepare data
    X = feature_df.values
    
    # Binary classification (Low vs High pain)
    # Convert ternary labels to binary
    binary_mask = (y == 0) | (y == 2)  # Low (0) and High (2) pain only
    X_binary = X[binary_mask]
    y_binary = y[binary_mask]
    y_binary = (y_binary == 2).astype(int)  # 0=Low, 1=High
    participants_binary = participants[binary_mask]
    
    logging.info(f"Binary classification: {len(X_binary)} samples")
    logging.info(f"Class distribution: {np.bincount(y_binary)}")
    
    # Scale features
    scaler = StandardScaler()
    X_binary_scaled = scaler.fit_transform(X_binary)
    
    # Initialize ensemble classifier
    ensemble_classifier = AdvancedClassifierEnsemble()
    
    # Hyperparameter optimization
    logging.info("Starting hyperparameter optimization (this will take time)...")
    ensemble_classifier.optimize_hyperparameters(X_binary_scaled, y_binary)
    
    # Create and train ensemble
    logging.info("Creating optimized ensemble...")
    ensemble = ensemble_classifier.create_ensemble()
    
    # Cross-validation evaluation
    logging.info("Evaluating ensemble with cross-validation...")
    cv_scores = cross_val_score(ensemble, X_binary_scaled, y_binary, cv=5, scoring='accuracy')
    
    print(f"\n{'='*60}")
    print("ADVANCED CLASSIFIER RESULTS")
    print(f"{'='*60}")
    print(f"Features extracted: {X_binary.shape[1]}")
    print(f"Samples: {len(X_binary)} (Binary: Low vs High pain)")
    print(f"Cross-validation accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    print(f"Individual CV scores: {[f'{s:.3f}' for s in cv_scores]}")
    
    # Train final model and show feature importance
    ensemble.fit(X_binary_scaled, y_binary)
    
    # Save results
    results_dir = Path("data/processed/advanced_classifier_results")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model and results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    with open(results_dir / f"advanced_ensemble_{timestamp}.pkl", 'wb') as f:
        pickle.dump({
            'ensemble': ensemble,
            'scaler': scaler,
            'feature_names': feature_df.columns.tolist(),
            'cv_scores': cv_scores,
            'best_params': ensemble_classifier.best_params
        }, f)
    
    # Save feature matrix
    feature_df_binary = feature_df.iloc[binary_mask].copy()
    feature_df_binary['participant'] = participants_binary
    feature_df_binary['label'] = y_binary
    feature_df_binary.to_csv(results_dir / f"advanced_features_{timestamp}.csv", index=False)
    
    logging.info(f"Results saved to {results_dir}")
    logging.info("Advanced classifier pipeline completed!")
    
    return ensemble, cv_scores

if __name__ == "__main__":
    ensemble, scores = main()
