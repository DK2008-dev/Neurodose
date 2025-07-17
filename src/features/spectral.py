"""
Feature extraction for EEG pain classification.

This module implements spectral feature extraction focusing on pain-relevant
frequency bands and electrode locations.
"""

import numpy as np
import mne
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class SpectralFeatureExtractor:
    """Extract spectral features from EEG epochs."""
    
    def __init__(self):
        """Initialize the feature extractor with standard frequency bands."""
        # Standard EEG frequency bands
        self.freq_bands = {
            'delta': (1, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 45)
        }
        
        # Pain-relevant electrode groups
        self.electrode_groups = {
            'central': ['C3', 'C4', 'CP3', 'CP4'],
            'vertex': ['Cz', 'FCz', 'CPz'],
            'frontocentral': ['Fz', 'FC1', 'FC2', 'AFz'],
            'pain_specific': ['C4', 'Cz', 'FCz', 'CPz']
        }
    
    def compute_psd(self, 
                   epoch_data: np.ndarray, 
                   sfreq: float,
                   fmin: float = 1.0,
                   fmax: float = 45.0,
                   n_fft: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute power spectral density for an epoch.
        
        Parameters:
        -----------
        epoch_data : np.ndarray
            EEG data (n_channels, n_samples)
        sfreq : float
            Sampling frequency
        fmin : float
            Minimum frequency
        fmax : float
            Maximum frequency
        n_fft : int, optional
            FFT length
            
        Returns:
        --------
        psds : np.ndarray
            Power spectral densities (n_channels, n_freqs)
        freqs : np.ndarray
            Frequency values
        """
        # Create a temporary Raw object for MNE functions
        info = mne.create_info(
            ch_names=[f'Ch{i}' for i in range(epoch_data.shape[0])],
            sfreq=sfreq,
            ch_types='eeg'
        )
        
        # Reshape data for MNE (add epoch dimension)
        data_reshaped = epoch_data[np.newaxis, :, :]
        
        # Compute PSD using Welch's method
        if n_fft is None:
            n_fft = min(256, epoch_data.shape[1] // 4)  # Default FFT length
        
        psds, freqs = mne.time_frequency.psd_array_welch(
            data_reshaped, sfreq, fmin=fmin, fmax=fmax, 
            n_fft=n_fft, verbose=False
        )
        
        return psds[0], freqs  # Remove epoch dimension
    
    def extract_band_powers(self, 
                           psds: np.ndarray, 
                           freqs: np.ndarray,
                           ch_names: List[str]) -> Dict[str, np.ndarray]:
        """
        Extract absolute and relative band powers.
        
        Parameters:
        -----------
        psds : np.ndarray
            Power spectral densities (n_channels, n_freqs)
        freqs : np.ndarray
            Frequency values
        ch_names : list
            Channel names
            
        Returns:
        --------
        features : dict
            Dictionary with band power features
        """
        features = {}
        
        # Compute absolute band powers
        for band_name, (fmin, fmax) in self.freq_bands.items():
            freq_mask = (freqs >= fmin) & (freqs <= fmax)
            if np.any(freq_mask):
                band_power = np.mean(psds[:, freq_mask], axis=1)
                features[f'{band_name}_abs'] = band_power
        
        # Compute relative band powers
        total_power = np.sum(psds, axis=1)
        for band_name in self.freq_bands.keys():
            if f'{band_name}_abs' in features:
                rel_power = features[f'{band_name}_abs'] / (total_power + 1e-10)
                features[f'{band_name}_rel'] = rel_power
        
        # Compute band power ratios
        if 'gamma_abs' in features and 'alpha_abs' in features:
            features['gamma_alpha_ratio'] = (features['gamma_abs'] / 
                                           (features['alpha_abs'] + 1e-10))
        
        if 'beta_abs' in features and 'alpha_abs' in features:
            features['beta_alpha_ratio'] = (features['beta_abs'] / 
                                          (features['alpha_abs'] + 1e-10))
        
        if 'theta_abs' in features and 'beta_abs' in features:
            features['theta_beta_ratio'] = (features['theta_abs'] / 
                                          (features['beta_abs'] + 1e-10))
        
        return features
    
    def extract_roi_features(self, 
                           features: Dict[str, np.ndarray],
                           ch_names: List[str]) -> Dict[str, float]:
        """
        Extract ROI-averaged features.
        
        Parameters:
        -----------
        features : dict
            Channel-wise features
        ch_names : list
            Channel names
            
        Returns:
        --------
        roi_features : dict
            ROI-averaged features
        """
        roi_features = {}
        
        # Create channel name to index mapping
        ch_idx_map = {ch: idx for idx, ch in enumerate(ch_names)}
        
        for roi_name, roi_channels in self.electrode_groups.items():
            # Find valid channels for this ROI
            valid_indices = []
            for ch in roi_channels:
                if ch in ch_idx_map:
                    valid_indices.append(ch_idx_map[ch])
            
            if valid_indices:
                # Average features across ROI channels
                for feature_name, feature_values in features.items():
                    if isinstance(feature_values, np.ndarray) and len(feature_values.shape) == 1:
                        roi_avg = np.mean(feature_values[valid_indices])
                        roi_features[f'{roi_name}_{feature_name}'] = roi_avg
        
        return roi_features
    
    def extract_temporal_features(self, 
                                epoch_data: np.ndarray,
                                sfreq: float,
                                early_window: Tuple[float, float] = (0.0, 1.0),
                                late_window: Tuple[float, float] = (1.0, 4.0)) -> Dict[str, np.ndarray]:
        """
        Extract features from early vs late time windows.
        
        Parameters:
        -----------
        epoch_data : np.ndarray
            EEG data (n_channels, n_samples)
        sfreq : float
            Sampling frequency
        early_window : tuple
            Early time window (start, end) in seconds
        late_window : tuple
            Late time window (start, end) in seconds
            
        Returns:
        --------
        temporal_features : dict
            Features from different temporal windows
        """
        temporal_features = {}
        n_samples = epoch_data.shape[1]
        
        # Convert time windows to sample indices
        early_start = int(early_window[0] * sfreq)
        early_end = min(int(early_window[1] * sfreq), n_samples)
        late_start = int(late_window[0] * sfreq)
        late_end = min(int(late_window[1] * sfreq), n_samples)
        
        # Extract early window features
        if early_end > early_start:
            early_data = epoch_data[:, early_start:early_end]
            early_psds, freqs = self.compute_psd(early_data, sfreq)
            early_features = self.extract_band_powers(early_psds, freqs, [])
            
            for key, value in early_features.items():
                temporal_features[f'early_{key}'] = value
        
        # Extract late window features
        if late_end > late_start:
            late_data = epoch_data[:, late_start:late_end]
            late_psds, freqs = self.compute_psd(late_data, sfreq)
            late_features = self.extract_band_powers(late_psds, freqs, [])
            
            for key, value in late_features.items():
                temporal_features[f'late_{key}'] = value
        
        return temporal_features
    
    def extract_features_single_epoch(self, 
                                    epoch_data: np.ndarray,
                                    sfreq: float,
                                    ch_names: List[str]) -> np.ndarray:
        """
        Extract all features for a single epoch.
        
        Parameters:
        -----------
        epoch_data : np.ndarray
            EEG data (n_channels, n_samples)
        sfreq : float
            Sampling frequency
        ch_names : list
            Channel names
            
        Returns:
        --------
        feature_vector : np.ndarray
            Concatenated feature vector
        """
        all_features = []
        
        # 1. Basic spectral features
        psds, freqs = self.compute_psd(epoch_data, sfreq)
        spectral_features = self.extract_band_powers(psds, freqs, ch_names)
        
        # Flatten channel-wise features
        for feature_name, feature_values in spectral_features.items():
            if isinstance(feature_values, np.ndarray):
                all_features.extend(feature_values.flatten())
            else:
                all_features.append(feature_values)
        
        # 2. ROI features
        roi_features = self.extract_roi_features(spectral_features, ch_names)
        all_features.extend(roi_features.values())
        
        # 3. Temporal features (early vs late)
        temporal_features = self.extract_temporal_features(epoch_data, sfreq)
        
        # ROI features for temporal windows
        for temp_key, temp_values in temporal_features.items():
            if isinstance(temp_values, np.ndarray):
                temp_roi_features = self.extract_roi_features(
                    {temp_key: temp_values}, ch_names
                )
                all_features.extend(temp_roi_features.values())
        
        return np.array(all_features, dtype=np.float32)
    
    def extract_features_batch(self, 
                             X: np.ndarray,
                             sfreq: float,
                             ch_names: List[str]) -> np.ndarray:
        """
        Extract features for a batch of epochs.
        
        Parameters:
        -----------
        X : np.ndarray
            EEG data (n_epochs, n_channels, n_samples)
        sfreq : float
            Sampling frequency
        ch_names : list
            Channel names
            
        Returns:
        --------
        features : np.ndarray
            Feature matrix (n_epochs, n_features)
        """
        logger.info(f"Extracting features from {X.shape[0]} epochs...")
        
        feature_list = []
        for i, epoch_data in enumerate(X):
            if i % 100 == 0:
                logger.info(f"Processing epoch {i}/{X.shape[0]}")
            
            features = self.extract_features_single_epoch(epoch_data, sfreq, ch_names)
            feature_list.append(features)
        
        feature_matrix = np.vstack(feature_list)
        logger.info(f"Extracted features shape: {feature_matrix.shape}")
        
        return feature_matrix
    
    def get_feature_names(self, ch_names: List[str]) -> List[str]:
        """
        Get feature names for interpretation.
        
        Parameters:
        -----------
        ch_names : list
            Channel names
            
        Returns:
        --------
        feature_names : list
            List of feature names
        """
        feature_names = []
        
        # Channel-wise spectral features
        for ch in ch_names:
            for band in self.freq_bands.keys():
                feature_names.extend([f'{ch}_{band}_abs', f'{ch}_{band}_rel'])
            
            # Ratios
            feature_names.extend([
                f'{ch}_gamma_alpha_ratio',
                f'{ch}_beta_alpha_ratio', 
                f'{ch}_theta_beta_ratio'
            ])
        
        # ROI features
        for roi_name in self.electrode_groups.keys():
            for band in self.freq_bands.keys():
                feature_names.extend([f'{roi_name}_{band}_abs', f'{roi_name}_{band}_rel'])
            
            feature_names.extend([
                f'{roi_name}_gamma_alpha_ratio',
                f'{roi_name}_beta_alpha_ratio',
                f'{roi_name}_theta_beta_ratio'
            ])
        
        # Temporal features (simplified)
        for prefix in ['early', 'late']:
            for roi_name in self.electrode_groups.keys():
                for band in self.freq_bands.keys():
                    feature_names.extend([
                        f'{roi_name}_{prefix}_{band}_abs',
                        f'{roi_name}_{prefix}_{band}_rel'
                    ])
        
        return feature_names
