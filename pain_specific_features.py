#!/usr/bin/env python3
"""
Pain-Specific EEG Feature Extraction
===================================

Based on established pain neuroscience literature:
1. Delta power increase during pain (1-4 Hz)
2. Gamma power increase during pain (30-100 Hz) 
3. Alpha/beta suppression during pain (8-30 Hz)
4. Pain-relevant channels: C4 (contralateral S1), Cz, FCz, CPz

Key References:
- Ploner et al. (2017) Brain rhythms of pain
- Hu et al. (2013) Gamma oscillations and pain
- May et al. (2012) Structural brain imaging of chronic pain
"""

import numpy as np
import mne
from typing import Dict, List, Tuple, Optional
import logging
from scipy import signal
from scipy.stats import pearsonr

logger = logging.getLogger(__name__)

class PainSpecificFeatureExtractor:
    """Extract pain-specific neurophysiological features from EEG"""
    
    def __init__(self):
        """Initialize with pain-specific frequency bands and channels"""
        
        # Pain-specific frequency bands (literature-based)
        self.pain_bands = {
            'delta': (1, 4),      # Increases during pain
            'theta': (4, 8),      # Mixed findings, include for completeness
            'alpha': (8, 13),     # Suppressed during pain
            'beta': (13, 30),     # Suppressed during pain  
            'gamma': (30, 50),    # Increases during pain
            'high_gamma': (50, 90) # Strong pain marker
        }
        
        # Pain-relevant channels (somatosensory and cingulate projections)
        self.pain_channels = {
            'primary_somatosensory': ['C3', 'C4', 'CP3', 'CP4'],
            'cingulate_vertex': ['Cz', 'FCz', 'CPz'],
            'contralateral_s1': ['C4'],  # Right hand stimulation → left S1 activation
            'bilateral_motor': ['C3', 'C4'],
            'pain_core': ['C4', 'Cz', 'FCz', 'CPz']  # Most pain-predictive
        }
        
        # Pain-specific feature combinations
        self.pain_ratios = {
            'pain_suppression': ('alpha', 'beta'),     # Alpha+Beta suppression
            'pain_activation': ('delta', 'gamma'),     # Delta+Gamma activation
            'arousal_index': ('gamma', 'alpha'),       # Gamma/Alpha ratio
            'pain_index': ('delta', 'alpha')           # Delta/Alpha ratio
        }
    
    def extract_pain_features(self, 
                            epoch_data: np.ndarray,
                            channel_names: List[str], 
                            sfreq: float) -> Dict[str, float]:
        """
        Extract comprehensive pain-specific features
        
        Parameters:
        -----------
        epoch_data : np.ndarray, shape (n_channels, n_samples)
            Single epoch EEG data
        channel_names : List[str]
            Channel names corresponding to epoch_data
        sfreq : float
            Sampling frequency
            
        Returns:
        --------
        Dict[str, float]
            Pain-specific features
        """
        features = {}
        
        # 1. Pain-specific power bands
        band_powers = self._compute_pain_band_powers(epoch_data, channel_names, sfreq)
        features.update(band_powers)
        
        # 2. Pain-specific ratios
        ratio_features = self._compute_pain_ratios(band_powers)
        features.update(ratio_features)
        
        # 3. Spatial pain patterns
        spatial_features = self._compute_spatial_pain_patterns(epoch_data, channel_names, sfreq)
        features.update(spatial_features)
        
        # 4. Temporal pain dynamics
        temporal_features = self._compute_temporal_pain_features(epoch_data, channel_names, sfreq)
        features.update(temporal_features)
        
        # 5. Pain-specific connectivity
        connectivity_features = self._compute_pain_connectivity(epoch_data, channel_names, sfreq)
        features.update(connectivity_features)
        
        return features
    
    def _compute_pain_band_powers(self, 
                                epoch_data: np.ndarray,
                                channel_names: List[str], 
                                sfreq: float) -> Dict[str, float]:
        """Compute power in pain-specific frequency bands"""
        features = {}
        
        # Get pain-relevant channel indices
        pain_channel_indices = self._get_channel_indices(channel_names, self.pain_channels['pain_core'])
        
        if not pain_channel_indices:
            logger.warning("No pain-relevant channels found")
            return features
        
        # Compute PSD for pain channels only
        pain_data = epoch_data[pain_channel_indices, :]
        
        try:
            # Use Welch method for robust PSD estimation
            freqs, psd = signal.welch(pain_data, fs=sfreq, nperseg=min(256, pain_data.shape[1]//4))
            
            # Extract power in each pain-specific band
            for band_name, (fmin, fmax) in self.pain_bands.items():
                freq_mask = (freqs >= fmin) & (freqs <= fmax)
                if not np.any(freq_mask):
                    continue
                    
                # Average power across channels and frequencies
                band_power = np.mean(psd[:, freq_mask])
                features[f'pain_{band_name}_power'] = float(band_power)
                
                # Individual channel powers for key channels
                for i, ch_idx in enumerate(pain_channel_indices):
                    ch_name = channel_names[ch_idx]
                    if ch_name in ['C4', 'Cz']:  # Most important channels
                        ch_power = np.mean(psd[i, freq_mask])
                        features[f'pain_{band_name}_{ch_name}'] = float(ch_power)
        
        except Exception as e:
            logger.error(f"Error computing pain band powers: {e}")
        
        return features
    
    def _compute_pain_ratios(self, band_powers: Dict[str, float]) -> Dict[str, float]:
        """Compute pain-specific power ratios"""
        features = {}
        
        try:
            # Pain suppression index (lower alpha/beta during pain)
            alpha_power = band_powers.get('pain_alpha_power', 1e-10)
            beta_power = band_powers.get('pain_beta_power', 1e-10)
            suppression_power = alpha_power + beta_power
            features['pain_suppression_index'] = float(suppression_power)
            
            # Pain activation index (higher delta/gamma during pain)
            delta_power = band_powers.get('pain_delta_power', 1e-10)
            gamma_power = band_powers.get('pain_gamma_power', 1e-10)
            activation_power = delta_power + gamma_power
            features['pain_activation_index'] = float(activation_power)
            
            # Pain ratio (activation / suppression)
            if suppression_power > 1e-10:
                features['pain_ratio'] = float(activation_power / suppression_power)
            
            # Arousal index (gamma / alpha)
            if alpha_power > 1e-10:
                features['arousal_index'] = float(gamma_power / alpha_power)
            
            # Delta/alpha ratio (established pain marker)
            if alpha_power > 1e-10:
                features['delta_alpha_ratio'] = float(delta_power / alpha_power)
            
            # C4-specific ratios (contralateral somatosensory)
            c4_delta = band_powers.get('pain_delta_C4', 1e-10)
            c4_alpha = band_powers.get('pain_alpha_C4', 1e-10)
            c4_gamma = band_powers.get('pain_gamma_C4', 1e-10)
            
            if c4_alpha > 1e-10:
                features['c4_pain_index'] = float((c4_delta + c4_gamma) / c4_alpha)
            
        except Exception as e:
            logger.error(f"Error computing pain ratios: {e}")
        
        return features
    
    def _compute_spatial_pain_patterns(self, 
                                     epoch_data: np.ndarray,
                                     channel_names: List[str], 
                                     sfreq: float) -> Dict[str, float]:
        """Compute spatial patterns specific to pain processing"""
        features = {}
        
        try:
            # Contralateral vs ipsilateral asymmetry (pain typically contralateral)
            c3_idx = self._get_channel_indices(channel_names, ['C3'])
            c4_idx = self._get_channel_indices(channel_names, ['C4'])
            
            if c3_idx and c4_idx:
                # Compute gamma power in C3 vs C4
                freqs, psd = signal.welch(epoch_data[[c3_idx[0], c4_idx[0]], :], 
                                        fs=sfreq, nperseg=min(256, epoch_data.shape[1]//4))
                
                gamma_mask = (freqs >= 30) & (freqs <= 50)
                if np.any(gamma_mask):
                    c3_gamma = np.mean(psd[0, gamma_mask])
                    c4_gamma = np.mean(psd[1, gamma_mask])
                    
                    # C4 should be higher for right hand pain
                    features['somatosensory_asymmetry'] = float(c4_gamma - c3_gamma)
                    features['somatosensory_ratio'] = float(c4_gamma / (c3_gamma + 1e-10))
            
            # Vertex activity (cingulate cortex pain processing)
            vertex_indices = self._get_channel_indices(channel_names, ['Cz', 'FCz', 'CPz'])
            if vertex_indices:
                vertex_data = epoch_data[vertex_indices, :]
                freqs, psd = signal.welch(vertex_data, fs=sfreq, nperseg=min(256, vertex_data.shape[1]//4))
                
                # Delta power in vertex (pain attention/salience)
                delta_mask = (freqs >= 1) & (freqs <= 4)
                if np.any(delta_mask):
                    vertex_delta = np.mean(psd[:, delta_mask])
                    features['vertex_delta_power'] = float(vertex_delta)
        
        except Exception as e:
            logger.error(f"Error computing spatial pain patterns: {e}")
        
        return features
    
    def _compute_temporal_pain_features(self, 
                                      epoch_data: np.ndarray,
                                      channel_names: List[str], 
                                      sfreq: float) -> Dict[str, float]:
        """Compute temporal dynamics of pain-related activity"""
        features = {}
        
        try:
            # Pain channels
            pain_indices = self._get_channel_indices(channel_names, self.pain_channels['pain_core'])
            if not pain_indices:
                return features
            
            pain_data = epoch_data[pain_indices, :]
            
            # Early vs late pain response (pain builds over time)
            n_samples = pain_data.shape[1]
            early_data = pain_data[:, :n_samples//2]
            late_data = pain_data[:, n_samples//2:]
            
            # Gamma power early vs late
            freqs_early, psd_early = signal.welch(early_data, fs=sfreq, nperseg=min(128, early_data.shape[1]//2))
            freqs_late, psd_late = signal.welch(late_data, fs=sfreq, nperseg=min(128, late_data.shape[1]//2))
            
            gamma_mask = (freqs_early >= 30) & (freqs_early <= 50)
            if np.any(gamma_mask):
                early_gamma = np.mean(psd_early[:, gamma_mask])
                late_gamma = np.mean(psd_late[:, gamma_mask])
                
                features['gamma_temporal_increase'] = float(late_gamma - early_gamma)
                features['gamma_temporal_ratio'] = float(late_gamma / (early_gamma + 1e-10))
            
            # Alpha suppression dynamics
            alpha_mask = (freqs_early >= 8) & (freqs_early <= 13)
            if np.any(alpha_mask):
                early_alpha = np.mean(psd_early[:, alpha_mask])
                late_alpha = np.mean(psd_late[:, alpha_mask])
                
                features['alpha_temporal_suppression'] = float(early_alpha - late_alpha)
        
        except Exception as e:
            logger.error(f"Error computing temporal pain features: {e}")
        
        return features
    
    def _compute_pain_connectivity(self, 
                                 epoch_data: np.ndarray,
                                 channel_names: List[str], 
                                 sfreq: float) -> Dict[str, float]:
        """Compute pain-specific connectivity measures"""
        features = {}
        
        try:
            # Connectivity between key pain channels
            c4_idx = self._get_channel_indices(channel_names, ['C4'])
            cz_idx = self._get_channel_indices(channel_names, ['Cz'])
            
            if c4_idx and cz_idx:
                c4_data = epoch_data[c4_idx[0], :]
                cz_data = epoch_data[cz_idx[0], :]
                
                # Gamma-band coherence (pain network connectivity)
                f, coh = signal.coherence(c4_data, cz_data, fs=sfreq, nperseg=min(256, len(c4_data)//4))
                gamma_mask = (f >= 30) & (f <= 50)
                if np.any(gamma_mask):
                    gamma_coherence = np.mean(coh[gamma_mask])
                    features['c4_cz_gamma_coherence'] = float(gamma_coherence)
                
                # Cross-correlation (temporal connectivity)
                max_lag = min(50, len(c4_data)//10)  # Max 50 samples lag
                correlation = np.correlate(c4_data, cz_data, mode='full')
                center = len(correlation) // 2
                lag_range = correlation[center-max_lag:center+max_lag+1]
                max_corr = np.max(np.abs(lag_range))
                features['c4_cz_max_correlation'] = float(max_corr)
        
        except Exception as e:
            logger.error(f"Error computing pain connectivity: {e}")
        
        return features
    
    def _get_channel_indices(self, channel_names: List[str], target_channels: List[str]) -> List[int]:
        """Get indices of target channels in the channel list"""
        indices = []
        for target in target_channels:
            try:
                idx = channel_names.index(target)
                indices.append(idx)
            except ValueError:
                continue
        return indices
    
    def extract_features_batch(self, 
                             epochs_data: np.ndarray,
                             channel_names: List[str], 
                             sfreq: float) -> np.ndarray:
        """Extract features for a batch of epochs"""
        n_epochs = epochs_data.shape[0]
        feature_list = []
        
        for i in range(n_epochs):
            epoch_features = self.extract_pain_features(epochs_data[i], channel_names, sfreq)
            feature_list.append(epoch_features)
        
        # Convert to consistent feature matrix
        if feature_list:
            # Get all unique feature names
            all_features = set()
            for features in feature_list:
                all_features.update(features.keys())
            
            feature_names = sorted(list(all_features))
            
            # Create feature matrix
            feature_matrix = np.zeros((n_epochs, len(feature_names)))
            for i, features in enumerate(feature_list):
                for j, feature_name in enumerate(feature_names):
                    feature_matrix[i, j] = features.get(feature_name, 0.0)
            
            return feature_matrix, feature_names
        
        return np.array([]), []

def test_pain_features():
    """Test the pain-specific feature extractor"""
    print("Testing Pain-Specific Feature Extraction...")
    
    # Create synthetic EEG data
    n_channels, n_samples = 68, 2000  # 4s at 500Hz
    sfreq = 500
    
    # Simulate EEG channel names
    channel_names = [f'C{i}' for i in range(1, 69)]
    channel_names[3] = 'C4'  # Make sure we have C4
    channel_names[31] = 'Cz'  # And Cz
    
    # Create synthetic data with pain-like patterns
    np.random.seed(42)
    data = np.random.randn(n_channels, n_samples) * 1e-6
    
    # Add pain-like features to C4 and Cz
    t = np.linspace(0, 4, n_samples)
    
    # Increased delta (2 Hz) in pain channels
    data[3, :] += 2e-6 * np.sin(2 * 2 * np.pi * t)  # C4
    data[31, :] += 1.5e-6 * np.sin(2 * 2 * np.pi * t)  # Cz
    
    # Increased gamma (40 Hz) in pain channels  
    data[3, :] += 1e-6 * np.sin(2 * 40 * np.pi * t)  # C4
    data[31, :] += 0.8e-6 * np.sin(2 * 40 * np.pi * t)  # Cz
    
    # Suppressed alpha (10 Hz) in pain channels
    data[3, :] += 0.5e-6 * np.sin(2 * 10 * np.pi * t)  # Lower alpha
    
    # Test feature extraction
    extractor = PainSpecificFeatureExtractor()
    features = extractor.extract_pain_features(data, channel_names, sfreq)
    
    print(f"\nExtracted {len(features)} pain-specific features:")
    for feature_name, value in sorted(features.items()):
        print(f"  {feature_name}: {value:.6f}")
    
    # Test key pain markers
    print(f"\nKey Pain Markers:")
    print(f"  Pain Activation Index: {features.get('pain_activation_index', 0):.6f}")
    print(f"  Pain Suppression Index: {features.get('pain_suppression_index', 0):.6f}")
    print(f"  Pain Ratio: {features.get('pain_ratio', 0):.6f}")
    print(f"  Delta/Alpha Ratio: {features.get('delta_alpha_ratio', 0):.6f}")
    print(f"  C4 Pain Index: {features.get('c4_pain_index', 0):.6f}")
    
    print(f"\n✅ Pain-specific feature extraction test completed!")

if __name__ == "__main__":
    test_pain_features()
