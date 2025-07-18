#!/usr/bin/env python3
"""
Prediction Script for EEG Pain Classification
Usage: python predict.py --input epoch.npy --model models/binary_model.pkl

Authors: Dhruv Kurup, Avid Patel
For: Journal of Emerging Investigators submission
"""

import argparse
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import sys

class FeatureExtractor:
    """Extract 78 neuroscience-aligned features for prediction."""
    
    def __init__(self, sfreq=500):
        self.sfreq = sfreq
        self.channels = ['Cz', 'FCz', 'C3', 'C4', 'Fz', 'Pz']
        self.freq_bands = {
            'delta': (1, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 45)
        }
    
    def extract_spectral_features(self, epoch_data, ch_names):
        """Extract spectral power features."""
        from scipy import signal
        
        features = {}
        
        # Get channel indices
        ch_indices = [ch_names.index(ch) for ch in self.channels if ch in ch_names]
        
        for ch_idx, ch_name in zip(ch_indices, [ch for ch in self.channels if ch in ch_names]):
            # Compute PSD
            freqs, psd = signal.welch(epoch_data[ch_idx], fs=self.sfreq, nperseg=min(256, len(epoch_data[ch_idx])))
            
            # Band power
            for band_name, (low, high) in self.freq_bands.items():
                band_mask = (freqs >= low) & (freqs <= high)
                band_power = np.log10(np.trapz(psd[band_mask], freqs[band_mask]) + 1e-12)
                features[f'{ch_name}_{band_name}_power'] = band_power
        
        return features
    
    def extract_ratio_features(self, spectral_features):
        """Extract frequency ratio features."""
        features = {}
        
        for ch in self.channels:
            # Check if all required bands exist
            required_bands = ['delta', 'alpha', 'gamma', 'beta', 'theta']
            if all(f'{ch}_{band}_power' in spectral_features for band in required_bands):
                # Delta/Alpha ratio
                features[f'{ch}_delta_alpha_ratio'] = (
                    spectral_features[f'{ch}_delta_power'] - spectral_features[f'{ch}_alpha_power']
                )
                
                # Gamma/Beta ratio
                features[f'{ch}_gamma_beta_ratio'] = (
                    spectral_features[f'{ch}_gamma_power'] - spectral_features[f'{ch}_beta_power']
                )
                
                # (Delta+Theta)/(Alpha+Beta) ratio
                numerator = spectral_features[f'{ch}_delta_power'] + spectral_features[f'{ch}_theta_power']
                denominator = spectral_features[f'{ch}_alpha_power'] + spectral_features[f'{ch}_beta_power']
                features[f'{ch}_low_high_ratio'] = numerator - denominator
        
        return features
    
    def extract_asymmetry_features(self, spectral_features):
        """Extract spatial asymmetry features."""
        features = {}
        
        # C4-C3 asymmetry for each band
        for band in self.freq_bands.keys():
            c4_key = f'C4_{band}_power'
            c3_key = f'C3_{band}_power'
            if c4_key in spectral_features and c3_key in spectral_features:
                features[f'C4_C3_{band}_asymmetry'] = spectral_features[c4_key] - spectral_features[c3_key]
        
        return features
    
    def extract_erp_features(self, epoch_data, ch_names):
        """Extract ERP component features."""
        features = {}
        
        # Time windows (in samples at 500Hz)
        n2_start, n2_end = int(0.15 * self.sfreq), int(0.25 * self.sfreq)  # 150-250ms
        p2_start, p2_end = int(0.20 * self.sfreq), int(0.35 * self.sfreq)  # 200-350ms
        
        for ch in ['Cz', 'FCz']:
            if ch in ch_names:
                ch_idx = ch_names.index(ch)
                
                # N2 component (negative peak)
                n2_window = epoch_data[ch_idx, n2_start:n2_end]
                features[f'{ch}_N2_amplitude'] = np.mean(n2_window)
                
                # P2 component (positive peak)
                p2_window = epoch_data[ch_idx, p2_start:p2_end]
                features[f'{ch}_P2_amplitude'] = np.mean(p2_window)
        
        return features
    
    def extract_temporal_features(self, epoch_data, ch_names):
        """Extract time-domain features."""
        features = {}
        
        ch_indices = [ch_names.index(ch) for ch in self.channels if ch in ch_names]
        
        for ch_idx, ch_name in zip(ch_indices, [ch for ch in self.channels if ch in ch_names]):
            signal_data = epoch_data[ch_idx]
            
            # RMS
            features[f'{ch_name}_RMS'] = np.sqrt(np.mean(signal_data**2))
            
            # Variance
            features[f'{ch_name}_variance'] = np.var(signal_data)
            
            # Zero-crossing rate
            zero_crossings = np.sum(np.diff(np.sign(signal_data)) != 0)
            features[f'{ch_name}_zero_crossing'] = zero_crossings / len(signal_data)
        
        return features
    
    def extract_all_features(self, epoch_data, ch_names):
        """Extract all 78 features from a single epoch."""
        all_features = {}
        
        # Spectral features (30)
        spectral_features = self.extract_spectral_features(epoch_data, ch_names)
        all_features.update(spectral_features)
        
        # Ratio features (18)
        ratio_features = self.extract_ratio_features(spectral_features)
        all_features.update(ratio_features)
        
        # Asymmetry features (5)
        asymmetry_features = self.extract_asymmetry_features(spectral_features)
        all_features.update(asymmetry_features)
        
        # ERP features (4)
        erp_features = self.extract_erp_features(epoch_data, ch_names)
        all_features.update(erp_features)
        
        # Temporal features (18)
        temporal_features = self.extract_temporal_features(epoch_data, ch_names)
        all_features.update(temporal_features)
        
        return all_features

def predict_pain_level(epoch_file, model_file):
    """
    Predict pain level from EEG epoch.
    
    Parameters:
    -----------
    epoch_file : str
        Path to .npy file containing epoch data (channels x samples)
    model_file : str
        Path to .pkl file containing trained model
        
    Returns:
    --------
    dict : Prediction results
    """
    
    # Load epoch data
    try:
        epoch_data = np.load(epoch_file)
        print(f"Loaded epoch data: {epoch_data.shape}")
    except Exception as e:
        print(f"Error loading epoch file: {e}")
        return None
    
    # Load model
    try:
        with open(model_file, 'rb') as f:
            model = pickle.load(f)
        print(f"Loaded model: {type(model).__name__}")
    except Exception as e:
        print(f"Error loading model file: {e}")
        return None
    
    # Standard channel names (68 channels)
    ch_names = ['Fp1', 'Fz', 'F3', 'F7', 'FT9', 'FC5', 'FC1', 'C3', 'T7', 'TP9', 
                'CP5', 'CP1', 'Pz', 'P3', 'P7', 'O1', 'Oz', 'O2', 'P4', 'P8', 
                'TP10', 'CP6', 'CP2', 'Cz', 'C4', 'T8', 'FT10', 'FC6', 'FC2', 'F4', 
                'F8', 'Fp2', 'AF7', 'AF3', 'AFz', 'F1', 'F5', 'FT7', 'FC3', 'FCz', 
                'C1', 'C5', 'TP7', 'CP3', 'P1', 'P5', 'PO7', 'PO3', 'POz', 'PO4', 
                'PO8', 'P6', 'P2', 'CPz', 'CP4', 'TP8', 'C6', 'C2', 'FC4', 'FT8', 
                'F6', 'F2', 'AF4', 'AF8', 'Fpz', 'VEOG', 'HEOG', 'EMG'][:epoch_data.shape[0]]
    
    # Extract features
    extractor = FeatureExtractor()
    features = extractor.extract_all_features(epoch_data, ch_names)
    
    # Convert to DataFrame (maintaining feature order)
    features_df = pd.DataFrame([features])
    
    # Make prediction
    try:
        prediction = model.predict(features_df)[0]
        probability = model.predict_proba(features_df)[0]
        
        # Interpret results
        pain_level = "High Pain" if prediction == 1 else "Low Pain"
        confidence = max(probability)
        
        results = {
            'prediction': int(prediction),
            'pain_level': pain_level,
            'probability_low': float(probability[0]),
            'probability_high': float(probability[1]),
            'confidence': float(confidence),
            'features_extracted': len(features)
        }
        
        return results
        
    except Exception as e:
        print(f"Error making prediction: {e}")
        return None

def main():
    """Command-line interface for pain prediction."""
    parser = argparse.ArgumentParser(description='Predict pain level from EEG epoch')
    parser.add_argument('--input', required=True, help='Input epoch file (.npy)')
    parser.add_argument('--model', required=True, help='Trained model file (.pkl)')
    parser.add_argument('--output', help='Output file for results (.json)')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not Path(args.input).exists():
        print(f"Error: Input file {args.input} not found")
        sys.exit(1)
    
    if not Path(args.model).exists():
        print(f"Error: Model file {args.model} not found")
        sys.exit(1)
    
    print("="*60)
    print("EEG PAIN CLASSIFICATION PREDICTION")
    print("="*60)
    print(f"Input epoch: {args.input}")
    print(f"Model: {args.model}")
    print()
    
    # Make prediction
    results = predict_pain_level(args.input, args.model)
    
    if results is None:
        print("Prediction failed!")
        sys.exit(1)
    
    # Display results
    print("PREDICTION RESULTS:")
    print(f"Pain Level: {results['pain_level']}")
    print(f"Confidence: {results['confidence']:.1%}")
    print(f"Probability Low Pain: {results['probability_low']:.1%}")
    print(f"Probability High Pain: {results['probability_high']:.1%}")
    print(f"Features Extracted: {results['features_extracted']}")
    
    # Save results if requested
    if args.output:
        import json
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")

if __name__ == "__main__":
    main()
