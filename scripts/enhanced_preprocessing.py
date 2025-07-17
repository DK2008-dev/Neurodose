#!/usr/bin/env python3
"""
Enhanced Preprocessing Script with Conservative ICA and Optional Wavelet Features

This script implements a balanced approach to preprocessing:
1. Core sliding window creation (essential)
2. Optional ICA artifact removal (conservative)
3. Optional wavelet feature extraction (literature-informed)

Users can control the level of preprocessing to avoid over-engineering.
"""

import os
import sys
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pickle
import pywt  # For wavelet transform

# Add src to path for imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / 'src'))

from src.data.loader import EEGDataLoader
from src.utils.helpers import load_config, setup_logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EnhancedPreprocessor:
    """
    Enhanced preprocessing with configurable options for ICA and wavelet features.
    
    Conservative approach: Essential preprocessing by default, optional enhancements.
    """
    
    def __init__(self, config_path: str = "config/cnn_config.yaml"):
        """Initialize with configuration."""
        self.config = load_config(config_path)
        self.loader = EEGDataLoader()
        
        # Window parameters (literature standard)
        self.window_length = 4.0  # seconds
        self.step_size = 1.0      # seconds
        self.baseline_before = 1.0  # seconds before laser onset
        self.response_after = 3.0   # seconds after laser onset
        
        # Processing options (conservative defaults)
        self.apply_ica = False      # Conservative: off by default
        self.extract_wavelet = False  # Conservative: off by default
        self.wavelet_type = 'db4'   # Literature standard
        self.wavelet_levels = 5     # Literature standard
        
        # Target sampling rate after preprocessing
        self.target_sfreq = 500.0  # Hz
        
        logger.info(f"Initialized EnhancedPreprocessor")
        logger.info(f"Window: {self.window_length}s, Step: {self.step_size}s")
        logger.info(f"ICA: {'Enabled' if self.apply_ica else 'Disabled'}")
        logger.info(f"Wavelet: {'Enabled' if self.extract_wavelet else 'Disabled'}")
    
    def configure_processing(self, apply_ica: bool = False, extract_wavelet: bool = False):
        """
        Configure preprocessing options.
        
        Args:
            apply_ica: Whether to apply ICA artifact removal
            extract_wavelet: Whether to extract wavelet features
        """
        self.apply_ica = apply_ica
        self.extract_wavelet = extract_wavelet
        
        logger.info(f"Updated processing options:")
        logger.info(f"  ICA: {'Enabled' if self.apply_ica else 'Disabled'}")
        logger.info(f"  Wavelet: {'Enabled' if self.extract_wavelet else 'Disabled'}")
    
    def process_participant(self, participant_id: str, data_dir: str) -> Dict:
        """
        Process single participant with configurable preprocessing.
        
        Args:
            participant_id: Participant identifier (e.g., 'vp01')
            data_dir: Directory containing BrainVision files
            
        Returns:
            Dictionary containing windows, features, labels, and metadata
        """
        logger.info(f"Processing {participant_id}...")
        
        # Construct file path
        base_filename = f"Exp_Mediation_Paradigm1_Perception_{participant_id}"
        file_path = os.path.join(data_dir, f"{base_filename}.vhdr")
        
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return None
        
        try:
            # Load and preprocess raw data
            raw = self.loader.load_raw_data(file_path)
            logger.info(f"Loaded raw data: {raw.info['nchan']} channels, {raw.info['sfreq']} Hz")
            
            # Optional ICA artifact removal
            if self.apply_ica:
                logger.info("Applying ICA artifact removal...")
                raw = self.loader.apply_ica_artifact_removal(raw)
            
            # Extract events and pain ratings
            events, pain_ratings = self.loader.extract_events(raw)
            logger.info(f"Found {len(events)} events, {len(pain_ratings)} pain ratings")
            
            # Validate we have the expected number of ratings
            if len(pain_ratings) != 60:
                logger.warning(f"Expected 60 pain ratings, got {len(pain_ratings)}")
            
            # Create sliding windows around laser onset events
            windows, labels, window_metadata = self._create_windows_around_events(
                raw, events, pain_ratings
            )
            
            # Optional wavelet feature extraction
            wavelet_features = None
            if self.extract_wavelet:
                logger.info("Extracting wavelet features...")
                wavelet_features = self._extract_wavelet_features(windows)
            
            # Create ternary labels using percentile-based approach
            ternary_labels = self._create_ternary_labels(labels, participant_id)
            
            # Package results
            result = {
                'participant_id': participant_id,
                'windows': windows,          # Shape: (n_windows, n_channels, n_samples)
                'pain_ratings': labels,      # Continuous pain ratings
                'ternary_labels': ternary_labels,  # Low/moderate/high (0/1/2)
                'window_metadata': window_metadata,
                'channel_names': raw.ch_names,
                'sfreq': raw.info['sfreq'],
                'n_windows': len(windows),
                'n_channels': raw.info['nchan'],
                'processing_options': {
                    'ica_applied': self.apply_ica,
                    'wavelet_extracted': self.extract_wavelet,
                    'wavelet_type': self.wavelet_type if self.extract_wavelet else None
                }
            }
            
            # Add wavelet features if extracted
            if wavelet_features is not None:
                result['wavelet_features'] = wavelet_features
                result['wavelet_feature_names'] = self._get_wavelet_feature_names()
            
            logger.info(f"Created {len(windows)} windows for {participant_id}")
            logger.info(f"Pain rating range: {np.min(labels):.1f} - {np.max(labels):.1f}")
            logger.info(f"Ternary label distribution: {np.bincount(ternary_labels)}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing {participant_id}: {str(e)}")
            return None
    
    def _create_windows_around_events(self, raw, events, pain_ratings) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        """
        Create sliding windows around laser onset events.
        
        Conservative approach: Extract windows centered on laser onset with consistent timing.
        """
        windows = []
        labels = []
        metadata = []
        
        sfreq = raw.info['sfreq']
        window_samples = int(self.window_length * sfreq)
        
        # Get data as numpy array
        data = raw.get_data()  # Shape: (n_channels, n_samples)
        
        # Process each laser onset event
        for i, (event_sample, event_id, pain_rating) in enumerate(zip(events[:, 0], events[:, 2], pain_ratings)):
            # Calculate window start (1s before laser onset)
            window_start = event_sample - int(self.baseline_before * sfreq)
            window_end = window_start + window_samples
            
            # Validate window bounds
            if window_start < 0 or window_end >= data.shape[1]:
                logger.warning(f"Skipping event {i}: window out of bounds")
                continue
            
            # Extract window
            window_data = data[:, window_start:window_end]
            
            # Store window and metadata
            windows.append(window_data)
            labels.append(pain_rating)
            metadata.append({
                'event_index': i,
                'event_sample': event_sample,
                'window_start': window_start,
                'window_end': window_end,
                'stimulus_intensity': event_id,
                'pain_rating': pain_rating
            })
        
        return np.array(windows), np.array(labels), metadata
    
    def _extract_wavelet_features(self, windows: np.ndarray) -> np.ndarray:
        """
        Extract wavelet features using db4 wavelet (literature standard).
        
        Args:
            windows: EEG windows of shape (n_windows, n_channels, n_samples)
            
        Returns:
            wavelet_features: Feature matrix of shape (n_windows, n_features)
        """
        n_windows, n_channels, n_samples = windows.shape
        features_per_channel = []
        
        logger.info(f"Extracting wavelet features from {n_windows} windows...")
        
        for window_idx in range(n_windows):
            window_features = []
            
            for ch_idx in range(n_channels):
                signal = windows[window_idx, ch_idx, :]
                
                # Wavelet decomposition using db4
                coeffs = pywt.wavedec(signal, self.wavelet_type, level=self.wavelet_levels)
                
                # Extract statistical features from each level
                channel_features = []
                for level_coeffs in coeffs:
                    # Statistical measures (literature standard)
                    stats = [
                        np.mean(level_coeffs),              # Mean
                        np.std(level_coeffs),               # Standard deviation
                        np.var(level_coeffs),               # Variance
                        np.median(level_coeffs),            # Median
                        np.percentile(level_coeffs, 25),    # 25th percentile
                        np.percentile(level_coeffs, 75),    # 75th percentile
                        np.sqrt(np.mean(level_coeffs**2)),  # RMS
                        self._zero_crossing_rate(level_coeffs)  # Zero-crossing rate
                    ]
                    channel_features.extend(stats)
                
                window_features.extend(channel_features)
            
            features_per_channel.append(window_features)
        
        wavelet_features = np.array(features_per_channel)
        logger.info(f"Extracted wavelet features: {wavelet_features.shape}")
        
        return wavelet_features
    
    def _zero_crossing_rate(self, signal: np.ndarray) -> float:
        """Calculate zero-crossing rate of a signal."""
        return np.sum(np.diff(np.sign(signal)) != 0) / len(signal)
    
    def _get_wavelet_feature_names(self) -> List[str]:
        """Generate feature names for wavelet features."""
        feature_names = []
        stats = ['mean', 'std', 'var', 'median', 'p25', 'p75', 'rms', 'zcr']
        
        # Assuming we have channel names from the loader
        for ch_idx in range(68):  # Typical number of channels
            for level in range(self.wavelet_levels + 1):  # +1 for approximation coefficients
                for stat in stats:
                    feature_names.append(f"ch{ch_idx:02d}_lvl{level}_{stat}")
        
        return feature_names
    
    def _create_ternary_labels(self, pain_ratings: np.ndarray, participant_id: str) -> np.ndarray:
        """
        Create ternary labels using percentile-based approach.
        
        Conservative approach: Use participant-specific percentiles rather than fixed thresholds.
        This respects individual differences in pain expression.
        """
        # Calculate percentiles for this participant
        p33 = np.percentile(pain_ratings, 33.33)
        p67 = np.percentile(pain_ratings, 66.67)
        
        # Create ternary labels
        ternary_labels = np.zeros(len(pain_ratings), dtype=int)
        ternary_labels[(pain_ratings > p33) & (pain_ratings <= p67)] = 1  # Moderate
        ternary_labels[pain_ratings > p67] = 2  # High
        
        logger.info(f"{participant_id} percentile thresholds: Low ≤{p33:.1f}, Moderate {p33:.1f}-{p67:.1f}, High >{p67:.1f}")
        
        return ternary_labels
    
    def process_multiple_participants(self, participant_ids: List[str], data_dir: str, output_dir: str):
        """
        Process multiple participants and save results.
        
        Args:
            participant_ids: List of participant IDs to process
            data_dir: Directory containing BrainVision files
            output_dir: Directory to save processed data
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        results = {}
        
        for participant_id in participant_ids:
            result = self.process_participant(participant_id, data_dir)
            
            if result is not None:
                results[participant_id] = result
                
                # Save individual participant data
                output_file = os.path.join(output_dir, f"{participant_id}_processed.pkl")
                with open(output_file, 'wb') as f:
                    pickle.dump(result, f)
                logger.info(f"Saved {participant_id} data to {output_file}")
            else:
                logger.error(f"Failed to process {participant_id}")
        
        # Save combined results summary
        summary = {
            'total_participants': len(results),
            'successful_participants': list(results.keys()),
            'total_windows': sum(r['n_windows'] for r in results.values()),
            'processing_config': {
                'window_length': self.window_length,
                'step_size': self.step_size,
                'baseline_before': self.baseline_before,
                'response_after': self.response_after,
                'target_sfreq': self.target_sfreq,
                'ica_applied': self.apply_ica,
                'wavelet_extracted': self.extract_wavelet,
                'wavelet_type': self.wavelet_type if self.extract_wavelet else None
            }
        }
        
        summary_file = os.path.join(output_dir, 'processing_summary.pkl')
        with open(summary_file, 'wb') as f:
            pickle.dump(summary, f)
        
        logger.info(f"Processing complete!")
        logger.info(f"Processed {len(results)} participants")
        logger.info(f"Total windows created: {summary['total_windows']}")
        logger.info(f"Results saved to: {output_dir}")
        
        return results


def main():
    """Main execution function with configurable preprocessing options."""
    
    # Configuration
    data_dir = "manual_upload/manual_upload"
    output_dir = "data/processed/enhanced_preprocessing"
    
    # Start with validated participants
    test_participants = ['vp01', 'vp02', 'vp03', 'vp04', 'vp05']
    
    logger.info("=== EEG Pain Classification: Enhanced Preprocessing ===")
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Participants: {test_participants}")
    
    # Initialize processor
    processor = EnhancedPreprocessor()
    
    # Configure preprocessing options
    print("\nPreprocessing Options:")
    print("1. Basic windows only (conservative)")
    print("2. Windows + ICA artifact removal")
    print("3. Windows + Wavelet features")
    print("4. Windows + ICA + Wavelet features (full)")
    
    choice = input("\nSelect option (1-4, default=1): ").strip()
    
    if choice == "2":
        processor.configure_processing(apply_ica=True, extract_wavelet=False)
    elif choice == "3":
        processor.configure_processing(apply_ica=False, extract_wavelet=True)
    elif choice == "4":
        processor.configure_processing(apply_ica=True, extract_wavelet=True)
    else:
        processor.configure_processing(apply_ica=False, extract_wavelet=False)
        logger.info("Using conservative preprocessing (basic windows only)")
    
    # Process participants
    try:
        results = processor.process_multiple_participants(
            test_participants, data_dir, output_dir
        )
        
        logger.info("✅ Enhanced preprocessing completed successfully!")
        logger.info("Next step: Model training with processed data")
        
    except Exception as e:
        logger.error(f"❌ Processing failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
