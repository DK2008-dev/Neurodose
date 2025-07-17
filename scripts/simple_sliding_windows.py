#!/usr/bin/env python3
"""
Simple Sliding Window Preprocessing Script

Conservative approach: Basic sliding window creation without external dependencies.
Creates 4-second sliding windows with 1-second steps around laser onset events.
"""

import os
import sys
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pickle

# Add project paths
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / 'src'))

from src.data.loader import EEGDataLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SimpleSlidingWindowProcessor:
    """
    Simple sliding window preprocessing for EEG pain classification.
    
    Conservative approach: Essential preprocessing only.
    """
    
    def __init__(self, data_dir: str = "manual_upload/manual_upload"):
        """Initialize with default parameters."""
        self.data_dir = data_dir
        # Use more liberal artifact threshold to accommodate all participants
        # Based on troubleshooting: vp02/vp04 need ~2500µV, vp05 needs ~1900µV
        self.loader = EEGDataLoader(raw_dir=data_dir, eeg_reject_thresh=2500e-6)  # 2500 µV
        
        # Window parameters (literature standard)
        self.window_length = 4.0  # seconds
        self.step_size = 1.0      # seconds
        self.baseline_before = 1.0  # seconds before laser onset
        self.response_after = 3.0   # seconds after laser onset
        
        # Target sampling rate after preprocessing
        self.target_sfreq = 500.0  # Hz
        
        logger.info(f"Initialized SimpleSlidingWindowProcessor")
        logger.info(f"Window: {self.window_length}s, Step: {self.step_size}s")
        logger.info(f"Baseline: {self.baseline_before}s, Response: {self.response_after}s")
        logger.info(f"Artifact threshold: {2500} µV (accommodates all participants)")
    
    def process_participant(self, participant_id: str, data_dir: str) -> Dict:
        """
        Process single participant to create sliding windows.
        
        Args:
            participant_id: Participant identifier (e.g., 'vp01')
            data_dir: Directory containing BrainVision files
            
        Returns:
            Dictionary containing windows, labels, and metadata
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
            
            # Extract events and pain ratings
            events, event_id, severity_map = self.loader.extract_events(raw)
            logger.info(f"Found {len(events)} events, severity mapping: {severity_map}")
            
            # Validate we have the expected number of ratings
            if len(severity_map) < 3:
                logger.warning(f"Expected severity mapping for 3 intensities, got {len(severity_map)}")
            
            # Create sliding windows around laser onset events using the loader's method
            windows, labels = self.loader.create_sliding_windows(
                raw, events, severity_map,
                window_length=self.window_length,
                step_size=self.step_size,
                use_laser_onset=True
            )
            
            logger.info(f"Created {len(windows)} windows with labels")
            
            # Check if we got any windows
            if len(windows) == 0:
                logger.warning(f"No windows created for {participant_id} - this may indicate event detection issues")
                return None
            
            # Create window metadata for consistency
            window_metadata = []
            for i in range(len(windows)):
                window_metadata.append({
                    'window_index': i,
                    'ternary_label': labels[i],
                    'processing_type': 'basic_laser_onset'
                })
            
            # Package results
            result = {
                'participant_id': participant_id,
                'windows': windows,          # Shape: (n_windows, n_channels, n_samples)
                'ternary_labels': labels,    # Low/moderate/high (0/1/2) from loader
                'window_metadata': window_metadata,
                'channel_names': raw.ch_names,
                'sfreq': raw.info['sfreq'],
                'n_windows': len(windows),
                'n_channels': raw.info['nchan'],
                'processing_type': 'basic_windows_only'
            }
            
            logger.info(f"Created {len(windows)} windows for {participant_id}")
            logger.info(f"Window shape: {windows.shape}")
            logger.info(f"Pain rating range: {np.min(labels):.1f} - {np.max(labels):.1f}")
            logger.info(f"Ternary label distribution: {np.bincount(labels)}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing {participant_id}: {str(e)}")
            import traceback
            traceback.print_exc()
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
        
        logger.info(f"Creating windows: {self.window_length}s ({window_samples} samples) at {sfreq} Hz")
        
        # Process each laser onset event
        for i, (event_sample, event_id, pain_rating) in enumerate(zip(events[:, 0], events[:, 2], pain_ratings)):
            # Calculate window start (1s before laser onset)
            window_start = event_sample - int(self.baseline_before * sfreq)
            window_end = window_start + window_samples
            
            # Validate window bounds
            if window_start < 0 or window_end >= data.shape[1]:
                logger.warning(f"Skipping event {i}: window out of bounds (start={window_start}, end={window_end}, data_length={data.shape[1]})")
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
                output_file = os.path.join(output_dir, f"{participant_id}_windows.pkl")
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
            'label_distribution': {},
            'processing_config': {
                'window_length': self.window_length,
                'step_size': self.step_size,
                'baseline_before': self.baseline_before,
                'response_after': self.response_after,
                'target_sfreq': self.target_sfreq,
                'processing_type': 'basic_windows_only'
            }
        }
        
        # Calculate overall label distribution
        all_labels = []
        for result in results.values():
            all_labels.extend(result['ternary_labels'])
        
        if all_labels:
            unique, counts = np.unique(all_labels, return_counts=True)
            for label, count in zip(unique, counts):
                label_names = {0: 'low', 1: 'moderate', 2: 'high'}
                summary['label_distribution'][label_names.get(label, f'label_{label}')] = int(count)
        
        summary_file = os.path.join(output_dir, 'processing_summary.pkl')
        with open(summary_file, 'wb') as f:
            pickle.dump(summary, f)
        
        logger.info(f"Processing complete!")
        logger.info(f"Processed {len(results)} participants")
        logger.info(f"Total windows created: {summary['total_windows']}")
        
        if summary['label_distribution']:
            logger.info("Label distribution:")
            for label, count in summary['label_distribution'].items():
                percentage = (count / summary['total_windows']) * 100
                logger.info(f"  {label}: {count} windows ({percentage:.1f}%)")
        
        logger.info(f"Results saved to: {output_dir}")
        
        return results


def main():
    """Main execution function."""
    
    # Configuration
    data_dir = "manual_upload/manual_upload"
    output_dir = "data/processed/basic_windows"
    
    # Start with validated participants
    test_participants = ['vp01', 'vp02', 'vp03', 'vp04', 'vp05']
    
    logger.info("=== EEG Pain Classification: Basic Sliding Window Preprocessing ===")
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Participants: {test_participants}")
    logger.info("Processing option: Basic windows only (conservative)")
    
    # Initialize processor
    processor = SimpleSlidingWindowProcessor(data_dir)
    
    # Process participants
    try:
        results = processor.process_multiple_participants(
            test_participants, data_dir, output_dir
        )
        
        logger.info("✅ Basic sliding window preprocessing completed successfully!")
        logger.info("Next step: Train models on processed windows")
        
        # Print summary statistics
        if results:
            total_windows = sum(r['n_windows'] for r in results.values())
            avg_windows = total_windows / len(results)
            logger.info(f"Summary: {len(results)} participants, {total_windows} total windows, {avg_windows:.1f} avg windows/participant")
        
    except Exception as e:
        logger.error(f"❌ Processing failed: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
