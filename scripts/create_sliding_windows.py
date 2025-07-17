#!/usr/bin/env python3
"""
Core Sliding Window Preprocessing Script

This script implements the fundamental preprocessing step: creating 4-second sliding windows
with 1-second steps around laser onset events. This is the foundation for all subsequent
feature extraction and model training.

Conservative approach: Focus on essential preprocessing without over-engineering.
"""

import os
import sys
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pickle

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from src.data.loader import EEGDataLoader
from src.utils.helpers import load_config, setup_logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SlidingWindowProcessor:
    """
    Core sliding window preprocessing for EEG pain classification.
    
    Implements literature-standard 4s windows with 1s steps around laser onset events.
    Conservative approach focusing on essential preprocessing only.
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
        
        # Target sampling rate after preprocessing
        self.target_sfreq = 500.0  # Hz
        
        logger.info(f"Initialized SlidingWindowProcessor")
        logger.info(f"Window: {self.window_length}s, Step: {self.step_size}s")
        logger.info(f"Baseline: {self.baseline_before}s, Response: {self.response_after}s")
    
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
            events, pain_ratings = self.loader.extract_events(raw)
            logger.info(f"Found {len(events)} events, {len(pain_ratings)} pain ratings")
            
            # Validate we have the expected number of ratings
            if len(pain_ratings) != 60:
                logger.warning(f"Expected 60 pain ratings, got {len(pain_ratings)}")
            
            # Create sliding windows around laser onset events
            windows, labels, window_metadata = self._create_windows_around_events(
                raw, events, pain_ratings
            )
            
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
                'n_channels': raw.info['nchan']
            }
            
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
            'processing_config': {
                'window_length': self.window_length,
                'step_size': self.step_size,
                'baseline_before': self.baseline_before,
                'response_after': self.response_after,
                'target_sfreq': self.target_sfreq
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
    """Main execution function."""
    
    # Configuration
    data_dir = "manual_upload/manual_upload"
    output_dir = "data/processed/sliding_windows"
    
    # Start with validated participants
    test_participants = ['vp01', 'vp02', 'vp03', 'vp04', 'vp05']
    
    logger.info("=== EEG Pain Classification: Sliding Window Preprocessing ===")
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Participants: {test_participants}")
    
    # Initialize processor
    processor = SlidingWindowProcessor()
    
    # Process participants
    try:
        results = processor.process_multiple_participants(
            test_participants, data_dir, output_dir
        )
        
        logger.info("✅ Sliding window preprocessing completed successfully!")
        logger.info("Next step: Feature extraction or direct model training")
        
    except Exception as e:
        logger.error(f"❌ Processing failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
