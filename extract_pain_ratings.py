#!/usr/bin/env python3
"""
Extract Original Pain Ratings for Literature-Standard Analysis

This script extracts the original continuous pain ratings (0-100 scale) 
from the raw EEG data processing pipeline to enable literature-standard
labeling comparison.
"""

import os
import sys
import numpy as np
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

# Add project paths
project_root = Path(__file__).parent
sys.path.append(str(project_root / 'src'))

from src.data.loader import EEGDataLoader

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def extract_pain_ratings_for_participant(participant_id: str, data_dir: str = "manual_upload/manual_upload") -> Tuple[np.ndarray, np.ndarray]:
    """Extract original pain ratings and create windows for a participant."""
    
    # Construct file path
    base_filename = f"Exp_Mediation_Paradigm1_Perception_{participant_id}"
    file_path = os.path.join(data_dir, f"{base_filename}.vhdr")
    
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return None, None
    
    # Initialize loader
    loader = EEGDataLoader(raw_dir=data_dir)
    
    try:
        # Load raw data
        raw = loader.load_raw_data(file_path)
        logger.info(f"Loaded {participant_id}: {raw.info['nchan']} channels, {raw.info['sfreq']} Hz")
        
        # Extract events - this should give us the pain ratings directly
        events, event_id, severity_map = loader.extract_events(raw)
        logger.info(f"Found {len(events)} events, severity mapping: {severity_map}")
        
        # Extract pain ratings from the event annotations directly
        pain_ratings = {}
        for desc, code in event_id.items():
            if 'Comment/' in desc:
                try:
                    # Extract numeric rating from description like "Comment/50"
                    rating_str = desc.split('Comment/')[-1].strip()
                    rating = int(rating_str)
                    # Find all events with this code and store their ratings
                    comment_events = events[events[:, 2] == code]
                    for event in comment_events:
                        pain_ratings[event[0]] = rating  # sample -> rating
                except (ValueError, IndexError):
                    pass
        
        logger.info(f"Extracted {len(pain_ratings)} pain ratings from comments")
        
        if len(pain_ratings) == 0:
            logger.warning(f"No pain ratings found for {participant_id}")
            return None, None
        
        # Extract original pain ratings from the events
        rating_values = list(pain_ratings.values())
        logger.info(f"Pain rating range: {min(rating_values)} - {max(rating_values)}")
        
        # Create 4-second windows around laser events  
        sfreq = raw.info['sfreq']
        window_samples = int(4.0 * sfreq)  # 4 seconds
        baseline_samples = int(1.0 * sfreq)  # 1 second before laser
        
        # Get data
        data = raw.get_data()
        
        windows = []
        extracted_ratings = []
        
        # Process each laser event with corresponding pain rating
        laser_events = [evt for evt in events if evt[2] in [1, 2, 3]]  # Laser onset codes
        
        for i, event in enumerate(laser_events):
            event_sample = event[0]
            
            # Find corresponding pain rating
            # Pain ratings are stored by sample number in the pain_ratings dict
            closest_rating_sample = min(pain_ratings.keys(), 
                                      key=lambda x: abs(x - event_sample))
            rating = pain_ratings[closest_rating_sample]
            
            # Extract window
            window_start = event_sample - baseline_samples
            window_end = window_start + window_samples
            
            if window_start >= 0 and window_end < data.shape[1]:
                window = data[:, window_start:window_end]
                windows.append(window)
                extracted_ratings.append(rating)
        
        logger.info(f"Extracted {len(windows)} windows with ratings for {participant_id}")
        return np.array(windows), np.array(extracted_ratings)
        
    except Exception as e:
        logger.error(f"Error processing {participant_id}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None


def main():
    """Extract pain ratings for all participants."""
    
    logger.info("🔍 Extracting Original Pain Ratings for Literature-Standard Analysis")
    logger.info("=" * 80)
    
    participants = ['vp01', 'vp02', 'vp03', 'vp04', 'vp05']
    all_windows = []
    all_ratings = []
    all_participants = []
    
    for participant in participants:
        windows, ratings = extract_pain_ratings_for_participant(participant)
        
        if windows is not None and ratings is not None:
            all_windows.append(windows)
            all_ratings.append(ratings)
            all_participants.extend([participant] * len(windows))
            
            logger.info(f"{participant}: {len(windows)} windows, rating range {ratings.min()}-{ratings.max()}")
    
    if all_windows:
        # Combine all data
        X = np.concatenate(all_windows, axis=0)
        pain_ratings = np.concatenate(all_ratings, axis=0)
        participants = np.array(all_participants)
        
        logger.info(f"\nCombined dataset:")
        logger.info(f"Total windows: {len(X)}")
        logger.info(f"Pain rating range: {pain_ratings.min()} - {pain_ratings.max()}")
        logger.info(f"Unique ratings: {len(np.unique(pain_ratings))}")
        
        # Save data with original pain ratings
        output_data = {
            'windows': X,
            'pain_ratings': pain_ratings,
            'participants': participants,
            'channel_names': [f'Ch{i}' for i in range(X.shape[1])],
            'sfreq': 500.0,
            'description': 'EEG windows with original continuous pain ratings (0-100 scale)'
        }
        
        output_file = 'data/processed/windows_with_pain_ratings.pkl'
        with open(output_file, 'wb') as f:
            pickle.dump(output_data, f)
        
        logger.info(f"\nData saved to: {output_file}")
        logger.info("✅ Ready for literature-standard Random Forest analysis")
        
        return output_file
    else:
        logger.error("No valid data extracted!")
        return None


if __name__ == "__main__":
    main()
