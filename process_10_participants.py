#!/usr/bin/env python3
"""
Fast Processing Script for 10 Participants
Optimized for rapid iteration and validation.
"""

import os
import sys
import time
import pickle
import logging
from pathlib import Path
from datetime import datetime

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / 'src'))

from src.data.loader import EEGDataLoader
import mne

def setup_logging():
    """Setup logging for 10-participant processing."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    mne.set_log_level('WARNING')

def process_participant_fast(vp_id, data_dir, output_dir):
    """Fast processing for a single participant."""
    try:
        start_time = time.time()
        
        # Initialize data loader
        loader = EEGDataLoader(data_dir)
        
        # Load and preprocess data
        base_name = f"Exp_Mediation_Paradigm1_Perception_{vp_id}"
        vhdr_file = Path(data_dir) / f"{base_name}.vhdr"
        
        if not vhdr_file.exists():
            return {'participant': vp_id, 'status': 'missing_file'}
        
        logging.info(f"{vp_id}: Loading and processing...")
        
        # Load raw data
        raw = loader.load_raw_data(str(vhdr_file))
        
        # Extract events and create windows
        events, pain_ratings = loader.extract_events(raw)
        
        # Create sliding windows with optimized settings
        windows, labels = loader.create_sliding_windows(
            raw, 
            events, 
            pain_ratings,  # This is the severity_map
            window_length=4.0,
            step_size=1.0,
            use_laser_onset=True
        )
        
        if len(windows) == 0:
            return {'participant': vp_id, 'status': 'no_windows'}
        
        # Create windows data structure
        windows_data = {
            'windows': windows,
            'labels': labels
        }
        
        # Save windows data
        output_file = Path(output_dir) / f"{vp_id}_windows.pkl"
        with open(output_file, 'wb') as f:
            pickle.dump(windows_data, f)
        
        processing_time = time.time() - start_time
        num_windows = len(windows_data['windows'])
        
        # Get label distribution
        labels = windows_data['labels']
        label_counts = {
            'low': sum(1 for l in labels if l == 0),
            'moderate': sum(1 for l in labels if l == 1),
            'high': sum(1 for l in labels if l == 2)
        }
        
        logging.info(f"{vp_id}: SUCCESS - {num_windows} windows, {label_counts}, {processing_time:.1f}s")
        
        return {
            'participant': vp_id,
            'status': 'success',
            'windows': num_windows,
            'labels': label_counts,
            'time': processing_time,
            'file': str(output_file)
        }
        
    except Exception as e:
        logging.error(f"{vp_id}: FAILED - {str(e)}")
        return {'participant': vp_id, 'status': 'failed', 'error': str(e)}

def main():
    """Process 10 participants for fast validation."""
    print("="*60)
    print("FAST PROCESSING: 10 PARTICIPANTS")
    print("="*60)
    
    setup_logging()
    
    # Configuration
    data_dir = "manual_upload/manual_upload"
    output_dir = Path("data/processed/fast_10_participants")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Select first 10 participants
    target_participants = [f"vp{i:02d}" for i in range(1, 11)]  # vp01 to vp10
    
    logging.info(f"Processing {len(target_participants)} participants: {target_participants}")
    logging.info(f"Estimated time: {len(target_participants) * 2.5:.1f} minutes")
    
    # Process participants
    results = []
    total_start = time.time()
    
    for i, vp_id in enumerate(target_participants, 1):
        logging.info(f"\n--- Processing {i}/{len(target_participants)}: {vp_id} ---")
        
        result = process_participant_fast(vp_id, data_dir, output_dir)
        results.append(result)
        
        # Progress update
        elapsed = time.time() - total_start
        avg_time = elapsed / i
        remaining = avg_time * (len(target_participants) - i)
        
        logging.info(f"Progress: {i}/{len(target_participants)} ({i/len(target_participants)*100:.1f}%)")
        logging.info(f"Elapsed: {elapsed/60:.1f}min, Remaining: {remaining/60:.1f}min")
    
    # Summary
    total_time = time.time() - total_start
    successful = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] == 'failed']
    missing = [r for r in results if r['status'] == 'missing_file']
    
    print(f"\n{'='*60}")
    print("PROCESSING COMPLETE!")
    print(f"{'='*60}")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Successful: {len(successful)}/{len(results)} participants")
    print(f"Failed: {len(failed)} participants")
    print(f"Missing files: {len(missing)} participants")
    
    if successful:
        total_windows = sum(r['windows'] for r in successful)
        print(f"Total windows created: {total_windows}")
        print(f"Average windows per participant: {total_windows/len(successful):.1f}")
        
        # Show individual results
        print(f"\nSuccessful participants:")
        for r in successful:
            print(f"  {r['participant']}: {r['windows']} windows, {r['labels']}")
    
    if failed:
        print(f"\nFailed participants: {[r['participant'] for r in failed]}")
    
    if missing:
        print(f"Missing files: {[r['participant'] for r in missing]}")
    
    # Save results
    results_file = output_dir / "processing_results.pkl"
    with open(results_file, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\nResults saved to: {results_file}")
    print(f"Window files saved to: {output_dir}")
    
    return results

if __name__ == "__main__":
    results = main()
