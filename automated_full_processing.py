#!/usr/bin/env python3
"""
Automated Full Dataset Processing Script
Processes all 51 participants independently with comprehensive monitoring,
error handling, and progress reporting.
"""

import os
import sys
import time
import pickle
import logging
import traceback
from pathlib import Path
from datetime import datetime
import subprocess

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / 'src'))

from src.data.loader import EEGDataLoader
import mne

# Configure logging for automated monitoring
def setup_logging():
    """Setup comprehensive logging for automated processing."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"full_processing_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Suppress MNE verbose output
    mne.set_log_level('WARNING')
    
    return str(log_file)

def check_available_participants(data_dir):
    """Check which participants have complete BrainVision files."""
    data_path = Path(data_dir)
    available = []
    
    for vp_num in range(1, 52):  # vp01 to vp51
        vp_id = f"vp{vp_num:02d}"
        base_name = f"Exp_Mediation_Paradigm1_Perception_{vp_id}"
        
        required_files = [
            data_path / f"{base_name}.vhdr",
            data_path / f"{base_name}.eeg", 
            data_path / f"{base_name}.vmrk"
        ]
        
        if all(f.exists() for f in required_files):
            available.append(vp_id)
        else:
            logging.warning(f"Missing files for {vp_id}")
    
    logging.info(f"Found {len(available)} participants with complete files")
    return available

def process_single_participant(vp_id, data_dir, output_dir):
    """Process a single participant with comprehensive error handling."""
    try:
        logging.info(f"Starting processing for {vp_id}")
        start_time = time.time()
        
        # Initialize data loader
        loader = EEGDataLoader(data_dir)
        
        # Load and preprocess data
        base_name = f"Exp_Mediation_Paradigm1_Perception_{vp_id}"
        vhdr_file = Path(data_dir) / f"{base_name}.vhdr"
        
        logging.info(f"{vp_id}: Loading raw data...")
        raw = loader.load_raw_data(str(vhdr_file))
        
        logging.info(f"{vp_id}: Extracting events...")
        events = loader.extract_events(raw)
        
        logging.info(f"{vp_id}: Creating sliding windows...")
        windows_data = loader.create_sliding_windows(raw, events)
        
        # Save processed data
        output_file = Path(output_dir) / f"{vp_id}_windows.pkl"
        with open(output_file, 'wb') as f:
            pickle.dump(windows_data, f)
        
        # Calculate statistics
        total_windows = len(windows_data['windows'])
        labels = windows_data['labels']
        label_counts = {
            'low': sum(1 for l in labels if l == 0),
            'moderate': sum(1 for l in labels if l == 1), 
            'high': sum(1 for l in labels if l == 2)
        }
        
        processing_time = time.time() - start_time
        file_size_mb = output_file.stat().st_size / (1024 * 1024)
        
        result = {
            'participant': vp_id,
            'status': 'success',
            'total_windows': total_windows,
            'label_distribution': label_counts,
            'processing_time': processing_time,
            'file_size_mb': file_size_mb,
            'output_file': str(output_file)
        }
        
        logging.info(f"{vp_id}: SUCCESS - {total_windows} windows, "
                    f"{label_counts}, {processing_time:.1f}s, {file_size_mb:.1f}MB")
        
        return result
        
    except Exception as e:
        error_msg = f"{vp_id}: FAILED - {str(e)}"
        logging.error(error_msg)
        logging.error(traceback.format_exc())
        
        return {
            'participant': vp_id,
            'status': 'failed',
            'error': str(e),
            'traceback': traceback.format_exc()
        }

def monitor_system_resources():
    """Monitor system resources and log warnings if needed."""
    try:
        import psutil
        
        # Check memory usage
        memory = psutil.virtual_memory()
        if memory.percent > 85:
            logging.warning(f"High memory usage: {memory.percent:.1f}%")
        
        # Check disk space
        disk = psutil.disk_usage('.')
        if disk.percent > 90:
            logging.warning(f"Low disk space: {disk.percent:.1f}% used")
            
        # Check CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        if cpu_percent > 90:
            logging.warning(f"High CPU usage: {cpu_percent:.1f}%")
            
    except ImportError:
        # psutil not available, skip monitoring
        pass
    except Exception as e:
        logging.warning(f"Resource monitoring failed: {e}")

def save_progress_checkpoint(results, output_dir):
    """Save current progress to allow resuming if needed."""
    checkpoint_file = Path(output_dir) / "processing_checkpoint.pkl"
    checkpoint_data = {
        'timestamp': datetime.now().isoformat(),
        'completed_participants': [r['participant'] for r in results if r['status'] == 'success'],
        'failed_participants': [r['participant'] for r in results if r['status'] == 'failed'],
        'results': results
    }
    
    with open(checkpoint_file, 'wb') as f:
        pickle.dump(checkpoint_data, f)
    
    logging.info(f"Progress checkpoint saved: {len(checkpoint_data['completed_participants'])} completed, "
                f"{len(checkpoint_data['failed_participants'])} failed")

def main():
    """Main automated processing function."""
    print("="*80)
    print("AUTOMATED FULL DATASET PROCESSING")
    print("="*80)
    
    # Setup logging
    log_file = setup_logging()
    logging.info("Starting automated full dataset processing")
    logging.info(f"Log file: {log_file}")
    
    # Configuration
    data_dir = "manual_upload/manual_upload"
    output_dir = Path("data/processed/full_dataset")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check available participants
    logging.info("Checking available participants...")
    available_participants = check_available_participants(data_dir)
    
    if not available_participants:
        logging.error("No participants found with complete files!")
        return
    
    logging.info(f"Will process {len(available_participants)} participants")
    logging.info(f"Estimated total time: {len(available_participants) * 2.5:.1f} minutes")
    
    # Process all participants
    results = []
    total_start_time = time.time()
    
    for i, vp_id in enumerate(available_participants, 1):
        logging.info(f"\n{'='*60}")
        logging.info(f"PROCESSING {i}/{len(available_participants)}: {vp_id}")
        logging.info(f"{'='*60}")
        
        # Monitor system resources every 10 participants
        if i % 10 == 0:
            monitor_system_resources()
        
        # Process participant
        result = process_single_participant(vp_id, data_dir, output_dir)
        results.append(result)
        
        # Save checkpoint every 5 participants
        if i % 5 == 0:
            save_progress_checkpoint(results, output_dir)
        
        # Calculate and display progress
        elapsed_time = time.time() - total_start_time
        avg_time_per_participant = elapsed_time / i
        estimated_remaining = avg_time_per_participant * (len(available_participants) - i)
        
        logging.info(f"Progress: {i}/{len(available_participants)} ({i/len(available_participants)*100:.1f}%)")
        logging.info(f"Elapsed: {elapsed_time/60:.1f}min, Remaining: {estimated_remaining/60:.1f}min")
        
        # Brief pause to prevent system overload
        time.sleep(1)
    
    # Final results summary
    total_time = time.time() - total_start_time
    successful = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] == 'failed']
    
    logging.info(f"\n{'='*80}")
    logging.info("PROCESSING COMPLETE!")
    logging.info(f"{'='*80}")
    logging.info(f"Total time: {total_time/60:.1f} minutes")
    logging.info(f"Successful: {len(successful)}/{len(results)} participants")
    logging.info(f"Failed: {len(failed)}/{len(results)} participants")
    
    if successful:
        total_windows = sum(r['total_windows'] for r in successful)
        total_size_mb = sum(r['file_size_mb'] for r in successful)
        
        # Calculate label distribution across all participants
        all_labels = {'low': 0, 'moderate': 0, 'high': 0}
        for r in successful:
            for label, count in r['label_distribution'].items():
                all_labels[label] += count
        
        logging.info(f"Total windows created: {total_windows}")
        logging.info(f"Label distribution: {all_labels}")
        logging.info(f"Total data size: {total_size_mb:.1f} MB")
        
        # Calculate balance score
        total_labels = sum(all_labels.values())
        if total_labels > 0:
            expected_per_class = total_labels / 3
            balance_score = 1 - max(abs(count - expected_per_class) for count in all_labels.values()) / expected_per_class
            logging.info(f"Balance score: {balance_score:.3f}")
    
    if failed:
        logging.warning(f"Failed participants: {[r['participant'] for r in failed]}")
        for r in failed:
            logging.error(f"{r['participant']}: {r['error']}")
    
    # Save final results
    final_results_file = output_dir / "final_processing_results.pkl"
    final_summary = {
        'timestamp': datetime.now().isoformat(),
        'total_time_minutes': total_time / 60,
        'total_participants': len(results),
        'successful_participants': len(successful),
        'failed_participants': len(failed),
        'results': results,
        'summary_stats': {
            'total_windows': sum(r['total_windows'] for r in successful),
            'total_size_mb': sum(r['file_size_mb'] for r in successful),
            'label_distribution': all_labels if successful else {},
            'balance_score': balance_score if successful and total_labels > 0 else 0
        }
    }
    
    with open(final_results_file, 'wb') as f:
        pickle.dump(final_summary, f)
    
    logging.info(f"Final results saved to: {final_results_file}")
    logging.info("Automated processing completed successfully!")
    
    return final_summary

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.info("Processing interrupted by user")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        logging.error(traceback.format_exc())
        sys.exit(1)
