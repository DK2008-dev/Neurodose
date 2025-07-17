#!/usr/bin/env python3
"""
Simplified Full Dataset Processing Script
Processes all 51 participants with basic monitoring.
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

# Import only essential modules
try:
    from src.data.loader import EEGDataLoader
    import mne
    mne.set_log_level('WARNING')
except ImportError as e:
    print(f"Import error: {e}")
    print("Waiting for dependencies to install...")
    time.sleep(30)
    try:
        from src.data.loader import EEGDataLoader
        import mne
        mne.set_log_level('WARNING')
    except ImportError:
        print("Dependencies still not available. Exiting.")
        sys.exit(1)

def setup_logging():
    """Setup logging."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"processing_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return str(log_file)

def main():
    """Process all participants automatically."""
    print("="*80)
    print("AUTOMATED DATASET PROCESSING")
    print("="*80)
    
    log_file = setup_logging()
    logging.info("Starting automated processing")
    
    # Configuration
    data_dir = "manual_upload/manual_upload"
    output_dir = Path("data/processed/full_dataset")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find available participants
    data_path = Path(data_dir)
    available = []
    
    for vp_num in range(1, 52):
        vp_id = f"vp{vp_num:02d}"
        base_name = f"Exp_Mediation_Paradigm1_Perception_{vp_id}"
        vhdr_file = data_path / f"{base_name}.vhdr"
        
        if vhdr_file.exists():
            available.append(vp_id)
    
    logging.info(f"Found {len(available)} participants")
    
    # Process each participant
    results = []
    start_time = time.time()
    
    for i, vp_id in enumerate(available, 1):
        try:
            logging.info(f"Processing {i}/{len(available)}: {vp_id}")
            
            # Initialize loader with correct parameters
            loader = EEGDataLoader(str(data_path))
            
            # Load data
            base_name = f"Exp_Mediation_Paradigm1_Perception_{vp_id}"
            vhdr_file = data_path / f"{base_name}.vhdr"
            
            raw = loader.load_raw_data(str(vhdr_file))
            events, event_id, severity_map = loader.extract_events(raw)
            X, y = loader.create_sliding_windows(raw, events, severity_map)
            
            # Create windows data structure for saving
            windows_data = {
                'windows': X,
                'labels': y,
                'participant': vp_id,
                'event_id': event_id,
                'severity_map': severity_map
            }
            
            # Save result
            output_file = output_dir / f"{vp_id}_windows.pkl"
            with open(output_file, 'wb') as f:
                pickle.dump(windows_data, f)
            
            # Log success
            total_windows = len(X)
            label_counts = {
                'low': sum(1 for l in y if l == 0),
                'moderate': sum(1 for l in y if l == 1),
                'high': sum(1 for l in y if l == 2)
            }
            
            result = {
                'participant': vp_id,
                'status': 'success',
                'total_windows': total_windows,
                'label_distribution': label_counts
            }
            results.append(result)
            
            logging.info(f"{vp_id}: SUCCESS - {total_windows} windows, {label_counts}")
            
        except Exception as e:
            logging.error(f"{vp_id}: FAILED - {str(e)}")
            results.append({
                'participant': vp_id,
                'status': 'failed',
                'error': str(e)
            })
        
        # Progress update
        elapsed = time.time() - start_time
        avg_time = elapsed / i
        remaining = avg_time * (len(available) - i)
        
        logging.info(f"Progress: {i}/{len(available)} ({i/len(available)*100:.1f}%), "
                    f"Elapsed: {elapsed/60:.1f}min, Remaining: {remaining/60:.1f}min")
    
    # Final summary
    total_time = time.time() - start_time
    successful = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] == 'failed']
    
    logging.info(f"PROCESSING COMPLETE!")
    logging.info(f"Total time: {total_time/60:.1f} minutes")
    logging.info(f"Successful: {len(successful)}/{len(results)}")
    logging.info(f"Failed: {len(failed)}/{len(results)}")
    
    if successful:
        total_windows = sum(r['total_windows'] for r in successful)
        logging.info(f"Total windows: {total_windows}")
    
    # Save final results
    final_results = {
        'timestamp': datetime.now().isoformat(),
        'total_time_minutes': total_time / 60,
        'successful_participants': len(successful),
        'failed_participants': len(failed),
        'results': results
    }
    
    with open(output_dir / "final_results.pkl", 'wb') as f:
        pickle.dump(final_results, f)
    
    logging.info("Results saved. Processing complete!")

if __name__ == "__main__":
    main()
