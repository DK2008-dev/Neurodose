#!/usr/bin/env python3
"""
Automated Progress Monitor
Monitors the automated processing and provides periodic status updates.
"""

import os
import time
import pickle
from pathlib import Path
from datetime import datetime
import subprocess
import sys

def check_processing_status():
    """Check current processing status and provide detailed report."""
    print(f"\n{'='*80}")
    print(f"PROCESSING STATUS CHECK - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}")
    
    # Check if processing is running
    try:
        result = subprocess.run(['tasklist', '/FI', 'IMAGENAME eq python.exe'], 
                              capture_output=True, text=True, shell=True)
        python_processes = result.stdout.count('python.exe')
        print(f"Active Python processes: {python_processes}")
    except:
        print("Could not check running processes")
    
    # Check checkpoint file
    checkpoint_file = Path("data/processed/full_dataset/processing_checkpoint.pkl")
    if checkpoint_file.exists():
        try:
            with open(checkpoint_file, 'rb') as f:
                checkpoint = pickle.load(f)
            
            completed = len(checkpoint['completed_participants'])
            failed = len(checkpoint['failed_participants'])
            total_attempted = completed + failed
            
            print(f"Progress from checkpoint:")
            print(f"  - Completed: {completed} participants")
            print(f"  - Failed: {failed} participants") 
            print(f"  - Total attempted: {total_attempted}")
            print(f"  - Last update: {checkpoint['timestamp']}")
            
            if completed > 0:
                # Calculate some stats from successful participants
                successful_results = [r for r in checkpoint['results'] if r['status'] == 'success']
                if successful_results:
                    total_windows = sum(r['total_windows'] for r in successful_results)
                    avg_processing_time = sum(r['processing_time'] for r in successful_results) / len(successful_results)
                    total_size_mb = sum(r['file_size_mb'] for r in successful_results)
                    
                    print(f"  - Total windows created: {total_windows}")
                    print(f"  - Average processing time: {avg_processing_time:.1f}s per participant")
                    print(f"  - Total data size: {total_size_mb:.1f} MB")
                    
                    # Estimate remaining time
                    if total_attempted < 51:
                        remaining = 51 - total_attempted
                        estimated_remaining_time = remaining * avg_processing_time / 60
                        print(f"  - Estimated remaining time: {estimated_remaining_time:.1f} minutes")
            
        except Exception as e:
            print(f"Error reading checkpoint: {e}")
    else:
        print("No checkpoint file found - processing may not have started yet")
    
    # Check output directory
    output_dir = Path("data/processed/full_dataset")
    if output_dir.exists():
        pickle_files = list(output_dir.glob("vp*_windows.pkl"))
        print(f"Completed participant files: {len(pickle_files)}")
        
        if pickle_files:
            # Show recent files
            recent_files = sorted(pickle_files, key=lambda x: x.stat().st_mtime, reverse=True)[:5]
            print("Most recently created files:")
            for f in recent_files:
                mod_time = datetime.fromtimestamp(f.stat().st_mtime)
                size_mb = f.stat().st_size / (1024 * 1024)
                print(f"  - {f.name}: {mod_time.strftime('%H:%M:%S')}, {size_mb:.1f}MB")
    
    # Check log files
    log_dir = Path("logs")
    if log_dir.exists():
        log_files = list(log_dir.glob("full_processing_*.log"))
        if log_files:
            latest_log = max(log_files, key=lambda x: x.stat().st_mtime)
            print(f"Latest log file: {latest_log.name}")
            
            # Show last few lines of log
            try:
                with open(latest_log, 'r') as f:
                    lines = f.readlines()
                    if len(lines) > 5:
                        print("Recent log entries:")
                        for line in lines[-5:]:
                            print(f"  {line.strip()}")
            except Exception as e:
                print(f"Could not read log file: {e}")
    
    print(f"{'='*80}\n")

def main():
    """Monitor processing with periodic updates."""
    print("AUTOMATED PROCESSING MONITOR STARTED")
    print("Will check status every 5 minutes...")
    print("Press Ctrl+C to stop monitoring\n")
    
    check_count = 0
    while True:
        try:
            check_count += 1
            print(f"Status Check #{check_count}")
            check_processing_status()
            
            # Check if processing is complete
            final_results = Path("data/processed/full_dataset/final_processing_results.pkl")
            if final_results.exists():
                print("🎉 PROCESSING APPEARS TO BE COMPLETE!")
                print("Checking final results...")
                
                try:
                    with open(final_results, 'rb') as f:
                        results = pickle.load(f)
                    
                    print(f"Final Summary:")
                    print(f"  - Total time: {results['total_time_minutes']:.1f} minutes")
                    print(f"  - Successful: {results['successful_participants']}/{results['total_participants']}")
                    print(f"  - Total windows: {results['summary_stats']['total_windows']}")
                    print(f"  - Total size: {results['summary_stats']['total_size_mb']:.1f} MB")
                    print(f"  - Balance score: {results['summary_stats']['balance_score']:.3f}")
                    
                    break
                except Exception as e:
                    print(f"Error reading final results: {e}")
            
            # Wait 5 minutes before next check
            print("Waiting 5 minutes before next check...\n")
            time.sleep(300)  # 5 minutes
            
        except KeyboardInterrupt:
            print("\nMonitoring stopped by user")
            break
        except Exception as e:
            print(f"Monitor error: {e}")
            time.sleep(60)  # Wait 1 minute and retry

if __name__ == "__main__":
    main()
