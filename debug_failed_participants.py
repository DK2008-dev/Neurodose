#!/usr/bin/env python3
"""
Comprehensive debug script for troubleshooting failed participants.
"""

import sys
import os
sys.path.append(os.path.abspath('.'))

import mne
import numpy as np
from src.data.loader import EEGDataLoader

def debug_participant(participant_id: str):
    """Debug a specific participant's data processing."""
    
    print(f"\n{'='*50}")
    print(f"DEBUGGING PARTICIPANT {participant_id}")
    print(f"{'='*50}")
    
    # Load data
    data_dir = "manual_upload/manual_upload"
    file_path = os.path.join(data_dir, f"Exp_Mediation_Paradigm1_Perception_{participant_id}.vhdr")
    
    try:
        # Initialize loader with liberal threshold (same as successful participants)
        loader = EEGDataLoader(data_dir, eeg_reject_thresh=1500e-6)
        
        # Load and preprocess
        print(f"Loading {file_path}...")
        raw = loader.load_raw_data(file_path)
        print(f"✓ Loaded raw data: {raw.info['nchan']} channels, {raw.info['sfreq']} Hz")
        print(f"  Duration: {raw.times[-1]:.1f} seconds")
        print(f"  Data range: {raw.get_data().min():.2e} to {raw.get_data().max():.2e} V")
        
        # Extract events
        events, event_id, severity_map = loader.extract_events(raw)
        print(f"✓ Events extracted: {len(events)} total")
        print(f"  Severity mapping: {severity_map}")
        
        # Get event breakdown
        unique_codes, counts = np.unique(events[:, 2], return_counts=True)
        print(f"  Event code breakdown:")
        for code, count in zip(unique_codes, counts):
            event_desc = [desc for desc, c in event_id.items() if c == code]
            print(f"    Code {code}: {count} events - {event_desc}")
        
        # Check laser events specifically
        laser_codes = []
        for desc, code in event_id.items():
            if 'L  1' in desc:
                laser_codes.append(code)
        print(f"  Laser codes found: {laser_codes}")
        
        # Test stimulus-laser pairing
        print(f"\n--- Stimulus-Laser Pairing Analysis ---")
        laser_events = []
        stim_count = 0
        
        for event in events:
            if event[2] in severity_map:
                stim_count += 1
                # Find corresponding laser event
                laser_time = None
                for other_event in events:
                    if (other_event[0] > event[0] and 
                        other_event[0] < event[0] + 2000 and
                        other_event[2] in laser_codes):
                        laser_time = other_event[0]
                        break
                
                if laser_time:
                    laser_events.append((laser_time, event[2]))
                    if stim_count <= 3:  # Show first 3
                        time_diff = laser_time - event[0]
                        print(f"  Stimulus {stim_count}: {event[0]} → Laser {laser_time} (diff: {time_diff} samples = {time_diff/500:.2f}s)")
                else:
                    print(f"  Stimulus {stim_count}: {event[0]} → NO LASER FOUND!")
                    if stim_count <= 3:  # Debug first few
                        window_events = events[(events[:, 0] > event[0]) & (events[:, 0] < event[0] + 2000)]
                        print(f"    Events in time window: {len(window_events)}")
                        for we in window_events[:3]:
                            we_desc = [desc for desc, code in event_id.items() if code == we[2]]
                            print(f"      {we[0]}: code {we[2]} - {we_desc}")
        
        print(f"  Total stimulus-laser pairs found: {len(laser_events)}")
        
        # Test artifact threshold
        print(f"\n--- Artifact Threshold Analysis ---")
        if laser_events:
            sfreq = raw.info['sfreq']
            window_samples = int(4.0 * sfreq)
            data = raw.get_data()
            
            valid_count = 0
            high_artifact_count = 0
            
            for i, (laser_sample, intensity_code) in enumerate(laser_events[:10]):  # Test first 10
                start = laser_sample - window_samples // 4
                end = start + window_samples
                
                if start >= 0 and end < data.shape[1]:
                    window_data = data[:, start:end]
                    peak_to_peak = np.ptp(window_data, axis=1)
                    max_ptp = np.max(peak_to_peak)
                    exceeds_thresh = np.any(peak_to_peak > loader.eeg_reject_thresh)
                    
                    if not exceeds_thresh:
                        valid_count += 1
                    else:
                        high_artifact_count += 1
                        if i < 3:  # Show details for first few problematic windows
                            bad_channel = np.argmax(peak_to_peak)
                            print(f"    Window {i+1}: Max P2P = {max_ptp*1e6:.1f} µV (channel {bad_channel})")
            
            print(f"  Valid windows: {valid_count}/{len(laser_events[:10])}")
            print(f"  High artifact windows: {high_artifact_count}/{len(laser_events[:10])}")
            print(f"  Artifact threshold: {loader.eeg_reject_thresh*1e6:.1f} µV")
            
            if valid_count == 0:
                print(f"  ⚠️  ALL WINDOWS REJECTED - Need higher threshold!")
                # Find actual max artifact level
                all_max_ptps = []
                for laser_sample, intensity_code in laser_events[:10]:
                    start = laser_sample - window_samples // 4
                    end = start + window_samples
                    if start >= 0 and end < data.shape[1]:
                        window_data = data[:, start:end]
                        peak_to_peak = np.ptp(window_data, axis=1)
                        all_max_ptps.append(np.max(peak_to_peak))
                
                if all_max_ptps:
                    min_threshold_needed = np.min(all_max_ptps)
                    max_threshold_needed = np.max(all_max_ptps)
                    print(f"  Suggested threshold range: {min_threshold_needed*1e6:.0f} - {max_threshold_needed*1e6:.0f} µV")
        
        return {
            'participant_id': participant_id,
            'success': True,
            'n_events': len(events),
            'n_laser_pairs': len(laser_events),
            'laser_codes': laser_codes,
            'valid_windows': valid_count if laser_events else 0,
            'error': None
        }
        
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            'participant_id': participant_id,
            'success': False,
            'error': str(e)
        }

def main():
    """Debug all failed participants."""
    
    failed_participants = ['vp02', 'vp04', 'vp05']
    successful_participants = ['vp01', 'vp03']
    
    print("TROUBLESHOOTING FAILED PARTICIPANTS")
    print("=" * 60)
    
    results = {}
    
    # First, check successful participants for reference
    print("\n🔍 REFERENCE: Successful participants")
    for pid in successful_participants:
        print(f"\n{pid} (SUCCESSFUL REFERENCE):")
        result = debug_participant(pid)
        results[pid] = result
    
    # Then debug failed participants
    print(f"\n\n🚨 DEBUGGING: Failed participants")
    for pid in failed_participants:
        result = debug_participant(pid)
        results[pid] = result
    
    # Summary
    print(f"\n\n{'='*60}")
    print("TROUBLESHOOTING SUMMARY")
    print(f"{'='*60}")
    
    print("\nSUCCESSFUL PARTICIPANTS:")
    for pid in successful_participants:
        if pid in results and results[pid]['success']:
            r = results[pid]
            print(f"  {pid}: {r['n_laser_pairs']} windows, laser codes {r['laser_codes']}")
    
    print("\nFAILED PARTICIPANTS:")
    for pid in failed_participants:
        if pid in results:
            r = results[pid]
            if r['success']:
                print(f"  {pid}: {r['n_laser_pairs']} windows, laser codes {r['laser_codes']} - FIXABLE")
            else:
                print(f"  {pid}: ERROR - {r['error']}")
    
    # Recommendations
    print(f"\n{'='*60}")
    print("RECOMMENDATIONS")
    print(f"{'='*60}")
    
    # Check if it's just threshold issue
    threshold_issues = []
    data_issues = []
    
    for pid in failed_participants:
        if pid in results and results[pid]['success']:
            if results[pid]['valid_windows'] == 0:
                threshold_issues.append(pid)
        elif pid in results and not results[pid]['success']:
            data_issues.append(pid)
    
    if threshold_issues:
        print(f"\n📈 THRESHOLD ISSUES: {threshold_issues}")
        print("   → Solution: Increase artifact rejection threshold")
        print("   → Suggested: Try 2000-3000 µV threshold")
    
    if data_issues:
        print(f"\n⚠️  DATA ISSUES: {data_issues}")
        print("   → Solution: Individual debugging needed")

if __name__ == "__main__":
    main()
