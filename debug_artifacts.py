#!/usr/bin/env python3
"""
Debug script to check artifact rejection thresholds.
"""

import sys
import os
sys.path.append(os.path.abspath('.'))

import mne
import numpy as np
from src.data.loader import EEGDataLoader

def debug_artifact_threshold():
    """Debug artifact rejection for vp01."""
    
    # Load data
    data_dir = "manual_upload/manual_upload"
    file_path = os.path.join(data_dir, "Exp_Mediation_Paradigm1_Perception_vp01.vhdr")
    
    # Initialize loader
    loader = EEGDataLoader(data_dir)
    
    # Load and preprocess
    raw = loader.load_raw_data(file_path)
    print(f"Current reject threshold: {loader.eeg_reject_thresh} V = {loader.eeg_reject_thresh*1e6} µV")
    
    # Extract events
    events, event_id, severity_map = loader.extract_events(raw)
    
    # Get laser codes  
    events_temp, event_id = mne.events_from_annotations(raw, verbose=False)
    laser_codes = []
    for desc, code in event_id.items():
        if 'L  1' in desc:
            laser_codes.append(code)
    
    # Find first few stimulus-laser pairs
    laser_events = []
    stim_count = 0
    for event in events:
        if event[2] in severity_map and stim_count < 10:  # Test first 10
            stim_count += 1
            for other_event in events:
                if (other_event[0] > event[0] and 
                    other_event[0] < event[0] + 2000 and
                    other_event[2] in laser_codes):
                    laser_events.append((other_event[0], event[2]))
                    break
    
    # Test window extraction and artifact checking
    sfreq = raw.info['sfreq']
    window_samples = int(4.0 * sfreq)  # 4 second windows
    data = raw.get_data()
    
    print(f"\n=== Artifact Threshold Analysis ===")
    print(f"Window length: {window_samples} samples ({window_samples/sfreq:.1f}s)")
    
    valid_count = 0
    for i, (laser_sample, intensity_code) in enumerate(laser_events[:5]):  # Test first 5
        start = laser_sample - window_samples // 4  # 1s before laser
        end = start + window_samples
        
        if start >= 0 and end < data.shape[1]:
            window_data = data[:, start:end]
            
            # Check artifacts (same logic as loader)
            peak_to_peak = np.ptp(window_data, axis=1)
            max_ptp = np.max(peak_to_peak)
            exceeds_thresh = np.any(peak_to_peak > loader.eeg_reject_thresh)
            
            print(f"\nWindow {i+1}: laser={laser_sample}, intensity={intensity_code}")
            print(f"  Peak-to-peak max: {max_ptp*1e6:.1f} µV")
            print(f"  Exceeds threshold: {exceeds_thresh}")
            print(f"  Channel with max: {np.argmax(peak_to_peak)}")
            
            if not exceeds_thresh:
                valid_count += 1
    
    print(f"\nValid windows (under threshold): {valid_count} out of {len(laser_events[:5])}")
    
    # Suggest better threshold
    print(f"\n=== Threshold Recommendations ===")
    print(f"Current threshold: {loader.eeg_reject_thresh*1e6:.1f} µV")
    print(f"Suggested thresholds to try:")
    print(f"  Conservative: 150 µV (150e-6)")
    print(f"  Moderate: 200 µV (200e-6)")
    print(f"  Liberal: 300 µV (300e-6)")

if __name__ == "__main__":
    debug_artifact_threshold()
