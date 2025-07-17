#!/usr/bin/env python3
"""
Debug script to understand why window creation is failing.
"""

import sys
import os
sys.path.append(os.path.abspath('.'))

import mne
import numpy as np
from src.data.loader import EEGDataLoader

def debug_window_creation():
    """Debug window creation for vp01."""
    
    # Load data
    data_dir = "manual_upload/manual_upload"
    file_path = os.path.join(data_dir, "Exp_Mediation_Paradigm1_Perception_vp01.vhdr")
    
    # Initialize loader
    loader = EEGDataLoader(data_dir)
    
    # Load and preprocess
    raw = loader.load_raw_data(file_path)
    print(f"Loaded raw data: {raw.info['nchan']} channels, {raw.info['sfreq']} Hz")
    
    # Extract events
    events, event_id, severity_map = loader.extract_events(raw)
    print(f"Events: {len(events)}, Severity map: {severity_map}")
    
    # Get laser codes
    events_temp, event_id = mne.events_from_annotations(raw, verbose=False)
    laser_codes = []
    for desc, code in event_id.items():
        if 'L  1' in desc:
            laser_codes.append(code)
    
    print(f"Laser codes: {laser_codes}")
    
    # Debug the stimulus-laser pairing logic
    print("\n=== Debug Stimulus-Laser Pairing ===")
    laser_events = []
    
    stim_count = 0
    for event in events:
        if event[2] in severity_map:
            stim_count += 1
            print(f"\nStimulus {stim_count}: time={event[0]}, code={event[2]}, intensity={severity_map[event[2]]}")
            
            # Find corresponding laser event
            laser_time = None
            for other_event in events:
                if (other_event[0] > event[0] and 
                    other_event[0] < event[0] + 2000 and
                    other_event[2] in laser_codes):
                    laser_time = other_event[0]
                    time_diff = laser_time - event[0]
                    print(f"  -> Found laser at {laser_time} (diff: {time_diff} samples = {time_diff/500:.2f}s)")
                    break
            
            if laser_time:
                laser_events.append((laser_time, event[2]))
            else:
                print(f"  -> No matching laser found!")
                
                # Debug: show what events exist in the time window
                window_events = events[(events[:, 0] > event[0]) & (events[:, 0] < event[0] + 2000)]
                print(f"  -> Events in time window: {len(window_events)}")
                for we in window_events[:5]:  # Show first 5
                    we_desc = [desc for desc, code in event_id.items() if code == we[2]]
                    print(f"     {we[0]}: code {we[2]} - {we_desc}")
    
    print(f"\nTotal laser events found: {len(laser_events)}")
    
    # Now test window creation
    if laser_events:
        print(f"\n=== Testing Window Creation ===")
        sfreq = raw.info['sfreq']
        window_samples = int(4.0 * sfreq)  # 4 second windows
        data = raw.get_data()
        n_samples = data.shape[1]
        
        valid_windows = 0
        for i, (laser_sample, intensity_code) in enumerate(laser_events[:3]):  # Test first 3
            start = laser_sample - window_samples // 4  # 1s before laser
            end = start + window_samples
            
            print(f"Window {i+1}: laser={laser_sample}, start={start}, end={end}")
            print(f"  Data range: 0 to {n_samples-1}")
            print(f"  Valid: {start >= 0 and end < n_samples}")
            
            if start >= 0 and end < n_samples:
                valid_windows += 1
        
        print(f"Valid windows: {valid_windows} out of {len(laser_events)}")

if __name__ == "__main__":
    debug_window_creation()
