#!/usr/bin/env python3
"""
Debug script to understand event structure in our EEG data.
"""

import sys
import os
sys.path.append(os.path.abspath('.'))

import mne
import numpy as np
from src.data.loader import EEGDataLoader

def debug_events():
    """Debug event structure for vp01."""
    
    # Load data
    data_dir = "manual_upload/manual_upload"
    file_path = os.path.join(data_dir, "Exp_Mediation_Paradigm1_Perception_vp01.vhdr")
    
    # Load raw data without preprocessing
    raw = mne.io.read_raw_brainvision(file_path, preload=True, verbose=False)
    
    # Get events
    events, event_id = mne.events_from_annotations(raw, verbose=False)
    
    print(f"Total events: {len(events)}")
    print(f"Event IDs: {event_id}")
    print()
    
    # Analyze event patterns
    unique_codes = np.unique(events[:, 2])
    print(f"Unique event codes: {unique_codes}")
    
    for code in unique_codes:
        code_events = events[events[:, 2] == code]
        event_desc = [desc for desc, c in event_id.items() if c == code]
        print(f"Code {code}: {len(code_events)} events - {event_desc}")
    
    print()
    
    # Look for stimulus and laser patterns
    print("=== Event Analysis ===")
    
    # Look for S 1, S 2, S 3 patterns
    stimulus_events = []
    laser_events = []
    
    for desc, code in event_id.items():
        if 'S  1' in desc or 'S  2' in desc or 'S  3' in desc:
            stimulus_events.append((code, desc))
        elif 'L  1' in desc:
            laser_events.append((code, desc))
    
    print(f"Stimulus events: {stimulus_events}")
    print(f"Laser events: {laser_events}")
    
    # Check temporal relationships
    if stimulus_events and laser_events:
        print("\n=== Temporal Analysis ===")
        
        for stim_code, stim_desc in stimulus_events[:3]:  # Check first few
            stim_times = events[events[:, 2] == stim_code][:, 0]
            print(f"\n{stim_desc} (code {stim_code}): {len(stim_times)} events")
            
            if len(stim_times) > 0:
                print(f"  First few times: {stim_times[:5]}")
                
                # Look for laser events after stimulus
                for laser_code, laser_desc in laser_events:
                    laser_times = events[events[:, 2] == laser_code][:, 0]
                    
                    if len(laser_times) > 0:
                        print(f"  {laser_desc} (code {laser_code}): {len(laser_times)} events")
                        print(f"    First few times: {laser_times[:5]}")
                        
                        # Check if they're close in time
                        for st in stim_times[:3]:
                            close_lasers = laser_times[(laser_times > st) & (laser_times < st + 2000)]
                            if len(close_lasers) > 0:
                                print(f"    Laser {close_lasers[0]} follows stimulus {st} (diff: {close_lasers[0] - st} samples)")

if __name__ == "__main__":
    debug_events()
