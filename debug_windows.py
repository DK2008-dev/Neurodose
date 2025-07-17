#!/usr/bin/env python3
"""
Debug script to understand why window creation is failing.
"""
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from data.loader import EEGDataLoader

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def debug_vp01():
    """Debug window creation for vp01."""
    data_dir = "manual_upload/manual_upload"
    data_path = Path(data_dir)
    
    # Initialize loader
    loader = EEGDataLoader(str(data_path))
    
    # Process vp01
    vp_id = "vp01"
    base_name = f"Exp_Mediation_Paradigm1_Perception_{vp_id}"
    vhdr_file = data_path / f"{base_name}.vhdr"
    
    print(f"Loading {vhdr_file}")
    
    # Load and process
    raw = loader.load_raw_data(str(vhdr_file))
    events, event_id, severity_map = loader.extract_events(raw)
    
    print(f"\nEvent analysis:")
    print(f"Total events: {len(events)}")
    print(f"Event ID mapping: {event_id}")
    print(f"Severity mapping: {severity_map}")
    
    # Check for laser events specifically
    laser_events = []
    stimulus_events = []
    comment_events = []
    
    for desc, code in event_id.items():
        if 'L  1' in desc:
            laser_events.append((desc, code))
        elif 'S  ' in desc:
            stimulus_events.append((desc, code))
        elif 'Comment' in desc:
            comment_events.append((desc, code))
    
    print(f"\nEvent types found:")
    print(f"Stimulus events: {stimulus_events}")
    print(f"Laser events: {laser_events}")
    print(f"Comment events: {len(comment_events)}")
    
    # Check actual event occurrences in the data
    print(f"\nEvent occurrences in data:")
    for desc, code in stimulus_events + laser_events:
        count = sum(1 for event in events if event[2] == code)
        print(f"  {desc} (code {code}): {count} occurrences")
    
    # Try to create windows
    print(f"\nTrying to create windows...")
    
    # Let's manually check the pairing logic first
    print(f"\nManual pairing analysis:")
    laser_code = 10010  # From the debug output
    
    stimulus_events = []
    laser_events_list = []
    
    for event in events:
        if event[2] in severity_map:  # Stimulus events (1, 2, 3)
            stimulus_events.append(event)
        elif event[2] == laser_code:  # Laser events
            laser_events_list.append(event)
    
    print(f"Found {len(stimulus_events)} stimulus events and {len(laser_events_list)} laser events")
    
    # Try to pair them
    pairs_found = 0
    for i, stim_event in enumerate(stimulus_events[:5]):  # First 5
        stim_sample, _, stim_code = stim_event
        print(f"\nStimulus {i}: Sample {stim_sample}, Code {stim_code}")
        
        # Find next laser within 2000 samples
        for laser_event in laser_events_list:
            laser_sample, _, laser_code_actual = laser_event
            time_diff = laser_sample - stim_sample
            
            if 0 < time_diff < 2000:  # Laser after stimulus within 2 seconds
                print(f"  -> Paired with laser at sample {laser_sample} (diff: {time_diff} samples)")
                pairs_found += 1
                break
        else:
            print(f"  -> No laser found within window")
    
    print(f"\nTotal pairs found manually: {pairs_found}")
    
    X, y = loader.create_sliding_windows(raw, events, severity_map)
    print(f"Created {len(X)} windows with {len(y)} labels")
    
    if len(X) == 0:
        print("\nInvestigating artifact rejection...")
        
        # Let's manually check one window
        laser_sample = 5528  # From sequence analysis
        window_length = 4.0  # seconds
        sfreq = raw.info['sfreq']
        window_samples = int(window_length * sfreq)
        
        start = laser_sample - window_samples // 4  # Start 1s before laser
        end = start + window_samples
        
        data = raw.get_data()
        window_data = data[:, start:end]
        
        # Check peak-to-peak values
        peak_to_peak = np.ptp(window_data, axis=1)
        print(f"Sample window peak-to-peak values (first 10 channels):")
        for i in range(min(10, len(peak_to_peak))):
            print(f"  Channel {i}: {peak_to_peak[i]:.2e} V")
        
        print(f"Max peak-to-peak: {np.max(peak_to_peak):.2e} V")
        print(f"Artifact threshold: {loader.eeg_reject_thresh:.2e} V")
        print(f"Rejected? {np.any(peak_to_peak > loader.eeg_reject_thresh)}")
    
    if len(X) == 0:
        print("No windows created - investigating...")
        
        # Let's manually check the sequence
        print(f"\nEvent sequence analysis:")
        sorted_events = events[events[:, 0].argsort()]  # Sort by sample time
        
        for i, event in enumerate(sorted_events[:20]):  # First 20 events
            sample, _, code = event
            desc = None
            for d, c in event_id.items():
                if c == code:
                    desc = d
                    break
            print(f"  {i}: Sample {sample}, Code {code}, Desc: {desc}")

if __name__ == "__main__":
    debug_vp01()
