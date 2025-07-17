"""
Test script to verify data loading from Tiemann et al. dataset.

This script loads a sample file and displays the data structure.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.loader import EEGDataLoader
import numpy as np
import matplotlib.pyplot as plt

def test_data_loading():
    """Test loading a single subject from the Tiemann dataset."""
    
    # Path to the raw data
    raw_dir = r"c:\Users\rohitmo\OneDrive\Neurodosing Model\manual_upload\manual_upload"
    
    # Initialize data loader
    loader = EEGDataLoader(raw_dir)
    
    # Test with one subject
    test_file = os.path.join(raw_dir, "Exp_Mediation_Paradigm1_Perception_vp01.vhdr")
    
    if not os.path.exists(test_file):
        print(f"Test file not found: {test_file}")
        return
    
    print("=" * 60)
    print("TESTING TIEMANN ET AL. DATASET LOADING")
    print("=" * 60)
    
    try:
        # Load raw data
        print("\n1. Loading raw data...")
        raw = loader.load_raw_data(test_file)
        print(f"   ✓ Loaded {len(raw.ch_names)} channels")
        print(f"   ✓ Sampling frequency: {raw.info['sfreq']} Hz")
        print(f"   ✓ Duration: {raw.times[-1]:.1f} seconds")
        print(f"   ✓ Data shape: {raw.get_data().shape}")
        
        # Check channel names
        print(f"\n2. Channel information...")
        pain_channels = ['C3', 'C4', 'Cz', 'FCz', 'CPz', 'CP3', 'CP4']
        found_channels = [ch for ch in pain_channels if ch in raw.ch_names]
        print(f"   ✓ Found pain-relevant channels: {found_channels}")
        
        # Extract events
        print(f"\n3. Extracting events...")
        events, event_id, severity_map = loader.extract_events(raw)
        print(f"   ✓ Found {len(events)} total events")
        print(f"   ✓ Event types: {list(event_id.keys())}")
        print(f"   ✓ Severity mapping: {severity_map}")
        
        # Analyze event structure
        print(f"\n4. Event analysis...")
        stimulus_events = [e for e in events if e[2] in severity_map]
        print(f"   ✓ Stimulus events: {len(stimulus_events)}")
        
        # Count by intensity
        intensity_counts = {}
        for event in stimulus_events:
            intensity = severity_map[event[2]]
            intensity_counts[intensity] = intensity_counts.get(intensity, 0) + 1
        print(f"   ✓ Intensity distribution: {intensity_counts}")
        
        # Apply ICA (optional for testing)
        print(f"\n5. Applying preprocessing...")
        raw_clean = loader.apply_ica_artifact_removal(raw)
        print(f"   ✓ ICA cleaning complete")
        
        # Create windows
        print(f"\n6. Creating data windows...")
        X, y = loader.create_sliding_windows(raw_clean, events, severity_map, 
                                           use_laser_onset=True)
        print(f"   ✓ Created {len(X)} windows")
        print(f"   ✓ Window shape: {X.shape}")
        print(f"   ✓ Label distribution: {np.bincount(y)}")
        
        # Display some statistics
        print(f"\n7. Data statistics...")
        print(f"   ✓ Mean EEG amplitude: {np.mean(X):.2f} µV")
        print(f"   ✓ Std EEG amplitude: {np.std(X):.2f} µV")
        print(f"   ✓ Min/Max amplitude: {np.min(X):.1f} / {np.max(X):.1f} µV")
        
        # Save sample for later use
        if len(X) > 0:
            print(f"\n8. Saving sample data...")
            np.savez('sample_data.npz', X=X[:10], y=y[:10])  # Save first 10 windows
            print(f"   ✓ Saved 10 sample windows to sample_data.npz")
        
        print(f"\n" + "=" * 60)
        print("✅ DATA LOADING SUCCESSFUL!")
        print("✅ Ready to process full dataset")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print(f"❌ DATA LOADING FAILED")
        return False

if __name__ == "__main__":
    success = test_data_loading()
    if success:
        print("\n🎉 All tests passed! The data loader is working correctly.")
    else:
        print("\n💥 Tests failed. Please check the error messages above.")
