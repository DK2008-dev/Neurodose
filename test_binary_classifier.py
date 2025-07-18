#!/usr/bin/env python3
"""
Test Binary Pain Classifier - Debug Version
"""

import pickle
import numpy as np
from pathlib import Path

def test_data_loading():
    """Test loading our processed data"""
    print("Testing data loading...")
    
    data_dir = "data/processed/basic_windows"
    data_path = Path(data_dir)
    participant_files = list(data_path.glob("vp*_windows.pkl"))
    
    print(f"Found {len(participant_files)} participant files:")
    for file_path in participant_files:
        print(f"  {file_path}")
    
    # Load first participant to check structure
    if participant_files:
        with open(participant_files[0], 'rb') as f:
            data = pickle.load(f)
        
        print(f"\nData structure for {participant_files[0].name}:")
        if isinstance(data, dict):
            print(f"  Type: Dictionary")
            print(f"  Keys: {list(data.keys())}")
            if 'windows' in data:
                print(f"  Windows shape: {data['windows'].shape}")
            if 'labels' in data:
                print(f"  Labels shape: {data['labels'].shape}")
                print(f"  Labels unique: {np.unique(data['labels'])}")
            if 'channel_names' in data:
                print(f"  Channels ({len(data['channel_names'])}): {data['channel_names'][:10]}...")
        else:
            print(f"  Type: {type(data)}")
            if isinstance(data, tuple) and len(data) == 2:
                print(f"  Windows shape: {data[0].shape}")
                print(f"  Labels shape: {data[1].shape}")

if __name__ == "__main__":
    test_data_loading()
