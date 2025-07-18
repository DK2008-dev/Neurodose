#!/usr/bin/env python3
"""
Debug Binary Pain Classifier - Channel Issue
"""

import pickle
import numpy as np
from pathlib import Path

def debug_channel_extraction():
    """Debug the channel extraction issue"""
    print("Debugging channel extraction...")
    
    # Load data
    with open('data/processed/basic_windows/vp01_windows.pkl', 'rb') as f:
        data = pickle.load(f)
    
    windows = data['windows']
    channel_names = data['channel_names']
    
    print(f"Windows shape: {windows.shape}")
    print(f"Channel names length: {len(channel_names)}")
    
    # Target channels
    channels_of_interest = ['Cz', 'CPz', 'C3', 'C4', 'Fz', 'Pz']
    
    # Find indices
    channel_indices = []
    for ch_name in channels_of_interest:
        if ch_name in channel_names:
            idx = channel_names.index(ch_name)
            channel_indices.append(idx)
            print(f"Found {ch_name} at index {idx}")
        else:
            print(f"Missing {ch_name}")
    
    print(f"Channel indices: {channel_indices}")
    
    if channel_indices:
        # Test feature extraction on one epoch
        epoch = windows[0]  # First epoch
        print(f"Epoch shape: {epoch.shape}")
        
        # Extract channel data
        for i, ch_idx in enumerate(channel_indices):
            ch_data = epoch[ch_idx, :]
            print(f"Channel {channels_of_interest[i]} (idx {ch_idx}): {ch_data.shape}, range [{ch_data.min():.2e}, {ch_data.max():.2e}]")

if __name__ == "__main__":
    debug_channel_extraction()
