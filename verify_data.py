#!/usr/bin/env python3
"""
Quick verification of processed data.
"""

import pickle
import numpy as np

def verify_processed_data():
    """Verify the structure of processed windows."""
    
    # Load one file
    with open('data/processed/basic_windows/vp01_windows.pkl', 'rb') as f:
        data = pickle.load(f)
    
    print("=== Processed Data Verification ===")
    print(f"Keys: {list(data.keys())}")
    print(f"Participant: {data['participant_id']}")
    print(f"Number of windows: {data['n_windows']}")
    print(f"Number of channels: {data['n_channels']}")
    print(f"Sampling frequency: {data['sfreq']} Hz")
    print(f"Processing type: {data['processing_type']}")
    
    print(f"\nWindows shape: {data['windows'].shape}")
    print(f"Labels shape: {data['ternary_labels'].shape}")
    print(f"Labels unique: {np.unique(data['ternary_labels'], return_counts=True)}")
    
    # Check a sample window
    sample_window = data['windows'][0]
    print(f"\nSample window (first):")
    print(f"  Shape: {sample_window.shape}")
    print(f"  Min/Max: {sample_window.min():.2e} / {sample_window.max():.2e}")
    print(f"  Mean: {sample_window.mean():.2e}")
    
    print(f"\nChannel names (first 10): {data['channel_names'][:10]}")
    
    # Load summary
    with open('data/processed/basic_windows/processing_summary.pkl', 'rb') as f:
        summary = pickle.load(f)
    
    print(f"\n=== Processing Summary ===")
    print(f"Total participants: {summary['total_participants']}")
    print(f"Successful participants: {summary['successful_participants']}")
    print(f"Total windows: {summary['total_windows']}")
    print(f"Label distribution: {summary['label_distribution']}")

if __name__ == "__main__":
    verify_processed_data()
