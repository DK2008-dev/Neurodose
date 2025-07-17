"""
Quick script to inspect the structure of the windows data file.
"""

import pickle
import numpy as np
from pathlib import Path

def inspect_windows_file():
    """Inspect the structure of the windows data file."""
    windows_file = "data/processed/windows_with_pain_ratings.pkl"
    
    if not Path(windows_file).exists():
        print(f"File not found: {windows_file}")
        return
    
    print(f"Loading {windows_file}...")
    with open(windows_file, 'rb') as f:
        data = pickle.load(f)
    
    print(f"Data type: {type(data)}")
    print(f"Data structure:")
    
    if isinstance(data, dict):
        print("  Dictionary keys:", list(data.keys()))
        for key, value in data.items():
            print(f"  {key}: {type(value)}")
            if hasattr(value, '__len__'):
                print(f"    Length: {len(value)}")
            if isinstance(value, (list, tuple)) and len(value) > 0:
                print(f"    First item type: {type(value[0])}")
                if isinstance(value[0], dict):
                    print(f"    First item keys: {list(value[0].keys())}")
    elif isinstance(data, (list, tuple)):
        print(f"  List/Tuple length: {len(data)}")
        if len(data) > 0:
            print(f"  First item type: {type(data[0])}")
            if isinstance(data[0], dict):
                print(f"  First item keys: {list(data[0].keys())}")
                # Show first few items structure
                for i in range(min(3, len(data))):
                    print(f"  Item {i}:")
                    for key, value in data[i].items():
                        print(f"    {key}: {type(value)} {getattr(value, 'shape', '') if hasattr(value, 'shape') else ''}")
    else:
        print(f"  Unexpected data type: {type(data)}")
    
    print("\nInspection complete.")

if __name__ == "__main__":
    inspect_windows_file()
