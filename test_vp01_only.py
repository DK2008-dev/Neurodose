#!/usr/bin/env python3
"""Test pain rating extraction for vp01 only."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.data.loader import EEGDataLoader
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_vp01_extraction():
    """Test extraction for vp01 specifically."""
    
    # Initialize loader
    loader = EEGDataLoader(
        raw_dir="manual_upload/manual_upload/",
        l_freq=1.0,
        h_freq=45.0,
        notch_freq=50.0,
        new_sfreq=500.0
    )
    
    # Load vp01 data
    vhdr_file = "manual_upload/manual_upload/Exp_Mediation_Paradigm1_Perception_vp01.vhdr"
    
    try:
        print(f"\n=== LOADING VP01 DATA ===")
        raw = loader.load_raw_data(vhdr_file)
        print(f"✓ Loaded raw data: {raw.info['nchan']} channels, {raw.times[-1]:.1f}s duration")
        
        print(f"\n=== EXTRACTING EVENTS ===")
        events, event_id, severity_map = loader.extract_events(raw)
        
        print(f"\n=== EVENT ANALYSIS ===")
        print(f"Total events found: {len(events)}")
        print(f"Event types found: {len(event_id)}")
        
        print(f"\n=== EVENT ID BREAKDOWN ===")
        stimulus_events = []
        laser_events = []
        comment_events = []
        
        for desc, code in event_id.items():
            count = len(events[events[:, 2] == code])
            print(f"  {desc}: {count} events (code {code})")
            
            if 'Stimulus' in desc:
                stimulus_events.append((desc, code, count))
            elif 'Laser' in desc:
                laser_events.append((desc, code, count))
            elif 'Comment' in desc:
                comment_events.append((desc, code, count))
        
        print(f"\n=== STIMULUS SUMMARY ===")
        print(f"Stimulus events: {len(stimulus_events)} types")
        for desc, code, count in stimulus_events:
            print(f"  {desc}: {count} events")
            
        print(f"\n=== LASER SUMMARY ===") 
        print(f"Laser events: {len(laser_events)} types")
        for desc, code, count in laser_events:
            print(f"  {desc}: {count} events")
            
        print(f"\n=== COMMENT SUMMARY ===")
        print(f"Comment events: {len(comment_events)} types") 
        total_comments = sum(count for _, _, count in comment_events)
        print(f"Total comment events: {total_comments}")
        
        for desc, code, count in comment_events:
            # Extract the rating from the description
            if 'Comment/' in desc:
                rating_str = desc.split('Comment/')[-1].strip()
                try:
                    rating = int(rating_str)
                    print(f"  Pain Rating {rating}: {count} events")
                except ValueError:
                    print(f"  {desc}: {count} events (could not parse rating)")
        
        print(f"\n=== SEVERITY MAPPING ===")
        print(f"Stimulus intensity codes: {severity_map}")
        
        print(f"\n=== EXPECTATIONS CHECK ===")
        print(f"Expected: ~60 pain ratings (comment events)")
        print(f"Found: {total_comments} comment events")
        
        if total_comments >= 50:
            print("✓ GOOD: Found expected number of pain ratings")
        else:
            print("⚠ WARNING: Lower than expected number of pain ratings")
            
        return True
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_vp01_extraction()
    print(f"\n=== FINAL RESULT ===")
    if success:
        print("✓ VP01 data extraction completed successfully")
    else:
        print("❌ VP01 data extraction failed")
