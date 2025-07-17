#!/usr/bin/env python3
"""Analyze pain rating distributions across multiple participants."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.data.loader import EEGDataLoader
import logging
import numpy as np
from collections import Counter

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_pain_ratings():
    """Analyze pain rating distributions across participants."""
    
    loader = EEGDataLoader(
        raw_dir="manual_upload/manual_upload/",
        l_freq=1.0,
        h_freq=45.0,
        notch_freq=50.0,
        new_sfreq=500.0
    )
    
    # Test first 5 participants
    participants = ['vp01', 'vp02', 'vp03', 'vp04', 'vp05']
    all_ratings = []
    participant_ratings = {}
    
    print(f"\n=== PAIN RATING ANALYSIS ===")
    
    for vp in participants:
        try:
            vhdr_file = f"manual_upload/manual_upload/Exp_Mediation_Paradigm1_Perception_{vp}.vhdr"
            raw = loader.load_raw_data(vhdr_file)
            events, event_id, severity_map = loader.extract_events(raw)
            
            # Extract pain ratings for this participant
            ratings = []
            for desc, code in event_id.items():
                if 'Comment/' in desc:
                    rating_str = desc.split('Comment/')[-1].strip()
                    try:
                        rating = int(rating_str)
                        comment_events = events[events[:, 2] == code]
                        for _ in comment_events:
                            ratings.append(rating)
                    except ValueError:
                        pass
            
            participant_ratings[vp] = sorted(ratings)
            all_ratings.extend(ratings)
            
            print(f"\n{vp.upper()}:")
            print(f"  Total ratings: {len(ratings)}")
            rating_counts = Counter(ratings)
            print(f"  Unique values: {sorted(rating_counts.keys())}")
            print(f"  Distribution: {dict(rating_counts)}")
            
            # Check for multiples of 10
            non_multiples = [r for r in ratings if r % 10 != 0]
            if non_multiples:
                print(f"  ⚠ Non-10-multiples found: {non_multiples}")
            else:
                print(f"  ✓ All ratings are multiples of 10")
                
        except Exception as e:
            print(f"❌ Error processing {vp}: {e}")
    
    print(f"\n=== OVERALL ANALYSIS ===")
    print(f"Total ratings collected: {len(all_ratings)}")
    
    overall_counts = Counter(all_ratings)
    print(f"All unique rating values: {sorted(overall_counts.keys())}")
    
    # Check rating range
    min_rating = min(all_ratings)
    max_rating = max(all_ratings)
    print(f"Rating range: {min_rating} - {max_rating}")
    
    # Check for gaps
    expected_values = list(range(min_rating, max_rating + 1, 10))
    actual_values = sorted(overall_counts.keys())
    missing_values = set(expected_values) - set(actual_values)
    
    print(f"Expected 10-multiples in range: {expected_values}")
    print(f"Actually observed values: {actual_values}")
    
    if missing_values:
        print(f"⚠ Missing expected values: {sorted(missing_values)}")
    else:
        print(f"✓ All expected 10-multiples present")
    
    # Check if any non-multiples of 10 exist
    non_multiples = [r for r in all_ratings if r % 10 != 0]
    if non_multiples:
        print(f"⚠ Found non-10-multiples: {sorted(set(non_multiples))}")
        print(f"  Count: {len(non_multiples)} out of {len(all_ratings)} total")
    else:
        print(f"✓ ALL ratings are exact multiples of 10")
    
    # Distribution analysis
    print(f"\n=== RATING DISTRIBUTION ===")
    for rating in sorted(overall_counts.keys()):
        count = overall_counts[rating]
        percentage = (count / len(all_ratings)) * 100
        print(f"  Rating {rating:2d}: {count:3d} times ({percentage:5.1f}%)")
    
    print(f"\n=== EXPERIMENTAL DESIGN IMPLICATIONS ===")
    if all(r % 10 == 0 for r in all_ratings):
        print("✓ FINDING: Ratings are constrained to 10-point intervals (0, 10, 20, ..., 90, 100)")
        print("✓ INTERPRETATION: This suggests a structured rating scale, not free-form input")
        print("✓ LIKELY METHOD: Visual Analog Scale (VAS) with discrete markers or")
        print("                 Numeric Rating Scale (NRS) with 11-point scale (0-100 in steps of 10)")
        print("✓ CLINICAL RELEVANCE: This is common in pain research for standardization")
    else:
        print("⚠ FINDING: Some ratings are not multiples of 10")
        print("⚠ INTERPRETATION: Mixed rating method or data entry variations")

if __name__ == "__main__":
    analyze_pain_ratings()
