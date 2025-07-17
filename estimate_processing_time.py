"""
Processing Time Estimation for 51 Participants

This script estimates the total processing time based on our current pipeline
and provides optimization recommendations.
"""

import os
import time
from pathlib import Path

def estimate_processing_time():
    """Estimate processing time for all 51 participants."""
    
    # Current processing metrics (based on 5 participants completed)
    participants_processed = 5
    windows_created = 281  # Total windows from 5 participants
    
    # File sizes and timing estimates
    avg_file_size_mb = 65  # Average .eeg file size
    processing_time_per_participant_minutes = 2.5  # Estimated from current runs
    
    # Calculate metrics
    total_participants = 51
    remaining_participants = total_participants - participants_processed
    
    # Time estimates
    sequential_time_hours = (remaining_participants * processing_time_per_participant_minutes) / 60
    
    # Expected outputs
    expected_windows_per_participant = 60  # Based on protocol
    total_expected_windows = total_participants * expected_windows_per_participant
    
    # Storage estimates
    avg_output_size_mb = 65  # Based on current pickle files
    total_storage_gb = (total_participants * avg_output_size_mb) / 1024
    
    print("=" * 70)
    print("PROCESSING TIME ESTIMATION FOR 51 PARTICIPANTS")
    print("=" * 70)
    
    print(f"\n📊 CURRENT STATUS:")
    print(f"   Completed: {participants_processed}/51 participants ({participants_processed/51*100:.1f}%)")
    print(f"   Windows created: {windows_created}")
    print(f"   Remaining: {remaining_participants} participants")
    
    print(f"\n⏱️  TIME ESTIMATES:")
    print(f"   Per participant: ~{processing_time_per_participant_minutes:.1f} minutes")
    print(f"   Remaining time: ~{sequential_time_hours:.1f} hours ({sequential_time_hours*60:.0f} minutes)")
    print(f"   Total project time: ~{(total_participants * processing_time_per_participant_minutes)/60:.1f} hours")
    
    print(f"\n💾 STORAGE ESTIMATES:")
    print(f"   Per participant: ~{avg_output_size_mb} MB")
    print(f"   Total storage needed: ~{total_storage_gb:.1f} GB")
    
    print(f"\n📈 EXPECTED OUTPUTS:")
    print(f"   Windows per participant: {expected_windows_per_participant}")
    print(f"   Total windows (51 participants): {total_expected_windows:,}")
    print(f"   Data increase: {total_expected_windows/windows_created:.1f}x current dataset")
    
    print(f"\n🚀 OPTIMIZATION RECOMMENDATIONS:")
    print(f"   1. Batch processing: Process 10 participants at a time")
    print(f"   2. Parallel processing: Could reduce time by 50-70%")
    print(f"   3. Progress monitoring: Save intermediate results")
    print(f"   4. Error handling: Skip failed participants, continue processing")
    
    print(f"\n⚠️  POTENTIAL ISSUES:")
    print(f"   1. Memory usage: Monitor RAM with large dataset")
    print(f"   2. Failed participants: Some may have data quality issues")
    print(f"   3. Disk space: Ensure {total_storage_gb:.1f}GB available")
    print(f"   4. Processing interruption: Implement checkpointing")
    
    print("=" * 70)
    
    return {
        'total_time_hours': sequential_time_hours,
        'total_storage_gb': total_storage_gb,
        'expected_windows': total_expected_windows
    }

def check_available_participants():
    """Check how many participants actually have Perception paradigm data."""
    
    manual_dir = Path("manual_upload/manual_upload")
    if not manual_dir.exists():
        print(f"❌ Directory not found: {manual_dir}")
        return
    
    participants = set()
    perception_files = []
    
    for file in manual_dir.glob("*.vhdr"):
        if "Perception" in file.name:
            # Extract participant number
            filename = file.name
            if "vp" in filename:
                # Extract the participant number (e.g., vp01, vp02, etc.)
                try:
                    vp_start = filename.find("vp") + 2
                    vp_num = filename[vp_start:vp_start+2]
                    if vp_num.isdigit():
                        participants.add(f"vp{vp_num}")
                        perception_files.append(file.name)
                except:
                    continue
    
    participants = sorted(participants)
    
    print(f"\n📁 AVAILABLE PARTICIPANTS SCAN:")
    print(f"   Perception paradigm files found: {len(perception_files)}")
    print(f"   Unique participants: {len(participants)}")
    
    if len(participants) > 0:
        print(f"   Participants available: {participants[:10]}{'...' if len(participants) > 10 else ''}")
        print(f"   Range: {participants[0]} to {participants[-1]}")
    
    return participants

if __name__ == "__main__":
    # Run estimations
    estimates = estimate_processing_time()
    available_participants = check_available_participants()
    
    if available_participants:
        actual_count = len(available_participants)
        if actual_count != 51:
            print(f"\n🔄 ADJUSTED ESTIMATE FOR {actual_count} PARTICIPANTS:")
            adjusted_time = (actual_count * 2.5) / 60
            adjusted_storage = (actual_count * 65) / 1024
            print(f"   Adjusted processing time: ~{adjusted_time:.1f} hours")
            print(f"   Adjusted storage needed: ~{adjusted_storage:.1f} GB")
