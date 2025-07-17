"""
Data loading and preprocessing for EEG pain classification.

This module handles loading BrainVision EEG files and basic preprocessing
following the methodology from Tiemann et al. (2018).
"""

import os
import logging
from typing import List, Tuple, Dict, Optional
import numpy as np
import mne
from mne.preprocessing import ICA
import pandas as pd
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EEGDataLoader:
    """Loads and preprocesses EEG data from BrainVision format."""
    
    def __init__(self, 
                 raw_dir: str,
                 l_freq: float = 1.0,
                 h_freq: float = 45.0,
                 notch_freq: float = 50.0,
                 new_sfreq: float = 500.0,
                 eeg_reject_thresh: float = 100e-6):
        """
        Initialize the EEG data loader.
        
        Parameters:
        -----------
        raw_dir : str
            Path to directory containing BrainVision files
        l_freq : float
            High-pass filter frequency (Hz)
        h_freq : float
            Low-pass filter frequency (Hz)
        notch_freq : float
            Notch filter frequency for power line noise (Hz)
        new_sfreq : float
            Target sampling frequency after resampling (Hz)
        eeg_reject_thresh : float
            Threshold for rejecting bad EEG epochs (V)
        """
        self.raw_dir = Path(raw_dir)
        self.l_freq = l_freq
        self.h_freq = h_freq
        self.notch_freq = notch_freq
        self.new_sfreq = new_sfreq
        self.eeg_reject_thresh = eeg_reject_thresh
        
        # Pain-relevant electrode locations based on Tiemann et al.
        self.pain_channels = ['C4', 'Cz', 'FCz', 'CPz', 'FC2', 'CP2']
        self.roi_groups = {
            'central': ['C3', 'C4', 'CP3', 'CP4'],
            'vertex': ['Cz', 'FCz', 'CPz'],
            'frontocentral': ['Fz', 'FC1', 'FC2', 'AFz']
        }
        
    def find_subject_files(self) -> List[str]:
        """Find all .vhdr files in the raw directory."""
        vhdr_files = list(self.raw_dir.glob('**/*.vhdr'))
        logger.info(f"Found {len(vhdr_files)} subject files")
        return [str(f) for f in vhdr_files]
    
    def load_raw_data(self, vhdr_file: str) -> mne.io.Raw:
        """
        Load a single BrainVision file and apply basic preprocessing.
        
        Parameters:
        -----------
        vhdr_file : str
            Path to .vhdr file
            
        Returns:
        --------
        raw : mne.io.Raw
            Preprocessed raw EEG data
        """
        logger.info(f"Loading {vhdr_file}")
        
        # Load BrainVision file
        raw = mne.io.read_raw_brainvision(vhdr_file, preload=True, verbose=False)
        
        # Keep only EEG channels (drop EOG if present)
        raw.pick_types(eeg=True, eog=False)
        
        # Apply basic preprocessing
        logger.info("Applying preprocessing filters...")
        raw.filter(l_freq=self.l_freq, h_freq=self.h_freq, verbose=False)
        raw.notch_filter(freqs=self.notch_freq, verbose=False)
        raw.set_eeg_reference('average', verbose=False)
        
        # Resample to reduce computational load
        if raw.info['sfreq'] != self.new_sfreq:
            raw.resample(self.new_sfreq, verbose=False)
        
        return raw
    
    def apply_ica_artifact_removal(self, raw: mne.io.Raw) -> mne.io.Raw:
        """
        Apply ICA for artifact removal.
        
        Parameters:
        -----------
        raw : mne.io.Raw
            Raw EEG data
            
        Returns:
        --------
        raw : mne.io.Raw
            Cleaned EEG data
        """
        logger.info("Applying ICA artifact removal...")
        
        # Fit ICA
        ica = ICA(method='infomax', random_state=42, verbose=False)
        ica.fit(raw)
        
        # Auto-detect artifact components (if EOG channels available)
        try:
            eog_indices, eog_scores = ica.find_bads_eog(raw, threshold=3.0, verbose=False)
            ica.exclude = eog_indices
            logger.info(f"Excluding {len(eog_indices)} ICA components")
        except Exception as e:
            logger.warning(f"Could not auto-detect artifacts: {e}")
        
        # Apply ICA
        raw = ica.apply(raw, verbose=False)
        
        return raw
    
    def extract_events(self, raw: mne.io.Raw) -> Tuple[np.ndarray, Dict, Dict]:
        """
        Extract stimulus events from annotations for Tiemann et al. dataset.
        
        Parameters:
        -----------
        raw : mne.io.Raw
            Raw EEG data with annotations
            
        Returns:
        --------
        events : np.ndarray
            Event array (samples x 3)
        event_id : dict
            Mapping from event names to codes
        severity_map : dict
            Mapping from event codes to intensity levels
        """
        events, event_id = mne.events_from_annotations(raw, verbose=False)
        
        # Create severity mapping for Tiemann dataset (S 1, S 2, S 3)
        severity_map = {}
        laser_events = []
        pain_ratings = {}
        
        for desc, code in event_id.items():
            if 'S  1' in desc:  # Low intensity
                severity_map[code] = 1
            elif 'S  2' in desc:  # Medium intensity
                severity_map[code] = 2
            elif 'S  3' in desc:  # High intensity
                severity_map[code] = 3
            elif 'L  1' in desc:  # Laser onset - use these for epoch timing
                laser_events.append(code)
        
        # Extract pain ratings from comments - ratings are in the event name after 'Comment/'
        for desc, code in event_id.items():
            if 'Comment/' in desc:
                try:
                    # Extract numeric rating from description like "Comment/50"
                    rating_str = desc.split('Comment/')[-1].strip()
                    rating = int(rating_str)
                    # Find all events with this code and store their ratings
                    comment_events = events[events[:, 2] == code]
                    for event in comment_events:
                        pain_ratings[event[0]] = rating  # sample -> rating
                except (ValueError, IndexError):
                    pass
        
        logger.info(f"Found {len(events)} events")
        logger.info(f"Stimulus intensity mapping: {severity_map}")
        logger.info(f"Laser event codes: {laser_events}")
        logger.info(f"Found {len(pain_ratings)} pain ratings")
        
        return events, event_id, severity_map
    
    def create_sliding_windows(self, 
                             raw: mne.io.Raw, 
                             events: np.ndarray,
                             severity_map: Dict,
                             window_length: float = 4.0,
                             step_size: float = 1.0,
                             use_laser_onset: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sliding windows with pain labels for Tiemann dataset.
        
        Parameters:
        -----------
        raw : mne.io.Raw
            Preprocessed EEG data
        events : np.ndarray
            Event array
        severity_map : dict
            Mapping from event codes to severity values
        window_length : float
            Window length in seconds
        step_size : float
            Step size in seconds
        use_laser_onset : bool
            If True, center windows on laser onset; if False, use sliding windows
            
        Returns:
        --------
        X : np.ndarray
            EEG data windows (n_windows, n_channels, n_samples)
        y : np.ndarray
            Pain severity labels (n_windows,)
        """
        logger.info("Creating sliding windows...")
        
        sfreq = raw.info['sfreq']
        window_samples = int(window_length * sfreq)
        step_samples = int(step_size * sfreq)
        
        data = raw.get_data()
        n_channels, n_samples = data.shape
        
        windows = []
        labels = []
        
        if use_laser_onset:
            # Extract windows centered on laser onset events
            laser_events = []
            for event in events:
                # Look for stimulus events (S 1, S 2, S 3) followed by laser (L 1)
                if event[2] in severity_map:
                    # Find corresponding laser event within reasonable time window
                    laser_time = None
                    for other_event in events:
                        if (other_event[0] > event[0] and 
                            other_event[0] < event[0] + 2000 and  # Within 2 seconds
                            other_event[2] not in severity_map):  # Not a stimulus event
                            laser_time = other_event[0]
                            break
                    
                    if laser_time:
                        laser_events.append((laser_time, event[2]))  # (sample, intensity_code)
            
            # Create windows around laser onsets
            for laser_sample, intensity_code in laser_events:
                start = laser_sample - window_samples // 4  # Start 1s before laser
                end = start + window_samples
                
                if start >= 0 and end < n_samples:
                    window_data = data[:, start:end]
                    
                    # Convert intensity code to ternary label
                    intensity = severity_map[intensity_code]
                    if intensity == 1:
                        label = 0  # low
                    elif intensity == 2:
                        label = 1  # moderate
                    else:  # intensity == 3
                        label = 2  # high
                    
                    # Check for artifacts
                    peak_to_peak = np.ptp(window_data, axis=1)
                    if np.any(peak_to_peak > self.eeg_reject_thresh):
                        continue
                    
                    windows.append(window_data)
                    labels.append(label)
        
        else:
            # Original sliding window approach
            for start in range(0, n_samples - window_samples + 1, step_samples):
                end = start + window_samples
                
                window_data = data[:, start:end]
                
                # Find events within this window
                window_events = events[(events[:, 0] >= start) & (events[:, 0] < end)]
                
                if len(window_events) > 0:
                    # Get severity values for events in this window
                    severities = [severity_map.get(event[2], 0) for event in window_events if event[2] in severity_map]
                    
                    if severities:
                        # Use modal (most common) severity
                        modal_severity = max(set(severities), key=severities.count)
                        
                        # Convert to ternary labels
                        if modal_severity == 1:
                            label = 0  # low
                        elif modal_severity == 2:
                            label = 1  # moderate
                        else:  # modal_severity == 3
                            label = 2  # high
                        
                        # Check for artifacts
                        peak_to_peak = np.ptp(window_data, axis=1)
                        if np.any(peak_to_peak > self.eeg_reject_thresh):
                            continue
                        
                        windows.append(window_data)
                        labels.append(label)
        
        X = np.array(windows)
        y = np.array(labels)
        
        logger.info(f"Created {len(X)} windows with label distribution: {np.bincount(y)}")
        
        return X, y
    
    def process_subject(self, vhdr_file: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process a single subject file end-to-end.
        
        Parameters:
        -----------
        vhdr_file : str
            Path to subject's .vhdr file
            
        Returns:
        --------
        X : np.ndarray
            EEG windows for this subject
        y : np.ndarray
            Pain labels for this subject
        """
        try:
            # Load and preprocess
            raw = self.load_raw_data(vhdr_file)
            raw = self.apply_ica_artifact_removal(raw)
            
            # Extract events
            events, event_id, severity_map = self.extract_events(raw)
            
            # Create sliding windows
            X, y = self.create_sliding_windows(raw, events, severity_map)
            
            return X, y
            
        except Exception as e:
            logger.error(f"Error processing {vhdr_file}: {e}")
            return np.array([]), np.array([])
    
    def process_all_subjects(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process all subjects in the raw directory.
        
        Returns:
        --------
        X : np.ndarray
            All EEG windows (n_total_windows, n_channels, n_samples)
        y : np.ndarray
            All pain labels (n_total_windows,)
        """
        vhdr_files = self.find_subject_files()
        
        all_X = []
        all_y = []
        
        for vhdr_file in vhdr_files:
            X, y = self.process_subject(vhdr_file)
            if len(X) > 0:
                all_X.append(X)
                all_y.append(y)
        
        if all_X:
            X_combined = np.concatenate(all_X, axis=0)
            y_combined = np.concatenate(all_y, axis=0)
            
            logger.info(f"Total dataset: {len(X_combined)} windows")
            logger.info(f"Shape: {X_combined.shape}")
            logger.info(f"Label distribution: {np.bincount(y_combined)}")
            
            return X_combined, y_combined
        else:
            logger.error("No valid data found!")
            return np.array([]), np.array([])
