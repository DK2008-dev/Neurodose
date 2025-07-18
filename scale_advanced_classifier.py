#!/usr/bin/env python3
"""
Scale Advanced EEG Pain Classifier to Full Dataset
Advanced features + Full 51-participant validation
"""

import numpy as np
import pandas as pd
import logging
from pathlib import Path
import time
from datetime import datetime
import pickle
import warnings
warnings.filterwarnings('ignore')

# Import our advanced classifier components
from advanced_pain_classifier import AdvancedFeatureExtractor, AdvancedClassifierEnsemble

def setup_logging():
    """Setup logging configuration."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"advanced_full_dataset_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def load_full_dataset():
    """Load all available participants from processed data."""
    # First check if full dataset exists
    full_data_dir = Path("data/processed/windows_all_participants")
    if full_data_dir.exists():
        logging.info("Loading from full dataset directory...")
        return load_processed_data(full_data_dir)
    
    # If not, try to load from existing basic windows
    basic_data_dir = Path("data/processed/basic_windows")
    if basic_data_dir.exists():
        logging.info("Loading from basic_windows directory (5 participants)...")
        return load_processed_data(basic_data_dir)
    
    # If neither exists, suggest processing
    logging.error("No processed data found. Please run data processing first.")
    return None, None, None

def load_processed_data(data_dir):
    """Load processed data from directory."""
    all_data = []
    all_labels = []
    participants = []
    
    participant_files = list(data_dir.glob("vp*_windows.pkl"))
    logging.info(f"Found {len(participant_files)} participant files")
    
    for file_path in sorted(participant_files):
        participant = file_path.stem.split('_')[0]
        
        try:
            with open(file_path, 'rb') as f:
                windows_data = pickle.load(f)
            
            # Handle different data structures
            if isinstance(windows_data, dict):
                if 'ternary_labels' in windows_data:
                    windows = windows_data['windows']
                    labels = windows_data['ternary_labels']
                elif 'labels' in windows_data:
                    windows = windows_data['windows']
                    labels = windows_data['labels']
                else:
                    logging.warning(f"Unknown data structure in {file_path}")
                    continue
            else:
                logging.warning(f"Unexpected data type in {file_path}")
                continue
            
            # Convert to binary classification (low vs high pain)
            binary_labels = []
            binary_windows = []
            
            for i, label in enumerate(labels):
                if label == 0:  # Low pain
                    binary_labels.append(0)
                    binary_windows.append(windows[i])
                elif label == 2:  # High pain
                    binary_labels.append(1)
                    binary_windows.append(windows[i])
                # Skip moderate pain (label == 1)
            
            all_data.extend(binary_windows)
            all_labels.extend(binary_labels)
            participants.extend([participant] * len(binary_windows))
            
            logging.info(f"Loaded {participant}: {len(binary_windows)} binary windows")
            
        except Exception as e:
            logging.error(f"Failed to load {participant}: {e}")
    
    return np.array(all_data), np.array(all_labels), np.array(participants)

def main():
    """Main advanced classifier pipeline with full dataset."""
    setup_logging()
    
    print("="*80)
    print("ADVANCED EEG PAIN CLASSIFIER - FULL DATASET")
    print("Wavelets + Connectivity + Hyperparameter Optimization + Ensemble")
    print("="*80)
    
    # Load data
    logging.info("Loading full dataset...")
    X_raw, y, participants = load_full_dataset()
    
    if X_raw is None or len(X_raw) == 0:
        logging.error("No data loaded. Exiting.")
        return
    
    unique_participants = np.unique(participants)
    logging.info(f"Loaded {len(X_raw)} windows from {len(unique_participants)} participants")
    logging.info(f"Class distribution: {np.bincount(y)}")
    
    # Initialize feature extractor
    feature_extractor = AdvancedFeatureExtractor()
    
    # Extract features for all windows
    logging.info("Extracting advanced features (this will take time for large dataset)...")
    start_time = time.time()
    
    all_features = []
    for i, window in enumerate(X_raw):
        if i % 100 == 0:
            elapsed = time.time() - start_time
            logging.info(f"Processing window {i+1}/{len(X_raw)} ({elapsed/60:.1f} minutes elapsed)")
        
        features = feature_extractor.extract_features(window)
        all_features.append(features)
    
    X_features = np.array(all_features)
    extraction_time = (time.time() - start_time) / 60
    logging.info(f"Feature extraction completed in {extraction_time:.1f} minutes")
    logging.info(f"Extracted {X_features.shape[1]} features")
    
    # Create binary dataset
    logging.info(f"Binary classification: {len(X_features)} samples")
    logging.info(f"Class distribution: {np.bincount(y)}")
    
    # Initialize ensemble classifier
    classifier = AdvancedClassifierEnsemble()
    
    # Train and evaluate
    logging.info("Starting hyperparameter optimization (this will take time)...")
    results = classifier.train_and_evaluate(X_features, y, participants)
    
    # Print results
    print("="*60)
    print("ADVANCED CLASSIFIER RESULTS - FULL DATASET")
    print("="*60)
    print(f"Features extracted: {X_features.shape[1]}")
    print(f"Samples: {len(X_features)} (Binary: Low vs High pain)")
    print(f"Participants: {len(unique_participants)}")
    print(f"Cross-validation accuracy: {results['ensemble_cv_score']:.3f} ± {results['ensemble_cv_std']:.3f}")
    print(f"Individual CV scores: {[f'{score:.3f}' for score in results['ensemble_cv_scores']]}")
    
    # Save results
    results_dir = Path("data/processed/advanced_full_dataset_results")
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save ensemble and results
    results_file = results_dir / f"advanced_full_ensemble_{timestamp}.pkl"
    with open(results_file, 'wb') as f:
        pickle.dump(results, f)
    
    # Save feature matrix
    feature_df = pd.DataFrame(X_features)
    feature_df['participant'] = participants
    feature_df['label'] = y
    features_file = results_dir / f"advanced_full_features_{timestamp}.csv"
    feature_df.to_csv(features_file, index=False)
    
    logging.info(f"Results saved to {results_dir}")
    logging.info("Advanced full dataset classifier pipeline completed!")

if __name__ == "__main__":
    main()
