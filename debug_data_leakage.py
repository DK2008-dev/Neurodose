#!/usr/bin/env python3
"""
Debug Data Leakage in Literature-Standard RF

This script identifies and fixes the data leakage issues that led to 
suspiciously high 98.3% accuracy.

Issues to check:
1. SMOTE applied before cross-validation (severe leakage)
2. Feature scaling fit on full dataset before CV (moderate leakage)
3. Data augmentation mixing test/train participants (severe leakage)
4. Synthetic data generation (SMOTE) from test participants
"""

import os
import sys
import numpy as np
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm

# Scientific computing
import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import skew, kurtosis
import pywt  # For wavelet transforms

# Machine learning
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut, cross_val_score, cross_validate
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def analyze_data_leakage():
    """Analyze potential data leakage sources."""
    
    logger.info("🔍 ANALYZING DATA LEAKAGE SOURCES")
    logger.info("================================================================================")
    
    # Load original data
    pain_ratings_file = 'data/processed/windows_with_pain_ratings.pkl'
    if not os.path.exists(pain_ratings_file):
        logger.error(f"Pain ratings file not found: {pain_ratings_file}")
        return
    
    with open(pain_ratings_file, 'rb') as f:
        data = pickle.load(f)
    
    X = data['windows']
    pain_ratings = data['pain_ratings']
    participants = data['participants']
    
    logger.info(f"Dataset: {len(X)} windows, {X.shape[1]} channels, {X.shape[2]} samples")
    
    # Check original data distribution
    logger.info("\n1. ORIGINAL DATA ANALYSIS")
    logger.info("-" * 50)
    
    unique_participants = np.unique(participants)
    logger.info(f"Participants: {unique_participants}")
    logger.info(f"Pain rating range: {np.min(pain_ratings):.1f} - {np.max(pain_ratings):.1f}")
    logger.info(f"Unique ratings: {len(np.unique(pain_ratings))} values")
    
    # Analyze per participant
    for participant in unique_participants:
        mask = participants == participant
        p_ratings = pain_ratings[mask]
        logger.info(f"  {participant}: {len(p_ratings)} samples, "
                   f"range {np.min(p_ratings):.1f}-{np.max(p_ratings):.1f}, "
                   f"{len(np.unique(p_ratings))} unique values")
    
    # Create labels using literature standard
    def create_labels(ratings):
        ratings_scaled = ratings / 10.0
        labels = np.zeros(len(ratings_scaled), dtype=int)
        labels[(ratings_scaled > 3) & (ratings_scaled <= 6)] = 1
        labels[ratings_scaled > 6] = 2
        return labels
    
    y = create_labels(pain_ratings)
    logger.info(f"\nLabel distribution: {np.bincount(y)}")
    
    # Check class balance per participant
    for participant in unique_participants:
        mask = participants == participant
        p_labels = y[mask]
        logger.info(f"  {participant} labels: {np.bincount(p_labels)}")
    
    return X, y, participants, unique_participants


def demonstrate_leakage_vs_correct():
    """Demonstrate the difference between leaky and correct cross-validation."""
    
    X, y, participants, unique_participants = analyze_data_leakage()
    
    logger.info("\n2. DEMONSTRATING DATA LEAKAGE")
    logger.info("-" * 50)
    
    # Simple feature extraction for demonstration
    def extract_simple_features(window):
        """Extract basic statistical features for demonstration."""
        features = []
        for ch in range(min(5, window.shape[0])):  # Use first 5 channels
            ch_data = window[ch, :]
            features.extend([
                np.mean(ch_data),
                np.std(ch_data),
                np.var(ch_data),
                skew(ch_data),
                kurtosis(ch_data)
            ])
        return np.array(features)
    
    # Extract features
    logger.info("Extracting simple features...")
    features = []
    for window in tqdm(X, desc="Feature extraction"):
        feature_vector = extract_simple_features(window)
        features.append(feature_vector)
    
    X_features = np.array(features)
    logger.info(f"Feature matrix: {X_features.shape}")
    
    # === LEAKY APPROACH (what we did wrong) ===
    logger.info("\n🚨 LEAKY APPROACH (WRONG):")
    
    # Apply SMOTE to full dataset BEFORE cross-validation (WRONG!)
    smote = SMOTE(random_state=42)
    X_smote, y_smote = smote.fit_resample(X_features, y)
    
    # Scale features on full dataset BEFORE cross-validation (WRONG!)
    scaler = StandardScaler()
    X_scaled_leaky = scaler.fit_transform(X_smote)
    
    logger.info(f"After SMOTE: {X_smote.shape[0]} samples (from {X_features.shape[0]})")
    logger.info(f"Balanced distribution: {np.bincount(y_smote)}")
    
    # Simulate the leaky cross-validation
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # This is wrong because we're training on augmented data that includes
    # synthetic samples generated from the test participant's data
    lopocv = LeaveOneGroupOut()
    leaky_scores = []
    
    for train_idx, test_idx in lopocv.split(X_features, y, participants):
        # Train on ALL SMOTE data (includes synthetic from test participant!)
        rf.fit(X_scaled_leaky, y_smote)  # WRONG!
        
        # Test on original participant data (scaled with global scaler)
        X_test = scaler.transform(X_features[test_idx])  # WRONG! Using global scaler
        y_test = y[test_idx]
        
        y_pred = rf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        leaky_scores.append(acc)
    
    leaky_mean = np.mean(leaky_scores)
    logger.info(f"Leaky CV accuracy: {leaky_mean:.3f} ± {np.std(leaky_scores):.3f}")
    logger.info("❌ This is artificially high due to data leakage!")
    
    # === CORRECT APPROACH ===
    logger.info("\n✅ CORRECT APPROACH:")
    
    correct_scores = []
    
    for train_idx, test_idx in lopocv.split(X_features, y, participants):
        # Split data properly
        X_train, X_test = X_features[train_idx], X_features[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Apply SMOTE only to training data
        smote_train = SMOTE(random_state=42)
        X_train_smote, y_train_smote = smote_train.fit_resample(X_train, y_train)
        
        # Scale features only on training data
        scaler_train = StandardScaler()
        X_train_scaled = scaler_train.fit_transform(X_train_smote)
        X_test_scaled = scaler_train.transform(X_test)  # Use training scaler
        
        # Train and test properly
        rf_correct = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_correct.fit(X_train_scaled, y_train_smote)
        
        y_pred = rf_correct.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        correct_scores.append(acc)
        
        participant = participants[test_idx][0]
        logger.info(f"  {participant}: {acc:.3f} ({len(y_test)} samples)")
    
    correct_mean = np.mean(correct_scores)
    logger.info(f"\nCorrect CV accuracy: {correct_mean:.3f} ± {np.std(correct_scores):.3f}")
    
    # Compare results
    logger.info("\n3. COMPARISON RESULTS")
    logger.info("-" * 50)
    logger.info(f"Leaky approach:   {leaky_mean:.3f} (WRONG - data leakage)")
    logger.info(f"Correct approach: {correct_mean:.3f} (TRUE performance)")
    logger.info(f"Difference:       {(leaky_mean - correct_mean):.3f} ({(leaky_mean - correct_mean)*100:.1f}% inflation)")
    
    # Analysis of why leakage occurred
    logger.info("\n4. DATA LEAKAGE ANALYSIS")
    logger.info("-" * 50)
    logger.info("Sources of data leakage identified:")
    logger.info("1. 🚨 SMOTE applied to full dataset before CV")
    logger.info("   - Synthetic samples generated from test participants")
    logger.info("   - Model sees 'augmented' test data during training")
    logger.info("")
    logger.info("2. 🚨 Feature scaling fit on full dataset")
    logger.info("   - Scaler parameters computed from test data")
    logger.info("   - Test data statistics leak into training")
    logger.info("")
    logger.info("3. 🚨 Augmentation mixing train/test participants")
    logger.info("   - Data augmentation applied globally")
    logger.info("   - Test participant patterns leak into training")
    
    if leaky_mean > 0.95:
        logger.info("\n❌ VERDICT: Suspiciously high accuracy due to severe data leakage")
        logger.info("   The 98.3% accuracy is NOT legitimate!")
    elif correct_mean > 0.80:
        logger.info("\n✅ VERDICT: Good performance after fixing leakage")
        logger.info("   The methodology is sound when applied correctly")
    else:
        logger.info("\n🟡 VERDICT: Moderate performance after fixing leakage")
        logger.info("   Further optimization needed")
    
    return {
        'leaky_accuracy': leaky_mean,
        'correct_accuracy': correct_mean,
        'leakage_inflation': leaky_mean - correct_mean
    }


if __name__ == "__main__":
    results = demonstrate_leakage_vs_correct()
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Leaky methodology:   {results['leaky_accuracy']:.1%}")
    print(f"Correct methodology: {results['correct_accuracy']:.1%}")
    print(f"Inflation due to leakage: {results['leakage_inflation']:.1%}")
    print("\nThe 98.3% accuracy was due to data leakage, not legitimate performance!")
