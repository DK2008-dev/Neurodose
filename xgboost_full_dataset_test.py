#!/usr/bin/env python3
"""
XGBoost Test on Full 51-Participant Dataset

This script tests XGBoost performance on the complete processed dataset
using the original methodology from the literature.
"""

import os
import sys
import numpy as np
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import time

# Scientific computing
import numpy as np
from scipy.signal import welch

# Machine learning
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, LeaveOneGroupOut
from sklearn.metrics import f1_score, roc_auc_score, classification_report, accuracy_score, confusion_matrix
import optuna
from xgboost import XGBClassifier

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ───────── CONFIG (from original code) ─────────
SF = 500
WINDOWS = [(0.0, 0.16), (0.16, 0.3), (0.3, 1.0)]
BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30), "gamma": (30, 90)}


def spectral_entropy(psd):
    """Spectral entropy calculation (from original code)."""
    p = psd / (psd.sum(axis=1, keepdims=True) + 1e-12)
    return -np.sum(p * np.log(p + 1e-12), axis=1) / np.log(p.shape[1])


def extract_features_original_method(epoch_data, sf=500):
    """
    Extract features using the exact method from the original XGBoost code.
    
    Args:
        epoch_data: EEG data (channels, samples) - 1 second epoch
        sf: Sampling frequency
        
    Returns:
        Feature vector
    """
    feats = []
    
    # Extract features for each time window
    for s0, s1 in WINDOWS:
        i0, i1 = int(s0 * sf), int(s1 * sf)
        if i1 > epoch_data.shape[1]:
            i1 = epoch_data.shape[1]
        if i0 >= i1:
            continue
            
        seg = epoch_data[:, i0:i1]
        if seg.shape[1] < 10:  # Skip very short segments
            continue
            
        freqs, psd = welch(seg, sf, nperseg=min(seg.shape[1], 256), axis=1)
        
        for lo, hi in BANDS.values():
            idx = (freqs >= lo) & (freqs <= hi)
            if np.any(idx):
                feats.append(psd[:, idx].mean(axis=1))
            else:
                feats.append(np.zeros(psd.shape[0]))
    
    # Full epoch features
    f_full, full_psd = welch(epoch_data, sf, nperseg=min(epoch_data.shape[1], 256), axis=1)
    
    # Band powers for ratios
    bm = {}
    for b, (lo, hi) in BANDS.items():
        idx = (f_full >= lo) & (f_full <= hi)
        if np.any(idx):
            bm[b] = full_psd[:, idx].mean(axis=1)
        else:
            bm[b] = np.zeros(full_psd.shape[0])
    
    # Band ratios (from original code)
    feats.append(bm["gamma"] / (bm["alpha"] + 1e-12))
    feats.append(bm["delta"] / (bm["theta"] + 1e-12))
    
    # Spectral entropy
    feats.append(spectral_entropy(full_psd))
    
    return np.hstack(feats)


def load_full_dataset():
    """Load the complete 51-participant dataset."""
    
    logger.info("🔄 Loading complete 51-participant dataset...")
    
    data_dir = Path("data/processed/full_dataset")
    if not data_dir.exists():
        logger.error(f"Dataset directory not found: {data_dir}")
        return None, None, None, None
    
    # Load all participant files
    participant_files = sorted(list(data_dir.glob("vp*_windows.pkl")))
    logger.info(f"Found {len(participant_files)} participant files")
    
    all_windows = []
    all_labels = []
    all_participants = []
    all_pain_ratings = []
    
    for file_path in participant_files:
        participant_id = file_path.stem.replace('_windows', '')
        
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            
            windows = data['windows']  # Shape: (n_windows, n_channels, n_samples)
            labels = data['labels']    # Pain intensity labels (0=low, 1=moderate, 2=high)
            
            # Skip participants with no windows (artifacts)
            if len(windows) == 0:
                logger.warning(f"{participant_id}: Skipping - no windows (excessive artifacts)")
                continue
            
            # Convert labels back to original pain ratings (approximate)
            # This is a rough conversion since we don't have exact ratings
            pain_ratings = []
            for label in labels:
                if label == 0:      # Low pain
                    pain_ratings.append(20)   # Representative low value
                elif label == 1:    # Moderate pain
                    pain_ratings.append(40)   # Representative moderate value
                else:               # High pain
                    pain_ratings.append(60)   # Representative high value
            
            all_windows.append(windows)
            all_labels.extend(labels)
            all_participants.extend([participant_id] * len(labels))
            all_pain_ratings.extend(pain_ratings)
            
            logger.info(f"{participant_id}: {len(labels)} windows, labels: {np.bincount(labels, minlength=3)}")
            
        except Exception as e:
            logger.warning(f"Failed to load {file_path}: {e}")
            continue
    
    if not all_windows:
        logger.error("No participant data loaded!")
        return None, None, None, None
    
    # Combine all data
    X_windows = np.vstack(all_windows)
    y_labels = np.array(all_labels)
    participants = np.array(all_participants)
    pain_ratings = np.array(all_pain_ratings)
    
    logger.info(f"Total dataset: {X_windows.shape[0]} windows, {X_windows.shape[1]} channels, {X_windows.shape[2]} samples")
    logger.info(f"Label distribution: {np.bincount(y_labels)}")
    logger.info(f"Participants: {len(np.unique(participants))}")
    
    return X_windows, y_labels, participants, pain_ratings


def process_for_xgboost(X_windows, pain_ratings, participants):
    """Process data for XGBoost using original methodology."""
    
    logger.info("🔧 Applying original XGBoost preprocessing...")
    logger.info("Binary classification: ≤30 = low (0), ≥50 = high (1), exclude 31-49")
    
    X_features = []
    y_binary = []
    participants_filtered = []
    
    for i, (window, rating, participant) in enumerate(zip(X_windows, pain_ratings, participants)):
        # Apply original binary classification logic
        if rating <= 30:
            cls = 0  # Low pain
        elif rating >= 50:
            cls = 1  # High pain
        else:
            continue  # Exclude moderate pain (31-49)
        
        # Use the first 1 second (500 samples) to match original approach
        epoch_1s = window[:, :500]  # First 1 second
        
        # Extract features using original method
        try:
            features = extract_features_original_method(epoch_1s, sf=SF)
            X_features.append(features)
            y_binary.append(cls)
            participants_filtered.append(participant)
        except Exception as e:
            logger.warning(f"Feature extraction failed for sample {i}: {e}")
            continue
    
    if len(X_features) == 0:
        logger.error("No valid features extracted!")
        return None, None, None
    
    X_features = np.vstack(X_features)
    y_binary = np.array(y_binary)
    participants_filtered = np.array(participants_filtered)
    
    logger.info(f"After binary filtering: {X_features.shape[0]} samples")
    logger.info(f"Feature dimension: {X_features.shape[1]}")
    logger.info(f"Class distribution - Low (0): {np.sum(y_binary == 0)}, High (1): {np.sum(y_binary == 1)}")
    
    # Show per-participant distribution
    unique_participants = np.unique(participants_filtered)
    logger.info(f"Participants after filtering: {len(unique_participants)}")
    
    for participant in unique_participants[:10]:  # Show first 10
        mask = participants_filtered == participant
        p_labels = y_binary[mask]
        logger.info(f"  {participant}: {len(p_labels)} samples, "
                   f"Low: {np.sum(p_labels == 0)}, High: {np.sum(p_labels == 1)}")
    
    return X_features, y_binary, participants_filtered


def objective(trial, X_tr, y_tr):
    """Optuna objective function (from original code)."""
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 300),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "subsample": trial.suggest_float("subsample", 0.7, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
        "gamma": trial.suggest_float("gamma", 0, 3),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 3),
        "reg_lambda": trial.suggest_float("reg_lambda", 1, 5)
    }
    
    model = XGBClassifier(**params, random_state=42, n_jobs=-1, verbosity=0)
    
    # Simple train/validation split for optimization
    X_train, X_val, y_train, y_val = train_test_split(X_tr, y_tr, test_size=0.2, 
                                                      random_state=42, stratify=y_tr)
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    return f1_score(y_val, y_pred)


def run_xgboost_experiment():
    """Run the complete XGBoost experiment."""
    
    start_time = time.time()
    
    # Load data
    X_windows, y_labels, participants, pain_ratings = load_full_dataset()
    if X_windows is None:
        return
    
    # Process for XGBoost
    X_features, y_binary, participants_filtered = process_for_xgboost(X_windows, pain_ratings, participants)
    if X_features is None:
        return
    
    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    X_features = imputer.fit_transform(X_features)
    
    logger.info("🎯 Starting XGBoost evaluation...")
    
    # 1. Simple train/test split (80/20) - like original validation
    logger.info("\n" + "="*60)
    logger.info("SIMPLE TRAIN/TEST SPLIT (80/20)")
    logger.info("="*60)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_features, y_binary, test_size=0.2, random_state=42, stratify=y_binary
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Hyperparameter optimization
    logger.info("Optimizing hyperparameters with Optuna...")
    study = optuna.create_study(direction='maximize', 
                               sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(lambda trial: objective(trial, X_train_scaled, y_train), 
                   n_trials=40, show_progress_bar=False)
    
    best_params = study.best_params
    logger.info(f"Best parameters: {best_params}")
    
    # Train final model
    final_model = XGBClassifier(**best_params, random_state=42, n_jobs=-1)
    final_model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = final_model.predict(X_test_scaled)
    y_pred_proba = final_model.predict_proba(X_test_scaled)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    logger.info(f"Simple Split Results:")
    logger.info(f"  Accuracy: {accuracy:.3f}")
    logger.info(f"  F1-Score: {f1:.3f}")
    logger.info(f"  AUC: {auc:.3f}")
    
    # 2. Leave-One-Participant-Out Cross-Validation
    logger.info("\n" + "="*60)
    logger.info("LEAVE-ONE-PARTICIPANT-OUT CROSS-VALIDATION")
    logger.info("="*60)
    
    logo = LeaveOneGroupOut()
    cv_accuracies = []
    cv_f1_scores = []
    
    fold = 0
    for train_idx, test_idx in logo.split(X_features, y_binary, participants_filtered):
        fold += 1
        test_participant = participants_filtered[test_idx[0]]
        
        X_train_cv, X_test_cv = X_features[train_idx], X_features[test_idx]
        y_train_cv, y_test_cv = y_binary[train_idx], y_binary[test_idx]
        
        # Check if test set has both classes
        if len(np.unique(y_test_cv)) < 2:
            logger.warning(f"Fold {fold} ({test_participant}): Only one class in test set, skipping...")
            continue
        
        # Scale features
        scaler_cv = StandardScaler()
        X_train_cv_scaled = scaler_cv.fit_transform(X_train_cv)
        X_test_cv_scaled = scaler_cv.transform(X_test_cv)
        
        # Train model with best params from simple split
        model_cv = XGBClassifier(**best_params, random_state=42, n_jobs=-1)
        model_cv.fit(X_train_cv_scaled, y_train_cv)
        
        # Predict
        y_pred_cv = model_cv.predict(X_test_cv_scaled)
        
        accuracy_cv = accuracy_score(y_test_cv, y_pred_cv)
        f1_cv = f1_score(y_test_cv, y_pred_cv, average='binary')
        
        cv_accuracies.append(accuracy_cv)
        cv_f1_scores.append(f1_cv)
        
        logger.info(f"Fold {fold:2d} ({test_participant}): Accuracy: {accuracy_cv:.3f}, F1: {f1_cv:.3f} "
                   f"({len(y_test_cv)} samples, {np.sum(y_test_cv == 0)}/{np.sum(y_test_cv == 1)} low/high)")
    
    # Final results
    logger.info("\n" + "="*60)
    logger.info("FINAL RESULTS SUMMARY")
    logger.info("="*60)
    
    mean_accuracy = np.mean(cv_accuracies)
    std_accuracy = np.std(cv_accuracies)
    mean_f1 = np.mean(cv_f1_scores)
    std_f1 = np.std(cv_f1_scores)
    
    elapsed_time = time.time() - start_time
    
    logger.info(f"Simple Train/Test Split (80/20):")
    logger.info(f"  Accuracy: {accuracy:.3f}")
    logger.info(f"  F1-Score: {f1:.3f}")
    logger.info(f"  AUC: {auc:.3f}")
    
    logger.info(f"\nLeave-One-Participant-Out CV:")
    logger.info(f"  Mean Accuracy: {mean_accuracy:.3f} ± {std_accuracy:.3f}")
    logger.info(f"  Mean F1-Score: {mean_f1:.3f} ± {std_f1:.3f}")
    logger.info(f"  Accuracy Range: {np.min(cv_accuracies):.3f} - {np.max(cv_accuracies):.3f}")
    
    logger.info(f"\nComparison to Literature:")
    logger.info(f"  Literature benchmark: ~87% accuracy")
    logger.info(f"  Our simple split: {accuracy*100:.1f}% accuracy")
    logger.info(f"  Our LOPOCV: {mean_accuracy*100:.1f}% ± {std_accuracy*100:.1f}% accuracy")
    logger.info(f"  Random baseline: 50% (binary classification)")
    
    logger.info(f"\nProcessing Time: {elapsed_time:.1f} seconds")
    
    # Save results
    results = {
        'simple_split': {
            'accuracy': accuracy,
            'f1_score': f1,
            'auc': auc,
            'best_params': best_params
        },
        'lopocv': {
            'mean_accuracy': mean_accuracy,
            'std_accuracy': std_accuracy,
            'mean_f1': mean_f1,
            'std_f1': std_f1,
            'all_accuracies': cv_accuracies,
            'all_f1_scores': cv_f1_scores
        },
        'dataset_info': {
            'total_samples': len(X_features),
            'n_features': X_features.shape[1],
            'n_participants': len(np.unique(participants_filtered)),
            'class_distribution': {
                'low': np.sum(y_binary == 0),
                'high': np.sum(y_binary == 1)
            }
        },
        'processing_time': elapsed_time
    }
    
    output_file = "data/processed/xgboost_full_dataset_results.pkl"
    with open(output_file, 'wb') as f:
        pickle.dump(results, f)
    logger.info(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    logger.info("Starting XGBoost Test on Full 51-Participant Dataset")
    run_xgboost_experiment()
