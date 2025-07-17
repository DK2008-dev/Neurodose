#!/usr/bin/env python3
"""
XGBoost Binary Classification Test

This script adapts the provided XGBoost code to work with our preprocessed data
to determine if our preprocessing pipeline is causing performance issues.

Key differences from original:
1. Uses our preprocessed data instead of raw files
2. Maintains their exact feature extraction methodology
3. Uses their binary classification approach (low vs high, excluding moderate)
4. Applies their Optuna hyperparameter optimization
"""

import os
import sys
import numpy as np
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import re

# Scientific computing
import numpy as np
from scipy.signal import welch

# Machine learning
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, LeaveOneGroupOut
from sklearn.metrics import f1_score, roc_auc_score, classification_report, accuracy_score
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


def load_and_process_our_data():
    """Load our preprocessed data and apply original XGBoost feature extraction."""
    
    logger.info("🔄 Loading our preprocessed data...")
    
    # Load our preprocessed data
    pain_ratings_file = 'data/processed/windows_with_pain_ratings.pkl'
    if not os.path.exists(pain_ratings_file):
        logger.error(f"Pain ratings file not found: {pain_ratings_file}")
        return None, None, None
    
    with open(pain_ratings_file, 'rb') as f:
        data = pickle.load(f)
    
    X_windows = data['windows']  # (n_samples, channels, 2000 samples = 4 seconds)
    pain_ratings = data['pain_ratings']
    participants = data['participants']
    
    logger.info(f"Loaded: {len(X_windows)} windows, {X_windows.shape[1]} channels, {X_windows.shape[2]} samples")
    logger.info(f"Pain rating range: {np.min(pain_ratings):.1f} - {np.max(pain_ratings):.1f}")
    
    # Apply original XGBoost binary classification logic
    # Original: rating<=30 = class 0 (low), rating>=50 = class 1 (high), exclude middle
    X_features = []
    y_binary = []
    participants_filtered = []
    
    logger.info("🔧 Applying original XGBoost preprocessing...")
    logger.info("Binary classification: ≤30 = low (0), ≥50 = high (1), exclude 31-49")
    
    kept_indices = []
    
    for i, (window, rating, participant) in enumerate(zip(X_windows, pain_ratings, participants)):
        # Apply original binary classification logic
        if rating <= 30:
            cls = 0  # Low pain
        elif rating >= 50:
            cls = 1  # High pain
        else:
            continue  # Exclude moderate pain (31-49)
        
        # The original code uses 1-second epochs, but our windows are 4 seconds
        # Let's use the first 1 second (500 samples) to match their approach
        epoch_1s = window[:, :500]  # First 1 second
        
        # Extract features using original method
        try:
            features = extract_features_original_method(epoch_1s, sf=SF)
            X_features.append(features)
            y_binary.append(cls)
            participants_filtered.append(participant)
            kept_indices.append(i)
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
    for participant in unique_participants:
        mask = participants_filtered == participant
        p_labels = y_binary[mask]
        logger.info(f"  {participant}: {len(p_labels)} samples, "
                   f"Low: {np.sum(p_labels == 0)}, High: {np.sum(p_labels == 1)}")
    
    return X_features, y_binary, participants_filtered


def objective(trial, X_tr, y_tr):
    """Optuna objective function (from original code)."""
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_loguniform("learning_rate", 1e-3, 0.3),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 5),
        "reg_lambda": trial.suggest_float("reg_lambda", 1, 10)
    }
    
    model = XGBClassifier(**params,
                          use_label_encoder=False,
                          eval_metric="logloss",
                          random_state=42)
    
    # Simple train/valid split for tuning
    if len(np.unique(y_tr)) < 2:
        return 0.0  # Can't train with only one class
    
    try:
        Xt, Xv, yt, yv = train_test_split(X_tr, y_tr,
                                          test_size=0.25,
                                          stratify=y_tr,
                                          random_state=42)
        
        model.fit(Xt, yt,
                  early_stopping_rounds=20,
                  eval_set=[(Xv, yv)],
                  verbose=False)
        
        preds = model.predict(Xv)
        return f1_score(yv, preds)
    except Exception:
        return 0.0


def run_xgboost_test():
    """Run the XGBoost test using original methodology on our data."""
    
    logger.info("🧪 XGBOOST TEST: Original Methodology on Our Data")
    logger.info("=" * 80)
    logger.info("Testing if our preprocessing is the performance bottleneck")
    logger.info("Using exact XGBoost approach that achieved ~87% accuracy")
    logger.info("=" * 80)
    
    # Load and process data
    X, y, participants = load_and_process_our_data()
    if X is None:
        return
    
    # Apply original preprocessing
    logger.info("\n🔧 Applying original preprocessing...")
    imp = SimpleImputer(strategy="mean")
    X = imp.fit_transform(X)
    
    # Test 1: Simple train/test split (like original)
    logger.info("\n📊 TEST 1: Simple Train/Test Split (80/20)")
    logger.info("-" * 50)
    
    try:
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                                 stratify=y, random_state=42)
        
        sc = StandardScaler().fit(X_tr)
        X_tr_scaled = sc.transform(X_tr)
        X_te_scaled = sc.transform(X_te)
        
        logger.info(f"Training set: {len(X_tr)} samples, Low: {np.sum(y_tr == 0)}, High: {np.sum(y_tr == 1)}")
        logger.info(f"Test set: {len(X_te)} samples, Low: {np.sum(y_te == 0)}, High: {np.sum(y_te == 1)}")
        
        # Optuna optimization
        logger.info("Running Optuna hyperparameter optimization...")
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: objective(trial, X_tr_scaled, y_tr), 
                      n_trials=40, n_jobs=1)
        
        logger.info(f"Best trial: {study.best_trial.params}")
        
        # Final evaluation
        best = XGBClassifier(**study.best_trial.params,
                           use_label_encoder=False,
                           eval_metric="logloss",
                           random_state=42)
        best.fit(X_tr_scaled, y_tr)
        y_pred = best.predict(X_te_scaled)
        y_prob = best.predict_proba(X_te_scaled)[:, 1]
        
        logger.info("\n📈 RESULTS - Simple Split:")
        logger.info(classification_report(y_te, y_pred, digits=3))
        logger.info(f"AUC-ROC: {roc_auc_score(y_te, y_prob):.3f}")
        
        simple_accuracy = accuracy_score(y_te, y_pred)
        simple_auc = roc_auc_score(y_te, y_prob)
        
    except Exception as e:
        logger.error(f"Simple split test failed: {e}")
        simple_accuracy = 0.0
        simple_auc = 0.0
    
    # Test 2: Leave-One-Participant-Out Cross-Validation
    logger.info("\n📊 TEST 2: Leave-One-Participant-Out Cross-Validation")
    logger.info("-" * 50)
    
    unique_participants = np.unique(participants)
    if len(unique_participants) < 2:
        logger.warning("Not enough participants for LOPOCV")
        lopocv_scores = []
        lopocv_mean = 0.0
    else:
        lopocv = LeaveOneGroupOut()
        lopocv_scores = []
        
        for fold, (train_idx, test_idx) in enumerate(lopocv.split(X, y, participants)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            test_participant = participants[test_idx][0]
            
            # Check if we have both classes in training
            if len(np.unique(y_train)) < 2:
                logger.warning(f"Fold {fold+1} ({test_participant}): Only one class in training, skipping")
                continue
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Use best parameters from previous optimization (simplified for speed)
            best_params = study.best_trial.params if 'study' in locals() else {
                'n_estimators': 200, 'max_depth': 6, 'learning_rate': 0.1
            }
            
            model = XGBClassifier(**best_params,
                                use_label_encoder=False,
                                eval_metric="logloss",
                                random_state=42)
            
            try:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                acc = accuracy_score(y_test, y_pred)
                lopocv_scores.append(acc)
                
                logger.info(f"  {test_participant}: {acc:.3f} accuracy ({len(y_test)} samples)")
            except Exception as e:
                logger.warning(f"Fold {fold+1} failed: {e}")
        
        lopocv_mean = np.mean(lopocv_scores) if lopocv_scores else 0.0
        lopocv_std = np.std(lopocv_scores) if lopocv_scores else 0.0
        
        logger.info(f"\n✅ LOPOCV Results: {lopocv_mean:.3f} ± {lopocv_std:.3f}")
        logger.info(f"Range: {np.min(lopocv_scores):.3f} - {np.max(lopocv_scores):.3f}")
    
    # Final comparison
    logger.info("\n" + "=" * 80)
    logger.info("FINAL COMPARISON")
    logger.info("=" * 80)
    logger.info(f"Original XGBoost (reported):     ~87% accuracy")
    logger.info(f"Our data + Original method:")
    logger.info(f"  Simple split (80/20):          {simple_accuracy:.1%}")
    logger.info(f"  LOPOCV (participant-independent): {lopocv_mean:.1%}")
    logger.info(f"  AUC-ROC (simple split):        {simple_auc:.3f}")
    
    if simple_accuracy > 0.8:
        logger.info("✅ SUCCESS: Our preprocessing is GOOD! Performance matches literature.")
    elif simple_accuracy > 0.6:
        logger.info("🟡 PARTIAL: Decent performance, some optimization needed.")
    else:
        logger.info("❌ ISSUE: Low performance suggests preprocessing problems.")
    
    if lopocv_mean < simple_accuracy - 0.2:
        logger.info("⚠️  WARNING: Large gap between simple split and LOPOCV suggests overfitting.")
    
    # Save results
    results = {
        'methodology': 'XGBoost Binary (Original Method)',
        'simple_split_accuracy': simple_accuracy,
        'simple_split_auc': simple_auc,
        'lopocv_scores': lopocv_scores,
        'lopocv_mean': lopocv_mean,
        'n_samples_total': len(X),
        'n_features': X.shape[1],
        'class_distribution': np.bincount(y)
    }
    
    output_file = 'data/processed/xgboost_original_method_results.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump(results, f)
    
    logger.info(f"\n💾 Results saved to: {output_file}")
    
    return results


if __name__ == "__main__":
    # Install xgboost if not available
    try:
        import xgboost
    except ImportError:
        logger.info("Installing XGBoost...")
        os.system("pip install xgboost")
        import xgboost
    
    try:
        import optuna
    except ImportError:
        logger.info("Installing Optuna...")
        os.system("pip install optuna")
        import optuna
    
    results = run_xgboost_test()
    
    if results:
        print("\n" + "="*80)
        print("XGBOOST ORIGINAL METHOD TEST SUMMARY")
        print("="*80)
        print(f"Simple split accuracy: {results['simple_split_accuracy']:.1%}")
        print(f"LOPOCV accuracy: {results['lopocv_mean']:.1%}")
        print(f"Literature benchmark: ~87%")
        print("\nThis test determines if our preprocessing is the bottleneck!")
