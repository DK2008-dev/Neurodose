#!/usr/bin/env python3
"""
XGBoost Test on Full EEG Pain Classification Dataset
===================================================

Tests XGBoost performance on the complete 51-participant dataset
with Leave-One-Participant-Out Cross-Validation (LOPOCV).
"""

import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA
import xgboost as xgb
import time
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_participant_data(data_dir):
    """Load all participant data and combine into single dataset."""
    all_windows = []
    all_labels = []
    all_participants = []
    
    data_path = Path(data_dir)
    participant_files = sorted([f for f in data_path.glob("vp*_windows.pkl")])
    
    logger.info(f"Found {len(participant_files)} participant files")
    
    for file_path in participant_files:
        participant_id = file_path.stem.replace('_windows', '')
        
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            
            windows = data['windows']  # Shape: (n_windows, n_channels, n_samples)
            labels = data['labels']    # Pain intensity labels (0=low, 1=moderate, 2=high)
            
            # Flatten windows for traditional ML (n_windows, n_features)
            windows_flat = windows.reshape(windows.shape[0], -1)
            
            all_windows.append(windows_flat)
            all_labels.extend(labels)
            all_participants.extend([participant_id] * len(labels))
            
            logger.info(f"{participant_id}: {len(labels)} windows, labels: {np.bincount(labels)}")
            
        except Exception as e:
            logger.warning(f"Failed to load {file_path}: {e}")
            continue
    
    # Combine all data
    X = np.vstack(all_windows)
    y = np.array(all_labels)
    groups = np.array(all_participants)
    
    logger.info(f"Total dataset: {X.shape[0]} windows, {X.shape[1]} features")
    logger.info(f"Label distribution: {np.bincount(y)}")
    logger.info(f"Participants: {len(np.unique(groups))}")
    
    return X, y, groups

def apply_pca_reduction(X_train, X_test, n_components=100):
    """Apply PCA dimensionality reduction."""
    pca = PCA(n_components=n_components, random_state=42)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    
    explained_variance = np.sum(pca.explained_variance_ratio_)
    logger.info(f"PCA: {n_components} components explain {explained_variance:.3f} of variance")
    
    return X_train_pca, X_test_pca

def run_lopocv_evaluation(X, y, groups):
    """Run Leave-One-Participant-Out Cross-Validation with XGBoost."""
    logo = LeaveOneGroupOut()
    
    all_accuracies = []
    all_predictions = []
    all_true_labels = []
    fold_results = []
    
    logger.info("Starting LOPOCV evaluation with XGBoost...")
    start_time = time.time()
    
    for fold, (train_idx, test_idx) in enumerate(logo.split(X, y, groups)):
        test_participant = groups[test_idx[0]]
        
        # Split data
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Apply PCA for dimensionality reduction
        X_train_pca, X_test_pca = apply_pca_reduction(X_train_scaled, X_test_scaled, n_components=100)
        
        # Train XGBoost
        xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='mlogloss',
            verbosity=0
        )
        
        xgb_model.fit(X_train_pca, y_train)
        
        # Predict
        y_pred = xgb_model.predict(X_test_pca)
        accuracy = accuracy_score(y_test, y_pred)
        
        all_accuracies.append(accuracy)
        all_predictions.extend(y_pred)
        all_true_labels.extend(y_test)
        
        fold_results.append({
            'participant': test_participant,
            'accuracy': accuracy,
            'n_test_samples': len(y_test),
            'test_distribution': np.bincount(y_test, minlength=3).tolist()
        })
        
        logger.info(f"Fold {fold+1:2d}: {test_participant} - Accuracy: {accuracy:.3f} ({len(y_test)} samples)")
    
    elapsed_time = time.time() - start_time
    
    # Overall results
    mean_accuracy = np.mean(all_accuracies)
    std_accuracy = np.std(all_accuracies)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"XGBOOST LOPOCV RESULTS SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Mean Accuracy: {mean_accuracy:.3f} ± {std_accuracy:.3f}")
    logger.info(f"Accuracy Range: {np.min(all_accuracies):.3f} - {np.max(all_accuracies):.3f}")
    logger.info(f"Total Time: {elapsed_time:.1f} seconds")
    logger.info(f"Random Baseline: 33.3% (3-class)")
    
    # Classification report
    print("\nOverall Classification Report:")
    print(classification_report(all_true_labels, all_predictions, 
                              target_names=['Low', 'Moderate', 'High']))
    
    # Confusion Matrix
    cm = confusion_matrix(all_true_labels, all_predictions)
    print("\nConfusion Matrix:")
    print("Predicted:  Low  Mod  High")
    for i, row_label in enumerate(['Low', 'Moderate', 'High']):
        print(f"True {row_label:>6}: {cm[i]}")
    
    return {
        'mean_accuracy': mean_accuracy,
        'std_accuracy': std_accuracy,
        'all_accuracies': all_accuracies,
        'fold_results': fold_results,
        'predictions': all_predictions,
        'true_labels': all_true_labels,
        'elapsed_time': elapsed_time
    }

def analyze_performance_by_participant(fold_results):
    """Analyze performance patterns by participant."""
    logger.info(f"\n{'='*60}")
    logger.info(f"PERFORMANCE BY PARTICIPANT")
    logger.info(f"{'='*60}")
    
    # Sort by accuracy
    sorted_results = sorted(fold_results, key=lambda x: x['accuracy'], reverse=True)
    
    print(f"{'Participant':<12} {'Accuracy':<10} {'Samples':<8} {'Low':<4} {'Mod':<4} {'High':<4}")
    print("-" * 50)
    
    for result in sorted_results:
        dist = result['test_distribution']
        print(f"{result['participant']:<12} {result['accuracy']:<10.3f} {result['n_test_samples']:<8} "
              f"{dist[0]:<4} {dist[1]:<4} {dist[2]:<4}")
    
    # Identify best/worst performers
    best = sorted_results[0]
    worst = sorted_results[-1]
    
    logger.info(f"\nBest performer: {best['participant']} ({best['accuracy']:.3f})")
    logger.info(f"Worst performer: {worst['participant']} ({worst['accuracy']:.3f})")

def compare_with_random_forest(xgb_results):
    """Compare XGBoost results with Random Forest if available."""
    rf_results_file = "data/processed/quick_rf_test_results.pkl"
    
    if os.path.exists(rf_results_file):
        try:
            with open(rf_results_file, 'rb') as f:
                rf_results = pickle.load(f)
            
            logger.info(f"\n{'='*60}")
            logger.info(f"MODEL COMPARISON")
            logger.info(f"{'='*60}")
            logger.info(f"XGBoost LOPOCV:     {xgb_results['mean_accuracy']:.3f} ± {xgb_results['std_accuracy']:.3f}")
            logger.info(f"Random Forest LOPOCV: {rf_results['mean_accuracy']:.3f} ± {rf_results['std_accuracy']:.3f}")
            
            improvement = (xgb_results['mean_accuracy'] - rf_results['mean_accuracy']) * 100
            logger.info(f"XGBoost vs RF:      {improvement:+.1f} percentage points")
            
        except Exception as e:
            logger.warning(f"Could not load RF results for comparison: {e}")

def main():
    """Main execution function."""
    logger.info("Starting XGBoost Test on EEG Pain Classification Data")
    
    # Data directory
    data_dir = "data/processed/full_dataset"
    
    if not os.path.exists(data_dir):
        logger.error(f"Data directory not found: {data_dir}")
        return
    
    try:
        # Load data
        logger.info("Loading participant data...")
        X, y, groups = load_participant_data(data_dir)
        
        # Run LOPOCV evaluation
        results = run_lopocv_evaluation(X, y, groups)
        
        # Analyze results
        analyze_performance_by_participant(results['fold_results'])
        
        # Compare with Random Forest
        compare_with_random_forest(results)
        
        # Save results
        output_file = "data/processed/quick_xgb_test_results.pkl"
        with open(output_file, 'wb') as f:
            pickle.dump(results, f)
        logger.info(f"\nResults saved to: {output_file}")
        
        # Performance summary
        logger.info(f"\n{'='*60}")
        logger.info(f"FINAL SUMMARY")
        logger.info(f"{'='*60}")
        logger.info(f"XGBoost LOPOCV Accuracy: {results['mean_accuracy']:.1f}% ± {results['std_accuracy']:.1f}%")
        logger.info(f"Performance vs Random Baseline: {results['mean_accuracy']*100 - 33.3:+.1f}%")
        logger.info(f"Total Processing Time: {results['elapsed_time']:.1f} seconds")
        
    except Exception as e:
        logger.error(f"Error during execution: {e}")
        raise

if __name__ == "__main__":
    main()
