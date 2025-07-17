#!/usr/bin/env python3
"""
Quick Random Forest baseline for EEG pain classification.
Fast validation of data quality before CNN training.
"""

import numpy as np
import pickle
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_preprocessed_data():
    """Load all preprocessed participant data."""
    data_dir = Path('data/processed/basic_windows')
    
    all_windows = []
    all_labels = []
    all_participants = []
    
    participants = ['vp01', 'vp02', 'vp03', 'vp04', 'vp05']
    
    logger.info("Loading preprocessed data...")
    
    for participant in participants:
        file_path = data_dir / f'{participant}_windows.pkl'
        
        if file_path.exists():
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            
            windows = data['windows']  # Shape: (n_windows, n_channels, n_samples)
            labels = data['ternary_labels']  # Shape: (n_windows,)
            
            # Flatten windows for RF (RF can't handle 2D features directly)
            windows_flat = windows.reshape(windows.shape[0], -1)  # Shape: (n_windows, n_channels * n_samples)
            
            all_windows.append(windows_flat)
            all_labels.append(labels)
            all_participants.extend([participant] * len(labels))
            
            logger.info(f"{participant}: {len(labels)} windows, labels: {np.bincount(labels)}")
        else:
            logger.warning(f"File not found: {file_path}")
    
    if not all_windows:
        raise ValueError("No data files found!")
    
    X = np.vstack(all_windows)
    y = np.hstack(all_labels)
    participants = np.array(all_participants)
    
    logger.info(f"Total dataset: {len(X)} windows, {X.shape[1]} features")
    logger.info(f"Label distribution: {np.bincount(y)}")
    
    return X, y, participants

def extract_simple_features(X):
    """Extract simple statistical features for faster RF training."""
    logger.info("Extracting statistical features...")
    
    n_samples, n_features = X.shape
    n_channels = 68
    n_timepoints = n_features // n_channels
    
    # Reshape back to (samples, channels, timepoints)
    X_reshaped = X.reshape(n_samples, n_channels, n_timepoints)
    
    features = []
    
    for i in tqdm(range(n_samples), desc="Feature extraction"):
        sample_features = []
        
        for ch in range(n_channels):
            channel_data = X_reshaped[i, ch, :]
            
            # Statistical features per channel
            sample_features.extend([
                np.mean(channel_data),      # Mean
                np.std(channel_data),       # Standard deviation
                np.var(channel_data),       # Variance
                np.min(channel_data),       # Minimum
                np.max(channel_data),       # Maximum
                np.median(channel_data),    # Median
                np.percentile(channel_data, 25),  # 25th percentile
                np.percentile(channel_data, 75),  # 75th percentile
            ])
        
        features.append(sample_features)
    
    features_array = np.array(features)
    logger.info(f"Extracted features shape: {features_array.shape}")
    
    return features_array

def run_rf_cross_validation(X, y, participants):
    """Run Random Forest with Leave-One-Participant-Out cross-validation."""
    
    logger.info("\n🌲 Random Forest Baseline Training")
    logger.info("="*60)
    logger.info("• Fast training (~1-2 minutes)")
    logger.info("• Baseline performance for CNN comparison")
    logger.info("• Feature importance analysis")
    logger.info("="*60)
    
    # Use statistical features instead of raw data for speed
    X_features = extract_simple_features(X)
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_features)
    
    # Setup cross-validation
    logo = LeaveOneGroupOut()
    
    # Random Forest model
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1,  # Use all cores
        class_weight='balanced'
    )
    
    # Cross-validation results
    fold_results = []
    
    logger.info(f"\nRunning {logo.get_n_splits(X_scaled, y, participants)}-fold cross-validation...")
    
    for fold, (train_idx, test_idx) in enumerate(tqdm(logo.split(X_scaled, y, participants), 
                                                     desc="RF Cross-validation")):
        
        # Split data
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        test_participant = participants[test_idx][0]
        
        # Train model
        rf.fit(X_train, y_train)
        
        # Predict
        y_pred = rf.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average='macro')
        f1_weighted = f1_score(y_test, y_pred, average='weighted')
        
        fold_results.append({
            'fold': fold + 1,
            'test_participant': test_participant,
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'y_true': y_test,
            'y_pred': y_pred
        })
        
        logger.info(f"Fold {fold+1} ({test_participant}): Acc={accuracy:.3f}, F1={f1_weighted:.3f}")
    
    return fold_results, rf, scaler

def analyze_rf_results(fold_results):
    """Analyze Random Forest cross-validation results."""
    
    logger.info("\n📊 RANDOM FOREST RESULTS")
    logger.info("="*60)
    
    # Calculate summary statistics
    accuracies = [r['accuracy'] for r in fold_results]
    f1_scores = [r['f1_weighted'] for r in fold_results]
    
    logger.info(f"Accuracy:  {np.mean(accuracies):.3f} ± {np.std(accuracies):.3f}")
    logger.info(f"F1 Score:  {np.mean(f1_scores):.3f} ± {np.std(f1_scores):.3f}")
    logger.info(f"Range:     {np.min(accuracies):.3f} - {np.max(accuracies):.3f}")
    
    # Detailed classification report
    all_true = np.concatenate([r['y_true'] for r in fold_results])
    all_pred = np.concatenate([r['y_pred'] for r in fold_results])
    
    logger.info("\nDetailed Classification Report:")
    logger.info(classification_report(all_true, all_pred, 
                                    target_names=['Low Pain', 'Moderate Pain', 'High Pain']))
    
    # Compare to literature
    mean_acc = np.mean(accuracies)
    if mean_acc > 0.6:
        logger.info(f"✅ Good baseline! RF achieved {mean_acc:.1%} accuracy")
        logger.info("   Data quality is good - proceed with CNN training")
    elif mean_acc > 0.4:
        logger.info(f"⚠️  Moderate baseline: {mean_acc:.1%} accuracy")
        logger.info("   Data has signal but CNN should perform much better")
    else:
        logger.info(f"❌ Low baseline: {mean_acc:.1%} accuracy")
        logger.info("   May indicate data quality issues - investigate before CNN")
    
    return {
        'mean_accuracy': np.mean(accuracies),
        'std_accuracy': np.std(accuracies),
        'mean_f1': np.mean(f1_scores),
        'std_f1': np.std(f1_scores),
        'fold_results': fold_results
    }

def main():
    """Main function for Random Forest baseline."""
    
    # Load data
    X, y, participants = load_preprocessed_data()
    
    # Run Random Forest cross-validation
    fold_results, trained_rf, scaler = run_rf_cross_validation(X, y, participants)
    
    # Analyze results
    summary = analyze_rf_results(fold_results)
    
    # Save results for comparison
    results_path = Path('data/processed/rf_baseline_results.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump({
            'summary': summary,
            'fold_results': fold_results,
            'model': trained_rf,
            'scaler': scaler
        }, f)
    
    logger.info(f"\nResults saved to: {results_path}")
    logger.info(f"RF Baseline: {summary['mean_accuracy']:.1%} ± {summary['std_accuracy']:.1%}")
    logger.info("\n🚀 Ready for CNN training!")

if __name__ == '__main__':
    main()
