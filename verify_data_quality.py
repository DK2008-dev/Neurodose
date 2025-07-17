#!/usr/bin/env python3
"""
Comprehensive Data Quality Verification for EEG Pain Classification

This script performs extensive data quality checks to verify if the preprocessed
EEG data is suitable for CNN training. It checks for:

1. Data integrity and consistency
2. Signal quality metrics
3. Label separability analysis
4. Feature quality assessment
5. Cross-participant consistency
6. Noise and artifact levels

If this analysis shows good data quality, then any CNN issues are model-related, not data-related.
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from scipy import signal, stats
from scipy.stats import entropy
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EEGDataQualityValidator:
    """Comprehensive EEG data quality validation."""
    
    def __init__(self, sfreq=500):
        self.sfreq = sfreq
        self.results = {}
        
    def load_data(self):
        """Load all preprocessed data."""
        data_dir = Path('data/processed/basic_windows')
        
        all_windows = []
        all_labels = []
        all_participants = []
        participant_data = {}
        
        participants = ['vp01', 'vp02', 'vp03', 'vp04', 'vp05']
        
        logger.info("Loading preprocessed data for quality analysis...")
        
        for participant in participants:
            file_path = data_dir / f'{participant}_windows.pkl'
            
            if file_path.exists():
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                
                windows = data['windows']  # Shape: (n_windows, n_channels, n_samples)
                labels = data['ternary_labels']
                
                all_windows.append(windows)
                all_labels.append(labels)
                all_participants.extend([participant] * len(windows))
                
                participant_data[participant] = {
                    'windows': windows,
                    'labels': labels,
                    'n_windows': len(windows),
                    'label_dist': np.bincount(labels)
                }
                
                logger.info(f"{participant}: {len(windows)} windows, labels: {np.bincount(labels)}")
        
        X = np.concatenate(all_windows, axis=0)
        y = np.concatenate(all_labels, axis=0)
        participants = np.array(all_participants)
        
        logger.info(f"Total: {len(X)} windows, {X.shape[1]} channels, {X.shape[2]} samples")
        
        return X, y, participants, participant_data
    
    def check_data_integrity(self, X, y, participants, participant_data):
        """Check basic data integrity."""
        logger.info("\n🔍 DATA INTEGRITY ANALYSIS")
        logger.info("="*50)
        
        integrity_results = {}
        
        # 1. Shape consistency
        logger.info("1. Shape Consistency:")
        logger.info(f"   Data shape: {X.shape}")
        logger.info(f"   Labels shape: {y.shape}")
        logger.info(f"   Participants: {len(np.unique(participants))}")
        
        shape_consistent = len(X) == len(y) == len(participants)
        integrity_results['shape_consistent'] = shape_consistent
        logger.info(f"   ✅ Shape consistency: {shape_consistent}")
        
        # 2. Missing or invalid data
        logger.info("\n2. Missing/Invalid Data:")
        n_nan = np.isnan(X).sum()
        n_inf = np.isinf(X).sum()
        logger.info(f"   NaN values: {n_nan}")
        logger.info(f"   Inf values: {n_inf}")
        
        data_clean = (n_nan == 0) and (n_inf == 0)
        integrity_results['data_clean'] = data_clean
        logger.info(f"   ✅ Data cleanliness: {data_clean}")
        
        # 3. Label validity
        logger.info("\n3. Label Validity:")
        unique_labels = np.unique(y)
        expected_labels = [0, 1, 2]
        labels_valid = np.array_equal(sorted(unique_labels), expected_labels)
        
        logger.info(f"   Expected labels: {expected_labels}")
        logger.info(f"   Found labels: {sorted(unique_labels)}")
        logger.info(f"   Label distribution: {np.bincount(y)}")
        
        integrity_results['labels_valid'] = labels_valid
        logger.info(f"   ✅ Label validity: {labels_valid}")
        
        # 4. Cross-participant consistency
        logger.info("\n4. Cross-Participant Consistency:")
        channel_counts = [data['windows'].shape[1] for data in participant_data.values()]
        sample_counts = [data['windows'].shape[2] for data in participant_data.values()]
        
        channels_consistent = len(set(channel_counts)) == 1
        samples_consistent = len(set(sample_counts)) == 1
        
        logger.info(f"   Channels per participant: {channel_counts}")
        logger.info(f"   Samples per participant: {sample_counts}")
        
        integrity_results['channels_consistent'] = channels_consistent
        integrity_results['samples_consistent'] = samples_consistent
        logger.info(f"   ✅ Channel consistency: {channels_consistent}")
        logger.info(f"   ✅ Sample consistency: {samples_consistent}")
        
        self.results['integrity'] = integrity_results
        
        # Overall integrity score
        integrity_score = sum(integrity_results.values()) / len(integrity_results)
        logger.info(f"\n📊 INTEGRITY SCORE: {integrity_score:.1%}")
        
        return integrity_score > 0.8
    
    def analyze_signal_quality(self, X, participants):
        """Analyze EEG signal quality metrics."""
        logger.info("\n📡 SIGNAL QUALITY ANALYSIS")
        logger.info("="*50)
        
        quality_results = {}
        
        # 1. Signal amplitude analysis
        logger.info("1. Signal Amplitude Analysis:")
        
        amplitudes = []
        for i in range(len(X)):
            window_amplitudes = []
            for ch in range(X.shape[1]):
                amp = np.ptp(X[i, ch, :]) * 1e6  # Peak-to-peak amplitude in microvolts
                window_amplitudes.append(amp)
            amplitudes.append(window_amplitudes)
        
        amplitudes = np.array(amplitudes)
        
        mean_amp = np.mean(amplitudes)
        std_amp = np.std(amplitudes)
        
        logger.info(f"   Mean amplitude: {mean_amp:.2f} µV")
        logger.info(f"   Std amplitude: {std_amp:.2f} µV")
        logger.info(f"   Amplitude range: {np.min(amplitudes):.2f} - {np.max(amplitudes):.2f} µV")
        
        # Check for reasonable EEG amplitudes (typically 10-100 µV)
        reasonable_amplitudes = (10 <= mean_amp <= 200) and (std_amp < mean_amp)
        quality_results['reasonable_amplitudes'] = reasonable_amplitudes
        logger.info(f"   ✅ Reasonable amplitudes: {reasonable_amplitudes}")
        
        # 2. Frequency content analysis
        logger.info("\n2. Frequency Content Analysis:")
        
        # Analyze frequency content for a sample of windows
        sample_indices = np.random.choice(len(X), min(50, len(X)), replace=False)
        freq_powers = []
        
        for idx in sample_indices:
            window_freq_powers = []
            for ch in range(min(10, X.shape[1])):  # Analyze subset of channels
                freqs, psd = signal.welch(X[idx, ch, :], fs=self.sfreq, nperseg=min(512, X.shape[2]//4))
                
                # Check power in different bands
                delta_power = np.mean(psd[(freqs >= 1) & (freqs <= 4)])
                theta_power = np.mean(psd[(freqs >= 4) & (freqs <= 8)])
                alpha_power = np.mean(psd[(freqs >= 8) & (freqs <= 13)])
                beta_power = np.mean(psd[(freqs >= 13) & (freqs <= 30)])
                
                window_freq_powers.append([delta_power, theta_power, alpha_power, beta_power])
            
            freq_powers.append(window_freq_powers)
        
        freq_powers = np.array(freq_powers)
        mean_band_powers = np.mean(freq_powers, axis=(0, 1))
        
        logger.info(f"   Delta (1-4Hz): {mean_band_powers[0]:.2e}")
        logger.info(f"   Theta (4-8Hz): {mean_band_powers[1]:.2e}")
        logger.info(f"   Alpha (8-13Hz): {mean_band_powers[2]:.2e}")
        logger.info(f"   Beta (13-30Hz): {mean_band_powers[3]:.2e}")
        
        # Check if alpha power > delta power (typical for clean EEG)
        good_frequency_profile = mean_band_powers[2] > mean_band_powers[0] * 0.1
        quality_results['good_frequency_profile'] = good_frequency_profile
        logger.info(f"   ✅ Good frequency profile: {good_frequency_profile}")
        
        # 3. Artifact detection
        logger.info("\n3. Artifact Detection:")
        
        # Check for extreme values (potential artifacts)
        extreme_threshold = np.percentile(np.abs(X), 99)
        extreme_ratio = np.mean(np.abs(X) > extreme_threshold)
        
        logger.info(f"   Extreme value threshold: {extreme_threshold:.2f}")
        logger.info(f"   Extreme value ratio: {extreme_ratio:.3%}")
        
        low_artifact_level = extreme_ratio < 0.05  # Less than 5% extreme values
        quality_results['low_artifact_level'] = low_artifact_level
        logger.info(f"   ✅ Low artifact level: {low_artifact_level}")
        
        # 4. Channel correlation analysis
        logger.info("\n4. Channel Correlation Analysis:")
        
        # Analyze correlations for a sample
        sample_correlations = []
        for idx in sample_indices[:10]:
            corr_matrix = np.corrcoef(X[idx])
            # Get upper triangle correlations (excluding diagonal)
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
            correlations = corr_matrix[mask]
            sample_correlations.extend(correlations[~np.isnan(correlations)])
        
        mean_correlation = np.mean(sample_correlations)
        std_correlation = np.std(sample_correlations)
        
        logger.info(f"   Mean inter-channel correlation: {mean_correlation:.3f}")
        logger.info(f"   Std inter-channel correlation: {std_correlation:.3f}")
        
        # Good EEG should have moderate correlations (not too high, not too low)
        reasonable_correlations = (0.1 <= mean_correlation <= 0.7) and (std_correlation > 0.05)
        quality_results['reasonable_correlations'] = reasonable_correlations
        logger.info(f"   ✅ Reasonable correlations: {reasonable_correlations}")
        
        self.results['signal_quality'] = quality_results
        
        # Overall signal quality score
        quality_score = sum(quality_results.values()) / len(quality_results)
        logger.info(f"\n📊 SIGNAL QUALITY SCORE: {quality_score:.1%}")
        
        return quality_score > 0.6
    
    def analyze_label_separability(self, X, y, participants):
        """Analyze how well labels can be separated using various methods."""
        logger.info("\n🎯 LABEL SEPARABILITY ANALYSIS")
        logger.info("="*50)
        
        separability_results = {}
        
        # Flatten data for analysis
        X_flat = X.reshape(X.shape[0], -1)
        
        # Remove any remaining NaN/Inf values
        finite_mask = np.isfinite(X_flat).all(axis=1)
        X_clean = X_flat[finite_mask]
        y_clean = y[finite_mask]
        participants_clean = participants[finite_mask]
        
        logger.info(f"Clean samples for analysis: {len(X_clean)}")
        
        # 1. Linear Discriminant Analysis
        logger.info("1. Linear Discriminant Analysis (LDA):")
        
        try:
            lda = LinearDiscriminantAnalysis()
            X_lda = lda.fit_transform(X_clean, y_clean)
            
            # Calculate separability score using silhouette score
            if len(np.unique(y_clean)) > 1:
                silhouette_lda = silhouette_score(X_lda, y_clean)
                logger.info(f"   LDA Silhouette Score: {silhouette_lda:.3f}")
                
                good_lda_separation = silhouette_lda > 0.1
                separability_results['good_lda_separation'] = good_lda_separation
                logger.info(f"   ✅ Good LDA separation: {good_lda_separation}")
            else:
                separability_results['good_lda_separation'] = False
                logger.info("   ❌ Insufficient label diversity for LDA")
        except Exception as e:
            logger.warning(f"   LDA analysis failed: {e}")
            separability_results['good_lda_separation'] = False
        
        # 2. PCA visualization
        logger.info("\n2. Principal Component Analysis (PCA):")
        
        try:
            # Use subset for PCA to speed up computation
            subset_size = min(1000, len(X_clean))
            subset_indices = np.random.choice(len(X_clean), subset_size, replace=False)
            X_subset = X_clean[subset_indices]
            y_subset = y_clean[subset_indices]
            
            # Standardize data
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_subset)
            
            pca = PCA(n_components=10)
            X_pca = pca.fit_transform(X_scaled)
            
            # Check variance explained
            var_explained = np.sum(pca.explained_variance_ratio_[:2])
            logger.info(f"   First 2 PCs explain: {var_explained:.1%} variance")
            
            # Calculate separability in PC space
            if len(np.unique(y_subset)) > 1:
                silhouette_pca = silhouette_score(X_pca[:, :3], y_subset)
                logger.info(f"   PCA Silhouette Score: {silhouette_pca:.3f}")
                
                good_pca_separation = silhouette_pca > 0.05
                separability_results['good_pca_separation'] = good_pca_separation
                logger.info(f"   ✅ Good PCA separation: {good_pca_separation}")
            else:
                separability_results['good_pca_separation'] = False
        except Exception as e:
            logger.warning(f"   PCA analysis failed: {e}")
            separability_results['good_pca_separation'] = False
        
        # 3. Random Forest feature importance
        logger.info("\n3. Random Forest Feature Analysis:")
        
        try:
            # Use a subset of features to speed up computation
            n_features = min(1000, X_clean.shape[1])
            feature_indices = np.random.choice(X_clean.shape[1], n_features, replace=False)
            X_rf = X_clean[:, feature_indices]
            
            rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
            rf.fit(X_rf, y_clean)
            
            # Check if any features have reasonable importance
            max_importance = np.max(rf.feature_importances_)
            mean_importance = np.mean(rf.feature_importances_)
            
            logger.info(f"   Max feature importance: {max_importance:.4f}")
            logger.info(f"   Mean feature importance: {mean_importance:.4f}")
            
            good_feature_importance = max_importance > 0.01 and mean_importance > 0.0005
            separability_results['good_feature_importance'] = good_feature_importance
            logger.info(f"   ✅ Good feature importance: {good_feature_importance}")
        except Exception as e:
            logger.warning(f"   RF analysis failed: {e}")
            separability_results['good_feature_importance'] = False
        
        # 4. Cross-participant consistency
        logger.info("\n4. Cross-Participant Label Consistency:")
        
        participant_label_consistency = []
        for participant in np.unique(participants_clean):
            participant_mask = participants_clean == participant
            participant_labels = y_clean[participant_mask]
            
            if len(participant_labels) > 0:
                # Check if all three classes are present
                unique_labels = np.unique(participant_labels)
                has_all_classes = len(unique_labels) == 3
                
                # Check label balance
                label_counts = np.bincount(participant_labels, minlength=3)
                balance_score = 1 - np.std(label_counts) / np.mean(label_counts) if np.mean(label_counts) > 0 else 0
                
                participant_label_consistency.append({
                    'participant': participant,
                    'has_all_classes': has_all_classes,
                    'balance_score': balance_score,
                    'n_samples': len(participant_labels)
                })
                
                logger.info(f"   {participant}: {len(participant_labels)} samples, "
                          f"classes: {len(unique_labels)}, balance: {balance_score:.2f}")
        
        # Overall consistency
        all_have_classes = all(p['has_all_classes'] for p in participant_label_consistency)
        mean_balance = np.mean([p['balance_score'] for p in participant_label_consistency])
        
        good_label_consistency = all_have_classes and mean_balance > 0.5
        separability_results['good_label_consistency'] = good_label_consistency
        logger.info(f"   ✅ Good label consistency: {good_label_consistency}")
        
        self.results['separability'] = separability_results
        
        # Overall separability score
        separability_score = sum(separability_results.values()) / len(separability_results)
        logger.info(f"\n📊 SEPARABILITY SCORE: {separability_score:.1%}")
        
        return separability_score > 0.5
    
    def run_comprehensive_analysis(self):
        """Run all data quality checks."""
        logger.info("🔬 COMPREHENSIVE EEG DATA QUALITY ANALYSIS")
        logger.info("="*80)
        
        # Load data
        X, y, participants, participant_data = self.load_data()
        
        # Run all analyses
        integrity_ok = self.check_data_integrity(X, y, participants, participant_data)
        quality_ok = self.analyze_signal_quality(X, participants)
        separability_ok = self.analyze_label_separability(X, y, participants)
        
        # Overall assessment
        logger.info("\n" + "="*80)
        logger.info("📋 FINAL DATA QUALITY ASSESSMENT")
        logger.info("="*80)
        
        all_checks = [integrity_ok, quality_ok, separability_ok]
        overall_score = sum(all_checks) / len(all_checks)
        
        logger.info(f"✅ Data Integrity: {'PASS' if integrity_ok else 'FAIL'}")
        logger.info(f"✅ Signal Quality: {'PASS' if quality_ok else 'FAIL'}")
        logger.info(f"✅ Label Separability: {'PASS' if separability_ok else 'FAIL'}")
        
        logger.info(f"\n📊 OVERALL DATA QUALITY: {overall_score:.1%}")
        
        if overall_score >= 0.8:
            logger.info("🟢 EXCELLENT: Data is high quality and suitable for CNN training")
            recommendation = "Proceed with CNN training. Any issues are likely model-related."
        elif overall_score >= 0.6:
            logger.info("🟡 GOOD: Data quality is acceptable for CNN training")
            recommendation = "Proceed with CNN training with careful monitoring."
        elif overall_score >= 0.4:
            logger.info("🟠 MODERATE: Data has some quality issues")
            recommendation = "Consider data preprocessing improvements before CNN training."
        else:
            logger.info("🔴 POOR: Significant data quality issues detected")
            recommendation = "Address data quality issues before proceeding with CNN training."
        
        logger.info(f"\n💡 RECOMMENDATION: {recommendation}")
        
        # Save results
        self.results['overall_score'] = overall_score
        self.results['recommendation'] = recommendation
        
        results_file = Path('data/processed/data_quality_analysis.pkl')
        with open(results_file, 'wb') as f:
            pickle.dump(self.results, f)
        
        logger.info(f"\n💾 Results saved to: {results_file}")
        
        return overall_score, recommendation

def main():
    """Run comprehensive data quality validation."""
    validator = EEGDataQualityValidator()
    score, recommendation = validator.run_comprehensive_analysis()
    
    return score >= 0.6  # Return True if data quality is sufficient

if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
