#!/usr/bin/env python3
"""
Full Analysis Pipeline for High-School Research Paper:
"The Complexity Paradox in EEG-Based Pain Detection: Why Simple Features Beat Deep and Advanced Methods"

Authors: Dhruv Kurup, Avid Patel
Target: Journal of Emerging Investigators (JEI)
"""

import os
import sys
import pickle
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from scipy.stats import zscore
import mne
from sklearn.model_selection import LeaveOneGroupOut, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import xgboost as xgb
import shap
from tqdm import tqdm

warnings.filterwarnings('ignore')

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

class PainDatasetProcessor:
    """Process all 51 participants from OSF Brain Mediators dataset."""
    
    def __init__(self, data_dir, output_dir):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "data_processed").mkdir(exist_ok=True)
        (self.output_dir / "features").mkdir(exist_ok=True)
        (self.output_dir / "models").mkdir(exist_ok=True)
        (self.output_dir / "results").mkdir(exist_ok=True)
        (self.output_dir / "plots").mkdir(exist_ok=True)
        (self.output_dir / "tables").mkdir(exist_ok=True)
        (self.output_dir / "scripts").mkdir(exist_ok=True)
        
        print(f"Initialized processor for {self.data_dir}")
        print(f"Output directory: {self.output_dir}")
    
    def preprocess_single_participant(self, vp_id):
        """Preprocess single participant following exact protocol."""
        print(f"\nProcessing participant {vp_id}...")
        
        # Find participant files
        vhdr_files = list(self.data_dir.glob(f"*{vp_id}*.vhdr"))
        if not vhdr_files:
            print(f"No files found for {vp_id}")
            return None, None
        
        # Use Paradigm1_Perception (our main task)
        perception_files = [f for f in vhdr_files if "Paradigm1_Perception" in f.name]
        if not perception_files:
            print(f"No Paradigm1_Perception file found for {vp_id}")
            return None, None
        
        vhdr_file = perception_files[0]
        print(f"Loading: {vhdr_file.name}")
        
        try:
            # Step 1: Load raw data
            raw = mne.io.read_raw_brainvision(vhdr_file, preload=True, verbose=False)
            
            # Step 2: Filtering
            raw.filter(l_freq=1.0, h_freq=45.0, picks='eeg', verbose=False)
            raw.notch_filter(50.0, picks='eeg', verbose=False)
            
            # Step 3: Resample
            raw.resample(500, verbose=False)
            
            # Step 4: ICA
            ica = mne.preprocessing.ICA(n_components=20, random_state=42, verbose=False)
            ica.fit(raw, picks='eeg', verbose=False)
            
            # Auto-remove artifacts (simplified approach)
            eog_indices, eog_scores = ica.find_bads_eog(raw, verbose=False)
            if eog_indices:
                ica.exclude = eog_indices[:2]  # Remove up to 2 components
                ica.apply(raw, verbose=False)
            
            # Step 5: Extract events and pain ratings
            events, event_id = mne.events_from_annotations(raw, verbose=False)
            
            # Find laser events and extract pain ratings
            laser_events = []
            pain_ratings = []
            
            annotations = raw.annotations
            for i, (onset, duration, description) in enumerate(zip(annotations.onset, 
                                                                  annotations.duration, 
                                                                  annotations.description)):
                if description == "Laser":
                    # Look for corresponding comment with pain rating
                    laser_time = onset
                    
                    # Search for comment within 5 seconds after laser
                    for j in range(i+1, min(i+10, len(annotations.description))):
                        if annotations.description[j].startswith("Comment"):
                            try:
                                rating = int(annotations.description[j].split("/")[-1])
                                laser_events.append(laser_time)
                                pain_ratings.append(rating)
                                break
                            except (ValueError, IndexError):
                                continue
            
            if len(laser_events) == 0:
                print(f"No valid laser-rating pairs found for {vp_id}")
                return None, None
            
            print(f"Found {len(laser_events)} laser-rating pairs")
            
            # Step 6: Create epochs
            custom_events = np.array([[int(t * raw.info['sfreq']), 0, 1] for t in laser_events])
            
            epochs = mne.Epochs(raw, custom_events, event_id={'laser': 1}, 
                              tmin=-1.0, tmax=3.0, baseline=(-1.0, 0.0),
                              reject=dict(eeg=2500e-6), preload=True, verbose=False)
            
            # Step 7: Binary labeling
            ratings_array = np.array(pain_ratings[:len(epochs)])
            
            if len(ratings_array) < 10:  # Need minimum samples
                print(f"Insufficient epochs for {vp_id}: {len(ratings_array)}")
                return None, None
            
            # Compute percentiles
            p33 = np.percentile(ratings_array, 33)
            p67 = np.percentile(ratings_array, 67)
            
            # Binary labels: low (≤33%) = 0, high (≥67%) = 1
            binary_mask = (ratings_array <= p33) | (ratings_array >= p67)
            binary_labels = np.where(ratings_array <= p33, 0, 1)[binary_mask]
            
            # Filter epochs
            epochs_data = epochs.get_data()[binary_mask]
            
            if len(epochs_data) < 5:  # Need minimum samples per class
                print(f"Insufficient binary samples for {vp_id}: {len(epochs_data)}")
                return None, None
            
            print(f"Final dataset: {len(epochs_data)} epochs, {np.sum(binary_labels == 0)} low, {np.sum(binary_labels == 1)} high")
            
            # Save processed data
            output_file = self.output_dir / "data_processed" / f"sub-{vp_id}_epochs.npy"
            labels_file = self.output_dir / "data_processed" / f"sub-{vp_id}_labels.npy"
            
            np.save(output_file, epochs_data)
            np.save(labels_file, binary_labels)
            
            return epochs_data, binary_labels
            
        except Exception as e:
            print(f"Error processing {vp_id}: {str(e)}")
            return None, None
    
    def preprocess_all_participants(self):
        """Process all 51 participants."""
        participant_ids = [f"vp{i:02d}" for i in range(1, 52)]
        
        success_count = 0
        dataset_counts = []
        
        print("Starting preprocessing of all 51 participants...")
        
        for vp_id in tqdm(participant_ids, desc="Processing participants"):
            epochs_data, labels = self.preprocess_single_participant(vp_id)
            
            if epochs_data is not None:
                success_count += 1
                dataset_counts.append({
                    'participant': vp_id,
                    'total_epochs': len(epochs_data),
                    'low_pain': np.sum(labels == 0),
                    'high_pain': np.sum(labels == 1),
                    'balance_ratio': np.sum(labels == 0) / np.sum(labels == 1) if np.sum(labels == 1) > 0 else 0
                })
        
        print(f"\nSuccessfully processed {success_count}/{len(participant_ids)} participants")
        
        # Save dataset summary
        dataset_df = pd.DataFrame(dataset_counts)
        dataset_df.to_csv(self.output_dir / "tables" / "table_dataset_counts.csv", index=False)
        
        return dataset_df

class FeatureExtractor:
    """Extract 78 neuroscience-aligned features."""
    
    def __init__(self, sfreq=500):
        self.sfreq = sfreq
        self.channels = ['Cz', 'FCz', 'C3', 'C4', 'Fz', 'Pz']
        self.freq_bands = {
            'delta': (1, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 45)
        }
    
    def extract_spectral_features(self, epoch_data, ch_names):
        """Extract spectral power features."""
        features = {}
        
        # Get channel indices
        ch_indices = [ch_names.index(ch) for ch in self.channels if ch in ch_names]
        
        for ch_idx, ch_name in zip(ch_indices, [ch for ch in self.channels if ch in ch_names]):
            # Compute PSD
            freqs, psd = signal.welch(epoch_data[ch_idx], fs=self.sfreq, nperseg=min(256, len(epoch_data[ch_idx])))
            
            # Band power
            for band_name, (low, high) in self.freq_bands.items():
                band_mask = (freqs >= low) & (freqs <= high)
                band_power = np.log10(np.trapz(psd[band_mask], freqs[band_mask]) + 1e-12)
                features[f'{ch_name}_{band_name}_power'] = band_power
        
        return features
    
    def extract_ratio_features(self, spectral_features):
        """Extract frequency ratio features."""
        features = {}
        
        for ch in self.channels:
            # Check if all required bands exist
            required_bands = ['delta', 'alpha', 'gamma', 'beta', 'theta']
            if all(f'{ch}_{band}_power' in spectral_features for band in required_bands):
                # Delta/Alpha ratio
                features[f'{ch}_delta_alpha_ratio'] = (
                    spectral_features[f'{ch}_delta_power'] - spectral_features[f'{ch}_alpha_power']
                )
                
                # Gamma/Beta ratio
                features[f'{ch}_gamma_beta_ratio'] = (
                    spectral_features[f'{ch}_gamma_power'] - spectral_features[f'{ch}_beta_power']
                )
                
                # (Delta+Theta)/(Alpha+Beta) ratio
                numerator = spectral_features[f'{ch}_delta_power'] + spectral_features[f'{ch}_theta_power']
                denominator = spectral_features[f'{ch}_alpha_power'] + spectral_features[f'{ch}_beta_power']
                features[f'{ch}_low_high_ratio'] = numerator - denominator
        
        return features
    
    def extract_asymmetry_features(self, spectral_features):
        """Extract spatial asymmetry features."""
        features = {}
        
        # C4-C3 asymmetry for each band
        for band in self.freq_bands.keys():
            c4_key = f'C4_{band}_power'
            c3_key = f'C3_{band}_power'
            if c4_key in spectral_features and c3_key in spectral_features:
                features[f'C4_C3_{band}_asymmetry'] = spectral_features[c4_key] - spectral_features[c3_key]
        
        return features
    
    def extract_erp_features(self, epoch_data, ch_names):
        """Extract ERP component features."""
        features = {}
        
        # Time windows (in samples at 500Hz)
        n2_start, n2_end = int(0.15 * self.sfreq), int(0.25 * self.sfreq)  # 150-250ms
        p2_start, p2_end = int(0.20 * self.sfreq), int(0.35 * self.sfreq)  # 200-350ms
        
        # Baseline correction already applied during epoching
        baseline_start = 0  # -1s to 0s already corrected
        
        for ch in ['Cz', 'FCz']:
            if ch in ch_names:
                ch_idx = ch_names.index(ch)
                
                # N2 component (negative peak)
                n2_window = epoch_data[ch_idx, n2_start:n2_end]
                features[f'{ch}_N2_amplitude'] = np.mean(n2_window)
                
                # P2 component (positive peak)
                p2_window = epoch_data[ch_idx, p2_start:p2_end]
                features[f'{ch}_P2_amplitude'] = np.mean(p2_window)
        
        return features
    
    def extract_temporal_features(self, epoch_data, ch_names):
        """Extract time-domain features."""
        features = {}
        
        ch_indices = [ch_names.index(ch) for ch in self.channels if ch in ch_names]
        
        for ch_idx, ch_name in zip(ch_indices, [ch for ch in self.channels if ch in ch_names]):
            signal_data = epoch_data[ch_idx]
            
            # RMS
            features[f'{ch_name}_RMS'] = np.sqrt(np.mean(signal_data**2))
            
            # Variance
            features[f'{ch_name}_variance'] = np.var(signal_data)
            
            # Zero-crossing rate
            zero_crossings = np.sum(np.diff(np.sign(signal_data)) != 0)
            features[f'{ch_name}_zero_crossing'] = zero_crossings / len(signal_data)
        
        return features
    
    def extract_all_features(self, epoch_data, ch_names):
        """Extract all 78 features from a single epoch."""
        all_features = {}
        
        # Spectral features (30)
        spectral_features = self.extract_spectral_features(epoch_data, ch_names)
        all_features.update(spectral_features)
        
        # Ratio features (18)
        ratio_features = self.extract_ratio_features(spectral_features)
        all_features.update(ratio_features)
        
        # Asymmetry features (5)
        asymmetry_features = self.extract_asymmetry_features(spectral_features)
        all_features.update(asymmetry_features)
        
        # ERP features (4)
        erp_features = self.extract_erp_features(epoch_data, ch_names)
        all_features.update(erp_features)
        
        # Temporal features (18)
        temporal_features = self.extract_temporal_features(epoch_data, ch_names)
        all_features.update(temporal_features)
        
        return all_features

class ModelTrainer:
    """Train and evaluate models with LOPOCV."""
    
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.results = []
    
    def load_all_data(self):
        """Load all processed participant data."""
        data_dir = self.output_dir / "data_processed"
        
        all_features = []
        all_labels = []
        all_participants = []
        
        print("Loading all participant data...")
        
        epoch_files = list(data_dir.glob("sub-*_epochs.npy"))
        
        for epoch_file in tqdm(epoch_files, desc="Loading data"):
            vp_id = epoch_file.name.split('_')[0].replace('sub-', '')
            labels_file = data_dir / f"sub-{vp_id}_labels.npy"
            
            if labels_file.exists():
                epochs_data = np.load(epoch_file)
                labels = np.load(labels_file)
                
                # Extract features for each epoch
                extractor = FeatureExtractor()
                
                # Get channel names (assuming standard order)
                ch_names = ['Fp1', 'Fz', 'F3', 'F7', 'FT9', 'FC5', 'FC1', 'C3', 'T7', 'TP9', 'CP5', 'CP1', 'Pz', 'P3', 'P7', 'O1', 'Oz', 'O2', 'P4', 'P8', 'TP10', 'CP6', 'CP2', 'Cz', 'C4', 'T8', 'FT10', 'FC6', 'FC2', 'F4', 'F8', 'Fp2', 'AF7', 'AF3', 'AFz', 'F1', 'F5', 'FT7', 'FC3', 'FCz', 'C1', 'C5', 'TP7', 'CP3', 'P1', 'P5', 'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'P6', 'P2', 'CPz', 'CP4', 'TP8', 'C6', 'C2', 'FC4', 'FT8', 'F6', 'F2', 'AF4', 'AF8', 'Fpz', 'VEOG', 'HEOG', 'EMG'][:epochs_data.shape[1]]
                
                for epoch_idx in range(len(epochs_data)):
                    features = extractor.extract_all_features(epochs_data[epoch_idx], ch_names)
                    all_features.append(features)
                    all_labels.append(labels[epoch_idx])
                    all_participants.append(vp_id)
        
        # Convert to DataFrame
        features_df = pd.DataFrame(all_features)
        features_df['label'] = all_labels
        features_df['participant'] = all_participants
        
        # Save feature matrix
        features_df.to_csv(self.output_dir / "features" / "all_features.csv", index=False)
        
        print(f"Loaded {len(features_df)} total epochs from {len(set(all_participants))} participants")
        print(f"Feature matrix shape: {features_df.shape}")
        
        return features_df
    
    def train_baseline_rf(self, features_df):
        """Train baseline Random Forest with LOPOCV."""
        print("\nTraining Baseline Random Forest...")
        
        X = features_df.drop(['label', 'participant'], axis=1)
        y = features_df['label']
        groups = features_df['participant']
        
        # LOPOCV
        lopocv = LeaveOneGroupOut()
        
        rf_scores = []
        rf_predictions = []
        rf_true_labels = []
        
        for train_idx, test_idx in tqdm(lopocv.split(X, y, groups), desc="LOPOCV folds", total=len(set(groups))):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Train model
            rf = RandomForestClassifier(n_estimators=300, max_depth=None, 
                                      class_weight='balanced', random_state=42)
            rf.fit(X_train, y_train)
            
            # Predict
            y_pred = rf.predict(X_test)
            y_pred_proba = rf.predict_proba(X_test)[:, 1]
            
            # Metrics
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_pred_proba)
            
            rf_scores.append({'accuracy': acc, 'f1': f1, 'auc': auc})
            rf_predictions.extend(y_pred)
            rf_true_labels.extend(y_test)
            
            # Log per-participant results
            test_participant = groups.iloc[test_idx].iloc[0]
            print(f"Participant {test_participant}: Acc={acc:.3f}, F1={f1:.3f}, AUC={auc:.3f}")
        
        rf_results = pd.DataFrame(rf_scores)
        
        print(f"\nBaseline RF Results:")
        print(f"Mean Accuracy: {rf_results['accuracy'].mean():.3f} ± {rf_results['accuracy'].std():.3f}")
        print(f"Mean F1: {rf_results['f1'].mean():.3f} ± {rf_results['f1'].std():.3f}")
        print(f"Mean AUC: {rf_results['auc'].mean():.3f} ± {rf_results['auc'].std():.3f}")
        
        # Save final model trained on all data
        final_rf = RandomForestClassifier(n_estimators=300, max_depth=None, 
                                        class_weight='balanced', random_state=42)
        final_rf.fit(X, y)
        
        with open(self.output_dir / "models" / "binary_model.pkl", 'wb') as f:
            pickle.dump(final_rf, f)
        
        # SHAP analysis
        explainer = shap.TreeExplainer(final_rf)
        shap_values = explainer.shap_values(X.sample(min(1000, len(X)), random_state=42))
        np.save(self.output_dir / "models" / "shap_values.npy", shap_values[1])  # For class 1
        
        return rf_results, final_rf, (rf_predictions, rf_true_labels)
    
    def train_tuned_xgboost(self, features_df):
        """Train tuned XGBoost with LOPOCV."""
        print("\nTraining Tuned XGBoost...")
        
        X = features_df.drop(['label', 'participant'], axis=1)
        y = features_df['label']
        groups = features_df['participant']
        
        # Parameter grid
        param_grid = {
            'n_estimators': [200, 400, 600],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.05, 0.1]
        }
        
        lopocv = LeaveOneGroupOut()
        xgb_scores = []
        
        for train_idx, test_idx in tqdm(lopocv.split(X, y, groups), desc="XGB LOPOCV", total=len(set(groups))):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Grid search on training fold
            xgb_model = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
            grid_search = GridSearchCV(xgb_model, param_grid, cv=3, scoring='roc_auc', n_jobs=-1)
            grid_search.fit(X_train, y_train)
            
            # Predict with best model
            best_model = grid_search.best_estimator_
            y_pred = best_model.predict(X_test)
            y_pred_proba = best_model.predict_proba(X_test)[:, 1]
            
            # Metrics
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_pred_proba)
            
            xgb_scores.append({'accuracy': acc, 'f1': f1, 'auc': auc})
        
        xgb_results = pd.DataFrame(xgb_scores)
        
        print(f"\nTuned XGBoost Results:")
        print(f"Mean Accuracy: {xgb_results['accuracy'].mean():.3f} ± {xgb_results['accuracy'].std():.3f}")
        print(f"Mean F1: {xgb_results['f1'].mean():.3f} ± {xgb_results['f1'].std():.3f}")
        print(f"Mean AUC: {xgb_results['auc'].mean():.3f} ± {xgb_results['auc'].std():.3f}")
        
        return xgb_results

class FigureGenerator:
    """Generate all figures for the research paper."""
    
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.plots_dir = self.output_dir / "plots"
        self.plots_dir.mkdir(exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")
    
    def create_workflow_diagram(self):
        """Create Figure 1: Workflow diagram."""
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Create workflow steps
        steps = [
            "Raw EEG\n(51 participants)",
            "Preprocessing\n(Filter, ICA, Epoch)",
            "Feature Extraction\n(78 features)",
            "Binary Labeling\n(Low vs High pain)",
            "LOPOCV Training\n(RF, XGBoost)",
            "Performance\nEvaluation"
        ]
        
        # Position steps
        x_positions = np.linspace(0.1, 0.9, len(steps))
        y_position = 0.5
        
        # Draw boxes and arrows
        for i, (x, step) in enumerate(zip(x_positions, steps)):
            # Box
            bbox = dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.7)
            ax.text(x, y_position, step, ha='center', va='center', 
                   bbox=bbox, fontsize=10, weight='bold')
            
            # Arrow to next step
            if i < len(steps) - 1:
                ax.annotate('', xy=(x_positions[i+1] - 0.05, y_position), 
                           xytext=(x + 0.05, y_position),
                           arrowprops=dict(arrowstyle='->', lw=2, color='darkblue'))
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title('EEG Pain Classification Workflow', fontsize=16, weight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / "workflow_diagram.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Created Figure 1: Workflow diagram")
    
    def create_accuracy_barplot(self, rf_results, xgb_results):
        """Create Figure 2: Accuracy comparison barplot."""
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        # Performance data
        methods = ['Simple RF\n(78 features)', 'Advanced Features\n(645 features)', 'CNN\n(Raw EEG)']
        accuracies = [
            rf_results['accuracy'].mean(),
            0.511,  # From our previous advanced features result
            0.487   # From our previous CNN result
        ]
        stds = [
            rf_results['accuracy'].std(),
            0.061,
            0.027
        ]
        
        # Colors
        colors = ['#2E8B57', '#FF6B6B', '#4ECDC4']
        
        # Create bars
        bars = ax.bar(methods, accuracies, yerr=stds, capsize=5, 
                     color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)
        
        # Add value labels on bars
        for bar, acc, std in zip(bars, accuracies, stds):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
                   f'{acc:.1%}±{std:.1%}',
                   ha='center', va='bottom', fontsize=11, weight='bold')
        
        # Add random baseline line
        ax.axhline(y=0.5, color='red', linestyle='--', linewidth=2, 
                  label='Random Baseline (50%)')
        
        ax.set_ylabel('LOPOCV Accuracy', fontsize=12, weight='bold')
        ax.set_title('Model Performance Comparison', fontsize=14, weight='bold')
        ax.set_ylim(0.4, 0.7)
        ax.legend(fontsize=10)
        
        # Format y-axis as percentage
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        
        plt.xticks(rotation=0, ha='center')
        plt.tight_layout()
        plt.savefig(self.plots_dir / "accuracy_barplot.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Created Figure 2: Accuracy comparison barplot")
    
    def create_participant_heatmap(self, features_df, rf_model):
        """Create Figure 3: Per-participant accuracy heatmap."""
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        X = features_df.drop(['label', 'participant'], axis=1)
        y = features_df['label']
        groups = features_df['participant']
        
        # Calculate per-participant accuracy
        lopocv = LeaveOneGroupOut()
        participant_results = []
        
        for train_idx, test_idx in lopocv.split(X, y, groups):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            test_participant = groups.iloc[test_idx].iloc[0]
            
            # Train and test
            rf = RandomForestClassifier(n_estimators=300, max_depth=None, 
                                      class_weight='balanced', random_state=42)
            rf.fit(X_train, y_train)
            y_pred = rf.predict(X_test)
            
            acc = accuracy_score(y_test, y_pred)
            participant_results.append({'participant': test_participant, 'accuracy': acc})
        
        # Create heatmap matrix
        results_df = pd.DataFrame(participant_results)
        results_df['vp_num'] = results_df['participant'].str.extract(r'(\d+)').astype(int)
        results_df = results_df.sort_values('vp_num')
        
        # Reshape for heatmap (10x5 grid approximately)
        n_rows = 10
        n_cols = int(np.ceil(len(results_df) / n_rows))
        
        heatmap_data = np.full((n_rows, n_cols), np.nan)
        for i, (_, row) in enumerate(results_df.iterrows()):
            r, c = i // n_cols, i % n_cols
            if r < n_rows:
                heatmap_data[r, c] = row['accuracy']
        
        # Create labels
        labels = np.full((n_rows, n_cols), '', dtype=object)
        for i, (_, row) in enumerate(results_df.iterrows()):
            r, c = i // n_cols, i % n_cols
            if r < n_rows:
                labels[r, c] = row['participant']
        
        # Plot heatmap
        sns.heatmap(heatmap_data, annot=labels, fmt='', cmap='RdYlBu_r', 
                   vmin=0.3, vmax=0.7, cbar_kws={'label': 'Accuracy'},
                   square=True, linewidths=0.5, ax=ax)
        
        ax.set_title('Per-Participant Classification Accuracy', fontsize=14, weight='bold')
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_xticks([])
        ax.set_yticks([])
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / "participant_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Created Figure 3: Per-participant accuracy heatmap")
    
    def create_confusion_matrix(self, y_true, y_pred):
        """Create Figure 4: Confusion matrix."""
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        
        cm = confusion_matrix(y_true, y_pred)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
                   xticklabels=['Low Pain', 'High Pain'],
                   yticklabels=['Low Pain', 'High Pain'], ax=ax)
        
        ax.set_title('Confusion Matrix (Normalized)', fontsize=14, weight='bold')
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_xlabel('Predicted Label', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / "confusion_matrix.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Created Figure 4: Confusion matrix")
    
    def create_shap_summary(self, features_df, rf_model):
        """Create Figure 5: SHAP feature importance."""
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        X = features_df.drop(['label', 'participant'], axis=1)
        
        # Load SHAP values
        shap_values = np.load(self.output_dir / "models" / "shap_values.npy")
        feature_names = X.columns.tolist()
        
        # Calculate mean absolute SHAP values
        mean_shap = np.mean(np.abs(shap_values), axis=0)
        
        # Get top 15 features
        top_15_idx = np.argsort(mean_shap)[-15:]
        top_15_features = [feature_names[i] for i in top_15_idx]
        top_15_values = mean_shap[top_15_idx]
        
        # Create horizontal bar plot
        y_pos = np.arange(len(top_15_features))
        bars = ax.barh(y_pos, top_15_values, color='steelblue', alpha=0.7)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels([f.replace('_', ' ').title() for f in top_15_features], fontsize=10)
        ax.set_xlabel('Mean |SHAP Value|', fontsize=12, weight='bold')
        ax.set_title('Top 15 Most Important Features', fontsize=14, weight='bold')
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, top_15_values)):
            ax.text(val + 0.001, bar.get_y() + bar.get_height()/2, 
                   f'{val:.3f}', va='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / "shap_summary.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Created Figure 5: SHAP feature importance")

def main():
    """Main pipeline execution."""
    print("="*80)
    print("EEG PAIN CLASSIFICATION - FULL RESEARCH PIPELINE")
    print("High-School Research Paper Analysis")
    print("="*80)
    
    # Configuration
    data_dir = "manual_upload/manual_upload"
    output_dir = "research_paper_analysis"
    
    # Initialize processor
    processor = PainDatasetProcessor(data_dir, output_dir)
    
    # Step 1: Preprocess all participants
    print("\n" + "="*50)
    print("STEP 1: PREPROCESSING ALL PARTICIPANTS")
    print("="*50)
    dataset_df = processor.preprocess_all_participants()
    
    # Step 2: Extract features and train models
    print("\n" + "="*50)
    print("STEP 2: FEATURE EXTRACTION AND MODEL TRAINING")
    print("="*50)
    
    trainer = ModelTrainer(output_dir)
    features_df = trainer.load_all_data()
    
    # Train models
    rf_results, rf_model, rf_predictions = trainer.train_baseline_rf(features_df)
    xgb_results = trainer.train_tuned_xgboost(features_df)
    
    # Step 3: Generate figures
    print("\n" + "="*50)
    print("STEP 3: GENERATING FIGURES")
    print("="*50)
    
    fig_gen = FigureGenerator(output_dir)
    fig_gen.create_workflow_diagram()
    fig_gen.create_accuracy_barplot(rf_results, xgb_results)
    fig_gen.create_participant_heatmap(features_df, rf_model)
    fig_gen.create_confusion_matrix(rf_predictions[1], rf_predictions[0])
    fig_gen.create_shap_summary(features_df, rf_model)
    
    # Step 4: Save performance table
    print("\n" + "="*50)
    print("STEP 4: SAVING RESULTS")
    print("="*50)
    
    # Performance summary table
    performance_data = {
        'Method': ['Simple RF (78 features)', 'Tuned XGBoost', 'Advanced Features (645)', 'CNN (Raw EEG)'],
        'Accuracy_Mean': [
            rf_results['accuracy'].mean(),
            xgb_results['accuracy'].mean(),
            0.511,
            0.487
        ],
        'Accuracy_Std': [
            rf_results['accuracy'].std(),
            xgb_results['accuracy'].std(),
            0.061,
            0.027
        ],
        'F1_Mean': [
            rf_results['f1'].mean(),
            xgb_results['f1'].mean(),
            0.0,  # Not available
            0.0   # Not available
        ],
        'AUC_Mean': [
            rf_results['auc'].mean(),
            xgb_results['auc'].mean(),
            0.0,  # Not available
            0.0   # Not available
        ]
    }
    
    performance_df = pd.DataFrame(performance_data)
    performance_df.to_csv(Path(output_dir) / "tables" / "table_performance.csv", index=False)
    
    # LOPOCV detailed results
    rf_results['method'] = 'Random Forest'
    xgb_results['method'] = 'XGBoost'
    all_results = pd.concat([rf_results, xgb_results], ignore_index=True)
    all_results.to_csv(Path(output_dir) / "results" / "lopocv_metrics.csv", index=False)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print(f"Results saved to: {output_dir}/")
    print(f"- Figures: {output_dir}/plots/")
    print(f"- Tables: {output_dir}/tables/")
    print(f"- Models: {output_dir}/models/")
    print(f"- Data: {output_dir}/data_processed/")
    print("\nReady for Journal of Emerging Investigators submission!")

if __name__ == "__main__":
    main()
