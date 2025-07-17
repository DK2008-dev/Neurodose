"""
Analyze Preprocessing Gap: Why 35% LOPOCV vs 87% Literature Benchmark?

This script investigates the specific differences between our preprocessing pipeline
and the original methodology that achieved 87% accuracy to identify optimization opportunities.
"""

import numpy as np
import pandas as pd
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.signal import welch, periodogram
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, LeaveOneGroupOut
from sklearn.metrics import classification_report, accuracy_score

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PreprocessingGapAnalyzer:
    """Analyze differences between our preprocessing and literature methodology."""
    
    def __init__(self, windows_file: str):
        """Initialize with preprocessed windows data."""
        self.windows_file = windows_file
        self.data = None
        self.results = {}
        
    def load_data(self) -> None:
        """Load preprocessed windows data."""
        logger.info(f"Loading data from {self.windows_file}")
        with open(self.windows_file, 'rb') as f:
            data = pickle.load(f)
        
        # Convert structured data to list format for compatibility
        self.data = []
        windows = data['windows']
        pain_ratings = data['pain_ratings']
        participants = data['participants']
        
        for i in range(len(windows)):
            window_dict = {
                'eeg_data': windows[i],  # Shape: (channels, time_points)
                'pain_rating': pain_ratings[i],
                'participant': participants[i]
            }
            self.data.append(window_dict)
        
        logger.info(f"Loaded {len(self.data)} windows from {len(set(participants))} participants")
        logger.info(f"EEG data shape: {windows[0].shape}, Sampling rate: {data.get('sfreq', 'unknown')} Hz")
        
    def analyze_data_characteristics(self) -> Dict[str, Any]:
        """Analyze fundamental data characteristics that might explain performance gaps."""
        logger.info("=== ANALYZING DATA CHARACTERISTICS ===")
        
        characteristics = {
            'participant_stats': {},
            'pain_distribution': {},
            'signal_quality': {},
            'temporal_patterns': {}
        }
        
        # Participant-level analysis
        participant_data = {}
        for window in self.data:
            participant = window['participant']
            pain_rating = window['pain_rating']
            
            if participant not in participant_data:
                participant_data[participant] = {'ratings': [], 'signal_stats': []}
            
            participant_data[participant]['ratings'].append(pain_rating)
            
            # Signal quality metrics
            eeg_data = window['eeg_data']  # Shape: (channels, time_points)
            signal_power = np.mean(np.var(eeg_data, axis=1))
            signal_range = np.mean(np.ptp(eeg_data, axis=1))
            
            participant_data[participant]['signal_stats'].append({
                'power': signal_power,
                'range': signal_range,
                'channels': eeg_data.shape[0],
                'time_points': eeg_data.shape[1]
            })
        
        # Analyze each participant
        for participant, data in participant_data.items():
            ratings = np.array(data['ratings'])
            stats_list = data['signal_stats']
            
            characteristics['participant_stats'][participant] = {
                'n_windows': len(ratings),
                'pain_mean': np.mean(ratings),
                'pain_std': np.std(ratings),
                'pain_range': [np.min(ratings), np.max(ratings)],
                'low_pain_pct': np.mean(ratings <= 30) * 100,
                'high_pain_pct': np.mean(ratings >= 50) * 100,
                'signal_power_mean': np.mean([s['power'] for s in stats_list]),
                'signal_range_mean': np.mean([s['range'] for s in stats_list]),
                'channels': stats_list[0]['channels'],
                'time_points': stats_list[0]['time_points']
            }
        
        # Overall distribution analysis
        all_ratings = [w['pain_rating'] for w in self.data]
        characteristics['pain_distribution'] = {
            'total_windows': len(all_ratings),
            'mean_rating': np.mean(all_ratings),
            'std_rating': np.std(all_ratings),
            'rating_range': [np.min(all_ratings), np.max(all_ratings)],
            'binary_distribution': {
                'low (≤30)': np.sum(np.array(all_ratings) <= 30),
                'excluded (31-49)': np.sum((np.array(all_ratings) > 30) & (np.array(all_ratings) < 50)),
                'high (≥50)': np.sum(np.array(all_ratings) >= 50)
            }
        }
        
        self.results['characteristics'] = characteristics
        return characteristics
    
    def compare_feature_extraction_methods(self) -> Dict[str, Any]:
        """Compare our feature extraction vs original methodology."""
        logger.info("=== COMPARING FEATURE EXTRACTION METHODS ===")
        
        # Filter to binary classification data
        binary_data = [w for w in self.data if w['pain_rating'] <= 30 or w['pain_rating'] >= 50]
        logger.info(f"Binary classification data: {len(binary_data)} windows")
        
        comparison = {
            'our_method': {},
            'original_method': {},
            'differences': {}
        }
        
        # Sample window for detailed analysis
        sample_window = binary_data[0]
        eeg_data = sample_window['eeg_data']  # Shape: (channels, time_points)
        sfreq = 500  # Hz
        
        logger.info(f"Sample window shape: {eeg_data.shape}")
        logger.info(f"Sampling frequency: {sfreq} Hz")
        
        # Our method: Full 4-second window spectral features
        our_features = self._extract_our_features(eeg_data, sfreq)
        
        # Original method: 1-second window with time segments
        # Use first 1 second (500 samples) as in the original
        eeg_1s = eeg_data[:, :500]
        original_features = self._extract_original_features(eeg_1s, sfreq)
        
        comparison['our_method'] = {
            'window_length': eeg_data.shape[1] / sfreq,
            'n_features': len(our_features),
            'feature_types': list(our_features.keys()),
            'sample_values': {k: v for k, v in list(our_features.items())[:5]}
        }
        
        comparison['original_method'] = {
            'window_length': eeg_1s.shape[1] / sfreq,
            'n_features': len(original_features),
            'feature_types': list(original_features.keys()),
            'sample_values': {k: v for k, v in list(original_features.items())[:5]}
        }
        
        comparison['differences'] = {
            'window_length_ratio': eeg_data.shape[1] / eeg_1s.shape[1],
            'feature_count_ratio': len(our_features) / len(original_features),
            'unique_our_features': set(our_features.keys()) - set(original_features.keys()),
            'unique_original_features': set(original_features.keys()) - set(our_features.keys())
        }
        
        self.results['feature_comparison'] = comparison
        return comparison
    
    def _extract_our_features(self, eeg_data: np.ndarray, sfreq: float) -> Dict[str, float]:
        """Extract features using our current methodology."""
        features = {}
        
        # Frequency bands (our method)
        bands = {
            'delta': (1, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 45)
        }
        
        for ch_idx in range(min(5, eeg_data.shape[0])):  # Sample first 5 channels
            ch_data = eeg_data[ch_idx, :]
            
            # Compute PSD
            freqs, psd = welch(ch_data, fs=sfreq, nperseg=min(256, len(ch_data)//4))
            
            for band_name, (low, high) in bands.items():
                band_mask = (freqs >= low) & (freqs <= high)
                band_power = np.mean(psd[band_mask])
                features[f'ch{ch_idx}_{band_name}_power'] = band_power
        
        return features
    
    def _extract_original_features(self, eeg_data: np.ndarray, sfreq: float) -> Dict[str, float]:
        """Extract features using original methodology."""
        features = {}
        
        # Original frequency bands
        bands = {
            'delta': (1, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 90)  # Extended gamma range
        }
        
        # Time windows (original method)
        time_windows = {
            'early': (0, 80),    # 0-0.16s
            'mid': (80, 150),    # 0.16-0.3s
            'late': (150, 500)   # 0.3-1.0s
        }
        
        for ch_idx in range(min(5, eeg_data.shape[0])):  # Sample first 5 channels
            for time_name, (start_idx, end_idx) in time_windows.items():
                ch_data = eeg_data[ch_idx, start_idx:end_idx]
                
                if len(ch_data) > 10:  # Ensure sufficient data
                    # Compute PSD for this time window
                    freqs, psd = welch(ch_data, fs=sfreq, nperseg=min(64, len(ch_data)//2))
                    
                    for band_name, (low, high) in bands.items():
                        band_mask = (freqs >= low) & (freqs <= high)
                        if np.any(band_mask):
                            band_power = np.mean(psd[band_mask])
                            features[f'ch{ch_idx}_{time_name}_{band_name}_power'] = band_power
        
        return features
    
    def analyze_class_imbalance_impact(self) -> Dict[str, Any]:
        """Analyze how class imbalance affects different participants."""
        logger.info("=== ANALYZING CLASS IMBALANCE IMPACT ===")
        
        imbalance_analysis = {}
        
        # Filter to binary classification
        binary_data = [w for w in self.data if w['pain_rating'] <= 30 or w['pain_rating'] >= 50]
        
        # Group by participant
        participant_groups = {}
        for window in binary_data:
            participant = window['participant']
            if participant not in participant_groups:
                participant_groups[participant] = []
            participant_groups[participant].append(window)
        
        for participant, windows in participant_groups.items():
            ratings = [w['pain_rating'] for w in windows]
            low_count = sum(1 for r in ratings if r <= 30)
            high_count = sum(1 for r in ratings if r >= 50)
            
            imbalance_ratio = min(low_count, high_count) / max(low_count, high_count) if max(low_count, high_count) > 0 else 0
            
            imbalance_analysis[participant] = {
                'total_windows': len(windows),
                'low_pain_count': low_count,
                'high_pain_count': high_count,
                'imbalance_ratio': imbalance_ratio,
                'minority_class': 'low' if low_count < high_count else 'high',
                'severity': 'severe' if imbalance_ratio < 0.3 else 'moderate' if imbalance_ratio < 0.7 else 'balanced'
            }
        
        # Overall statistics
        all_ratios = [data['imbalance_ratio'] for data in imbalance_analysis.values()]
        imbalance_analysis['overall'] = {
            'mean_imbalance_ratio': np.mean(all_ratios),
            'std_imbalance_ratio': np.std(all_ratios),
            'severely_imbalanced_participants': sum(1 for r in all_ratios if r < 0.3),
            'balanced_participants': sum(1 for r in all_ratios if r >= 0.7)
        }
        
        self.results['imbalance_analysis'] = imbalance_analysis
        return imbalance_analysis
    
    def investigate_signal_quality_differences(self) -> Dict[str, Any]:
        """Investigate signal quality metrics that might affect performance."""
        logger.info("=== INVESTIGATING SIGNAL QUALITY ===")
        
        quality_analysis = {}
        
        # Analyze signal characteristics per participant
        participant_quality = {}
        
        for window in self.data:
            participant = window['participant']
            eeg_data = window['eeg_data']
            
            if participant not in participant_quality:
                participant_quality[participant] = {
                    'snr_estimates': [],
                    'artifact_levels': [],
                    'signal_variance': [],
                    'channel_consistency': []
                }
            
            # Signal-to-noise ratio estimate (simplified)
            signal_power = np.mean(np.var(eeg_data, axis=1))
            noise_estimate = np.mean(np.var(np.diff(eeg_data, axis=1), axis=1))
            snr_estimate = signal_power / (noise_estimate + 1e-10)
            
            # Artifact level (high frequency content)
            artifact_level = np.mean(np.var(eeg_data[:, ::10], axis=1))  # Downsampled variance
            
            # Signal variance across channels
            channel_vars = np.var(eeg_data, axis=1)
            
            # Channel consistency (correlation between channels)
            if eeg_data.shape[0] > 1:
                corr_matrix = np.corrcoef(eeg_data)
                mean_correlation = np.mean(corr_matrix[np.triu_indices_from(corr_matrix, k=1)])
            else:
                mean_correlation = 1.0
            
            participant_quality[participant]['snr_estimates'].append(snr_estimate)
            participant_quality[participant]['artifact_levels'].append(artifact_level)
            participant_quality[participant]['signal_variance'].append(np.mean(channel_vars))
            participant_quality[participant]['channel_consistency'].append(mean_correlation)
        
        # Summarize quality metrics per participant
        for participant, metrics in participant_quality.items():
            quality_analysis[participant] = {
                'mean_snr': np.mean(metrics['snr_estimates']),
                'mean_artifacts': np.mean(metrics['artifact_levels']),
                'mean_variance': np.mean(metrics['signal_variance']),
                'mean_consistency': np.mean(metrics['channel_consistency']),
                'snr_std': np.std(metrics['snr_estimates']),
                'n_windows': len(metrics['snr_estimates'])
            }
        
        # Overall quality assessment
        all_snr = [q['mean_snr'] for q in quality_analysis.values()]
        all_consistency = [q['mean_consistency'] for q in quality_analysis.values()]
        
        quality_analysis['overall'] = {
            'mean_snr_across_participants': np.mean(all_snr),
            'std_snr_across_participants': np.std(all_snr),
            'mean_consistency_across_participants': np.mean(all_consistency),
            'std_consistency_across_participants': np.std(all_consistency),
            'quality_range': [np.min(all_snr), np.max(all_snr)]
        }
        
        self.results['quality_analysis'] = quality_analysis
        return quality_analysis
    
    def generate_optimization_recommendations(self) -> Dict[str, List[str]]:
        """Generate specific recommendations to close the performance gap."""
        logger.info("=== GENERATING OPTIMIZATION RECOMMENDATIONS ===")
        
        recommendations = {
            'immediate_actions': [],
            'feature_engineering': [],
            'data_handling': [],
            'model_optimization': [],
            'evaluation_strategy': []
        }
        
        # Analyze results to generate recommendations
        if 'imbalance_analysis' in self.results:
            imbalance = self.results['imbalance_analysis']
            severely_imbalanced = imbalance['overall']['severely_imbalanced_participants']
            
            if severely_imbalanced > 0:
                recommendations['immediate_actions'].append(
                    f"🚨 Address severe class imbalance in {severely_imbalanced} participants"
                )
                recommendations['data_handling'].extend([
                    "Implement participant-specific SMOTE with careful CV isolation",
                    "Consider stratified sampling within each participant",
                    "Evaluate excluding severely imbalanced participants from training"
                ])
        
        if 'feature_comparison' in self.results:
            comparison = self.results['feature_comparison']
            window_ratio = comparison['differences']['window_length_ratio']
            
            if window_ratio > 2:
                recommendations['feature_engineering'].extend([
                    f"Consider reducing window length from {window_ratio:.1f}x to match literature (1s)",
                    "Implement time-windowed feature extraction (early/mid/late segments)",
                    "Add temporal dynamics features within shorter windows"
                ])
            
            feature_ratio = comparison['differences']['feature_count_ratio']
            if feature_ratio > 2:
                recommendations['feature_engineering'].append(
                    "Reduce feature dimensionality - too many features may cause overfitting"
                )
        
        if 'quality_analysis' in self.results:
            quality = self.results['quality_analysis']
            snr_std = quality['overall']['std_snr_across_participants']
            
            if snr_std > np.mean([q['mean_snr'] for q in quality.values() if isinstance(q, dict) and 'mean_snr' in q]) * 0.5:
                recommendations['data_handling'].extend([
                    "Implement participant-specific signal quality normalization",
                    "Consider adaptive preprocessing based on signal quality metrics"
                ])
        
        # General recommendations based on literature gaps
        recommendations['model_optimization'].extend([
            "Implement Optuna hyperparameter optimization (40+ trials)",
            "Use XGBoost with participant-aware regularization",
            "Consider ensemble methods to handle participant variability"
        ])
        
        recommendations['evaluation_strategy'].extend([
            "Implement stratified LOPOCV to handle class imbalance",
            "Use participant-weighted metrics to account for data imbalance",
            "Report both simple split and LOPOCV results for comparison"
        ])
        
        self.results['recommendations'] = recommendations
        return recommendations
    
    def create_comprehensive_report(self) -> None:
        """Create a comprehensive analysis report."""
        logger.info("=== CREATING COMPREHENSIVE REPORT ===")
        
        report_path = Path("data/processed/preprocessing_gap_analysis.txt")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("PREPROCESSING GAP ANALYSIS REPORT\n")
            f.write("Why 35% LOPOCV vs 87% Literature Benchmark?\n")
            f.write("=" * 80 + "\n\n")
            
            # Data characteristics
            if 'characteristics' in self.results:
                f.write("1. DATA CHARACTERISTICS ANALYSIS\n")
                f.write("-" * 40 + "\n")
                chars = self.results['characteristics']
                
                f.write(f"Overall Dataset:\n")
                f.write(f"  - Total windows: {chars['pain_distribution']['total_windows']}\n")
                f.write(f"  - Mean pain rating: {chars['pain_distribution']['mean_rating']:.1f}\n")
                f.write(f"  - Binary distribution:\n")
                for category, count in chars['pain_distribution']['binary_distribution'].items():
                    f.write(f"    - {category}: {count} windows\n")
                
                f.write(f"\nParticipant Analysis:\n")
                for participant, stats in chars['participant_stats'].items():
                    f.write(f"  {participant}: {stats['n_windows']} windows, "
                           f"{stats['low_pain_pct']:.1f}% low, {stats['high_pain_pct']:.1f}% high pain\n")
                f.write("\n")
            
            # Feature comparison
            if 'feature_comparison' in self.results:
                f.write("2. FEATURE EXTRACTION COMPARISON\n")
                f.write("-" * 40 + "\n")
                comp = self.results['feature_comparison']
                
                f.write(f"Our Method:\n")
                f.write(f"  - Window length: {comp['our_method']['window_length']:.1f}s\n")
                f.write(f"  - Features: {comp['our_method']['n_features']}\n")
                
                f.write(f"\nOriginal Method:\n")
                f.write(f"  - Window length: {comp['original_method']['window_length']:.1f}s\n")
                f.write(f"  - Features: {comp['original_method']['n_features']}\n")
                
                f.write(f"\nKey Differences:\n")
                f.write(f"  - Window length ratio: {comp['differences']['window_length_ratio']:.1f}x\n")
                f.write(f"  - Feature count ratio: {comp['differences']['feature_count_ratio']:.1f}x\n")
                f.write("\n")
            
            # Class imbalance
            if 'imbalance_analysis' in self.results:
                f.write("3. CLASS IMBALANCE IMPACT\n")
                f.write("-" * 40 + "\n")
                imbalance = self.results['imbalance_analysis']
                
                f.write(f"Overall Imbalance Statistics:\n")
                f.write(f"  - Mean imbalance ratio: {imbalance['overall']['mean_imbalance_ratio']:.3f}\n")
                f.write(f"  - Severely imbalanced participants: {imbalance['overall']['severely_imbalanced_participants']}\n")
                f.write(f"  - Balanced participants: {imbalance['overall']['balanced_participants']}\n")
                
                f.write(f"\nParticipant Imbalance Details:\n")
                for participant, data in imbalance.items():
                    if participant != 'overall':
                        f.write(f"  {participant}: {data['severity']} "
                               f"(ratio: {data['imbalance_ratio']:.3f}, "
                               f"{data['low_pain_count']}L/{data['high_pain_count']}H)\n")
                f.write("\n")
            
            # Recommendations
            if 'recommendations' in self.results:
                f.write("4. OPTIMIZATION RECOMMENDATIONS\n")
                f.write("-" * 40 + "\n")
                recs = self.results['recommendations']
                
                for category, items in recs.items():
                    if items:
                        f.write(f"\n{category.upper().replace('_', ' ')}:\n")
                        for item in items:
                            f.write(f"  • {item}\n")
                f.write("\n")
            
            f.write("=" * 80 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 80 + "\n")
        
        logger.info(f"Comprehensive report saved to {report_path}")
    
    def run_full_analysis(self) -> None:
        """Run complete preprocessing gap analysis."""
        logger.info("Starting comprehensive preprocessing gap analysis...")
        
        self.load_data()
        self.analyze_data_characteristics()
        self.compare_feature_extraction_methods()
        self.analyze_class_imbalance_impact()
        self.investigate_signal_quality_differences()
        self.generate_optimization_recommendations()
        self.create_comprehensive_report()
        
        # Save detailed results
        results_path = Path("data/processed/preprocessing_gap_analysis.pkl")
        with open(results_path, 'wb') as f:
            pickle.dump(self.results, f)
        logger.info(f"Detailed results saved to {results_path}")
        
        logger.info("=== ANALYSIS COMPLETE ===")
        logger.info("Key findings:")
        logger.info("1. Check data/processed/preprocessing_gap_analysis.txt for full report")
        logger.info("2. Check data/processed/preprocessing_gap_analysis.pkl for detailed results")
        logger.info("3. Focus on class imbalance and feature extraction optimization")

def main():
    """Main execution function."""
    windows_file = "data/processed/windows_with_pain_ratings.pkl"
    
    if not Path(windows_file).exists():
        logger.error(f"Windows file not found: {windows_file}")
        logger.error("Please run preprocessing first to generate the windows file.")
        return
    
    analyzer = PreprocessingGapAnalyzer(windows_file)
    analyzer.run_full_analysis()

if __name__ == "__main__":
    main()
