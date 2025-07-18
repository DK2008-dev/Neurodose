#!/usr/bin/env python3
"""
Comprehensive Results Backup System
==================================

This script provides multiple ways to save and backup CNN validation results:
1. Automatic intermediate saves during training
2. Manual backup creation from terminal output
3. Results recovery and analysis tools
4. Progress monitoring and logging
"""

import pickle
import os
import json
import csv
from datetime import datetime
import re

class CNNResultsSaver:
    """Comprehensive results saving and backup system"""
    
    def __init__(self, results_dir="data/processed"):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
    
    def save_automatic_backup(self, results_data, suffix=""):
        """Save automatic backup with comprehensive metadata"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_name = f"cnn_backup_{timestamp}{suffix}"
        
        files_created = []
        
        # 1. Pickle file (Python objects)
        pickle_file = os.path.join(self.results_dir, f"{base_name}.pkl")
        with open(pickle_file, 'wb') as f:
            pickle.dump(results_data, f)
        files_created.append(pickle_file)
        
        # 2. JSON file (human-readable)
        json_file = os.path.join(self.results_dir, f"{base_name}.json")
        json_data = self._convert_to_json_serializable(results_data)
        with open(json_file, 'w') as f:
            json.dump(json_data, f, indent=2)
        files_created.append(json_file)
        
        # 3. CSV file (spreadsheet compatible)
        csv_file = os.path.join(self.results_dir, f"{base_name}.csv")
        self._save_results_to_csv(results_data, csv_file)
        files_created.append(csv_file)
        
        # 4. Detailed text summary
        summary_file = os.path.join(self.results_dir, f"{base_name}_summary.txt")
        self._create_detailed_summary(results_data, summary_file)
        files_created.append(summary_file)
        
        return files_created
    
    def save_current_progress(self, architecture="EEGNet", completed_folds=None):
        """Save current progress from running validation"""
        
        # Default data if not provided
        if completed_folds is None:
            completed_folds = self._get_current_completed_folds()
        
        results_data = {
            'experiment_info': {
                'architecture': architecture,
                'validation_type': 'Leave-One-Participant-Out Cross-Validation',
                'start_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'dataset': 'OSF Brain Mediators for Pain',
                'total_participants': 49,
                'baseline_xgboost': 0.511,
                'baseline_random': 0.333
            },
            'progress': {
                'completed_folds': len(completed_folds),
                'total_folds': 49,
                'completion_percentage': len(completed_folds) / 49 * 100,
                'status': 'IN_PROGRESS'
            },
            'results': completed_folds,
            'analysis': self._analyze_results(completed_folds),
            'metadata': {
                'saved_by': 'CNNResultsSaver',
                'save_timestamp': datetime.now().isoformat(),
                'save_type': 'progress_backup'
            }
        }
        
        return self.save_automatic_backup(results_data, "_progress")
    
    def _get_current_completed_folds(self):
        """Extract current results from the known progress"""
        return {
            'vp01': {'accuracy': 0.433, 'time_seconds': 3024.2, 'test_samples': 60},
            'vp02': {'accuracy': 0.220, 'time_seconds': 2667.7, 'test_samples': 41},
            'vp03': {'accuracy': 0.367, 'time_seconds': 2635.6, 'test_samples': 60},
            'vp04': {'accuracy': 0.367, 'time_seconds': 2647.5, 'test_samples': 60},
            'vp05': {'accuracy': 0.317, 'time_seconds': 2542.9, 'test_samples': 60},
            'vp06': {'accuracy': 0.283, 'time_seconds': 2524.0, 'test_samples': 60},
            'vp07': {'accuracy': 0.367, 'time_seconds': 2584.5, 'test_samples': 60},
            'vp08': {'accuracy': 0.300, 'time_seconds': 2613.3, 'test_samples': 60},
            'vp09': {'accuracy': 0.267, 'time_seconds': 2476.2, 'test_samples': 60},
            'vp10': {'accuracy': 0.383, 'time_seconds': 2370.3, 'test_samples': 60},
            'vp11': {'accuracy': 0.367, 'time_seconds': 2169.4, 'test_samples': 60}
        }
    
    def _analyze_results(self, results):
        """Analyze completed results"""
        if not results:
            return {}
        
        accuracies = [r['accuracy'] for r in results.values()]
        times = [r['time_seconds'] for r in results.values()]
        
        return {
            'accuracy_stats': {
                'mean': sum(accuracies) / len(accuracies),
                'min': min(accuracies),
                'max': max(accuracies),
                'std': (sum([(x - sum(accuracies)/len(accuracies))**2 for x in accuracies]) / len(accuracies))**0.5,
                'range': max(accuracies) - min(accuracies)
            },
            'timing_stats': {
                'mean_minutes': sum(times) / len(times) / 60,
                'total_hours': sum(times) / 3600,
                'estimated_remaining_hours': (49 - len(results)) * (sum(times) / len(times)) / 3600
            },
            'performance_evaluation': {
                'vs_random_baseline': sum(accuracies) / len(accuracies) - 0.333,
                'vs_xgboost_baseline': sum(accuracies) / len(accuracies) - 0.511,
                'best_participant': max(results.keys(), key=lambda k: results[k]['accuracy']),
                'worst_participant': min(results.keys(), key=lambda k: results[k]['accuracy'])
            }
        }
    
    def _convert_to_json_serializable(self, obj):
        """Convert object to JSON-serializable format"""
        if isinstance(obj, dict):
            return {k: self._convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_json_serializable(item) for item in obj]
        elif hasattr(obj, 'isoformat'):  # datetime objects
            return obj.isoformat()
        else:
            return obj
    
    def _save_results_to_csv(self, results_data, csv_file):
        """Save results in CSV format for spreadsheet analysis"""
        if 'results' not in results_data:
            return
        
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow(['Participant', 'Accuracy', 'Time_Minutes', 'Test_Samples', 'Training_Samples'])
            
            # Data rows
            for participant, result in results_data['results'].items():
                writer.writerow([
                    participant,
                    f"{result['accuracy']:.3f}",
                    f"{result.get('time_seconds', 0) / 60:.1f}",
                    result.get('test_samples', ''),
                    result.get('training_samples', '')
                ])
    
    def _create_detailed_summary(self, results_data, summary_file):
        """Create detailed text summary"""
        with open(summary_file, 'w') as f:
            f.write("CNN VALIDATION RESULTS - DETAILED SUMMARY\n")
            f.write("=" * 60 + "\n\n")
            
            # Experiment info
            if 'experiment_info' in results_data:
                f.write("EXPERIMENT INFORMATION:\n")
                f.write("-" * 30 + "\n")
                for key, value in results_data['experiment_info'].items():
                    f.write(f"  {key}: {value}\n")
                f.write("\n")
            
            # Progress
            if 'progress' in results_data:
                f.write("PROGRESS STATUS:\n")
                f.write("-" * 30 + "\n")
                for key, value in results_data['progress'].items():
                    f.write(f"  {key}: {value}\n")
                f.write("\n")
            
            # Analysis
            if 'analysis' in results_data:
                f.write("PERFORMANCE ANALYSIS:\n")
                f.write("-" * 30 + "\n")
                analysis = results_data['analysis']
                
                if 'accuracy_stats' in analysis:
                    f.write("  Accuracy Statistics:\n")
                    for key, value in analysis['accuracy_stats'].items():
                        f.write(f"    {key}: {value:.3f}\n")
                    f.write("\n")
                
                if 'timing_stats' in analysis:
                    f.write("  Timing Statistics:\n")
                    for key, value in analysis['timing_stats'].items():
                        f.write(f"    {key}: {value:.2f}\n")
                    f.write("\n")
                
                if 'performance_evaluation' in analysis:
                    f.write("  Performance Evaluation:\n")
                    for key, value in analysis['performance_evaluation'].items():
                        f.write(f"    {key}: {value}\n")
                    f.write("\n")
            
            # Individual results
            if 'results' in results_data:
                f.write("INDIVIDUAL PARTICIPANT RESULTS:\n")
                f.write("-" * 30 + "\n")
                for participant, result in results_data['results'].items():
                    f.write(f"  {participant}: {result['accuracy']:.1%} ")
                    f.write(f"({result.get('time_seconds', 0)/60:.1f} min)\n")

    def load_results(self, file_path):
        """Load previously saved results"""
        if file_path.endswith('.pkl'):
            with open(file_path, 'rb') as f:
                return pickle.load(f)
        elif file_path.endswith('.json'):
            with open(file_path, 'r') as f:
                return json.load(f)
        else:
            raise ValueError("Unsupported file format. Use .pkl or .json")
    
    def list_saved_results(self):
        """List all saved result files"""
        files = []
        for file in os.listdir(self.results_dir):
            if 'cnn' in file.lower() and any(file.endswith(ext) for ext in ['.pkl', '.json', '.csv']):
                files.append(os.path.join(self.results_dir, file))
        return sorted(files)

def quick_save_current_progress():
    """Quick function to save current progress"""
    saver = CNNResultsSaver()
    files = saver.save_current_progress()
    
    print("✅ CURRENT PROGRESS SAVED TO MULTIPLE FORMATS:")
    for file in files:
        print(f"   📁 {os.path.basename(file)}")
    
    return files

if __name__ == "__main__":
    # Save current progress in multiple formats
    quick_save_current_progress()
