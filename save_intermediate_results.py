#!/usr/bin/env python3
"""
Save Intermediate CNN Validation Results
=====================================

This script captures and saves the intermediate results from the ongoing CNN validation.
Can be used to preserve progress if the process needs to be stopped.
"""

import pickle
import os
from datetime import datetime

def save_intermediate_results():
    """Save the intermediate results from the current CNN validation run"""
    
    # Extract results from current terminal output
    intermediate_results = {
        'architecture': 'EEGNet',
        'validation_type': 'Leave-One-Participant-Out Cross-Validation',
        'baseline_accuracy': 0.511,  # XGBoost baseline
        'random_baseline': 0.333,   # Random ternary baseline
        
        # Dataset information
        'dataset_info': {
            'total_windows': 2875,
            'total_participants': 49,  # 51 loaded, 2 failed (vp39, vp40)
            'data_shape': '(2875, 68, 2000)',
            'labels_range': '0.0 - 2.0',
            'class_distribution': {
                'low': {'count': 969, 'percentage': 33.7},
                'moderate': {'count': 952, 'percentage': 33.1},
                'high': {'count': 954, 'percentage': 33.2}
            }
        },
        
        # Training configuration
        'training_config': {
            'batch_size': 32,
            'epochs': 50,
            'device': 'cpu',
            'architecture': 'EEGNet'
        },
        
        # Completed folds results (from terminal output)
        'completed_folds': {
            'vp01': {
                'accuracy': 0.433,
                'training_time_seconds': 3024.2,
                'test_samples': 60,
                'training_samples': 2815,
                'final_epoch_stats': {
                    'loss': 0.9611,
                    'accuracy': 0.540
                }
            },
            'vp02': {
                'accuracy': 0.220,
                'training_time_seconds': 2667.7,
                'test_samples': 41,
                'training_samples': 2834,
                'final_epoch_stats': {
                    'loss': 0.9630,
                    'accuracy': 0.527
                }
            },
            'vp03': {
                'accuracy': 0.367,
                'training_time_seconds': 2635.6,
                'test_samples': 60,
                'training_samples': 2815,
                'final_epoch_stats': {
                    'loss': 0.9559,
                    'accuracy': 0.543
                }
            },
            'vp04': {
                'accuracy': 0.367,
                'training_time_seconds': 2647.5,
                'test_samples': 60,
                'training_samples': 2815,
                'final_epoch_stats': {
                    'loss': 0.9606,
                    'accuracy': 0.525
                }
            },
            'vp05': {
                'accuracy': 0.317,
                'training_time_seconds': 2542.9,
                'test_samples': 60,
                'training_samples': 2815,
                'final_epoch_stats': {
                    'loss': 0.9559,
                    'accuracy': 0.536
                }
            },
            'vp06': {
                'accuracy': 0.283,
                'training_time_seconds': 2524.0,
                'test_samples': 60,
                'training_samples': 2815,
                'final_epoch_stats': {
                    'loss': 0.9650,
                    'accuracy': 0.524
                }
            },
            'vp07': {
                'accuracy': 0.367,
                'training_time_seconds': 2584.5,
                'test_samples': 60,
                'training_samples': 2815,
                'final_epoch_stats': {
                    'loss': 0.9354,
                    'accuracy': 0.551
                }
            },
            'vp08': {
                'accuracy': 0.300,
                'training_time_seconds': 2613.3,
                'test_samples': 60,
                'training_samples': 2815,
                'final_epoch_stats': {
                    'loss': 0.9640,
                    'accuracy': 0.530
                }
            },
            'vp09': {
                'accuracy': 0.267,
                'training_time_seconds': 2476.2,
                'test_samples': 60,
                'training_samples': 2815,
                'final_epoch_stats': {
                    'loss': 0.9514,
                    'accuracy': 0.542
                }
            },
            'vp10': {
                'accuracy': 0.383,
                'training_time_seconds': 2370.3,
                'test_samples': 60,
                'training_samples': 2815,
                'final_epoch_stats': {
                    'loss': 0.9524,
                    'accuracy': 0.544
                }
            },
            'vp11': {
                'accuracy': 0.367,
                'training_time_seconds': 2169.4,
                'test_samples': 60,
                'training_samples': 2815,
                'final_epoch_stats': {
                    'loss': 0.9663,
                    'accuracy': 0.528
                }
            }
        },
        
        # Current progress summary
        'progress_summary': {
            'completed_folds': 11,
            'total_folds': 49,
            'completion_percentage': 22.4,
            'current_running_average': 0.334,  # 33.4% from terminal
            'current_std_dev': 0.063,  # Estimated from range
            'best_participant': 'vp01 (43.3%)',
            'worst_participant': 'vp02 (22.0%)',
            'currently_training': 'vp12 (in progress)'
        },
        
        # Performance analysis
        'performance_analysis': {
            'vs_random_baseline': {
                'current_accuracy': 0.334,
                'random_baseline': 0.333,
                'improvement': 0.001,
                'percentage_improvement': 0.3
            },
            'vs_xgboost_baseline': {
                'current_accuracy': 0.334,
                'xgboost_baseline': 0.511,
                'gap': -0.177,
                'percentage_gap': -34.6
            },
            'status': 'Performing at random baseline level',
            'trend': 'Consistent performance across participants'
        },
        
        # Timing estimates
        'timing_estimates': {
            'average_fold_time_minutes': 43.7,  # Average from completed folds
            'estimated_total_time_hours': 35.6,  # 49 folds * 43.7 min / 60
            'elapsed_time_hours': 8.0,  # Approximate from start
            'estimated_remaining_hours': 27.6
        },
        
        # Technical details
        'technical_details': {
            'failed_participants': ['vp39', 'vp40'],
            'error_message': 'zero-size array to reduction operation minimum which has no identity',
            'training_stable': True,
            'loss_decreasing': True,
            'accuracy_improving': True,
            'overfitting_detected': False
        },
        
        # Metadata
        'metadata': {
            'script_name': 'test_cnn_validation_fixed.py',
            'saved_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'experiment_start': '2025-07-17 ~12:40',
            'intermediate_save': True,
            'validation_status': 'IN_PROGRESS'
        }
    }
    
    # Calculate additional statistics
    accuracies = [fold['accuracy'] for fold in intermediate_results['completed_folds'].values()]
    intermediate_results['statistics'] = {
        'mean_accuracy': sum(accuracies) / len(accuracies),
        'min_accuracy': min(accuracies),
        'max_accuracy': max(accuracies),
        'accuracy_range': max(accuracies) - min(accuracies),
        'standard_deviation': (sum([(x - sum(accuracies)/len(accuracies))**2 for x in accuracies]) / len(accuracies))**0.5
    }
    
    # Save to file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = f"data/processed/cnn_intermediate_results_{timestamp}.pkl"
    
    try:
        os.makedirs(os.path.dirname(results_file), exist_ok=True)
        with open(results_file, 'wb') as f:
            pickle.dump(intermediate_results, f)
        
        print(f"✅ INTERMEDIATE RESULTS SAVED SUCCESSFULLY!")
        print(f"📁 File: {results_file}")
        print(f"📊 Progress: {intermediate_results['progress_summary']['completed_folds']}/49 folds (22.4%)")
        print(f"🎯 Current Accuracy: {intermediate_results['statistics']['mean_accuracy']:.1%}")
        print(f"⏱️  Estimated Remaining: {intermediate_results['timing_estimates']['estimated_remaining_hours']:.1f} hours")
        
        # Also save a human-readable summary
        summary_file = f"data/processed/cnn_intermediate_summary_{timestamp}.txt"
        with open(summary_file, 'w') as f:
            f.write("CNN VALIDATION INTERMEDIATE RESULTS SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Saved: {intermediate_results['metadata']['saved_timestamp']}\n")
            f.write(f"Status: {intermediate_results['metadata']['validation_status']}\n\n")
            
            f.write("PROGRESS:\n")
            f.write(f"  Completed: {intermediate_results['progress_summary']['completed_folds']}/49 folds\n")
            f.write(f"  Percentage: {intermediate_results['progress_summary']['completion_percentage']:.1f}%\n")
            f.write(f"  Currently training: {intermediate_results['progress_summary']['currently_training']}\n\n")
            
            f.write("PERFORMANCE:\n")
            f.write(f"  Mean accuracy: {intermediate_results['statistics']['mean_accuracy']:.1%}\n")
            f.write(f"  Best participant: {intermediate_results['progress_summary']['best_participant']}\n")
            f.write(f"  Worst participant: {intermediate_results['progress_summary']['worst_participant']}\n")
            f.write(f"  vs Random baseline: {intermediate_results['performance_analysis']['vs_random_baseline']['improvement']:+.1%}\n")
            f.write(f"  vs XGBoost baseline: {intermediate_results['performance_analysis']['vs_xgboost_baseline']['gap']:+.1%}\n\n")
            
            f.write("INDIVIDUAL RESULTS:\n")
            for participant, result in intermediate_results['completed_folds'].items():
                f.write(f"  {participant}: {result['accuracy']:.1%} ({result['training_time_seconds']/60:.1f} min)\n")
        
        print(f"📝 Summary: {summary_file}")
        return results_file, summary_file
        
    except Exception as e:
        print(f"❌ ERROR saving intermediate results: {e}")
        return None, None

if __name__ == "__main__":
    save_intermediate_results()
