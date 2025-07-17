import pickle
import os

def load_and_display_results():
    # Load Random Forest results
    rf_files = [
        'comprehensive_rf_results.pkl',
        'corrected_literature_rf_results.pkl', 
        'quick_rf_test_results.pkl'
    ]
    
    print("=== RANDOM FOREST RESULTS ===")
    for rf_file in rf_files:
        path = f'data/processed/{rf_file}'
        if os.path.exists(path):
            try:
                with open(path, 'rb') as f:
                    data = pickle.load(f)
                print(f"\n{rf_file}:")
                print(f"  Simple Split Accuracy: {data.get('simple_split_results', {}).get('accuracy', 'N/A')}")
                print(f"  LOPOCV Mean Accuracy: {data.get('lopocv_results', {}).get('mean_accuracy', 'N/A')}")
                print(f"  Total Samples: {data.get('dataset_info', {}).get('total_samples', 'N/A')}")
                print(f"  Participants: {data.get('dataset_info', {}).get('n_participants', 'N/A')}")
            except Exception as e:
                print(f"  Error loading {rf_file}: {e}")
    
    # Load XGBoost results
    xgb_files = [
        'xgboost_full_dataset_results.pkl',
        'quick_xgb_test_results.pkl'
    ]
    
    print("\n=== XGBOOST RESULTS ===")
    for xgb_file in xgb_files:
        path = f'data/processed/{xgb_file}'
        if os.path.exists(path):
            try:
                with open(path, 'rb') as f:
                    data = pickle.load(f)
                print(f"\n{xgb_file}:")
                print(f"  Simple Split Accuracy: {data.get('simple_split_results', {}).get('accuracy', 'N/A')}")
                print(f"  LOPOCV Mean Accuracy: {data.get('lopocv_results', {}).get('mean_accuracy', 'N/A')}")
                print(f"  Total Samples: {data.get('dataset_info', {}).get('total_samples', 'N/A')}")
                print(f"  Participants: {data.get('dataset_info', {}).get('n_participants', 'N/A')}")
            except Exception as e:
                print(f"  Error loading {xgb_file}: {e}")

if __name__ == "__main__":
    load_and_display_results()
