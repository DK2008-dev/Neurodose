import pickle
import json

def load_detailed_results():
    # Load XGBoost full dataset results
    print("=== XGBOOST FULL DATASET RESULTS (July 17, 2025) ===")
    try:
        with open('data/processed/xgboost_full_dataset_results.pkl', 'rb') as f:
            xgb_data = pickle.load(f)
        
        print("\nDataset Info:")
        dataset_info = xgb_data.get('dataset_info', {})
        for key, value in dataset_info.items():
            print(f"  {key}: {value}")
        
        print("\nSimple Split Results (80/20):")
        simple_split = xgb_data.get('simple_split', {})
        for key, value in simple_split.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.3f}")
            else:
                print(f"  {key}: {value}")
        
        print("\nLeave-One-Participant-Out CV Results:")
        lopocv = xgb_data.get('lopocv', {})
        for key, value in lopocv.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.3f}")
            elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], float):
                print(f"  {key}: {[f'{v:.3f}' for v in value]}")
            else:
                print(f"  {key}: {value}")
        
        print(f"\nProcessing Time: {xgb_data.get('processing_time', 'N/A'):.1f} seconds")
        
    except Exception as e:
        print(f"Error loading XGBoost results: {e}")
    
    # Load Random Forest results if available
    rf_files = ['comprehensive_rf_results.pkl', 'corrected_literature_rf_results.pkl']
    
    for rf_file in rf_files:
        print(f"\n=== {rf_file.upper()} ===")
        try:
            with open(f'data/processed/{rf_file}', 'rb') as f:
                rf_data = pickle.load(f)
            
            print("Keys available:", list(rf_data.keys()))
            
            # Print summary if available
            if 'results' in rf_data:
                results = rf_data['results']
                print(f"LOPOCV Accuracy: {results.get('lopocv_accuracy', 'N/A')}")
                print(f"Processing Time: {results.get('processing_time', 'N/A')}")
            
        except Exception as e:
            print(f"Error loading {rf_file}: {e}")

if __name__ == "__main__":
    load_detailed_results()
