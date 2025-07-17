import pickle

# Load and format Random Forest results for documentation
def format_rf_results():
    try:
        # Comprehensive RF results
        with open('data/processed/comprehensive_rf_results.pkl', 'rb') as f:
            comp_rf = pickle.load(f)
        
        print("=== RANDOM FOREST COMPREHENSIVE RESULTS ===")
        print(f"LOPOCV Accuracy: {comp_rf['lopocv_mean']:.3f} ± {comp_rf['lopocv_std']:.3f}")
        print(f"CV Accuracy: {comp_rf['cv_mean']:.3f} ± {comp_rf['cv_std']:.3f}")
        print(f"Features: {comp_rf['n_features']}")
        print(f"Accuracy Range: {min(comp_rf['lopocv_scores']):.3f} - {max(comp_rf['lopocv_scores']):.3f}")
        
    except Exception as e:
        print(f"Error loading comprehensive RF: {e}")
    
    try:
        # Corrected Literature RF results
        with open('data/processed/corrected_literature_rf_results.pkl', 'rb') as f:
            lit_rf = pickle.load(f)
        
        print("\n=== RANDOM FOREST LITERATURE METHOD RESULTS ===")
        print(f"CV Accuracy: {lit_rf['cv_mean']:.3f} ± {lit_rf['cv_std']:.3f}")
        print(f"Features: {lit_rf['n_features']}")
        print(f"Baseline Accuracy: {lit_rf['baseline_accuracy']:.3f}")
        print(f"Improvement: {lit_rf['improvement_over_baseline']:.3f}")
        print(f"Methodology: {lit_rf['methodology']}")
        
    except Exception as e:
        print(f"Error loading literature RF: {e}")

if __name__ == "__main__":
    format_rf_results()
