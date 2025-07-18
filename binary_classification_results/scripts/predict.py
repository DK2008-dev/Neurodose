#!/usr/bin/env python3
"""
Binary Pain Classifier - Prediction Script
Usage: python predict.py <input_file.npy>
"""

import pickle
import numpy as np
import sys
from pathlib import Path

def predict_pain(input_file):
    # Load model
    model_path = Path(__file__).parent.parent / 'models' / 'binary_model.pkl'
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    model = model_data['model']
    scaler = model_data['scaler']
    
    # Load input data
    if input_file.endswith('.npy'):
        features = np.load(input_file)
    else:
        print("Unsupported format. Use .npy file with extracted features.")
        return
    
    # Scale and predict
    features_scaled = scaler.transform(features.reshape(1, -1))
    label = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0]
    
    print(f"Predicted Label: {label} ({'High Pain' if label == 1 else 'Low Pain'})")
    print(f"Low Pain Probability: {probability[0]:.3f}")
    print(f"High Pain Probability: {probability[1]:.3f}")
    print(f"Confidence: {np.max(probability):.3f}")

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python predict.py <input_file.npy>")
        sys.exit(1)
    predict_pain(sys.argv[1])
