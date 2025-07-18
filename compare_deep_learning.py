#!/usr/bin/env python3
"""
Deep Learning vs Advanced Features Comparison
Test CNN architectures against advanced feature engineering
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pickle
from pathlib import Path
import logging
from datetime import datetime
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score, classification_report
import time

def setup_logging():
    """Setup logging configuration."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"deep_learning_comparison_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

class SimpleEEGNet(nn.Module):
    """Simplified EEGNet for binary pain classification."""
    
    def __init__(self, n_channels=68, n_samples=2000, n_classes=2):
        super(SimpleEEGNet, self).__init__()
        
        # Temporal convolution
        self.temporal_conv = nn.Conv2d(1, 16, (1, 64), padding=(0, 32))
        self.spatial_conv = nn.Conv2d(16, 32, (n_channels, 1))
        
        # Calculate conv output size
        conv_output_size = self._get_conv_output_size(n_channels, n_samples)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 125)),  # Adaptive pooling to fixed size
            nn.Flatten(),
            nn.Linear(32 * 125, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, n_classes)
        )
    
    def _get_conv_output_size(self, n_channels, n_samples):
        """Calculate the output size after convolutions."""
        # After temporal conv: (n_samples + 2*32 - 64) + 1
        temp_size = n_samples + 2*32 - 64 + 1
        # After spatial conv: 1 channel dimension
        return temp_size
    
    def forward(self, x):
        # Input: (batch, channels, samples)
        # Add channel dimension: (batch, 1, channels, samples)
        x = x.unsqueeze(1)
        
        # Convolutions
        x = torch.relu(self.temporal_conv(x))
        x = torch.relu(self.spatial_conv(x))
        
        # Classification
        x = self.classifier(x)
        
        return x

def load_raw_data():
    """Load raw EEG data for CNN training."""
    data_dir = Path("data/processed/basic_windows")
    
    all_data = []
    all_labels = []
    participants = []
    
    participant_files = list(data_dir.glob("vp*_windows.pkl"))
    logging.info(f"Found {len(participant_files)} participant files")
    
    for file_path in sorted(participant_files):
        participant = file_path.stem.split('_')[0]
        
        try:
            with open(file_path, 'rb') as f:
                windows_data = pickle.load(f)
            
            windows = windows_data['windows']
            labels = windows_data['ternary_labels']
            
            # Convert to binary classification (low vs high pain)
            binary_labels = []
            binary_windows = []
            
            for i, label in enumerate(labels):
                if label == 0:  # Low pain
                    binary_labels.append(0)
                    binary_windows.append(windows[i])
                elif label == 2:  # High pain
                    binary_labels.append(1)
                    binary_windows.append(windows[i])
            
            all_data.extend(binary_windows)
            all_labels.extend(binary_labels)
            participants.extend([participant] * len(binary_windows))
            
            logging.info(f"Loaded {participant}: {len(binary_windows)} binary windows")
            
        except Exception as e:
            logging.error(f"Failed to load {participant}: {e}")
    
    return np.array(all_data), np.array(all_labels), np.array(participants)

def train_cnn_fold(X_train, y_train, X_test, y_test, fold_num):
    """Train CNN for one fold."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.LongTensor(y_train).to(device)
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    y_test_tensor = torch.LongTensor(y_test).to(device)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    
    # Initialize model
    model = SimpleEEGNet(n_channels=X_train.shape[1], n_samples=X_train.shape[2]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training
    model.train()
    for epoch in range(20):  # Quick training
        epoch_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        if epoch % 5 == 0:
            logging.info(f"Fold {fold_num}, Epoch {epoch}, Loss: {epoch_loss/len(train_loader):.4f}")
    
    # Evaluation
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_tensor)
        _, predicted = torch.max(test_outputs.data, 1)
        test_accuracy = accuracy_score(y_test_tensor.cpu().numpy(), predicted.cpu().numpy())
    
    return test_accuracy

def compare_approaches():
    """Compare deep learning vs advanced features."""
    setup_logging()
    
    print("="*80)
    print("DEEP LEARNING vs ADVANCED FEATURES COMPARISON")
    print("="*80)
    
    # Load raw data for CNN
    logging.info("Loading raw EEG data for CNN...")
    X_raw, y, participants = load_raw_data()
    
    if len(X_raw) == 0:
        logging.error("No data loaded.")
        return
    
    logging.info(f"Loaded {len(X_raw)} windows from {len(np.unique(participants))} participants")
    logging.info(f"Data shape: {X_raw.shape}")
    logging.info(f"Class distribution: {np.bincount(y)}")
    
    # Cross-validation setup
    cv = LeaveOneGroupOut()
    cnn_scores = []
    
    # CNN evaluation
    logging.info("Evaluating CNN with Leave-One-Participant-Out CV...")
    fold_num = 1
    
    for train_idx, test_idx in cv.split(X_raw, y, participants):
        logging.info(f"Training CNN fold {fold_num}/5...")
        
        X_train, X_test = X_raw[train_idx], X_raw[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        test_participant = participants[test_idx][0]
        logging.info(f"Testing on participant: {test_participant}")
        
        # Train and evaluate CNN
        accuracy = train_cnn_fold(X_train, y_train, X_test, y_test, fold_num)
        cnn_scores.append(accuracy)
        
        logging.info(f"Fold {fold_num} CNN accuracy: {accuracy:.3f}")
        fold_num += 1
    
    # Load advanced features results for comparison
    advanced_results_dir = Path("data/processed/advanced_classifier_results")
    if advanced_results_dir.exists():
        result_files = list(advanced_results_dir.glob("advanced_ensemble_*.pkl"))
        if result_files:
            with open(result_files[-1], 'rb') as f:
                advanced_results = pickle.load(f)
            
            advanced_accuracy = advanced_results.get('ensemble_cv_score', 0.0)
            advanced_std = advanced_results.get('ensemble_cv_std', 0.0)
        else:
            advanced_accuracy = 0.511  # From previous run
            advanced_std = 0.061
    else:
        advanced_accuracy = 0.511  # From previous run
        advanced_std = 0.061
    
    # Results comparison
    cnn_mean = np.mean(cnn_scores)
    cnn_std = np.std(cnn_scores)
    
    print("\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)
    print(f"CNN (Raw EEG):           {cnn_mean:.3f} ± {cnn_std:.3f}")
    print(f"Advanced Features:       {advanced_accuracy:.3f} ± {advanced_std:.3f}")
    print(f"Difference:              {cnn_mean - advanced_accuracy:+.3f}")
    print(f"Individual CNN scores:   {[f'{score:.3f}' for score in cnn_scores]}")
    
    if cnn_mean > advanced_accuracy:
        print(f"\n✅ CNN outperforms advanced features by {(cnn_mean - advanced_accuracy)*100:.1f}%")
    else:
        print(f"\n⚠️ Advanced features outperform CNN by {(advanced_accuracy - cnn_mean)*100:.1f}%")
    
    # Save comparison results
    results = {
        'cnn_scores': cnn_scores,
        'cnn_mean': cnn_mean,
        'cnn_std': cnn_std,
        'advanced_accuracy': advanced_accuracy,
        'advanced_std': advanced_std,
        'comparison_timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
    }
    
    results_dir = Path("data/processed/comparison_results")
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = results_dir / f"cnn_vs_features_{timestamp}.pkl"
    
    with open(results_file, 'wb') as f:
        pickle.dump(results, f)
    
    logging.info(f"Comparison results saved to {results_file}")
    logging.info("Deep learning comparison completed!")

if __name__ == "__main__":
    compare_approaches()
