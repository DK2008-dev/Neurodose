"""
Example notebook for EEG pain classification analysis.

This notebook demonstrates the complete pipeline from data loading
to model training and evaluation.
"""

# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Add project root to path
project_root = Path.cwd().parent
sys.path.append(str(project_root))

from src.data.loader import EEGDataLoader
from src.features.spectral import SpectralFeatureExtractor
from src.models.cnn import create_model
from src.utils.helpers import *

print("EEG Pain Classification Analysis")
print("="*40)

# %%
# Configuration
DATA_DIR = project_root / "data" / "raw"
PROCESSED_DIR = project_root / "data" / "processed"

# Create sample data if raw data not available
if not DATA_DIR.exists():
    print("Raw data directory not found. Please download the OSF dataset.")
    print("Dataset: https://osf.io/bsv86/")
    print("Place .vhdr, .eeg, and .vmrk files in:", DATA_DIR)

# %%
# Example: Create synthetic data for demonstration
def create_synthetic_eeg_data(n_subjects=5, n_epochs_per_subject=100):
    """Create synthetic EEG data for demonstration."""
    np.random.seed(42)
    
    n_channels = 64
    n_samples = 2000  # 4 seconds at 500 Hz
    
    all_X = []
    all_y = []
    
    for subject in range(n_subjects):
        X_subject = []
        y_subject = []
        
        for epoch in range(n_epochs_per_subject):
            # Generate realistic EEG-like signal
            t = np.linspace(0, 4, n_samples)
            
            # Base noise
            signal = np.random.normal(0, 10, (n_channels, n_samples))
            
            # Add alpha rhythm (8-13 Hz) - stronger in posterior channels
            alpha_freq = np.random.uniform(8, 13)
            alpha_channels = list(range(50, 64))  # Posterior channels
            for ch in alpha_channels:
                signal[ch] += 20 * np.sin(2 * np.pi * alpha_freq * t)
            
            # Add pain-related modulation
            pain_level = np.random.choice([0, 1, 2])  # low, moderate, high
            
            # Pain affects central channels (C3, C4, Cz)
            pain_channels = [30, 31, 32]  # Approximate central locations
            pain_modulation = (pain_level + 1) * 5  # Stronger for higher pain
            
            for ch in pain_channels:
                # Add gamma activity for pain
                gamma_freq = np.random.uniform(30, 45)
                signal[ch] += pain_modulation * np.sin(2 * np.pi * gamma_freq * t)
                
                # Modulate alpha suppression
                alpha_suppression = pain_level * 0.3
                signal[ch] -= alpha_suppression * 15 * np.sin(2 * np.pi * 10 * t)
            
            X_subject.append(signal)
            y_subject.append(pain_level)
        
        all_X.append(np.array(X_subject))
        all_y.append(np.array(y_subject))
    
    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    
    print(f"Created synthetic dataset: {X.shape}")
    print(f"Label distribution: {np.bincount(y)}")
    
    return X, y

# Create synthetic data
X_synthetic, y_synthetic = create_synthetic_eeg_data()

# %%
# Data exploration
print("Data Shape:", X_synthetic.shape)
print("Labels:", np.unique(y_synthetic))
print("Label distribution:", {i: count for i, count in enumerate(np.bincount(y_synthetic))})

# Plot sample epochs
fig, axes = plt.subplots(3, 1, figsize=(12, 8))

for i, pain_level in enumerate(['Low', 'Moderate', 'High']):
    # Find first epoch of this pain level
    epoch_idx = np.where(y_synthetic == i)[0][0]
    epoch_data = X_synthetic[epoch_idx]
    
    # Plot first 10 channels
    for ch in range(10):
        axes[i].plot(epoch_data[ch] + ch * 50, alpha=0.7)
    
    axes[i].set_title(f'{pain_level} Pain (Epoch {epoch_idx})')
    axes[i].set_ylabel('Amplitude (μV)')
    if i == 2:
        axes[i].set_xlabel('Time (samples)')

plt.tight_layout()
plt.show()

# %%
# Feature extraction example
feature_extractor = SpectralFeatureExtractor()

# Extract features for a sample epoch
sample_epoch = X_synthetic[0]
sfreq = 500  # 500 Hz
ch_names = [f'Ch{i}' for i in range(64)]

features = feature_extractor.extract_features_single_epoch(
    sample_epoch, sfreq, ch_names
)

print(f"Extracted {len(features)} features")
print("Feature shape:", features.shape)

# %%
# Extract features for all epochs (subset for demo)
subset_size = 100
X_subset = X_synthetic[:subset_size]
y_subset = y_synthetic[:subset_size]

print(f"Extracting features for {subset_size} epochs...")
feature_matrix = feature_extractor.extract_features_batch(
    X_subset, sfreq, ch_names
)

print(f"Feature matrix shape: {feature_matrix.shape}")

# %%
# Visualize feature distributions
pain_labels = ['Low', 'Moderate', 'High']

# Plot some key features
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Select a few interesting features
feature_indices = [0, 10, 20, 30]  # Arbitrary selection for demo

for i, feat_idx in enumerate(feature_indices):
    ax = axes[i//2, i%2]
    
    for pain_level in range(3):
        mask = y_subset == pain_level
        values = feature_matrix[mask, feat_idx]
        ax.hist(values, alpha=0.6, label=pain_labels[pain_level], bins=20)
    
    ax.set_title(f'Feature {feat_idx}')
    ax.set_xlabel('Feature Value')
    ax.set_ylabel('Count')
    ax.legend()

plt.tight_layout()
plt.show()

# %%
# Model creation and architecture
model_configs = {
    'eegnet': {
        'n_channels': 64,
        'n_samples': 2000,
        'n_classes': 3,
        'dropout_rate': 0.5
    },
    'shallow': {
        'n_channels': 64,
        'n_samples': 2000,
        'n_classes': 3,
        'dropout_rate': 0.5
    }
}

# Create models
models = {}
for model_type, config in model_configs.items():
    models[model_type] = create_model(model_type, **config)
    print(f"\n{model_type.upper()} Model:")
    print_model_summary(models[model_type], (config['n_channels'], config['n_samples']))

# %%
# Quick training example with synthetic data
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

# Prepare data
X_train, X_val, X_test, y_train, y_val, y_test = split_data(
    X_synthetic, y_synthetic, test_size=0.2, val_size=0.1
)

# Create data loaders
train_loader = create_data_loader(X_train, y_train, batch_size=16, shuffle=True)
val_loader = create_data_loader(X_val, y_val, batch_size=16, shuffle=False)

# Train a simple model (reduced epochs for demo)
model = models['eegnet']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

print(f"Training on device: {device}")
print("Quick training demo (5 epochs)...")

model.train()
for epoch in range(5):
    total_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/5 - Loss: {avg_loss:.4f}")

print("Training demo completed!")

# %%
# Evaluation on validation set
model.eval()
correct = 0
total = 0
predictions = []
targets = []

with torch.no_grad():
    for data, target in val_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        pred = output.argmax(dim=1)
        
        correct += pred.eq(target).sum().item()
        total += target.size(0)
        
        predictions.extend(pred.cpu().numpy())
        targets.extend(target.cpu().numpy())

accuracy = 100. * correct / total
print(f"Validation Accuracy: {accuracy:.2f}%")

# Plot confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(targets, predictions)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=pain_labels, yticklabels=pain_labels)
plt.title('Confusion Matrix (Synthetic Data Demo)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# %%
print("Analysis completed!")
print("\nNext steps:")
print("1. Download real EEG data from OSF: https://osf.io/bsv86/")
print("2. Run preprocessing: python scripts/preprocess_data.py --data_dir data/raw")
print("3. Train model: python scripts/train_model.py")
print("4. Test real-time: python scripts/real_time_predict.py --simulate")
