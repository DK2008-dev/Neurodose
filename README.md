# EEG Pain Classification System

A real-time EEG-based pain severity classification system using deep learning and Lab Streaming Layer (LSL) integration.

## Overview

This project implements an end-to-end pipeline for:
- **Offline Model Training**: Train a ternary pain classifier (low/moderate/high) using raw EEG data
- **Real-Time Deployment**: Live pain assessment via LSL streaming with 4-second sliding windows
- **Clinical Application**: Objective pain measurement for non-communicative patients

## Dataset

Uses the OSF "Brain Mediators for Pain" dataset (Tiemann et al., 2018) with:
- 51 participants with laser-evoked pain stimuli
- 65 EEG channels + 2 EOG channels at 1000 Hz
- BrainVision format (.vhdr, .eeg, .vmrk files)
- Three pain intensities: 30%, 50%, 70% of individual pain threshold

## Key Features

### Preprocessing Pipeline
- 1 Hz high-pass filter, 50 Hz notch filter
- 500 Hz resampling, common average reference
- ICA-based artifact removal
- 4-second sliding windows with 1-second steps

### Feature Extraction
- Spectral band powers (delta, theta, alpha, beta, gamma)
- Pain-relevant electrode selection (C4, Cz, FCz)
- ROI-based feature aggregation
- Band power ratios (gamma/alpha, beta/alpha)

### Deep Learning Model
- CNN architecture for raw EEG input
- Ternary classification: low/moderate/high pain
- End-to-end learning from spatio-temporal signals

### Real-Time System
- LSL integration for live EEG streaming
- Continuous 4-second window processing
- Real-time pain severity predictions

## Project Structure

```
├── src/
│   ├── data/              # Data loading and preprocessing
│   ├── features/          # Feature extraction methods
│   ├── models/            # CNN architectures and training
│   ├── streaming/         # LSL real-time processing
│   └── utils/             # Utility functions
├── config/                # Configuration files
├── notebooks/             # Jupyter analysis notebooks
├── scripts/               # Training and evaluation scripts
├── tests/                 # Unit tests
└── data/                  # Raw and processed data
```

## Quick Start

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Download Dataset**
   - Download from OSF: https://osf.io/bsv86/
   - Place in `data/raw/` directory

3. **Preprocess Data**
   ```bash
   python scripts/preprocess_data.py --data_dir data/raw --output_dir data/processed
   ```

4. **Train Model**
   ```bash
   python scripts/train_model.py --config config/cnn_config.yaml
   ```

5. **Real-Time Streaming**
   ```bash
   python scripts/real_time_predict.py --model_path models/best_model.pth
   ```

## Citation

Based on the methodology from:
- Tiemann et al. (2018). "A pain-based brain signature for quantifying pain" 
- Al-Nafjan et al. (2017). "Review and Classification of Emotion Recognition based on EEG"

## License

This project is for research purposes only.
