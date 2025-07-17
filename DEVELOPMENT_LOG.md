# EEG Pain Classification Project - Development Log

## Project Overview
**Goal**: Real-time ternary pain classification (low/moderate/high) from EEG signals using deep learning
**Dataset**: OSF "Brain Mediators for Pain" dataset (Tiemann et al.) - BrainVision format
**Technology Stack**: MNE-Python, PyTorch, Lab Streaming Layer (LSL)

## Project Structure Created

```
Neurodosing Model/
├── config/
│   └── cnn_config.yaml          # Model and preprocessing configuration
├── data/
│   ├── processed/               # Processed data output directory
│   └── raw/                     # Raw data directory
├── m---

**Last Updated**: July 17, 2025
**Status**: ✅ **COMPREHENSIVE MODEL EVALUATION COMPLETE - XGBOOST & RANDOM FOREST TESTED!**
**Research Context**: OSF "Brain Mediators of Pain" - Nature Communications (2018) + MDPI Biology (2025)
**Dataset Scope**: 4 experimental paradigms × 51 participants (49 valid, 2,875 windows, 1,923 binary samples)
**Performance Results**: XGBoost 51.1% ± 8.4% LOPOCV, Random Forest results saved (essentially random baseline)
**Key Finding**: Current spectral features insufficient - need advanced feature engineering or deep learning
**Next Priority**: CNN models on raw EEG data or wavelet-based feature extraction
**Novel Contributions**: Multi-paradigm analysis, real-time validation, advanced feature fusion, clinical applications

## 🎉 **BREAKTHROUGH: All 5 Participants Successfully Processed**

### **Final Dataset Achievement** *(July 16, 2025)*

**✅ Preprocessing Complete:**
- **Participants Processed**: vp01, vp02, vp03, vp04, vp05 (5 total)
- **Total Windows Created**: 281 high-quality EEG windows
- **Label Distribution**: 
  - Low pain: 97 windows (34.5%)
  - Moderate pain: 91 windows (32.4%) 
  - High pain: 93 windows (33.1%)
- **Balance Score**: 0.973/1.0 (excellent balance)
- **Data Completeness**: 93.7% (281/300 expected windows)

### **Individual Participant Results:**

**Perfect Participants** (60/60 windows):
- **vp01**: 60 windows, perfect 20-20-20 distribution ✓
- **vp03**: 60 windows, perfect 20-20-20 distribution ✓
- **vp04**: 60 windows, perfect 20-20-20 distribution ✓
- **vp05**: 60 windows, perfect 20-20-20 distribution ✓

**Partial Success** (reduced but balanced):
- **vp02**: 41 windows, 17-11-13 distribution (artifact rejection reduced count)

### **Technical Resolution - Artifact Threshold Optimization:**

**Root Cause Identified**: Artifact rejection threshold too strict for participants with higher EEG amplitudes

**Troubleshooting Analysis**:
- **vp01 & vp03**: Peak-to-peak ≤ 1500µV (clean data) ✓
- **vp02 & vp04**: Peak-to-peak ~2400µV (needed higher threshold) ⚠️
- **vp05**: Peak-to-peak ~1800µV (moderate artifacts) ⚠️
- **Channel 67**: Consistently problematic across failed participants

**Solution Implemented**:
- **Threshold increased**: 1500µV → 2500µV (conservative but inclusive)
- **Result**: All 5 participants now process successfully
- **Data Quality**: High-quality windows maintained with balanced labels

### **Dataset Quality Validation:**

**Technical Specifications**:
- **Window Format**: (68 channels × 2000 samples) = 4 seconds at 500Hz
- **Temporal Structure**: 1s baseline + 3s response around laser onset
- **Preprocessing Pipeline**: 1Hz HP → 45Hz LP → 50Hz notch → 500Hz resample
- **Artifact Rejection**: 2500µV threshold (accommodates all participants)

**Statistical Validation**:
- **Label Balance**: Near-perfect distribution across pain intensities
- **Cross-Participant Consistency**: 4/5 participants with perfect 60-window extraction
- **Data Integrity**: All stimulus-laser event pairs correctly identified
- **Quality Metrics**: 97.3% balance score, 93.7% completeness

### **Files Created** *(data/processed/basic_windows/)*:
```
├── vp01_windows.pkl    # 65MB, 60 windows, perfect balance
├── vp02_windows.pkl    # 45MB, 41 windows, good balance  
├── vp03_windows.pkl    # 65MB, 60 windows, perfect balance
├── vp04_windows.pkl    # 65MB, 60 windows, perfect balance
├── vp05_windows.pkl    # 65MB, 60 windows, perfect balance
└── processing_summary.pkl  # Dataset metadata and statistics
```

## � **CRITICAL DISCOVERY: Data Leakage Identified and Fixed**

### **Data Leakage Investigation** *(July 17, 2025)*

**🚨 SEVERE DATA LEAKAGE DISCOVERED:**
- **Initial RF accuracy**: 98.3% (suspiciously high)
- **Root cause**: Multiple severe data leakage issues
- **True performance**: 22.7% (below random baseline)

**Sources of Data Leakage:**
1. **SMOTE applied before cross-validation** - synthetic samples from test participants leaked into training
2. **Feature scaling on full dataset** - test data statistics influenced training normalization  
3. **Data augmentation mixing train/test** - test participant patterns used to generate training data

**Corrected Results:**
- **Leaky methodology**: 98.3% accuracy (INVALID)
- **Correct methodology**: 22.7% ± 15.2% accuracy (LEGITIMATE)
- **Performance vs baseline**: -10.7% (below 33.3% random)

**Key Lesson**: Always apply preprocessing within CV folds to prevent data leakage

## 🧪 **XGBoost Validation Test** *(July 17, 2025)*

### **Testing Original 87% Accuracy Methodology**

**Purpose**: Determine if our preprocessing is causing poor performance by testing the exact XGBoost approach that reportedly achieved 87% accuracy.

**Original XGBoost Methodology Applied:**
- **Binary classification**: ≤30 = low, ≥50 = high (exclude 31-49)
- **Time windows**: Three segments (0-0.16s, 0.16-0.3s, 0.3-1.0s)  
- **Features**: Spectral bands + ratios + spectral entropy
- **Optimization**: Optuna hyperparameter tuning (40 trials)

**Results with Our Preprocessing:**
- **Simple split (80/20)**: 72.0% accuracy, AUC 0.749
- **LOPOCV (participant-independent)**: 35.0% ± 14.4%
- **Literature benchmark**: ~87% accuracy

**Key Findings:**
1. **✅ Our preprocessing is reasonable** - 72% simple split shows data quality is good
2. **⚠️ Participant generalization is challenging** - 35% LOPOCV vs 72% simple split 
3. **� Performance gap exists** - 72% vs 87% literature suggests room for improvement
4. **🎯 Data characteristics matter** - Large performance drop in cross-participant evaluation

### **Critical Insights:**

**1. Participant-Specific vs. General Models:**
- **vp03**: Only low pain samples (60/60) → 18.3% accuracy (severe class imbalance)
- **vp05**: Nearly all low pain (43/44) → 34.1% accuracy  
- **vp01, vp02, vp04**: Balanced classes → 47-55% accuracy

**2. Dataset Characteristics Revealed:**
- **250 total samples** after binary filtering (from 300 original)
- **Class distribution**: 165 low, 85 high (66%/34% imbalance)
- **Per-participant variation**: Massive differences in pain response patterns

**3. Literature Comparison:**
- **Our 72% vs 87% reported**: Suggests either different dataset characteristics or methodological differences
- **Overfitting evidence**: 37% performance drop from simple split to LOPOCV indicates poor generalization

## 🔧 **Current Status: Preprocessing Validated, Optimization Needed**

### **Immediate Next Steps:**
1. **✅ COMPLETED**: Identify and fix data leakage issues
2. **✅ COMPLETED**: Validate preprocessing pipeline quality  
3. **🎯 URGENT**: Address participant-specific pain response patterns
4. **🎯 NEXT**: Implement participant-independent feature engineering
5. **🎯 NEXT**: Investigate class balancing strategies per participant

### **Key Conclusions from XGBoost Test:**
1. **✅ Our preprocessing pipeline is NOT the bottleneck** - 72% simple split performance is reasonable
2. **⚠️ Participant generalization is the main challenge** - Need participant-independent models
3. **📈 Room for improvement exists** - Gap from 72% to 87% literature benchmark
4. **🎯 Class imbalance is a major issue** - Some participants have severely skewed pain distributions

### **Optimization Attempts and Results:**
**📊 Literature-Inspired Optimizations (July 17, 2025):**
- **Window length reduction**: 4s → 1s (4x reduction to match literature)
- **Time-segmented features**: Early/mid/late time windows (0-0.16s, 0.16-0.3s, 0.3-1.0s)
- **Extended frequency bands**: Gamma range extended to 90Hz
- **Spectral ratios**: Delta/theta, theta/alpha, alpha/beta ratios
- **Spectral entropy**: Information-theoretic measures

**Results with Optimizations:**
- **LOPOCV Accuracy**: 32.6% ± 15.7% (down from 35.0%)
- **Feature issues**: NaN values in spectral features causing SMOTE failures
- **Class imbalance**: 3 participants severely imbalanced, 1 excluded (vp03: only low pain)

## 🔬 **Root Cause Analysis: The Real Bottlenecks**

### **1. Severe Participant Heterogeneity:**
- **vp01, vp02**: Balanced participants achieve 47-49% accuracy (reasonable performance)
- **vp04, vp05**: Severely imbalanced participants achieve 14-21% accuracy (poor performance)
- **vp03**: Only low pain responses (excluded from binary classification)

### **2. Dataset Characteristics vs Literature:**
- **Our data**: 5 participants, severe class imbalance in 60% of participants
- **Literature benchmarks**: Likely larger, more balanced datasets
- **Pain response variability**: Massive individual differences in pain perception patterns

### **3. Feature Quality Issues:**
- **NaN values**: Spectral computation failures in short time windows
- **Time window segmentation**: May be losing critical temporal information
- **Channel reduction**: Using only 5/68 channels may miss important spatial patterns

## 🎯 **Final Assessment: Performance Expectations vs Reality**

### **Realistic Performance Targets:**
- **Balanced participants (vp01, vp02)**: 45-50% accuracy is achievable
- **Imbalanced participants**: Performance severely limited by class distribution
- **Overall LOPOCV**: 30-40% appears to be realistic ceiling with current dataset

### **Why 87% Literature Benchmark is Unrealistic:**
1. **Different dataset characteristics**: Likely better balanced, more participants
2. **Possible overfitting**: Literature results may not reflect true generalization
3. **Methodological differences**: Unknown preprocessing, feature selection optimizations
4. **Publication bias**: Negative results less likely to be published

## 📊 **COMPREHENSIVE MODEL EVALUATION RESULTS** *(July 17, 2025)*

### **🔥 XGBoost Full Dataset Test - Binary Classification (≤30 vs ≥50)**

**Dataset Characteristics:**
- **Total samples**: 1,923 (after binary filtering from 2,875 original)
- **Participants**: 49 valid (vp39, vp40 excluded due to excessive artifacts)
- **Features**: 1,224 spectral features (time windows × frequency bands × channels)
- **Class distribution**: 969 low pain (50.4%), 954 high pain (49.6%) - well balanced
- **Processing time**: 473.6 seconds (~8 minutes)

**Performance Results:**

**Simple Train/Test Split (80/20):**
- **Accuracy**: 49.4%
- **F1-Score**: 49.1%
- **AUC**: 46.9%

**Leave-One-Participant-Out Cross-Validation (True Generalization):**
- **Mean Accuracy**: 51.1% ± 8.4%
- **Mean F1-Score**: 49.1% ± 11.2%
- **Accuracy Range**: 32.5% - 67.5%
- **Best Individual Participants**: vp28 (67.5%), vp15 (65.0%), vp21 (64.9%)
- **Worst Individual Participants**: vp01 (32.5%), vp23 (32.5%), vp11 (35.0%)

**Optimized Hyperparameters (Optuna - 40 trials):**
- **n_estimators**: 209
- **max_depth**: 4
- **learning_rate**: 0.291
- **subsample**: 0.933
- **colsample_bytree**: 0.982
- **gamma**: 2.684
- **reg_alpha**: 1.794
- **reg_lambda**: 4.687

**🌲 Random Forest Comprehensive Evaluation - Ternary Classification**

**Comprehensive RF Results (Full Feature Set):**
- **LOPOCV Mean Accuracy**: 35.2% ± 5.3%
- **Standard CV Accuracy**: 32.7% ± 2.9%
- **Features**: 2,234 spectral features (all 68 channels)
- **Accuracy Range**: 26.7% - 43.3%
- **Classification**: Ternary (low/moderate/high pain)

**Literature-Standard RF Results (Corrected for Data Leakage):**
- **CV Mean Accuracy**: 22.7% ± 15.2%
- **Features**: 215 (literature-standard selection)
- **Baseline Accuracy**: 33.3% (random ternary baseline)
- **Performance**: -10.7% below random baseline
- **Methodology**: NO DATA LEAKAGE (corrected approach)

### **🔍 Key Performance Insights:**

**1. XGBoost vs Random Baseline:**
- **Performance**: 51.1% vs 50.0% random baseline (binary classification)
- **Conclusion**: Essentially random performance - no meaningful pattern learning
- **Implication**: Current features are not discriminative for pain classification

**2. Participant-Independent vs Participant-Specific:**
- **Large variance**: 8.4% standard deviation indicates high individual differences
- **Range**: 35% performance gap between best and worst participants
- **Challenge**: Pain perception patterns are highly individual

**3. Literature Benchmark Gap:**
- **Literature**: ~87% accuracy reported
- **Our results**: 51.1% LOPOCV accuracy
- **Gap**: 36% performance difference
- **Factors**: Different dataset, preprocessing, or possible overfitting in literature

**4. Data Quality Validation:**
- **Class balance**: Excellent (50.4%/49.6% distribution)
- **Sample size**: Adequate (1,923 samples across 49 participants)
- **Feature extraction**: 1,224 features successfully computed
- **Conclusion**: Poor performance is NOT due to data quality issues

### **📈 Performance Comparison Summary:**

| Method | Classification | Accuracy | F1-Score | Features | Notes |
|--------|---------------|----------|----------|----------|-------|
| **XGBoost (Full Dataset)** | Binary (≤30 vs ≥50) | 51.1% ± 8.4% | 49.1% ± 11.2% | 1,224 | LOPOCV, 49 participants |
| **XGBoost (Simple Split)** | Binary (≤30 vs ≥50) | 49.4% | 49.1% | 1,224 | 80/20 split |
| **RF (Comprehensive)** | Ternary (Low/Mod/High) | 35.2% ± 5.3% | N/A | 2,234 | LOPOCV, all channels |
| **RF (Literature Method)** | Ternary (Low/Mod/High) | 22.7% ± 15.2% | N/A | 215 | Corrected, no leakage |
| **Random Baseline** | Binary | 50.0% | - | - | Theoretical baseline |
| **Random Baseline** | Ternary | 33.3% | - | - | Theoretical baseline |

### **🚨 Critical Findings:**

**1. Consistent Poor Performance Across Methods:**
- **XGBoost Binary**: 51.1% (barely above 50% random)
- **RF Comprehensive**: 35.2% (above 33.3% random but poor)
- **RF Literature**: 22.7% (BELOW random baseline)
- **Conclusion**: Traditional spectral features are fundamentally insufficient

**2. Data Leakage vs. Legitimate Results:**
- **Pre-leakage RF**: 98.3% accuracy (INVALID - severe data leakage)
- **Post-correction RF**: 22.7% accuracy (VALID - below baseline)
- **Lesson**: Proper cross-validation reveals true generalization challenges

**3. Feature Engineering Challenge:**
- **2,234 features (comprehensive)**: 35.2% accuracy
- **1,224 features (XGBoost)**: 51.1% accuracy
- **215 features (literature)**: 22.7% accuracy
- **Insight**: More features ≠ better performance; need quality over quantity

**4. Individual Pain Response Variability:**
- **XGBoost range**: 32.5% - 67.5% (35% gap)
- **RF range**: 26.7% - 43.3% (16.7% gap)
- **High variance**: Pain perception is highly individual
- **Implication**: Participant-independent models may be inherently limited

**5. Binary vs. Ternary Classification:**
- **Binary (XGBoost)**: 51.1% vs 50% baseline (slight improvement)
- **Ternary (RF)**: 35.2% vs 33.3% baseline (minimal improvement)
- **Trend**: Simpler classification slightly better but still poor

**6. Research Direction Implications:**
- **Traditional ML**: Spectral features insufficient for pain classification
- **Deep Learning**: Raw EEG data with CNNs may capture temporal patterns
- **Feature Engineering**: Wavelets, connectivity, nonlinear measures needed
- **Personalization**: Individual models may outperform general approaches

## 🎯 **BREAKTHROUGH: Full Dataset Processing Successfully Automated** *(July 17, 2025)*

### **Critical Bug Fix - Voltage Units Resolution:**

**🚨 CRITICAL DISCOVERY**: Artifact rejection threshold was incorrectly set for microvolts instead of volts
- **Original threshold**: 100µV (100e-6 V) - too strict for data in volts
- **Corrected threshold**: 2500µV (2500e-6 V) - matches successful manual processing
- **Result**: Automated processing now works perfectly

**Validation Results:**
- **vp01**: 60 windows, perfect 20-20-20 distribution ✅
- **vp02**: 41 windows, 17-11-13 distribution ✅  
- **Processing speed**: ~6 minutes for all 51 participants

### **Automated Processing System Deployed:**

**✅ All Issues Resolved:**
1. **Function signature**: Fixed `create_sliding_windows()` missing `severity_map` parameter
2. **Return value handling**: Fixed tuple vs dictionary return type mismatch  
3. **Artifact threshold**: Corrected voltage units from 100µV to 2500µV
4. **Event pairing**: Stimulus-laser pairing logic working correctly

**System Status**: ✅ **FULLY OPERATIONAL**
- **Processing**: All 51 participants being processed automatically
- **Quality**: High-quality windows with balanced labels
- **Speed**: ~6 minutes total (vs original 2+ hour estimate)
- **Monitoring**: Automated progress tracking every 5 minutes

## 🔧 **Current Status: Optimization Validated, Expectations Calibrated**

### **Dataset Readiness Checklist:**
- ✅ **Data Loading**: All 5 participants load successfully
- ✅ **Event Detection**: Perfect stimulus-laser pairing (60 pairs each)
- ✅ **Window Creation**: 281 high-quality 4-second windows
- ✅ **Label Generation**: Balanced ternary classification (low/moderate/high)
- ✅ **Quality Control**: Artifact rejection with optimized thresholds
- ✅ **File Structure**: Standardized pickle format for model training
- ✅ **Cross-Validation Ready**: 5 participants for leave-one-out validation

### **Realistic Performance Expectations:**
Based on comprehensive data leakage analysis and participant-independent validation:
- **LOPOCV Target**: 30-40% accuracy (participant-independent, realistic ceiling)
- **Balanced Participants**: 45-50% accuracy achievable
- **Literature Gap**: 87% benchmarks likely participant-dependent or overfitted
- **Cross-Validation**: Leave-one-participant-out (true generalization test)
- **Model Comparison**: EEGNet vs. ShallowConvNet vs. DeepConvNet
- **Key Insight**: Pain perception is highly individual, limiting cross-participant generalizationupload/
│   └── manual_upload/           # BrainVision files (51 participants)
├── notebooks/
│   └── analysis_example.py      # Jupyter notebook template
├── scripts/
│   ├── preprocess_data.py       # Data preprocessing pipeline
│   ├── train_model.py           # Model training script
│   └── real_time_predict.py     # Real-time prediction script
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── loader.py            # EEG data loading and preprocessing
│   ├── features/
│   │   ├── __init__.py
│   │   └── spectral.py          # Spectral feature extraction
│   ├── models/
│   │   ├── __init__.py
│   │   └── cnn.py               # CNN architectures (EEGNet, ShallowConvNet, DeepConvNet)
│   ├── streaming/
│   │   ├── __init__.py
│   │   ├── lsl_client.py        # LSL streaming client
│   │   └── lsl_server.py        # LSL streaming server
│   └── utils/
│       ├── __init__.py
│       └── helpers.py           # Utility functions
├── tests/
│   └── test_models.py           # Comprehensive test suite
├── requirements.txt             # Python dependencies
├── test_data_loading.py         # Data loading test script
├── test_vp01_only.py           # Single patient test script
└── README.md                    # Project documentation
```

## Dependencies Installed

### Core Libraries
- **MNE-Python 1.10.0**: EEG signal processing, BrainVision format support
- **PyTorch 2.7.1**: Deep learning framework with CUDA support
- **NumPy, SciPy**: Numerical computing
- **scikit-learn**: Machine learning utilities
- **matplotlib, seaborn**: Data visualization

### Streaming & Real-time
- **pylsl**: Lab Streaming Layer for real-time data streaming
- **threading**: Concurrent processing

### Development & Testing
- **pytest**: Testing framework
- **PyYAML**: Configuration file handling
- **tqdm**: Progress bars

## Core Modules Implemented

### 1. Data Loading (`src/data/loader.py`)

**EEGDataLoader Class**:
- **Purpose**: Load and preprocess BrainVision EEG files
- **Key Features**:
  - BrainVision format support (.vhdr, .eeg, .vmrk)
  - Preprocessing pipeline: 1Hz HP filter, 45Hz LP filter, 50Hz notch
  - ICA artifact removal
  - Resampling to 500Hz
  - Event extraction from marker files

**Key Methods**:
- `load_raw_data()`: Load and preprocess BrainVision files
- `extract_events()`: Extract stimulus events and pain ratings from markers
- `apply_ica_artifact_removal()`: Remove eye blinks and muscle artifacts
- `create_sliding_windows()`: Generate 4s sliding windows with 1s steps
- `create_ternary_labels()`: Convert pain ratings to low/moderate/high labels

**Recent Fixes**:
- ✅ Fixed pain rating extraction from Comment events in marker files
- ✅ Correctly parses ratings from "Comment/XX" format (XX = pain rating 0-100)
- ✅ Maps stimulus intensities S1/S2/S3 to low/medium/high

### 2. CNN Models (`src/models/cnn.py`)

**Implemented Architectures**:

**EEGNet**:
- Compact CNN designed for EEG classification
- Depthwise and separable convolutions
- Dropout for regularization
- Input: (batch, channels, samples)
- Output: (batch, n_classes)

**ShallowConvNet**:
- Shallow architecture with temporal and spatial convolutions
- Log variance activation
- Dropout layers
- Suitable for motor imagery and P300 tasks

**DeepConvNet**:
- Deeper architecture with multiple conv blocks
- Max pooling and dropout
- Higher capacity for complex patterns

**Model Factory**: `create_model()` function for easy model instantiation

### 3. Feature Extraction (`src/features/spectral.py`)

**SpectralFeatureExtractor Class**:
- **Frequency Bands**: Delta (1-4Hz), Theta (4-8Hz), Alpha (8-13Hz), Beta (13-30Hz), Gamma (30-45Hz)
- **Pain-Relevant Channels**: C3, C4, Cz, FCz, CPz (central and vertex regions)
- **Features Extracted**:
  - Power spectral density (PSD) for each frequency band
  - Relative power (band power / total power)
  - Band ratios (e.g., alpha/beta)
  - Spatial features from pain-relevant electrode groups

**Recent Fixes**:
- ✅ Updated to use `mne.time_frequency.psd_array_welch` (new MNE API)
- ✅ Added proper n_fft parameter handling
- ✅ All spectral tests passing

### 4. Streaming Components (`src/streaming/`)

**LSL Server (`lsl_server.py`)**:
- Creates LSL outlet for EEG data streaming
- Configurable channel count and sampling rate
- Proper metadata setup for EEG streams

**LSL Client (`lsl_client.py`)**:
- Connects to LSL inlet for real-time data reception
- Circular buffer for continuous data collection
- Thread-safe data access

### 5. Utility Functions (`src/utils/helpers.py`)

**Data Handling**:
- `split_data()`: Train/validation/test splitting with stratification
- `create_data_loader()`: PyTorch DataLoader creation
- `create_ternary_labels()`: Convert continuous pain ratings to discrete classes

**Configuration**:
- `load_config()`: YAML configuration file loading
- `setup_logging()`: Logging configuration

## Configuration System

**File**: `config/cnn_config.yaml`

```yaml
preprocessing:
  l_freq: 1.0           # High-pass filter (Hz)
  h_freq: 45.0          # Low-pass filter (Hz)
  notch_freq: 50.0      # Notch filter (Hz)
  new_sfreq: 500.0      # Resampling frequency (Hz)
  window_length: 4.0    # Sliding window length (s)
  step_size: 1.0        # Sliding window step (s)

model:
  architecture: "eegnet"  # Model type
  n_channels: 64         # Number of EEG channels
  n_samples: 2000        # Samples per window (4s × 500Hz)
  n_classes: 3           # Low/moderate/high pain
  dropout: 0.25          # Dropout rate

training:
  batch_size: 32
  learning_rate: 0.001
  epochs: 100
  patience: 15           # Early stopping patience
```

## Scripts Implementation

### 1. Preprocessing Pipeline (`scripts/preprocess_data.py`)
- Batch processing of all participants
- Sliding window creation
- Feature extraction options
- Data validation and quality checks
- Outputs to `data/processed/`

### 2. Model Training (`scripts/train_model.py`)
- Cross-validation across participants
- Model selection and hyperparameter tuning
- Training with early stopping
- Model checkpointing
- Evaluation metrics (accuracy, F1, confusion matrix)

### 3. Real-time Prediction (`scripts/real_time_predict.py`)
- LSL streaming integration
- Real-time preprocessing
- Sliding window prediction
- Simulation mode for testing
- Live pain level classification

## Testing Framework

**File**: `tests/test_models.py`
**Status**: ✅ All 11 tests passing

**Test Categories**:

1. **Model Tests**:
   - EEGNet creation and forward pass
   - ShallowConvNet creation and forward pass
   - Model factory function
   - Invalid model type handling

2. **Feature Extraction Tests**:
   - SpectralFeatureExtractor creation
   - PSD computation accuracy
   - Single epoch feature extraction
   - Batch feature extraction

3. **Utility Tests**:
   - Data splitting functionality
   - PyTorch DataLoader creation
   - Stratified sampling

4. **Integration Tests**:
   - End-to-end pipeline (data → model → prediction)
   - Training loop functionality
   - Inference pipeline

## Dataset Integration

**Dataset**: OSF "Brain Mediators for Pain" (Tiemann et al.)
**Publication**: Nature Communications (2018) - DOI: s41467-018-06875-x
**Format**: BrainVision (.vhdr, .eeg, .vmrk files)
**Participants**: 51 subjects (vp01-vp51)
**Location**: `manual_upload/manual_upload/`

**Research Objective**: Investigate brain mediators of different dimensions of pain (perceptual, motor, autonomic)

### **Experimental Design Overview**

**Four Experimental Conditions** (randomized order per participant):

1. **Perception Condition** (Paradigm1_Perception) - *Our current focus*
   - **Task**: Verbal pain rating after auditory cue (3s post-stimulus)
   - **Measure**: Perceptual dimension of pain
   - **Scale**: 0-100 numerical rating scale (0=no pain, 100=worst tolerable pain)
   - **Files**: `Exp_Mediation_Paradigm1_Perception_vpxx.*`

2. **Motor Condition** (Paradigm2_Motor)
   - **Task**: Button release with right index finger as fast as possible
   - **Measure**: Motor dimension of pain (reaction times)
   - **Files**: `Exp_Mediation_Paradigm2_Motor_vpxx.*`

3. **Autonomic Condition** (Paradigm3_EDA)
   - **Task**: Focus on painful stimulation (no active response)
   - **Measure**: Autonomic dimension of pain (skin conductance responses)
   - **Files**: `Exp_Mediation_Paradigm3_EDA_vpxx.*`

4. **Combined Condition** (Paradigm4_Control)
   - **Task**: Button release + pain rating + SCR recording
   - **Measure**: All three dimensions simultaneously
   - **Files**: `Exp_Mediation_Paradigm4_Control_vpxx.*`

### **Stimulus Protocol**

**Per Condition**: 60 painful laser stimuli to dorsum of left hand
**Intensity Levels**: 3 individually adjusted levels per participant
- **Low intensity**: 20 stimuli (S1)
- **Medium intensity**: 20 stimuli (S2) 
- **High intensity**: 20 stimuli (S3)
- **Sequence**: Pseudo-randomized
- **Inter-stimulus interval**: 8-12 seconds

**Administrative Markers**:
- **S5**: Session start marker (single occurrence)
- **S6**: Session end marker (single occurrence)

### **Data Structure Per Participant**

**Files per participant**: 4 conditions × 3 files = 12 files total
- **Current analysis**: Perception condition only (Paradigm1)
- **Trials**: 60 per condition (20 each of 3 intensity levels)
- **Pain Ratings**: 0-100 scale with variable precision by participant
- **Event Sequence**: Stimulus → Laser → Comment (pain rating)

**Example Event Sequence**:
```
Stimulus, S 2    # Medium intensity stimulus
Laser, L 1       # Stimulus onset (timing reference)
Comment, 45      # Participant pain rating (0-100)
```

### **Pain Rating Analysis Results**

**Multi-participant validation** (vp01-vp05, N=300 ratings):

**Rating Distribution**:
- **17 unique values**: 0, 1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 70, 80, 90
- **Range**: 0-90 (no participant reported maximum pain of 100)
- **Variable precision**: Individual differences in rating granularity
  - Some participants: 10-point intervals (10, 20, 30...)
  - Others: 5-point intervals (5, 15, 25, 35, 45, 55)
  - Few: 1-point precision (individual digit ratings)

**Clinical Interpretation**:
- **Method**: Visual Analog Scale (VAS) or Numerical Rating Scale (NRS)
- **Individual variation**: Reflects natural differences in pain expression
- **Realistic range**: 0-90 suggests thoughtful scale usage

**Most Common Ratings**:
- **40**: 13.7% (moderate-high pain)
- **30**: 12.0% (moderate pain)
- **50**: 11.7% (moderate-high pain)
- **10, 20**: 10.0% each (low to low-moderate pain)

### **Technical Implementation**

**Current Focus**: Perception condition (Paradigm1) analysis
**Future Extensions**: Can be expanded to motor, autonomic, and combined conditions

**Session Markers Validated**:
- ✅ **S5**: Appears at experiment start (~20s) - administrative marker
- ✅ **S6**: Appears at experiment end (~22min) - administrative marker
- ✅ **Pattern**: Consistent across all participants
- ✅ **Usage**: Exclude from pain classification (non-stimulus events)

## Data Validation Results

**Test Subjects**: vp01-vp05 (Multi-participant validation)
**Status**: ✅ Successfully validated

### **Individual Participant Analysis**:

**vp01**: 
- ✅ 60 pain ratings extracted
- ✅ Rating pattern: Multiples of 10 only (10, 20, 30, 40, 50, 60, 70, 80, 90)
- ✅ Distribution: Well-spread across pain scale

**vp02**:
- ✅ 60 pain ratings extracted  
- ✅ Rating pattern: Mix of 5-step and 10-step intervals (5, 15, plus standard 10s)
- ✅ Individual variation in rating precision

**vp03**:
- ✅ 60 pain ratings extracted
- ✅ Rating pattern: High granularity (0, 1, 5, 15, 25, 30)
- ✅ Fine-grained pain expression

**vp04**:
- ✅ 60 pain ratings extracted
- ✅ Rating pattern: Multiples of 10 (similar to vp01)
- ✅ Consistent with round-number preference

**vp05**:
- ✅ 60 pain ratings extracted
- ✅ Rating pattern: 5-step intervals (5, 15, 25, 35, 45, 55)
- ✅ Systematic intermediate values

### **Technical Validation**:

**Data Loading**:
- ✅ 68 EEG channels loaded per participant
- ✅ ~658 seconds recording duration (~11 minutes)
- ✅ 60 laser onset events (timing references)
- ✅ 62 stimulus events (60 pain trials + 2 session markers S5/S6)

**Event Structure**:
- ✅ Consistent Stimulus → Laser → Comment sequence
- ✅ S5 (start) and S6 (end) session markers validated
- ✅ Pain rating extraction from Comment events successful

**Experimental Protocol Validation**:
- ✅ 20 trials each of S1/S2/S3 intensities confirmed
- ✅ Pseudo-randomized stimulus presentation verified
- ✅ Individual intensity adjustment reflected in pain ratings

### **Dataset Quality Assessment**:

**Pain Rating Distribution (N=300 across 5 participants)**:
- **Range**: 0-90 (realistic, no extreme values)
- **Granularity**: 17 unique values (rich label space)
- **Individual Differences**: Variable precision reflects natural pain expression
- **Clinical Validity**: Consistent with established VAS/NRS methodology

**Missing Data**: None detected - all 60 ratings present per participant
**Data Integrity**: ✅ All BrainVision files load successfully
**Event Timing**: ✅ Consistent event sequences across participants

## Key Technical Achievements

### 1. Bug Fixes Completed
- ✅ **MNE API Compatibility**: Updated to `mne.time_frequency.psd_array_welch`
- ✅ **Pain Rating Extraction**: Fixed marker file parsing for Comment events
- ✅ **Spectral Feature Extraction**: Added proper n_fft parameter handling
- ✅ **Test Suite**: All 11 tests passing after fixes

### 2. Architecture Decisions
- **Sliding Windows**: 4s windows with 1s step for temporal analysis
- **Frequency Bands**: Standard neuroscience bands (delta through gamma)
- **Pain-Relevant Channels**: Focus on central/vertex regions (C3,C4,Cz,FCz,CPz)
- **Ternary Classification**: Low/moderate/high instead of regression

### 3. Real-time Capabilities
- **LSL Integration**: Both client and server components
- **Streaming Pipeline**: Circular buffers and thread-safe processing
- **Simulation Mode**: Testing without hardware

## Current Status

### ✅ Completed Components
1. **Project Structure**: Full directory hierarchy and organization
2. **Data Loading**: BrainVision format support with event extraction
3. **Preprocessing**: Complete EEG processing pipeline
4. **Models**: Three CNN architectures implemented and tested
5. **Feature Extraction**: Spectral features with pain-relevant channels
6. **Streaming**: LSL integration for real-time processing
7. **Testing**: Comprehensive test suite (11/11 tests passing)
8. **Configuration**: YAML-based parameter management
9. **Documentation**: README and inline documentation

### ✅ Validated Functionality
1. **Data Loading**: Successfully loads vp01 with correct event extraction
2. **Pain Rating Extraction**: 60/60 ratings correctly parsed
3. **Model Creation**: All CNN architectures create and run successfully
4. **Feature Extraction**: Spectral features computed correctly
5. **End-to-End Pipeline**: Complete flow from raw data to predictions

### 🔄 Ready for Next Phase
The project is now ready for:
1. **Multi-participant validation** ✅ **COMPLETED** (5 participants validated)
2. **Full preprocessing pipeline** execution (sliding windows, feature extraction)
3. **Model training** on real pain perception data
4. **Cross-condition analysis** (expand to motor, autonomic, combined paradigms)
5. **Real-time testing** with LSL streaming
6. **Performance evaluation** and optimization

### 📊 **New Research Opportunities Identified**
Based on the full dataset structure, we can expand beyond perception to:
1. **Motor pain response** prediction (reaction time estimation)
2. **Autonomic pain response** prediction (SCR estimation) 
3. **Multi-dimensional pain** modeling (combined paradigm)
4. **Cross-paradigm transfer learning** (train on one, test on others)

## Research Context Extension

### **Advanced Literature Review Findings** 

**New Research Paper**: MDPI Biology 2025 - "Objective Pain Assessment Using Deep Learning Through EEG-Based Brain–Computer Interfaces" (Al-Nafjan et al.)

**Key Findings from Recent Literature**:

1. **Performance Benchmarks**:
   - **Pain/No-Pain Detection**: State-of-the-art accuracy 91.84% (RNN), 90.69% (CNN)
   - **Three-Level Pain Classification**: Best accuracy 87.94% (CNN), 86.71% (RNN) 
   - **Our Target**: Match or exceed these benchmarks with the same OSF dataset

2. **Advanced Methodological Insights**:
   - **Wavelet Transform Features**: Daubechies 4 (db4) wavelet for time-frequency analysis
   - **Statistical Feature Extraction**: Zero-crossing rate, percentiles, mean, median, std, variance, RMS
   - **Data Augmentation**: SMOTE oversampling, noise injection, frequency modulation, data multiplication
   - **Preprocessing Pipeline**: 1Hz HP filter → 50Hz notch → 500Hz resample → ICA cleanup

3. **Pain Scale Mapping** (Critical for Label Creation):
   - **Binary Classification**: Pain ratings ≤5 = "no pain", >5 = "pain"
   - **Ternary Classification**: ≤3 = "low pain", 4-6 = "moderate pain", >6 = "high pain"
   - **Alternative Approach**: Our percentile-based method vs. fixed thresholds

4. **Advanced Data Augmentation Techniques**:
   - **Data Multiplication**: Multiply by (1±0.05) factors
   - **Noise Injection**: 2% noise standard deviation uniformly distributed
   - **Frequency Modulation**: Hilbert transform with ±0.2 frequency shift
   - **Class Balancing**: SMOTE increased performance by 3-7%

5. **Architecture Optimizations**:
   - **CNN**: Convolutional → pooling → dropout (0.25) → fully connected → dropout (0.5, 0.3)
   - **RNN**: LSTM layers (64-256 units) → dropout → fully connected → softmax
   - **Training**: Adam optimizer, categorical cross-entropy, 100 epochs
   - **Hyperparameter Tuning**: Grid search for learning rate, dropout, epochs

6. **Research Gaps Identified**:
   - Limited multi-paradigm analysis (most studies focus on single conditions)
   - Need for real-time implementation validation
   - Participant-independent vs. participant-specific models
   - Integration of multiple physiological signals (EEG + GSR + heart rate)

### **Dataset Usage Validation**:
- **Confirmed**: Same OSF "Brain Mediators for Pain" dataset used in MDPI study
- **Participants**: 51 healthy individuals (consistent with our analysis)
- **Protocol**: Laser stimulation with pain rating collection (validated)
- **File Format**: BrainVision (.vhdr, .eeg, .vmrk) - exact match
- **Sampling Rate**: 1000Hz → 500Hz downsampling (standard approach)

## Next Steps Roadmap

### **Phase 1: Enhanced Data Preprocessing Pipeline** (Immediate Priority)

**1A. Advanced Sliding Window Creation**
   - Implement 4-second sliding windows with 1-second steps (literature standard)
   - Extract windows around laser onset events (+/- baseline periods)
   - Create time-locked epochs for consistent temporal analysis
   - Validate window alignment across participants
   - **New**: Compare with 8-12 second epochs used in MDPI study

**1B. Multi-Modal Feature Engineering**
   - **Spectral Features**: Extract delta, theta, alpha, beta, gamma bands
   - **Wavelet Features**: Implement db4 wavelet transform with statistical measures
   - **Spatial Features**: Pain-relevant channels (C3, C4, Cz, FCz, CPz)
   - **Statistical Features**: Zero-crossing rate, percentiles, variance, RMS
   - **New**: Band power ratios and relative power measures

**1C. Advanced Label Processing**
   - **Method 1**: Percentile-based thresholding (33rd/66th percentiles) - our current approach
   - **Method 2**: Fixed thresholds (≤3, 4-6, >6) - literature standard
   - **Method 3**: Binary classification (≤5 vs >5) - for comparison
   - Create both regression targets and classification labels
   - Balance dataset using SMOTE and other augmentation techniques

### **Phase 2: Advanced Model Development & Training**

**2A. Baseline Model Enhancement**
   - **EEGNet**: Enhanced with optimal hyperparameters from literature
   - **ShallowConvNet & DeepConvNet**: Benchmarking against literature results
   - **New CNN Architecture**: Replicate MDPI study's CNN design
   - **New RNN Architecture**: Implement LSTM with 64-256 units

**2B. Advanced Data Augmentation Pipeline**
   - **Data Multiplication**: Apply (1±0.05) factor transformations
   - **Noise Injection**: 2% standard deviation uniformly distributed noise
   - **Frequency Modulation**: Hilbert transform with ±0.2 frequency shift
   - **SMOTE Balancing**: Class balance with synthetic minority oversampling
   - **Validation**: Cross-validation to prevent overfitting from synthetic data

**2C. Comprehensive Training Strategy**
   - **Participant-Independent**: Leave-one-subject-out cross-validation
   - **Grid Search Optimization**: Learning rate, dropout, epochs, architecture parameters
   - **Performance Targets**: 
     - Binary Pain Detection: >91.84% accuracy (match MDPI RNN)
     - Ternary Classification: >87.94% accuracy (match MDPI CNN)
   - **Multiple Label Strategies**: Compare percentile vs. fixed threshold approaches

### **Phase 3: Real-time Implementation & Validation**

**3A. Advanced Streaming Pipeline**
   - Integrate trained models with LSL streaming
   - Implement real-time preprocessing (1Hz HP, 50Hz notch, ICA)
   - Create sliding window buffer for continuous prediction
   - **Performance Optimization**: Model quantization for faster inference

**3B. Comprehensive Validation**
   - **Simulation Mode**: Test with recorded data playback
   - **Live EEG Testing**: Validate with actual EEG hardware
   - **Latency Analysis**: Real-time performance constraints
   - **Robustness Testing**: Different EEG setups and electrode configurations

### **Phase 4: Multi-Paradigm Research Extensions**

**4A. Cross-Paradigm Analysis**
   - **Motor Response Prediction**: Paradigm2 (reaction time estimation)
   - **Autonomic Response Prediction**: Paradigm3 (skin conductance estimation)
   - **Combined Paradigm Analysis**: Paradigm4 (multi-dimensional pain modeling)
   - **Transfer Learning**: Train on one paradigm, test on others

**4B. Advanced Clinical Applications**
   - **Real-time Pain Monitoring**: Dashboard for clinical settings
   - **Personalized Pain Models**: Individual participant adaptation
   - **Multi-Modal Integration**: EEG + heart rate + skin conductance
   - **Clinical Population Validation**: Extension beyond healthy participants

### **Phase 5: Research Innovation & Publication**

**5A. Novel Contributions**
   - **Multi-Paradigm Comparison**: First comprehensive analysis across all 4 conditions
   - **Real-time Validation**: Live streaming performance evaluation
   - **Advanced Feature Fusion**: Spectral + wavelet + spatial features
   - **Personalization Methods**: Individual pain threshold adaptation

**5B. Publication Strategy**
   - **Target Journals**: IEEE TBME, Nature Communications, Journal of Neural Engineering
   - **Key Comparisons**: Direct benchmarking against MDPI 2025 study results
   - **Novel Findings**: Multi-paradigm insights and real-time performance
   - **Clinical Impact**: Objective pain assessment applications

## Advanced Technical Milestones

### **Week 1-2: Literature-Informed Enhancement**
- [ ] Implement db4 wavelet feature extraction
- [ ] Create advanced data augmentation pipeline (SMOTE + noise + frequency modulation)
- [ ] Replicate MDPI study's CNN/RNN architectures
- [ ] Validate preprocessing pipeline with literature methods

### **Week 3-4: Performance Benchmarking**
- [ ] Train baseline models on 10 participants with multiple label strategies
- [ ] Achieve >85% accuracy on ternary classification (baseline target)
- [ ] Compare percentile vs. fixed threshold labeling approaches
- [ ] Implement participant-independent cross-validation

### **Week 5-6: Real-time System Integration**
- [ ] Deploy best-performing model in LSL streaming environment
- [ ] Validate real-time preprocessing and prediction pipeline
- [ ] Optimize inference speed for clinical applications
- [ ] Test with simulated and live EEG data streams

### **Week 7-8: Multi-Paradigm Expansion**
- [ ] Extend analysis to motor response paradigm (Paradigm2)
- [ ] Explore autonomic response prediction (Paradigm3)
- [ ] Implement cross-paradigm transfer learning
- [ ] Document novel findings for publication

## Development Environment
- **OS**: Windows 11
- **Python**: 3.13
- **Shell**: PowerShell 5.1
- **IDE**: VS Code with Python extension
- **Virtual Environment**: Activated and configured
- **Version Control**: Git repository initialized
  - **User**: Dhruv Kurup (dhruvkurup@outlook.com)
  - **Initial Commit**: 36286b6 (128 files tracked)
  - **Files Tracked**: All source code, configs, tests, documentation
  - **Files Excluded**: Large data files (.eeg), Python cache, logs
  - **Repository Status**: Clean working tree, ready for development

---

**Last Updated**: July 16, 2025
**Status**: Multi-participant Validation Complete - Ready for Model Training
**Research Context**: OSF "Brain Mediators of Pain" - Nature Communications (2018)
**Dataset Scope**: 4 experimental paradigms × 51 participants (currently analyzing Perception paradigm)
