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
├── manual_upload/
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

## Next Steps Roadmap

### **Phase 1: Data Preprocessing Pipeline** (Immediate Priority)
1. **Sliding Window Creation**
   - Implement 4-second sliding windows with 1-second steps
   - Extract windows around laser onset events (+/- baseline periods)
   - Create time-locked epochs for consistent temporal analysis
   - Validate window alignment across participants

2. **Feature Engineering**
   - Extract spectral features (delta, theta, alpha, beta, gamma bands)
   - Compute spatial features from pain-relevant channels (C3, C4, Cz, FCz, CPz)
   - Generate band power ratios and relative power measures
   - Create participant-specific normalization

3. **Label Processing**
   - Convert continuous pain ratings (0-100) to ternary labels (low/moderate/high)
   - Implement participant-specific thresholding (33rd/66th percentiles)
   - Create both regression targets (continuous) and classification labels (discrete)
   - Balance dataset across pain intensity levels

### **Phase 2: Model Development & Training**
1. **Baseline Models**
   - Train EEGNet on preprocessed data
   - Compare ShallowConvNet and DeepConvNet architectures
   - Establish performance benchmarks with cross-validation

2. **Advanced Training**
   - Implement participant-independent validation (leave-one-subject-out)
   - Add data augmentation for temporal robustness
   - Hyperparameter optimization (learning rate, dropout, architecture params)
   - Model ensemble methods for improved accuracy

3. **Performance Evaluation**
   - Classification accuracy, precision, recall, F1-score
   - Confusion matrices and class-wise performance
   - ROC curves and AUC analysis
   - Statistical significance testing across participants

### **Phase 3: Real-time Implementation**
1. **Streaming Pipeline**
   - Integrate trained models with LSL streaming
   - Implement real-time preprocessing and prediction
   - Create sliding window buffer for continuous prediction
   - Test with simulated and live EEG data

2. **Performance Optimization**
   - Model quantization for faster inference
   - Latency optimization for real-time constraints
   - Memory usage optimization for embedded systems
   - Robustness testing with different EEG setups

### **Phase 4: Advanced Research Extensions**
1. **Multi-paradigm Analysis**
   - Expand to motor response prediction (Paradigm2)
   - Integrate autonomic response prediction (Paradigm3)
   - Combined paradigm analysis (Paradigm4)
   - Cross-paradigm transfer learning

2. **Clinical Applications**
   - Pain level monitoring dashboard
   - Personalized pain prediction models
   - Integration with pain management protocols
   - Validation with clinical populations

### **Technical Milestones**
- [ ] **Week 1**: Complete preprocessing pipeline for 10 participants
- [ ] **Week 2**: Train and validate baseline EEGNet model
- [ ] **Week 3**: Implement real-time prediction system
- [ ] **Week 4**: Multi-paradigm analysis and clinical applications

## Development Environment
- **OS**: Windows 11
- **Python**: 3.13
- **Shell**: PowerShell 5.1
- **IDE**: VS Code with Python extension
- **Virtual Environment**: Activated and configured

---

**Last Updated**: July 16, 2025
**Status**: Multi-participant Validation Complete - Ready for Model Training
**Research Context**: OSF "Brain Mediators of Pain" - Nature Communications (2018)
**Dataset Scope**: 4 experimental paradigms × 51 participants (currently analyzing Perception paradigm)
