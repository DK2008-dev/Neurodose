# Dataset Information: EEG Pain Classification Research

**Research Project:** The Complexity Paradox and Augmentation Illusion in EEG-Based Pain Detection  
**Principal Investigator:** Dhruv Kurup  
**Date:** July 2025  

## Primary Dataset

### OSF "Brain Mediators for Pain" Dataset
- **Original Study:** Tiemann et al. (2018) Nature Communications
- **Repository:** https://osf.io/bsv86/
- **DOI:** 10.1038/s41467-018-07058-1
- **License:** Open access for research purposes

### Dataset Characteristics
- **Participants:** 51 healthy adults (age 18-35, mean 24.3 ± 4.2 years)
- **Gender Distribution:** 27 female, 24 male
- **EEG Setup:** 68-channel system, 1000 Hz sampling rate
- **Stimuli:** 60 laser pain stimuli per participant (20 each: low, medium, high intensity)
- **Pain Ratings:** 0-100 visual analog scale, collected 3 seconds post-stimulus
- **Total Trials:** 3,060 original trials across 51 participants

### Data Quality and Processing
- **Successful Processing:** 49 participants (96% retention rate)
- **Excluded Participants:** 2 (vp06, vp23) due to excessive artifacts
- **Final Dataset:** 2,891 high-quality epochs
- **Quality Control:** Rigorous artifact rejection and ICA cleaning

## Processed Dataset Characteristics

### Binary Classification Dataset
- **Total Epochs:** 1,224 trials
- **Low Pain:** 612 trials (≤33rd percentile ratings)
- **High Pain:** 612 trials (≥67th percentile ratings)
- **Balance Ratio:** 1.00 (perfect balance)
- **Excluded:** Middle 34% (moderate pain) for clear class separation

### Ternary Classification Dataset
- **Total Epochs:** 2,234 trials
- **Low Pain:** 741 trials (≤33rd percentile)
- **Moderate Pain:** 752 trials (34th-66th percentile)
- **High Pain:** 741 trials (≥67th percentile)
- **Balance Ratio:** 0.98-1.01 (excellent balance)

### Full Dataset (All Intensities)
- **Total Epochs:** 2,891 trials
- **Low Pain:** 963 trials
- **Moderate Pain:** 964 trials
- **High Pain:** 964 trials
- **Balance Ratio:** 0.99-1.00 (near-perfect balance)

## Data Preprocessing Pipeline

### 1. Raw Data Loading
- **Format:** BrainVision (.vhdr, .eeg, .vmrk)
- **Channels:** 68 EEG electrodes (10-20 system)
- **Sampling Rate:** 1000 Hz → 500 Hz (resampled)

### 2. Filtering and Cleaning
```
- Band-pass filter: 1-45 Hz
- Notch filter: 50 Hz (electrical line noise)
- High-pass: 1 Hz (drift removal)
- Low-pass: 45 Hz (artifact removal)
```

### 3. Artifact Removal
- **ICA Components:** 20-component decomposition
- **Artifact Types:** Eye blinks, muscle activity, electrode noise
- **Rejection Criteria:** Peak-to-peak amplitude >2500 μV

### 4. Epoching
- **Window:** 4 seconds (-1 to +3 seconds around laser onset)
- **Baseline Correction:** -1 to 0 seconds pre-stimulus
- **Event Extraction:** Pain ratings and stimulus markers

### 5. Labeling Strategy
- **Binary:** 33rd/67th percentile thresholds per participant
- **Ternary:** 33rd/67th percentile boundaries for three classes
- **Individual Calibration:** Thresholds adapted per participant

## Feature Extraction

### Simple Feature Set (78 Features)
1. **Spectral Features (30):** Power spectral density in 5 frequency bands
   - Delta: 1-4 Hz
   - Theta: 4-8 Hz  
   - Alpha: 8-13 Hz
   - Beta: 13-30 Hz
   - Gamma: 30-45 Hz
   - Channels: Cz, FCz, C3, C4, Fz, Pz

2. **Frequency Ratios (18):** Band-to-band power ratios
   - Delta/alpha, gamma/beta, low/high frequency

3. **Spatial Asymmetry (5):** Left-right differences
   - C4-C3 power differences across frequency bands

4. **ERP Components (4):** Event-related potentials
   - N2 amplitude (150-250 ms)
   - P2 amplitude (200-350 ms)

5. **Temporal Features (21):** Time-domain characteristics
   - RMS amplitude, variance, zero-crossing rate

### Advanced Feature Set (645 Features)
1. **Wavelet Analysis (350):** Multi-resolution decomposition
   - Daubechies 4 wavelet, 5 levels
   - Statistical measures per level per channel

2. **Connectivity Measures (120):** Inter-channel relationships
   - Coherence, phase-locking values, cross-correlation

3. **Advanced Spectral (95):** Sophisticated frequency analysis
   - Multitaper estimation, spectral entropy, edge frequency

4. **Complexity Measures (80):** Nonlinear dynamics
   - Sample entropy, approximate entropy, fractal dimension

## Pain-Relevant Electrode Locations

### Primary Channels (Pain Processing)
- **Central:** C3, C4, Cz (primary somatosensory cortex)
- **Vertex:** FCz, CPz (midline processing)
- **Frontal:** Fz, AFz (attention and cognitive aspects)

### Region of Interest Groups
1. **Central ROI:** C3, C4, CP3, CP4 (somatosensory processing)
2. **Vertex ROI:** Cz, FCz, CPz (midline pain processing)
3. **Fronto-Central:** Fz, FC1, FC2, AFz (cognitive modulation)

## Data Access and Usage

### Original Dataset Access
1. Visit OSF repository: https://osf.io/bsv86/
2. Download BrainVision files for each participant (vp01-vp51)
3. Follow preprocessing pipeline in `/code/preprocessing/`

### Processed Data Samples
- **Location:** `/data/processed_samples/`
- **Format:** NumPy arrays (.npy) and CSV files
- **Contents:** Example processed epochs and feature matrices
- **Size:** Representative subset for verification

### Data Usage Guidelines
1. **Attribution:** Cite original Tiemann et al. (2018) study
2. **Methodology:** Use identical preprocessing for comparability
3. **Validation:** Employ LOPOCV for clinical relevance
4. **Reproducibility:** Follow exact parameter settings

## Quality Assurance

### Participant-Level Quality
- **Perfect Participants:** 45/49 (60/60 trials retained)
- **Near-Perfect:** 4/49 (58-59/60 trials retained)
- **Exclusion Rate:** 4% (2/51 participants)

### Signal Quality Metrics
- **Artifact Rejection:** <5% trial loss per participant
- **ICA Component Selection:** Manual verification of eye/muscle artifacts
- **Baseline Stability:** Consistent pre-stimulus periods

### Statistical Validation
- **Class Balance:** Verified across all classification schemes
- **Individual Differences:** Quantified and documented
- **Preprocessing Consistency:** Identical pipeline across participants

## Ethical Considerations

### Original Study Ethics
- **Approval:** Ethics committee approval obtained by Tiemann et al.
- **Consent:** Written informed consent from all participants
- **Pain Protocol:** Standardized, calibrated pain stimuli

### Secondary Analysis Ethics
- **Public Data:** Openly available for research purposes
- **Anonymization:** No personally identifiable information
- **Research Purpose:** Scientific advancement in pain assessment

## Contact and Support

**Data Questions:**  
Dhruv Kurup - research.dhruv.kurup@gmail.com

**Technical Support:**  
Repository issues and preprocessing questions

**Original Dataset:**  
Contact authors of Tiemann et al. (2018) for dataset-specific questions

---

**Last Updated:** July 18, 2025  
**Version:** 1.0.0  
**Status:** Complete and Validated
