<!-- Use this file to provide workspace-specific custom instructions to Copilot. For more details, visit https://code.visualstudio.com/docs/copilot/copilot-customization#_use-a-githubcopilotinstructionsmd-file -->

# EEG Pain Classification Project Instructions

This is a neuroscience project focused on EEG-based pain classification using deep learning.

## Project Context
- **Domain**: Neuroscience, EEG signal processing, pain research
- **Technology Stack**: MNE-Python, TensorFlow/PyTorch, LSL (Lab Streaming Layer)
- **Data**: BrainVision EEG format from OSF "Brain Mediators for Pain" dataset
- **Goal**: Real-time ternary pain classification (low/moderate/high)

## Key Conventions
- Use MNE-Python conventions for EEG processing
- Follow neuroscience naming conventions (epochs, channels, events)
- Implement sliding window approach (4s windows, 1s steps)
- Use standard frequency bands: delta (1-4Hz), theta (4-8Hz), alpha (8-13Hz), beta (13-30Hz), gamma (30-45Hz)

## Important Electrode Locations
- **Pain-relevant channels**: C4 (contralateral S1), Cz, FCz, CPz
- **ROI groups**: Central (C3,C4,CP3,CP4), Vertex (Cz,FCz,CPz), Fronto-central (Fz,FC1,FC2,AFz)

## Processing Pipeline
1. Load BrainVision files (.vhdr, .eeg, .vmrk)
2. Preprocess: 1Hz HP filter, 50Hz notch, 500Hz resample, ICA cleanup
3. Extract events from annotations (S30/S50/S70 → pain intensities)
4. Create sliding windows with overlap
5. Extract spectral features or train CNN on raw data
6. Classify into ternary labels (0=low, 1=moderate, 2=high)

## Code Style
- Use type hints for function parameters
- Include docstrings with parameter descriptions
- Handle edge cases in EEG processing (artifacts, bad channels)
- Use logging for processing steps
- Implement robust error handling for real-time streaming

## LSL Integration
- When working with LSL, use proper stream naming conventions
- Implement timestamp synchronization
- Handle real-time data buffering correctly
- Use appropriate LSL data types for EEG streams
