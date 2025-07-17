#!/usr/bin/env python3
"""
Data preprocessing script for EEG pain classification.

This script processes the OSF Brain Mediators for Pain dataset and creates
sliding windows with pain labels for model training.
"""

import argparse
import logging
from pathlib import Path
import numpy as np
import yaml

import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.data.loader import EEGDataLoader
from src.utils.helpers import setup_logging, save_data, set_seed


def main():
    parser = argparse.ArgumentParser(description='Preprocess EEG pain data')
    parser.add_argument('--config', type=str, default='config/cnn_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to raw data directory')
    parser.add_argument('--output_dir', type=str, default='data/processed',
                       help='Path to output directory')
    parser.add_argument('--output_name', type=str, default='pain_epochs.npz',
                       help='Output filename')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug logging')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup logging
    log_level = 'DEBUG' if args.debug else config['logging']['level']
    setup_logging(level=log_level)
    logger = logging.getLogger(__name__)
    
    # Set random seed
    set_seed(config['training']['random_state'])
    
    logger.info("Starting EEG data preprocessing...")
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    
    # Initialize data loader with configuration
    data_config = config['data']
    loader = EEGDataLoader(
        raw_dir=args.data_dir,
        l_freq=data_config['preprocessing']['l_freq'],
        h_freq=data_config['preprocessing']['h_freq'],
        notch_freq=data_config['preprocessing']['notch_freq'],
        new_sfreq=data_config['preprocessing']['new_sfreq'],
        eeg_reject_thresh=data_config['preprocessing']['eeg_reject_thresh']
    )
    
    # Process all subjects
    logger.info("Processing all subjects...")
    X, y = loader.process_all_subjects()
    
    if len(X) == 0:
        logger.error("No data was processed successfully!")
        return
    
    # Save processed data
    output_path = Path(args.output_dir) / args.output_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    data_dict = {
        'X': X,
        'y': y,
        'config': config,
        'info': {
            'n_epochs': len(X),
            'n_channels': X.shape[1],
            'n_samples': X.shape[2],
            'sampling_rate': data_config['preprocessing']['new_sfreq'],
            'window_length': data_config['windowing']['window_length'],
            'step_size': data_config['windowing']['step_size'],
            'label_distribution': np.bincount(y).tolist()
        }
    }
    
    save_data(data_dict, output_path)
    
    # Print summary
    logger.info("\n" + "="*50)
    logger.info("PREPROCESSING SUMMARY")
    logger.info("="*50)
    logger.info(f"Total epochs: {len(X)}")
    logger.info(f"Data shape: {X.shape}")
    logger.info(f"Sampling rate: {data_config['preprocessing']['new_sfreq']} Hz")
    logger.info(f"Window length: {data_config['windowing']['window_length']} s")
    logger.info(f"Label distribution: {dict(zip(['low', 'moderate', 'high'], np.bincount(y)))}")
    logger.info(f"Data saved to: {output_path}")
    logger.info("="*50)


if __name__ == "__main__":
    main()
