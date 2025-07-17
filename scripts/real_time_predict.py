#!/usr/bin/env python3
"""
Real-time pain prediction script using LSL streams.

This script loads a trained model and performs real-time pain classification
on incoming EEG data via Lab Streaming Layer (LSL).
"""

import argparse
import logging
import time
from pathlib import Path
import torch
import yaml
import signal
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.models.cnn import create_model
from src.streaming.lsl_processor import EEGStreamProcessor, StreamConfig, LSLDataSimulator
from src.utils.helpers import setup_logging, load_data


class RealTimePainClassifier:
    """Real-time pain classification system."""
    
    def __init__(self, model_path: str, config: dict = None):
        """
        Initialize the real-time classifier.
        
        Parameters:
        -----------
        model_path : str
            Path to trained model file
        config : dict, optional
            Configuration dictionary
        """
        self.logger = logging.getLogger(__name__)
        
        # Load model
        self.logger.info(f"Loading model from {model_path}")
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Create model
        model_type = checkpoint['model_type']
        model_config = checkpoint['model_config']
        self.model = create_model(model_type, **model_config)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Setup stream configuration
        if config:
            stream_config = config.get('streaming', {})
        else:
            stream_config = checkpoint.get('config', {}).get('streaming', {})
        
        self.stream_config = StreamConfig(
            stream_name=stream_config.get('stream_name', 'EEG'),
            stream_type=stream_config.get('stream_type', 'EEG'),
            n_channels=stream_config.get('n_channels', 64),
            sampling_rate=stream_config.get('sampling_rate', 500.0),
            window_length=stream_config.get('window_length', 4.0),
            step_size=stream_config.get('step_size', 1.0),
            buffer_length=stream_config.get('buffer_length', 10.0)
        )
        
        # Initialize stream processor
        self.processor = EEGStreamProcessor(
            model=self.model,
            config=self.stream_config,
            device='auto'
        )
        
        # Control variables
        self.is_running = False
        self.simulator = None
        
        self.logger.info("Real-time pain classifier initialized")
        self.logger.info(f"Model: {model_type}")
        self.logger.info(f"Expected input: {model_config['n_channels']} channels, "
                        f"{model_config['n_samples']} samples")
    
    def start_simulation(self):
        """Start EEG data simulation for testing."""
        self.logger.info("Starting EEG data simulation...")
        
        self.simulator = LSLDataSimulator(
            n_channels=self.stream_config.n_channels,
            sampling_rate=self.stream_config.sampling_rate,
            stream_name=self.stream_config.stream_name
        )
        
        self.simulator.start()
        self.logger.info("EEG simulation started")
    
    def stop_simulation(self):
        """Stop EEG data simulation."""
        if self.simulator:
            self.simulator.stop()
            self.simulator = None
            self.logger.info("EEG simulation stopped")
    
    def start_processing(self, simulate: bool = False):
        """
        Start real-time processing.
        
        Parameters:
        -----------
        simulate : bool
            Whether to start data simulation
        """
        if self.is_running:
            self.logger.warning("Processing already running")
            return
        
        try:
            # Start simulation if requested
            if simulate:
                self.start_simulation()
                time.sleep(2)  # Give simulation time to start
            
            # Start stream processing
            if not self.processor.start_streaming():
                self.logger.error("Failed to start stream processing")
                return
            
            self.is_running = True
            self.logger.info("Real-time processing started")
            
            # Main processing loop
            self.run_processing_loop()
            
        except Exception as e:
            self.logger.error(f"Error in processing: {e}")
        finally:
            self.stop_processing()
    
    def run_processing_loop(self):
        """Main processing loop with statistics reporting."""
        self.logger.info("Processing loop started. Press Ctrl+C to stop.")
        
        last_stats_time = time.time()
        stats_interval = 30  # Report stats every 30 seconds
        
        try:
            while self.is_running:
                time.sleep(1)
                
                # Report statistics periodically
                current_time = time.time()
                if current_time - last_stats_time >= stats_interval:
                    stats = self.processor.get_statistics()
                    self.logger.info(f"Stats: {stats['processed_windows']} windows processed, "
                                   f"avg processing time: {stats['avg_processing_time']:.3f}s, "
                                   f"buffer size: {stats['buffer_size']}")
                    last_stats_time = current_time
                    
        except KeyboardInterrupt:
            self.logger.info("Interrupted by user")
        except Exception as e:
            self.logger.error(f"Error in processing loop: {e}")
    
    def stop_processing(self):
        """Stop real-time processing."""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Stop stream processing
        self.processor.stop_streaming()
        
        # Stop simulation
        self.stop_simulation()
        
        self.logger.info("Real-time processing stopped")


def signal_handler(signum, frame):
    """Handle interrupt signals gracefully."""
    logger = logging.getLogger(__name__)
    logger.info("Received interrupt signal, stopping...")
    global classifier
    if classifier:
        classifier.stop_processing()
    sys.exit(0)


def main():
    parser = argparse.ArgumentParser(description='Real-time EEG pain classification')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model file')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to configuration file (optional)')
    parser.add_argument('--simulate', action='store_true',
                       help='Simulate EEG data for testing')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug logging')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = 'DEBUG' if args.debug else 'INFO'
    setup_logging(level=log_level)
    logger = logging.getLogger(__name__)
    
    # Load configuration if provided
    config = None
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Initialize classifier
    global classifier
    try:
        classifier = RealTimePainClassifier(args.model_path, config)
        
        # Start processing
        logger.info("Starting real-time pain classification...")
        if args.simulate:
            logger.info("Using simulated EEG data")
        else:
            logger.info("Waiting for real EEG stream...")
        
        classifier.start_processing(simulate=args.simulate)
        
    except Exception as e:
        logger.error(f"Failed to start classifier: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
