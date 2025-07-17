"""
Lab Streaming Layer (LSL) integration for real-time EEG processing.

This module handles real-time EEG data acquisition and pain classification
using LSL streams.
"""

import numpy as np
import torch
import time
import threading
from collections import deque
from typing import Optional, Callable, Dict, Any
import logging
from dataclasses import dataclass

try:
    import pylsl
    LSL_AVAILABLE = True
except ImportError:
    LSL_AVAILABLE = False
    logging.warning("pylsl not available. Real-time streaming will not work.")

logger = logging.getLogger(__name__)


@dataclass
class StreamConfig:
    """Configuration for LSL stream parameters."""
    stream_name: str = "EEG"
    stream_type: str = "EEG"
    n_channels: int = 64
    sampling_rate: float = 500.0
    window_length: float = 4.0  # seconds
    step_size: float = 1.0      # seconds
    buffer_length: float = 10.0  # seconds


class EEGStreamProcessor:
    """Real-time EEG stream processor with pain classification."""
    
    def __init__(self, 
                 model: torch.nn.Module,
                 config: StreamConfig,
                 preprocessing_func: Optional[Callable] = None,
                 device: str = 'auto'):
        """
        Initialize the stream processor.
        
        Parameters:
        -----------
        model : torch.nn.Module
            Trained pain classification model
        config : StreamConfig
            Stream configuration parameters
        preprocessing_func : callable, optional
            Function to preprocess EEG data chunks
        device : str
            Device for model inference
        """
        if not LSL_AVAILABLE:
            raise ImportError("pylsl is required for real-time streaming")
        
        self.config = config
        self.preprocessing_func = preprocessing_func
        
        # Model setup
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model = model.to(self.device)
        self.model.eval()
        
        # Stream variables
        self.inlet = None
        self.outlet = None
        self.is_streaming = False
        self.processing_thread = None
        
        # Data buffer
        buffer_samples = int(config.buffer_length * config.sampling_rate)
        self.data_buffer = deque(maxlen=buffer_samples)
        self.timestamp_buffer = deque(maxlen=buffer_samples)
        
        # Window parameters
        self.window_samples = int(config.window_length * config.sampling_rate)
        self.step_samples = int(config.step_size * config.sampling_rate)
        
        # Pain labels
        self.pain_labels = {0: 'low', 1: 'moderate', 2: 'high'}
        
        # Statistics
        self.processed_windows = 0
        self.processing_times = deque(maxlen=100)
        
        logger.info(f"Stream processor initialized on device: {self.device}")
        logger.info(f"Window: {config.window_length}s, Step: {config.step_size}s")
    
    def find_eeg_stream(self, timeout: float = 5.0) -> bool:
        """
        Find and connect to an EEG stream.
        
        Parameters:
        -----------
        timeout : float
            Timeout in seconds to wait for stream
            
        Returns:
        --------
        success : bool
            True if stream found and connected
        """
        logger.info("Searching for EEG streams...")
        
        # Look for streams
        streams = pylsl.resolve_stream('type', self.config.stream_type, timeout=timeout)
        
        if not streams:
            logger.error(f"No {self.config.stream_type} streams found within {timeout}s")
            return False
        
        # Connect to the first available stream
        stream_info = streams[0]
        logger.info(f"Found stream: {stream_info.name()} ({stream_info.type()})")
        logger.info(f"Channels: {stream_info.channel_count()}, "
                   f"Sampling rate: {stream_info.nominal_srate()} Hz")
        
        # Create inlet
        self.inlet = pylsl.StreamInlet(stream_info, 
                                      max_buflen=int(self.config.buffer_length),
                                      max_chunklen=0,
                                      recover=True)
        
        return True
    
    def create_output_stream(self):
        """Create LSL output stream for pain predictions."""
        # Create stream info for pain predictions
        info = pylsl.StreamInfo(
            name='PainClassification',
            type='Markers',
            channel_count=1,
            nominal_srate=pylsl.IRREGULAR_RATE,
            channel_format=pylsl.cf_string,
            source_id='pain_classifier'
        )
        
        # Add metadata
        desc = info.desc()
        desc.append_child_value("manufacturer", "EEG Pain Classifier")
        
        channels = desc.append_child("channels")
        ch = channels.append_child("channel")
        ch.append_child_value("label", "pain_level")
        ch.append_child_value("unit", "categorical")
        ch.append_child_value("type", "classification")
        
        # Create outlet
        self.outlet = pylsl.StreamOutlet(info)
        logger.info("Created pain classification output stream")
    
    def preprocess_chunk(self, data_chunk: np.ndarray) -> np.ndarray:
        """
        Preprocess a data chunk for model input.
        
        Parameters:
        -----------
        data_chunk : np.ndarray
            Raw EEG data (n_channels, n_samples)
            
        Returns:
        --------
        preprocessed : np.ndarray
            Preprocessed data ready for model
        """
        if self.preprocessing_func:
            return self.preprocessing_func(data_chunk)
        
        # Simple preprocessing: just normalize
        normalized = (data_chunk - np.mean(data_chunk, axis=1, keepdims=True)) / \
                    (np.std(data_chunk, axis=1, keepdims=True) + 1e-8)
        
        return normalized
    
    def classify_window(self, data_window: np.ndarray) -> Dict[str, Any]:
        """
        Classify a single data window.
        
        Parameters:
        -----------
        data_window : np.ndarray
            EEG data window (n_channels, n_samples)
            
        Returns:
        --------
        result : dict
            Classification result with prediction and confidence
        """
        start_time = time.time()
        
        try:
            # Preprocess
            preprocessed = self.preprocess_chunk(data_window)
            
            # Convert to tensor
            x = torch.FloatTensor(preprocessed).unsqueeze(0).to(self.device)
            
            # Model prediction
            with torch.no_grad():
                logits = self.model(x)
                probabilities = torch.softmax(logits, dim=1)
                predicted_class = torch.argmax(logits, dim=1).item()
                confidence = torch.max(probabilities, dim=1)[0].item()
            
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            
            result = {
                'prediction': predicted_class,
                'pain_level': self.pain_labels[predicted_class],
                'confidence': confidence,
                'probabilities': probabilities.cpu().numpy()[0],
                'processing_time': processing_time,
                'timestamp': time.time()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error in classification: {e}")
            return {
                'prediction': -1,
                'pain_level': 'error',
                'confidence': 0.0,
                'error': str(e),
                'timestamp': time.time()
            }
    
    def process_stream(self):
        """Main processing loop for real-time classification."""
        logger.info("Starting real-time processing...")
        
        last_window_start = 0
        
        while self.is_streaming:
            try:
                # Pull samples from the inlet
                chunk, timestamps = self.inlet.pull_chunk(timeout=0.1, max_samples=100)
                
                if chunk:
                    # Add to buffer
                    for sample, timestamp in zip(chunk, timestamps):
                        self.data_buffer.append(sample)
                        self.timestamp_buffer.append(timestamp)
                
                # Check if we have enough data for a window
                current_buffer_size = len(self.data_buffer)
                
                if current_buffer_size >= self.window_samples:
                    # Calculate next window position
                    current_time = len(self.data_buffer)
                    
                    if current_time >= last_window_start + self.step_samples:
                        # Extract window
                        window_start = current_time - self.window_samples
                        window_data = np.array(list(self.data_buffer))[window_start:]
                        window_data = window_data.T  # Shape: (n_channels, n_samples)
                        
                        # Classify window
                        result = self.classify_window(window_data)
                        
                        # Send prediction via LSL
                        if self.outlet and result['prediction'] >= 0:
                            self.outlet.push_sample([result['pain_level']], 
                                                   timestamp=result['timestamp'])
                        
                        # Log result
                        self.processed_windows += 1
                        if self.processed_windows % 10 == 0:
                            avg_processing_time = np.mean(self.processing_times)
                            logger.info(f"Window {self.processed_windows}: "
                                      f"{result['pain_level']} "
                                      f"(conf: {result['confidence']:.3f}, "
                                      f"time: {avg_processing_time:.3f}s)")
                        
                        last_window_start = current_time - self.window_samples + self.step_samples
                
                # Small sleep to prevent excessive CPU usage
                time.sleep(0.01)
                
            except Exception as e:
                logger.error(f"Error in processing loop: {e}")
                time.sleep(0.1)
    
    def start_streaming(self, create_output: bool = True) -> bool:
        """
        Start real-time streaming and processing.
        
        Parameters:
        -----------
        create_output : bool
            Whether to create output stream for predictions
            
        Returns:
        --------
        success : bool
            True if streaming started successfully
        """
        if self.is_streaming:
            logger.warning("Streaming already active")
            return False
        
        # Find and connect to EEG stream
        if not self.find_eeg_stream():
            return False
        
        # Create output stream
        if create_output:
            self.create_output_stream()
        
        # Start processing thread
        self.is_streaming = True
        self.processing_thread = threading.Thread(target=self.process_stream)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        logger.info("Real-time streaming started")
        return True
    
    def stop_streaming(self):
        """Stop real-time streaming."""
        if not self.is_streaming:
            logger.warning("Streaming not active")
            return
        
        self.is_streaming = False
        
        if self.processing_thread:
            self.processing_thread.join(timeout=5.0)
        
        if self.inlet:
            self.inlet.close_stream()
            self.inlet = None
        
        if self.outlet:
            del self.outlet
            self.outlet = None
        
        logger.info("Streaming stopped")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics."""
        stats = {
            'processed_windows': self.processed_windows,
            'buffer_size': len(self.data_buffer),
            'is_streaming': self.is_streaming,
            'avg_processing_time': np.mean(self.processing_times) if self.processing_times else 0,
            'max_processing_time': np.max(self.processing_times) if self.processing_times else 0,
            'device': str(self.device)
        }
        
        return stats


class LSLDataSimulator:
    """Simulate EEG data stream for testing purposes."""
    
    def __init__(self, 
                 n_channels: int = 64,
                 sampling_rate: float = 500.0,
                 stream_name: str = "SimulatedEEG"):
        """
        Initialize the data simulator.
        
        Parameters:
        -----------
        n_channels : int
            Number of EEG channels
        sampling_rate : float
            Sampling rate in Hz
        stream_name : str
            Name of the simulated stream
        """
        if not LSL_AVAILABLE:
            raise ImportError("pylsl is required for data simulation")
        
        self.n_channels = n_channels
        self.sampling_rate = sampling_rate
        self.stream_name = stream_name
        
        # Create stream info
        info = pylsl.StreamInfo(
            name=stream_name,
            type='EEG',
            channel_count=n_channels,
            nominal_srate=sampling_rate,
            channel_format=pylsl.cf_float32,
            source_id='simulator'
        )
        
        # Create outlet
        self.outlet = pylsl.StreamOutlet(info)
        self.is_running = False
        self.thread = None
        
        logger.info(f"Created simulated EEG stream: {stream_name}")
    
    def generate_sample(self) -> np.ndarray:
        """Generate a single EEG sample with realistic noise and artifacts."""
        # Base noise
        sample = np.random.normal(0, 10, self.n_channels)
        
        # Add some alpha rhythm (8-13 Hz) - strongest in occipital channels
        t = time.time()
        alpha_freq = 10  # 10 Hz alpha
        alpha_amplitude = 20
        alpha_channels = list(range(max(0, self.n_channels - 8), self.n_channels))  # Posterior channels
        
        for ch in alpha_channels:
            sample[ch] += alpha_amplitude * np.sin(2 * np.pi * alpha_freq * t)
        
        # Add some 50 Hz power line noise
        powerline_amplitude = 2
        sample += powerline_amplitude * np.sin(2 * np.pi * 50 * t)
        
        # Occasionally add larger artifacts
        if np.random.random() < 0.01:  # 1% chance
            artifact_channels = np.random.choice(self.n_channels, 
                                               size=np.random.randint(1, 5), 
                                               replace=False)
            sample[artifact_channels] += np.random.normal(0, 100, len(artifact_channels))
        
        return sample.astype(np.float32)
    
    def streaming_loop(self):
        """Main streaming loop."""
        logger.info("Starting data simulation...")
        
        sample_interval = 1.0 / self.sampling_rate
        next_sample_time = time.time()
        
        while self.is_running:
            current_time = time.time()
            
            if current_time >= next_sample_time:
                # Generate and send sample
                sample = self.generate_sample()
                self.outlet.push_sample(sample.tolist())
                
                next_sample_time += sample_interval
            else:
                # Sleep until next sample
                time.sleep(max(0, next_sample_time - current_time))
    
    def start(self):
        """Start the data simulation."""
        if self.is_running:
            logger.warning("Simulator already running")
            return
        
        self.is_running = True
        self.thread = threading.Thread(target=self.streaming_loop)
        self.thread.daemon = True
        self.thread.start()
        
        logger.info("Data simulation started")
    
    def stop(self):
        """Stop the data simulation."""
        if not self.is_running:
            logger.warning("Simulator not running")
            return
        
        self.is_running = False
        
        if self.thread:
            self.thread.join(timeout=2.0)
        
        logger.info("Data simulation stopped")
