# -*- coding: utf-8 -*-
"""Test module for EEG pain classification."""

import pytest
import numpy as np
import torch
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from src.models.cnn import EEGNet, ShallowConvNet, create_model
from src.features.spectral import SpectralFeatureExtractor
from src.utils.helpers import split_data, create_data_loader


class TestModels:
    """Test CNN models."""
    
    def test_eegnet_creation(self):
        """Test EEGNet model creation."""
        model = EEGNet(n_channels=64, n_samples=2000, n_classes=3)
        assert model is not None
        
        # Test forward pass
        x = torch.randn(4, 64, 2000)  # batch_size=4
        output = model(x)
        assert output.shape == (4, 3)
    
    def test_shallow_convnet_creation(self):
        """Test Shallow ConvNet creation."""
        model = ShallowConvNet(n_channels=64, n_samples=2000, n_classes=3)
        assert model is not None
        
        # Test forward pass
        x = torch.randn(4, 64, 2000)
        output = model(x)
        assert output.shape == (4, 3)
    
    def test_model_factory(self):
        """Test model factory function."""
        model = create_model('eegnet', n_channels=32, n_samples=1000, n_classes=3)
        assert isinstance(model, EEGNet)
        
        model = create_model('shallow', n_channels=32, n_samples=1000, n_classes=3)
        assert isinstance(model, ShallowConvNet)
    
    def test_invalid_model_type(self):
        """Test invalid model type."""
        with pytest.raises(ValueError):
            create_model('invalid_model')


class TestFeatureExtraction:
    """Test feature extraction."""
    
    def test_spectral_extractor_creation(self):
        """Test spectral feature extractor creation."""
        extractor = SpectralFeatureExtractor()
        assert extractor is not None
        assert len(extractor.freq_bands) == 5
    
    def test_psd_computation(self):
        """Test PSD computation."""
        extractor = SpectralFeatureExtractor()
        
        # Create synthetic data
        np.random.seed(42)
        epoch_data = np.random.randn(64, 2000)  # 64 channels, 2000 samples
        
        psds, freqs = extractor.compute_psd(epoch_data, sfreq=500)
        
        assert psds.shape[0] == 64  # Same number of channels
        assert len(freqs) == psds.shape[1]
        assert freqs[0] >= 1.0  # Minimum frequency
        assert freqs[-1] <= 45.0  # Maximum frequency
    
    def test_feature_extraction_single_epoch(self):
        """Test feature extraction for single epoch."""
        extractor = SpectralFeatureExtractor()
        
        # Create synthetic data
        np.random.seed(42)
        epoch_data = np.random.randn(64, 2000)
        sfreq = 500
        ch_names = [f'Ch{i}' for i in range(64)]
        
        features = extractor.extract_features_single_epoch(epoch_data, sfreq, ch_names)
        
        assert isinstance(features, np.ndarray)
        assert features.dtype == np.float32
        assert len(features) > 0
    
    def test_feature_extraction_batch(self):
        """Test batch feature extraction."""
        extractor = SpectralFeatureExtractor()
        
        # Create synthetic batch
        np.random.seed(42)
        X = np.random.randn(10, 64, 2000)  # 10 epochs
        sfreq = 500
        ch_names = [f'Ch{i}' for i in range(64)]
        
        features = extractor.extract_features_batch(X, sfreq, ch_names)
        
        assert features.shape[0] == 10  # Same number of epochs
        assert features.shape[1] > 0   # Some features extracted


class TestUtilities:
    """Test utility functions."""
    
    def test_data_splitting(self):
        """Test data splitting function."""
        # Create synthetic data
        np.random.seed(42)
        X = np.random.randn(100, 64, 2000)
        y = np.random.randint(0, 3, 100)
        
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(
            X, y, test_size=0.2, val_size=0.1, random_state=42
        )
        
        # Check sizes
        assert len(X_train) == 70  # 70% for training
        assert len(X_val) == 10    # 10% for validation
        assert len(X_test) == 20   # 20% for testing
        
        # Check no data leakage
        total_samples = len(X_train) + len(X_val) + len(X_test)
        assert total_samples == len(X)
    
    def test_data_loader_creation(self):
        """Test PyTorch DataLoader creation."""
        # Create synthetic data
        np.random.seed(42)
        X = np.random.randn(50, 64, 2000)
        y = np.random.randint(0, 3, 50)
        
        loader = create_data_loader(X, y, batch_size=8, shuffle=True)
        
        # Test one batch
        batch_x, batch_y = next(iter(loader))
        assert batch_x.shape == (8, 64, 2000)
        assert batch_y.shape == (8,)
        assert batch_x.dtype == torch.float32
        assert batch_y.dtype == torch.int64


class TestIntegration:
    """Integration tests."""
    
    def test_end_to_end_pipeline(self):
        """Test complete pipeline from data to prediction."""
        # Create synthetic data
        np.random.seed(42)
        X = np.random.randn(20, 64, 2000)
        y = np.random.randint(0, 3, 20)
        
        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(
            X, y, test_size=0.3, val_size=0.2, random_state=42
        )
        
        # Create model
        model = create_model('eegnet', n_channels=64, n_samples=2000, n_classes=3)
        
        # Create data loader
        train_loader = create_data_loader(X_train, y_train, batch_size=4)
        
        # Test training step
        model.train()
        optimizer = torch.optim.Adam(model.parameters())
        criterion = torch.nn.CrossEntropyLoss()
        
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            break  # Just test one batch
        
        # Test inference
        model.eval()
        with torch.no_grad():
            test_input = torch.FloatTensor(X_test[:4])
            predictions = model(test_input)
            predicted_classes = predictions.argmax(dim=1)
            
            assert predictions.shape == (4, 3)
            assert predicted_classes.shape == (4,)
            assert all(0 <= p <= 2 for p in predicted_classes)


if __name__ == "__main__":
    pytest.main([__file__])
