"""
Utility functions for EEG pain classification project.
"""

import numpy as np
import torch
import pickle
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Union
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


def setup_logging(level: str = 'INFO', log_file: str = None):
    """
    Set up logging configuration.
    
    Parameters:
    -----------
    level : str
        Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
    log_file : str, optional
        Path to log file
    """
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {level}')
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Configure root logger
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[]
    )
    
    # Add console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logging.getLogger().addHandler(console_handler)
    
    # Add file handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logging.getLogger().addHandler(file_handler)


def save_data(data: Dict[str, Any], filepath: str):
    """
    Save data to file (supports .npz, .pkl, .json).
    
    Parameters:
    -----------
    data : dict
        Data to save
    filepath : str
        Output file path
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    if filepath.suffix == '.npz':
        np.savez_compressed(filepath, **data)
    elif filepath.suffix == '.pkl':
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    elif filepath.suffix == '.json':
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    else:
        raise ValueError(f"Unsupported file format: {filepath.suffix}")
    
    logger.info(f"Data saved to {filepath}")


def load_data(filepath: str) -> Dict[str, Any]:
    """
    Load data from file.
    
    Parameters:
    -----------
    filepath : str
        Input file path
        
    Returns:
    --------
    data : dict
        Loaded data
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    if filepath.suffix == '.npz':
        data = dict(np.load(filepath, allow_pickle=True))
    elif filepath.suffix == '.pkl':
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
    elif filepath.suffix == '.json':
        with open(filepath, 'r') as f:
            data = json.load(f)
    else:
        raise ValueError(f"Unsupported file format: {filepath.suffix}")
    
    logger.info(f"Data loaded from {filepath}")
    return data


def split_data(X: np.ndarray, 
               y: np.ndarray,
               test_size: float = 0.2,
               val_size: float = 0.1,
               random_state: int = 42,
               stratify: bool = True) -> Tuple[np.ndarray, ...]:
    """
    Split data into train/validation/test sets.
    
    Parameters:
    -----------
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Target labels
    test_size : float
        Proportion of test set
    val_size : float
        Proportion of validation set
    random_state : int
        Random seed
    stratify : bool
        Whether to stratify split by labels
        
    Returns:
    --------
    X_train, X_val, X_test, y_train, y_val, y_test : np.ndarray
        Split datasets
    """
    stratify_arg = y if stratify else None
    
    # First split: train+val vs test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, 
        stratify=stratify_arg
    )
    
    # Second split: train vs val
    val_size_adjusted = val_size / (1 - test_size)
    stratify_temp = y_temp if stratify else None
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, 
        random_state=random_state, stratify=stratify_temp
    )
    
    logger.info(f"Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    logger.info(f"Train labels: {np.bincount(y_train)}")
    logger.info(f"Val labels: {np.bincount(y_val)}")
    logger.info(f"Test labels: {np.bincount(y_test)}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def normalize_features(X_train: np.ndarray, 
                      X_val: np.ndarray = None, 
                      X_test: np.ndarray = None) -> Tuple[np.ndarray, ...]:
    """
    Normalize features using StandardScaler.
    
    Parameters:
    -----------
    X_train : np.ndarray
        Training features
    X_val : np.ndarray, optional
        Validation features
    X_test : np.ndarray, optional
        Test features
        
    Returns:
    --------
    Normalized arrays and the fitted scaler
    """
    scaler = StandardScaler()
    
    # Fit on training data only
    X_train_norm = scaler.fit_transform(X_train)
    
    results = [X_train_norm, scaler]
    
    if X_val is not None:
        X_val_norm = scaler.transform(X_val)
        results.append(X_val_norm)
    
    if X_test is not None:
        X_test_norm = scaler.transform(X_test)
        results.append(X_test_norm)
    
    return tuple(results)


def evaluate_model(y_true: np.ndarray, 
                  y_pred: np.ndarray,
                  class_names: List[str] = None) -> Dict[str, Any]:
    """
    Evaluate model performance.
    
    Parameters:
    -----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    class_names : list, optional
        Class names for reporting
        
    Returns:
    --------
    results : dict
        Evaluation metrics
    """
    if class_names is None:
        class_names = ['low', 'moderate', 'high']
    
    # Classification report
    report = classification_report(y_true, y_pred, 
                                 target_names=class_names, 
                                 output_dict=True)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Overall accuracy
    accuracy = np.mean(y_true == y_pred)
    
    results = {
        'accuracy': accuracy,
        'classification_report': report,
        'confusion_matrix': cm,
        'class_names': class_names
    }
    
    logger.info(f"Model Accuracy: {accuracy:.4f}")
    logger.info("\nClassification Report:")
    logger.info(classification_report(y_true, y_pred, target_names=class_names))
    
    return results


def plot_confusion_matrix(cm: np.ndarray, 
                         class_names: List[str],
                         title: str = 'Confusion Matrix',
                         save_path: str = None,
                         figsize: Tuple[int, int] = (8, 6)):
    """
    Plot confusion matrix.
    
    Parameters:
    -----------
    cm : np.ndarray
        Confusion matrix
    class_names : list
        Class names
    title : str
        Plot title
    save_path : str, optional
        Path to save figure
    figsize : tuple
        Figure size
    """
    plt.figure(figsize=figsize)
    
    # Normalize confusion matrix
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create heatmap
    sns.heatmap(cm_norm, 
                annot=True, 
                fmt='.2f',
                cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names,
                cbar_kws={'label': 'Normalized Count'})
    
    plt.title(title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Confusion matrix saved to {save_path}")
    
    plt.show()


def plot_training_history(history: Dict[str, List[float]],
                         save_path: str = None,
                         figsize: Tuple[int, int] = (12, 4)):
    """
    Plot training history.
    
    Parameters:
    -----------
    history : dict
        Training history with 'train_loss', 'val_loss', 'train_acc', 'val_acc'
    save_path : str, optional
        Path to save figure
    figsize : tuple
        Figure size
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot loss
    epochs = range(1, len(history['train_loss']) + 1)
    ax1.plot(epochs, history['train_loss'], 'bo-', label='Training Loss')
    if 'val_loss' in history:
        ax1.plot(epochs, history['val_loss'], 'ro-', label='Validation Loss')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracy
    if 'train_acc' in history:
        ax2.plot(epochs, history['train_acc'], 'bo-', label='Training Accuracy')
    if 'val_acc' in history:
        ax2.plot(epochs, history['val_acc'], 'ro-', label='Validation Accuracy')
    ax2.set_title('Model Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Training history saved to {save_path}")
    
    plt.show()


def create_pytorch_dataset(X: np.ndarray, y: np.ndarray) -> torch.utils.data.Dataset:
    """
    Create PyTorch dataset from numpy arrays.
    
    Parameters:
    -----------
    X : np.ndarray
        Feature data
    y : np.ndarray
        Target labels
        
    Returns:
    --------
    dataset : torch.utils.data.Dataset
        PyTorch dataset
    """
    class EEGDataset(torch.utils.data.Dataset):
        def __init__(self, X, y):
            self.X = torch.FloatTensor(X)
            self.y = torch.LongTensor(y)
        
        def __len__(self):
            return len(self.X)
        
        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]
    
    return EEGDataset(X, y)


def create_data_loader(X: np.ndarray, 
                      y: np.ndarray,
                      batch_size: int = 32,
                      shuffle: bool = True,
                      num_workers: int = 0) -> torch.utils.data.DataLoader:
    """
    Create PyTorch DataLoader.
    
    Parameters:
    -----------
    X : np.ndarray
        Feature data
    y : np.ndarray
        Target labels
    batch_size : int
        Batch size
    shuffle : bool
        Whether to shuffle data
    num_workers : int
        Number of worker processes
        
    Returns:
    --------
    loader : torch.utils.data.DataLoader
        PyTorch DataLoader
    """
    dataset = create_pytorch_dataset(X, y)
    loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    return loader


def get_class_weights(y: np.ndarray) -> torch.Tensor:
    """
    Calculate class weights for imbalanced datasets.
    
    Parameters:
    -----------
    y : np.ndarray
        Target labels
        
    Returns:
    --------
    weights : torch.Tensor
        Class weights
    """
    from sklearn.utils.class_weight import compute_class_weight
    
    classes = np.unique(y)
    weights = compute_class_weight('balanced', classes=classes, y=y)
    
    return torch.FloatTensor(weights)


def print_model_summary(model: torch.nn.Module, 
                       input_shape: Tuple[int, ...]):
    """
    Print model summary.
    
    Parameters:
    -----------
    model : torch.nn.Module
        PyTorch model
    input_shape : tuple
        Input shape (channels, samples)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model Summary:")
    print(f"Input shape: {input_shape}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size (MB): {total_params * 4 / 1024 / 1024:.2f}")
    
    # Print layer details
    print("\nModel Architecture:")
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Only leaf modules
            print(f"  {name}: {module}")


def set_seed(seed: int = 42):
    """
    Set random seeds for reproducibility.
    
    Parameters:
    -----------
    seed : int
        Random seed
    """
    import random
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    logger.info(f"Random seed set to {seed}")
