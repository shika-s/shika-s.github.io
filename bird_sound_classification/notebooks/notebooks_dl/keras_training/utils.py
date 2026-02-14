"""
Utility functions for BirdCLEF 2025 Keras training pipeline.
"""

import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from typing import Dict, List, Tuple, Optional, Any
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import logging
import time
from pathlib import Path
import cv2
import librosa
import gc


def set_random_seeds(seed: int = 42):
    """
    Set random seeds for reproducibility across all libraries.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # Set deterministic behavior for TensorFlow
    tf.config.experimental.enable_op_determinism()
    
    print(f"Random seeds set to {seed}")


def setup_logging(log_dir: str, log_level: str = 'INFO') -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        log_dir: Directory to save log files
        log_level: Logging level
        
    Returns:
        Configured logger
    """
    os.makedirs(log_dir, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger('birdclef_keras')
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # File handler
    log_file = os.path.join(log_dir, f'training_{int(time.time())}.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(getattr(logging, log_level.upper()))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger


def calculate_auc_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate average AUC score for multi-label classification.
    
    Args:
        y_true: True labels (batch_size, num_classes)
        y_pred: Predicted probabilities (batch_size, num_classes)
        
    Returns:
        Average AUC score across all classes
    """
    if len(y_true.shape) == 1:
        y_true = y_true.reshape(-1, 1)
    if len(y_pred.shape) == 1:
        y_pred = y_pred.reshape(-1, 1)
    
    num_classes = y_true.shape[1]
    aucs = []
    
    for i in range(num_classes):
        # Only calculate AUC if there are positive samples
        if np.sum(y_true[:, i]) > 0 and np.sum(y_true[:, i]) < len(y_true[:, i]):
            try:
                auc = roc_auc_score(y_true[:, i], y_pred[:, i])
                aucs.append(auc)
            except ValueError:
                # Skip classes with issues
                continue
    
    return np.mean(aucs) if aucs else 0.0


class AUCCallback(tf.keras.callbacks.Callback):
    """Custom callback to calculate AUC during training."""
    
    def __init__(self, validation_data: tf.data.Dataset, name: str = "auc"):
        super().__init__()
        self.validation_data = validation_data
        self.name = name
        
    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
            
        # Collect all validation predictions and labels
        y_true_list = []
        y_pred_list = []
        
        for batch in self.validation_data:
            if isinstance(batch, dict):
                x_batch = batch['melspec']
                y_batch = batch['target']
            else:
                x_batch, y_batch = batch
                
            y_pred_batch = self.model(x_batch, training=False)
            
            # Apply sigmoid to get probabilities
            y_pred_batch = tf.nn.sigmoid(y_pred_batch)
            
            y_true_list.append(y_batch.numpy())
            y_pred_list.append(y_pred_batch.numpy())
        
        # Concatenate all batches
        y_true = np.concatenate(y_true_list, axis=0)
        y_pred = np.concatenate(y_pred_list, axis=0)
        
        # Calculate AUC
        auc_score = calculate_auc_score(y_true, y_pred)
        logs[f'val_{self.name}'] = auc_score
        
        print(f' - val_{self.name}: {auc_score:.4f}')


def create_stratified_folds(
    df: pd.DataFrame, 
    n_splits: int = 5, 
    random_state: int = 42
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Create stratified folds for cross-validation.
    
    Args:
        df: DataFrame with 'primary_label' column
        n_splits: Number of folds
        random_state: Random state for reproducibility
        
    Returns:
        List of (train_indices, val_indices) tuples
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    folds = []
    for train_idx, val_idx in skf.split(df, df['primary_label']):
        folds.append((train_idx, val_idx))
    
    return folds


def audio_to_melspec(
    audio_data: np.ndarray, 
    sample_rate: int = 32000,
    n_fft: int = 1024,
    hop_length: int = 512,
    n_mels: int = 128,
    fmin: float = 50.0,
    fmax: float = 14000.0,
    target_shape: Tuple[int, int] = (256, 256)
) -> np.ndarray:
    """
    Convert audio data to mel spectrogram.
    
    Args:
        audio_data: Audio waveform
        sample_rate: Audio sample rate
        n_fft: FFT window size
        hop_length: Hop length for STFT
        n_mels: Number of mel bands
        fmin: Minimum frequency
        fmax: Maximum frequency
        target_shape: Target shape for spectrogram
        
    Returns:
        Normalized mel spectrogram
    """
    # Handle NaN values
    if np.isnan(audio_data).any():
        mean_signal = np.nanmean(audio_data)
        audio_data = np.nan_to_num(audio_data, nan=mean_signal)
    
    # Generate mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=audio_data,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        power=2.0
    )
    
    # Convert to dB
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Normalize to [0, 1]
    mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)
    
    # Resize to target shape
    if mel_spec_norm.shape != target_shape:
        mel_spec_norm = cv2.resize(mel_spec_norm, target_shape, interpolation=cv2.INTER_LINEAR)
    
    return mel_spec_norm.astype(np.float32)


def process_audio_file(
    audio_path: str,
    sample_rate: int = 32000,
    target_duration: float = 5.0,
    target_shape: Tuple[int, int] = (256, 256),
    **mel_kwargs
) -> Optional[np.ndarray]:
    """
    Process a single audio file to get mel spectrogram.
    
    Args:
        audio_path: Path to audio file
        sample_rate: Target sample rate
        target_duration: Target duration in seconds
        target_shape: Target spectrogram shape
        **mel_kwargs: Additional arguments for mel spectrogram generation
        
    Returns:
        Processed mel spectrogram or None if error
    """
    try:
        # Load audio
        audio_data, _ = librosa.load(audio_path, sr=sample_rate)
        
        # Calculate target samples
        target_samples = int(target_duration * sample_rate)
        
        # Repeat audio if too short
        if len(audio_data) < target_samples:
            n_copy = int(np.ceil(target_samples / len(audio_data)))
            if n_copy > 1:
                audio_data = np.concatenate([audio_data] * n_copy)
        
        # Extract center segment
        start_idx = max(0, int(len(audio_data) / 2 - target_samples / 2))
        end_idx = min(len(audio_data), start_idx + target_samples)
        center_audio = audio_data[start_idx:end_idx]
        
        # Pad if necessary
        if len(center_audio) < target_samples:
            center_audio = np.pad(
                center_audio, 
                (0, target_samples - len(center_audio)), 
                mode='constant'
            )
        
        # Generate mel spectrogram
        mel_spec = audio_to_melspec(
            center_audio, 
            sample_rate=sample_rate,
            target_shape=target_shape,
            **mel_kwargs
        )
        
        return mel_spec
        
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None


def clear_memory():
    """Clear memory and run garbage collection."""
    gc.collect()
    
    # Clear TensorFlow backend
    tf.keras.backend.clear_session()
    
    # Clear GPU memory if available
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.reset_memory_growth(gpu)
        except:
            pass


def get_model_size(model: tf.keras.Model) -> Dict[str, Any]:
    """
    Get model size information.
    
    Args:
        model: Keras model
        
    Returns:
        Dictionary with model size information
    """
    total_params = model.count_params()
    trainable_params = np.sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    non_trainable_params = total_params - trainable_params
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'non_trainable_params': non_trainable_params,
        'model_size_mb': total_params * 4 / (1024 * 1024)  # Assuming float32
    }


def print_model_summary(model: tf.keras.Model, name: str = "Model"):
    """
    Print detailed model summary.
    
    Args:
        model: Keras model
        name: Model name for printing
    """
    print(f"\n{name} Summary:")
    print("=" * 50)
    
    # Basic summary
    model.summary()
    
    # Size information
    size_info = get_model_size(model)
    print(f"\nModel Size Information:")
    print(f"Total parameters: {size_info['total_params']:,}")
    print(f"Trainable parameters: {size_info['trainable_params']:,}")
    print(f"Non-trainable parameters: {size_info['non_trainable_params']:,}")
    print(f"Estimated model size: {size_info['model_size_mb']:.2f} MB")
    print("=" * 50)


def save_training_history(
    history: tf.keras.callbacks.History,
    save_path: str,
    fold: Optional[int] = None
):
    """
    Save training history to CSV file.
    
    Args:
        history: Keras training history
        save_path: Path to save the history
        fold: Fold number (optional)
    """
    if hasattr(history, 'history'):
        history_dict = history.history
    else:
        history_dict = history
    
    # Convert to DataFrame
    df = pd.DataFrame(history_dict)
    
    # Add fold information if provided
    if fold is not None:
        df['fold'] = fold
    
    # Add epoch number
    df['epoch'] = range(len(df))
    
    # Save to CSV
    df.to_csv(save_path, index=False)
    print(f"Training history saved to {save_path}")


def load_precomputed_spectrograms(npy_path: str) -> Optional[Dict[str, np.ndarray]]:
    """
    Load pre-computed spectrograms from NPY file.
    
    Args:
        npy_path: Path to NPY file
        
    Returns:
        Dictionary of spectrograms or None if error
    """
    try:
        spectrograms = np.load(npy_path, allow_pickle=True).item()
        print(f"Loaded {len(spectrograms)} pre-computed spectrograms")
        return spectrograms
    except Exception as e:
        print(f"Error loading pre-computed spectrograms: {e}")
        return None


def create_sample_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create sample names from filename column.
    
    Args:
        df: DataFrame with 'filename' column
        
    Returns:
        DataFrame with 'samplename' column added
    """
    df = df.copy()
    df['samplename'] = df['filename'].apply(
        lambda x: x.split('/')[0] + '-' + x.split('/')[-1].split('.')[0]
    )
    return df


def monitor_gpu_memory():
    """Monitor and print GPU memory usage."""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for i, gpu in enumerate(gpus):
                memory_info = tf.config.experimental.get_memory_info(gpu.name)
                current_mb = memory_info['current'] / (1024 * 1024)
                peak_mb = memory_info['peak'] / (1024 * 1024)
                print(f"GPU {i}: Current memory: {current_mb:.2f} MB, Peak memory: {peak_mb:.2f} MB")
        except:
            print("Could not get GPU memory info")
    else:
        print("No GPUs available")


def format_time(seconds: float) -> str:
    """
    Format seconds into human-readable time string.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    else:
        return f"{minutes:02d}:{seconds:02d}"


class Timer:
    """Simple timer context manager."""
    
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = None
        
    def __enter__(self):
        self.start_time = time.time()
        print(f"Starting {self.name}...")
        return self
        
    def __exit__(self, *args):
        elapsed = time.time() - self.start_time
        print(f"{self.name} completed in {format_time(elapsed)}")


def create_model_checkpoint_callback(
    filepath: str,
    monitor: str = 'val_auc',
    mode: str = 'max',
    save_best_only: bool = True,
    save_weights_only: bool = False,
    verbose: int = 1
) -> tf.keras.callbacks.ModelCheckpoint:
    """
    Create model checkpoint callback.
    
    Args:
        filepath: Path to save model
        monitor: Metric to monitor
        mode: 'min' or 'max'
        save_best_only: Whether to save only best model
        save_weights_only: Whether to save only weights
        verbose: Verbosity level
        
    Returns:
        ModelCheckpoint callback
    """
    return tf.keras.callbacks.ModelCheckpoint(
        filepath=filepath,
        monitor=monitor,
        mode=mode,
        save_best_only=save_best_only,
        save_weights_only=save_weights_only,
        verbose=verbose
    )


def create_early_stopping_callback(
    monitor: str = 'val_auc',
    patience: int = 5,
    mode: str = 'max',
    restore_best_weights: bool = True,
    verbose: int = 1
) -> tf.keras.callbacks.EarlyStopping:
    """
    Create early stopping callback.
    
    Args:
        monitor: Metric to monitor
        patience: Number of epochs with no improvement
        mode: 'min' or 'max'
        restore_best_weights: Whether to restore best weights
        verbose: Verbosity level
        
    Returns:
        EarlyStopping callback
    """
    return tf.keras.callbacks.EarlyStopping(
        monitor=monitor,
        patience=patience,
        mode=mode,
        restore_best_weights=restore_best_weights,
        verbose=verbose
    )


def create_reduce_lr_callback(
    monitor: str = 'val_loss',
    factor: float = 0.5,
    patience: int = 3,
    min_lr: float = 1e-7,
    mode: str = 'min',
    verbose: int = 1
) -> tf.keras.callbacks.ReduceLROnPlateau:
    """
    Create reduce learning rate callback.
    
    Args:
        monitor: Metric to monitor
        factor: Factor to reduce learning rate
        patience: Number of epochs with no improvement
        min_lr: Minimum learning rate
        mode: 'min' or 'max'
        verbose: Verbosity level
        
    Returns:
        ReduceLROnPlateau callback
    """
    return tf.keras.callbacks.ReduceLROnPlateau(
        monitor=monitor,
        factor=factor,
        patience=patience,
        min_lr=min_lr,
        mode=mode,
        verbose=verbose
    )


if __name__ == "__main__":
    # Test utility functions
    print("Testing utility functions...")
    
    # Test random seed setting
    set_random_seeds(42)
    
    # Test AUC calculation
    y_true = np.random.randint(0, 2, (100, 5))
    y_pred = np.random.random((100, 5))
    auc = calculate_auc_score(y_true, y_pred)
    print(f"Test AUC score: {auc:.4f}")
    
    # Test timer
    with Timer("Test operation"):
        time.sleep(1)
    
    print("Utility functions test completed!")