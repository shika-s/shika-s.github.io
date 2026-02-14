"""
Configuration management for BirdCLEF 2025 Keras training pipeline.
"""

import os
import yaml
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path


@dataclass
class PathConfig:
    """Configuration for file paths."""
    
    # Auto-detect project root
    project_root: str = None
    
    def __post_init__(self):
        if self.project_root is None:
            # Auto-detect project root based on common paths
            if os.path.exists('/home/tealeave/projects/mids-207/DATASCI207_Bird_Sounds/'):
                self.project_root = '/home/tealeave/projects/mids-207/DATASCI207_Bird_Sounds/'
            else:
                self.project_root = '/pub/ddlin/projects/mids/DATASCI207_Bird_Sounds/'
        
        # Set all derived paths
        self.data_dir = os.path.join(self.project_root, 'data')
        self.raw_data_dir = os.path.join(self.data_dir, 'raw')
        self.working_data_dir = os.path.join(self.data_dir, 'working')
        
        # Data files
        self.train_audio_dir = os.path.join(self.raw_data_dir, 'train_audio')
        self.train_csv = os.path.join(self.raw_data_dir, 'train.csv')
        self.test_soundscapes_dir = os.path.join(self.raw_data_dir, 'test_soundscapes')
        self.submission_csv = os.path.join(self.raw_data_dir, 'sample_submission.csv')
        self.taxonomy_csv = os.path.join(self.raw_data_dir, 'taxonomy.csv')
        
        # Pre-computed spectrograms
        self.spectrogram_npy = os.path.join(
            self.working_data_dir, 
            'birdclef25-mel-spectrograms/birdclef2025_melspec_5sec_256_256.npy'
        )
        
        # Output directories
        self.output_dir = os.path.join(self.working_data_dir, 'keras_training_output')
        self.model_dir = os.path.join(self.output_dir, 'models')
        self.logs_dir = os.path.join(self.output_dir, 'logs')
        self.tuner_dir = os.path.join(self.output_dir, 'keras_tuner')
        
        # Create output directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs(self.tuner_dir, exist_ok=True)


@dataclass
class AudioConfig:
    """Configuration for audio processing."""
    
    # Audio parameters
    sample_rate: int = 32000
    target_duration: float = 5.0
    target_shape: Tuple[int, int] = (256, 256)
    
    # Mel spectrogram parameters
    n_fft: int = 1024
    hop_length: int = 512
    n_mels: int = 128
    fmin: float = 50.0
    fmax: float = 14000.0
    
    # Data loading
    load_precomputed: bool = True  # Use pre-computed spectrograms if available


@dataclass
class ModelConfig:
    """Configuration for model architecture."""
    
    # Model architecture
    model_name: str = 'efficientnet_b0'
    pretrained: bool = True
    in_channels: int = 1
    dropout_rate: float = 0.2
    drop_path_rate: float = 0.2
    
    # Classification head
    num_classes: int = 182  # Will be updated from taxonomy.csv
    
    # Mixed precision
    use_mixed_precision: bool = True


@dataclass
class TrainingConfig:
    """Configuration for training parameters."""
    
    # Basic training parameters
    epochs: int = 10
    batch_size: int = 32  # Match PyTorch implementation
    learning_rate: float = 5e-4
    weight_decay: float = 1e-5
    
    # Cross-validation
    n_folds: int = 5
    selected_folds: List[int] = None
    
    def __post_init__(self):
        if self.selected_folds is None:
            self.selected_folds = list(range(self.n_folds))
    
    # Optimizer and scheduler
    optimizer: str = 'adamw'  # 'adam', 'adamw', 'rmsprop'
    scheduler: str = 'cosine'  # 'cosine', 'exponential', 'polynomial'
    min_lr: float = 1e-6
    
    # Loss function
    loss_function: str = 'binary_crossentropy'
    
    # Data augmentation
    mixup_alpha: float = 0.5
    augmentation_prob: float = 0.5
    
    # Early stopping and checkpointing
    patience: int = 5
    monitor_metric: str = 'val_auc'
    save_best_only: bool = True
    
    # Hardware
    num_workers: int = 1
    prefetch_buffer_size: int = 2


@dataclass
class HyperparameterSpace:
    """Configuration for hyperparameter tuning space."""
    
    # Learning rate range
    lr_min: float = 1e-5
    lr_max: float = 1e-2
    
    # Batch size options
    batch_sizes: List[int] = None
    
    # Dropout range
    dropout_min: float = 0.1
    dropout_max: float = 0.5
    
    # Mixup alpha range
    mixup_alpha_min: float = 0.1
    mixup_alpha_max: float = 1.0
    
    # Optimizer options
    optimizers: List[str] = None
    
    # Scheduler options
    schedulers: List[str] = None
    
    # Augmentation probability range
    aug_prob_min: float = 0.2
    aug_prob_max: float = 0.8
    
    def __post_init__(self):
        if self.batch_sizes is None:
            self.batch_sizes = [16, 24, 32]  # Include 32 to match PyTorch
        if self.optimizers is None:
            self.optimizers = ['adam', 'adamw', 'rmsprop']
        if self.schedulers is None:
            self.schedulers = ['cosine', 'exponential', 'polynomial']


@dataclass
class TunerConfig:
    """Configuration for Keras Tuner."""
    
    # Tuner type
    tuner_type: str = 'random_search'  # 'random_search', 'hyperband', 'bayesian'
    
    # Search parameters
    max_trials: int = 50
    executions_per_trial: int = 1
    
    # Hyperband specific
    hyperband_iterations: int = 2
    
    # Objective
    objective: str = 'val_auc'
    direction: str = 'max'  # 'max' or 'min'
    
    # Early stopping for trials
    trial_patience: int = 3
    trial_min_epochs: int = 3


@dataclass
class DebugConfig:
    """Configuration for debugging and development."""
    
    debug: bool = False
    max_samples_debug: int = 1000
    debug_epochs: int = 2
    debug_folds: List[int] = None
    
    def __post_init__(self):
        if self.debug_folds is None:
            self.debug_folds = [0]


class Config:
    """Main configuration class that combines all configurations."""
    
    def __init__(self, config_file: Optional[str] = None):
        # Initialize all configuration sections
        self.paths = PathConfig()
        self.audio = AudioConfig()
        self.model = ModelConfig()
        self.training = TrainingConfig()
        self.hyperparams = HyperparameterSpace()
        self.tuner = TunerConfig()
        self.debug = DebugConfig()
        
        # Load from YAML if provided
        if config_file and os.path.exists(config_file):
            self.load_from_yaml(config_file)
        
        # Apply debug settings if enabled
        if self.debug.debug:
            self.apply_debug_settings()
        
        # Update num_classes from taxonomy.csv
        self.update_num_classes()
        
        # Set random seed
        self.seed = 42
    
    def apply_debug_settings(self):
        """Apply debug-specific settings."""
        self.training.epochs = self.debug.debug_epochs
        self.training.selected_folds = self.debug.debug_folds
        self.tuner.max_trials = 3
        self.tuner.executions_per_trial = 1
    
    def update_num_classes(self):
        """Update number of classes from taxonomy.csv."""
        try:
            import pandas as pd
            taxonomy_df = pd.read_csv(self.paths.taxonomy_csv)
            self.model.num_classes = len(taxonomy_df)
            print(f"Updated num_classes to {self.model.num_classes} from taxonomy.csv")
        except Exception as e:
            print(f"Warning: Could not update num_classes from taxonomy.csv: {e}")
            print(f"Using default num_classes: {self.model.num_classes}")
    
    def load_from_yaml(self, config_file: str):
        """Load configuration from YAML file."""
        with open(config_file, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Update configurations
        for section_name, section_config in config_dict.items():
            if hasattr(self, section_name):
                section = getattr(self, section_name)
                for key, value in section_config.items():
                    if hasattr(section, key):
                        setattr(section, key, value)
    
    def save_to_yaml(self, config_file: str):
        """Save configuration to YAML file."""
        config_dict = {
            'paths': asdict(self.paths),
            'audio': asdict(self.audio),
            'model': asdict(self.model),
            'training': asdict(self.training),
            'hyperparams': asdict(self.hyperparams),
            'tuner': asdict(self.tuner),
            'debug': asdict(self.debug)
        }
        
        with open(config_file, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    
    def get_device_strategy(self):
        """Get TensorFlow distributed strategy."""
        import tensorflow as tf
        
        # Check for GPU availability
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                # Enable memory growth
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                
                # Use MirroredStrategy for multi-GPU or single GPU
                if len(gpus) > 1:
                    strategy = tf.distribute.MirroredStrategy()
                    print(f"Using MirroredStrategy with {len(gpus)} GPUs")
                else:
                    strategy = tf.distribute.get_strategy()
                    print(f"Using single GPU: {gpus[0]}")
                    
            except RuntimeError as e:
                print(f"GPU setup error: {e}")
                strategy = tf.distribute.get_strategy()
        else:
            strategy = tf.distribute.get_strategy()
            print("Using CPU")
        
        return strategy
    
    def setup_mixed_precision(self):
        """Setup mixed precision training."""
        if self.model.use_mixed_precision:
            import tensorflow as tf
            
            # Check if GPU supports mixed precision
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                try:
                    # Enable mixed precision
                    policy = tf.keras.mixed_precision.Policy('mixed_float16')
                    tf.keras.mixed_precision.set_global_policy(policy)
                    print("Mixed precision training enabled")
                except Exception as e:
                    print(f"Could not enable mixed precision: {e}")
                    self.model.use_mixed_precision = False
    
    def validate(self):
        """Validate configuration parameters."""
        errors = []
        
        # Check if required paths exist
        if not os.path.exists(self.paths.train_csv):
            errors.append(f"Train CSV not found: {self.paths.train_csv}")
        
        if not os.path.exists(self.paths.taxonomy_csv):
            errors.append(f"Taxonomy CSV not found: {self.paths.taxonomy_csv}")
        
        if not os.path.exists(self.paths.train_audio_dir):
            errors.append(f"Train audio directory not found: {self.paths.train_audio_dir}")
        
        # Check if pre-computed spectrograms exist if enabled
        if self.audio.load_precomputed and not os.path.exists(self.paths.spectrogram_npy):
            print(f"Warning: Pre-computed spectrograms not found: {self.paths.spectrogram_npy}")
            print("Will generate spectrograms on-the-fly")
            self.audio.load_precomputed = False
        
        # Validate hyperparameter ranges
        if self.hyperparams.lr_min >= self.hyperparams.lr_max:
            errors.append("Learning rate min must be less than max")
        
        if self.hyperparams.dropout_min >= self.hyperparams.dropout_max:
            errors.append("Dropout min must be less than max")
        
        # Validate training parameters
        if self.training.batch_size <= 0:
            errors.append("Batch size must be positive")
        
        if self.training.epochs <= 0:
            errors.append("Epochs must be positive")
        
        if errors:
            raise ValueError("Configuration validation failed:\n" + "\n".join(errors))
        
        print("Configuration validation passed")
    
    def __str__(self):
        """String representation of configuration."""
        sections = []
        for attr_name in ['paths', 'audio', 'model', 'training', 'hyperparams', 'tuner', 'debug']:
            if hasattr(self, attr_name):
                attr = getattr(self, attr_name)
                sections.append(f"{attr_name.upper()}:")
                for key, value in asdict(attr).items():
                    sections.append(f"  {key}: {value}")
                sections.append("")
        
        return "\n".join(sections)


def create_default_config() -> Config:
    """Create a default configuration."""
    return Config()


def load_config(config_file: str) -> Config:
    """Load configuration from YAML file."""
    return Config(config_file)


if __name__ == "__main__":
    # Test configuration
    config = create_default_config()
    print("Default configuration:")
    print(config)
    
    # Validate configuration
    try:
        config.validate()
        print("Configuration is valid!")
    except ValueError as e:
        print(f"Configuration validation failed: {e}")