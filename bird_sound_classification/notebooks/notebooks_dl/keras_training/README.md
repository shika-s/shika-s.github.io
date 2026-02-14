# BirdCLEF 2025 Keras Training Pipeline

A comprehensive TensorFlow/Keras implementation for BirdCLEF 2025 bird sound classification with systematic hyperparameter optimization and SLURM job submission support.

## 🎯 Overview

This pipeline converts your PyTorch implementation to a highly optimized Keras solution featuring:

- **Systematic Hyperparameter Optimization** with Keras Tuner (RandomSearch, Hyperband, Bayesian)
- **EfficientNet-B0** backbone with custom classification head (206 bird species)
- **Advanced Data Pipeline** with TensorFlow for optimal GPU utilization
- **SLURM Integration** for seamless cluster job submission (`sbatch -p free-gpu --gres=gpu:V100:1`)
- **5-Fold Cross-Validation** with ensemble capability
- **Mixed Precision Training** for V100 efficiency
- **Comprehensive Augmentations** (Mixup, Time/Frequency masking, Brightness/Contrast)

## 📁 Project Structure

```
keras_training/
├── config.py                    # Configuration management
├── utils.py                     # Utility functions (AUC, seeds, logging)
├── data_pipeline.py             # TensorFlow data pipeline
├── model_definitions.py         # EfficientNet model with custom head
├── train_keras_tuner.py         # Hyperparameter optimization script
├── train_single_model.py        # Single model training script
├── test_pipeline.py             # Pipeline testing script
├── slurm_scripts/
│   ├── setup_environment.sh     # Environment setup
│   ├── submit_job.sh            # Job submission convenience script
│   ├── run_hyperparameter_tuning.sh  # SLURM hyperparameter tuning
│   └── run_training.sh          # SLURM model training
└── results/                     # Output directory for models and logs
```

## 🏗️ Code Architecture

### Core Components Overview

The codebase is organized into several key components that work together to create a complete machine learning pipeline:

#### 1. **Configuration System** (`config.py`)
- **Purpose**: Centralized configuration management using Python dataclasses
- **Key Classes**:
  - `Config`: Main configuration class that combines all settings
  - `PathConfig`: File and directory path management with auto-detection
  - `AudioConfig`: Audio processing parameters (sample rate, spectrogram settings)
  - `ModelConfig`: Neural network architecture settings
  - `TrainingConfig`: Training parameters (epochs, batch size, optimization)
  - `TunerConfig`: Hyperparameter tuning configuration
  - `DebugConfig`: Debug mode settings (reduced samples, epochs, trials)

#### 2. **Data Pipeline** (`data_pipeline.py`)
- **Purpose**: TensorFlow-based data loading and preprocessing pipeline
- **Key Class**: `BirdCLEFDataPipeline`
  - Handles both pre-computed spectrograms and on-the-fly audio processing
  - Creates cross-validation splits using stratified sampling
  - Applies data augmentations (mixup, SpecAugment, brightness/contrast)
  - Manages multi-label encoding for 206 bird species
  - Optimizes GPU utilization with TensorFlow `tf.data` API

#### 3. **Model Architecture** (`model_definitions.py`)
- **Purpose**: Neural network model definition and training utilities
- **Key Classes**:
  - `BirdCLEFModel`: Main model class with EfficientNet backbone
  - `ClassWiseAUC`: Custom AUC metric for multi-label classification
  - `HyperModel`: Keras Tuner integration for hyperparameter optimization
- **Architecture**: EfficientNet-B0 backbone → Global Average Pooling → Dropout → Dense(206)

#### 4. **Training Orchestration** (`train_single_model.py`)
- **Purpose**: Manages the complete training process with cross-validation
- **Key Class**: `BirdCLEFTrainer`
  - Coordinates configuration, data pipeline, and model training
  - Handles 5-fold cross-validation
  - Manages model checkpointing and early stopping
  - Calculates and logs performance metrics

#### 5. **Hyperparameter Optimization** (`train_keras_tuner.py`)
- **Purpose**: Automated hyperparameter search using Keras Tuner
- **Key Class**: `BirdCLEFTuner`
  - Supports multiple search algorithms (RandomSearch, Hyperband, Bayesian)
  - Optimizes learning rate, batch size, dropout, optimizer, scheduler
  - Handles distributed training and memory management

#### 6. **Utilities** (`utils.py`)
- **Purpose**: Shared utility functions and helper classes
- **Key Functions**:
  - Audio processing: `process_audio_file()`, `audio_to_melspec()`
  - Metrics: `calculate_auc_score()`, `AUCCallback`
  - System: `set_random_seeds()`, `clear_memory()`, `monitor_gpu_memory()`
  - Logging: `setup_logging()`, `Timer` context manager

### Component Interaction Flow

```
┌─────────────┐    ┌────────────────┐    ┌─────────────────┐
│   Config    │───▶│  Data Pipeline │───▶│     Model       │
│   System    │    │   (tf.data)    │    │ (EfficientNet)  │
└─────────────┘    └────────────────┘    └─────────────────┘
       │                     │                      │
       │                     │                      │
       ▼                     ▼                      ▼
┌─────────────┐    ┌────────────────┐    ┌─────────────────┐
│    SLURM    │    │   Training     │    │    Results      │
│   Scripts   │    │   (Trainer)    │    │   & Metrics     │
└─────────────┘    └────────────────┘    └─────────────────┘
```

### Key Design Patterns

1. **Configuration-Driven**: All components use the centralized `Config` system
2. **Pipeline Pattern**: Data flows through well-defined processing stages
3. **Strategy Pattern**: Multiple tuner types and optimizers supported
4. **Factory Pattern**: Model and optimizer creation through factory functions
5. **Observer Pattern**: Callbacks for monitoring training progress

### Debug Mode Architecture

The debug mode is implemented at the configuration level and affects all components:
- **Data Pipeline**: Uses only 1,000 samples instead of full dataset (~28K)
- **Training**: Runs for 2 epochs instead of 10-20
- **Hyperparameter Tuning**: Tests only 3 trials instead of 50+
- **Cross-Validation**: Uses only fold 0 instead of all 5 folds
- **Same Code Path**: All components use identical logic, just with reduced parameters

## ⚙️ Configuration System Deep Dive

### Config Class Structure

The configuration system is built using Python dataclasses for type safety and clarity. Each component has its own configuration dataclass that gets combined into a main `Config` class:

```python
from config import Config, create_default_config

# Create default configuration
config = create_default_config()

# Access different configuration sections
print(f"Project root: {config.paths.project_root}")
print(f"Number of classes: {config.model.num_classes}")
print(f"Training epochs: {config.training.epochs}")
print(f"Debug mode: {config.debug.debug}")
```

### PathConfig - File and Directory Management

```python
@dataclass
class PathConfig:
    project_root: str = None  # Auto-detected
    data_dir: str             # /path/to/data
    train_csv: str            # /path/to/train.csv
    spectrogram_npy: str      # /path/to/precomputed/spectrograms.npy
    output_dir: str           # /path/to/output
    model_dir: str            # /path/to/models
    logs_dir: str             # /path/to/logs
    tuner_dir: str            # /path/to/keras_tuner
```

**Key Features**:
- **Auto-detection**: Automatically detects project root based on common paths
- **Directory creation**: Creates output directories if they don't exist
- **Path validation**: Validates that required input files exist

### AudioConfig - Audio Processing Parameters

```python
@dataclass
class AudioConfig:
    sample_rate: int = 32000           # Target sample rate
    target_duration: float = 5.0       # Audio segment duration in seconds
    target_shape: Tuple[int, int] = (256, 256)  # Spectrogram dimensions
    n_fft: int = 1024                 # FFT window size
    hop_length: int = 512             # STFT hop length
    n_mels: int = 128                 # Number of mel frequency bands
    fmin: float = 50.0                # Minimum frequency
    fmax: float = 14000.0             # Maximum frequency
    load_precomputed: bool = True     # Use pre-computed spectrograms
```

**Usage Example**:
```python
# Modify audio processing parameters
config.audio.sample_rate = 44100
config.audio.target_duration = 10.0
config.audio.target_shape = (512, 512)
```

### ModelConfig - Neural Network Architecture

```python
@dataclass
class ModelConfig:
    model_name: str = 'efficientnet_b0'    # EfficientNet variant
    pretrained: bool = True                # Use ImageNet pretrained weights
    in_channels: int = 1                   # Input channels (grayscale spectrograms)
    dropout_rate: float = 0.2              # Dropout rate in classification head
    num_classes: int = 206                 # Number of bird species (auto-updated)
    use_mixed_precision: bool = True       # FP16 training for V100 efficiency
```

### TrainingConfig - Training Parameters

```python
@dataclass
class TrainingConfig:
    epochs: int = 10                       # Number of training epochs
    batch_size: int = 32                   # Training batch size
    learning_rate: float = 5e-4            # Initial learning rate
    weight_decay: float = 1e-5             # L2 regularization strength
    n_folds: int = 5                       # Cross-validation folds
    selected_folds: List[int] = [0,1,2,3,4] # Which folds to train
    optimizer: str = 'adamw'               # Optimizer type
    scheduler: str = 'cosine'              # Learning rate scheduler
    mixup_alpha: float = 0.5               # Mixup augmentation strength
    patience: int = 5                      # Early stopping patience
    monitor_metric: str = 'val_auc'        # Metric to monitor for best model
```

### TunerConfig - Hyperparameter Optimization

```python
@dataclass
class TunerConfig:
    tuner_type: str = 'random_search'      # 'random_search', 'hyperband', 'bayesian'
    max_trials: int = 50                   # Maximum number of trials
    executions_per_trial: int = 1          # Executions per trial for averaging
    objective: str = 'val_auc'             # Optimization objective
    direction: str = 'max'                 # 'max' or 'min'
    trial_patience: int = 3                # Early stopping patience per trial
```

### HyperparameterSpace - Search Space Definition

```python
@dataclass
class HyperparameterSpace:
    lr_min: float = 1e-5                   # Learning rate range
    lr_max: float = 1e-2
    batch_sizes: List[int] = [16, 32, 64]  # Batch size options
    dropout_min: float = 0.1               # Dropout rate range
    dropout_max: float = 0.5
    optimizers: List[str] = ['adam', 'adamw', 'rmsprop']  # Optimizer options
    schedulers: List[str] = ['cosine', 'exponential']     # Scheduler options
```

### DebugConfig - Debug Mode Settings

```python
@dataclass
class DebugConfig:
    debug: bool = False                    # Enable debug mode
    max_samples_debug: int = 1000          # Max samples in debug mode
    debug_epochs: int = 2                  # Epochs in debug mode
    debug_folds: List[int] = [0]           # Folds to use in debug mode
```

### Configuration Usage Examples

#### Creating and Customizing Configuration

```python
from config import Config, create_default_config

# Method 1: Create default config
config = create_default_config()

# Method 2: Load from YAML file
config = Config(config_file='custom_config.yaml')

# Method 3: Create and customize
config = Config()
config.debug.debug = True                  # Enable debug mode
config.training.epochs = 5                 # Reduce epochs
config.training.batch_size = 16            # Reduce batch size
config.model.dropout_rate = 0.3            # Increase dropout
```

#### Environment Variable Integration

The configuration system automatically reads environment variables in SLURM scripts:

```bash
# In SLURM script
export DEBUG=true
export EPOCHS=10
export BATCH_SIZE=32
export TUNER_TYPE=bayesian

# Python code automatically applies these via argument parsing
```

#### Configuration Validation

```python
# Validate configuration before use
try:
    config.validate()
    print("Configuration is valid!")
except ValueError as e:
    print(f"Configuration error: {e}")
```

#### Saving and Loading Configuration

```python
# Save configuration to YAML
config.save_to_yaml('my_config.yaml')

# Load configuration from YAML
config = Config('my_config.yaml')

# Configuration sections are automatically updated
```

### Advanced Configuration Features

#### Automatic Path Detection

The `PathConfig` class automatically detects the project root:

```python
# Tries multiple common paths
if os.path.exists('/home/tealeave/projects/mids-207/DATASCI207_Bird_Sounds/'):
    project_root = '/home/tealeave/projects/mids-207/DATASCI207_Bird_Sounds/'
else:
    project_root = '/pub/ddlin/projects/mids/DATASCI207_Bird_Sounds/'
```

#### Mixed Precision Setup

```python
# Automatically configured based on GPU availability
config.setup_mixed_precision()
if config.model.use_mixed_precision:
    print("Mixed precision enabled for faster training")
```

#### Debug Mode Application

```python
# Debug settings are automatically applied
if config.debug.debug:
    config.apply_debug_settings()
    # This reduces epochs, samples, trials, and folds
```

## 🔄 Data Pipeline Deep Dive

### BirdCLEFDataPipeline Class

The `BirdCLEFDataPipeline` class in `data_pipeline.py` manages all data loading, preprocessing, and augmentation. It's designed to handle both pre-computed spectrograms and on-the-fly audio processing efficiently.

#### Core Architecture

```python
class BirdCLEFDataPipeline:
    def __init__(self, config: Config):
        self.config = config
        self.spectrograms = None              # Pre-computed spectrograms dict
        self.species_to_idx = {}              # Label encoding mapping
        self.idx_to_species = {}              # Reverse mapping
        self.num_classes = 206                # Number of bird species
        
        self._load_taxonomy()                 # Load species labels
        self._load_precomputed_spectrograms() # Load pre-computed data if available
```

#### Key Methods and Their Functions

##### 1. Data Loading and Preparation

```python
def prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepares the DataFrame for training by:
    1. Adding file paths to audio files
    2. Creating sample names for spectrogram lookup
    3. Applying debug mode filtering (1000 samples)
    4. Filtering for available pre-computed spectrograms
    """
    df = df.copy()
    if 'filepath' not in df.columns:
        df['filepath'] = df['filename'].apply(
            lambda x: os.path.join(self.config.paths.train_audio_dir, x)
        )
    
    # Create sample names for spectrogram lookup
    df = create_sample_names(df)
    
    # Apply debug mode filtering
    if self.config.debug.debug:
        df = df.sample(min(1000, len(df)), random_state=42)
    
    return df
```

##### 2. Label Encoding

```python
def encode_labels(self, df: pd.DataFrame) -> np.ndarray:
    """
    Converts bird species labels to multi-hot encoded arrays:
    - Primary labels: Always encoded (main species)
    - Secondary labels: Additional species in the same audio clip
    - Output shape: (num_samples, 206) for 206 bird species
    """
    labels = np.zeros((len(df), self.num_classes), dtype=np.float32)
    
    for i, row in df.iterrows():
        # Encode primary label
        if row['primary_label'] in self.species_to_idx:
            labels[i, self.species_to_idx[row['primary_label']]] = 1.0
        
        # Encode secondary labels
        if 'secondary_labels' in row and pd.notna(row['secondary_labels']):
            secondary_labels = eval(row['secondary_labels'])
            for label in secondary_labels:
                if label in self.species_to_idx:
                    labels[i, self.species_to_idx[label]] = 1.0
    
    return labels
```

##### 3. Spectrogram Loading

```python
def _get_spectrogram_tf(self, sample_name: str, filepath: str) -> tf.Tensor:
    """
    Retrieves spectrograms using two methods:
    1. Pre-computed: Fast lookup from loaded .npy file
    2. On-the-fly: Dynamic audio processing using librosa
    
    Returns: TensorFlow tensor of shape (256, 256, 1)
    """
    def get_precomputed(name_tensor):
        return self.spectrograms.get(
            name_tensor.numpy().decode(), 
            np.zeros((256, 256), dtype=np.float32)
        )
    
    def generate_on_the_fly(path_tensor):
        return self._process_audio_tf(path_tensor.numpy().decode())
    
    if self.config.audio.load_precomputed and self.spectrograms:
        spectrogram = tf.py_function(get_precomputed, [sample_name], tf.float32)
    else:
        spectrogram = tf.py_function(generate_on_the_fly, [filepath], tf.float32)
    
    spectrogram.set_shape((256, 256))
    return tf.expand_dims(spectrogram, axis=-1)  # Add channel dimension
```

### Data Augmentation Pipeline

The pipeline implements several augmentation techniques to improve model generalization:

#### 1. Mixup Augmentation

```python
def _apply_mixup(self, features: tf.Tensor, labels: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Implements mixup augmentation:
    - Samples lambda from Beta distribution
    - Mixes features: mixed = λ * x1 + (1-λ) * x2
    - Mixes labels: mixed = λ * y1 + (1-λ) * y2
    - Applied at batch level after batching
    """
    lam = tf.random.gamma([1], alpha=0.5, beta=0.5)[0]  # Beta distribution
    lam = tf.clip_by_value(lam, 0.0, 1.0)
    
    indices = tf.random.shuffle(tf.range(tf.shape(features)[0]))
    
    mixed_features = lam * features + (1 - lam) * tf.gather(features, indices)
    mixed_labels = lam * labels + (1 - lam) * tf.gather(labels, indices)
    
    return mixed_features, mixed_labels
```

#### 2. SpecAugment (Time and Frequency Masking)

```python
def _apply_spectrogram_augmentations(self, spectrogram: tf.Tensor) -> tf.Tensor:
    """
    Applies SpecAugment augmentations:
    1. Time masking: Horizontal stripes (mask time steps)
    2. Frequency masking: Vertical stripes (mask frequency bins)
    3. Brightness/contrast: Random intensity adjustments
    """
    # Time masking (horizontal stripes)
    if tf.random.uniform([]) < 0.5:
        mask_width = tf.random.uniform([], minval=5, maxval=21, dtype=tf.int32)
        # ... masking logic
    
    # Frequency masking (vertical stripes)
    if tf.random.uniform([]) < 0.5:
        mask_height = tf.random.uniform([], minval=5, maxval=21, dtype=tf.int32)
        # ... masking logic
    
    # Brightness/contrast adjustment
    if tf.random.uniform([]) < 0.5:
        gain = tf.random.uniform([], minval=0.8, maxval=1.2)
        bias = tf.random.uniform([], minval=-0.1, maxval=0.1)
        spectrogram = tf.clip_by_value(spectrogram * gain + bias, 0.0, 1.0)
    
    return spectrogram
```

### Cross-Validation Data Splitting

```python
def create_fold_datasets(self, df: pd.DataFrame, fold: int) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """
    Creates stratified train/validation splits:
    1. Uses StratifiedKFold to ensure balanced species distribution
    2. Maintains same species ratios in train and validation sets
    3. Creates separate tf.data.Dataset for each split
    """
    from sklearn.model_selection import StratifiedKFold
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, val_idx = list(skf.split(df, df['primary_label']))[fold]
    
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)
    
    train_dataset = self.create_dataset(train_df, is_training=True)
    val_dataset = self.create_dataset(val_df, is_training=False)
    
    return train_dataset, val_dataset
```

### TensorFlow Dataset Creation

```python
def create_dataset(self, df: pd.DataFrame, is_training: bool = True) -> tf.data.Dataset:
    """
    Creates optimized tf.data.Dataset with the following pipeline:
    
    1. Create dataset from DataFrame
    2. Shuffle (if training)
    3. Map: Load spectrograms and apply augmentations
    4. Batch: Group samples into batches
    5. Mixup: Apply batch-level mixup augmentation
    6. Prefetch: Overlap data loading with training
    """
    # Step 1: Create dataset from tensors
    dataset = tf.data.Dataset.from_tensor_slices({
        "samplename": df['samplename'].values,
        "filepath": df['filepath'].values,
        "label": self.encode_labels(df)
    })
    
    # Step 2: Shuffle for training
    if is_training:
        dataset = dataset.shuffle(buffer_size=1000, seed=42)
    
    # Step 3: Map function to load spectrograms and augment
    def map_fn(sample):
        spectrogram = self._get_spectrogram_tf(sample['samplename'], sample['filepath'])
        
        # Apply spectrogram augmentations during training
        if is_training and tf.random.uniform([]) < 0.5:
            spectrogram = self._apply_spectrogram_augmentations(spectrogram)
        
        return {'melspec': spectrogram}, sample['label']
    
    dataset = dataset.map(map_fn, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Step 4: Batch the dataset
    dataset = dataset.batch(self.config.training.batch_size)
    
    # Step 5: Apply mixup at batch level
    if is_training and self.config.training.mixup_alpha > 0:
        def mixup_batch(features, labels):
            mixed_features, mixed_labels = self._apply_mixup(features['melspec'], labels)
            return {'melspec': mixed_features}, mixed_labels
        
        dataset = dataset.map(mixup_batch, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Step 6: Prefetch for performance
    return dataset.prefetch(tf.data.AUTOTUNE)
```

### Performance Optimizations

#### 1. TensorFlow Data API Optimizations

```python
# Parallel processing
dataset = dataset.map(map_fn, num_parallel_calls=tf.data.AUTOTUNE)

# Prefetching
dataset = dataset.prefetch(tf.data.AUTOTUNE)

# Batching with drop_remainder for consistent shapes
dataset = dataset.batch(batch_size, drop_remainder=True)
```

#### 2. Pre-computed Spectrograms

```python
def _load_precomputed_spectrograms(self):
    """
    Loads pre-computed spectrograms from .npy file:
    - Faster than on-the-fly processing
    - Consistent preprocessing across runs
    - Reduces GPU memory usage during training
    """
    if os.path.exists(self.config.paths.spectrogram_npy):
        self.spectrograms = np.load(self.config.paths.spectrogram_npy, allow_pickle=True).item()
        print(f"Loaded {len(self.spectrograms)} pre-computed spectrograms")
    else:
        print("No pre-computed spectrograms found, will generate on-the-fly")
```

#### 3. Memory Management

```python
# Efficient data types
labels = np.zeros((len(df), self.num_classes), dtype=np.float32)  # Float32 instead of Float64

# Proper tensor shapes
spectrogram.set_shape((256, 256))  # Set static shape for optimization
```

### Data Pipeline Usage Examples

#### Basic Usage

```python
from data_pipeline import create_data_pipeline
from config import create_default_config

# Create configuration and data pipeline
config = create_default_config()
pipeline = create_data_pipeline(config)

# Load training data
train_df = pd.read_csv(config.paths.train_csv)

# Create training dataset
train_dataset = pipeline.create_dataset(train_df, is_training=True)

# Iterate through batches
for batch_features, batch_labels in train_dataset.take(1):
    print(f"Batch features shape: {batch_features['melspec'].shape}")
    print(f"Batch labels shape: {batch_labels.shape}")
```

#### Cross-Validation Usage

```python
# Create datasets for specific fold
train_dataset, val_dataset = pipeline.create_fold_datasets(train_df, fold=0)

# Get dataset sizes
train_size = len(train_df) * 4 // 5  # 80% for training
val_size = len(train_df) // 5        # 20% for validation

print(f"Training samples: {train_size}")
print(f"Validation samples: {val_size}")
```

#### Debug Mode Usage

```python
# Enable debug mode for faster iteration
config.debug.debug = True
config.apply_debug_settings()

# Create pipeline with debug settings
pipeline = create_data_pipeline(config)

# Now uses only 1000 samples instead of full dataset
debug_dataset = pipeline.create_dataset(train_df, is_training=True)
```

### Key Design Decisions

1. **Dual Loading Strategy**: Supports both pre-computed and on-the-fly processing
2. **TensorFlow Integration**: Uses tf.data API for optimal GPU utilization
3. **Stratified Sampling**: Maintains balanced species distribution in cross-validation
4. **Batch-Level Mixup**: Applies mixup after batching for efficiency
5. **Configurable Augmentations**: All augmentation parameters controlled via config
6. **Memory Efficiency**: Uses appropriate data types and prefetching

## 🧠 Model Architecture Deep Dive

### BirdCLEFModel Class

The `BirdCLEFModel` class in `model_definitions.py` implements a transfer learning approach using EfficientNet as the backbone with a custom classification head designed for multi-label bird species classification.

#### Core Architecture Overview

```python
class BirdCLEFModel(tf.keras.Model):
    def __init__(self, config: Config, **kwargs):
        super().__init__(**kwargs)
        
        self.config = config
        self.num_classes = config.model.num_classes  # 206 bird species
        self.dropout_rate = config.model.dropout_rate
        
        # Build model components
        self._build_backbone()    # EfficientNet-B0
        self._build_head()        # Classification head
```

#### Model Components

##### 1. **Backbone: EfficientNet-B0**

```python
def _build_backbone(self):
    """
    Creates EfficientNet-B0 backbone for feature extraction:
    - Pre-trained on ImageNet for transfer learning
    - Modified input to accept 1-channel spectrograms
    - Includes top layers removed for custom classification
    """
    from tensorflow.keras.applications import EfficientNetB0
    
    self.backbone = EfficientNetB0(
        weights='imagenet',                    # Pre-trained weights
        include_top=False,                     # Remove classification head
        input_shape=(256, 256, 1),             # 1-channel spectrograms
        pooling=None                           # We'll add custom pooling
    )
    
    # Enable fine-tuning
    self.backbone.trainable = True
```

**Key Features**:
- **Transfer Learning**: Leverages ImageNet pre-trained weights
- **Channel Adaptation**: Automatically adapts from 3-channel RGB to 1-channel spectrograms
- **Efficient Architecture**: EfficientNet-B0 balances performance and computational efficiency
- **Feature Extraction**: Outputs feature maps of shape (batch_size, 8, 8, 1280)

##### 2. **Classification Head**

```python
def _build_head(self):
    """
    Creates custom classification head:
    - Global Average Pooling for spatial dimension reduction
    - Dropout for regularization
    - Dense layer for multi-label classification
    """
    # Global pooling to reduce spatial dimensions
    self.global_pool = tf.keras.layers.GlobalAveragePooling2D(
        name='global_avg_pool'
    )
    
    # Dropout for regularization
    self.dropout = tf.keras.layers.Dropout(
        self.dropout_rate, 
        name='head_dropout'
    )
    
    # Final classification layer
    self.classifier = tf.keras.layers.Dense(
        self.num_classes,           # 206 bird species
        activation=None,            # Raw logits for binary cross-entropy
        name='classifier'
    )
```

#### Forward Pass Implementation

```python
def call(self, inputs, training=None):
    """
    Defines the forward pass through the model:
    
    Input: (batch_size, 256, 256, 1) spectrograms
    ↓
    EfficientNet Backbone: Feature extraction
    ↓
    Global Average Pooling: (batch_size, 1280)
    ↓
    Dropout: Regularization
    ↓
    Dense Layer: (batch_size, 206) logits
    """
    # Handle dictionary input from data pipeline
    x = inputs['melspec'] if isinstance(inputs, dict) else inputs
    
    # 1. Extract features using EfficientNet backbone
    x = self.backbone(x, training=training)  # (batch_size, 8, 8, 1280)
    
    # 2. Global average pooling
    x = self.global_pool(x)                  # (batch_size, 1280)
    
    # 3. Apply dropout
    x = self.dropout(x, training=training)   # (batch_size, 1280)
    
    # 4. Classification
    logits = self.classifier(x)              # (batch_size, 206)
    
    return logits
```

### Custom AUC Metric

The model uses a custom AUC metric designed for multi-label classification:

```python
class ClassWiseAUC(tf.keras.metrics.Metric):
    """
    Custom AUC metric that calculates per-class AUC and averages them.
    This matches the evaluation methodology used in PyTorch implementation.
    """
    
    def __init__(self, num_classes: int, name='class_wise_auc', **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self._y_true_list = []
        self._y_pred_list = []
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        """Store predictions for batch-wise calculation"""
        y_pred = tf.nn.sigmoid(y_pred)  # Convert logits to probabilities
        self._y_true_list.append(y_true)
        self._y_pred_list.append(y_pred)
    
    def result(self):
        """Calculate final AUC score"""
        y_true = tf.concat(self._y_true_list, axis=0)
        y_pred = tf.concat(self._y_pred_list, axis=0)
        
        # Calculate AUC for each class
        aucs = []
        for i in range(self.num_classes):
            if tf.reduce_sum(y_true[:, i]) > 0:  # Skip classes with no positive samples
                try:
                    auc = roc_auc_score(y_true[:, i].numpy(), y_pred[:, i].numpy())
                    aucs.append(auc)
                except ValueError:
                    continue
        
        return np.mean(aucs) if aucs else 0.0
```

### Model Creation and Compilation

```python
def create_model(config: Config, hp: Optional[kt.HyperParameters] = None) -> tf.keras.Model:
    """
    Factory function to create, compile, and return the BirdCLEF model:
    
    1. Create model instance
    2. Configure optimizer
    3. Compile with loss function and metrics
    4. Return ready-to-train model
    """
    # Extract hyperparameters
    if hp:
        dropout_rate = hp.Float('dropout_rate', 0.1, 0.5)
        learning_rate = hp.Float('learning_rate', 1e-5, 1e-2, sampling='LOG')
        optimizer_name = hp.Choice('optimizer', ['adam', 'adamw', 'rmsprop'])
    else:
        dropout_rate = config.model.dropout_rate
        learning_rate = config.training.learning_rate
        optimizer_name = config.training.optimizer
    
    # Create model
    model = BirdCLEFModel(
        config=config,
        dropout_rate=dropout_rate
    )
    
    # Create optimizer
    optimizer = create_optimizer(optimizer_name, learning_rate, config)
    
    # Compile model
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),  # Multi-label
        metrics=[ClassWiseAUC(config.model.num_classes)]
    )
    
    return model
```

### Optimizer Configuration

```python
def create_optimizer(optimizer_name: str, learning_rate: float, config: Config) -> tf.keras.optimizers.Optimizer:
    """
    Creates optimizers with appropriate configuration:
    - Adam: Standard adaptive optimizer
    - AdamW: Adam with weight decay (L2 regularization)
    - RMSprop: Alternative adaptive optimizer
    """
    weight_decay = config.training.weight_decay
    
    if optimizer_name.lower() == 'adam':
        return tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_name.lower() == 'adamw':
        return tf.keras.optimizers.AdamW(
            learning_rate=learning_rate, 
            weight_decay=weight_decay
        )
    elif optimizer_name.lower() == 'rmsprop':
        return tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    else:
        raise ValueError(f"Optimizer {optimizer_name} not supported")
```

### Learning Rate Scheduling

```python
def create_lr_scheduler(scheduler_name: str, optimizer: tf.keras.optimizers.Optimizer, config: Config, steps_per_epoch: int) -> Optional[tf.keras.callbacks.Callback]:
    """
    Creates learning rate schedulers:
    - Cosine: Cosine annealing with warm restarts
    - Exponential: Exponential decay
    - Polynomial: Polynomial decay
    - Reduce on Plateau: Adaptive reduction
    """
    if scheduler_name.lower() == 'cosine':
        return tf.keras.callbacks.CosineRestartDecay(
            initial_learning_rate=config.training.learning_rate,
            first_decay_steps=config.training.epochs * steps_per_epoch
        )
    
    elif scheduler_name.lower() == 'exponential':
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=config.training.learning_rate,
            decay_steps=steps_per_epoch,
            decay_rate=0.9
        )
        optimizer.learning_rate = lr_schedule
        return None
    
    elif scheduler_name.lower() == 'reduce_on_plateau':
        return tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=config.training.min_lr
        )
    
    return None
```

### Training Callbacks

```python
def create_callbacks(config: Config, model_path: str, log_dir: str) -> list:
    """
    Creates training callbacks for monitoring and control:
    - ModelCheckpoint: Save best model
    - EarlyStopping: Prevent overfitting
    - TensorBoard: Training visualization
    - ReduceLROnPlateau: Adaptive learning rate
    """
    callbacks_list = []
    
    # Save best model
    callbacks_list.append(tf.keras.callbacks.ModelCheckpoint(
        filepath=model_path,
        monitor='val_auc',
        mode='max',
        save_best_only=True,
        verbose=1
    ))
    
    # Early stopping
    callbacks_list.append(tf.keras.callbacks.EarlyStopping(
        monitor='val_auc',
        patience=config.training.patience,
        mode='max',
        restore_best_weights=True
    ))
    
    # TensorBoard logging
    callbacks_list.append(tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
        update_freq='epoch'
    ))
    
    return callbacks_list
```

### Hyperparameter Optimization Integration

```python
class HyperModel(kt.HyperModel):
    """
    Keras Tuner integration for hyperparameter optimization.
    Defines the search space and builds models with different hyperparameters.
    """
    
    def __init__(self, config: Config):
        self.config = config
    
    def build(self, hp: kt.HyperParameters) -> tf.keras.Model:
        """
        Build model with hyperparameters:
        - Learning rate: 1e-5 to 1e-2 (log scale)
        - Dropout rate: 0.1 to 0.5
        - Optimizer: Adam, AdamW, RMSprop
        - Batch size: 16, 32, 64
        """
        # Define hyperparameter search space
        learning_rate = hp.Float('learning_rate', 1e-5, 1e-2, sampling='LOG')
        dropout_rate = hp.Float('dropout_rate', 0.1, 0.5)
        optimizer_name = hp.Choice('optimizer', ['adam', 'adamw', 'rmsprop'])
        
        # Create model with hyperparameters
        return create_model(self.config, hp)
```

### Model Architecture Visualization

```
Input: (batch_size, 256, 256, 1) Spectrograms
                    ↓
┌─────────────────────────────────────────────────────┐
│                EfficientNet-B0                      │
│                   Backbone                          │
│  (Pre-trained on ImageNet, fine-tuned)            │
│                                                     │
│  Input → Conv2D → MBConv Blocks → Output           │
│  (256,256,1) → ... → (8,8,1280)                   │
└─────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────┐
│            Global Average Pooling                   │
│              (8,8,1280) → (1280)                   │
└─────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────┐
│                 Dropout                             │
│            (rate = 0.2 default)                     │
└─────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────┐
│              Dense Layer                            │
│          (1280) → (206) logits                     │
│        (206 bird species)                          │
└─────────────────────────────────────────────────────┘
                    ↓
Output: (batch_size, 206) Raw logits for multi-label classification
```

### Model Usage Examples

#### Basic Model Creation

```python
from model_definitions import create_model
from config import create_default_config

# Create configuration
config = create_default_config()

# Create model
model = create_model(config)

# Print model summary
model.summary()

# Model parameters
print(f"Total parameters: {model.count_params():,}")
print(f"Input shape: {model.input_shape}")
print(f"Output shape: {model.output_shape}")
```

#### Model Training

```python
# Compile model (already done in create_model)
model = create_model(config)

# Train model
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=config.training.epochs,
    callbacks=create_callbacks(config, 'model.h5', 'logs/')
)
```

#### Model Prediction

```python
# Load trained model
model = tf.keras.models.load_model('best_model.h5')

# Make predictions
predictions = model.predict(test_dataset)

# Convert logits to probabilities
probabilities = tf.nn.sigmoid(predictions)

# Get top predictions for each sample
top_predictions = tf.nn.top_k(probabilities, k=5)
```

### Key Architecture Design Decisions

1. **Transfer Learning**: Uses EfficientNet-B0 pre-trained on ImageNet for feature extraction
2. **Multi-Label Classification**: Designed for multiple bird species in single audio clip
3. **Custom AUC Metric**: Implements class-wise AUC calculation for proper evaluation
4. **Configurable Architecture**: All parameters controlled through configuration system
5. **Mixed Precision**: Supports FP16 training for V100 efficiency
6. **Flexible Optimization**: Supports multiple optimizers and learning rate schedulers

### Performance Characteristics

- **Model Size**: ~4.3M parameters (lightweight for production)
- **Input Processing**: 256x256 mel spectrograms (1-channel)
- **Training Speed**: ~2-3 minutes per epoch on V100 (debug mode)
- **Memory Usage**: ~8GB GPU memory for batch_size=32
- **Accuracy**: Target AUC > 0.85 on validation set

## 🎯 Training Process Deep Dive

### BirdCLEFTrainer Class

The `BirdCLEFTrainer` class in `train_single_model.py` orchestrates the complete training process, handling cross-validation, model training, evaluation, and result aggregation.

#### Core Architecture

```python
class BirdCLEFTrainer:
    def __init__(self, config: Config, hyperparameters: dict = None):
        self.config = config
        self.hyperparameters = hyperparameters or {}
        self.logger = None
        self.data_pipeline = None
        
        # Setup environment, logging, and data pipeline
        self._setup_environment()
        self._setup_logging()
        self._setup_data_pipeline()
        self._apply_hyperparameters()
```

#### Key Methods and Training Flow

##### 1. Environment Setup

```python
def _setup_environment(self):
    """
    Configures the training environment:
    1. Sets random seeds for reproducibility
    2. Configures distributed training strategy
    3. Enables mixed precision if available
    4. Monitors GPU memory usage
    """
    # Reproducibility
    set_random_seeds(self.config.seed)
    
    # Distributed training strategy
    self.strategy = self.config.get_device_strategy()
    
    # Mixed precision for V100 efficiency
    self.config.setup_mixed_precision()
    
    # Environment monitoring
    monitor_gpu_memory()
```

##### 2. Hyperparameter Application

```python
def _apply_hyperparameters(self):
    """
    Applies hyperparameters from tuning to configuration:
    - Learning rate, batch size, optimizer
    - Dropout rate, mixup alpha
    - Scheduler type and parameters
    """
    if not self.hyperparameters:
        return
    
    hp_mapping = {
        'learning_rate': ('training', 'learning_rate'),
        'batch_size': ('training', 'batch_size'),
        'optimizer': ('training', 'optimizer'),
        'dropout_rate': ('model', 'dropout_rate'),
        'mixup_alpha': ('training', 'mixup_alpha')
    }
    
    for hp_key, (section, config_key) in hp_mapping.items():
        if hp_key in self.hyperparameters:
            section_obj = getattr(self.config, section)
            setattr(section_obj, config_key, self.hyperparameters[hp_key])
```

##### 3. Single Fold Training

```python
def train_single_fold(self, fold: int, train_df: pd.DataFrame) -> dict:
    """
    Trains a model on a single cross-validation fold:
    
    1. Create train/validation datasets
    2. Build model within distributed strategy scope
    3. Configure callbacks and learning rate scheduler
    4. Train model with early stopping
    5. Evaluate and save results
    """
    # Create datasets
    train_dataset, val_dataset = self.data_pipeline.create_fold_datasets(
        train_df, fold
    )
    
    # Model creation within strategy scope
    with self.strategy.scope():
        model = create_model(self.config, **self.hyperparameters)
        
        # Learning rate scheduler
        lr_scheduler = create_lr_scheduler(
            self.config.training.scheduler,
            model.optimizer,
            self.config,
            steps_per_epoch
        )
    
    # Training callbacks
    callbacks_list = create_callbacks(
        self.config,
        model_path=f'best_model_fold_{fold}.h5',
        log_dir=f'logs/fold_{fold}'
    )
    
    # Training
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=self.config.training.epochs,
        callbacks=callbacks_list,
        verbose=1
    )
    
    # Evaluation
    val_loss = model.evaluate(val_dataset, verbose=0)
    val_auc = self._calculate_validation_auc(model, val_dataset)
    
    return {
        'fold': fold,
        'best_val_loss': float(min(history.history['val_loss'])),
        'val_auc': float(val_auc),
        'model_path': model_path,
        'training_time': elapsed_time
    }
```

##### 4. Cross-Validation Training

```python
def train_cross_validation(self, train_df: pd.DataFrame) -> dict:
    """
    Orchestrates training across all cross-validation folds:
    
    1. Iterates through selected folds
    2. Trains model on each fold
    3. Collects performance metrics
    4. Aggregates results with mean and std
    5. Saves comprehensive results
    """
    all_results = {}
    fold_scores = {'val_loss': [], 'val_auc': [], 'train_loss': []}
    
    for fold in self.config.training.selected_folds:
        try:
            # Train single fold
            fold_results = self.train_single_fold(fold, train_df)
            all_results[f'fold_{fold}'] = fold_results
            
            # Collect scores
            fold_scores['val_loss'].append(fold_results['best_val_loss'])
            fold_scores['val_auc'].append(fold_results['val_auc'])
            
        except Exception as e:
            self.logger.error(f"Error in fold {fold}: {str(e)}")
            continue
        
        finally:
            # Memory cleanup
            tf.keras.backend.clear_session()
            gc.collect()
    
    # Aggregate results
    aggregated_results = {
        'cv_results': {
            'mean_val_loss': float(np.mean(fold_scores['val_loss'])),
            'std_val_loss': float(np.std(fold_scores['val_loss'])),
            'mean_val_auc': float(np.mean(fold_scores['val_auc'])),
            'std_val_auc': float(np.std(fold_scores['val_auc']))
        },
        'individual_folds': all_results,
        'num_completed_folds': len(fold_scores['val_loss'])
    }
    
    return aggregated_results
```

##### 5. AUC Calculation

```python
def _calculate_validation_auc(self, model: tf.keras.Model, val_dataset: tf.data.Dataset) -> float:
    """
    Calculates AUC on validation dataset:
    1. Iterates through validation batches
    2. Collects predictions and true labels
    3. Converts logits to probabilities
    4. Calculates class-wise AUC and averages
    """
    y_true_list = []
    y_pred_list = []
    
    for batch in val_dataset:
        if isinstance(batch, dict):
            x_batch = batch['melspec']
            y_batch = batch['target']
        else:
            x_batch, y_batch = batch
        
        # Get predictions
        y_pred_batch = model(x_batch, training=False)
        y_pred_batch = tf.nn.sigmoid(y_pred_batch)
        
        y_true_list.append(y_batch.numpy())
        y_pred_list.append(y_pred_batch.numpy())
    
    # Concatenate and calculate AUC
    y_true = np.concatenate(y_true_list, axis=0)
    y_pred = np.concatenate(y_pred_list, axis=0)
    
    return calculate_auc_score(y_true, y_pred)
```

### Training Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    Training Pipeline Flow                       │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  1. Environment Setup                                           │
│     • Set random seeds                                          │
│     • Configure distributed strategy                            │
│     • Enable mixed precision                                    │
│     • Monitor GPU memory                                        │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  2. Data Pipeline Setup                                         │
│     • Load taxonomy and create label mappings                   │
│     • Load pre-computed spectrograms if available               │
│     • Prepare DataFrame with debug filtering                    │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  3. Cross-Validation Loop                                       │
│     For each fold (0-4):                                        │
│                                                                 │
│     ┌─────────────────────────────────────────────────────────┐ │
│     │  3.1. Create Fold Datasets                             │ │
│     │       • Stratified train/val split                     │ │
│     │       • Apply augmentations to training data           │ │
│     │       • Create tf.data.Dataset objects                 │ │
│     └─────────────────────────────────────────────────────────┘ │
│                                │                                │
│                                ▼                                │
│     ┌─────────────────────────────────────────────────────────┐ │
│     │  3.2. Model Creation                                   │ │
│     │       • Build model within strategy scope              │ │
│     │       • Apply hyperparameters                          │ │
│     │       • Create optimizer and scheduler                 │ │
│     └─────────────────────────────────────────────────────────┘ │
│                                │                                │
│                                ▼                                │
│     ┌─────────────────────────────────────────────────────────┐ │
│     │  3.3. Training                                         │ │
│     │       • Configure callbacks                            │ │
│     │       • Train with early stopping                      │ │
│     │       • Monitor validation AUC                         │ │
│     │       • Save best model                                │ │
│     └─────────────────────────────────────────────────────────┘ │
│                                │                                │
│                                ▼                                │
│     ┌─────────────────────────────────────────────────────────┐ │
│     │  3.4. Evaluation                                       │ │
│     │       • Load best model weights                        │ │
│     │       • Calculate validation AUC                       │ │
│     │       • Save training history                          │ │
│     │       • Collect fold results                           │ │
│     └─────────────────────────────────────────────────────────┘ │
│                                │                                │
│                                ▼                                │
│     ┌─────────────────────────────────────────────────────────┐ │
│     │  3.5. Memory Cleanup                                   │ │
│     │       • Clear TensorFlow session                       │ │
│     │       • Run garbage collection                         │ │
│     │       • Monitor memory usage                           │ │
│     └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  4. Results Aggregation                                         │
│     • Calculate mean and std across folds                       │
│     • Compile comprehensive results                             │
│     • Save to JSON file                                         │
│     • Generate training summary                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Hyperparameter Integration

The trainer seamlessly integrates with hyperparameter optimization results:

```python
def load_hyperparameters(hp_path: str) -> dict:
    """
    Loads hyperparameters from JSON file produced by tuning:
    1. Reads tuning results JSON
    2. Extracts best hyperparameters
    3. Returns dict for trainer use
    """
    with open(hp_path, 'r') as f:
        data = json.load(f)
    
    # Extract best hyperparameters
    if 'best_overall_hyperparameters' in data:
        return data['best_overall_hyperparameters']
    elif 'best_hyperparameters' in data:
        return data['best_hyperparameters']
    else:
        return data

# Usage in training script
hyperparameters = load_hyperparameters('results/hyperparameter_tuning_results.json')
trainer = BirdCLEFTrainer(config, hyperparameters)
```

### Training Monitoring and Callbacks

#### 1. Model Checkpointing

```python
tf.keras.callbacks.ModelCheckpoint(
    filepath='best_model_fold_{fold}.h5',
    monitor='val_auc',           # Monitor validation AUC
    mode='max',                  # Maximize AUC
    save_best_only=True,         # Only save best model
    save_weights_only=False,     # Save full model
    verbose=1
)
```

#### 2. Early Stopping

```python
tf.keras.callbacks.EarlyStopping(
    monitor='val_auc',
    patience=5,                  # Stop after 5 epochs without improvement
    mode='max',
    restore_best_weights=True,   # Restore best weights when stopping
    verbose=1
)
```

#### 3. TensorBoard Logging

```python
tf.keras.callbacks.TensorBoard(
    log_dir=f'logs/fold_{fold}',
    histogram_freq=1,            # Log histograms every epoch
    update_freq='epoch'          # Update logs every epoch
)
```

#### 4. Learning Rate Scheduling

```python
# Example: Cosine annealing
tf.keras.callbacks.CosineRestartDecay(
    initial_learning_rate=5e-4,
    first_decay_steps=epochs * steps_per_epoch,
    alpha=1e-6                   # Minimum learning rate
)
```

### Training Usage Examples

#### Basic Training

```python
from train_single_model import BirdCLEFTrainer
from config import create_default_config
import pandas as pd

# Setup
config = create_default_config()
trainer = BirdCLEFTrainer(config)

# Load data
train_df = pd.read_csv(config.paths.train_csv)

# Train with cross-validation
results = trainer.train_cross_validation(train_df)

print(f"Mean val AUC: {results['cv_results']['mean_val_auc']:.4f}")
print(f"Std val AUC: {results['cv_results']['std_val_auc']:.4f}")
```

#### Training with Hyperparameters

```python
from train_single_model import BirdCLEFTrainer, load_hyperparameters

# Load best hyperparameters from tuning
hyperparameters = load_hyperparameters('results/hyperparameter_tuning_results.json')

# Create trainer with hyperparameters
trainer = BirdCLEFTrainer(config, hyperparameters)

# Train
results = trainer.train_cross_validation(train_df)
```

#### Single Fold Training

```python
# Train only specific fold
config.training.selected_folds = [0]
trainer = BirdCLEFTrainer(config)

# Train single fold
results = trainer.train_single_fold(0, train_df)
print(f"Fold 0 AUC: {results['val_auc']:.4f}")
```

#### Debug Mode Training

```python
# Enable debug mode
config.debug.debug = True
config.apply_debug_settings()

# Create trainer
trainer = BirdCLEFTrainer(config)

# Quick debug training (1000 samples, 2 epochs, 1 fold)
results = trainer.train_cross_validation(train_df)
```

### Training Results Structure

The trainer produces comprehensive results in JSON format:

```json
{
  "cv_results": {
    "mean_val_loss": 0.1234,
    "std_val_loss": 0.0123,
    "mean_val_auc": 0.8567,
    "std_val_auc": 0.0234,
    "mean_train_loss": 0.0987,
    "std_train_loss": 0.0098
  },
  "individual_folds": {
    "fold_0": {
      "fold": 0,
      "best_epoch": 15,
      "best_val_loss": 0.1198,
      "val_auc": 0.8634,
      "model_path": "results/models/best_model_fold_0.h5",
      "training_time": 1234.56
    }
  },
  "num_completed_folds": 5,
  "training_config": {
    "epochs": 20,
    "batch_size": 32,
    "learning_rate": 0.0005,
    "optimizer": "adamw"
  },
  "hyperparameters": {
    "learning_rate": 0.0003,
    "dropout_rate": 0.3,
    "mixup_alpha": 0.4
  }
}
```

### Error Handling and Recovery

The trainer includes robust error handling:

```python
try:
    fold_results = self.train_single_fold(fold, train_df)
    all_results[f'fold_{fold}'] = fold_results
except Exception as e:
    self.logger.error(f"Error in fold {fold}: {str(e)}")
    import traceback
    self.logger.error(traceback.format_exc())
    continue  # Continue with next fold
finally:
    # Always cleanup memory
    tf.keras.backend.clear_session()
    gc.collect()
```

### Key Training Features

1. **Cross-Validation**: Robust 5-fold stratified cross-validation
2. **Hyperparameter Integration**: Seamless integration with tuning results
3. **Memory Management**: Automatic cleanup between folds
4. **Monitoring**: Comprehensive logging and TensorBoard integration
5. **Error Recovery**: Continues training even if individual folds fail
6. **Reproducibility**: Consistent results through proper seeding
7. **Distributed Training**: Automatic GPU utilization and mixed precision
8. **Debug Mode**: Fast iteration with reduced data and epochs

## 🔬 Complete ML Pipeline Technical Explanation

### Pipeline Overview: From Audio to Predictions

This section explains how all components work together to create a complete machine learning pipeline for bird sound classification.

```
Raw Audio Files → Spectrograms → Features → Model → Predictions
     ↓              ↓            ↓         ↓         ↓
  librosa      TF Pipeline   EfficientNet  Sigmoid  Species
 processing    + Augments    Backbone     Activation Labels
```

### Step-by-Step Pipeline Flow

#### 1. **Data Ingestion and Preparation**

```python
# The pipeline starts with raw audio files and metadata
train_df = pd.read_csv('train.csv')  # Contains filename, primary_label, secondary_labels
audio_files = glob.glob('train_audio/**/*.ogg')  # ~28,000 audio files

# BirdCLEFDataPipeline handles the complexity
pipeline = BirdCLEFDataPipeline(config)
pipeline.prepare_dataframe(train_df)  # Adds filepaths, creates sample names

# In debug mode: Only uses 1000 samples for fast iteration
if config.debug.debug:
    train_df = train_df.sample(1000)
```

#### 2. **Audio Processing and Spectrogram Generation**

```python
# Two processing pathways:

# Option A: Pre-computed spectrograms (FAST)
spectrograms = np.load('precomputed_spectrograms.npy')  # 28K spectrograms
spectrogram = spectrograms[sample_name]  # Direct lookup

# Option B: On-the-fly processing (FLEXIBLE)
def process_audio_file(audio_path):
    # Load audio with librosa
    audio, sr = librosa.load(audio_path, sr=32000)
    
    # Extract 5-second center segment
    center_audio = audio[start:start+5*32000]
    
    # Generate mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=center_audio,
        sr=32000,
        n_fft=1024,
        hop_length=512,
        n_mels=128
    )
    
    # Convert to dB and normalize
    mel_spec_db = librosa.power_to_db(mel_spec)
    normalized = (mel_spec_db - min) / (max - min)
    
    # Resize to 256x256
    resized = cv2.resize(normalized, (256, 256))
    
    return resized.astype(np.float32)
```

#### 3. **Label Encoding and Multi-Label Setup**

```python
# Convert bird species names to multi-hot encoded vectors
def encode_labels(df):
    labels = np.zeros((len(df), 206), dtype=np.float32)  # 206 bird species
    
    for i, row in df.iterrows():
        # Primary label (always present)
        species_idx = species_to_idx[row['primary_label']]
        labels[i, species_idx] = 1.0
        
        # Secondary labels (multiple birds in same clip)
        for secondary_species in row['secondary_labels']:
            if secondary_species in species_to_idx:
                labels[i, species_to_idx[secondary_species]] = 1.0
    
    return labels

# Example: A clip might contain both "Common Robin" and "Blue Jay"
# labels[i] = [0, 0, 1, 0, ..., 1, 0, ...]  # Two 1s for two species
```

#### 4. **TensorFlow Dataset Creation and Optimization**

```python
# Create optimized tf.data.Dataset
dataset = tf.data.Dataset.from_tensor_slices({
    'samplename': df['samplename'].values,
    'filepath': df['filepath'].values,
    'label': encoded_labels
})

# Shuffle for training
dataset = dataset.shuffle(buffer_size=1000)

# Map function loads spectrograms
def load_spectrogram(sample):
    spec = get_spectrogram_tf(sample['samplename'], sample['filepath'])
    return {'melspec': spec}, sample['label']

# Parallel processing for performance
dataset = dataset.map(load_spectrogram, num_parallel_calls=tf.data.AUTOTUNE)

# Batch the data
dataset = dataset.batch(32)

# Prefetch for GPU utilization
dataset = dataset.prefetch(tf.data.AUTOTUNE)
```

#### 5. **Data Augmentation Pipeline**

```python
# Applied during training for better generalization

# A. Spectrogram-level augmentations (applied to individual samples)
def augment_spectrogram(spec):
    # Time masking: mask horizontal stripes (time steps)
    if random.random() < 0.5:
        mask_width = random.randint(5, 20)
        start_time = random.randint(0, 256 - mask_width)
        spec[:, start_time:start_time+mask_width] = 0
    
    # Frequency masking: mask vertical stripes (frequency bins)
    if random.random() < 0.5:
        mask_height = random.randint(5, 20)
        start_freq = random.randint(0, 256 - mask_height)
        spec[start_freq:start_freq+mask_height, :] = 0
    
    # Brightness/contrast adjustment
    if random.random() < 0.5:
        gain = random.uniform(0.8, 1.2)
        bias = random.uniform(-0.1, 0.1)
        spec = np.clip(spec * gain + bias, 0, 1)
    
    return spec

# B. Mixup augmentation (applied to batches)
def mixup_batch(batch_x, batch_y):
    # Sample mixing coefficient from Beta distribution
    lam = np.random.beta(0.5, 0.5)
    
    # Shuffle batch indices
    indices = np.random.permutation(len(batch_x))
    
    # Mix features and labels
    mixed_x = lam * batch_x + (1 - lam) * batch_x[indices]
    mixed_y = lam * batch_y + (1 - lam) * batch_y[indices]
    
    return mixed_x, mixed_y
```

#### 6. **Model Architecture and Forward Pass**

```python
# Model processes batches of spectrograms
def forward_pass(batch):
    # Input: (batch_size, 256, 256, 1) spectrograms
    x = batch['melspec']
    
    # 1. EfficientNet backbone feature extraction
    # Automatically handles 1-channel → 3-channel conversion
    features = efficientnet_backbone(x)  # → (batch_size, 8, 8, 1280)
    
    # 2. Global average pooling
    pooled = tf.reduce_mean(features, axis=[1, 2])  # → (batch_size, 1280)
    
    # 3. Dropout for regularization
    if training:
        pooled = tf.nn.dropout(pooled, rate=0.2)
    
    # 4. Final classification layer
    logits = tf.matmul(pooled, classifier_weights)  # → (batch_size, 206)
    
    return logits

# Multi-label classification: Each logit represents probability of that species
# probabilities = tf.nn.sigmoid(logits)  # Convert to [0,1] probabilities
```

#### 7. **Training Process and Optimization**

```python
# Cross-validation training loop
for fold in range(5):
    # 1. Create stratified train/validation split
    train_idx, val_idx = stratified_split(df, fold)
    
    # 2. Create datasets
    train_dataset = create_dataset(df.iloc[train_idx], training=True)
    val_dataset = create_dataset(df.iloc[val_idx], training=False)
    
    # 3. Create model
    model = create_model(config)
    
    # 4. Train with callbacks
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=20,
        callbacks=[
            ModelCheckpoint(monitor='val_auc', save_best_only=True),
            EarlyStopping(monitor='val_auc', patience=5),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5)
        ]
    )
    
    # 5. Evaluate
    val_predictions = model.predict(val_dataset)
    val_auc = calculate_auc(val_true, val_predictions)
    
    fold_results.append(val_auc)

# Aggregate results
mean_auc = np.mean(fold_results)
std_auc = np.std(fold_results)
```

#### 8. **Hyperparameter Optimization Integration**

```python
# Keras Tuner searches hyperparameter space
def build_model(hp):
    # Search space
    learning_rate = hp.Float('lr', 1e-5, 1e-2, sampling='LOG')
    dropout_rate = hp.Float('dropout', 0.1, 0.5)
    batch_size = hp.Choice('batch_size', [16, 32, 64])
    optimizer = hp.Choice('optimizer', ['adam', 'adamw', 'rmsprop'])
    
    # Build model with hyperparameters
    model = create_model(dropout_rate=dropout_rate)
    
    # Compile with chosen optimizer
    model.compile(
        optimizer=create_optimizer(optimizer, learning_rate),
        loss='binary_crossentropy',
        metrics=['auc']
    )
    
    return model

# Tuner evaluates different combinations
tuner = kt.RandomSearch(
    build_model,
    objective='val_auc',
    max_trials=50
)

# Best hyperparameters are saved and used for final training
best_hp = tuner.get_best_hyperparameters()[0]
```

#### 9. **Evaluation and Metrics Calculation**

```python
# Custom AUC calculation for multi-label classification
def calculate_auc(y_true, y_pred):
    # Convert logits to probabilities
    y_pred_proba = tf.nn.sigmoid(y_pred)
    
    # Calculate AUC for each class
    aucs = []
    for class_idx in range(206):
        # Skip classes with no positive samples
        if np.sum(y_true[:, class_idx]) > 0:
            class_auc = roc_auc_score(
                y_true[:, class_idx], 
                y_pred_proba[:, class_idx]
            )
            aucs.append(class_auc)
    
    # Average across classes
    return np.mean(aucs)

# Final evaluation across all folds
final_auc = np.mean([fold_auc for fold_auc in fold_results])
```

### Key Pipeline Design Principles

#### 1. **Modularity**
- Each component (Config, Data Pipeline, Model, Trainer) is independent
- Components communicate through well-defined interfaces
- Easy to swap implementations (e.g., different models, optimizers)

#### 2. **Scalability**
- TensorFlow `tf.data` API for efficient data loading
- Distributed training support for multi-GPU
- Mixed precision for memory efficiency
- Pre-computed spectrograms for speed

#### 3. **Reproducibility**
- Fixed random seeds across all components
- Deterministic operations where possible
- Stratified cross-validation for consistent evaluation
- Configuration-driven approach

#### 4. **Robustness**
- Error handling and recovery at each step
- Memory management and cleanup
- Graceful degradation (pre-computed → on-the-fly)
- Comprehensive logging and monitoring

#### 5. **Flexibility**
- Debug mode for fast iteration
- Configurable hyperparameters
- Multiple tuning algorithms
- Different optimization strategies

### Performance Optimization Strategies

#### 1. **Data Loading Optimization**
```python
# Parallel data loading
dataset = dataset.map(load_fn, num_parallel_calls=tf.data.AUTOTUNE)

# Prefetching
dataset = dataset.prefetch(tf.data.AUTOTUNE)

# Batch-level operations
dataset = dataset.batch(batch_size).map(batch_augment)
```

#### 2. **Memory Management**
```python
# Efficient data types
labels = np.zeros((n_samples, n_classes), dtype=np.float32)  # Not float64

# Clear memory between folds
tf.keras.backend.clear_session()
gc.collect()

# Gradient accumulation for large effective batch sizes
```

#### 3. **Mixed Precision Training**
```python
# Enable mixed precision for V100 GPUs
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)

# Model automatically uses FP16 for forward pass, FP32 for loss calculation
```

### Debug Mode vs Production Mode

| Component | Debug Mode | Production Mode |
|-----------|------------|-----------------|
| **Data Samples** | 1,000 | 28,564 |
| **Epochs** | 2 | 20 |
| **Folds** | 1 (fold 0) | 5 (all folds) |
| **HP Trials** | 3 | 50 |
| **Expected Time** | 5-10 minutes | 4-8 hours |
| **Memory Usage** | ~4GB | ~8GB |
| **Purpose** | Fast iteration | Final model |

### Error Handling and Recovery

The pipeline includes comprehensive error handling:

```python
# Graceful degradation
if not os.path.exists(precomputed_spectrograms):
    print("Using on-the-fly processing")
    config.audio.load_precomputed = False

# Continue training even if individual folds fail
for fold in folds:
    try:
        results = train_fold(fold)
    except Exception as e:
        logger.error(f"Fold {fold} failed: {e}")
        continue  # Continue with next fold
    finally:
        cleanup_memory()

# Automatic checkpoint recovery
if os.path.exists(checkpoint_path):
    model.load_weights(checkpoint_path)
    logger.info("Resumed from checkpoint")
```

This comprehensive pipeline ensures reliable, efficient, and reproducible training of bird sound classification models while maintaining flexibility for experimentation and production deployment.

## 🖥️ SLURM Integration and Job Management

### SLURM Scripts Overview

The `slurm_scripts/` directory contains shell scripts that integrate the Python codebase with SLURM (Simple Linux Utility for Resource Management) for efficient cluster job submission and management.

#### Script Architecture

```
slurm_scripts/
├── setup_environment.sh       # One-time environment setup
├── submit_job.sh              # Convenient wrapper for job submission
├── run_hyperparameter_tuning.sh  # Hyperparameter optimization job
└── run_training.sh            # Model training job
```

### Script Detailed Breakdown

#### 1. **setup_environment.sh** - Environment Preparation

```bash
#!/bin/bash
# Purpose: One-time setup of the Python environment and dependencies

# Key functions:
# 1. Check Poetry installation
# 2. Install Python dependencies
# 3. Verify CUDA/GPU availability
# 4. Test configuration loading
# 5. Create necessary directories
# 6. Set script permissions

# Usage:
bash slurm_scripts/setup_environment.sh
```

**What it does:**
- Installs Python dependencies via Poetry
- Verifies GPU availability with `nvidia-smi`
- Tests configuration loading
- Creates output directories (`logs/`, `results/`, etc.)
- Makes scripts executable
- Prints usage examples

#### 2. **submit_job.sh** - Job Submission Wrapper

```bash
#!/bin/bash
# Purpose: Convenient wrapper for submitting different types of jobs

# Key features:
# - Argument parsing for job types (tune, train, setup)
# - Environment variable configuration
# - Automatic resource allocation
# - Job status reporting

# Usage examples:
./slurm_scripts/submit_job.sh setup                    # Environment setup
./slurm_scripts/submit_job.sh --debug tune             # Debug hyperparameter tuning
./slurm_scripts/submit_job.sh --max-trials 30 tune     # Custom hyperparameter tuning
./slurm_scripts/submit_job.sh train                    # Model training
./slurm_scripts/submit_job.sh --fold 0 train           # Single fold training
```

**Supported Arguments:**
- `--debug`: Enable debug mode
- `--tuner-type`: RandomSearch, Hyperband, or Bayesian
- `--max-trials`: Number of hyperparameter trials
- `--epochs`: Number of training epochs
- `--batch-size`: Training batch size
- `--learning-rate`: Learning rate
- `--config`: Custom configuration file
- `--hyperparameters`: Hyperparameters JSON file
- `--fold`: Specific fold to train

#### 3. **run_hyperparameter_tuning.sh** - Hyperparameter Optimization

```bash
#!/bin/bash
#SBATCH --job-name=birdclef_hp_tuning
#SBATCH --partition=free-gpu
#SBATCH --gres=gpu:V100:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/hp_tuning_%j.out
#SBATCH --error=logs/hp_tuning_%j.err

# Environment setup
export PYTHONHASHSEED=42
export TF_DETERMINISTIC_OPS=1
export TF_FORCE_GPU_ALLOW_GROWTH=true

# Load modules
module load cuda/12.2.0
module load python/3.10.2

# Activate environment
source $(poetry env info --path)/bin/activate

# Run hyperparameter tuning
python train_keras_tuner.py \
    --tuner-type $TUNER_TYPE \
    --max-trials $MAX_TRIALS \
    --epochs $EPOCHS \
    ${DEBUG:+--debug} \
    ${CONFIG_FILE:+--config $CONFIG_FILE} \
    ${FOLD:+--fold $FOLD}
```

**Environment Variables:**
- `TUNER_TYPE`: random_search, hyperband, bayesian (default: random_search)
- `MAX_TRIALS`: Maximum trials (default: 50)
- `EPOCHS`: Epochs per trial (default: 10)
- `DEBUG`: Enable debug mode (default: false)
- `CONFIG_FILE`: Custom configuration file
- `FOLD`: Specific fold for tuning

#### 4. **run_training.sh** - Model Training

```bash
#!/bin/bash
#SBATCH --job-name=birdclef_training
#SBATCH --partition=free-gpu
#SBATCH --gres=gpu:V100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/training_%j.out
#SBATCH --error=logs/training_%j.err

# Environment setup (same as hyperparameter tuning)
# ...

# Check for hyperparameters file
HP_FILE=""
if [ -f "results/hyperparameter_tuning_results.json" ]; then
    HP_FILE="results/hyperparameter_tuning_results.json"
fi

# Run training
python train_single_model.py \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --learning-rate $LEARNING_RATE \
    ${DEBUG:+--debug} \
    ${CONFIG_FILE:+--config $CONFIG_FILE} \
    ${HP_FILE:+--hyperparameters $HP_FILE} \
    ${FOLD:+--fold $FOLD}
```

### Python-SLURM Integration

#### How Environment Variables Flow

```
SLURM Script → Environment Variables → Python Arguments → Configuration
     ↓                    ↓                    ↓               ↓
export DEBUG=true → DEBUG=true → --debug → config.debug.debug = True
export EPOCHS=5   → EPOCHS=5   → --epochs 5 → config.training.epochs = 5
```

#### Argument Parsing in Python Scripts

```python
# In train_keras_tuner.py and train_single_model.py
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--tuner-type', choices=['random_search', 'hyperband', 'bayesian'])
    parser.add_argument('--max-trials', type=int, default=50)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--learning-rate', type=float, default=5e-4)
    parser.add_argument('--config', type=str, help='YAML config file')
    parser.add_argument('--hyperparameters', type=str, help='JSON hyperparameters file')
    parser.add_argument('--fold', type=int, help='Specific fold to train')
    return parser.parse_args()

# Configuration override
args = parse_arguments()
if args.debug:
    config.debug.debug = True
    config.apply_debug_settings()
```

### Job Submission Workflow

#### Complete Workflow Example

```bash
# 1. Initial setup (run once)
bash slurm_scripts/setup_environment.sh

# 2. Submit debug hyperparameter tuning
./slurm_scripts/submit_job.sh --debug --max-trials 3 tune

# 3. Monitor job
squeue -u $USER
tail -f logs/hp_tuning_*.out

# 4. Submit production hyperparameter tuning
./slurm_scripts/submit_job.sh --tuner-type bayesian --max-trials 50 tune

# 5. Submit training with best hyperparameters
./slurm_scripts/submit_job.sh train

# 6. Monitor training
tail -f logs/training_*.out
```

#### Resource Management

```bash
# Check resource usage
squeue -u $USER -o "%.8i %.8j %.8T %.10M %.6D %.20S %.20e"

# View job details
scontrol show job <JOB_ID>

# Cancel job if needed
scancel <JOB_ID>

# Check GPU usage
ssh <node> nvidia-smi
```

### Environment Variables Reference

#### Global Environment Variables

```bash
# Reproducibility
export PYTHONHASHSEED=42
export TF_DETERMINISTIC_OPS=1
export TF_CUDNN_DETERMINISTIC=1

# GPU optimization
export TF_FORCE_GPU_ALLOW_GROWTH=true
export CUDA_VISIBLE_DEVICES=0  # Set by SLURM

# CUDA paths
export CUDA_HOME=/opt/apps/cuda/12.2.0
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

#### Job-Specific Environment Variables

| Variable | Default | Purpose | Example |
|----------|---------|---------|---------|
| `DEBUG` | false | Enable debug mode | `DEBUG=true` |
| `TUNER_TYPE` | random_search | Hyperparameter tuner | `TUNER_TYPE=bayesian` |
| `MAX_TRIALS` | 50 | Number of trials | `MAX_TRIALS=30` |
| `EPOCHS` | 10/20 | Training epochs | `EPOCHS=15` |
| `BATCH_SIZE` | 32 | Training batch size | `BATCH_SIZE=64` |
| `LEARNING_RATE` | 5e-4 | Learning rate | `LEARNING_RATE=1e-3` |
| `CONFIG_FILE` | - | Custom config | `CONFIG_FILE=custom.yaml` |
| `HYPERPARAMETERS_FILE` | - | HP file | `HYPERPARAMETERS_FILE=best_hp.json` |
| `FOLD` | - | Specific fold | `FOLD=0` |

### Job Output and Logging

#### Log File Structure

```
logs/
├── hp_tuning_40121751.out      # Hyperparameter tuning stdout
├── hp_tuning_40121751.err      # Hyperparameter tuning stderr
├── training_40121752.out       # Training stdout
├── training_40121752.err       # Training stderr
└── setup_environment.log       # Environment setup log
```

#### Log Content Examples

**Hyperparameter Tuning Log:**
```
Job ID: 40121751
Node: gpu-node-05
GPU: 0
Start time: 2024-01-15 10:30:00

Python version: 3.10.2
TensorFlow version: 2.13.0
Keras Tuner version: 1.4.6
Number of GPUs: 1

Hyperparameter tuning configuration:
  Tuner type: random_search
  Max trials: 50
  Epochs per trial: 10
  Debug mode: false

Trial 1/50
Trial 1 - Learning rate: 0.0003, Dropout: 0.2, Optimizer: adamw
Epoch 1/10: loss: 0.1234, val_loss: 0.1456, val_auc: 0.7890
...
Best trial: Trial 23 with val_auc: 0.8567
```

**Training Log:**
```
Job ID: 40121752
Using hyperparameters from: results/hyperparameter_tuning_results.json

Training configuration:
  Epochs: 20
  Batch size: 32
  Learning rate: 0.0003
  Optimizer: adamw

FOLD 0
Training fold 0...
Epoch 1/20: loss: 0.0987, val_loss: 0.1234, val_auc: 0.8234
...
Fold 0 completed with best val_auc: 0.8567

Cross-validation results:
Mean val AUC: 0.8534 ± 0.0123
```

### Advanced SLURM Features

#### Job Arrays for Parallel Fold Training

```bash
# Submit array job for parallel fold training
sbatch --array=0-4 <<EOF
#!/bin/bash
#SBATCH --job-name=birdclef_fold_array
#SBATCH --partition=free-gpu
#SBATCH --gres=gpu:V100:1
#SBATCH --output=logs/fold_%A_%a.out
#SBATCH --error=logs/fold_%A_%a.err

export FOLD=\$SLURM_ARRAY_TASK_ID
python train_single_model.py --fold \$FOLD --epochs 20
EOF
```

#### Job Dependencies

```bash
# Submit hyperparameter tuning
HP_JOB_ID=$(sbatch --parsable slurm_scripts/run_hyperparameter_tuning.sh)

# Submit training dependent on hyperparameter tuning completion
sbatch --dependency=afterok:$HP_JOB_ID slurm_scripts/run_training.sh
```

#### Resource Monitoring

```bash
# Monitor GPU usage during training
srun --job-name=gpu_monitor --pty watch -n 1 nvidia-smi

# Check job efficiency
seff <JOB_ID>

# Detailed job accounting
sacct -j <JOB_ID> --format=JobID,JobName,Partition,Account,AllocCPUS,State,ExitCode,Start,End,Elapsed,MaxRSS,MaxVMSize
```

### Troubleshooting Common Issues

#### 1. **Module Loading Errors**

```bash
# Problem: Module not found
module load cuda/12.2.0
# Error: Module 'cuda/12.2.0' not found

# Solution: Check available modules
module avail cuda
module load cuda/11.8.0  # Use available version
```

#### 2. **GPU Memory Issues**

```bash
# Problem: Out of memory
# Solution: Enable memory growth
export TF_FORCE_GPU_ALLOW_GROWTH=true

# Or reduce batch size
export BATCH_SIZE=16
```

#### 3. **Poetry Environment Issues**

```bash
# Problem: Poetry environment not found
# Solution: Recreate environment
poetry env remove python
poetry install
```

#### 4. **Job Submission Failures**

```bash
# Problem: Job won't submit
# Solution: Check partition and resources
sinfo                    # Check available partitions
squeue -p free-gpu       # Check queue status
```

### Best Practices for SLURM Usage

#### 1. **Resource Requests**
- **CPU**: Request 4 CPUs for data loading parallelism
- **Memory**: 32GB for full dataset, 16GB for debug mode
- **GPU**: Single V100 is sufficient for this model size
- **Time**: 24 hours for hyperparameter tuning, 12 hours for training

#### 2. **Job Organization**
- Use descriptive job names: `birdclef_debug_tune`, `birdclef_prod_train`
- Organize logs by job type and date
- Use job arrays for parallel fold training
- Set up job dependencies for workflows

#### 3. **Monitoring and Debugging**
- Always check logs with `tail -f`
- Monitor GPU usage with `nvidia-smi`
- Use debug mode for quick iterations
- Save intermediate results frequently

#### 4. **Error Recovery**
- Design jobs to be resumable
- Save checkpoints frequently
- Use graceful error handling
- Implement automatic retry logic

This SLURM integration provides a robust, scalable framework for running machine learning workloads on cluster resources while maintaining reproducibility and efficient resource utilization.

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Run once to set up the environment
cd /pub/ddlin/projects/mids/DATASCI207_Bird_Sounds/notebooks/notebooks_dl/keras_training
bash slurm_scripts/setup_environment.sh
```

### 2. Test the Pipeline

```bash
# Test all components
poetry run python test_pipeline.py
```

### 3. Submit Jobs

#### Hyperparameter Tuning
```bash
# Basic hyperparameter tuning (50 trials, 10 epochs each)
sbatch -p free-gpu --gres=gpu:V100:1 slurm_scripts/run_hyperparameter_tuning.sh

# Quick debug tuning (3 trials, 2 epochs each)
DEBUG=true sbatch -p free-gpu --gres=gpu:V100:1 slurm_scripts/run_hyperparameter_tuning.sh

# Custom tuning parameters
TUNER_TYPE=bayesian MAX_TRIALS=30 EPOCHS=15 sbatch -p free-gpu --gres=gpu:V100:1 slurm_scripts/run_hyperparameter_tuning.sh
```

#### Model Training
```bash
# Train with best hyperparameters (automatically loads from tuning results)
sbatch -p free-gpu --gres=gpu:V100:1 slurm_scripts/run_training.sh

# Debug training (2 epochs, single fold)
DEBUG=true sbatch -p free-gpu --gres=gpu:V100:1 slurm_scripts/run_training.sh

# Train specific fold only
FOLD=0 sbatch -p free-gpu --gres=gpu:V100:1 slurm_scripts/run_training.sh
```

#### Using the Convenience Script
```bash
# Setup environment
./slurm_scripts/submit_job.sh setup

# Quick debug tuning
./slurm_scripts/submit_job.sh --debug --max-trials 5 tune

# Production hyperparameter tuning
./slurm_scripts/submit_job.sh --tuner-type bayesian --max-trials 50 tune

# Train with best hyperparameters
./slurm_scripts/submit_job.sh train

# Train specific fold
./slurm_scripts/submit_job.sh --fold 0 train
```

## ⚙️ Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DEBUG` | `false` | Enable debug mode (fewer samples, epochs) |
| `TUNER_TYPE` | `random_search` | `random_search`, `hyperband`, `bayesian` |
| `MAX_TRIALS` | `50` | Maximum hyperparameter trials |
| `EPOCHS` | `10`/`20` | Epochs per trial/final training |
| `BATCH_SIZE` | `32` | Training batch size |
| `LEARNING_RATE` | `5e-4` | Learning rate |
| `CONFIG_FILE` | - | Path to custom YAML config |
| `HYPERPARAMETERS_FILE` | - | Path to JSON hyperparameters |
| `FOLD` | - | Specific fold to train (0-4) |

### Hyperparameter Search Space

- **Learning Rate**: 1e-5 to 1e-2 (log scale)
- **Batch Size**: [16, 32, 64]
- **Dropout Rate**: 0.1 to 0.5
- **Optimizer**: [Adam, AdamW, RMSprop]
- **Scheduler**: [Cosine, Exponential, Polynomial]
- **Mixup Alpha**: 0.1 to 1.0
- **Augmentation Probability**: 0.2 to 0.8

## 📊 Model Architecture

### BirdCLEF Model Components

1. **Channel Adapter**: Converts 1-channel spectrograms to 3-channel for EfficientNet
2. **Backbone**: EfficientNet-B0 (ImageNet pretrained)
3. **Global Pooling**: Average, Max, or Both
4. **Dropout**: Configurable rate
5. **Classification Head**: 206 bird species (multi-label)

### Model Statistics
- **Total Parameters**: ~4.3M
- **Input Shape**: (256, 256, 1) mel spectrograms
- **Output**: 206-class multi-label probabilities
- **Mixed Precision**: Automatic on V100

## 🔄 Data Pipeline Features

### Data Sources
- **Pre-computed Spectrograms**: Fast loading from NPY file (28,564 samples)
- **On-the-fly Generation**: Dynamic audio processing as fallback
- **Stratified Sampling**: Balanced cross-validation splits

### Augmentations
- **Mixup**: Random sample mixing with configurable alpha
- **Time Masking**: Horizontal spectrogram stripes
- **Frequency Masking**: Vertical spectrogram stripes
- **Brightness/Contrast**: Random intensity adjustments

### Performance Optimizations
- **TensorFlow Data API**: Optimal GPU utilization
- **Prefetching**: Overlap data loading with training
- **Parallel Processing**: Multi-threaded data preparation
- **Memory Management**: Efficient batch handling

## 📈 Training Features

### Cross-Validation
- **5-Fold Stratified**: Balanced splits by bird species
- **Ensemble Ready**: Individual fold models saved
- **Aggregated Metrics**: Mean ± std across folds

### Monitoring
- **TensorBoard**: Real-time training visualization
- **Custom AUC**: Multi-label AUC calculation
- **Model Checkpointing**: Best model saving
- **Early Stopping**: Configurable patience

### Optimization
- **Mixed Precision**: FP16 training on V100
- **Learning Rate Scheduling**: Cosine, Exponential, Polynomial
- **Weight Decay**: AdamW optimization
- **Gradient Clipping**: Stable training

## 📋 Job Management

### Monitor Jobs
```bash
# Check job status
squeue -u $USER

# View job details
scontrol show job <JOB_ID>

# Cancel job
scancel <JOB_ID>
```

### View Results
```bash
# Training logs
tail -f logs/training_<JOB_ID>.out

# Hyperparameter tuning logs
tail -f logs/hp_tuning_<JOB_ID>.out

# TensorBoard
tensorboard --logdir=results/logs/

# Results files
ls -la results/
```

## 🎯 Expected Results

### Hyperparameter Tuning Output
```json
{
  "mean_val_loss": 0.1234,
  "best_overall_hyperparameters": {
    "learning_rate": 0.0003,
    "batch_size": 32,
    "optimizer": "adamw",
    "dropout_rate": 0.3,
    "mixup_alpha": 0.5
  }
}
```

### Cross-Validation Output
```json
{
  "cv_results": {
    "mean_val_auc": 0.85,
    "std_val_auc": 0.02,
    "mean_val_loss": 0.12,
    "std_val_loss": 0.01
  },
  "num_completed_folds": 5
}
```

## 🛠️ Troubleshooting

### Common Issues

1. **Import Errors**: Run `poetry install` to install dependencies
2. **GPU Memory**: Reduce batch size or enable mixed precision
3. **CUDA Errors**: Check GPU availability with `nvidia-smi`
4. **File Not Found**: Verify data paths in config
5. **Permission Denied**: Run `chmod +x slurm_scripts/*.sh`

### Debug Mode

Enable debug mode for faster iteration:
```bash
DEBUG=true ./slurm_scripts/submit_job.sh tune
```

Debug mode changes:
- 1000 samples max
- 2 epochs only
- 3 hyperparameter trials
- Single fold training

### Performance Tips

1. **Pre-computed Spectrograms**: Use for faster training
2. **Mixed Precision**: Enable for V100 efficiency
3. **Batch Size**: Increase if GPU memory allows
4. **Workers**: Adjust `num_workers` based on CPU cores
5. **Prefetch**: Tune `prefetch_buffer_size` for I/O

## 📚 Key Differences from PyTorch

| Aspect | PyTorch | Keras |
|--------|---------|-------|
| Data Loading | DataLoader | tf.data.Dataset |
| Model Definition | nn.Module | tf.keras.Model |
| Mixed Precision | autocast/GradScaler | Policy |
| Optimization | Manual loops | model.fit() |
| Callbacks | Custom | Built-in + Custom |
| Hyperparameter Tuning | Manual | Keras Tuner |

## 🔗 Integration with Original Code

This Keras implementation maintains compatibility with your PyTorch workflow:

- **Same Data**: Uses identical training data and preprocessing
- **Same Metrics**: AUC calculation matches PyTorch version
- **Same Augmentations**: Equivalent mixup and spectrogram masking
- **Same Architecture**: EfficientNet-B0 with similar head design
- **Same Cross-Validation**: 5-fold stratified splits

## 📞 Support

For issues or questions:
1. Check the troubleshooting section
2. Review SLURM job logs
3. Test with debug mode first
4. Verify environment setup

The pipeline is designed for robust, scalable training with systematic hyperparameter optimization. All components are tested and ready for production use on SLURM clusters with V100 GPUs.

#### Quick Start - Hyperparameter Tuning

```bash
# Navigate to the Keras training directory
cd notebooks/notebooks_dl/keras_training

# Submit hyperparameter tuning job
sbatch slurm_scripts/run_hyperparameter_tuning.sh
```

#### Configuration Options

You can customize the training with environment variables:

```bash
# Quick debug run (3 trials, 2 epochs)
export DEBUG=true
sbatch slurm_scripts/run_hyperparameter_tuning.sh

# Production hyperparameter tuning
export TUNER_TYPE="bayesian"
export MAX_TRIALS=50
export EPOCHS=15
sbatch slurm_scripts/run_hyperparameter_tuning.sh

# Run specific fold only
export FOLD=0
sbatch slurm_scripts/run_hyperparameter_tuning.sh
```

#### Monitor Jobs

```bash
# Check job status
squeue -u $USER

# View live logs
tail -f logs/hp_tuning_*.out

# View results
ls -la results/
```

#### Expected Outputs

The hyperparameter tuning produces:
- `results/hyperparameter_tuning_results.json`: Best hyperparameters and validation metrics
- `results/best_hyperparameters_fold_*.json`: Per-fold optimization results
- `logs/hp_tuning_*.out`: Training logs and GPU usage

For detailed documentation, see [`notebooks/notebooks_dl/keras_training/README.md`](notebooks/notebooks_dl/keras_training/README.md)

## Project Structure

- `src/`: source code modules
- `notebooks/`: exploratory and deep learning notebooks
  - `notebooks_dl/keras_training/`: Keras-based training pipeline with SLURM support
- `data/`: raw and processed datasets
- `models/`: trained models and checkpoints
- `logs/`: training logs and metrics

## Contributing

Feel free to open issues or submit pull requests.