"""
TensorFlow data pipeline for BirdCLEF 2025 Keras training.
"""

import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from typing import Tuple, Dict

try:
    # Use relative imports if this is part of a package
    from .config import Config
    from .utils import (
        process_audio_file,
        load_precomputed_spectrograms,
        create_sample_names
    )
except ImportError:
    # Fallback for standalone script execution
    from config import Config
    from utils import (
        process_audio_file,
        load_precomputed_spectrograms,
        create_sample_names
    )


class BirdCLEFDataPipeline:
    """
    TensorFlow data pipeline for BirdCLEF 2025 dataset.
    Handles both pre-computed and on-the-fly spectrogram generation.
    """

    def __init__(self, config: Config):
        """
        Initializes the data pipeline.

        Args:
            config: A configuration object with all necessary parameters.
        """
        self.config = config
        self.spectrograms = None
        self.species_to_idx = {}
        self.idx_to_species = {}
        self.num_classes = 0

        self._load_taxonomy()
        if config.audio.load_precomputed:
            self._load_precomputed_spectrograms()

    def _load_taxonomy(self):
        """Load taxonomy from CSV and create label-to-index mappings."""
        try:
            taxonomy_df = pd.read_csv(self.config.paths.taxonomy_csv)
            # Sort species for consistent mapping across runs
            species_list = sorted(taxonomy_df['primary_label'].unique())
            self.species_to_idx = {species: idx for idx, species in enumerate(species_list)}
            self.idx_to_species = {idx: species for species, idx in self.species_to_idx.items()}
            self.num_classes = len(species_list)
            print(f"Loaded taxonomy with {self.num_classes} species")
        except Exception as e:
            print(f"Error loading taxonomy: {e}")
            raise

    def _load_precomputed_spectrograms(self):
        """Load pre-computed spectrograms from a .npy file if available."""
        if os.path.exists(self.config.paths.spectrogram_npy):
            self.spectrograms = load_precomputed_spectrograms(self.config.paths.spectrogram_npy)
            if self.spectrograms is None:
                print("Failed to load pre-computed spectrograms, will generate on-the-fly.")
                self.config.audio.load_precomputed = False
        else:
            print("Pre-computed spectrograms not found, will generate on-the-fly.")
            self.config.audio.load_precomputed = False

    def prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepares the DataFrame by adding file paths, creating sample names, and
        filtering for debug mode or available pre-computed spectrograms.
        """
        df = df.copy()
        if 'filepath' not in df.columns:
            df['filepath'] = df['filename'].apply(lambda x: os.path.join(self.config.paths.train_audio_dir, x))
        df = create_sample_names(df)

        if self.config.debug.debug:
            df = df.sample(
                min(self.config.debug.max_samples_debug, len(df)),
                random_state=self.config.seed
            ).reset_index(drop=True)
            print(f"Debug mode: Using {len(df)} samples")

        if self.config.audio.load_precomputed and self.spectrograms:
            available_samples = df['samplename'].isin(self.spectrograms.keys())
            if available_samples.sum() < len(df):
                print(f"Warning: Only {available_samples.sum()}/{len(df)} samples have pre-computed spectrograms")
                df = df[available_samples].reset_index(drop=True)
        return df

    def encode_labels(self, df: pd.DataFrame) -> np.ndarray:
        """Converts primary and secondary labels into a multi-hot encoded array."""
        labels = np.zeros((len(df), self.num_classes), dtype=np.float32)
        for i, row in df.iterrows():
            # Encode primary label
            if row['primary_label'] in self.species_to_idx:
                labels[i, self.species_to_idx[row['primary_label']]] = 1.0
            
            # Encode secondary labels
            if 'secondary_labels' in row and pd.notna(row['secondary_labels']):
                try:
                    # Safely evaluate string representation of list
                    secondary_labels = eval(row['secondary_labels']) if isinstance(row['secondary_labels'], str) else row.get('secondary_labels', [])
                    for label in secondary_labels:
                        if label in self.species_to_idx:
                            labels[i, self.species_to_idx[label]] = 1.0
                except (NameError, SyntaxError):
                    # Handle cases where secondary_labels might be malformed
                    pass
        return labels

    def _get_spectrogram_tf(self, sample_name: str, filepath: str) -> tf.Tensor:
        """
        Retrieves a spectrogram, either from the pre-computed dictionary or
        by generating it on-the-fly, wrapped in a tf.py_function.
        """
        def get_precomputed(name_tensor):
            """Helper function to look up a pre-computed spectrogram."""
            return self.spectrograms.get(name_tensor.numpy().decode(), np.zeros(self.config.audio.target_shape, dtype=np.float32))

        def generate_on_the_fly(path_tensor):
            """Helper function to generate a spectrogram from an audio file."""
            return self._process_audio_tf(path_tensor.numpy().decode())

        if self.config.audio.load_precomputed and self.spectrograms is not None:
            spectrogram = tf.py_function(get_precomputed, [sample_name], tf.float32)
        else:
            spectrogram = tf.py_function(generate_on_the_fly, [filepath], tf.float32)

        spectrogram.set_shape(self.config.audio.target_shape)
        return tf.expand_dims(spectrogram, axis=-1)

    def _process_audio_tf(self, audio_path: str) -> np.ndarray:
        """
        Processes a single audio file into a spectrogram. This is a wrapper
        for the utility function to be used within a tf.py_function.
        """
        spec = process_audio_file(
            audio_path,
            sample_rate=self.config.audio.sample_rate,
            target_duration=self.config.audio.target_duration,
            target_shape=self.config.audio.target_shape,
            n_fft=self.config.audio.n_fft,
            hop_length=self.config.audio.hop_length,
            n_mels=self.config.audio.n_mels,
            fmin=self.config.audio.fmin,
            fmax=self.config.audio.fmax
        )
        # Return a zero-array if processing fails
        return spec if spec is not None else np.zeros(self.config.audio.target_shape, dtype=np.float32)

    def _apply_mixup(self, features: tf.Tensor, labels: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Applies MixUp augmentation to a batch of features and labels."""
        # Sample lambda from a Beta distribution
        lam = tf.random.gamma([1], self.config.training.mixup_alpha, self.config.training.mixup_alpha)[0]
        lam = tf.clip_by_value(lam, 0.0, 1.0)
        
        # Shuffle indices to mix with different samples in the batch
        indices = tf.random.shuffle(tf.range(tf.shape(features)[0]))
        
        mixed_features = lam * features + (1 - lam) * tf.gather(features, indices)
        mixed_labels = lam * labels + (1 - lam) * tf.gather(labels, indices)
        return mixed_features, mixed_labels

    def _apply_spectrogram_augmentations(self, spectrogram: tf.Tensor) -> tf.Tensor:
        """Applies SpecAugment - time and frequency masking to spectrograms."""
        # SpecAugment implementation matching PyTorch version
        
        # Time masking (horizontal stripes) - mask time steps
        if tf.random.uniform([]) < 0.5:
            num_masks = tf.random.uniform([], minval=1, maxval=4, dtype=tf.int32)  # 1-3 masks
            for _ in range(3):  # Max 3 iterations
                # Only apply mask if we haven't exceeded num_masks
                if tf.random.uniform([]) < (tf.cast(num_masks, tf.float32) / 3.0):
                    mask_width = tf.random.uniform([], minval=5, maxval=21, dtype=tf.int32)  # 5-20 pixels
                    spec_width = tf.shape(spectrogram)[1]  # time dimension
                    max_start = tf.maximum(0, spec_width - mask_width)
                    start_idx = tf.random.uniform([], minval=0, maxval=max_start + 1, dtype=tf.int32)
                    
                    # Create mask
                    mask = tf.ones_like(spectrogram)
                    indices = tf.range(spec_width)
                    time_mask = tf.logical_or(indices < start_idx, indices >= start_idx + mask_width)
                    time_mask = tf.reshape(time_mask, [1, -1, 1])
                    mask = tf.where(time_mask, mask, tf.zeros_like(mask))
                    spectrogram = spectrogram * mask
        
        # Frequency masking (vertical stripes) - mask frequency bins
        if tf.random.uniform([]) < 0.5:
            num_masks = tf.random.uniform([], minval=1, maxval=4, dtype=tf.int32)  # 1-3 masks
            for _ in range(3):  # Max 3 iterations
                # Only apply mask if we haven't exceeded num_masks
                if tf.random.uniform([]) < (tf.cast(num_masks, tf.float32) / 3.0):
                    mask_height = tf.random.uniform([], minval=5, maxval=21, dtype=tf.int32)  # 5-20 pixels
                    spec_height = tf.shape(spectrogram)[0]  # frequency dimension
                    max_start = tf.maximum(0, spec_height - mask_height)
                    start_idx = tf.random.uniform([], minval=0, maxval=max_start + 1, dtype=tf.int32)
                    
                    # Create mask
                    mask = tf.ones_like(spectrogram)
                    indices = tf.range(spec_height)
                    freq_mask = tf.logical_or(indices < start_idx, indices >= start_idx + mask_height)
                    freq_mask = tf.reshape(freq_mask, [-1, 1, 1])
                    mask = tf.where(freq_mask, mask, tf.zeros_like(mask))
                    spectrogram = spectrogram * mask
        
        # Random brightness/contrast adjustment (matching PyTorch)
        if tf.random.uniform([]) < 0.5:
            gain = tf.random.uniform([], minval=0.8, maxval=1.2)
            bias = tf.random.uniform([], minval=-0.1, maxval=0.1)
            spectrogram = spectrogram * gain + bias
            spectrogram = tf.clip_by_value(spectrogram, 0.0, 1.0)
        
        return spectrogram

    def create_dataset(self, df: pd.DataFrame, is_training: bool = True) -> tf.data.Dataset:
        """
        Creates a complete TensorFlow dataset from a DataFrame. The dataset yields
        tuples of (features, labels) where features is a dictionary.
        """
        df = self.prepare_dataframe(df)
        labels = self.encode_labels(df)

        dataset = tf.data.Dataset.from_tensor_slices({
            "samplename": df['samplename'].values,
            "filepath": df['filepath'].values,
            "label": labels
        })

        if is_training:
            dataset = dataset.shuffle(buffer_size=min(1000, len(df)), seed=self.config.seed)

        def _map_fn(sample):
            """The main mapping function to process each item in the dataset."""
            spectrogram = self._get_spectrogram_tf(sample['samplename'], sample['filepath'])
            if is_training and tf.random.uniform([]) < self.config.training.augmentation_prob:
                spectrogram = self._apply_spectrogram_augmentations(spectrogram)
            
            # Keras models typically expect a (features, labels) tuple.
            # The features can be a dictionary if the model has named inputs.
            features = {'melspec': spectrogram}
            label = sample['label']
            return features, label

        dataset = dataset.map(_map_fn, num_parallel_calls=tf.data.AUTOTUNE)
        
        dataset = dataset.batch(self.config.training.batch_size, drop_remainder=is_training)

        # Apply Mixup to the entire batch after batching
        if is_training and self.config.training.mixup_alpha > 0:
            def _mixup_batch(features, labels):
                mixed_features, mixed_labels = self._apply_mixup(features['melspec'], labels)
                # Return in the same (features, labels) format
                return {'melspec': mixed_features}, mixed_labels

            dataset = dataset.map(_mixup_batch, num_parallel_calls=tf.data.AUTOTUNE)

        return dataset.prefetch(tf.data.AUTOTUNE)

    def create_fold_datasets(self, df: pd.DataFrame, fold: int) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        """Creates training and validation datasets for a specific cross-validation fold."""
        skf = StratifiedKFold(n_splits=self.config.training.n_folds, shuffle=True, random_state=self.config.seed)
        # Stratify by the primary label to ensure balanced folds
        train_idx, val_idx = list(skf.split(df, df['primary_label']))[fold]
        
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)
        
        print(f"Fold {fold}: Train={len(train_df)}, Val={len(val_df)}")
        
        train_dataset = self.create_dataset(train_df, is_training=True)
        val_dataset = self.create_dataset(val_df, is_training=False)
        return train_dataset, val_dataset

    def get_steps_per_epoch(self, dataset_size: int) -> int:
        """Calculate steps per epoch based on dataset size and batch size."""
        return max(1, dataset_size // self.config.training.batch_size)

    def create_inference_dataset(self, df: pd.DataFrame) -> tf.data.Dataset:
        """
        Creates a dataset for inference (no labels, no shuffling, no augmentations).
        The dataset yields dictionaries of features, suitable for `model.predict()`.
        """
        df = self.prepare_dataframe(df)

        dataset = tf.data.Dataset.from_tensor_slices({
            'samplename': df['samplename'].values,
            'filepath': df['filepath'].values
        })

        def load_spectrogram(sample):
            """Mapping function for inference to load only the spectrogram."""
            spectrogram = self._get_spectrogram_tf(sample['sample_name'], sample['filepath'])
            return {'melspec': spectrogram}

        dataset = dataset.map(load_spectrogram, num_parallel_calls=tf.data.AUTOTUNE)
        
        dataset = dataset.batch(self.config.training.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset

    def save_label_mappings(self, save_path: str):
        """Saves the species-to-index mappings to a JSON file."""
        mappings = {
            'species_to_idx': self.species_to_idx,
            'idx_to_species': self.idx_to_species,
            'num_classes': self.num_classes
        }
        with open(save_path, 'w') as f:
            json.dump(mappings, f, indent=4)
        print(f"Label mappings saved to {save_path}")

    def load_label_mappings(self, load_path: str):
        """Loads species-to-index mappings from a JSON file."""
        with open(load_path, 'r') as f:
            mappings = json.load(f)

        self.species_to_idx = mappings['species_to_idx']
        # Convert JSON string keys back to integer keys for idx_to_species
        self.idx_to_species = {int(k): v for k, v in mappings['idx_to_species'].items()}
        self.num_classes = mappings['num_classes']
        print(f"Label mappings loaded from {load_path}")


def create_data_pipeline(config: Config) -> BirdCLEFDataPipeline:
    """Factory function to create the data pipeline."""
    return BirdCLEFDataPipeline(config)


if __name__ == "__main__":
    from config import create_default_config
    print("Testing data pipeline...")

    # Create a default configuration (ensure your config.py is set up)
    config = create_default_config()
    # Enable debug mode to run on a small subset of data
    config.debug.debug = True
    config.training.mixup_alpha = 0.4 # Enable mixup for testing
    
    # Manually set num_classes in config for the test assertions
    # In a real run, this would be set after loading the taxonomy.
    temp_taxonomy_df = pd.read_csv(config.paths.taxonomy_csv)
    config.model.num_classes = len(temp_taxonomy_df['primary_label'].unique())


    pipeline = create_data_pipeline(config)
    
    try:
        train_df = pd.read_csv(config.paths.train_csv)
        # Use a small slice of the data for a quick test
        dataset = pipeline.create_dataset(train_df.head(32), is_training=True)
        
        # Test iteration by taking one batch from the dataset
        for features, labels in dataset.take(1):
            print("\n✅ Data pipeline test successful!")
            print(f"    Features batch type: {type(features)}")
            print(f"    Features dictionary keys: {list(features.keys())}")
            print(f"    Melspec tensor shape: {features['melspec'].shape}")
            print(f"    Labels batch type: {type(labels)}")
            print(f"    Labels tensor shape: {labels.shape}")
            
            # Check if shapes match expectations
            batch_size = features['melspec'].shape[0]
            expected_spec_shape = (batch_size, *config.audio.target_shape, 1)
            # The num_classes in the pipeline is set during its initialization
            expected_label_shape = (batch_size, pipeline.num_classes)
            
            assert features['melspec'].shape == expected_spec_shape, "Spectrogram shape mismatch!"
            assert labels.shape == expected_label_shape, "Labels shape mismatch!"
            print("\n    Shapes are correct.")
            break
            
    except Exception as e:
        print(f"\n❌ Data pipeline test failed: {e}")
        import traceback
        traceback.print_exc()