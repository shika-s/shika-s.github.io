"""
Data processing utilities for audio classification pipeline.
This module contains functions for loading, processing, and analyzing
audio classification datasets with class imbalance handling.
For junior data scientists:
- Start with main_data_processing_pipeline for end-to-end data prep.
- Use compute_oversampling_factors to handle imbalance.
- Always pass cfg for configuration.
"""
import pandas as pd
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from typing import Dict, Optional, Tuple, Callable, Any
from sklearn.preprocessing import StandardScaler


def load_and_prepare_data(cfg: Any) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and split data into train/val and test sets.
    
    Args:
        cfg: Configuration object with working_df_path.
    
    Returns:
        tuple: (combined_train_val_df, test_df)
        
    Example:
        train_val_df, test_df = load_and_prepare_data(cfg)
    """
    print("\nLoading full dataset...")
    full_df = pd.read_csv(cfg.working_df_path)
    if 'filename' in full_df.columns and 'samplename' not in full_df.columns:
        full_df['samplename'] = full_df['filename'].str.replace('.ogg', '', regex=False).str.replace('/', '-')
    
    # Filter for train/val splits
    combined_train_val_df = (
        full_df
        .loc[full_df['split'].isin(['train', 'val'])]
        .reset_index(drop=True)
    )
    
    # Filter for test split
    test_df = (
        full_df
        .loc[full_df['split'] == 'test']
        .reset_index(drop=True)
    )
    
    print(f"Train/Val samples: {len(combined_train_val_df):,}")
    print(f"Test samples: {len(test_df):,}")
    
    return combined_train_val_df, test_df



def preprocess_metadata(df: pd.DataFrame, cfg: Any) -> Dict[str, np.ndarray]:
    """
    Preprocesses tabular metadata into numerical feature vectors.

    - Converts latitude and longitude to 3D Cartesian coordinates.
    - Scales the new x, y, z coordinates.
    - Converts boolean flags to float32.
    - One-hot encodes the 'category' column.

    Args:
        df: DataFrame containing the metadata.
        cfg: Configuration object.

    Returns:
        A dictionary mapping 'samplename' to its processed metadata vector.
    """
    print("\nPreprocessing metadata with 3D coordinate conversion...")
    
    # Create a copy to avoid modifying the original DataFrame
    df_processed = df.copy()

    # 1. Convert Latitude/Longitude to 3D Cartesian (Geocentric) coordinates
    # First, convert degrees to radians
    lon_rad = np.deg2rad(df_processed['longitude'])
    lat_rad = np.deg2rad(df_processed['latitude'])

    # Apply the conversion formulas
    df_processed['x'] = np.cos(lat_rad) * np.cos(lon_rad)
    df_processed['y'] = np.cos(lat_rad) * np.sin(lon_rad)
    df_processed['z'] = np.sin(lat_rad)

    # Define new numerical, boolean, and categorical columns
    # We now use x, y, z instead of latitude, longitude
    numerical_cols = ['x', 'y', 'z']
    boolean_cols = [
        'has_uncertain', 'song/canto', 'call', 'uncertain', 
        'mating/groups', 'hatching', 'immitation', 'noise/drum'
    ]
    categorical_cols = ['category']

    # 2. Scale Numerical Features (the new x, y, z)
    scaler = StandardScaler()
    df_processed[numerical_cols] = scaler.fit_transform(df_processed[numerical_cols])

    # 3. Convert Boolean Features to Float
    for col in boolean_cols:
        df_processed[col] = df_processed[col].astype(np.float32)

    # 4. One-Hot Encode Categorical Features
    df_processed = pd.get_dummies(df_processed, columns=categorical_cols, prefix='cat')

    # Combine all feature columns into a list
    feature_cols = numerical_cols + boolean_cols + [col for col in df_processed.columns if col.startswith('cat_')]
    
    # Update CFG with the metadata dimension
    cfg.metadata_dim = len(feature_cols)
    print(f"Metadata feature dimension: {cfg.metadata_dim}")

    # Create the dictionary mapping samplename -> feature vector
    metadata_features = {}
    for _, row in df_processed.iterrows():
        metadata_vector = row[feature_cols].values.astype(np.float32)
        metadata_features[row['samplename']] = metadata_vector
        
    return metadata_features


def compute_oversampling_factors(
    df: pd.DataFrame,
    target_col: str = 'y_species_encoded',
    max_factor: int = 50,
    majority_class: int = 1
) -> Dict[int, int]:
    """
    Compute oversampling factors for class balancing.
    
    Caps max oversampling at `max_factor` to avoid extreme repetition.
    Forces downsampling for majority class if specified.
    
    Args:
        df: DataFrame containing the data
        target_col: Name of the target column
        max_factor: Maximum oversampling factor to prevent extreme oversampling
        majority_class: Class to downsample (typically the majority class)
    
    Returns:
        Dictionary mapping class labels to oversampling factors
        
    Example:
        factors = compute_oversampling_factors(train_df)
        # {0: 47, 1: 1, 2: 50, 3: 50}
    """
    class_counts = df[target_col].value_counts().to_dict()
    print(f"Class counts in train/val: {class_counts}")
    
    if not class_counts:
        raise ValueError("No class counts found in DataFrame.")
    
    max_count = max(class_counts.values())
    
    oversampling_factors = {
        cls: min(round(max_count / count), max_factor) if count > 0 else 0
        for cls, count in class_counts.items()
    }
    
    # Force downsampling of majority class (typically Aves)
    oversampling_factors[majority_class] = 1
    
    print(f"Computed oversampling factors: {oversampling_factors}")
    return oversampling_factors


def analyze_class_distribution(
    df: pd.DataFrame,
    target_col: str = 'y_species_encoded'
) -> None:
    """Analyze and print class distribution statistics.
    
    Args:
        df: DataFrame to analyze
        target_col: Column with encoded labels
        
    Example:
        analyze_class_distribution(train_df)
    """
    class_counts = df[target_col].value_counts().sort_index()
    total = class_counts.sum()
    
    class_mapping = {
        0: 'Amphibia',
        1: 'Aves',
        2: 'Insecta',
        3: 'Mammalia'
    }
    
    print(f"\nFull dataset class distribution:")
    for idx, count in class_counts.items():
        class_name = class_mapping.get(idx, f"Class_{idx}")
        percentage = (count / total) * 100
        print(f" {idx}: {class_name} - {count:,} samples ({percentage:.1f}%)")
    
    # Highlight imbalance if Aves is majority
    if 1 in class_counts:
        aves_ratio = class_counts[1] / total
        print(f"Imbalance ratio: Aves / Total = {aves_ratio:.2%}")


def compute_class_weights(
    df: pd.DataFrame,
    target_col: str = 'y_species_encoded',
    use_class_weights: bool = True,
    num_classes: int = 4
) -> Dict[int, float]:
    """
    Compute class weights for handling class imbalance.
    
    Args:
        df: DataFrame containing the data
        target_col: Name of the target column
        use_class_weights: Whether to compute balanced weights or use uniform weights
        num_classes: Total number of classes
    
    Returns:
        Dictionary mapping class labels to weights
        
    Example:
        weights = compute_class_weights(train_df)
    """
    if use_class_weights:
        print("\nComputing balanced class weights...")
        classes = np.unique(df[target_col])
        labels = df[target_col].values
        weights = compute_class_weight('balanced', classes=classes, y=labels)
        class_weights_dict = dict(zip(classes, weights))
    else:
        print("\nUsing uniform class weights.")
        class_weights_dict = {i: 1.0 for i in range(num_classes)}
    
    print(f"Final class weights: {class_weights_dict}")
    return class_weights_dict


def inspect_data_generator(
    df: pd.DataFrame,
    cfg: Any,
    spectrograms: Dict[str, np.ndarray],
    class_weights: Optional[Dict[int, float]] = None,
    data_generator_func: Callable = None
) -> None:
    """Inspect one batch from the data generator for sanity checking.
    
    Args:
        df: Full DataFrame
        cfg: Configuration
        spectrograms: Spectrogram dict
        class_weights: Optional weights
        data_generator_func: The generator function
        
    Example:
        inspect_data_generator(train_df, cfg, spectrograms, weights)
    """
    if data_generator_func is None:
        raise ValueError("data_generator_func must be provided.")
    
    print("\nInspecting one batch from the data generator:")
    
    train_df = df[df['split'] == 'train']
    
    gen = data_generator_func(
        train_df,
        cfg,
        spectrograms,
        class_weights=class_weights if cfg.use_class_weights else None,
        is_train=True,
        yield_weight=False
    )
    
    batch_specs, batch_labels = next(gen)
    print(f"Batch shape: {batch_specs.shape}")
    print(f"Example batch labels (one-hot): {batch_labels}")
    print("Class order: [Amphibia, Aves, Insecta, Mammalia]")


def main_data_processing_pipeline(
    cfg: Any,
    spectrograms: Dict[str, np.ndarray],
    data_generator_func: Callable
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[int, float], Dict[str, np.ndarray]]:
    """
    Main pipeline for data processing and analysis.
    
    Runs loading, oversampling computation, distribution analysis,
    class weights, and generator inspection.
    
    Args:
        cfg: Configuration object
        spectrograms: Preloaded spectrogram data
        data_generator_func: Data generator function (e.g., from data_augmentor_and_generator)
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, Dict[int, float], Dict[str, np.ndarray]]: 
            (combined_train_val_df, test_df, class_weights_dict, filtered_spectrograms)
            
    Example:
        train_val, test, weights, filtered_specs = main_data_processing_pipeline(cfg, specs, data_generator)
    """
    
    # 1. Load and prepare data
    combined_train_val_df, test_df = load_and_prepare_data(cfg)
    
    # Filter spectrograms to only include train/val/test samples
    needed_keys = set(combined_train_val_df['samplename'].tolist() + test_df['samplename'].tolist())
    filtered_spectrograms = {k: spectrograms[k] for k in needed_keys if k in spectrograms}
    print(f"Filtered spectrograms to {len(filtered_spectrograms)} entries")
    
    # 2. Compute oversampling factors
    oversampling_factors = compute_oversampling_factors(combined_train_val_df)
    cfg.oversampling_factors = oversampling_factors # Update cfg dynamically
    
    # 3. Analyze class distribution
    analyze_class_distribution(combined_train_val_df)
    
    # 4. Compute class weights
    class_weights_dict = compute_class_weights(
        combined_train_val_df,
        use_class_weights=cfg.use_class_weights,
        num_classes=cfg.num_classes
    )
    
    # 5. Sanity check with data generator
    inspect_data_generator(
        combined_train_val_df,
        cfg,
        filtered_spectrograms,
        class_weights_dict,
        data_generator_func
    )
    
    return combined_train_val_df, test_df, class_weights_dict, filtered_spectrograms