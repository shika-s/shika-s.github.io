"""
Spectrogram Augmentation and Data Generation Pipeline

This module implements SpecAugment-style data augmentation for spectrograms
along with a flexible data generator supporting class balancing, mixup augmentation,
and sample weighting for audio classification tasks.

Key Features:
- Time and frequency masking (SpecAugment)
- MixUp augmentation for improved generalization
- Class-aware oversampling/downsampling
- Sample weighting for imbalanced datasets

For junior data scientists:
- Use this for generating augmented data batches during training.
- Toggle features via CFG flags (e.g., cfg.use_augmentation, cfg.use_oversampling).
- Example: See data_generator docstring for usage.
"""

import math
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
from scipy import stats
import random
from typing import Dict, Generator, Tuple, Optional, Any
import pandas as pd
import librosa

@tf.keras.utils.register_keras_serializable()
class SpecTimeMask(layers.Layer):
    """
    SpecAugment Time Masking Layer
    
    Applies time masking by zeroing out a random contiguous time window
    along the width axis of spectrograms. This helps the model become
    invariant to temporal variations and reduces overfitting.
    
    Based on: "SpecAugment: A Simple Data Augmentation Method for 
    Automatic Speech Recognition" (Park et al., 2019)
    
    Args:
        max_frac (float): Maximum fraction of time steps to mask (0.0-1.0)
        **kw: Additional keyword arguments for the parent Layer class
        
    Input shape:
        4D tensor: (batch_size, height, width, channels)
        Where width represents the time dimension
        
    Output shape:
        Same as input shape with masked time regions set to zero
        
    Example:
        time_mask = SpecTimeMask(max_frac=0.15)
        augmented_spec = time_mask(spectrogram, training=True)
    """
    
    def __init__(self, max_frac: float = 0.15, **kw):
        super().__init__(**kw)
        self.max_frac = max_frac
        
        # Validate parameters
        if not 0 <= max_frac <= 1:
            raise ValueError(f"max_frac must be between 0 and 1, got {max_frac}")

    def call(self, x: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        """
        Apply time masking during training only.
        
        Args:
            x: Input spectrogram tensor of shape (batch, height, width, channels)
            training: Boolean indicating training mode
            
        Returns:
            Masked spectrogram tensor with same shape as input
        """
        if not training:
            return x
            
        # Get time dimension (width)
        t = tf.shape(x)[2]
        
        # Sample mask fraction uniformly between 0 and max_frac
        mask_frac = tf.random.uniform([], 0, self.max_frac)
        
        # Calculate mask width in time steps
        mask_width = tf.cast(mask_frac * tf.cast(t, tf.float32), tf.int32)
        
        # Sample random start position ensuring mask fits within bounds
        start_pos = tf.random.uniform([], 0, t - mask_width, dtype=tf.int32)
        
        # Create zero mask for the selected time window
        zeros = tf.zeros_like(x[:, :, start_pos:start_pos + mask_width, :])
        
        # Concatenate unmasked regions with masked region
        return tf.concat([
            x[:, :, :start_pos, :],           # Before mask
            zeros,                             # Masked region
            x[:, :, start_pos + mask_width:, :] # After mask
        ], axis=2)

@tf.keras.utils.register_keras_serializable()
class SpecFreqMask(layers.Layer):
    """
    SpecAugment Frequency Masking Layer
    
    Applies frequency masking by zeroing out a random contiguous frequency band
    along the height axis of spectrograms. This encourages the model to not
    over-rely on specific frequency patterns.
    
    Args:
        max_frac (float): Maximum fraction of frequency bins to mask (0.0-1.0)
        **kw: Additional keyword arguments for the parent Layer class
        
    Input shape:
        4D tensor: (batch_size, height, width, channels)
        Where height represents the frequency dimension
        
    Output shape:
        Same as input shape with masked frequency regions set to zero
        
    Example:
        freq_mask = SpecFreqMask(max_frac=0.15)
        augmented_spec = freq_mask(spectrogram, training=True)
    """
    
    def __init__(self, max_frac: float = 0.15, **kw):
        super().__init__(**kw)
        self.max_frac = max_frac
        
        # Validate parameters
        if not 0 <= max_frac <= 1:
            raise ValueError(f"max_frac must be between 0 and 1, got {max_frac}")

    def call(self, x: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        """
        Apply frequency masking during training only.
        
        Args:
            x: Input spectrogram tensor of shape (batch, height, width, channels)
            training: Boolean indicating training mode
            
        Returns:
            Masked spectrogram tensor with same shape as input
        """
        if not training:
            return x
            
        # Get frequency dimension (height)
        h = tf.shape(x)[1]
        
        # Sample mask fraction uniformly between 0 and max_frac
        mask_frac = tf.random.uniform([], 0, self.max_frac)
        
        # Calculate mask height in frequency bins
        mask_height = tf.cast(mask_frac * tf.cast(h, tf.float32), tf.int32)
        
        # Sample random start position ensuring mask fits within bounds
        start_pos = tf.random.uniform([], 0, h - mask_height, dtype=tf.int32)
        
        # Create zero mask for the selected frequency band
        zeros = tf.zeros_like(x[:, start_pos:start_pos + mask_height, :, :])
        
        # Concatenate unmasked regions with masked region
        return tf.concat([
            x[:, :start_pos, :, :],              # Before mask
            zeros,                                # Masked region
            x[:, start_pos + mask_height:, :, :] # After mask
        ], axis=1)


def data_generator(
    df: pd.DataFrame,
    cfg: Any,
    spectrograms: Dict[str, np.ndarray],
    class_weights: Optional[Dict[int, float]] = None,
    is_train: bool = False,
    yield_weight: bool = False,
    batch_size: int = 32,
):
    """
    Yields *batches* rather than single samples.
    If `yield_weight=True`, returns (X, y, sample_weight) where
    `sample_weight.shape == (batch_size,)`.
    """
    if yield_weight and class_weights is None:
        raise ValueError("class_weights required if yield_weight=True")

    # Convenience alias
    H, W = cfg.TARGET_SHAPE

    while True:
        rows = df.sample(frac=1, replace=False).to_dict("records")   # shuffle

        # -------- oversampling (unchanged) ---------------------------------
        if is_train and cfg.use_oversampling and hasattr(cfg, "oversampling_factors"):
            class_bins = {c: [] for c in cfg.oversampling_factors}
            for r in rows:
                class_bins[r["y_species_encoded"]].append(r)
            rows = []
            for c, bucket in class_bins.items():
                factor = cfg.oversampling_factors.get(c, 1)
                if factor > 0 and bucket:
                    rows.extend(random.choices(bucket, k=len(bucket) * factor))
            random.shuffle(rows)
        # -------------------------------------------------------------------

        # Batch buffers
        X_buf, y_buf, w_buf = [], [], []

        for row in rows:
            sample_id = row["samplename"]

            # ---------- load / pad spectrogram -----------------------------
            spec = spectrograms.get(sample_id, np.zeros(cfg.TARGET_SHAPE, np.float32))
            if spec.ndim == 2 and spec.shape[0] != H:
                spec = spec.T
            pad_h, pad_w = max(0, H - spec.shape[0]), max(0, W - spec.shape[1])
            spec = np.pad(spec, ((0, pad_h), (0, pad_w)), mode="constant")[:H, :W]
            if spec.ndim == 2:
                spec = np.expand_dims(spec, -1)           # (H, W, 1)
            # ----------------------------------------------------------------

            label = to_categorical(row["y_species_encoded"], cfg.num_classes)

            # ---------- MixUp ----------------------------------------------
            if (
                is_train
                and cfg.use_augmentation
                and np.random.rand() < cfg.aug_prob
                and cfg.mixup_alpha > 0
            ):
                other = random.choice(rows)
                other_spec = spectrograms.get(
                    other["samplename"], np.zeros(cfg.TARGET_SHAPE, np.float32)
                )
                if other_spec.ndim == 2 and other_spec.shape[0] != H:
                    other_spec = other_spec.T
                other_spec = np.pad(
                    other_spec,
                    ((0, max(0, H - other_spec.shape[0])),
                     (0, max(0, W - other_spec.shape[1]))),
                    mode="constant"
                )[:H, :W]
                if other_spec.ndim == 2:
                    other_spec = np.expand_dims(other_spec, -1)
                other_label = to_categorical(other["y_species_encoded"], cfg.num_classes)

                lam = np.random.beta(cfg.mixup_alpha, cfg.mixup_alpha)
                spec = lam * spec + (1 - lam) * other_spec
                label = lam * label + (1 - lam) * other_label
            # ----------------------------------------------------------------

            X_buf.append(spec)
            y_buf.append(label)
            if yield_weight:
                w_buf.append(class_weights[row["y_species_encoded"]])

            # ------- when buffer full → yield ------------------------------
            if len(X_buf) == batch_size:
                X = np.stack(X_buf).astype(np.float32)
                y = np.stack(y_buf).astype(np.float32)
                if yield_weight:
                    w = np.array(w_buf, dtype=np.float32)
                    yield X, y, w
                else:
                    yield X, y
                X_buf, y_buf, w_buf = [], [], []



# --- NEW: multimodal batch generator ----------------------------------------
def data_generator_with_metadata(
        df: pd.DataFrame,
        cfg: Any,
        spectrograms: Dict[str, np.ndarray],
        metadata_features: Dict[str, np.ndarray],
        class_weights: Optional[Dict[int, float]] = None,
        is_train: bool = False,
        yield_weight: bool = False,
        batch_size: int = 32,
):
    """
    Batched generator that returns ((spec_batch, meta_batch), label_batch [, w]).
    Shape:
        spec_batch  : (B, H, W, C)
        meta_batch  : (B, cfg.metadata_dim)
        label_batch : (B, num_classes)
        w           : (B,) - optional sample-weights
    """
    if yield_weight and class_weights is None:
        raise ValueError("class_weights required if yield_weight=True")

    H, W = cfg.TARGET_SHAPE

    while True:
        # -------- shuffle + (optionally) oversample -------------------------
        rows = df.sample(frac=1, replace=False).to_dict("records")
        if is_train and cfg.use_oversampling and hasattr(cfg, "oversampling_factors"):
            class_bins = {c: [] for c in cfg.oversampling_factors}
            for r in rows:
                class_bins[r["y_species_encoded"]].append(r)
            rows = []
            for c, bucket in class_bins.items():
                k = len(bucket) * cfg.oversampling_factors.get(c, 1)
                if k > 0 and bucket:
                    rows.extend(random.choices(bucket, k=k))
            random.shuffle(rows)
        # -------------------------------------------------------------------

        spec_buf, meta_buf, lbl_buf, w_buf = [], [], [], []

        for r in rows:
            sid = r["samplename"]

            # --- spectrogram ------------------------------------------------
            spec = spectrograms.get(sid, np.zeros(cfg.TARGET_SHAPE, np.float32))
            if spec.ndim == 2 and spec.shape[0] != H:
                spec = spec.T
            pad_h, pad_w = max(0, H - spec.shape[0]), max(0, W - spec.shape[1])
            spec = np.pad(spec, ((0, pad_h), (0, pad_w)), mode="constant")[:H, :W]
            if spec.ndim == 2:
                spec = np.expand_dims(spec, -1)          # (H,W,1)

            # --- metadata vector -------------------------------------------
            meta = metadata_features.get(sid, np.zeros(cfg.metadata_dim, np.float32))

            # --- label ------------------------------------------------------
            lbl = to_categorical(r["y_species_encoded"], cfg.num_classes)

            # --- MixUp ------------------------------------------------------
            if (is_train and cfg.use_augmentation and
                    np.random.rand() < cfg.aug_prob and cfg.mixup_alpha > 0):
                other = random.choice(rows)
                osid = other["samplename"]

                #   spec
                o_spec = spectrograms.get(osid, np.zeros(cfg.TARGET_SHAPE, np.float32))
                if o_spec.ndim == 2 and o_spec.shape[0] != H:
                    o_spec = o_spec.T
                o_spec = np.pad(o_spec,
                                ((0, max(0, H - o_spec.shape[0])),
                                 (0, max(0, W - o_spec.shape[1]))),
                                mode="constant")[:H, :W]
                if o_spec.ndim == 2:
                    o_spec = np.expand_dims(o_spec, -1)

                #   meta
                o_meta = metadata_features.get(osid,
                                               np.zeros(cfg.metadata_dim, np.float32))

                #   label
                o_lbl = to_categorical(other["y_species_encoded"], cfg.num_classes)

                lam = np.random.beta(cfg.mixup_alpha, cfg.mixup_alpha)
                spec = lam * spec + (1 - lam) * o_spec
                meta = lam * meta + (1 - lam) * o_meta
                lbl  = lam * lbl  + (1 - lam) * o_lbl

            # --- accumulate -------------------------------------------------
            spec_buf.append(spec)
            meta_buf.append(meta.astype(np.float32))
            lbl_buf.append(lbl.astype(np.float32))
            if yield_weight:
                w_buf.append(class_weights[r["y_species_encoded"]])

            # --- emit when full --------------------------------------------
            if len(spec_buf) == batch_size:
                X_spec = np.stack(spec_buf)
                X_meta = np.stack(meta_buf)
                y      = np.stack(lbl_buf)
                if yield_weight:
                    sw = np.array(w_buf, np.float32)
                    yield (X_spec, X_meta), y, sw
                else:
                    yield (X_spec, X_meta), y
                spec_buf, meta_buf, lbl_buf, w_buf = [], [], [], []


def get_steps(df, cfg, batch_size=32):
    """
    Estimate batches per epoch after oversampling.
    Works for both train and val splits.
    """
    if cfg.use_oversampling and hasattr(cfg, "oversampling_factors"):
        counts = (
            df["y_species_encoded"]
            .value_counts()
            .to_dict()
        )
        total = sum(
            counts.get(cls, 0) * cfg.oversampling_factors.get(cls, 1)
            for cls in counts
        )
    else:
        total = len(df)

    return math.ceil(total / batch_size)