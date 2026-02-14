"""
Custom neural network components and utilities for deep learning models.

This module provides custom Keras layers, loss functions, and metrics
commonly used in computer vision and classification tasks.

For junior data scientists:
- Use get_vit_model for building the Vision Transformer.
- Custom layers like AddClsToken are for ViT-specific needs.
- Metrics/losses are tailored for imbalanced multi-class problems.
"""
import gc
import numpy as np
from pathlib import Path
from typing import Callable
from typing import Optional, Callable, Dict, Any
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, applications, backend as K
from tensorflow.keras.optimizers.schedules import CosineDecay
import keras_tuner as kt
import keras_cv
import keras_hub
from .data_augmentor_and_generator import SpecTimeMask, SpecFreqMask

@tf.keras.utils.register_keras_serializable()
class AddClsToken(layers.Layer):
    """
    Prepends a learnable [CLS] token to patch sequences for Vision Transformers.
    
    The [CLS] token is commonly used in transformer architectures as a global
    representation that aggregates information from all patches/tokens.
    
    Args:
        dim: Embedding dimension of the input patches
        **kwargs: Additional keyword arguments passed to the parent Layer
        
    Input shape:
        (batch_size, n_patches, embedding_dim)
        
    Output shape:
        (batch_size, n_patches + 1, embedding_dim)
        
    Example:
        >>> cls_layer = AddClsToken(dim=768)
        >>> patches_with_cls = cls_layer(patch_embeddings)  # (32, 196, 768) -> (32, 197, 768)
    """
    
    def __init__(self, dim: int, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        
        # Initialize learnable [CLS] token with zeros
        # Shape: (1, 1, dim) for broadcasting across batch
        self.cls_token = self.add_weight(
            shape=(1, 1, dim),
            initializer="zeros",
            trainable=True,
            name="cls_token"
        )

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """
        Forward pass: prepend [CLS] token to input sequence.
        
        Args:
            x: Input tensor of shape (batch_size, n_patches, dim)
            
        Returns:
            Tensor with [CLS] token prepended: (batch_size, n_patches + 1, dim)
        """
        batch_size = tf.shape(x)[0]
        embedding_dim = tf.shape(x)[-1]
        
        # Broadcast [CLS] token across batch dimension
        cls_tokens = tf.broadcast_to(
            self.cls_token, 
            [batch_size, 1, embedding_dim]
        )
        
        # Concatenate [CLS] token at the beginning of sequence
        return tf.concat([cls_tokens, x], axis=1)

    def get_config(self) -> dict:
        """Return layer configuration for serialization."""
        config = super().get_config()
        config.update({"dim": self.dim})
        return config


@tf.keras.utils.register_keras_serializable()
def macro_f1_fn(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Dynamic macro F1 that automatically detects number of classes.
    
    More flexible version that works with any number of classes.
    Calculates F1 for each class independently and averages them (macro avg).
    
    Args:
        y_true: True labels in one-hot format, shape (batch_size, num_classes)
        y_pred: Predicted probabilities, shape (batch_size, num_classes)
        
    Returns:
        Scalar tensor representing macro F1 score
        
    Example:
        model.compile(metrics=[macro_f1_fn])
    """
    y_true_labels = tf.argmax(y_true, axis=-1)
    y_pred_labels = tf.argmax(y_pred, axis=-1)
    
    num_classes = tf.shape(y_true)[-1]
    
    def compute_class_f1(class_id: tf.Tensor) -> tf.Tensor:
        true_mask = tf.equal(y_true_labels, class_id)
        pred_mask = tf.equal(y_pred_labels, class_id)
        
        tp = tf.reduce_sum(tf.cast(tf.logical_and(true_mask, pred_mask), tf.float32))
        fp = tf.reduce_sum(tf.cast(tf.logical_and(tf.logical_not(true_mask), pred_mask), tf.float32))
        fn = tf.reduce_sum(tf.cast(tf.logical_and(true_mask, tf.logical_not(pred_mask)), tf.float32))
        
        eps = K.epsilon()
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        return 2 * precision * recall / (precision + recall + eps)

    # Use tf.map_fn for dynamic number of classes
    class_indices = tf.range(num_classes, dtype=tf.int64)  # Add dtype=tf.int64 here
    f1_scores = tf.map_fn(
        compute_class_f1, 
        class_indices, 
        fn_output_signature=tf.float32
    )
        
    return tf.reduce_mean(f1_scores)

@tf.keras.utils.register_keras_serializable()
def categorical_focal_loss(
    alpha: float = 0.25, 
    gamma: float = 2.0
) -> Callable[[tf.Tensor, tf.Tensor], tf.Tensor]:
    """
    Create a categorical focal loss function for handling class imbalance.
    
    Focal loss addresses class imbalance by down-weighting easy examples
    and focusing learning on hard negatives. The loss is defined as:
    
    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    
    Where:
    - p_t is the model's estimated probability for the true class
    - α_t is a weighting factor for class t
    - γ (gamma) is the focusing parameter
    
    Args:
        alpha: Weighting factor for rare class (typically 0.25)
               Higher values give more weight to minority classes
        gamma: Focusing parameter (typically 2.0)
               Higher values focus more on hard examples
               
    Returns:
        Compiled loss function that can be used with model.compile()
        
    Example:
        >>> model.compile(
        ...     optimizer='adam',
        ...     loss=categorical_focal_loss(alpha=0.25, gamma=2.0),
        ...     metrics=['accuracy']
        ... )
        
    References:
        Lin, T. Y., et al. "Focal loss for dense object detection." ICCV, 2017.
    """
    
    def focal_loss_fn(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Internal focal loss computation.
        
        Args:
            y_true: Ground truth labels (one-hot encoded)
            y_pred: Predicted class probabilities
            
        Returns:
            Focal loss value
        """
        # Clip predictions to prevent log(0)
        y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
        
        # Standard cross-entropy loss
        cross_entropy = -y_true * K.log(y_pred)
        
        # Focal loss weighting: α * (1 - p_t)^γ
        focal_weight = alpha * y_true * K.pow((1 - y_pred), gamma)
        
        # Apply focal weighting to cross-entropy
        focal_loss = focal_weight * cross_entropy
        
        # Return mean loss across batch and classes
        return K.mean(K.sum(focal_loss, axis=-1))
    
    # Set function name for better debugging/logging
    focal_loss_fn.__name__ = f'categorical_focal_loss_a{alpha}_g{gamma}'
    
    return focal_loss_fn


def build_gpu_augmenter(cfg: Any) -> keras.Sequential:
    """
    Builds a sequential augmentation layer for GPU-accelerated operations.
    
    Includes Gaussian noise, random translation, time/freq masking.
    
    Args:
        cfg: Configuration with noise_std, max_freq_shift, etc.
        
    Returns:
        Sequential model for augmentation.
    """
    return keras.Sequential([
        layers.GaussianNoise(stddev=cfg.noise_std),
        # width = time, height = frequency in your mel-specs
        layers.RandomTranslation(
            height_factor=cfg.max_freq_shift / cfg.TARGET_SHAPE[0],
            width_factor=cfg.max_time_shift / cfg.TARGET_SHAPE[1],
            fill_mode='reflect'),
        SpecTimeMask(max_frac=cfg.max_mask_time),
        SpecFreqMask(max_frac=cfg.max_mask_freq),
    ], name="gpu_augment")




@tf.keras.utils.register_keras_serializable(package="Custom")
class UnfreezeBackbone(keras.callbacks.Callback):
    """
    Unfreeze the backbone after `freeze_epochs` and (optionally) scale the
    optimizer LR by `lr_mult`.

    * Works with constant LR, tf.Variable LR, or any LearningRateSchedule.
    * Avoids the TypeError raised when the optimizer was built with a schedule.
    """
    def __init__(self,
                 freeze_epochs: int = 3,
                 key: str = "vi_t_backbone",
                 lr_mult: float = 0.1,
                 **kwargs):
        super().__init__(**kwargs)
        self.freeze_epochs = int(freeze_epochs)
        self.key           = key.lower()
        self.lr_mult       = float(lr_mult)
        self._done         = False

    # ── helpers ────────────────────────────────────────────────────────────
    def _all_layers(self, layer):
        yield layer
        if isinstance(layer, keras.Model):
            for sub in layer.layers:
                yield from self._all_layers(sub)

    def _find_backbone(self):
        for lyr in self._all_layers(self.model):
            if self.key in lyr.name.lower():
                return lyr
        return None

    # ── main hook ──────────────────────────────────────────────────────────
    def on_epoch_begin(self, epoch, logs=None):
        if self._done or epoch < self.freeze_epochs:
            return

        # 1) Unfreeze backbone
        bb = self._find_backbone()
        if bb is None:
            tf.print(f"[WARN] Backbone containing '{self.key}' not found; skip unfreeze.")
        else:
            for lyr in self._all_layers(bb):
                lyr.trainable = True
            tf.print(f"[INFO] Unfroze backbone with {len(list(self._all_layers(bb)))} sub-layers.")

        # 2) Scale LR safely
        lr = self.model.optimizer.learning_rate
        try:
            if isinstance(lr, (float, int, np.floating)):
                self.model.optimizer.learning_rate = lr * self.lr_mult

            elif isinstance(lr, tf.Variable):
                lr.assign(lr * self.lr_mult)

            elif isinstance(lr, tf.keras.optimizers.schedules.LearningRateSchedule):
                # Wrap once; no TypeError because we use _set_hyper
                scaled = ScaledLearningRateSchedule(lr, self.lr_mult)
                self.model.optimizer._set_hyper("learning_rate", scaled)
                tf.print(f"[INFO] Applied ScaledLearningRateSchedule (×{self.lr_mult}).")

            else:
                tf.print("[WARN] Could not scale LR — unrecognised type:", type(lr))

        except TypeError as e:
            # Fallback: schedule is immutable → just warn
            tf.print("[WARN] Optimiser LR is immutable schedule; skipping LR scaling.", e)

        self._done = True

    # ── serialization ──────────────────────────────────────────────────────
    def get_config(self):
        cfg = super().get_config()
        cfg.update(dict(
            freeze_epochs=self.freeze_epochs,
            key=self.key,
            lr_mult=self.lr_mult,
        ))
        return cfg




def get_cosine_decay_scheduler(
        train_df: pd.DataFrame,  # Assuming this is passed to estimate steps
        cfg: Any
        ) -> CosineDecay:
    """
    Creates a Cosine Decay learning rate scheduler.
    
    Decays from initial LR to min_lr over total steps following a cosine curve.
    No warm restarts (standard annealing).
    
    Args:
        cfg: Configuration with lr, min_lr, epochs (used to estimate steps if steps_per_epoch not provided).
        
    Returns:
        CosineDecay scheduler instance for use in optimizer.
        
    Example (in notebook):
        scheduler = get_cosine_decay_scheduler(cfg)
        optimizer = keras.optimizers.AdamW(learning_rate=scheduler)
    """
    # Estimate total decay steps (adjust if you have exact steps_per_epoch)
    # Here, assuming ~ len(train_df) / batch_size; replace with actual if known
    steps_per_epoch = len(train_df) // cfg.batch_size  # From your notebook; make global or pass
    decay_steps = cfg.epochs * steps_per_epoch
    
    # Alpha: Minimum LR as fraction of initial (decays to min_lr)
    alpha = cfg.min_lr / cfg.lr
    
    return CosineDecay(
        initial_learning_rate=cfg.lr,
        decay_steps=decay_steps,
        alpha=alpha,
        name="cosine_decay_scheduler"
    )


@tf.keras.utils.register_keras_serializable()
class GrayToRGB(layers.Layer):
    """
    Converts grayscale (1-channel) inputs to RGB by repeating the channel dimension.
    
    Input shape: (batch_size, height, width, 1)
    Output shape: (batch_size, height, width, 3)
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def call(self, x: tf.Tensor) -> tf.Tensor:
        return tf.repeat(x, repeats=3, axis=-1)
    
    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (3,)
    
    def get_config(self) -> dict:
        return super().get_config()
    

def get_vit_model(cfg: Any) -> tf.keras.Model:
    """
    Improved ViT/ResCNN classifier: Supports both with adaptive configs and transfer learning.
    
    Key Features:
    - Model type via cfg.model_type: 'ViT' or 'ResCNN'.
    - Transfer learning for both: ViT uses keras_hub ViT, ResCNN uses tf.keras ResNet50.
    - Adaptive patch_size for ViT.
    - Input normalization: Since spectrograms are min-max normed [0,1], extract/summarizes based on the provided instructions.
    - From-scratch paths for both when transfer is disabled.
    """
    input_shape = (*cfg.TARGET_SHAPE, cfg.in_channels)  # e.g., (32, 32, 1)
    inputs = layers.Input(shape=input_shape, name="spec_input")
    x = inputs

    # Light normalization (trainable; adapts even if min-max normed)
    x = layers.Normalization(mean=0.0, variance=1.0, name="spec_norm")(x)

    # Optional grayscale → RGB (for transfer learning compatibility)
    if cfg.use_transfer_learning and cfg.in_channels == 1:
        x = GrayToRGB(name="gray2rgb")(x)

    # Augmentation (unchanged)
    if cfg.use_augmentation:
        x = build_gpu_augmenter(cfg)(x, training=True)

    # Adaptive params based on resolution
    res = cfg.TARGET_SHAPE[0]  # Assume square
    if res <= 32:
        patch_size = 4
    elif res <= 64:
        patch_size = 8
    else:  # 256x256
        patch_size = 16

    if cfg.use_transfer_learning:
        # Shared transfer logic: Resize to common size (224 for standard presets)
        x = layers.Resizing(224, 224, interpolation='bilinear')(x)
        
        if cfg.model_type == 'vit':
            # ViT Transfer: Use tiny preset for small data
            vit_preset = "vit_base_patch16_224_imagenet21k"
            backbone = keras_hub.models.ViTBackbone.from_preset(vit_preset, load_weights=True, key="vit_backbone")
            backbone.trainable = False
            x = backbone(x, training=False)
            # Pooling for ViT
            if len(x.shape) == 3:
                x = layers.GlobalAveragePooling1D(name="gap1d")(x)
            else:
                raise ValueError(f"Unexpected ViT output shape: {x.shape}")
        
        elif cfg.model_type == 'rescnn':
            # ResCNN Transfer: Use ResNet50 or EfficientNetB0 (lightweight, pre-trained on ImageNet)
            backbone = applications.EfficientNetB0(
                include_top=False,
                weights='imagenet',
                pooling=None  # We'll add our own pooling
            )
            backbone.trainable = False
            x = backbone(x, training=False)
            # Pooling for CNN
            x = layers.GlobalAveragePooling2D(name="gap2d")(x)
        
        else:
            raise ValueError(f"Unsupported model_type: {cfg.model_type}")

    else:
        # From-scratch paths
        if cfg.model_type == 'vit':
            proj_dim = 64 if res < 128 else 128
            num_patches = (res // patch_size) ** 2
            
            patches = layers.Conv2D(proj_dim, patch_size, strides=patch_size, padding='same', name="patch_conv")(x)
            patches = layers.Reshape((num_patches, proj_dim), name="patch_flatten")(patches)
            
            x = AddClsToken(proj_dim)(patches)
            pos = tf.range(0, num_patches + 1)
            x = x + layers.Embedding(input_dim=num_patches + 1, output_dim=proj_dim, name="pos_embed")(pos)
            
            num_blocks = 3 if res < 128 else 4
            for _ in range(num_blocks):
                attn = layers.MultiHeadAttention(num_heads=4, key_dim=proj_dim // 4)(x, x)
                attn = layers.Dropout(cfg.dropout_rate)(attn)
                x = layers.LayerNormalization(epsilon=1e-6)(x + attn)
                
                mlp = layers.Dense(128 if res < 128 else 256, activation="gelu")(x)
                mlp = layers.Dense(proj_dim)(mlp)
                mlp = layers.Dropout(cfg.dropout_rate)(mlp)
                x = layers.LayerNormalization(epsilon=1e-6)(x + mlp)
            
            x = layers.LayerNormalization(epsilon=1e-6)(x)[:, 0, :]

        elif cfg.model_type == 'rescnn':
            # From-scratch ResCNN: Fixed with shortcut projection
            def res_block(y, filters):
                shortcut = y
                input_channels = shortcut.shape[-1]  # Use shape for dynamic
                y = layers.Conv2D(filters, 3, padding='same', activation='relu')(y)
                y = layers.Conv2D(filters, 3, padding='same')(y)
                y = layers.BatchNormalization()(y)
                if input_channels != filters:
                    shortcut = layers.Conv2D(filters, 1, padding='same')(shortcut)
                y = layers.Add()([shortcut, y])
                y = layers.Activation('relu')(y)
                return y
            
            x = layers.Conv2D(32, 3, padding='same', activation='relu')(x)
            x = layers.MaxPooling2D()(x)
            x = res_block(x, 64)
            x = layers.MaxPooling2D()(x)
            x = res_block(x, 128)
            x = layers.GlobalAveragePooling2D()(x)
            x = layers.Dropout(cfg.dropout_rate)(x)  # Consistent dropout

        else:
            raise ValueError(f"Unsupported model_type: {cfg.model_type}")

    # Classifier head
    x = layers.Dropout(cfg.dropout_rate)(x)
    outputs = layers.Dense(cfg.num_classes, activation="softmax")(x)
    
    model = models.Model(inputs, outputs)
    return model


def get_multimodal_vit_model(cfg: Any) -> tf.keras.Model:
    """
    Vision-+-tabular hybrid.
    Returns
        model(inputs=[spec_input, meta_input]) -> softmax(num_classes)
    """
    # ---------- Vision branch ----------------------------------------------
    vit_base = get_vit_model(cfg)          # full ViT (includes head)
    vit_feat = vit_base.layers[-2].output  # features right before Dense

    vision_branch = models.Model(
        inputs=vit_base.input,
        outputs=vit_feat,
        name="vision_branch"
    )

    spec_input = vision_branch.input
    spec_out   = vision_branch(spec_input)

    # ---------- Metadata branch --------------------------------------------
    meta_input = layers.Input(shape=(cfg.metadata_dim,), name="meta_input")
    x = layers.Dense(128, activation="relu")(meta_input)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    meta_out = layers.Dense(64, activation="relu")(x)

    # ---------- Fusion & head ----------------------------------------------
    x = layers.Concatenate(name="fusion_layer")([spec_out, meta_out])
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    logits = layers.Dense(cfg.num_classes, activation="softmax",
                          name="final_output")(x)

    return models.Model(
        inputs=[spec_input, meta_input],
        outputs=logits,
        name="multimodal_vit"
    )


class SafeCheckpointTuner(kt.Hyperband):
    """Hyperband that (1) clears sessions each trial, (2) guarantees a checkpoint."""
    def run_trial(self, trial, *fit_args, **fit_kwargs):
        # ---- 3.1 clear memory from previous trial --------------------------------
        tf.keras.backend.clear_session()
        gc.collect()

        # ---- 3.2 attach ModelCheckpoint ------------------------------------------
        trial_dir = Path(self.project_dir) / trial.trial_id
        trial_dir.mkdir(parents=True, exist_ok=True)
        ckpt_cb = tf.keras.callbacks.ModelCheckpoint(
            filepath=str(trial_dir / "checkpoint.weights.h5"),
            save_weights_only=True,
            save_best_only=True,
            monitor="val_macro_f1_fn",
            mode="max",
            verbose=0,
        )
        fit_kwargs.setdefault("callbacks", []).append(ckpt_cb)

        # ---- 3.3 run the actual fit ----------------------------------------------
        return super().run_trial(trial, *fit_args, **fit_kwargs)
    

from tensorflow.keras import layers, models

def get_multimodal_vit_model_softattn(cfg) -> tf.keras.Model:
    """
    Vision-plus-tabular hybrid with *soft attention* over the metadata vector.

    Returns
        model(inputs=[spec_input, meta_input]) → softmax(num_classes)
    """

    # ------------------------------------------------------------------
    # 1. Vision branch (ViT backbone → feature vector)
    # ------------------------------------------------------------------
    vit_base  = get_vit_model(cfg)                  # full ViT (includes head)
    vit_feat  = vit_base.layers[-2].output          # features just before Dense
    vision_branch = models.Model(
        inputs  = vit_base.input,
        outputs = vit_feat,
        name    = "vision_branch",
    )

    spec_input = vision_branch.input               # (time, freq, channels)
    spec_out   = vision_branch(spec_input)         # (embed_dim,)

    # ------------------------------------------------------------------
    # 2. Metadata branch with soft attention
    # ------------------------------------------------------------------
    meta_input = layers.Input(shape=(cfg.metadata_dim,), name="meta_input")

    # ---- soft attention weights --------------------------------------
    # Produces a weight α_i for each raw feature m_i; softmax gives Σ α_i = 1
    attn_logits = layers.Dense(cfg.metadata_dim, use_bias=True,
                               name="meta_attn_dense")(meta_input)
    attn_scores = layers.Activation("softmax", name="meta_attn_softmax")(attn_logits)

    # ---- weighted (attended) metadata vector --------------------------
    attended_meta = layers.Multiply(name="meta_weighted")([meta_input, attn_scores])
    # Optional reduction (sums to a single scalar)—commented out here
    # attended_meta = layers.Lambda(lambda x: tf.reduce_sum(x, axis=-1, keepdims=True))(weighted_meta)

    # ---- downstream processing ----------------------------------------
    x = layers.Dense(128, activation="relu")(attended_meta)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    meta_out = layers.Dense(64, activation="relu")(x)          # final meta embedding

    # ------------------------------------------------------------------
    # 3. Fusion & classification head
    # ------------------------------------------------------------------
    fusion = layers.Concatenate(name="fusion_layer")([spec_out, meta_out])
    x = layers.Dense(256, activation="relu")(fusion)
    x = layers.Dropout(0.5)(x)
    logits = layers.Dense(cfg.num_classes, activation="softmax",
                          name="final_output")(x)

    # ------------------------------------------------------------------
    # 4. Build
    # ------------------------------------------------------------------
    return models.Model(
        inputs=[spec_input, meta_input],
        outputs=logits,
        name="multimodal_vit_softattn",
    )
