"""
Model definitions for BirdCLEF 2025 Keras training.
"""

import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers, callbacks
from tensorflow.keras.applications import EfficientNetB0, EfficientNetB1, EfficientNetB2
import keras_tuner as kt
from typing import Dict, Any, Optional, Tuple, Union
import numpy as np
from sklearn.metrics import roc_auc_score

try:
    from .config import Config
except ImportError:
    from config import Config


class ClassWiseAUC(tf.keras.metrics.Metric):
    """
    Custom AUC metric that calculates per-class AUC and averages them,
    matching the PyTorch implementation behavior.
    """
    
    def __init__(self, num_classes: int, name='class_wise_auc', **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.true_positives = self.add_weight('true_positives', initializer='zeros')
        self.true_negatives = self.add_weight('true_negatives', initializer='zeros')
        self.false_positives = self.add_weight('false_positives', initializer='zeros')
        self.false_negatives = self.add_weight('false_negatives', initializer='zeros')
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        """Update metric state - store predictions for final calculation."""
        # Convert logits to probabilities
        y_pred = tf.nn.sigmoid(y_pred)
        
        # Store for final calculation
        if not hasattr(self, '_y_true_list'):
            self._y_true_list = []
            self._y_pred_list = []
        
        self._y_true_list.append(y_true)
        self._y_pred_list.append(y_pred)
        
        # Dummy update for consistency
        self.true_positives.assign_add(0.0)
    
    def result(self):
        """Calculate the final AUC result matching PyTorch implementation."""
        if not hasattr(self, '_y_true_list') or len(self._y_true_list) == 0:
            return 0.0
            
        # Concatenate all predictions and targets
        y_true = tf.concat(self._y_true_list, axis=0)
        y_pred = tf.concat(self._y_pred_list, axis=0)
        
        # Convert to numpy for sklearn calculation
        y_true_np = y_true.numpy()
        y_pred_np = y_pred.numpy()
        
        aucs = []
        for i in range(self.num_classes):
            # Skip classes with no positive samples (matches PyTorch logic)
            if np.sum(y_true_np[:, i]) > 0:
                try:
                    class_auc = roc_auc_score(y_true_np[:, i], y_pred_np[:, i])
                    aucs.append(class_auc)
                except ValueError:
                    # Handle edge cases where AUC cannot be computed
                    continue
        
        return np.mean(aucs) if aucs else 0.0
    
    def reset_state(self):
        """Reset metric state."""
        self.true_positives.assign(0.0)
        self.true_negatives.assign(0.0)
        self.false_positives.assign(0.0)
        self.false_negatives.assign(0.0)
        self._y_true_list = []
        self._y_pred_list = []


class BirdCLEFModel(tf.keras.Model):
    """
    BirdCLEF model with EfficientNet backbone and custom classification head.
    Supports mixup training and different pooling strategies.
    """

    def __init__(
        self,
        config: Config,
        model_name: str = 'efficientnet_b0',
        dropout_rate: float = 0.2,
        num_classes: Optional[int] = None,
        pooling: str = 'avg',
        use_mixup: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.config = config
        self.model_name = model_name
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes or config.model.num_classes
        self.pooling = pooling
        self.use_mixup = use_mixup

        # Build model components
        self._build_backbone()
        self._build_head()

    def _build_backbone(self):
        """Build the backbone model - simplified approach matching PyTorch timm."""
        model_map = {
            'efficientnet_b0': EfficientNetB0,
            'efficientnet_b1': EfficientNetB1,
            'efficientnet_b2': EfficientNetB2,
        }
        if self.model_name not in model_map:
            raise ValueError(f"Model {self.model_name} not supported. Available: {list(model_map.keys())}")

        backbone_fn = model_map[self.model_name]
        
        # Simple direct creation - let Keras handle channel adaptation automatically
        # This matches the PyTorch timm approach: timm.create_model(..., in_chans=1)
        self.backbone = backbone_fn(
            weights='imagenet',
            include_top=False,
            input_shape=(*self.config.audio.target_shape, 1),  # 1 channel for spectrograms
            pooling=None
        )
        
        self.backbone.trainable = True
        if hasattr(self.config.model, 'freeze_layers') and self.config.model.freeze_layers > 0:
            for layer in self.backbone.layers[:self.config.model.freeze_layers]:
                layer.trainable = False

    def _build_head(self):
        """Build the classification head."""
        # Simplified to only use average pooling like PyTorch
        self.global_pool = layers.GlobalAveragePooling2D(name='global_avg_pool')

        self.dropout = layers.Dropout(self.dropout_rate, name='head_dropout')
        self.classifier = layers.Dense(
            self.num_classes,
            activation=None,  # Using from_logits=True in the loss function
            name='classifier'
        )

    def call(self, inputs, training=None):
        """Defines the forward pass of the model."""
        # Handle dictionary input if necessary
        x = inputs['melspec'] if isinstance(inputs, dict) else inputs

        # 1. Pass through the backbone (now accepts 1-channel input directly)
        x = self.backbone(x, training=training)

        # 2. Apply pooling (simplified to average pooling only)
        x = self.global_pool(x)

        # 3. Apply dropout
        x = self.dropout(x, training=training)

        # 4. Get predictions (logits)
        return self.classifier(x)

    def get_config(self):
        """Get model configuration for saving."""
        config = super().get_config()
        config.update({
            'model_name': self.model_name,
            'dropout_rate': self.dropout_rate,
            'num_classes': self.num_classes,
            'pooling': self.pooling,
            'use_mixup': self.use_mixup
        })
        return config


def create_model(
    config: Config,
    hp: Optional[kt.HyperParameters] = None,
    **kwargs
) -> tf.keras.Model:
    """Creates, compiles, and returns the BirdCLEF model."""
    if hp:
        model_name = 'efficientnet_b0'  # Fixed to B0 to match PyTorch
        dropout_rate = hp.Float('dropout_rate', config.hyperparams.dropout_min, config.hyperparams.dropout_max)
        pooling = 'avg'  # Fixed to average pooling to match PyTorch
        learning_rate = hp.Float('learning_rate', config.hyperparams.lr_min, config.hyperparams.lr_max, sampling='LOG')
        optimizer_name = hp.Choice('optimizer', config.hyperparams.optimizers, default='adamw')
    else:
        model_name = kwargs.get('model_name', config.model.model_name)
        dropout_rate = kwargs.get('dropout_rate', config.model.dropout_rate)
        pooling = 'avg'  # Fixed to average pooling to match PyTorch
        learning_rate = kwargs.get('learning_rate', config.training.learning_rate)
        optimizer_name = kwargs.get('optimizer', config.training.optimizer)

    model = BirdCLEFModel(
        config=config,
        model_name=model_name,
        dropout_rate=dropout_rate,
        pooling=pooling,
        **kwargs
    )

    # Keras will build the model automatically on the first call to fit() or predict().
    optimizer = create_optimizer(optimizer_name, learning_rate, config)

    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[ClassWiseAUC(config.model.num_classes)] # Custom AUC matching PyTorch
    )

    return model


def create_optimizer(
    optimizer_name: str,
    learning_rate: float,
    config: Config
) -> tf.keras.optimizers.Optimizer:
    """Create optimizer based on name and configuration."""
    weight_decay = config.training.weight_decay

    if optimizer_name.lower() == 'adam':
        return optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_name.lower() == 'adamw':
        return optimizers.AdamW(learning_rate=learning_rate, weight_decay=weight_decay)
    elif optimizer_name.lower() == 'rmsprop':
        return optimizers.RMSprop(learning_rate=learning_rate)
    else:
        raise ValueError(f"Optimizer {optimizer_name} not supported")


def create_lr_scheduler(
    scheduler_name: str,
    optimizer: tf.keras.optimizers.Optimizer,
    config: Config,
    steps_per_epoch: int,
    hp: Optional[kt.HyperParameters] = None
) -> Optional[tf.keras.callbacks.Callback]:
    """
    Create learning rate scheduler.
    
    Args:
        scheduler_name: Name of scheduler
        optimizer: Optimizer to apply scheduler to
        config: Configuration object
        steps_per_epoch: Steps per epoch
        hp: Keras Tuner hyperparameters
        
    Returns:
        Learning rate scheduler callback or None
    """
    if scheduler_name.lower() == 'cosine':
        return callbacks.CosineRestartDecay(
            initial_learning_rate=config.training.learning_rate,
            first_decay_steps=config.training.epochs * steps_per_epoch,
            t_mul=1.0,
            m_mul=1.0,
            alpha=config.training.min_lr / config.training.learning_rate
        )
    
    elif scheduler_name.lower() == 'exponential':
        decay_rate = 0.9 if hp is None else hp.Float('decay_rate', 0.8, 0.95)
        decay_steps = steps_per_epoch
        
        lr_schedule = optimizers.schedules.ExponentialDecay(
            initial_learning_rate=config.training.learning_rate,
            decay_steps=decay_steps,
            decay_rate=decay_rate,
            staircase=False
        )
        
        # Update optimizer with new schedule
        optimizer.learning_rate = lr_schedule
        return None
    
    elif scheduler_name.lower() == 'polynomial':
        end_learning_rate = config.training.min_lr
        decay_steps = config.training.epochs * steps_per_epoch
        power = 1.0 if hp is None else hp.Float('poly_power', 0.5, 2.0)
        
        lr_schedule = optimizers.schedules.PolynomialDecay(
            initial_learning_rate=config.training.learning_rate,
            decay_steps=decay_steps,
            end_learning_rate=end_learning_rate,
            power=power
        )
        
        # Update optimizer with new schedule
        optimizer.learning_rate = lr_schedule
        return None
    
    elif scheduler_name.lower() == 'reduce_on_plateau':
        return callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=config.training.min_lr,
            verbose=1
        )
    
    else:
        return None


def create_callbacks(
    config: Config,
    model_path: str,
    log_dir: str,
    validation_data: Optional[tf.data.Dataset] = None,
    hp: Optional[kt.HyperParameters] = None
) -> list:
    """Create training callbacks."""
    callbacks_list = []

    callbacks_list.append(callbacks.ModelCheckpoint(
        filepath=model_path,
        monitor=config.training.monitor_metric,
        mode='max' if 'auc' in config.training.monitor_metric.lower() else 'min',
        save_best_only=config.training.save_best_only,
        save_weights_only=False,
        verbose=1
    ))

    callbacks_list.append(callbacks.EarlyStopping(
        monitor=config.training.monitor_metric,
        patience=config.training.patience,
        mode='max' if 'auc' in config.training.monitor_metric.lower() else 'min',
        restore_best_weights=True,
        verbose=1
    ))

    callbacks_list.append(callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
        update_freq='epoch'
    ))

    if config.training.scheduler == 'reduce_on_plateau':
        callbacks_list.append(callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=config.training.min_lr,
            verbose=1
        ))

    return callbacks_list


class HyperModel(kt.HyperModel):
    """HyperModel for Keras Tuner optimization."""
    def __init__(self, config: Config):
        self.config = config

    def build(self, hp: kt.HyperParameters) -> tf.keras.Model:
        """Build model with hyperparameters."""
        return create_model(self.config, hp)

    def fit(self, hp: kt.HyperParameters, model: tf.keras.Model, *args, **kwargs):
        """Fit model with hyperparameters."""
        # This is a simplified fit method.
        # For dynamic batch sizes, you would need to rebuild the dataset here.
        return model.fit(*args, **kwargs)


def load_model_from_checkpoint(checkpoint_path: str, config: Config) -> tf.keras.Model:
    """
    Load model from checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint
        config: Configuration object
        
    Returns:
        Loaded model
    """
    try:
        # Try loading the full model first
        model = tf.keras.models.load_model(checkpoint_path, compile=False)
        
        # Recompile with current configuration
        optimizer = create_optimizer(config.training.optimizer, config.training.learning_rate, config)
        model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
            metrics=[]
        )
        
        print(f"Loaded full model from {checkpoint_path}")
        return model
        
    except Exception as e:
        print(f"Could not load full model: {e}")
        
        try:
            # Create new model and load weights
            model = create_model(config)
            model.load_weights(checkpoint_path)
            print(f"Loaded model weights from {checkpoint_path}")
            return model
            
        except Exception as e:
            print(f"Could not load model weights: {e}")
            raise


def get_model_summary_string(model: tf.keras.Model) -> str:
    """
    Get model summary as string.
    
    Args:
        model: Keras model
        
    Returns:
        Model summary string
    """
    summary_lines = []
    model.summary(print_fn=summary_lines.append)
    return '\n'.join(summary_lines)


def freeze_backbone_layers(model: BirdCLEFModel, num_layers: int):
    """
    Freeze the first N layers of the backbone.
    
    Args:
        model: BirdCLEF model
        num_layers: Number of layers to freeze
    """
    for i, layer in enumerate(model.backbone.layers):
        if i < num_layers:
            layer.trainable = False
        else:
            layer.trainable = True
    
    print(f"Frozen first {num_layers} layers of backbone")


def unfreeze_backbone_layers(model: BirdCLEFModel):
    """
    Unfreeze all backbone layers.
    
    Args:
        model: BirdCLEF model
    """
    for layer in model.backbone.layers:
        layer.trainable = True
    
    print("Unfrozen all backbone layers")


if __name__ == "__main__":
    # Test model creation
    from config import create_default_config
    print("Testing model creation...")
    config = create_default_config()

    try:
        model = create_model(config)
        model.summary()
        dummy_input = tf.random.normal((2, *config.audio.target_shape, 1))
        output = model(dummy_input)
        print(f"\nOutput shape: {output.shape}")
        print(f"Expected shape: (2, {config.model.num_classes})")
        assert output.shape == (2, config.model.num_classes)
        print("\n✅ Model creation test successful!")
    except Exception as e:
        print(f"\n❌ Model creation test failed: {e}")
        import traceback
        traceback.print_exc()