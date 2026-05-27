# tuner_utils.py
from pathlib import Path
import gc, math, json
import tensorflow as tf
from tensorflow import keras
import keras_tuner as kt

from .model_architect import (
    get_multimodal_vit_model,
    macro_f1_fn,
    categorical_focal_loss,
    UnfreezeBackbone,
    get_cosine_decay_scheduler,
    SafeCheckpointTuner,
)
from .data_augmentor_and_generator import (
    data_generator_with_metadata,
    get_steps,
)

class MultimodalHyperModel(kt.HyperModel):
    """
    Custom HyperModel for building and fitting the multimodal ViT with HP-dependent elements.
    """
    def __init__(self, cfg, train_df, spectrograms, metadata_features, class_weights):
        self.cfg = cfg
        self.train_df = train_df
        self.spectrograms = spectrograms
        self.metadata_features = metadata_features
        self.class_weights = class_weights

    def build(self, hp):
        # ------------------------------------------------------------------ H-P space
        base_lr = hp.Float("lr", 1e-5, 5e-4, sampling="log", default=1e-4)
        dropout = hp.Float("dropout", 0.2, 0.6, step=0.1, default=0.3)
        vision_mult = hp.Choice("vision_lr_mult", [0.1, 0.25, 0.5], default=0.1)
        dense_width = hp.Choice("dense_width", [128, 256, 384], default=256)
        freeze_epoch = hp.Int("freeze_epochs", 0, 5, default=3)  # used by callback

        # Push chosen dropout back into cfg so get_vit_model sees it -----------
        self.cfg.dropout_rate = dropout

        # ---------------------------------------------------------------- build model
        model = get_multimodal_vit_model(self.cfg)

        # LR schedule (cosine)  -----------------------------------------------
        steps_per_epoch = get_steps(self.train_df, self.cfg, self.cfg.batch_size)
        decay_steps = self.cfg.epochs * steps_per_epoch
        alpha = self.cfg.min_lr / base_lr
        cosine = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=base_lr,
            decay_steps=decay_steps,
            alpha=alpha,
            name="cosine_decay_scheduler"
        )

        opt = keras.optimizers.AdamW(learning_rate=cosine,
                                     weight_decay=hp.Float("weight_decay",
                                                           1e-6, 1e-4,
                                                           sampling="log",
                                                           default=1e-5))
        model.compile(
            optimizer=opt,
            loss=categorical_focal_loss(alpha=0.25, gamma=2.0),
            metrics=["accuracy", macro_f1_fn]
        )

        return model

    def fit(self, hp, model, *args, **kwargs):
        # ---------------------------------------------------------------- callbacks
        # ① unfreeze backbone
        unfreeze_cb = UnfreezeBackbone(
            freeze_epochs=hp.Int("freeze_epochs", 0, 5, default=3),
            key="vi_t_backbone",  # Adjust if your backbone name differs
            lr_mult=hp.Choice("vision_lr_mult", [0.1, 0.25, 0.5], default=0.1)
        )

        # ② early stop (patient on macro-F1)
        early_cb = keras.callbacks.EarlyStopping(
            monitor="val_macro_f1_fn",
            patience=8,
            mode="max",
            restore_best_weights=True
        )

        # Add to existing callbacks (SafeCheckpointTuner adds checkpoint)
        if 'callbacks' not in kwargs:
            kwargs['callbacks'] = []
        kwargs['callbacks'].extend([unfreeze_cb, early_cb])

        return model.fit(*args, **kwargs)


def run_multimodal_tuner(cfg,
                         train_df,
                         val_df,
                         spectrograms,
                         metadata_features,
                         class_weights):
    """
    Creates a SafeCheckpointTuner, launches search, and returns the best model.
    """
    project_dir = Path(cfg.OUTPUT_DIR) / f'{cfg.model_save_name.replace(".keras", "")}_tuner'
    project_dir.mkdir(parents=True, exist_ok=True)

    # Compute fixed data generators and steps outside (HP-independent)
    train_gen = data_generator_with_metadata(
        train_df, cfg, spectrograms, metadata_features,
        class_weights=class_weights,
        is_train=True,
        yield_weight=False,
        batch_size=cfg.batch_size,
    )
    val_gen = data_generator_with_metadata(
        val_df, cfg, spectrograms, metadata_features,
        class_weights=None,
        is_train=False,
        yield_weight=False,
        batch_size=cfg.batch_size,
    )
    steps_per_epoch = get_steps(train_df, cfg, cfg.batch_size)
    val_steps = get_steps(val_df, cfg, cfg.batch_size)

    # Create HyperModel instance
    hypermodel = MultimodalHyperModel(cfg, train_df, spectrograms, metadata_features, class_weights)

    tuner = SafeCheckpointTuner(
        hypermodel=hypermodel,
        objective=kt.Objective("val_macro_f1_fn", "max"),
        max_epochs=cfg.epochs,
        factor=3,
        hyperband_iterations=2,             # ➟ total ~ (1 + factor) epochs
        executions_per_trial=1,
        directory=str(project_dir),
        project_name=f'{cfg.model_save_name.replace(".keras", "")}',
        # overwrite=True,
    )

    print("\n⏳  Starting hyper-parameter search …")
    tuner.search(
        x=train_gen,
        validation_data=val_gen,
        epochs=cfg.epochs,
        steps_per_epoch=steps_per_epoch,
        validation_steps=val_steps,
        verbose=2,
    )
    print("✅  Search finished!")

    # Reload best weights
    best_trial = tuner.get_best_models(num_models=1)[0]
    # — optional: save to disk
    best_path = project_dir / f'best_{cfg.model_save_name}'
    best_trial.save(best_path)
    print(f"Best model saved ➜  {best_path}")

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0].values
    hp_json_path = project_dir / f"best_hps_{cfg.model_save_name.replace('.keras', '')}.json"
    with open(hp_json_path, 'w') as f:
        json.dump(best_hps, f, indent=4)
    print(f"Best hyperparameters saved to: {hp_json_path}")

    return best_trial, tuner

