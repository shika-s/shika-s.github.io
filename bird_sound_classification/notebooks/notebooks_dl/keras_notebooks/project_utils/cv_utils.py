# ------------------------------------------------------------------------
# utils/cv_runner.py
# ------------------------------------------------------------------------
import json, gc, numpy as np, pandas as pd, tensorflow as tf
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from keras_tuner.engine.hyperparameters import HyperParameters
from tensorflow import keras
from .tuner_utils import MultimodalHyperModel
from .data_augmentor_and_generator import data_generator_with_metadata, get_steps
from .model_architect import UnfreezeBackbone, macro_f1_fn

# ------------------------ helper: callbacks -----------------------------
def _build_callbacks(cfg, fold_id, model=None, monitor="val_macro_f1_fn"):
    ckpt_path = (
        Path(cfg.MODELS_DIR) /
        f"{cfg.model_save_name.replace('.keras', '')}_fold{fold_id}.keras"
    )

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            ckpt_path, monitor=monitor, mode="max",
            save_best_only=True
        ),
        keras.callbacks.EarlyStopping(
            monitor=monitor, mode="max",
            patience=getattr(cfg, "es_patience", 4),
            restore_best_weights=True
        ),
        UnfreezeBackbone(
            freeze_epochs=getattr(cfg, "freeze_epochs", 3),
            key="vi_t_backbone", lr_mult=0.1
        ),
    ]

    return callbacks


# ------------------------ main runner -----------------------------------
def cross_validate_best_hp(
    cfg,
    full_df: pd.DataFrame,
    spectrograms: dict,
    metadata_features: dict,
    class_weights: dict | None,
    hp_json_path: str | Path,
    n_splits: int = 5,
    random_state: int = 42,
):
    """
    n-fold CV using **exactly** the hyper-parameters that won the tuner search.
    Returns a list[dict] with metrics per fold.
    """
    # ---------- load best HPs -------------------------------------------
    with open(hp_json_path, "r", encoding="utf-8") as f:
        hp_vals = json.load(f)

    hp = HyperParameters()
    for k, v in hp_vals.items():
        hp.values[k] = v

    # make epochs configurable via HP if it was tuned
    cv_epochs = int(hp.values.get("epochs", cfg.epochs))

    # ---------- set up CV splits ----------------------------------------
    skf = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )
    y = full_df["y_species_encoded"].values
    metrics_per_fold: list[dict] = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(full_df, y), start=1):
        print(f"\n==========  Fold {fold}/{n_splits}  ==========")
        train_df = full_df.iloc[tr_idx].reset_index(drop=True)
        val_df   = full_df.iloc[va_idx].reset_index(drop=True)

        # ---------- build model -----------------------------------------
        hypermodel = MultimodalHyperModel(
            cfg, train_df, spectrograms, metadata_features, class_weights
        )
        model = hypermodel.build(hp)

        # ---------- data generators -------------------------------------
        train_gen = data_generator_with_metadata(
            train_df, cfg, spectrograms, metadata_features,
            class_weights=class_weights, is_train=True,
            yield_weight=False, batch_size=cfg.batch_size,
        )
        val_gen   = data_generator_with_metadata(
            val_df, cfg, spectrograms, metadata_features,
            class_weights=None, is_train=False,
            yield_weight=False, batch_size=cfg.batch_size,
        )
        steps_per_epoch = get_steps(train_df, cfg, cfg.batch_size)
        val_steps       = get_steps(val_df,   cfg, cfg.batch_size)

        # ---------- fit ---------------------------------------------------
        callbacks = _build_callbacks(cfg, fold, model=model)
        history = hypermodel.fit(
            hp, model,
            train_gen,
            validation_data=val_gen,
            epochs=cv_epochs,
            steps_per_epoch=steps_per_epoch,
            validation_steps=val_steps,
            verbose=1,
            callbacks=callbacks
        )
        # ---------- metrics ---------------------------------------------
        best_epoch = int(np.argmax(history.history["val_macro_f1_fn"]))
        metrics_per_fold.append({
            "fold"         : fold,
            "best_epoch"   : best_epoch,
            "val_loss"     : float(history.history["val_loss"][best_epoch]),
            "val_accuracy" : float(history.history["val_accuracy"][best_epoch]),
            "val_macro_f1" : float(history.history["val_macro_f1_fn"][best_epoch]),
        })

        tf.keras.backend.clear_session(); gc.collect()

    # ---------- summary -------------------------------------------------
    df = pd.DataFrame(metrics_per_fold)
    print("\n===== CV summary =====")
    print(df.describe().round(4))
    return metrics_per_fold