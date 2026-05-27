Here’s a detailed guide of `02_vit_training_trials_aug_tl_kt.ipynb` to understand, reproduce, and extend its Vision-Transformer (ViT) training pipeline with augmentation, transfer learning, and Keras Tuner.

---

## 1. Notebook Overview

This notebook loads precomputed mel-spectrograms, metadata, and then:

1. Prepares data (splits, oversampling factors, class weights)
2. Builds a multimodal ViT + metadata fusion model
3. Runs a Hyperband search over learning rate, dropout, backbone-unfreeze schedule, etc.
4. Applies the best hyperparameters in a 5-fold cross-validation
5. Evaluates and visualizes results on the held-out test set

---

## 2. Configuration (`CFG` class)

All high-level settings live in the `CFG` class near the top of the notebook:

* **`model_type`**: `"vit"` or `"rescnn"`
* **`input_dim`**: spectrogram resolution (e.g. 32, 64, 256)
* **`use_augmentation`**, **`use_transfer_learning`**, **`use_class_weights`**: toggles for each feature
* **`use_tuner`**: whether to invoke Keras Tuner
* **`n_fold`**, **`epochs`**, **`batch_size`**, **`lr`**, **`min_lr`**: training and search settings
* **`OUTPUT_DIR`**, **`spectrogram_npy`**: paths for artifacts

> **Tip:** Tweak these flags to turn on/off components without rewriting code.

---

## 3. Data Preparation

1. **Load spectrograms** from `.npy`:

   ```python
   spectrograms = np.load(cfg.spectrogram_npy, allow_pickle=True).item()
   ```

2. **Main pipeline**:

   ```python
   combined_train_val_df, test_df,
   class_weights_dict, filtered_spectrograms = \
       main_data_processing_pipeline(cfg, spectrograms, data_generator)
   ```

   * Splits into train/val/test
   * Computes `oversampling_factors` (capped at 50×)
   * Computes balanced `class_weights` if enabled
   * Prints class distribution and inspects one batch from `data_generator`

3. **Metadata encoding** via 3D conversion + scaling + one-hot:

   ```python
   metadata_features = preprocess_metadata(combined_train_val_df, cfg)
   ```

   Updates `cfg.metadata_dim` automatically.

---

## 4. Train/Validation Split

A reproducible 80/20 stratified split on `y_species_encoded`:

```python
train_df, val_df = train_test_split(
    combined_train_val_df, test_size=0.2,
    stratify=combined_train_val_df["y_species_encoded"],
    random_state=cfg.seed
)
```

---

## 5. Hyperparameter Search with Keras Tuner

If `cfg.use_tuner` is `True`, the notebook runs:

```python
best_model, tuner = run_multimodal_tuner(
    cfg, train_df, val_df,
    filtered_spectrograms,
    metadata_features,
    class_weights
)
```

Key points:

* **Search space** (in `MultimodalHyperModel.build`):

  * `lr`: 1e-5 → 5e-4 (log scale)
  * `dropout`: 0.2 → 0.6
  * `vision_lr_mult`: {0.1, 0.25, 0.5}
  * `dense_width`: {128, 256, 384}
  * `weight_decay`: 1e-6 → 1e-4
* **Bandwidth**: Hyperband with factor=3, `max_epochs=cfg.epochs`
* **Callbacks**:

  * `ModelCheckpoint` on `val_macro_f1_fn`
  * `EarlyStopping` (patience=8)
  * `UnfreezeBackbone` after `freeze_epochs`

> **Pro tip:** Monitor GPU memory manually—ViTs at 256×256 can spike to ≈12 GB.

---

## 6. Inspecting & Saving Best H-Ps

After search:

```python
hp_json_path = project_dir / f"best_hps_{cfg.model_save_name.stem}.json"
# pretty-print:
for name, val in tuner.get_best_hyperparameters()[0].values.items():
    print(f"  {name}: {val}")
```

---

## 7. 5-Fold Cross-Validation

When `cfg.train_model` is `True`, the notebook loads those JSON hyperparameters and runs:

```python
metrics = cross_validate_best_hp(
    cfg=cfg,
    full_df=combined_train_val_df,
    spectrograms=filtered_spectrograms,
    metadata_features=metadata_features,
    class_weights=class_weights,
    hp_json_path=hp_json_path,
    n_splits=cfg.n_fold
)
```

This yields per-fold best epoch, loss, accuracy, and macro-F1, then prints a summary table.

---

## 8. Final Evaluation & Visualization

1. **Model summary & architecture plot**:

   ```python
   best_model.summary()
   plot_model(best_model, show_shapes=True, expand_nested=True)
   ```
2. **Test-set predictions**:

   ```python
   y_true, y_pred_prob = evaluate_on_test_multimodal(
       best_model, cfg, filtered_spectrograms, metadata_features, test_df
   )
   ```
3. **Diagnostics**:

   * Learning curves (`plot_training_diagnostics`)
   * Confusion matrix & per-class bars
   * Probability distributions (`plot_test_evaluation`)

---

## 9. Custom Components & Tips

* **SpecAugment layers**: `SpecTimeMask` & `SpecFreqMask` zero out random time/frequency bands.
* **UnfreezeBackbone callback**: unfreezes the ViT after N epochs and safely scales LR.
* **Data generators** support mixup, on-the-fly masking, and oversampling.

> **Common pitfalls:**
>
> * Forgetting to clear sessions (`tf.keras.backend.clear_session()`) → OOM
> * Mismatched shapes if changing `input_dim` without adjusting `TARGET_SHAPE`
> * Very long epochs when oversampling extreme classes—cap factors wisely

---

## 10. How to Extend

* **Add new HPs**: modify `MultimodalHyperModel.build()`
* **Swap backbone**: change `vit_preset` or use EfficientNet/EfficientNetV2
* **Custom augmentations**: add to `build_gpu_augmenter(cfg)`
* **Late-fusion or attention**: alter `get_multimodal_vit_model()`

