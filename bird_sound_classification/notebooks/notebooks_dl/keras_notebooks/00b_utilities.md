Below is a quick reference to each of the key utility scripts that underpin the ViT training notebook—what they provide, how the notebook uses them, and tips for extending or debugging.

---

### `common_imports.py`

* **Purpose**: Centralize all standard library, ML/DL, and environment setup (logging, warnings, mixed-precision).
* **Key contents**:

  * Imports for NumPy, Pandas, TensorFlow/Keras, scikit-learn, etc.
  * `mixed_precision.set_global_policy("mixed_float16")` to halve memory use.
  * Logging & warning filters.
* **Notebook use**: Simply imported at the top to ensure a consistent environment.
* **Tip**: If you add new dependencies (e.g. a custom augmentation library), import them here and adjust global policies as needed.

---

### `data_augmentor_and_generator.py`

* **Purpose**: Build on-the-fly spectrogram augmentation + a flexible Keras data generator that handles oversampling, mixup, and per-sample weighting.
* **Key components**:

  * `SpecTimeMask` & `SpecFreqMask` layers (SpecAugment).
  * `data_generator_with_metadata(df, cfg, spectrograms, metadata_features, class_weights)` yields `(X_spec, X_meta), y, sample_weight`.
  * `get_steps(df, batch_size, cfg)` computes train/val steps given oversampling factors.
* **Notebook use**:

  * Called by `run_multimodal_tuner()` and CV routines to feed batches during search & training.
  * Controlled by `cfg.use_augmentation`, `cfg.use_oversampling`, `cfg.yield_weight`, etc.
* **Tip**: To add a new augmentation (e.g. time warp), wrap it as a Keras layer and insert into the generator loop.

---

### `loading_utils.py`

* **Purpose**: End-to-end data loading, split, imbalance handling, and generator sanity checks.
* **Key functions**:

  * `main_data_processing_pipeline(cfg, spectrograms, data_generator)`: returns `(train_val_df, test_df, class_weights, filtered_spectrograms)`.
  * `compute_oversampling_factors(df, cap)`, `compute_class_weights(...)`.
  * `inspect_data_generator(...)`: prints a sample batch for validation.
* **Notebook use**:

  * Drives all of Section 3 (Data Preparation).
  * Computes `cfg.oversampling_factors` and `class_weights_dict`.
* **Tip**: If your metadata format changes, adjust `preprocess_metadata` here and update `cfg.metadata_dim`.

---

### `cv_utils.py`

* **Purpose**: Wrap a 5-fold stratified cross-validation loop around a fixed set of hyperparameters.
* **Key functions**:

  * `cross_validate_best_hp(...)`: for each fold, rebuilds the model, fits, records best epoch metrics, and clears session.
  * `_build_callbacks(...)`: checkpointing & early stopping per fold.
* **Notebook use**: Called when `cfg.train_model=True` to produce fold-wise macro-F1, loss, and accuracy.
* **Tip**: To switch to a different CV strategy (e.g. GroupKFold), swap out the `StratifiedKFold` instantiation.

---

### `tuner_utils.py`

* **Purpose**: Define the HyperModel and orchestrate Keras Tuner’s Hyperband search.
* **Key components**:

  * `MultimodalHyperModel(kt.HyperModel)`: implements `build(hp)` to sample LR, dropout, unfreeze schedule, etc., and `run_trial` to attach checkpoint callbacks.
  * `run_multimodal_tuner(...)`: configures `kt.Hyperband`, runs the search, saves the best model & hyperparameters JSON.
* **Notebook use**: Used directly in the “Hyperparameter Search” section.
* **Tip**: To add a new hyperparameter (e.g. attention heads), update `MultimodalHyperModel.build()` and rerun the tuner.

---

### `model_architect.py`

* **Purpose**: Assemble the actual Keras model architecture (ViT + metadata branch) and custom callbacks/metrics.
* **Key definitions**:

  * **Layers & callbacks**: `AddClsToken`, `UnfreezeBackbone` (unfreezes ViT after N epochs), `SafeCheckpointTuner`.
  * **Metrics & losses**: `macro_f1_fn`, `categorical_focal_loss(alpha, gamma)`, `get_cosine_decay_scheduler`.
  * **Model builder**: `get_multimodal_vit_model(cfg)`, which fuses CLS-pooled ViT features with a metadata MLP (optionally with soft-attention).
* **Notebook use**: Imported by both tuner and CV routines to instantiate/freeze/unfreeze the backbone and compile with the right loss/metrics.
* **Tip**: Swap in a different Transformer backbone by changing the `keras_hub` URL or replacing `get_vit_model`.

---

### `evaluation_utils.py`

* **Purpose**: Evaluate the final trained model on the test set and generate publication-ready plots.
* **Key functions**:

  * `evaluate_on_test_multimodal(...)`: runs `model.predict`, computes loss/accuracy/AUC, and returns `(y_true, y_pred_prob)`.
  * Plotters: `plot_training_diagnostics()`, `plot_confusion_matrix_prob()`, plus utilities to save classification reports and probability histograms.
* **Notebook use**: Called in the final evaluation section to visualize learning curves, confusion matrices, and per-class performance.
* **Tip**: If you need to log metrics to TensorBoard or WandB instead, wrap these functions or insert callbacks in your training loop.

---

With this map of utilities, one should be able to:

1. **Locate** where each piece of functionality lives.
2. **Understand** which `cfg` flags toggle which behaviors.
3. **Extend** any module—whether adding new augmentations, hyperparameters, backbone models, or evaluation plots—without modifying the core notebook.
