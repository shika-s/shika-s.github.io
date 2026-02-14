# Project Plan: Deep Learning Models for BirdCLEF 2025

This document outlines the plan for developing and training deep learning models for the BirdCLEF 2025 competition.

## 1. Data Transformations, Augmentations & Feature Engineering

Our primary feature representation will be Mel-spectrograms generated from the raw audio files.

*   **Spectrogram Generation**:
    *   **Audio Sampling Rate**: 32,000 Hz
    *   **FFT Window Size (n_fft)**: 1024
    *   **Hop Length**: 512
    *   **Number of Mel Bands (n_mels)**: 128
    *   **Frequency Range**: 50 Hz to 14,000 Hz
*   **Preprocessing**:
    *   Audio clips will be processed into 5-second segments.
    *   Spectrograms will be resized to a fixed shape of (256, 256).
    *   Spectrograms will be normalized to a [0, 1] range after being converted to decibels.
*   **Augmentations**: To improve model generalization, we will apply the following augmentations during training:
    *   **Mixup**: Linearly interpolating between samples and their labels. (alpha = 0.5)
    *   **SpecAugment**:
        *   **Time Masking**: Applying horizontal masks to the spectrogram.
        *   **Frequency Masking**: Applying vertical masks to the spectrogram.
    *   **Brightness/Contrast**: Randomly adjusting the brightness and contrast of the spectrograms.

## 2. Train/Validation/Test Split

We will use a **5-fold stratified cross-validation** strategy.

*   **Splitting**: The data will be split based on the `primary_label` to ensure that the class distribution is maintained across all folds.
*   **Ratio**: Within each fold, the data will be split into an **80% training set** and a **20% validation set**.
*   **Test Set**: The final evaluation will be performed on the official Kaggle test set.

## 3. Strategies to Manage Computational Cost

*   **Mixed Precision Training**: We will use `mixed_float16` precision for training on V100 GPUs to accelerate training and reduce memory usage.
*   **Pre-computed Spectrograms**: To speed up experimentation, we will use pre-computed mel-spectrograms stored in a `.npy` file. The pipeline also supports on-the-fly generation if needed.
*   **Efficient Data Pipeline**: The `tf.data` API will be used to optimize the data loading and preprocessing pipeline for efficient GPU utilization.
*   **Debug Mode**: A debug mode is available in the configuration to run experiments on a small subset of the data (1000 samples, 2 epochs) for quick sanity checks.

## 4. Handling Class Imbalance

*   **Stratified Sampling**: As mentioned above, we will use stratified k-fold cross-validation to ensure each fold has a similar class distribution to the overall dataset.
*   **Evaluation Metric**: The Kaggle competition uses a mean class-wise AUC. By optimizing for this metric, we are inherently paying more attention to the performance on minority classes.
*   **Future Strategies**: We can explore other techniques if class imbalance remains a significant issue:
    *   **Class Weights**: Applying weights to the loss function to give more importance to under-represented classes.
    *   **Focal Loss**: A variant of cross-entropy loss that focuses training on hard-to-classify examples.
    *   **Oversampling/Undersampling**: Techniques like SMOTE (Synthetic Minority Over-sampling Technique) could be considered.

## 5. Experiment Tracking & Reproducibility

*   **Configuration Management**: A centralized `config.py` file will be used to manage all parameters for reproducibility.
*   **Random Seeds**: We will set a fixed random seed (42) for all libraries (numpy, tensorflow, random) to ensure deterministic behavior.
*   **SLURM Integration**: The `slurm_scripts` directory provides scripts for submitting and managing training jobs on a SLURM cluster.
*   **Experiment Tracking Tools**: We should consider using a dedicated experiment tracking tool like **MLflow** or **Weights & Biases** to log parameters, metrics, and model artifacts for better comparison and organization.

## 6. Metrics for Comparing Against Baseline

*   **Primary Metric**: **Mean Class-wise AUC**. This aligns with the Kaggle competition's evaluation metric and is a good measure for multi-label classification with imbalanced classes. Our Keras implementation includes a custom `ClassWiseAUC` metric for this.
*   **Secondary Metric**: **Binary Cross-Entropy Loss**. This will be monitored during training to ensure the model is converging.

## 7. My Models: RNN and ViT

As I am in charge of testing RNN and Vision Transformer (ViT) models, I will follow this plan:

*   **Recurrent Neural Network (RNN)**:
    *   **Input**: I will use the mel-spectrograms as input, treating the time axis as a sequence. The input shape will be (batch_size, time_steps, n_mels), e.g., (32, 256, 128).
    *   **Architecture**: I will start with a simple architecture, such as 2-3 layers of LSTMs or GRUs, followed by a Dense classification head.
    *   **Data Pipeline**: I will adapt the existing `data_pipeline.py` to provide data in the required sequential format.
*   **Vision Transformer (ViT)**:
    *   **Input**: The 256x256 spectrogram images will be treated as input images.
    *   **Architecture**: I will use a pre-trained ViT model (e.g., from `timm` or TensorFlow Hub) and add a custom classification head for our 206 classes.
    *   **Data Pipeline**: The existing data pipeline can be used as-is, since it already provides image-like spectrograms.

## 8. Heads-up

*   **Competition Metric**: We should definitely switch to using the **average AUC per class** as our primary metric, as it is what the Kaggle competition uses.
*   **Latitude/Longitude**: The current plan does not use latitude and longitude information for training, so the fact that it's not available in the test set is not an issue for our current models. We can consider incorporating this metadata in future iterations if needed.
