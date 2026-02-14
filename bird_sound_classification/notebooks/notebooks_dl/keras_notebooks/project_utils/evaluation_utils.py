"""
Evaluation and plotting utilities for model performance.

This module includes functions for test set evaluation and visualization of results.
Class mapping is defined here for consistency.

For junior data scientists:
- Use evaluate_on_test to get predictions on test data.
- Plot functions generate visuals; save them via cfg.PLOTS_DIR.
"""
import gc
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from typing import Tuple, Dict, Any
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.metrics import AUC, CategoricalAccuracy
from tensorflow.keras.losses import CategoricalCrossentropy
from .data_augmentor_and_generator import data_generator, data_generator_with_metadata

# Consistent class mapping
CLASS_MAPPING_DICT = {0: 'Amphibia', 1: 'Aves', 2: 'Insecta', 3: 'Mammalia'}

def evaluate_on_test(
    model: tf.keras.Model, 
    cfg: Any, 
    spectrograms: Dict[str, np.ndarray], 
    test_df: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Evaluate the model on the test dataset.
    
    Parameters:
    -----------
    model: Trained Keras model
    cfg: Configuration object with attributes like TARGET_SHAPE, in_channels, num_classes, batch_size
    spectrograms: Dictionary of precomputed spectrograms
    test_df: DataFrame with test samples
    
    Returns:
    --------
    y_true: One-hot encoded true labels (numpy array)
    y_pred_prob: Predicted probabilities (numpy array) # Updated docstring
    """
    # Create test dataset (no aug, no weights)
    test_ds = tf.data.Dataset.from_generator(
        lambda: data_generator(test_df, cfg, spectrograms, is_train=False, yield_weight=False),
        output_signature=(
            tf.TensorSpec(shape=(*cfg.TARGET_SHAPE, cfg.in_channels), dtype=tf.float32),
            tf.TensorSpec(shape=(cfg.num_classes,), dtype=tf.float32)
        )
    )
    test_ds = test_ds.batch(cfg.batch_size).prefetch(tf.data.AUTOTUNE)
    
    # Ground truth labels (one-hot encoded)
    y_true = np.array([to_categorical(row['y_species_encoded'], num_classes=cfg.num_classes)
                       for _, row in test_df.iterrows()])
    
    # Get predictions (probabilities)
    preds_prob = model.predict(test_ds, verbose=1)
    
    # Clear session and collect garbage
    tf.keras.backend.clear_session()
    gc.collect()
    
    # Compute overall metrics using probabilities
    test_loss = CategoricalCrossentropy()(y_true, preds_prob).numpy()
    test_auc = AUC(multi_label=False)(y_true, preds_prob).numpy()  # Multiclass AUC (one-vs-rest)
    test_acc = CategoricalAccuracy()(y_true, preds_prob).numpy()
    
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test AUC: {test_auc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    
    return y_true, preds_prob  # Return probabilities for downstream use

def get_config_str(cfg: Any) -> str:
    """
    Generate a string summarizing model configurations.
    """
    return (
        f"Model: {cfg.model_type}, Aug: {cfg.use_augmentation}, "
        f"OS: {cfg.use_oversampling}, CW: {cfg.use_class_weights}, "
        f"TL: {cfg.use_transfer_learning}"
    )

def get_suffix(cfg: Any) -> str:
    """
    Extract the dynamic suffix from model_save_name (e.g., '_aug').
    """
    return cfg.model_save_name.replace(f"base_model_{cfg.input_dim}", "").replace(".keras", "")


def plot_training_diagnostics(
    history: tf.keras.callbacks.History, 
    cfg: Any, 
    y_true: np.ndarray, 
    y_pred_prob: np.ndarray, 
    class_mapping: Dict[int, str]
) -> None:
    """
    Plot training diagnostics: learning curves, normalized confusion matrix, and per-class metrics bar plot for test evaluation.
    Saves plots to disk after displaying.
    
    Parameters:
    -----------
    history: Keras history object from model.fit()
    cfg: Configuration object with PLOTS_DIR, input_dim, model_save_name
    y_true: One-hot encoded true labels (numpy array)
    y_pred: Predicted probabilities (numpy array)
    class_mapping_dict: dict mapping class indices to names
    """
    suffix = get_suffix(cfg)
    config_str = get_config_str(cfg)
    
    # Combined Learning Curves
    metrics_to_plot = ['loss', 'accuracy', 'auroc']  # Add more if available in history
    fig, axs = plt.subplots(1, len(metrics_to_plot), figsize=(4 * len(metrics_to_plot), 4))
    if not isinstance(axs, np.ndarray):
        axs = np.array([axs])  # Make iterable if single
    for i, metric in enumerate(metrics_to_plot):
        if metric in history.history:
            axs[i].plot(history.history[metric], label=f'Train {metric}')
            if f'val_{metric}' in history.history:
                axs[i].plot(history.history[f'val_{metric}'], label=f'Val {metric}')
            axs[i].set_title(f'{metric.capitalize()} Curve')
            axs[i].set_xlabel('Epoch')
            axs[i].set_ylabel(metric.capitalize())
            axs[i].legend()
    plt.suptitle(f"Training History ({config_str})")
    plt.tight_layout()
    
    # Save before show
    plot_path = cfg.PLOTS_DIR / f"training_curves_{cfg.input_dim}{suffix}.png"
    fig.savefig(plot_path)  # Use fig.savefig() for explicit control
    print(f"Training curves saved to: {plot_path}")
    
    plt.show()  # Display after saving
    plt.close(fig)  # Close to free memory
    
    # Convert to class labels for CM and report
    y_true_labels = np.argmax(y_true, axis=1)
    y_pred_labels = np.argmax(y_pred_prob, axis=1)
    class_names = list(class_mapping.values())
    
    # Confusion Matrix (normalized)
    cm = confusion_matrix(y_true_labels, y_pred_labels)
    fig_cm, ax_cm = plt.subplots(figsize=(6, 4.5))
    sns.heatmap(cm / np.sum(cm, axis=1)[:, np.newaxis], annot=True, fmt='.2%', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax_cm)
    ax_cm.set_title('Normalized Confusion Matrix')
    ax_cm.set_xlabel('Predicted')
    ax_cm.set_ylabel('True')
    
    # Save before show
    cm_plot_path = cfg.PLOTS_DIR / f"confusion_matrix_{cfg.input_dim}{suffix}.png"
    fig_cm.savefig(cm_plot_path)
    print(f"Confusion matrix saved to: {cm_plot_path}")
    
    plt.show()  # Display after saving
    plt.close(fig_cm)
    
    # Per-Class Bar Plot
    report = classification_report(y_true_labels, y_pred_labels, target_names=class_names, output_dict=True)
    df_report = pd.DataFrame(report).transpose().iloc[:-3, :3]  # Precision, Recall, F1 per class
    fig_bar, ax_bar = plt.subplots(figsize=(5, 5))
    df_report.plot(kind='bar', ax=ax_bar)
    ax_bar.set_title('Per-Class Precision, Recall, F1')
    plt.tight_layout()
    
    # Save before show
    per_class_plot_path = cfg.PLOTS_DIR / f"per_class_metrics_{cfg.input_dim}{suffix}.png"
    fig_bar.savefig(per_class_plot_path)
    print(f"Per-class metrics saved to: {per_class_plot_path}")
    
    plt.show()  # Display after saving
    plt.close(fig_bar)


def plot_test_evaluation(
    y_true: np.ndarray, 
    y_pred_prob: np.ndarray, 
    class_mapping: Dict[int, str], 
    cfg: Any
) -> None:
    """
    Plot confusion matrix with predicted class probability distribution on the side,
    and save classification report for test set.
    
    Args:
        y_true: True labels
        y_pred_prob: Predicted probs
        class_mapping: Dict for class names
        cfg: Configuration (for save paths and configs)
        
    Saves report CSV and confusion matrix plot with dynamic filenames including configs.
    """
    # Default mapping if none provided
    if class_mapping is None:
        class_mapping = {i: f'Class_{i}' for i in range(y_true.shape[1])}
    
    # Get class names in order
    class_names = [class_mapping[i] for i in range(len(class_mapping))]
    
    # Convert to class labels
    y_true_labels = np.argmax(y_true, axis=1)
    y_pred_labels = np.argmax(y_pred_prob, axis=1)
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    # Plot 1: Confusion Matrix
    cm = confusion_matrix(y_true_labels, y_pred_labels)
    im = axes[0].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    axes[0].figure.colorbar(im, ax=axes[0])
    axes[0].set(xticks=np.arange(cm.shape[1]),
                yticks=np.arange(cm.shape[0]),
                xticklabels=class_names,
                yticklabels=class_names,
                title='Confusion Matrix',
                ylabel='True label',
                xlabel='Predicted label')
    
    # Rotate x-tick labels
    plt.setp(axes[0].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations to CM
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            axes[0].text(j, i, format(cm[i, j], 'd'),
                         ha="center", va="center",
                         color="white" if cm[i, j] > thresh else "black")
    
    # Plot 2: Per-class probability distributions
    colors = ['blue', 'orange', 'green', 'brown']  # Matching example
    for idx, class_name in enumerate(class_names):
        axes[1].hist(y_pred_prob[:, idx], bins=30, alpha=0.5, label=class_name, color=colors[idx % len(colors)])
    axes[1].set_xlabel('Predicted Probability')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Distribution of Predicted Probabilities by Class')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    suffix = get_suffix(cfg)
    config_str = get_config_str(cfg)
    plt.suptitle(f"Test Evaluation ({config_str})")
    plt.tight_layout()
    
    # Save before show
    eval_plot_path = cfg.PLOTS_DIR / f'test_evaluation_{cfg.input_dim}{suffix}.png'
    fig.savefig(eval_plot_path)
    print(f"Test evaluation plot saved to: {eval_plot_path}")
    
    plt.show()  # Display after saving
    plt.close(fig)
    
    # Save classification report as CSV if cfg is provided
    report = classification_report(y_true_labels, y_pred_labels, target_names=class_names, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    csv_path = cfg.RESULTS_CSV_DIR / f'classification_report_{cfg.input_dim}{suffix}.csv'
    df_report.to_csv(csv_path)
    print(f"Classification report saved to: {csv_path}")
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_true_labels, y_pred_labels, target_names=class_names))


def evaluate_on_test_multimodal(
    model: tf.keras.Model, 
    cfg: Any, 
    spectrograms: Dict[str, np.ndarray], 
    metadata_features: Dict[str, np.ndarray],
    test_df: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Multimodal version of evaluate_on_test. Returns y_true (one-hot) and y_pred_prob.
    """
    # Fix: Use batched signature (from Step 2)
    test_ds = tf.data.Dataset.from_generator(
        lambda: data_generator_with_metadata(test_df, cfg, spectrograms, metadata_features, is_train=False, yield_weight=False),
        output_signature=(
            (tf.TensorSpec(shape=(None, *cfg.TARGET_SHAPE, cfg.in_channels), dtype=tf.float32),
             tf.TensorSpec(shape=(None, cfg.metadata_dim), dtype=tf.float32)),
            tf.TensorSpec(shape=(None, cfg.num_classes), dtype=tf.float32)
        )
    ).prefetch(tf.data.AUTOTUNE)
    
    # Ground truth (one-hot)
    y_true = np.array([to_categorical(row['y_species_encoded'], num_classes=cfg.num_classes)
                       for _, row in test_df.iterrows()])
    
    # Predict with steps to handle finite data (prevent infinite loop)
    steps = math.ceil(len(test_df) / cfg.batch_size)
    y_pred_prob = model.predict(test_ds, steps=steps, verbose=1)
    
    # Trim if extra padded samples
    if len(y_pred_prob) > len(y_true):
        y_pred_prob = y_pred_prob[:len(y_true)]
    
    # Compute metrics
    test_loss = CategoricalCrossentropy()(y_true, y_pred_prob).numpy()
    test_auc = AUC(multi_label=False)(y_true, y_pred_prob).numpy()
    test_acc = CategoricalAccuracy()(y_true, y_pred_prob).numpy()
    
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test AUC: {test_auc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    
    tf.keras.backend.clear_session()
    gc.collect()
    
    return y_true, y_pred_prob