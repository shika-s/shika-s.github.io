#!/usr/bin/env python3
"""
Single model training for BirdCLEF 2025 with best hyperparameters.
"""

import os
import sys
import argparse
import json
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime
from pathlib import Path

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config, create_default_config, load_config
from data_pipeline import create_data_pipeline
from model_definitions import create_model, create_callbacks, create_lr_scheduler
from utils import (
    set_random_seeds,
    setup_logging,
    clear_memory,
    Timer,
    format_time,
    monitor_gpu_memory,
    calculate_auc_score,
    save_training_history,
    print_model_summary
)


class BirdCLEFTrainer:
    """
    Main class for training BirdCLEF models with cross-validation.
    """
    
    def __init__(self, config: Config, hyperparameters: dict = None):
        self.config = config
        self.hyperparameters = hyperparameters or {}
        self.logger = None
        self.data_pipeline = None
        
        # Setup
        self._setup_environment()
        self._setup_logging()
        self._setup_data_pipeline()
        
        # Apply hyperparameters to config
        self._apply_hyperparameters()
    
    def _setup_environment(self):
        """Setup TensorFlow environment and mixed precision."""
        # Set random seeds
        set_random_seeds(self.config.seed)
        
        # Setup distributed strategy
        self.strategy = self.config.get_device_strategy()
        
        # Setup mixed precision
        self.config.setup_mixed_precision()
        
        # Print environment info
        print(f"TensorFlow version: {tf.__version__}")
        print(f"Number of GPUs: {len(tf.config.experimental.list_physical_devices('GPU'))}")
        print(f"Mixed precision: {self.config.model.use_mixed_precision}")
        print(f"Strategy: {type(self.strategy).__name__}")
        
        # Monitor GPU memory
        monitor_gpu_memory()
    
    def _setup_logging(self):
        """Setup logging."""
        self.logger = setup_logging(self.config.paths.logs_dir, 'INFO')
        self.logger.info("Starting model training")
        self.logger.info(f"Configuration: {self.config}")
        if self.hyperparameters:
            self.logger.info(f"Hyperparameters: {self.hyperparameters}")
    
    def _setup_data_pipeline(self):
        """Setup data pipeline."""
        self.data_pipeline = create_data_pipeline(self.config)
        self.logger.info("Data pipeline setup completed")
    
    def _apply_hyperparameters(self):
        """Apply hyperparameters to configuration."""
        if not self.hyperparameters:
            return
        
        # Update training configuration with hyperparameters
        hp_mapping = {
            'learning_rate': ('training', 'learning_rate'),
            'batch_size': ('training', 'batch_size'),
            'optimizer': ('training', 'optimizer'),
            'scheduler': ('training', 'scheduler'),
            'mixup_alpha': ('training', 'mixup_alpha'),
            'dropout_rate': ('model', 'dropout_rate'),
        }
        
        for hp_key, (section, config_key) in hp_mapping.items():
            if hp_key in self.hyperparameters:
                section_obj = getattr(self.config, section)
                setattr(section_obj, config_key, self.hyperparameters[hp_key])
                self.logger.info(f"Applied hyperparameter: {hp_key} = {self.hyperparameters[hp_key]}")
    
    def train_single_fold(
        self, 
        fold: int, 
        train_df: pd.DataFrame
    ) -> dict:
        """
        Train model on a single fold.
        
        Args:
            fold: Fold number
            train_df: Training DataFrame
            
        Returns:
            Dictionary with training results
        """
        self.logger.info(f"Training fold {fold}")
        
        # Create fold datasets
        with Timer(f"Creating datasets for fold {fold}"):
            train_dataset, val_dataset = self.data_pipeline.create_fold_datasets(
                train_df, fold, self.config.training.n_folds
            )
        
        # Calculate dataset sizes and steps
        train_size = len(train_df) * (self.config.training.n_folds - 1) // self.config.training.n_folds
        val_size = len(train_df) // self.config.training.n_folds
        steps_per_epoch = self.data_pipeline.get_steps_per_epoch(train_size)
        
        self.logger.info(f"Train size: {train_size}, Val size: {val_size}")
        self.logger.info(f"Steps per epoch: {steps_per_epoch}")
        
        # Create model within strategy scope
        with self.strategy.scope():
            model = create_model(
                self.config,
                **self.hyperparameters
            )
            
            # Create learning rate scheduler if needed
            lr_scheduler = create_lr_scheduler(
                self.config.training.scheduler,
                model.optimizer,
                self.config,
                steps_per_epoch
            )
        
        # Print model summary
        if fold == 0:  # Only print for first fold
            print_model_summary(model, f"BirdCLEF Model (Fold {fold})")
        
        # Create callbacks
        model_path = os.path.join(
            self.config.paths.model_dir,
            f'best_model_fold_{fold}.h5'
        )
        
        log_dir = os.path.join(
            self.config.paths.logs_dir,
            f'fold_{fold}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        )
        
        callbacks_list = create_callbacks(
            self.config,
            model_path,
            log_dir,
            validation_data=val_dataset
        )
        
        # Add learning rate scheduler if created
        if lr_scheduler:
            callbacks_list.append(lr_scheduler)
        
        # Train model
        self.logger.info(f"Starting training for fold {fold}")
        
        with Timer(f"Training fold {fold}"):
            history = model.fit(
                train_dataset,
                validation_data=val_dataset,
                epochs=self.config.training.epochs,
                callbacks=callbacks_list,
                verbose=1
            )
        
        # Load best model weights
        if os.path.exists(model_path):
            model.load_weights(model_path)
            self.logger.info(f"Loaded best weights for fold {fold}")
        
        # Evaluate on validation set
        self.logger.info(f"Evaluating fold {fold}")
        val_loss = model.evaluate(val_dataset, verbose=0)
        
        # Calculate AUC on validation set
        val_auc = self._calculate_validation_auc(model, val_dataset)
        
        # Save training history
        history_path = os.path.join(
            self.config.paths.output_dir,
            f'training_history_fold_{fold}.csv'
        )
        save_training_history(history, history_path, fold)
        
        # Get best metrics from history
        best_epoch = np.argmin(history.history['val_loss'])
        best_val_loss = min(history.history['val_loss'])
        best_train_loss = history.history['loss'][best_epoch]
        
        # Compile results
        results = {
            'fold': fold,
            'best_epoch': int(best_epoch + 1),
            'best_val_loss': float(best_val_loss),
            'best_train_loss': float(best_train_loss),
            'final_val_loss': float(val_loss),
            'val_auc': float(val_auc),
            'model_path': model_path,
            'history_path': history_path,
            'log_dir': log_dir,
            'train_size': train_size,
            'val_size': val_size
        }
        
        self.logger.info(f"Fold {fold} results:")
        self.logger.info(f"  Best epoch: {results['best_epoch']}")
        self.logger.info(f"  Best val loss: {results['best_val_loss']:.4f}")
        self.logger.info(f"  Val AUC: {results['val_auc']:.4f}")
        
        # Clear memory
        del model
        clear_memory()
        
        return results
    
    def _calculate_validation_auc(self, model: tf.keras.Model, val_dataset: tf.data.Dataset) -> float:
        """
        Calculate AUC on validation dataset.
        
        Args:
            model: Trained model
            val_dataset: Validation dataset
            
        Returns:
            AUC score
        """
        y_true_list = []
        y_pred_list = []
        
        for batch in val_dataset:
            if isinstance(batch, dict):
                x_batch = batch['melspec']
                y_batch = batch['target']
            else:
                x_batch, y_batch = batch
            
            # Get predictions
            y_pred_batch = model(x_batch, training=False)
            y_pred_batch = tf.nn.sigmoid(y_pred_batch)  # Convert logits to probabilities
            
            y_true_list.append(y_batch.numpy())
            y_pred_list.append(y_pred_batch.numpy())
        
        # Concatenate all batches
        y_true = np.concatenate(y_true_list, axis=0)
        y_pred = np.concatenate(y_pred_list, axis=0)
        
        # Calculate AUC
        return calculate_auc_score(y_true, y_pred)
    
    def train_cross_validation(self, train_df: pd.DataFrame) -> dict:
        """
        Train models using cross-validation.
        
        Args:
            train_df: Training DataFrame
            
        Returns:
            Dictionary with aggregated results
        """
        self.logger.info("Starting cross-validation training")
        
        all_results = {}
        fold_scores = {
            'val_loss': [],
            'val_auc': [],
            'train_loss': []
        }
        
        for fold in self.config.training.selected_folds:
            self.logger.info(f"\n{'='*50}")
            self.logger.info(f"FOLD {fold}")
            self.logger.info(f"{'='*50}")
            
            try:
                fold_results = self.train_single_fold(fold, train_df)
                all_results[f'fold_{fold}'] = fold_results
                
                # Collect scores for averaging
                fold_scores['val_loss'].append(fold_results['best_val_loss'])
                fold_scores['val_auc'].append(fold_results['val_auc'])
                fold_scores['train_loss'].append(fold_results['best_train_loss'])
                
            except Exception as e:
                self.logger.error(f"Error in fold {fold}: {str(e)}")
                import traceback
                self.logger.error(traceback.format_exc())
                continue
        
        # Calculate aggregated metrics
        if fold_scores['val_loss']:
            aggregated_results = {
                'cv_results': {
                    'mean_val_loss': float(np.mean(fold_scores['val_loss'])),
                    'std_val_loss': float(np.std(fold_scores['val_loss'])),
                    'mean_val_auc': float(np.mean(fold_scores['val_auc'])),
                    'std_val_auc': float(np.std(fold_scores['val_auc'])),
                    'mean_train_loss': float(np.mean(fold_scores['train_loss'])),
                    'std_train_loss': float(np.std(fold_scores['train_loss']))
                },
                'individual_folds': all_results,
                'num_completed_folds': len(fold_scores['val_loss']),
                'training_config': {
                    'epochs': self.config.training.epochs,
                    'batch_size': self.config.training.batch_size,
                    'learning_rate': self.config.training.learning_rate,
                    'optimizer': self.config.training.optimizer,
                    'scheduler': self.config.training.scheduler
                }
            }
            
            if self.hyperparameters:
                aggregated_results['hyperparameters'] = self.hyperparameters
            
            self.logger.info(f"\nCross-validation completed!")
            self.logger.info(f"Mean val loss: {aggregated_results['cv_results']['mean_val_loss']:.4f} ± {aggregated_results['cv_results']['std_val_loss']:.4f}")
            self.logger.info(f"Mean val AUC: {aggregated_results['cv_results']['mean_val_auc']:.4f} ± {aggregated_results['cv_results']['std_val_auc']:.4f}")
            
            # Save results
            results_path = os.path.join(
                self.config.paths.output_dir,
                'cross_validation_results.json'
            )
            with open(results_path, 'w') as f:
                json.dump(aggregated_results, f, indent=2, default=str)
            
            self.logger.info(f"Results saved to {results_path}")
            
            return aggregated_results
        
        else:
            self.logger.error("No folds completed successfully")
            return {}
    
    def train_single_model(self, train_df: pd.DataFrame, fold: int = 0) -> dict:
        """
        Train a single model on one fold.
        
        Args:
            train_df: Training DataFrame
            fold: Fold to train on
            
        Returns:
            Training results
        """
        return self.train_single_fold(fold, train_df)


def load_hyperparameters(hp_path: str) -> dict:
    """
    Load hyperparameters from JSON file.
    
    Args:
        hp_path: Path to hyperparameters JSON file
        
    Returns:
        Hyperparameters dictionary
    """
    with open(hp_path, 'r') as f:
        data = json.load(f)
    
    # Extract hyperparameters based on file structure
    if 'best_overall_hyperparameters' in data:
        return data['best_overall_hyperparameters']
    elif 'best_hyperparameters' in data:
        return data['best_hyperparameters']
    else:
        return data


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='BirdCLEF Model Training')
    
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to configuration YAML file'
    )
    
    parser.add_argument(
        '--hyperparameters',
        type=str,
        default=None,
        help='Path to hyperparameters JSON file'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode'
    )
    
    parser.add_argument(
        '--fold',
        type=int,
        default=None,
        help='Train single fold only (0-4)'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Number of epochs'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Batch size'
    )
    
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=None,
        help='Learning rate'
    )
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_arguments()
    
    # Create or load configuration
    if args.config:
        config = load_config(args.config)
    else:
        config = create_default_config()
    
    # Load hyperparameters if provided
    hyperparameters = {}
    if args.hyperparameters:
        if os.path.exists(args.hyperparameters):
            hyperparameters = load_hyperparameters(args.hyperparameters)
            print(f"Loaded hyperparameters from {args.hyperparameters}")
        else:
            print(f"Warning: Hyperparameters file not found: {args.hyperparameters}")
    
    # Override configuration with command line arguments
    if args.debug:
        config.debug.debug = True
    
    if args.epochs:
        config.training.epochs = args.epochs
    
    if args.batch_size:
        config.training.batch_size = args.batch_size
    
    if args.learning_rate:
        config.training.learning_rate = args.learning_rate
    
    if args.fold is not None:
        config.training.selected_folds = [args.fold]
    
    # Apply debug settings
    if config.debug.debug:
        config.apply_debug_settings()
        print("Debug mode enabled")
    
    # Validate configuration
    try:
        config.validate()
    except ValueError as e:
        print(f"Configuration validation failed: {e}")
        return 1
    
    # Print configuration summary
    print("\nTraining Configuration:")
    print(f"Epochs: {config.training.epochs}")
    print(f"Batch size: {config.training.batch_size}")
    print(f"Learning rate: {config.training.learning_rate}")
    print(f"Optimizer: {config.training.optimizer}")
    print(f"Scheduler: {config.training.scheduler}")
    print(f"Selected folds: {config.training.selected_folds}")
    print(f"Debug mode: {config.debug.debug}")
    
    if hyperparameters:
        print(f"\nUsing hyperparameters: {hyperparameters}")
    
    # Load training data
    try:
        train_df = pd.read_csv(config.paths.train_csv)
        print(f"\nLoaded {len(train_df)} training samples")
    except Exception as e:
        print(f"Error loading training data: {e}")
        return 1
    
    # Create trainer
    trainer = BirdCLEFTrainer(config, hyperparameters)
    
    try:
        start_time = time.time()
        
        if len(config.training.selected_folds) == 1:
            # Train single fold
            fold = config.training.selected_folds[0]
            results = trainer.train_single_model(train_df, fold)
            print(f"\nSingle fold training completed!")
            print(f"Fold {fold} val AUC: {results['val_auc']:.4f}")
        else:
            # Train with cross-validation
            results = trainer.train_cross_validation(train_df)
            print(f"\nCross-validation training completed!")
            if results:
                print(f"Mean val AUC: {results['cv_results']['mean_val_auc']:.4f} ± {results['cv_results']['std_val_auc']:.4f}")
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"Total training time: {format_time(total_time)}")
        print(f"Results saved to: {config.paths.output_dir}")
        
        return 0
        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)