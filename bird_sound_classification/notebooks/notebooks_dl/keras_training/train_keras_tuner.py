#!/usr/bin/env python3
"""
Hyperparameter optimization for BirdCLEF 2025 using Keras Tuner.
"""

import os
import sys
import argparse
import json
import time
import gc
import psutil
from pathlib import Path
import pandas as pd
import numpy as np
import tensorflow as tf
import keras_tuner as kt
from datetime import datetime

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config, create_default_config, load_config
from data_pipeline import create_data_pipeline
from model_definitions import HyperModel, create_model, create_callbacks
from utils import (
    set_random_seeds,
    setup_logging,
    clear_memory,
    Timer,
    format_time,
    monitor_gpu_memory
)


def log_memory_usage(stage=""):
    """Log current memory usage."""
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024
    print(f"Memory usage {stage}: {memory_mb:.2f} MB")
    return memory_mb


class BirdCLEFTuner:
    """
    Main class for hyperparameter tuning of BirdCLEF models.
    """

    def __init__(self, config: Config):
        self.config = config
        self.logger = None
        self.data_pipeline = None
        self.tuner = None

        # Setup
        self._setup_environment()
        self._setup_logging()
        self._setup_data_pipeline()

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
        print(f"Keras Tuner version: {kt.__version__}")
        print(f"Number of GPUs: {len(tf.config.experimental.list_physical_devices('GPU'))}")
        print(f"Mixed precision: {self.config.model.use_mixed_precision}")
        print(f"Strategy: {type(self.strategy).__name__}")

        # Monitor GPU memory
        monitor_gpu_memory()
        log_memory_usage("after environment setup")

    def _setup_logging(self):
        """Setup logging."""
        self.logger = setup_logging(self.config.paths.logs_dir, 'INFO')
        self.logger.info("Starting hyperparameter tuning")
        self.logger.info(f"Configuration: {self.config}")

    def _setup_data_pipeline(self):
        """Setup data pipeline."""
        self.data_pipeline = create_data_pipeline(self.config)
        self.logger.info("Data pipeline setup completed")
        log_memory_usage("after data pipeline setup")

    def create_tuner(self, train_dataset: tf.data.Dataset) -> kt.Tuner:
        """
        Create Keras Tuner instance.
        """
        hypermodel = HyperModel(self.config)
        tuner_dir = os.path.join(
            self.config.paths.tuner_dir,
            f"tuner_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

        tuner_class_map = {
            'random_search': kt.RandomSearch,
            'hyperband': kt.Hyperband,
            'bayesian': kt.BayesianOptimization
        }

        if self.config.tuner.tuner_type not in tuner_class_map:
            raise ValueError(f"Tuner type {self.config.tuner.tuner_type} not supported")

        tuner_class = tuner_class_map[self.config.tuner.tuner_type]
        tuner_kwargs = {
            'hypermodel': hypermodel,
            'objective': kt.Objective(self.config.tuner.objective, direction=self.config.tuner.direction),
            'max_trials': self.config.tuner.max_trials,
            'directory': tuner_dir,
            'project_name': 'birdclef_tuning',
            'overwrite': True
        }

        if self.config.tuner.tuner_type == 'hyperband':
            tuner_kwargs['max_epochs'] = self.config.training.epochs
            tuner_kwargs['hyperband_iterations'] = self.config.tuner.hyperband_iterations
        else:
            tuner_kwargs['executions_per_trial'] = self.config.tuner.executions_per_trial

        tuner = tuner_class(**tuner_kwargs)
        self.logger.info(f"Created {self.config.tuner.tuner_type} tuner with {self.config.tuner.max_trials} trials")
        log_memory_usage("after tuner creation")
        return tuner

    def run_single_fold_tuning(self, fold: int = 0) -> dict:
        """
        Run hyperparameter tuning on a single fold.
        """
        self.logger.info(f"Starting hyperparameter tuning on fold {fold}")
        log_memory_usage("at start of fold tuning")

        train_df = pd.read_csv(self.config.paths.train_csv)
        self.logger.info(f"Loaded {len(train_df)} training samples")
        log_memory_usage("after loading training data")

        with Timer("Dataset creation"):
            train_dataset, val_dataset = self.data_pipeline.create_fold_datasets(
                train_df, fold
            )
        log_memory_usage("after dataset creation")

        # Create tuner
        tuner = self.create_tuner(train_dataset)

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=self.config.tuner.trial_patience,
            restore_best_weights=True
        )

        with Timer("Hyperparameter search"):
            tuner.search(
                train_dataset,
                validation_data=val_dataset,
                epochs=self.config.training.epochs,
                callbacks=[early_stopping],
                verbose=1
            )
        log_memory_usage("after hyperparameter search")

        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        self.logger.info("Best hyperparameters found:")
        for key, value in best_hps.values.items():
            self.logger.info(f"  {key}: {value}")

        best_model = tuner.get_best_models(num_models=1)[0]
        self.logger.info("Evaluating best model...")
        evaluation_results = best_model.evaluate(val_dataset, verbose=0, return_dict=True)
        val_loss = evaluation_results['loss']

        results = {
            'fold': fold,
            'best_hyperparameters': best_hps.values,
            'best_val_loss': val_loss,
            'best_val_metrics': evaluation_results,
            'tuner_directory': tuner.directory,
        }

        hp_save_path = os.path.join(
            self.config.paths.output_dir,
            f'best_hyperparameters_fold_{fold}.json'
        )
        with open(hp_save_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        self.logger.info(f"Best hyperparameters saved to {hp_save_path}")

        clear_memory()
        tf.keras.backend.clear_session()
        gc.collect()
        log_memory_usage("after memory cleanup")

        return results

    def run_full_tuning(self) -> dict:
        """
        Run hyperparameter tuning across all selected folds.
        """
        self.logger.info("Starting full hyperparameter tuning")
        log_memory_usage("at start of full tuning")

        all_results = {}
        best_scores = []

        for fold in self.config.training.selected_folds:
            self.logger.info(f"\n{'='*50}")
            self.logger.info(f"FOLD {fold}")
            self.logger.info(f"{'='*50}")

            try:
                fold_results = self.run_single_fold_tuning(fold)
                all_results[f'fold_{fold}'] = fold_results
                best_scores.append(fold_results['best_val_loss'])
                self.logger.info(f"Fold {fold} completed with best val_loss: {fold_results['best_val_loss']:.4f}")

            except Exception as e:
                self.logger.error(f"Error in fold {fold}: {str(e)}")
                import traceback
                self.logger.error(traceback.format_exc())

            finally:
                # Cleanup even on error or success
                tf.keras.backend.clear_session()
                gc.collect()
                log_memory_usage(f"after fold {fold} cleanup")

        if best_scores:
            aggregated_results = {
                'mean_val_loss': float(np.mean(best_scores)),
                'std_val_loss': float(np.std(best_scores)),
                'individual_folds': all_results,
            }
            best_fold = min(all_results.keys(), key=lambda x: all_results[x]['best_val_loss'])
            aggregated_results['best_overall_hyperparameters'] = all_results[best_fold]['best_hyperparameters']

            self.logger.info(f"\nTuning completed!")
            self.logger.info(f"Mean validation loss: {aggregated_results['mean_val_loss']:.4f} ± {aggregated_results['std_val_loss']:.4f}")

            results_save_path = os.path.join(
                self.config.paths.output_dir,
                'hyperparameter_tuning_results.json'
            )
            with open(results_save_path, 'w') as f:
                json.dump(aggregated_results, f, indent=2, default=str)
            self.logger.info(f"Aggregated results saved to {results_save_path}")

            return aggregated_results
        else:
            self.logger.error("No folds completed successfully")
            return {}

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='BirdCLEF Hyperparameter Tuning')
    parser.add_argument('--config', type=str, default=None, help='Path to configuration YAML file')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--fold', type=int, default=None, help='Run tuning on specific fold only (0-4)')
    parser.add_argument('--tuner-type', type=str, choices=['random_search', 'hyperband', 'bayesian'], default=None, help='Type of tuner to use')
    parser.add_argument('--max-trials', type=int, default=None, help='Maximum number of trials')
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs per trial')
    return parser.parse_args()


def main():
    """Main function."""
    # --- ADDED: GPU Check ---
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        print(f"✅ GPU(s) found: {len(gpus)}")
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("   Memory growth enabled.")
        except RuntimeError as e:
            print(f"   Error setting memory growth: {e}")
    else:
        print("❌ No GPU found. Running on CPU.")
    # --- End of Additions ---

    args = parse_arguments()
    log_memory_usage("at start of main")

    config = load_config(args.config) if args.config else create_default_config()

    if args.debug: config.debug.debug = True
    if args.tuner_type: config.tuner.tuner_type = args.tuner_type
    if args.max_trials: config.tuner.max_trials = args.max_trials
    if args.epochs: config.training.epochs = args.epochs
    if config.debug.debug:
        config.apply_debug_settings()
        print("Debug mode enabled")

    try:
        config.validate()
    except ValueError as e:
        print(f"Configuration validation failed: {e}")
        return 1

    tuner = BirdCLEFTuner(config)
    
    try:
        start_time = time.time()
        if args.fold is not None:
            results = tuner.run_single_fold_tuning(args.fold)
        else:
            results = tuner.run_full_tuning()
        
        total_time = time.time() - start_time
        print(f"\nHyperparameter tuning completed!")
        print(f"Total time: {format_time(total_time)}")
        
        return 0
    except Exception as e:
        print(f"Error during hyperparameter tuning: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())