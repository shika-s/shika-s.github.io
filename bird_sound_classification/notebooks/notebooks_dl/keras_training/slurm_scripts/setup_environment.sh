#!/bin/bash
# Environment setup script for BirdCLEF Keras training
# This script should be run once to set up the environment

set -e  # Exit on error

echo "Setting up BirdCLEF Keras training environment..."

# Navigate to project root
PROJECT_ROOT="/pub/ddlin/projects/mids/DATASCI207_Bird_Sounds"
cd "$PROJECT_ROOT"

echo "Project directory: $(pwd)"

# Check if poetry is available
if ! command -v poetry &> /dev/null; then
    echo "Error: Poetry is not installed or not in PATH"
    echo "Please install Poetry first: https://python-poetry.org/docs/#installation"
    exit 1
fi

echo "Poetry version: $(poetry --version)"

# Install dependencies
echo "Installing Python dependencies..."
poetry install

# Check if CUDA is available
echo "Checking CUDA availability..."
nvidia-smi || echo "Warning: NVIDIA GPU not detected"

# Check Python and key packages
echo "Checking Python environment..."
poetry run python -c "
import sys
import tensorflow as tf
import keras_tuner as kt
import librosa
import cv2

print(f'Python version: {sys.version}')
print(f'TensorFlow version: {tf.__version__}')
print(f'Keras Tuner version: {kt.__version__}')
print(f'Librosa version: {librosa.__version__}')
print(f'OpenCV version: {cv2.__version__}')

# Check GPU availability
gpus = tf.config.experimental.list_physical_devices('GPU')
print(f'Number of GPUs: {len(gpus)}')
for i, gpu in enumerate(gpus):
    print(f'  GPU {i}: {gpu}')

if gpus:
    print('GPU support: Available')
else:
    print('GPU support: Not available (CPU only)')
"

# Create necessary directories
echo "Creating output directories..."
mkdir -p "$PROJECT_ROOT/notebooks/notebooks_dl/keras_training/logs"
mkdir -p "$PROJECT_ROOT/data/working/keras_training_output"
mkdir -p "$PROJECT_ROOT/data/working/keras_training_output/models"
mkdir -p "$PROJECT_ROOT/data/working/keras_training_output/logs"
mkdir -p "$PROJECT_ROOT/data/working/keras_training_output/keras_tuner"

# Make scripts executable
echo "Making scripts executable..."
chmod +x "$PROJECT_ROOT/notebooks/notebooks_dl/keras_training/slurm_scripts/"*.sh
chmod +x "$PROJECT_ROOT/notebooks/notebooks_dl/keras_training/"*.py

# Test configuration loading
echo "Testing configuration..."
cd "$PROJECT_ROOT/notebooks/notebooks_dl/keras_training"
poetry run python -c "
from config import create_default_config
try:
    config = create_default_config()
    config.validate()
    print('Configuration test: PASSED')
    print(f'Number of classes: {config.model.num_classes}')
    print(f'Data paths exist: {config.paths.train_csv}')
except Exception as e:
    print(f'Configuration test: FAILED - {e}')
    exit(1)
"

echo "Environment setup completed successfully!"
echo ""
echo "Usage examples:"
echo "1. Hyperparameter tuning:"
echo "   sbatch -p free-gpu --gres=gpu:V100:1 slurm_scripts/run_hyperparameter_tuning.sh"
echo ""
echo "2. Model training:"
echo "   sbatch -p free-gpu --gres=gpu:V100:1 slurm_scripts/run_training.sh"
echo ""
echo "3. Debug mode (faster):"
echo "   DEBUG=true sbatch -p free-gpu --gres=gpu:V100:1 slurm_scripts/run_hyperparameter_tuning.sh"
echo ""
echo "4. Specific configuration:"
echo "   CONFIG_FILE=custom_config.yaml sbatch -p free-gpu --gres=gpu:V100:1 slurm_scripts/run_training.sh"
echo ""
echo "Environment variables you can set:"
echo "  - DEBUG: true/false (default: false)"
echo "  - TUNER_TYPE: random_search/hyperband/bayesian (default: random_search)"
echo "  - MAX_TRIALS: number (default: 50)"
echo "  - EPOCHS: number (default: 10 for tuning, 20 for training)"
echo "  - BATCH_SIZE: number (default: 32)"
echo "  - LEARNING_RATE: float (default: 5e-4)"
echo "  - CONFIG_FILE: path to YAML config file"
echo "  - HYPERPARAMETERS_FILE: path to JSON hyperparameters file"
echo "  - FOLD: fold number to train (default: all folds)"