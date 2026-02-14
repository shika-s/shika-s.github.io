#!/bin/bash
#SBATCH --job-name=birdclef_hp_tuning
#SBATCH --partition=free-gpu
#SBATCH --gres=gpu:V100:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/hp_tuning_%j.out
#SBATCH --error=logs/hp_tuning_%j.err

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"

# Set environment variables for reproducibility
export PYTHONHASHSEED=42
export TF_DETERMINISTIC_OPS=1
export TF_CUDNN_DETERMINISTIC=1

# Set CUDA memory growth to avoid OOM errors
export TF_FORCE_GPU_ALLOW_GROWTH=true

# Load modules first
module load cuda/12.2.0
module load python/3.10.2

# Set environment variables for TensorFlow and CUDA
export CUDA_DIR=/opt/apps/cuda/12.2.0
export XLA_FLAGS="--xla_gpu_cuda_data_dir=${CUDA_DIR}/nvvm/libdevice"
echo "XLA_FLAGS = $TF_XLA_FLAGS"

# Set CUDA environment variables
export CUDA_HOME=/opt/apps/cuda/12.2.0
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Check GPU availability
nvidia-smi
echo "CUDA version: $(nvcc --version)"

# Navigate to project directory
cd /pub/ddlin/projects/mids/DATASCI207_Bird_Sounds/notebooks/notebooks_dl/keras_training

# Create logs directory if it doesn't exist
mkdir -p logs

# Activate poetry environment
echo "Activating poetry environment..."
source $(poetry env info --path)/bin/activate

# Update TensorFlow if needed
echo "Updating TensorFlow to compatible version..."
poetry update tensorflow

# Check Python and package versions
echo "Python version: $(python --version)"
echo "TensorFlow version: $(python -c 'import tensorflow as tf; print(tf.__version__)')"
echo "Keras Tuner version: $(python -c 'import keras_tuner as kt; print(kt.__version__)')"

# Default hyperparameter tuning arguments
TUNER_TYPE=${TUNER_TYPE:-"random_search"}
MAX_TRIALS=${MAX_TRIALS:-50}
EPOCHS=${EPOCHS:-10}
DEBUG=${DEBUG:-false}

echo "Hyperparameter tuning configuration:"
echo "  Tuner type: $TUNER_TYPE"
echo "  Max trials: $MAX_TRIALS"
echo "  Epochs per trial: $EPOCHS"
echo "  Debug mode: $DEBUG"

# Run hyperparameter tuning
echo "Starting hyperparameter tuning..."

# Build command with arguments
CMD="python train_keras_tuner.py"
CMD="$CMD --tuner-type $TUNER_TYPE"
CMD="$CMD --max-trials $MAX_TRIALS"
CMD="$CMD --epochs $EPOCHS"

if [ "$DEBUG" = true ]; then
    CMD="$CMD --debug"
fi

# Add config file if specified
if [ ! -z "$CONFIG_FILE" ]; then
    CMD="$CMD --config $CONFIG_FILE"
fi

# Add specific fold if specified
if [ ! -z "$FOLD" ]; then
    CMD="$CMD --fold $FOLD"
fi

echo "Running command: $CMD"
$CMD

# Check exit code
EXIT_CODE=$?
echo "Hyperparameter tuning exit code: $EXIT_CODE"

if [ $EXIT_CODE -eq 0 ]; then
    echo "Hyperparameter tuning completed successfully!"
    
    # Print results summary
    echo "Results location: $(pwd)/results/"
    if [ -f "results/hyperparameter_tuning_results.json" ]; then
        echo "Results summary:"
        python -c "
import json
try:
    with open('results/hyperparameter_tuning_results.json', 'r') as f:
        results = json.load(f)
    print(f\"Mean validation loss: {results.get('mean_val_loss', 'N/A'):.4f}\")
    print(f\"Best hyperparameters: {results.get('best_overall_hyperparameters', 'N/A')}\")
except Exception as e:
    print(f\"Could not read results: {e}\")
"
    fi
else
    echo "Hyperparameter tuning failed with exit code: $EXIT_CODE"
fi

echo "End time: $(date)"

# Print GPU memory usage
echo "Final GPU memory usage:"
nvidia-smi

exit $EXIT_CODE