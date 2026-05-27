#!/bin/bash
#SBATCH --job-name=birdclef_training
#SBATCH --partition=free-gpu
#SBATCH --gres=gpu:V100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/training_%j.out
#SBATCH --error=logs/training_%j.err

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

# Check Python and package versions
echo "Python version: $(python --version)"
echo "TensorFlow version: $(python -c 'import tensorflow as tf; print(tf.__version__)')"

# Default training arguments
EPOCHS=${EPOCHS:-20}
BATCH_SIZE=${BATCH_SIZE:-32}
LEARNING_RATE=${LEARNING_RATE:-5e-4}
DEBUG=${DEBUG:-false}

echo "Training configuration:"
echo "  Epochs: $EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  Learning rate: $LEARNING_RATE"
echo "  Debug mode: $DEBUG"

# Check for hyperparameters file
HP_FILE=""
if [ ! -z "$HYPERPARAMETERS_FILE" ] && [ -f "$HYPERPARAMETERS_FILE" ]; then
    HP_FILE="$HYPERPARAMETERS_FILE"
    echo "  Using hyperparameters from: $HP_FILE"
elif [ -f "results/hyperparameter_tuning_results.json" ]; then
    HP_FILE="results/hyperparameter_tuning_results.json"
    echo "  Using hyperparameters from: $HP_FILE"
else
    echo "  No hyperparameters file found, using defaults"
fi

# Run training
echo "Starting model training..."

# Build command with arguments
CMD="python train_single_model.py"
CMD="$CMD --epochs $EPOCHS"
CMD="$CMD --batch-size $BATCH_SIZE"
CMD="$CMD --learning-rate $LEARNING_RATE"

if [ "$DEBUG" = true ]; then
    CMD="$CMD --debug"
fi

# Add config file if specified
if [ ! -z "$CONFIG_FILE" ]; then
    CMD="$CMD --config $CONFIG_FILE"
fi

# Add hyperparameters file if found
if [ ! -z "$HP_FILE" ]; then
    CMD="$CMD --hyperparameters $HP_FILE"
fi

# Add specific fold if specified
if [ ! -z "$FOLD" ]; then
    CMD="$CMD --fold $FOLD"
fi

echo "Running command: $CMD"
$CMD

# Check exit code
EXIT_CODE=$?
echo "Training exit code: $EXIT_CODE"

if [ $EXIT_CODE -eq 0 ]; then
    echo "Training completed successfully!"
    
    # Print results summary
    echo "Results location: $(pwd)/results/"
    if [ -f "results/cross_validation_results.json" ]; then
        echo "Cross-validation results summary:"
        python -c "
import json
try:
    with open('results/cross_validation_results.json', 'r') as f:
        results = json.load(f)
    cv_results = results.get('cv_results', {})
    print(f\"Mean val AUC: {cv_results.get('mean_val_auc', 'N/A'):.4f} ± {cv_results.get('std_val_auc', 'N/A'):.4f}\")
    print(f\"Mean val loss: {cv_results.get('mean_val_loss', 'N/A'):.4f} ± {cv_results.get('std_val_loss', 'N/A'):.4f}\")
    print(f\"Completed folds: {results.get('num_completed_folds', 'N/A')}\")
except Exception as e:
    print(f\"Could not read results: {e}\")
"
    fi
    
    # List saved models
    echo "Saved models:"
    ls -la results/models/ 2>/dev/null || echo "No models directory found"
    
else
    echo "Training failed with exit code: $EXIT_CODE"
fi

echo "End time: $(date)"

# Print GPU memory usage
echo "Final GPU memory usage:"
nvidia-smi

exit $EXIT_CODE