#!/bin/bash
# Convenience script for submitting BirdCLEF training jobs

set -e

# Default values
JOB_TYPE=""
DEBUG=false
TUNER_TYPE="random_search"
MAX_TRIALS=50
EPOCHS=""
BATCH_SIZE=32
LEARNING_RATE=5e-4
CONFIG_FILE=""
HYPERPARAMETERS_FILE=""
FOLD=""

# Function to display usage
usage() {
    echo "Usage: $0 [OPTIONS] JOB_TYPE"
    echo ""
    echo "JOB_TYPE:"
    echo "  tune    - Run hyperparameter tuning"
    echo "  train   - Run model training"
    echo "  setup   - Setup environment (no GPU needed)"
    echo ""
    echo "OPTIONS:"
    echo "  -d, --debug                Enable debug mode"
    echo "  -t, --tuner-type TYPE      Tuner type (random_search, hyperband, bayesian)"
    echo "  -n, --max-trials NUM       Maximum number of trials for tuning"
    echo "  -e, --epochs NUM           Number of epochs"
    echo "  -b, --batch-size NUM       Batch size"
    echo "  -l, --learning-rate FLOAT  Learning rate"
    echo "  -c, --config FILE          Configuration YAML file"
    echo "  -h, --hyperparameters FILE Hyperparameters JSON file"
    echo "  -f, --fold NUM             Specific fold to train (0-4)"
    echo "  --help                     Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 setup                           # Setup environment"
    echo "  $0 tune                            # Basic hyperparameter tuning"
    echo "  $0 --debug tune                    # Debug hyperparameter tuning"
    echo "  $0 --epochs 5 --max-trials 10 tune # Quick tuning"
    echo "  $0 train                           # Train with best hyperparameters"
    echo "  $0 --fold 0 train                  # Train only fold 0"
    echo "  $0 --debug --epochs 2 train        # Debug training"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--debug)
            DEBUG=true
            shift
            ;;
        -t|--tuner-type)
            TUNER_TYPE="$2"
            shift 2
            ;;
        -n|--max-trials)
            MAX_TRIALS="$2"
            shift 2
            ;;
        -e|--epochs)
            EPOCHS="$2"
            shift 2
            ;;
        -b|--batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        -l|--learning-rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        -c|--config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        -h|--hyperparameters)
            HYPERPARAMETERS_FILE="$2"
            shift 2
            ;;
        -f|--fold)
            FOLD="$2"
            shift 2
            ;;
        --help)
            usage
            exit 0
            ;;
        tune|train|setup)
            JOB_TYPE="$1"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Check if job type is specified
if [ -z "$JOB_TYPE" ]; then
    echo "Error: Job type not specified"
    usage
    exit 1
fi

# Navigate to script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR/.."

echo "BirdCLEF Job Submission"
echo "======================="
echo "Job type: $JOB_TYPE"
echo "Debug mode: $DEBUG"

# Set default epochs based on job type
if [ -z "$EPOCHS" ]; then
    if [ "$JOB_TYPE" = "tune" ]; then
        EPOCHS=10
    else
        EPOCHS=20
    fi
fi

# Handle different job types
case $JOB_TYPE in
    setup)
        echo "Setting up environment..."
        bash slurm_scripts/setup_environment.sh
        exit $?
        ;;
    
    tune)
        echo "Submitting hyperparameter tuning job..."
        echo "Configuration:"
        echo "  Tuner type: $TUNER_TYPE"
        echo "  Max trials: $MAX_TRIALS"
        echo "  Epochs: $EPOCHS"
        
        # Set environment variables
        export DEBUG=$DEBUG
        export TUNER_TYPE=$TUNER_TYPE
        export MAX_TRIALS=$MAX_TRIALS
        export EPOCHS=$EPOCHS
        
        if [ ! -z "$CONFIG_FILE" ]; then
            export CONFIG_FILE=$CONFIG_FILE
            echo "  Config file: $CONFIG_FILE"
        fi
        
        if [ ! -z "$FOLD" ]; then
            export FOLD=$FOLD
            echo "  Fold: $FOLD"
        fi
        
        # Submit job
        sbatch -p free-gpu --gres=gpu:V100:1 slurm_scripts/run_hyperparameter_tuning.sh
        ;;
    
    train)
        echo "Submitting training job..."
        echo "Configuration:"
        echo "  Epochs: $EPOCHS"
        echo "  Batch size: $BATCH_SIZE"
        echo "  Learning rate: $LEARNING_RATE"
        
        # Set environment variables
        export DEBUG=$DEBUG
        export EPOCHS=$EPOCHS
        export BATCH_SIZE=$BATCH_SIZE
        export LEARNING_RATE=$LEARNING_RATE
        
        if [ ! -z "$CONFIG_FILE" ]; then
            export CONFIG_FILE=$CONFIG_FILE
            echo "  Config file: $CONFIG_FILE"
        fi
        
        if [ ! -z "$HYPERPARAMETERS_FILE" ]; then
            export HYPERPARAMETERS_FILE=$HYPERPARAMETERS_FILE
            echo "  Hyperparameters file: $HYPERPARAMETERS_FILE"
        fi
        
        if [ ! -z "$FOLD" ]; then
            export FOLD=$FOLD
            echo "  Fold: $FOLD"
        fi
        
        # Submit job
        sbatch -p free-gpu --gres=gpu:V100:1 slurm_scripts/run_training.sh
        ;;
    
    *)
        echo "Error: Unknown job type: $JOB_TYPE"
        usage
        exit 1
        ;;
esac

echo ""
echo "Job submitted! Check status with: squeue -u $USER"
echo "View logs in: logs/"