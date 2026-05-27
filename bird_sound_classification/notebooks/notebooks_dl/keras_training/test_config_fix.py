#!/usr/bin/env python3
"""
Test script to verify config fixes.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import create_default_config
import tensorflow as tf

def test_config_hyperparams():
    """Test that hyperparameter configuration is properly initialized."""
    print("Testing configuration hyperparameters...")
    
    config = create_default_config()
    
    # Test that optimizers is properly set
    print(f"Optimizers: {config.hyperparams.optimizers}")
    assert config.hyperparams.optimizers is not None, "optimizers should not be None"
    assert len(config.hyperparams.optimizers) > 0, "optimizers should not be empty"
    
    # Test that schedulers is properly set
    print(f"Schedulers: {config.hyperparams.schedulers}")
    assert config.hyperparams.schedulers is not None, "schedulers should not be None"
    assert len(config.hyperparams.schedulers) > 0, "schedulers should not be empty"
    
    # Test that batch_sizes is properly set
    print(f"Batch sizes: {config.hyperparams.batch_sizes}")
    assert config.hyperparams.batch_sizes is not None, "batch_sizes should not be None"
    assert len(config.hyperparams.batch_sizes) > 0, "batch_sizes should not be empty"
    
    print("✓ All hyperparameter configuration tests passed!")
    return True

def test_gpu_detection():
    """Test GPU detection."""
    print("\nTesting GPU detection...")
    
    # Check GPU availability
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print(f"Number of GPUs detected: {len(gpus)}")
    
    if gpus:
        print("GPU devices:")
        for i, gpu in enumerate(gpus):
            print(f"  GPU {i}: {gpu.name}")
    else:
        print("No GPUs detected")
    
    # Check device strategy
    strategy = tf.distribute.get_strategy()
    print(f"Strategy: {type(strategy).__name__}")
    
    return True

if __name__ == "__main__":
    print("Running configuration and GPU tests...")
    
    try:
        test_config_hyperparams()
        test_gpu_detection()
        print("\n✅ All tests passed!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)