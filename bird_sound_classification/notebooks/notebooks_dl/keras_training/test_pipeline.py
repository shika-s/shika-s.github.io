#!/usr/bin/env python3
"""
Test script for BirdCLEF Keras training pipeline.
"""

import os
import sys
import pandas as pd
import numpy as np
import tensorflow as tf

def test_imports():
    """Test all module imports."""
    print("Testing imports...")
    
    try:
        from config import create_default_config
        print("✓ Config import successful")
        
        from utils import set_random_seeds, calculate_auc_score
        print("✓ Utils import successful")
        
        from data_pipeline import create_data_pipeline
        print("✓ Data pipeline import successful")
        
        from model_definitions import create_model
        print("✓ Model definitions import successful")
        
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config():
    """Test configuration creation and validation."""
    print("\nTesting configuration...")
    
    try:
        from config import create_default_config
        
        config = create_default_config()
        config.debug.debug = True
        config.apply_debug_settings()
        
        print(f"✓ Config created (debug mode: {config.debug.debug})")
        print(f"✓ Number of classes: {config.model.num_classes}")
        print(f"✓ Training epochs: {config.training.epochs}")
        
        # Test validation
        config.validate()
        print("✓ Config validation passed")
        
        return config
    except Exception as e:
        print(f"❌ Config error: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_data_pipeline(config):
    """Test data pipeline creation."""
    print("\nTesting data pipeline...")
    
    try:
        from data_pipeline import create_data_pipeline
        
        # Create pipeline
        pipeline = create_data_pipeline(config)
        print(f"✓ Data pipeline created")
        print(f"✓ Number of classes: {pipeline.num_classes}")
        
        # Check if training data exists
        if os.path.exists(config.paths.train_csv):
            print("✓ Training CSV found")
            
            # Load small sample
            train_df = pd.read_csv(config.paths.train_csv)
            print(f"✓ Loaded {len(train_df)} training samples")
            
            # Test with small subset
            test_df = train_df.head(5)
            
            # Test dataset creation
            dataset = pipeline.create_dataset(test_df, is_training=True)
            print("✓ Dataset creation successful")
            
            # Test iteration
            for batch in dataset.take(1):
                print(f"✓ Batch melspec shape: {batch['melspec'].shape}")
                print(f"✓ Batch target shape: {batch['target'].shape}")
                break
                
        else:
            print("⚠ Training CSV not found, skipping dataset tests")
        
        return True
    except Exception as e:
        print(f"❌ Data pipeline error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model(config):
    """Test model creation."""
    print("\nTesting model creation...")
    
    try:
        from model_definitions import create_model
        
        # Create model
        model = create_model(config)
        print("✓ Model created successfully")
        
        # Test forward pass
        dummy_input = tf.random.normal((2, *config.audio.target_shape, 1))
        output = model(dummy_input)
        print(f"✓ Forward pass successful")
        print(f"✓ Output shape: {output.shape}")
        print(f"✓ Expected shape: (2, {config.model.num_classes})")
        
        # Test model summary
        total_params = model.count_params()
        print(f"✓ Total parameters: {total_params:,}")
        
        return True
    except Exception as e:
        print(f"❌ Model error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_utilities():
    """Test utility functions."""
    print("\nTesting utilities...")
    
    try:
        from utils import set_random_seeds, calculate_auc_score
        
        # Test random seed setting
        set_random_seeds(42)
        print("✓ Random seeds set")
        
        # Test AUC calculation
        y_true = np.random.randint(0, 2, (50, 5))
        y_pred = np.random.random((50, 5))
        auc = calculate_auc_score(y_true, y_pred)
        print(f"✓ AUC calculation: {auc:.4f}")
        
        return True
    except Exception as e:
        print(f"❌ Utilities error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test function."""
    print("BirdCLEF Keras Pipeline Test")
    print("=" * 30)
    
    # Test environment
    print(f"Python version: {sys.version}")
    print(f"TensorFlow version: {tf.__version__}")
    print(f"GPU available: {len(tf.config.experimental.list_physical_devices('GPU')) > 0}")
    
    # Run tests
    tests_passed = 0
    total_tests = 5
    
    if test_imports():
        tests_passed += 1
    
    config = test_config()
    if config is not None:
        tests_passed += 1
        
        if test_utilities():
            tests_passed += 1
            
        if test_data_pipeline(config):
            tests_passed += 1
            
        if test_model(config):
            tests_passed += 1
    
    # Summary
    print(f"\nTest Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! Pipeline is ready for training.")
        return 0
    else:
        print("❌ Some tests failed. Check the errors above.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)