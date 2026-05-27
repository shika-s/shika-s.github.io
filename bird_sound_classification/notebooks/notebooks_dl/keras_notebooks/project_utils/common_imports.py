"""Common imports and global configurations for the bird sounds project.

This module centralizes all standard imports and sets up global policies like mixed precision.
It includes basic, data processing, ML, and DL libraries available in the environment.
Note: No internet access, so only pre-installed packages are imported here.
"""

# Standard library
import gc
import logging
import os
import random
import sys
import time
import json
import math
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Generator

# Data science stack
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

# ML imports
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

# Deep learning
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import layers, models, metrics
from tensorflow.keras.callbacks import (
    CSVLogger, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
)
from tensorflow.keras.utils import to_categorical

# Updated: Added missing imports for ViT transfer learning
import keras_cv
import keras_hub
import huggingface_hub

# Configuration
from tensorflow.keras import mixed_precision

# Setup global policies and warnings
mixed_precision.set_global_policy("mixed_float16")  # Halves memory usage for efficiency
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)
# os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'