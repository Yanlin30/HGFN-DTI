import os
import time
import random
import numpy as np


def seed_tf(seed=0):
    # Set random seed for Python's random module
    import tensorflow as tf
    random.seed(seed)
    
    # Set random seed for NumPy
    np.random.seed(seed)
    
    # Set random seed for TensorFlow
    tf.random.set_seed(seed)
    
    # Configure GPU for deterministic operations (if available)
    # Note: TensorFlow does not have a direct equivalent to PyTorch's cudnn deterministic settings
    # But we can control some deterministic operations using the following configuration:
    
    # Disable certain optimizations for reproducibility on GPU
    # These settings might affect performance but are useful for determinism
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
        # Optionally, configure GPU for deterministic operations if using certain libraries
        tf.config.set_logical_device_configuration(physical_devices[0], [])

# Example usage


def seed_torch(seed=0):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
