"""
utils/seed.py
-------------
Global seed setter for reproducibility.
"""
import os
import random

import numpy as np


def set_global_seed(seed: int = 42) -> None:
    """Set seeds for Python, NumPy, and TensorFlow (if available)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass
