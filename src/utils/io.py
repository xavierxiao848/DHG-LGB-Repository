"""
File I/O utilities for DHG-LGB framework.

Provides standardized loading and saving functions for various file formats.
"""

import os
import pickle
import numpy as np
import yaml
from typing import Any, Dict


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load YAML configuration file.

    Parameters
    ----------
    config_path : str
        Path to YAML config file

    Returns
    -------
    dict
        Configuration dictionary

    Examples
    --------
    >>> config = load_config('config/config.yaml')
    >>> hgnn_params = config['hgnn']
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def save_pickle(obj: Any, filepath: str) -> None:
    """Save object to pickle file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(filepath: str) -> Any:
    """Load object from pickle file."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def save_numpy(array: np.ndarray, filepath: str, fmt: str = '%.8f') -> None:
    """
    Save numpy array to text file.

    Parameters
    ----------
    array : np.ndarray
        Array to save
    filepath : str
        Output file path
    fmt : str, default='%.8f'
        Format for floating point numbers
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    np.savetxt(filepath, array, fmt=fmt)


def load_numpy(filepath: str, dtype: type = np.float32) -> np.ndarray:
    """
    Load numpy array from text file.

    Parameters
    ----------
    filepath : str
        Input file path
    dtype : type, default=np.float32
        Data type for array

    Returns
    -------
    np.ndarray
        Loaded array
    """
    return np.loadtxt(filepath, dtype=dtype)
