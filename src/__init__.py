"""
DHG-LGB: Disease-Hypergraph Integrated with LightGBM

A framework for predicting metabolite-disease associations using
hypergraph neural networks and gradient boosting.

Usage:
    # Import specific modules as needed
    from src.models import HGNNModel, LightGBMClassifier
    from src.evaluation import compute_metrics
    from src.utils import load_config
"""

__version__ = '1.0.0'
__author__ = 'Your Name'
__email__ = 'your.email@institution.edu'

# Lazy imports - submodules are imported only when accessed
# This allows the package to be imported even if some dependencies are missing
__all__ = [
    'preprocessing',
    'models',
    'training',
    'evaluation',
    'visualization',
    'utils',
]
