"""
DeepGIS: A deep learning toolkit for remote sensing image analysis.

This package provides tools for:
- Remote sensing image classification
- Model training and evaluation
- Prediction and analysis
"""

from .config.config import Config
from .model.model_factory import load_model
from .trainer import Trainer
from .predictor import Predictor
from .analysis import ApplicationAnalyzer

__all__ = [
    'Config',
    'load_model',
    'Trainer',
    'Predictor'
    'ApplicationAnalyzer',
]

__version__ = '0.1.0'