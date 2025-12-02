"""
Deep Learning Models for HFT Trading

Contains neural network architectures specifically designed for
high-frequency trading applications, including:
- DeepLOB: CNN-LSTM hybrid for limit order book analysis
- Feature generators for order book data
"""

from .deeplob import (
    DeepLOBConfig,
    DeepLOBModel,
    InceptionModule,
    DeepLOBTrainer,
    LOBFeatureGenerator,
)

__all__ = [
    'DeepLOBConfig',
    'DeepLOBModel',
    'InceptionModule',
    'DeepLOBTrainer',
    'LOBFeatureGenerator',
]
