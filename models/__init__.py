"""
Models package for ML-based trading predictions.

Contains gradient boosting, ensemble models, and deep learning modules.
"""

from typing import Dict, Any

# Import from submodules
from .ml.gradient_boost import GradientBoostingModels
from .ml.ensemble import EnsembleModel

# Deep learning imports (optional - requires PyTorch)
try:
    from .deep_learning import (
        DeepLOBConfig,
        DeepLOBModel,
        DeepLOBTrainer,
        LOBFeatureGenerator,
    )
    _HAS_DEEP_LEARNING = True
except ImportError:
    _HAS_DEEP_LEARNING = False
    DeepLOBConfig = None
    DeepLOBModel = None
    DeepLOBTrainer = None
    LOBFeatureGenerator = None

__all__ = [
    'GradientBoostingModels',
    'EnsembleModel',
    # Deep learning (optional)
    'DeepLOBConfig',
    'DeepLOBModel',
    'DeepLOBTrainer',
    'LOBFeatureGenerator',
    '_HAS_DEEP_LEARNING',
]
