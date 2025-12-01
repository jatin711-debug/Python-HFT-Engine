"""
Strategies module for institutional-grade trading.

Includes:
- Momentum strategies
- Mean reversion strategies
- Breakout strategies
- Regime-switching meta-strategies
- Strategy ensembles
"""

from .institutional_strategies import (
    StrategyType,
    StrategySignal,
    MomentumStrategy,
    MeanReversionStrategy,
    BreakoutStrategy,
    RegimeSwitchingStrategy,
    StrategyEnsemble,
)

__all__ = [
    'StrategyType',
    'StrategySignal',
    'MomentumStrategy',
    'MeanReversionStrategy',
    'BreakoutStrategy',
    'RegimeSwitchingStrategy',
    'StrategyEnsemble',
]
