"""
Strategies module for institutional-grade trading.

Includes:
- Momentum strategies
- Mean reversion strategies
- Breakout strategies
- Regime-switching meta-strategies
- Strategy ensembles
- HFT strategies (Statistical Arbitrage, LOB Imbalance, Market Making)
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

from .hft_strategies import (
    HFTSignal,
    LOBSnapshot,
    StatisticalArbitrageEWLR,
    OrderBookImbalanceStrategy,
    IntelligentMarketMaker,
    TechnicalStrategySearch,
    IndexArbitrageStrategy,
    HFTStrategyEnsemble,
)

__all__ = [
    # Institutional strategies
    'StrategyType',
    'StrategySignal',
    'MomentumStrategy',
    'MeanReversionStrategy',
    'BreakoutStrategy',
    'RegimeSwitchingStrategy',
    'StrategyEnsemble',
    # HFT strategies
    'HFTSignal',
    'LOBSnapshot',
    'StatisticalArbitrageEWLR',
    'OrderBookImbalanceStrategy',
    'IntelligentMarketMaker',
    'TechnicalStrategySearch',
    'IndexArbitrageStrategy',
    'HFTStrategyEnsemble',
]
