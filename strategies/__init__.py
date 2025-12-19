"""
Strategies module for institutional-grade trading.

Includes:
- Momentum strategies
- Mean reversion strategies
- Breakout strategies
- Regime-switching meta-strategies
- Strategy ensembles
- HFT strategies (Statistical Arbitrage, LOB Imbalance, Market Making)
- Micro strategies (1-5 second crypto trading)
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

from .micro_strategies import (
    MicroSignal,
    MicroConfig,
    MicroStrategyEnsemble,
    MomentumBurstStrategy,
    MicroMeanReversionStrategy,
    VolatilityBreakoutStrategy,
    OrderFlowEdgeStrategy,
    create_micro_ensemble,
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
    # Micro strategies
    'MicroSignal',
    'MicroConfig',
    'MicroStrategyEnsemble',
    'MomentumBurstStrategy',
    'MicroMeanReversionStrategy',
    'VolatilityBreakoutStrategy',
    'OrderFlowEdgeStrategy',
    'create_micro_ensemble',
]
