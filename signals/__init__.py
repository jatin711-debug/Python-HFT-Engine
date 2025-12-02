"""
Signals package for trading signal generation.
"""

from .generator import SignalGenerator, SignalConfig, TradingSignal, SignalType, PositionType

# HFT Unified Signal Aggregator (optional)
try:
    from .unified_signal_aggregator import (
        UnifiedSignalAggregator, 
        AggregatorConfig, 
        UnifiedSignal,
        SignalSource,
        ExecutionPath,
    )
    HFT_AGGREGATOR_AVAILABLE = True
except ImportError:
    HFT_AGGREGATOR_AVAILABLE = False

__all__ = [
    'SignalGenerator', 
    'SignalConfig', 
    'TradingSignal', 
    'SignalType', 
    'PositionType',
    'UnifiedSignalAggregator',
    'AggregatorConfig',
    'UnifiedSignal',
    'SignalSource',
    'ExecutionPath',
    'HFT_AGGREGATOR_AVAILABLE',
]
