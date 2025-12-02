"""
Configuration module for the Trading Engine.

Includes:
- General settings (data, trading, risk)
- Model parameters (ML/DL hyperparameters)
- HFT-specific settings
"""

from .settings import (
    settings,
    Settings,
    DataSettings,
    TradingSettings,
    RiskSettings,
    SignalSettings,
    SentimentSettings,
    FeatureSettings,
    BacktestSettings,
)
from .model_params import (
    model_params,
    ModelParams,
    XGBoostParams,
    LightGBMParams,
    RandomForestParams,
    LSTMParams,
    TransformerParams,
    EnsembleParams,
    HyperoptParams,
)
from .hft_settings import (
    TradingMode,
    RiskLevel,
    HFTConfig,
    StatArbSettings,
    LOBImbalanceSettings,
    MarketMakingSettings,
    TechnicalSearchSettings,
    ArbitrageSettings,
    DeepLOBSettings,
    RiskSettings as HFTRiskSettings,
    ExecutionSettings,
    get_hft_config,
    get_paper_trading_config,
    get_backtesting_config,
    get_live_trading_config,
)

__all__ = [
    # General settings
    'settings',
    'Settings',
    'DataSettings',
    'TradingSettings',
    'RiskSettings',
    'SignalSettings',
    'SentimentSettings',
    'FeatureSettings',
    'BacktestSettings',
    # Model params
    'model_params',
    'ModelParams',
    'XGBoostParams',
    'LightGBMParams',
    'RandomForestParams',
    'LSTMParams',
    'TransformerParams',
    'EnsembleParams',
    'HyperoptParams',
    # HFT settings
    'TradingMode',
    'RiskLevel',
    'HFTConfig',
    'StatArbSettings',
    'LOBImbalanceSettings',
    'MarketMakingSettings',
    'TechnicalSearchSettings',
    'ArbitrageSettings',
    'DeepLOBSettings',
    'HFTRiskSettings',
    'ExecutionSettings',
    'get_hft_config',
    'get_paper_trading_config',
    'get_backtesting_config',
    'get_live_trading_config',
]
