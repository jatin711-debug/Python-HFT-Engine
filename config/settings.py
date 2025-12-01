"""
Global settings and configuration for the Trading Engine.

This module contains all configurable parameters for the trading system,
including data sources, trading parameters, risk limits, and API keys.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from pathlib import Path
import os


@dataclass
class DataSettings:
    """Settings for data sources and storage."""
    
    # Data directories
    base_dir: Path = Path("data")
    cache_dir: Path = Path("data/cache")
    
    # Market data settings
    default_source: str = "yfinance"  # Options: yfinance, alpha_vantage
    default_interval: str = "1d"       # Options: 1m, 5m, 15m, 1h, 1d
    lookback_days: int = 365 * 5       # 5 years of historical data
    
    # API Keys (load from environment)
    alpha_vantage_key: str = field(
        default_factory=lambda: os.getenv("ALPHA_VANTAGE_API_KEY", "")
    )
    news_api_key: str = field(
        default_factory=lambda: os.getenv("NEWS_API_KEY", "")
    )


@dataclass
class TradingSettings:
    """Settings for trading execution."""
    
    # Initial capital
    initial_capital: float = 100_000.0
    
    # Transaction costs
    commission_pct: float = 0.001      # 0.1% per trade
    slippage_pct: float = 0.0005       # 0.05% slippage
    
    # Position sizing
    max_position_pct: float = 0.10     # Max 10% of portfolio per position
    min_position_pct: float = 0.02     # Min 2% of portfolio per position
    
    # Leverage
    max_leverage: float = 1.0          # No leverage by default
    allow_shorting: bool = True        # Allow short positions
    
    # Trade frequency
    min_holding_period: int = 1        # Minimum days to hold position
    max_trades_per_day: int = 10       # Maximum trades per day


@dataclass
class RiskSettings:
    """Settings for risk management."""
    
    # Stop loss / Take profit
    default_stop_loss_atr: float = 2.0      # Stop loss at 2x ATR
    default_take_profit_atr: float = 3.0    # Take profit at 3x ATR
    use_trailing_stop: bool = True
    trailing_stop_atr: float = 1.5
    
    # Portfolio risk limits
    max_portfolio_risk: float = 0.02        # Risk 2% per trade
    max_daily_loss: float = 0.03            # Max 3% daily loss
    max_weekly_loss: float = 0.05           # Max 5% weekly loss
    max_drawdown: float = 0.15              # Pause at 15% drawdown
    
    # Position limits
    max_sector_exposure: float = 0.30       # Max 30% in one sector
    max_correlation: float = 0.70           # Max correlation between positions
    
    # Volatility adjustments
    high_volatility_threshold: float = 30   # VIX level for high vol
    low_volatility_threshold: float = 15    # VIX level for low vol
    volatility_size_adjustment: float = 0.5 # Reduce size by 50% in high vol


@dataclass
class SignalSettings:
    """Settings for signal generation."""
    
    # Signal thresholds (adjusted for realistic ML probability outputs)
    strong_buy_threshold: float = 0.65      # Strong buy above 65%
    buy_threshold: float = 0.52             # Buy above 52% (slight edge)
    sell_threshold: float = 0.48            # Sell below 48%
    strong_sell_threshold: float = 0.35     # Strong sell below 35%
    
    # Signal values
    strong_buy_value: int = 2
    buy_value: int = 1
    hold_value: int = 0
    sell_value: int = -1
    strong_sell_value: int = -2
    
    # Minimum confidence to trade (lowered for more signals)
    min_confidence: float = 0.30
    
    # Signal smoothing
    signal_smoothing_window: int = 3        # Days to smooth signals
    
    # Sentiment weight in final decision
    sentiment_weight: float = 0.20          # 20% weight for sentiment


@dataclass
class SentimentSettings:
    """Settings for sentiment analysis."""
    
    # News fetching
    max_articles_per_symbol: int = 100
    news_lookback_days: int = 7
    min_article_length: int = 100           # Minimum words in article
    
    # News sources (priority order)
    news_sources: List[str] = field(default_factory=lambda: [
        "google_news",
        "yahoo_finance",
        "reuters",
        "bloomberg",
        "marketwatch",
        "seekingalpha",
        "benzinga",
    ])
    
    # Sentiment models
    use_vader: bool = True
    use_textblob: bool = True
    use_finbert: bool = False              # Slower but more accurate
    
    # Model weights for ensemble
    vader_weight: float = 0.40
    textblob_weight: float = 0.20
    finbert_weight: float = 0.40
    
    # Aggregation settings
    recency_decay: float = 0.9              # Decay factor for older news
    min_articles_for_signal: int = 5        # Min articles to generate signal


@dataclass
class FeatureSettings:
    """Settings for feature engineering."""
    
    # Technical indicators - Moving Averages
    sma_periods: List[int] = field(default_factory=lambda: [5, 10, 20, 50, 100, 200])
    ema_periods: List[int] = field(default_factory=lambda: [5, 10, 20, 50, 100])
    
    # Technical indicators - Momentum
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    stoch_k: int = 14
    stoch_d: int = 3
    cci_period: int = 20
    williams_period: int = 14
    mom_period: int = 10
    roc_period: int = 10
    
    # Technical indicators - Volatility
    atr_period: int = 14
    bb_period: int = 20
    bb_std: float = 2.0
    
    # Technical indicators - Volume
    obv_enabled: bool = True
    mfi_period: int = 14
    adl_enabled: bool = True
    
    # Technical indicators - Trend
    adx_period: int = 14
    aroon_period: int = 25
    sar_acceleration: float = 0.02
    sar_maximum: float = 0.2
    
    # Pattern recognition
    use_candlestick_patterns: bool = True
    
    # Statistical features
    rolling_windows: List[int] = field(default_factory=lambda: [5, 10, 20, 60])
    zscore_window: int = 20
    beta_window: int = 60
    correlation_window: int = 20


@dataclass
class BacktestSettings:
    """Settings for backtesting."""
    
    # Time periods
    train_start: str = "2015-01-01"
    train_end: str = "2022-12-31"
    test_start: str = "2023-01-01"
    test_end: str = "2024-12-31"
    
    # Walk-forward settings
    use_walk_forward: bool = True
    train_window_days: int = 252 * 2       # 2 years training
    test_window_days: int = 63             # ~3 months testing
    step_days: int = 21                    # Step forward 1 month
    
    # Monte Carlo settings
    monte_carlo_runs: int = 1000
    confidence_level: float = 0.95
    
    # Benchmark
    benchmark_symbol: str = "SPY"
    
    # Output
    save_trades: bool = True
    save_equity_curve: bool = True
    generate_report: bool = True


@dataclass
class Settings:
    """Main settings container."""
    
    data: DataSettings = field(default_factory=DataSettings)
    trading: TradingSettings = field(default_factory=TradingSettings)
    risk: RiskSettings = field(default_factory=RiskSettings)
    signals: SignalSettings = field(default_factory=SignalSettings)
    sentiment: SentimentSettings = field(default_factory=SentimentSettings)
    features: FeatureSettings = field(default_factory=FeatureSettings)
    backtest: BacktestSettings = field(default_factory=BacktestSettings)
    
    # Logging
    log_level: str = "INFO"
    log_file: str = "trading_engine.log"
    
    # Parallel processing
    n_jobs: int = -1                        # Use all cores
    
    # Random seed for reproducibility
    random_seed: int = 42


# Global settings instance
settings = Settings()


def load_settings_from_yaml(filepath: str) -> Settings:
    """Load settings from a YAML file."""
    import yaml
    
    with open(filepath, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update settings from config dict
    # This is a simplified version - can be extended
    return Settings(**config)


def save_settings_to_yaml(settings: Settings, filepath: str) -> None:
    """Save settings to a YAML file."""
    import yaml
    from dataclasses import asdict
    
    with open(filepath, 'w') as f:
        yaml.dump(asdict(settings), f, default_flow_style=False)
