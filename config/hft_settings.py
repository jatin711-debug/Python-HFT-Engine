"""
HFT-Specific Configuration Settings

Centralized configuration for all HFT-related parameters including:
- Strategy parameters
- Risk limits
- Market making settings
- Deep learning model configs
- Execution parameters
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import json
from pathlib import Path


class TradingMode(Enum):
    """Trading mode enumeration"""
    PAPER = "paper"
    LIVE = "live"
    BACKTEST = "backtest"
    SIMULATION = "simulation"


class RiskLevel(Enum):
    """Risk tolerance levels"""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"


@dataclass
class StatArbSettings:
    """Statistical Arbitrage Strategy Settings"""
    
    # EWLR parameters
    ewlr_halflife: int = 20  # Half-life for exponential weights
    lookback_period: int = 60  # Lookback for regression
    correlation_threshold: float = 0.7  # Min correlation for pairs
    
    # Z-score thresholds
    entry_zscore: float = 2.0  # Entry threshold
    exit_zscore: float = 0.5   # Exit threshold
    stop_zscore: float = 4.0   # Stop loss threshold
    
    # Position sizing
    max_position_pct: float = 0.1  # Max 10% of capital per leg
    hedge_ratio_bounds: Tuple[float, float] = (0.5, 2.0)
    
    # Cointegration
    cointegration_pvalue: float = 0.05
    min_halflife: int = 5
    max_halflife: int = 120
    
    # Pairs to track
    default_pairs: List[Tuple[str, str]] = field(default_factory=lambda: [
        ("AAPL", "MSFT"),
        ("GOOGL", "META"),
        ("JPM", "BAC"),
        ("XOM", "CVX"),
        ("KO", "PEP")
    ])


@dataclass
class LOBImbalanceSettings:
    """Limit Order Book Imbalance Strategy Settings"""
    
    # Imbalance calculation
    num_levels: int = 10  # Number of LOB levels to consider
    imbalance_threshold: float = 0.3  # Signal threshold
    volume_weighted: bool = True
    
    # Signal parameters
    lookback_ticks: int = 100
    signal_decay: float = 0.95
    min_spread_bps: float = 1.0  # Min spread in basis points
    
    # Position management
    max_position_size: int = 1000
    entry_timeout_ms: int = 100
    
    # Level weights (importance of each level)
    level_weights: List[float] = field(default_factory=lambda: [
        1.0, 0.8, 0.6, 0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1
    ])


@dataclass
class MarketMakingSettings:
    """Intelligent Market Making Strategy Settings"""
    
    # Spread management
    base_spread_bps: float = 2.0  # Base spread in basis points
    min_spread_bps: float = 1.0
    max_spread_bps: float = 10.0
    spread_adjustment_factor: float = 0.5
    
    # Inventory management
    target_inventory: int = 0
    max_inventory: int = 5000
    inventory_skew_factor: float = 0.001  # Price skew per unit
    
    # Quote parameters
    quote_size: int = 100
    min_quote_size: int = 10
    max_quote_size: int = 1000
    quote_refresh_ms: int = 100
    
    # Volatility adjustment
    volatility_lookback: int = 20
    volatility_multiplier: float = 2.0
    
    # Adverse selection protection
    adverse_selection_threshold: float = 0.6
    fade_aggressive_orders: bool = True
    
    # ML model for spread prediction
    use_ml_spread: bool = True
    ml_features: List[str] = field(default_factory=lambda: [
        "volatility", "spread_ma", "imbalance", "volume", "time_of_day"
    ])


@dataclass
class TechnicalSearchSettings:
    """Automated Technical Strategy Search Settings"""
    
    # Search space
    indicator_set: List[str] = field(default_factory=lambda: [
        "SMA", "EMA", "RSI", "MACD", "BB", "ATR", "ADX", "STOCH"
    ])
    
    # Parameter ranges
    sma_periods: Tuple[int, int] = (5, 200)
    ema_periods: Tuple[int, int] = (5, 100)
    rsi_periods: Tuple[int, int] = (7, 21)
    
    # Overfitting mitigation
    train_test_split: float = 0.7
    min_trades_required: int = 30
    walk_forward_periods: int = 5
    out_of_sample_validation: bool = True
    
    # Cross-validation
    n_folds: int = 5
    purge_gap: int = 10  # Gap between train/test to avoid leakage
    
    # Deflated Sharpe Ratio
    deflate_sharpe: bool = True
    num_trials_threshold: int = 100


@dataclass  
class ArbitrageSettings:
    """Index Fund Arbitrage Settings"""
    
    # ETF tracking
    target_etf: str = "SPY"
    constituent_count: int = 50  # Top N constituents to track
    
    # Arbitrage thresholds
    nav_deviation_threshold: float = 0.001  # 10 bps
    execution_slippage_estimate: float = 0.0005  # 5 bps
    
    # Rebalancing
    rebalance_threshold: float = 0.02  # 2% weight deviation
    max_constituents_traded: int = 20
    
    # Timing
    creation_redemption_cutoff: str = "15:30"  # ET
    min_holding_period_minutes: int = 5


@dataclass
class DeepLOBSettings:
    """DeepLOB Neural Network Settings"""
    
    # Architecture
    num_levels: int = 10
    sequence_length: int = 100
    hidden_dim: int = 64
    num_classes: int = 3  # Up, Down, Stationary
    
    # Inception modules
    inception_filters: List[int] = field(default_factory=lambda: [32, 32, 32])
    conv_kernels: List[int] = field(default_factory=lambda: [1, 3, 5])
    
    # LSTM
    lstm_hidden_size: int = 64
    lstm_num_layers: int = 1
    
    # Training
    batch_size: int = 64
    learning_rate: float = 0.001
    epochs: int = 50
    early_stopping_patience: int = 10
    
    # Prediction horizon
    prediction_horizon: int = 10  # ticks ahead
    label_smoothing: float = 0.1
    
    # Feature engineering
    normalize_features: bool = True
    add_technical_features: bool = True
    
    # Model paths
    model_save_dir: str = "models/saved/deeplob"
    checkpoint_interval: int = 5


@dataclass
class RiskSettings:
    """HFT Risk Management Settings"""
    
    # Position limits
    max_position_value: float = 100000.0
    max_position_shares: int = 10000
    max_daily_trades: int = 10000
    
    # Loss limits
    max_daily_loss: float = 5000.0
    max_drawdown_pct: float = 0.02  # 2%
    max_loss_per_trade: float = 500.0
    
    # Order limits
    max_order_value: float = 50000.0
    max_order_shares: int = 5000
    max_orders_per_second: int = 100
    
    # Kill switch triggers
    kill_switch_loss_threshold: float = 10000.0
    kill_switch_drawdown_pct: float = 0.05
    kill_switch_error_rate: float = 0.1
    
    # VaR settings
    var_confidence: float = 0.99
    var_horizon_days: int = 1
    var_lookback_days: int = 252
    
    # Recovery
    cooldown_period_seconds: int = 300
    gradual_recovery_steps: int = 5


@dataclass
class ExecutionSettings:
    """Order Execution Settings"""
    
    # Latency budgets (milliseconds)
    max_signal_to_order_ms: float = 10.0
    max_order_to_fill_ms: float = 50.0
    stale_quote_threshold_ms: float = 100.0
    
    # Smart Order Routing
    use_smart_routing: bool = True
    preferred_venues: List[str] = field(default_factory=lambda: [
        "NYSE", "NASDAQ", "BATS", "IEX"
    ])
    
    # Execution algos
    default_algo: str = "TWAP"  # TWAP, VWAP, IS, POV
    algo_participation_rate: float = 0.1  # 10% of volume
    
    # Transaction costs
    commission_per_share: float = 0.001
    exchange_fee_per_share: float = 0.0003
    slippage_estimate_bps: float = 1.0


@dataclass
class HFTConfig:
    """
    Master HFT Configuration
    
    Aggregates all HFT-related settings into a single configuration object.
    Can be serialized to/from JSON for persistence.
    """
    
    # Mode and environment
    trading_mode: TradingMode = TradingMode.PAPER
    risk_level: RiskLevel = RiskLevel.MODERATE
    
    # Strategy settings
    stat_arb: StatArbSettings = field(default_factory=StatArbSettings)
    lob_imbalance: LOBImbalanceSettings = field(default_factory=LOBImbalanceSettings)
    market_making: MarketMakingSettings = field(default_factory=MarketMakingSettings)
    tech_search: TechnicalSearchSettings = field(default_factory=TechnicalSearchSettings)
    arbitrage: ArbitrageSettings = field(default_factory=ArbitrageSettings)
    
    # Model settings
    deeplob: DeepLOBSettings = field(default_factory=DeepLOBSettings)
    
    # Risk and execution
    risk: RiskSettings = field(default_factory=RiskSettings)
    execution: ExecutionSettings = field(default_factory=ExecutionSettings)
    
    # Enabled strategies
    enabled_strategies: List[str] = field(default_factory=lambda: [
        "stat_arb", "lob_imbalance", "market_making"
    ])
    
    # Symbols to trade
    symbols: List[str] = field(default_factory=lambda: [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"
    ])
    
    # Data settings
    data_source: str = "polygon"  # polygon, alpaca, ib
    use_level2_data: bool = True
    
    def to_dict(self) -> Dict:
        """Convert config to dictionary"""
        result = {}
        for key, value in self.__dict__.items():
            if hasattr(value, '__dict__'):
                if isinstance(value, Enum):
                    result[key] = value.value
                else:
                    result[key] = {k: v for k, v in value.__dict__.items()}
            elif isinstance(value, list):
                result[key] = value
            else:
                result[key] = value
        return result
    
    def save(self, filepath: str) -> None:
        """Save config to JSON file"""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
    
    @classmethod
    def load(cls, filepath: str) -> 'HFTConfig':
        """Load config from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        config = cls()
        # Map loaded values to config
        if 'trading_mode' in data:
            config.trading_mode = TradingMode(data['trading_mode'])
        if 'risk_level' in data:
            config.risk_level = RiskLevel(data['risk_level'])
        if 'enabled_strategies' in data:
            config.enabled_strategies = data['enabled_strategies']
        if 'symbols' in data:
            config.symbols = data['symbols']
            
        return config
    
    def apply_risk_level_presets(self):
        """Apply preset values based on risk level"""
        if self.risk_level == RiskLevel.CONSERVATIVE:
            self.risk.max_position_value = 50000.0
            self.risk.max_daily_loss = 2500.0
            self.risk.max_drawdown_pct = 0.01
            self.stat_arb.entry_zscore = 2.5
            self.market_making.base_spread_bps = 3.0
            
        elif self.risk_level == RiskLevel.AGGRESSIVE:
            self.risk.max_position_value = 200000.0
            self.risk.max_daily_loss = 10000.0
            self.risk.max_drawdown_pct = 0.03
            self.stat_arb.entry_zscore = 1.5
            self.market_making.base_spread_bps = 1.5
            
    def validate(self) -> List[str]:
        """Validate configuration, return list of issues"""
        issues = []
        
        # Risk validation
        if self.risk.max_daily_loss > self.risk.kill_switch_loss_threshold:
            issues.append("max_daily_loss should be less than kill_switch_loss_threshold")
            
        if self.risk.max_drawdown_pct > self.risk.kill_switch_drawdown_pct:
            issues.append("max_drawdown_pct should be less than kill_switch_drawdown_pct")
            
        # Strategy validation
        if self.stat_arb.entry_zscore <= self.stat_arb.exit_zscore:
            issues.append("stat_arb entry_zscore must be greater than exit_zscore")
            
        if self.market_making.min_spread_bps >= self.market_making.max_spread_bps:
            issues.append("market_making min_spread_bps must be less than max_spread_bps")
            
        # Execution validation
        if not self.execution.preferred_venues:
            issues.append("At least one preferred venue must be specified")
            
        return issues


# Default configuration instances for different use cases
def get_paper_trading_config() -> HFTConfig:
    """Get configuration for paper trading"""
    config = HFTConfig(
        trading_mode=TradingMode.PAPER,
        risk_level=RiskLevel.MODERATE
    )
    return config


def get_backtesting_config() -> HFTConfig:
    """Get configuration for backtesting"""
    config = HFTConfig(
        trading_mode=TradingMode.BACKTEST,
        risk_level=RiskLevel.MODERATE
    )
    # Disable real-time features for backtesting
    config.execution.use_smart_routing = False
    config.lob_imbalance.entry_timeout_ms = 0
    return config


def get_live_trading_config() -> HFTConfig:
    """Get configuration for live trading (use with caution!)"""
    config = HFTConfig(
        trading_mode=TradingMode.LIVE,
        risk_level=RiskLevel.CONSERVATIVE  # Start conservative
    )
    config.apply_risk_level_presets()
    
    # Stricter risk limits for live
    config.risk.max_daily_loss *= 0.5
    config.risk.max_position_value *= 0.5
    
    return config


# Convenience function to get default config
def get_hft_config(mode: str = "paper") -> HFTConfig:
    """
    Get HFT configuration by mode
    
    Args:
        mode: 'paper', 'backtest', 'live', or 'simulation'
    
    Returns:
        Configured HFTConfig instance
    """
    mode_map = {
        "paper": get_paper_trading_config,
        "backtest": get_backtesting_config,
        "live": get_live_trading_config,
        "simulation": get_paper_trading_config
    }
    
    if mode not in mode_map:
        raise ValueError(f"Unknown mode: {mode}. Use one of {list(mode_map.keys())}")
    
    return mode_map[mode]()


if __name__ == "__main__":
    # Example usage and validation
    config = get_paper_trading_config()
    
    print("HFT Configuration")
    print("=" * 50)
    print(f"Trading Mode: {config.trading_mode.value}")
    print(f"Risk Level: {config.risk_level.value}")
    print(f"Enabled Strategies: {config.enabled_strategies}")
    print(f"Symbols: {config.symbols}")
    print()
    
    # Validate
    issues = config.validate()
    if issues:
        print("Configuration Issues:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("Configuration is valid!")
    
    # Save example
    config.save("config/hft_config_example.json")
    print("\nConfig saved to config/hft_config_example.json")
