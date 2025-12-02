"""
HFT Trading Engine Integration Example

This example demonstrates how to integrate all HFT components:
- HFT Strategies (Statistical Arbitrage, LOB Imbalance, Market Making)
- DeepLOB Neural Network
- Risk Management System
- Configuration Management

NOTE: This is a research/paper-trading implementation in Python.
For true HFT (microsecond latency), C++/FPGA implementation is required.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import HFT modules
from config.hft_settings import (
    HFTConfig, 
    get_hft_config, 
    TradingMode, 
    RiskLevel
)
from strategies.hft_strategies import (
    StatisticalArbitrageStrategy,
    LOBImbalanceStrategy,
    IntelligentMarketMaker,
    HFTStrategyEnsemble,
    HFTSignal
)
from risk.hft_risk_management import (
    HFTRiskManager,
    Order,
    OrderRiskLimits
)

# Optional: Deep Learning (requires PyTorch)
try:
    from models.deep_learning.deeplob import (
        DeepLOBConfig,
        DeepLOBModel,
        LOBFeatureGenerator
    )
    HAS_PYTORCH = True
except ImportError:
    HAS_PYTORCH = False
    logger.warning("PyTorch not available - DeepLOB features disabled")


class HFTTradingEngine:
    """
    High-Frequency Trading Engine
    
    Orchestrates all HFT components for signal generation, risk management,
    and order execution (simulated in Python).
    """
    
    def __init__(self, config: Optional[HFTConfig] = None):
        """
        Initialize HFT Trading Engine
        
        Args:
            config: HFT configuration (uses paper trading defaults if None)
        """
        self.config = config or get_hft_config("paper")
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Validate configuration
        issues = self.config.validate()
        if issues:
            for issue in issues:
                self.logger.warning(f"Config issue: {issue}")
        
        # Initialize components
        self._init_strategies()
        self._init_risk_manager()
        self._init_deep_learning()
        
        # State tracking
        self.positions: Dict[str, int] = {}
        self.orders: List[Order] = []
        self.signals_history: List[HFTSignal] = []
        self.is_running = False
        
        self.logger.info(f"HFT Engine initialized in {self.config.trading_mode.value} mode")
        
    def _init_strategies(self):
        """Initialize trading strategies based on config"""
        self.strategies = {}
        
        if "stat_arb" in self.config.enabled_strategies:
            self.strategies["stat_arb"] = StatisticalArbitrageStrategy(
                pairs=self.config.stat_arb.default_pairs,
                lookback_period=self.config.stat_arb.lookback_period,
                entry_zscore=self.config.stat_arb.entry_zscore,
                exit_zscore=self.config.stat_arb.exit_zscore
            )
            self.logger.info("Statistical Arbitrage strategy enabled")
            
        if "lob_imbalance" in self.config.enabled_strategies:
            self.strategies["lob_imbalance"] = LOBImbalanceStrategy(
                imbalance_threshold=self.config.lob_imbalance.imbalance_threshold,
                num_levels=self.config.lob_imbalance.num_levels,
                lookback_ticks=self.config.lob_imbalance.lookback_ticks
            )
            self.logger.info("LOB Imbalance strategy enabled")
            
        if "market_making" in self.config.enabled_strategies:
            self.strategies["market_making"] = IntelligentMarketMaker(
                base_spread_bps=self.config.market_making.base_spread_bps,
                max_inventory=self.config.market_making.max_inventory,
                inventory_skew_factor=self.config.market_making.inventory_skew_factor
            )
            self.logger.info("Intelligent Market Making strategy enabled")
            
        # Create ensemble if multiple strategies
        if len(self.strategies) > 1:
            self.ensemble = HFTStrategyEnsemble(list(self.strategies.values()))
            self.logger.info(f"Strategy ensemble created with {len(self.strategies)} strategies")
        else:
            self.ensemble = None
            
    def _init_risk_manager(self):
        """Initialize risk management system"""
        risk_limits = OrderRiskLimits(
            max_order_value=self.config.risk.max_order_value,
            max_order_shares=self.config.risk.max_order_shares,
            max_position_value=self.config.risk.max_position_value,
            max_position_shares=self.config.risk.max_position_shares,
            max_daily_trades=self.config.risk.max_daily_trades,
            max_daily_loss=self.config.risk.max_daily_loss,
            max_orders_per_second=self.config.risk.max_orders_per_second,
            max_drawdown_pct=self.config.risk.max_drawdown_pct
        )
        
        self.risk_manager = HFTRiskManager(
            limits=risk_limits,
            kill_switch_loss_threshold=self.config.risk.kill_switch_loss_threshold,
            kill_switch_drawdown_pct=self.config.risk.kill_switch_drawdown_pct
        )
        self.logger.info("Risk management system initialized")
        
    def _init_deep_learning(self):
        """Initialize DeepLOB model if PyTorch available"""
        self.deeplob_model = None
        self.lob_feature_generator = None
        
        if HAS_PYTORCH and self.config.use_level2_data:
            try:
                dl_config = DeepLOBConfig(
                    num_levels=self.config.deeplob.num_levels,
                    sequence_length=self.config.deeplob.sequence_length,
                    hidden_dim=self.config.deeplob.hidden_dim,
                    num_classes=self.config.deeplob.num_classes
                )
                self.deeplob_model = DeepLOBModel(dl_config)
                self.lob_feature_generator = LOBFeatureGenerator(
                    num_levels=self.config.deeplob.num_levels
                )
                self.logger.info("DeepLOB model initialized")
            except Exception as e:
                self.logger.error(f"Failed to initialize DeepLOB: {e}")
                
    def process_market_data(self, data: pd.DataFrame) -> List[HFTSignal]:
        """
        Process incoming market data and generate signals
        
        Args:
            data: Market data DataFrame with OHLCV columns
            
        Returns:
            List of trading signals from all strategies
        """
        if not self.risk_manager.can_trade():
            self.logger.warning("Trading halted by risk manager")
            return []
            
        signals = []
        
        # Generate signals from each strategy
        for name, strategy in self.strategies.items():
            try:
                if hasattr(strategy, 'generate_signals'):
                    strategy_signals = strategy.generate_signals(data)
                    for signal in strategy_signals:
                        signal.strategy_name = name
                        signals.append(signal)
            except Exception as e:
                self.logger.error(f"Error in {name} strategy: {e}")
                
        # Use ensemble if available
        if self.ensemble and len(signals) > 1:
            # Ensemble combines signals with weighted voting
            combined_signal = self.ensemble.combine_signals(data, {})
            if combined_signal:
                signals.append(combined_signal)
                
        self.signals_history.extend(signals)
        return signals
        
    def process_lob_data(self, lob_snapshot: Dict) -> Optional[HFTSignal]:
        """
        Process limit order book snapshot
        
        Args:
            lob_snapshot: Dictionary with 'bids' and 'asks' price/volume levels
            
        Returns:
            Signal from LOB analysis
        """
        signal = None
        
        # LOB Imbalance strategy
        if "lob_imbalance" in self.strategies:
            strategy = self.strategies["lob_imbalance"]
            signal = strategy.calculate_imbalance_signal(lob_snapshot)
            
        # DeepLOB prediction (if model trained)
        if self.deeplob_model and self.lob_feature_generator:
            try:
                features = self.lob_feature_generator.generate_features(lob_snapshot)
                # Note: Would need sequence of features for actual prediction
                # This is simplified for demonstration
            except Exception as e:
                self.logger.debug(f"DeepLOB feature generation: {e}")
                
        return signal
        
    def validate_and_submit_order(self, signal: HFTSignal) -> Optional[Order]:
        """
        Validate signal through risk checks and submit order
        
        Args:
            signal: Trading signal to convert to order
            
        Returns:
            Order object if validated and submitted, None otherwise
        """
        # Convert signal to order
        order = Order(
            symbol=signal.symbol,
            side="BUY" if signal.direction > 0 else "SELL",
            quantity=signal.suggested_quantity,
            price=signal.price,
            order_type="LIMIT",
            timestamp=datetime.now()
        )
        
        # Get current position
        current_position = self.positions.get(signal.symbol, 0)
        
        # Validate through risk manager
        is_valid, rejection_reason = self.risk_manager.validate_order(
            order, 
            current_position
        )
        
        if not is_valid:
            self.logger.warning(f"Order rejected: {rejection_reason}")
            return None
            
        # Submit order (simulated in paper trading)
        self.orders.append(order)
        
        # Update position (simulated fill)
        if self.config.trading_mode == TradingMode.PAPER:
            position_delta = order.quantity if order.side == "BUY" else -order.quantity
            self.positions[signal.symbol] = current_position + position_delta
            
        self.logger.info(f"Order submitted: {order.side} {order.quantity} {order.symbol} @ {order.price}")
        return order
        
    def update_portfolio_value(self, prices: Dict[str, float]) -> float:
        """
        Update portfolio value with current prices
        
        Args:
            prices: Dictionary of symbol -> current price
            
        Returns:
            Current portfolio value
        """
        total_value = 0.0
        for symbol, position in self.positions.items():
            if symbol in prices and position != 0:
                value = position * prices[symbol]
                total_value += value
                
        # Update risk manager
        self.risk_manager.update_portfolio_value(total_value)
        
        return total_value
        
    def get_status(self) -> Dict:
        """Get current engine status"""
        return {
            "mode": self.config.trading_mode.value,
            "risk_level": self.config.risk_level.value,
            "can_trade": self.risk_manager.can_trade(),
            "positions": self.positions.copy(),
            "total_orders": len(self.orders),
            "signals_generated": len(self.signals_history),
            "strategies_enabled": list(self.strategies.keys()),
            "deeplob_available": self.deeplob_model is not None
        }
        
    def run_backtest(self, historical_data: pd.DataFrame) -> Dict:
        """
        Run backtest on historical data
        
        Args:
            historical_data: DataFrame with OHLCV data
            
        Returns:
            Backtest results dictionary
        """
        self.logger.info("Starting backtest...")
        
        results = {
            "signals": [],
            "trades": [],
            "pnl": 0.0,
            "num_trades": 0
        }
        
        # Process data in chunks (simulating real-time)
        window_size = 100
        for i in range(window_size, len(historical_data)):
            window = historical_data.iloc[i-window_size:i]
            
            # Generate signals
            signals = self.process_market_data(window)
            results["signals"].extend(signals)
            
            # Process signals
            for signal in signals:
                if signal.confidence > 0.6:  # Minimum confidence threshold
                    order = self.validate_and_submit_order(signal)
                    if order:
                        results["trades"].append({
                            "timestamp": historical_data.index[i],
                            "symbol": order.symbol,
                            "side": order.side,
                            "quantity": order.quantity,
                            "price": order.price
                        })
                        results["num_trades"] += 1
                        
        self.logger.info(f"Backtest complete: {results['num_trades']} trades")
        return results


def run_demo():
    """Run demonstration of HFT Engine"""
    print("=" * 60)
    print("HFT Trading Engine Demo")
    print("=" * 60)
    
    # Create configuration
    config = get_hft_config("paper")
    config.enabled_strategies = ["stat_arb", "lob_imbalance", "market_making"]
    config.symbols = ["AAPL", "MSFT", "GOOGL"]
    
    print(f"\nConfiguration:")
    print(f"  Mode: {config.trading_mode.value}")
    print(f"  Risk Level: {config.risk_level.value}")
    print(f"  Strategies: {config.enabled_strategies}")
    print(f"  Symbols: {config.symbols}")
    
    # Initialize engine
    engine = HFTTradingEngine(config)
    
    print(f"\nEngine Status:")
    status = engine.get_status()
    for key, value in status.items():
        print(f"  {key}: {value}")
        
    # Generate sample data for demonstration
    print("\nGenerating sample market data...")
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=500, freq='1min')
    
    sample_data = pd.DataFrame({
        'Open': 150 + np.random.randn(500).cumsum() * 0.1,
        'High': 151 + np.random.randn(500).cumsum() * 0.1,
        'Low': 149 + np.random.randn(500).cumsum() * 0.1,
        'Close': 150 + np.random.randn(500).cumsum() * 0.1,
        'Volume': np.random.randint(1000, 10000, 500)
    }, index=dates)
    sample_data['High'] = sample_data[['Open', 'High', 'Close']].max(axis=1) + 0.1
    sample_data['Low'] = sample_data[['Open', 'Low', 'Close']].min(axis=1) - 0.1
    
    # Run backtest
    print("\nRunning backtest...")
    results = engine.run_backtest(sample_data)
    
    print(f"\nBacktest Results:")
    print(f"  Total Signals: {len(results['signals'])}")
    print(f"  Total Trades: {results['num_trades']}")
    
    # Show sample LOB processing
    print("\n" + "-" * 40)
    print("Sample LOB Processing:")
    sample_lob = {
        'bids': [(149.95, 1000), (149.90, 2000), (149.85, 3000)],
        'asks': [(150.05, 800), (150.10, 1500), (150.15, 2500)]
    }
    print(f"  LOB Snapshot: {sample_lob}")
    
    lob_signal = engine.process_lob_data(sample_lob)
    if lob_signal:
        print(f"  Signal: {lob_signal.direction} with confidence {lob_signal.confidence:.2f}")
    else:
        print("  No signal generated")
        
    print("\n" + "=" * 60)
    print("Demo Complete!")
    print("=" * 60)


if __name__ == "__main__":
    run_demo()
