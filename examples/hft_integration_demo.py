"""
HFT Integration Demo

This example demonstrates how to use the HFT strategies integrated
with the main trading engine.

There are 3 ways to use HFT strategies:

1. Command Line (--hft flag):
   python main.py --symbol AAPL --mode backtest --hft

2. Programmatic (run method):
   engine.run(symbol="AAPL", mode="backtest", use_hft=True)

3. Standalone (direct usage):
   aggregator = UnifiedSignalAggregator(ml_model=model, use_hft=True)
   signal = aggregator.generate_unified_signal(df, "AAPL")
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime

# Add parent to path
sys.path.insert(0, '.')

def demo_standalone_hft():
    """Demo: Using HFT strategies standalone without ML model."""
    print("\n" + "="*60)
    print("DEMO 1: Standalone HFT Signals (No ML)")
    print("="*60)
    
    from strategies.hft_strategies import (
        HFTStrategyEnsemble,
        StatisticalArbitrageEWLR,
        OrderBookImbalanceStrategy,
    )
    
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=200, freq='D')
    
    df = pd.DataFrame({
        'open': 150 + np.random.randn(200).cumsum() * 0.5,
        'high': 151 + np.random.randn(200).cumsum() * 0.5,
        'low': 149 + np.random.randn(200).cumsum() * 0.5,
        'close': 150 + np.random.randn(200).cumsum() * 0.5,
        'volume': np.random.randint(1000000, 10000000, 200),
    }, index=dates)
    
    df['high'] = df[['open', 'close', 'high']].max(axis=1) + np.random.rand(200) * 0.5
    df['low'] = df[['open', 'close', 'low']].min(axis=1) - np.random.rand(200) * 0.5
    
    # Use HFT Ensemble
    ensemble = HFTStrategyEnsemble()
    
    # Generate signals for last few days
    print("\nRecent HFT Signals:")
    print("-" * 60)
    
    for idx in range(-5, 0):
        signal = ensemble.generate_ensemble_signal(df, current_idx=idx)
        print(f"Date: {df.index[idx].strftime('%Y-%m-%d')} | "
              f"Direction: {signal.direction:+d} | "
              f"Strength: {signal.strength:.2f} | "
              f"Confidence: {signal.confidence:.2f}")
        if signal.reasoning:
            for reason in signal.reasoning[:2]:  # First 2 reasons
                print(f"    → {reason}")


def demo_lob_imbalance():
    """Demo: LOB Imbalance strategy for intraday signals."""
    print("\n" + "="*60)
    print("DEMO 2: LOB Imbalance (Intraday Signals)")
    print("="*60)
    
    from strategies.hft_strategies import OrderBookImbalanceStrategy, LOBSnapshot
    
    # Create intraday data
    np.random.seed(123)
    
    strategy = OrderBookImbalanceStrategy(
        imbalance_threshold=0.2,
        momentum_periods=5,
    )
    
    # Simulate intraday bars
    minutes = pd.date_range('2024-01-15 09:30', periods=60, freq='1min')
    price = 150.0
    prices = []
    
    for _ in range(60):
        price += np.random.randn() * 0.05
        prices.append(price)
    
    df = pd.DataFrame({
        'open': prices,
        'high': [p + np.random.rand() * 0.1 for p in prices],
        'low': [p - np.random.rand() * 0.1 for p in prices],
        'close': prices,
        'volume': np.random.randint(10000, 100000, 60),
    }, index=minutes)
    
    print("\nLOB-based signals (last 10 minutes):")
    print("-" * 60)
    
    for idx in range(-10, 0):
        signal = strategy.generate_signal(df, current_idx=idx)
        features = strategy.calculate_imbalance_features(df, current_idx=idx)
        
        dir_emoji = "🟢" if signal.direction > 0 else "🔴" if signal.direction < 0 else "⚪"
        print(f"{df.index[idx].strftime('%H:%M')} | {dir_emoji} | "
              f"Imbalance: {features['imbalance']:+.3f} | "
              f"Momentum: {features['imbalance_momentum']:+.3f}")


def demo_unified_aggregator():
    """Demo: Full unified aggregator with all signal sources."""
    print("\n" + "="*60)
    print("DEMO 3: Unified Signal Aggregator")
    print("="*60)
    
    from signals.unified_signal_aggregator import (
        UnifiedSignalAggregator,
        AggregatorConfig,
    )
    
    # Create sample data with features
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=200, freq='D')
    
    close_prices = 150 + np.random.randn(200).cumsum() * 0.5
    
    df = pd.DataFrame({
        'open': close_prices - np.random.rand(200) * 0.5,
        'high': close_prices + np.random.rand(200) * 1.0,
        'low': close_prices - np.random.rand(200) * 1.0,
        'close': close_prices,
        'volume': np.random.randint(1000000, 10000000, 200),
    }, index=dates)
    
    # Add some basic features
    df['sma_20'] = df['close'].rolling(20).mean()
    df['sma_50'] = df['close'].rolling(50).mean()
    df['rsi'] = 50 + np.random.randn(200) * 15  # Simulated RSI
    df['adx'] = 20 + np.random.rand(200) * 20  # Simulated ADX
    df['atr'] = df['high'] - df['low']
    
    # Configure aggregator
    config = AggregatorConfig(
        ml_weight=0.30,  # Lower ML weight (no model in demo)
        institutional_weight=0.40,
        hft_weight=0.30,
        require_agreement=True,
        min_agreement_score=0.5,
    )
    
    # Create aggregator without ML model
    aggregator = UnifiedSignalAggregator(
        ml_model=None,  # No ML model in this demo
        config=config,
        use_hft=True,
    )
    
    print("\nUnified Signals (last 5 days):")
    print("-" * 60)
    
    for idx in range(-5, 0):
        signal = aggregator.generate_unified_signal(
            df=df,
            symbol="DEMO",
            current_idx=idx,
        )
        
        dir_str = "BUY " if signal.direction > 0 else "SELL" if signal.direction < 0 else "HOLD"
        regime_emoji = {
            'bull_trend': '📈',
            'bear_trend': '📉',
            'high_volatility': '⚡',
            'low_volatility': '😴',
            'ranging': '↔️',
        }.get(signal.regime, '❓')
        
        print(f"{df.index[idx].strftime('%Y-%m-%d')} | {dir_str} | "
              f"Confidence: {signal.confidence:.2f} | "
              f"Agreement: {signal.agreement_score:.0%} | "
              f"Regime: {regime_emoji} {signal.regime}")


def demo_command_line_usage():
    """Show command line usage examples."""
    print("\n" + "="*60)
    print("COMMAND LINE USAGE")
    print("="*60)
    
    print("""
# Run backtest with HFT strategies enabled:
python main.py --symbol AAPL --mode backtest --hft

# Run with HFT and news sentiment:
python main.py --symbol TSLA --mode backtest --hft --fetch-news

# Live mode with HFT signals:
python main.py --symbol NVDA --mode live --hft

# Walk-forward validation with HFT:
python main.py --symbol MSFT --mode walkforward --hft

# With all features:
python main.py --symbol AAPL --mode backtest --hft --fetch-news --alternative-data --save-results
""")


def demo_programmatic_usage():
    """Show programmatic usage example."""
    print("\n" + "="*60)
    print("PROGRAMMATIC USAGE")
    print("="*60)
    
    print("""
from main import TradingEngine

# Create engine
engine = TradingEngine()

# Run with HFT enabled
results = engine.run(
    symbol="AAPL",
    mode="backtest",
    use_hft=True,  # Enable HFT strategies
)

# Access HFT-specific results
signals_df = results['signals']

# Check if HFT was used
if results.get('hft_enabled'):
    print("HFT strategies were integrated!")
    
    # HFT-specific columns in signals_df:
    # - unified_direction: Final direction from all strategies
    # - unified_strength: Combined signal strength
    # - unified_confidence: Combined confidence score
    # - agreement_score: How much strategies agree
    # - regime: Detected market regime
    # - execution_path: Recommended execution path
    
    # Filter for high-agreement signals
    high_agreement = signals_df[signals_df['agreement_score'] > 0.7]
    print(f"High-agreement signals: {len(high_agreement)}")
""")


if __name__ == '__main__':
    print("="*60)
    print("        HFT INTEGRATION EXAMPLES")
    print("="*60)
    
    # Run demos
    demo_standalone_hft()
    demo_lob_imbalance()
    demo_unified_aggregator()
    demo_command_line_usage()
    demo_programmatic_usage()
    
    print("\n" + "="*60)
    print("ALL DEMOS COMPLETE!")
    print("="*60)
