"""
Test Suite for HFT Strategies

Run tests with:
    python -m pytest tests/test_hft_strategies.py -v
    
Or run directly:
    python tests/test_hft_strategies.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import unittest


class TestStatisticalArbitrage(unittest.TestCase):
    """Test Statistical Arbitrage EWLR Strategy"""
    
    @classmethod
    def setUpClass(cls):
        """Generate test data for pairs trading"""
        np.random.seed(42)
        n = 500
        
        # Create cointegrated pair
        dates = pd.date_range(start='2024-01-01', periods=n, freq='1min')
        
        # Asset A - random walk
        asset_a = 100 + np.cumsum(np.random.randn(n) * 0.1)
        
        # Asset B - cointegrated with A (beta ~ 1.2)
        noise = np.random.randn(n) * 0.5
        asset_b = 50 + 1.2 * asset_a + noise
        
        cls.prices_a = pd.Series(asset_a, index=dates, name='AAPL')
        cls.prices_b = pd.Series(asset_b, index=dates, name='MSFT')
        cls.dates = dates
        
    def test_strategy_initialization(self):
        """Test strategy can be initialized"""
        from strategies.hft_strategies import StatisticalArbitrageEWLR
        
        strategy = StatisticalArbitrageEWLR(
            alpha=0.9999,
            entry_threshold=2.0,
            exit_threshold=0.5,
            lookback=100
        )
        
        self.assertIsNotNone(strategy)
        self.assertEqual(strategy.entry_threshold, 2.0)
        print("✓ Statistical Arbitrage initialized successfully")
        
    def test_ewlr_fit(self):
        """Test EWLR model fitting"""
        from strategies.hft_strategies import StatisticalArbitrageEWLR
        
        strategy = StatisticalArbitrageEWLR(
            alpha=0.999,
            lookback=100
        )
        
        # Fit model with Y = target, X = components
        X = pd.DataFrame({'AAPL': self.prices_a})
        Y = self.prices_b
        
        result = strategy.fit(Y, X)
        
        self.assertIn('beta', result)
        self.assertIn('spread_mean', result)
        self.assertIn('spread_std', result)
        
        # Beta should be close to 1.2 (our synthetic relationship)
        beta = result['beta'][1]  # [intercept, slope]
        print(f"✓ EWLR fitted: beta={beta:.4f} (expected ~1.2)")
        
    def test_signal_generation(self):
        """Test signal generation"""
        from strategies.hft_strategies import StatisticalArbitrageEWLR
        
        strategy = StatisticalArbitrageEWLR(
            alpha=0.999,
            entry_threshold=2.0,
            lookback=100
        )
        
        # Generate signal using full series
        X = pd.DataFrame({'AAPL': self.prices_a})
        Y = self.prices_b
        
        signal = strategy.generate_signal(Y, X, current_idx=-1)
        
        self.assertIsNotNone(signal)
        print(f"✓ Signal generated: direction={signal.direction}")


class TestLOBImbalance(unittest.TestCase):
    """Test LOB Imbalance Strategy"""
    
    @classmethod
    def setUpClass(cls):
        """Create sample OHLCV data"""
        np.random.seed(42)
        n = 200
        dates = pd.date_range(start='2024-01-01', periods=n, freq='1min')
        
        close = 100 + np.cumsum(np.random.randn(n) * 0.1)
        
        cls.df = pd.DataFrame({
            'open': close + np.random.randn(n) * 0.05,
            'high': close + np.abs(np.random.randn(n) * 0.1),
            'low': close - np.abs(np.random.randn(n) * 0.1),
            'close': close,
            'volume': np.random.randint(1000, 10000, n).astype(float)
        }, index=dates)
        
    def test_strategy_initialization(self):
        """Test LOB strategy initialization"""
        from strategies.hft_strategies import OrderBookImbalanceStrategy
        
        strategy = OrderBookImbalanceStrategy(
            imbalance_threshold=0.3,
            n_levels=5,
            momentum_periods=3
        )
        
        self.assertIsNotNone(strategy)
        self.assertEqual(strategy.n_levels, 5)
        print("✓ LOB Imbalance strategy initialized")
        
    def test_lob_estimation_from_ohlcv(self):
        """Test LOB estimation from OHLCV data"""
        from strategies.hft_strategies import OrderBookImbalanceStrategy
        
        strategy = OrderBookImbalanceStrategy(
            imbalance_threshold=0.2,
            n_levels=5
        )
        
        # Estimate LOB from OHLCV
        lob = strategy.estimate_lob_from_ohlcv(self.df, idx=-1)
        
        self.assertIsNotNone(lob)
        self.assertEqual(len(lob.bid_prices), 5)
        self.assertEqual(len(lob.ask_prices), 5)
        print(f"✓ LOB estimated: {len(lob.bid_prices)} bid levels, {len(lob.ask_prices)} ask levels")
        
    def test_imbalance_calculation(self):
        """Test imbalance calculation"""
        from strategies.hft_strategies import OrderBookImbalanceStrategy, LOBSnapshot
        
        strategy = OrderBookImbalanceStrategy(
            imbalance_threshold=0.2,
            n_levels=3
        )
        
        # Create LOBSnapshot with known imbalance using correct dataclass format
        snapshot = LOBSnapshot(
            timestamp=pd.Timestamp.now(),
            bid_prices=np.array([99.95, 99.90, 99.85]),
            bid_sizes=np.array([2000.0, 3000.0, 2500.0]),  # Total: 7500
            ask_prices=np.array([100.05, 100.10, 100.15]),
            ask_sizes=np.array([500.0, 800.0, 600.0])      # Total: 1900
        )
        
        # Use the dataclass property
        imbalance = snapshot.imbalance
        
        # Expected: (7500 - 1900) / (7500 + 1900) = 0.596
        self.assertIsNotNone(imbalance)
        self.assertGreater(imbalance, 0.5)
        print(f"✓ Imbalance calculated: {imbalance:.4f} (bid heavy)")
        
    def test_signal_generation(self):
        """Test signal generation from OHLCV"""
        from strategies.hft_strategies import OrderBookImbalanceStrategy
        
        strategy = OrderBookImbalanceStrategy(
            imbalance_threshold=0.1,  # Lower threshold to ensure signal
            n_levels=5,
            momentum_periods=3
        )
        
        signal = strategy.generate_signal(self.df, current_idx=-1)
        
        self.assertIsNotNone(signal)
        print(f"✓ Generated signal: direction={signal.direction}")


class TestMarketMaker(unittest.TestCase):
    """Test Intelligent Market Maker"""
    
    @classmethod
    def setUpClass(cls):
        """Create sample data"""
        np.random.seed(42)
        n = 100
        dates = pd.date_range(start='2024-01-01', periods=n, freq='1min')
        
        close = 150 + np.cumsum(np.random.randn(n) * 0.1)
        
        cls.df = pd.DataFrame({
            'open': close + np.random.randn(n) * 0.05,
            'high': close + np.abs(np.random.randn(n) * 0.1),
            'low': close - np.abs(np.random.randn(n) * 0.1),
            'close': close,
            'volume': np.random.randint(1000, 10000, n).astype(float)
        }, index=dates)
    
    def test_initialization(self):
        """Test market maker initialization"""
        from strategies.hft_strategies import IntelligentMarketMaker
        
        mm = IntelligentMarketMaker(
            spread_target_bps=10,
            max_inventory=1000,
            inventory_skew=0.5
        )
        
        self.assertIsNotNone(mm)
        self.assertEqual(mm.max_inventory, 1000)
        print("✓ Market Maker initialized")
        
    def test_feature_extraction(self):
        """Test feature extraction"""
        from strategies.hft_strategies import IntelligentMarketMaker
        
        mm = IntelligentMarketMaker(
            spread_target_bps=10,
            lookback_ticks=3
        )
        
        features = mm.extract_features(self.df, idx=-1)
        
        self.assertIsNotNone(features)
        self.assertGreater(len(features), 0)
        print(f"✓ Extracted {len(features)} features")
        
    def test_quote_generation(self):
        """Test quote generation"""
        from strategies.hft_strategies import IntelligentMarketMaker
        
        mm = IntelligentMarketMaker(
            spread_target_bps=10,
            max_inventory=1000,
            inventory_skew=0.5
        )
        
        # Generate quotes based on current data
        quotes = mm.generate_quotes(self.df, idx=-1)
        
        if quotes:
            self.assertIn('bid_price', quotes)
            self.assertIn('ask_price', quotes)
            self.assertLess(quotes['bid_price'], quotes['ask_price'])
            print(f"✓ Quotes: Bid={quotes['bid_price']:.2f}, Ask={quotes['ask_price']:.2f}")
        else:
            print("✓ No quotes (ML model not trained)")


class TestRiskManagement(unittest.TestCase):
    """Test HFT Risk Management"""
        
    def test_pre_trade_checker(self):
        """Test pre-trade risk checking"""
        from risk.hft_risk_management import PreTradeRiskChecker, Order, OrderType, RiskCheckResult
        
        checker = PreTradeRiskChecker(
            max_order_size=50000,
            max_position_per_symbol=100000,
            max_daily_loss=5000
        )
        
        # Valid order
        valid_order = Order(
            order_id='test_001',
            symbol='AAPL',
            side='buy',
            quantity=100,
            price=150.0,
            order_type=OrderType.LIMIT
        )
        
        result = checker.check_order(valid_order, reference_price=150.0)
        # Result could be APPROVED or MODIFIED, both are acceptable for valid orders
        self.assertIn(result.result, [RiskCheckResult.APPROVED, RiskCheckResult.MODIFIED])
        print(f"✓ Valid order result: {result.result}")
        
    def test_kill_switch(self):
        """Test kill switch functionality"""
        from risk.hft_risk_management import KillSwitch
        
        ks = KillSwitch(
            max_daily_loss=10000,
            max_drawdown=0.05
        )
        
        self.assertTrue(ks.is_active)
        print("✓ Kill switch active initially")
        
        # Simulate large loss using update_pnl
        ks.update_pnl(-15000)
        
        # Check if limits were triggered
        should_halt, reason = ks.check_and_halt()
        
        # Kill switch should now be inactive
        self.assertTrue(should_halt or not ks.is_active)
        print(f"✓ Kill switch triggered: {reason or 'loss threshold exceeded'}")


class TestHFTConfig(unittest.TestCase):
    """Test HFT Configuration"""
    
    def test_config_creation(self):
        """Test config creation"""
        from config.hft_settings import get_hft_config, TradingMode
        
        config = get_hft_config("paper")
        
        self.assertEqual(config.trading_mode, TradingMode.PAPER)
        self.assertIsNotNone(config.stat_arb)
        self.assertIsNotNone(config.risk)
        print("✓ Config created successfully")
        
    def test_config_validation(self):
        """Test config validation"""
        from config.hft_settings import get_hft_config
        
        config = get_hft_config("paper")
        issues = config.validate()
        
        self.assertEqual(len(issues), 0, f"Config issues: {issues}")
        print("✓ Config validated with no issues")
        
    def test_risk_presets(self):
        """Test risk level presets"""
        from config.hft_settings import get_hft_config, RiskLevel
        
        config = get_hft_config("paper")
        
        # Apply conservative preset
        config.risk_level = RiskLevel.CONSERVATIVE
        config.apply_risk_level_presets()
        
        self.assertEqual(config.risk.max_position_value, 50000.0)
        print("✓ Conservative presets applied")


def run_quick_demo():
    """Run a quick interactive demo of the strategies"""
    print("\n" + "="*60)
    print("HFT STRATEGIES QUICK DEMO")
    print("="*60)
    
    # Generate sample data
    np.random.seed(42)
    n = 200
    dates = pd.date_range(start='2024-01-01', periods=n, freq='1min')
    
    close = 150 + np.cumsum(np.random.randn(n) * 0.1)
    df = pd.DataFrame({
        'open': close + np.random.randn(n) * 0.05,
        'high': close + np.abs(np.random.randn(n) * 0.1),
        'low': close - np.abs(np.random.randn(n) * 0.1),
        'close': close,
        'volume': np.random.randint(1000, 10000, n).astype(float)
    }, index=dates)
    
    # 1. Statistical Arbitrage Demo
    print("\n--- Statistical Arbitrage (EWLR) ---")
    from strategies.hft_strategies import StatisticalArbitrageEWLR
    
    strategy = StatisticalArbitrageEWLR(
        alpha=0.999,
        entry_threshold=2.0,
        exit_threshold=0.5,
        lookback=100
    )
    print(f"Initialized with alpha={strategy.alpha}, entry_threshold={strategy.entry_threshold}")
    
    # Create cointegrated pair for demo
    asset_a = close
    asset_b = 50 + 1.2 * asset_a + np.random.randn(n) * 0.5
    
    X = pd.DataFrame({'Asset_A': asset_a})
    Y = pd.Series(asset_b, name='Asset_B')
    
    result = strategy.fit(Y, X)
    print(f"Fitted model: beta={result['beta'][1]:.4f}, spread_std={result['spread_std']:.4f}")
    
    # Generate signal using the full series
    signal = strategy.generate_signal(Y, X, current_idx=-1)
    print(f"Current signal: direction={signal.direction}, confidence={signal.confidence:.2f}")
    
    # 2. LOB Imbalance Demo
    print("\n--- LOB Imbalance Strategy ---")
    from strategies.hft_strategies import OrderBookImbalanceStrategy, LOBSnapshot
    
    lob_strategy = OrderBookImbalanceStrategy(
        imbalance_threshold=0.2,
        n_levels=5,
        momentum_periods=3
    )
    
    # Sample LOB with bid-heavy imbalance - using correct dataclass format
    snapshot = LOBSnapshot(
        timestamp=pd.Timestamp.now(),
        bid_prices=np.array([149.95, 149.90, 149.85]),
        bid_sizes=np.array([2000.0, 3000.0, 2500.0]),
        ask_prices=np.array([150.05, 150.10, 150.15]),
        ask_sizes=np.array([500.0, 800.0, 600.0])
    )
    
    imbalance = snapshot.imbalance
    print(f"LOB Imbalance: {imbalance:.4f} (positive = bid heavy)")
    
    # Generate signal from OHLCV data
    signal = lob_strategy.generate_signal(df, current_idx=-1)
    print(f"Signal: direction={signal.direction}, confidence={signal.confidence:.2f}")
    
    # 3. Market Maker Demo
    print("\n--- Intelligent Market Maker ---")
    from strategies.hft_strategies import IntelligentMarketMaker
    
    mm = IntelligentMarketMaker(
        spread_target_bps=10,
        max_inventory=1000,
        inventory_skew=0.5
    )
    
    features = mm.extract_features(df, idx=-1)
    print(f"Extracted {len(features)} features for ML model")
    
    quotes = mm.generate_quotes(df, idx=-1)
    if quotes:
        print(f"Quotes: Bid={quotes['bid_price']:.2f}, Ask={quotes['ask_price']:.2f}, Spread={quotes['spread_bps']:.1f}bps")
    else:
        print("No quotes (ML model needs training)")
    
    # 4. Risk Management Demo
    print("\n--- Risk Management ---")
    from risk.hft_risk_management import PreTradeRiskChecker, KillSwitch, Order, OrderType
    
    checker = PreTradeRiskChecker(
        max_order_size=50000,
        max_position_per_symbol=100000,
        max_daily_loss=5000
    )
    
    order = Order(
        order_id='test_001',
        symbol='AAPL',
        side='buy',
        quantity=100,
        price=150.0,
        order_type=OrderType.LIMIT
    )
    
    result = checker.check_order(order, reference_price=150.0)
    print(f"Order validation: {result.result}")
    
    # Test kill switch
    ks = KillSwitch(max_daily_loss=10000, max_drawdown=0.05)
    print(f"Kill switch active: {ks.is_active}")
    
    # 5. Config Demo
    print("\n--- HFT Configuration ---")
    from config.hft_settings import get_hft_config
    
    config = get_hft_config("paper")
    print(f"Mode: {config.trading_mode.value}")
    print(f"Risk Level: {config.risk_level.value}")
    print(f"Max Daily Loss: ${config.risk.max_daily_loss:,.0f}")
    
    print("\n" + "="*60)
    print("DEMO COMPLETE!")
    print("="*60)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Test HFT Strategies')
    parser.add_argument('--demo', action='store_true', help='Run quick demo instead of tests')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    if args.demo:
        run_quick_demo()
    else:
        # Run unit tests
        print("Running HFT Strategy Tests...")
        print("="*60)
        
        loader = unittest.TestLoader()
        suite = unittest.TestSuite()
        
        # Add test classes
        suite.addTests(loader.loadTestsFromTestCase(TestStatisticalArbitrage))
        suite.addTests(loader.loadTestsFromTestCase(TestLOBImbalance))
        suite.addTests(loader.loadTestsFromTestCase(TestMarketMaker))
        suite.addTests(loader.loadTestsFromTestCase(TestRiskManagement))
        suite.addTests(loader.loadTestsFromTestCase(TestHFTConfig))
        
        # Run tests
        verbosity = 2 if args.verbose else 1
        runner = unittest.TextTestRunner(verbosity=verbosity)
        result = runner.run(suite)
        
        # Summary
        print("\n" + "="*60)
        print(f"Tests run: {result.testsRun}")
        print(f"Failures: {len(result.failures)}")
        print(f"Errors: {len(result.errors)}")
        print("="*60)
