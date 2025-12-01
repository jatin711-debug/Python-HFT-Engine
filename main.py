"""
Trading Engine - Main Entry Point

A sophisticated ML-powered trading engine for generating
buy/sell/hold signals with backtesting capabilities.

Features:
- 150+ technical indicators via TA-Lib
- ML ensemble (XGBoost, LightGBM, CatBoost, RandomForest)
- News sentiment analysis (VADER, TextBlob, FinBERT)
- High-performance backtesting via VectorBT
- Long/short position support
- Advanced features (regime detection, fractal, momentum factors)
- Alternative data (Reddit sentiment, SEC filings)
- Institutional strategy ensemble
- Walk-forward validation

Usage:
    python main.py --symbol AAPL --mode backtest
    python main.py --symbol TSLA --mode live --fetch-news
    python main.py --symbol MSFT --mode walkforward
"""

import argparse
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any

import pandas as pd
import numpy as np

# Local imports
from config import DataSettings, TradingSettings, RiskSettings, FeatureSettings, SignalSettings
from config import XGBoostParams, LightGBMParams, EnsembleParams
from data.fetchers import MarketDataFetcher, NewsFetcher
from data.preprocessors import DataCleaner
from features import TechnicalFeatures, StatisticalFeatures, SentimentFeatures
from features import AdvancedFeatures, OptionsFeatures, MacroFeatures
from models.ml import GradientBoostingModels, EnsembleModel, EnsembleConfig
from signals import SignalGenerator, SignalConfig
from backtesting import BacktestEngine, BacktestConfig, BacktestResult
from backtesting import WalkForwardBacktest, MonteCarloValidator, WalkForwardResult
from strategies import (
    StrategyEnsemble,
    MomentumStrategy,
    MeanReversionStrategy,
    BreakoutStrategy,
    RegimeSwitchingStrategy,
)
from data.alternative import RedditSentimentFetcher, SECFetcher

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('trading_engine.log'),
    ]
)
logger = logging.getLogger(__name__)


class TradingEngine:
    """
    Main trading engine orchestrating all components.
    
    Pipeline:
    1. Fetch market data and news
    2. Clean and preprocess data
    3. Generate technical, statistical, and sentiment features
    4. Add advanced features (regime, fractal, momentum factors)
    5. Integrate alternative data (Reddit, SEC)
    6. Train/load ML models
    7. Generate trading signals via strategy ensemble
    8. Backtest or execute signals
    
    Example:
        >>> engine = TradingEngine()
        >>> engine.run(symbol="AAPL", mode="backtest")
    """
    
    def __init__(
        self,
        data_settings: DataSettings = None,
        trading_settings: TradingSettings = None,
        risk_settings: RiskSettings = None,
        feature_settings: FeatureSettings = None,
        signal_settings: SignalSettings = None,
    ):
        """
        Initialize trading engine.
        
        Args:
            data_settings: Data fetching configuration
            trading_settings: Trading parameters
            risk_settings: Risk management settings
            feature_settings: Feature engineering settings
            signal_settings: Signal generation settings
        """
        # Settings
        self.data_settings = data_settings or DataSettings()
        self.trading_settings = trading_settings or TradingSettings()
        self.risk_settings = risk_settings or RiskSettings()
        self.feature_settings = feature_settings or FeatureSettings()
        self.signal_settings = signal_settings or SignalSettings()
        
        # Components (initialized lazily)
        self.market_data_fetcher = None
        self.news_fetcher = None
        self.data_cleaner = None
        self.technical_features = None
        self.statistical_features = None
        self.sentiment_features = None
        self.advanced_features = None
        self.reddit_fetcher = None
        self.sec_fetcher = None
        self.ml_model = None
        self.signal_generator = None
        self.strategy_ensemble = None
        self.backtest_engine = None
        self.walk_forward_engine = None
        
        # State
        self.is_initialized = False
        self.model_path = Path("models/saved")
        self.model_path.mkdir(parents=True, exist_ok=True)
    
    def initialize(self, fetch_news: bool = True, use_alternative_data: bool = True):
        """Initialize all components."""
        logger.info("Initializing trading engine components...")
        
        # Data fetchers
        self.market_data_fetcher = MarketDataFetcher(
            source=self.data_settings.default_source,
        )
        
        if fetch_news:
            self.news_fetcher = NewsFetcher()
        
        # Preprocessor
        self.data_cleaner = DataCleaner()
        
        # Feature generators
        self.technical_features = TechnicalFeatures(
            sma_periods=self.feature_settings.sma_periods,
            ema_periods=self.feature_settings.ema_periods,
            rsi_period=self.feature_settings.rsi_period,
            macd_fast=self.feature_settings.macd_fast,
            macd_slow=self.feature_settings.macd_slow,
            macd_signal=self.feature_settings.macd_signal,
        )
        
        self.statistical_features = StatisticalFeatures(
            rolling_windows=self.feature_settings.rolling_windows,
        )
        
        if fetch_news:
            self.sentiment_features = SentimentFeatures(
                vader_weight=0.4,
                textblob_weight=0.2,
                finbert_weight=0.4,
            )
        
        # Advanced features (institutional-grade)
        self.advanced_features = AdvancedFeatures()
        
        # Alternative data sources
        if use_alternative_data:
            try:
                self.reddit_fetcher = RedditSentimentFetcher()
                logger.info("Reddit sentiment fetcher initialized")
            except Exception as e:
                logger.warning(f"Reddit fetcher not available: {e}")
                self.reddit_fetcher = None
            
            try:
                self.sec_fetcher = SECFetcher()
                logger.info("SEC filings fetcher initialized")
            except Exception as e:
                logger.warning(f"SEC fetcher not available: {e}")
                self.sec_fetcher = None
        
        # Strategy ensemble (institutional-grade)
        self.strategy_ensemble = StrategyEnsemble()
        
        # Backtesting
        self.backtest_engine = BacktestEngine(
            config=BacktestConfig(
                initial_capital=self.trading_settings.initial_capital,
                commission=self.trading_settings.commission_pct,
                slippage=self.trading_settings.slippage_pct,
                position_size=self.trading_settings.max_position_pct,
                allow_shorting=self.trading_settings.allow_shorting,
            )
        )
        
        # Walk-forward backtesting
        self.walk_forward_engine = WalkForwardBacktest(
            train_period=252,  # 1 year training
            test_period=63,    # 3 months testing
            step_size=21,      # Roll monthly
        )
        
        self.is_initialized = True
        logger.info("Trading engine initialized successfully")
    
    def fetch_data(
        self,
        symbol: str,
        start_date: datetime = None,
        end_date: datetime = None,
        fetch_news: bool = True,
    ) -> Dict[str, Any]:
        """
        Fetch market data and news for a symbol.
        
        Args:
            symbol: Stock symbol
            start_date: Start date for data
            end_date: End date for data
            fetch_news: Whether to fetch news
            
        Returns:
            Dictionary with price_data and news_data
        """
        end_date = end_date or datetime.now()
        start_date = start_date or (end_date - timedelta(days=365*5))  # 5 years for sufficient data
        
        logger.info(f"Fetching data for {symbol} from {start_date} to {end_date}")
        
        # Fetch price data
        price_data = self.market_data_fetcher.fetch(
            symbols=symbol,
            start=start_date.strftime("%Y-%m-%d") if start_date else None,
            end=end_date.strftime("%Y-%m-%d") if end_date else None,
            interval=self.data_settings.default_interval,
        )
        
        logger.info(f"Fetched {len(price_data)} price records")
        
        # Fetch news if enabled
        news_data = {}
        if fetch_news and self.news_fetcher:
            try:
                articles = self.news_fetcher.fetch_news(
                    symbol=symbol,
                    max_articles=100,
                )
                
                # Organize by date
                for article in articles:
                    date_str = article.published_at.strftime('%Y-%m-%d')
                    if date_str not in news_data:
                        news_data[date_str] = []
                    news_data[date_str].append(article.__dict__)
                
                logger.info(f"Fetched {len(articles)} news articles")
            except Exception as e:
                logger.warning(f"Failed to fetch news: {e}")
        
        return {
            'price_data': price_data,
            'news_data': news_data,
        }
    
    def prepare_features(
        self,
        price_data: pd.DataFrame,
        news_data: Dict = None,
        symbol: str = None,
        use_alternative_data: bool = True,
    ) -> pd.DataFrame:
        """
        Prepare all features for ML model.
        
        Args:
            price_data: OHLCV data
            news_data: News articles by date
            symbol: Stock symbol for alternative data
            use_alternative_data: Whether to add Reddit/SEC data
            
        Returns:
            DataFrame with all features
        """
        logger.info("Preparing features...")
        
        # Clean data
        df = self.data_cleaner.clean_ohlcv(price_data)
        logger.info(f"Cleaned data: {len(df)} rows")
        
        # Add technical indicators
        df = self.technical_features.add_all_indicators(df)
        logger.info(f"Added technical indicators: {df.shape[1]} columns")
        
        # Add statistical features
        df = self.statistical_features.add_all_features(df)
        logger.info(f"Added statistical features: {df.shape[1]} columns")
        
        # Defragment DataFrame to fix performance warning
        df = df.copy()
        
        # Add sentiment features if available
        if news_data and self.sentiment_features:
            df = self.sentiment_features.add_sentiment_features(df, news_data)
            logger.info(f"Added sentiment features: {df.shape[1]} columns")
        
        # Add advanced institutional features
        if self.advanced_features:
            try:
                df = self.advanced_features.add_all_advanced_features(df)
                logger.info(f"Added advanced features: {df.shape[1]} columns")
            except Exception as e:
                logger.warning(f"Could not add advanced features: {e}")
        
        # Add alternative data features
        if use_alternative_data and symbol:
            # Reddit sentiment
            if self.reddit_fetcher:
                try:
                    df = self.reddit_fetcher.get_wsb_sentiment_features(df, symbol)
                    logger.info(f"Added Reddit sentiment features: {df.shape[1]} columns")
                except Exception as e:
                    logger.warning(f"Could not add Reddit features: {e}")
            
            # SEC filings
            if self.sec_fetcher:
                try:
                    df = self.sec_fetcher.get_sec_features(df, symbol)
                    logger.info(f"Added SEC filing features: {df.shape[1]} columns")
                except Exception as e:
                    logger.warning(f"Could not add SEC features: {e}")
        
        # Create target variable (next day return direction)
        df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
        
        # Drop rows with NaN - only drop rows where critical columns are NaN
        initial_len = len(df)
        # Keep rows where we have at least the basic features
        df = df.dropna(subset=['close', 'target'])
        # Fill remaining NaN with 0 for features (after warmup period)
        df = df.fillna(0)
        logger.info(f"Dropped {initial_len - len(df)} rows with NaN values, {len(df)} rows remaining")
        
        return df
    
    def train_model(
        self,
        df: pd.DataFrame,
        validation_split: float = 0.2,
    ) -> EnsembleModel:
        """
        Train ML ensemble model.
        
        Args:
            df: DataFrame with features and target
            validation_split: Fraction of data for validation
            
        Returns:
            Trained EnsembleModel
        """
        logger.info("Training ML ensemble model...")
        
        # Prepare features and target
        feature_cols = [c for c in df.columns if c not in [
            'open', 'high', 'low', 'close', 'volume', 'target', 'date'
        ]]
        
        X = df[feature_cols]
        y = df['target']
        
        # Time-series split
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
        
        logger.info(f"Training set: {len(X_train)}, Validation set: {len(X_val)}")
        
        # Create and train ensemble
        config = EnsembleConfig(
            use_xgb=True,
            use_lgb=True,
            use_catboost=True,
            use_random_forest=True,
            use_stacking=True,
            task='classification',
        )
        
        self.ml_model = EnsembleModel(config=config)
        self.ml_model.fit(X_train, y_train, X_val, y_val)
        
        # Evaluate
        metrics = self.ml_model.evaluate(X_val, y_val)
        logger.info(f"Validation metrics: AUC={metrics.roc_auc:.4f}, F1={metrics.f1:.4f}")
        
        return self.ml_model
    
    def generate_signals(
        self,
        df: pd.DataFrame,
        symbol: str,
        use_strategy_ensemble: bool = True,
    ) -> pd.DataFrame:
        """
        Generate trading signals.
        
        Args:
            df: DataFrame with features
            symbol: Stock symbol
            use_strategy_ensemble: Whether to use institutional strategy ensemble
            
        Returns:
            DataFrame with signals
        """
        logger.info("Generating trading signals...")
        
        # Initialize signal generator
        signal_config = SignalConfig(
            strong_buy_threshold=self.signal_settings.strong_buy_threshold,
            buy_threshold=self.signal_settings.buy_threshold,
            sell_threshold=self.signal_settings.sell_threshold,
            strong_sell_threshold=self.signal_settings.strong_sell_threshold,
            min_confidence=self.signal_settings.min_confidence,
        )
        
        self.signal_generator = SignalGenerator(
            ml_model=self.ml_model,
            sentiment_analyzer=self.sentiment_features,
            config=signal_config,
        )
        
        # Generate ML-based signals
        signals = self.signal_generator.generate_signals(df, symbol=symbol)
        signals_df = self.signal_generator.signals_to_dataframe(signals)
        
        # Enhance with strategy ensemble if enabled
        if use_strategy_ensemble and self.strategy_ensemble:
            try:
                # Generate signals for each row
                ensemble_results = []
                for idx in range(len(df)):
                    try:
                        result = self.strategy_ensemble.generate_ensemble_signal(df, idx)
                        ensemble_results.append(result)
                    except Exception:
                        ensemble_results.append({
                            'direction': 0,
                            'strength': 0,
                            'confidence': 0,
                        })
                
                # Create ensemble signals dataframe
                ensemble_signals = pd.DataFrame(ensemble_results)
                
                # Merge with ML signals
                if 'strength' in ensemble_signals.columns:
                    signals_df = signals_df.reset_index(drop=True)
                    
                    # Combine: weight ML 60%, strategy ensemble 40%
                    ml_strength = signals_df['signal_value'].values[:len(ensemble_signals)]
                    strategy_strength = ensemble_signals['strength'].fillna(0).values
                    strategy_direction = ensemble_signals['direction'].fillna(0).values
                    strategy_confidence = ensemble_signals['confidence'].fillna(0).values if 'confidence' in ensemble_signals.columns else np.ones_like(strategy_direction) * 0.5
                    
                    # Combine signal with direction
                    strategy_signal = strategy_strength * strategy_direction
                    combined_strength = 0.6 * ml_strength + 0.4 * strategy_signal[:len(ml_strength)]
                    
                    # Apply confidence filter - only trade when both ML and strategy agree
                    # and have reasonable confidence
                    agreement_bonus = np.sign(ml_strength) == np.sign(strategy_signal[:len(ml_strength)])
                    combined_strength = combined_strength * (1 + 0.2 * agreement_bonus.astype(float))
                    
                    # Reduce signal when ML and strategy disagree
                    combined_strength = combined_strength * (0.5 + 0.5 * agreement_bonus.astype(float))
                    
                    signals_df['combined_signal'] = combined_strength
                    signals_df['strategy_signal'] = strategy_signal[:len(ml_strength)]
                    signals_df['ml_strategy_agreement'] = agreement_bonus.astype(int)
                    
                    logger.info("Strategy ensemble signals integrated")
                    logger.info(f"ML-Strategy agreement rate: {agreement_bonus.mean()*100:.1f}%")
                    
            except Exception as e:
                logger.warning(f"Could not integrate strategy ensemble: {e}")
        
        # Summary
        summary = self.signal_generator.get_signal_summary(signals)
        logger.info(f"Signal summary: {summary}")
        
        return signals_df
    
    def backtest(
        self,
        price_data: pd.DataFrame,
        signals_df: pd.DataFrame,
    ) -> BacktestResult:
        """
        Run backtest on signals.
        
        Args:
            price_data: OHLCV data
            signals_df: DataFrame with signals
            
        Returns:
            BacktestResult
        """
        logger.info("Running backtest...")
        
        # Extract signal values (use combined if available)
        if 'combined_signal' in signals_df.columns:
            raw_signals = signals_df['combined_signal'].values
        else:
            raw_signals = signals_df['signal_value'].values
        
        # Get agreement filter if available
        agreement = None
        if 'ml_strategy_agreement' in signals_df.columns:
            agreement = signals_df['ml_strategy_agreement'].values
        
        # Convert to discrete signals: 1 (buy), -1 (sell), 0 (hold)
        # Use very selective thresholds - only trade on strong conviction
        buy_threshold = 0.35   # Higher threshold for stronger signals
        sell_threshold = -0.35
        
        signals = np.zeros(len(raw_signals))
        signals[raw_signals > buy_threshold] = 1
        signals[raw_signals < sell_threshold] = -1
        
        # Align data FIRST before applying filters
        min_len = min(len(price_data), len(signals))
        price_data = price_data.iloc[:min_len].copy()
        signals = signals[:min_len]
        if agreement is not None:
            agreement = agreement[:min_len]
        
        # Apply agreement filter - only trade when ML and strategy agree
        if agreement is not None:
            # Reduce signals where there's disagreement
            signals[agreement == 0] = 0
            logger.info("Applied ML-Strategy agreement filter")
        
        # Add momentum confirmation filter
        if len(price_data) > 20:
            close = price_data['close'].values
            
            # RSI filter - avoid overbought/oversold trades against momentum
            if 'rsi' in price_data.columns:
                rsi = price_data['rsi'].values
            else:
                delta = pd.Series(close).diff()
                gain = delta.where(delta > 0, 0).rolling(14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                rsi = (100 - (100 / (1 + gain / (loss + 1e-10)))).values
            
            # Don't buy overbought or sell oversold
            signals[(signals == 1) & (rsi > 70)] = 0   # No buys when overbought
            signals[(signals == -1) & (rsi < 30)] = 0  # No shorts when oversold
        
        # Add trend filter: only trade in direction of longer-term trend
        if len(price_data) > 50:
            close = price_data['close'].values
            sma_50 = pd.Series(close).rolling(50).mean().values
            sma_20 = pd.Series(close).rolling(20).mean().values
            
            # Trend conditions
            uptrend = (close > sma_50) & (sma_20 > sma_50)
            downtrend = (close < sma_50) & (sma_20 < sma_50)
            
            # Only allow trades in clear trend direction
            signals[(signals == -1) & uptrend] = 0   # No shorts in uptrend
            signals[(signals == 1) & downtrend] = 0  # No longs in downtrend
            logger.info("Applied trend filter")
        
        # Minimum holding period - prevent rapid switching
        min_hold_bars = 5  # Hold for at least 5 bars
        last_trade_bar = -min_hold_bars
        for i in range(len(signals)):
            if signals[i] != 0:
                if i - last_trade_bar < min_hold_bars:
                    signals[i] = 0  # Cancel signal if too soon
                else:
                    last_trade_bar = i
        
        logger.info(f"Signal distribution: Buy={np.sum(signals == 1)}, Sell={np.sum(signals == -1)}, Hold={np.sum(signals == 0)}")
        
        # Calculate ATR for dynamic stop loss
        if 'high' in price_data.columns and 'low' in price_data.columns:
            high = price_data['high'].values
            low = price_data['low'].values
            close = price_data['close'].values
            tr = np.maximum(high - low, np.maximum(abs(high - np.roll(close, 1)), abs(low - np.roll(close, 1))))
            atr = pd.Series(tr).rolling(14).mean().values
            atr_pct = atr / close
            # Tighter stops with better risk-reward
            stop_loss = np.clip(1.5 * atr_pct, 0.01, 0.03)  # 1-3% based on ATR
            take_profit = np.clip(4.0 * atr_pct, 0.02, 0.10)  # 2.5x risk-reward minimum
            avg_sl = np.nanmean(stop_loss)
            avg_tp = np.nanmean(take_profit)
            logger.info(f"Dynamic stops: SL={avg_sl*100:.1f}%, TP={avg_tp*100:.1f}%")
        else:
            stop_loss = 0.015  # 1.5% default stop loss
            take_profit = 0.06  # 6% default take profit (4:1 risk-reward)
        
        # Run backtest with risk management
        result = self.backtest_engine.run(
            prices=price_data,
            signals=signals,
            stop_loss=stop_loss if isinstance(stop_loss, float) else avg_sl,
            take_profit=take_profit if isinstance(take_profit, float) else avg_tp,
        )
        
        # Print summary
        self.backtest_engine.print_summary(result)
        
        return result
    
    def walk_forward_backtest(
        self,
        df: pd.DataFrame,
        symbol: str,
    ) -> WalkForwardResult:
        """
        Run walk-forward backtest with rolling optimization.
        
        This provides more realistic performance estimates by
        continuously re-training on past data and testing on future data.
        
        Args:
            df: DataFrame with features
            symbol: Stock symbol
            
        Returns:
            WalkForwardResult
        """
        logger.info("Running walk-forward backtest...")
        
        def signal_func(data: pd.DataFrame) -> pd.Series:
            """Generate signals for a data window."""
            # Get feature columns
            feature_cols = [c for c in data.columns if c not in [
                'open', 'high', 'low', 'close', 'volume', 'target', 'date'
            ]]
            
            if self.ml_model and hasattr(self.ml_model, 'predict_proba'):
                # Use ML model probabilities
                X = data[feature_cols]
                probs = self.ml_model.predict_proba(X)
                signals = (probs - 0.5) * 2  # Scale to -1 to 1
            else:
                # Fallback to simple momentum
                signals = np.sign(data['close'].pct_change(5))
            
            return pd.Series(signals, index=data.index)
        
        def backtest_func(data: pd.DataFrame, signals: pd.Series) -> Dict:
            """Run backtest on a window."""
            try:
                result = self.backtest_engine.run(
                    prices=data,
                    signals=signals.values,
                )
                return {
                    'total_return': result.total_return,
                    'sharpe_ratio': result.sharpe_ratio,
                    'total_trades': result.total_trades,
                }
            except Exception as e:
                logger.debug(f"Backtest error: {e}")
                return {
                    'total_return': 0,
                    'sharpe_ratio': 0,
                    'total_trades': 0,
                }
        
        def optimize_func(train_data: pd.DataFrame) -> Dict:
            """Optimize parameters on training data."""
            # Re-train model on training data
            feature_cols = [c for c in train_data.columns if c not in [
                'open', 'high', 'low', 'close', 'volume', 'target', 'date'
            ]]
            
            if 'target' in train_data.columns:
                X = train_data[feature_cols]
                y = train_data['target']
                
                if len(X) > 100:  # Minimum samples for training
                    # Quick training with reduced parameters
                    config = EnsembleConfig(
                        use_xgb=True,
                        use_lgb=True,
                        use_catboost=False,  # Skip for speed
                        use_random_forest=False,
                        use_stacking=False,
                        task='classification',
                    )
                    
                    self.ml_model = EnsembleModel(config=config)
                    self.ml_model.fit(X, y)
            
            return {}
        
        # Run walk-forward analysis
        result = self.walk_forward_engine.run(
            df=df,
            signal_func=signal_func,
            backtest_func=backtest_func,
            optimize_func=optimize_func,
        )
        
        # Print summary
        self.walk_forward_engine.print_summary(result)
        
        # Run Monte Carlo validation
        if result.all_trades is not None and len(result.all_trades) > 0:
            validator = MonteCarloValidator(n_simulations=1000)
            validation = validator.validate_trades(result.all_trades)
            
            logger.info(f"\n📊 Monte Carlo Validation:")
            logger.info(f"  Significant: {validation['is_significant']}")
            logger.info(f"  P-Value: {validation['p_value']:.4f}")
            logger.info(f"  95% CI: ({validation['confidence_interval'][0]:.2f}, {validation['confidence_interval'][1]:.2f})")
        
        return result
    
    def run(
        self,
        symbol: str,
        mode: str = 'backtest',
        start_date: datetime = None,
        end_date: datetime = None,
        fetch_news: bool = True,
        use_alternative_data: bool = True,
        train_model: bool = True,
        save_model: bool = True,
        plot_results: bool = True,
    ) -> Dict[str, Any]:
        """
        Run the complete trading pipeline.
        
        Args:
            symbol: Stock symbol to analyze
            mode: 'backtest', 'live', or 'walkforward'
            start_date: Start date for data
            end_date: End date for data
            fetch_news: Whether to fetch news data
            use_alternative_data: Whether to use Reddit/SEC data
            train_model: Whether to train a new model
            save_model: Whether to save the trained model
            plot_results: Whether to plot backtest results
            
        Returns:
            Dictionary with results
        """
        logger.info(f"Running trading engine for {symbol} in {mode} mode")
        
        # Initialize if needed
        if not self.is_initialized:
            self.initialize(fetch_news=fetch_news, use_alternative_data=use_alternative_data)
        
        # Fetch data
        data = self.fetch_data(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            fetch_news=fetch_news,
        )
        
        # Prepare features
        df = self.prepare_features(
            price_data=data['price_data'],
            news_data=data['news_data'],
            symbol=symbol,
            use_alternative_data=use_alternative_data,
        )
        
        # Train or load model
        if train_model and mode != 'walkforward':
            self.train_model(df)
            
            if save_model:
                model_path = self.model_path / f"{symbol}_model"
                self.ml_model.save(str(model_path))
                logger.info(f"Model saved to {model_path}")
        elif not train_model:
            # Try to load existing model
            model_path = self.model_path / f"{symbol}_model"
            if model_path.exists():
                self.ml_model = EnsembleModel()
                self.ml_model.load(str(model_path))
                logger.info(f"Model loaded from {model_path}")
            else:
                logger.warning("No saved model found, training new model")
                self.train_model(df)
        
        # Handle different modes
        if mode == 'walkforward':
            # Walk-forward backtesting
            wf_result = self.walk_forward_backtest(df, symbol)
            
            return {
                'symbol': symbol,
                'mode': mode,
                'data': data,
                'features': df,
                'walkforward_result': wf_result,
                'backtest_result': None,
            }
        
        # Generate signals
        signals_df = self.generate_signals(df, symbol)
        
        # Run backtest
        if mode == 'backtest':
            result = self.backtest(data['price_data'], signals_df)
            
            if plot_results:
                try:
                    self.backtest_engine.plot_results()
                except Exception as e:
                    logger.warning(f"Could not plot results: {e}")
        else:
            result = None
            # For live mode, get latest signals
            latest_signal = signals_df.iloc[-1]
            logger.info(f"\n📊 LATEST SIGNAL FOR {symbol}:")
            logger.info(f"  Signal: {latest_signal['signal']}")
            logger.info(f"  Confidence: {latest_signal['confidence']:.2%}")
            logger.info(f"  Entry Price: ${latest_signal['entry_price']:.2f}")
            if latest_signal['stop_loss']:
                logger.info(f"  Stop Loss: ${latest_signal['stop_loss']:.2f}")
            if latest_signal['take_profit']:
                logger.info(f"  Take Profit: ${latest_signal['take_profit']:.2f}")
            
            # Show strategy ensemble info if available
            if 'dominant_strategy' in signals_df.columns:
                logger.info(f"  Dominant Strategy: {latest_signal.get('dominant_strategy', 'N/A')}")
            if 'market_regime' in signals_df.columns:
                logger.info(f"  Market Regime: {latest_signal.get('market_regime', 'N/A')}")
        
        return {
            'symbol': symbol,
            'mode': mode,
            'data': data,
            'features': df,
            'signals': signals_df,
            'backtest_result': result,
        }
    
    def save_results(
        self,
        results: Dict[str, Any],
        output_dir: str = 'results',
    ):
        """Save results to files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        symbol = results['symbol']
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save signals
        signals_path = output_path / f"{symbol}_signals_{timestamp}.csv"
        results['signals'].to_csv(signals_path, index=False)
        logger.info(f"Signals saved to {signals_path}")
        
        # Save features (sample)
        features_path = output_path / f"{symbol}_features_{timestamp}.csv"
        results['features'].tail(100).to_csv(features_path)
        logger.info(f"Features saved to {features_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Sophisticated ML-powered trading engine'
    )
    parser.add_argument(
        '--symbol', '-s',
        type=str,
        default='AAPL',
        help='Stock symbol to analyze'
    )
    parser.add_argument(
        '--mode', '-m',
        type=str,
        choices=['backtest', 'live', 'walkforward'],
        default='backtest',
        help='Running mode: backtest, live, or walkforward'
    )
    parser.add_argument(
        '--start-date',
        type=str,
        help='Start date (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--end-date',
        type=str,
        help='End date (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--fetch-news',
        action='store_true',
        help='Fetch news for sentiment analysis'
    )
    parser.add_argument(
        '--train-model',
        action='store_true',
        default=True,
        help='Train a new model'
    )
    parser.add_argument(
        '--no-train',
        action='store_true',
        help='Use existing model instead of training'
    )
    parser.add_argument(
        '--save-results',
        action='store_true',
        help='Save results to files'
    )
    parser.add_argument(
        '--no-plot',
        action='store_true',
        help='Disable plotting'
    )
    parser.add_argument(
        '--alternative-data',
        action='store_true',
        help='Fetch alternative data (Reddit, SEC filings)'
    )
    parser.add_argument(
        '--no-alternative-data',
        action='store_true',
        help='Disable alternative data fetching'
    )
    
    args = parser.parse_args()
    
    # Parse dates
    start_date = None
    end_date = None
    if args.start_date:
        start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
    if args.end_date:
        end_date = datetime.strptime(args.end_date, '%Y-%m-%d')
    
    # Determine alternative data usage
    use_alt_data = args.alternative_data and not args.no_alternative_data
    
    # Run engine
    engine = TradingEngine()
    
    results = engine.run(
        symbol=args.symbol,
        mode=args.mode,
        start_date=start_date,
        end_date=end_date,
        fetch_news=args.fetch_news,
        use_alternative_data=use_alt_data,
        train_model=not args.no_train,
        plot_results=not args.no_plot,
    )
    
    if args.save_results:
        engine.save_results(results)
    
    return results


if __name__ == '__main__':
    main()
