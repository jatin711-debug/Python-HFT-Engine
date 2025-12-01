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

Usage:
    python main.py --symbol AAPL --mode backtest
    python main.py --symbol TSLA --mode live --fetch-news
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
from models.ml import GradientBoostingModels, EnsembleModel, EnsembleConfig
from signals import SignalGenerator, SignalConfig
from backtesting import BacktestEngine, BacktestConfig, BacktestResult

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
    4. Train/load ML models
    5. Generate trading signals
    6. Backtest or execute signals
    
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
        self.ml_model = None
        self.signal_generator = None
        self.backtest_engine = None
        
        # State
        self.is_initialized = False
        self.model_path = Path("models/saved")
        self.model_path.mkdir(parents=True, exist_ok=True)
    
    def initialize(self, fetch_news: bool = True):
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
    ) -> pd.DataFrame:
        """
        Prepare all features for ML model.
        
        Args:
            price_data: OHLCV data
            news_data: News articles by date
            
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
    ) -> pd.DataFrame:
        """
        Generate trading signals.
        
        Args:
            df: DataFrame with features
            symbol: Stock symbol
            
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
        
        # Generate signals
        signals = self.signal_generator.generate_signals(df, symbol=symbol)
        signals_df = self.signal_generator.signals_to_dataframe(signals)
        
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
        
        # Extract signal values
        signals = signals_df['signal_value'].values
        
        # Align data
        min_len = min(len(price_data), len(signals))
        price_data = price_data.iloc[:min_len]
        signals = signals[:min_len]
        
        # Run backtest
        result = self.backtest_engine.run(
            prices=price_data,
            signals=signals,
        )
        
        # Print summary
        self.backtest_engine.print_summary(result)
        
        return result
    
    def run(
        self,
        symbol: str,
        mode: str = 'backtest',
        start_date: datetime = None,
        end_date: datetime = None,
        fetch_news: bool = True,
        train_model: bool = True,
        save_model: bool = True,
        plot_results: bool = True,
    ) -> Dict[str, Any]:
        """
        Run the complete trading pipeline.
        
        Args:
            symbol: Stock symbol to analyze
            mode: 'backtest' or 'live'
            start_date: Start date for data
            end_date: End date for data
            fetch_news: Whether to fetch news data
            train_model: Whether to train a new model
            save_model: Whether to save the trained model
            plot_results: Whether to plot backtest results
            
        Returns:
            Dictionary with results
        """
        logger.info(f"Running trading engine for {symbol} in {mode} mode")
        
        # Initialize if needed
        if not self.is_initialized:
            self.initialize(fetch_news=fetch_news)
        
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
        )
        
        # Train or load model
        if train_model:
            self.train_model(df)
            
            if save_model:
                model_path = self.model_path / f"{symbol}_model"
                self.ml_model.save(str(model_path))
                logger.info(f"Model saved to {model_path}")
        else:
            # Try to load existing model
            model_path = self.model_path / f"{symbol}_model"
            if model_path.exists():
                self.ml_model = EnsembleModel()
                self.ml_model.load(str(model_path))
                logger.info(f"Model loaded from {model_path}")
            else:
                logger.warning("No saved model found, training new model")
                self.train_model(df)
        
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
        choices=['backtest', 'live'],
        default='backtest',
        help='Running mode'
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
    
    args = parser.parse_args()
    
    # Parse dates
    start_date = None
    end_date = None
    if args.start_date:
        start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
    if args.end_date:
        end_date = datetime.strptime(args.end_date, '%Y-%m-%d')
    
    # Run engine
    engine = TradingEngine()
    
    results = engine.run(
        symbol=args.symbol,
        mode=args.mode,
        start_date=start_date,
        end_date=end_date,
        fetch_news=args.fetch_news,
        train_model=not args.no_train,
        plot_results=not args.no_plot,
    )
    
    if args.save_results:
        engine.save_results(results)
    
    return results


if __name__ == '__main__':
    main()
