"""
Microstructure Features for High-Frequency Trading.

Features specifically designed for 1-5 second candle trading on crypto markets.
These replace the 150+ traditional indicators with ~20 high-signal features
optimized for short-term price prediction.

Key Features:
- Volume/trade imbalance (buy vs sell pressure)
- Micro momentum (ultra-short-term trends)
- Spread dynamics (liquidity changes)
- VWAP deviation (fair value)
- Volatility regime (risk assessment)
- Order flow signals (smart money detection)

Usage:
    from features.microstructure_features import MicrostructureFeatures
    
    micro = MicrostructureFeatures()
    df = micro.add_all_features(df, sentiment=crypto_sentiment)
"""

import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numba import jit

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class MicrostructureConfig:
    """Configuration for microstructure features."""
    
    # Momentum periods (in candles, so 5 = 5 seconds for 1s candles)
    momentum_periods: tuple = (1, 3, 5, 10, 30)
    
    # VWAP lookback
    vwap_period: int = 60  # 1 minute for 1s candles
    
    # Volatility windows
    volatility_windows: tuple = (10, 30, 60)
    
    # Volume imbalance windows
    volume_imbalance_windows: tuple = (5, 10, 30)
    
    # Large trade threshold (as multiple of average)
    whale_threshold: float = 5.0
    
    # Spread normalization period
    spread_ma_period: int = 30


# =============================================================================
# NUMBA-OPTIMIZED FUNCTIONS (for speed)
# =============================================================================

@jit(nopython=True, cache=True)
def calc_returns(prices: np.ndarray) -> np.ndarray:
    """Fast return calculation."""
    returns = np.zeros(len(prices))
    returns[1:] = (prices[1:] - prices[:-1]) / prices[:-1]
    return returns


@jit(nopython=True, cache=True)
def calc_rolling_std(values: np.ndarray, window: int) -> np.ndarray:
    """Fast rolling standard deviation."""
    result = np.zeros(len(values))
    result[:window] = np.nan
    
    for i in range(window, len(values)):
        result[i] = np.std(values[i-window:i])
    
    return result


@jit(nopython=True, cache=True)
def calc_rolling_zscore(values: np.ndarray, window: int) -> np.ndarray:
    """Fast rolling z-score."""
    result = np.zeros(len(values))
    result[:window] = np.nan
    
    for i in range(window, len(values)):
        window_vals = values[i-window:i]
        mean = np.mean(window_vals)
        std = np.std(window_vals)
        if std > 0:
            result[i] = (values[i] - mean) / std
        else:
            result[i] = 0
    
    return result


@jit(nopython=True, cache=True)
def calc_rolling_corr(x: np.ndarray, y: np.ndarray, window: int) -> np.ndarray:
    """Fast rolling correlation."""
    result = np.zeros(len(x))
    result[:window] = np.nan
    
    for i in range(window, len(x)):
        x_win = x[i-window:i]
        y_win = y[i-window:i]
        
        x_mean = np.mean(x_win)
        y_mean = np.mean(y_win)
        
        numerator = np.sum((x_win - x_mean) * (y_win - y_mean))
        denominator = np.sqrt(np.sum((x_win - x_mean)**2) * np.sum((y_win - y_mean)**2))
        
        if denominator > 0:
            result[i] = numerator / denominator
        else:
            result[i] = 0
    
    return result


@jit(nopython=True, cache=True)
def calc_vwap(prices: np.ndarray, volumes: np.ndarray, window: int) -> np.ndarray:
    """Calculate VWAP."""
    result = np.zeros(len(prices))
    result[:window] = np.nan
    
    for i in range(window, len(prices)):
        vol_sum = np.sum(volumes[i-window:i])
        if vol_sum > 0:
            result[i] = np.sum(prices[i-window:i] * volumes[i-window:i]) / vol_sum
        else:
            result[i] = prices[i]
    
    return result


# =============================================================================
# MICROSTRUCTURE FEATURES CLASS
# =============================================================================

class MicrostructureFeatures:
    """
    High-signal features for 1-5 second trading.
    
    Replaces 150+ traditional indicators with ~20 focused features:
    1. Micro Momentum (returns at multiple short horizons)
    2. Volume Imbalance (buy vs sell pressure)
    3. VWAP Deviation (fair value signal)
    4. Spread Dynamics (liquidity changes)
    5. Volatility Regime (risk assessment)
    6. Order Flow (smart money detection)
    7. Return Autocorrelation (mean reversion signal)
    8. Trade Intensity (activity bursts)
    """
    
    def __init__(self, config: MicrostructureConfig = None):
        self.config = config or MicrostructureConfig()
        logger.info("MicrostructureFeatures initialized")
    
    def add_all_features(
        self,
        df: pd.DataFrame,
        sentiment: Optional[Any] = None,  # CryptoSentiment from Binance handler
    ) -> pd.DataFrame:
        """
        Add all microstructure features to DataFrame.
        
        Args:
            df: OHLCV DataFrame with columns: open, high, low, close, volume
            sentiment: Optional CryptoSentiment object for trade flow features
            
        Returns:
            DataFrame with added features
        """
        df = df.copy()
        
        # Ensure we have required columns
        required = ['open', 'high', 'low', 'close', 'volume']
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        logger.info(f"Adding microstructure features to {len(df)} rows...")
        
        # 1. Micro Momentum Features
        df = self._add_momentum_features(df)
        
        # 2. Volume Imbalance Features
        df = self._add_volume_imbalance_features(df)
        
        # 3. VWAP Features
        df = self._add_vwap_features(df)
        
        # 4. Spread/Range Features
        df = self._add_spread_features(df)
        
        # 5. Volatility Features
        df = self._add_volatility_features(df)
        
        # 6. Return Autocorrelation (mean reversion signal)
        df = self._add_autocorrelation_features(df)
        
        # 7. Trade Intensity Features
        df = self._add_trade_intensity_features(df)
        
        # 8. Candle Pattern Features (simplified)
        df = self._add_candle_patterns(df)
        
        # 9. External Sentiment (if provided)
        if sentiment is not None:
            df = self._add_sentiment_features(df, sentiment)
            
        # 10. Technical Indicators (RSI, ADX for filtering)
        df = self._add_technical_features(df)
        
        # 11. Composite Signals
        df = self._add_composite_signals(df)
        
        # Fill NaN with 0 (happens at start due to lookback)
        df = df.fillna(0)
        
        logger.info(f"Added features. Total columns: {len(df.columns)}")
        return df
    
    # -------------------------------------------------------------------------
    # FEATURE GROUPS
    # -------------------------------------------------------------------------
    
    def _add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add micro momentum features at multiple horizons."""
        close = df['close'].values
        
        for period in self.config.momentum_periods:
            # Simple return
            df[f'return_{period}'] = df['close'].pct_change(period)
            
            # Log return (more stable)
            df[f'logret_{period}'] = np.log(df['close'] / df['close'].shift(period))
            
        # Momentum acceleration (2nd derivative)
        df['momentum_accel'] = df['return_5'] - df['return_5'].shift(5)
        
        # Momentum consistency (sign agreement across horizons)
        signs = []
        for period in [1, 3, 5, 10]:
            if f'return_{period}' in df.columns:
                signs.append(np.sign(df[f'return_{period}']))
        
        if signs:
            df['momentum_agreement'] = sum(signs) / len(signs)
        
        return df
    
    def _add_volume_imbalance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume imbalance features (proxy for buy/sell pressure)."""
        
        # Estimate buy/sell volume from candle structure
        # If close > open: mostly buying, else mostly selling
        df['is_bullish'] = (df['close'] > df['open']).astype(float)
        
        # Volume split estimate
        df['buy_volume_est'] = df['volume'] * df['is_bullish']
        df['sell_volume_est'] = df['volume'] * (1 - df['is_bullish'])
        
        # Rolling volume imbalance at different windows
        for window in self.config.volume_imbalance_windows:
            buy_sum = df['buy_volume_est'].rolling(window).sum()
            sell_sum = df['sell_volume_est'].rolling(window).sum()
            total = buy_sum + sell_sum
            
            df[f'vol_imbalance_{window}'] = (buy_sum - sell_sum) / total.replace(0, 1)
        
        # Taker buy ratio (if available from Binance)
        if 'taker_buy_volume' in df.columns:
            df['taker_buy_ratio'] = df['taker_buy_volume'] / df['volume'].replace(0, 1)
            df['taker_imbalance'] = df['taker_buy_ratio'] - 0.5  # Center around 0
        
        # Volume surge detection
        vol_ma = df['volume'].rolling(30).mean()
        df['volume_surge'] = df['volume'] / vol_ma.replace(0, 1)
        df['is_high_volume'] = (df['volume_surge'] > 2.0).astype(float)
        
        return df
    
    def _add_vwap_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add VWAP-based features."""
        close = df['close'].values
        volume = df['volume'].values
        
        # VWAP at config period
        vwap = calc_vwap(close, volume, self.config.vwap_period)
        df['vwap'] = vwap
        
        # Price deviation from VWAP (fair value signal)
        df['vwap_deviation'] = (df['close'] - df['vwap']) / df['vwap'].replace(0, 1)
        
        # Z-score of VWAP deviation
        df['vwap_zscore'] = calc_rolling_zscore(
            df['vwap_deviation'].values, 
            self.config.vwap_period
        )
        
        # VWAP trend
        df['vwap_trend'] = df['vwap'].pct_change(10, fill_method=None)
        
        return df
    
    def _add_spread_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add spread and range features."""
        
        # True Range (proxy for spread in crypto)
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        
        tr = np.maximum(
            high - low,
            np.maximum(
                np.abs(high - np.roll(close, 1)),
                np.abs(low - np.roll(close, 1))
            )
        )
        df['true_range'] = tr
        
        # Relative range (% of price)
        df['range_pct'] = (df['high'] - df['low']) / df['close'].replace(0, 1)
        
        # Spread dynamics (normalized by moving average)
        range_ma = df['range_pct'].rolling(self.config.spread_ma_period).mean()
        df['spread_ratio'] = df['range_pct'] / range_ma.replace(0, 1)
        
        # Spread expansion/contraction
        df['spread_change'] = df['range_pct'].pct_change(5)
        
        # Squeeze detection (Bollinger Band squeeze proxy)
        bb_width = df['range_pct'].rolling(20).std()
        bb_width_ma = bb_width.rolling(20).mean()
        df['is_squeeze'] = (bb_width < bb_width_ma * 0.5).astype(float)
        
        return df
    
    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility regime features."""
        returns = df['close'].pct_change().values
        
        for window in self.config.volatility_windows:
            # Realized volatility
            df[f'volatility_{window}'] = calc_rolling_std(returns, window)
            
        # Volatility ratio (short/long)
        if 'volatility_10' in df.columns and 'volatility_60' in df.columns:
            df['vol_ratio'] = df['volatility_10'] / df['volatility_60'].replace(0, 1)
        
        # Volatility regime (z-score of current vol)
        df['vol_zscore'] = calc_rolling_zscore(df['volatility_30'].values, 60)
        
        # High/low volatility regime
        df['high_vol_regime'] = (df['vol_zscore'] > 1.5).astype(float)
        df['low_vol_regime'] = (df['vol_zscore'] < -1.0).astype(float)
        
        # Volatility trend
        df['vol_trend'] = df['volatility_30'].pct_change(10)
        
        return df
    
    def _add_autocorrelation_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add return autocorrelation (mean reversion signal)."""
        returns = df['close'].pct_change().values
        
        # Lag-1 autocorrelation (mean reversion indicator)
        lagged_returns = np.roll(returns, 1)
        lagged_returns[0] = 0
        
        df['return_autocorr_10'] = calc_rolling_corr(returns, lagged_returns, 10)
        df['return_autocorr_30'] = calc_rolling_corr(returns, lagged_returns, 30)
        
        # Strong negative autocorr = mean reverting
        df['mean_reverting'] = (df['return_autocorr_30'] < -0.3).astype(float)
        
        # Strong positive autocorr = trending
        df['trending'] = (df['return_autocorr_30'] > 0.3).astype(float)
        
        return df
    
    def _add_trade_intensity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add trade intensity features."""
        
        # If we have trade count
        if 'trades' in df.columns:
            trades_ma = df['trades'].rolling(30).mean()
            df['trade_intensity'] = df['trades'] / trades_ma.replace(0, 1)
            
            # Trade spike detection
            df['trade_spike'] = (df['trade_intensity'] > 2.0).astype(float)
        
        # Volume per trade (if available)
        if 'trades' in df.columns and df['trades'].sum() > 0:
            df['avg_trade_size'] = df['volume'] / df['trades'].replace(0, 1)
            trade_size_ma = df['avg_trade_size'].rolling(30).mean()
            df['trade_size_ratio'] = df['avg_trade_size'] / trade_size_ma.replace(0, 1)
            
            # Large trade indicator (whale detection)
            df['large_trades'] = (df['trade_size_ratio'] > self.config.whale_threshold).astype(float)
        
        return df
    
    def _add_candle_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add simplified candle pattern features."""
        
        # Candle body size
        df['body_size'] = abs(df['close'] - df['open']) / df['close'].replace(0, 1)
        
        # Upper/lower shadow
        df['upper_shadow'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['close'].replace(0, 1)
        df['lower_shadow'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['close'].replace(0, 1)
        
        # Doji (small body, big shadows)
        df['is_doji'] = ((df['body_size'] < 0.001) & 
                         ((df['upper_shadow'] > 0.002) | (df['lower_shadow'] > 0.002))).astype(float)
        
        # Hammer/shooting star
        df['is_hammer'] = ((df['lower_shadow'] > df['body_size'] * 2) & 
                          (df['is_bullish'] == 0)).astype(float)
        df['is_shooting_star'] = ((df['upper_shadow'] > df['body_size'] * 2) & 
                                  (df['is_bullish'] == 1)).astype(float)
        
        # Engulfing (current body larger than previous)
        prev_body = df['body_size'].shift(1)
        df['is_bullish_engulf'] = ((df['is_bullish'] == 1) & 
                                    (df['body_size'] > prev_body * 1.5) &
                                    (df['is_bullish'].shift(1) == 0)).astype(float)
        df['is_bearish_engulf'] = ((df['is_bullish'] == 0) & 
                                    (df['body_size'] > prev_body * 1.5) &
                                    (df['is_bullish'].shift(1) == 1)).astype(float)
        
        return df
    
    def _add_sentiment_features(
        self, 
        df: pd.DataFrame, 
        sentiment: Any,
    ) -> pd.DataFrame:
        """Add crypto sentiment features from trade flow."""
        
        # These come from the CryptoSentiment object
        df['sentiment_vol_imbalance'] = sentiment.volume_imbalance
        df['sentiment_trade_imbalance'] = sentiment.trade_imbalance
        df['sentiment_whale_signal'] = sentiment.whale_signal
        
        # Combine into overall sentiment score
        df['crypto_sentiment'] = (
            0.4 * sentiment.volume_imbalance +
            0.3 * sentiment.trade_imbalance +
            0.3 * sentiment.whale_signal
        )
        
        return df

    def _add_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add basic technical indicators for strategy filtering."""
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        
        # --- RSI (14) ---
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).fillna(0)
        loss = (-delta.where(delta < 0, 0)).fillna(0)
        
        avg_gain = gain.rolling(window=14, min_periods=1).mean()
        avg_loss = loss.rolling(window=14, min_periods=1).mean()
        
        rs = avg_gain / avg_loss.replace(0, 1)
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # --- ADX (14) ---
        # 1. True Range
        tr = np.maximum(
            high - low,
            np.maximum(
                np.abs(high - np.roll(close, 1)),
                np.abs(low - np.roll(close, 1))
            )
        )
        
        # 2. Directional Movement
        up_move = high - np.roll(high, 1)
        down_move = np.roll(low, 1) - low
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # 3. Smoothed (using simple rolling mean for speed/stability)
        tr14 = pd.Series(tr).rolling(14).sum()
        plus_di14 = pd.Series(plus_dm).rolling(14).sum()
        minus_di14 = pd.Series(minus_dm).rolling(14).sum()
        
        plus_di = 100 * (plus_di14 / tr14.replace(0, 1))
        minus_di = 100 * (minus_di14 / tr14.replace(0, 1))
        
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, 1)
        df['adx'] = dx.rolling(14).mean().fillna(0).values
        
        # --- Bollinger Bands (20, 2) ---
        sma_20 = df['close'].rolling(window=20).mean()
        std_20 = df['close'].rolling(window=20).std()
        
        df['bb_middle'] = sma_20
        df['bb_upper'] = sma_20 + (std_20 * 2)
        df['bb_lower'] = sma_20 - (std_20 * 2)
        
        # --- EMA (12, 26) ---
        df['ema_fast'] = df['close'].ewm(span=12, adjust=False).mean()
        df['ema_slow'] = df['close'].ewm(span=26, adjust=False).mean()
        
        # --- MACD (12, 26, 9) ---
        df['macd'] = df['ema_fast'] - df['ema_slow']
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        
        return df
    
    def _add_composite_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add composite signals combining multiple features."""
        
        # Momentum score (combine multiple momentum signals)
        momentum_cols = [c for c in df.columns if c.startswith('return_')]
        if momentum_cols:
            df['momentum_score'] = sum(np.sign(df[c]) for c in momentum_cols) / len(momentum_cols)
        
        # Volume-confirmed momentum
        if 'momentum_score' in df.columns:
            df['vol_confirmed_momentum'] = df['momentum_score'] * df.get('volume_surge', 1)
        
        # Mean reversion signal (VWAP deviation + autocorr)
        if 'vwap_zscore' in df.columns and 'mean_reverting' in df.columns:
            # Strong reversion signal when price far from VWAP in mean-reverting regime
            df['mean_reversion_signal'] = np.where(
                df['mean_reverting'] == 1,
                -df['vwap_zscore'],  # Fade the move
                0
            )
        
        # Breakout signal (squeeze + momentum)
        if 'is_squeeze' in df.columns and 'momentum_score' in df.columns:
            # Look for squeeze release with momentum
            df['breakout_signal'] = np.where(
                (df['is_squeeze'].shift(1) == 1) & (df['is_squeeze'] == 0),
                df['momentum_score'] * 2,  # Amplify signal on squeeze release
                0
            )
        
        # Overall micro signal (weighted combination)
        signal_components = []
        weights = []
        
        if 'momentum_score' in df.columns:
            signal_components.append(df['momentum_score'])
            weights.append(0.3)
        
        if 'vol_imbalance_10' in df.columns:
            signal_components.append(df['vol_imbalance_10'])
            weights.append(0.25)
        
        if 'vwap_deviation' in df.columns:
            # Normalize VWAP deviation
            vwap_signal = -np.clip(df['vwap_zscore'], -2, 2) / 2
            signal_components.append(vwap_signal)
            weights.append(0.15)
        
        if 'taker_imbalance' in df.columns:
            signal_components.append(df['taker_imbalance'] * 2)
            weights.append(0.3)
        
        if signal_components:
            df['micro_signal'] = sum(s * w for s, w in zip(signal_components, weights))
        
        return df
    
    # -------------------------------------------------------------------------
    # FEATURE SELECTION HELPER
    # -------------------------------------------------------------------------
    
    def get_feature_names(self) -> list:
        """Get list of all feature names this class generates."""
        return [
            # Momentum
            'return_1', 'return_3', 'return_5', 'return_10', 'return_30',
            'logret_1', 'logret_3', 'logret_5', 'logret_10', 'logret_30',
            'momentum_accel', 'momentum_agreement',
            
            # Volume imbalance
            'is_bullish', 'buy_volume_est', 'sell_volume_est',
            'vol_imbalance_5', 'vol_imbalance_10', 'vol_imbalance_30',
            'taker_buy_ratio', 'taker_imbalance',
            'volume_surge', 'is_high_volume',
            
            # VWAP
            'vwap', 'vwap_deviation', 'vwap_zscore', 'vwap_trend',
            
            # Spread
            'true_range', 'range_pct', 'spread_ratio', 'spread_change', 'is_squeeze',
            
            # Volatility
            'volatility_10', 'volatility_30', 'volatility_60',
            'vol_ratio', 'vol_zscore', 'high_vol_regime', 'low_vol_regime', 'vol_trend',
            
            # Autocorrelation
            'return_autocorr_10', 'return_autocorr_30', 'mean_reverting', 'trending',
            
            # Trade intensity
            'trade_intensity', 'trade_spike', 'avg_trade_size', 
            'trade_size_ratio', 'large_trades',
            
            # Candle patterns
            'body_size', 'upper_shadow', 'lower_shadow',
            'is_doji', 'is_hammer', 'is_shooting_star',
            'is_bullish_engulf', 'is_bearish_engulf',
            
            # Sentiment
            'sentiment_vol_imbalance', 'sentiment_trade_imbalance',
            'sentiment_whale_signal', 'crypto_sentiment',
            
            # Composite
            'momentum_score', 'vol_confirmed_momentum',
            'mean_reversion_signal', 'breakout_signal', 'micro_signal',
        ]
    
    def get_most_important_features(self) -> list:
        """Get the most important features for ML training."""
        return [
            # Core momentum
            'return_1', 'return_5', 'return_10', 'momentum_agreement',
            
            # Volume/flow (most predictive for crypto)
            'vol_imbalance_10', 'taker_imbalance', 'volume_surge',
            
            # Fair value
            'vwap_zscore',
            
            # Volatility regime
            'vol_ratio', 'vol_zscore',
            
            # Market structure
            'return_autocorr_30', 'is_squeeze',
            
            # Composite
            'micro_signal',
        ]


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def create_microstructure_features(
    df: pd.DataFrame,
    sentiment: Optional[Any] = None,
) -> pd.DataFrame:
    """
    Quick function to add microstructure features.
    
    Args:
        df: OHLCV DataFrame
        sentiment: Optional CryptoSentiment
        
    Returns:
        DataFrame with features added
    """
    micro = MicrostructureFeatures()
    return micro.add_all_features(df, sentiment)


if __name__ == '__main__':
    # Demo
    import numpy as np
    
    # Generate sample 1-second data
    np.random.seed(42)
    n = 500
    
    close = 50000 + np.cumsum(np.random.randn(n) * 10)
    
    df = pd.DataFrame({
        'open': close + np.random.randn(n) * 5,
        'high': close + np.abs(np.random.randn(n) * 20),
        'low': close - np.abs(np.random.randn(n) * 20),
        'close': close,
        'volume': np.random.uniform(10, 100, n),
        'trades': np.random.randint(10, 500, n),
        'taker_buy_volume': np.random.uniform(5, 50, n),
    })
    
    # Add features
    micro = MicrostructureFeatures()
    df_features = micro.add_all_features(df)
    
    print(f"Original columns: {len(df.columns)}")
    print(f"After features: {len(df_features.columns)}")
    print(f"\nFeature columns added: {len(df_features.columns) - len(df.columns)}")
    print(f"\nMost important features: {micro.get_most_important_features()}")
    print(f"\nSample micro_signal values:\n{df_features['micro_signal'].tail(10)}")
