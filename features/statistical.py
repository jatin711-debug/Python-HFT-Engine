"""
Statistical Features Module.

This module provides statistical feature calculations
for trading strategies, including rolling statistics,
z-scores, correlations, and regime detection.
"""

import warnings
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy import stats
import logging

logger = logging.getLogger(__name__)


class StatisticalFeatures:
    """
    Statistical feature calculator for trading data.
    
    Provides:
    - Rolling statistics (mean, std, skew, kurtosis)
    - Z-score calculations
    - Correlation features
    - Regime detection
    - Distribution features
    - Time-series features
    
    Example:
        >>> features = StatisticalFeatures()
        >>> df_with_stats = features.add_all_features(ohlcv_data)
    """
    
    def __init__(
        self,
        rolling_windows: List[int] = None,
        zscore_windows: List[int] = None,
        correlation_window: int = 20,
    ):
        """
        Initialize statistical features calculator.
        
        Args:
            rolling_windows: Windows for rolling statistics
            zscore_windows: Windows for z-score calculations
            correlation_window: Window for correlation calculations
        """
        self.rolling_windows = rolling_windows or [5, 10, 20, 50]
        self.zscore_windows = zscore_windows or [10, 20, 50]
        self.correlation_window = correlation_window
    
    def add_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add all statistical features to DataFrame.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with all statistical features added
        """
        # Suppress fragmentation warnings during feature generation
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)
            
            df = df.copy()
            df.columns = df.columns.str.lower()
            
            logger.info("Adding rolling statistics...")
            df = self._add_rolling_statistics(df)
            
            logger.info("Adding z-score features...")
            df = self._add_zscore_features(df)
            
            logger.info("Adding distribution features...")
            df = self._add_distribution_features(df)
            
            logger.info("Adding correlation features...")
            df = self._add_correlation_features(df)
            
            logger.info("Adding regime detection...")
            df = self._add_regime_features(df)
            
            logger.info("Adding lag features...")
            df = self._add_lag_features(df)
        
        # Defragment DataFrame after all column additions
        return df.copy()
    
    def _add_rolling_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rolling statistical measures."""
        
        close = df['close']
        returns = close.pct_change()
        
        for window in self.rolling_windows:
            prefix = f'rolling_{window}'
            
            # Basic rolling statistics
            df[f'{prefix}_mean'] = close.rolling(window).mean()
            df[f'{prefix}_std'] = close.rolling(window).std()
            df[f'{prefix}_var'] = close.rolling(window).var()
            df[f'{prefix}_min'] = close.rolling(window).min()
            df[f'{prefix}_max'] = close.rolling(window).max()
            df[f'{prefix}_median'] = close.rolling(window).median()
            
            # Range and position within range
            rolling_min = close.rolling(window).min()
            rolling_max = close.rolling(window).max()
            df[f'{prefix}_range'] = rolling_max - rolling_min
            df[f'{prefix}_range_pct'] = (close - rolling_min) / (rolling_max - rolling_min + 1e-8)
            
            # Rolling returns statistics
            df[f'{prefix}_return_mean'] = returns.rolling(window).mean()
            df[f'{prefix}_return_std'] = returns.rolling(window).std()
            
            # Higher moments
            df[f'{prefix}_skew'] = returns.rolling(window).skew()
            df[f'{prefix}_kurt'] = returns.rolling(window).kurt()
            
            # Coefficient of variation
            df[f'{prefix}_cv'] = df[f'{prefix}_std'] / (df[f'{prefix}_mean'] + 1e-8)
            
            # Quantiles
            df[f'{prefix}_q25'] = close.rolling(window).quantile(0.25)
            df[f'{prefix}_q75'] = close.rolling(window).quantile(0.75)
            df[f'{prefix}_iqr'] = df[f'{prefix}_q75'] - df[f'{prefix}_q25']
        
        return df
    
    def _add_zscore_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add z-score features for mean reversion signals."""
        
        close = df['close']
        
        for window in self.zscore_windows:
            rolling_mean = close.rolling(window).mean()
            rolling_std = close.rolling(window).std()
            
            # Price z-score
            df[f'zscore_{window}'] = (close - rolling_mean) / (rolling_std + 1e-8)
            
            # Extreme z-score signals
            df[f'zscore_{window}_extreme_high'] = (df[f'zscore_{window}'] > 2).astype(int)
            df[f'zscore_{window}_extreme_low'] = (df[f'zscore_{window}'] < -2).astype(int)
        
        # Volume z-score
        volume = df['volume']
        for window in self.zscore_windows:
            vol_mean = volume.rolling(window).mean()
            vol_std = volume.rolling(window).std()
            df[f'volume_zscore_{window}'] = (volume - vol_mean) / (vol_std + 1e-8)
        
        return df
    
    def _add_distribution_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add features based on return distribution."""
        
        returns = df['close'].pct_change()
        
        # Rolling percentile rank
        df['return_percentile_20'] = returns.rolling(20).apply(
            lambda x: stats.percentileofscore(x, x.iloc[-1]) / 100
        )
        
        # Rolling Sharpe-like ratio (without risk-free rate)
        for window in [20, 50]:
            mean_return = returns.rolling(window).mean()
            std_return = returns.rolling(window).std()
            df[f'sharpe_ratio_{window}'] = np.sqrt(252) * mean_return / (std_return + 1e-8)
        
        # Rolling Sortino-like ratio (downside risk)
        for window in [20, 50]:
            mean_return = returns.rolling(window).mean()
            negative_returns = returns.copy()
            negative_returns[negative_returns > 0] = 0
            downside_std = negative_returns.rolling(window).std()
            df[f'sortino_ratio_{window}'] = np.sqrt(252) * mean_return / (downside_std + 1e-8)
        
        # Maximum drawdown in rolling window
        df['rolling_max_drawdown_20'] = self._rolling_max_drawdown(df['close'], 20)
        df['rolling_max_drawdown_50'] = self._rolling_max_drawdown(df['close'], 50)
        
        # Calmar-like ratio
        df['calmar_ratio_50'] = (
            returns.rolling(50).mean() * 252 / 
            (abs(df['rolling_max_drawdown_50']) + 1e-8)
        )
        
        return df
    
    def _add_correlation_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add correlation-based features."""
        
        close = df['close']
        volume = df['volume']
        returns = close.pct_change()
        
        # Price-volume correlation
        df['price_volume_corr'] = returns.rolling(self.correlation_window).corr(
            volume.pct_change()
        )
        
        # High-Low correlation with volume
        high_low_range = df['high'] - df['low']
        df['range_volume_corr'] = high_low_range.rolling(self.correlation_window).corr(
            volume
        )
        
        # Autocorrelation of returns
        for lag in [1, 5, 10]:
            df[f'return_autocorr_lag{lag}'] = returns.rolling(20).apply(
                lambda x: x.autocorr(lag=lag) if len(x.dropna()) > lag else np.nan
            )
        
        # Beta-like measure (correlation * volatility ratio)
        if 'close' in df.columns:
            market_vol = returns.rolling(20).std()
            df['relative_volatility'] = returns.rolling(20).std() / (market_vol + 1e-8)
        
        return df
    
    def _add_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market regime detection features."""
        
        close = df['close']
        returns = close.pct_change()
        
        # Volatility regime based on rolling std
        vol_20 = returns.rolling(20).std()
        vol_50 = returns.rolling(50).std()
        vol_100 = returns.rolling(100).std()
        
        # Volatility relative to longer-term average
        df['volatility_ratio_20_50'] = vol_20 / (vol_50 + 1e-8)
        df['volatility_ratio_20_100'] = vol_20 / (vol_100 + 1e-8)
        
        # Volatility regime classification
        vol_percentile = vol_20.rolling(100).apply(
            lambda x: stats.percentileofscore(x.dropna(), x.iloc[-1]) if len(x.dropna()) > 0 else 50
        )
        df['volatility_regime'] = pd.cut(
            vol_percentile,
            bins=[0, 25, 50, 75, 100],
            labels=[0, 1, 2, 3],
            include_lowest=True
        ).astype(float)
        
        # Trend regime based on moving average position
        sma_20 = close.rolling(20).mean()
        sma_50 = close.rolling(50).mean()
        sma_100 = close.rolling(100).mean()
        
        df['trend_regime'] = (
            (close > sma_20).astype(int) +
            (close > sma_50).astype(int) +
            (close > sma_100).astype(int) +
            (sma_20 > sma_50).astype(int) +
            (sma_50 > sma_100).astype(int)
        )
        
        # Mean reversion regime (high z-score = potential mean reversion)
        zscore = (close - sma_50) / (returns.rolling(50).std() * np.sqrt(50) + 1e-8)
        df['mean_reversion_score'] = abs(zscore)
        
        # Momentum regime (consistent direction)
        df['momentum_regime'] = returns.rolling(20).apply(
            lambda x: (x > 0).sum() / len(x)
        )
        
        # Range-bound vs trending (ADR ratio)
        adr = (df['high'] - df['low']).rolling(20).mean()
        tr = pd.concat([
            df['high'] - df['low'],
            abs(df['high'] - close.shift()),
            abs(df['low'] - close.shift())
        ], axis=1).max(axis=1).rolling(20).mean()
        
        df['range_trend_ratio'] = adr / (tr + 1e-8)
        
        return df
    
    def _add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add lagged features for time-series modeling."""
        
        close = df['close']
        returns = close.pct_change()
        volume = df['volume']
        
        # Lagged returns
        for lag in [1, 2, 3, 5, 10]:
            df[f'return_lag_{lag}'] = returns.shift(lag)
        
        # Lagged volume ratio
        for lag in [1, 2, 3]:
            df[f'volume_ratio_lag_{lag}'] = (volume / volume.rolling(20).mean()).shift(lag)
        
        # Lagged high-low range
        range_pct = (df['high'] - df['low']) / close
        for lag in [1, 2, 3]:
            df[f'range_lag_{lag}'] = range_pct.shift(lag)
        
        # Cumulative return features
        df['cum_return_3d'] = close.pct_change(3)
        df['cum_return_5d'] = close.pct_change(5)
        df['cum_return_10d'] = close.pct_change(10)
        
        # Return momentum (acceleration)
        df['return_acceleration'] = returns.diff()
        
        # Consecutive up/down days
        up_day = (returns > 0).astype(int)
        down_day = (returns < 0).astype(int)
        
        df['consecutive_up'] = up_day.groupby(
            (up_day != up_day.shift()).cumsum()
        ).cumsum() * up_day
        
        df['consecutive_down'] = down_day.groupby(
            (down_day != down_day.shift()).cumsum()
        ).cumsum() * down_day
        
        return df
    
    def _rolling_max_drawdown(
        self, prices: pd.Series, window: int
    ) -> pd.Series:
        """Calculate rolling maximum drawdown."""
        
        def max_dd(x):
            cumulative = (1 + x.pct_change()).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            return drawdown.min()
        
        return prices.rolling(window).apply(max_dd, raw=False)
    
    def get_feature_names(self) -> List[str]:
        """Get list of all feature names that will be generated."""
        features = []
        
        # Rolling statistics
        for window in self.rolling_windows:
            prefix = f'rolling_{window}'
            features.extend([
                f'{prefix}_mean', f'{prefix}_std', f'{prefix}_var',
                f'{prefix}_min', f'{prefix}_max', f'{prefix}_median',
                f'{prefix}_range', f'{prefix}_range_pct',
                f'{prefix}_return_mean', f'{prefix}_return_std',
                f'{prefix}_skew', f'{prefix}_kurt', f'{prefix}_cv',
                f'{prefix}_q25', f'{prefix}_q75', f'{prefix}_iqr'
            ])
        
        # Z-scores
        for window in self.zscore_windows:
            features.extend([
                f'zscore_{window}',
                f'zscore_{window}_extreme_high',
                f'zscore_{window}_extreme_low',
                f'volume_zscore_{window}'
            ])
        
        # Distribution features
        features.extend([
            'return_percentile_20',
            'sharpe_ratio_20', 'sharpe_ratio_50',
            'sortino_ratio_20', 'sortino_ratio_50',
            'rolling_max_drawdown_20', 'rolling_max_drawdown_50',
            'calmar_ratio_50'
        ])
        
        # Correlation features
        features.extend([
            'price_volume_corr', 'range_volume_corr',
            'return_autocorr_lag1', 'return_autocorr_lag5', 'return_autocorr_lag10',
            'relative_volatility'
        ])
        
        # Regime features
        features.extend([
            'volatility_ratio_20_50', 'volatility_ratio_20_100',
            'volatility_regime', 'trend_regime',
            'mean_reversion_score', 'momentum_regime', 'range_trend_ratio'
        ])
        
        # Lag features
        for lag in [1, 2, 3, 5, 10]:
            features.append(f'return_lag_{lag}')
        for lag in [1, 2, 3]:
            features.extend([f'volume_ratio_lag_{lag}', f'range_lag_{lag}'])
        
        features.extend([
            'cum_return_3d', 'cum_return_5d', 'cum_return_10d',
            'return_acceleration', 'consecutive_up', 'consecutive_down'
        ])
        
        return features
