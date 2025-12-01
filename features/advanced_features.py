"""
Advanced Features Module - Institutional Grade.

Features used by top quant funds that go beyond basic technical analysis:
- Cross-asset correlations
- Options-derived signals
- Factor exposures
- Market regime detection
- Macro indicators
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy import stats
from scipy.signal import argrelextrema
import logging

logger = logging.getLogger(__name__)


class AdvancedFeatures:
    """
    Institutional-grade feature engineering.
    
    These are the types of features that differentiate
    sophisticated quant funds from retail traders.
    """
    
    def __init__(self):
        self.vix_data = None
        self.treasury_data = None
        self.sector_data = None
    
    def add_all_advanced_features(
        self,
        df: pd.DataFrame,
        market_data: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> pd.DataFrame:
        """Add all advanced features to the dataframe."""
        
        df = df.copy()
        
        logger.info("Adding market regime features...")
        df = self._add_regime_features(df)
        
        logger.info("Adding fractal/chaos features...")
        df = self._add_fractal_features(df)
        
        logger.info("Adding order flow proxy features...")
        df = self._add_order_flow_features(df)
        
        logger.info("Adding support/resistance features...")
        df = self._add_support_resistance_features(df)
        
        logger.info("Adding mean reversion features...")
        df = self._add_mean_reversion_features(df)
        
        logger.info("Adding momentum factor features...")
        df = self._add_momentum_factors(df)
        
        logger.info("Adding volatility regime features...")
        df = self._add_volatility_regime_features(df)
        
        if market_data:
            logger.info("Adding cross-asset features...")
            df = self._add_cross_asset_features(df, market_data)
        
        return df
    
    # =========================================================================
    # MARKET REGIME DETECTION
    # =========================================================================
    
    def _add_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect market regimes using Hidden Markov Model proxy.
        
        Regimes:
        - Bull (trending up)
        - Bear (trending down)
        - Sideways (range-bound)
        - High volatility
        - Low volatility
        """
        close = df['close']
        returns = close.pct_change()
        
        # Rolling regime detection based on returns distribution
        for window in [20, 50]:
            # Mean return regime
            rolling_mean = returns.rolling(window).mean()
            rolling_std = returns.rolling(window).std()
            
            # Regime score: positive = bullish, negative = bearish
            df[f'regime_score_{window}'] = rolling_mean / (rolling_std + 1e-8)
            
            # Trend strength using ADX-like calculation
            high_low_range = df['high'] - df['low']
            avg_range = high_low_range.rolling(window).mean()
            df[f'trend_strength_{window}'] = avg_range / close
            
            # Regime classification
            df[f'regime_bull_{window}'] = (df[f'regime_score_{window}'] > 0.5).astype(int)
            df[f'regime_bear_{window}'] = (df[f'regime_score_{window}'] < -0.5).astype(int)
            df[f'regime_sideways_{window}'] = (
                (df[f'regime_score_{window}'] >= -0.5) & 
                (df[f'regime_score_{window}'] <= 0.5)
            ).astype(int)
        
        # Volatility regime using rolling percentile
        vol_20 = returns.rolling(20).std()
        vol_percentile = vol_20.rolling(252).apply(
            lambda x: stats.percentileofscore(x.dropna(), x.iloc[-1]) if len(x.dropna()) > 0 else 50
        )
        df['vol_regime_percentile'] = vol_percentile
        df['high_vol_regime'] = (vol_percentile > 80).astype(int)
        df['low_vol_regime'] = (vol_percentile < 20).astype(int)
        
        # Regime change detection
        df['regime_change'] = (
            df['regime_bull_20'].diff().abs() + 
            df['regime_bear_20'].diff().abs()
        ).fillna(0)
        
        return df
    
    # =========================================================================
    # FRACTAL / CHAOS THEORY FEATURES
    # =========================================================================
    
    def _add_fractal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add features based on fractal market hypothesis.
        
        These capture the self-similar, chaotic nature of markets
        that traditional indicators miss.
        """
        close = df['close']
        
        # Hurst Exponent (approximation) - measures trend persistence
        # H > 0.5: trending, H < 0.5: mean reverting, H = 0.5: random walk
        for window in [20, 50, 100]:
            df[f'hurst_proxy_{window}'] = self._calculate_hurst_proxy(close, window)
        
        # Fractal Dimension (box-counting approximation)
        df['fractal_dimension'] = self._calculate_fractal_dimension(close)
        
        # Lyapunov Exponent proxy (chaos measure)
        # Positive = chaotic, Negative = stable
        returns = close.pct_change()
        df['lyapunov_proxy'] = returns.rolling(50).apply(
            lambda x: np.mean(np.log(np.abs(np.diff(x)) + 1e-10)) if len(x) > 1 else 0
        )
        
        # Entropy (information content)
        df['price_entropy'] = self._calculate_entropy(close, window=20)
        
        return df
    
    def _calculate_hurst_proxy(self, series: pd.Series, window: int) -> pd.Series:
        """Calculate Hurst exponent using R/S method (simplified)."""
        def hurst(x):
            if len(x) < 20:
                return 0.5
            
            n = len(x)
            mean = np.mean(x)
            std = np.std(x)
            
            if std == 0:
                return 0.5
            
            # Cumulative deviation from mean
            cumdev = np.cumsum(x - mean)
            
            # Range
            R = np.max(cumdev) - np.min(cumdev)
            
            # R/S statistic
            rs = R / std if std > 0 else 0
            
            # Hurst approximation
            if rs > 0 and n > 0:
                H = np.log(rs) / np.log(n)
                return np.clip(H, 0, 1)
            return 0.5
        
        return series.rolling(window).apply(hurst, raw=True)
    
    def _calculate_fractal_dimension(self, series: pd.Series, window: int = 30) -> pd.Series:
        """Calculate fractal dimension using box-counting approximation."""
        def box_dim(x):
            if len(x) < 10:
                return 1.5
            
            n = len(x)
            # Normalize to [0,1]
            x_norm = (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-10)
            
            # Count boxes at different scales
            scales = [2, 4, 8, 16]
            counts = []
            
            for scale in scales:
                if scale >= n:
                    continue
                boxes = set()
                for i in range(0, n, scale):
                    box_x = i // scale
                    box_y = int(x_norm[i] * scale)
                    boxes.add((box_x, box_y))
                counts.append((np.log(scale), np.log(len(boxes) + 1)))
            
            if len(counts) < 2:
                return 1.5
            
            # Linear regression to get dimension
            scales_log = [c[0] for c in counts]
            counts_log = [c[1] for c in counts]
            
            slope, _ = np.polyfit(scales_log, counts_log, 1)
            return np.clip(-slope, 1, 2)
        
        return series.rolling(window).apply(box_dim, raw=True)
    
    def _calculate_entropy(self, series: pd.Series, window: int = 20) -> pd.Series:
        """Calculate Shannon entropy of price changes."""
        returns = series.pct_change()
        
        def entropy(x):
            if len(x) < 5:
                return 0
            
            # Bin returns into categories
            bins = 10
            hist, _ = np.histogram(x, bins=bins, density=True)
            hist = hist[hist > 0]  # Remove zeros
            
            if len(hist) == 0:
                return 0
            
            # Shannon entropy
            return -np.sum(hist * np.log2(hist + 1e-10))
        
        return returns.rolling(window).apply(entropy, raw=True)
    
    # =========================================================================
    # ORDER FLOW PROXY FEATURES
    # =========================================================================
    
    def _add_order_flow_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Proxy for order flow analysis without Level 2 data.
        
        These approximate what institutional traders see in order books.
        """
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']
        
        # Volume-weighted price movement (buying vs selling pressure)
        price_change = close.diff()
        df['buying_pressure'] = np.where(price_change > 0, volume, 0)
        df['selling_pressure'] = np.where(price_change < 0, volume, 0)
        
        # Cumulative volume delta
        df['volume_delta'] = df['buying_pressure'] - df['selling_pressure']
        df['cum_volume_delta'] = df['volume_delta'].cumsum()
        
        # Volume delta momentum
        for window in [5, 10, 20]:
            df[f'volume_delta_ma_{window}'] = df['volume_delta'].rolling(window).mean()
            df[f'volume_delta_divergence_{window}'] = (
                df['volume_delta'] - df[f'volume_delta_ma_{window}']
            )
        
        # Price-Volume Trend (PVT)
        df['pvt'] = ((close - close.shift(1)) / close.shift(1) * volume).cumsum()
        df['pvt_signal'] = df['pvt'].rolling(20).mean()
        
        # Ease of Movement
        distance_moved = ((high + low) / 2) - ((high.shift(1) + low.shift(1)) / 2)
        box_ratio = (volume / 1e6) / (high - low + 1e-10)
        df['ease_of_movement'] = distance_moved / box_ratio
        df['eom_ma'] = df['ease_of_movement'].rolling(14).mean()
        
        # Volume Price Confirmation Indicator
        price_trend = close.rolling(20).mean().diff()
        volume_trend = volume.rolling(20).mean().diff()
        df['vpci'] = np.sign(price_trend) * np.sign(volume_trend)
        
        # Intraday intensity (proxy)
        df['intraday_intensity'] = (
            (2 * close - high - low) / (high - low + 1e-10) * volume
        )
        df['ii_cumsum'] = df['intraday_intensity'].rolling(20).sum()
        
        return df
    
    # =========================================================================
    # SUPPORT / RESISTANCE FEATURES
    # =========================================================================
    
    def _add_support_resistance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Identify support and resistance levels.
        
        These are key price levels where supply/demand imbalances occur.
        """
        close = df['close']
        high = df['high']
        low = df['low']
        
        # Find local minima and maxima
        order = 5  # Number of points to compare
        
        # Local maxima (resistance)
        local_max_idx = argrelextrema(high.values, np.greater, order=order)[0]
        df['is_resistance'] = 0
        if len(local_max_idx) > 0:
            df.iloc[local_max_idx, df.columns.get_loc('is_resistance')] = 1
        
        # Local minima (support)
        local_min_idx = argrelextrema(low.values, np.less, order=order)[0]
        df['is_support'] = 0
        if len(local_min_idx) > 0:
            df.iloc[local_min_idx, df.columns.get_loc('is_support')] = 1
        
        # Distance to nearest support/resistance
        df['dist_to_resistance'] = np.nan
        df['dist_to_support'] = np.nan
        
        last_resistance = None
        last_support = None
        
        for i in range(len(df)):
            if df['is_resistance'].iloc[i] == 1:
                last_resistance = high.iloc[i]
            if df['is_support'].iloc[i] == 1:
                last_support = low.iloc[i]
            
            if last_resistance is not None:
                df.iloc[i, df.columns.get_loc('dist_to_resistance')] = (
                    (last_resistance - close.iloc[i]) / close.iloc[i]
                )
            if last_support is not None:
                df.iloc[i, df.columns.get_loc('dist_to_support')] = (
                    (close.iloc[i] - last_support) / close.iloc[i]
                )
        
        # Pivot points (classic floor trader pivots)
        df['pivot'] = (high.shift(1) + low.shift(1) + close.shift(1)) / 3
        df['r1'] = 2 * df['pivot'] - low.shift(1)
        df['s1'] = 2 * df['pivot'] - high.shift(1)
        df['r2'] = df['pivot'] + (high.shift(1) - low.shift(1))
        df['s2'] = df['pivot'] - (high.shift(1) - low.shift(1))
        
        # Distance to pivot levels (normalized)
        df['dist_to_pivot'] = (close - df['pivot']) / close
        df['dist_to_r1'] = (df['r1'] - close) / close
        df['dist_to_s1'] = (close - df['s1']) / close
        
        # Price position relative to range
        rolling_high = high.rolling(20).max()
        rolling_low = low.rolling(20).min()
        df['price_position'] = (close - rolling_low) / (rolling_high - rolling_low + 1e-10)
        
        return df
    
    # =========================================================================
    # MEAN REVERSION FEATURES
    # =========================================================================
    
    def _add_mean_reversion_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Features for mean reversion strategies.
        
        Detects overbought/oversold conditions and reversion probability.
        """
        close = df['close']
        returns = close.pct_change()
        
        # Z-score at multiple timescales
        for window in [10, 20, 50, 100]:
            rolling_mean = close.rolling(window).mean()
            rolling_std = close.rolling(window).std()
            df[f'zscore_{window}'] = (close - rolling_mean) / (rolling_std + 1e-10)
            
            # Mean reversion probability (based on historical z-score behavior)
            df[f'mr_probability_{window}'] = 1 - stats.norm.cdf(abs(df[f'zscore_{window}']))
        
        # Ornstein-Uhlenbeck parameters (mean reversion speed)
        # Simplified estimation
        for window in [50, 100]:
            price_diff = close.diff()
            price_lag = close.shift(1)
            
            def estimate_ou_theta(y, x):
                if len(y) < 10 or np.std(x) == 0:
                    return 0
                slope, _ = np.polyfit(x, y, 1)
                return -slope  # Mean reversion speed
            
            df[f'ou_theta_{window}'] = pd.Series(
                [estimate_ou_theta(
                    price_diff.iloc[max(0, i-window):i].values,
                    price_lag.iloc[max(0, i-window):i].values
                ) for i in range(len(df))],
                index=df.index
            )
        
        # RSI divergence (price makes new high but RSI doesn't)
        # Calculate RSI
        delta = close.diff()
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).rolling(14).mean()
        avg_loss = pd.Series(loss).rolling(14).mean()
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        df['rsi_14'] = rsi.values
        
        # Detect divergence
        price_high = close.rolling(20).max() == close
        rsi_high = pd.Series(rsi).rolling(20).max() == rsi
        df['bearish_divergence'] = (price_high & ~rsi_high).astype(int)
        
        price_low = close.rolling(20).min() == close
        rsi_low = pd.Series(rsi).rolling(20).min() == rsi
        df['bullish_divergence'] = (price_low & ~rsi_low).astype(int)
        
        # Consecutive up/down days (extreme values suggest reversion)
        up_days = (returns > 0).astype(int)
        down_days = (returns < 0).astype(int)
        
        df['consecutive_up'] = up_days.groupby(
            (up_days != up_days.shift()).cumsum()
        ).cumsum() * up_days
        
        df['consecutive_down'] = down_days.groupby(
            (down_days != down_days.shift()).cumsum()
        ).cumsum() * down_days
        
        # Extreme move indicator (suggests reversion)
        df['extreme_move'] = (abs(returns) > returns.rolling(50).std() * 2).astype(int)
        
        return df
    
    # =========================================================================
    # MOMENTUM FACTOR FEATURES
    # =========================================================================
    
    def _add_momentum_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fama-French style momentum factors.
        
        These are the systematic risk factors that explain returns.
        """
        close = df['close']
        returns = close.pct_change()
        
        # Classic momentum (12-1 month)
        # Skip most recent month (short-term reversal)
        df['momentum_12_1'] = close.shift(21) / close.shift(252) - 1
        
        # Intermediate momentum (6 months)
        df['momentum_6m'] = close / close.shift(126) - 1
        
        # Short-term momentum (1 month)
        df['momentum_1m'] = close / close.shift(21) - 1
        
        # Time-series momentum (trend following)
        for window in [20, 50, 100, 200]:
            df[f'ts_momentum_{window}'] = close / close.rolling(window).mean() - 1
        
        # Momentum quality (consistency of returns)
        for window in [20, 50]:
            positive_days = (returns > 0).rolling(window).sum()
            df[f'momentum_quality_{window}'] = positive_days / window
        
        # Acceleration (change in momentum)
        df['momentum_acceleration'] = df['momentum_1m'].diff()
        
        # 52-week high/low proximity
        high_52w = close.rolling(252).max()
        low_52w = close.rolling(252).min()
        df['pct_from_52w_high'] = (close - high_52w) / high_52w
        df['pct_from_52w_low'] = (close - low_52w) / low_52w
        
        # Momentum breadth (how consistent is the trend)
        sma_20 = close.rolling(20).mean()
        sma_50 = close.rolling(50).mean()
        sma_100 = close.rolling(100).mean()
        sma_200 = close.rolling(200).mean()
        
        df['momentum_breadth'] = (
            (close > sma_20).astype(int) +
            (close > sma_50).astype(int) +
            (close > sma_100).astype(int) +
            (close > sma_200).astype(int)
        ) / 4
        
        return df
    
    # =========================================================================
    # VOLATILITY REGIME FEATURES
    # =========================================================================
    
    def _add_volatility_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Advanced volatility features for regime detection.
        """
        close = df['close']
        high = df['high']
        low = df['low']
        returns = close.pct_change()
        
        # Parkinson volatility (more efficient than close-to-close)
        parkinson_vol = np.sqrt(
            (1 / (4 * np.log(2))) * 
            ((np.log(high / low)) ** 2).rolling(20).mean()
        )
        df['parkinson_volatility'] = parkinson_vol
        
        # Garman-Klass volatility
        open_price = df['open']
        log_hl = np.log(high / low) ** 2
        log_co = np.log(close / open_price) ** 2
        gk_vol = np.sqrt(
            0.5 * log_hl.rolling(20).mean() - 
            (2 * np.log(2) - 1) * log_co.rolling(20).mean()
        )
        df['gk_volatility'] = gk_vol
        
        # Volatility of volatility (VVIX proxy)
        realized_vol = returns.rolling(20).std()
        df['vol_of_vol'] = realized_vol.rolling(20).std()
        
        # Volatility term structure (short vs long vol)
        vol_5 = returns.rolling(5).std()
        vol_20 = returns.rolling(20).std()
        vol_60 = returns.rolling(60).std()
        
        df['vol_term_5_20'] = vol_5 / (vol_20 + 1e-10)
        df['vol_term_20_60'] = vol_20 / (vol_60 + 1e-10)
        
        # Volatility skew (asymmetry in up vs down moves)
        up_vol = returns[returns > 0].rolling(20).std()
        down_vol = returns[returns < 0].rolling(20).std()
        df['vol_skew'] = (up_vol - down_vol).reindex(df.index).ffill()
        
        # GARCH proxy (volatility clustering)
        df['vol_persistence'] = realized_vol.rolling(5).mean() / (realized_vol.rolling(50).mean() + 1e-10)
        
        # Jump detection (sudden volatility spikes)
        vol_zscore = (realized_vol - realized_vol.rolling(50).mean()) / (realized_vol.rolling(50).std() + 1e-10)
        df['vol_jump'] = (vol_zscore > 2).astype(int)
        
        return df
    
    # =========================================================================
    # CROSS-ASSET FEATURES
    # =========================================================================
    
    def _add_cross_asset_features(
        self,
        df: pd.DataFrame,
        market_data: Dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        """
        Cross-asset correlation and relative strength features.
        
        Args:
            df: Main price dataframe
            market_data: Dictionary with keys like 'spy', 'vix', 'tlt', etc.
        """
        close = df['close']
        
        # Beta to market (SPY)
        if 'spy' in market_data:
            spy = market_data['spy']['close'].reindex(df.index).ffill()
            stock_returns = close.pct_change()
            market_returns = spy.pct_change()
            
            # Rolling beta
            for window in [20, 60]:
                covariance = stock_returns.rolling(window).cov(market_returns)
                market_var = market_returns.rolling(window).var()
                df[f'beta_{window}'] = covariance / (market_var + 1e-10)
            
            # Alpha (Jensen's alpha)
            df['alpha_60'] = stock_returns.rolling(60).mean() - df['beta_60'] * market_returns.rolling(60).mean()
            
            # Relative strength
            df['relative_strength_spy'] = close / spy
            df['rs_momentum'] = df['relative_strength_spy'].pct_change(20)
        
        # VIX correlation (fear gauge)
        if 'vix' in market_data:
            vix = market_data['vix']['close'].reindex(df.index).ffill()
            stock_returns = close.pct_change()
            vix_returns = vix.pct_change()
            
            df['vix_correlation'] = stock_returns.rolling(20).corr(vix_returns)
            df['vix_level'] = vix
            df['vix_percentile'] = vix.rolling(252).apply(
                lambda x: stats.percentileofscore(x.dropna(), x.iloc[-1]) if len(x.dropna()) > 0 else 50
            )
        
        # Interest rate sensitivity (TLT - Treasury bonds)
        if 'tlt' in market_data:
            tlt = market_data['tlt']['close'].reindex(df.index).ffill()
            stock_returns = close.pct_change()
            bond_returns = tlt.pct_change()
            
            df['bond_correlation'] = stock_returns.rolling(20).corr(bond_returns)
            
            # Risk-on/risk-off indicator
            df['risk_on_off'] = (stock_returns.rolling(10).mean() - bond_returns.rolling(10).mean())
        
        # Sector rotation signals
        if 'xlf' in market_data and 'xlu' in market_data:
            # Financials vs Utilities (risk appetite indicator)
            xlf = market_data['xlf']['close'].reindex(df.index).ffill()
            xlu = market_data['xlu']['close'].reindex(df.index).ffill()
            
            df['sector_rotation'] = (xlf / xlu).pct_change(20)
        
        return df


class OptionsFeatures:
    """
    Options-derived signals (requires options data).
    
    These provide forward-looking information that price data alone cannot.
    """
    
    def __init__(self):
        pass
    
    def add_options_features(
        self,
        df: pd.DataFrame,
        options_data: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Add options-derived features.
        
        If options_data not available, use proxy calculations.
        """
        close = df['close']
        returns = close.pct_change()
        
        # Implied volatility proxy (using realized vol percentile)
        realized_vol = returns.rolling(20).std() * np.sqrt(252)
        df['iv_proxy'] = realized_vol.rolling(252).apply(
            lambda x: np.percentile(x.dropna(), 75) if len(x.dropna()) > 0 else x.iloc[-1]
        )
        
        # IV/RV ratio (volatility risk premium)
        df['vol_risk_premium'] = df['iv_proxy'] / (realized_vol + 1e-10)
        
        # Put/Call ratio proxy (based on price action)
        # When price drops on high volume, put activity is likely higher
        price_down = (returns < 0).astype(int)
        volume_above_avg = (df['volume'] > df['volume'].rolling(20).mean()).astype(int)
        df['put_call_proxy'] = (price_down * volume_above_avg).rolling(10).mean()
        
        # Skew proxy (tail risk indicator)
        # Large negative returns relative to positive ones
        neg_returns = np.where(returns < 0, returns, 0)
        pos_returns = np.where(returns > 0, returns, 0)
        
        df['skew_proxy'] = (
            pd.Series(neg_returns).rolling(20).std() / 
            (pd.Series(pos_returns).rolling(20).std() + 1e-10)
        ).values
        
        # Max pain proxy (gravitational price level)
        # Price tends toward highest open interest strike
        # Proxy: volume-weighted average price level
        df['max_pain_proxy'] = (
            (df['high'] + df['low'] + close) / 3 * df['volume']
        ).rolling(20).sum() / df['volume'].rolling(20).sum()
        
        df['dist_to_max_pain'] = (close - df['max_pain_proxy']) / close
        
        return df


class MacroFeatures:
    """
    Macroeconomic regime features.
    
    These capture the broader economic environment that affects all stocks.
    """
    
    def __init__(self):
        pass
    
    def add_macro_features(
        self,
        df: pd.DataFrame,
        macro_data: Optional[Dict[str, pd.Series]] = None,
    ) -> pd.DataFrame:
        """
        Add macro-derived features.
        
        If macro_data not available, use market-based proxies.
        """
        close = df['close']
        
        # Yield curve proxy (using price momentum as economic indicator)
        # Strong momentum = expanding economy, weak = contracting
        df['economic_proxy'] = close.pct_change(60)
        
        # Recession probability proxy
        # Based on momentum turning negative after extended positive
        momentum_60 = close.pct_change(60)
        momentum_negative = (momentum_60 < 0).astype(int)
        df['recession_proxy'] = momentum_negative.rolling(3).mean()
        
        # Inflation proxy (using price volatility)
        # High volatility often accompanies inflationary periods
        vol_60 = close.pct_change().rolling(60).std()
        df['inflation_proxy'] = vol_60.rolling(252).apply(
            lambda x: stats.percentileofscore(x.dropna(), x.iloc[-1]) if len(x.dropna()) > 0 else 50
        )
        
        # Credit stress proxy (using drawdown depth)
        cummax = close.cummax()
        drawdown = (close - cummax) / cummax
        df['credit_stress_proxy'] = -drawdown.rolling(20).min()
        
        # Liquidity proxy (using volume relative to volatility)
        volume_ma = df['volume'].rolling(20).mean()
        volatility = close.pct_change().rolling(20).std()
        df['liquidity_proxy'] = volume_ma / (volatility * close + 1e-10)
        
        return df
