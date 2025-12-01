"""
Technical Indicators Module using TA-Lib.

This module provides a comprehensive set of technical indicators
used for feature engineering in trading strategies.

Includes 150+ indicators across categories:
- Trend indicators
- Momentum indicators
- Volatility indicators
- Volume indicators
- Pattern recognition
"""

import warnings
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

# Try to import TA-Lib
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    logging.warning(
        "TA-Lib not available. Install with: pip install TA-Lib\n"
        "Note: TA-Lib requires the underlying C library to be installed first."
    )

logger = logging.getLogger(__name__)


class TechnicalFeatures:
    """
    Comprehensive technical indicator calculator using TA-Lib.
    
    Features 150+ indicators including:
    - Moving averages (SMA, EMA, WMA, DEMA, TEMA, KAMA, etc.)
    - Momentum (RSI, MACD, Stochastic, CCI, MOM, ROC, etc.)
    - Volatility (ATR, Bollinger Bands, Keltner Channel)
    - Volume (OBV, MFI, AD, ADOSC)
    - Trend (ADX, Aroon, Parabolic SAR)
    - Pattern recognition (61 candlestick patterns)
    
    Example:
        >>> features = TechnicalFeatures()
        >>> df_with_indicators = features.add_all_indicators(ohlcv_data)
    """
    
    def __init__(
        self,
        sma_periods: List[int] = None,
        ema_periods: List[int] = None,
        rsi_period: int = 14,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        bb_period: int = 20,
        bb_std: float = 2.0,
        atr_period: int = 14,
        adx_period: int = 14,
    ):
        """
        Initialize technical features calculator.
        
        Args:
            sma_periods: Periods for Simple Moving Averages
            ema_periods: Periods for Exponential Moving Averages
            rsi_period: Period for RSI calculation
            macd_fast: Fast period for MACD
            macd_slow: Slow period for MACD
            macd_signal: Signal period for MACD
            bb_period: Period for Bollinger Bands
            bb_std: Standard deviations for Bollinger Bands
            atr_period: Period for ATR calculation
            adx_period: Period for ADX calculation
        """
        self.sma_periods = sma_periods or [5, 10, 20, 50, 100, 200]
        self.ema_periods = ema_periods or [5, 10, 20, 50, 100]
        self.rsi_period = rsi_period
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.atr_period = atr_period
        self.adx_period = adx_period
        
        if not TALIB_AVAILABLE:
            logger.warning("TA-Lib not available, using fallback implementations")
    
    def add_all_indicators(
        self,
        df: pd.DataFrame,
        include_patterns: bool = True,
    ) -> pd.DataFrame:
        """
        Add all technical indicators to DataFrame.
        
        Args:
            df: DataFrame with OHLCV data
            include_patterns: Whether to include candlestick patterns
            
        Returns:
            DataFrame with all indicators added
        """
        # Suppress fragmentation warnings during feature generation
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)
            
            df = df.copy()
            
            # Ensure column names are lowercase
            df.columns = df.columns.str.lower()
            
            # Extract OHLCV data
            open_price = df['open'].values
            high = df['high'].values
            low = df['low'].values
            close = df['close'].values
            volume = df['volume'].values.astype(float)
            
            logger.info("Adding trend indicators...")
            df = self._add_trend_indicators(df, open_price, high, low, close)
            
            logger.info("Adding momentum indicators...")
            df = self._add_momentum_indicators(df, open_price, high, low, close, volume)
            
            logger.info("Adding volatility indicators...")
            df = self._add_volatility_indicators(df, high, low, close)
            
            logger.info("Adding volume indicators...")
            df = self._add_volume_indicators(df, high, low, close, volume)
            
            if include_patterns and TALIB_AVAILABLE:
                logger.info("Adding candlestick patterns...")
                df = self._add_candlestick_patterns(df, open_price, high, low, close)
            
            logger.info("Adding derived features...")
            df = self._add_derived_features(df)
        
        # Defragment DataFrame after all column additions
        return df.copy()
    
    def _add_trend_indicators(
        self,
        df: pd.DataFrame,
        open_price: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
    ) -> pd.DataFrame:
        """Add trend-following indicators."""
        
        if TALIB_AVAILABLE:
            # Simple Moving Averages
            for period in self.sma_periods:
                df[f'sma_{period}'] = talib.SMA(close, timeperiod=period)
            
            # Exponential Moving Averages
            for period in self.ema_periods:
                df[f'ema_{period}'] = talib.EMA(close, timeperiod=period)
            
            # Double EMA
            df['dema_20'] = talib.DEMA(close, timeperiod=20)
            
            # Triple EMA
            df['tema_20'] = talib.TEMA(close, timeperiod=20)
            
            # Kaufman Adaptive Moving Average
            df['kama_20'] = talib.KAMA(close, timeperiod=20)
            
            # Triangular Moving Average
            df['trima_20'] = talib.TRIMA(close, timeperiod=20)
            
            # Weighted Moving Average
            df['wma_20'] = talib.WMA(close, timeperiod=20)
            
            # T3 - Triple Exponential Moving Average
            df['t3_10'] = talib.T3(close, timeperiod=10, vfactor=0.7)
            
            # MESA Adaptive Moving Average
            df['mama'], df['fama'] = talib.MAMA(close, fastlimit=0.5, slowlimit=0.05)
            
            # Hilbert Transform - Instantaneous Trendline
            df['ht_trendline'] = talib.HT_TRENDLINE(close)
            
            # Parabolic SAR
            df['sar'] = talib.SAR(high, low, acceleration=0.02, maximum=0.2)
            df['sar_signal'] = np.where(close > df['sar'], 1, -1)
            
            # ADX - Average Directional Movement Index
            df['adx'] = talib.ADX(high, low, close, timeperiod=self.adx_period)
            df['plus_di'] = talib.PLUS_DI(high, low, close, timeperiod=self.adx_period)
            df['minus_di'] = talib.MINUS_DI(high, low, close, timeperiod=self.adx_period)
            df['adxr'] = talib.ADXR(high, low, close, timeperiod=self.adx_period)
            
            # Aroon indicators
            df['aroon_up'], df['aroon_down'] = talib.AROON(high, low, timeperiod=25)
            df['aroon_osc'] = talib.AROONOSC(high, low, timeperiod=25)
            
        else:
            # Fallback implementations
            for period in self.sma_periods:
                df[f'sma_{period}'] = pd.Series(close).rolling(period).mean()
            
            for period in self.ema_periods:
                df[f'ema_{period}'] = pd.Series(close).ewm(span=period).mean()
            
            # Simple ADX approximation
            df['adx'] = self._calculate_adx_fallback(high, low, close)
        
        # Moving average crossover signals
        if 'sma_20' in df.columns and 'sma_50' in df.columns:
            df['sma_cross_20_50'] = np.where(df['sma_20'] > df['sma_50'], 1, -1)
        
        if 'ema_10' in df.columns and 'ema_20' in df.columns:
            df['ema_cross_10_20'] = np.where(df['ema_10'] > df['ema_20'], 1, -1)
        
        return df
    
    def _add_momentum_indicators(
        self,
        df: pd.DataFrame,
        open_price: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        volume: np.ndarray,
    ) -> pd.DataFrame:
        """Add momentum indicators."""
        
        if TALIB_AVAILABLE:
            # RSI
            df['rsi'] = talib.RSI(close, timeperiod=self.rsi_period)
            df['rsi_7'] = talib.RSI(close, timeperiod=7)
            df['rsi_21'] = talib.RSI(close, timeperiod=21)
            
            # MACD
            df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(
                close,
                fastperiod=self.macd_fast,
                slowperiod=self.macd_slow,
                signalperiod=self.macd_signal
            )
            
            # Stochastic
            df['stoch_k'], df['stoch_d'] = talib.STOCH(
                high, low, close,
                fastk_period=14, slowk_period=3, slowk_matype=0,
                slowd_period=3, slowd_matype=0
            )
            
            # Stochastic Fast
            df['stoch_fastk'], df['stoch_fastd'] = talib.STOCHF(
                high, low, close,
                fastk_period=14, fastd_period=3, fastd_matype=0
            )
            
            # Stochastic RSI
            df['stoch_rsi_k'], df['stoch_rsi_d'] = talib.STOCHRSI(
                close, timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0
            )
            
            # CCI - Commodity Channel Index
            df['cci'] = talib.CCI(high, low, close, timeperiod=20)
            df['cci_14'] = talib.CCI(high, low, close, timeperiod=14)
            
            # CMO - Chande Momentum Oscillator
            df['cmo'] = talib.CMO(close, timeperiod=14)
            
            # MOM - Momentum
            df['mom_10'] = talib.MOM(close, timeperiod=10)
            df['mom_20'] = talib.MOM(close, timeperiod=20)
            
            # ROC - Rate of Change
            df['roc'] = talib.ROC(close, timeperiod=10)
            df['rocp'] = talib.ROCP(close, timeperiod=10)
            df['rocr'] = talib.ROCR(close, timeperiod=10)
            
            # Williams %R
            df['willr'] = talib.WILLR(high, low, close, timeperiod=14)
            
            # Ultimate Oscillator
            df['ultosc'] = talib.ULTOSC(high, low, close, timeperiod1=7, timeperiod2=14, timeperiod3=28)
            
            # Balance of Power
            df['bop'] = talib.BOP(open_price, high, low, close)
            
            # PPO - Percentage Price Oscillator
            df['ppo'] = talib.PPO(close, fastperiod=12, slowperiod=26, matype=0)
            
            # APO - Absolute Price Oscillator
            df['apo'] = talib.APO(close, fastperiod=12, slowperiod=26, matype=0)
            
            # DX - Directional Movement Index
            df['dx'] = talib.DX(high, low, close, timeperiod=14)
            
            # TRIX - 1-day Rate-Of-Change of Triple Smooth EMA
            df['trix'] = talib.TRIX(close, timeperiod=15)
            
            # MFI - Money Flow Index
            df['mfi'] = talib.MFI(high, low, close, volume, timeperiod=14)
            
        else:
            # Fallback RSI
            df['rsi'] = self._calculate_rsi_fallback(close, self.rsi_period)
            df['rsi_7'] = self._calculate_rsi_fallback(close, 7)
            
            # Fallback MACD
            ema_fast = pd.Series(close).ewm(span=self.macd_fast).mean()
            ema_slow = pd.Series(close).ewm(span=self.macd_slow).mean()
            df['macd'] = ema_fast - ema_slow
            df['macd_signal'] = df['macd'].ewm(span=self.macd_signal).mean()
            df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # Add RSI overbought/oversold signals
        df['rsi_overbought'] = (df['rsi'] > 70).astype(int)
        df['rsi_oversold'] = (df['rsi'] < 30).astype(int)
        
        return df
    
    def _add_volatility_indicators(
        self,
        df: pd.DataFrame,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
    ) -> pd.DataFrame:
        """Add volatility indicators."""
        
        if TALIB_AVAILABLE:
            # ATR - Average True Range
            df['atr'] = talib.ATR(high, low, close, timeperiod=self.atr_period)
            df['atr_7'] = talib.ATR(high, low, close, timeperiod=7)
            df['atr_21'] = talib.ATR(high, low, close, timeperiod=21)
            
            # NATR - Normalized ATR
            df['natr'] = talib.NATR(high, low, close, timeperiod=self.atr_period)
            
            # True Range
            df['trange'] = talib.TRANGE(high, low, close)
            
            # Bollinger Bands
            df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(
                close,
                timeperiod=self.bb_period,
                nbdevup=self.bb_std,
                nbdevdn=self.bb_std,
                matype=0
            )
            
            # Bollinger Band Width
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
            
            # %B - Where price is within Bollinger Bands
            df['bb_pct'] = (close - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            
        else:
            # Fallback ATR
            df['atr'] = self._calculate_atr_fallback(high, low, close, self.atr_period)
            
            # Fallback Bollinger Bands
            rolling_mean = pd.Series(close).rolling(self.bb_period).mean()
            rolling_std = pd.Series(close).rolling(self.bb_period).std()
            df['bb_upper'] = rolling_mean + (self.bb_std * rolling_std)
            df['bb_middle'] = rolling_mean
            df['bb_lower'] = rolling_mean - (self.bb_std * rolling_std)
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        # Add volatility regime
        df['high_volatility'] = (df['atr'] > df['atr'].rolling(50).mean() * 1.5).astype(int)
        
        return df
    
    def _add_volume_indicators(
        self,
        df: pd.DataFrame,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        volume: np.ndarray,
    ) -> pd.DataFrame:
        """Add volume indicators."""
        
        if TALIB_AVAILABLE:
            # OBV - On Balance Volume
            df['obv'] = talib.OBV(close, volume)
            
            # AD - Chaikin A/D Line
            df['ad'] = talib.AD(high, low, close, volume)
            
            # ADOSC - Chaikin A/D Oscillator
            df['adosc'] = talib.ADOSC(high, low, close, volume, fastperiod=3, slowperiod=10)
            
        else:
            # Fallback OBV
            df['obv'] = self._calculate_obv_fallback(close, volume)
        
        # Volume moving averages
        df['volume_sma_20'] = pd.Series(volume).rolling(20).mean()
        df['volume_ratio'] = volume / df['volume_sma_20']
        
        # Volume trend
        df['volume_trend'] = pd.Series(volume).rolling(5).mean() / pd.Series(volume).rolling(20).mean()
        
        return df
    
    def _add_candlestick_patterns(
        self,
        df: pd.DataFrame,
        open_price: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
    ) -> pd.DataFrame:
        """Add candlestick pattern recognition."""
        
        if not TALIB_AVAILABLE:
            return df
        
        # All 61 candlestick pattern functions
        pattern_functions = {
            'cdl_2crows': talib.CDL2CROWS,
            'cdl_3blackcrows': talib.CDL3BLACKCROWS,
            'cdl_3inside': talib.CDL3INSIDE,
            'cdl_3linestrike': talib.CDL3LINESTRIKE,
            'cdl_3outside': talib.CDL3OUTSIDE,
            'cdl_3starsinsouth': talib.CDL3STARSINSOUTH,
            'cdl_3whitesoldiers': talib.CDL3WHITESOLDIERS,
            'cdl_abandonedbaby': talib.CDLABANDONEDBABY,
            'cdl_advanceblock': talib.CDLADVANCEBLOCK,
            'cdl_belthold': talib.CDLBELTHOLD,
            'cdl_breakaway': talib.CDLBREAKAWAY,
            'cdl_closingmarubozu': talib.CDLCLOSINGMARUBOZU,
            'cdl_concealbabyswall': talib.CDLCONCEALBABYSWALL,
            'cdl_counterattack': talib.CDLCOUNTERATTACK,
            'cdl_darkcloudcover': talib.CDLDARKCLOUDCOVER,
            'cdl_doji': talib.CDLDOJI,
            'cdl_dojistar': talib.CDLDOJISTAR,
            'cdl_dragonflydoji': talib.CDLDRAGONFLYDOJI,
            'cdl_engulfing': talib.CDLENGULFING,
            'cdl_eveningdojistar': talib.CDLEVENINGDOJISTAR,
            'cdl_eveningstar': talib.CDLEVENINGSTAR,
            'cdl_gapsidesidewhite': talib.CDLGAPSIDESIDEWHITE,
            'cdl_gravestonedoji': talib.CDLGRAVESTONEDOJI,
            'cdl_hammer': talib.CDLHAMMER,
            'cdl_hangingman': talib.CDLHANGINGMAN,
            'cdl_harami': talib.CDLHARAMI,
            'cdl_haramicross': talib.CDLHARAMICROSS,
            'cdl_highwave': talib.CDLHIGHWAVE,
            'cdl_hikkake': talib.CDLHIKKAKE,
            'cdl_hikkakemod': talib.CDLHIKKAKEMOD,
            'cdl_homingpigeon': talib.CDLHOMINGPIGEON,
            'cdl_identical3crows': talib.CDLIDENTICAL3CROWS,
            'cdl_inneck': talib.CDLINNECK,
            'cdl_invertedhammer': talib.CDLINVERTEDHAMMER,
            'cdl_kicking': talib.CDLKICKING,
            'cdl_kickingbylength': talib.CDLKICKINGBYLENGTH,
            'cdl_ladderbottom': talib.CDLLADDERBOTTOM,
            'cdl_longleggeddoji': talib.CDLLONGLEGGEDDOJI,
            'cdl_longline': talib.CDLLONGLINE,
            'cdl_marubozu': talib.CDLMARUBOZU,
            'cdl_matchinglow': talib.CDLMATCHINGLOW,
            'cdl_mathold': talib.CDLMATHOLD,
            'cdl_morningdojistar': talib.CDLMORNINGDOJISTAR,
            'cdl_morningstar': talib.CDLMORNINGSTAR,
            'cdl_onneck': talib.CDLONNECK,
            'cdl_piercing': talib.CDLPIERCING,
            'cdl_rickshawman': talib.CDLRICKSHAWMAN,
            'cdl_risefall3methods': talib.CDLRISEFALL3METHODS,
            'cdl_separatinglines': talib.CDLSEPARATINGLINES,
            'cdl_shootingstar': talib.CDLSHOOTINGSTAR,
            'cdl_shortline': talib.CDLSHORTLINE,
            'cdl_spinningtop': talib.CDLSPINNINGTOP,
            'cdl_stalledpattern': talib.CDLSTALLEDPATTERN,
            'cdl_sticksandwich': talib.CDLSTICKSANDWICH,
            'cdl_takuri': talib.CDLTAKURI,
            'cdl_tasukigap': talib.CDLTASUKIGAP,
            'cdl_thrusting': talib.CDLTHRUSTING,
            'cdl_tristar': talib.CDLTRISTAR,
            'cdl_unique3river': talib.CDLUNIQUE3RIVER,
            'cdl_upsidegap2crows': talib.CDLUPSIDEGAP2CROWS,
            'cdl_xsidegap3methods': talib.CDLXSIDEGAP3METHODS,
        }
        
        for name, func in pattern_functions.items():
            try:
                df[name] = func(open_price, high, low, close)
            except Exception as e:
                logger.warning(f"Error calculating {name}: {str(e)}")
                df[name] = 0
        
        # Aggregate bullish and bearish patterns
        bullish_patterns = [col for col in df.columns if col.startswith('cdl_') and (df[col] > 0).any()]
        bearish_patterns = [col for col in df.columns if col.startswith('cdl_') and (df[col] < 0).any()]
        
        df['bullish_pattern_count'] = (df[[col for col in df.columns if col.startswith('cdl_')]] > 0).sum(axis=1)
        df['bearish_pattern_count'] = (df[[col for col in df.columns if col.startswith('cdl_')]] < 0).sum(axis=1)
        
        return df
    
    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived features from existing indicators."""
        
        close = df['close']
        
        # Price momentum features
        df['returns_1d'] = close.pct_change(1)
        df['returns_5d'] = close.pct_change(5)
        df['returns_20d'] = close.pct_change(20)
        
        # Log returns
        df['log_returns_1d'] = np.log(close / close.shift(1))
        
        # Volatility of returns
        df['volatility_20d'] = df['returns_1d'].rolling(20).std()
        df['volatility_60d'] = df['returns_1d'].rolling(60).std()
        
        # Distance from moving averages
        if 'sma_20' in df.columns:
            df['dist_sma_20'] = (close - df['sma_20']) / df['sma_20']
        if 'sma_50' in df.columns:
            df['dist_sma_50'] = (close - df['sma_50']) / df['sma_50']
        if 'sma_200' in df.columns:
            df['dist_sma_200'] = (close - df['sma_200']) / df['sma_200']
        
        # RSI divergence
        if 'rsi' in df.columns:
            df['rsi_slope'] = df['rsi'].diff(5)
            df['price_slope'] = close.pct_change(5)
            df['rsi_divergence'] = np.sign(df['rsi_slope']) != np.sign(df['price_slope'])
        
        # MACD histogram trend
        if 'macd_hist' in df.columns:
            df['macd_hist_slope'] = df['macd_hist'].diff(3)
        
        # Trend strength
        if 'adx' in df.columns:
            df['strong_trend'] = (df['adx'] > 25).astype(int)
            df['very_strong_trend'] = (df['adx'] > 40).astype(int)
        
        return df
    
    # Fallback implementations when TA-Lib is not available
    def _calculate_rsi_fallback(self, close: np.ndarray, period: int = 14) -> pd.Series:
        """Calculate RSI without TA-Lib."""
        delta = pd.Series(close).diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_atr_fallback(
        self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14
    ) -> pd.Series:
        """Calculate ATR without TA-Lib."""
        high = pd.Series(high)
        low = pd.Series(low)
        close = pd.Series(close)
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(period).mean()
    
    def _calculate_adx_fallback(
        self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14
    ) -> pd.Series:
        """Simplified ADX calculation without TA-Lib."""
        high = pd.Series(high)
        low = pd.Series(low)
        
        plus_dm = high.diff()
        minus_dm = -low.diff()
        
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
        
        atr = self._calculate_atr_fallback(high.values, low.values, close, period)
        
        plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(period).mean() / atr)
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(period).mean()
        
        return adx
    
    def _calculate_obv_fallback(self, close: np.ndarray, volume: np.ndarray) -> pd.Series:
        """Calculate OBV without TA-Lib."""
        close = pd.Series(close)
        volume = pd.Series(volume)
        
        direction = np.sign(close.diff())
        direction.iloc[0] = 0
        
        return (direction * volume).cumsum()
    
    def get_feature_names(self) -> List[str]:
        """Get list of all feature names that will be generated."""
        features = []
        
        # Moving averages
        features.extend([f'sma_{p}' for p in self.sma_periods])
        features.extend([f'ema_{p}' for p in self.ema_periods])
        
        # Other indicators
        features.extend([
            'dema_20', 'tema_20', 'kama_20', 'trima_20', 'wma_20', 't3_10',
            'mama', 'fama', 'ht_trendline', 'sar', 'sar_signal',
            'adx', 'plus_di', 'minus_di', 'adxr',
            'aroon_up', 'aroon_down', 'aroon_osc',
            'rsi', 'rsi_7', 'rsi_21',
            'macd', 'macd_signal', 'macd_hist',
            'stoch_k', 'stoch_d', 'stoch_fastk', 'stoch_fastd',
            'cci', 'cci_14', 'cmo', 'mom_10', 'mom_20',
            'roc', 'rocp', 'rocr', 'willr', 'ultosc', 'bop', 'ppo', 'apo',
            'atr', 'atr_7', 'atr_21', 'natr', 'trange',
            'bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'bb_pct',
            'obv', 'ad', 'adosc', 'volume_sma_20', 'volume_ratio', 'volume_trend',
            'returns_1d', 'returns_5d', 'returns_20d', 'log_returns_1d',
            'volatility_20d', 'volatility_60d',
        ])
        
        return features
