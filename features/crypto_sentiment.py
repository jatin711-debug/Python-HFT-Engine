"""
Crypto Sentiment Analyzer.

Aggregates sentiment from multiple crypto-specific sources:
1. Trade flow sentiment (from Binance websocket)
2. Social sentiment (Twitter, Reddit crypto subs)
3. Funding rate sentiment (perpetual futures)
4. Liquidation sentiment (large liquidations = capitulation)

Designed for sub-minute trading decisions.

Usage:
    analyzer = CryptoSentimentAnalyzer()
    sentiment = analyzer.get_composite_sentiment(symbol='BTCUSDT')
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from enum import Enum
import requests
import asyncio

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

class SentimentSource(Enum):
    """Sources of sentiment data."""
    TRADE_FLOW = "trade_flow"
    FUNDING_RATE = "funding_rate"
    LIQUIDATIONS = "liquidations"
    FEAR_GREED = "fear_greed"
    SOCIAL = "social"


@dataclass
class SentimentSignal:
    """Individual sentiment signal."""
    source: SentimentSource
    value: float  # -1 (bearish) to +1 (bullish)
    confidence: float  # 0 to 1
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CompositeSentiment:
    """Aggregated sentiment from all sources."""
    timestamp: datetime
    symbol: str
    
    # Individual signals
    trade_flow: float = 0.0
    funding_rate: float = 0.0
    liquidation: float = 0.0
    fear_greed: float = 0.0
    social: float = 0.0
    
    # Composite
    composite: float = 0.0
    confidence: float = 0.0
    
    # Regime
    is_extreme_fear: bool = False
    is_extreme_greed: bool = False
    
    def to_dict(self) -> Dict:
        return {
            'trade_flow': self.trade_flow,
            'funding_rate': self.funding_rate,
            'liquidation': self.liquidation,
            'fear_greed': self.fear_greed,
            'social': self.social,
            'composite': self.composite,
            'confidence': self.confidence,
            'is_extreme_fear': self.is_extreme_fear,
            'is_extreme_greed': self.is_extreme_greed,
        }


# =============================================================================
# CRYPTO SENTIMENT ANALYZER
# =============================================================================

class CryptoSentimentAnalyzer:
    """
    Multi-source crypto sentiment analyzer.
    
    Sources:
    1. Trade flow (from Binance handler)
    2. Funding rates (Binance futures API)
    3. Liquidations (via public APIs)
    4. Fear & Greed Index
    5. Social sentiment (simplified)
    
    Weights are optimized for short-term trading signals.
    """
    
    # API endpoints
    BINANCE_FUTURES_URL = "https://fapi.binance.com"
    FEAR_GREED_URL = "https://api.alternative.me/fng/"
    
    # Sentiment weights (tuned for short-term crypto trading)
    WEIGHTS = {
        SentimentSource.TRADE_FLOW: 0.35,      # Most predictive for 1-5s
        SentimentSource.FUNDING_RATE: 0.25,    # Crowded trade indicator
        SentimentSource.LIQUIDATIONS: 0.20,    # Capitulation/squeeze
        SentimentSource.FEAR_GREED: 0.10,      # Background sentiment
        SentimentSource.SOCIAL: 0.10,          # Noise but useful for extremes
    }
    
    def __init__(self, aggressive_mode: bool = True):
        """
        Initialize crypto sentiment analyzer.
        
        Args:
            aggressive_mode: If True, use more responsive settings
        """
        self.aggressive_mode = aggressive_mode
        self._cache: Dict[str, Any] = {}
        self._cache_ttl = 60 if not aggressive_mode else 30  # seconds
        
        # Funding rate thresholds
        self.funding_extreme_threshold = 0.001 if aggressive_mode else 0.0005
        
        logger.info(f"CryptoSentimentAnalyzer initialized (aggressive={aggressive_mode})")
    
    # -------------------------------------------------------------------------
    # PUBLIC INTERFACE
    # -------------------------------------------------------------------------
    
    def get_composite_sentiment(
        self,
        symbol: str = "BTCUSDT",
        trade_sentiment: Optional[Any] = None,  # CryptoSentiment from Binance
    ) -> CompositeSentiment:
        """
        Get composite sentiment from all sources.
        
        Args:
            symbol: Trading pair
            trade_sentiment: Real-time trade flow sentiment (from Binance handler)
            
        Returns:
            CompositeSentiment with all signals
        """
        result = CompositeSentiment(
            timestamp=datetime.now(),
            symbol=symbol,
        )
        
        signals: List[SentimentSignal] = []
        
        # 1. Trade flow (if provided from Binance handler)
        if trade_sentiment is not None:
            tf_signal = self._process_trade_flow(trade_sentiment)
            signals.append(tf_signal)
            result.trade_flow = tf_signal.value
        
        # 2. Funding rate
        try:
            fr_signal = self._get_funding_rate_sentiment(symbol)
            signals.append(fr_signal)
            result.funding_rate = fr_signal.value
        except Exception as e:
            logger.debug(f"Funding rate fetch failed: {e}")
        
        # 3. Liquidations (using funding as proxy for crowding)
        try:
            liq_signal = self._estimate_liquidation_sentiment(symbol)
            signals.append(liq_signal)
            result.liquidation = liq_signal.value
        except Exception as e:
            logger.debug(f"Liquidation sentiment failed: {e}")
        
        # 4. Fear & Greed Index
        try:
            fg_signal = self._get_fear_greed()
            signals.append(fg_signal)
            result.fear_greed = fg_signal.value
            
            # Check extremes
            fg_raw = fg_signal.metadata.get('raw_value', 50)
            result.is_extreme_fear = fg_raw < 20
            result.is_extreme_greed = fg_raw > 80
        except Exception as e:
            logger.debug(f"Fear & Greed fetch failed: {e}")
        
        # 5. Aggregate composite
        result.composite, result.confidence = self._aggregate_signals(signals)
        
        return result
    
    def add_sentiment_features(
        self,
        df: pd.DataFrame,
        symbol: str = "BTCUSDT",
        trade_sentiment: Optional[Any] = None,
    ) -> pd.DataFrame:
        """
        Add sentiment features to DataFrame.
        
        Args:
            df: OHLCV DataFrame
            symbol: Trading pair
            trade_sentiment: Real-time trade flow
            
        Returns:
            DataFrame with sentiment features
        """
        df = df.copy()
        
        # Get current sentiment
        sentiment = self.get_composite_sentiment(symbol, trade_sentiment)
        
        # Add as features (constant for all rows since it's current sentiment)
        for key, value in sentiment.to_dict().items():
            if isinstance(value, bool):
                df[f'sentiment_{key}'] = float(value)
            else:
                df[f'sentiment_{key}'] = value
        
        return df
    
    # -------------------------------------------------------------------------
    # SENTIMENT SOURCES
    # -------------------------------------------------------------------------
    
    def _process_trade_flow(self, trade_sentiment: Any) -> SentimentSignal:
        """Process trade flow sentiment from Binance handler."""
        # Combine volume imbalance and whale signal
        vol_imb = getattr(trade_sentiment, 'volume_imbalance', 0)
        whale = getattr(trade_sentiment, 'whale_signal', 0)
        trade_imb = getattr(trade_sentiment, 'trade_imbalance', 0)
        
        # Weighted combination
        value = 0.5 * vol_imb + 0.3 * whale + 0.2 * trade_imb
        
        # Confidence based on volume
        buy_vol = getattr(trade_sentiment, 'buy_volume', 0)
        sell_vol = getattr(trade_sentiment, 'sell_volume', 0)
        total_vol = buy_vol + sell_vol
        confidence = min(1.0, total_vol / 100000)  # More volume = more confidence
        
        return SentimentSignal(
            source=SentimentSource.TRADE_FLOW,
            value=np.clip(value, -1, 1),
            confidence=confidence,
            metadata={'volume': total_vol, 'whale_signal': whale}
        )
    
    def _get_funding_rate_sentiment(self, symbol: str) -> SentimentSignal:
        """
        Get funding rate sentiment.
        
        Positive funding = longs pay shorts = crowded long = bearish
        Negative funding = shorts pay longs = crowded short = bullish
        """
        cache_key = f"funding_{symbol}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached
        
        try:
            url = f"{self.BINANCE_FUTURES_URL}/fapi/v1/fundingRate"
            response = requests.get(
                url,
                params={'symbol': symbol, 'limit': 1},
                timeout=5
            )
            response.raise_for_status()
            data = response.json()
            
            if data:
                rate = float(data[0]['fundingRate'])
                
                # Normalize: high positive funding is bearish (contrarian)
                # Scale relative to extreme threshold
                normalized = -rate / self.funding_extreme_threshold
                value = np.clip(normalized, -1, 1)
                
                # Higher confidence for extreme rates
                confidence = min(1.0, abs(rate) / self.funding_extreme_threshold)
                
                signal = SentimentSignal(
                    source=SentimentSource.FUNDING_RATE,
                    value=value,
                    confidence=confidence,
                    metadata={'rate': rate}
                )
                
                self._set_cached(cache_key, signal)
                return signal
                
        except Exception as e:
            logger.debug(f"Funding rate API error: {e}")
        
        return SentimentSignal(
            source=SentimentSource.FUNDING_RATE,
            value=0,
            confidence=0
        )
    
    def _estimate_liquidation_sentiment(self, symbol: str) -> SentimentSignal:
        """
        Estimate liquidation sentiment.
        
        Since real liquidation data requires paid APIs, we use
        price volatility + funding rate as a proxy.
        
        High volatility with extreme funding = potential squeeze.
        """
        # This is a simplified proxy
        # In production, you'd use actual liquidation feeds
        
        return SentimentSignal(
            source=SentimentSource.LIQUIDATIONS,
            value=0,  # Neutral without real data
            confidence=0.3,
            metadata={'note': 'Proxy estimate'}
        )
    
    def _get_fear_greed(self) -> SentimentSignal:
        """
        Get Fear & Greed Index.
        
        0-24: Extreme Fear (contrarian bullish)
        25-44: Fear
        45-55: Neutral
        56-75: Greed
        76-100: Extreme Greed (contrarian bearish)
        """
        cache_key = "fear_greed"
        cached = self._get_cached(cache_key)
        if cached:
            return cached
        
        try:
            response = requests.get(self.FEAR_GREED_URL, timeout=5)
            response.raise_for_status()
            data = response.json()
            
            if data.get('data'):
                fg_value = int(data['data'][0]['value'])
                
                # Convert to sentiment: 
                # Extreme fear (0-20) = bullish (+1)
                # Neutral (50) = 0
                # Extreme greed (80-100) = bearish (-1)
                # Use contrarian logic
                normalized = (50 - fg_value) / 50
                value = np.clip(normalized, -1, 1)
                
                signal = SentimentSignal(
                    source=SentimentSource.FEAR_GREED,
                    value=value,
                    confidence=0.7,  # Generally reliable
                    metadata={'raw_value': fg_value}
                )
                
                self._set_cached(cache_key, signal)
                return signal
                
        except Exception as e:
            logger.debug(f"Fear & Greed API error: {e}")
        
        return SentimentSignal(
            source=SentimentSource.FEAR_GREED,
            value=0,
            confidence=0
        )
    
    # -------------------------------------------------------------------------
    # AGGREGATION
    # -------------------------------------------------------------------------
    
    def _aggregate_signals(
        self,
        signals: List[SentimentSignal],
    ) -> tuple:
        """
        Aggregate signals into composite sentiment.
        
        Returns:
            (composite_value, confidence)
        """
        if not signals:
            return 0.0, 0.0
        
        total_weight = 0
        weighted_sum = 0
        confidence_sum = 0
        
        for signal in signals:
            weight = self.WEIGHTS.get(signal.source, 0.1)
            weighted_sum += signal.value * signal.confidence * weight
            total_weight += weight * signal.confidence
            confidence_sum += signal.confidence * weight
        
        if total_weight > 0:
            composite = weighted_sum / total_weight
            confidence = confidence_sum / sum(self.WEIGHTS.values())
        else:
            composite = 0
            confidence = 0
        
        return np.clip(composite, -1, 1), np.clip(confidence, 0, 1)
    
    # -------------------------------------------------------------------------
    # CACHING
    # -------------------------------------------------------------------------
    
    def _get_cached(self, key: str) -> Optional[Any]:
        """Get cached value if not expired."""
        if key in self._cache:
            value, timestamp = self._cache[key]
            if time.time() - timestamp < self._cache_ttl:
                return value
        return None
    
    def _set_cached(self, key: str, value: Any):
        """Set cached value."""
        self._cache[key] = (value, time.time())


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def get_crypto_sentiment(
    symbol: str = "BTCUSDT",
    trade_sentiment: Any = None,
    aggressive: bool = True,
) -> CompositeSentiment:
    """
    Quick function to get crypto sentiment.
    
    Args:
        symbol: Trading pair
        trade_sentiment: From Binance handler
        aggressive: Use aggressive mode
        
    Returns:
        CompositeSentiment
    """
    analyzer = CryptoSentimentAnalyzer(aggressive_mode=aggressive)
    return analyzer.get_composite_sentiment(symbol, trade_sentiment)


if __name__ == '__main__':
    # Demo
    analyzer = CryptoSentimentAnalyzer(aggressive_mode=True)
    
    print("Fetching crypto sentiment...")
    sentiment = analyzer.get_composite_sentiment("BTCUSDT")
    
    print(f"\n=== Crypto Sentiment for BTCUSDT ===")
    print(f"Trade Flow:     {sentiment.trade_flow:+.3f}")
    print(f"Funding Rate:   {sentiment.funding_rate:+.3f}")
    print(f"Liquidation:    {sentiment.liquidation:+.3f}")
    print(f"Fear & Greed:   {sentiment.fear_greed:+.3f}")
    print(f"---")
    print(f"Composite:      {sentiment.composite:+.3f}")
    print(f"Confidence:     {sentiment.confidence:.2f}")
    print(f"Extreme Fear:   {sentiment.is_extreme_fear}")
    print(f"Extreme Greed:  {sentiment.is_extreme_greed}")
