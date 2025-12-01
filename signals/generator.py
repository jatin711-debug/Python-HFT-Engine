"""
Signal Generator Module.

Generates buy/sell/hold signals by combining:
- ML model predictions
- Sentiment analysis
- Technical indicators
- Risk management rules
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class SignalType(Enum):
    """Trading signal types."""
    STRONG_BUY = 2
    BUY = 1
    HOLD = 0
    SELL = -1
    STRONG_SELL = -2


class PositionType(Enum):
    """Position types."""
    LONG = 1
    SHORT = -1
    FLAT = 0


@dataclass
class TradingSignal:
    """Container for trading signal with metadata."""
    timestamp: datetime
    symbol: str
    signal: SignalType
    position: PositionType
    ml_probability: float
    sentiment_score: float
    technical_score: float
    confidence: float
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    risk_reward_ratio: Optional[float] = None
    notes: List[str] = field(default_factory=list)


@dataclass
class SignalConfig:
    """Configuration for signal generation."""
    # Thresholds (adjusted for realistic ML probability outputs ~0.50-0.55)
    strong_buy_threshold: float = 0.58      # Strong buy above 58%
    buy_threshold: float = 0.52             # Buy above 52%
    sell_threshold: float = 0.48            # Sell below 48%
    strong_sell_threshold: float = 0.42     # Strong sell below 42%
    
    # Weights for combining scores
    ml_weight: float = 0.5
    sentiment_weight: float = 0.2
    technical_weight: float = 0.3
    
    # Confidence requirements (lowered for more signals)
    min_confidence: float = 0.25
    min_sentiment_confidence: float = 0.1
    
    # Risk management
    default_stop_loss_pct: float = 0.02  # 2%
    default_take_profit_pct: float = 0.04  # 4% (2:1 RR)
    max_position_size: float = 0.1  # 10% of portfolio
    
    # Sentiment thresholds
    sentiment_bullish_threshold: float = 0.1
    sentiment_bearish_threshold: float = -0.1
    
    # Technical confirmation (disabled for more signals)
    require_trend_confirmation: bool = False
    require_volume_confirmation: bool = False


class SignalGenerator:
    """
    Sophisticated signal generator for trading decisions.
    
    Combines:
    - ML model predictions (probability of price increase)
    - Sentiment analysis (news sentiment scores)
    - Technical indicators (trend, momentum, volatility)
    - Risk management rules (position sizing, stop loss)
    
    Example:
        >>> generator = SignalGenerator(model, sentiment_analyzer)
        >>> signals = generator.generate_signals(data)
        >>> for signal in signals:
        ...     print(f"{signal.symbol}: {signal.signal.name} at {signal.entry_price}")
    """
    
    def __init__(
        self,
        ml_model: Any = None,
        sentiment_analyzer: Any = None,
        config: SignalConfig = None,
    ):
        """
        Initialize signal generator.
        
        Args:
            ml_model: Trained ML model for predictions
            sentiment_analyzer: Sentiment analyzer for news
            config: Signal generation configuration
        """
        self.ml_model = ml_model
        self.sentiment_analyzer = sentiment_analyzer
        self.config = config or SignalConfig()
        
        # Signal history for analysis
        self.signal_history: List[TradingSignal] = []
    
    def generate_signals(
        self,
        data: pd.DataFrame,
        symbol: str = "UNKNOWN",
        news_data: Optional[Dict] = None,
    ) -> List[TradingSignal]:
        """
        Generate trading signals for given data.
        
        Args:
            data: DataFrame with OHLCV and features
            symbol: Stock symbol
            news_data: Optional news data for sentiment
            
        Returns:
            List of TradingSignal objects
        """
        signals = []
        
        # Ensure we have required columns
        required_cols = ['close', 'high', 'low', 'volume']
        if not all(col in data.columns for col in required_cols):
            raise ValueError(f"Data must contain columns: {required_cols}")
        
        # Get ML predictions
        ml_probs = self._get_ml_predictions(data)
        
        # Get sentiment scores
        sentiment_scores = self._get_sentiment_scores(data, news_data)
        
        # Get technical scores
        technical_scores = self._get_technical_scores(data)
        
        # Generate signals for each row
        for i in range(len(data)):
            idx = data.index[i]
            row = data.iloc[i]
            
            # Skip if not enough data for calculations
            if i < 50:  # Need history for indicators
                continue
            
            # Get scores
            ml_prob = ml_probs[i] if ml_probs is not None else 0.5
            sentiment = sentiment_scores[i] if sentiment_scores is not None else 0.0
            technical = technical_scores[i] if technical_scores is not None else 0.5
            
            # Combine scores
            combined_score = self._combine_scores(ml_prob, sentiment, technical)
            
            # Calculate confidence
            confidence = self._calculate_confidence(
                ml_prob, sentiment, technical, row
            )
            
            # Determine signal
            signal_type, position = self._determine_signal(
                combined_score, confidence, row, data.iloc[:i+1]
            )
            
            # Calculate risk parameters
            entry_price = row['close']
            stop_loss, take_profit = self._calculate_risk_levels(
                entry_price, signal_type, row, data.iloc[:i+1]
            )
            
            # Create signal object
            signal = TradingSignal(
                timestamp=idx if isinstance(idx, datetime) else datetime.now(),
                symbol=symbol,
                signal=signal_type,
                position=position,
                ml_probability=ml_prob,
                sentiment_score=sentiment,
                technical_score=technical,
                confidence=confidence,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                risk_reward_ratio=self._calculate_rr_ratio(
                    entry_price, stop_loss, take_profit
                ),
                notes=self._generate_notes(ml_prob, sentiment, technical, row),
            )
            
            signals.append(signal)
        
        self.signal_history.extend(signals)
        return signals
    
    def _get_ml_predictions(self, data: pd.DataFrame) -> Optional[np.ndarray]:
        """Get ML model predictions."""
        if self.ml_model is None:
            return None
        
        try:
            # Identify feature columns (exclude OHLCV and target)
            exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'target', 'date']
            feature_cols = [c for c in data.columns if c.lower() not in exclude_cols]
            
            if not feature_cols:
                logger.warning("No feature columns found for ML prediction")
                return np.full(len(data), 0.5)
            
            X = data[feature_cols].fillna(0)
            
            # Get predictions
            if hasattr(self.ml_model, 'predict_with_confidence'):
                probs, _ = self.ml_model.predict_with_confidence(X)
            else:
                probs = self.ml_model.predict(X)
            
            return probs
            
        except Exception as e:
            logger.warning(f"ML prediction failed: {e}")
            return np.full(len(data), 0.5)
    
    def _get_sentiment_scores(
        self,
        data: pd.DataFrame,
        news_data: Optional[Dict],
    ) -> np.ndarray:
        """Get sentiment scores for each row."""
        # Check if sentiment is already in data
        if 'sentiment_mean' in data.columns:
            return data['sentiment_mean'].fillna(0).values
        
        if 'sentiment_weighted' in data.columns:
            return data['sentiment_weighted'].fillna(0).values
        
        # If we have news data and analyzer, calculate sentiment
        if news_data and self.sentiment_analyzer:
            scores = []
            for idx in data.index:
                date_str = idx.strftime('%Y-%m-%d') if isinstance(idx, datetime) else str(idx)
                articles = news_data.get(date_str, [])
                
                if articles:
                    result = self.sentiment_analyzer.analyze_articles(articles)
                    scores.append(result['weighted_sentiment'])
                else:
                    scores.append(0.0)
            
            return np.array(scores)
        
        return np.zeros(len(data))
    
    def _get_technical_scores(self, data: pd.DataFrame) -> np.ndarray:
        """Calculate technical analysis scores."""
        scores = []
        
        for i in range(len(data)):
            row = data.iloc[i]
            score = 0.5  # Neutral base
            components = []
            
            # Trend indicators
            if 'adx' in data.columns:
                # Strong trend confirmation
                if row.get('adx', 0) > 25:
                    if row.get('plus_di', 0) > row.get('minus_di', 0):
                        score += 0.1
                        components.append('bullish_trend')
                    else:
                        score -= 0.1
                        components.append('bearish_trend')
            
            # RSI
            if 'rsi' in data.columns:
                rsi = row.get('rsi', 50)
                if rsi < 30:
                    score += 0.1  # Oversold = bullish
                elif rsi > 70:
                    score -= 0.1  # Overbought = bearish
            
            # MACD
            if 'macd_hist' in data.columns:
                macd_hist = row.get('macd_hist', 0)
                if macd_hist > 0:
                    score += 0.05
                elif macd_hist < 0:
                    score -= 0.05
            
            # Bollinger Bands position
            if 'bb_pct' in data.columns:
                bb_pct = row.get('bb_pct', 0.5)
                if bb_pct < 0.1:  # Near lower band
                    score += 0.1
                elif bb_pct > 0.9:  # Near upper band
                    score -= 0.1
            
            # Moving average crossovers
            if 'sma_20' in data.columns and 'sma_50' in data.columns:
                if row.get('sma_20', 0) > row.get('sma_50', 0):
                    score += 0.05
                else:
                    score -= 0.05
            
            # Volume confirmation
            if 'volume_ratio' in data.columns:
                vol_ratio = row.get('volume_ratio', 1)
                if vol_ratio > 1.5:  # High volume confirms moves
                    # Amplify the current direction
                    if score > 0.5:
                        score += 0.05
                    elif score < 0.5:
                        score -= 0.05
            
            # Normalize score to [0, 1]
            score = np.clip(score, 0, 1)
            scores.append(score)
        
        return np.array(scores)
    
    def _combine_scores(
        self,
        ml_prob: float,
        sentiment: float,
        technical: float,
    ) -> float:
        """Combine all scores using weighted average."""
        # Normalize sentiment from [-1, 1] to [0, 1]
        sentiment_normalized = (sentiment + 1) / 2
        
        combined = (
            self.config.ml_weight * ml_prob +
            self.config.sentiment_weight * sentiment_normalized +
            self.config.technical_weight * technical
        )
        
        return combined
    
    def _calculate_confidence(
        self,
        ml_prob: float,
        sentiment: float,
        technical: float,
        row: pd.Series,
    ) -> float:
        """Calculate confidence score for the signal."""
        confidence_factors = []
        
        # ML confidence: how far from 0.5
        ml_confidence = abs(ml_prob - 0.5) * 2
        confidence_factors.append(ml_confidence)
        
        # Sentiment confidence: strength of sentiment
        sentiment_confidence = abs(sentiment)
        confidence_factors.append(sentiment_confidence)
        
        # Technical confidence: how far from neutral
        tech_confidence = abs(technical - 0.5) * 2
        confidence_factors.append(tech_confidence)
        
        # Agreement factor: do all signals agree?
        ml_direction = 1 if ml_prob > 0.5 else -1
        sentiment_direction = 1 if sentiment > 0 else -1 if sentiment < 0 else 0
        tech_direction = 1 if technical > 0.5 else -1
        
        directions = [ml_direction, sentiment_direction, tech_direction]
        agreement = abs(sum(directions)) / 3
        confidence_factors.append(agreement)
        
        # Volatility factor: lower volatility = higher confidence
        if 'atr' in row.index and 'close' in row.index:
            atr_pct = row['atr'] / row['close']
            vol_confidence = max(0, 1 - atr_pct * 20)  # Normalize
            confidence_factors.append(vol_confidence)
        
        return np.mean(confidence_factors)
    
    def _determine_signal(
        self,
        combined_score: float,
        confidence: float,
        row: pd.Series,
        history: pd.DataFrame,
    ) -> Tuple[SignalType, PositionType]:
        """Determine signal type based on combined score and filters."""
        
        # Check minimum confidence
        if confidence < self.config.min_confidence:
            return SignalType.HOLD, PositionType.FLAT
        
        # Determine base signal from score
        if combined_score >= self.config.strong_buy_threshold:
            signal = SignalType.STRONG_BUY
            position = PositionType.LONG
        elif combined_score >= self.config.buy_threshold:
            signal = SignalType.BUY
            position = PositionType.LONG
        elif combined_score <= self.config.strong_sell_threshold:
            signal = SignalType.STRONG_SELL
            position = PositionType.SHORT
        elif combined_score <= self.config.sell_threshold:
            signal = SignalType.SELL
            position = PositionType.SHORT
        else:
            signal = SignalType.HOLD
            position = PositionType.FLAT
        
        # Apply filters
        if self.config.require_trend_confirmation:
            if not self._check_trend_confirmation(position, row, history):
                return SignalType.HOLD, PositionType.FLAT
        
        if self.config.require_volume_confirmation:
            if not self._check_volume_confirmation(row):
                # Downgrade strong signals
                if signal == SignalType.STRONG_BUY:
                    signal = SignalType.BUY
                elif signal == SignalType.STRONG_SELL:
                    signal = SignalType.SELL
        
        return signal, position
    
    def _check_trend_confirmation(
        self,
        position: PositionType,
        row: pd.Series,
        history: pd.DataFrame,
    ) -> bool:
        """Check if trend confirms the signal direction."""
        if position == PositionType.FLAT:
            return True
        
        # Check moving average alignment
        if 'sma_20' in row.index and 'sma_50' in row.index:
            ma_bullish = row['sma_20'] > row['sma_50']
            
            if position == PositionType.LONG and not ma_bullish:
                return False
            if position == PositionType.SHORT and ma_bullish:
                return False
        
        # Check ADX trend strength
        if 'adx' in row.index:
            if row['adx'] < 20:  # Weak trend
                return False
        
        return True
    
    def _check_volume_confirmation(self, row: pd.Series) -> bool:
        """Check if volume confirms the move."""
        if 'volume_ratio' in row.index:
            return row['volume_ratio'] > 0.8  # At least 80% of average
        return True
    
    def _calculate_risk_levels(
        self,
        entry_price: float,
        signal: SignalType,
        row: pd.Series,
        history: pd.DataFrame,
    ) -> Tuple[float, float]:
        """Calculate stop loss and take profit levels."""
        
        # Use ATR for dynamic stop loss if available
        if 'atr' in row.index:
            atr = row['atr']
            atr_multiplier = 2.0
        else:
            atr = entry_price * self.config.default_stop_loss_pct
            atr_multiplier = 1.0
        
        if signal in [SignalType.STRONG_BUY, SignalType.BUY]:
            # Long position
            stop_loss = entry_price - (atr * atr_multiplier)
            take_profit = entry_price + (atr * atr_multiplier * 2)  # 2:1 RR
            
            # Adjust stop loss to recent swing low
            if len(history) >= 20:
                recent_low = history['low'].tail(20).min()
                stop_loss = max(stop_loss, recent_low * 0.99)
        
        elif signal in [SignalType.STRONG_SELL, SignalType.SELL]:
            # Short position
            stop_loss = entry_price + (atr * atr_multiplier)
            take_profit = entry_price - (atr * atr_multiplier * 2)  # 2:1 RR
            
            # Adjust stop loss to recent swing high
            if len(history) >= 20:
                recent_high = history['high'].tail(20).max()
                stop_loss = min(stop_loss, recent_high * 1.01)
        
        else:
            stop_loss = None
            take_profit = None
        
        return stop_loss, take_profit
    
    def _calculate_rr_ratio(
        self,
        entry: float,
        stop_loss: Optional[float],
        take_profit: Optional[float],
    ) -> Optional[float]:
        """Calculate risk/reward ratio."""
        if not all([entry, stop_loss, take_profit]):
            return None
        
        risk = abs(entry - stop_loss)
        reward = abs(take_profit - entry)
        
        if risk > 0:
            return reward / risk
        return None
    
    def _generate_notes(
        self,
        ml_prob: float,
        sentiment: float,
        technical: float,
        row: pd.Series,
    ) -> List[str]:
        """Generate explanatory notes for the signal."""
        notes = []
        
        # ML model notes
        if ml_prob > 0.7:
            notes.append(f"Strong ML bullish signal ({ml_prob:.2%})")
        elif ml_prob < 0.3:
            notes.append(f"Strong ML bearish signal ({ml_prob:.2%})")
        
        # Sentiment notes
        if sentiment > self.config.sentiment_bullish_threshold:
            notes.append(f"Positive news sentiment ({sentiment:+.2f})")
        elif sentiment < self.config.sentiment_bearish_threshold:
            notes.append(f"Negative news sentiment ({sentiment:+.2f})")
        
        # Technical notes
        if 'rsi' in row.index:
            rsi = row['rsi']
            if rsi < 30:
                notes.append(f"RSI oversold ({rsi:.1f})")
            elif rsi > 70:
                notes.append(f"RSI overbought ({rsi:.1f})")
        
        if 'adx' in row.index:
            adx = row['adx']
            if adx > 40:
                notes.append(f"Very strong trend (ADX={adx:.1f})")
            elif adx > 25:
                notes.append(f"Strong trend (ADX={adx:.1f})")
        
        return notes
    
    def get_signal_summary(
        self,
        signals: List[TradingSignal],
    ) -> Dict[str, Any]:
        """Get summary statistics for signals."""
        if not signals:
            return {}
        
        total = len(signals)
        
        buy_signals = [s for s in signals if s.signal in [SignalType.BUY, SignalType.STRONG_BUY]]
        sell_signals = [s for s in signals if s.signal in [SignalType.SELL, SignalType.STRONG_SELL]]
        hold_signals = [s for s in signals if s.signal == SignalType.HOLD]
        
        avg_confidence = np.mean([s.confidence for s in signals])
        high_confidence = [s for s in signals if s.confidence > 0.7]
        
        return {
            'total_signals': total,
            'buy_count': len(buy_signals),
            'sell_count': len(sell_signals),
            'hold_count': len(hold_signals),
            'buy_percentage': len(buy_signals) / total * 100,
            'sell_percentage': len(sell_signals) / total * 100,
            'hold_percentage': len(hold_signals) / total * 100,
            'avg_confidence': avg_confidence,
            'high_confidence_count': len(high_confidence),
            'avg_ml_probability': np.mean([s.ml_probability for s in signals]),
            'avg_sentiment': np.mean([s.sentiment_score for s in signals]),
            'avg_technical': np.mean([s.technical_score for s in signals]),
        }
    
    def signals_to_dataframe(
        self,
        signals: List[TradingSignal],
    ) -> pd.DataFrame:
        """Convert signals to DataFrame."""
        data = []
        
        for s in signals:
            data.append({
                'timestamp': s.timestamp,
                'symbol': s.symbol,
                'signal': s.signal.name,
                'signal_value': s.signal.value,
                'position': s.position.name,
                'ml_probability': s.ml_probability,
                'sentiment_score': s.sentiment_score,
                'technical_score': s.technical_score,
                'confidence': s.confidence,
                'entry_price': s.entry_price,
                'stop_loss': s.stop_loss,
                'take_profit': s.take_profit,
                'risk_reward_ratio': s.risk_reward_ratio,
                'notes': '; '.join(s.notes),
            })
        
        return pd.DataFrame(data)
