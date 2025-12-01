"""
Institutional Trading Strategies Module.

Strategies used by top hedge funds and quant firms:
- Momentum (trend following)
- Mean Reversion
- Statistical Arbitrage
- Factor-based
- Risk Parity
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class StrategyType(Enum):
    """Types of trading strategies."""
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    BREAKOUT = "breakout"
    PAIRS_TRADING = "pairs_trading"
    FACTOR_BASED = "factor_based"
    REGIME_SWITCHING = "regime_switching"


@dataclass
class StrategySignal:
    """Signal from a strategy with metadata."""
    strategy: StrategyType
    direction: int  # 1 = long, -1 = short, 0 = neutral
    strength: float  # 0 to 1
    confidence: float  # 0 to 1
    entry_price: float
    stop_loss: float
    take_profit: float
    reasoning: List[str]


class MomentumStrategy:
    """
    Institutional Momentum Strategy.
    
    Based on academic research:
    - Jegadeesh & Titman (1993): 12-month momentum
    - Carhart (1997): Momentum factor
    - Moskowitz, Ooi & Pedersen (2012): Time-series momentum
    
    Key insights:
    - Skip most recent month (short-term reversal)
    - Combine multiple timeframes
    - Account for volatility
    - Avoid crowded trades
    """
    
    def __init__(
        self,
        lookback_periods: List[int] = None,
        holding_period: int = 21,
        vol_target: float = 0.15,
    ):
        self.lookback_periods = lookback_periods or [21, 63, 126, 252]
        self.holding_period = holding_period
        self.vol_target = vol_target
    
    def generate_signal(
        self,
        df: pd.DataFrame,
        current_idx: int = -1,
    ) -> StrategySignal:
        """Generate momentum-based signal."""
        
        if current_idx == -1:
            current_idx = len(df) - 1
        
        close = df['close']
        current_price = close.iloc[current_idx]
        
        reasoning = []
        momentum_scores = []
        
        # Multi-timeframe momentum
        for period in self.lookback_periods:
            if current_idx < period:
                continue
            
            # Classic momentum (skip most recent week for reversal effect)
            skip_period = 5
            mom = (close.iloc[current_idx - skip_period] / 
                   close.iloc[current_idx - period] - 1)
            
            momentum_scores.append(mom)
            
            if mom > 0.1:
                reasoning.append(f"Strong bullish momentum ({period}d): {mom:.1%}")
            elif mom < -0.1:
                reasoning.append(f"Strong bearish momentum ({period}d): {mom:.1%}")
        
        if not momentum_scores:
            return self._neutral_signal(current_price, ["Insufficient data"])
        
        # Composite momentum score
        avg_momentum = np.mean(momentum_scores)
        
        # Volatility adjustment
        returns = close.pct_change()
        realized_vol = returns.iloc[max(0, current_idx-63):current_idx].std() * np.sqrt(252)
        
        if realized_vol > 0:
            vol_scalar = self.vol_target / realized_vol
            vol_scalar = np.clip(vol_scalar, 0.5, 2.0)  # Limit scaling
        else:
            vol_scalar = 1.0
        
        # Trend strength (ADX proxy)
        if 'adx' in df.columns:
            adx = df['adx'].iloc[current_idx]
            if adx > 25:
                reasoning.append(f"Strong trend (ADX={adx:.0f})")
            elif adx < 15:
                reasoning.append(f"Weak trend (ADX={adx:.0f}), reducing position")
                vol_scalar *= 0.5
        
        # Final direction and strength
        if avg_momentum > 0.02:
            direction = 1
            strength = min(abs(avg_momentum) * 10, 1.0) * vol_scalar
        elif avg_momentum < -0.02:
            direction = -1
            strength = min(abs(avg_momentum) * 10, 1.0) * vol_scalar
        else:
            direction = 0
            strength = 0
        
        # Risk management
        atr = self._calculate_atr(df, current_idx)
        stop_loss = current_price - (direction * 2 * atr)
        take_profit = current_price + (direction * 4 * atr)
        
        # Confidence based on momentum consistency
        momentum_agreement = sum(1 for m in momentum_scores if np.sign(m) == np.sign(avg_momentum))
        confidence = momentum_agreement / len(momentum_scores)
        
        return StrategySignal(
            strategy=StrategyType.MOMENTUM,
            direction=direction,
            strength=strength,
            confidence=confidence,
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            reasoning=reasoning,
        )
    
    def _calculate_atr(self, df: pd.DataFrame, idx: int, period: int = 14) -> float:
        """Calculate Average True Range."""
        if idx < period:
            return df['close'].iloc[idx] * 0.02  # Default 2%
        
        high = df['high'].iloc[max(0, idx-period):idx]
        low = df['low'].iloc[max(0, idx-period):idx]
        close = df['close'].iloc[max(0, idx-period):idx]
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.mean()
    
    def _neutral_signal(self, price: float, reasoning: List[str]) -> StrategySignal:
        """Return neutral signal."""
        return StrategySignal(
            strategy=StrategyType.MOMENTUM,
            direction=0,
            strength=0,
            confidence=0.5,
            entry_price=price,
            stop_loss=price,
            take_profit=price,
            reasoning=reasoning,
        )


class MeanReversionStrategy:
    """
    Institutional Mean Reversion Strategy.
    
    Based on:
    - Statistical arbitrage principles
    - Ornstein-Uhlenbeck process
    - Bollinger Band mean reversion
    - RSI oversold/overbought
    
    Key insights:
    - Works best in range-bound markets
    - Requires regime detection
    - Position sizing based on deviation
    """
    
    def __init__(
        self,
        zscore_entry: float = 2.0,
        zscore_exit: float = 0.5,
        max_holding_period: int = 20,
        lookback: int = 50,
    ):
        self.zscore_entry = zscore_entry
        self.zscore_exit = zscore_exit
        self.max_holding_period = max_holding_period
        self.lookback = lookback
    
    def generate_signal(
        self,
        df: pd.DataFrame,
        current_idx: int = -1,
    ) -> StrategySignal:
        """Generate mean reversion signal."""
        
        if current_idx == -1:
            current_idx = len(df) - 1
        
        close = df['close']
        current_price = close.iloc[current_idx]
        
        if current_idx < self.lookback:
            return self._neutral_signal(current_price, ["Insufficient data"])
        
        reasoning = []
        
        # Calculate z-score
        window_data = close.iloc[current_idx - self.lookback:current_idx + 1]
        mean = window_data.mean()
        std = window_data.std()
        
        if std == 0:
            return self._neutral_signal(current_price, ["Zero volatility"])
        
        zscore = (current_price - mean) / std
        reasoning.append(f"Z-score: {zscore:.2f}")
        
        # Check if we're in a mean-reverting regime
        # Use Hurst exponent proxy
        hurst = self._estimate_hurst(window_data)
        reasoning.append(f"Hurst: {hurst:.2f} ({'Mean reverting' if hurst < 0.5 else 'Trending'})")
        
        if hurst > 0.6:
            # Market is trending, not good for mean reversion
            return self._neutral_signal(current_price, 
                                        reasoning + ["Market trending - skip mean reversion"])
        
        # RSI confirmation
        rsi = self._calculate_rsi(close.iloc[:current_idx + 1])
        reasoning.append(f"RSI: {rsi:.0f}")
        
        # Generate signal
        if zscore < -self.zscore_entry and rsi < 30:
            direction = 1  # Buy (price below mean, expect rise)
            strength = min(abs(zscore) / 3, 1.0)
            reasoning.append("OVERSOLD - Buy signal")
        elif zscore > self.zscore_entry and rsi > 70:
            direction = -1  # Sell (price above mean, expect fall)
            strength = min(abs(zscore) / 3, 1.0)
            reasoning.append("OVERBOUGHT - Sell signal")
        else:
            direction = 0
            strength = 0
        
        # Confidence based on multiple confirmations
        confirmations = 0
        if abs(zscore) > self.zscore_entry:
            confirmations += 1
        if (direction == 1 and rsi < 30) or (direction == -1 and rsi > 70):
            confirmations += 1
        if hurst < 0.5:
            confirmations += 1
        
        confidence = confirmations / 3
        
        # Risk management
        stop_loss = current_price - (direction * std * 3)
        take_profit = mean  # Target is the mean
        
        return StrategySignal(
            strategy=StrategyType.MEAN_REVERSION,
            direction=direction,
            strength=strength,
            confidence=confidence,
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            reasoning=reasoning,
        )
    
    def _estimate_hurst(self, series: pd.Series) -> float:
        """Estimate Hurst exponent."""
        n = len(series)
        if n < 20:
            return 0.5
        
        mean = series.mean()
        std = series.std()
        
        if std == 0:
            return 0.5
        
        cumdev = (series - mean).cumsum()
        R = cumdev.max() - cumdev.min()
        
        if R == 0:
            return 0.5
        
        rs = R / std
        H = np.log(rs) / np.log(n)
        
        return np.clip(H, 0, 1)
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI."""
        delta = prices.diff()
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = pd.Series(gain).rolling(period).mean().iloc[-1]
        avg_loss = pd.Series(loss).rolling(period).mean().iloc[-1]
        
        if avg_loss == 0:
            return 100
        
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    
    def _neutral_signal(self, price: float, reasoning: List[str]) -> StrategySignal:
        """Return neutral signal."""
        return StrategySignal(
            strategy=StrategyType.MEAN_REVERSION,
            direction=0,
            strength=0,
            confidence=0.5,
            entry_price=price,
            stop_loss=price,
            take_profit=price,
            reasoning=reasoning,
        )


class BreakoutStrategy:
    """
    Institutional Breakout Strategy.
    
    Based on:
    - Donchian channels
    - Volatility breakouts
    - Volume confirmation
    
    Key insights:
    - Breakouts work best after consolidation
    - Volume confirmation is critical
    - False breakouts are common - need filters
    """
    
    def __init__(
        self,
        lookback: int = 20,
        atr_multiplier: float = 2.0,
        volume_threshold: float = 1.5,
    ):
        self.lookback = lookback
        self.atr_multiplier = atr_multiplier
        self.volume_threshold = volume_threshold
    
    def generate_signal(
        self,
        df: pd.DataFrame,
        current_idx: int = -1,
    ) -> StrategySignal:
        """Generate breakout signal."""
        
        if current_idx == -1:
            current_idx = len(df) - 1
        
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']
        current_price = close.iloc[current_idx]
        
        if current_idx < self.lookback:
            return self._neutral_signal(current_price, ["Insufficient data"])
        
        reasoning = []
        
        # Calculate channel boundaries
        window = slice(current_idx - self.lookback, current_idx)
        channel_high = high.iloc[window].max()
        channel_low = low.iloc[window].min()
        
        reasoning.append(f"Channel: [{channel_low:.2f}, {channel_high:.2f}]")
        
        # Check for breakout
        breakout_up = current_price > channel_high
        breakout_down = current_price < channel_low
        
        # Volume confirmation
        current_volume = volume.iloc[current_idx]
        avg_volume = volume.iloc[window].mean()
        volume_ratio = current_volume / avg_volume
        
        volume_confirmed = volume_ratio > self.volume_threshold
        reasoning.append(f"Volume ratio: {volume_ratio:.1f}x")
        
        # Volatility contraction before breakout (squeeze)
        recent_atr = self._calculate_atr(df, current_idx, 10)
        longer_atr = self._calculate_atr(df, current_idx, 20)
        
        squeeze = recent_atr < longer_atr * 0.8
        if squeeze:
            reasoning.append("Volatility squeeze detected - high quality breakout")
        
        # Generate signal
        if breakout_up and volume_confirmed:
            direction = 1
            strength = 0.8 if squeeze else 0.6
            reasoning.append("BULLISH BREAKOUT")
        elif breakout_down and volume_confirmed:
            direction = -1
            strength = 0.8 if squeeze else 0.6
            reasoning.append("BEARISH BREAKOUT")
        elif breakout_up or breakout_down:
            # Breakout without volume - lower confidence
            direction = 1 if breakout_up else -1
            strength = 0.3
            reasoning.append("Breakout without volume confirmation - LOW confidence")
        else:
            direction = 0
            strength = 0
        
        # Confidence
        confidence = 0.0
        if direction != 0:
            if volume_confirmed:
                confidence += 0.4
            if squeeze:
                confidence += 0.3
            confidence += 0.3  # Base confidence for breakout
        
        # Risk management
        atr = self._calculate_atr(df, current_idx)
        stop_loss = current_price - (direction * self.atr_multiplier * atr)
        take_profit = current_price + (direction * self.atr_multiplier * 2 * atr)
        
        return StrategySignal(
            strategy=StrategyType.BREAKOUT,
            direction=direction,
            strength=strength,
            confidence=confidence,
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            reasoning=reasoning,
        )
    
    def _calculate_atr(self, df: pd.DataFrame, idx: int, period: int = 14) -> float:
        """Calculate ATR."""
        if idx < period:
            return df['close'].iloc[idx] * 0.02
        
        high = df['high'].iloc[idx - period:idx]
        low = df['low'].iloc[idx - period:idx]
        close = df['close'].iloc[idx - period:idx]
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.mean()
    
    def _neutral_signal(self, price: float, reasoning: List[str]) -> StrategySignal:
        """Return neutral signal."""
        return StrategySignal(
            strategy=StrategyType.BREAKOUT,
            direction=0,
            strength=0,
            confidence=0.5,
            entry_price=price,
            stop_loss=price,
            take_profit=price,
            reasoning=reasoning,
        )


class RegimeSwitchingStrategy:
    """
    Regime-Switching Meta-Strategy.
    
    This is what sophisticated quant funds do:
    - Detect the current market regime
    - Apply the appropriate sub-strategy
    - Dynamically adjust based on conditions
    
    Regimes:
    - Bull trend → Momentum strategy
    - Bear trend → Short momentum or cash
    - Mean reverting → Mean reversion strategy
    - High volatility → Reduce exposure
    - Low volatility → Breakout strategy
    """
    
    def __init__(self):
        self.momentum_strategy = MomentumStrategy()
        self.mean_reversion_strategy = MeanReversionStrategy()
        self.breakout_strategy = BreakoutStrategy()
        
        self.current_regime = None
    
    def detect_regime(self, df: pd.DataFrame, current_idx: int = -1) -> str:
        """
        Detect current market regime.
        
        Returns one of: 'bull_trend', 'bear_trend', 'mean_reverting', 
                       'high_volatility', 'low_volatility'
        """
        if current_idx == -1:
            current_idx = len(df) - 1
        
        close = df['close']
        returns = close.pct_change()
        
        if current_idx < 100:
            return 'unknown'
        
        # Trend detection
        sma_50 = close.iloc[current_idx - 50:current_idx].mean()
        sma_200 = close.iloc[current_idx - 200:current_idx].mean() if current_idx >= 200 else sma_50
        current_price = close.iloc[current_idx]
        
        # Volatility regime
        recent_vol = returns.iloc[current_idx - 20:current_idx].std() * np.sqrt(252)
        historical_vol = returns.iloc[current_idx - 252:current_idx].std() * np.sqrt(252) if current_idx >= 252 else recent_vol
        
        vol_percentile = (recent_vol - historical_vol * 0.5) / (historical_vol * 0.5 + 0.01)
        
        # Hurst exponent for mean reversion detection
        hurst = self._estimate_hurst(close.iloc[current_idx - 100:current_idx])
        
        # Regime classification
        if vol_percentile > 0.5:
            return 'high_volatility'
        elif vol_percentile < -0.3:
            return 'low_volatility'
        elif hurst < 0.45:
            return 'mean_reverting'
        elif current_price > sma_50 > sma_200:
            return 'bull_trend'
        elif current_price < sma_50 < sma_200:
            return 'bear_trend'
        else:
            return 'neutral'
    
    def generate_signal(
        self,
        df: pd.DataFrame,
        current_idx: int = -1,
    ) -> StrategySignal:
        """
        Generate signal based on detected regime.
        """
        regime = self.detect_regime(df, current_idx)
        self.current_regime = regime
        
        # Select strategy based on regime
        if regime == 'bull_trend':
            signal = self.momentum_strategy.generate_signal(df, current_idx)
            signal.reasoning.insert(0, f"Regime: BULL TREND → Using Momentum Strategy")
            
        elif regime == 'bear_trend':
            signal = self.momentum_strategy.generate_signal(df, current_idx)
            # In bear trend, prefer short signals
            if signal.direction == 0:
                signal.direction = -1
                signal.strength = 0.3
                signal.reasoning.insert(0, f"Regime: BEAR TREND → Defensive positioning")
            else:
                signal.reasoning.insert(0, f"Regime: BEAR TREND → Using Momentum Strategy")
                
        elif regime == 'mean_reverting':
            signal = self.mean_reversion_strategy.generate_signal(df, current_idx)
            signal.reasoning.insert(0, f"Regime: MEAN REVERTING → Using Mean Reversion Strategy")
            
        elif regime == 'high_volatility':
            # Reduce all positions in high volatility
            signal = self.momentum_strategy.generate_signal(df, current_idx)
            signal.strength *= 0.3  # Reduce position size
            signal.reasoning.insert(0, f"Regime: HIGH VOLATILITY → Reduced exposure (30%)")
            
        elif regime == 'low_volatility':
            signal = self.breakout_strategy.generate_signal(df, current_idx)
            signal.reasoning.insert(0, f"Regime: LOW VOLATILITY → Using Breakout Strategy")
            
        else:
            # Neutral - no strong signal
            close = df['close']
            current_price = close.iloc[current_idx] if current_idx != -1 else close.iloc[-1]
            signal = StrategySignal(
                strategy=StrategyType.REGIME_SWITCHING,
                direction=0,
                strength=0,
                confidence=0.3,
                entry_price=current_price,
                stop_loss=current_price,
                take_profit=current_price,
                reasoning=[f"Regime: NEUTRAL → No clear signal"],
            )
        
        signal.strategy = StrategyType.REGIME_SWITCHING
        return signal
    
    def _estimate_hurst(self, series: pd.Series) -> float:
        """Estimate Hurst exponent."""
        n = len(series)
        if n < 20:
            return 0.5
        
        mean = series.mean()
        std = series.std()
        
        if std == 0:
            return 0.5
        
        cumdev = (series - mean).cumsum()
        R = cumdev.max() - cumdev.min()
        
        if R == 0:
            return 0.5
        
        rs = R / std
        H = np.log(rs) / np.log(n)
        
        return np.clip(H, 0, 1)


class StrategyEnsemble:
    """
    Ensemble of multiple strategies with dynamic weighting.
    
    This is how top quant funds combine multiple strategies:
    - Each strategy votes independently
    - Weights adjusted based on recent performance
    - Risk parity allocation
    """
    
    def __init__(self):
        self.strategies = {
            'momentum': MomentumStrategy(),
            'mean_reversion': MeanReversionStrategy(),
            'breakout': BreakoutStrategy(),
            'regime_switching': RegimeSwitchingStrategy(),
        }
        
        # Performance tracking for dynamic weighting
        self.strategy_performance: Dict[str, List[float]] = {
            name: [] for name in self.strategies
        }
        
        # Default weights (equal)
        self.weights = {name: 0.25 for name in self.strategies}
    
    def generate_ensemble_signal(
        self,
        df: pd.DataFrame,
        current_idx: int = -1,
    ) -> Dict[str, Any]:
        """
        Generate ensemble signal from all strategies.
        """
        signals = {}
        
        # Get signal from each strategy
        for name, strategy in self.strategies.items():
            try:
                signals[name] = strategy.generate_signal(df, current_idx)
            except Exception as e:
                logger.warning(f"Strategy {name} failed: {e}")
                continue
        
        if not signals:
            return {'direction': 0, 'strength': 0, 'confidence': 0, 'reasoning': []}
        
        # Weighted voting
        total_direction = 0
        total_strength = 0
        total_confidence = 0
        total_weight = 0
        all_reasoning = []
        
        for name, signal in signals.items():
            weight = self.weights[name]
            total_direction += signal.direction * weight * signal.confidence
            total_strength += signal.strength * weight
            total_confidence += signal.confidence * weight
            total_weight += weight
            
            all_reasoning.append(f"[{name}] {signal.direction:+d} @ {signal.confidence:.0%}")
        
        if total_weight > 0:
            final_direction = np.sign(total_direction / total_weight)
            final_strength = total_strength / total_weight
            final_confidence = total_confidence / total_weight
        else:
            final_direction = 0
            final_strength = 0
            final_confidence = 0
        
        # Get price for risk levels
        close = df['close']
        current_price = close.iloc[current_idx] if current_idx != -1 else close.iloc[-1]
        
        # Use regime switching strategy's levels as base
        regime_signal = signals.get('regime_switching')
        if regime_signal:
            stop_loss = regime_signal.stop_loss
            take_profit = regime_signal.take_profit
        else:
            atr_proxy = close.pct_change().rolling(20).std().iloc[-1] * current_price * 2
            stop_loss = current_price - final_direction * atr_proxy
            take_profit = current_price + final_direction * atr_proxy * 2
        
        return {
            'direction': int(final_direction),
            'strength': final_strength,
            'confidence': final_confidence,
            'entry_price': current_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'reasoning': all_reasoning,
            'individual_signals': signals,
        }
    
    def update_weights(self, strategy_name: str, pnl: float):
        """
        Update strategy weights based on performance.
        
        Uses exponential weighting of recent performance.
        """
        self.strategy_performance[strategy_name].append(pnl)
        
        # Keep last 50 trades
        if len(self.strategy_performance[strategy_name]) > 50:
            self.strategy_performance[strategy_name] = \
                self.strategy_performance[strategy_name][-50:]
        
        # Recalculate weights based on Sharpe ratio
        sharpe_ratios = {}
        for name, returns in self.strategy_performance.items():
            if len(returns) < 10:
                sharpe_ratios[name] = 1.0  # Default
            else:
                mean_ret = np.mean(returns)
                std_ret = np.std(returns)
                sharpe_ratios[name] = mean_ret / (std_ret + 0.01)
        
        # Normalize to weights (softmax with temperature)
        temperature = 0.5
        exp_sharpes = {k: np.exp(v / temperature) for k, v in sharpe_ratios.items()}
        total = sum(exp_sharpes.values())
        
        self.weights = {k: v / total for k, v in exp_sharpes.items()}
        
        logger.info(f"Updated strategy weights: {self.weights}")
