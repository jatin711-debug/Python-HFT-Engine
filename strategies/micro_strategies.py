"""
Micro Trading Strategies for 1-5 Second Candles.

Strategies optimized for ultra-short-term crypto trading on Binance.
Each strategy is designed for quick entries and exits with tight risk management.

Strategies:
1. Momentum Burst - Ride short-term price explosions
2. Micro Mean Reversion - Fade overextended moves
3. Volatility Breakout - Trade squeeze expansions
4. Order Flow Edge - Trade based on buy/sell imbalance

Risk Parameters (Aggressive Mode, $500 capital):
- Max position: 20% of capital ($100)
- Stop loss: 0.1-0.2%
- Take profit: 0.2-0.5%
- Max trades per minute: 5

Usage:
    from strategies.micro_strategies import MicroStrategyEnsemble
    
    ensemble = MicroStrategyEnsemble(capital=500, aggressive=True)
    signal = ensemble.generate_signal(df, sentiment)
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

class StrategyType(Enum):
    """Types of micro strategies."""
    MOMENTUM_BURST = "momentum_burst"
    MEAN_REVERSION = "mean_reversion"
    VOLATILITY_BREAKOUT = "volatility_breakout"
    ORDER_FLOW = "order_flow"


@dataclass
class MicroSignal:
    """Signal from a micro strategy."""
    strategy: StrategyType
    direction: int  # 1 = long, -1 = short, 0 = neutral
    strength: float  # 0 to 1
    confidence: float  # 0 to 1
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size_pct: float  # % of capital
    reasoning: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def risk_reward(self) -> float:
        """Calculate risk/reward ratio."""
        if self.direction == 0:
            return 0
        
        if self.direction == 1:
            risk = self.entry_price - self.stop_loss
            reward = self.take_profit - self.entry_price
        else:
            risk = self.stop_loss - self.entry_price
            reward = self.entry_price - self.take_profit
        
        return reward / risk if risk > 0 else 0


@dataclass
class MicroConfig:
    """Configuration for micro strategies."""
    
    # Capital and position sizing
    capital: float = 500.0
    max_position_pct: float = 0.20  # 20% max per trade
    min_position_pct: float = 0.05  # 5% min per trade
    
    # Risk management
    default_stop_loss_pct: float = 0.002  # 0.2%
    default_take_profit_pct: float = 0.005  # 0.5%
    min_risk_reward: float = 2.0
    
    # Trade frequency limits
    max_trades_per_minute: int = 5
    cooldown_seconds: float = 10.0
    
    # Strategy-specific thresholds
    momentum_threshold: float = 0.003  # 0.3% move to trigger
    mean_reversion_zscore: float = 2.0  # Z-score for entry
    volatility_squeeze_mult: float = 0.5  # Squeeze detection
    volume_surge_mult: float = 2.0  # Volume spike detection
    
    # Aggressive mode multipliers
    aggressive: bool = True


# =============================================================================
# STRATEGY 1: MOMENTUM BURST
# =============================================================================

class MomentumBurstStrategy:
    """
    Trade sudden price explosions with volume confirmation.
    
    Entry:
    - Price moves > threshold in last N candles
    - Volume surge (> 2x average)
    - Momentum agreement across timeframes
    
    Exit:
    - Stop loss: 0.1-0.2%
    - Take profit: 0.3-0.5%
    - Time-based: Exit after 30-60 seconds if no move
    """
    
    def __init__(self, config: MicroConfig):
        self.config = config
        self.name = "MomentumBurst"
    
    def generate_signal(
        self,
        df: pd.DataFrame,
        current_idx: int = -1,
    ) -> MicroSignal:
        """Generate momentum burst signal."""
        if len(df) < 30:
            return self._neutral_signal(df['close'].iloc[current_idx])
        
        # Get recent data
        close = df['close'].values
        volume = df['volume'].values
        high = df['high'].values
        low = df['low'].values
        
        idx = current_idx if current_idx != -1 else len(df) - 1
        current_price = close[idx]
        
        reasoning = []
        
        # 1. Check for price explosion (last 5 candles)
        if idx >= 5:
            recent_return = (close[idx] - close[idx-5]) / close[idx-5]
        else:
            return self._neutral_signal(current_price)
        
        # 2. Check volume surge
        if idx >= 30:
            vol_avg = np.mean(volume[idx-30:idx])
            vol_ratio = volume[idx] / vol_avg if vol_avg > 0 else 1
        else:
            vol_ratio = 1
        
        # 3. Check momentum consistency
        ret_1 = (close[idx] - close[idx-1]) / close[idx-1] if idx >= 1 else 0
        ret_3 = (close[idx] - close[idx-3]) / close[idx-3] if idx >= 3 else 0
        ret_5 = recent_return
        
        momentum_signs = np.sign([ret_1, ret_3, ret_5])
        momentum_agreement = np.mean(momentum_signs)
        
        # Entry conditions
        threshold = self.config.momentum_threshold
        vol_threshold = self.config.volume_surge_mult
        
        # Long signal
        if (recent_return > threshold and 
            vol_ratio > vol_threshold and 
            momentum_agreement > 0.5):
            
            direction = 1
            strength = min(1.0, recent_return / (threshold * 2))
            confidence = min(1.0, vol_ratio / (vol_threshold * 2))
            
            reasoning.append(f"Price burst: {recent_return*100:.2f}%")
            reasoning.append(f"Volume surge: {vol_ratio:.1f}x")
            reasoning.append(f"Momentum aligned: {momentum_agreement:.2f}")
        
        # Short signal
        elif (recent_return < -threshold and 
              vol_ratio > vol_threshold and 
              momentum_agreement < -0.5):
            
            direction = -1
            strength = min(1.0, abs(recent_return) / (threshold * 2))
            confidence = min(1.0, vol_ratio / (vol_threshold * 2))
            
            reasoning.append(f"Price crash: {recent_return*100:.2f}%")
            reasoning.append(f"Volume surge: {vol_ratio:.1f}x")
            reasoning.append(f"Momentum aligned: {momentum_agreement:.2f}")
        
        else:
            return self._neutral_signal(current_price)
        
        # Calculate stops
        atr = np.mean(high[idx-10:idx] - low[idx-10:idx]) if idx >= 10 else current_price * 0.002
        
        if direction == 1:
            stop_loss = current_price - (atr * 1.5)
            take_profit = current_price + (atr * 3.0)  # 2:1 risk/reward
        else:
            stop_loss = current_price + (atr * 1.5)
            take_profit = current_price - (atr * 3.0)
        
        # Position sizing based on confidence
        base_size = self.config.max_position_pct if self.config.aggressive else self.config.max_position_pct * 0.5
        position_size = base_size * strength * confidence
        
        return MicroSignal(
            strategy=StrategyType.MOMENTUM_BURST,
            direction=direction,
            strength=strength,
            confidence=confidence,
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size_pct=position_size,
            reasoning=reasoning,
            metadata={
                'recent_return': recent_return,
                'vol_ratio': vol_ratio,
                'atr': atr,
            }
        )
    
    def _neutral_signal(self, price: float) -> MicroSignal:
        return MicroSignal(
            strategy=StrategyType.MOMENTUM_BURST,
            direction=0,
            strength=0,
            confidence=0,
            entry_price=price,
            stop_loss=price,
            take_profit=price,
            position_size_pct=0,
        )


# =============================================================================
# STRATEGY 2: MICRO MEAN REVERSION
# =============================================================================

class MicroMeanReversionStrategy:
    """
    Fade overextended moves that are likely to revert.
    
    Entry:
    - Price deviation from VWAP > 2 std
    - Return autocorrelation is negative (mean-reverting regime)
    - No strong volume (smart money not pushing)
    
    Exit:
    - Target: Return to VWAP or 0.5 Z-score
    - Stop: 1.5x entry deviation
    """
    
    def __init__(self, config: MicroConfig):
        self.config = config
        self.name = "MicroMeanReversion"
    
    def generate_signal(
        self,
        df: pd.DataFrame,
        current_idx: int = -1,
    ) -> MicroSignal:
        """Generate mean reversion signal."""
        if len(df) < 60:
            return self._neutral_signal(df['close'].iloc[-1])
        
        close = df['close'].values
        volume = df['volume'].values
        
        idx = current_idx if current_idx != -1 else len(df) - 1
        current_price = close[idx]
        
        reasoning = []
        
        # 1. Calculate VWAP and deviation
        lookback = min(60, idx)
        vwap = np.average(close[idx-lookback:idx], weights=volume[idx-lookback:idx])
        deviation = (current_price - vwap) / vwap
        
        # Z-score of deviation
        deviations = [(close[i] - np.average(close[max(0,i-60):i], weights=volume[max(0,i-60):i])) / 
                     np.average(close[max(0,i-60):i], weights=volume[max(0,i-60):i])
                     for i in range(max(60, idx-30), idx)]
        
        if len(deviations) > 0:
            dev_mean = np.mean(deviations)
            dev_std = np.std(deviations)
            zscore = (deviation - dev_mean) / dev_std if dev_std > 0 else 0
        else:
            zscore = 0
        
        # 2. Check for mean-reverting regime (negative autocorrelation)
        if idx >= 31:
            close_slice = close[idx-30:idx+1]  # Get 31 values to have 30 returns
            if len(close_slice) >= 2:
                returns = np.diff(close_slice) / close_slice[:-1]
            else:
                returns = np.array([0])
        else:
            returns = np.array([0])
        
        if len(returns) > 5:
            lagged = returns[:-1]
            current_ret = returns[1:]
            if len(lagged) == len(current_ret) and len(lagged) > 1:
                autocorr = np.corrcoef(lagged, current_ret)[0, 1]
                if np.isnan(autocorr):
                    autocorr = 0
            else:
                autocorr = 0
        else:
            autocorr = 0
        
        # 3. Check volume (prefer low volume for mean reversion)
        vol_avg = np.mean(volume[idx-30:idx]) if idx >= 30 else volume[idx]
        vol_ratio = volume[idx] / vol_avg if vol_avg > 0 else 1
        
        # Entry conditions
        zscore_threshold = self.config.mean_reversion_zscore
        
        # Short signal (price extended above VWAP)
        if (zscore > zscore_threshold and 
            autocorr < 0 and 
            vol_ratio < 1.5):
            
            direction = -1
            strength = min(1.0, abs(zscore) / (zscore_threshold * 2))
            confidence = min(1.0, abs(autocorr) * 2)
            
            reasoning.append(f"Overextended: Z={zscore:.2f}")
            reasoning.append(f"Mean reverting: autocorr={autocorr:.2f}")
            reasoning.append(f"Low volume: {vol_ratio:.1f}x")
        
        # Long signal (price extended below VWAP)
        elif (zscore < -zscore_threshold and 
              autocorr < 0 and 
              vol_ratio < 1.5):
            
            direction = 1
            strength = min(1.0, abs(zscore) / (zscore_threshold * 2))
            confidence = min(1.0, abs(autocorr) * 2)
            
            reasoning.append(f"Oversold: Z={zscore:.2f}")
            reasoning.append(f"Mean reverting: autocorr={autocorr:.2f}")
            reasoning.append(f"Low volume: {vol_ratio:.1f}x")
        
        else:
            return self._neutral_signal(current_price)
        
        # Calculate stops - tighter for mean reversion
        stop_dist = abs(deviation) * 1.5
        target_dist = abs(deviation) * 0.7  # Target 70% reversion
        
        if direction == 1:
            stop_loss = current_price * (1 - stop_dist)
            take_profit = current_price * (1 + target_dist)
        else:
            stop_loss = current_price * (1 + stop_dist)
            take_profit = current_price * (1 - target_dist)
        
        # Smaller position for mean reversion (counter-trend)
        base_size = self.config.max_position_pct * 0.6
        position_size = base_size * strength * confidence
        
        return MicroSignal(
            strategy=StrategyType.MEAN_REVERSION,
            direction=direction,
            strength=strength,
            confidence=confidence,
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size_pct=position_size,
            reasoning=reasoning,
            metadata={
                'vwap': vwap,
                'zscore': zscore,
                'autocorr': autocorr,
            }
        )
    
    def _neutral_signal(self, price: float) -> MicroSignal:
        return MicroSignal(
            strategy=StrategyType.MEAN_REVERSION,
            direction=0,
            strength=0,
            confidence=0,
            entry_price=price,
            stop_loss=price,
            take_profit=price,
            position_size_pct=0,
        )


# =============================================================================
# STRATEGY 3: VOLATILITY BREAKOUT
# =============================================================================

class VolatilityBreakoutStrategy:
    """
    Trade volatility squeeze expansions.
    
    Entry:
    - Volatility squeeze detected (BB width < 50% of average)
    - Squeeze releases with volume
    - Direction confirmed by close position in range
    
    Exit:
    - Trailing stop based on ATR
    - Take profit at 2x ATR
    """
    
    def __init__(self, config: MicroConfig):
        self.config = config
        self.name = "VolatilityBreakout"
    
    def generate_signal(
        self,
        df: pd.DataFrame,
        current_idx: int = -1,
    ) -> MicroSignal:
        """Generate volatility breakout signal."""
        if len(df) < 60:
            return self._neutral_signal(df['close'].iloc[-1])
        
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        volume = df['volume'].values
        
        idx = current_idx if current_idx != -1 else len(df) - 1
        current_price = close[idx]
        
        reasoning = []
        
        # 1. Calculate Bollinger Band width
        lookback = 20
        if idx < lookback:
            return self._neutral_signal(current_price)
        
        sma = np.mean(close[idx-lookback:idx])
        std = np.std(close[idx-lookback:idx])
        bb_width = (2 * std) / sma if sma > 0 else 0
        
        # Average BB width
        bb_widths = []
        for i in range(max(lookback, idx-60), idx):
            s = np.std(close[i-lookback:i])
            m = np.mean(close[i-lookback:i])
            bb_widths.append((2 * s) / m if m > 0 else 0)
        
        avg_bb_width = np.mean(bb_widths) if bb_widths else bb_width
        
        # 2. Detect squeeze (current width < 50% of average)
        squeeze_mult = self.config.volatility_squeeze_mult
        is_squeeze = bb_width < avg_bb_width * squeeze_mult
        was_squeeze = False
        
        if idx >= 1:
            prev_std = np.std(close[idx-lookback-1:idx-1])
            prev_sma = np.mean(close[idx-lookback-1:idx-1])
            prev_width = (2 * prev_std) / prev_sma if prev_sma > 0 else 0
            was_squeeze = prev_width < avg_bb_width * squeeze_mult
        
        # 3. Squeeze release detection
        squeeze_release = was_squeeze and not is_squeeze
        
        if not squeeze_release:
            # Also check for expansion with strong momentum
            expansion = bb_width > avg_bb_width * 1.2
            strong_move = abs(close[idx] - close[idx-1]) > std
            if not (expansion and strong_move):
                return self._neutral_signal(current_price)
        
        # 4. Determine direction from close position
        upper_band = sma + 2 * std
        lower_band = sma - 2 * std
        band_range = upper_band - lower_band
        
        position_in_band = (current_price - lower_band) / band_range if band_range > 0 else 0.5
        
        # 5. Volume confirmation
        vol_avg = np.mean(volume[idx-30:idx]) if idx >= 30 else volume[idx]
        vol_ratio = volume[idx] / vol_avg if vol_avg > 0 else 1
        
        # Entry logic
        if position_in_band > 0.7 and vol_ratio > 1.3:
            # Breakout to upside
            direction = 1
            strength = min(1.0, (position_in_band - 0.5) * 2)
            confidence = min(1.0, vol_ratio / 2)
            
            reasoning.append("Squeeze release UP")
            reasoning.append(f"Band position: {position_in_band:.2f}")
            reasoning.append(f"Volume: {vol_ratio:.1f}x")
        
        elif position_in_band < 0.3 and vol_ratio > 1.3:
            # Breakout to downside
            direction = -1
            strength = min(1.0, (0.5 - position_in_band) * 2)
            confidence = min(1.0, vol_ratio / 2)
            
            reasoning.append("Squeeze release DOWN")
            reasoning.append(f"Band position: {position_in_band:.2f}")
            reasoning.append(f"Volume: {vol_ratio:.1f}x")
        
        else:
            return self._neutral_signal(current_price)
        
        # Calculate stops using ATR
        atr = np.mean(high[idx-14:idx] - low[idx-14:idx]) if idx >= 14 else std
        
        if direction == 1:
            stop_loss = current_price - (atr * 1.5)
            take_profit = current_price + (atr * 3.0)
        else:
            stop_loss = current_price + (atr * 1.5)
            take_profit = current_price - (atr * 3.0)
        
        # Position sizing - larger for breakouts with momentum
        base_size = self.config.max_position_pct
        position_size = base_size * strength * confidence
        
        return MicroSignal(
            strategy=StrategyType.VOLATILITY_BREAKOUT,
            direction=direction,
            strength=strength,
            confidence=confidence,
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size_pct=position_size,
            reasoning=reasoning,
            metadata={
                'bb_width': bb_width,
                'avg_width': avg_bb_width,
                'squeeze_release': squeeze_release,
                'atr': atr,
            }
        )
    
    def _neutral_signal(self, price: float) -> MicroSignal:
        return MicroSignal(
            strategy=StrategyType.VOLATILITY_BREAKOUT,
            direction=0,
            strength=0,
            confidence=0,
            entry_price=price,
            stop_loss=price,
            take_profit=price,
            position_size_pct=0,
        )


# =============================================================================
# STRATEGY 4: ORDER FLOW EDGE
# =============================================================================

class OrderFlowEdgeStrategy:
    """
    Trade based on buy/sell imbalance from trade flow.
    
    Entry:
    - Strong volume imbalance (> 60% one side)
    - Whale activity confirmation
    - Momentum alignment
    
    Exit:
    - Quick scalp: 0.1-0.2% target
    - Tight stop: 0.1%
    """
    
    def __init__(self, config: MicroConfig):
        self.config = config
        self.name = "OrderFlowEdge"
    
    def generate_signal(
        self,
        df: pd.DataFrame,
        sentiment: Optional[Any] = None,  # CryptoSentiment
        current_idx: int = -1,
    ) -> MicroSignal:
        """Generate order flow signal."""
        idx = current_idx if current_idx != -1 else len(df) - 1
        current_price = df['close'].iloc[idx]
        
        if sentiment is None:
            return self._neutral_signal(current_price)
        
        reasoning = []
        
        # Get sentiment values
        vol_imbalance = getattr(sentiment, 'volume_imbalance', 0)
        trade_imbalance = getattr(sentiment, 'trade_imbalance', 0)
        whale_signal = getattr(sentiment, 'whale_signal', 0)
        
        # Combined flow signal
        flow_signal = 0.5 * vol_imbalance + 0.3 * trade_imbalance + 0.2 * whale_signal
        
        # Check momentum alignment
        if len(df) >= 5:
            recent_return = (df['close'].iloc[idx] - df['close'].iloc[idx-5]) / df['close'].iloc[idx-5]
        else:
            recent_return = 0
        
        momentum_aligned = np.sign(flow_signal) == np.sign(recent_return)
        
        # Entry conditions
        imbalance_threshold = 0.3  # 30% imbalance
        
        # Long signal
        if (flow_signal > imbalance_threshold and momentum_aligned):
            direction = 1
            strength = min(1.0, abs(flow_signal) / 0.6)
            confidence = 0.7 if whale_signal > 0.2 else 0.5
            
            reasoning.append(f"Buy flow: {vol_imbalance:.2f}")
            reasoning.append(f"Whale buying: {whale_signal:.2f}")
            reasoning.append("Momentum aligned")
        
        # Short signal
        elif (flow_signal < -imbalance_threshold and momentum_aligned):
            direction = -1
            strength = min(1.0, abs(flow_signal) / 0.6)
            confidence = 0.7 if whale_signal < -0.2 else 0.5
            
            reasoning.append(f"Sell flow: {vol_imbalance:.2f}")
            reasoning.append(f"Whale selling: {whale_signal:.2f}")
            reasoning.append("Momentum aligned")
        
        else:
            return self._neutral_signal(current_price)
        
        # Tight stops for scalping
        stop_pct = 0.001  # 0.1%
        take_pct = 0.002  # 0.2%
        
        if direction == 1:
            stop_loss = current_price * (1 - stop_pct)
            take_profit = current_price * (1 + take_pct)
        else:
            stop_loss = current_price * (1 + stop_pct)
            take_profit = current_price * (1 - take_pct)
        
        # Position sizing - smaller for scalps
        base_size = self.config.max_position_pct * 0.5
        position_size = base_size * strength * confidence
        
        return MicroSignal(
            strategy=StrategyType.ORDER_FLOW,
            direction=direction,
            strength=strength,
            confidence=confidence,
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size_pct=position_size,
            reasoning=reasoning,
            metadata={
                'vol_imbalance': vol_imbalance,
                'whale_signal': whale_signal,
                'flow_signal': flow_signal,
            }
        )
    
    def _neutral_signal(self, price: float) -> MicroSignal:
        return MicroSignal(
            strategy=StrategyType.ORDER_FLOW,
            direction=0,
            strength=0,
            confidence=0,
            entry_price=price,
            stop_loss=price,
            take_profit=price,
            position_size_pct=0,
        )


# =============================================================================
# STRATEGY ENSEMBLE
# =============================================================================

class MicroStrategyEnsemble:
    """
    Ensemble of micro strategies for 1-5 second trading.
    
    Combines signals from all strategies with dynamic weighting
    based on recent performance and market conditions.
    """
    
    # Strategy weights (can be dynamically adjusted)
    DEFAULT_WEIGHTS = {
        StrategyType.MOMENTUM_BURST: 0.30,
        StrategyType.MEAN_REVERSION: 0.20,
        StrategyType.VOLATILITY_BREAKOUT: 0.25,
        StrategyType.ORDER_FLOW: 0.25,
    }
    
    def __init__(
        self,
        capital: float = 500.0,
        aggressive: bool = True,
    ):
        """
        Initialize micro strategy ensemble.
        
        Args:
            capital: Starting capital
            aggressive: Use aggressive settings
        """
        self.config = MicroConfig(
            capital=capital,
            aggressive=aggressive,
            max_position_pct=0.20 if aggressive else 0.10,
        )
        
        # Initialize strategies
        self.momentum = MomentumBurstStrategy(self.config)
        self.mean_reversion = MicroMeanReversionStrategy(self.config)
        self.volatility = VolatilityBreakoutStrategy(self.config)
        self.order_flow = OrderFlowEdgeStrategy(self.config)
        
        self.weights = self.DEFAULT_WEIGHTS.copy()
        
        # Performance tracking
        self._performance: Dict[StrategyType, List[float]] = {
            st: [] for st in StrategyType
        }
        
        logger.info(f"MicroStrategyEnsemble initialized (capital=${capital}, aggressive={aggressive})")
    
    def generate_signal(
        self,
        df: pd.DataFrame,
        sentiment: Optional[Any] = None,
        current_idx: int = -1,
    ) -> MicroSignal:
        """
        Generate ensemble signal from all strategies.
        
        Args:
            df: OHLCV DataFrame
            sentiment: CryptoSentiment for order flow
            current_idx: Index to generate signal for
            
        Returns:
            Best signal or combined signal
        """
        # Collect signals from all strategies
        signals: Dict[StrategyType, MicroSignal] = {}
        
        signals[StrategyType.MOMENTUM_BURST] = self.momentum.generate_signal(df, current_idx)
        signals[StrategyType.MEAN_REVERSION] = self.mean_reversion.generate_signal(df, current_idx)
        signals[StrategyType.VOLATILITY_BREAKOUT] = self.volatility.generate_signal(df, current_idx)
        signals[StrategyType.ORDER_FLOW] = self.order_flow.generate_signal(df, sentiment, current_idx)
        
        # Filter active signals
        active_signals = {
            st: sig for st, sig in signals.items() 
            if sig.direction != 0
        }
        
        if not active_signals:
            # No signals - return neutral
            price = df['close'].iloc[current_idx if current_idx != -1 else -1]
            return MicroSignal(
                strategy=StrategyType.MOMENTUM_BURST,
                direction=0,
                strength=0,
                confidence=0,
                entry_price=price,
                stop_loss=price,
                take_profit=price,
                position_size_pct=0,
                reasoning=["No active signals"],
            )
        
        # Check for agreement
        directions = [sig.direction for sig in active_signals.values()]
        agreement = len(set(directions)) == 1  # All same direction
        
        if agreement:
            # Strong signal - all agree
            return self._combine_signals(active_signals)
        else:
            # Conflicting signals - take strongest
            best_strategy = max(
                active_signals.items(),
                key=lambda x: x[1].strength * x[1].confidence * self.weights[x[0]]
            )
            signal = best_strategy[1]
            signal.reasoning.append("Strongest single signal (conflict)")
            return signal
    
    def _combine_signals(
        self,
        signals: Dict[StrategyType, MicroSignal],
    ) -> MicroSignal:
        """Combine agreeing signals into one."""
        if not signals:
            return self.momentum._neutral_signal(0)
        
        # Use the first signal as base
        base_signal = list(signals.values())[0]
        
        # Calculate weighted averages
        total_weight = sum(self.weights[st] for st in signals.keys())
        
        weighted_strength = sum(
            sig.strength * self.weights[st] 
            for st, sig in signals.items()
        ) / total_weight
        
        weighted_confidence = sum(
            sig.confidence * self.weights[st] 
            for st, sig in signals.items()
        ) / total_weight
        
        # Boost for agreement
        agreement_boost = 1.0 + 0.2 * len(signals)  # Up to 1.8x for all 4
        
        # Combine reasoning
        all_reasons = []
        for st, sig in signals.items():
            all_reasons.extend([f"[{st.value}] {r}" for r in sig.reasoning])
        all_reasons.append(f"Strategy agreement: {len(signals)}/4")
        
        # Use best risk/reward
        best_rr = max(signals.values(), key=lambda s: s.risk_reward)
        
        return MicroSignal(
            strategy=StrategyType.MOMENTUM_BURST,  # Ensemble
            direction=base_signal.direction,
            strength=min(1.0, weighted_strength * agreement_boost),
            confidence=min(1.0, weighted_confidence * agreement_boost),
            entry_price=base_signal.entry_price,
            stop_loss=best_rr.stop_loss,
            take_profit=best_rr.take_profit,
            position_size_pct=min(
                self.config.max_position_pct,
                base_signal.position_size_pct * agreement_boost
            ),
            reasoning=all_reasons,
            metadata={
                'num_signals': len(signals),
                'strategies': [st.value for st in signals.keys()],
            }
        )
    
    def update_performance(
        self,
        strategy_type: StrategyType,
        pnl: float,
    ):
        """Update performance tracking for dynamic weighting."""
        self._performance[strategy_type].append(pnl)
        
        # Keep last 100 trades
        if len(self._performance[strategy_type]) > 100:
            self._performance[strategy_type].pop(0)
        
        # Update weights based on performance
        self._update_weights()
    
    def _update_weights(self):
        """Dynamically adjust weights based on recent performance."""
        # Calculate average PnL for each strategy
        avg_pnl = {}
        for st, pnls in self._performance.items():
            if len(pnls) >= 10:  # Need minimum trades
                avg_pnl[st] = np.mean(pnls)
        
        if not avg_pnl:
            return
        
        # Normalize and update weights
        total = sum(max(0, p) for p in avg_pnl.values())
        if total > 0:
            for st, pnl in avg_pnl.items():
                # Blend with default weights
                performance_weight = max(0, pnl) / total
                self.weights[st] = 0.7 * self.DEFAULT_WEIGHTS[st] + 0.3 * performance_weight


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def create_micro_ensemble(
    capital: float = 500.0,
    aggressive: bool = True,
) -> MicroStrategyEnsemble:
    """Create a micro strategy ensemble."""
    return MicroStrategyEnsemble(capital=capital, aggressive=aggressive)


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
    })
    
    # Create ensemble
    ensemble = MicroStrategyEnsemble(capital=500, aggressive=True)
    
    # Generate signal
    signal = ensemble.generate_signal(df)
    
    print(f"\n=== Micro Strategy Signal ===")
    print(f"Direction: {signal.direction}")
    print(f"Strength: {signal.strength:.2f}")
    print(f"Confidence: {signal.confidence:.2f}")
    print(f"Entry: ${signal.entry_price:.2f}")
    print(f"Stop Loss: ${signal.stop_loss:.2f}")
    print(f"Take Profit: ${signal.take_profit:.2f}")
    print(f"Position Size: {signal.position_size_pct*100:.1f}%")
    print(f"Risk/Reward: {signal.risk_reward:.2f}")
    print(f"\nReasoning:")
    for r in signal.reasoning:
        print(f"  - {r}")
