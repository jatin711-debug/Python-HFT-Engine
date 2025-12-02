"""
Unified Signal Aggregator Module.

Combines signals from multiple strategy types:
1. ML Model predictions
2. Institutional strategies (momentum, mean reversion, breakout)
3. HFT strategies (stat arb, LOB imbalance, market making)

This creates a unified decision-making framework that can:
- Weight signals based on market regime
- Apply appropriate risk management
- Route signals to different execution paths
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import logging

# Local imports
from signals.generator import SignalGenerator, TradingSignal, SignalType, SignalConfig
from strategies.institutional_strategies import (
    StrategyEnsemble as InstitutionalEnsemble,
    StrategySignal,
    StrategyType,
)
from strategies.hft_strategies import (
    HFTStrategyEnsemble,
    StatisticalArbitrageEWLR,
    OrderBookImbalanceStrategy,
    IntelligentMarketMaker,
    HFTSignal,
    LOBSnapshot,
)

logger = logging.getLogger(__name__)


class SignalSource(Enum):
    """Source of the trading signal."""
    ML_MODEL = "ml_model"
    INSTITUTIONAL = "institutional"
    HFT = "hft"
    ENSEMBLE = "ensemble"


class ExecutionPath(Enum):
    """Where to route the signal for execution."""
    STANDARD = "standard"      # Regular order execution
    LOW_LATENCY = "low_latency"  # Fast execution path
    PAPER = "paper"            # Paper trading only


@dataclass
class UnifiedSignal:
    """
    Unified signal combining all strategy inputs.
    
    This is the final output that goes to execution.
    """
    timestamp: datetime
    symbol: str
    
    # Final decision
    direction: int  # 1 = long, -1 = short, 0 = neutral
    strength: float  # 0 to 1
    confidence: float  # 0 to 1
    
    # Execution details
    entry_price: float
    stop_loss: Optional[float]
    take_profit: Optional[float]
    position_size_pct: float  # % of portfolio
    execution_path: ExecutionPath
    
    # Component signals
    ml_signal: Optional[float] = None
    institutional_signal: Optional[Dict] = None
    hft_signal: Optional[HFTSignal] = None
    
    # Metadata
    reasoning: List[str] = field(default_factory=list)
    regime: Optional[str] = None
    agreement_score: float = 0.0  # How much strategies agree
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'symbol': self.symbol,
            'direction': self.direction,
            'strength': self.strength,
            'confidence': self.confidence,
            'entry_price': self.entry_price,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'position_size_pct': self.position_size_pct,
            'execution_path': self.execution_path.value,
            'ml_signal': self.ml_signal,
            'regime': self.regime,
            'agreement_score': self.agreement_score,
            'reasoning': self.reasoning,
        }


@dataclass
class AggregatorConfig:
    """Configuration for signal aggregation."""
    # Weights for different signal sources
    ml_weight: float = 0.35
    institutional_weight: float = 0.35
    hft_weight: float = 0.30
    
    # Confidence thresholds
    min_confidence: float = 0.4
    high_confidence: float = 0.7
    
    # Agreement requirements
    require_agreement: bool = True  # Only trade when strategies agree
    min_agreement_score: float = 0.5
    
    # Position sizing
    base_position_pct: float = 0.05  # 5% base position
    max_position_pct: float = 0.15  # 15% max position
    
    # HFT-specific
    use_hft_for_timing: bool = True  # Use HFT signals for entry timing
    hft_execution_threshold: float = 0.7  # High HFT confidence → low latency
    
    # Risk adjustments
    reduce_in_high_vol: bool = True
    vol_reduction_factor: float = 0.5


class UnifiedSignalAggregator:
    """
    Aggregates signals from all strategy types into unified trading decisions.
    
    This is the central decision-making component that:
    1. Collects signals from ML, institutional, and HFT strategies
    2. Weighs them based on market conditions
    3. Resolves conflicts between strategies
    4. Applies risk management
    5. Routes to appropriate execution path
    
    Example:
        >>> aggregator = UnifiedSignalAggregator(ml_model=model)
        >>> signal = aggregator.generate_unified_signal(df, "AAPL")
        >>> if signal.direction != 0:
        ...     execute_trade(signal)
    """
    
    def __init__(
        self,
        ml_model: Any = None,
        config: AggregatorConfig = None,
        use_hft: bool = True,
    ):
        """
        Initialize the unified signal aggregator.
        
        Args:
            ml_model: Trained ML model for predictions
            config: Aggregation configuration
            use_hft: Whether to include HFT strategies
        """
        self.ml_model = ml_model
        self.config = config or AggregatorConfig()
        self.use_hft = use_hft
        
        # Initialize strategy components
        self.signal_generator = SignalGenerator(ml_model=ml_model)
        self.institutional_ensemble = InstitutionalEnsemble()
        
        if use_hft:
            self.hft_ensemble = HFTStrategyEnsemble()
            self.stat_arb = StatisticalArbitrageEWLR()
            self.lob_strategy = OrderBookImbalanceStrategy()
            self.market_maker = IntelligentMarketMaker()
        else:
            self.hft_ensemble = None
        
        # State tracking
        self.current_regime = None
        self.signal_history: List[UnifiedSignal] = []
    
    def generate_unified_signal(
        self,
        df: pd.DataFrame,
        symbol: str,
        current_idx: int = -1,
        lob_data: Optional[List[LOBSnapshot]] = None,
        component_prices: Optional[pd.DataFrame] = None,
    ) -> UnifiedSignal:
        """
        Generate a unified trading signal by combining all sources.
        
        Args:
            df: OHLCV DataFrame with features
            symbol: Stock symbol
            current_idx: Index to generate signal for (-1 for latest)
            lob_data: Optional LOB snapshots for HFT strategies
            component_prices: Optional component prices for stat arb
            
        Returns:
            UnifiedSignal with final trading decision
        """
        if current_idx == -1:
            current_idx = len(df) - 1
        
        current_price = df['close'].iloc[current_idx]
        timestamp = df.index[current_idx] if hasattr(df.index[current_idx], 'strftime') else datetime.now()
        
        reasoning = []
        
        # 1. Detect market regime
        regime = self._detect_regime(df, current_idx)
        self.current_regime = regime
        reasoning.append(f"Market Regime: {regime}")
        
        # 2. Get ML signal
        ml_signal = self._get_ml_signal(df, current_idx)
        if ml_signal is not None:
            reasoning.append(f"ML Signal: {ml_signal:+.2f}")
        
        # 3. Get institutional strategy signal
        inst_signal = self._get_institutional_signal(df, current_idx)
        if inst_signal:
            reasoning.append(f"Institutional: dir={inst_signal['direction']:+d}, conf={inst_signal['confidence']:.2f}")
        
        # 4. Get HFT signal (if enabled)
        hft_signal = None
        if self.use_hft and self.hft_ensemble:
            hft_signal = self._get_hft_signal(df, current_idx, lob_data, component_prices)
            if hft_signal:
                reasoning.append(f"HFT: dir={hft_signal.direction:+d}, conf={hft_signal.confidence:.2f}")
        
        # 5. Combine signals with regime-adjusted weights
        weights = self._get_regime_adjusted_weights(regime)
        
        combined_direction, combined_strength, combined_confidence = self._combine_signals(
            ml_signal=ml_signal,
            inst_signal=inst_signal,
            hft_signal=hft_signal,
            weights=weights,
        )
        
        # 6. Calculate agreement score
        agreement_score = self._calculate_agreement(ml_signal, inst_signal, hft_signal)
        reasoning.append(f"Strategy Agreement: {agreement_score:.0%}")
        
        # 7. Apply agreement filter
        if self.config.require_agreement and agreement_score < self.config.min_agreement_score:
            reasoning.append(f"⚠️ Low agreement - reducing position")
            combined_strength *= 0.5
            combined_confidence *= 0.7
        
        # 8. Final direction decision
        if combined_confidence < self.config.min_confidence:
            final_direction = 0
            reasoning.append("Confidence below threshold - HOLD")
        else:
            final_direction = int(np.sign(combined_direction)) if abs(combined_direction) > 0.1 else 0
        
        # 9. Position sizing
        position_size = self._calculate_position_size(
            combined_strength,
            combined_confidence,
            regime,
            df,
            current_idx,
        )
        
        # 10. Risk levels
        stop_loss, take_profit = self._calculate_risk_levels(
            current_price,
            final_direction,
            df,
            current_idx,
        )
        
        # 11. Determine execution path
        execution_path = self._determine_execution_path(
            hft_signal,
            combined_confidence,
            regime,
        )
        
        # Create unified signal
        unified_signal = UnifiedSignal(
            timestamp=timestamp,
            symbol=symbol,
            direction=final_direction,
            strength=combined_strength,
            confidence=combined_confidence,
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size_pct=position_size,
            execution_path=execution_path,
            ml_signal=ml_signal,
            institutional_signal=inst_signal,
            hft_signal=hft_signal,
            reasoning=reasoning,
            regime=regime,
            agreement_score=agreement_score,
        )
        
        self.signal_history.append(unified_signal)
        return unified_signal
    
    def _detect_regime(self, df: pd.DataFrame, idx: int) -> str:
        """Detect current market regime."""
        if idx < 100:
            return "unknown"
        
        close = df['close']
        returns = close.pct_change()
        
        # Trend detection
        sma_20 = close.iloc[idx-20:idx].mean()
        sma_50 = close.iloc[idx-50:idx].mean()
        current = close.iloc[idx]
        
        # Volatility
        recent_vol = returns.iloc[idx-20:idx].std() * np.sqrt(252)
        hist_vol = returns.iloc[max(0,idx-252):idx].std() * np.sqrt(252)
        
        vol_ratio = recent_vol / (hist_vol + 0.01)
        
        # ADX for trend strength
        if 'adx' in df.columns:
            adx = df['adx'].iloc[idx]
        else:
            adx = 25  # Default neutral
        
        # Classify regime
        if vol_ratio > 1.5:
            return "high_volatility"
        elif vol_ratio < 0.7:
            return "low_volatility"
        elif adx > 30 and current > sma_20 > sma_50:
            return "bull_trend"
        elif adx > 30 and current < sma_20 < sma_50:
            return "bear_trend"
        elif adx < 20:
            return "ranging"
        else:
            return "neutral"
    
    def _get_ml_signal(self, df: pd.DataFrame, idx: int) -> Optional[float]:
        """Get ML model prediction."""
        if self.ml_model is None:
            return None
        
        try:
            exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'target', 'date']
            feature_cols = [c for c in df.columns if c.lower() not in exclude_cols]
            
            if not feature_cols:
                return None
            
            X = df[feature_cols].iloc[[idx]].fillna(0)
            
            if hasattr(self.ml_model, 'predict_proba'):
                prob = self.ml_model.predict_proba(X)[0]
                # Convert probability to signal: 0.5 → 0, 1.0 → 1, 0.0 → -1
                return (prob - 0.5) * 2
            else:
                pred = self.ml_model.predict(X)[0]
                return pred if isinstance(pred, (int, float)) else 0
                
        except Exception as e:
            logger.warning(f"ML prediction failed: {e}")
            return None
    
    def _get_institutional_signal(self, df: pd.DataFrame, idx: int) -> Optional[Dict]:
        """Get institutional strategy ensemble signal."""
        try:
            result = self.institutional_ensemble.generate_ensemble_signal(df, idx)
            return result
        except Exception as e:
            logger.warning(f"Institutional signal failed: {e}")
            return None
    
    def _get_hft_signal(
        self,
        df: pd.DataFrame,
        idx: int,
        lob_data: Optional[List[LOBSnapshot]] = None,
        component_prices: Optional[pd.DataFrame] = None,
    ) -> Optional[HFTSignal]:
        """Get HFT strategy ensemble signal."""
        if self.hft_ensemble is None:
            return None
        
        try:
            signal = self.hft_ensemble.generate_ensemble_signal(
                df=df,
                component_prices=component_prices,
                current_idx=idx,
            )
            return signal
        except Exception as e:
            logger.warning(f"HFT signal failed: {e}")
            return None
    
    def _get_regime_adjusted_weights(self, regime: str) -> Dict[str, float]:
        """Adjust signal weights based on market regime."""
        base_weights = {
            'ml': self.config.ml_weight,
            'institutional': self.config.institutional_weight,
            'hft': self.config.hft_weight,
        }
        
        # Regime-specific adjustments
        if regime == "high_volatility":
            # In high vol, trust HFT less (too noisy), trust ML more
            base_weights['hft'] *= 0.5
            base_weights['ml'] *= 1.2
            
        elif regime == "bull_trend" or regime == "bear_trend":
            # In trending markets, institutional momentum strategies shine
            base_weights['institutional'] *= 1.3
            base_weights['hft'] *= 0.8
            
        elif regime == "ranging":
            # In ranging markets, HFT mean-reversion and stat arb work well
            base_weights['hft'] *= 1.3
            base_weights['institutional'] *= 0.8
            
        elif regime == "low_volatility":
            # Low vol = good for breakout and HFT
            base_weights['hft'] *= 1.2
        
        # Normalize
        total = sum(base_weights.values())
        return {k: v/total for k, v in base_weights.items()}
    
    def _combine_signals(
        self,
        ml_signal: Optional[float],
        inst_signal: Optional[Dict],
        hft_signal: Optional[HFTSignal],
        weights: Dict[str, float],
    ) -> Tuple[float, float, float]:
        """
        Combine signals from all sources.
        
        Returns: (direction, strength, confidence)
        """
        directions = []
        strengths = []
        confidences = []
        total_weight = 0
        
        # ML signal
        if ml_signal is not None:
            directions.append(ml_signal * weights['ml'])
            strengths.append(abs(ml_signal) * weights['ml'])
            confidences.append(min(abs(ml_signal) + 0.5, 1.0) * weights['ml'])
            total_weight += weights['ml']
        
        # Institutional signal
        if inst_signal:
            directions.append(inst_signal['direction'] * inst_signal.get('confidence', 0.5) * weights['institutional'])
            strengths.append(inst_signal.get('strength', 0.5) * weights['institutional'])
            confidences.append(inst_signal.get('confidence', 0.5) * weights['institutional'])
            total_weight += weights['institutional']
        
        # HFT signal
        if hft_signal:
            directions.append(hft_signal.direction * hft_signal.confidence * weights['hft'])
            strengths.append(hft_signal.strength * weights['hft'])
            confidences.append(hft_signal.confidence * weights['hft'])
            total_weight += weights['hft']
        
        if total_weight == 0:
            return (0, 0, 0)
        
        combined_direction = sum(directions) / total_weight
        combined_strength = sum(strengths) / total_weight
        combined_confidence = sum(confidences) / total_weight
        
        return (combined_direction, combined_strength, combined_confidence)
    
    def _calculate_agreement(
        self,
        ml_signal: Optional[float],
        inst_signal: Optional[Dict],
        hft_signal: Optional[HFTSignal],
    ) -> float:
        """Calculate how much the strategies agree."""
        directions = []
        
        if ml_signal is not None:
            directions.append(np.sign(ml_signal))
        
        if inst_signal:
            directions.append(inst_signal['direction'])
        
        if hft_signal:
            directions.append(hft_signal.direction)
        
        if len(directions) < 2:
            return 1.0  # Can't disagree with yourself
        
        # Remove neutral signals for agreement calculation
        non_neutral = [d for d in directions if d != 0]
        
        if len(non_neutral) == 0:
            return 1.0  # All neutral = agreement
        
        if len(non_neutral) == 1:
            return 0.7  # Only one has opinion
        
        # Check if all agree
        if all(d > 0 for d in non_neutral) or all(d < 0 for d in non_neutral):
            return 1.0
        
        # Partial agreement
        positive = sum(1 for d in non_neutral if d > 0)
        return max(positive, len(non_neutral) - positive) / len(non_neutral)
    
    def _calculate_position_size(
        self,
        strength: float,
        confidence: float,
        regime: str,
        df: pd.DataFrame,
        idx: int,
    ) -> float:
        """Calculate position size as % of portfolio."""
        base_size = self.config.base_position_pct
        
        # Scale by strength and confidence
        size = base_size * strength * confidence
        
        # Reduce in high volatility
        if self.config.reduce_in_high_vol and regime == "high_volatility":
            size *= self.config.vol_reduction_factor
        
        # Cap at maximum
        size = min(size, self.config.max_position_pct)
        
        return size
    
    def _calculate_risk_levels(
        self,
        price: float,
        direction: int,
        df: pd.DataFrame,
        idx: int,
    ) -> Tuple[Optional[float], Optional[float]]:
        """Calculate stop loss and take profit levels."""
        if direction == 0:
            return (None, None)
        
        # Use ATR for dynamic levels
        if 'atr' in df.columns:
            atr = df['atr'].iloc[idx]
        else:
            # Calculate ATR
            if idx < 14:
                atr = price * 0.02
            else:
                high = df['high'].iloc[idx-14:idx]
                low = df['low'].iloc[idx-14:idx]
                close = df['close'].iloc[idx-14:idx]
                tr = pd.concat([
                    high - low,
                    abs(high - close.shift()),
                    abs(low - close.shift())
                ], axis=1).max(axis=1)
                atr = tr.mean()
        
        # 2:1 risk reward minimum
        if direction == 1:  # Long
            stop_loss = price - (2 * atr)
            take_profit = price + (4 * atr)
        else:  # Short
            stop_loss = price + (2 * atr)
            take_profit = price - (4 * atr)
        
        return (stop_loss, take_profit)
    
    def _determine_execution_path(
        self,
        hft_signal: Optional[HFTSignal],
        confidence: float,
        regime: str,
    ) -> ExecutionPath:
        """Determine which execution path to use."""
        # High confidence HFT signal → low latency
        if (hft_signal and 
            hft_signal.confidence > self.config.hft_execution_threshold and
            self.config.use_hft_for_timing):
            return ExecutionPath.LOW_LATENCY
        
        # High volatility → be careful
        if regime == "high_volatility":
            return ExecutionPath.STANDARD
        
        # High overall confidence → standard execution is fine
        if confidence > self.config.high_confidence:
            return ExecutionPath.STANDARD
        
        # Default to standard
        return ExecutionPath.STANDARD
    
    def get_signal_summary(self, n_last: int = 10) -> Dict:
        """Get summary of recent signals."""
        if not self.signal_history:
            return {}
        
        recent = self.signal_history[-n_last:]
        
        return {
            'total_signals': len(self.signal_history),
            'recent_signals': len(recent),
            'long_signals': sum(1 for s in recent if s.direction == 1),
            'short_signals': sum(1 for s in recent if s.direction == -1),
            'neutral_signals': sum(1 for s in recent if s.direction == 0),
            'avg_confidence': np.mean([s.confidence for s in recent]),
            'avg_agreement': np.mean([s.agreement_score for s in recent]),
            'regimes': {s.regime for s in recent},
        }
    
    def signals_to_dataframe(self) -> pd.DataFrame:
        """Convert signal history to DataFrame."""
        return pd.DataFrame([s.to_dict() for s in self.signal_history])


def integrate_with_main_engine(
    engine: Any,
    use_hft: bool = True,
) -> UnifiedSignalAggregator:
    """
    Helper function to integrate UnifiedSignalAggregator with TradingEngine.
    
    Args:
        engine: TradingEngine instance
        use_hft: Whether to enable HFT strategies
        
    Returns:
        Configured UnifiedSignalAggregator
    """
    aggregator = UnifiedSignalAggregator(
        ml_model=engine.ml_model if hasattr(engine, 'ml_model') else None,
        use_hft=use_hft,
    )
    
    return aggregator
