"""
High-Frequency Trading Strategies Module.

Implements HFT strategies adapted for Python-based quantitative trading:
1. Statistical Arbitrage with EWLR (Exponentially Weighted Linear Regression)
2. Order Book Imbalance / LOB Pressure
3. Market Making (Intelligent)
4. Index Arbitrage
5. Technical Strategy Search (Overfitting Mitigation)

Note: True HFT requires C++/FPGA. This module provides:
- Research & backtesting framework
- Signal generation for downstream execution
- Strategy validation before production deployment

References:
- Sources from NotebookLLM HFT research (53+ sources)
- Jegadeesh & Titman momentum research
- Market microstructure literature
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from scipy import stats
from scipy.optimize import minimize
import warnings

logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class LOBSnapshot:
    """
    Limit Order Book Snapshot.
    
    Represents the state of the order book at a point in time.
    In real HFT, this comes from Level 2/3 market data feeds.
    """
    timestamp: pd.Timestamp
    bid_prices: np.ndarray  # Best bid to Nth level
    bid_sizes: np.ndarray   # Sizes at each bid level
    ask_prices: np.ndarray  # Best ask to Nth level
    ask_sizes: np.ndarray   # Sizes at each ask level
    
    @property
    def mid_price(self) -> float:
        """Calculate mid price."""
        return (self.bid_prices[0] + self.ask_prices[0]) / 2
    
    @property
    def spread(self) -> float:
        """Calculate bid-ask spread."""
        return self.ask_prices[0] - self.bid_prices[0]
    
    @property
    def spread_bps(self) -> float:
        """Spread in basis points."""
        return (self.spread / self.mid_price) * 10000
    
    @property
    def imbalance(self) -> float:
        """
        Calculate LOB imbalance.
        I_t = (Q_B - Q_A) / (Q_B + Q_A)
        
        Positive = more buying pressure
        Negative = more selling pressure
        """
        total_bid = self.bid_sizes.sum()
        total_ask = self.ask_sizes.sum()
        if total_bid + total_ask == 0:
            return 0
        return (total_bid - total_ask) / (total_bid + total_ask)
    
    @property
    def weighted_imbalance(self) -> float:
        """
        Distance-weighted imbalance.
        Levels closer to mid-price get higher weight.
        """
        n_levels = len(self.bid_prices)
        weights = np.exp(-np.arange(n_levels) * 0.5)  # Exponential decay
        weights = weights / weights.sum()
        
        weighted_bid = (self.bid_sizes * weights).sum()
        weighted_ask = (self.ask_sizes * weights).sum()
        
        if weighted_bid + weighted_ask == 0:
            return 0
        return (weighted_bid - weighted_ask) / (weighted_bid + weighted_ask)


@dataclass
class HFTSignal:
    """Signal from HFT strategy."""
    strategy: str
    direction: int  # 1 = buy, -1 = sell, 0 = neutral
    strength: float  # Signal strength 0-1
    confidence: float  # Confidence 0-1
    entry_price: float
    stop_loss_bps: float  # Stop loss in basis points
    take_profit_bps: float  # Take profit in basis points
    quantity_pct: float  # Position size as % of max
    reasoning: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# STRATEGY 1: STATISTICAL ARBITRAGE WITH EWLR
# =============================================================================

class StatisticalArbitrageEWLR:
    """
    Statistical Arbitrage using Exponentially Weighted Linear Regression.
    
    From your research:
    - Model: Estimated Y = βX where β = (X'WX)^(-1)(X'WY)
    - W is diagonal matrix with exponential weights
    - Trade when actual price deviates from estimated
    - Mean reversion assumption
    
    Use cases:
    - Pairs trading (stock vs ETF, related stocks)
    - Index arbitrage (ETF vs components)
    - Cross-asset relationships
    """
    
    def __init__(
        self,
        alpha: float = 0.9999,  # Exponential weighting factor
        entry_threshold: float = 2.0,  # Z-score for entry
        exit_threshold: float = 0.5,  # Z-score for exit
        lookback: int = 500,  # Historical window
        half_life: int = 20,  # Mean reversion half-life
    ):
        """
        Initialize EWLR Statistical Arbitrage.
        
        Args:
            alpha: Weighting factor (0.9999 typical for HFT)
            entry_threshold: Standard deviations to trigger trade
            exit_threshold: Standard deviations to close trade
            lookback: Number of historical points to use
            half_life: Expected mean reversion half-life
        """
        self.alpha = alpha
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.lookback = lookback
        self.half_life = half_life
        
        # State
        self.beta = None
        self.spread_mean = None
        self.spread_std = None
        self.position = 0  # Current position
    
    def fit(
        self,
        Y: pd.Series,  # Target asset prices
        X: pd.DataFrame,  # Component asset prices
    ) -> Dict[str, float]:
        """
        Fit EWLR model to historical data.
        
        Args:
            Y: Target asset price series
            X: Component asset price matrix
            
        Returns:
            Dict with beta coefficients and model stats
        """
        n = len(Y)
        
        # Create exponential weights
        # w_i = alpha^(n-i) for i = 1...n
        weights = np.array([self.alpha ** (n - i - 1) for i in range(n)])
        W = np.diag(weights)
        
        # Add constant term to X
        X_with_const = np.column_stack([np.ones(n), X.values])
        
        # EWLR: β = (X'WX)^(-1)(X'WY)
        try:
            XtW = X_with_const.T @ W
            XtWX = XtW @ X_with_const
            XtWY = XtW @ Y.values
            
            # Solve for beta
            self.beta = np.linalg.solve(XtWX, XtWY)
            
        except np.linalg.LinAlgError:
            logger.warning("Singular matrix in EWLR - using pseudo-inverse")
            XtW = X_with_const.T @ W
            XtWX = XtW @ X_with_const
            XtWY = XtW @ Y.values
            self.beta = np.linalg.pinv(XtWX) @ XtWY
        
        # Calculate spread (residuals)
        Y_estimated = X_with_const @ self.beta
        spread = Y.values - Y_estimated
        
        # Exponentially weighted spread statistics
        self.spread_mean = np.average(spread, weights=weights)
        variance = np.average((spread - self.spread_mean) ** 2, weights=weights)
        self.spread_std = np.sqrt(variance)
        
        return {
            'beta': self.beta.tolist(),
            'spread_mean': self.spread_mean,
            'spread_std': self.spread_std,
            'r_squared': 1 - variance / np.var(Y.values),
        }
    
    def get_zscore(
        self,
        Y_current: float,
        X_current: np.ndarray,
    ) -> float:
        """
        Calculate current Z-score of spread.
        
        Args:
            Y_current: Current price of target asset
            X_current: Current prices of component assets
            
        Returns:
            Z-score indicating deviation from fair value
        """
        if self.beta is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Add constant term
        X_with_const = np.concatenate([[1], X_current])
        
        # Estimated fair value
        Y_estimated = X_with_const @ self.beta
        
        # Current spread
        spread = Y_current - Y_estimated
        
        # Z-score
        if self.spread_std == 0:
            return 0
        return (spread - self.spread_mean) / self.spread_std
    
    def generate_signal(
        self,
        Y_series: pd.Series,
        X_df: pd.DataFrame,
        current_idx: int = -1,
    ) -> HFTSignal:
        """
        Generate trading signal based on spread deviation.
        
        Trading Logic (from your research):
        - BUY when actual Y << Estimated Y (undervalued)
        - SELL when actual Y >> Estimated Y (overvalued)
        - CLOSE when spread reverts to near mean
        """
        if current_idx == -1:
            current_idx = len(Y_series) - 1
        
        # Fit on rolling window
        start_idx = max(0, current_idx - self.lookback)
        Y_window = Y_series.iloc[start_idx:current_idx]
        X_window = X_df.iloc[start_idx:current_idx]
        
        if len(Y_window) < 50:  # Minimum data requirement
            return self._neutral_signal(Y_series.iloc[current_idx])
        
        self.fit(Y_window, X_window)
        
        # Current observation
        Y_current = Y_series.iloc[current_idx]
        X_current = X_df.iloc[current_idx].values
        
        zscore = self.get_zscore(Y_current, X_current)
        
        reasoning = [f"Spread Z-score: {zscore:.2f}"]
        
        # Generate signal
        if self.position == 0:  # No position
            if zscore < -self.entry_threshold:
                # Spread too negative = Y undervalued = BUY Y
                direction = 1
                strength = min(abs(zscore) / 4, 1.0)
                reasoning.append(f"ENTRY: Y undervalued (zscore={zscore:.2f} < -{self.entry_threshold})")
                self.position = 1
            elif zscore > self.entry_threshold:
                # Spread too positive = Y overvalued = SELL Y
                direction = -1
                strength = min(abs(zscore) / 4, 1.0)
                reasoning.append(f"ENTRY: Y overvalued (zscore={zscore:.2f} > {self.entry_threshold})")
                self.position = -1
            else:
                direction = 0
                strength = 0
                reasoning.append("No entry signal - spread within bounds")
        
        else:  # Have position
            # Check for exit
            if abs(zscore) < self.exit_threshold:
                direction = 0  # Close position
                strength = 1.0
                reasoning.append(f"EXIT: Spread reverted (zscore={zscore:.2f})")
                self.position = 0
            elif self.position == 1 and zscore > self.entry_threshold:
                # Stop out - spread went wrong way
                direction = 0
                strength = 1.0
                reasoning.append(f"STOP: Spread diverged further")
                self.position = 0
            elif self.position == -1 and zscore < -self.entry_threshold:
                direction = 0
                strength = 1.0
                reasoning.append(f"STOP: Spread diverged further")
                self.position = 0
            else:
                # Hold position
                direction = self.position
                strength = 0.5
                reasoning.append(f"HOLD: Waiting for reversion")
        
        # Confidence based on model fit and zscore extremity
        confidence = min(0.5 + abs(zscore) / 10, 0.95)
        
        return HFTSignal(
            strategy="StatArb_EWLR",
            direction=direction,
            strength=strength,
            confidence=confidence,
            entry_price=Y_current,
            stop_loss_bps=100,  # 1% stop
            take_profit_bps=50,  # 0.5% profit target
            quantity_pct=strength,
            reasoning=reasoning,
            metadata={
                'zscore': zscore,
                'beta': self.beta.tolist() if self.beta is not None else [],
                'spread_std': self.spread_std,
            }
        )
    
    def _neutral_signal(self, price: float) -> HFTSignal:
        return HFTSignal(
            strategy="StatArb_EWLR",
            direction=0,
            strength=0,
            confidence=0.5,
            entry_price=price,
            stop_loss_bps=100,
            take_profit_bps=50,
            quantity_pct=0,
            reasoning=["Insufficient data for model"],
        )


# =============================================================================
# STRATEGY 2: ORDER BOOK IMBALANCE
# =============================================================================

class OrderBookImbalanceStrategy:
    """
    LOB Imbalance Strategy.
    
    From your research:
    - I_t = (Q_B - Q_A) / (Q_B + Q_A)
    - If demand (bid) >> supply (ask), price expected to rise
    - Uses multiple snapshots for momentum confirmation
    - Requires Level 2 data (or proxy)
    
    This implementation works with:
    1. Real LOB data if available
    2. Proxy from OHLCV + volume data
    """
    
    def __init__(
        self,
        imbalance_threshold: float = 0.3,  # Minimum imbalance to trade
        momentum_periods: int = 3,  # Number of snapshots for momentum
        snapshot_interval: int = 1,  # Bars between snapshots
        profit_target_bps: float = 50,  # 50 basis points
        stop_loss_bps: float = 100,  # 100 basis points
        n_levels: int = 5,  # LOB depth levels to consider
        decay_factor: float = 0.5,  # Decay for weighting deeper levels
    ):
        self.imbalance_threshold = imbalance_threshold
        self.momentum_periods = momentum_periods
        self.snapshot_interval = snapshot_interval
        self.profit_target_bps = profit_target_bps
        self.stop_loss_bps = stop_loss_bps
        self.n_levels = n_levels
        self.decay_factor = decay_factor
        
        # State
        self.imbalance_history: List[float] = []
    
    def estimate_lob_from_ohlcv(
        self,
        df: pd.DataFrame,
        idx: int,
    ) -> LOBSnapshot:
        """
        Estimate LOB state from OHLCV data.
        
        This is a PROXY when real Level 2 data isn't available.
        In production HFT, you'd use actual market data feeds.
        """
        if idx < 0:
            idx = len(df) + idx
        
        row = df.iloc[idx]
        close = row['close']
        high = row['high']
        low = row['low']
        volume = row['volume']
        
        # Estimate spread from high-low
        typical_spread = (high - low) * 0.1  # Approximate
        
        # Estimate bid/ask from close position in range
        range_position = (close - low) / (high - low + 1e-10)
        
        # Generate synthetic LOB levels
        bid_prices = np.array([
            close - typical_spread * (i + 0.5) 
            for i in range(self.n_levels)
        ])
        ask_prices = np.array([
            close + typical_spread * (i + 0.5)
            for i in range(self.n_levels)
        ])
        
        # Estimate sizes based on volume and price position
        # Higher range_position = more buying = larger bids
        avg_size = volume / (2 * self.n_levels)
        
        bid_multiplier = 0.5 + range_position  # 0.5 to 1.5
        ask_multiplier = 1.5 - range_position
        
        bid_sizes = np.array([
            avg_size * bid_multiplier * np.exp(-i * 0.3)
            for i in range(self.n_levels)
        ])
        ask_sizes = np.array([
            avg_size * ask_multiplier * np.exp(-i * 0.3)
            for i in range(self.n_levels)
        ])
        
        return LOBSnapshot(
            timestamp=df.index[idx] if hasattr(df.index[idx], 'strftime') else pd.Timestamp.now(),
            bid_prices=bid_prices,
            bid_sizes=bid_sizes,
            ask_prices=ask_prices,
            ask_sizes=ask_sizes,
        )
    
    def calculate_imbalance_features(
        self,
        df: pd.DataFrame,
        current_idx: int = -1,
    ) -> Dict[str, float]:
        """
        Calculate imbalance-based features.
        
        Returns multiple imbalance metrics for ML features.
        """
        if current_idx == -1:
            current_idx = len(df) - 1
        
        # Current LOB estimate
        lob = self.estimate_lob_from_ohlcv(df, current_idx)
        
        # Get historical imbalances for momentum
        imbalances = []
        for i in range(self.momentum_periods):
            hist_idx = current_idx - i * self.snapshot_interval
            if hist_idx >= 0:
                hist_lob = self.estimate_lob_from_ohlcv(df, hist_idx)
                imbalances.append(hist_lob.imbalance)
        
        # Calculate momentum (change in imbalance)
        if len(imbalances) >= 2:
            imbalance_momentum = imbalances[0] - imbalances[-1]
        else:
            imbalance_momentum = 0
        
        return {
            'imbalance': lob.imbalance,
            'weighted_imbalance': lob.weighted_imbalance,
            'imbalance_momentum': imbalance_momentum,
            'spread_bps': lob.spread_bps,
            'mid_price': lob.mid_price,
        }
    
    def generate_signal(
        self,
        df: pd.DataFrame,
        current_idx: int = -1,
        lob_data: Optional[List[LOBSnapshot]] = None,
    ) -> HFTSignal:
        """
        Generate signal based on LOB imbalance.
        
        From your research - Signal Logic:
        1. |I_t| > threshold AND
        2. Sufficient momentum in imbalance change
        """
        if current_idx == -1:
            current_idx = len(df) - 1
        
        if current_idx < self.momentum_periods * self.snapshot_interval:
            return self._neutral_signal(df['close'].iloc[current_idx])
        
        # Use real LOB data if provided, else estimate
        if lob_data:
            current_lob = lob_data[-1]
            imbalances = [lob.imbalance for lob in lob_data[-self.momentum_periods:]]
        else:
            features = self.calculate_imbalance_features(df, current_idx)
            current_imbalance = features['imbalance']
            imbalances = [current_imbalance]
            
            for i in range(1, self.momentum_periods):
                hist_idx = current_idx - i * self.snapshot_interval
                if hist_idx >= 0:
                    hist_features = self.calculate_imbalance_features(df, hist_idx)
                    imbalances.append(hist_features['imbalance'])
        
        current_imbalance = imbalances[0]
        
        reasoning = [f"Current imbalance: {current_imbalance:.3f}"]
        
        # Check threshold condition
        if abs(current_imbalance) < self.imbalance_threshold:
            return self._neutral_signal(
                df['close'].iloc[current_idx],
                [f"Imbalance {current_imbalance:.3f} below threshold {self.imbalance_threshold}"]
            )
        
        # Check momentum condition
        if len(imbalances) >= self.momentum_periods:
            # All imbalances should be moving in same direction
            momentum_consistent = all(
                np.sign(imbalances[i]) == np.sign(imbalances[0])
                for i in range(min(3, len(imbalances)))
            )
            momentum_increasing = (
                abs(imbalances[0]) > abs(imbalances[-1]) * 0.9
            )
        else:
            momentum_consistent = True
            momentum_increasing = True
        
        if not (momentum_consistent and momentum_increasing):
            reasoning.append("Momentum not confirmed - signal weakened")
        
        # Generate signal
        direction = 1 if current_imbalance > 0 else -1
        
        # Signal strength proportional to imbalance magnitude
        base_strength = min(abs(current_imbalance) / 0.5, 1.0)
        if momentum_consistent and momentum_increasing:
            strength = base_strength
            confidence = 0.7
        else:
            strength = base_strength * 0.5
            confidence = 0.5
        
        current_price = df['close'].iloc[current_idx]
        reasoning.append(f"Direction: {'BUY' if direction > 0 else 'SELL'}")
        reasoning.append(f"Signal strength: {strength:.2f}")
        
        return HFTSignal(
            strategy="LOB_Imbalance",
            direction=direction,
            strength=strength,
            confidence=confidence,
            entry_price=current_price,
            stop_loss_bps=self.stop_loss_bps,
            take_profit_bps=self.profit_target_bps,
            quantity_pct=strength,
            reasoning=reasoning,
            metadata={
                'imbalance': current_imbalance,
                'imbalance_history': imbalances,
                'momentum_consistent': momentum_consistent,
            }
        )
    
    def _neutral_signal(self, price: float, reasoning: List[str] = None) -> HFTSignal:
        return HFTSignal(
            strategy="LOB_Imbalance",
            direction=0,
            strength=0,
            confidence=0.5,
            entry_price=price,
            stop_loss_bps=self.stop_loss_bps,
            take_profit_bps=self.profit_target_bps,
            quantity_pct=0,
            reasoning=reasoning or ["No signal"],
        )


# =============================================================================
# STRATEGY 3: INTELLIGENT MARKET MAKING
# =============================================================================

class IntelligentMarketMaker:
    """
    ML-Driven Market Making Strategy.
    
    From your research:
    - Uses Random Forest to predict optimal bid/ask placement
    - Features: spread, weighted midpoints, volume, trade rates
    - Immediately reverses position after fill (risk averse)
    - Manages inventory risk
    
    Note: True market making requires:
    - Sub-millisecond execution
    - Direct market access
    - Significant capital for inventory
    """
    
    def __init__(
        self,
        spread_target_bps: float = 10,  # Target spread capture
        max_inventory: float = 1000,  # Max position size
        inventory_skew: float = 0.5,  # How much to skew quotes based on inventory
        lookback_ticks: int = 3,  # Historical ticks for features
    ):
        self.spread_target_bps = spread_target_bps
        self.max_inventory = max_inventory
        self.inventory_skew = inventory_skew
        self.lookback_ticks = lookback_ticks
        
        # State
        self.inventory = 0
        self.ml_model = None
        self.feature_scaler = None
    
    def extract_features(
        self,
        df: pd.DataFrame,
        lob: Optional[LOBSnapshot] = None,
        idx: int = -1,
    ) -> np.ndarray:
        """
        Extract 54 features for market making prediction.
        
        Feature categories (from your research):
        - Spread and midpoint
        - Bid/ask deltas at multiple levels
        - Volumes
        - Recent trade rates
        """
        if idx == -1:
            idx = len(df) - 1
        
        features = []
        
        for tick in range(self.lookback_ticks):
            tick_idx = idx - tick
            if tick_idx < 0:
                # Pad with zeros
                features.extend([0] * 18)
                continue
            
            row = df.iloc[tick_idx]
            prev_row = df.iloc[tick_idx - 1] if tick_idx > 0 else row
            
            # Basic price features
            close = row['close']
            open_p = row['open']
            high = row['high']
            low = row['low']
            volume = row['volume']
            
            # Spread proxy
            spread = (high - low) / close
            
            # Midpoint
            midpoint = (high + low) / 2
            
            # VWAP proxy
            vwap = (high + low + close) / 3
            
            # Price changes (deltas)
            price_delta = close - prev_row['close']
            range_delta = (high - low) - (prev_row['high'] - prev_row['low'])
            
            # Volume features
            volume_ratio = volume / (df['volume'].iloc[max(0, tick_idx-20):tick_idx].mean() + 1)
            
            # Trade rate proxy (based on volume and volatility)
            trade_rate = volume / ((high - low) * close + 1)
            
            # Bid/ask proxies at multiple "levels"
            # Level 1: closest to mid
            bid_1 = close - spread * 0.3 * close
            ask_1 = close + spread * 0.7 * close
            
            # Level 2
            bid_2 = close - spread * 0.6 * close
            ask_2 = close + spread * 1.4 * close
            
            # Level 3
            bid_3 = close - spread * 1.0 * close
            ask_3 = close + spread * 2.0 * close
            
            tick_features = [
                spread,
                midpoint / close,  # Normalized
                vwap / close,
                price_delta / close,
                range_delta / close,
                volume_ratio,
                trade_rate / 1e6,  # Scale down
                (bid_1 - prev_row['close']) / close,  # Delta
                (ask_1 - prev_row['close']) / close,
                (bid_2 - prev_row['close']) / close,
                (ask_2 - prev_row['close']) / close,
                (bid_3 - prev_row['close']) / close,
                (ask_3 - prev_row['close']) / close,
                high / close - 1,
                low / close - 1,
                open_p / close - 1,
                volume / 1e6,  # Scaled volume
                (close - open_p) / close,  # Body
            ]
            
            features.extend(tick_features)
        
        return np.array(features)
    
    def fit_model(
        self,
        df: pd.DataFrame,
        n_samples: int = 1000,
    ):
        """
        Train Random Forest model for quote prediction.
        
        The model predicts the probability of a favorable fill
        at different price levels.
        """
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
        
        X = []
        y = []
        
        for i in range(self.lookback_ticks + 10, min(len(df) - 1, n_samples + self.lookback_ticks + 10)):
            features = self.extract_features(df, idx=i)
            X.append(features)
            
            # Target: was next price up or down?
            next_return = (df['close'].iloc[i + 1] - df['close'].iloc[i]) / df['close'].iloc[i]
            y.append(1 if next_return > 0 else 0)
        
        X = np.array(X)
        y = np.array(y)
        
        # Scale features
        self.feature_scaler = StandardScaler()
        X_scaled = self.feature_scaler.fit_transform(X)
        
        # Train model
        self.ml_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=20,
            random_state=42,
        )
        self.ml_model.fit(X_scaled, y)
        
        logger.info(f"Market making model trained on {len(X)} samples")
    
    def generate_quotes(
        self,
        df: pd.DataFrame,
        idx: int = -1,
    ) -> Dict[str, float]:
        """
        Generate optimal bid/ask quotes.
        
        Returns dict with:
        - bid_price, ask_price
        - bid_size, ask_size
        - expected_edge
        """
        if idx == -1:
            idx = len(df) - 1
        
        current_price = df['close'].iloc[idx]
        spread_pct = self.spread_target_bps / 10000
        
        # Inventory-based skew
        inventory_pct = self.inventory / self.max_inventory
        skew = inventory_pct * self.inventory_skew * spread_pct
        
        # If we have ML model, use it to adjust
        if self.ml_model is not None:
            features = self.extract_features(df, idx=idx)
            features_scaled = self.feature_scaler.transform(features.reshape(1, -1))
            prob_up = self.ml_model.predict_proba(features_scaled)[0][1]
            
            # Adjust skew based on prediction
            # If prob_up > 0.5, we want to be more aggressive on bid (buy)
            ml_skew = (prob_up - 0.5) * spread_pct
        else:
            prob_up = 0.5
            ml_skew = 0
        
        # Calculate quotes
        mid_skew = skew + ml_skew
        bid_price = current_price * (1 - spread_pct / 2 + mid_skew)
        ask_price = current_price * (1 + spread_pct / 2 + mid_skew)
        
        # Size based on inventory
        base_size = self.max_inventory * 0.1
        bid_size = base_size * (1 - inventory_pct)  # Less buying if long
        ask_size = base_size * (1 + inventory_pct)  # More selling if long
        
        return {
            'bid_price': bid_price,
            'ask_price': ask_price,
            'bid_size': max(0, bid_size),
            'ask_size': max(0, ask_size),
            'mid_price': current_price,
            'spread_bps': (ask_price - bid_price) / current_price * 10000,
            'prob_up': prob_up,
            'inventory': self.inventory,
        }
    
    def generate_signal(
        self,
        df: pd.DataFrame,
        idx: int = -1,
    ) -> HFTSignal:
        """
        Generate market making signal.
        
        Note: In real market making, you'd place both bid and ask.
        Here we generate a directional signal based on where we
        expect to get filled.
        """
        quotes = self.generate_quotes(df, idx)
        
        # Determine direction based on probability and inventory
        prob_up = quotes['prob_up']
        inventory_pct = self.inventory / self.max_inventory
        
        # If strongly believe price going up AND not too long
        if prob_up > 0.6 and inventory_pct < 0.5:
            direction = 1
            strength = (prob_up - 0.5) * 2
            reasoning = [f"Buy signal: prob_up={prob_up:.2f}, inventory={inventory_pct:.2f}"]
        elif prob_up < 0.4 and inventory_pct > -0.5:
            direction = -1
            strength = (0.5 - prob_up) * 2
            reasoning = [f"Sell signal: prob_up={prob_up:.2f}, inventory={inventory_pct:.2f}"]
        else:
            direction = 0
            strength = 0
            reasoning = [f"No signal: prob_up={prob_up:.2f}, inventory={inventory_pct:.2f}"]
        
        return HFTSignal(
            strategy="Market_Making",
            direction=direction,
            strength=strength,
            confidence=abs(prob_up - 0.5) * 2,
            entry_price=quotes['mid_price'],
            stop_loss_bps=50,  # Tight stops for MM
            take_profit_bps=quotes['spread_bps'] / 2,  # Capture half spread
            quantity_pct=strength * 0.5,  # Conservative sizing
            reasoning=reasoning,
            metadata=quotes,
        )
    
    def update_inventory(self, fill_direction: int, fill_size: float):
        """Update inventory after a fill."""
        self.inventory += fill_direction * fill_size
        self.inventory = np.clip(self.inventory, -self.max_inventory, self.max_inventory)


# =============================================================================
# STRATEGY 4: INDEX ARBITRAGE
# =============================================================================

class IndexArbitrageStrategy:
    """
    Index Fund Arbitrage Strategy.
    
    From your research:
    - Trade ETF vs underlying index value
    - Buy ETF when current ratio < trailing average ratio
    - Sell ETF when current ratio > trailing average ratio
    - Only trade on significant discrepancies (e.g., $0.01+)
    """
    
    def __init__(
        self,
        lookback: int = 100,  # Periods for trailing average
        entry_threshold: float = 0.001,  # 0.1% minimum discrepancy
        exit_threshold: float = 0.0003,  # 0.03% to close
        min_profit_cents: float = 1.0,  # Minimum $0.01 expected profit
    ):
        self.lookback = lookback
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.min_profit_cents = min_profit_cents
        
        self.position = 0
    
    def calculate_nav(
        self,
        component_prices: pd.DataFrame,
        weights: Dict[str, float],
    ) -> pd.Series:
        """
        Calculate Net Asset Value of index from components.
        
        Args:
            component_prices: DataFrame with component prices
            weights: Dict of symbol -> weight in index
        """
        nav = pd.Series(0.0, index=component_prices.index)
        
        for symbol, weight in weights.items():
            if symbol in component_prices.columns:
                nav += component_prices[symbol] * weight
        
        return nav
    
    def generate_signal(
        self,
        etf_prices: pd.Series,
        nav_prices: pd.Series,
        current_idx: int = -1,
    ) -> HFTSignal:
        """
        Generate arbitrage signal between ETF and NAV.
        
        Trading Logic:
        - If ETF trading at discount to NAV → Buy ETF
        - If ETF trading at premium to NAV → Sell ETF
        """
        if current_idx == -1:
            current_idx = len(etf_prices) - 1
        
        if current_idx < self.lookback:
            return self._neutral_signal(etf_prices.iloc[current_idx])
        
        # Calculate price ratio (ETF / NAV)
        ratio = etf_prices / nav_prices
        current_ratio = ratio.iloc[current_idx]
        
        # Trailing average ratio
        trailing_ratio = ratio.iloc[current_idx - self.lookback:current_idx].mean()
        
        # Deviation from fair value
        deviation = (current_ratio - trailing_ratio) / trailing_ratio
        
        # Price difference in cents
        etf_price = etf_prices.iloc[current_idx]
        nav_price = nav_prices.iloc[current_idx]
        price_diff = (etf_price - nav_price * trailing_ratio) * 100  # In cents
        
        reasoning = [
            f"ETF/NAV ratio: {current_ratio:.4f}",
            f"Trailing ratio: {trailing_ratio:.4f}",
            f"Deviation: {deviation:.4%}",
            f"Price diff: ${price_diff/100:.4f}",
        ]
        
        # Check for arbitrage opportunity
        if self.position == 0:
            if deviation < -self.entry_threshold and abs(price_diff) >= self.min_profit_cents:
                # ETF undervalued relative to NAV
                direction = 1  # Buy ETF
                strength = min(abs(deviation) / 0.005, 1.0)
                reasoning.append("ETF at DISCOUNT - BUY arbitrage")
                self.position = 1
            elif deviation > self.entry_threshold and abs(price_diff) >= self.min_profit_cents:
                # ETF overvalued relative to NAV
                direction = -1  # Sell ETF
                strength = min(abs(deviation) / 0.005, 1.0)
                reasoning.append("ETF at PREMIUM - SELL arbitrage")
                self.position = -1
            else:
                direction = 0
                strength = 0
                reasoning.append("No arbitrage opportunity")
        else:
            # Check for exit
            if abs(deviation) < self.exit_threshold:
                direction = 0
                strength = 1.0
                reasoning.append("Arbitrage converged - EXIT")
                self.position = 0
            else:
                direction = self.position
                strength = 0.5
                reasoning.append("Holding arbitrage position")
        
        return HFTSignal(
            strategy="Index_Arbitrage",
            direction=direction,
            strength=strength,
            confidence=min(abs(deviation) / 0.003, 0.95),
            entry_price=etf_price,
            stop_loss_bps=30,
            take_profit_bps=10,
            quantity_pct=strength,
            reasoning=reasoning,
            metadata={
                'ratio': current_ratio,
                'trailing_ratio': trailing_ratio,
                'deviation': deviation,
                'price_diff_cents': price_diff,
            }
        )
    
    def _neutral_signal(self, price: float) -> HFTSignal:
        return HFTSignal(
            strategy="Index_Arbitrage",
            direction=0,
            strength=0,
            confidence=0.5,
            entry_price=price,
            stop_loss_bps=30,
            take_profit_bps=10,
            quantity_pct=0,
            reasoning=["Insufficient data"],
        )


# =============================================================================
# STRATEGY 5: AUTOMATED TECHNICAL STRATEGY SEARCH
# =============================================================================

class TechnicalStrategySearch:
    """
    Automated Technical Strategy Search with Overfitting Mitigation.
    
    From your research:
    - Enumerate possible strategy space
    - Benchmark against random walk with same volatility
    - Only select strategies that beat the baseline significantly
    
    This prevents data mining / overfitting biases.
    """
    
    def __init__(
        self,
        n_random_walks: int = 100,  # Number of random walks for benchmark
        significance_level: float = 0.05,  # P-value threshold
        min_sharpe_threshold: float = 0.5,  # Minimum Sharpe on real data
    ):
        self.n_random_walks = n_random_walks
        self.significance_level = significance_level
        self.min_sharpe_threshold = min_sharpe_threshold
        
        # Store validated strategies
        self.validated_strategies: List[Dict] = []
    
    def generate_random_walk(
        self,
        df: pd.DataFrame,
        seed: int = None,
    ) -> pd.Series:
        """
        Generate random walk with same statistical properties as real data.
        
        Uses bootstrap from empirical distribution of returns.
        """
        if seed:
            np.random.seed(seed)
        
        returns = df['close'].pct_change().dropna()
        
        # Bootstrap returns
        random_returns = np.random.choice(returns.values, size=len(returns), replace=True)
        
        # Generate price series
        random_prices = df['close'].iloc[0] * (1 + random_returns).cumprod()
        
        return pd.Series(random_prices, index=df.index[1:])
    
    def evaluate_strategy(
        self,
        prices: pd.Series,
        signals: np.ndarray,
    ) -> Dict[str, float]:
        """Evaluate a strategy's performance."""
        # Align
        min_len = min(len(prices), len(signals))
        prices = prices.iloc[-min_len:]
        signals = signals[-min_len:]
        
        # Calculate returns
        price_returns = prices.pct_change().values
        strategy_returns = signals[:-1] * price_returns[1:]
        strategy_returns = strategy_returns[~np.isnan(strategy_returns)]
        
        if len(strategy_returns) < 10:
            return {'sharpe': 0, 'total_return': 0, 'win_rate': 0}
        
        # Metrics
        sharpe = np.sqrt(252) * np.mean(strategy_returns) / (np.std(strategy_returns) + 1e-10)
        total_return = np.prod(1 + strategy_returns) - 1
        win_rate = np.sum(strategy_returns > 0) / len(strategy_returns)
        
        return {
            'sharpe': sharpe,
            'total_return': total_return,
            'win_rate': win_rate,
        }
    
    def benchmark_strategy(
        self,
        df: pd.DataFrame,
        signal_func: Callable,
    ) -> Tuple[bool, Dict]:
        """
        Benchmark strategy against random walks.
        
        Args:
            df: Real price data
            signal_func: Function that generates signals from prices
            
        Returns:
            (is_valid, results_dict)
        """
        # Evaluate on real data
        real_signals = signal_func(df)
        real_metrics = self.evaluate_strategy(df['close'], real_signals)
        
        logger.info(f"Real data Sharpe: {real_metrics['sharpe']:.2f}")
        
        # Minimum threshold check
        if real_metrics['sharpe'] < self.min_sharpe_threshold:
            return False, {
                'reason': 'Below minimum Sharpe threshold',
                'real_sharpe': real_metrics['sharpe'],
            }
        
        # Generate random walk benchmarks
        random_sharpes = []
        for i in range(self.n_random_walks):
            random_prices = self.generate_random_walk(df, seed=i)
            
            # Create a df-like structure for the signal function
            random_df = df.copy()
            random_df['close'] = random_prices.reindex(df.index).ffill().bfill()
            
            random_signals = signal_func(random_df)
            random_metrics = self.evaluate_strategy(random_df['close'], random_signals)
            random_sharpes.append(random_metrics['sharpe'])
        
        # Statistical test
        percentile = stats.percentileofscore(random_sharpes, real_metrics['sharpe'])
        p_value = 1 - percentile / 100
        
        is_valid = p_value < self.significance_level
        
        return is_valid, {
            'real_sharpe': real_metrics['sharpe'],
            'random_sharpe_mean': np.mean(random_sharpes),
            'random_sharpe_std': np.std(random_sharpes),
            'percentile': percentile,
            'p_value': p_value,
            'is_significant': is_valid,
        }
    
    def search_momentum_strategies(
        self,
        df: pd.DataFrame,
    ) -> List[Dict]:
        """
        Search through momentum strategy space.
        
        Strategy space: all combinations of lookback periods.
        """
        results = []
        
        lookback_periods = [5, 10, 20, 40, 60, 100]
        
        for lookback in lookback_periods:
            def signal_func(data, lb=lookback):
                returns = data['close'].pct_change(lb)
                signals = np.sign(returns.values)
                return signals
            
            is_valid, metrics = self.benchmark_strategy(df, signal_func)
            
            result = {
                'strategy': f'Momentum_{lookback}',
                'lookback': lookback,
                'is_valid': is_valid,
                **metrics,
            }
            results.append(result)
            
            if is_valid:
                self.validated_strategies.append(result)
                logger.info(f"✓ Valid strategy found: Momentum_{lookback}")
        
        return results
    
    def search_mean_reversion_strategies(
        self,
        df: pd.DataFrame,
    ) -> List[Dict]:
        """Search through mean reversion strategy space."""
        results = []
        
        lookback_periods = [10, 20, 40, 60]
        zscore_thresholds = [1.5, 2.0, 2.5]
        
        for lookback in lookback_periods:
            for threshold in zscore_thresholds:
                def signal_func(data, lb=lookback, th=threshold):
                    close = data['close']
                    zscore = (close - close.rolling(lb).mean()) / (close.rolling(lb).std() + 1e-10)
                    signals = np.zeros(len(zscore))
                    signals[zscore < -th] = 1  # Buy oversold
                    signals[zscore > th] = -1  # Sell overbought
                    return signals
                
                is_valid, metrics = self.benchmark_strategy(df, signal_func)
                
                result = {
                    'strategy': f'MeanRev_{lookback}_{threshold}',
                    'lookback': lookback,
                    'threshold': threshold,
                    'is_valid': is_valid,
                    **metrics,
                }
                results.append(result)
                
                if is_valid:
                    self.validated_strategies.append(result)
                    logger.info(f"✓ Valid strategy found: MeanRev_{lookback}_{threshold}")
        
        return results


# =============================================================================
# HFT STRATEGY ENSEMBLE
# =============================================================================

class HFTStrategyEnsemble:
    """
    Ensemble of all HFT strategies with dynamic weighting.
    
    Combines signals from:
    1. Statistical Arbitrage (EWLR)
    2. LOB Imbalance
    3. Market Making
    4. Index Arbitrage
    5. Validated Technical Strategies
    """
    
    def __init__(self):
        self.stat_arb = StatisticalArbitrageEWLR()
        self.lob_imbalance = OrderBookImbalanceStrategy()
        self.market_maker = IntelligentMarketMaker()
        
        # Strategy weights (can be optimized)
        self.weights = {
            'stat_arb': 0.3,
            'lob_imbalance': 0.3,
            'market_making': 0.2,
            'momentum': 0.2,
        }
        
        # Performance tracking
        self.performance_history: Dict[str, List[float]] = {
            name: [] for name in self.weights
        }
    
    def generate_ensemble_signal(
        self,
        df: pd.DataFrame,
        component_prices: Optional[pd.DataFrame] = None,
        current_idx: int = -1,
    ) -> HFTSignal:
        """
        Generate combined signal from all HFT strategies.
        """
        signals = {}
        
        # 1. LOB Imbalance (always available with OHLCV)
        try:
            signals['lob_imbalance'] = self.lob_imbalance.generate_signal(df, current_idx)
        except Exception as e:
            logger.warning(f"LOB Imbalance error: {e}")
        
        # 2. Market Making
        try:
            signals['market_making'] = self.market_maker.generate_signal(df, current_idx)
        except Exception as e:
            logger.warning(f"Market Making error: {e}")
        
        # 3. Statistical Arbitrage (if component prices available)
        if component_prices is not None and len(component_prices.columns) > 0:
            try:
                signals['stat_arb'] = self.stat_arb.generate_signal(
                    df['close'], component_prices, current_idx
                )
            except Exception as e:
                logger.warning(f"Stat Arb error: {e}")
        
        # 4. Simple momentum (as fallback/validation)
        try:
            signals['momentum'] = self._simple_momentum_signal(df, current_idx)
        except Exception as e:
            logger.warning(f"Momentum error: {e}")
        
        if not signals:
            current_price = df['close'].iloc[current_idx] if current_idx != -1 else df['close'].iloc[-1]
            return HFTSignal(
                strategy="HFT_Ensemble",
                direction=0,
                strength=0,
                confidence=0,
                entry_price=current_price,
                stop_loss_bps=100,
                take_profit_bps=50,
                quantity_pct=0,
                reasoning=["No valid signals from any strategy"],
            )
        
        # Weighted combination
        total_direction = 0
        total_strength = 0
        total_confidence = 0
        all_reasoning = []
        
        for name, signal in signals.items():
            weight = self.weights.get(name, 0.1)
            total_direction += signal.direction * weight * signal.confidence
            total_strength += signal.strength * weight
            total_confidence += signal.confidence * weight
            all_reasoning.append(
                f"[{name}] dir={signal.direction:+d}, str={signal.strength:.2f}, conf={signal.confidence:.2f}"
            )
        
        total_weight = sum(self.weights.get(name, 0.1) for name in signals)
        
        final_direction = int(np.sign(total_direction / total_weight))
        final_strength = total_strength / total_weight
        final_confidence = total_confidence / total_weight
        
        current_price = df['close'].iloc[current_idx] if current_idx != -1 else df['close'].iloc[-1]
        
        return HFTSignal(
            strategy="HFT_Ensemble",
            direction=final_direction,
            strength=final_strength,
            confidence=final_confidence,
            entry_price=current_price,
            stop_loss_bps=100,
            take_profit_bps=50,
            quantity_pct=final_strength,
            reasoning=all_reasoning,
            metadata={
                'individual_signals': {k: v.direction for k, v in signals.items()},
            }
        )
    
    def _simple_momentum_signal(self, df: pd.DataFrame, idx: int) -> HFTSignal:
        """Simple momentum signal for validation."""
        close = df['close']
        
        if idx == -1:
            idx = len(close) - 1
        
        if idx < 20:
            return HFTSignal(
                strategy="Momentum",
                direction=0, strength=0, confidence=0.5,
                entry_price=close.iloc[idx],
                stop_loss_bps=100, take_profit_bps=50, quantity_pct=0,
                reasoning=["Insufficient data"]
            )
        
        # Multi-timeframe momentum
        mom_5 = close.iloc[idx] / close.iloc[idx - 5] - 1
        mom_10 = close.iloc[idx] / close.iloc[idx - 10] - 1
        mom_20 = close.iloc[idx] / close.iloc[idx - 20] - 1
        
        avg_mom = (mom_5 + mom_10 + mom_20) / 3
        
        direction = int(np.sign(avg_mom)) if abs(avg_mom) > 0.001 else 0
        strength = min(abs(avg_mom) * 10, 1.0)
        
        # Confidence based on momentum consistency
        same_sign = sum([
            np.sign(mom_5) == np.sign(avg_mom),
            np.sign(mom_10) == np.sign(avg_mom),
            np.sign(mom_20) == np.sign(avg_mom),
        ])
        confidence = same_sign / 3
        
        return HFTSignal(
            strategy="Momentum",
            direction=direction,
            strength=strength,
            confidence=confidence,
            entry_price=close.iloc[idx],
            stop_loss_bps=100,
            take_profit_bps=50,
            quantity_pct=strength,
            reasoning=[f"Momentum: 5d={mom_5:.2%}, 10d={mom_10:.2%}, 20d={mom_20:.2%}"]
        )
    
    def update_weights(self, strategy_name: str, pnl: float):
        """Update strategy weights based on performance."""
        self.performance_history[strategy_name].append(pnl)
        
        # Keep last 100 trades
        if len(self.performance_history[strategy_name]) > 100:
            self.performance_history[strategy_name] = self.performance_history[strategy_name][-100:]
        
        # Recalculate weights based on Sharpe-like metric
        new_weights = {}
        for name, returns in self.performance_history.items():
            if len(returns) < 10:
                new_weights[name] = 0.25
            else:
                mean_ret = np.mean(returns)
                std_ret = np.std(returns) + 1e-10
                sharpe = mean_ret / std_ret
                new_weights[name] = max(0.05, sharpe + 1)  # Ensure positive
        
        # Normalize
        total = sum(new_weights.values())
        self.weights = {k: v / total for k, v in new_weights.items()}
