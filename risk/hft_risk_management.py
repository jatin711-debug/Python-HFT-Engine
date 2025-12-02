"""
HFT Risk Management Module.

Implements institutional-grade risk controls for high-frequency trading:
1. Pre-Trade Risk Checks - Validate orders before submission
2. Kill Switches (Circuit Breakers) - Automatic halt on anomalies
3. Position Limits - Prevent excessive exposure
4. Order Rate Limits - Prevent runaway algorithms
5. Real-time P&L Monitoring - Track performance continuously

From your research:
- Pre-trade checks operate in nanoseconds
- Kill switches halt trading instantly on threshold breach
- Critical for regulatory compliance and capital protection

Note: In production HFT, these would be implemented in hardware (FPGA)
or highly optimized C++. This Python implementation is for:
- Backtesting and research
- Signal generation with risk filtering
- Integration with slower trading systems
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import deque
import logging
import threading
import time

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS AND DATA STRUCTURES
# =============================================================================

class RiskLevel(Enum):
    """Risk severity levels."""
    NORMAL = "normal"
    ELEVATED = "elevated"
    HIGH = "high"
    CRITICAL = "critical"
    HALT = "halt"


class OrderType(Enum):
    """Order types."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class RiskCheckResult(Enum):
    """Results of pre-trade risk check."""
    APPROVED = "approved"
    REJECTED = "rejected"
    MODIFIED = "modified"  # Order modified to comply
    PENDING_REVIEW = "pending_review"


@dataclass
class Order:
    """Order representation for risk checking."""
    order_id: str
    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: float
    price: Optional[float]
    order_type: OrderType
    timestamp: datetime = field(default_factory=datetime.now)
    
    @property
    def notional_value(self) -> float:
        """Estimated notional value."""
        return self.quantity * (self.price or 0)


@dataclass
class Position:
    """Current position in a symbol."""
    symbol: str
    quantity: float
    avg_entry_price: float
    current_price: float
    unrealized_pnl: float = 0
    realized_pnl: float = 0
    
    @property
    def notional_value(self) -> float:
        return abs(self.quantity) * self.current_price
    
    @property
    def is_long(self) -> bool:
        return self.quantity > 0
    
    @property
    def is_short(self) -> bool:
        return self.quantity < 0


@dataclass
class RiskCheckReport:
    """Report from pre-trade risk check."""
    result: RiskCheckResult
    order: Order
    checks_passed: List[str]
    checks_failed: List[str]
    warnings: List[str]
    modified_order: Optional[Order] = None
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RiskMetrics:
    """Real-time risk metrics snapshot."""
    timestamp: datetime
    total_pnl: float
    daily_pnl: float
    drawdown: float
    max_drawdown: float
    var_95: float
    exposure: float
    position_count: int
    order_count: int
    risk_level: RiskLevel


# =============================================================================
# PRE-TRADE RISK CHECKS
# =============================================================================

class PreTradeRiskChecker:
    """
    Pre-Trade Risk Checks.
    
    From your research:
    - Validate orders against position limits, order rate limits, credit limits
    - Must operate in nanoseconds (in production)
    - Prevents oversized or erroneous trades
    
    Checks performed:
    1. Position limits (max size per symbol, sector, portfolio)
    2. Order size limits (max single order size)
    3. Price reasonability (deviation from market)
    4. Order rate limits (orders per second/minute)
    5. Credit/margin limits
    6. Notional limits
    """
    
    def __init__(
        self,
        # Position limits
        max_position_per_symbol: float = 100_000,  # Max $ per symbol
        max_position_pct: float = 0.10,  # Max 10% of portfolio in one position
        max_portfolio_notional: float = 1_000_000,  # Max total exposure
        
        # Order limits
        max_order_size: float = 10_000,  # Max single order $
        max_order_pct: float = 0.05,  # Max 5% of portfolio per order
        
        # Price limits
        max_price_deviation: float = 0.05,  # 5% from reference price
        
        # Rate limits
        max_orders_per_second: int = 10,
        max_orders_per_minute: int = 100,
        
        # P&L limits
        max_daily_loss: float = 10_000,  # $ loss limit
        max_daily_loss_pct: float = 0.03,  # 3% of portfolio
    ):
        self.max_position_per_symbol = max_position_per_symbol
        self.max_position_pct = max_position_pct
        self.max_portfolio_notional = max_portfolio_notional
        self.max_order_size = max_order_size
        self.max_order_pct = max_order_pct
        self.max_price_deviation = max_price_deviation
        self.max_orders_per_second = max_orders_per_second
        self.max_orders_per_minute = max_orders_per_minute
        self.max_daily_loss = max_daily_loss
        self.max_daily_loss_pct = max_daily_loss_pct
        
        # Order tracking for rate limits
        self.order_timestamps: deque = deque(maxlen=1000)
        
        # State
        self.current_positions: Dict[str, Position] = {}
        self.daily_pnl: float = 0
        self.portfolio_value: float = 100_000  # Default
        self.reference_prices: Dict[str, float] = {}
    
    def check_order(
        self,
        order: Order,
        reference_price: Optional[float] = None,
    ) -> RiskCheckReport:
        """
        Perform all pre-trade risk checks on an order.
        
        Args:
            order: Order to validate
            reference_price: Current market price for the symbol
            
        Returns:
            RiskCheckReport with result and details
        """
        checks_passed = []
        checks_failed = []
        warnings = []
        details = {}
        
        # Store reference price
        if reference_price:
            self.reference_prices[order.symbol] = reference_price
        ref_price = self.reference_prices.get(order.symbol, order.price)
        
        # 1. Order Size Check
        if self._check_order_size(order, details):
            checks_passed.append("order_size")
        else:
            checks_failed.append("order_size")
        
        # 2. Position Limit Check
        if self._check_position_limits(order, ref_price, details):
            checks_passed.append("position_limits")
        else:
            checks_failed.append("position_limits")
        
        # 3. Price Reasonability Check
        if self._check_price_reasonability(order, ref_price, details):
            checks_passed.append("price_reasonability")
        else:
            checks_failed.append("price_reasonability")
        
        # 4. Order Rate Check
        if self._check_order_rate(details):
            checks_passed.append("order_rate")
        else:
            checks_failed.append("order_rate")
        
        # 5. Daily Loss Check
        if self._check_daily_loss(details):
            checks_passed.append("daily_loss")
        else:
            checks_failed.append("daily_loss")
        
        # 6. Portfolio Exposure Check
        if self._check_portfolio_exposure(order, ref_price, details):
            checks_passed.append("portfolio_exposure")
        else:
            checks_failed.append("portfolio_exposure")
        
        # Determine result
        if len(checks_failed) == 0:
            result = RiskCheckResult.APPROVED
        elif 'daily_loss' in checks_failed or 'order_rate' in checks_failed:
            result = RiskCheckResult.REJECTED
        else:
            # Try to modify order to comply
            modified = self._try_modify_order(order, ref_price, checks_failed)
            if modified:
                result = RiskCheckResult.MODIFIED
                warnings.append("Order modified to comply with risk limits")
            else:
                result = RiskCheckResult.REJECTED
        
        # Record order timestamp for rate limiting
        if result in [RiskCheckResult.APPROVED, RiskCheckResult.MODIFIED]:
            self.order_timestamps.append(datetime.now())
        
        return RiskCheckReport(
            result=result,
            order=order,
            checks_passed=checks_passed,
            checks_failed=checks_failed,
            warnings=warnings,
            modified_order=modified if result == RiskCheckResult.MODIFIED else None,
            details=details,
        )
    
    def _check_order_size(self, order: Order, details: Dict) -> bool:
        """Check order size limits."""
        notional = order.notional_value
        
        # Absolute limit
        if notional > self.max_order_size:
            details['order_size_violation'] = f"${notional:.0f} > ${self.max_order_size:.0f}"
            return False
        
        # Portfolio percentage limit
        pct = notional / self.portfolio_value
        if pct > self.max_order_pct:
            details['order_pct_violation'] = f"{pct:.1%} > {self.max_order_pct:.1%}"
            return False
        
        return True
    
    def _check_position_limits(
        self,
        order: Order,
        ref_price: float,
        details: Dict,
    ) -> bool:
        """Check position limits."""
        current_pos = self.current_positions.get(order.symbol)
        
        # Calculate resulting position
        order_qty = order.quantity if order.side == 'buy' else -order.quantity
        current_qty = current_pos.quantity if current_pos else 0
        resulting_qty = current_qty + order_qty
        resulting_notional = abs(resulting_qty * ref_price)
        
        # Per-symbol limit
        if resulting_notional > self.max_position_per_symbol:
            details['position_limit_violation'] = (
                f"${resulting_notional:.0f} > ${self.max_position_per_symbol:.0f}"
            )
            return False
        
        # Portfolio percentage limit
        pct = resulting_notional / self.portfolio_value
        if pct > self.max_position_pct:
            details['position_pct_violation'] = f"{pct:.1%} > {self.max_position_pct:.1%}"
            return False
        
        return True
    
    def _check_price_reasonability(
        self,
        order: Order,
        ref_price: float,
        details: Dict,
    ) -> bool:
        """Check if order price is reasonable."""
        if order.order_type == OrderType.MARKET or order.price is None:
            return True  # Market orders don't have price
        
        if ref_price is None or ref_price == 0:
            return True  # No reference to check against
        
        deviation = abs(order.price - ref_price) / ref_price
        
        if deviation > self.max_price_deviation:
            details['price_deviation'] = f"{deviation:.1%} > {self.max_price_deviation:.1%}"
            return False
        
        return True
    
    def _check_order_rate(self, details: Dict) -> bool:
        """Check order rate limits."""
        now = datetime.now()
        
        # Orders in last second
        recent_second = sum(
            1 for ts in self.order_timestamps
            if (now - ts).total_seconds() < 1
        )
        
        if recent_second >= self.max_orders_per_second:
            details['rate_limit_second'] = f"{recent_second} >= {self.max_orders_per_second}"
            return False
        
        # Orders in last minute
        recent_minute = sum(
            1 for ts in self.order_timestamps
            if (now - ts).total_seconds() < 60
        )
        
        if recent_minute >= self.max_orders_per_minute:
            details['rate_limit_minute'] = f"{recent_minute} >= {self.max_orders_per_minute}"
            return False
        
        return True
    
    def _check_daily_loss(self, details: Dict) -> bool:
        """Check daily loss limits."""
        # Absolute loss limit
        if self.daily_pnl < -self.max_daily_loss:
            details['daily_loss_absolute'] = f"${self.daily_pnl:.0f} < -${self.max_daily_loss:.0f}"
            return False
        
        # Percentage loss limit
        loss_pct = -self.daily_pnl / self.portfolio_value
        if loss_pct > self.max_daily_loss_pct:
            details['daily_loss_pct'] = f"{loss_pct:.1%} > {self.max_daily_loss_pct:.1%}"
            return False
        
        return True
    
    def _check_portfolio_exposure(
        self,
        order: Order,
        ref_price: float,
        details: Dict,
    ) -> bool:
        """Check total portfolio exposure."""
        # Calculate current exposure
        current_exposure = sum(
            pos.notional_value for pos in self.current_positions.values()
        )
        
        # Add this order's exposure
        order_notional = order.quantity * (ref_price or order.price or 0)
        total_exposure = current_exposure + order_notional
        
        if total_exposure > self.max_portfolio_notional:
            details['exposure_violation'] = (
                f"${total_exposure:.0f} > ${self.max_portfolio_notional:.0f}"
            )
            return False
        
        return True
    
    def _try_modify_order(
        self,
        order: Order,
        ref_price: float,
        failed_checks: List[str],
    ) -> Optional[Order]:
        """Try to modify order to comply with limits."""
        modified_qty = order.quantity
        modified_price = order.price
        
        # Calculate max allowed quantity
        if 'order_size' in failed_checks:
            max_qty = self.max_order_size / (ref_price or order.price or 1)
            modified_qty = min(modified_qty, max_qty)
        
        if 'position_limits' in failed_checks:
            current_pos = self.current_positions.get(order.symbol)
            current_qty = current_pos.quantity if current_pos else 0
            
            if order.side == 'buy':
                max_resulting = self.max_position_per_symbol / ref_price
                modified_qty = min(modified_qty, max(0, max_resulting - current_qty))
            else:
                max_resulting = self.max_position_per_symbol / ref_price
                modified_qty = min(modified_qty, max(0, max_resulting + current_qty))
        
        if 'price_reasonability' in failed_checks and modified_price:
            # Adjust price to be within limits
            if modified_price > ref_price:
                modified_price = ref_price * (1 + self.max_price_deviation * 0.9)
            else:
                modified_price = ref_price * (1 - self.max_price_deviation * 0.9)
        
        # Check if modification is viable
        if modified_qty < order.quantity * 0.1:  # Less than 10% of original
            return None
        
        return Order(
            order_id=order.order_id + "_modified",
            symbol=order.symbol,
            side=order.side,
            quantity=modified_qty,
            price=modified_price,
            order_type=order.order_type,
            timestamp=order.timestamp,
        )
    
    def update_position(self, position: Position):
        """Update current position for a symbol."""
        self.current_positions[position.symbol] = position
    
    def update_pnl(self, pnl_change: float):
        """Update daily P&L."""
        self.daily_pnl += pnl_change
    
    def reset_daily(self):
        """Reset daily counters."""
        self.daily_pnl = 0
        self.order_timestamps.clear()


# =============================================================================
# KILL SWITCH / CIRCUIT BREAKER
# =============================================================================

class KillSwitch:
    """
    Automated Kill Switch (Circuit Breaker).
    
    From your research:
    - Essential loss limitation system
    - Automatically halts trading when thresholds breached
    - Must operate instantly
    
    Triggers:
    1. Maximum loss threshold
    2. Maximum drawdown
    3. Anomaly detection (unusual activity)
    4. Manual override
    5. Market conditions (VIX, circuit breakers)
    """
    
    def __init__(
        self,
        # Loss thresholds
        max_daily_loss: float = 10_000,
        max_daily_loss_pct: float = 0.03,
        max_weekly_loss: float = 25_000,
        max_drawdown: float = 0.10,  # 10%
        
        # Activity thresholds
        max_consecutive_losses: int = 10,
        max_loss_velocity: float = 1000,  # $ per minute
        
        # Market condition thresholds
        vix_halt_threshold: float = 35,
        
        # Callbacks
        on_halt: Optional[Callable] = None,
        on_warning: Optional[Callable] = None,
    ):
        self.max_daily_loss = max_daily_loss
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_weekly_loss = max_weekly_loss
        self.max_drawdown = max_drawdown
        self.max_consecutive_losses = max_consecutive_losses
        self.max_loss_velocity = max_loss_velocity
        self.vix_halt_threshold = vix_halt_threshold
        
        self.on_halt = on_halt
        self.on_warning = on_warning
        
        # State
        self.is_active = True  # Trading allowed
        self.halt_reason: Optional[str] = None
        self.warnings: List[str] = []
        
        # Tracking
        self.daily_pnl: float = 0
        self.weekly_pnl: float = 0
        self.peak_equity: float = 100_000
        self.current_equity: float = 100_000
        self.consecutive_losses: int = 0
        self.pnl_history: deque = deque(maxlen=100)
        self.loss_timestamps: deque = deque(maxlen=100)
        
        # Manual override
        self.manual_halt: bool = False
    
    def check_and_halt(self) -> Tuple[bool, Optional[str]]:
        """
        Check all halt conditions.
        
        Returns:
            (should_continue_trading, halt_reason if any)
        """
        if self.manual_halt:
            return False, "Manual halt activated"
        
        reasons = []
        
        # 1. Daily loss check
        if self.daily_pnl < -self.max_daily_loss:
            reasons.append(f"Daily loss ${-self.daily_pnl:.0f} > ${self.max_daily_loss:.0f}")
        
        daily_loss_pct = -self.daily_pnl / self.peak_equity
        if daily_loss_pct > self.max_daily_loss_pct:
            reasons.append(f"Daily loss {daily_loss_pct:.1%} > {self.max_daily_loss_pct:.1%}")
        
        # 2. Weekly loss check
        if self.weekly_pnl < -self.max_weekly_loss:
            reasons.append(f"Weekly loss ${-self.weekly_pnl:.0f} > ${self.max_weekly_loss:.0f}")
        
        # 3. Drawdown check
        current_dd = (self.peak_equity - self.current_equity) / self.peak_equity
        if current_dd > self.max_drawdown:
            reasons.append(f"Drawdown {current_dd:.1%} > {self.max_drawdown:.1%}")
        
        # 4. Consecutive losses check
        if self.consecutive_losses >= self.max_consecutive_losses:
            reasons.append(f"Consecutive losses {self.consecutive_losses} >= {self.max_consecutive_losses}")
        
        # 5. Loss velocity check
        loss_velocity = self._calculate_loss_velocity()
        if loss_velocity > self.max_loss_velocity:
            reasons.append(f"Loss velocity ${loss_velocity:.0f}/min > ${self.max_loss_velocity:.0f}/min")
        
        if reasons:
            self.is_active = False
            self.halt_reason = "; ".join(reasons)
            
            if self.on_halt:
                self.on_halt(self.halt_reason)
            
            logger.critical(f"🛑 KILL SWITCH ACTIVATED: {self.halt_reason}")
            return False, self.halt_reason
        
        # Check for warnings (approaching limits)
        self._check_warnings()
        
        return True, None
    
    def _calculate_loss_velocity(self) -> float:
        """Calculate rate of loss per minute."""
        now = datetime.now()
        
        recent_losses = [
            pnl for pnl, ts in zip(self.pnl_history, self.loss_timestamps)
            if pnl < 0 and (now - ts).total_seconds() < 60
        ]
        
        if not recent_losses:
            return 0
        
        return -sum(recent_losses)  # Positive value = loss rate
    
    def _check_warnings(self):
        """Check for approaching thresholds and issue warnings."""
        self.warnings = []
        
        # 80% of daily loss
        if self.daily_pnl < -self.max_daily_loss * 0.8:
            self.warnings.append(f"Approaching daily loss limit ({self.daily_pnl:.0f})")
        
        # 80% of drawdown
        current_dd = (self.peak_equity - self.current_equity) / self.peak_equity
        if current_dd > self.max_drawdown * 0.8:
            self.warnings.append(f"Approaching drawdown limit ({current_dd:.1%})")
        
        # 80% of consecutive losses
        if self.consecutive_losses >= self.max_consecutive_losses * 0.8:
            self.warnings.append(f"Approaching consecutive loss limit ({self.consecutive_losses})")
        
        if self.warnings and self.on_warning:
            for warning in self.warnings:
                self.on_warning(warning)
    
    def update_pnl(self, pnl_change: float, timestamp: datetime = None):
        """
        Update P&L tracking.
        
        Args:
            pnl_change: P&L change from last trade
            timestamp: Time of the trade
        """
        timestamp = timestamp or datetime.now()
        
        self.daily_pnl += pnl_change
        self.weekly_pnl += pnl_change
        self.current_equity += pnl_change
        
        self.pnl_history.append(pnl_change)
        self.loss_timestamps.append(timestamp)
        
        # Update peak
        if self.current_equity > self.peak_equity:
            self.peak_equity = self.current_equity
        
        # Track consecutive losses
        if pnl_change < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
    
    def update_market_conditions(self, vix: float):
        """Check market-wide circuit breakers."""
        if vix > self.vix_halt_threshold:
            self.is_active = False
            self.halt_reason = f"VIX {vix:.1f} > {self.vix_halt_threshold}"
            
            if self.on_halt:
                self.on_halt(self.halt_reason)
            
            logger.warning(f"⚠️ Trading halted due to high VIX: {vix:.1f}")
    
    def activate_manual_halt(self, reason: str = "Manual override"):
        """Manually halt trading."""
        self.manual_halt = True
        self.is_active = False
        self.halt_reason = reason
        logger.warning(f"🛑 Manual halt activated: {reason}")
    
    def resume_trading(self, require_confirmation: bool = True):
        """Resume trading after halt."""
        if require_confirmation:
            logger.warning("⚠️ Confirm trading resumption - risk checks passed?")
        
        self.is_active = True
        self.manual_halt = False
        self.halt_reason = None
        self.warnings = []
        logger.info("✅ Trading resumed")
    
    def reset_daily(self):
        """Reset daily counters."""
        self.daily_pnl = 0
        self.consecutive_losses = 0
        self.pnl_history.clear()
        self.loss_timestamps.clear()
        
        if not self.manual_halt:
            self.is_active = True
            self.halt_reason = None
    
    def reset_weekly(self):
        """Reset weekly counters."""
        self.weekly_pnl = 0
        self.reset_daily()


# =============================================================================
# REAL-TIME RISK MONITOR
# =============================================================================

class RealTimeRiskMonitor:
    """
    Real-time risk monitoring system.
    
    Continuously monitors:
    - P&L and drawdown
    - Position exposure
    - Market conditions
    - System health
    
    Provides risk metrics and alerts.
    """
    
    def __init__(
        self,
        portfolio_value: float = 100_000,
        update_interval_seconds: float = 1.0,
    ):
        self.portfolio_value = portfolio_value
        self.update_interval = update_interval_seconds
        
        # Components
        self.pre_trade_checker = PreTradeRiskChecker()
        self.kill_switch = KillSwitch()
        
        # State
        self.positions: Dict[str, Position] = {}
        self.trades: List[Dict] = []
        self.metrics_history: List[RiskMetrics] = []
        
        # Running totals
        self.total_pnl = 0
        self.daily_pnl = 0
        self.peak_equity = portfolio_value
        self.current_equity = portfolio_value
        self.order_count = 0
        
        # Sync components
        self.pre_trade_checker.portfolio_value = portfolio_value
        self.kill_switch.peak_equity = portfolio_value
        self.kill_switch.current_equity = portfolio_value
    
    def check_order(self, order: Order, ref_price: float = None) -> RiskCheckReport:
        """
        Run pre-trade risk checks.
        
        Returns:
            RiskCheckReport with approval/rejection
        """
        # First check kill switch
        can_trade, halt_reason = self.kill_switch.check_and_halt()
        
        if not can_trade:
            return RiskCheckReport(
                result=RiskCheckResult.REJECTED,
                order=order,
                checks_passed=[],
                checks_failed=["kill_switch"],
                warnings=[],
                details={'halt_reason': halt_reason},
            )
        
        # Run pre-trade checks
        report = self.pre_trade_checker.check_order(order, ref_price)
        
        if report.result == RiskCheckResult.APPROVED:
            self.order_count += 1
        
        return report
    
    def record_trade(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        pnl: float = 0,
    ):
        """
        Record a completed trade.
        
        Updates all risk trackers.
        """
        trade = {
            'timestamp': datetime.now(),
            'symbol': symbol,
            'side': side,
            'quantity': quantity,
            'price': price,
            'pnl': pnl,
        }
        self.trades.append(trade)
        
        # Update P&L
        self.total_pnl += pnl
        self.daily_pnl += pnl
        self.current_equity += pnl
        
        if self.current_equity > self.peak_equity:
            self.peak_equity = self.current_equity
        
        # Update components
        self.pre_trade_checker.update_pnl(pnl)
        self.kill_switch.update_pnl(pnl)
        
        # Update position
        self._update_position(symbol, side, quantity, price)
        
        # Check kill switch after trade
        self.kill_switch.check_and_halt()
    
    def _update_position(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
    ):
        """Update position after trade."""
        current = self.positions.get(symbol)
        
        qty_change = quantity if side == 'buy' else -quantity
        
        if current is None:
            self.positions[symbol] = Position(
                symbol=symbol,
                quantity=qty_change,
                avg_entry_price=price,
                current_price=price,
            )
        else:
            new_qty = current.quantity + qty_change
            
            if new_qty == 0:
                del self.positions[symbol]
            else:
                # Update average price
                if np.sign(qty_change) == np.sign(current.quantity):
                    # Adding to position
                    total_cost = current.quantity * current.avg_entry_price + qty_change * price
                    current.avg_entry_price = total_cost / new_qty
                
                current.quantity = new_qty
                current.current_price = price
        
        # Sync with pre-trade checker
        if symbol in self.positions:
            self.pre_trade_checker.update_position(self.positions[symbol])
    
    def get_current_metrics(self) -> RiskMetrics:
        """Calculate current risk metrics."""
        # Drawdown
        drawdown = (self.peak_equity - self.current_equity) / self.peak_equity
        max_dd = max(
            m.max_drawdown for m in self.metrics_history
        ) if self.metrics_history else drawdown
        max_dd = max(max_dd, drawdown)
        
        # VaR (from recent P&L if available)
        recent_pnls = [t['pnl'] for t in self.trades[-100:] if t['pnl'] != 0]
        if len(recent_pnls) >= 10:
            var_95 = np.percentile(recent_pnls, 5)
        else:
            var_95 = -self.portfolio_value * 0.02  # Default 2%
        
        # Total exposure
        exposure = sum(pos.notional_value for pos in self.positions.values())
        
        # Risk level
        if not self.kill_switch.is_active:
            risk_level = RiskLevel.HALT
        elif drawdown > 0.08 or self.daily_pnl < -self.portfolio_value * 0.02:
            risk_level = RiskLevel.CRITICAL
        elif drawdown > 0.05 or self.daily_pnl < -self.portfolio_value * 0.01:
            risk_level = RiskLevel.HIGH
        elif drawdown > 0.02 or len(self.kill_switch.warnings) > 0:
            risk_level = RiskLevel.ELEVATED
        else:
            risk_level = RiskLevel.NORMAL
        
        metrics = RiskMetrics(
            timestamp=datetime.now(),
            total_pnl=self.total_pnl,
            daily_pnl=self.daily_pnl,
            drawdown=drawdown,
            max_drawdown=max_dd,
            var_95=var_95,
            exposure=exposure,
            position_count=len(self.positions),
            order_count=self.order_count,
            risk_level=risk_level,
        )
        
        self.metrics_history.append(metrics)
        
        return metrics
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """Get human-readable risk summary."""
        metrics = self.get_current_metrics()
        
        return {
            'status': '🟢 ACTIVE' if self.kill_switch.is_active else '🔴 HALTED',
            'risk_level': metrics.risk_level.value.upper(),
            'daily_pnl': f"${metrics.daily_pnl:,.2f}",
            'total_pnl': f"${metrics.total_pnl:,.2f}",
            'drawdown': f"{metrics.drawdown:.2%}",
            'max_drawdown': f"{metrics.max_drawdown:.2%}",
            'var_95': f"${metrics.var_95:,.2f}",
            'exposure': f"${metrics.exposure:,.2f}",
            'positions': metrics.position_count,
            'orders_today': metrics.order_count,
            'warnings': self.kill_switch.warnings,
            'halt_reason': self.kill_switch.halt_reason,
        }
    
    def print_risk_dashboard(self):
        """Print risk dashboard to console."""
        summary = self.get_risk_summary()
        
        print("\n" + "=" * 60)
        print("📊 RISK DASHBOARD")
        print("=" * 60)
        print(f"Status: {summary['status']}")
        print(f"Risk Level: {summary['risk_level']}")
        print("-" * 40)
        print(f"Daily P&L:    {summary['daily_pnl']}")
        print(f"Total P&L:    {summary['total_pnl']}")
        print(f"Drawdown:     {summary['drawdown']}")
        print(f"Max Drawdown: {summary['max_drawdown']}")
        print(f"VaR (95%):    {summary['var_95']}")
        print("-" * 40)
        print(f"Exposure:     {summary['exposure']}")
        print(f"Positions:    {summary['positions']}")
        print(f"Orders Today: {summary['orders_today']}")
        
        if summary['warnings']:
            print("-" * 40)
            print("⚠️ WARNINGS:")
            for w in summary['warnings']:
                print(f"  - {w}")
        
        if summary['halt_reason']:
            print("-" * 40)
            print(f"🛑 HALT REASON: {summary['halt_reason']}")
        
        print("=" * 60)
    
    def reset_daily(self):
        """Reset daily metrics."""
        self.daily_pnl = 0
        self.order_count = 0
        self.pre_trade_checker.reset_daily()
        self.kill_switch.reset_daily()
        logger.info("Daily risk metrics reset")


# =============================================================================
# SMART ORDER ROUTER (SOR) PLACEHOLDER
# =============================================================================

class SmartOrderRouter:
    """
    Smart Order Router Placeholder.
    
    From your research:
    - Decides optimal execution venue and method
    - Minimizes slippage and market impact
    - Uses real-time market microstructure data
    
    Note: Full SOR implementation requires:
    - Multiple exchange connections
    - Real-time market data from all venues
    - Sub-millisecond latency
    
    This is a placeholder for the routing logic.
    """
    
    def __init__(
        self,
        venues: List[str] = None,
        default_venue: str = "primary",
    ):
        self.venues = venues or ["primary", "secondary"]
        self.default_venue = default_venue
        
        # Venue statistics (would be real-time in production)
        self.venue_stats = {
            venue: {
                'avg_latency_ms': 1.0,
                'fill_rate': 0.95,
                'avg_slippage_bps': 1.0,
            }
            for venue in self.venues
        }
    
    def route_order(
        self,
        order: Order,
        market_data: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Determine optimal routing for an order.
        
        In production, this would consider:
        - Real-time spreads at each venue
        - Order book depth
        - Historical fill rates
        - Latency to each venue
        - Maker/taker fees
        
        Returns:
            Routing decision with venue and execution strategy
        """
        # Simple routing logic (placeholder)
        best_venue = self.default_venue
        best_score = float('inf')
        
        for venue, stats in self.venue_stats.items():
            # Score = latency + slippage (lower is better)
            score = stats['avg_latency_ms'] + stats['avg_slippage_bps']
            if score < best_score:
                best_score = score
                best_venue = venue
        
        # Determine execution strategy
        if order.quantity * (order.price or 100) > 50000:
            # Large order - use TWAP/VWAP
            execution_strategy = "TWAP"
            slice_count = 5
        else:
            execution_strategy = "immediate"
            slice_count = 1
        
        return {
            'venue': best_venue,
            'execution_strategy': execution_strategy,
            'slice_count': slice_count,
            'expected_slippage_bps': self.venue_stats[best_venue]['avg_slippage_bps'],
            'expected_latency_ms': self.venue_stats[best_venue]['avg_latency_ms'],
        }
