"""
Stock Trading Dashboard - Backend Server.

Adapted from the crypto trading server for US stocks.

Features:
- Multiple stock support (AAPL, TSLA, MSFT, etc.)
- Simulated paper trading (no real orders)
- SQLite persistence for trades/positions
- Yahoo Finance data integration
- Real-time WebSocket updates

Usage:
    python dashboard/stock_server.py
    
Runs on port 8001 (crypto runs on 8000)
"""

import asyncio
import json
import logging
import sys
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from collections import deque
from contextlib import asynccontextmanager
from enum import Enum
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from dashboard.stock_data_handler import StockDataHandler, StockSentiment
from dashboard.database import Database, TradeRecord, PositionRecord, DailyPnLRecord
from features import MicrostructureFeatures
from strategies import MicroStrategyEnsemble

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class StockTradingConfig:
    """Stock trading configuration."""
    # Rate limiting
    max_new_positions_per_window: int = 5
    rate_limit_window_seconds: int = 60
    min_seconds_between_trades: int = 15
    cooldown_after_loss_seconds: int = 60
    
    # Fees (typical broker fees for stocks)
    fee_per_trade: float = 0.0  # Most brokers are commission-free now
    sec_fee_rate: float = 0.0000278  # SEC fee per $ of sale
    
    # Position limits
    max_positions_per_stock: int = 5
    max_total_positions: int = 15
    
    # Risk
    capital: float = 1000.0
    max_position_pct: float = 0.15  # 15% per position
    min_confidence: float = 0.80  # 80% confidence required
    
    # Trend Filter
    enable_trend_filter: bool = True
    
    # Trailing Stop Settings
    trailing_stop_atr_mult: float = 2.0  # 2x ATR for stocks (larger moves)
    breakeven_threshold_pct: float = 0.5  # Move to breakeven after 50% to target
    enable_trailing_stops: bool = True
    
    # Data settings
    data_refresh_seconds: int = 60  # Refresh Yahoo data every 60s


# =============================================================================
# DATA STRUCTURES
# =============================================================================

class PositionSide(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"


@dataclass
class Position:
    """Trading position with trailing stop support."""
    id: str
    symbol: str
    side: PositionSide
    entry_price: float
    size: float
    entry_time: datetime
    stop_loss: float
    take_profit: float
    unrealized_pnl: float = 0.0
    highest_price: float = 0.0
    lowest_price: float = 0.0
    initial_stop_loss: float = 0.0
    trailing_active: bool = False
    
    def to_dict(self):
        return {
            'id': self.id,
            'symbol': self.symbol,
            'side': self.side.value,
            'entry_price': self.entry_price,
            'size': self.size,
            'entry_time': self.entry_time.strftime('%H:%M:%S'),
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'unrealized_pnl': self.unrealized_pnl,
            'trailing_active': self.trailing_active,
        }
    
    def to_db_record(self) -> PositionRecord:
        """Convert to database record."""
        return PositionRecord(
            id=self.id,
            symbol=self.symbol,
            side=self.side.value,
            entry_price=self.entry_price,
            size=self.size,
            entry_time=self.entry_time.strftime('%H:%M:%S'),
            stop_loss=self.stop_loss,
            take_profit=self.take_profit,
            highest_price=self.highest_price,
            lowest_price=self.lowest_price,
            initial_stop_loss=self.initial_stop_loss,
            trailing_active=self.trailing_active,
        )


@dataclass
class Trade:
    """Completed trade."""
    id: str
    symbol: str
    side: str
    entry_price: float
    exit_price: float
    size: float
    gross_pnl: float
    fee: float
    net_pnl: float
    entry_time: str
    exit_time: str
    reason: str
    
    def to_dict(self):
        return asdict(self)
    
    def to_db_record(self) -> TradeRecord:
        """Convert to database record."""
        return TradeRecord(
            id=self.id,
            symbol=self.symbol,
            side=self.side,
            entry_price=self.entry_price,
            exit_price=self.exit_price,
            size=self.size,
            gross_pnl=self.gross_pnl,
            fee=self.fee,
            net_pnl=self.net_pnl,
            entry_time=self.entry_time,
            exit_time=self.exit_time,
            reason=self.reason,
        )


@dataclass
class StockState:
    """State for a single stock."""
    symbol: str
    price: float = 0.0
    price_change_pct: float = 0.0
    positions: List[Position] = field(default_factory=list)
    prices: List[float] = field(default_factory=list)
    timestamps: List[str] = field(default_factory=list)
    
    # Auto-trading control
    auto_trade_enabled: bool = False
    
    # Signals
    signal_direction: str = "HOLD"
    signal_confidence: float = 0.0
    signal_strength: float = 0.0
    
    # Chart data (OHLCV candles)
    candles: List[dict] = field(default_factory=list)
    
    # Indicators for visualization
    indicators: dict = field(default_factory=dict)
    
    # Current ATR for trailing stops
    current_atr: float = 0.0


@dataclass  
class DashboardState:
    """Full dashboard state."""
    timestamp: str
    connected: bool
    market_type: str  # 'stocks' or 'crypto'
    
    # Active stock
    active_symbol: str
    stocks: Dict[str, dict]
    
    # Global stats
    total_pnl: float
    total_fees: float
    net_pnl: float
    win_rate: float
    total_trades: int
    open_positions: int
    
    # Rate limiting
    trades_in_window: int
    can_trade: bool
    next_trade_in: int
    
    # Market status
    market_open: bool
    
    # Recent trades (all stocks)
    trades: List[dict]


# =============================================================================
# TRADING ENGINE
# =============================================================================

class StockTradingEngine:
    """Stock trading engine with SQLite persistence."""
    
    DEFAULT_STOCKS = ['AAPL', 'TSLA', 'MSFT']
    
    def __init__(self, config: StockTradingConfig = None):
        self.config = config or StockTradingConfig()
        
        # Database for persistence
        db_path = os.path.join(os.path.dirname(__file__), "stock_trading.db")
        self.db = Database(db_path)
        
        # Stocks
        self.stocks: Dict[str, StockState] = {}
        self.active_symbol = 'AAPL'
        
        # Components
        self.data_handler: Optional[StockDataHandler] = None
        self.feature_generators: Dict[str, MicrostructureFeatures] = {}
        self.strategies: Dict[str, MicroStrategyEnsemble] = {}
        
        # Global state
        self.all_positions: List[Position] = []
        self.all_trades: List[Trade] = []
        self.total_fees: float = 0.0
        
        # Load capital from database or use default
        self.capital = self.db.get_capital(self.config.capital)
        
        # Rate limiting
        self.trade_timestamps: deque = deque(maxlen=100)
        self.last_trade_time: Optional[datetime] = None
        self.last_loss_time: Optional[datetime] = None
        self.consecutive_losses: int = 0
        
        # Position ID counter
        self._position_id = 0
        
        # WebSocket clients
        self.clients: List[WebSocket] = []
        self.is_running = False
        
        # Last data refresh time
        self.last_data_refresh: datetime = datetime.min
    
    def _next_position_id(self) -> str:
        self._position_id += 1
        return f"STK-{self._position_id:04d}"
    
    async def initialize(self):
        """Initialize all components."""
        logger.info("Initializing Stock Trading Engine...")
        
        # Initialize data handler
        self.data_handler = StockDataHandler(self.DEFAULT_STOCKS)
        
        # Preload historical data
        logger.info("Loading historical stock data (this may take a moment)...")
        self.data_handler.preload_historical(days=5, interval="5m")
        
        # Initialize stocks
        for symbol in self.DEFAULT_STOCKS:
            self._add_stock(symbol)
        
        # Load persisted data from database
        await self._load_from_database()
        
        logger.info(f"Initialized {len(self.stocks)} stocks")
        logger.info(f"Capital: ${self.capital:.2f}")
        logger.info(f"Recovered {len(self.all_positions)} open positions")
    
    def _add_stock(self, symbol: str):
        """Add a stock to track."""
        if symbol in self.stocks:
            return
        
        self.stocks[symbol] = StockState(symbol=symbol)
        self.feature_generators[symbol] = MicrostructureFeatures()
        self.strategies[symbol] = MicroStrategyEnsemble(
            capital=self.config.capital,
            aggressive=False,  # Less aggressive for stocks
        )
        
        logger.info(f"Added stock: {symbol}")
    
    async def _load_from_database(self):
        """Load persisted positions and trades from database."""
        # Load positions
        db_positions = self.db.load_all_positions()
        for db_pos in db_positions:
            try:
                pos = Position(
                    id=db_pos.id,
                    symbol=db_pos.symbol,
                    side=PositionSide(db_pos.side),
                    entry_price=db_pos.entry_price,
                    size=db_pos.size,
                    entry_time=datetime.strptime(db_pos.entry_time, '%H:%M:%S').replace(
                        year=datetime.now().year,
                        month=datetime.now().month,
                        day=datetime.now().day
                    ),
                    stop_loss=db_pos.stop_loss,
                    take_profit=db_pos.take_profit,
                    highest_price=db_pos.highest_price,
                    lowest_price=db_pos.lowest_price,
                    initial_stop_loss=db_pos.initial_stop_loss,
                    trailing_active=db_pos.trailing_active,
                )
                
                self.all_positions.append(pos)
                
                if pos.symbol in self.stocks:
                    self.stocks[pos.symbol].positions.append(pos)
                
                # Update position ID counter
                try:
                    pos_num = int(pos.id.split('-')[1])
                    self._position_id = max(self._position_id, pos_num)
                except:
                    pass
                    
            except Exception as e:
                logger.error(f"Error loading position {db_pos.id}: {e}")
        
        # Load recent trades for display
        db_trades = self.db.load_all_trades(limit=100)
        for db_trade in db_trades:
            trade = Trade(
                id=db_trade.id,
                symbol=db_trade.symbol,
                side=db_trade.side,
                entry_price=db_trade.entry_price,
                exit_price=db_trade.exit_price,
                size=db_trade.size,
                gross_pnl=db_trade.gross_pnl,
                fee=db_trade.fee,
                net_pnl=db_trade.net_pnl,
                entry_time=db_trade.entry_time,
                exit_time=db_trade.exit_time,
                reason=db_trade.reason,
            )
            self.all_trades.append(trade)
        
        # Load totals from database
        self.total_fees = self.db.get_total_fees()
    
    async def start(self):
        """Start trading loop."""
        await self.initialize()
        
        self.is_running = True
        logger.info("Stock trading loop started")
        
        while self.is_running:
            try:
                await self._trading_iteration()
            except Exception as e:
                logger.error(f"Trading error: {e}")
            
            await asyncio.sleep(3)  # 3-second tick
    
    async def stop(self):
        """Stop trading."""
        self.is_running = False
        
        # Save capital
        self.db.set_capital(self.capital)
        
        logger.info("Stock trading stopped")
    
    async def _trading_iteration(self):
        """Process all stocks."""
        # Refresh data periodically
        now = datetime.now()
        if (now - self.last_data_refresh).seconds >= self.config.data_refresh_seconds:
            if self.data_handler.is_market_open():
                self.data_handler.refresh_data(interval="1m")
            self.last_data_refresh = now
        
        # Process each stock
        for symbol in self.stocks:
            await self._process_stock(symbol)
        
        # Broadcast state
        state = self._build_state()
        await self._broadcast(state)
    
    async def _process_stock(self, symbol: str):
        """Process a single stock."""
        if not self.data_handler:
            return
        
        # Get data
        df = self.data_handler.get_buffer(symbol, n_bars=100)
        if len(df) < 30:
            return
        
        current_price = float(df['close'].iloc[-1])
        stock = self.stocks[symbol]
        
        # Update price history
        stock.price = current_price
        stock.prices.append(current_price)
        stock.timestamps.append(datetime.now().strftime('%H:%M:%S'))
        
        if len(stock.prices) > 100:
            stock.prices.pop(0)
            stock.timestamps.pop(0)
        
        # Price change
        if len(stock.prices) >= 2:
            stock.price_change_pct = (current_price - stock.prices[0]) / stock.prices[0] * 100
        
        # Calculate ATR
        if len(df) >= 14:
            high = df['high'].values
            low = df['low'].values
            close = df['close'].values
            
            tr_values = []
            for i in range(1, min(15, len(df))):
                tr = max(
                    high[-i] - low[-i],
                    abs(high[-i] - close[-i-1]) if i < len(close) else 0,
                    abs(low[-i] - close[-i-1]) if i < len(close) else 0
                )
                tr_values.append(tr)
            
            stock.current_atr = sum(tr_values) / len(tr_values) if tr_values else 0
        
        # Add features
        df_features = self.feature_generators[symbol].add_all_features(df)
        
        # Build candle data with indicators
        candles = []
        for i in range(max(0, len(df_features) - 100), len(df_features)):
            c_time = df_features.index[i].strftime('%H:%M:%S') if hasattr(df_features.index[i], 'strftime') else str(i)
            
            def get_val(col):
                if col in df_features.columns:
                    val = df_features[col].iloc[i]
                    return float(val) if not pd.isna(val) else None
                return None

            candles.append({
                'time': c_time,
                'open': float(df_features['open'].iloc[i]),
                'high': float(df_features['high'].iloc[i]),
                'low': float(df_features['low'].iloc[i]),
                'close': float(df_features['close'].iloc[i]),
                'volume': float(df_features['volume'].iloc[i]),
                'rsi': get_val('rsi'),
                'macd': get_val('macd'),
                'macd_signal': get_val('macd_signal'),
                'bb_upper': get_val('bb_upper'),
                'bb_middle': get_val('bb_middle'),
                'bb_lower': get_val('bb_lower'),
                'ema_fast': get_val('ema_fast'),
                'ema_slow': get_val('ema_slow'),
                'adx': get_val('adx'),
            })
        stock.candles = candles
        
        # Extract latest indicators
        if candles:
            stock.indicators = {
                'rsi': candles[-1]['rsi'],
                'macd': candles[-1]['macd'],
                'macd_signal': candles[-1]['macd_signal'],
                'bb_upper': candles[-1]['bb_upper'],
                'bb_middle': candles[-1]['bb_middle'],
                'bb_lower': candles[-1]['bb_lower'],
                'ema_fast': candles[-1]['ema_fast'],
                'ema_slow': candles[-1]['ema_slow'],
                'adx': candles[-1]['adx'],
            }
        
        # Get sentiment
        sentiment = self.data_handler.get_sentiment(symbol)
        
        class SentimentWrapper:
            def __init__(self, s):
                self.volume_imbalance = s.volume_imbalance
                self.trade_imbalance = s.trade_imbalance
                self.whale_signal = s.whale_signal
        
        # Generate signal
        signal = self.strategies[symbol].generate_signal(df_features, SentimentWrapper(sentiment))
        
        stock.signal_direction = "BUY" if signal.direction == 1 else "SELL" if signal.direction == -1 else "HOLD"
        stock.signal_confidence = signal.confidence
        stock.signal_strength = signal.strength
        
        # Manage existing positions
        await self._manage_positions(symbol, current_price)
        
        # Look for new entries
        if (stock.auto_trade_enabled and 
            self._can_open_position() and 
            signal.direction != 0 and 
            signal.confidence >= self.config.min_confidence):
            
            # Trend filter
            if self.config.enable_trend_filter and stock.indicators.get('ema_fast'):
                ema = stock.indicators['ema_fast']
                if signal.direction == 1 and current_price < ema:
                    logger.info(f"🚫 TREND FILTER: Blocked LONG {symbol}")
                    return
                elif signal.direction == -1 and current_price > ema:
                    logger.info(f"🚫 TREND FILTER: Blocked SHORT {symbol}")
                    return
            
            # Momentum filter
            if len(stock.prices) >= 10:
                recent_prices = stock.prices[-10:]
                price_momentum = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
                
                if signal.direction == -1 and price_momentum > 0.001:
                    logger.info(f"🚫 MOMENTUM FILTER: Blocked SHORT {symbol}")
                    return
                elif signal.direction == 1 and price_momentum < -0.001:
                    logger.info(f"🚫 MOMENTUM FILTER: Blocked LONG {symbol}")
                    return
            
            await self._open_position(symbol, signal, current_price)
    
    def _can_open_position(self) -> bool:
        """Check rate limiting."""
        now = datetime.now()
        
        if self.last_loss_time:
            cooldown = 300 if self.consecutive_losses >= 3 else self.config.cooldown_after_loss_seconds
            if now < self.last_loss_time + timedelta(seconds=cooldown):
                return False
        
        if self.last_trade_time:
            if now < self.last_trade_time + timedelta(seconds=self.config.min_seconds_between_trades):
                return False
        
        window_start = now - timedelta(seconds=self.config.rate_limit_window_seconds)
        recent_trades = [t for t in self.trade_timestamps if t > window_start]
        
        if len(recent_trades) >= self.config.max_new_positions_per_window:
            return False
        
        if len(self.all_positions) >= self.config.max_total_positions:
            return False
        
        return True
    
    def _get_next_trade_in(self) -> int:
        """Get seconds until next trade allowed."""
        now = datetime.now()
        
        if self.last_loss_time:
            cooldown = 300 if self.consecutive_losses >= 3 else self.config.cooldown_after_loss_seconds
            cooldown_end = self.last_loss_time + timedelta(seconds=cooldown)
            if now < cooldown_end:
                return int((cooldown_end - now).total_seconds())
        
        if self.last_trade_time:
            min_time = self.last_trade_time + timedelta(seconds=self.config.min_seconds_between_trades)
            if now < min_time:
                return int((min_time - now).total_seconds())
        
        return 0
    
    def _trades_in_window(self) -> int:
        """Count trades in current window."""
        now = datetime.now()
        window_start = now - timedelta(seconds=self.config.rate_limit_window_seconds)
        return len([t for t in self.trade_timestamps if t > window_start])
    
    async def _open_position(self, symbol: str, signal, price: float):
        """Open a new position."""
        side = PositionSide.LONG if signal.direction == 1 else PositionSide.SHORT
        
        position_value = self.capital * self.config.max_position_pct
        size = position_value / price
        
        position = Position(
            id=self._next_position_id(),
            symbol=symbol,
            side=side,
            entry_price=price,
            size=size,
            entry_time=datetime.now(),
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
        )
        
        self.all_positions.append(position)
        self.stocks[symbol].positions.append(position)
        
        # Persist to database
        self.db.save_position(position.to_db_record())
        
        # Track for rate limiting
        self.trade_timestamps.append(datetime.now())
        self.last_trade_time = datetime.now()
        
        logger.info(f"📈 Opened {side.value} {symbol} @ ${price:.2f} [{position.id}]")
    
    async def _manage_positions(self, symbol: str, current_price: float):
        """Manage positions with trailing stops."""
        positions_to_close = []
        stock = self.stocks.get(symbol)
        atr = stock.current_atr if stock else 0
        
        for pos in self.all_positions:
            if pos.symbol != symbol:
                continue
            
            # Initialize tracking
            if pos.highest_price == 0:
                pos.highest_price = current_price
                pos.initial_stop_loss = pos.stop_loss
            if pos.lowest_price == 0:
                pos.lowest_price = current_price
            
            # Calculate P&L
            if pos.side == PositionSide.LONG:
                pos.unrealized_pnl = (current_price - pos.entry_price) * pos.size
                pos.highest_price = max(pos.highest_price, current_price)
                
                # Trailing stop
                target_distance = pos.take_profit - pos.entry_price
                current_profit = current_price - pos.entry_price
                progress_pct = current_profit / target_distance if target_distance > 0 else 0
                
                if self.config.enable_trailing_stops and atr > 0 and progress_pct > self.config.breakeven_threshold_pct:
                    pos.trailing_active = True
                    trailing_stop = pos.highest_price - (atr * self.config.trailing_stop_atr_mult)
                    if trailing_stop > pos.stop_loss:
                        pos.stop_loss = trailing_stop
                        self.db.update_position(pos.to_db_record())
                
                hit_stop = current_price <= pos.stop_loss
                hit_target = current_price >= pos.take_profit
                
            else:  # SHORT
                pos.unrealized_pnl = (pos.entry_price - current_price) * pos.size
                pos.lowest_price = min(pos.lowest_price, current_price)
                
                # Trailing stop
                target_distance = pos.entry_price - pos.take_profit
                current_profit = pos.entry_price - current_price
                progress_pct = current_profit / target_distance if target_distance > 0 else 0
                
                if self.config.enable_trailing_stops and atr > 0 and progress_pct > self.config.breakeven_threshold_pct:
                    pos.trailing_active = True
                    trailing_stop = pos.lowest_price + (atr * self.config.trailing_stop_atr_mult)
                    if trailing_stop < pos.stop_loss:
                        pos.stop_loss = trailing_stop
                        self.db.update_position(pos.to_db_record())
                
                hit_stop = current_price >= pos.stop_loss
                hit_target = current_price <= pos.take_profit
            
            # Check exits
            reason = None
            if hit_stop:
                reason = "Trailing Stop" if pos.trailing_active else "Stop Loss"
            elif hit_target:
                reason = "Take Profit"
            elif (datetime.now() - pos.entry_time).seconds > 3600:  # 1 hour timeout for stocks
                reason = "Timeout"
            
            if reason:
                positions_to_close.append((pos, current_price, reason))
        
        for pos, price, reason in positions_to_close:
            await self._close_position(pos, price, reason)
    
    async def _close_position(self, pos: Position, exit_price: float, reason: str):
        """Close a position."""
        # Calculate P&L
        if pos.side == PositionSide.LONG:
            gross_pnl = (exit_price - pos.entry_price) * pos.size
        else:
            gross_pnl = (pos.entry_price - exit_price) * pos.size
        
        # Calculate fee (SEC fee on sales only)
        fee = 0.0
        if pos.side == PositionSide.SHORT or exit_price > pos.entry_price:
            fee = exit_price * pos.size * self.config.sec_fee_rate
        
        net_pnl = gross_pnl - fee
        
        # Update capital
        self.capital += net_pnl
        self.total_fees += fee
        
        # Save capital to database
        self.db.set_capital(self.capital)
        
        # Track losses
        if net_pnl < 0:
            self.last_loss_time = datetime.now()
            self.consecutive_losses += 1
            if self.consecutive_losses >= 3:
                logger.warning(f"⚠️ 3 Consecutive losses - 5 minute cooldown")
        else:
            self.consecutive_losses = 0
        
        # Record trade
        trade = Trade(
            id=pos.id,
            symbol=pos.symbol,
            side=pos.side.value,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            size=pos.size,
            gross_pnl=gross_pnl,
            fee=fee,
            net_pnl=net_pnl,
            entry_time=pos.entry_time.strftime('%H:%M:%S'),
            exit_time=datetime.now().strftime('%H:%M:%S'),
            reason=reason,
        )
        self.all_trades.append(trade)
        
        # Persist trade to database
        self.db.save_trade(trade.to_db_record())
        
        # Remove position from database
        self.db.delete_position(pos.id)
        
        # Remove from lists
        self.all_positions.remove(pos)
        if pos in self.stocks[pos.symbol].positions:
            self.stocks[pos.symbol].positions.remove(pos)
        
        result = "✅ WIN" if net_pnl > 0 else "❌ LOSS"
        logger.info(f"{result} Closed {pos.side.value} {pos.symbol} @ ${exit_price:.2f} | Net: ${net_pnl:+.2f}")
    
    def _build_state(self) -> DashboardState:
        """Build state for dashboard."""
        # Calculate totals from database for accuracy
        stats = self.db.get_statistics()
        
        total_pnl = self.capital - self.config.capital
        net_pnl = total_pnl
        
        win_rate = stats['win_rate']
        
        # Build stock states
        stocks_dict = {}
        for symbol, stock in self.stocks.items():
            stocks_dict[symbol] = {
                'symbol': symbol,
                'price': stock.price,
                'price_change_pct': stock.price_change_pct,
                'positions': [p.to_dict() for p in stock.positions],
                'prices': stock.prices[-50:],
                'timestamps': stock.timestamps[-50:],
                'candles': stock.candles,
                'indicators': stock.indicators,
                'signal_direction': stock.signal_direction,
                'signal_confidence': stock.signal_confidence,
                'signal_strength': stock.signal_strength,
                'position_count': len(stock.positions),
                'long_count': len([p for p in stock.positions if p.side == PositionSide.LONG]),
                'short_count': len([p for p in stock.positions if p.side == PositionSide.SHORT]),
                'auto_trade_enabled': stock.auto_trade_enabled,
                'current_atr': stock.current_atr,
            }
        
        return DashboardState(
            timestamp=datetime.now().isoformat(),
            connected=True,
            market_type='stocks',
            active_symbol=self.active_symbol,
            stocks=stocks_dict,
            total_pnl=total_pnl,
            total_fees=self.total_fees,
            net_pnl=net_pnl,
            win_rate=win_rate,
            total_trades=len(self.all_trades),
            open_positions=len(self.all_positions),
            trades_in_window=self._trades_in_window(),
            can_trade=self._can_open_position(),
            next_trade_in=self._get_next_trade_in(),
            market_open=self.data_handler.is_market_open() if self.data_handler else False,
            trades=[t.to_dict() for t in self.all_trades[-20:]],
        )
    
    async def _broadcast(self, state: DashboardState):
        """Send state to all clients."""
        if not self.clients:
            return
        
        data = json.dumps(asdict(state))
        
        disconnected = []
        for client in self.clients:
            try:
                await client.send_text(data)
            except:
                disconnected.append(client)
        
        for client in disconnected:
            self.clients.remove(client)
    
    async def add_stock(self, symbol: str):
        """Add a new stock."""
        symbol = symbol.upper()
        
        if symbol in self.stocks:
            return
        
        self._add_stock(symbol)
        
        if self.data_handler:
            self.data_handler.add_symbol(symbol, days=5, interval="5m")
    
    def set_active_stock(self, symbol: str):
        """Set the active stock."""
        if symbol in self.stocks:
            self.active_symbol = symbol


# =============================================================================
# FASTAPI APP
# =============================================================================

engine: Optional[StockTradingEngine] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    engine = StockTradingEngine()
    asyncio.create_task(engine.start())
    yield
    if engine:
        await engine.stop()


app = FastAPI(title="Stock Trading Dashboard", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    if engine:
        engine.clients.append(websocket)
        logger.info(f"Stock client connected ({len(engine.clients)} total)")
    
    try:
        while True:
            data = await websocket.receive_text()
            
            try:
                cmd = json.loads(data)
                if cmd.get('action') == 'set_active':
                    engine.set_active_stock(cmd.get('symbol', 'AAPL'))
                elif cmd.get('action') == 'add_stock':
                    await engine.add_stock(cmd.get('symbol', ''))
                elif cmd.get('action') == 'toggle_auto_trade':
                    symbol = cmd.get('symbol', '')
                    if symbol in engine.stocks:
                        engine.stocks[symbol].auto_trade_enabled = not engine.stocks[symbol].auto_trade_enabled
                        status = "enabled" if engine.stocks[symbol].auto_trade_enabled else "disabled"
                        logger.info(f"Auto-trade {status} for {symbol}")
            except:
                pass
                
    except WebSocketDisconnect:
        if engine and websocket in engine.clients:
            engine.clients.remove(websocket)
        logger.info("Stock client disconnected")


@app.get("/api/status")
async def get_status():
    if engine:
        return {
            "running": engine.is_running,
            "market_type": "stocks",
            "stocks": list(engine.stocks.keys()),
            "positions": len(engine.all_positions),
            "trades": len(engine.all_trades),
            "capital": engine.capital,
            "market_open": engine.data_handler.is_market_open() if engine.data_handler else False,
        }
    return {"running": False}


@app.get("/api/statistics")
async def get_statistics():
    if engine:
        return engine.db.get_statistics()
    return {}


@app.post("/api/add_stock/{symbol}")
async def add_stock(symbol: str):
    if engine:
        await engine.add_stock(symbol)
        return {"success": True, "symbol": symbol}
    return {"success": False}


if __name__ == "__main__":
    uvicorn.run("stock_server:app", host="0.0.0.0", port=8001, reload=False)
