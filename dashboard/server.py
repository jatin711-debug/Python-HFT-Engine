"""
Pro Crypto Trading Dashboard - Backend Server.

Features:
- Multiple coin support (BTC, ETH, SOL, etc.)
- Multi-position trading (unlimited LONGs + SHORTs)
- Trade rate limiting (max 5 new positions per 60 seconds)
- Fee tracking
- Real-time WebSocket updates

Usage:
    python dashboard/server.py
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from data.fetchers import BinanceDataHandler
from features import MicrostructureFeatures
from strategies import MicroStrategyEnsemble

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class TradingConfig:
    """Trading configuration."""
    # Rate limiting
    max_new_positions_per_window: int = 5
    rate_limit_window_seconds: int = 60
    min_seconds_between_trades: int = 10
    cooldown_after_loss_seconds: int = 30
    
    # Fees
    fee_rate: float = 0.001  # 0.1% per trade
    
    # Position limits
    max_positions_per_coin: int = 10
    max_total_positions: int = 20
    
    # Risk
    capital: float = 500.0
    max_position_pct: float = 0.10  # 10% per position
    min_confidence: float = 0.70  # Increased from 0.5 to 0.70 (70%)
    
    # Trailing Stop Settings
    trailing_stop_atr_mult: float = 1.5  # 1.5x ATR trailing distance
    breakeven_threshold_pct: float = 0.3  # Move to breakeven after 30% to target
    enable_trailing_stops: bool = True


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
    # Trailing stop tracking
    highest_price: float = 0.0  # Track highest price since entry (for LONG)
    lowest_price: float = 0.0  # Track lowest price since entry (for SHORT) - 0 means uninitialized
    initial_stop_loss: float = 0.0  # Original stop loss before trailing
    trailing_active: bool = False  # Whether trailing stop is active
    
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


@dataclass
class CoinState:
    """State for a single coin."""
    symbol: str
    price: float = 0.0
    price_change_pct: float = 0.0
    positions: List[Position] = field(default_factory=list)
    prices: List[float] = field(default_factory=list)
    timestamps: List[str] = field(default_factory=list)
    
    # Auto-trading control
    auto_trade_enabled: bool = True
    
    # Signals
    signal_direction: str = "HOLD"
    signal_confidence: float = 0.0
    signal_strength: float = 0.0
    
    # Chart data (OHLCV candles) - now supports multi-timeframe
    candles: List[dict] = field(default_factory=list)  # 1s candles (raw)
    candles_1m: List[dict] = field(default_factory=list)  # Aggregated 1m candles
    candles_5m: List[dict] = field(default_factory=list)  # Aggregated 5m candles
    
    # Indicators for visualization
    indicators: dict = field(default_factory=dict)
    
    # Current ATR for trailing stops
    current_atr: float = 0.0


@dataclass  
class DashboardState:
    """Full dashboard state."""
    timestamp: str
    connected: bool
    
    # Active coin
    active_symbol: str
    coins: Dict[str, dict]
    
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
    next_trade_in: int  # seconds
    
    # Recent trades (all coins)
    trades: List[dict]


# =============================================================================
# TRADING ENGINE
# =============================================================================

class ProTradingEngine:
    """Multi-coin, multi-position trading engine."""
    
    DEFAULT_COINS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']
    
    def __init__(self, config: TradingConfig = None):
        self.config = config or TradingConfig()
        
        # Coins
        self.coins: Dict[str, CoinState] = {}
        self.active_symbol = 'BTCUSDT'
        
        # Components per coin
        self.data_handlers: Dict[str, BinanceDataHandler] = {}
        self.feature_generators: Dict[str, MicrostructureFeatures] = {}
        self.strategies: Dict[str, MicroStrategyEnsemble] = {}
        
        # Global state
        self.all_positions: List[Position] = []
        self.all_trades: List[Trade] = []
        self.total_fees: float = 0.0
        self.capital = self.config.capital
        
        # Rate limiting
        self.trade_timestamps: deque = deque(maxlen=100)
        self.last_trade_time: Optional[datetime] = None
        self.last_loss_time: Optional[datetime] = None
        
        # Position ID counter
        self._position_id = 0
        
        # WebSocket clients
        self.clients: List[WebSocket] = []
        self.is_running = False
    
    def _next_position_id(self) -> str:
        self._position_id += 1
        return f"POS-{self._position_id:04d}"
    
    async def initialize(self):
        """Initialize all components."""
        logger.info("Initializing Pro Trading Engine...")
        
        for symbol in self.DEFAULT_COINS:
            await self._add_coin(symbol)
        
        logger.info(f"Initialized {len(self.coins)} coins")
    
    async def _add_coin(self, symbol: str):
        """Add a coin to track."""
        if symbol in self.coins:
            return
        
        self.coins[symbol] = CoinState(symbol=symbol)
        
        # Data handler
        handler = BinanceDataHandler([symbol])
        handler.preload_historical(interval='1m', n_candles=200)
        self.data_handlers[symbol] = handler
        
        # Features
        self.feature_generators[symbol] = MicrostructureFeatures()
        
        # Strategy
        self.strategies[symbol] = MicroStrategyEnsemble(
            capital=self.config.capital,
            aggressive=True,
        )
        
        logger.info(f"Added coin: {symbol}")
    
    async def start(self):
        """Start trading loop."""
        await self.initialize()
        
        # Connect all data handlers
        for symbol, handler in self.data_handlers.items():
            await handler.connect()
            await handler.subscribe_klines('1s')
            await handler.subscribe_trades()
        
        self.is_running = True
        logger.info("Trading loop started")
        
        while self.is_running:
            try:
                await self._trading_iteration()
            except Exception as e:
                logger.error(f"Trading error: {e}")
            
            await asyncio.sleep(3)  # 3-second tick
    
    async def stop(self):
        """Stop trading."""
        self.is_running = False
        for handler in self.data_handlers.values():
            await handler.disconnect()
        logger.info("Trading stopped")
    
    async def _trading_iteration(self):
        """Process all coins."""
        for symbol in self.coins:
            await self._process_coin(symbol)
        
        # Broadcast state
        state = self._build_state()
        await self._broadcast(state)
    
    async def _process_coin(self, symbol: str):
        """Process a single coin."""
        handler = self.data_handlers.get(symbol)
        if not handler:
            return
        
        # Get data
        df = handler.get_buffer(symbol, n_bars=100, interval='1s')
        if len(df) < 60:
            return
        
        current_price = df['close'].iloc[-1]
        coin = self.coins[symbol]
        
        # Update price history
        coin.price = current_price
        coin.prices.append(current_price)
        coin.timestamps.append(datetime.now().strftime('%H:%M:%S'))
        
        if len(coin.prices) > 100:
            coin.prices.pop(0)
            coin.timestamps.pop(0)
        
        # Price change
        if len(coin.prices) >= 2:
            coin.price_change_pct = (current_price - coin.prices[0]) / coin.prices[0] * 100
        
        # Calculate current ATR for trailing stops (14-period)
        if len(df) >= 14:
            high = df['high'].values
            low = df['low'].values
            close = df['close'].values
            
            # True Range calculation
            tr_values = []
            for i in range(1, min(15, len(df))):
                tr = max(
                    high[-i] - low[-i],
                    abs(high[-i] - close[-i-1]) if i < len(close) else 0,
                    abs(low[-i] - close[-i-1]) if i < len(close) else 0
                )
                tr_values.append(tr)
            
            coin.current_atr = sum(tr_values) / len(tr_values) if tr_values else 0
        
        # Build candle data (last 100 candles for chart - supports multi-timeframe aggregation)
        candles = []
        for i in range(max(0, len(df) - 100), len(df)):
            candles.append({
                'time': df.index[i].strftime('%H:%M:%S') if hasattr(df.index[i], 'strftime') else str(i),
                'open': float(df['open'].iloc[i]),
                'high': float(df['high'].iloc[i]),
                'low': float(df['low'].iloc[i]),
                'close': float(df['close'].iloc[i]),
                'volume': float(df['volume'].iloc[i]),
            })
        coin.candles = candles
        
        # Add features
        df_features = self.feature_generators[symbol].add_all_features(df)
        
        # Extract indicators for visualization
        if len(df_features) > 0:
            last_idx = len(df_features) - 1
            coin.indicators = {
                'rsi': float(df_features['rsi'].iloc[last_idx]) if 'rsi' in df_features else None,
                'macd': float(df_features['macd'].iloc[last_idx]) if 'macd' in df_features else None,
                'macd_signal': float(df_features['macd_signal'].iloc[last_idx]) if 'macd_signal' in df_features else None,
                'bb_upper': float(df_features['bb_upper'].iloc[last_idx]) if 'bb_upper' in df_features else None,
                'bb_middle': float(df_features['bb_middle'].iloc[last_idx]) if 'bb_middle' in df_features else None,
                'bb_lower': float(df_features['bb_lower'].iloc[last_idx]) if 'bb_lower' in df_features else None,
                'ema_fast': float(df_features['ema_fast'].iloc[last_idx]) if 'ema_fast' in df_features else None,
                'ema_slow': float(df_features['ema_slow'].iloc[last_idx]) if 'ema_slow' in df_features else None,
            }
        
        # Get sentiment
        sentiment = handler.get_sentiment(symbol, lookback_seconds=10)
        
        class SentimentWrapper:
            def __init__(self, s):
                self.volume_imbalance = s.volume_imbalance
                self.trade_imbalance = s.trade_imbalance
                self.whale_signal = s.whale_signal
        
        # Generate signal
        signal = self.strategies[symbol].generate_signal(df_features, SentimentWrapper(sentiment))
        
        coin.signal_direction = "BUY" if signal.direction == 1 else "SELL" if signal.direction == -1 else "HOLD"
        coin.signal_confidence = signal.confidence
        coin.signal_strength = signal.strength
        
        # Manage existing positions
        await self._manage_positions(symbol, current_price)
        
        # Look for new entries (if AUTO-TRADE ENABLED and rate limit allows)
        if coin.auto_trade_enabled and self._can_open_position() and signal.direction != 0 and signal.confidence >= self.config.min_confidence:
            await self._open_position(symbol, signal, current_price)
    
    def _can_open_position(self) -> bool:
        """Check if we can open a new position (rate limiting)."""
        now = datetime.now()
        
        # Check cooldown after loss
        if self.last_loss_time:
            cooldown_end = self.last_loss_time + timedelta(seconds=self.config.cooldown_after_loss_seconds)
            if now < cooldown_end:
                return False
        
        # Check min time between trades
        if self.last_trade_time:
            min_time = self.last_trade_time + timedelta(seconds=self.config.min_seconds_between_trades)
            if now < min_time:
                return False
        
        # Check rate limit window
        window_start = now - timedelta(seconds=self.config.rate_limit_window_seconds)
        recent_trades = [t for t in self.trade_timestamps if t > window_start]
        
        if len(recent_trades) >= self.config.max_new_positions_per_window:
            return False
        
        # Check total positions
        if len(self.all_positions) >= self.config.max_total_positions:
            return False
        
        return True
    
    def _get_next_trade_in(self) -> int:
        """Get seconds until next trade allowed."""
        now = datetime.now()
        
        # Check cooldown
        if self.last_loss_time:
            cooldown_end = self.last_loss_time + timedelta(seconds=self.config.cooldown_after_loss_seconds)
            if now < cooldown_end:
                return int((cooldown_end - now).total_seconds())
        
        # Check min time
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
        
        # === FEE-AWARE PROFIT FILTER ===
        # Calculate expected profit and fee
        if side == PositionSide.LONG:
            expected_profit = (signal.take_profit - price) * size
        else:
            expected_profit = (price - signal.take_profit) * size
        
        # Calculate round-trip fee
        trade_value = price * size + signal.take_profit * size
        round_trip_fee = trade_value * self.config.fee_rate
        
        # BLOCK TRADE IF PROFIT < 3x FEE
        min_profit_required = round_trip_fee * 3.0
        
        if expected_profit < min_profit_required:
            logger.warning(
                f"❌ BLOCKED {side.value} {symbol}: Expected profit ${expected_profit:.2f} < "
                f"Min required ${min_profit_required:.2f} (3x fee of ${round_trip_fee:.2f})"
            )
            return
        
        # Profit is adequate - proceed with trade
        logger.info(
            f"✅ APPROVED {side.value} {symbol}: Expected profit ${expected_profit:.2f} > "
            f"Min required ${min_profit_required:.2f}"
        )
        
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
        self.coins[symbol].positions.append(position)
        
        # Track for rate limiting
        self.trade_timestamps.append(datetime.now())
        self.last_trade_time = datetime.now()
        
        logger.info(f"Opened {side.value} {symbol} @ ${price:.2f} [{position.id}]")
    
    async def _manage_positions(self, symbol: str, current_price: float):
        """Manage positions for a coin with trailing stops."""
        positions_to_close = []
        coin = self.coins.get(symbol)
        atr = coin.current_atr if coin else 0
        
        for pos in self.all_positions:
            if pos.symbol != symbol:
                continue
            
            # Initialize tracking prices on first update
            if pos.highest_price == 0:
                pos.highest_price = current_price
                pos.initial_stop_loss = pos.stop_loss
            if pos.lowest_price == 0:
                pos.lowest_price = current_price
            
            # Calculate unrealized P&L
            if pos.side == PositionSide.LONG:
                pos.unrealized_pnl = (current_price - pos.entry_price) * pos.size
                pos.highest_price = max(pos.highest_price, current_price)
                
                # Calculate progress to target
                target_distance = pos.take_profit - pos.entry_price
                current_profit = current_price - pos.entry_price
                progress_pct = current_profit / target_distance if target_distance > 0 else 0
                
                # Trailing stop logic
                if self.config.enable_trailing_stops and atr > 0 and progress_pct > self.config.breakeven_threshold_pct:
                    pos.trailing_active = True
                    # Trail 1.5x ATR behind highest price
                    trailing_stop = pos.highest_price - (atr * self.config.trailing_stop_atr_mult)
                    # Never lower the stop, only raise it
                    if trailing_stop > pos.stop_loss:
                        old_stop = pos.stop_loss
                        pos.stop_loss = trailing_stop
                        logger.debug(f"[TRAIL] {pos.id} stop raised: ${old_stop:.2f} → ${pos.stop_loss:.2f}")
                
                hit_stop = current_price <= pos.stop_loss
                hit_target = current_price >= pos.take_profit
                
            else:  # SHORT
                pos.unrealized_pnl = (pos.entry_price - current_price) * pos.size
                pos.lowest_price = min(pos.lowest_price, current_price)
                
                # Calculate progress to target
                target_distance = pos.entry_price - pos.take_profit
                current_profit = pos.entry_price - current_price
                progress_pct = current_profit / target_distance if target_distance > 0 else 0
                
                # Trailing stop logic
                if self.config.enable_trailing_stops and atr > 0 and progress_pct > self.config.breakeven_threshold_pct:
                    pos.trailing_active = True
                    # Trail 1.5x ATR above lowest price
                    trailing_stop = pos.lowest_price + (atr * self.config.trailing_stop_atr_mult)
                    # Never raise the stop for shorts, only lower it
                    if trailing_stop < pos.stop_loss:
                        old_stop = pos.stop_loss
                        pos.stop_loss = trailing_stop
                        logger.debug(f"[TRAIL] {pos.id} stop lowered: ${old_stop:.2f} → ${pos.stop_loss:.2f}")
                
                hit_stop = current_price >= pos.stop_loss
                hit_target = current_price <= pos.take_profit
            
            # Check exits
            reason = None
            if hit_stop:
                reason = "Trailing Stop" if pos.trailing_active else "Stop Loss"
            elif hit_target:
                reason = "Take Profit"
            elif (datetime.now() - pos.entry_time).seconds > 180:
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
        
        # Calculate fee
        trade_value = pos.entry_price * pos.size + exit_price * pos.size
        fee = trade_value * self.config.fee_rate
        net_pnl = gross_pnl - fee
        
        # Update capital
        self.capital += net_pnl
        self.total_fees += fee
        
        # Track loss cooldown
        if net_pnl < 0:
            self.last_loss_time = datetime.now()
        
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
        
        # Remove from positions
        self.all_positions.remove(pos)
        if pos in self.coins[pos.symbol].positions:
            self.coins[pos.symbol].positions.remove(pos)
        
        result = "WIN" if net_pnl > 0 else "LOSS"
        logger.info(f"[{result}] Closed {pos.side.value} {pos.symbol} @ ${exit_price:.2f} | Net: ${net_pnl:+.2f} [{pos.id}]")
    
    def _build_state(self) -> DashboardState:
        """Build state for dashboard."""
        # Calculate totals
        total_pnl = self.capital - self.config.capital
        net_pnl = total_pnl  # Already net after fees
        
        wins = [t for t in self.all_trades if t.net_pnl > 0]
        win_rate = len(wins) / len(self.all_trades) * 100 if self.all_trades else 0
        
        # Build coin states
        coins_dict = {}
        for symbol, coin in self.coins.items():
            coins_dict[symbol] = {
                'symbol': symbol,
                'price': coin.price,
                'price_change_pct': coin.price_change_pct,
                'positions': [p.to_dict() for p in coin.positions],
                'prices': coin.prices[-50:],
                'timestamps': coin.timestamps[-50:],
                'candles': coin.candles,  # OHLCV data (100 candles for multi-timeframe)
                'indicators': coin.indicators,  # Technical indicators
                'signal_direction': coin.signal_direction,
                'signal_confidence': coin.signal_confidence,
                'signal_strength': coin.signal_strength,
                'position_count': len(coin.positions),
                'long_count': len([p for p in coin.positions if p.side == PositionSide.LONG]),
                'short_count': len([p for p in coin.positions if p.side == PositionSide.SHORT]),
                'auto_trade_enabled': coin.auto_trade_enabled,
                'current_atr': coin.current_atr,  # ATR for trailing stop reference
            }
        
        return DashboardState(
            timestamp=datetime.now().isoformat(),
            connected=True,
            active_symbol=self.active_symbol,
            coins=coins_dict,
            total_pnl=total_pnl,
            total_fees=self.total_fees,
            net_pnl=net_pnl,
            win_rate=win_rate,
            total_trades=len(self.all_trades),
            open_positions=len(self.all_positions),
            trades_in_window=self._trades_in_window(),
            can_trade=self._can_open_position(),
            next_trade_in=self._get_next_trade_in(),
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
    
    async def add_coin(self, symbol: str):
        """Add a new coin."""
        symbol = symbol.upper()
        if not symbol.endswith('USDT'):
            symbol += 'USDT'
        
        await self._add_coin(symbol)
        
        handler = self.data_handlers[symbol]
        await handler.connect()
        await handler.subscribe_klines('1s')
        await handler.subscribe_trades()
    
    def set_active_coin(self, symbol: str):
        """Set the active coin."""
        if symbol in self.coins:
            self.active_symbol = symbol


# =============================================================================
# FASTAPI APP
# =============================================================================

engine: Optional[ProTradingEngine] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    engine = ProTradingEngine()
    asyncio.create_task(engine.start())
    yield
    if engine:
        await engine.stop()


app = FastAPI(title="Pro Crypto Trading Dashboard", lifespan=lifespan)

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
        logger.info(f"Client connected ({len(engine.clients)} total)")
    
    try:
        while True:
            data = await websocket.receive_text()
            
            # Handle commands
            try:
                cmd = json.loads(data)
                if cmd.get('action') == 'set_active':
                    engine.set_active_coin(cmd.get('symbol', 'BTCUSDT'))
                elif cmd.get('action') == 'add_coin':
                    await engine.add_coin(cmd.get('symbol', ''))
                elif cmd.get('action') == 'toggle_auto_trade':
                    symbol = cmd.get('symbol', '')
                    if symbol in engine.coins:
                        engine.coins[symbol].auto_trade_enabled = not engine.coins[symbol].auto_trade_enabled
                        status = "enabled" if engine.coins[symbol].auto_trade_enabled else "disabled"
                        logger.info(f"Auto-trade {status} for {symbol}")
            except:
                pass
                
    except WebSocketDisconnect:
        if engine and websocket in engine.clients:
            engine.clients.remove(websocket)
        logger.info("Client disconnected")


@app.get("/api/status")
async def get_status():
    if engine:
        return {
            "running": engine.is_running,
            "coins": list(engine.coins.keys()),
            "positions": len(engine.all_positions),
            "trades": len(engine.all_trades),
        }
    return {"running": False}


@app.post("/api/add_coin/{symbol}")
async def add_coin(symbol: str):
    if engine:
        await engine.add_coin(symbol)
        return {"success": True, "symbol": symbol}
    return {"success": False}


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
