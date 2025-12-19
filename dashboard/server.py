"""
Crypto Trading Dashboard - Backend Server.

FastAPI server that:
1. Connects to Binance for live data
2. Runs trading strategies
3. Pushes updates to React dashboard via WebSocket

Usage:
    python dashboard/server.py
    
Then open React app at http://localhost:5173
"""

import asyncio
import json
import logging
import sys
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from contextlib import asynccontextmanager

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn

from data.fetchers import BinanceDataHandler, create_binance_handler
from features import MicrostructureFeatures
from strategies import MicroStrategyEnsemble

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class DashboardState:
    """Current state to send to dashboard."""
    timestamp: str
    symbol: str
    price: float
    price_change_pct: float
    
    # Position
    position_side: str  # "LONG", "SHORT", "NONE"
    position_entry: float
    position_size: float
    position_pnl: float
    position_pnl_pct: float
    
    # P&L
    total_pnl: float
    total_pnl_pct: float
    win_rate: float
    trade_count: int
    
    # Signal
    signal_direction: str  # "BUY", "SELL", "HOLD"
    signal_confidence: float
    signal_strength: float
    
    # Strategy signals
    momentum_signal: float
    mean_reversion_signal: float
    volatility_signal: float
    order_flow_signal: float
    
    # Recent prices for chart
    prices: List[float]
    timestamps: List[str]
    
    # Recent trades
    trades: List[Dict]


@dataclass
class Position:
    """Current trading position."""
    side: str  # "LONG", "SHORT"
    entry_price: float
    size: float
    entry_time: datetime
    stop_loss: float
    take_profit: float


@dataclass
class Trade:
    """Completed trade record."""
    side: str
    entry_price: float
    exit_price: float
    size: float
    pnl: float
    pnl_pct: float
    entry_time: str
    exit_time: str
    reason: str


# =============================================================================
# TRADING ENGINE
# =============================================================================

class DashboardTradingEngine:
    """Trading engine for dashboard."""
    
    def __init__(self, symbol: str = "BTCUSDT", capital: float = 500.0):
        self.symbol = symbol
        self.capital = capital
        self.initial_capital = capital
        
        # Components
        self.data_handler: Optional[BinanceDataHandler] = None
        self.features: Optional[MicrostructureFeatures] = None
        self.strategy: Optional[MicroStrategyEnsemble] = None
        
        # State
        self.position: Optional[Position] = None
        self.trades: List[Trade] = []
        self.prices: List[float] = []
        self.timestamps: List[str] = []
        
        # Connected clients
        self.clients: List[WebSocket] = []
        
        self.is_running = False
    
    async def initialize(self):
        """Initialize components."""
        logger.info(f"Initializing trading engine for {self.symbol}...")
        
        self.data_handler = BinanceDataHandler([self.symbol])
        self.features = MicrostructureFeatures()
        self.strategy = MicroStrategyEnsemble(capital=self.capital, aggressive=True)
        
        # Preload historical data
        self.data_handler.preload_historical(interval='1m', n_candles=200)
        
        logger.info("Trading engine initialized")
    
    async def start(self):
        """Start the trading loop."""
        await self.initialize()
        
        # Connect to Binance
        await self.data_handler.connect()
        await self.data_handler.subscribe_klines('1s')
        await self.data_handler.subscribe_trades()
        
        self.is_running = True
        logger.info("Trading loop started")
        
        # Main loop
        while self.is_running:
            try:
                await self._trading_iteration()
            except Exception as e:
                logger.error(f"Trading iteration error: {e}")
            
            await asyncio.sleep(5)  # Update every 5 seconds
    
    async def stop(self):
        """Stop the trading loop."""
        self.is_running = False
        if self.data_handler:
            await self.data_handler.disconnect()
        logger.info("Trading loop stopped")
    
    async def _trading_iteration(self):
        """Single trading iteration."""
        # Get candle data
        df = self.data_handler.get_buffer(self.symbol, n_bars=100, interval='1s')
        
        if len(df) < 60:
            return
        
        current_price = df['close'].iloc[-1]
        
        # Store price history
        self.prices.append(current_price)
        self.timestamps.append(datetime.now().strftime('%H:%M:%S'))
        
        # Keep last 100 prices
        if len(self.prices) > 100:
            self.prices.pop(0)
            self.timestamps.pop(0)
        
        # Add features
        df_features = self.features.add_all_features(df)
        
        # Get sentiment
        sentiment = self.data_handler.get_sentiment(self.symbol, lookback_seconds=10)
        
        # Create sentiment wrapper
        class SentimentWrapper:
            def __init__(self, s):
                self.volume_imbalance = s.volume_imbalance
                self.trade_imbalance = s.trade_imbalance
                self.whale_signal = s.whale_signal
        
        sentiment_obj = SentimentWrapper(sentiment)
        
        # Generate signal
        signal = self.strategy.generate_signal(df_features, sentiment_obj)
        
        # Manage position
        if self.position:
            await self._manage_position(current_price)
        elif signal.direction != 0 and signal.confidence >= 0.5:
            await self._open_position(signal, current_price)
        
        # Build and broadcast state
        state = self._build_state(current_price, signal)
        await self._broadcast(state)
    
    async def _open_position(self, signal, price: float):
        """Open a new position."""
        side = "LONG" if signal.direction == 1 else "SHORT"
        size = self.capital * signal.position_size_pct / price
        
        self.position = Position(
            side=side,
            entry_price=price,
            size=size,
            entry_time=datetime.now(),
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
        )
        
        logger.info(f"Opened {side} position at ${price:.2f}")
    
    async def _manage_position(self, current_price: float):
        """Manage existing position."""
        pos = self.position
        
        # Calculate P&L
        if pos.side == "LONG":
            pnl = (current_price - pos.entry_price) * pos.size
            hit_stop = current_price <= pos.stop_loss
            hit_target = current_price >= pos.take_profit
        else:
            pnl = (pos.entry_price - current_price) * pos.size
            hit_stop = current_price >= pos.stop_loss
            hit_target = current_price <= pos.take_profit
        
        # Check exits
        reason = None
        if hit_stop:
            reason = "Stop Loss"
        elif hit_target:
            reason = "Take Profit"
        elif (datetime.now() - pos.entry_time).seconds > 120:
            reason = "Timeout"
        
        if reason:
            await self._close_position(current_price, reason)
    
    async def _close_position(self, price: float, reason: str):
        """Close position and record trade."""
        pos = self.position
        
        if pos.side == "LONG":
            pnl = (price - pos.entry_price) * pos.size
        else:
            pnl = (pos.entry_price - price) * pos.size
        
        pnl_pct = pnl / (pos.entry_price * pos.size)
        
        trade = Trade(
            side=pos.side,
            entry_price=pos.entry_price,
            exit_price=price,
            size=pos.size,
            pnl=pnl,
            pnl_pct=pnl_pct,
            entry_time=pos.entry_time.strftime('%H:%M:%S'),
            exit_time=datetime.now().strftime('%H:%M:%S'),
            reason=reason,
        )
        
        self.trades.append(trade)
        self.capital += pnl
        self.position = None
        
        logger.info(f"Closed {pos.side} position: ${pnl:+.2f} ({reason})")
    
    def _build_state(self, current_price: float, signal) -> DashboardState:
        """Build state for dashboard."""
        # Calculate totals
        total_pnl = self.capital - self.initial_capital
        total_pnl_pct = total_pnl / self.initial_capital * 100
        
        wins = [t for t in self.trades if t.pnl > 0]
        win_rate = len(wins) / len(self.trades) * 100 if self.trades else 0
        
        # Position info
        if self.position:
            pos_side = self.position.side
            pos_entry = self.position.entry_price
            pos_size = self.position.size
            
            if pos_side == "LONG":
                pos_pnl = (current_price - pos_entry) * pos_size
            else:
                pos_pnl = (pos_entry - current_price) * pos_size
            
            pos_pnl_pct = pos_pnl / (pos_entry * pos_size) * 100
        else:
            pos_side = "NONE"
            pos_entry = 0
            pos_size = 0
            pos_pnl = 0
            pos_pnl_pct = 0
        
        # Price change
        if len(self.prices) >= 2:
            price_change = (current_price - self.prices[0]) / self.prices[0] * 100
        else:
            price_change = 0
        
        # Signal direction
        if signal.direction == 1:
            sig_dir = "BUY"
        elif signal.direction == -1:
            sig_dir = "SELL"
        else:
            sig_dir = "HOLD"
        
        return DashboardState(
            timestamp=datetime.now().isoformat(),
            symbol=self.symbol,
            price=current_price,
            price_change_pct=price_change,
            
            position_side=pos_side,
            position_entry=pos_entry,
            position_size=pos_size,
            position_pnl=pos_pnl,
            position_pnl_pct=pos_pnl_pct,
            
            total_pnl=total_pnl,
            total_pnl_pct=total_pnl_pct,
            win_rate=win_rate,
            trade_count=len(self.trades),
            
            signal_direction=sig_dir,
            signal_confidence=signal.confidence,
            signal_strength=signal.strength,
            
            momentum_signal=signal.metadata.get('momentum', 0) if hasattr(signal, 'metadata') else 0,
            mean_reversion_signal=0,
            volatility_signal=0,
            order_flow_signal=0,
            
            prices=self.prices[-50:],
            timestamps=self.timestamps[-50:],
            
            trades=[asdict(t) for t in self.trades[-10:]],
        )
    
    async def _broadcast(self, state: DashboardState):
        """Broadcast state to all connected clients."""
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


# =============================================================================
# FASTAPI APP
# =============================================================================

# Global engine instance
engine: Optional[DashboardTradingEngine] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown."""
    global engine
    engine = DashboardTradingEngine(symbol="BTCUSDT", capital=500.0)
    
    # Start trading loop in background
    asyncio.create_task(engine.start())
    
    yield
    
    # Shutdown
    if engine:
        await engine.stop()


app = FastAPI(title="Crypto Trading Dashboard", lifespan=lifespan)

# CORS for React dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates."""
    await websocket.accept()
    
    if engine:
        engine.clients.append(websocket)
        logger.info(f"Client connected. Total: {len(engine.clients)}")
    
    try:
        while True:
            # Keep connection alive, receive any commands
            data = await websocket.receive_text()
            # Could handle commands like "START", "STOP", etc.
    except WebSocketDisconnect:
        if engine and websocket in engine.clients:
            engine.clients.remove(websocket)
        logger.info("Client disconnected")


@app.get("/api/status")
async def get_status():
    """Get current status."""
    if engine:
        return {
            "running": engine.is_running,
            "symbol": engine.symbol,
            "capital": engine.capital,
            "position": engine.position.side if engine.position else None,
            "trades": len(engine.trades),
        }
    return {"running": False}


if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )
