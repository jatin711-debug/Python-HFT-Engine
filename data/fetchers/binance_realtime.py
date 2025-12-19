"""
Binance Real-Time Data Handler.

Provides WebSocket-based streaming of candles and trades from Binance.
Designed for 1-5 second candle trading with efficient data buffering.

Usage:
    handler = BinanceDataHandler(symbols=['BTCUSDT', 'ETHUSDT'])
    await handler.connect()
    await handler.subscribe_klines(interval='1s')
    
    # Get recent data for feature calculation
    df = handler.get_buffer('BTCUSDT', n_bars=100)
"""

import asyncio
import json
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Dict, List, Optional, Any
from enum import Enum

import numpy as np
import pandas as pd
import aiohttp
import requests

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS AND DATA STRUCTURES
# =============================================================================

class TimeFrame(Enum):
    """Supported Binance kline intervals."""
    SECOND_1 = '1s'
    SECOND_5 = '5s'   # Note: Binance doesn't have 5s, we aggregate from 1s
    MINUTE_1 = '1m'
    MINUTE_5 = '5m'
    MINUTE_15 = '15m'
    HOUR_1 = '1h'
    DAY_1 = '1d'


@dataclass
class Candle:
    """Single OHLCV candle."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    quote_volume: float = 0.0
    trades: int = 0
    taker_buy_volume: float = 0.0
    taker_buy_quote_volume: float = 0.0
    is_closed: bool = True


@dataclass
class Trade:
    """Single trade (for sentiment/flow analysis)."""
    timestamp: datetime
    price: float
    quantity: float
    is_buyer_maker: bool  # True = sell order, False = buy order
    
    @property
    def is_buy(self) -> bool:
        return not self.is_buyer_maker
    
    @property
    def value(self) -> float:
        return self.price * self.quantity


@dataclass 
class OrderBookUpdate:
    """Order book update for LOB analysis."""
    timestamp: datetime
    bids: List[List[float]]  # [[price, qty], ...]
    asks: List[List[float]]  # [[price, qty], ...]
    
    @property
    def mid_price(self) -> float:
        if self.bids and self.asks:
            return (self.bids[0][0] + self.asks[0][0]) / 2
        return 0.0
    
    @property
    def spread(self) -> float:
        if self.bids and self.asks:
            return self.asks[0][0] - self.bids[0][0]
        return 0.0
    
    @property
    def imbalance(self) -> float:
        """LOB imbalance: (bid_vol - ask_vol) / (bid_vol + ask_vol)"""
        bid_vol = sum(b[1] for b in self.bids[:5]) if self.bids else 0
        ask_vol = sum(a[1] for a in self.asks[:5]) if self.asks else 0
        total = bid_vol + ask_vol
        return (bid_vol - ask_vol) / total if total > 0 else 0


@dataclass
class CryptoSentiment:
    """Crypto-specific sentiment from trade flow."""
    timestamp: datetime
    symbol: str
    buy_volume: float = 0.0
    sell_volume: float = 0.0
    buy_trades: int = 0
    sell_trades: int = 0
    large_buy_volume: float = 0.0  # Whale buys
    large_sell_volume: float = 0.0  # Whale sells
    
    @property
    def volume_imbalance(self) -> float:
        """Buy/sell volume imbalance."""
        total = self.buy_volume + self.sell_volume
        return (self.buy_volume - self.sell_volume) / total if total > 0 else 0
    
    @property
    def trade_imbalance(self) -> float:
        """Buy/sell trade count imbalance."""
        total = self.buy_trades + self.sell_trades
        return (self.buy_trades - self.sell_trades) / total if total > 0 else 0
    
    @property
    def whale_signal(self) -> float:
        """Large order imbalance (-1 to 1)."""
        total = self.large_buy_volume + self.large_sell_volume
        return (self.large_buy_volume - self.large_sell_volume) / total if total > 0 else 0


# =============================================================================
# CIRCULAR BUFFER FOR EFFICIENT ROLLING CALCULATIONS
# =============================================================================

class CandleBuffer:
    """Efficient circular buffer for candle data."""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self._buffer: deque = deque(maxlen=max_size)
        self._lock = asyncio.Lock()
    
    async def append(self, candle: Candle):
        """Add candle to buffer."""
        async with self._lock:
            self._buffer.append(candle)
    
    def append_sync(self, candle: Candle):
        """Synchronous append for non-async contexts."""
        self._buffer.append(candle)
    
    def get_latest(self, n: int = 1) -> List[Candle]:
        """Get last n candles."""
        if n >= len(self._buffer):
            return list(self._buffer)
        return list(self._buffer)[-n:]
    
    def to_dataframe(self, n: Optional[int] = None) -> pd.DataFrame:
        """Convert buffer to DataFrame."""
        candles = self.get_latest(n) if n else list(self._buffer)
        
        if not candles:
            return pd.DataFrame(columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'quote_volume', 'trades', 'taker_buy_volume'
            ])
        
        data = {
            'timestamp': [c.timestamp for c in candles],
            'open': [c.open for c in candles],
            'high': [c.high for c in candles],
            'low': [c.low for c in candles],
            'close': [c.close for c in candles],
            'volume': [c.volume for c in candles],
            'quote_volume': [c.quote_volume for c in candles],
            'trades': [c.trades for c in candles],
            'taker_buy_volume': [c.taker_buy_volume for c in candles],
        }
        
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        return df
    
    def __len__(self) -> int:
        return len(self._buffer)


class TradeBuffer:
    """Buffer for trade data (for sentiment analysis)."""
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self._buffer: deque = deque(maxlen=max_size)
    
    def append(self, trade: Trade):
        self._buffer.append(trade)
    
    def get_sentiment(self, lookback_seconds: float = 10.0) -> CryptoSentiment:
        """Calculate sentiment from recent trades."""
        now = datetime.now()
        cutoff = now.timestamp() - lookback_seconds
        
        sentiment = CryptoSentiment(timestamp=now, symbol='')
        
        for trade in reversed(self._buffer):
            if trade.timestamp.timestamp() < cutoff:
                break
            
            if trade.is_buy:
                sentiment.buy_volume += trade.value
                sentiment.buy_trades += 1
                if trade.value > 10000:  # $10k+ = whale
                    sentiment.large_buy_volume += trade.value
            else:
                sentiment.sell_volume += trade.value
                sentiment.sell_trades += 1
                if trade.value > 10000:
                    sentiment.large_sell_volume += trade.value
        
        return sentiment


# =============================================================================
# BINANCE DATA HANDLER
# =============================================================================

class BinanceDataHandler:
    """
    Real-time Binance data handler via WebSocket.
    
    Features:
    - Kline (candlestick) streaming for 1s/1m intervals
    - Trade streaming for sentiment analysis
    - Order book streaming for LOB analysis
    - Efficient circular buffers for rolling calculations
    
    Example:
        handler = BinanceDataHandler(['BTCUSDT', 'ETHUSDT'])
        await handler.connect()
        await handler.subscribe_klines('1s')
        
        # In your trading loop:
        df = handler.get_buffer('BTCUSDT', n_bars=100)
        sentiment = handler.get_sentiment('BTCUSDT')
    """
    
    BASE_WS_URL = "wss://stream.binance.com:9443/ws"
    BASE_REST_URL = "https://api.binance.com/api/v3"
    
    def __init__(
        self,
        symbols: List[str],
        buffer_size: int = 1000,
        trade_buffer_size: int = 10000,
    ):
        """
        Initialize Binance data handler.
        
        Args:
            symbols: List of trading pairs (e.g., ['BTCUSDT', 'ETHUSDT'])
            buffer_size: Max candles to keep in buffer
            trade_buffer_size: Max trades to keep for sentiment
        """
        self.symbols = [s.upper() for s in symbols]
        self.buffer_size = buffer_size
        
        # Buffers for each symbol and timeframe
        self._candle_buffers: Dict[str, Dict[str, CandleBuffer]] = {}
        self._trade_buffers: Dict[str, TradeBuffer] = {}
        self._orderbook: Dict[str, OrderBookUpdate] = {}
        
        # Initialize buffers
        for symbol in self.symbols:
            self._candle_buffers[symbol] = {}
            self._trade_buffers[symbol] = TradeBuffer(trade_buffer_size)
        
        # WebSocket sessions
        self._ws_session: Optional[aiohttp.ClientSession] = None
        self._ws_connections: Dict[str, aiohttp.ClientWebSocketResponse] = {}
        self._running = False
        
        # Callbacks
        self._candle_callbacks: List[Callable] = []
        self._trade_callbacks: List[Callable] = []
        
        # 5-second candle aggregation (Binance doesn't have 5s natively)
        self._5s_aggregators: Dict[str, List[Candle]] = {}
        
        logger.info(f"Initialized BinanceDataHandler for {self.symbols}")
    
    # -------------------------------------------------------------------------
    # CONNECTION MANAGEMENT
    # -------------------------------------------------------------------------
    
    async def connect(self):
        """Connect to Binance WebSocket."""
        if self._running:
            logger.warning("Already connected")
            return
        
        self._ws_session = aiohttp.ClientSession()
        self._running = True
        logger.info("Connected to Binance WebSocket")
    
    async def disconnect(self):
        """Disconnect from Binance WebSocket."""
        self._running = False
        
        for ws_name, ws in self._ws_connections.items():
            if not ws.closed:
                await ws.close()
                logger.info(f"Closed {ws_name} connection")
        
        self._ws_connections.clear()
        
        if self._ws_session:
            await self._ws_session.close()
            self._ws_session = None
        
        logger.info("Disconnected from Binance")
    
    # -------------------------------------------------------------------------
    # SUBSCRIPTION METHODS
    # -------------------------------------------------------------------------
    
    async def subscribe_klines(self, interval: str = '1s'):
        """
        Subscribe to kline (candlestick) updates.
        
        Args:
            interval: '1s', '1m', '5m', etc.
        """
        for symbol in self.symbols:
            symbol_lower = symbol.lower()
            stream_name = f"{symbol_lower}@kline_{interval}"
            
            # Initialize buffer for this interval
            if interval not in self._candle_buffers[symbol]:
                self._candle_buffers[symbol][interval] = CandleBuffer(self.buffer_size)
            
            # Start stream
            asyncio.create_task(
                self._stream_handler(stream_name, self._handle_kline)
            )
            logger.info(f"Subscribed to {symbol} klines ({interval})")
    
    async def subscribe_trades(self):
        """Subscribe to trade updates for sentiment analysis."""
        for symbol in self.symbols:
            symbol_lower = symbol.lower()
            stream_name = f"{symbol_lower}@trade"
            
            asyncio.create_task(
                self._stream_handler(stream_name, self._handle_trade)
            )
            logger.info(f"Subscribed to {symbol} trades")
    
    async def subscribe_orderbook(self, depth: int = 10):
        """Subscribe to order book updates."""
        for symbol in self.symbols:
            symbol_lower = symbol.lower()
            stream_name = f"{symbol_lower}@depth{depth}@100ms"
            
            asyncio.create_task(
                self._stream_handler(stream_name, self._handle_orderbook)
            )
            logger.info(f"Subscribed to {symbol} orderbook (depth={depth})")
    
    # -------------------------------------------------------------------------
    # STREAM HANDLERS
    # -------------------------------------------------------------------------
    
    async def _stream_handler(self, stream_name: str, handler: Callable):
        """Generic stream handler with reconnection."""
        url = f"{self.BASE_WS_URL}/{stream_name}"
        
        while self._running:
            try:
                async with self._ws_session.ws_connect(url) as ws:
                    self._ws_connections[stream_name] = ws
                    logger.info(f"Connected to stream: {stream_name}")
                    
                    async for msg in ws:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            data = json.loads(msg.data)
                            await handler(data)
                        elif msg.type == aiohttp.WSMsgType.ERROR:
                            logger.error(f"WebSocket error: {ws.exception()}")
                            break
                        elif msg.type == aiohttp.WSMsgType.CLOSED:
                            break
                            
            except Exception as e:
                logger.error(f"Stream {stream_name} error: {e}")
                if self._running:
                    await asyncio.sleep(5)  # Reconnect delay
    
    async def _handle_kline(self, data: dict):
        """Handle kline update."""
        k = data.get('k', {})
        symbol = data.get('s', '').upper()
        interval = k.get('i', '1m')
        
        candle = Candle(
            timestamp=datetime.fromtimestamp(k['t'] / 1000),
            open=float(k['o']),
            high=float(k['h']),
            low=float(k['l']),
            close=float(k['c']),
            volume=float(k['v']),
            quote_volume=float(k.get('q', 0)),
            trades=int(k.get('n', 0)),
            taker_buy_volume=float(k.get('V', 0)),
            taker_buy_quote_volume=float(k.get('Q', 0)),
            is_closed=k.get('x', False),
        )
        
        if symbol in self._candle_buffers:
            if interval not in self._candle_buffers[symbol]:
                self._candle_buffers[symbol][interval] = CandleBuffer(self.buffer_size)
            
            # Only add closed candles to avoid duplicates
            if candle.is_closed:
                await self._candle_buffers[symbol][interval].append(candle)
                
                # Aggregate 5s candles from 1s
                if interval == '1s':
                    await self._aggregate_5s_candle(symbol, candle)
                
                # Trigger callbacks
                for callback in self._candle_callbacks:
                    try:
                        await callback(symbol, interval, candle)
                    except Exception as e:
                        logger.error(f"Candle callback error: {e}")
    
    async def _aggregate_5s_candle(self, symbol: str, candle: Candle):
        """Aggregate 1s candles into 5s candles."""
        if symbol not in self._5s_aggregators:
            self._5s_aggregators[symbol] = []
        
        self._5s_aggregators[symbol].append(candle)
        
        # Every 5 candles, create a 5s aggregated candle
        if len(self._5s_aggregators[symbol]) >= 5:
            candles = self._5s_aggregators[symbol]
            
            agg_candle = Candle(
                timestamp=candles[0].timestamp,
                open=candles[0].open,
                high=max(c.high for c in candles),
                low=min(c.low for c in candles),
                close=candles[-1].close,
                volume=sum(c.volume for c in candles),
                quote_volume=sum(c.quote_volume for c in candles),
                trades=sum(c.trades for c in candles),
                taker_buy_volume=sum(c.taker_buy_volume for c in candles),
                is_closed=True,
            )
            
            if '5s' not in self._candle_buffers[symbol]:
                self._candle_buffers[symbol]['5s'] = CandleBuffer(self.buffer_size)
            
            await self._candle_buffers[symbol]['5s'].append(agg_candle)
            self._5s_aggregators[symbol] = []
    
    async def _handle_trade(self, data: dict):
        """Handle trade update for sentiment analysis."""
        symbol = data.get('s', '').upper()
        
        trade = Trade(
            timestamp=datetime.fromtimestamp(data['T'] / 1000),
            price=float(data['p']),
            quantity=float(data['q']),
            is_buyer_maker=data['m'],
        )
        
        if symbol in self._trade_buffers:
            self._trade_buffers[symbol].append(trade)
            
            # Trigger callbacks
            for callback in self._trade_callbacks:
                try:
                    await callback(symbol, trade)
                except Exception as e:
                    logger.error(f"Trade callback error: {e}")
    
    async def _handle_orderbook(self, data: dict):
        """Handle order book update."""
        # Extract symbol from stream data
        # For depth streams, we need to track which symbol this belongs to
        bids = [[float(p), float(q)] for p, q in data.get('bids', [])]
        asks = [[float(p), float(q)] for p, q in data.get('asks', [])]
        
        # Note: Binance depth stream doesn't include symbol in payload
        # We need to track this from the stream subscription
        # For now, store with a placeholder that gets updated
        update = OrderBookUpdate(
            timestamp=datetime.now(),
            bids=bids,
            asks=asks,
        )
        
        # This will be called with the correct symbol context
        return update
    
    # -------------------------------------------------------------------------
    # DATA ACCESS METHODS
    # -------------------------------------------------------------------------
    
    def get_buffer(
        self,
        symbol: str,
        n_bars: Optional[int] = None,
        interval: str = '1s',
    ) -> pd.DataFrame:
        """
        Get candle data as DataFrame.
        
        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            n_bars: Number of bars to return (None = all)
            interval: Timeframe ('1s', '5s', '1m', etc.)
            
        Returns:
            DataFrame with OHLCV data
        """
        symbol = symbol.upper()
        
        if symbol not in self._candle_buffers:
            logger.warning(f"No buffer for {symbol}")
            return pd.DataFrame()
        
        if interval not in self._candle_buffers[symbol]:
            logger.warning(f"No {interval} data for {symbol}")
            return pd.DataFrame()
        
        return self._candle_buffers[symbol][interval].to_dataframe(n_bars)
    
    def get_sentiment(
        self,
        symbol: str,
        lookback_seconds: float = 10.0,
    ) -> CryptoSentiment:
        """
        Get sentiment from recent trade flow.
        
        Args:
            symbol: Trading pair
            lookback_seconds: How far back to analyze
            
        Returns:
            CryptoSentiment with buy/sell imbalances
        """
        symbol = symbol.upper()
        
        if symbol not in self._trade_buffers:
            return CryptoSentiment(timestamp=datetime.now(), symbol=symbol)
        
        sentiment = self._trade_buffers[symbol].get_sentiment(lookback_seconds)
        sentiment.symbol = symbol
        return sentiment
    
    def get_orderbook(self, symbol: str) -> Optional[OrderBookUpdate]:
        """Get latest order book snapshot."""
        return self._orderbook.get(symbol.upper())
    
    def get_latest_price(self, symbol: str, interval: str = '1s') -> Optional[float]:
        """Get the latest close price."""
        symbol = symbol.upper()
        
        if symbol in self._candle_buffers:
            if interval in self._candle_buffers[symbol]:
                candles = self._candle_buffers[symbol][interval].get_latest(1)
                if candles:
                    return candles[0].close
        
        return None
    
    # -------------------------------------------------------------------------
    # HISTORICAL DATA (REST API)
    # -------------------------------------------------------------------------
    
    def fetch_historical_klines(
        self,
        symbol: str,
        interval: str = '1m',
        limit: int = 500,
    ) -> pd.DataFrame:
        """
        Fetch historical klines via REST API.
        
        Args:
            symbol: Trading pair
            interval: Timeframe
            limit: Number of candles (max 1000)
            
        Returns:
            DataFrame with historical OHLCV
        """
        url = f"{self.BASE_REST_URL}/klines"
        params = {
            'symbol': symbol.upper(),
            'interval': interval,
            'limit': min(limit, 1000),
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades',
                'taker_buy_volume', 'taker_buy_quote_volume', 'ignore'
            ])
            
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            for col in ['open', 'high', 'low', 'close', 'volume', 'quote_volume', 'taker_buy_volume']:
                df[col] = df[col].astype(float)
            
            df['trades'] = df['trades'].astype(int)
            
            # Keep relevant columns
            df = df[['open', 'high', 'low', 'close', 'volume', 'quote_volume', 'trades', 'taker_buy_volume']]
            
            logger.info(f"Fetched {len(df)} historical klines for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching historical data: {e}")
            return pd.DataFrame()
    
    def preload_historical(
        self,
        interval: str = '1m',
        n_candles: int = 500,
    ):
        """
        Preload historical data into buffers.
        
        Call this before starting real-time streaming to have
        enough data for feature calculations.
        """
        for symbol in self.symbols:
            df = self.fetch_historical_klines(symbol, interval, n_candles)
            
            if df.empty:
                continue
            
            if interval not in self._candle_buffers[symbol]:
                self._candle_buffers[symbol][interval] = CandleBuffer(self.buffer_size)
            
            buffer = self._candle_buffers[symbol][interval]
            
            for ts, row in df.iterrows():
                candle = Candle(
                    timestamp=ts,
                    open=row['open'],
                    high=row['high'],
                    low=row['low'],
                    close=row['close'],
                    volume=row['volume'],
                    quote_volume=row['quote_volume'],
                    trades=row['trades'],
                    taker_buy_volume=row['taker_buy_volume'],
                    is_closed=True,
                )
                buffer.append_sync(candle)
            
            logger.info(f"Preloaded {len(df)} candles for {symbol} ({interval})")
    
    # -------------------------------------------------------------------------
    # CALLBACK REGISTRATION
    # -------------------------------------------------------------------------
    
    def on_candle(self, callback: Callable):
        """Register callback for new candles."""
        self._candle_callbacks.append(callback)
    
    def on_trade(self, callback: Callable):
        """Register callback for new trades."""
        self._trade_callbacks.append(callback)
    
    # -------------------------------------------------------------------------
    # UTILITY METHODS
    # -------------------------------------------------------------------------
    
    @property
    def is_connected(self) -> bool:
        """Check if connected."""
        return self._running and self._ws_session is not None
    
    def get_buffer_stats(self) -> Dict[str, Dict[str, int]]:
        """Get buffer sizes for all symbols."""
        stats = {}
        for symbol, intervals in self._candle_buffers.items():
            stats[symbol] = {}
            for interval, buffer in intervals.items():
                stats[symbol][interval] = len(buffer)
            stats[symbol]['trades'] = len(self._trade_buffers.get(symbol, []))
        return stats


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def create_binance_handler(
    symbols: List[str] = None,
    preload: bool = True,
) -> BinanceDataHandler:
    """
    Create and initialize a Binance data handler.
    
    Args:
        symbols: Trading pairs (default: BTC and ETH)
        preload: Whether to preload historical data
        
    Returns:
        Initialized BinanceDataHandler
    """
    if symbols is None:
        symbols = ['BTCUSDT', 'ETHUSDT']
    
    handler = BinanceDataHandler(symbols)
    
    if preload:
        handler.preload_historical(interval='1m', n_candles=500)
    
    return handler


# =============================================================================
# ASYNC RUNNER FOR TESTING
# =============================================================================

async def _demo():
    """Demo the Binance data handler."""
    handler = BinanceDataHandler(['BTCUSDT', 'ETHUSDT'])
    
    # Preload historical data
    handler.preload_historical(interval='1m', n_candles=100)
    
    print(f"Buffer stats: {handler.get_buffer_stats()}")
    
    # Get data as DataFrame
    df = handler.get_buffer('BTCUSDT', n_bars=10, interval='1m')
    print(f"\nLatest 10 candles:\n{df}")
    
    # Connect for real-time
    await handler.connect()
    
    # Subscribe to streams
    await handler.subscribe_klines('1s')
    await handler.subscribe_trades()
    
    # Run for 30 seconds
    print("\nStreaming for 30 seconds...")
    await asyncio.sleep(30)
    
    # Check buffers
    print(f"\nBuffer stats after streaming: {handler.get_buffer_stats()}")
    
    # Get sentiment
    sentiment = handler.get_sentiment('BTCUSDT', lookback_seconds=10)
    print(f"\nBTCUSDT Sentiment:")
    print(f"  Volume imbalance: {sentiment.volume_imbalance:.3f}")
    print(f"  Trade imbalance: {sentiment.trade_imbalance:.3f}")
    print(f"  Whale signal: {sentiment.whale_signal:.3f}")
    
    await handler.disconnect()


if __name__ == '__main__':
    asyncio.run(_demo())
