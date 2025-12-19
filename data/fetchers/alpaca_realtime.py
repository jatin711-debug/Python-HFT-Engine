"""
Alpaca Real-Time Stock Data Handler.

Provides WebSocket-based streaming of trades and bars from Alpaca Markets.
Aggregates trades into custom 5-second candles for short-term trading.

Features:
- Real-time trade streaming (free with Alpaca account)
- Custom candle aggregation (5s, 10s, 30s)
- Volume/trade flow analysis
- Integration with existing HFT strategies

Requirements:
- Alpaca account (free): https://alpaca.markets
- Set environment variables:
    ALPACA_API_KEY=your_key
    ALPACA_SECRET_KEY=your_secret

Usage:
    handler = AlpacaDataHandler(symbols=['AAPL', 'TSLA'])
    await handler.connect()
    
    # Get 5-second candles
    df = handler.get_buffer('AAPL', n_bars=100, interval='5s')
"""

import asyncio
import json
import logging
import os
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable, Dict, List, Optional, Any
from enum import Enum

import numpy as np
import pandas as pd
import aiohttp
import requests

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class AlpacaConfig:
    """Alpaca API configuration."""
    api_key: str = ""
    secret_key: str = ""
    paper: bool = True  # Use paper trading endpoint
    
    # Data feed
    feed: str = "iex"  # 'iex' (free) or 'sip' (paid, more complete)
    
    # Custom candle intervals (in seconds)
    custom_intervals: tuple = (5, 10, 30)
    
    def __post_init__(self):
        # Load from environment if not provided
        if not self.api_key:
            self.api_key = os.getenv('ALPACA_API_KEY', '')
        if not self.secret_key:
            self.secret_key = os.getenv('ALPACA_SECRET_KEY', '')
    
    @property
    def base_url(self) -> str:
        if self.paper:
            return "https://paper-api.alpaca.markets"
        return "https://api.alpaca.markets"
    
    @property
    def data_url(self) -> str:
        return "https://data.alpaca.markets"
    
    @property
    def stream_url(self) -> str:
        return "wss://stream.data.alpaca.markets/v2"


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class StockTrade:
    """Single stock trade."""
    timestamp: datetime
    symbol: str
    price: float
    size: int
    exchange: str
    conditions: List[str] = field(default_factory=list)
    
    @property
    def value(self) -> float:
        return self.price * self.size


@dataclass
class StockCandle:
    """Aggregated candle."""
    timestamp: datetime
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: int
    trade_count: int
    vwap: float
    buy_volume: int = 0  # Estimated from upticks
    sell_volume: int = 0  # Estimated from downticks
    is_closed: bool = True


@dataclass
class StockSentiment:
    """Stock-specific sentiment from trade flow."""
    timestamp: datetime
    symbol: str
    buy_volume: int = 0
    sell_volume: int = 0
    buy_trades: int = 0
    sell_trades: int = 0
    large_buy_volume: int = 0  # Institutional buys
    large_sell_volume: int = 0  # Institutional sells
    
    @property
    def volume_imbalance(self) -> float:
        total = self.buy_volume + self.sell_volume
        return (self.buy_volume - self.sell_volume) / total if total > 0 else 0
    
    @property
    def trade_imbalance(self) -> float:
        total = self.buy_trades + self.sell_trades
        return (self.buy_trades - self.sell_trades) / total if total > 0 else 0
    
    @property
    def institutional_signal(self) -> float:
        total = self.large_buy_volume + self.large_sell_volume
        return (self.large_buy_volume - self.large_sell_volume) / total if total > 0 else 0


# =============================================================================
# CANDLE AGGREGATOR
# =============================================================================

class CandleAggregator:
    """
    Aggregates raw trades into custom interval candles.
    
    Unlike crypto, stock trades need to be aggregated into candles
    since Alpaca free tier doesn't provide sub-minute bars.
    """
    
    def __init__(self, interval_seconds: int = 5, max_candles: int = 1000):
        self.interval_seconds = interval_seconds
        self.max_candles = max_candles
        
        # Current candle being built
        self._current_candle: Optional[Dict] = None
        self._current_interval_start: Optional[datetime] = None
        
        # Completed candles
        self._candles: deque = deque(maxlen=max_candles)
        
        # Trade tracking for buy/sell classification
        self._last_price: float = 0
    
    def add_trade(self, trade: StockTrade) -> Optional[StockCandle]:
        """
        Add a trade and return completed candle if interval ended.
        
        Uses tick rule for buy/sell classification:
        - Uptick (price > last) = buy
        - Downtick (price < last) = sell
        """
        # Determine interval start
        ts = trade.timestamp
        interval_start = ts.replace(
            second=(ts.second // self.interval_seconds) * self.interval_seconds,
            microsecond=0
        )
        
        # Check if we need to start a new candle
        completed_candle = None
        
        if self._current_interval_start is None or interval_start > self._current_interval_start:
            # Complete the current candle
            if self._current_candle is not None:
                completed_candle = self._finalize_candle()
            
            # Start new candle
            self._current_interval_start = interval_start
            self._current_candle = {
                'timestamp': interval_start,
                'symbol': trade.symbol,
                'open': trade.price,
                'high': trade.price,
                'low': trade.price,
                'close': trade.price,
                'volume': 0,
                'trade_count': 0,
                'total_value': 0.0,
                'buy_volume': 0,
                'sell_volume': 0,
            }
        
        # Update current candle
        c = self._current_candle
        c['high'] = max(c['high'], trade.price)
        c['low'] = min(c['low'], trade.price)
        c['close'] = trade.price
        c['volume'] += trade.size
        c['trade_count'] += 1
        c['total_value'] += trade.value
        
        # Classify as buy or sell using tick rule
        if self._last_price > 0:
            if trade.price > self._last_price:
                c['buy_volume'] += trade.size
            elif trade.price < self._last_price:
                c['sell_volume'] += trade.size
            else:
                # Same price - split evenly
                c['buy_volume'] += trade.size // 2
                c['sell_volume'] += trade.size // 2
        
        self._last_price = trade.price
        
        return completed_candle
    
    def _finalize_candle(self) -> StockCandle:
        """Finalize and return the current candle."""
        c = self._current_candle
        
        vwap = c['total_value'] / c['volume'] if c['volume'] > 0 else c['close']
        
        candle = StockCandle(
            timestamp=c['timestamp'],
            symbol=c['symbol'],
            open=c['open'],
            high=c['high'],
            low=c['low'],
            close=c['close'],
            volume=c['volume'],
            trade_count=c['trade_count'],
            vwap=vwap,
            buy_volume=c['buy_volume'],
            sell_volume=c['sell_volume'],
            is_closed=True,
        )
        
        self._candles.append(candle)
        return candle
    
    def get_candles(self, n: Optional[int] = None) -> List[StockCandle]:
        """Get last n candles."""
        if n is None or n >= len(self._candles):
            return list(self._candles)
        return list(self._candles)[-n:]
    
    def to_dataframe(self, n: Optional[int] = None) -> pd.DataFrame:
        """Convert candles to DataFrame."""
        candles = self.get_candles(n)
        
        if not candles:
            return pd.DataFrame(columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'trade_count', 'vwap', 'buy_volume', 'sell_volume'
            ])
        
        data = {
            'timestamp': [c.timestamp for c in candles],
            'open': [c.open for c in candles],
            'high': [c.high for c in candles],
            'low': [c.low for c in candles],
            'close': [c.close for c in candles],
            'volume': [c.volume for c in candles],
            'trade_count': [c.trade_count for c in candles],
            'vwap': [c.vwap for c in candles],
            'buy_volume': [c.buy_volume for c in candles],
            'sell_volume': [c.sell_volume for c in candles],
        }
        
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        return df
    
    def __len__(self) -> int:
        return len(self._candles)


# =============================================================================
# ALPACA DATA HANDLER
# =============================================================================

class AlpacaDataHandler:
    """
    Real-time Alpaca stock data handler.
    
    Features:
    - WebSocket trade streaming
    - Custom candle aggregation (5s, 10s, 30s)
    - Trade flow sentiment analysis
    - Historical data preloading
    
    Example:
        handler = AlpacaDataHandler(['AAPL', 'TSLA', 'MSFT'])
        await handler.connect()
        
        # Get 5-second candles
        df = handler.get_buffer('AAPL', n_bars=100, interval='5s')
    """
    
    def __init__(
        self,
        symbols: List[str],
        config: AlpacaConfig = None,
        buffer_size: int = 1000,
    ):
        """
        Initialize Alpaca data handler.
        
        Args:
            symbols: List of stock symbols (e.g., ['AAPL', 'TSLA'])
            config: API configuration
            buffer_size: Max candles per aggregator
        """
        self.symbols = [s.upper() for s in symbols]
        self.config = config or AlpacaConfig()
        self.buffer_size = buffer_size
        
        # Candle aggregators for each symbol and interval
        self._aggregators: Dict[str, Dict[int, CandleAggregator]] = {}
        self._trade_buffers: Dict[str, deque] = {}
        
        # Initialize aggregators
        for symbol in self.symbols:
            self._aggregators[symbol] = {}
            for interval in self.config.custom_intervals:
                self._aggregators[symbol][interval] = CandleAggregator(
                    interval_seconds=interval,
                    max_candles=buffer_size,
                )
            self._trade_buffers[symbol] = deque(maxlen=10000)
        
        # WebSocket
        self._ws_session: Optional[aiohttp.ClientSession] = None
        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self._running = False
        
        # Callbacks
        self._candle_callbacks: List[Callable] = []
        self._trade_callbacks: List[Callable] = []
        
        # Check credentials
        if not self.config.api_key or not self.config.secret_key:
            logger.warning("Alpaca API credentials not set! Set ALPACA_API_KEY and ALPACA_SECRET_KEY")
        
        logger.info(f"Initialized AlpacaDataHandler for {self.symbols}")
    
    # -------------------------------------------------------------------------
    # CONNECTION
    # -------------------------------------------------------------------------
    
    async def connect(self):
        """Connect to Alpaca WebSocket."""
        if self._running:
            return
        
        self._ws_session = aiohttp.ClientSession()
        self._running = True
        
        # Start WebSocket handler
        asyncio.create_task(self._websocket_handler())
        
        logger.info("Connecting to Alpaca...")
    
    async def disconnect(self):
        """Disconnect from Alpaca."""
        self._running = False
        
        if self._ws and not self._ws.closed:
            await self._ws.close()
        
        if self._ws_session:
            await self._ws_session.close()
            self._ws_session = None
        
        logger.info("Disconnected from Alpaca")
    
    async def _websocket_handler(self):
        """Handle WebSocket connection and messages."""
        feed = self.config.feed
        url = f"{self.config.stream_url}/{feed}"
        
        while self._running:
            try:
                async with self._ws_session.ws_connect(url) as ws:
                    self._ws = ws
                    
                    # Authenticate
                    auth_msg = {
                        "action": "auth",
                        "key": self.config.api_key,
                        "secret": self.config.secret_key,
                    }
                    await ws.send_json(auth_msg)
                    
                    # Wait for auth response
                    auth_response = await ws.receive_json()
                    if auth_response[0].get('msg') == 'authenticated':
                        logger.info("Authenticated with Alpaca")
                    else:
                        logger.error(f"Auth failed: {auth_response}")
                        break
                    
                    # Subscribe to trades
                    sub_msg = {
                        "action": "subscribe",
                        "trades": self.symbols,
                    }
                    await ws.send_json(sub_msg)
                    
                    # Read messages
                    async for msg in ws:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            data = json.loads(msg.data)
                            await self._handle_messages(data)
                        elif msg.type == aiohttp.WSMsgType.ERROR:
                            logger.error(f"WebSocket error: {ws.exception()}")
                            break
                            
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                if self._running:
                    await asyncio.sleep(5)
    
    async def _handle_messages(self, messages: List[dict]):
        """Handle incoming messages."""
        for msg in messages:
            msg_type = msg.get('T')
            
            if msg_type == 't':  # Trade
                await self._handle_trade(msg)
    
    async def _handle_trade(self, data: dict):
        """Handle trade message."""
        try:
            trade = StockTrade(
                timestamp=datetime.fromisoformat(data['t'].replace('Z', '+00:00')),
                symbol=data['S'],
                price=float(data['p']),
                size=int(data['s']),
                exchange=data.get('x', ''),
                conditions=data.get('c', []),
            )
            
            symbol = trade.symbol
            
            # Add to trade buffer
            if symbol in self._trade_buffers:
                self._trade_buffers[symbol].append(trade)
            
            # Aggregate into candles
            if symbol in self._aggregators:
                for interval, aggregator in self._aggregators[symbol].items():
                    completed = aggregator.add_trade(trade)
                    
                    if completed:
                        for callback in self._candle_callbacks:
                            try:
                                await callback(symbol, interval, completed)
                            except Exception as e:
                                logger.error(f"Candle callback error: {e}")
            
            # Trade callbacks
            for callback in self._trade_callbacks:
                try:
                    await callback(symbol, trade)
                except Exception as e:
                    logger.error(f"Trade callback error: {e}")
                    
        except Exception as e:
            logger.error(f"Trade parsing error: {e}")
    
    # -------------------------------------------------------------------------
    # DATA ACCESS
    # -------------------------------------------------------------------------
    
    def get_buffer(
        self,
        symbol: str,
        n_bars: Optional[int] = None,
        interval: str = '5s',
    ) -> pd.DataFrame:
        """
        Get candle data as DataFrame.
        
        Args:
            symbol: Stock symbol
            n_bars: Number of bars (None = all)
            interval: '5s', '10s', '30s', or '1m'
            
        Returns:
            DataFrame with OHLCV data
        """
        symbol = symbol.upper()
        
        # Parse interval
        if interval.endswith('s'):
            interval_secs = int(interval[:-1])
        elif interval.endswith('m'):
            interval_secs = int(interval[:-1]) * 60
        else:
            interval_secs = 5
        
        if symbol not in self._aggregators:
            return pd.DataFrame()
        
        if interval_secs not in self._aggregators[symbol]:
            logger.warning(f"Interval {interval} not available for {symbol}")
            return pd.DataFrame()
        
        return self._aggregators[symbol][interval_secs].to_dataframe(n_bars)
    
    def get_sentiment(
        self,
        symbol: str,
        lookback_seconds: float = 30.0,
    ) -> StockSentiment:
        """
        Get sentiment from recent trade flow.
        
        Args:
            symbol: Stock symbol
            lookback_seconds: How far back to analyze
            
        Returns:
            StockSentiment with buy/sell analysis
        """
        symbol = symbol.upper()
        
        if symbol not in self._trade_buffers:
            return StockSentiment(timestamp=datetime.now(), symbol=symbol)
        
        now = datetime.now()
        cutoff = now.timestamp() - lookback_seconds
        
        sentiment = StockSentiment(timestamp=now, symbol=symbol)
        
        last_price = 0
        for trade in reversed(self._trade_buffers[symbol]):
            if trade.timestamp.timestamp() < cutoff:
                break
            
            # Classify using tick rule
            if last_price > 0:
                if trade.price > last_price:
                    sentiment.buy_volume += trade.size
                    sentiment.buy_trades += 1
                    if trade.value > 50000:  # $50k+ = institutional
                        sentiment.large_buy_volume += trade.size
                elif trade.price < last_price:
                    sentiment.sell_volume += trade.size
                    sentiment.sell_trades += 1
                    if trade.value > 50000:
                        sentiment.large_sell_volume += trade.size
            
            last_price = trade.price
        
        return sentiment
    
    def get_latest_price(self, symbol: str) -> Optional[float]:
        """Get latest trade price."""
        symbol = symbol.upper()
        
        if symbol in self._trade_buffers and self._trade_buffers[symbol]:
            return self._trade_buffers[symbol][-1].price
        
        return None
    
    # -------------------------------------------------------------------------
    # HISTORICAL DATA
    # -------------------------------------------------------------------------
    
    def fetch_historical_bars(
        self,
        symbol: str,
        timeframe: str = '1Min',
        limit: int = 500,
    ) -> pd.DataFrame:
        """
        Fetch historical bars via REST API.
        
        Args:
            symbol: Stock symbol
            timeframe: '1Min', '5Min', '15Min', '1Hour', '1Day'
            limit: Number of bars
            
        Returns:
            DataFrame with historical OHLCV
        """
        if not self.config.api_key:
            logger.error("API key not set")
            return pd.DataFrame()
        
        url = f"{self.config.data_url}/v2/stocks/{symbol}/bars"
        headers = {
            'APCA-API-KEY-ID': self.config.api_key,
            'APCA-API-SECRET-KEY': self.config.secret_key,
        }
        
        # Calculate start time
        end = datetime.now()
        if timeframe == '1Min':
            start = end - timedelta(minutes=limit)
        elif timeframe == '5Min':
            start = end - timedelta(minutes=limit * 5)
        else:
            start = end - timedelta(days=limit)
        
        params = {
            'timeframe': timeframe,
            'start': start.isoformat() + 'Z',
            'end': end.isoformat() + 'Z',
            'limit': limit,
            'feed': self.config.feed,
        }
        
        try:
            response = requests.get(url, headers=headers, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            bars = data.get('bars', [])
            if not bars:
                return pd.DataFrame()
            
            df = pd.DataFrame(bars)
            df['timestamp'] = pd.to_datetime(df['t'])
            df.set_index('timestamp', inplace=True)
            
            df = df.rename(columns={
                'o': 'open',
                'h': 'high',
                'l': 'low',
                'c': 'close',
                'v': 'volume',
                'n': 'trade_count',
                'vw': 'vwap',
            })
            
            df = df[['open', 'high', 'low', 'close', 'volume', 'trade_count', 'vwap']]
            
            logger.info(f"Fetched {len(df)} historical bars for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Historical data error: {e}")
            return pd.DataFrame()
    
    def preload_historical(self, timeframe: str = '1Min', n_bars: int = 500):
        """Preload historical data for all symbols."""
        for symbol in self.symbols:
            df = self.fetch_historical_bars(symbol, timeframe, n_bars)
            if not df.empty:
                logger.info(f"Preloaded {len(df)} bars for {symbol}")
    
    # -------------------------------------------------------------------------
    # CALLBACKS
    # -------------------------------------------------------------------------
    
    def on_candle(self, callback: Callable):
        """Register callback for new candles."""
        self._candle_callbacks.append(callback)
    
    def on_trade(self, callback: Callable):
        """Register callback for new trades."""
        self._trade_callbacks.append(callback)
    
    # -------------------------------------------------------------------------
    # UTILITY
    # -------------------------------------------------------------------------
    
    @property
    def is_connected(self) -> bool:
        return self._running and self._ws is not None and not self._ws.closed
    
    def get_buffer_stats(self) -> Dict[str, Dict[str, int]]:
        """Get buffer sizes."""
        stats = {}
        for symbol in self.symbols:
            stats[symbol] = {}
            for interval, agg in self._aggregators[symbol].items():
                stats[symbol][f'{interval}s'] = len(agg)
            stats[symbol]['trades'] = len(self._trade_buffers[symbol])
        return stats


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def create_alpaca_handler(
    symbols: List[str] = None,
    api_key: str = None,
    secret_key: str = None,
) -> AlpacaDataHandler:
    """
    Create and initialize an Alpaca data handler.
    
    Args:
        symbols: Stock symbols (default: AAPL, TSLA, MSFT)
        api_key: Alpaca API key (or set ALPACA_API_KEY env)
        secret_key: Alpaca secret key (or set ALPACA_SECRET_KEY env)
        
    Returns:
        Configured AlpacaDataHandler
    """
    if symbols is None:
        symbols = ['AAPL', 'TSLA', 'MSFT']
    
    config = AlpacaConfig(
        api_key=api_key or '',
        secret_key=secret_key or '',
    )
    
    return AlpacaDataHandler(symbols, config)


# =============================================================================
# DEMO
# =============================================================================

async def _demo():
    """Demo the Alpaca data handler."""
    print("=== Alpaca Data Handler Demo ===\n")
    
    # Check for credentials
    api_key = os.getenv('ALPACA_API_KEY')
    secret_key = os.getenv('ALPACA_SECRET_KEY')
    
    if not api_key or not secret_key:
        print("⚠️  Set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables")
        print("   Get free API keys at: https://alpaca.markets")
        return
    
    handler = AlpacaDataHandler(['AAPL', 'TSLA'])
    
    # Fetch some historical data first
    print("Fetching historical data...")
    handler.preload_historical('1Min', 100)
    
    # Connect for real-time
    await handler.connect()
    
    print("Streaming trades for 60 seconds...")
    await asyncio.sleep(60)
    
    # Check buffers
    print(f"\nBuffer stats: {handler.get_buffer_stats()}")
    
    # Get 5-second candles
    df = handler.get_buffer('AAPL', n_bars=10, interval='5s')
    print(f"\nLatest 5s candles for AAPL:\n{df}")
    
    # Get sentiment
    sentiment = handler.get_sentiment('AAPL', lookback_seconds=30)
    print(f"\nAAPL Trade Flow Sentiment:")
    print(f"  Volume imbalance: {sentiment.volume_imbalance:.3f}")
    print(f"  Trade imbalance: {sentiment.trade_imbalance:.3f}")
    print(f"  Institutional signal: {sentiment.institutional_signal:.3f}")
    
    await handler.disconnect()


if __name__ == '__main__':
    asyncio.run(_demo())
