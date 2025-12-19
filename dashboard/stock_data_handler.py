"""
Stock Data Handler for Yahoo Finance.

Provides real-time-like stock data by:
1. Fetching intraday data from Yahoo Finance
2. Buffering for feature calculation
3. Polling for updates during market hours

Usage:
    handler = StockDataHandler(['AAPL', 'TSLA', 'MSFT'])
    handler.preload_historical(days=5)
    
    # Get data for feature calculation
    df = handler.get_buffer('AAPL', n_bars=100)
    
    # Get latest price
    price = handler.get_latest_price('AAPL')
    
    # Refresh data (call every 60 seconds)
    handler.refresh_data()
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import logging
import time

try:
    import yfinance as yf
    YF_AVAILABLE = True
except ImportError:
    YF_AVAILABLE = False
    logging.warning("yfinance not installed. Run: pip install yfinance")

logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class StockSentiment:
    """
    Stock sentiment based on price action and volume.
    
    Mirrors the crypto sentiment structure for compatibility.
    """
    timestamp: datetime
    symbol: str
    buy_volume: int = 0
    sell_volume: int = 0
    buy_trades: int = 0
    sell_trades: int = 0
    large_buy_volume: int = 0
    large_sell_volume: int = 0
    
    @property
    def volume_imbalance(self) -> float:
        """Buy/sell volume imbalance (-1 to 1)."""
        total = self.buy_volume + self.sell_volume
        if total == 0:
            return 0
        return (self.buy_volume - self.sell_volume) / total
    
    @property
    def trade_imbalance(self) -> float:
        """Buy/sell trade count imbalance (-1 to 1)."""
        total = self.buy_trades + self.sell_trades
        if total == 0:
            return 0
        return (self.buy_trades - self.sell_trades) / total
    
    @property
    def whale_signal(self) -> float:
        """Large order imbalance (-1 to 1)."""
        total = self.large_buy_volume + self.large_sell_volume
        if total == 0:
            return 0
        return (self.large_buy_volume - self.large_sell_volume) / total


# =============================================================================
# STOCK DATA HANDLER
# =============================================================================

class StockDataHandler:
    """
    Stock data handler using Yahoo Finance.
    
    Features:
    - Intraday data fetching (1-minute bars)
    - Data buffering for feature calculation
    - Sentiment approximation from volume/price
    - Automatic refresh during market hours
    """
    
    def __init__(
        self,
        symbols: List[str],
        buffer_size: int = 500,
    ):
        """
        Initialize stock data handler.
        
        Args:
            symbols: List of stock symbols (e.g., ['AAPL', 'TSLA', 'MSFT'])
            buffer_size: Maximum number of bars to keep in buffer
        """
        if not YF_AVAILABLE:
            raise ImportError("yfinance is required. Install with: pip install yfinance")
        
        self.symbols = [s.upper() for s in symbols]
        self.buffer_size = buffer_size
        
        # Data buffers per symbol
        self.buffers: Dict[str, pd.DataFrame] = {}
        
        # Last refresh time
        self.last_refresh: Dict[str, datetime] = {}
        
        # Tickers cache
        self._tickers: Dict[str, Any] = {}
        
        logger.info(f"StockDataHandler initialized for: {self.symbols}")
    
    def preload_historical(self, days: int = 5, interval: str = "1m"):
        """
        Preload historical intraday data.
        
        Args:
            days: Number of days of history to load
            interval: Data interval ('1m', '5m', '15m', '1h', '1d')
        
        Note:
            Yahoo Finance limits 1-minute data to last 7 days.
            For longer periods, use '5m' or '1h'.
        """
        # Adjust interval based on days (Yahoo limits)
        if days > 7 and interval == "1m":
            logger.warning(f"Yahoo Finance limits 1m data to 7 days. Using 5m instead.")
            interval = "5m"
        
        if days > 60 and interval in ["1m", "5m"]:
            logger.warning(f"Yahoo Finance limits intraday data to 60 days. Using 1h.")
            interval = "1h"
        
        for symbol in self.symbols:
            try:
                logger.info(f"Loading {days}d of {interval} data for {symbol}...")
                
                ticker = yf.Ticker(symbol)
                self._tickers[symbol] = ticker
                
                # Fetch data
                df = ticker.history(period=f"{days}d", interval=interval)
                
                if df.empty:
                    logger.warning(f"No data returned for {symbol}")
                    continue
                
                # Standardize columns
                df = self._standardize_columns(df)
                
                # Store in buffer
                self.buffers[symbol] = df.tail(self.buffer_size)
                self.last_refresh[symbol] = datetime.now()
                
                logger.info(f"Loaded {len(self.buffers[symbol])} bars for {symbol}")
                
                # Rate limiting
                time.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Error loading {symbol}: {e}")
    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names to lowercase."""
        df = df.copy()
        
        # Rename columns to lowercase
        df.columns = [c.lower() for c in df.columns]
        
        # Ensure required columns exist
        required = ['open', 'high', 'low', 'close', 'volume']
        for col in required:
            if col not in df.columns:
                logger.warning(f"Missing column: {col}")
        
        return df
    
    def get_buffer(self, symbol: str, n_bars: int = 100) -> pd.DataFrame:
        """
        Get last N bars from buffer.
        
        Args:
            symbol: Stock symbol
            n_bars: Number of bars to return
        
        Returns:
            DataFrame with OHLCV data
        """
        symbol = symbol.upper()
        
        if symbol not in self.buffers:
            logger.warning(f"No buffer for {symbol}")
            return pd.DataFrame()
        
        return self.buffers[symbol].tail(n_bars).copy()
    
    def get_latest_price(self, symbol: str) -> float:
        """Get the latest closing price."""
        symbol = symbol.upper()
        
        if symbol not in self.buffers or len(self.buffers[symbol]) == 0:
            return 0.0
        
        return float(self.buffers[symbol]['close'].iloc[-1])
    
    def get_latest_bar(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get the latest bar as a dictionary."""
        symbol = symbol.upper()
        
        if symbol not in self.buffers or len(self.buffers[symbol]) == 0:
            return None
        
        bar = self.buffers[symbol].iloc[-1]
        return {
            'open': float(bar['open']),
            'high': float(bar['high']),
            'low': float(bar['low']),
            'close': float(bar['close']),
            'volume': int(bar['volume']),
            'timestamp': bar.name.isoformat() if hasattr(bar.name, 'isoformat') else str(bar.name),
        }
    
    def refresh_data(self, interval: str = "1m"):
        """
        Refresh data for all symbols.
        
        Call this periodically (every 60 seconds) during market hours.
        """
        for symbol in self.symbols:
            try:
                ticker = self._tickers.get(symbol) or yf.Ticker(symbol)
                self._tickers[symbol] = ticker
                
                # Fetch latest data (last 1 day)
                df = ticker.history(period="1d", interval=interval)
                
                if df.empty:
                    logger.debug(f"No new data for {symbol}")
                    continue
                
                # Standardize
                df = self._standardize_columns(df)
                
                # Append new bars to buffer
                if symbol in self.buffers:
                    # Find new rows only
                    existing_times = set(self.buffers[symbol].index)
                    new_rows = df[~df.index.isin(existing_times)]
                    
                    if not new_rows.empty:
                        self.buffers[symbol] = pd.concat([
                            self.buffers[symbol],
                            new_rows
                        ]).tail(self.buffer_size)
                        
                        logger.debug(f"Added {len(new_rows)} new bars for {symbol}")
                else:
                    self.buffers[symbol] = df.tail(self.buffer_size)
                
                self.last_refresh[symbol] = datetime.now()
                
            except Exception as e:
                logger.error(f"Error refreshing {symbol}: {e}")
            
            # Rate limiting
            time.sleep(0.2)
    
    def get_sentiment(
        self,
        symbol: str,
        lookback_seconds: int = 60,
    ) -> StockSentiment:
        """
        Estimate sentiment from price action and volume.
        
        Since we don't have tick data for stocks, we approximate
        buy/sell pressure from close position within bar.
        
        Args:
            symbol: Stock symbol
            lookback_seconds: Not used (kept for API compatibility)
        
        Returns:
            StockSentiment object
        """
        symbol = symbol.upper()
        
        if symbol not in self.buffers or len(self.buffers[symbol]) < 10:
            return StockSentiment(
                timestamp=datetime.now(),
                symbol=symbol,
            )
        
        # Get recent bars
        df = self.buffers[symbol].tail(10)
        
        buy_volume = 0
        sell_volume = 0
        buy_trades = 0
        sell_trades = 0
        large_buy = 0
        large_sell = 0
        
        avg_volume = df['volume'].mean()
        
        for _, bar in df.iterrows():
            # Close position in range (0 = at low, 1 = at high)
            bar_range = bar['high'] - bar['low']
            if bar_range > 0:
                close_position = (bar['close'] - bar['low']) / bar_range
            else:
                close_position = 0.5
            
            vol = bar['volume']
            
            # Classify as buy or sell based on close position
            if close_position > 0.6:  # Closed near high = buying pressure
                buy_volume += vol
                buy_trades += 1
                if vol > avg_volume * 1.5:
                    large_buy += vol
            elif close_position < 0.4:  # Closed near low = selling pressure
                sell_volume += vol
                sell_trades += 1
                if vol > avg_volume * 1.5:
                    large_sell += vol
            else:  # Neutral
                buy_volume += vol // 2
                sell_volume += vol // 2
        
        return StockSentiment(
            timestamp=datetime.now(),
            symbol=symbol,
            buy_volume=int(buy_volume),
            sell_volume=int(sell_volume),
            buy_trades=buy_trades,
            sell_trades=sell_trades,
            large_buy_volume=int(large_buy),
            large_sell_volume=int(large_sell),
        )
    
    def is_market_open(self) -> bool:
        """
        Check if US stock market is currently open.
        
        Market hours: 9:30 AM - 4:00 PM Eastern Time
        """
        now = datetime.now()
        
        # Check weekday (0 = Monday, 6 = Sunday)
        if now.weekday() >= 5:  # Weekend
            return False
        
        # Check time (rough approximation, not accounting for holidays)
        # This is in local time - adjust if needed
        hour = now.hour
        minute = now.minute
        
        market_open = 9 * 60 + 30  # 9:30 AM
        market_close = 16 * 60     # 4:00 PM
        current_time = hour * 60 + minute
        
        return market_open <= current_time <= market_close
    
    def add_symbol(self, symbol: str, days: int = 5, interval: str = "1m"):
        """Add a new symbol to track."""
        symbol = symbol.upper()
        
        if symbol in self.symbols:
            logger.info(f"{symbol} already being tracked")
            return
        
        self.symbols.append(symbol)
        
        # Load historical data
        try:
            ticker = yf.Ticker(symbol)
            self._tickers[symbol] = ticker
            
            df = ticker.history(period=f"{days}d", interval=interval)
            
            if not df.empty:
                df = self._standardize_columns(df)
                self.buffers[symbol] = df.tail(self.buffer_size)
                self.last_refresh[symbol] = datetime.now()
                logger.info(f"Added {symbol} with {len(self.buffers[symbol])} bars")
            else:
                logger.warning(f"No data available for {symbol}")
                
        except Exception as e:
            logger.error(f"Error adding {symbol}: {e}")
    
    def remove_symbol(self, symbol: str):
        """Remove a symbol from tracking."""
        symbol = symbol.upper()
        
        if symbol in self.symbols:
            self.symbols.remove(symbol)
        if symbol in self.buffers:
            del self.buffers[symbol]
        if symbol in self._tickers:
            del self._tickers[symbol]
        if symbol in self.last_refresh:
            del self.last_refresh[symbol]
        
        logger.info(f"Removed {symbol} from tracking")
    
    def get_all_prices(self) -> Dict[str, float]:
        """Get latest prices for all symbols."""
        return {
            symbol: self.get_latest_price(symbol)
            for symbol in self.symbols
        }


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test the handler
    handler = StockDataHandler(['AAPL', 'TSLA', 'MSFT'])
    
    print("Loading historical data...")
    handler.preload_historical(days=5, interval="5m")
    
    print("\nBuffer sizes:")
    for symbol in handler.symbols:
        if symbol in handler.buffers:
            print(f"  {symbol}: {len(handler.buffers[symbol])} bars")
    
    print("\nLatest prices:")
    for symbol, price in handler.get_all_prices().items():
        print(f"  {symbol}: ${price:.2f}")
    
    print("\nLatest bar for AAPL:")
    bar = handler.get_latest_bar('AAPL')
    if bar:
        print(f"  O: ${bar['open']:.2f}, H: ${bar['high']:.2f}, L: ${bar['low']:.2f}, C: ${bar['close']:.2f}")
    
    print("\nSentiment for AAPL:")
    sentiment = handler.get_sentiment('AAPL')
    print(f"  Volume imbalance: {sentiment.volume_imbalance:.2f}")
    print(f"  Trade imbalance: {sentiment.trade_imbalance:.2f}")
    print(f"  Whale signal: {sentiment.whale_signal:.2f}")
    
    print("\nMarket open:", handler.is_market_open())
    
    print("\n✅ Stock data handler tests passed!")
