"""
Market Data Fetcher Module.

This module handles fetching OHLCV (Open, High, Low, Close, Volume) data
from various data sources including Yahoo Finance and Alpha Vantage.
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Union, Dict, Any
from datetime import datetime, timedelta
from pathlib import Path
import logging
import time

try:
    import yfinance as yf
except ImportError:
    yf = None

logger = logging.getLogger(__name__)


class MarketDataFetcher:
    """
    Fetches market data from various sources.
    
    Supports:
    - Yahoo Finance (free, reliable for daily data)
    - Alpha Vantage (API key required, better intraday)
    - Local CSV files (for backtesting)
    
    Example:
        >>> fetcher = MarketDataFetcher()
        >>> data = fetcher.fetch("AAPL", start="2020-01-01", end="2024-01-01")
        >>> print(data.head())
    """
    
    def __init__(
        self,
        source: str = "yfinance",
        cache_dir: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        """
        Initialize the market data fetcher.
        
        Args:
            source: Data source - 'yfinance', 'alpha_vantage', or 'csv'
            cache_dir: Directory to cache downloaded data
            api_key: API key for premium data sources
        """
        self.source = source.lower()
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.api_key = api_key
        
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            
        # Validate source
        valid_sources = ['yfinance', 'alpha_vantage', 'csv']
        if self.source not in valid_sources:
            raise ValueError(f"Source must be one of {valid_sources}")
            
        # Check yfinance availability
        if self.source == 'yfinance' and yf is None:
            raise ImportError("yfinance is not installed. Run: pip install yfinance")
    
    def fetch(
        self,
        symbols: Union[str, List[str]],
        start: Optional[str] = None,
        end: Optional[str] = None,
        interval: str = "1d",
        auto_adjust: bool = True,
    ) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Fetch OHLCV data for given symbol(s).
        
        Args:
            symbols: Single symbol or list of symbols
            start: Start date (YYYY-MM-DD format)
            end: End date (YYYY-MM-DD format)
            interval: Data interval - 1m, 5m, 15m, 1h, 1d, 1wk, 1mo
            auto_adjust: Whether to auto-adjust for splits and dividends
            
        Returns:
            DataFrame with OHLCV data, or dict of DataFrames for multiple symbols
        """
        # Handle single symbol as list
        if isinstance(symbols, str):
            symbols = [symbols]
            single_symbol = True
        else:
            single_symbol = False
        
        # Set default dates if not provided
        if end is None:
            end = datetime.now().strftime("%Y-%m-%d")
        if start is None:
            start = (datetime.now() - timedelta(days=365 * 5)).strftime("%Y-%m-%d")
        
        # Fetch data based on source
        if self.source == 'yfinance':
            results = self._fetch_yfinance(symbols, start, end, interval, auto_adjust)
        elif self.source == 'alpha_vantage':
            results = self._fetch_alpha_vantage(symbols, start, end, interval)
        else:
            results = self._fetch_csv(symbols, start, end)
        
        # Return single DataFrame if single symbol was requested
        if single_symbol and len(results) == 1:
            return list(results.values())[0]
        
        return results
    
    def _fetch_yfinance(
        self,
        symbols: List[str],
        start: str,
        end: str,
        interval: str,
        auto_adjust: bool,
    ) -> Dict[str, pd.DataFrame]:
        """Fetch data from Yahoo Finance."""
        results = {}
        
        for symbol in symbols:
            try:
                # Check cache first
                cached_data = self._load_from_cache(symbol, start, end, interval)
                if cached_data is not None:
                    logger.info(f"Loaded {symbol} from cache")
                    results[symbol] = cached_data
                    continue
                
                logger.info(f"Fetching {symbol} from Yahoo Finance...")
                
                ticker = yf.Ticker(symbol)
                df = ticker.history(
                    start=start,
                    end=end,
                    interval=interval,
                    auto_adjust=auto_adjust,
                )
                
                if df.empty:
                    logger.warning(f"No data returned for {symbol}")
                    continue
                
                # Standardize column names
                df = self._standardize_columns(df)
                
                # Save to cache
                self._save_to_cache(df, symbol, interval)
                
                results[symbol] = df
                
                # Rate limiting for API
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error fetching {symbol}: {str(e)}")
                continue
        
        return results
    
    def _fetch_alpha_vantage(
        self,
        symbols: List[str],
        start: str,
        end: str,
        interval: str,
    ) -> Dict[str, pd.DataFrame]:
        """Fetch data from Alpha Vantage."""
        import requests
        
        if not self.api_key:
            raise ValueError("Alpha Vantage requires an API key")
        
        results = {}
        base_url = "https://www.alphavantage.co/query"
        
        # Map interval to Alpha Vantage function
        interval_map = {
            '1d': 'TIME_SERIES_DAILY_ADJUSTED',
            '1wk': 'TIME_SERIES_WEEKLY_ADJUSTED',
            '1mo': 'TIME_SERIES_MONTHLY_ADJUSTED',
        }
        
        function = interval_map.get(interval, 'TIME_SERIES_DAILY_ADJUSTED')
        
        for symbol in symbols:
            try:
                params = {
                    'function': function,
                    'symbol': symbol,
                    'apikey': self.api_key,
                    'outputsize': 'full',
                }
                
                response = requests.get(base_url, params=params)
                data = response.json()
                
                # Parse the response
                time_series_key = [k for k in data.keys() if 'Time Series' in k]
                if not time_series_key:
                    logger.warning(f"No data for {symbol} from Alpha Vantage")
                    continue
                
                time_series = data[time_series_key[0]]
                
                # Convert to DataFrame
                df = pd.DataFrame.from_dict(time_series, orient='index')
                df.index = pd.to_datetime(df.index)
                df = df.sort_index()
                
                # Rename columns
                column_map = {
                    '1. open': 'open',
                    '2. high': 'high',
                    '3. low': 'low',
                    '4. close': 'close',
                    '5. adjusted close': 'adj_close',
                    '6. volume': 'volume',
                }
                df = df.rename(columns=column_map)
                df = df.astype(float)
                
                # Filter by date range
                df = df.loc[start:end]
                
                results[symbol] = df
                
                # Rate limit (5 calls per minute for free tier)
                time.sleep(12)
                
            except Exception as e:
                logger.error(f"Error fetching {symbol} from Alpha Vantage: {str(e)}")
                continue
        
        return results
    
    def _fetch_csv(
        self,
        symbols: List[str],
        start: str,
        end: str,
    ) -> Dict[str, pd.DataFrame]:
        """Fetch data from local CSV files."""
        results = {}
        
        for symbol in symbols:
            try:
                filepath = self.cache_dir / f"{symbol}.csv"
                
                if not filepath.exists():
                    logger.warning(f"CSV file not found for {symbol}: {filepath}")
                    continue
                
                df = pd.read_csv(filepath, index_col=0, parse_dates=True)
                df = self._standardize_columns(df)
                
                # Filter by date range
                df = df.loc[start:end]
                
                results[symbol] = df
                
            except Exception as e:
                logger.error(f"Error loading CSV for {symbol}: {str(e)}")
                continue
        
        return results
    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names to lowercase."""
        df.columns = df.columns.str.lower()
        
        # Common column mappings
        column_map = {
            'adj close': 'adj_close',
            'adj. close': 'adj_close',
            'adjusted close': 'adj_close',
        }
        
        df = df.rename(columns=column_map)
        
        # Ensure required columns exist
        required = ['open', 'high', 'low', 'close', 'volume']
        missing = [col for col in required if col not in df.columns]
        
        if missing:
            logger.warning(f"Missing columns: {missing}")
        
        return df
    
    def _load_from_cache(
        self,
        symbol: str,
        start: str,
        end: str,
        interval: str,
    ) -> Optional[pd.DataFrame]:
        """Load data from cache if available and fresh."""
        if not self.cache_dir:
            return None
        
        cache_file = self.cache_dir / f"{symbol}_{interval}.parquet"
        
        if not cache_file.exists():
            return None
        
        # Check if cache is fresh (less than 1 day old for daily data)
        cache_age = datetime.now().timestamp() - cache_file.stat().st_mtime
        max_age = 86400 if interval == '1d' else 3600  # 1 day or 1 hour
        
        if cache_age > max_age:
            return None
        
        try:
            df = pd.read_parquet(cache_file)
            
            # Filter by requested date range
            df = df.loc[start:end]
            
            if df.empty:
                return None
            
            return df
            
        except Exception as e:
            logger.warning(f"Failed to load cache for {symbol}: {str(e)}")
            return None
    
    def _save_to_cache(
        self,
        df: pd.DataFrame,
        symbol: str,
        interval: str,
    ) -> None:
        """Save data to cache."""
        if not self.cache_dir:
            return
        
        try:
            cache_file = self.cache_dir / f"{symbol}_{interval}.parquet"
            df.to_parquet(cache_file)
            logger.debug(f"Saved {symbol} to cache")
        except Exception as e:
            logger.warning(f"Failed to save cache for {symbol}: {str(e)}")
    
    def fetch_multiple(
        self,
        symbols: List[str],
        start: Optional[str] = None,
        end: Optional[str] = None,
        interval: str = "1d",
        combine: bool = True,
    ) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Fetch data for multiple symbols efficiently.
        
        Args:
            symbols: List of symbols
            start: Start date
            end: End date
            interval: Data interval
            combine: If True, combine into single DataFrame with MultiIndex columns
            
        Returns:
            Combined DataFrame or dict of DataFrames
        """
        results = self.fetch(symbols, start, end, interval)
        
        if not combine:
            return results
        
        # Combine into single DataFrame with MultiIndex columns
        combined = pd.DataFrame()
        for symbol, df in results.items():
            for col in df.columns:
                combined[(symbol, col)] = df[col]
        
        combined.columns = pd.MultiIndex.from_tuples(combined.columns)
        
        return combined
    
    def get_available_symbols(self, exchange: str = "NYSE") -> List[str]:
        """
        Get list of available symbols for an exchange.
        
        Note: This is a simplified version. For production, use a proper
        symbol database or API.
        """
        # Common US stock symbols
        common_symbols = {
            'NYSE': [
                'JPM', 'BAC', 'C', 'WFC', 'GS',  # Banks
                'XOM', 'CVX', 'COP',  # Energy
                'JNJ', 'PFE', 'UNH', 'MRK',  # Healthcare
                'WMT', 'HD', 'MCD', 'NKE',  # Consumer
                'DIS', 'VZ', 'T',  # Media/Telecom
            ],
            'NASDAQ': [
                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META',  # Tech giants
                'NVDA', 'AMD', 'INTC', 'AVGO',  # Semiconductors
                'TSLA', 'NFLX', 'PYPL', 'ADBE',  # Growth
                'COST', 'SBUX', 'PEP', 'CMCSA',  # Consumer
            ],
            'ETF': [
                'SPY', 'QQQ', 'IWM', 'DIA',  # Index ETFs
                'GLD', 'SLV', 'USO',  # Commodities
                'TLT', 'IEF', 'LQD',  # Bonds
                'XLF', 'XLK', 'XLE', 'XLV',  # Sector ETFs
                'VXX', 'UVXY',  # Volatility
            ],
            'CRYPTO': [
                'BTC-USD', 'ETH-USD', 'BNB-USD', 'XRP-USD',
                'ADA-USD', 'SOL-USD', 'DOT-USD', 'DOGE-USD',
            ],
        }
        
        return common_symbols.get(exchange.upper(), [])


# Convenience functions
def fetch_stock(
    symbol: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    interval: str = "1d",
) -> pd.DataFrame:
    """Quick function to fetch stock data."""
    fetcher = MarketDataFetcher()
    return fetcher.fetch(symbol, start, end, interval)


def fetch_portfolio(
    symbols: List[str],
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
    """Quick function to fetch data for a portfolio of stocks."""
    fetcher = MarketDataFetcher()
    return fetcher.fetch(symbols, start, end)
