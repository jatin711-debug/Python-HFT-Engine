"""
Data fetchers module.
"""

from .market_data import MarketDataFetcher
from .news_fetcher import NewsFetcher

__all__ = ['MarketDataFetcher', 'NewsFetcher']
