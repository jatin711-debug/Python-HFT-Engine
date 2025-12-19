"""
Data fetchers module.
"""

from .market_data import MarketDataFetcher
from .news_fetcher import NewsFetcher
from .binance_realtime import (
    BinanceDataHandler,
    CandleBuffer,
    TradeBuffer,
    Candle,
    Trade,
    CryptoSentiment,
    create_binance_handler,
)
from .alpaca_realtime import (
    AlpacaDataHandler,
    AlpacaConfig,
    StockTrade,
    StockCandle,
    StockSentiment,
    CandleAggregator,
    create_alpaca_handler,
)

__all__ = [
    'MarketDataFetcher',
    'NewsFetcher',
    # Binance
    'BinanceDataHandler',
    'CandleBuffer',
    'TradeBuffer',
    'Candle',
    'Trade',
    'CryptoSentiment',
    'create_binance_handler',
    # Alpaca
    'AlpacaDataHandler',
    'AlpacaConfig',
    'StockTrade',
    'StockCandle',
    'StockSentiment',
    'CandleAggregator',
    'create_alpaca_handler',
]
