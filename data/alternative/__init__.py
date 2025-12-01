"""
Alternative Data Sources Module.

Provides access to alternative data sources used by institutional investors:
- Reddit sentiment (retail sentiment indicator)
- SEC EDGAR filings (insider transactions, material events)
"""

from .reddit_fetcher import (
    RedditSentimentFetcher,
    RedditPost,
)

from .sec_fetcher import (
    SECFetcher,
    SECFiling,
    InsiderTransaction,
    InstitutionalHolding,
    InstitutionalHoldingsAnalyzer,
)

__all__ = [
    # Reddit
    'RedditSentimentFetcher',
    'RedditPost',
    # SEC
    'SECFetcher',
    'SECFiling',
    'InsiderTransaction',
    'InstitutionalHolding',
    'InstitutionalHoldingsAnalyzer',
]
