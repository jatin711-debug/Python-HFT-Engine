"""
News Fetcher Module.

This module handles fetching news articles from multiple sources
for sentiment analysis. It aggregates the top 100 news articles
per stock symbol from various reliable sources.
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging
import time
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from gnews import GNews
except ImportError:
    GNews = None

try:
    from newspaper import Article
except ImportError:
    Article = None

try:
    import feedparser
except ImportError:
    feedparser = None

try:
    import requests
except ImportError:
    requests = None

logger = logging.getLogger(__name__)


@dataclass
class NewsArticle:
    """Represents a single news article."""
    title: str
    description: str
    content: str
    url: str
    source: str
    published_date: datetime
    symbol: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'title': self.title,
            'description': self.description,
            'content': self.content,
            'url': self.url,
            'source': self.source,
            'published_date': self.published_date,
            'symbol': self.symbol,
        }


class NewsFetcher:
    """
    Fetches news articles from multiple sources for sentiment analysis.
    
    Sources supported:
    - Google News (via GNews library)
    - Yahoo Finance RSS
    - Financial news RSS feeds (Reuters, MarketWatch, etc.)
    - NewsAPI (if API key provided)
    
    Example:
        >>> fetcher = NewsFetcher()
        >>> articles = fetcher.fetch_news("AAPL", max_articles=100)
        >>> print(f"Fetched {len(articles)} articles")
    """
    
    # RSS feed URLs for major financial news sources
    RSS_FEEDS = {
        'yahoo_finance': 'https://finance.yahoo.com/rss/headline?s={symbol}',
        'google_finance': 'https://news.google.com/rss/search?q={symbol}+stock&hl=en-US&gl=US&ceid=US:en',
        'marketwatch': 'https://feeds.marketwatch.com/marketwatch/topstories/',
        'reuters_business': 'https://www.reutersagency.com/feed/?best-topics=business-finance&post_type=best',
        'cnbc': 'https://www.cnbc.com/id/100003114/device/rss/rss.html',
        'bloomberg': 'https://feeds.bloomberg.com/markets/news.rss',
        'seeking_alpha': 'https://seekingalpha.com/feed.xml',
        'benzinga': 'https://www.benzinga.com/feed',
    }
    
    # Company name mappings for better search
    COMPANY_NAMES = {
        'AAPL': 'Apple',
        'MSFT': 'Microsoft',
        'GOOGL': 'Google Alphabet',
        'AMZN': 'Amazon',
        'META': 'Meta Facebook',
        'TSLA': 'Tesla',
        'NVDA': 'Nvidia',
        'JPM': 'JPMorgan Chase',
        'V': 'Visa',
        'JNJ': 'Johnson Johnson',
        'WMT': 'Walmart',
        'PG': 'Procter Gamble',
        'MA': 'Mastercard',
        'HD': 'Home Depot',
        'DIS': 'Disney',
        'BAC': 'Bank of America',
        'XOM': 'Exxon Mobil',
        'NFLX': 'Netflix',
        'ADBE': 'Adobe',
        'CRM': 'Salesforce',
    }
    
    def __init__(
        self,
        news_api_key: Optional[str] = None,
        max_workers: int = 5,
        request_timeout: int = 10,
    ):
        """
        Initialize the news fetcher.
        
        Args:
            news_api_key: Optional API key for NewsAPI
            max_workers: Maximum concurrent threads for fetching
            request_timeout: Timeout for HTTP requests in seconds
        """
        self.news_api_key = news_api_key
        self.max_workers = max_workers
        self.request_timeout = request_timeout
        
        # Initialize GNews if available
        if GNews:
            self.gnews = GNews(
                language='en',
                country='US',
                period='7d',  # Last 7 days
                max_results=50,
            )
        else:
            self.gnews = None
            logger.warning("GNews not available. Install with: pip install gnews")
    
    def fetch_news(
        self,
        symbol: str,
        max_articles: int = 10,
        days_back: int = 7,
        include_content: bool = True,
    ) -> List[NewsArticle]:
        """
        Fetch news articles for a given stock symbol.
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            max_articles: Maximum number of articles to fetch
            days_back: How many days back to search
            include_content: Whether to fetch full article content
            
        Returns:
            List of NewsArticle objects
        """
        all_articles = []
        
        # Get company name for better search results
        company_name = self.COMPANY_NAMES.get(symbol, symbol)
        
        logger.info(f"Fetching news for {symbol} ({company_name})...")
        
        # 1. Fetch from Google News
        google_articles = self._fetch_google_news(symbol, company_name, max_articles // 3)
        all_articles.extend(google_articles)
        
        # 2. Fetch from RSS feeds
        rss_articles = self._fetch_rss_feeds(symbol, company_name, max_articles // 3)
        all_articles.extend(rss_articles)
        
        # 3. Fetch from NewsAPI if available
        if self.news_api_key:
            newsapi_articles = self._fetch_newsapi(symbol, company_name, max_articles // 3)
            all_articles.extend(newsapi_articles)
        
        # Remove duplicates based on URL
        seen_urls = set()
        unique_articles = []
        for article in all_articles:
            if article.url not in seen_urls:
                seen_urls.add(article.url)
                unique_articles.append(article)
        
        # Fetch full content if requested
        if include_content:
            unique_articles = self._fetch_article_content(unique_articles)
        
        # Sort by date (most recent first)
        unique_articles.sort(key=lambda x: x.published_date, reverse=True)
        
        # Limit to max_articles
        unique_articles = unique_articles[:max_articles]
        
        logger.info(f"Fetched {len(unique_articles)} articles for {symbol}")
        
        return unique_articles
    
    def _fetch_google_news(
        self,
        symbol: str,
        company_name: str,
        max_articles: int,
    ) -> List[NewsArticle]:
        """Fetch news from Google News using GNews library."""
        articles = []
        
        if not self.gnews:
            return articles
        
        try:
            # Search for stock symbol
            search_queries = [
                f"{symbol} stock",
                f"{company_name} stock",
                f"{symbol} shares",
            ]
            
            for query in search_queries:
                try:
                    results = self.gnews.get_news(query)
                    
                    for result in results[:max_articles // len(search_queries)]:
                        article = NewsArticle(
                            title=result.get('title', ''),
                            description=result.get('description', ''),
                            content='',  # Will be fetched later if needed
                            url=result.get('url', ''),
                            source=result.get('publisher', {}).get('title', 'Google News'),
                            published_date=self._parse_date(result.get('published date', '')),
                            symbol=symbol,
                        )
                        articles.append(article)
                        
                except Exception as e:
                    logger.warning(f"Error fetching Google News for query '{query}': {str(e)}")
                    continue
                
                time.sleep(0.5)  # Rate limiting
                
        except Exception as e:
            logger.error(f"Error fetching Google News: {str(e)}")
        
        return articles
    
    def _fetch_rss_feeds(
        self,
        symbol: str,
        company_name: str,
        max_articles: int,
    ) -> List[NewsArticle]:
        """Fetch news from RSS feeds."""
        articles = []
        
        if not feedparser:
            logger.warning("feedparser not available. Install with: pip install feedparser")
            return articles
        
        # Symbol-specific feeds
        symbol_feeds = {
            'yahoo': self.RSS_FEEDS['yahoo_finance'].format(symbol=symbol),
            'google': self.RSS_FEEDS['google_finance'].format(symbol=symbol),
        }
        
        # General market feeds
        general_feeds = {
            'marketwatch': self.RSS_FEEDS['marketwatch'],
            'cnbc': self.RSS_FEEDS['cnbc'],
        }
        
        all_feeds = {**symbol_feeds, **general_feeds}
        
        for feed_name, feed_url in all_feeds.items():
            try:
                feed = feedparser.parse(feed_url)
                
                for entry in feed.entries[:max_articles // len(all_feeds)]:
                    # Check if article is related to our symbol
                    title = entry.get('title', '')
                    summary = entry.get('summary', '')
                    
                    # For general feeds, filter by symbol/company name
                    if feed_name in ['marketwatch', 'cnbc']:
                        text = (title + ' ' + summary).lower()
                        if symbol.lower() not in text and company_name.lower() not in text:
                            continue
                    
                    article = NewsArticle(
                        title=title,
                        description=summary,
                        content='',
                        url=entry.get('link', ''),
                        source=feed_name,
                        published_date=self._parse_date(entry.get('published', '')),
                        symbol=symbol,
                    )
                    articles.append(article)
                    
            except Exception as e:
                logger.warning(f"Error fetching RSS feed '{feed_name}': {str(e)}")
                continue
        
        return articles
    
    def _fetch_newsapi(
        self,
        symbol: str,
        company_name: str,
        max_articles: int,
    ) -> List[NewsArticle]:
        """Fetch news from NewsAPI."""
        articles = []
        
        if not requests:
            return articles
        
        try:
            url = 'https://newsapi.org/v2/everything'
            
            params = {
                'q': f'{symbol} OR {company_name}',
                'language': 'en',
                'sortBy': 'publishedAt',
                'pageSize': min(max_articles, 100),  # NewsAPI limit
                'apiKey': self.news_api_key,
            }
            
            response = requests.get(url, params=params, timeout=self.request_timeout)
            data = response.json()
            
            if data.get('status') == 'ok':
                for item in data.get('articles', []):
                    article = NewsArticle(
                        title=item.get('title', ''),
                        description=item.get('description', ''),
                        content=item.get('content', ''),
                        url=item.get('url', ''),
                        source=item.get('source', {}).get('name', 'NewsAPI'),
                        published_date=self._parse_date(item.get('publishedAt', '')),
                        symbol=symbol,
                    )
                    articles.append(article)
                    
        except Exception as e:
            logger.error(f"Error fetching from NewsAPI: {str(e)}")
        
        return articles
    
    def _fetch_article_content(
        self,
        articles: List[NewsArticle],
    ) -> List[NewsArticle]:
        """Fetch full content for articles using newspaper3k."""
        if not Article:
            logger.warning("newspaper3k not available. Install with: pip install newspaper3k")
            return articles
        
        def fetch_content(article: NewsArticle) -> NewsArticle:
            try:
                news_article = Article(article.url)
                news_article.download()
                news_article.parse()
                article.content = news_article.text
            except Exception:
                # Keep the description as content if full fetch fails
                article.content = article.description
            return article
        
        # Use thread pool for concurrent fetching
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(fetch_content, article): article for article in articles}
            
            result_articles = []
            for future in as_completed(futures):
                try:
                    result_articles.append(future.result())
                except Exception:
                    result_articles.append(futures[future])
        
        return result_articles
    
    def _parse_date(self, date_str: str) -> datetime:
        """Parse date string to datetime object (timezone-naive)."""
        if not date_str:
            return datetime.now()
        
        # Common date formats
        formats = [
            '%Y-%m-%dT%H:%M:%SZ',
            '%Y-%m-%dT%H:%M:%S.%fZ',
            '%a, %d %b %Y %H:%M:%S %z',
            '%a, %d %b %Y %H:%M:%S GMT',
            '%Y-%m-%d %H:%M:%S',
            '%Y-%m-%d',
        ]
        
        for fmt in formats:
            try:
                parsed = datetime.strptime(date_str.strip(), fmt)
                # Convert to timezone-naive if timezone-aware
                if parsed.tzinfo is not None:
                    parsed = parsed.replace(tzinfo=None)
                return parsed
            except ValueError:
                continue
        
        # If no format matches, return current time
        return datetime.now()
    
    def fetch_batch(
        self,
        symbols: List[str],
        max_articles_per_symbol: int = 100,
    ) -> Dict[str, List[NewsArticle]]:
        """
        Fetch news for multiple symbols.
        
        Args:
            symbols: List of stock symbols
            max_articles_per_symbol: Maximum articles per symbol
            
        Returns:
            Dictionary mapping symbols to their articles
        """
        results = {}
        
        for symbol in symbols:
            articles = self.fetch_news(symbol, max_articles_per_symbol)
            results[symbol] = articles
            time.sleep(1)  # Rate limiting between symbols
        
        return results
    
    def to_dataframe(self, articles: List[NewsArticle]) -> pd.DataFrame:
        """Convert list of articles to pandas DataFrame."""
        if not articles:
            return pd.DataFrame()
        
        data = [article.to_dict() for article in articles]
        df = pd.DataFrame(data)
        df['published_date'] = pd.to_datetime(df['published_date'])
        df = df.sort_values('published_date', ascending=False)
        
        return df


# Convenience function
def fetch_stock_news(
    symbol: str,
    max_articles: int = 100,
    days_back: int = 7,
) -> List[NewsArticle]:
    """Quick function to fetch news for a stock."""
    fetcher = NewsFetcher()
    return fetcher.fetch_news(symbol, max_articles, days_back)
