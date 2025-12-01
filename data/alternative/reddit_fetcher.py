"""
Reddit Sentiment Fetcher Module.

Fetches sentiment data from Reddit communities:
- r/wallstreetbets (retail sentiment, meme stocks)
- r/stocks (general stock discussion)
- r/investing (long-term investing)
- r/options (options sentiment)
- r/SecurityAnalysis (fundamental analysis)

Uses PRAW (Python Reddit API Wrapper) for data fetching.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging
import time
import re
from collections import defaultdict

try:
    import praw
    PRAW_AVAILABLE = True
except ImportError:
    PRAW_AVAILABLE = False
    praw = None

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class RedditPost:
    """Container for Reddit post data."""
    title: str
    selftext: str
    score: int
    num_comments: int
    created_utc: datetime
    subreddit: str
    author: str
    url: str
    upvote_ratio: float
    symbols_mentioned: List[str]
    sentiment_score: float = 0.0


class RedditSentimentFetcher:
    """
    Fetches and analyzes sentiment from Reddit financial communities.
    
    This captures retail investor sentiment which can be a contrarian
    indicator or early signal for momentum in certain stocks.
    
    Example:
        >>> fetcher = RedditSentimentFetcher(
        ...     client_id='your_id',
        ...     client_secret='your_secret'
        ... )
        >>> sentiment = fetcher.get_symbol_sentiment('AAPL')
    """
    
    # Financial subreddits to monitor
    SUBREDDITS = [
        'wallstreetbets',
        'stocks', 
        'investing',
        'options',
        'SecurityAnalysis',
        'stockmarket',
        'ValueInvesting',
        'Daytrading',
    ]
    
    # Common stock symbol patterns
    SYMBOL_PATTERN = re.compile(r'\$([A-Z]{1,5})\b|\b([A-Z]{2,5})\b')
    
    # Exclude common words that look like tickers
    EXCLUDED_WORDS = {
        'A', 'I', 'AM', 'PM', 'CEO', 'CFO', 'IPO', 'USA', 'UK', 'EU',
        'GDP', 'CPI', 'FED', 'SEC', 'NYSE', 'NASDAQ', 'ETF', 'SPAC',
        'DD', 'TL', 'DR', 'TLDR', 'OP', 'IMO', 'IMHO', 'FOMO', 'YOLO',
        'LOL', 'WTF', 'ATH', 'ATL', 'EOD', 'EOW', 'EOM', 'IV', 'DTE',
        'OTM', 'ITM', 'ATM', 'PE', 'EPS', 'PS', 'PB', 'ROE', 'ROI',
        'EBITDA', 'FCF', 'DCF', 'NPV', 'IRR', 'CAGR', 'YOY', 'QOQ',
        'THE', 'AND', 'FOR', 'ARE', 'BUT', 'NOT', 'YOU', 'ALL', 'CAN',
        'HER', 'WAS', 'ONE', 'OUR', 'OUT', 'HAS', 'HIS', 'HOW', 'ITS',
        'MAY', 'NEW', 'NOW', 'OLD', 'SEE', 'WAY', 'WHO', 'BOY', 'DID',
        'GET', 'LET', 'PUT', 'SAY', 'SHE', 'TOO', 'USE', 'AI', 'ML',
    }
    
    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        user_agent: str = "TradingEngine/1.0",
    ):
        """
        Initialize Reddit sentiment fetcher.
        
        Args:
            client_id: Reddit API client ID
            client_secret: Reddit API client secret
            user_agent: User agent string for API requests
        """
        self.client_id = client_id
        self.client_secret = client_secret
        self.user_agent = user_agent
        
        self.reddit = None
        if PRAW_AVAILABLE and client_id and client_secret:
            try:
                self.reddit = praw.Reddit(
                    client_id=client_id,
                    client_secret=client_secret,
                    user_agent=user_agent,
                )
                logger.info("Reddit API initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize Reddit API: {e}")
        else:
            logger.warning(
                "PRAW not available or credentials missing. "
                "Install with: pip install praw"
            )
        
        # Initialize sentiment analyzer
        if VADER_AVAILABLE:
            self.sentiment_analyzer = SentimentIntensityAnalyzer()
        else:
            self.sentiment_analyzer = None
            logger.warning("VADER not available for sentiment analysis")
    
    def fetch_subreddit_posts(
        self,
        subreddit_name: str,
        time_filter: str = 'day',
        limit: int = 100,
    ) -> List[RedditPost]:
        """
        Fetch posts from a specific subreddit.
        
        Args:
            subreddit_name: Name of subreddit
            time_filter: 'hour', 'day', 'week', 'month', 'year', 'all'
            limit: Maximum number of posts to fetch
            
        Returns:
            List of RedditPost objects
        """
        if not self.reddit:
            return self._fetch_without_api(subreddit_name, limit)
        
        posts = []
        
        try:
            subreddit = self.reddit.subreddit(subreddit_name)
            
            for submission in subreddit.top(time_filter=time_filter, limit=limit):
                # Extract symbols mentioned
                text = f"{submission.title} {submission.selftext}"
                symbols = self._extract_symbols(text)
                
                # Calculate sentiment
                sentiment = self._analyze_sentiment(text)
                
                post = RedditPost(
                    title=submission.title,
                    selftext=submission.selftext[:1000] if submission.selftext else "",
                    score=submission.score,
                    num_comments=submission.num_comments,
                    created_utc=datetime.fromtimestamp(submission.created_utc),
                    subreddit=subreddit_name,
                    author=str(submission.author) if submission.author else "[deleted]",
                    url=submission.url,
                    upvote_ratio=submission.upvote_ratio,
                    symbols_mentioned=symbols,
                    sentiment_score=sentiment,
                )
                posts.append(post)
                
        except Exception as e:
            logger.warning(f"Error fetching from r/{subreddit_name}: {e}")
        
        return posts
    
    def _fetch_without_api(
        self,
        subreddit_name: str,
        limit: int,
    ) -> List[RedditPost]:
        """
        Fetch posts without API using public JSON endpoint.
        
        Note: This is rate-limited and less reliable than the API.
        """
        import requests
        
        posts = []
        url = f"https://www.reddit.com/r/{subreddit_name}/top.json?t=day&limit={limit}"
        
        headers = {'User-Agent': self.user_agent}
        
        try:
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                for child in data.get('data', {}).get('children', []):
                    post_data = child.get('data', {})
                    
                    text = f"{post_data.get('title', '')} {post_data.get('selftext', '')}"
                    symbols = self._extract_symbols(text)
                    sentiment = self._analyze_sentiment(text)
                    
                    post = RedditPost(
                        title=post_data.get('title', ''),
                        selftext=post_data.get('selftext', '')[:1000],
                        score=post_data.get('score', 0),
                        num_comments=post_data.get('num_comments', 0),
                        created_utc=datetime.fromtimestamp(post_data.get('created_utc', 0)),
                        subreddit=subreddit_name,
                        author=post_data.get('author', '[deleted]'),
                        url=post_data.get('url', ''),
                        upvote_ratio=post_data.get('upvote_ratio', 0.5),
                        symbols_mentioned=symbols,
                        sentiment_score=sentiment,
                    )
                    posts.append(post)
                    
            else:
                logger.warning(f"Reddit API returned status {response.status_code}")
                
        except Exception as e:
            logger.warning(f"Error fetching Reddit data: {e}")
        
        return posts
    
    def get_symbol_sentiment(
        self,
        symbol: str,
        days_back: int = 7,
        subreddits: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Get aggregated sentiment for a specific symbol.
        
        Args:
            symbol: Stock symbol to analyze
            days_back: How many days of data to analyze
            subreddits: Specific subreddits to search (default: all)
            
        Returns:
            Dictionary with sentiment metrics
        """
        subreddits = subreddits or self.SUBREDDITS
        
        all_posts = []
        symbol_upper = symbol.upper()
        
        for subreddit in subreddits:
            posts = self.fetch_subreddit_posts(
                subreddit,
                time_filter='week' if days_back > 1 else 'day',
                limit=100,
            )
            
            # Filter posts mentioning this symbol
            relevant_posts = [
                p for p in posts 
                if symbol_upper in p.symbols_mentioned
            ]
            
            all_posts.extend(relevant_posts)
            
            # Rate limiting
            time.sleep(0.5)
        
        if not all_posts:
            return {
                'symbol': symbol,
                'mention_count': 0,
                'avg_sentiment': 0.0,
                'sentiment_std': 0.0,
                'bullish_ratio': 0.5,
                'total_score': 0,
                'avg_score': 0,
                'total_comments': 0,
                'trending': False,
            }
        
        # Calculate metrics
        sentiments = [p.sentiment_score for p in all_posts]
        scores = [p.score for p in all_posts]
        
        # Weight sentiment by post score (engagement)
        weighted_sentiment = sum(
            p.sentiment_score * np.log1p(p.score + 1) 
            for p in all_posts
        ) / sum(np.log1p(p.score + 1) for p in all_posts)
        
        return {
            'symbol': symbol,
            'mention_count': len(all_posts),
            'avg_sentiment': np.mean(sentiments),
            'weighted_sentiment': weighted_sentiment,
            'sentiment_std': np.std(sentiments),
            'bullish_ratio': sum(1 for s in sentiments if s > 0.05) / len(sentiments),
            'total_score': sum(scores),
            'avg_score': np.mean(scores),
            'total_comments': sum(p.num_comments for p in all_posts),
            'trending': len(all_posts) > 10 and np.mean(scores) > 100,
            'top_posts': sorted(all_posts, key=lambda x: x.score, reverse=True)[:5],
        }
    
    def get_trending_symbols(
        self,
        subreddits: Optional[List[str]] = None,
        min_mentions: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Get currently trending symbols on Reddit.
        
        Returns:
            List of trending symbols with metrics
        """
        subreddits = subreddits or self.SUBREDDITS
        symbol_data = defaultdict(lambda: {
            'mentions': 0,
            'total_score': 0,
            'sentiments': [],
            'subreddits': set(),
        })
        
        for subreddit in subreddits:
            posts = self.fetch_subreddit_posts(subreddit, time_filter='day', limit=100)
            
            for post in posts:
                for symbol in post.symbols_mentioned:
                    symbol_data[symbol]['mentions'] += 1
                    symbol_data[symbol]['total_score'] += post.score
                    symbol_data[symbol]['sentiments'].append(post.sentiment_score)
                    symbol_data[symbol]['subreddits'].add(subreddit)
            
            time.sleep(0.5)
        
        # Filter and sort by mentions
        trending = []
        for symbol, data in symbol_data.items():
            if data['mentions'] >= min_mentions:
                trending.append({
                    'symbol': symbol,
                    'mentions': data['mentions'],
                    'total_score': data['total_score'],
                    'avg_sentiment': np.mean(data['sentiments']),
                    'sentiment_std': np.std(data['sentiments']),
                    'subreddit_count': len(data['subreddits']),
                    'bullish_ratio': sum(1 for s in data['sentiments'] if s > 0.05) / len(data['sentiments']),
                })
        
        # Sort by engagement (mentions * avg score)
        trending.sort(
            key=lambda x: x['mentions'] * np.log1p(x['total_score']),
            reverse=True
        )
        
        return trending[:20]
    
    def _extract_symbols(self, text: str) -> List[str]:
        """Extract stock symbols from text."""
        symbols = set()
        
        # Find $SYMBOL patterns
        matches = self.SYMBOL_PATTERN.findall(text)
        
        for match in matches:
            # match is a tuple from regex groups
            symbol = match[0] or match[1]
            if symbol and symbol not in self.EXCLUDED_WORDS:
                symbols.add(symbol)
        
        return list(symbols)
    
    def _analyze_sentiment(self, text: str) -> float:
        """Analyze sentiment of text."""
        if not self.sentiment_analyzer or not text:
            return 0.0
        
        try:
            scores = self.sentiment_analyzer.polarity_scores(text)
            return scores['compound']
        except Exception:
            return 0.0
    
    def get_wsb_sentiment_features(
        self,
        df: pd.DataFrame,
        symbol: str,
    ) -> pd.DataFrame:
        """
        Add Reddit sentiment features to a dataframe.
        
        Args:
            df: Price dataframe with datetime index
            symbol: Stock symbol
            
        Returns:
            DataFrame with added sentiment features
        """
        df = df.copy()
        
        # Get sentiment data
        sentiment_data = self.get_symbol_sentiment(symbol, days_back=7)
        
        # Add static features (will be same for all rows in recent data)
        df['reddit_mentions'] = sentiment_data['mention_count']
        df['reddit_sentiment'] = sentiment_data.get('weighted_sentiment', 0)
        df['reddit_bullish_ratio'] = sentiment_data['bullish_ratio']
        df['reddit_trending'] = int(sentiment_data['trending'])
        
        # Normalize mentions to a score
        df['reddit_hype_score'] = np.clip(
            sentiment_data['mention_count'] / 50,  # Normalize
            0, 1
        )
        
        # Combined Reddit signal
        df['reddit_signal'] = (
            df['reddit_sentiment'] * 0.5 +
            (df['reddit_bullish_ratio'] - 0.5) * 0.3 +
            df['reddit_hype_score'] * 0.2
        )
        
        return df
