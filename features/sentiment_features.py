"""
Sentiment Features Module.

This module provides sentiment analysis features from news articles
using multiple NLP models for robust sentiment scoring.

Ensemble approach:
- VADER: Fast, rule-based (40% weight)
- TextBlob: Pattern-based (20% weight)
- FinBERT: Deep learning, finance-specific (40% weight)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

# Try to import sentiment libraries
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False
    logger.warning("VADER not available. Install with: pip install vaderSentiment")

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False
    logger.warning("TextBlob not available. Install with: pip install textblob")

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("Transformers not available. Install with: pip install transformers torch")


@dataclass
class SentimentScore:
    """Container for sentiment analysis results."""
    vader_score: float
    textblob_score: float
    finbert_score: float
    ensemble_score: float
    confidence: float
    label: str  # 'bullish', 'bearish', 'neutral'
    

class SentimentFeatures:
    """
    Multi-model sentiment analyzer for financial news.
    
    Uses an ensemble of:
    - VADER: Fast rule-based analyzer (40% weight)
    - TextBlob: Pattern-based analyzer (20% weight)
    - FinBERT: Transformer model fine-tuned for finance (40% weight)
    
    Example:
        >>> analyzer = SentimentFeatures()
        >>> score = analyzer.analyze_text("Apple beats Q3 earnings expectations")
        >>> print(f"Sentiment: {score.label} ({score.ensemble_score:.2f})")
    """
    
    def __init__(
        self,
        vader_weight: float = 0.4,
        textblob_weight: float = 0.2,
        finbert_weight: float = 0.4,
        finbert_model: str = "ProsusAI/finbert",
        use_gpu: bool = True,
        max_sequence_length: int = 512,
    ):
        """
        Initialize sentiment analyzer.
        
        Args:
            vader_weight: Weight for VADER scores
            textblob_weight: Weight for TextBlob scores
            finbert_weight: Weight for FinBERT scores
            finbert_model: HuggingFace model ID for FinBERT
            use_gpu: Whether to use GPU for transformer models
            max_sequence_length: Maximum token length for FinBERT
        """
        self.vader_weight = vader_weight
        self.textblob_weight = textblob_weight
        self.finbert_weight = finbert_weight
        self.max_sequence_length = max_sequence_length
        
        # Initialize VADER
        self.vader_analyzer = None
        if VADER_AVAILABLE:
            self.vader_analyzer = SentimentIntensityAnalyzer()
            logger.info("VADER sentiment analyzer initialized")
        
        # Initialize FinBERT
        self.finbert_tokenizer = None
        self.finbert_model = None
        self.device = None
        
        if TRANSFORMERS_AVAILABLE:
            try:
                self.device = torch.device(
                    "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
                )
                self.finbert_tokenizer = AutoTokenizer.from_pretrained(finbert_model)
                self.finbert_model = AutoModelForSequenceClassification.from_pretrained(
                    finbert_model
                )
                self.finbert_model.to(self.device)
                self.finbert_model.eval()
                logger.info(f"FinBERT initialized on {self.device}")
            except Exception as e:
                logger.warning(f"Failed to initialize FinBERT: {e}")
                self.finbert_model = None
        
        # Adjust weights if models are not available
        self._adjust_weights()
    
    def _adjust_weights(self):
        """Adjust weights based on available models."""
        total_weight = 0.0
        
        if VADER_AVAILABLE and self.vader_analyzer:
            total_weight += self.vader_weight
        else:
            self.vader_weight = 0.0
        
        if TEXTBLOB_AVAILABLE:
            total_weight += self.textblob_weight
        else:
            self.textblob_weight = 0.0
        
        if self.finbert_model is not None:
            total_weight += self.finbert_weight
        else:
            self.finbert_weight = 0.0
        
        # Normalize weights
        if total_weight > 0:
            self.vader_weight /= total_weight
            self.textblob_weight /= total_weight
            self.finbert_weight /= total_weight
            
        logger.info(
            f"Sentiment weights: VADER={self.vader_weight:.2f}, "
            f"TextBlob={self.textblob_weight:.2f}, "
            f"FinBERT={self.finbert_weight:.2f}"
        )
    
    def analyze_text(self, text: str) -> SentimentScore:
        """
        Analyze sentiment of a single text.
        
        Args:
            text: Text to analyze
            
        Returns:
            SentimentScore with all model scores and ensemble
        """
        if not text or not isinstance(text, str):
            return SentimentScore(
                vader_score=0.0,
                textblob_score=0.0,
                finbert_score=0.0,
                ensemble_score=0.0,
                confidence=0.0,
                label='neutral'
            )
        
        # Get individual scores
        vader_score = self._get_vader_score(text)
        textblob_score = self._get_textblob_score(text)
        finbert_score, finbert_conf = self._get_finbert_score(text)
        
        # Calculate ensemble score
        ensemble_score = (
            self.vader_weight * vader_score +
            self.textblob_weight * textblob_score +
            self.finbert_weight * finbert_score
        )
        
        # Calculate confidence (weighted average of individual confidences)
        confidence = self._calculate_confidence(
            vader_score, textblob_score, finbert_score, finbert_conf
        )
        
        # Determine label
        if ensemble_score > 0.15:
            label = 'bullish'
        elif ensemble_score < -0.15:
            label = 'bearish'
        else:
            label = 'neutral'
        
        return SentimentScore(
            vader_score=vader_score,
            textblob_score=textblob_score,
            finbert_score=finbert_score,
            ensemble_score=ensemble_score,
            confidence=confidence,
            label=label
        )
    
    def _get_vader_score(self, text: str) -> float:
        """Get VADER compound score normalized to [-1, 1]."""
        if not self.vader_analyzer:
            return 0.0
        
        try:
            scores = self.vader_analyzer.polarity_scores(text)
            return scores['compound']  # Already in [-1, 1]
        except Exception as e:
            logger.warning(f"VADER error: {e}")
            return 0.0
    
    def _get_textblob_score(self, text: str) -> float:
        """Get TextBlob polarity score."""
        if not TEXTBLOB_AVAILABLE:
            return 0.0
        
        try:
            blob = TextBlob(text)
            return blob.sentiment.polarity  # Already in [-1, 1]
        except Exception as e:
            logger.warning(f"TextBlob error: {e}")
            return 0.0
    
    def _get_finbert_score(self, text: str) -> Tuple[float, float]:
        """
        Get FinBERT sentiment score.
        
        Returns:
            Tuple of (score, confidence)
        """
        if not self.finbert_model:
            return 0.0, 0.0
        
        try:
            # Tokenize
            inputs = self.finbert_tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_sequence_length,
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get prediction
            with torch.no_grad():
                outputs = self.finbert_model(**inputs)
                probs = torch.softmax(outputs.logits, dim=1)
            
            # FinBERT outputs: [positive, negative, neutral]
            # Convert to score in [-1, 1]
            probs = probs.cpu().numpy()[0]
            
            # Score = positive - negative
            score = probs[0] - probs[1]
            
            # Confidence = max probability
            confidence = float(probs.max())
            
            return float(score), confidence
            
        except Exception as e:
            logger.warning(f"FinBERT error: {e}")
            return 0.0, 0.0
    
    def _calculate_confidence(
        self,
        vader_score: float,
        textblob_score: float,
        finbert_score: float,
        finbert_conf: float
    ) -> float:
        """Calculate overall confidence based on model agreement."""
        scores = [vader_score, textblob_score, finbert_score]
        scores = [s for s in scores if s != 0.0]  # Filter out unavailable
        
        if not scores:
            return 0.0
        
        # Agreement: all scores have same sign
        signs = [np.sign(s) for s in scores]
        agreement = len(set(signs)) == 1
        
        # Magnitude: average absolute score
        magnitude = np.mean([abs(s) for s in scores])
        
        # Combine agreement and magnitude
        if agreement:
            confidence = min(magnitude + 0.3, 1.0)
        else:
            confidence = magnitude * 0.5
        
        return confidence
    
    def analyze_articles(
        self,
        articles: List[Dict[str, Any]],
        text_field: str = 'content',
        title_field: str = 'title',
        title_weight: float = 0.3,
    ) -> Dict[str, Any]:
        """
        Analyze sentiment of multiple articles.
        
        Args:
            articles: List of article dictionaries
            text_field: Field containing article text
            title_field: Field containing article title
            title_weight: Weight for title vs content
            
        Returns:
            Dictionary with aggregate sentiment metrics
        """
        if not articles:
            return self._empty_aggregate()
        
        scores = []
        for article in articles:
            title = article.get(title_field, '')
            content = article.get(text_field, '')
            
            # Analyze title and content separately
            title_score = self.analyze_text(title) if title else None
            content_score = self.analyze_text(content) if content else None
            
            # Combine title and content scores
            if title_score and content_score:
                combined_score = (
                    title_weight * title_score.ensemble_score +
                    (1 - title_weight) * content_score.ensemble_score
                )
                confidence = (
                    title_weight * title_score.confidence +
                    (1 - title_weight) * content_score.confidence
                )
            elif title_score:
                combined_score = title_score.ensemble_score
                confidence = title_score.confidence
            elif content_score:
                combined_score = content_score.ensemble_score
                confidence = content_score.confidence
            else:
                continue
            
            scores.append({
                'score': combined_score,
                'confidence': confidence,
                'timestamp': article.get('published_at', datetime.now())
            })
        
        return self._aggregate_scores(scores)
    
    def _aggregate_scores(self, scores: List[Dict]) -> Dict[str, Any]:
        """Aggregate individual article scores."""
        if not scores:
            return self._empty_aggregate()
        
        sentiment_values = [s['score'] for s in scores]
        confidence_values = [s['confidence'] for s in scores]
        
        # Basic statistics
        mean_sentiment = np.mean(sentiment_values)
        std_sentiment = np.std(sentiment_values)
        median_sentiment = np.median(sentiment_values)
        
        # Weighted by confidence
        if sum(confidence_values) > 0:
            weighted_sentiment = np.average(sentiment_values, weights=confidence_values)
        else:
            weighted_sentiment = mean_sentiment
        
        # Sentiment distribution
        bullish_count = sum(1 for s in sentiment_values if s > 0.15)
        bearish_count = sum(1 for s in sentiment_values if s < -0.15)
        neutral_count = len(sentiment_values) - bullish_count - bearish_count
        
        bullish_ratio = bullish_count / len(sentiment_values)
        bearish_ratio = bearish_count / len(sentiment_values)
        
        # Sentiment momentum (if timestamps available)
        sentiment_momentum = self._calculate_sentiment_momentum(scores)
        
        return {
            'mean_sentiment': mean_sentiment,
            'weighted_sentiment': weighted_sentiment,
            'median_sentiment': median_sentiment,
            'std_sentiment': std_sentiment,
            'min_sentiment': min(sentiment_values),
            'max_sentiment': max(sentiment_values),
            'sentiment_range': max(sentiment_values) - min(sentiment_values),
            'bullish_count': bullish_count,
            'bearish_count': bearish_count,
            'neutral_count': neutral_count,
            'bullish_ratio': bullish_ratio,
            'bearish_ratio': bearish_ratio,
            'sentiment_momentum': sentiment_momentum,
            'mean_confidence': np.mean(confidence_values),
            'article_count': len(scores),
        }
    
    def _calculate_sentiment_momentum(self, scores: List[Dict]) -> float:
        """Calculate sentiment trend over time."""
        if len(scores) < 2:
            return 0.0
        
        # Sort by timestamp
        sorted_scores = sorted(scores, key=lambda x: x.get('timestamp', datetime.now()))
        
        # Simple: compare recent vs older sentiment
        mid = len(sorted_scores) // 2
        older_sentiment = np.mean([s['score'] for s in sorted_scores[:mid]])
        recent_sentiment = np.mean([s['score'] for s in sorted_scores[mid:]])
        
        return recent_sentiment - older_sentiment
    
    def _empty_aggregate(self) -> Dict[str, Any]:
        """Return empty aggregate when no articles available."""
        return {
            'mean_sentiment': 0.0,
            'weighted_sentiment': 0.0,
            'median_sentiment': 0.0,
            'std_sentiment': 0.0,
            'min_sentiment': 0.0,
            'max_sentiment': 0.0,
            'sentiment_range': 0.0,
            'bullish_count': 0,
            'bearish_count': 0,
            'neutral_count': 0,
            'bullish_ratio': 0.0,
            'bearish_ratio': 0.0,
            'sentiment_momentum': 0.0,
            'mean_confidence': 0.0,
            'article_count': 0,
        }
    
    def add_sentiment_features(
        self,
        df: pd.DataFrame,
        news_data: Dict[str, List[Dict]],
        date_column: str = 'date',
    ) -> pd.DataFrame:
        """
        Add sentiment features to OHLCV DataFrame.
        
        Args:
            df: DataFrame with OHLCV data
            news_data: Dictionary mapping dates to list of articles
            date_column: Column containing dates
            
        Returns:
            DataFrame with sentiment features added
        """
        df = df.copy()
        
        # Initialize sentiment columns
        sentiment_columns = [
            'sentiment_mean', 'sentiment_weighted', 'sentiment_std',
            'sentiment_bullish_ratio', 'sentiment_bearish_ratio',
            'sentiment_momentum', 'sentiment_confidence', 'news_count'
        ]
        
        for col in sentiment_columns:
            df[col] = 0.0
        
        # Process each date
        for idx, row in df.iterrows():
            date = row[date_column] if date_column in df.columns else idx
            
            # Get articles for this date
            if isinstance(date, datetime):
                date_str = date.strftime('%Y-%m-%d')
            else:
                date_str = str(date)
            
            articles = news_data.get(date_str, [])
            
            if articles:
                sentiment = self.analyze_articles(articles)
                
                df.at[idx, 'sentiment_mean'] = sentiment['mean_sentiment']
                df.at[idx, 'sentiment_weighted'] = sentiment['weighted_sentiment']
                df.at[idx, 'sentiment_std'] = sentiment['std_sentiment']
                df.at[idx, 'sentiment_bullish_ratio'] = sentiment['bullish_ratio']
                df.at[idx, 'sentiment_bearish_ratio'] = sentiment['bearish_ratio']
                df.at[idx, 'sentiment_momentum'] = sentiment['sentiment_momentum']
                df.at[idx, 'sentiment_confidence'] = sentiment['mean_confidence']
                df.at[idx, 'news_count'] = sentiment['article_count']
        
        # Add derived sentiment features
        df['sentiment_ma_5'] = df['sentiment_mean'].rolling(5).mean()
        df['sentiment_ma_10'] = df['sentiment_mean'].rolling(10).mean()
        df['sentiment_change'] = df['sentiment_mean'].diff()
        df['sentiment_zscore'] = (
            (df['sentiment_mean'] - df['sentiment_mean'].rolling(20).mean()) /
            (df['sentiment_mean'].rolling(20).std() + 1e-8)
        )
        
        # Sentiment-price divergence
        if 'close' in df.columns:
            returns = df['close'].pct_change()
            df['sentiment_price_divergence'] = (
                np.sign(df['sentiment_mean']) != np.sign(returns)
            ).astype(int)
        
        return df
    
    def get_feature_names(self) -> List[str]:
        """Get list of all sentiment feature names."""
        return [
            'sentiment_mean', 'sentiment_weighted', 'sentiment_std',
            'sentiment_bullish_ratio', 'sentiment_bearish_ratio',
            'sentiment_momentum', 'sentiment_confidence', 'news_count',
            'sentiment_ma_5', 'sentiment_ma_10', 'sentiment_change',
            'sentiment_zscore', 'sentiment_price_divergence'
        ]
