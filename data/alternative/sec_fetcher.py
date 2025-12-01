"""
SEC EDGAR Filings Fetcher Module.

Fetches and analyzes SEC filings for institutional insights:
- 10-K (Annual reports)
- 10-Q (Quarterly reports)
- 8-K (Current events)
- 13F (Institutional holdings)
- 4 (Insider transactions)
- SC 13D/13G (Activist/institutional ownership)

Uses SEC EDGAR API and RSS feeds.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging
import time
import re
import xml.etree.ElementTree as ET

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class SECFiling:
    """Container for SEC filing data."""
    form_type: str
    filed_date: datetime
    accepted_date: datetime
    accession_number: str
    file_url: str
    description: str
    company_name: str
    cik: str
    items: List[str]  # For 8-K items
    sentiment_score: float = 0.0


@dataclass
class InsiderTransaction:
    """Container for insider transaction data."""
    filer_name: str
    filer_relation: str  # CEO, CFO, Director, etc.
    transaction_type: str  # Buy, Sell, Option Exercise
    transaction_date: datetime
    shares: int
    price_per_share: float
    total_value: float
    shares_owned_after: int
    form_url: str


@dataclass 
class InstitutionalHolding:
    """Container for 13F institutional holding data."""
    institution_name: str
    cik: str
    shares: int
    value: float
    quarter: str
    change_shares: int
    change_pct: float
    portfolio_pct: float


class SECFetcher:
    """
    Fetches and analyzes SEC EDGAR filings.
    
    Provides institutional-grade insights from regulatory filings
    including sentiment analysis and trend detection.
    
    Example:
        >>> fetcher = SECFetcher()
        >>> filings = fetcher.get_recent_filings('AAPL')
        >>> insider = fetcher.get_insider_transactions('AAPL')
    """
    
    # SEC EDGAR base URLs
    EDGAR_BASE = "https://www.sec.gov"
    EDGAR_SEARCH = "https://efts.sec.gov/LATEST/search-index"
    EDGAR_COMPANY = "https://data.sec.gov/submissions"
    EDGAR_RSS = "https://www.sec.gov/cgi-bin/browse-edgar"
    
    # SEC requires user agent identification
    USER_AGENT = "TradingEngine/1.0 (contact@example.com)"
    
    # 8-K Item codes and descriptions
    FORM_8K_ITEMS = {
        '1.01': 'Entry into Material Agreement',
        '1.02': 'Termination of Material Agreement',
        '1.03': 'Bankruptcy or Receivership',
        '2.01': 'Completion of Acquisition/Disposition',
        '2.02': 'Results of Operations (Earnings)',
        '2.03': 'Creation of Direct Obligation',
        '2.04': 'Triggering Events for Acceleration',
        '2.05': 'Exit Activities/Material Impairments',
        '2.06': 'Material Impairments',
        '3.01': 'Delisting/Transfer',
        '3.02': 'Unregistered Sales of Equity',
        '3.03': 'Material Modification to Shareholder Rights',
        '4.01': 'Changes in Accountant',
        '4.02': 'Non-Reliance on Financial Statements',
        '5.01': 'Changes in Control',
        '5.02': 'Departure/Election of Directors/Officers',
        '5.03': 'Amendments to Articles/Bylaws',
        '5.05': 'Amendments to Code of Ethics',
        '5.07': 'Shareholder Vote Results',
        '5.08': 'Shareholder Director Nominations',
        '7.01': 'Regulation FD Disclosure',
        '8.01': 'Other Events',
        '9.01': 'Financial Statements and Exhibits',
    }
    
    # Sentiment weights for different filing types
    FILING_SENTIMENT_WEIGHTS = {
        '10-K': 1.0,
        '10-Q': 0.8,
        '8-K': 0.9,
        '4': 0.7,
        '13F': 0.6,
        'SC 13D': 0.9,
        'SC 13G': 0.6,
    }
    
    def __init__(self, user_agent: Optional[str] = None):
        """
        Initialize SEC fetcher.
        
        Args:
            user_agent: Custom user agent string (SEC requires identification)
        """
        self.user_agent = user_agent or self.USER_AGENT
        self.headers = {'User-Agent': self.user_agent}
        
        if VADER_AVAILABLE:
            self.sentiment_analyzer = SentimentIntensityAnalyzer()
        else:
            self.sentiment_analyzer = None
            
        # Cache for CIK lookups
        self._cik_cache: Dict[str, str] = {}
        
        logger.info("SEC EDGAR fetcher initialized")
    
    def get_cik(self, symbol: str) -> Optional[str]:
        """
        Get CIK (Central Index Key) for a stock symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            CIK number as string, or None if not found
        """
        if symbol in self._cik_cache:
            return self._cik_cache[symbol]
        
        try:
            # SEC company tickers JSON
            url = "https://www.sec.gov/files/company_tickers.json"
            response = requests.get(url, headers=self.headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                for entry in data.values():
                    if entry.get('ticker', '').upper() == symbol.upper():
                        cik = str(entry.get('cik_str', '')).zfill(10)
                        self._cik_cache[symbol] = cik
                        return cik
                        
        except Exception as e:
            logger.warning(f"Error getting CIK for {symbol}: {e}")
        
        return None
    
    def get_company_filings(
        self,
        symbol: str,
        form_types: Optional[List[str]] = None,
        limit: int = 20,
    ) -> List[SECFiling]:
        """
        Get recent SEC filings for a company.
        
        Args:
            symbol: Stock symbol
            form_types: List of form types to fetch (e.g., ['10-K', '10-Q', '8-K'])
            limit: Maximum number of filings to return
            
        Returns:
            List of SECFiling objects
        """
        cik = self.get_cik(symbol)
        if not cik:
            logger.warning(f"Could not find CIK for {symbol}")
            return []
        
        form_types = form_types or ['10-K', '10-Q', '8-K', '4']
        filings = []
        
        try:
            # SEC submissions endpoint
            url = f"{self.EDGAR_COMPANY}/CIK{cik}.json"
            response = requests.get(url, headers=self.headers, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                company_name = data.get('name', '')
                
                recent = data.get('filings', {}).get('recent', {})
                
                forms = recent.get('form', [])
                filed_dates = recent.get('filingDate', [])
                accessions = recent.get('accessionNumber', [])
                primary_docs = recent.get('primaryDocument', [])
                
                count = 0
                for i, form in enumerate(forms):
                    if count >= limit:
                        break
                    
                    if form in form_types:
                        accession = accessions[i].replace('-', '')
                        
                        filing = SECFiling(
                            form_type=form,
                            filed_date=datetime.strptime(filed_dates[i], '%Y-%m-%d'),
                            accepted_date=datetime.strptime(filed_dates[i], '%Y-%m-%d'),
                            accession_number=accessions[i],
                            file_url=f"{self.EDGAR_BASE}/Archives/edgar/data/{cik.lstrip('0')}/{accession}/{primary_docs[i]}",
                            description=f"{form} filing",
                            company_name=company_name,
                            cik=cik,
                            items=[],
                        )
                        filings.append(filing)
                        count += 1
                        
        except Exception as e:
            logger.warning(f"Error fetching filings for {symbol}: {e}")
        
        return filings
    
    def get_8k_items(self, filing: SECFiling) -> List[str]:
        """
        Extract item codes from an 8-K filing.
        
        8-K items indicate the nature of the material event.
        """
        if filing.form_type != '8-K':
            return []
        
        items = []
        
        try:
            # Fetch the index page
            idx_url = filing.file_url.replace('.htm', '-index.htm')
            response = requests.get(idx_url, headers=self.headers, timeout=10)
            
            if response.status_code == 200 and BS4_AVAILABLE:
                soup = BeautifulSoup(response.text, 'html.parser')
                text = soup.get_text()
                
                # Find item codes
                for item_code in self.FORM_8K_ITEMS.keys():
                    if f"Item {item_code}" in text or f"ITEM {item_code}" in text:
                        items.append(item_code)
                        
        except Exception as e:
            logger.debug(f"Error extracting 8-K items: {e}")
        
        return items
    
    def get_insider_transactions(
        self,
        symbol: str,
        days_back: int = 90,
        limit: int = 50,
    ) -> List[InsiderTransaction]:
        """
        Get insider transactions (Form 4 filings).
        
        Args:
            symbol: Stock symbol
            days_back: How many days of transactions to fetch
            limit: Maximum number of transactions
            
        Returns:
            List of InsiderTransaction objects
        """
        cik = self.get_cik(symbol)
        if not cik:
            return []
        
        transactions = []
        
        # Fetch Form 4 filings
        filings = self.get_company_filings(symbol, form_types=['4'], limit=limit)
        
        for filing in filings:
            if filing.filed_date < datetime.now() - timedelta(days=days_back):
                continue
            
            try:
                # Parse Form 4 XML
                xml_url = filing.file_url.replace('.htm', '.xml')
                response = requests.get(xml_url, headers=self.headers, timeout=10)
                
                if response.status_code == 200:
                    txns = self._parse_form4_xml(response.text, filing.file_url)
                    transactions.extend(txns)
                    
            except Exception as e:
                logger.debug(f"Error parsing Form 4: {e}")
        
        return transactions
    
    def _parse_form4_xml(
        self,
        xml_content: str,
        form_url: str,
    ) -> List[InsiderTransaction]:
        """Parse Form 4 XML to extract transaction details."""
        transactions = []
        
        try:
            root = ET.fromstring(xml_content)
            
            # Get filer info
            filer_name = ""
            filer_relation = ""
            
            owner = root.find('.//reportingOwner')
            if owner is not None:
                name_elem = owner.find('.//rptOwnerName')
                filer_name = name_elem.text if name_elem is not None else ""
                
                # Get relationship
                rel_elem = owner.find('.//reportingOwnerRelationship')
                if rel_elem is not None:
                    if rel_elem.find('isDirector') is not None:
                        if rel_elem.find('isDirector').text == '1':
                            filer_relation = 'Director'
                    if rel_elem.find('isOfficer') is not None:
                        if rel_elem.find('isOfficer').text == '1':
                            title = rel_elem.find('officerTitle')
                            filer_relation = title.text if title is not None else 'Officer'
            
            # Get non-derivative transactions
            for txn in root.findall('.//nonDerivativeTransaction'):
                txn_data = self._extract_transaction(txn)
                if txn_data:
                    transactions.append(InsiderTransaction(
                        filer_name=filer_name,
                        filer_relation=filer_relation,
                        transaction_type=txn_data['type'],
                        transaction_date=txn_data['date'],
                        shares=txn_data['shares'],
                        price_per_share=txn_data['price'],
                        total_value=txn_data['shares'] * txn_data['price'],
                        shares_owned_after=txn_data['shares_after'],
                        form_url=form_url,
                    ))
                    
        except Exception as e:
            logger.debug(f"Error parsing Form 4 XML: {e}")
        
        return transactions
    
    def _extract_transaction(self, txn_elem) -> Optional[Dict]:
        """Extract transaction details from XML element."""
        try:
            # Get transaction date
            date_elem = txn_elem.find('.//transactionDate/value')
            if date_elem is None:
                return None
            txn_date = datetime.strptime(date_elem.text, '%Y-%m-%d')
            
            # Get transaction type
            code_elem = txn_elem.find('.//transactionCoding/transactionCode')
            txn_code = code_elem.text if code_elem is not None else ''
            
            # Map transaction codes
            txn_type_map = {
                'P': 'Buy',
                'S': 'Sell',
                'A': 'Grant',
                'D': 'Disposition',
                'F': 'Tax Payment',
                'M': 'Option Exercise',
                'C': 'Conversion',
                'G': 'Gift',
            }
            txn_type = txn_type_map.get(txn_code, txn_code)
            
            # Get shares
            shares_elem = txn_elem.find('.//transactionAmounts/transactionShares/value')
            shares = int(float(shares_elem.text)) if shares_elem is not None else 0
            
            # Get price
            price_elem = txn_elem.find('.//transactionAmounts/transactionPricePerShare/value')
            price = float(price_elem.text) if price_elem is not None else 0.0
            
            # Get shares owned after
            after_elem = txn_elem.find('.//postTransactionAmounts/sharesOwnedFollowingTransaction/value')
            shares_after = int(float(after_elem.text)) if after_elem is not None else 0
            
            return {
                'date': txn_date,
                'type': txn_type,
                'shares': shares,
                'price': price,
                'shares_after': shares_after,
            }
            
        except Exception:
            return None
    
    def analyze_insider_activity(
        self,
        symbol: str,
        days_back: int = 90,
    ) -> Dict[str, Any]:
        """
        Analyze insider trading patterns.
        
        Returns:
            Dictionary with insider activity metrics
        """
        transactions = self.get_insider_transactions(symbol, days_back)
        
        if not transactions:
            return {
                'symbol': symbol,
                'total_transactions': 0,
                'net_shares': 0,
                'net_value': 0,
                'buy_count': 0,
                'sell_count': 0,
                'insider_sentiment': 0.0,
            }
        
        buys = [t for t in transactions if t.transaction_type == 'Buy']
        sells = [t for t in transactions if t.transaction_type == 'Sell']
        
        buy_shares = sum(t.shares for t in buys)
        sell_shares = sum(t.shares for t in sells)
        buy_value = sum(t.total_value for t in buys)
        sell_value = sum(t.total_value for t in sells)
        
        net_shares = buy_shares - sell_shares
        net_value = buy_value - sell_value
        
        # Calculate insider sentiment (-1 to 1)
        total_value = buy_value + sell_value
        if total_value > 0:
            insider_sentiment = (buy_value - sell_value) / total_value
        else:
            insider_sentiment = 0.0
        
        return {
            'symbol': symbol,
            'total_transactions': len(transactions),
            'net_shares': net_shares,
            'net_value': net_value,
            'buy_count': len(buys),
            'sell_count': len(sells),
            'buy_shares': buy_shares,
            'sell_shares': sell_shares,
            'buy_value': buy_value,
            'sell_value': sell_value,
            'insider_sentiment': insider_sentiment,
            'recent_transactions': transactions[:10],
            # C-suite specific
            'executive_buys': len([t for t in buys if 'CEO' in t.filer_relation or 'CFO' in t.filer_relation]),
            'executive_sells': len([t for t in sells if 'CEO' in t.filer_relation or 'CFO' in t.filer_relation]),
        }
    
    def analyze_filing_sentiment(
        self,
        filing: SECFiling,
        max_chars: int = 50000,
    ) -> float:
        """
        Analyze sentiment of a filing's text content.
        
        Args:
            filing: SECFiling object
            max_chars: Maximum characters to analyze
            
        Returns:
            Sentiment score (-1 to 1)
        """
        if not self.sentiment_analyzer:
            return 0.0
        
        try:
            response = requests.get(filing.file_url, headers=self.headers, timeout=30)
            
            if response.status_code == 200:
                if BS4_AVAILABLE:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    text = soup.get_text()[:max_chars]
                else:
                    # Basic HTML stripping
                    text = re.sub(r'<[^>]+>', '', response.text)[:max_chars]
                
                # Analyze in chunks (VADER works better on sentences)
                chunks = [text[i:i+5000] for i in range(0, len(text), 5000)]
                sentiments = []
                
                for chunk in chunks:
                    score = self.sentiment_analyzer.polarity_scores(chunk)
                    sentiments.append(score['compound'])
                
                return np.mean(sentiments)
                
        except Exception as e:
            logger.debug(f"Error analyzing filing sentiment: {e}")
        
        return 0.0
    
    def get_sec_features(
        self,
        df: pd.DataFrame,
        symbol: str,
    ) -> pd.DataFrame:
        """
        Add SEC-based features to a dataframe.
        
        Args:
            df: Price dataframe with datetime index
            symbol: Stock symbol
            
        Returns:
            DataFrame with added SEC features
        """
        df = df.copy()
        
        # Get insider activity
        insider = self.analyze_insider_activity(symbol, days_back=90)
        
        # Add insider features
        df['insider_sentiment'] = insider['insider_sentiment']
        df['insider_buy_count'] = insider['buy_count']
        df['insider_sell_count'] = insider['sell_count']
        df['insider_net_value'] = insider['net_value']
        df['executive_activity'] = insider['executive_buys'] - insider['executive_sells']
        
        # Normalize insider signal
        df['insider_signal'] = np.clip(insider['insider_sentiment'], -1, 1)
        
        # Get recent filings
        filings = self.get_company_filings(symbol, form_types=['8-K'], limit=5)
        
        # Check for significant 8-K events
        significant_events = 0
        earnings_filed = 0
        
        recent_date = datetime.now() - timedelta(days=30)
        
        for filing in filings:
            if filing.filed_date >= recent_date:
                items = self.get_8k_items(filing)
                
                # Count significant events
                significant_items = {'1.01', '1.02', '2.01', '2.05', '3.01', '5.01', '5.02'}
                if any(item in significant_items for item in items):
                    significant_events += 1
                
                if '2.02' in items:
                    earnings_filed += 1
        
        df['recent_8k_events'] = significant_events
        df['recent_earnings_8k'] = earnings_filed
        
        # Combined SEC signal
        df['sec_signal'] = (
            df['insider_signal'] * 0.6 +
            np.clip(df['executive_activity'] * 0.1, -0.2, 0.2) +
            np.clip(-significant_events * 0.05, -0.2, 0.2)  # Negative for many events (uncertainty)
        )
        
        return df


class InstitutionalHoldingsAnalyzer:
    """
    Analyzes 13F institutional holdings filings.
    
    Tracks what hedge funds and institutions are buying/selling.
    """
    
    def __init__(self, user_agent: Optional[str] = None):
        """Initialize analyzer."""
        self.user_agent = user_agent or SECFetcher.USER_AGENT
        self.headers = {'User-Agent': self.user_agent}
    
    def get_institutional_holders(
        self,
        symbol: str,
        quarter: Optional[str] = None,
    ) -> List[InstitutionalHolding]:
        """
        Get institutional holders for a stock.
        
        Note: This would typically require a data provider like 
        Whale Wisdom or SEC EDGAR parsing of 13F filings.
        
        Args:
            symbol: Stock symbol
            quarter: Quarter in format 'YYYY-Q#' (e.g., '2024-Q1')
            
        Returns:
            List of institutional holdings
        """
        # Note: Full 13F parsing is complex - this is a placeholder
        # Real implementation would parse 13F-HR filings
        logger.info(f"Institutional holdings analysis for {symbol}")
        
        return []
    
    def analyze_institutional_changes(
        self,
        symbol: str,
    ) -> Dict[str, Any]:
        """
        Analyze quarter-over-quarter changes in institutional holdings.
        
        Returns:
            Dictionary with institutional change metrics
        """
        # Placeholder for institutional analysis
        return {
            'symbol': symbol,
            'total_institutional_holders': 0,
            'institutional_ownership_pct': 0.0,
            'quarter_change_pct': 0.0,
            'new_positions': 0,
            'closed_positions': 0,
            'increased_positions': 0,
            'decreased_positions': 0,
        }
