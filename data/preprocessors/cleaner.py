"""
Data Cleaner Module.

This module handles data preprocessing including:
- Missing value imputation
- Outlier detection and handling
- Data normalization
- Feature scaling
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Union, Tuple
from scipy import stats
import logging

logger = logging.getLogger(__name__)


class DataCleaner:
    """
    Comprehensive data cleaning and preprocessing pipeline.
    
    Handles:
    - Missing values (forward fill, interpolation, etc.)
    - Outliers (z-score, IQR methods)
    - Duplicate removal
    - Data type conversion
    - Index alignment
    
    Example:
        >>> cleaner = DataCleaner()
        >>> clean_data = cleaner.clean(raw_data)
    """
    
    def __init__(
        self,
        fillna_method: str = 'ffill',
        outlier_method: str = 'zscore',
        outlier_threshold: float = 5.0,
        remove_duplicates: bool = True,
    ):
        """
        Initialize the data cleaner.
        
        Args:
            fillna_method: Method for filling missing values
                Options: 'ffill', 'bfill', 'interpolate', 'mean', 'median', 'zero'
            outlier_method: Method for detecting outliers
                Options: 'zscore', 'iqr', 'mad', 'none'
            outlier_threshold: Threshold for outlier detection
            remove_duplicates: Whether to remove duplicate rows
        """
        self.fillna_method = fillna_method
        self.outlier_method = outlier_method
        self.outlier_threshold = outlier_threshold
        self.remove_duplicates = remove_duplicates
    
    def clean(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Apply full cleaning pipeline to DataFrame.
        
        Args:
            df: Input DataFrame
            columns: Specific columns to clean (None = all)
            
        Returns:
            Cleaned DataFrame
        """
        df = df.copy()
        
        # Select columns
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        logger.info(f"Cleaning {len(columns)} columns...")
        
        # Step 1: Remove duplicates
        if self.remove_duplicates:
            original_len = len(df)
            df = df[~df.index.duplicated(keep='first')]
            removed = original_len - len(df)
            if removed > 0:
                logger.info(f"Removed {removed} duplicate rows")
        
        # Step 2: Handle missing values
        df = self._handle_missing(df, columns)
        
        # Step 3: Handle outliers
        df = self._handle_outliers(df, columns)
        
        # Step 4: Ensure sorted index
        if not df.index.is_monotonic_increasing:
            df = df.sort_index()
        
        return df
    
    def _handle_missing(
        self,
        df: pd.DataFrame,
        columns: List[str],
    ) -> pd.DataFrame:
        """Handle missing values in specified columns."""
        for col in columns:
            if col not in df.columns:
                continue
                
            missing_count = df[col].isna().sum()
            
            if missing_count == 0:
                continue
            
            missing_pct = missing_count / len(df) * 100
            logger.debug(f"Column '{col}': {missing_count} missing ({missing_pct:.2f}%)")
            
            if self.fillna_method == 'ffill':
                df[col] = df[col].fillna(method='ffill')
                # Backfill any remaining NaN at the start
                df[col] = df[col].fillna(method='bfill')
                
            elif self.fillna_method == 'bfill':
                df[col] = df[col].fillna(method='bfill')
                df[col] = df[col].fillna(method='ffill')
                
            elif self.fillna_method == 'interpolate':
                df[col] = df[col].interpolate(method='linear')
                df[col] = df[col].fillna(method='bfill')
                df[col] = df[col].fillna(method='ffill')
                
            elif self.fillna_method == 'mean':
                df[col] = df[col].fillna(df[col].mean())
                
            elif self.fillna_method == 'median':
                df[col] = df[col].fillna(df[col].median())
                
            elif self.fillna_method == 'zero':
                df[col] = df[col].fillna(0)
        
        return df
    
    def _handle_outliers(
        self,
        df: pd.DataFrame,
        columns: List[str],
    ) -> pd.DataFrame:
        """Handle outliers in specified columns."""
        if self.outlier_method == 'none':
            return df
        
        for col in columns:
            if col not in df.columns:
                continue
            
            # Detect outliers
            if self.outlier_method == 'zscore':
                outlier_mask = self._zscore_outliers(df[col])
            elif self.outlier_method == 'iqr':
                outlier_mask = self._iqr_outliers(df[col])
            elif self.outlier_method == 'mad':
                outlier_mask = self._mad_outliers(df[col])
            else:
                continue
            
            outlier_count = outlier_mask.sum()
            
            if outlier_count > 0:
                outlier_pct = outlier_count / len(df) * 100
                logger.debug(f"Column '{col}': {outlier_count} outliers ({outlier_pct:.2f}%)")
                
                # Cap outliers at threshold instead of removing
                df.loc[outlier_mask, col] = np.nan
                df[col] = df[col].fillna(method='ffill')
        
        return df
    
    def _zscore_outliers(self, series: pd.Series) -> pd.Series:
        """Detect outliers using z-score method."""
        z_scores = np.abs(stats.zscore(series.dropna()))
        outlier_mask = pd.Series(False, index=series.index)
        outlier_mask.loc[series.dropna().index] = z_scores > self.outlier_threshold
        return outlier_mask
    
    def _iqr_outliers(self, series: pd.Series) -> pd.Series:
        """Detect outliers using IQR method."""
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - self.outlier_threshold * IQR
        upper_bound = Q3 + self.outlier_threshold * IQR
        
        return (series < lower_bound) | (series > upper_bound)
    
    def _mad_outliers(self, series: pd.Series) -> pd.Series:
        """Detect outliers using Median Absolute Deviation."""
        median = series.median()
        mad = np.abs(series - median).median()
        
        if mad == 0:
            return pd.Series(False, index=series.index)
        
        modified_z_scores = 0.6745 * (series - median) / mad
        return np.abs(modified_z_scores) > self.outlier_threshold
    
    def clean_ohlcv(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean OHLCV data with specific rules for price data.
        
        Ensures:
        - High >= Low
        - Open and Close within High-Low range
        - Volume >= 0
        - No zero prices
        """
        df = df.copy()
        
        # Ensure required columns exist
        required = ['open', 'high', 'low', 'close', 'volume']
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Handle missing values
        df = self._handle_missing(df, required)
        
        # Fix high/low inversions
        inverted = df['high'] < df['low']
        if inverted.any():
            logger.warning(f"Found {inverted.sum()} rows with high < low, fixing...")
            df.loc[inverted, ['high', 'low']] = df.loc[inverted, ['low', 'high']].values
        
        # Ensure open is within range
        df['open'] = df['open'].clip(lower=df['low'], upper=df['high'])
        
        # Ensure close is within range
        df['close'] = df['close'].clip(lower=df['low'], upper=df['high'])
        
        # Replace zero prices with NaN and forward fill
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            zero_mask = df[col] <= 0
            if zero_mask.any():
                logger.warning(f"Found {zero_mask.sum()} zero/negative prices in {col}")
                df.loc[zero_mask, col] = np.nan
                df[col] = df[col].fillna(method='ffill')
        
        # Ensure volume is non-negative
        df['volume'] = df['volume'].clip(lower=0)
        
        # Remove extreme price outliers
        df = self._handle_outliers(df, price_cols)
        
        return df
    
    def calculate_returns(
        self,
        df: pd.DataFrame,
        price_col: str = 'close',
        method: str = 'log',
    ) -> pd.Series:
        """
        Calculate returns from price series.
        
        Args:
            df: DataFrame with price column
            price_col: Name of price column
            method: 'log' for log returns, 'simple' for simple returns
            
        Returns:
            Series of returns
        """
        prices = df[price_col]
        
        if method == 'log':
            returns = np.log(prices / prices.shift(1))
        else:
            returns = prices.pct_change()
        
        return returns
    
    def align_dataframes(
        self,
        dfs: List[pd.DataFrame],
        method: str = 'inner',
    ) -> List[pd.DataFrame]:
        """
        Align multiple DataFrames to common index.
        
        Args:
            dfs: List of DataFrames
            method: 'inner' for intersection, 'outer' for union
            
        Returns:
            List of aligned DataFrames
        """
        if not dfs:
            return dfs
        
        # Get common index
        if method == 'inner':
            common_index = dfs[0].index
            for df in dfs[1:]:
                common_index = common_index.intersection(df.index)
        else:
            common_index = dfs[0].index
            for df in dfs[1:]:
                common_index = common_index.union(df.index)
        
        # Reindex all DataFrames
        aligned = []
        for df in dfs:
            aligned.append(df.reindex(common_index))
        
        return aligned


def clean_market_data(
    df: pd.DataFrame,
    fillna_method: str = 'ffill',
) -> pd.DataFrame:
    """Convenience function to clean market data."""
    cleaner = DataCleaner(fillna_method=fillna_method)
    return cleaner.clean_ohlcv(df)
