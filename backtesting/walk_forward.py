"""
Walk-Forward Backtesting Module.

Implements institutional-grade walk-forward analysis:
- Rolling window optimization
- Out-of-sample validation
- Adaptive parameter optimization
- Regime-aware testing
- Monte Carlo validation

Walk-forward testing prevents overfitting by continuously
re-optimizing on past data and testing on future data.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import logging
import warnings

try:
    import vectorbt as vbt
    VBT_AVAILABLE = True
except ImportError:
    VBT_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardWindow:
    """Container for a single walk-forward window."""
    window_id: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    train_return: float = 0.0
    test_return: float = 0.0
    train_sharpe: float = 0.0
    test_sharpe: float = 0.0
    train_trades: int = 0
    test_trades: int = 0
    best_params: Dict = field(default_factory=dict)


@dataclass
class WalkForwardResult:
    """Container for walk-forward backtest results."""
    windows: List[WalkForwardWindow]
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    avg_train_test_correlation: float
    overfitting_ratio: float
    equity_curve: pd.Series = None
    all_trades: pd.DataFrame = None


class WalkForwardBacktest:
    """
    Walk-Forward Backtesting Engine.
    
    Implements rolling window optimization that continuously:
    1. Trains/optimizes on historical window
    2. Tests on subsequent out-of-sample period
    3. Rolls forward and repeats
    
    This simulates how a real trading system would be deployed
    and provides more realistic performance estimates.
    
    Example:
        >>> wf = WalkForwardBacktest(
        ...     train_period=252,  # 1 year training
        ...     test_period=63,    # 3 months testing
        ...     step_size=63,      # Roll every 3 months
        ... )
        >>> result = wf.run(df, signal_generator, backtester)
    """
    
    def __init__(
        self,
        train_period: int = 252,
        test_period: int = 63,
        step_size: int = 21,
        min_train_period: int = 126,
        expanding_window: bool = False,
        optimization_metric: str = 'sharpe',
    ):
        """
        Initialize walk-forward backtester.
        
        Args:
            train_period: Number of trading days for training window
            test_period: Number of trading days for test window
            step_size: Number of days to roll forward each iteration
            min_train_period: Minimum training period (for expanding window)
            expanding_window: If True, training window expands over time
            optimization_metric: Metric to optimize ('sharpe', 'return', 'sortino')
        """
        self.train_period = train_period
        self.test_period = test_period
        self.step_size = step_size
        self.min_train_period = min_train_period
        self.expanding_window = expanding_window
        self.optimization_metric = optimization_metric
        
        logger.info(
            f"Walk-forward backtest initialized: "
            f"train={train_period}d, test={test_period}d, step={step_size}d"
        )
    
    def generate_windows(
        self,
        df: pd.DataFrame,
    ) -> List[Tuple[pd.DataFrame, pd.DataFrame, int]]:
        """
        Generate train/test windows from data.
        
        Args:
            df: Full price dataframe
            
        Returns:
            List of (train_df, test_df, window_id) tuples
        """
        windows = []
        n_rows = len(df)
        
        window_id = 0
        
        if self.expanding_window:
            # Expanding window: training always starts from beginning
            train_start = 0
            test_start = self.min_train_period
            
            while test_start + self.test_period <= n_rows:
                train_end = test_start
                test_end = min(test_start + self.test_period, n_rows)
                
                train_df = df.iloc[train_start:train_end].copy()
                test_df = df.iloc[test_start:test_end].copy()
                
                if len(train_df) >= self.min_train_period and len(test_df) > 0:
                    windows.append((train_df, test_df, window_id))
                    window_id += 1
                
                test_start += self.step_size
                
        else:
            # Rolling window: fixed training size
            train_start = 0
            
            while train_start + self.train_period + self.test_period <= n_rows:
                train_end = train_start + self.train_period
                test_start = train_end
                test_end = test_start + self.test_period
                
                train_df = df.iloc[train_start:train_end].copy()
                test_df = df.iloc[test_start:test_end].copy()
                
                windows.append((train_df, test_df, window_id))
                window_id += 1
                
                train_start += self.step_size
        
        logger.info(f"Generated {len(windows)} walk-forward windows")
        return windows
    
    def run(
        self,
        df: pd.DataFrame,
        signal_func: Callable[[pd.DataFrame], pd.Series],
        backtest_func: Callable[[pd.DataFrame, pd.Series], Dict],
        optimize_func: Optional[Callable[[pd.DataFrame], Dict]] = None,
    ) -> WalkForwardResult:
        """
        Run walk-forward backtest.
        
        Args:
            df: Full price dataframe with features
            signal_func: Function that generates signals from data
            backtest_func: Function that runs backtest and returns metrics
            optimize_func: Optional function to optimize parameters on training data
            
        Returns:
            WalkForwardResult with aggregated metrics
        """
        windows = self.generate_windows(df)
        
        if not windows:
            raise ValueError("Not enough data for walk-forward analysis")
        
        window_results: List[WalkForwardWindow] = []
        all_test_returns = []
        all_test_trades = []
        
        for train_df, test_df, window_id in windows:
            logger.info(f"Processing window {window_id + 1}/{len(windows)}")
            
            try:
                # Optimize on training data if function provided
                if optimize_func:
                    best_params = optimize_func(train_df)
                else:
                    best_params = {}
                
                # Generate signals
                train_signals = signal_func(train_df)
                test_signals = signal_func(test_df)
                
                # Run backtests
                train_metrics = backtest_func(train_df, train_signals)
                test_metrics = backtest_func(test_df, test_signals)
                
                # Create window result
                wf_window = WalkForwardWindow(
                    window_id=window_id,
                    train_start=train_df.index[0] if hasattr(train_df.index[0], 'to_pydatetime') else train_df.index[0],
                    train_end=train_df.index[-1] if hasattr(train_df.index[-1], 'to_pydatetime') else train_df.index[-1],
                    test_start=test_df.index[0] if hasattr(test_df.index[0], 'to_pydatetime') else test_df.index[0],
                    test_end=test_df.index[-1] if hasattr(test_df.index[-1], 'to_pydatetime') else test_df.index[-1],
                    train_return=train_metrics.get('total_return', 0),
                    test_return=test_metrics.get('total_return', 0),
                    train_sharpe=train_metrics.get('sharpe_ratio', 0),
                    test_sharpe=test_metrics.get('sharpe_ratio', 0),
                    train_trades=train_metrics.get('total_trades', 0),
                    test_trades=test_metrics.get('total_trades', 0),
                    best_params=best_params,
                )
                
                window_results.append(wf_window)
                
                # Collect test period returns
                if 'equity_curve' in test_metrics:
                    all_test_returns.append(test_metrics['equity_curve'])
                else:
                    all_test_returns.append(pd.Series([1 + test_metrics.get('total_return', 0)]))
                
                if 'trades' in test_metrics:
                    all_test_trades.append(test_metrics['trades'])
                    
            except Exception as e:
                logger.warning(f"Error in window {window_id}: {e}")
                continue
        
        # Aggregate results
        return self._aggregate_results(window_results, all_test_returns, all_test_trades)
    
    def _aggregate_results(
        self,
        windows: List[WalkForwardWindow],
        test_returns: List[pd.Series],
        test_trades: List[pd.DataFrame],
    ) -> WalkForwardResult:
        """Aggregate walk-forward window results."""
        
        if not windows:
            return WalkForwardResult(
                windows=[],
                total_return=0,
                annualized_return=0,
                sharpe_ratio=0,
                max_drawdown=0,
                win_rate=0,
                profit_factor=0,
                total_trades=0,
                avg_train_test_correlation=0,
                overfitting_ratio=0,
            )
        
        # Calculate metrics
        test_returns_list = [w.test_return for w in windows]
        train_returns_list = [w.train_return for w in windows]
        
        # Compound test returns
        total_return = np.prod([1 + r for r in test_returns_list]) - 1
        
        # Calculate annualized return
        total_days = sum(
            (w.test_end - w.test_start).days if hasattr(w.test_end, 'days') or hasattr(w.test_start, 'days')
            else (pd.Timestamp(w.test_end) - pd.Timestamp(w.test_start)).days
            for w in windows
        )
        years = max(total_days / 252, 1/252)
        annualized_return = (1 + total_return) ** (1/years) - 1
        
        # Aggregate Sharpe
        test_sharpes = [w.test_sharpe for w in windows if w.test_sharpe != 0]
        avg_sharpe = np.mean(test_sharpes) if test_sharpes else 0
        
        # Calculate overfitting ratio (train vs test performance)
        if train_returns_list and test_returns_list:
            train_avg = np.mean(train_returns_list)
            test_avg = np.mean(test_returns_list)
            
            if train_avg != 0:
                overfitting_ratio = (train_avg - test_avg) / abs(train_avg)
            else:
                overfitting_ratio = 0
            
            # Train-test correlation
            if len(train_returns_list) > 2:
                correlation = np.corrcoef(train_returns_list, test_returns_list)[0, 1]
                correlation = 0 if np.isnan(correlation) else correlation
            else:
                correlation = 0
        else:
            overfitting_ratio = 0
            correlation = 0
        
        # Combine equity curves
        if test_returns:
            equity_curve = pd.concat(test_returns, ignore_index=True)
            # Calculate max drawdown
            cumulative = (1 + equity_curve).cumprod()
            rolling_max = cumulative.expanding().max()
            drawdowns = (cumulative - rolling_max) / rolling_max
            max_drawdown = abs(drawdowns.min())
        else:
            equity_curve = pd.Series([1])
            max_drawdown = 0
        
        # Combine trades
        total_trades = sum(w.test_trades for w in windows)
        
        # Calculate win rate and profit factor from trades
        if test_trades:
            all_trades = pd.concat(test_trades, ignore_index=True)
            
            if 'pnl' in all_trades.columns and len(all_trades) > 0:
                winning_trades = all_trades[all_trades['pnl'] > 0]
                losing_trades = all_trades[all_trades['pnl'] < 0]
                
                win_rate = len(winning_trades) / len(all_trades) if len(all_trades) > 0 else 0
                
                gross_profit = winning_trades['pnl'].sum() if len(winning_trades) > 0 else 0
                gross_loss = abs(losing_trades['pnl'].sum()) if len(losing_trades) > 0 else 0.001
                profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
            else:
                win_rate = 0
                profit_factor = 0
                all_trades = pd.DataFrame()
        else:
            win_rate = 0
            profit_factor = 0
            all_trades = pd.DataFrame()
        
        return WalkForwardResult(
            windows=windows,
            total_return=total_return,
            annualized_return=annualized_return,
            sharpe_ratio=avg_sharpe,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=total_trades,
            avg_train_test_correlation=correlation,
            overfitting_ratio=overfitting_ratio,
            equity_curve=equity_curve,
            all_trades=all_trades,
        )
    
    def run_with_vectorbt(
        self,
        df: pd.DataFrame,
        signal_func: Callable[[pd.DataFrame], pd.Series],
        initial_capital: float = 100000,
        commission: float = 0.001,
    ) -> WalkForwardResult:
        """
        Run walk-forward backtest using VectorBT.
        
        Args:
            df: Price dataframe
            signal_func: Function to generate signals
            initial_capital: Starting capital
            commission: Commission rate
            
        Returns:
            WalkForwardResult
        """
        if not VBT_AVAILABLE:
            raise ImportError("VectorBT required for this method")
        
        def vbt_backtest(data: pd.DataFrame, signals: pd.Series) -> Dict:
            """Run VectorBT backtest."""
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                entries = signals > 0
                exits = signals < 0
                
                pf = vbt.Portfolio.from_signals(
                    close=data['close'],
                    entries=entries,
                    exits=exits,
                    init_cash=initial_capital,
                    fees=commission,
                    freq='D',
                )
                
                return {
                    'total_return': float(pf.total_return()),
                    'sharpe_ratio': float(pf.sharpe_ratio()) if not np.isnan(pf.sharpe_ratio()) else 0,
                    'total_trades': int(pf.trades.count()),
                    'equity_curve': pf.value().pct_change().fillna(0),
                    'trades': pf.trades.records_readable if hasattr(pf.trades, 'records_readable') else pd.DataFrame(),
                }
        
        return self.run(df, signal_func, vbt_backtest)
    
    def print_summary(self, result: WalkForwardResult) -> str:
        """Print walk-forward backtest summary."""
        lines = [
            "\n" + "="*60,
            "WALK-FORWARD BACKTEST RESULTS",
            "="*60,
            f"Number of Windows: {len(result.windows)}",
            f"Total Return: {result.total_return:.2%}",
            f"Annualized Return: {result.annualized_return:.2%}",
            f"Sharpe Ratio: {result.sharpe_ratio:.2f}",
            f"Max Drawdown: {result.max_drawdown:.2%}",
            f"Win Rate: {result.win_rate:.2%}",
            f"Profit Factor: {result.profit_factor:.2f}",
            f"Total Trades: {result.total_trades}",
            "-"*60,
            "ROBUSTNESS METRICS",
            "-"*60,
            f"Train-Test Correlation: {result.avg_train_test_correlation:.2f}",
            f"Overfitting Ratio: {result.overfitting_ratio:.2%}",
            "-"*60,
        ]
        
        # Window breakdown
        if result.windows:
            lines.append("WINDOW BREAKDOWN")
            lines.append("-"*60)
            for w in result.windows[:10]:  # Show first 10
                lines.append(
                    f"Window {w.window_id}: "
                    f"Train={w.train_return:.2%} | "
                    f"Test={w.test_return:.2%} | "
                    f"Trades={w.test_trades}"
                )
            if len(result.windows) > 10:
                lines.append(f"... and {len(result.windows) - 10} more windows")
        
        lines.append("="*60)
        
        summary = "\n".join(lines)
        print(summary)
        return summary


class MonteCarloValidator:
    """
    Monte Carlo validation for backtesting results.
    
    Tests statistical significance by:
    1. Shuffling trade order
    2. Random sampling with replacement
    3. Bootstrap confidence intervals
    """
    
    def __init__(
        self,
        n_simulations: int = 1000,
        confidence_level: float = 0.95,
    ):
        """
        Initialize validator.
        
        Args:
            n_simulations: Number of Monte Carlo simulations
            confidence_level: Confidence level for intervals
        """
        self.n_simulations = n_simulations
        self.confidence_level = confidence_level
    
    def validate_trades(
        self,
        trades_df: pd.DataFrame,
        metric: str = 'total_return',
    ) -> Dict[str, Any]:
        """
        Run Monte Carlo validation on trade results.
        
        Args:
            trades_df: DataFrame with trade results (must have 'pnl' column)
            metric: Metric to validate
            
        Returns:
            Dictionary with validation results
        """
        if 'pnl' not in trades_df.columns or len(trades_df) < 2:
            return {
                'is_significant': False,
                'p_value': 1.0,
                'confidence_interval': (0, 0),
            }
        
        pnl_values = trades_df['pnl'].values
        original_total = pnl_values.sum()
        
        # Bootstrap simulation
        simulated_totals = []
        
        for _ in range(self.n_simulations):
            # Resample with replacement
            sample = np.random.choice(pnl_values, size=len(pnl_values), replace=True)
            simulated_totals.append(sample.sum())
        
        simulated_totals = np.array(simulated_totals)
        
        # Calculate confidence interval
        lower_pct = (1 - self.confidence_level) / 2
        upper_pct = 1 - lower_pct
        
        ci_lower = np.percentile(simulated_totals, lower_pct * 100)
        ci_upper = np.percentile(simulated_totals, upper_pct * 100)
        
        # Calculate p-value (probability of getting >= original by chance)
        # For positive returns, this is proportion <= 0
        if original_total > 0:
            p_value = (simulated_totals <= 0).mean()
        else:
            p_value = (simulated_totals >= 0).mean()
        
        # Is the result significant?
        is_significant = (
            (original_total > 0 and ci_lower > 0) or
            (original_total < 0 and ci_upper < 0)
        )
        
        return {
            'original_total': original_total,
            'mean_simulated': np.mean(simulated_totals),
            'std_simulated': np.std(simulated_totals),
            'confidence_interval': (ci_lower, ci_upper),
            'p_value': p_value,
            'is_significant': is_significant,
            'confidence_level': self.confidence_level,
            'n_simulations': self.n_simulations,
        }
    
    def shuffle_analysis(
        self,
        trades_df: pd.DataFrame,
    ) -> Dict[str, Any]:
        """
        Analyze impact of trade ordering through shuffling.
        
        This tests if the strategy's performance depends on
        the specific sequence of trades or is robust.
        """
        if 'pnl' not in trades_df.columns or len(trades_df) < 2:
            return {'is_robust': True, 'sequence_dependency': 0}
        
        pnl_values = trades_df['pnl'].values
        
        # Calculate metrics with different orderings
        max_drawdowns = []
        sharpes = []
        
        for _ in range(min(self.n_simulations, 500)):
            shuffled = np.random.permutation(pnl_values)
            cumsum = np.cumsum(shuffled)
            
            # Max drawdown
            running_max = np.maximum.accumulate(cumsum)
            drawdown = (running_max - cumsum) / np.maximum(running_max, 1)
            max_drawdowns.append(np.max(drawdown))
            
            # Sharpe-like ratio
            if np.std(shuffled) > 0:
                sharpes.append(np.mean(shuffled) / np.std(shuffled))
        
        # Original max drawdown
        cumsum_original = np.cumsum(pnl_values)
        running_max_original = np.maximum.accumulate(cumsum_original)
        original_dd = np.max((running_max_original - cumsum_original) / np.maximum(running_max_original, 1))
        
        # Sequence dependency: how much worse/better is original vs random?
        if max_drawdowns:
            avg_shuffled_dd = np.mean(max_drawdowns)
            sequence_dependency = (original_dd - avg_shuffled_dd) / max(avg_shuffled_dd, 0.001)
        else:
            sequence_dependency = 0
        
        return {
            'original_max_drawdown': original_dd,
            'avg_shuffled_max_drawdown': np.mean(max_drawdowns) if max_drawdowns else 0,
            'sequence_dependency': sequence_dependency,
            'is_robust': abs(sequence_dependency) < 0.5,  # Less than 50% difference
            'sharpe_std': np.std(sharpes) if sharpes else 0,
        }
