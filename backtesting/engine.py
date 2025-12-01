"""
Backtesting Engine Module.

High-performance backtesting using VectorBT for vectorized operations.
Supports long/short positions, stop loss/take profit, and comprehensive metrics.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
import logging
from datetime import datetime

# Try to import VectorBT
try:
    import vectorbt as vbt
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    logging.warning("VectorBT not available. Install with: pip install vectorbt")

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """Configuration for backtesting."""
    initial_capital: float = 100_000.0
    commission: float = 0.001  # 0.1% per trade
    slippage: float = 0.0005  # 0.05% slippage
    position_size: float = 0.1  # 10% of capital per trade
    max_positions: int = 5  # Maximum concurrent positions
    use_stop_loss: bool = True
    use_take_profit: bool = True
    allow_shorting: bool = True
    freq: str = '1D'  # Data frequency


@dataclass
class BacktestResult:
    """Container for backtest results."""
    # Performance metrics
    total_return: float = 0.0
    annual_return: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0  # in days
    
    # Trade statistics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    avg_trade_duration: float = 0.0
    
    # Risk metrics
    volatility: float = 0.0
    var_95: float = 0.0  # Value at Risk
    cvar_95: float = 0.0  # Conditional VaR
    beta: float = 0.0
    alpha: float = 0.0
    
    # Additional data
    equity_curve: pd.Series = None
    trades: pd.DataFrame = None
    daily_returns: pd.Series = None
    monthly_returns: pd.DataFrame = None


class BacktestEngine:
    """
    High-performance backtesting engine using VectorBT.
    
    Features:
    - Vectorized operations for speed (100x faster than event-driven)
    - Long and short position support
    - Stop loss and take profit orders
    - Comprehensive performance metrics
    - Visualization support
    
    Example:
        >>> engine = BacktestEngine()
        >>> result = engine.run(price_data, signals)
        >>> print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
        >>> engine.plot_results()
    """
    
    def __init__(
        self,
        config: BacktestConfig = None,
    ):
        """
        Initialize backtest engine.
        
        Args:
            config: Backtest configuration
        """
        self.config = config or BacktestConfig()
        self.result = None
        self.portfolio = None
        
        if not VECTORBT_AVAILABLE:
            logger.warning(
                "VectorBT not available. Using fallback backtesting. "
                "Install with: pip install vectorbt"
            )
    
    def run(
        self,
        prices: pd.DataFrame,
        signals: Union[pd.Series, np.ndarray],
        stop_loss: Optional[Union[pd.Series, float]] = None,
        take_profit: Optional[Union[pd.Series, float]] = None,
    ) -> BacktestResult:
        """
        Run backtest on price data with signals.
        
        Args:
            prices: DataFrame with OHLC data (or at least 'close')
            signals: Trading signals (1: buy, -1: sell, 0: hold)
            stop_loss: Stop loss levels or percentage
            take_profit: Take profit levels or percentage
            
        Returns:
            BacktestResult with all metrics
        """
        logger.info("Running backtest...")
        
        # Prepare data
        if isinstance(prices, pd.DataFrame):
            close = prices['close'] if 'close' in prices.columns else prices.iloc[:, 0]
        else:
            close = prices
        
        if isinstance(signals, pd.Series):
            signals = signals.values
        
        # Align signals with prices
        if len(signals) != len(close):
            logger.warning(
                f"Signal length ({len(signals)}) doesn't match price length ({len(close)}). "
                "Truncating to shorter length."
            )
            min_len = min(len(signals), len(close))
            signals = signals[-min_len:]
            close = close.iloc[-min_len:]
        
        if VECTORBT_AVAILABLE:
            result = self._run_vectorbt(close, signals, stop_loss, take_profit, prices)
        else:
            result = self._run_fallback(close, signals, stop_loss, take_profit)
        
        self.result = result
        return result
    
    def _run_vectorbt(
        self,
        close: pd.Series,
        signals: np.ndarray,
        stop_loss: Optional[Union[pd.Series, float]],
        take_profit: Optional[Union[pd.Series, float]],
        prices: pd.DataFrame,
    ) -> BacktestResult:
        """Run backtest using VectorBT."""
        
        # Create entry signals
        entries = signals == 1  # Long entries
        exits = signals == -1  # Exit signals
        
        # Also create short entries if allowed
        if self.config.allow_shorting:
            short_entries = signals == -1
            short_exits = signals == 1
        
        # Prepare stop loss and take profit
        sl_stop = None
        tp_stop = None
        
        if self.config.use_stop_loss and stop_loss is not None:
            if isinstance(stop_loss, (int, float)):
                sl_stop = stop_loss
            else:
                # For variable stop loss, use the mean as approximation
                sl_stop = float(np.nanmean(stop_loss))
        
        if self.config.use_take_profit and take_profit is not None:
            if isinstance(take_profit, (int, float)):
                tp_stop = take_profit
            else:
                tp_stop = float(np.nanmean(take_profit))
        
        # Run long strategy
        try:
            long_portfolio = vbt.Portfolio.from_signals(
                close=close,
                entries=entries,
                exits=exits,
                init_cash=self.config.initial_capital,
                size=self.config.position_size,
                size_type='percent',
                fees=self.config.commission,
                slippage=self.config.slippage,
                sl_stop=sl_stop,
                tp_stop=tp_stop,
                freq=self.config.freq,
            )
            
            # Run short strategy if allowed
            if self.config.allow_shorting:
                short_portfolio = vbt.Portfolio.from_signals(
                    close=close,
                    entries=short_entries,
                    exits=short_exits,
                    init_cash=self.config.initial_capital,
                    size=self.config.position_size,
                    size_type='percent',
                    fees=self.config.commission,
                    slippage=self.config.slippage,
                    direction='shortonly',  # Updated for newer VectorBT API
                    sl_stop=sl_stop,
                    tp_stop=tp_stop,
                    freq=self.config.freq,
                )
                
                # Combine portfolios (simplified: use long only for now)
                self.portfolio = long_portfolio
            else:
                self.portfolio = long_portfolio
            
        except Exception as e:
            logger.warning(f"VectorBT error: {e}. Using simplified backtest.")
            # Simplified backtest without stops
            self.portfolio = vbt.Portfolio.from_signals(
                close=close,
                entries=entries,
                exits=exits,
                init_cash=self.config.initial_capital,
                size=self.config.position_size,
                size_type='percent',
                fees=self.config.commission,
                freq=self.config.freq,
            )
        
        # Extract metrics
        result = self._extract_vectorbt_metrics(self.portfolio, close)
        
        return result
    
    def _extract_vectorbt_metrics(
        self,
        portfolio: Any,
        close: pd.Series,
    ) -> BacktestResult:
        """Extract metrics from VectorBT portfolio."""
        
        stats = portfolio.stats()
        
        # Get trades
        try:
            trades_df = portfolio.trades.records_readable
        except:
            trades_df = pd.DataFrame()
        
        # Calculate additional metrics
        returns = portfolio.returns()
        equity = portfolio.value()
        
        # Drawdown analysis
        drawdown = portfolio.drawdown()
        max_dd = float(drawdown.max())
        
        # Calculate trade statistics
        if len(trades_df) > 0:
            pnl = trades_df['PnL'] if 'PnL' in trades_df.columns else pd.Series([0])
            winning = pnl[pnl > 0]
            losing = pnl[pnl < 0]
            
            total_trades = len(trades_df)
            winning_trades = len(winning)
            losing_trades = len(losing)
            
            avg_win = float(winning.mean()) if len(winning) > 0 else 0
            avg_loss = float(losing.mean()) if len(losing) > 0 else 0
            
            gross_profit = winning.sum() if len(winning) > 0 else 0
            gross_loss = abs(losing.sum()) if len(losing) > 0 else 1
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        else:
            total_trades = 0
            winning_trades = 0
            losing_trades = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0
        
        # VaR and CVaR
        var_95 = float(returns.quantile(0.05))
        cvar_95 = float(returns[returns <= var_95].mean()) if len(returns[returns <= var_95]) > 0 else var_95
        
        # Monthly returns
        try:
            monthly = returns.resample('ME').apply(lambda x: (1 + x).prod() - 1)
            monthly_df = monthly.to_frame('returns')
            monthly_df['year'] = monthly_df.index.year
            monthly_df['month'] = monthly_df.index.month
            monthly_returns = monthly_df.pivot(index='year', columns='month', values='returns')
        except:
            monthly_returns = pd.DataFrame()
        
        # Handle max drawdown duration (can be Timedelta or numeric)
        max_dd_duration_raw = stats.get('Max Drawdown Duration', 0)
        if hasattr(max_dd_duration_raw, 'days'):
            max_dd_duration = int(max_dd_duration_raw.days)
        elif max_dd_duration_raw is None or pd.isna(max_dd_duration_raw):
            max_dd_duration = 0
        else:
            max_dd_duration = int(max_dd_duration_raw)
        
        # Handle avg trade duration (can be Timedelta or numeric)
        avg_trade_duration_raw = stats.get('Avg Winning Trade Duration', 0)
        if hasattr(avg_trade_duration_raw, 'days'):
            avg_trade_duration = float(avg_trade_duration_raw.days)
        elif avg_trade_duration_raw is None or pd.isna(avg_trade_duration_raw):
            avg_trade_duration = 0.0
        else:
            avg_trade_duration = float(avg_trade_duration_raw)
        
        result = BacktestResult(
            total_return=float(stats.get('Total Return [%]', 0)) / 100,
            annual_return=float(stats.get('Annualized Return [%]', 0)) / 100,
            sharpe_ratio=float(stats.get('Sharpe Ratio', 0)),
            sortino_ratio=float(stats.get('Sortino Ratio', 0)),
            calmar_ratio=float(stats.get('Calmar Ratio', 0)),
            max_drawdown=max_dd,
            max_drawdown_duration=max_dd_duration,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=winning_trades / total_trades if total_trades > 0 else 0,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            avg_trade_duration=avg_trade_duration,
            volatility=float(stats.get('Annualized Volatility [%]', 0)) / 100,
            var_95=var_95,
            cvar_95=cvar_95,
            equity_curve=equity,
            trades=trades_df,
            daily_returns=returns,
            monthly_returns=monthly_returns,
        )
        
        return result
    
    def _run_fallback(
        self,
        close: pd.Series,
        signals: np.ndarray,
        stop_loss: Optional[Union[pd.Series, float]],
        take_profit: Optional[Union[pd.Series, float]],
    ) -> BacktestResult:
        """Fallback backtesting without VectorBT."""
        
        logger.info("Using fallback backtesting engine")
        
        cash = self.config.initial_capital
        position = 0
        entry_price = 0
        equity_curve = []
        trades = []
        
        for i in range(len(close)):
            price = close.iloc[i]
            signal = signals[i]
            
            # Calculate current equity
            if position > 0:
                current_value = cash + position * price
            elif position < 0:
                current_value = cash + position * price
            else:
                current_value = cash
            
            equity_curve.append(current_value)
            
            # Check stop loss / take profit
            if position != 0:
                pnl_pct = (price - entry_price) / entry_price * np.sign(position)
                
                # Stop loss
                if self.config.use_stop_loss and stop_loss is not None:
                    sl = stop_loss if isinstance(stop_loss, float) else 0.02
                    if pnl_pct < -sl:
                        # Close position
                        trades.append({
                            'entry_price': entry_price,
                            'exit_price': price,
                            'position': position,
                            'pnl': position * (price - entry_price),
                        })
                        cash += position * price * (1 - self.config.commission)
                        position = 0
                        continue
                
                # Take profit
                if self.config.use_take_profit and take_profit is not None:
                    tp = take_profit if isinstance(take_profit, float) else 0.04
                    if pnl_pct > tp:
                        trades.append({
                            'entry_price': entry_price,
                            'exit_price': price,
                            'position': position,
                            'pnl': position * (price - entry_price),
                        })
                        cash += position * price * (1 - self.config.commission)
                        position = 0
                        continue
            
            # Process signals
            if signal == 1 and position <= 0:  # Buy
                if position < 0:  # Close short first
                    cash += position * price * (1 - self.config.commission)
                    trades.append({
                        'entry_price': entry_price,
                        'exit_price': price,
                        'position': position,
                        'pnl': position * (price - entry_price),
                    })
                
                # Open long
                position_value = cash * self.config.position_size
                position = position_value / price
                cash -= position_value * (1 + self.config.commission)
                entry_price = price
            
            elif signal == -1 and position >= 0:  # Sell
                if position > 0:  # Close long first
                    cash += position * price * (1 - self.config.commission)
                    trades.append({
                        'entry_price': entry_price,
                        'exit_price': price,
                        'position': position,
                        'pnl': position * (price - entry_price),
                    })
                
                if self.config.allow_shorting:
                    # Open short
                    position_value = cash * self.config.position_size
                    position = -position_value / price
                    cash += position_value * (1 - self.config.commission)
                    entry_price = price
                else:
                    position = 0
        
        # Calculate metrics
        equity = pd.Series(equity_curve, index=close.index)
        returns = equity.pct_change().dropna()
        
        total_return = (equity.iloc[-1] - equity.iloc[0]) / equity.iloc[0]
        
        # Sharpe ratio
        if returns.std() > 0:
            sharpe = np.sqrt(252) * returns.mean() / returns.std()
        else:
            sharpe = 0
        
        # Sortino ratio
        neg_returns = returns[returns < 0]
        if len(neg_returns) > 0 and neg_returns.std() > 0:
            sortino = np.sqrt(252) * returns.mean() / neg_returns.std()
        else:
            sortino = 0
        
        # Max drawdown
        peak = equity.expanding().max()
        drawdown = (equity - peak) / peak
        max_dd = abs(drawdown.min())
        
        # Trade statistics
        trades_df = pd.DataFrame(trades)
        if len(trades_df) > 0:
            winning = trades_df[trades_df['pnl'] > 0]
            losing = trades_df[trades_df['pnl'] < 0]
        else:
            winning = pd.DataFrame()
            losing = pd.DataFrame()
        
        result = BacktestResult(
            total_return=total_return,
            annual_return=total_return * 252 / len(close),
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_dd,
            total_trades=len(trades),
            winning_trades=len(winning),
            losing_trades=len(losing),
            win_rate=len(winning) / len(trades) if trades else 0,
            avg_win=float(winning['pnl'].mean()) if len(winning) > 0 else 0,
            avg_loss=float(losing['pnl'].mean()) if len(losing) > 0 else 0,
            volatility=returns.std() * np.sqrt(252),
            equity_curve=equity,
            trades=trades_df,
            daily_returns=returns,
        )
        
        return result
    
    def print_summary(self, result: BacktestResult = None):
        """Print backtest summary."""
        result = result or self.result
        
        if result is None:
            logger.warning("No backtest results available")
            return
        
        print("\n" + "="*60)
        print("BACKTEST RESULTS")
        print("="*60)
        
        print("\n📈 PERFORMANCE METRICS")
        print("-"*40)
        print(f"Total Return:        {result.total_return:>10.2%}")
        print(f"Annual Return:       {result.annual_return:>10.2%}")
        print(f"Sharpe Ratio:        {result.sharpe_ratio:>10.2f}")
        print(f"Sortino Ratio:       {result.sortino_ratio:>10.2f}")
        print(f"Calmar Ratio:        {result.calmar_ratio:>10.2f}")
        print(f"Max Drawdown:        {result.max_drawdown:>10.2%}")
        print(f"Volatility (Ann.):   {result.volatility:>10.2%}")
        
        print("\n📊 TRADE STATISTICS")
        print("-"*40)
        print(f"Total Trades:        {result.total_trades:>10}")
        print(f"Winning Trades:      {result.winning_trades:>10}")
        print(f"Losing Trades:       {result.losing_trades:>10}")
        print(f"Win Rate:            {result.win_rate:>10.2%}")
        print(f"Profit Factor:       {result.profit_factor:>10.2f}")
        print(f"Avg Win:             {result.avg_win:>10.2f}")
        print(f"Avg Loss:            {result.avg_loss:>10.2f}")
        
        print("\n⚠️ RISK METRICS")
        print("-"*40)
        print(f"VaR (95%):           {result.var_95:>10.2%}")
        print(f"CVaR (95%):          {result.cvar_95:>10.2%}")
        
        print("\n" + "="*60)
    
    def plot_results(self, result: BacktestResult = None, save_html: bool = True):
        """Plot backtest results."""
        result = result or self.result
        
        if result is None:
            logger.warning("No backtest results available")
            return
        
        if VECTORBT_AVAILABLE and self.portfolio is not None:
            # Use VectorBT plotting - save to HTML file instead of showing in terminal
            try:
                fig = self.portfolio.plot()
                if save_html:
                    # Save to HTML file for viewing in browser
                    output_path = "backtest_results.html"
                    fig.write_html(output_path)
                    logger.info(f"📊 Backtest chart saved to: {output_path}")
                else:
                    # Only show() if in interactive environment
                    fig.show()
            except Exception as e:
                logger.warning(f"VectorBT plotting failed: {e}")
                self._plot_fallback(result)
        else:
            self._plot_fallback(result)
    
    def _plot_fallback(self, result: BacktestResult):
        """Fallback plotting using matplotlib."""
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(3, 1, figsize=(14, 10))
            
            # Equity curve
            ax1 = axes[0]
            result.equity_curve.plot(ax=ax1)
            ax1.set_title('Equity Curve')
            ax1.set_ylabel('Portfolio Value ($)')
            ax1.grid(True, alpha=0.3)
            
            # Drawdown
            ax2 = axes[1]
            peak = result.equity_curve.expanding().max()
            drawdown = (result.equity_curve - peak) / peak
            drawdown.plot(ax=ax2, color='red')
            ax2.fill_between(drawdown.index, 0, drawdown, color='red', alpha=0.3)
            ax2.set_title('Drawdown')
            ax2.set_ylabel('Drawdown (%)')
            ax2.grid(True, alpha=0.3)
            
            # Daily returns
            ax3 = axes[2]
            result.daily_returns.plot(ax=ax3, alpha=0.7)
            ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax3.set_title('Daily Returns')
            ax3.set_ylabel('Return (%)')
            ax3.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            logger.warning("matplotlib not available for plotting")
    
    def compare_strategies(
        self,
        strategies: Dict[str, Tuple[pd.Series, np.ndarray]],
    ) -> pd.DataFrame:
        """
        Compare multiple strategies.
        
        Args:
            strategies: Dict of strategy name -> (prices, signals)
            
        Returns:
            DataFrame with comparison metrics
        """
        results = {}
        
        for name, (prices, signals) in strategies.items():
            result = self.run(prices, signals)
            results[name] = {
                'Total Return': result.total_return,
                'Annual Return': result.annual_return,
                'Sharpe Ratio': result.sharpe_ratio,
                'Sortino Ratio': result.sortino_ratio,
                'Max Drawdown': result.max_drawdown,
                'Win Rate': result.win_rate,
                'Profit Factor': result.profit_factor,
                'Total Trades': result.total_trades,
            }
        
        comparison = pd.DataFrame(results).T
        comparison = comparison.round(4)
        
        return comparison
    
    def walk_forward_analysis(
        self,
        prices: pd.DataFrame,
        signals: np.ndarray,
        train_period: int = 252,  # 1 year
        test_period: int = 63,    # 3 months
    ) -> List[BacktestResult]:
        """
        Perform walk-forward analysis.
        
        Args:
            prices: Price data
            signals: Trading signals
            train_period: Training period length
            test_period: Test period length
            
        Returns:
            List of results for each test period
        """
        results = []
        
        total_periods = len(prices)
        start = train_period
        
        while start + test_period <= total_periods:
            # Test period
            test_prices = prices.iloc[start:start + test_period]
            test_signals = signals[start:start + test_period]
            
            result = self.run(test_prices, test_signals)
            result.start_date = test_prices.index[0]
            result.end_date = test_prices.index[-1]
            results.append(result)
            
            start += test_period
        
        # Aggregate results
        if results:
            logger.info(f"\nWalk-Forward Analysis ({len(results)} periods):")
            returns = [r.total_return for r in results]
            sharpes = [r.sharpe_ratio for r in results]
            logger.info(f"  Mean Return: {np.mean(returns):.2%} (+/- {np.std(returns):.2%})")
            logger.info(f"  Mean Sharpe: {np.mean(sharpes):.2f} (+/- {np.std(sharpes):.2f})")
        
        return results
