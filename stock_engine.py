"""
Stock Trading Engine for 5-Second Candles.

Main entry point for high-frequency stock trading on Alpaca.
Integrates:
- Real-time Alpaca trade streaming
- Custom 5-second candle aggregation
- Microstructure features (adapted for stocks)
- Existing HFT strategies
- Micro trading strategies

Requirements:
- Alpaca account (free): https://alpaca.markets
- Set environment variables:
    ALPACA_API_KEY=your_key
    ALPACA_SECRET_KEY=your_secret

Usage:
    # Paper trading (recommended)
    python stock_engine.py --symbols AAPL TSLA --mode paper
    
    # Backtest
    python stock_engine.py --symbols AAPL --mode backtest
"""

import argparse
import asyncio
import logging
import signal
import sys
import os
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd

# Local imports
from data.fetchers import AlpacaDataHandler, AlpacaConfig, StockSentiment, create_alpaca_handler
from features import MicrostructureFeatures, MicrostructureConfig
from strategies import MicroStrategyEnsemble, MicroSignal, HFTStrategyEnsemble

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('stock_engine.log'),
    ]
)
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class StockEngineConfig:
    """Configuration for the stock trading engine."""
    
    # Trading parameters
    symbols: List[str] = field(default_factory=lambda: ['AAPL', 'TSLA'])
    candle_interval: int = 5  # 5-second candles
    
    # Capital and risk (adjusted for stocks)
    capital: float = 500.0
    max_position_pct: float = 0.25  # 25% max per trade
    max_daily_loss_pct: float = 0.03  # 3% max daily loss
    max_trades_per_hour: int = 30
    
    # Strategy settings
    aggressive: bool = True
    min_confidence: float = 0.5
    min_strength: float = 0.3
    
    # Risk management
    use_trailing_stop: bool = True
    trailing_stop_pct: float = 0.003  # 0.3%
    position_timeout_seconds: float = 300  # 5 min for stocks
    
    # Alpaca settings
    paper_trading: bool = True
    alpaca_api_key: str = ""
    alpaca_secret_key: str = ""
    data_feed: str = "iex"  # 'iex' (free) or 'sip' (paid)
    
    def __post_init__(self):
        if not self.alpaca_api_key:
            self.alpaca_api_key = os.getenv('ALPACA_API_KEY', '')
        if not self.alpaca_secret_key:
            self.alpaca_secret_key = os.getenv('ALPACA_SECRET_KEY', '')


@dataclass
class StockPosition:
    """Current stock position."""
    symbol: str
    direction: int  # 1 = long, -1 = short
    entry_price: float
    shares: int
    stop_loss: float
    take_profit: float
    entry_time: datetime
    unrealized_pnl: float = 0.0
    highest_price: float = 0.0
    lowest_price: float = 0.0
    
    @property
    def value(self) -> float:
        return self.entry_price * self.shares


@dataclass
class StockTradeRecord:
    """Completed trade record."""
    symbol: str
    direction: int
    entry_price: float
    exit_price: float
    shares: int
    pnl: float
    pnl_pct: float
    entry_time: datetime
    exit_time: datetime
    reason: str
    strategy: str


# =============================================================================
# STOCK TRADING ENGINE
# =============================================================================

class StockTradingEngine:
    """
    High-frequency stock trading engine for Alpaca.
    
    Features:
    - Real-time trade streaming → 5-second candles
    - Microstructure features (volume imbalance, VWAP, etc.)
    - Micro strategies + existing HFT strategies
    - Paper trading mode for testing
    
    Note: Due to PDT rules, this is best used with:
    - Paper trading for testing
    - $25k+ accounts for live day trading
    - Swing trading (hold overnight) with smaller accounts
    """
    
    def __init__(self, config: StockEngineConfig = None):
        self.config = config or StockEngineConfig()
        
        # Components
        self.data_handler: Optional[AlpacaDataHandler] = None
        self.feature_generator: Optional[MicrostructureFeatures] = None
        self.strategy_ensemble: Optional[MicroStrategyEnsemble] = None
        
        # State
        self.positions: Dict[str, StockPosition] = {}
        self.trade_history: List[StockTradeRecord] = []
        self.daily_pnl: float = 0.0
        self.total_pnl: float = 0.0
        self.trades_today: int = 0
        self.is_running: bool = False
        
        # Risk controls
        self.is_halted: bool = False
        self.halt_reason: str = ""
        
        logger.info(f"StockTradingEngine initialized")
        logger.info(f"  Symbols: {self.config.symbols}")
        logger.info(f"  Capital: ${self.config.capital}")
        logger.info(f"  Mode: {'Paper' if self.config.paper_trading else 'LIVE'}")
        logger.info(f"  Candle Interval: {self.config.candle_interval}s")
    
    # -------------------------------------------------------------------------
    # INITIALIZATION
    # -------------------------------------------------------------------------
    
    async def initialize(self):
        """Initialize all components."""
        logger.info("Initializing engine components...")
        
        # Check API credentials
        if not self.config.alpaca_api_key or not self.config.alpaca_secret_key:
            logger.error("❌ Alpaca API credentials not set!")
            logger.error("   Set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables")
            logger.error("   Get free keys at: https://alpaca.markets")
            raise ValueError("Alpaca credentials required")
        
        # 1. Data handler
        alpaca_config = AlpacaConfig(
            api_key=self.config.alpaca_api_key,
            secret_key=self.config.alpaca_secret_key,
            paper=self.config.paper_trading,
            feed=self.config.data_feed,
            custom_intervals=(5, 10, 30),
        )
        
        self.data_handler = AlpacaDataHandler(
            symbols=self.config.symbols,
            config=alpaca_config,
        )
        
        # Preload historical data
        logger.info("Preloading historical data...")
        self.data_handler.preload_historical('1Min', 500)
        
        # 2. Feature generator
        self.feature_generator = MicrostructureFeatures()
        
        # 3. Strategy ensemble
        self.strategy_ensemble = MicroStrategyEnsemble(
            capital=self.config.capital,
            aggressive=self.config.aggressive,
        )
        
        logger.info("✅ Engine initialized successfully")
    
    # -------------------------------------------------------------------------
    # TRADING LOOP
    # -------------------------------------------------------------------------
    
    async def run(self):
        """Main trading loop."""
        await self.initialize()
        
        # Connect to Alpaca
        await self.data_handler.connect()
        
        logger.info("🚀 Starting trading loop...")
        logger.info("   Waiting for trades to aggregate into candles...")
        self.is_running = True
        
        try:
            while self.is_running:
                await self._trading_iteration()
                await asyncio.sleep(self.config.candle_interval)
                
        except asyncio.CancelledError:
            logger.info("Trading loop cancelled")
        except Exception as e:
            logger.error(f"Trading loop error: {e}")
            raise
        finally:
            await self.shutdown()
    
    async def _trading_iteration(self):
        """Single trading iteration."""
        if self.is_halted:
            return
        
        # Check daily loss limit
        if self.daily_pnl < -self.config.capital * self.config.max_daily_loss_pct:
            self.halt("Daily loss limit reached")
            return
        
        for symbol in self.config.symbols:
            try:
                await self._process_symbol(symbol)
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
    
    async def _process_symbol(self, symbol: str):
        """Process a single symbol."""
        # 1. Get candle data
        interval = f'{self.config.candle_interval}s'
        df = self.data_handler.get_buffer(symbol, n_bars=100, interval=interval)
        
        if len(df) < 60:
            return  # Wait for more data
        
        # 2. Add microstructure features
        df = self.feature_generator.add_all_features(df)
        
        # 3. Get trade flow sentiment
        sentiment = self.data_handler.get_sentiment(symbol, lookback_seconds=30)
        
        # 4. Manage existing position or look for entry
        if symbol in self.positions:
            await self._manage_position(symbol, df, sentiment)
        else:
            await self._look_for_entry(symbol, df, sentiment)
    
    async def _look_for_entry(self, symbol: str, df: pd.DataFrame, sentiment: StockSentiment):
        """Look for entry signal."""
        if self.trades_today >= self.config.max_trades_per_hour:
            return
        
        # Create sentiment-like object for strategy
        class SentimentWrapper:
            def __init__(self, s: StockSentiment):
                self.volume_imbalance = s.volume_imbalance
                self.trade_imbalance = s.trade_imbalance
                self.whale_signal = s.institutional_signal
        
        sentiment_obj = SentimentWrapper(sentiment)
        
        # Generate signal
        signal = self.strategy_ensemble.generate_signal(df, sentiment_obj)
        
        # Filter
        if (signal.direction == 0 or 
            signal.confidence < self.config.min_confidence or
            signal.strength < self.config.min_strength):
            return
        
        # Calculate position size
        current_price = df['close'].iloc[-1]
        position_value = self.config.capital * signal.position_size_pct
        shares = int(position_value / current_price)
        
        if shares < 1:
            return  # Can't buy fractional shares on Alpaca basic
        
        # Execute entry
        await self._execute_entry(
            symbol=symbol,
            direction=signal.direction,
            price=current_price,
            shares=shares,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            signal=signal,
        )
    
    async def _execute_entry(
        self,
        symbol: str,
        direction: int,
        price: float,
        shares: int,
        stop_loss: float,
        take_profit: float,
        signal: MicroSignal,
    ):
        """Execute entry trade."""
        side = 'BUY' if direction == 1 else 'SELL'
        
        if self.config.paper_trading:
            logger.info(f"📝 [PAPER] {side} {shares} {symbol} @ ${price:.2f}")
        else:
            # TODO: Implement actual Alpaca order execution
            logger.warning("Live trading not yet implemented")
            return
        
        # Record position
        self.positions[symbol] = StockPosition(
            symbol=symbol,
            direction=direction,
            entry_price=price,
            shares=shares,
            stop_loss=stop_loss,
            take_profit=take_profit,
            entry_time=datetime.now(),
            highest_price=price,
            lowest_price=price,
        )
        
        self.trades_today += 1
        
        logger.info(f"  Value: ${price * shares:.2f}")
        logger.info(f"  Stop: ${stop_loss:.2f}")
        logger.info(f"  Target: ${take_profit:.2f}")
        logger.info(f"  Confidence: {signal.confidence:.2%}")
    
    async def _manage_position(self, symbol: str, df: pd.DataFrame, sentiment: StockSentiment):
        """Manage existing position."""
        position = self.positions[symbol]
        current_price = df['close'].iloc[-1]
        
        # Update tracking
        position.highest_price = max(position.highest_price, current_price)
        if position.lowest_price == 0:
            position.lowest_price = current_price
        else:
            position.lowest_price = min(position.lowest_price, current_price)
        
        # Calculate unrealized P&L
        if position.direction == 1:
            position.unrealized_pnl = (current_price - position.entry_price) * position.shares
        else:
            position.unrealized_pnl = (position.entry_price - current_price) * position.shares
        
        exit_reason = None
        
        # Check stop loss
        if position.direction == 1 and current_price <= position.stop_loss:
            exit_reason = "Stop Loss"
        elif position.direction == -1 and current_price >= position.stop_loss:
            exit_reason = "Stop Loss"
        
        # Check take profit
        elif position.direction == 1 and current_price >= position.take_profit:
            exit_reason = "Take Profit"
        elif position.direction == -1 and current_price <= position.take_profit:
            exit_reason = "Take Profit"
        
        # Trailing stop
        elif self.config.use_trailing_stop and position.unrealized_pnl > 0:
            if position.direction == 1:
                trailing = position.highest_price * (1 - self.config.trailing_stop_pct)
                if current_price <= trailing:
                    exit_reason = "Trailing Stop"
            else:
                trailing = position.lowest_price * (1 + self.config.trailing_stop_pct)
                if current_price >= trailing:
                    exit_reason = "Trailing Stop"
        
        # Timeout
        time_held = (datetime.now() - position.entry_time).total_seconds()
        if time_held > self.config.position_timeout_seconds:
            exit_reason = "Timeout"
        
        if exit_reason:
            await self._execute_exit(symbol, current_price, exit_reason)
    
    async def _execute_exit(self, symbol: str, price: float, reason: str):
        """Execute exit trade."""
        position = self.positions[symbol]
        
        # Calculate P&L
        if position.direction == 1:
            pnl = (price - position.entry_price) * position.shares
        else:
            pnl = (position.entry_price - price) * position.shares
        
        pnl_pct = pnl / (position.entry_price * position.shares)
        
        # Update totals
        self.daily_pnl += pnl
        self.total_pnl += pnl
        
        # Log
        result = "WIN" if pnl >= 0 else "LOSS"
        logger.info(f"📤 [{result}] {symbol} - {reason}")
        logger.info(f"  Exit: ${price:.2f}")
        logger.info(f"  P&L: ${pnl:+.2f} ({pnl_pct*100:+.2f}%)")
        logger.info(f"  Daily P&L: ${self.daily_pnl:.2f}")
        
        # Record trade
        trade = StockTradeRecord(
            symbol=symbol,
            direction=position.direction,
            entry_price=position.entry_price,
            exit_price=price,
            shares=position.shares,
            pnl=pnl,
            pnl_pct=pnl_pct,
            entry_time=position.entry_time,
            exit_time=datetime.now(),
            reason=reason,
            strategy="MicroEnsemble",
        )
        self.trade_history.append(trade)
        
        del self.positions[symbol]
    
    # -------------------------------------------------------------------------
    # RISK CONTROLS
    # -------------------------------------------------------------------------
    
    def halt(self, reason: str):
        """Halt trading."""
        self.is_halted = True
        self.halt_reason = reason
        logger.warning(f"🛑 TRADING HALTED: {reason}")
    
    # -------------------------------------------------------------------------
    # SHUTDOWN
    # -------------------------------------------------------------------------
    
    async def shutdown(self):
        """Shutdown gracefully."""
        logger.info("Shutting down...")
        self.is_running = False
        
        # Close positions
        for symbol in list(self.positions.keys()):
            price = self.data_handler.get_latest_price(symbol)
            if price:
                await self._execute_exit(symbol, price, "Shutdown")
        
        if self.data_handler:
            await self.data_handler.disconnect()
        
        self._print_summary()
    
    def _print_summary(self):
        """Print session summary."""
        print("\n" + "="*60)
        print("STOCK TRADING SESSION SUMMARY")
        print("="*60)
        
        print(f"\n📊 PERFORMANCE")
        print(f"  Total P&L: ${self.total_pnl:.2f}")
        print(f"  Total Trades: {len(self.trade_history)}")
        
        if self.trade_history:
            wins = [t for t in self.trade_history if t.pnl > 0]
            losses = [t for t in self.trade_history if t.pnl <= 0]
            
            print(f"  Win Rate: {len(wins)/len(self.trade_history)*100:.1f}%")
            
            if wins:
                print(f"  Avg Win: ${np.mean([t.pnl for t in wins]):.2f}")
            if losses:
                print(f"  Avg Loss: ${np.mean([t.pnl for t in losses]):.2f}")
        
        print("="*60)
    
    # -------------------------------------------------------------------------
    # BACKTEST
    # -------------------------------------------------------------------------
    
    def backtest(
        self,
        symbol: str = "AAPL",
        n_bars: int = 500,
    ) -> Dict[str, Any]:
        """
        Run backtest on historical minute bars.
        
        Note: Uses 1-minute bars since that's the finest granularity
        available from Alpaca's free tier for historical data.
        """
        logger.info(f"Running backtest on {symbol}...")
        
        # Check credentials
        if not self.config.alpaca_api_key:
            logger.error("Alpaca API key required for backtest")
            return {}
        
        # Initialize
        self.data_handler = AlpacaDataHandler(
            [symbol],
            AlpacaConfig(
                api_key=self.config.alpaca_api_key,
                secret_key=self.config.alpaca_secret_key,
            )
        )
        
        self.feature_generator = MicrostructureFeatures()
        self.strategy_ensemble = MicroStrategyEnsemble(
            capital=self.config.capital,
            aggressive=self.config.aggressive,
        )
        
        # Fetch data
        df = self.data_handler.fetch_historical_bars(symbol, '1Min', n_bars)
        
        if df.empty:
            logger.error("No historical data available")
            return {}
        
        # Add features
        df = self.feature_generator.add_all_features(df)
        
        # Simulate
        equity = [self.config.capital]
        trades = []
        position = None
        
        for i in range(60, len(df)):
            data_slice = df.iloc[:i+1].copy()
            current_price = df['close'].iloc[i]
            
            # Manage position
            if position is not None:
                exit_triggered = False
                exit_reason = ""
                
                if position['direction'] == 1:
                    if current_price <= position['stop_loss']:
                        exit_triggered, exit_reason = True, "Stop Loss"
                    elif current_price >= position['take_profit']:
                        exit_triggered, exit_reason = True, "Take Profit"
                else:
                    if current_price >= position['stop_loss']:
                        exit_triggered, exit_reason = True, "Stop Loss"
                    elif current_price <= position['take_profit']:
                        exit_triggered, exit_reason = True, "Take Profit"
                
                if exit_triggered:
                    if position['direction'] == 1:
                        pnl = (current_price - position['entry']) * position['shares']
                    else:
                        pnl = (position['entry'] - current_price) * position['shares']
                    
                    trades.append({
                        'entry': position['entry'],
                        'exit': current_price,
                        'direction': position['direction'],
                        'pnl': pnl,
                        'reason': exit_reason,
                    })
                    
                    equity.append(equity[-1] + pnl)
                    position = None
                else:
                    equity.append(equity[-1])
            else:
                # Look for entry
                signal = self.strategy_ensemble.generate_signal(data_slice)
                
                if (signal.direction != 0 and 
                    signal.confidence >= self.config.min_confidence):
                    
                    position_value = equity[-1] * signal.position_size_pct
                    shares = int(position_value / current_price)
                    
                    if shares >= 1:
                        position = {
                            'direction': signal.direction,
                            'entry': current_price,
                            'shares': shares,
                            'stop_loss': signal.stop_loss,
                            'take_profit': signal.take_profit,
                        }
                
                equity.append(equity[-1])
        
        # Calculate metrics
        equity = np.array(equity)
        returns = np.diff(equity) / equity[:-1]
        
        total_return = (equity[-1] - equity[0]) / equity[0]
        max_dd = np.min(equity / np.maximum.accumulate(equity) - 1)
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252 * 390) if np.std(returns) > 0 else 0
        
        wins = [t for t in trades if t['pnl'] > 0]
        losses = [t for t in trades if t['pnl'] <= 0]
        
        results = {
            'total_return': total_return,
            'total_pnl': equity[-1] - equity[0],
            'max_drawdown': max_dd,
            'sharpe_ratio': sharpe,
            'total_trades': len(trades),
            'win_rate': len(wins) / len(trades) if trades else 0,
            'profit_factor': sum(t['pnl'] for t in wins) / abs(sum(t['pnl'] for t in losses)) if losses and sum(t['pnl'] for t in losses) != 0 else 0,
            'equity_curve': equity,
            'trades': trades,
        }
        
        # Print results
        print("\n" + "="*60)
        print(f"BACKTEST RESULTS - {symbol}")
        print("="*60)
        print(f"\n📊 PERFORMANCE")
        print(f"  Total Return: {results['total_return']*100:.2f}%")
        print(f"  Total P&L: ${results['total_pnl']:.2f}")
        print(f"  Max Drawdown: {results['max_drawdown']*100:.2f}%")
        print(f"  Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        print(f"\n📈 TRADE STATISTICS")
        print(f"  Total Trades: {results['total_trades']}")
        print(f"  Win Rate: {results['win_rate']*100:.1f}%")
        print(f"  Profit Factor: {results['profit_factor']:.2f}")
        print("="*60)
        
        return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Stock Trading Engine for 5-Second Candles'
    )
    parser.add_argument(
        '--symbols', '-s',
        nargs='+',
        default=['AAPL', 'TSLA'],
        help='Stock symbols to trade'
    )
    parser.add_argument(
        '--mode', '-m',
        choices=['paper', 'live', 'backtest'],
        default='paper',
        help='Trading mode'
    )
    parser.add_argument(
        '--capital',
        type=float,
        default=500.0,
        help='Starting capital'
    )
    parser.add_argument(
        '--interval',
        type=int,
        default=5,
        help='Candle interval in seconds (5, 10, or 30)'
    )
    
    args = parser.parse_args()
    
    config = StockEngineConfig(
        symbols=[s.upper() for s in args.symbols],
        capital=args.capital,
        candle_interval=args.interval,
        paper_trading=(args.mode != 'live'),
    )
    
    engine = StockTradingEngine(config)
    
    if args.mode == 'backtest':
        for symbol in args.symbols:
            engine.backtest(symbol.upper(), n_bars=500)
    else:
        loop = asyncio.get_event_loop()
        
        def signal_handler(sig, frame):
            logger.info("Shutdown signal received...")
            loop.create_task(engine.shutdown())
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        try:
            loop.run_until_complete(engine.run())
        except KeyboardInterrupt:
            pass
        finally:
            loop.close()


if __name__ == '__main__':
    main()
