"""
Crypto Trading Engine for 1-5 Second Candles.

Main entry point for high-frequency crypto trading on Binance.
Integrates all new components:
- Real-time Binance data streaming
- Microstructure features
- Crypto sentiment analysis
- Micro trading strategies
- Aggressive risk management for $500 capital

Usage:
    # Live trading
    python crypto_engine.py --symbol BTCUSDT --mode live
    
    # Paper trading (recommended first)
    python crypto_engine.py --symbol BTCUSDT --mode paper
    
    # Backtest
    python crypto_engine.py --symbol BTCUSDT --mode backtest
"""

import argparse
import asyncio
import logging
import signal
import sys
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

import numpy as np
import pandas as pd

# Local imports
from data.fetchers import BinanceDataHandler, create_binance_handler, CryptoSentiment
from features import MicrostructureFeatures, MicrostructureConfig
from features.crypto_sentiment import CryptoSentimentAnalyzer, get_crypto_sentiment
from strategies import MicroStrategyEnsemble, MicroSignal, create_micro_ensemble

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('crypto_engine.log'),
    ]
)
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class CryptoEngineConfig:
    """Configuration for the crypto trading engine."""
    
    # Trading parameters
    symbols: List[str] = field(default_factory=lambda: ['BTCUSDT'])
    timeframe: str = '1s'  # 1-second candles
    
    # Capital and risk
    capital: float = 500.0
    max_position_pct: float = 0.20  # 20% max per trade
    max_daily_loss_pct: float = 0.05  # 5% max daily loss ($25)
    max_trades_per_hour: int = 60
    
    # Strategy settings
    aggressive: bool = True
    min_confidence: float = 0.5
    min_strength: float = 0.3
    
    # Risk management
    use_trailing_stop: bool = True
    trailing_stop_pct: float = 0.002  # 0.2%
    position_timeout_seconds: float = 120  # Auto-close after 2 min
    
    # Data settings
    buffer_size: int = 1000
    preload_candles: int = 500
    
    # Modes
    paper_trading: bool = True  # Start with paper trading


@dataclass
class Position:
    """Current position."""
    symbol: str
    direction: int  # 1 = long, -1 = short
    entry_price: float
    quantity: float
    stop_loss: float
    take_profit: float
    entry_time: datetime
    unrealized_pnl: float = 0.0
    highest_price: float = 0.0  # For trailing stop
    lowest_price: float = 0.0   # For trailing stop
    
    @property
    def value(self) -> float:
        return self.entry_price * self.quantity


@dataclass
class TradeRecord:
    """Record of a completed trade."""
    symbol: str
    direction: int
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    pnl_pct: float
    entry_time: datetime
    exit_time: datetime
    reason: str
    strategy: str


# =============================================================================
# CRYPTO TRADING ENGINE
# =============================================================================

class CryptoTradingEngine:
    """
    High-frequency crypto trading engine for Binance.
    
    Designed for 1-5 second candle trading with $500 capital.
    Uses aggressive settings with tight risk management.
    
    Pipeline:
    1. Stream real-time data from Binance
    2. Calculate microstructure features
    3. Get crypto sentiment
    4. Generate signals from micro strategy ensemble
    5. Execute trades (paper or live)
    6. Manage positions with trailing stops
    """
    
    def __init__(self, config: CryptoEngineConfig = None):
        self.config = config or CryptoEngineConfig()
        
        # Components
        self.data_handler: Optional[BinanceDataHandler] = None
        self.feature_generator: Optional[MicrostructureFeatures] = None
        self.sentiment_analyzer: Optional[CryptoSentimentAnalyzer] = None
        self.strategy_ensemble: Optional[MicroStrategyEnsemble] = None
        
        # State
        self.positions: Dict[str, Position] = {}
        self.trade_history: List[TradeRecord] = []
        self.daily_pnl: float = 0.0
        self.total_pnl: float = 0.0
        self.trades_today: int = 0
        self.is_running: bool = False
        
        # Risk controls
        self.is_halted: bool = False
        self.halt_reason: str = ""
        
        logger.info(f"CryptoTradingEngine initialized")
        logger.info(f"  Symbols: {self.config.symbols}")
        logger.info(f"  Capital: ${self.config.capital}")
        logger.info(f"  Mode: {'Paper' if self.config.paper_trading else 'LIVE'}")
        logger.info(f"  Aggressive: {self.config.aggressive}")
    
    # -------------------------------------------------------------------------
    # INITIALIZATION
    # -------------------------------------------------------------------------
    
    async def initialize(self):
        """Initialize all components."""
        logger.info("Initializing engine components...")
        
        # 1. Data handler
        self.data_handler = BinanceDataHandler(
            symbols=self.config.symbols,
            buffer_size=self.config.buffer_size,
        )
        
        # Preload historical data
        logger.info("Preloading historical data...")
        self.data_handler.preload_historical(
            interval='1m',
            n_candles=self.config.preload_candles,
        )
        
        # 2. Feature generator
        self.feature_generator = MicrostructureFeatures(
            config=MicrostructureConfig()
        )
        
        # 3. Sentiment analyzer
        self.sentiment_analyzer = CryptoSentimentAnalyzer(
            aggressive_mode=self.config.aggressive
        )
        
        # 4. Strategy ensemble
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
        
        # Connect to Binance
        await self.data_handler.connect()
        
        # Subscribe to streams
        for symbol in self.config.symbols:
            await self.data_handler.subscribe_klines(self.config.timeframe)
            await self.data_handler.subscribe_trades()
        
        logger.info("🚀 Starting trading loop...")
        self.is_running = True
        
        try:
            while self.is_running:
                await self._trading_iteration()
                await asyncio.sleep(1)  # 1-second loop
                
        except asyncio.CancelledError:
            logger.info("Trading loop cancelled")
        except Exception as e:
            logger.error(f"Trading loop error: {e}")
            raise
        finally:
            await self.shutdown()
    
    async def _trading_iteration(self):
        """Single trading iteration."""
        # Check if halted
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
        # 1. Get latest data
        df = self.data_handler.get_buffer(symbol, n_bars=100, interval=self.config.timeframe)
        
        if len(df) < 60:
            return  # Not enough data
        
        # 2. Add microstructure features
        df = self.feature_generator.add_all_features(df)
        
        # 3. Get sentiment
        trade_sentiment = self.data_handler.get_sentiment(symbol, lookback_seconds=10)
        sentiment = self.sentiment_analyzer.get_composite_sentiment(symbol, trade_sentiment)
        
        # 4. Check existing position
        if symbol in self.positions:
            await self._manage_position(symbol, df, sentiment)
        else:
            # 5. Look for new signals
            await self._look_for_entry(symbol, df, sentiment)
    
    async def _look_for_entry(self, symbol: str, df: pd.DataFrame, sentiment: Any):
        """Look for entry signals."""
        # Check trade limit
        if self.trades_today >= self.config.max_trades_per_hour:
            return
        
        # Generate signal
        signal = self.strategy_ensemble.generate_signal(df, sentiment)
        
        # Filter by confidence and strength
        if (signal.direction == 0 or 
            signal.confidence < self.config.min_confidence or
            signal.strength < self.config.min_strength):
            return
        
        # Calculate position size
        position_value = self.config.capital * signal.position_size_pct
        current_price = df['close'].iloc[-1]
        quantity = position_value / current_price
        
        # Execute trade
        await self._execute_entry(
            symbol=symbol,
            direction=signal.direction,
            price=current_price,
            quantity=quantity,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            signal=signal,
        )
    
    async def _execute_entry(
        self,
        symbol: str,
        direction: int,
        price: float,
        quantity: float,
        stop_loss: float,
        take_profit: float,
        signal: MicroSignal,
    ):
        """Execute entry trade."""
        if self.config.paper_trading:
            logger.info(f"📝 [PAPER] {'BUY' if direction == 1 else 'SELL'} {symbol}")
        else:
            # TODO: Implement actual Binance order execution
            logger.warning("Live trading not implemented - using paper mode")
        
        # Record position
        self.positions[symbol] = Position(
            symbol=symbol,
            direction=direction,
            entry_price=price,
            quantity=quantity,
            stop_loss=stop_loss,
            take_profit=take_profit,
            entry_time=datetime.now(),
            highest_price=price,
            lowest_price=price,
        )
        
        self.trades_today += 1
        
        logger.info(f"  Entry: ${price:.2f}")
        logger.info(f"  Size: ${price * quantity:.2f} ({quantity:.6f})")
        logger.info(f"  Stop: ${stop_loss:.2f}")
        logger.info(f"  Target: ${take_profit:.2f}")
        logger.info(f"  Confidence: {signal.confidence:.2%}")
    
    async def _manage_position(self, symbol: str, df: pd.DataFrame, sentiment: Any):
        """Manage existing position."""
        position = self.positions[symbol]
        current_price = df['close'].iloc[-1]
        
        # Update tracking prices
        if position.direction == 1:
            position.highest_price = max(position.highest_price, current_price)
        else:
            position.lowest_price = min(position.lowest_price, current_price) if position.lowest_price > 0 else current_price
        
        # Calculate unrealized P&L
        if position.direction == 1:
            position.unrealized_pnl = (current_price - position.entry_price) * position.quantity
        else:
            position.unrealized_pnl = (position.entry_price - current_price) * position.quantity
        
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
        elif self.config.use_trailing_stop:
            if position.direction == 1:
                trailing_stop = position.highest_price * (1 - self.config.trailing_stop_pct)
                if current_price <= trailing_stop and position.unrealized_pnl > 0:
                    exit_reason = "Trailing Stop"
            else:
                trailing_stop = position.lowest_price * (1 + self.config.trailing_stop_pct)
                if current_price >= trailing_stop and position.unrealized_pnl > 0:
                    exit_reason = "Trailing Stop"
        
        # Position timeout
        time_in_position = (datetime.now() - position.entry_time).total_seconds()
        if time_in_position > self.config.position_timeout_seconds:
            exit_reason = "Timeout"
        
        # Exit if triggered
        if exit_reason:
            await self._execute_exit(symbol, current_price, exit_reason)
    
    async def _execute_exit(self, symbol: str, price: float, reason: str):
        """Execute exit trade."""
        position = self.positions[symbol]
        
        # Calculate P&L
        if position.direction == 1:
            pnl = (price - position.entry_price) * position.quantity
        else:
            pnl = (position.entry_price - price) * position.quantity
        
        pnl_pct = pnl / (position.entry_price * position.quantity)
        
        # Update totals
        self.daily_pnl += pnl
        self.total_pnl += pnl
        
        # Log
        pnl_str = f"+${pnl:.2f}" if pnl >= 0 else f"-${abs(pnl):.2f}"
        logger.info(f"📤 [{'WIN' if pnl >= 0 else 'LOSS'}] {symbol} - {reason}")
        logger.info(f"  Exit: ${price:.2f}")
        logger.info(f"  P&L: {pnl_str} ({pnl_pct*100:.2f}%)")
        logger.info(f"  Daily P&L: ${self.daily_pnl:.2f}")
        
        # Record trade
        trade = TradeRecord(
            symbol=symbol,
            direction=position.direction,
            entry_price=position.entry_price,
            exit_price=price,
            quantity=position.quantity,
            pnl=pnl,
            pnl_pct=pnl_pct,
            entry_time=position.entry_time,
            exit_time=datetime.now(),
            reason=reason,
            strategy="MicroEnsemble",
        )
        self.trade_history.append(trade)
        
        # Update strategy performance
        # self.strategy_ensemble.update_performance(...)
        
        # Remove position
        del self.positions[symbol]
    
    # -------------------------------------------------------------------------
    # RISK CONTROLS
    # -------------------------------------------------------------------------
    
    def halt(self, reason: str):
        """Halt trading."""
        self.is_halted = True
        self.halt_reason = reason
        logger.warning(f"🛑 TRADING HALTED: {reason}")
    
    def resume(self):
        """Resume trading."""
        self.is_halted = False
        self.halt_reason = ""
        logger.info("✅ Trading resumed")
    
    # -------------------------------------------------------------------------
    # SHUTDOWN
    # -------------------------------------------------------------------------
    
    async def shutdown(self):
        """Shutdown engine gracefully."""
        logger.info("Shutting down engine...")
        
        self.is_running = False
        
        # Close any open positions
        for symbol in list(self.positions.keys()):
            df = self.data_handler.get_buffer(symbol, n_bars=1)
            if len(df) > 0:
                current_price = df['close'].iloc[-1]
                await self._execute_exit(symbol, current_price, "Shutdown")
        
        # Disconnect
        if self.data_handler:
            await self.data_handler.disconnect()
        
        # Print summary
        self._print_summary()
    
    def _print_summary(self):
        """Print trading summary."""
        print("\n" + "="*60)
        print("TRADING SESSION SUMMARY")
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
            
            # Profit factor
            total_wins = sum(t.pnl for t in wins)
            total_losses = abs(sum(t.pnl for t in losses))
            if total_losses > 0:
                print(f"  Profit Factor: {total_wins/total_losses:.2f}")
        
        print("="*60)
    
    # -------------------------------------------------------------------------
    # BACKTEST MODE
    # -------------------------------------------------------------------------
    
    def backtest(
        self,
        symbol: str = "BTCUSDT",
        interval: str = "1m",
        n_candles: int = 1000,
    ) -> Dict[str, Any]:
        """
        Run backtest on historical data.
        
        Args:
            symbol: Trading pair
            interval: Candle interval
            n_candles: Number of candles to backtest
            
        Returns:
            Backtest results
        """
        logger.info(f"Running backtest on {symbol} ({interval})...")
        
        # Initialize components (sync)
        self.data_handler = BinanceDataHandler([symbol])
        self.feature_generator = MicrostructureFeatures()
        self.sentiment_analyzer = CryptoSentimentAnalyzer(aggressive_mode=True)
        self.strategy_ensemble = MicroStrategyEnsemble(
            capital=self.config.capital,
            aggressive=self.config.aggressive,
        )
        
        # Fetch historical data
        df = self.data_handler.fetch_historical_klines(symbol, interval, n_candles)
        
        if df.empty:
            logger.error("No historical data available")
            return {}
        
        # Add features
        df = self.feature_generator.add_all_features(df)
        
        # Simulate trading
        equity = [self.config.capital]
        signals = []
        trades = []
        
        position = None
        
        for i in range(60, len(df)):
            # Get slice up to current bar
            data_slice = df.iloc[:i+1].copy()
            current_price = df['close'].iloc[i]
            
            # Manage existing position
            if position is not None:
                # Check exits
                exit_triggered = False
                exit_reason = ""
                
                if position['direction'] == 1:
                    if current_price <= position['stop_loss']:
                        exit_triggered = True
                        exit_reason = "Stop Loss"
                    elif current_price >= position['take_profit']:
                        exit_triggered = True
                        exit_reason = "Take Profit"
                else:
                    if current_price >= position['stop_loss']:
                        exit_triggered = True
                        exit_reason = "Stop Loss"
                    elif current_price <= position['take_profit']:
                        exit_triggered = True
                        exit_reason = "Take Profit"
                
                if exit_triggered:
                    # Calculate P&L
                    if position['direction'] == 1:
                        pnl = (current_price - position['entry']) * position['qty']
                    else:
                        pnl = (position['entry'] - current_price) * position['qty']
                    
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
                    signal.confidence >= self.config.min_confidence and
                    signal.strength >= self.config.min_strength):
                    
                    position_value = equity[-1] * signal.position_size_pct
                    qty = position_value / current_price
                    
                    position = {
                        'direction': signal.direction,
                        'entry': current_price,
                        'qty': qty,
                        'stop_loss': signal.stop_loss,
                        'take_profit': signal.take_profit,
                    }
                    
                    signals.append({
                        'idx': i,
                        'direction': signal.direction,
                        'confidence': signal.confidence,
                        'strength': signal.strength,
                    })
                
                equity.append(equity[-1])
        
        # Calculate metrics
        equity = np.array(equity)
        returns = np.diff(equity) / equity[:-1]
        
        total_return = (equity[-1] - equity[0]) / equity[0]
        max_dd = np.min(equity / np.maximum.accumulate(equity) - 1)
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252 * 24 * 60) if np.std(returns) > 0 else 0
        
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
            'avg_win': np.mean([t['pnl'] for t in wins]) if wins else 0,
            'avg_loss': np.mean([t['pnl'] for t in losses]) if losses else 0,
            'equity_curve': equity,
            'trades': trades,
        }
        
        # Print results
        print("\n" + "="*60)
        print("BACKTEST RESULTS")
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
        print(f"  Avg Win: ${results['avg_win']:.2f}")
        print(f"  Avg Loss: ${results['avg_loss']:.2f}")
        print("="*60)
        
        return results


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Crypto Trading Engine for 1-5 Second Candles'
    )
    parser.add_argument(
        '--symbol', '-s',
        type=str,
        default='BTCUSDT',
        help='Trading pair (e.g., BTCUSDT, ETHUSDT)'
    )
    parser.add_argument(
        '--mode', '-m',
        type=str,
        choices=['live', 'paper', 'backtest'],
        default='paper',
        help='Trading mode'
    )
    parser.add_argument(
        '--capital',
        type=float,
        default=500.0,
        help='Starting capital in USD'
    )
    parser.add_argument(
        '--aggressive',
        action='store_true',
        default=True,
        help='Use aggressive settings'
    )
    parser.add_argument(
        '--conservative',
        action='store_true',
        help='Use conservative settings'
    )
    
    args = parser.parse_args()
    
    # Create config
    config = CryptoEngineConfig(
        symbols=[args.symbol.upper()],
        capital=args.capital,
        aggressive=not args.conservative,
        paper_trading=(args.mode != 'live'),
    )
    
    # Create engine
    engine = CryptoTradingEngine(config)
    
    if args.mode == 'backtest':
        # Run backtest
        results = engine.backtest(
            symbol=args.symbol.upper(),
            interval='1m',
            n_candles=1000,
        )
    else:
        # Run live/paper trading
        loop = asyncio.get_event_loop()
        
        # Handle graceful shutdown
        def signal_handler(sig, frame):
            logger.info("Received shutdown signal...")
            loop.create_task(engine.shutdown())
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        try:
            loop.run_until_complete(engine.run())
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
        finally:
            loop.close()


if __name__ == '__main__':
    main()
