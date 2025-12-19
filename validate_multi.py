"""
Multi-Symbol Validation Script.

Runs backtests across multiple symbols and aggregates results.
"""
# Suppress TensorFlow warnings
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Import trading engine
from main import TradingEngine

# Top 16 liquid stocks to test
SYMBOLS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META',
    'NVDA', 'AMD', 'NFLX', 'TSLA', 'JPM',
    'BAC', 'V', 'MA', 'DIS', 'WMT', 'HD'
]


def run_backtest(symbol: str, use_hft: bool = True) -> dict:
    """Run backtest for a single symbol."""
    try:
        engine = TradingEngine()
        engine.initialize(fetch_news=False, use_alternative_data=False)
        
        result = engine.run(
            symbol=symbol,
            mode='backtest',
            fetch_news=False,
            use_alternative_data=False,
            train_model=True,
            save_model=False,
            plot_results=False,
            use_hft=use_hft,
        )
        
        if result['backtest_result']:
            br = result['backtest_result']
            return {
                'symbol': symbol,
                'total_return': br.total_return,
                'annual_return': br.annual_return,
                'sharpe': br.sharpe_ratio,
                'sortino': br.sortino_ratio,
                'max_drawdown': br.max_drawdown,
                'volatility': br.volatility,
                'win_rate': br.win_rate,
                'profit_factor': br.profit_factor,
                'total_trades': br.total_trades,
                'status': 'SUCCESS'
            }
    except Exception as e:
        logger.error(f"{symbol} failed: {str(e)[:50]}")
        return {
            'symbol': symbol,
            'total_return': 0,
            'annual_return': 0,
            'sharpe': 0,
            'sortino': 0,
            'max_drawdown': 0,
            'volatility': 0,
            'win_rate': 0,
            'profit_factor': 0,
            'total_trades': 0,
            'status': f'FAILED: {str(e)[:30]}'
        }


def main():
    """Run multi-symbol validation."""
    print("\n" + "="*70)
    print("🚀 MULTI-SYMBOL VALIDATION TEST")
    print("="*70 + "\n")
    
    results = []
    
    for i, symbol in enumerate(SYMBOLS, 1):
        print(f"\n[{i}/{len(SYMBOLS)}] Testing {symbol}...")
        print("-" * 40)
        
        result = run_backtest(symbol, use_hft=True)
        results.append(result)
        
        if result['status'] == 'SUCCESS':
            print(f"  ✅ Return: {result['total_return']:.2%}, Sharpe: {result['sharpe']:.2f}, "
                  f"WinRate: {result['win_rate']:.1%}, Trades: {result['total_trades']}")
        else:
            print(f"  ❌ {result['status']}")
    
    # Create results DataFrame
    df = pd.DataFrame(results)
    
    # Summary statistics
    successful = df[df['status'] == 'SUCCESS']
    
    print("\n" + "="*70)
    print("📊 VALIDATION RESULTS SUMMARY")
    print("="*70)
    
    print(f"\nSymbols Tested: {len(SYMBOLS)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(df) - len(successful)}")
    
    if len(successful) > 0:
        print(f"\n{'Symbol':<8} {'Return':>10} {'Annual':>10} {'Sharpe':>8} {'Sortino':>8} {'MaxDD':>8} {'WinRate':>8} {'PF':>6} {'Trades':>7}")
        print("-" * 85)
        
        for _, row in successful.sort_values('sharpe', ascending=False).iterrows():
            print(f"{row['symbol']:<8} {row['total_return']:>9.2%} {row['annual_return']:>9.2%} "
                  f"{row['sharpe']:>8.2f} {row['sortino']:>8.2f} {row['max_drawdown']:>7.2%} "
                  f"{row['win_rate']:>7.1%} {row['profit_factor']:>6.2f} {row['total_trades']:>7}")
        
        print("-" * 85)
        print(f"{'AVERAGE':<8} {successful['total_return'].mean():>9.2%} {successful['annual_return'].mean():>9.2%} "
              f"{successful['sharpe'].mean():>8.2f} {successful['sortino'].mean():>8.2f} {successful['max_drawdown'].mean():>7.2%} "
              f"{successful['win_rate'].mean():>7.1%} {successful['profit_factor'].mean():>6.2f} {successful['total_trades'].mean():>7.0f}")
        
        print("\n📈 KEY METRICS:")
        print(f"  • Avg Sharpe Ratio: {successful['sharpe'].mean():.2f}")
        print(f"  • Profitable Symbols: {len(successful[successful['total_return'] > 0])}/{len(successful)}")
        print(f"  • Best Performer: {successful.loc[successful['sharpe'].idxmax(), 'symbol']} (Sharpe={successful['sharpe'].max():.2f})")
        print(f"  • Worst Performer: {successful.loc[successful['sharpe'].idxmin(), 'symbol']} (Sharpe={successful['sharpe'].min():.2f})")
        print(f"  • Avg Max Drawdown: {successful['max_drawdown'].mean():.2%}")
        print(f"  • Avg Win Rate: {successful['win_rate'].mean():.1%}")
    
    # Save results
    df.to_csv('validation_results.csv', index=False)
    print(f"\n💾 Results saved to: validation_results.csv")
    print("="*70 + "\n")
    
    return df


if __name__ == '__main__':
    main()
