# Sophisticated ML Trading Engine

A **professional-grade, institutional-quality** ML-powered trading engine that generates **buy/sell/hold signals** with **long/short position support** and comprehensive **backtesting capabilities**.

Built with techniques used by JP Morgan, Two Sigma, and other top quant funds.

## 🌟 Features

### 📈 Technical Analysis (150+ Indicators via TA-Lib)
- **Trend Indicators**: SMA, EMA, DEMA, TEMA, KAMA, Parabolic SAR, ADX, Aroon
- **Momentum**: RSI, MACD, Stochastic, CCI, Williams %R, Ultimate Oscillator
- **Volatility**: ATR, Bollinger Bands, Keltner Channels
- **Volume**: OBV, MFI, Chaikin A/D, VWAP
- **Patterns**: 61 candlestick pattern recognition

### 🤖 Machine Learning Ensemble
- **XGBoost**: Gradient boosting with regularization
- **LightGBM**: Fast, distributed gradient boosting
- **CatBoost**: Handles categorical features natively
- **Random Forest**: Robust, interpretable predictions
- **Stacking Meta-learner**: Combines all models optimally

### 📰 Sentiment Analysis
- **VADER**: Fast rule-based sentiment (40% weight)
- **TextBlob**: Pattern-based NLP (20% weight)
- **FinBERT**: Finance-specific transformer (40% weight)
- Fetches top 100 news articles per stock

### 🏦 Institutional-Grade Features (NEW)
- **Market Regime Detection**: Hidden Markov Models, volatility clustering
- **Fractal Analysis**: Hurst exponent, fractal dimension
- **Order Flow Proxies**: Volume imbalance, smart money indicators
- **Support/Resistance Detection**: Automatic level identification
- **Momentum Factors**: Fama-French style factor analysis

### 📊 Alternative Data Sources (NEW)
- **Reddit Sentiment**: r/wallstreetbets, r/stocks, r/investing sentiment
- **SEC EDGAR Filings**: Form 4 insider transactions, 8-K events
- **Institutional Holdings**: 13F filing analysis

### 🎯 Strategy Ensemble (NEW)
- **Momentum Strategy**: Trend-following with breakout confirmation
- **Mean Reversion Strategy**: Oversold/overbought opportunities
- **Breakout Strategy**: Volume-confirmed range breakouts
- **Regime-Switching Meta-Strategy**: Adapts to market conditions

### 📊 Backtesting (VectorBT)
- **100x faster** than event-driven backtesting
- Long and short position support
- Stop loss and take profit orders
- **Walk-Forward Analysis**: Prevents overfitting
- **Monte Carlo Validation**: Statistical significance testing

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/trading-engine.git
cd trading-engine

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### TA-Lib Installation (Required)

TA-Lib requires the underlying C library:

**Windows:**
```bash
# Download pre-built wheel from https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib
pip install TA_Lib‑0.4.28‑cp311‑cp311‑win_amd64.whl
```

**macOS:**
```bash
brew install ta-lib
pip install TA-Lib
```

**Linux:**
```bash
sudo apt-get install ta-lib
pip install TA-Lib
```

### Running the Engine

```bash
# Basic backtest
python main.py --symbol AAPL --mode backtest

# With news sentiment
python main.py --symbol TSLA --mode backtest --fetch-news

# With alternative data (Reddit, SEC)
python main.py --symbol MSFT --mode backtest --alternative-data

# Walk-forward backtest (prevents overfitting)
python main.py --symbol GOOGL --mode walkforward

# Live signals
python main.py --symbol AMZN --mode live --fetch-news

# Custom date range
python main.py --symbol NVDA --start-date 2022-01-01 --end-date 2024-01-01
```

## 📁 Project Structure

```
trading-engine/
├── main.py                    # Entry point
├── requirements.txt           # Dependencies
├── ARCHITECTURE.md            # Detailed architecture docs
├── config/
│   ├── settings.py            # Configuration dataclasses
│   └── model_params.py        # ML hyperparameters
├── data/
│   ├── fetchers/
│   │   ├── market_data.py     # OHLCV data fetching
│   │   └── news_fetcher.py    # News aggregation
│   ├── preprocessors/
│   │   └── cleaner.py         # Data cleaning
│   └── alternative/           # NEW: Alternative data
│       ├── reddit_fetcher.py  # Reddit sentiment
│       └── sec_fetcher.py     # SEC EDGAR filings
├── features/
│   ├── technical.py           # TA-Lib indicators
│   ├── statistical.py         # Rolling stats, z-scores
│   ├── sentiment_features.py  # NLP sentiment analysis
│   └── advanced_features.py   # NEW: Institutional features
├── models/
│   └── ml/
│       ├── gradient_boost.py  # XGBoost, LightGBM, CatBoost
│       └── ensemble.py        # Stacking ensemble
├── signals/
│   └── generator.py           # Buy/sell/hold signals
├── strategies/                # NEW: Strategy ensemble
│   └── institutional_strategies.py
└── backtesting/
    ├── engine.py              # VectorBT backtesting
    └── walk_forward.py        # NEW: Walk-forward analysis
```

## 📊 Signal Types

| Signal | Value | Description |
|--------|-------|-------------|
| STRONG_BUY | +2 | High confidence long |
| BUY | +1 | Standard long signal |
| HOLD | 0 | No action |
| SELL | -1 | Standard short signal |
| STRONG_SELL | -2 | High confidence short |

## 🔧 Configuration

### Trading Settings
```python
from config import TradingSettings

settings = TradingSettings(
    strong_buy_threshold=0.75,
    buy_threshold=0.6,
    sell_threshold=0.4,
    strong_sell_threshold=0.25,
    min_confidence=0.5,
    allow_short=True,
)
```

### Risk Settings
```python
from config import RiskSettings

risk = RiskSettings(
    max_position_size=0.1,      # 10% max per position
    max_portfolio_risk=0.02,    # 2% daily VaR limit
    stop_loss_pct=0.02,         # 2% stop loss
    take_profit_pct=0.04,       # 4% take profit (2:1 RR)
)
```

## 📈 Backtest Results Example

```
============================================================
BACKTEST RESULTS
============================================================

📈 PERFORMANCE METRICS
----------------------------------------
Total Return:             -2.15%
Annual Return:             0.00%
Sharpe Ratio:              -0.26
Max Drawdown:              0.00%
Volatility (Ann.):         0.00%

📊 TRADE STATISTICS
----------------------------------------
Total Trades:                220
Winning Trades:              112
Losing Trades:               108
Win Rate:                  50.91%
Profit Factor:              0.90
Avg Win:                  $169.59
Avg Loss:                -$195.75

============================================================
```

## 🧠 How It Works

### 1. Data Pipeline
```
Market Data → Cleaning → Technical Features → Statistical Features → Advanced Features → Alternative Data → Sentiment
```

### 2. ML Ensemble
```
Features → XGBoost  ─┐
Features → LightGBM ─┤→ Meta-Learner → Final Prediction
Features → CatBoost ─┤
Features → RF ───────┘
```

### 3. Strategy Ensemble
```
Momentum Strategy ─────┐
Mean Reversion Strategy┤→ Weighted Voting → Strategy Signal
Breakout Strategy ─────┤
Regime Switching ──────┘
```

### 4. Combined Signal
```
ML Prediction (60%) + Strategy Ensemble (40%) → Combined Signal
```

### 5. Risk Management
```
Signal → Position Sizing → Stop Loss/Take Profit → Execute
```

## 📚 API Usage

```python
from main import TradingEngine

# Initialize engine
engine = TradingEngine()

# Run analysis
results = engine.run(
    symbol="AAPL",
    mode="backtest",
    fetch_news=True,
    use_alternative_data=True,
    train_model=True,
)

# Access results
signals = results['signals']
backtest = results['backtest_result']

# Get latest signal
latest = signals.iloc[-1]
print(f"Signal: {latest['signal']}, Confidence: {latest['confidence']:.2%}")
```

## 🔬 Advanced Usage

### Walk-Forward Backtest
```python
from main import TradingEngine

engine = TradingEngine()
results = engine.run(
    symbol="AAPL",
    mode="walkforward",  # Use walk-forward mode
)

wf_result = results['walkforward_result']
print(f"Total Return: {wf_result.total_return:.2%}")
print(f"Overfitting Ratio: {wf_result.overfitting_ratio:.2%}")
```

### Reddit Sentiment Analysis
```python
from data.alternative import RedditSentimentFetcher

fetcher = RedditSentimentFetcher()
sentiment = fetcher.get_symbol_sentiment("TSLA", days_back=7)

print(f"Mentions: {sentiment['mention_count']}")
print(f"Sentiment: {sentiment['weighted_sentiment']:.2f}")
print(f"Bullish Ratio: {sentiment['bullish_ratio']:.2%}")
print(f"Trending: {sentiment['trending']}")
```

### SEC Insider Analysis
```python
from data.alternative import SECFetcher

fetcher = SECFetcher()
insider = fetcher.analyze_insider_activity("AAPL", days_back=90)

print(f"Buy Count: {insider['buy_count']}")
print(f"Sell Count: {insider['sell_count']}")
print(f"Net Value: ${insider['net_value']:,.0f}")
print(f"Insider Sentiment: {insider['insider_sentiment']:.2f}")
```

### Strategy Ensemble
```python
from strategies import StrategyEnsemble

ensemble = StrategyEnsemble()
signal = ensemble.generate_ensemble_signal(df)

print(f"Direction: {signal['direction']}")
print(f"Strength: {signal['strength']:.2f}")
print(f"Confidence: {signal['confidence']:.2%}")
```

## ⚠️ Disclaimer

This software is for **educational and research purposes only**. It is not financial advice. Trading stocks and other financial instruments involves risk of loss. Past performance does not guarantee future results.

Always:
- Do your own research
- Start with paper trading
- Never risk more than you can afford to lose
- Consult a financial advisor

## 📄 License

MIT License - See [LICENSE](LICENSE) for details.

## 🤝 Contributing

Contributions are welcome! Please open an issue on GitHub.

## 📧 Contact

For questions or suggestions, please open an issue on GitHub.
