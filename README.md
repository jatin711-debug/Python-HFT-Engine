# Sophisticated ML Trading Engine

A professional-grade, ML-powered trading engine that generates **buy/sell/hold signals** with **long/short position support** and comprehensive **backtesting capabilities**.

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

### 📊 Backtesting (VectorBT)
- **100x faster** than event-driven backtesting
- Long and short position support
- Stop loss and take profit orders
- Comprehensive performance metrics
- Walk-forward analysis

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

# Live signals
python main.py --symbol MSFT --mode live --fetch-news

# Custom date range
python main.py --symbol GOOGL --start-date 2022-01-01 --end-date 2024-01-01
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
│   └── preprocessors/
│       └── cleaner.py         # Data cleaning
├── features/
│   ├── technical.py           # TA-Lib indicators
│   ├── statistical.py         # Rolling stats, z-scores
│   └── sentiment_features.py  # NLP sentiment analysis
├── models/
│   └── ml/
│       ├── gradient_boost.py  # XGBoost, LightGBM, CatBoost
│       └── ensemble.py        # Stacking ensemble
├── signals/
│   └── generator.py           # Buy/sell/hold signals
└── backtesting/
    └── engine.py              # VectorBT backtesting
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
Total Return:            156.32%
Annual Return:            48.21%
Sharpe Ratio:              2.15
Sortino Ratio:             3.42
Calmar Ratio:              1.89
Max Drawdown:            25.47%
Volatility (Ann.):       22.35%

📊 TRADE STATISTICS
----------------------------------------
Total Trades:                234
Winning Trades:              142
Losing Trades:                92
Win Rate:                  60.68%
Profit Factor:              2.31
Avg Win:                  $842.15
Avg Loss:                -$512.33

⚠️ RISK METRICS
----------------------------------------
VaR (95%):               -2.15%
CVaR (95%):              -3.42%

============================================================
```

## 🧠 How It Works

### 1. Data Pipeline
```
Market Data → Cleaning → Technical Features → Statistical Features → Sentiment
```

### 2. ML Ensemble
```
Features → XGBoost ─┐
Features → LightGBM ─┤→ Meta-Learner → Final Prediction
Features → CatBoost ─┤
Features → RF ───────┘
```

### 3. Signal Generation
```
ML Prediction (50%) + Sentiment (20%) + Technical (30%) → Combined Score → Signal
```

### 4. Risk Management
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

### Custom Model Training
```python
from models.ml import EnsembleModel, EnsembleConfig

config = EnsembleConfig(
    use_xgb=True,
    use_lgb=True,
    use_catboost=True,
    use_stacking=True,
)

model = EnsembleModel(config)
model.fit(X_train, y_train, X_val, y_val)

# Get predictions with confidence
predictions, confidence = model.predict_with_confidence(X_test)
```

### Walk-Forward Analysis
```python
from backtesting import BacktestEngine

engine = BacktestEngine()
results = engine.walk_forward_analysis(
    prices=price_data,
    signals=signals,
    train_period=252,  # 1 year
    test_period=63,    # 3 months
)
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

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📧 Contact

For questions or suggestions, please open an issue on GitHub.
