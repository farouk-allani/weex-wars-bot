# WEEX AI Wars II — Trading Bot

**Autonomous futures trading bot for the WEEX AI Wars II competition.**

## Strategy

Multi-strategy adaptive bot that switches between trend-following and mean-reversion based on market regime.

### Core Components

1. **Market Regime Detection** — ADX-based classification (trending vs ranging)
2. **Trend Following** — MACD + Bollinger Band breakouts in trending markets
3. **Mean Reversion** — RSI extremes + Bollinger Band bounces in ranging markets
4. **Risk Management** — Half-Kelly sizing, 2% per-trade risk, 20% max drawdown kill-switch
5. **Multi-Timeframe Analysis** — 1H for signals, 4H for trend confirmation, 15m for entries

### Risk Controls

- Max 2% capital risk per trade
- Max 20% drawdown from peak (hard stop)
- Max 8x leverage
- 5 trade cooldown after consecutive losses
- Dynamic stop-loss based on ATR
- Scale-in/out position management

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your WEEX API keys
python -m src.main
```

## Configuration

Edit `config.yaml` for strategy parameters, risk limits, and trading pairs.

## Architecture

```
src/
├── core/
│   ├── engine.py         # Main trading engine
│   ├── exchange.py       # WEEX API client
│   └── models.py         # Data models
├── strategies/
│   ├── base.py           # Strategy interface
│   ├── trend_follow.py   # Trend-following strategy
│   ├── mean_reversion.py # Mean-reversion strategy
│   └── composite.py      # Multi-strategy orchestrator
├── indicators/
│   ├── technical.py      # RSI, MACD, BB, ATR, VWAP
│   └── regime.py         # Market regime detection
├── risk/
│   ├── manager.py        # Risk management engine
│   ├── position_sizer.py # Kelly criterion sizing
│   └── drawdown.py       # Drawdown monitor
├── data/
│   ├── fetcher.py        # Market data fetcher
│   └── cache.py          # Local data cache
├── utils/
│   ├── logger.py         # Structured logging
│   └── helpers.py        # Utility functions
└── main.py               # Entry point
```

## Author

**Farouk Allani** — [@farouk_allani](https://x.com/farouk_allani)
