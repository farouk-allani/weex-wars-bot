# WEEX AI Wars II — Trading Bot v8

Autonomous futures trading bot for the **WEEX AI Wars II** competition.

## Strategy (v8)

Multi-mode adaptive bot:

1. **High-conviction trend rider** — ADX + EMA stack + ≥2 edge confluence (1h + 4h)
2. **Mean reversion** — RSI extremes at Bollinger bands in ranging regimes
3. **SOL keep-alive** — VWAP bounce / BB touch at tiny size for activity rules
4. **Risk engine** — strength-scaled sizing, half-Kelly, 15% kill-switch, trailing stops

### Risk controls

- Max **1.5%** equity risk per trade (scaled by signal strength)
- Max **15%** drawdown kill-switch
- Max **2** open positions, **5–8x** leverage
- Loss cooldown after 3 consecutive losses
- Trailing stop activates only after **+1.2%** price move
- BTC/ETH correlation guard (no same-direction double stack)

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your WEEX API keys
python -m src.main
```

### Modes

- `trading.mode: paper` — simulated fills with **real** SL/TP tracking
- `trading.mode: live` — real orders + exchange brackets + software backup stops

## Configuration

Edit `config.yaml` for pairs, risk limits, and strategy knobs.

Key sections:

| Section | Purpose |
|--------|---------|
| `trading` | mode, symbols, 1h/4h timeframes |
| `risk` | DD kill, daily loss, trailing |
| `strategy.trend_follow` | ADX threshold, BB |
| `strategy.mean_reversion` | RSI/BB ranges |
| `strategy.keepalive` | SOL activity trades |
| `edges` | funding extremes |

## Architecture

```
src/
├── core/
│   ├── engine.py       # Main loop + HTF fetch + position management
│   ├── exchange.py     # WEEX via ccxt (paper + live SL/TP)
│   └── models.py       # Signal, Position, AccountState
├── strategies/
│   ├── composite.py    # Conviction + MR + keep-alive
│   └── edges.py        # Funding, volume, MTF, session, cascade
├── indicators/
│   └── technical.py    # RSI, MACD, BB, ATR, ADX, VWAP, regime
├── risk/
│   └── manager.py      # Sizing, cooldowns, chandelier trail
├── backtest/
│   └── engine.py       # Historical simulation
└── main.py
```

## Backtest

```bash
python run_backtest.py
# or multi-pair:
python run_multi_backtest.py
```

Data is fetched via public OHLCV (Binance proxy for history); live trading uses WEEX.

## Smoke test

```bash
python test_bot.py
```

## Author

**Farouk Allani** — [@farouk_allani](https://x.com/farouk_allani)
