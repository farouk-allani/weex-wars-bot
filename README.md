# WEEX AI Wars II — Trading Bot v8.3

Competition-oriented futures bot with fixed execution plumbing, multi-timeframe bias, and a tuned mean-reversion core.

## What v8.3 learned (90d multi-pair tune)

| Version | Net (BTC+ETH+SOL) | Notes |
|--------|-------------------|--------|
| v8 baseline | **-$125** | Late trend entries + false breakouts |
| v8.2 | **-$102** | Still breakout-heavy |
| **v8.3 mr_only_htf** | **-$29** | Best; max DD ~0.4% |
| BTC mean-reversion alone | **+$11** (67% WR) | Strongest pocket |

Honest takeaway: last 90 days were hostile to momentum. The bot is now **survival-first** with a real MR edge on BTC, ETH sized down, breakouts off, and 4h bias so we do not fight the higher timeframe.

## Strategy stack

1. **Mean reversion (primary)** — RSI + Bollinger + Stoch turn, ADX capped, mid-band targets  
2. **SOL trend pullback** — deep EMA pullback only on SOL (`enabled_pairs`)  
3. **SOL keep-alive** — tiny VWAP/BB size for activity rules  
4. **Breakouts** — **disabled** (false breaks bled equity)

## Risk

- 1.2% base risk × signal strength (keep-alive ~0.10 → micro size)  
- 15% drawdown kill-switch, 2.5% daily loss limit  
- Max 2 positions, half-Kelly, correlation guard BTC/ETH  
- Trailing only after +1.2% price move  
- Software SL/TP always stored (paper + live bracket cache)

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# add WEEX_API_KEY / WEEX_API_SECRET / WEEX_API_PASSPHRASE
python test_bot.py
python check_ready.py
python -m src.main          # paper mode
```

### Backtest / tune

```bash
python run_tune.py --days 90 --apply-best
python run_multi_backtest.py
```

## Paper → live checklist

1. `trading.mode: paper` in `config.yaml`  
2. `python test_bot.py` + `python check_ready.py`  
3. Run `python -m src.main` — every fill must log **Stop** and **TP**  
4. Fill `.env` keys when going live  
5. Set `trading.mode: live` only after 24h paper looks clean  
6. Start leverage **3–5** (max 8)

## Architecture

```
src/core/       engine, exchange (SL/TP), models
src/strategies/ composite v8.3, edges (funding/MTF/session)
src/risk/       strength sizing, trail, kill-switch
src/backtest/   HTF resample, strategy PnL breakdown
src/indicators/ RSI, MACD, BB, ATR, ADX, VWAP, regime
```

## Author

**Farouk Allani** — [@farouk_allani](https://x.com/farouk_allani)
