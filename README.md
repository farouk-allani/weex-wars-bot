# WEEX AI Wars II — Trading Bot v8.4

Competition futures bot with portfolio risk, partial take-profit, adaptive weights, and a tuned mean-reversion core.

## Progress (90d portfolio, shared $10k)

| Stage | Net closed PnL | Max DD | Notes |
|-------|----------------|--------|--------|
| v8 baseline (sum pairs) | **-$125** | ~0.9% | Late trends + false breakouts |
| v8.3 MR focus | **-$29** | ~0.4% | Isolated BTC edge |
| v8.4 portfolio (KA tax) | **-$38** | 0.3% | Keep-alive over-trading |
| **v8.4 final** | **+$2.58** | **0.2%** | Sharpe **2.19**, PF **1.18** |

Final capital ~**$10,031** (partial scale-outs bank extra).  
**BTC** +$10.9 | **mean_reversion** +$8.2 (57% WR) | keep-alive still a small tax.

## Strategy stack

1. **Mean reversion (primary)** — z-score BB stretch + RSI/Stoch turn + volume filter  
2. **Partial TP at 1R** — bank 50%, stop → breakeven, trail rest  
3. **HTF 4h bias** — don’t fight the higher timeframe  
4. **SOL keep-alive** — micro size, max 3/week, only if book is quiet  
5. **Breakouts OFF**

## Risk & execution

- 1.2% base risk × strength × pair weight × strategy weight  
- 15% DD kill-switch, 2.5% daily loss, **time-based** 6h loss cooldown  
- Keep-alive losses **do not** trigger portfolio cooldown  
- State saved to `data/bot_state.json` (survives restarts)  
- Logs → `logs/trading.log`  
- Paper + live SL/TP brackets  

## Commands

```bash
pip install -r requirements.txt
cp .env.example .env          # add WEEX keys for live
python test_bot.py
python check_ready.py
python run_portfolio_backtest.py --days 90
python run_tune.py --days 90 --apply-best
python -m src.main            # paper by default
```

## Paper → live

1. `trading.mode: paper`  
2. Confirm every fill logs **Stop** + **TP** (+ partial TP)  
3. Fill `.env` keys  
4. After clean paper, set `mode: live`, leverage 3–5  

## Architecture

```
src/core/        engine v8.4, exchange, models
src/strategies/  composite (MR + KA), edges
src/risk/        adaptive sizing, partial TP, state
src/backtest/    single + portfolio shared capital
src/utils/       logger, state persistence
```

## Author

**Farouk Allani** — [@farouk_allani](https://x.com/farouk_allani)
