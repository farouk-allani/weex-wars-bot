# WEEX AI Wars II — Trading Bot v8.5

Competition futures bot: portfolio risk, partial take-profit runners, walk-forward mode selection, adaptive weights.

## Results snapshot

### 90d portfolio (BTC + SOL, shared $10k) — current config

| Metric | Value |
|--------|-------|
| Closed PnL | **+$2.13** |
| Final capital | **~$10,030** |
| Win rate | **53.6%** |
| Max DD | **0.2%** |
| Sharpe | **1.99** |
| Profit factor | **1.16** |
| mean_reversion | **+$6.53** |
| BTC | **+$10.21** |

### Journey

| Version | Result | Lesson |
|---------|--------|--------|
| v8 | -$125 | Late trends / false breakouts |
| v8.3 | -$29 | MR pocket on BTC |
| v8.4 | +$2.58 | Cap keep-alive tax |
| v8.5 WFO 120d | pick **comp_no_eth** | ETH drag; wick as bonus only |
| **v8.5.1 90d** | **+$2.13 / 54% WR** | BTC+SOL competition profile |

## Modes

| Profile | File | Use |
|---------|------|-----|
| **Competition** (default) | `config.yaml` | BTC+SOL, tiny KA activity |
| Pure edge BTC | `config.edge.yaml` | No KA, max edge research |

```bash
# Competition paper
python -m src.main

# Pure edge config
# copy config.edge.yaml → config.yaml  (or pass path when you wire it)
```

## Commands

```bash
pip install -r requirements.txt
python test_bot.py
python paper_checklist.py
python run_portfolio_backtest.py --days 90
python run_walk_forward.py --days 120 --apply-best
python -m src.main
```

## What’s new in v8.5

- **Walk-forward** mode comparison (`run_walk_forward.py`)
- **Wick quality** boosts size (not a hard gate)
- **Tighter runner trail** after partial TP
- **ETH dropped** (WFO + 90d evidence)
- **Disk cache** for OHLCV/funding (`data/cache/`)
- **Paper checklist** (`paper_checklist.py`)
- State save + file logging (from v8.4)

## Risk

- 1.2% risk × strength × pair × strategy weights  
- 15% kill-switch, 6h time-based loss cooldown  
- KA losses do **not** trigger cooldown  
- Partial TP 50% @ 1R → BE + tight trail  

## Paper → live

1. `python paper_checklist.py`  
2. `python -m src.main` (mode: paper)  
3. First fill must log Stop + TP + Partial  
4. Review `logs/trading.log` after 24h  
5. Live only when clean — leverage 3–5  

## Author

**Farouk Allani** — [@farouk_allani](https://x.com/farouk_allani)
