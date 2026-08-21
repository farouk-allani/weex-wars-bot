# WEEX Competition Rehearsal — Trading Bot v8.5

AI-driven futures rehearsal bot with hard portfolio-risk vetoes, restart-safe
venue protection, durable decision logs, and supervised live arming. WEEX has
published AI Wars II pre-registration and Trader Skill onboarding, but dates,
scoring details, event tasks, and final eligibility rules remain TBA. This repo
does not treat pre-registration as final competition authorization.

## Current AI evidence: no live GO

The live-config point-in-time replay recorded **168 trades, -31.61%, 27.4% win
rate, PF 0.64, and t=-2.69**. It does not demonstrate trading alpha. Keep the bot
in paper mode until a locked future sample passes the readiness gates and the
exact current event rules are verified. See [RESEARCH.md](RESEARCH.md#2h-the-live-configuration-measured-properly-2026-08-07-supersedes-2g).
Readiness counts only trades carrying the current `evaluation.policy_id`; bump that
ID whenever trading behavior changes instead of pooling incompatible history.

## Historical rules-engine benchmark

The table below is a legacy mechanical portfolio backtest. It is useful for cost
and risk plumbing, but it is **not** a backtest of the deployed DeepSeek decision
layer and must not be used as proof that the live AI bot is profitable. Use
`run_ai_replay.py` for point-in-time AI decision scoring.

### 90d portfolio (BTC + SOL, shared $10k) — legacy experiment

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
| **Rehearsal** (default) | `config.yaml` | AI decisions across the historical S1 8-symbol set; keepalive off |
| Pure edge BTC | `config.edge.yaml` | No KA, max edge research |

```bash
# Rehearsal paper
python -m src.main

# Pure edge config
# copy config.edge.yaml → config.yaml  (or pass path when you wire it)
```

## Commands

```bash
pip install -r requirements.txt
python test_bot.py
python paper_checklist.py
python check_ready.py --target live
python run_portfolio_backtest.py --days 90
python run_walk_forward.py --days 120 --apply-best
python -m src.main
```

## Dashboard (Command Center)

Polished local web UI for paper/live monitoring:

```bash
pip install -r requirements.txt
python run_dashboard.py
# open http://127.0.0.1:8787
```

In another terminal, run the bot:

```bash
python -m src.main
```

The dashboard auto-refreshes every **5s** from:

- `data/bot_state.json` — equity, trades, risk, strategy stats  
- `logs/trading.log` — live log stream  
- `config.yaml` — mode, pairs, risk limits  

If the bot hasn’t written state yet, the UI shows **demo metrics** so you can preview the layout.

## Deploy & CI/CD (push → live)

**Recommended:** Docker Compose on a VPS + GitHub Actions.

```bash
# local
git push origin main   # CI must pass before the exact tested commit deploys
```

Deploy notes live in local **`DEPLOY.md`** (gitignored — not pushed to GitHub).

Quick local prod-like stack:

```bash
cp .env.example .env
docker compose up -d --build
# bot + dashboard on :8787
```

## What’s new in v8.5

- Live bracket metadata and software trails survive restarts
- Resting live entries carry an atomic venue stop
- Live closes are accounted from the actual taker/maker result
- Official `UploadAiLog` delivery is durable and blocks new live entries on a gap
- Live readiness requires authenticated funds and a successful no-order AI-log
  allowlist probe; populated environment variables alone cannot turn the gate green
- Dashboard/deploy health fails on stale AI, compliance, or execution state
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
- Mechanical keepalive is disabled in AI mode
- Partial TP is disabled by measurement; venue SL and full target stay active

## Paper → live

1. `python paper_checklist.py`  
2. `python -m src.main` (mode: paper)  
3. `python probe_ai_log.py` (uploads the latest authentic decision with
   `orderId: null`; it cannot place an order)
4. `python check_ready.py --target live` while mode is still `paper`; this makes
   read-only private calls and requires positive USDT futures balance
5. Supervise one minimum-notional live rehearsal: venue SL visible and UploadAiLog successful
6. Return to paper if either check fails; never retry the trade to retry a log
7. Arm live only after the rehearsal and watch the first three round trips

## Author

**Farouk Allani** — [@farouk_allani](https://x.com/farouk_allani)
