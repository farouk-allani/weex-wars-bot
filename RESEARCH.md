# RESEARCH.md — everything measured, built, and decided (do NOT start from zero)

> Living document. Last full update: **2026-07-21**.
> Method for ALL hypotheses: measure IC/event-study → out-of-sample split → cost bar →
> forward validation on fresh data → only then wire into the bot. Never trust one green cell.

## 1. The competition (AI Wars II — corrected 2026-07-21)

- **5 weekly rounds of LIVE futures trading**, dates TBA — rounds start ~late Aug 2026
  (DoraHacks: "submission" 2026-08-31 is a formality; ranking = live trading only).
  **Everything before that is preseason.** Registration + WEEX API key: DONE.
- **Scoring is MULTI-METRIC: profits + risk management + strategy stability.**
  NOT cumulative-PnL-only (that was AI Wars I). A max-variance "strike" plan is
  therefore WRONG for this event — steady-positive, tight-drawdown, consistent
  behavior is the rank-optimal profile.
- Team AI compliance: official skill (github.com/weex-labs/weex-agent-skills-ai-wars),
  and **every AI-driven live order needs an ai-log.json uploaded** (schema: stage /
  exact provider model id / complete verbatim prompt messages / output matching the
  final order params / explanation ≤1000 chars). DQ for wash trading & manipulation.
- Losing team still shares 30% of each round's pool.

## 2. Edge scoreboard (what was tested and what happened)

| Hypothesis | Tool | Verdict | Key numbers |
|---|---|---|---|
| 1h/daily TA, positioning, macro | run_signal_scan / run_macro_validate / run_ai_replay | **DEAD** (2026-07-14) | 1,266 replayed AI trades: edge < cost in every arm; only sp500/vix 1d shock survives OOS (0.126%/trade < 0.22% market cost). The −0.32 hourly macro IC was ONE Fed event counted 5000× — flips sign OOS. |
| Cross-sectional momentum | run_rv_scan | **DEAD** | all variants reversal-signed or unprofitable |
| vol_14d cross-sectional | run_rv_scan | real IC, unmonetizable | IC −0.06..−0.09 both halves, L/S sim doesn't pay |
| **Funding-carry RV** | run_rv_scan → **run_carry_weex** | **ALIVE, small** | On WEEX's own funding (360d): only the **3d-hold GATED** variant survives: +4.29%/360d, OOS half +4.91%, Sharpe 1.13, maxDD 5.6% — but carry leg is just +0.87%/yr; price leg (unproven) carries it. Forward paper book LIVE on VPS. |
| Binance→WEEX lead-lag | run_leadlag_record/analyze | **DEAD** (2026-07-15) | loud IC (t to −8) but **negative gross PnL** = bid-ask bounce artifact. WEEX swaps have NO quote websocket (trades only). |
| Liquidation-cascade FADE (OI proxy) | run_liq_scan | misleading | 15m buy-flush looked great on 8 pairs (n=9, t=2.6) — **failed replication** on top-40 (t=0.3). Universe was contaminated by tokenized-equity perps; time-clustered events inflate t. |
| Liquidation-cascade FADE (real forced orders) | run_liq_forward | **DEAD** (2026-07-20) | 89.5h, 15,393 real forced orders: raw fade "profit" (60m +0.18%, t 3.3) is **pure market beta** — beta-neutral deduped: −0.077%, t=−2.7 at $250k. Bigger flush → CONTINUES, not reverts. |
| **Cascade CONTINUATION (trade WITH the flush)** | run_liq_forward `--since 2026-07-20 --direction with` | **PRE-DECLARED, judging on fresh data only** | Registered 2026-07-20 after the fade's −2.7t. Primary cell: $250k/180s, 60m, beta-neutral deduped episodes. Auto-evaluates every 8h on the VPS → `data/continuation_eval.txt`. NO backfitting to pre-07-20 data. |

**The recurring lesson:** significant IC/t-stat + negative gross PnL = artifact
(bounce, beta, clustering). Demand *money*, in both halves, after maker cost,
then again on a fresh window.

## 3. Live systems (VPS 45.88.191.129, docker compose: bot / dashboard / collectors)

- **bot** — paper trading, AI (DeepSeek) hourly decisions, maker-entry execution
  (post-only, chase ≤0.5 ATR, abandon on RR degradation), risk engine, keepalive on.
  State: `data/bot_state.json` (volume `bot-data`, survives deploys).
- **collectors** — 24/7: real Binance forced-order recorder (`run_liq_record.py`,
  24 crypto perps + 1s mids for the 8 pairs → `data/liq_forward/*.jsonl`, ~5k
  orders/day) + every 8h: carry paper step + funding snapshot + continuation eval.
- **dashboard** :8787 — Edge Lab panel (`/api/edges`): carry gate verdict + basket,
  paper equity, liquidation collector stats.
- Deploy: `git push origin main` → CI → SSH deploy → `docker compose up -d`.
  Config changes need a bot restart (config is a read-only mount).

## 4. Compliance layer (built 2026-07-21, preseason-tested)

- `src/ai/logbook.py` records per decision: exact model id (provider-resolved),
  verbatim message array, full context, raw response, reasoning; `link_order()`
  binds OrderId→decision and **emits a WEEX-schema ai-log file** per order to
  `data/ai_logs/` via `src/ai/wars_log.py` (schema-tested).
- Remaining before round 1: hook the official skill's upload flow for LIVE orders
  (`--ai-log @file.json`), and a supervised tiny-size LIVE maker-path test.

## 5. Machine/env gotchas (cost hours — don't rediscover)

- **Windows dev box:** aiodns/c-ares can't read DNS → all ccxt.pro/aiohttp needs
  `TCPConnector(resolver=ThreadedResolver())`. Python 3.13 strict SSL rejects
  Binance fapi + WEEX cert chains → sync ccxt needs `truststore.inject_into_ssl()`,
  aiohttp needs `truststore.SSLContext` (see `make_exchange` in run_liq_record.py).
  Linux/Docker: neither issue exists (truststore import is optional-guarded).
- WEEX API: funding history max 365d back, **7-day windows per call**; klines
  intervals limited (no 8h); spot host cert differs from contract host.
- Binance: OI history keeps ~30 days only (5m gran); all-market liquidation
  subscription broken in current ccxt (`[]` → TypeError) — subscribe per-symbol.
  ccxt leaves `amount` empty on forced orders — size is in `info.o.q`.
- "Top by volume" perp universes are contaminated with tokenized equity/ETF/
  commodity perps (MSTR, SOXL, QQQ, XAU, AMD, TSM…) — market-hour gaps fake
  crypto signals. Hand-verify crypto universes.

## 6. Preseason roadmap (~41 days to rounds)

1. **Continuation verdict**: auto-eval runs every 8h; first meaningful read ~1 week
   of fresh data (≈2026-07-27), then weekly windows. Bar: net-positive at maker,
   same sign across ≥2 independent weeks, |t|≥2 on deduped beta-neutral episodes.
2. **Carry**: forward paper record accumulates; judge after 2-3 weeks of snapshots.
3. **Live rehearsal**: supervised real-money maker test at minimum size; then ai-log
   upload integration against the official skill.
4. **Dress rehearsals**: weekly self-scored rounds (profit / maxDD / stability),
   tune AI gating to multi-metric, **code freeze before round 1**.
5. Watch DoraHacks for the official event link, round durations, metric weights,
   and task list; re-read rules for keepalive/wash-trading implications.

## 7. Where the numbers live

- Memory (Claude sessions): `~/.claude/projects/c--Users-DELL-Desktop-weexbwarsbot/memory/`
- Forward records: `data/carry_forward.jsonl`, `data/carry_paper_*.json(l)`,
  `data/liq_forward/*.jsonl`, `data/continuation_eval.txt`, `data/ai_logs/` (all on the
  VPS `bot-data` volume; `data/` is gitignored — large/regenerable).
- AI decision log: `logs/ai_decisions.jsonl` (bot volume).
