# RESEARCH.md — everything measured, built, and decided (do NOT start from zero)

> Living document. Last full update: **2026-08-01**.
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
| 1h OHLCV entry families, re-judged under the FIXED exit geometry | run_entry_scan → run_entry_artifact_check | **DEAD** (2026-07-29) | 30 pre-declared cells (momentum/zscore/breakout/ema/rsi/vol-surge × both signs). 3 cleared "net>0 in full sample AND both OOS halves" — **all 3 died to beta + episode dedup**. See §2c. |

**The recurring lesson:** significant IC/t-stat + negative gross PnL = artifact
(bounce, beta, clustering). Demand *money*, in both halves, after maker cost,
then again on a fresh window.

### 2b. The exit geometry was setting an impossible bar (2026-07-29)

Every entry above was judged against a bar the *exits* had quietly raised. Measured
with `run_exit_lab.py` / `run_stop_width.py` / `run_exit_robust.py` (8 pairs, 120d 1h,
risk-based sizing, maker TP legs, stop wins intra-bar ties):

| | break-even directional accuracy |
|---|---|
| config as deployed 2026-07-16..29 | **62.1%** |
| after the three fixes below | **54.8%** |

62% accuracy on 8 liquid majors is not reachable. **The bot could not have been
profitable in that window no matter how good the entries were** — which reframes the
20-trade live result (30% WR, 0.63 payoff, −2.37%) as partly a geometry artifact
rather than pure model failure.

The three defects, in order of measured damage:

1. **Partial TP at the midpoint to TP** — banked half the position exactly where the
   right tail begins, paid an extra round trip to do it, and flipped the position into
   the post-partial regime whose 0.45% trail closed the remainder at +0.3..0.9%. Its
   cost **scales with edge**: +0.01/trade at zero alpha, +0.13/trade at 60% accuracy.
   That signature — harmless when there is nothing to destroy, expensive when there is
   — is exactly what a right-tail-truncating rule looks like. Now `partial_tp_enabled:
   false`.
2. **A 1.2x-ATR stop, when 7/8 pairs preferred ≥1.6x and the optimum was a plateau at
   2.0-2.5x.** Under risk-based sizing a tight stop also buys a *larger* position, so it
   paid more fees to die more often. Both OOS halves agreed. `min_stop_atr` is now
   **1.8 and widens rather than rejects** (rejecting would cost pace, a scored metric);
   the target scales with it so the model's intended R:R survives.
3. **`trail = max(chandelier, pct_trail)` took the tighter candidate**, so the fixed
   0.6% percent trail won essentially always and the ATR chandelier was dead code. No
   winner could ever give back more than 0.6% regardless of the pair's volatility —
   the mechanism behind a trailing stop that **never fired once in 20 live trades**.
   Now takes the looser; `trailing_stop_distance` widened to 0.02 as an outer bound.
   Breakeven also moved off 1R (surrendered a full R the instant a trade worked) to a
   configurable `be_trigger_r: 1.75`.

Result on the shipped combination: payoff 0.84 → 1.33, per-trade Sharpe −0.039 →
+0.013, fees/trade −9%, and `take_profit` exits doubled at the expense of capped
`be_stop` exits (883 → 183 per 2649 trades).

**What this does NOT do:** it does not create alpha. At 50% accuracy the shipped
geometry still loses −0.65/trade to costs. It lowers the bar from unreachable to
merely hard — the entry-edge search in §2 is still the binding problem, and the crude
1h z-score mean-reversion proxy tested here was *anti*-predictive (gross −0.073/trade
vs random's −0.001), consistent with everything else on the scoreboard.

### 2c. Entry families re-tested under the corrected bar — still nothing (2026-07-29)

Fixing the exits (§2b) lowered the required accuracy 62.1% → 54.8%, which legitimately
reopens entry hypotheses that were rejected as "edge < cost" against the old bar: the
cost side of that comparison moved. Asked once, with the grid pre-declared:
`run_entry_scan.py` — 30 cells (momentum 6/12/24/48, z-score fade 20/50, breakout
12/24/48, EMA cross 8/16, RSI extreme 10/14, volume surge 24/48, **both signs each**),
judged in net money through the shipped exit geometry, bar = net > 0 on the full sample
**and** in both OOS halves independently.

Three cells cleared it: `breakout_24_against` (+0.18/trade), `vol_surge_24_against`
(+0.45), `vol_surge_48_against` (+0.40). All three t < 1.3 — and `vol_surge_*_against`
is close kin to the liquidation-cascade FADE that died on 2026-07-20, so it inherited
that suspicion rather than a clean slate. `run_entry_artifact_check.py` applied the same
scrub that killed the cascade fade:

| cell | raw | beta-neutral | +episode dedup |
|---|---|---|---|
| breakout_24_against | +0.18 (t 0.48) | +0.92 (t 0.70) | +0.45 (t 0.58) |
| vol_surge_24_against | +0.45 (t 1.27) | **−0.20** (t −0.75) | **−0.18** (t −0.75) |
| vol_surge_48_against | +0.40 (t 1.06) | **−0.20** (t −0.74) | **−0.20** (t −0.64) |

**All three are artifacts.** The window fell −21.9% equal-weight over 120d, and although
the fade signals were ~55% long *by count*, their short legs harvested that drift — so
removing the basket move flips the sign. Episode dedup independently halves the sample
(957→450, 1067→453): one market-wide event was being counted once per correlated pair.

**The scrub is symmetric, and that matters.** The continuation cells looked *strongly*
negative (breakout_24_with −1.76 at t=−4.76; vol_surge_24_with −1.77 at t=−5.18) and
they collapse too — to t=−1.69 and **t=+0.21**. So there is no robust "1h momentum is
anti-predictive" claim either; the −21.9% drift was manufacturing loud t-stats in *both*
directions. Any future candidate must pass `run_entry_artifact_check.py` before it is
believed, including ones whose sign looks intuitively right.

Net position: the exits no longer set an impossible bar, and there is still no entry
edge in cheap 1h OHLCV features. That is consistent with §2 and narrows where to look —
the remaining untested directions (event-reaction latency, order-book microstructure,
on-chain flows) all require data this scan does not have.

### 2d. The position cap was manufacturing the exit leak (VPS audit 2026-08-01)

Full audit of the running stack, 2.9 days after the §2b exit fix shipped. Infra was
clean: 3 containers up 2d with **0 restarts**, host 59d, disk 32%, deployed commit ==
`main`, config byte-identical, AI healthy on `deepseek-v4-pro` (1 parse error in 67
cycles), compliance 34/34 orders logged, maker fills 12/14 rested (86%).

**Paper result: 992.78 from 1000 (−0.72%), maxDD 1.65%, 11 closed trades, fees 2.65
(20% of the loss).** The §2b fix has *not* reproduced live, and n is far too small to
say either way — but it is not yet visible in P&L, and the honest table is:

| | pre-fix live (n=20) | post-fix live (n=11) | §2b predicted |
|---|---|---|---|
| payoff | 0.74 | **0.48** | 0.84 → 1.33 |
| break-even WR | 57.5% | **67.6%** | 62.1% → **54.8%** |
| win rate | 30.0% | 36.4% | — |

Mechanically the fix did what it said: stop *frequency* fell (35% → 18% of exits) and
`take_profit` fired for the first time ever. But each surviving stop got bigger
(−5.57% → −7.46% of margin), so stop damage per trade barely moved (−1.24 → −1.02),
and in a low-vol chop the right tail never showed up. **Re-judge at n≥40.**

**The finding that is actionable now — `max_open_positions: 3` was binding, and the
model paid to work around it.** 19 of 67 decision cycles were blocked from entering
(15 by the cap). On 2026-07-31 00:02 the model closed the SOL short at −3.03 with the
stated reason *"Closing the short to free a position slot"*, then re-opened the same
SOL short 62 minutes later. It realised a loss and paid two extra round trips to end
up holding the position it started with, and only came out ahead by fill luck.

The prompt had explicitly authorised this — EXIT DISCIPLINE listed *"the margin slot
is needed for a clearly better opportunity"* as a valid close — and nothing anywhere
checked the *"clearly better"* part. So `ai_close` being the top leak (5 of 11 exits,
−5.82 = 44% of the period loss) is partly not model twitchiness at all: **scarce slots
plus a pace requirement equals churn**, and no prompt edit can fix that, because with
a full book closing is the only way the model can act on a new idea.

Fixed three ways, all shipped together:

1. **Counts stop being the working constraint.** `max_open_positions` 3 → 5,
   `max_same_side_positions` 2 → 3. Measured at this bot's real sizing (0.32x equity
   notional per leg, over the 11 live trades) and the measured median pair correlation
   of +0.64, `max_correlated_notional` bites on the **4th** same-side leg — 0.84x at 3
   legs, 1.09x at 4. The count of 3 was therefore strictly tighter than the budget and
   was doing all the binding, crudely. The budget *scales* a crowded leg instead of
   forcing an exit, and still lets a genuinely diversified book hold five.
2. **The swap is priced.** With the book full, a `close` proposed alongside a new entry
   is treated as a swap and only executes if the replacement's conviction beats the
   incumbent's *entry* conviction by `risk.swap_conviction_margin` (0.15). Entry
   conviction is now tracked per position and persisted across restarts. Closes with no
   replacement competing for their slot are untouched — those are ordinary thesis exits.
   Incumbents from before tracking fall back to `min_conviction`, so the guard never
   blocks on missing data.
3. **The prompt tells the truth about it.** The "margin slot" clause is gone, replaced
   by the rule as enforced, plus `entry_conviction` on each open position so the model
   can check a swap before proposing one.

Regression-tested in `test_bot.py` (`CAPACITY-MOTIVATED CLOSES ARE REFUSED`): the exact
observed churn is refused, a genuinely better idea still gets its slot, thesis closes
are untouched, resting maker orders count toward "full", and margin 0 disables it.

**Still producing nothing, both unchanged:**
- **Carry paper book: zero positions in 16 days** (`equity 1.0, book null`). The
  snapshot prints "TRADE 7d" but `paper_step()` is hardcoded `HOLD_DAYS = 3`
  (run_carry_weex.py:268) — the only cell that passed OOS — and the 3d gate needs
  >0.10% while it prints 0.039–0.053%. Consistent by design, but it is a permanently
  closed gate, not a running experiment: it cannot generate evidence either way.
- **Continuation: 12 days in, still failing its pre-declared bar.** $250k/180s deduped
  episodes t = 0.6–1.5 (needs |t|≥2), no monotonic dose-response (5m +0.038% →
  15m +0.059% → 30m +0.032%), net-maker negative at 3 of 4 horizons. $1000k cell n=5.

**Second-order observation, not yet acted on:** pace is bimodal — 11 trades in the
first 1.6 days, then **35 hours with zero entries** (24 consecutive declines while
holding one position). Same trade count, much worse *stability*, which is a scored
metric (§1). Also note the single entry taken under pace pressure justified itself as
*"acceptable as a range-fade to meet the trade minimum"* and is the book's only
winner — n=1, but a hint that `min_conviction` may be set too high.

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

**Audited 2026-07-27 (commit `005df32`) — it was NOT actually covering every order.**
Measured on the VPS: **7 of 18 linked orders had no ai-log file.** The misses all
predate `wars_log.py` (2026-07-21), so live coverage was 11/11, but the mechanism
was still able to fire:

- `link_order()` looked the decision up in `DecisionLog._recent` (process memory)
  and swallowed a miss with `except: pass`. A maker entry rests up to
  `entry_ttl_minutes` (45) and can fill **after a restart/deploy** → no ai-log, no
  message. Fixed: `_read_decision()` recovers it from the append-only log; failures
  count, set `last_ailog_error`, and log at ERROR.
- `explanation` looked for `reason`/`reasoning` but the model emits **`rationale`**.
  It never matched, so every log shipped the cycle's raw chain-of-thought truncated
  at exactly 1000 chars instead of the per-symbol reasoning. Fixed + boundary-aware
  truncation.
- `quantity`/`price` were raw floats (`0.5436481177706018`). The schema wants the
  parameters that **match the submitted request**, so orders are now rounded to
  venue increments *before* being built (`normalize_amount`/`normalize_price`);
  paper rounds identically to live.
- Monitoring: `data/compliance.json` written every cycle, alarms on a new gap,
  exposed on `/api/health`. `run_compliance_audit.py [--backfill]` reconciles
  order_link records against files on disk and rebuilds recoverable logs (never
  fabricates one). **Backfilled all 7 on 2026-07-27 → 18/18 orders have a log.**
- **Presence is the weak invariant; usability is the real one.** `wars_log.validate()`
  checks stage/model/`input.messages`/`input.market_context`/output
  symbol+side+quantity/explanation≤1000. **7 of the 18 logs have an EMPTY verbatim
  message array** — their decisions predate the logbook capturing messages, so they
  cannot be completed from anything on disk. They are classified
  `ai_logs_unrepairable_historical` (derived from the source decision, not an
  allowlist) and excluded from `compliant` / the alarm, so permanent history cannot
  desensitise us to a NEW emitter failure. They are preseason PAPER orders and will
  never be submitted. Current live state: `compliant: true`, 0 missing, 0 repairable.

**Related state-loading bug, same silent-corruption signature:**
`TradeResult.timestamp` defaulted to `utcnow()`, so trades restored from a state
file written before timestamps were persisted took **boot time**. 8 of 14 trades on
the VPS were stamped inside 0.078ms of each other, with durations contradicting the
decision log — and the book never held >3 positions at once, so those simultaneous
closes never happened. Undated trades now load as `None`.

**The stamps are microsecond-SEQUENTIAL, not identical** (`10:37:53.148001` →
`.148054`): the old loader ran the default factory once per trade inside a for-loop.
The first version of the quarantine grouped by exact `isoformat()` and silently
never fired — the audit that suggested equality had truncated to seconds for
display. Detection is now a run of ≥4 stamps within **50ms**, which no close path
can produce (even paper does P&L arithmetic, a state write, logging and console
output). Window kept far tighter than any real burst: wrongly discarding a genuine
close time is worse than leaving one artifact.

**Why it mattered:** in a live round a boot-time stamp falls *inside* the round
window, so 8 phantom trades would have read as 11/10 against the 10-trade minimum
and told the model to stop trading with 3 real trades on the board. `_trade_pace`
counts only dated trades. Both covered in `test_bot.py` (incl. real closes 30s apart
surviving).

**Method note:** every one of these was found by checking what the artifacts
actually contain, not what the code intends to write — and two of the fixes were
themselves wrong on the first pass and only caught by verifying against the live
VPS. Verify the fix, not just the bug.

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
