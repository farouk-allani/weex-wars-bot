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

### 2e. The paper simulator was flattering us in two ways (2026-08-03)

Triggered by an outside code review. Three real defects, verified in the source before
being believed — and the two that matter both bias **in the same direction**, so the
paper book has been reporting a book slightly better than the one live would have run.

**1. Funding was never charged.** `fetch_funding_rate` existed and fed the model as
*context*; the paper ledger never debited it. A paper position was free to hold and a
live one is not. This is not a rounding error in the research sense — it silently
invalidated the **carry** arm above (§2, "one surviving candidate"): a funding-carry
book measured in a simulator that does not charge funding is measuring nothing. Now
settled to `balance` at 00/08/16 UTC, once per boundary, anchored per position and
persisted so a restart cannot skip the settlements it slept through.

**2. A touch counted as a full fill.** `check_entry_fill` filled a resting order the
moment `last` reached the limit. At the touch we are at the *back* of the queue, and
most touches reverse without clearing the book down to us — so the old rule granted
free entries at precisely the local reversals a real queue never gives you, which are
also the best-looking entries in any sample. **The Aug 1 audit's "maker fills 12/14
rested (86%)" is an artifact of this rule and should not be quoted again.** Now
requires trade-through by `execution.paper_fill_through_ticks` (default 1). That is
the pessimistic bracket, not a queue model; truth is between the two, and the honest
choice is the bracket that cannot flatter us.

**3. Resting orders were invisible to the correlation budgets.** `correlation_scale`
read only `account.positions`, so N same-side maker orders were each sized as if the
others did not exist, then filled together in the one correlated move the budgets
exist to survive. With `max_open_positions` now 5 (§2d) that was up to five unbudgeted
legs. Fixed at the source (both budgets) *and* in the AI's context, which was showing
the model free capacity the risk engine was about to refuse.

Also fixed: the container healthcheck treated a **missing** `ai_health.json` as
healthy (`d is None or ...`) — the exact hole its own comment claimed to close, and
the same failure mode as the 16h silent outage. The engine now publishes the file at
startup (without stamping a fake `last_success`), so absence can only mean the writer
is dead, and the check demands it.

**Consequence for the record: paper results from before this date are not comparable
to results after it, and the −0.72% / 11 trades in §2d is the optimistic reading.**
The next audit's numbers should be expected to look *worse*, and that is the fix
working. Re-measure the maker fill rate before drawing any conclusion from it.

### 2f. PRE-DECLARED: does the AI decision layer have a general edge? (registered 2026-08-07, before the run)

**Why this exists.** The live paper book is up **+2.13% ($999.94 → $1021.20) on 18
trades over 8.6 days** — 50% win rate, payoff 1.63, maxDD 1.65%, net directional beta
0.02x. It reads well and it means nothing yet:

- Per-trade sd is $6.09, so the 95% CI on the +$18.12 is **−$32.5 to +$68.8**.
  **t = 0.70.** At this effect size, t=2 needs ~147 trades.
- **+$15 of the +$18 is ADA** — the one pair that trended in the window (+20.7% while
  BTC/ETH/SOL moved ±0.2%). Drop the top 3 trades and the book is **−$16.36**. Drop
  the two ADA take-profits and the break-even win rate rises to 52.4% against an
  observed 50%, i.e. underwater.
- **13 of the 18 trades closed before the §2e sim fix** (deployed Aug 3 14:13 UTC).
  The trustworthy sample is n=5.

At the observed 2.08 trades/day, waiting for n≈150 live is ~20-25 days and the rounds
are late August. `run_ai_replay.py` is the way to get n without spending the runway:
no lookahead (context is `candles[:i]`, outcome resolved on `candles[i+1:]`), fees and
slippage charged at config rates, stop assumed on ambiguous bars, fixed notional per
decision so it measures the **brain, not the book**.

**Scope, fixed before the run.** Primary: `--days 90 --every 8` (all 8 pairs, intel +
macro + oscillators on, default model). Out-of-sample: the same command with
`--offset 90`, run **once**, after the primary is scored, with **zero** prompt or
config changes in between.

**The bar. All of these are pre-declared; none are negotiable after seeing the output.**

| # | Test | Pass condition |
|---|------|----------------|
| 1 | Sample | ≥120 scored trades in the primary window (else the run is void — resample denser, don't judge it) |
| 2 | **Significance** | **t ≥ 2.0** on per-trade PnL (mean / (sd/√n)). Positive PnL alone is *not* a pass — positive is free at small n |
| 3 | Generality (a) | Drop the single best-contributing symbol → total PnL still positive **and** t ≥ 1.5 |
| 4 | Generality (b) | ≥5 of 8 symbols individually net-positive |
| 5 | Cost survival | Profit factor ≥ 1.25 (a PF near 1.0 dies to any live execution slip) |
| 6 | Pace | ≥10 trades per 14d (the round minimum; the script prints this) |
| 7 | **OOS** | Held-out window: **same sign**, and t ≥ 1.0 |

**Pre-declared actions, so the result cannot be rationalised afterwards:**

- **Fails #2** → the decision layer has no demonstrated edge. We do **not** tune the
  prompt until the window turns green; that is overfitting in English, and the script
  says so itself. The round strategy instead becomes: minimise drawdown, clear the
  10-trade minimum, and compete on the risk/stability metrics rather than on profit.
- **Passes #2, fails #3/#4** → the edge is symbol-specific, not general. Restrict the
  traded universe to the symbols that carry it; do not claim a general edge.
- **Passes primary, fails #7** → unproven. Treated exactly as a #2 failure.
- **Shorts net-negative at t ≤ −2** → declare shorts a structural leak and gate them
  off in config. (Live currently shows longs +$20.88/8, shorts −$2.76/10 — far too
  thin to act on, which is the point of measuring it at n in the hundreds.)

**Two limits of this rig, stated before the result so they can't be produced later as
an excuse:**

1. **The model may carry training-data knowledge of these windows.** That makes a PASS
   *suggestive, not conclusive* — but it makes a FAIL damning: a model that cannot
   make money on price action it may have already seen will not make it live. The
   asymmetry is the reason this test is worth running despite the contamination.
2. **`resolve()` models only SL / TP / 72h-timeout.** It does not simulate the live
   book's trailing and break-even stops (§2b), maker entry queue (§2e), position cap
   (§2d) or correlation budgets. It scores *decision quality* under a simplified exit,
   which is the intended isolation — but it is not a forecast of live book PnL.

**The live paper book keeps running untouched during all of this — as the control, not
the experiment. Nothing gets tuned on replay output and then declared "confirmed" by
the live run.**

### 2g. RESULT: the bar was missed, and not narrowly (2026-08-07)

> **CORRECTION, same day, before acting on any of this.** This run scored the wrong
> configuration. `run_ai_replay.py` defaulted `--no-osc` to OFF — i.e. oscillators
> **ON** — while `config.yaml:51` sets `include_oscillators: false` and the live engine
> reads it (`engine.py:93`). So this measured a brain the bot does not run, and
> specifically the one arm already on record for triggering the memorised *"price at
> upper Bollinger → short to VWAP"* reflex that had the model reciting a fade on ~80%
> of trades. Corroborating: the earlier 1,266-trade measurement of the live arm gave a
> ~36% win rate against a ~38.5% random baseline (≈random); this run gave 24.1%
> (well below). That gap is consistent with the reflex being live in this arm.
> **The numbers below stand as a measurement of the oscillators-ON arm and nothing
> more.** Root cause fixed: the replay now defaults to the config value and prints a
> loud warning when a flag overrides it. The live-configuration arm is re-running; its
> result lands in §2h and supersedes this section's verdict.
>
> What survives the correction regardless of the re-run: the three harness checks
> (ambiguous-bar, fee overcharge, conviction), the `--offset` truncation bug, and the
> observation that the live book's break-even stops are doing the work.

`--days 90 --every 6`, window 2026-05-06 → 2026-08-04, 360 decision points, all 8
pairs. Scored by `run_replay_score.py` (frozen at d2724bc, before the data existed).

| Clause | Measured | |
|---|---|---|
| #1 n ≥ 120 | 240 | PASS |
| **#2 t ≥ 2.0** | **−3.66** | **FAIL** |
| #3 drop best symbol (ADA) → PnL > 0 | −$552.13 | FAIL |
| #3 drop best symbol → t ≥ 1.5 | −4.23 | FAIL |
| #4 ≥5/8 pairs net-positive | **1/8** | FAIL |
| #5 profit factor ≥ 1.25 | 0.59 | FAIL |
| #6 pace ≥ 10/14d | 37.3 | PASS |

**−53.78% on 240 trades. Win rate 27.1%. Both directions negative (long −$332 at
t=−3.37, short −$206 at t=−1.88). Seven of eight pairs negative; the eighth is ADA at
+$14.30, t=+0.21 — noise.**

**It is worse than a coin flip.** Measuring the true geometry from `r_multiple` rather
than from realised PnL (the realised-outcome estimator is conditional and biased —
checked because the first pass used it): median stop = 1.119R (the 0.119 is the fee and
slippage load, which is the estimator's sanity check), median target = 1.755R, so the
take-profit sits **1.57× the stop distance**. A driftless random entry therefore hits
its target first **38.9%** of the time. The model managed **24.1%** (55 of 228
resolved). **z = −4.59, p < 0.0001. All 8 pairs sit below the random-entry win rate.**
Correcting the estimator made the result worse, not better.

**Three ways it could have been the harness, all checked and all rejected:**

1. *The pessimistic ambiguous-bar rule.* Only **3% of stops resolved on the first bar**
   (median 11 bars; targets median 22). If the assume-the-stop rule were manufacturing
   this, stops would fire on bar 1 constantly. They do not.
2. *Taker fees on both legs.* Replay charges `commission_rate` 0.0006 round-trip while
   live pays maker 0.0002 on entry. Maximum possible correction is ~**0.02R/trade
   against a −0.34R average** — it recovers ~6% of the loss and changes no verdict.
3. *Conviction carrying the signal.* It does not. The 0.50–0.65 bucket holds 169 of
   240 trades at **t = −4.19**. The only positive bucket is 0.80–1.01: +$59.90 on
   **n=11, t=+0.97**. That is not a filter, it is eleven trades.

**Why the live book still shows +2.13%.** Not a contradiction — the two measure
different things, and the gap is itself the finding. Live n=18 against replay n=240,
and more importantly the live book runs the §2b exit machinery: **5 of its 18 trades
exited at `be_stop` for a combined +$0.61**, i.e. scratched. In replay those same
trades take the full −1R. Live avg loss is −$3.18; replay's is −$7.58. **The exits are
doing the work.** The entry signal is negative and the break-even stop is quietly
converting losers into scratches. That is a system that loses slowly, not one that
wins — and it explains why the live equity curve looks calm (maxDD 1.65%) while the
underlying decisions are actively harmful. The +$15-of-+$18 ADA concentration is the
same story: ADA is also the only non-negative pair in replay, during a 90d window in
which it was the only pair that trended.

**PRE-DECLARED CONSEQUENCE, now in force.** The decision layer has no demonstrated
edge — it has a demonstrated *anti*-edge. Per §2f: **we do not tune the prompt until
this window turns green.** Round strategy shifts to drawdown-minimisation and clearing
the trade minimum, competing on the risk and stability metrics that the official
DoraHacks page confirms are scored ("rankings consider profits, risk management, and
strategy stability").

**Registered as a hypothesis, NOT acted on: does fading it work?** A signal reliably
worse than random contains information with the sign flipped. The arithmetic is
suggestive — but it was generated *from this window*, so testing it here is
backfitting, and it does not survive naive inversion for free: a faded trade pays the
same fees, and mirroring the levels inverts the reward:risk to 1/1.57 = 0.64, which
raises the break-even hit rate to 61.5%. It needs its own pre-declared bar, its own
`--fade` implementation, and a window that did not generate the hypothesis. See §2h.
**Do not enable anything resembling this in config before that test exists.**

**Bug found while setting up the held-out arm:** `--offset` was ignored by the candle,
funding and sentiment fetches (only `fetch_macro_history` accounted for it). At
`--days 90 --offset 90` the fetch returned 105 days, `end` landed at bar 288, `start`
clamped to `LOOKBACK`, and the run would have scored **15 decision points over ~4 days
while printing them as the requested 90-day window** — silently. Fixed, plus a hard
guard that exits if the realised point count falls below 90% of what was asked for.
**Any previous `--offset` result in this repo should be treated as void.**

### 2h. THE LIVE CONFIGURATION, MEASURED PROPERLY (2026-08-07, supersedes §2g)

Same window (2026-05-06 → 2026-08-04), 360 points, `include_oscillators` now following
config (false) — i.e. the brain the bot actually runs. **168 trades. −31.61%. t = −2.69.
Win rate 27.4%. PF 0.64. 7 of 8 pairs negative. Every clause of the §2f bar failed
except sample size and pace.** The verdict is unchanged by the §2g correction.

**Two claims in §2g were overstated and are withdrawn.** Both came from estimating the
reward:risk ratio off *realised* outcomes, which is conditional on which level was hit
and biased toward trades with nearer targets. `to_signal` lets the model choose its own
R:R per trade (floor `min_rr` 1.35), so the ratio had to be measured unconditionally —
from the declared `stop_loss`/`take_profit` in `logs/ai_replay.jsonl` across **all 443
intended entries**, winners and losers alike:

| | conditional (wrong) | unconditional (correct) |
|---|---|---|
| declared R:R | 1.59 | **2.26 mean / 2.00 median** |
| random hit rate | 38.7% | **32.1%** |
| live-config z | −3.44 | **−1.85 (p = 0.064)** |

**The live decision layer's entries are statistically INDISTINGUISHABLE FROM RANDOM.**
Not "worse than a coin flip" — that claim was an artifact of the biased estimator, it
was wrong twice, and it does not survive correct measurement. (The oscillators-ON arm
*does* stay significantly sub-random at z = −2.58, p = 0.010, which is independent
corroboration that the memorised Bollinger-fade reflex is real and costly. Turning
oscillators off changed *what* it traded — shorts fell from 53% to 36% of entries — but
not *how well*: −0.306R vs −0.337R, t = +0.23, not significant.)

**Why it still loses decisively, which is the finding that matters.** Per-trade
economics at the measured geometry:

- a **random** entry: `0.321 × 2.26 − 0.679 × 1 − 0.119 fees = −0.073R`
- the **model**: **−0.306R** (t = −3.00 vs zero)
- model minus random: −0.234R, t = −2.29 — marginally worse, but the headline is that
  **random already loses.** The geometry plus costs is a −0.073R/trade tax, and the
  model adds no directional information to pay it with.

So the correct diagnosis is not "the AI is actively harmful." It is: **the AI has no
measurable directional edge, and a coin flip cannot fund this cost structure.** That is
a different and more tractable problem than an inverted signal.

**The fade hypothesis registered in §2g is weakened, not strengthened.** Mirrored-level
arithmetic gives `0.747 × 1 − 0.253 × 2.26 − 0.119 = +0.056R/trade` — thin, and resting
entirely on a hit-rate deficit significant only at **p = 0.064**, derived from the same
window that generated the idea. **Not actionable.** It would need the deficit to
replicate on a held-out window first, and §2g's larger apparent edge was a product of
the same estimator error.

**Standing consequence (unchanged from §2f).** No prompt tuning to make a window turn
green. Round strategy competes on the risk and stability metrics that the DoraHacks
page confirms are scored, alongside profit. The live paper book's calm equity curve is
its exit machinery scratching losers, not entries earning.

**Method note worth keeping:** the conditional-estimator trap bit twice in one session
and in the same direction both times — toward a more dramatic conclusion. Any future
"better/worse than random" claim in this repo must be computed from **declared** levels
over **all** intended entries, never from realised winners.

### 2i. PRE-DECLARED: held-out replication + the one filter worth testing (registered 2026-08-08, before the run)

`--days 90 --every 6 --offset 90` — window ≈2026-02-05 → 2026-05-06, live config
(oscillators follow config), model and prompt **unchanged** since §2h. This is the
first `--offset` run since the truncation bug was fixed, so it is also the first
genuinely out-of-sample window this repo has produced.

**Question 1 — does "no edge" replicate?** §2h could still be one bad quarter.
- **Overturned if** the held-out window returns PnL t ≥ +2.0. Then the §2f consequence
  lifts and the whole no-edge reading is wrong.
- **Confirmed if** t < 2.0 *and* the hit rate again sits within noise of its
  geometry-implied random baseline (computed from **declared** levels over **all**
  intended entries — never from realised winners, per §2h).

**Question 2 — the trend-alignment filter.** §2h's live-config run split as
**WITH 4h trend −0.20R (n=126)** vs **AGAINST/flat −0.63R (n=42)**. That is the only
split in the data that looks like it separates anything. It was found *in* that window,
so it is a hypothesis, not a result. Pre-declared bar, on the held-out window only:

| # | Test | Pass condition |
|---|------|----------------|
| 1 | Separation is real | (WITH − AGAINST) avg R, **t ≥ 2.0** |
| 2 | Direction holds | WITH-trend avg R > AGAINST avg R (sign must match §2h) |
| 3 | Worth acting on | WITH-trend avg R > **−0.073R** (the random-entry bleed rate; a filter that still loses more than a coin flip is not a filter) |
| 4 | Leaves a viable book | WITH-trend pace ≥ 10 trades/14d, so the round minimum is still reachable |

**Pre-declared honesty clause: passing this does NOT produce an edge.** The best
available outcome is *bleeding more slowly*. Clause 3 caps the claim deliberately — if
WITH-trend comes back at, say, −0.05R, that is still a losing book, just a cheaper one.
Nothing here may be described as an edge without clearing the §2f bar, which this test
does not attempt.

**Failing it changes nothing** — the §2f consequence is already in force. It only
removes one candidate.

> **STATUS 2026-08-08: THIS TEST HAS NOT RUN. The first attempt is VOID.** All 360
> calls returned `402 Insufficient Balance` — the DeepSeek account is out of credit.
> Zero decisions were made. **No conclusion of any kind may be drawn from that run**,
> and in particular the trend filter remains untested.
>
> The failure mode is the dangerous part and is now fixed. `AITrader.decide()` fails
> closed, returning no decisions, so 360 consecutive API failures rendered as *"all
> hold"* and the script printed a confident verdict: *"The model took ZERO trades
> across the whole replay. It cannot reach the 10-trade minimum, so it would be
> disqualified regardless of how good its reasoning reads."* That sentence describes a
> model that was never asked a single question. It is the same silent-failure class as
> the 16h dead-model outage in §2/[[ai-mode-silent-failure-modes]]: a fail-closed path
> that is indistinguishable from a decision.
>
> `run_ai_replay` now raises on `trader.last_error` instead of scoring the point, and
> aborts the whole run with exit code 2 if **any** point failed — a run that lost a
> share of its decision points is not a smaller sample but a biased one, since API
> failures are not randomly distributed in time. Verified against the live 402.
>
> **Blocked on: topping up the DeepSeek balance.** Re-run unchanged when credit is
> available; the bar above stands as written and is not to be revised in the meantime.

### 2j. The arithmetic that needs no test: with no edge, trade count IS the strategy

This does not require a held-out window because it is not an empirical claim about
markets — it follows from §2h plus the round rules.

With entries indistinguishable from random, expected PnL is `−n × bleed`, and the
measured bleed floor is **−0.073R/trade** even for a perfect coin flip (geometry +
fees), with the model realising −0.306R. Every additional trade has negative expectancy
and adds variance. The round requires a **10-trade minimum**. Therefore:

> **The optimal trade count is the smallest number that clears the minimum.**

**CORRECTED before acting on it — the first draft of this section had a unit error.**
`risk.min_trades: 10` is per **weekly** round, so the floor is **10 per 7d**, not per
14d. `run_ai_replay` reported "Trades / 14d" and compared it against 10, flattering
pace by exactly 2× — it would have called a disqualifying book compliant. §2f bar #6
inherited the same error. Both fixed; no recorded verdict changes, because every run
so far cleared the stricter reading anyway.

With the units right, the headroom is **modest, not large**:

| | pace | vs 10/round floor |
|---|---|---|
| replay, live config | 168/90d = **13.1/round** | +31% |
| live paper book | 2.08/day = **14.6/round** | +46% |

So cutting to a safe ~11-12/round removes roughly **3 trades/week**, not the majority
of them. At `max_risk_per_trade` 0.012 on $1,000, 1R ≈ $12, so the saving is bounded by:

- at the observed −0.306R/trade: ~**$11/week**, ≈ **5.5% of the account** over 5 rounds
- at the random-entry floor of −0.073R: ~**$2.6/week**, ≈ **1.3%**

Real, worth taking, and nowhere near a fix. **The honest framing is that trade-count
reduction is a cheap partial mitigation, not a strategy.** Cutting harder is not
available: the 10-trade floor is a hard disqualification line and a quiet week needs
buffer above it.

The larger controllable term is inside the bleed itself. Of the −0.073R random-entry
floor, **0.119R is fees and slippage measured in stop units** — so widening stops
mechanically shrinks it (same cost over a larger denominator). `min_stop_atr` is
currently 1.8 while `run_stop_width.py` found the optimum on a **2.0–2.5 plateau**
across 8 pairs × 120d with both OOS halves agreeing. Moving to ~2.2 is the one config
change already backed by prior out-of-sample measurement rather than by this window.
**Not applied yet** — it interacts with the §2i trend-filter test, and both should be
decided together once the held-out window reports.

Note this inverts the §2 "round pace" note, which worried about trading *too little*.
That was written when the edge sign was unknown. It is now measured, and the pace
problem runs the other way — though by less than the first draft of this section
claimed.

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
