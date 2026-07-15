"""Replay historical market states through the AI and score its decisions.

The only honest way to know whether the model has an edge before risking money.
At each sampled point in history it sees ONLY the candles up to that moment, makes
a call, and we then walk the real future forward to see whether the stop or the
target came first.

Strictly no lookahead: the context is built from candles[:i], and the outcome is
resolved on candles[i+1:]. Fees and slippage are charged at the config rates.

Each decision is scored independently (fixed notional equity per trade, no
compounding, no position limits). That isolates *decision quality* from portfolio
construction — we are measuring the brain, not the book.

    python run_ai_replay.py --days 60 --every 12
    python run_ai_replay.py --days 30 --every 8 --model deepseek-reasoner
"""

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import numpy as np
import yaml
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

sys.path.insert(0, ".")

from src.ai import AITrader, DecisionLog, DeepSeekClient, build_context
from src.ai.context import symbol_snapshot
from src.backtest.engine import resample_to_htf
from src.core.models import AccountState, Side
from src.data.fetcher import fetch_funding_map, fetch_ohlcv, interpolate_funding
from src.data.intel import fetch_fear_greed, fetch_positioning_history, positioning_snapshot
from src.data.macro import fetch_macro_history, macro_snapshot
from src.indicators.technical import calculate_atr
from src.risk.manager import RiskManager
from src.strategies.edges import EdgeStrategies

console = Console()
LOOKBACK = 200          # candles the model sees
MAX_HOLD_HOURS = 72     # force-exit an unresolved trade rather than let it run forever


def resolve(candles, start_i, side, entry, sl, tp, commission, slippage):
    """Walk the real future. Which came first — the stop or the target?

    Conservative on the ambiguous bar: if a single candle spans both levels we
    assume the stop filled. We cannot know the intrabar path, and assuming the
    happy one is how backtests learn to lie.
    """
    for i in range(start_i + 1, min(start_i + 1 + MAX_HOLD_HOURS, len(candles))):
        c = candles[i]
        if side == Side.LONG:
            hit_sl, hit_tp = c.low <= sl, c.high >= tp
        else:
            hit_sl, hit_tp = c.high >= sl, c.low <= tp
        if hit_sl:
            return sl, "stop_loss", i - start_i
        if hit_tp:
            return tp, "take_profit", i - start_i
    last = min(start_i + MAX_HOLD_HOURS, len(candles) - 1)
    return candles[last].close, "timeout", last - start_i


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--days", type=int, default=60)
    ap.add_argument("--every", type=int, default=12, help="hours between decision points")
    ap.add_argument("--model", default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-intel", action="store_true",
                    help="withhold positioning/sentiment")
    ap.add_argument("--no-macro", action="store_true",
                    help="withhold macro (dollar, yields, equities, Nikkei)")
    ap.add_argument("--no-osc", action="store_true",
                    help="withhold RSI/StochRSI/Bollinger/VWAP + edge signals")
    ap.add_argument("--offset", type=int, default=0,
                    help="end the window N days earlier — for out-of-sample validation")
    ap.add_argument("--tag", default="", help="label for the saved results file")
    args = ap.parse_args()
    use_intel = not args.no_intel
    use_macro = not args.no_macro
    use_osc = not args.no_osc

    cfg = yaml.safe_load(open(args.config)) or {}
    cfg.setdefault("ai", {})["enabled"] = True
    if args.model:
        cfg["ai"]["model"] = args.model

    symbols = cfg["trading"]["symbols"]
    equity = float(cfg["backtest"]["initial_capital"])
    commission = float(cfg["backtest"]["commission_rate"])
    slippage = float(cfg["backtest"]["slippage_pct"])
    max_risk = float(cfg["risk"]["max_risk_per_trade"])

    client = DeepSeekClient(cfg)
    trader = AITrader(cfg, client, DecisionLog("logs/ai_replay.jsonl"))
    risk = RiskManager(cfg)
    edges = EdgeStrategies(cfg)

    console.print(Panel.fit(
        f"[bold]AI REPLAY — historical decision scoring[/]\n"
        f"model: [cyan]{client.model}[/]  |  {args.days}d  |  every {args.every}h\n"
        f"intel: [{'green' if use_intel else 'red'}]"
        f"{'positioning + sentiment ON' if use_intel else 'OFF (control arm)'}[/]\n"
        f"[dim]no lookahead; stop assumed on ambiguous bars[/]",
        title="Replay",
    ))

    fng = fetch_fear_greed(args.days + 5) if use_intel else {}
    macro_hist = fetch_macro_history(args.days + args.offset + 20) if use_macro else {}
    if use_macro:
        console.print(f"[green]  macro: {sum(len(v) for v in macro_hist.values())} rows "
                      f"across {len([k for k,v in macro_hist.items() if v])} series[/]")

    data = {}
    for s in symbols:
        c = fetch_ohlcv(s, "1h", args.days + 15, use_cache=True)
        if len(c) < LOOKBACK + 50:
            console.print(f"[yellow]skip {s}: {len(c)} candles[/]")
            continue
        fmap = fetch_funding_map(s, args.days + 15, use_cache=True)
        pos = fetch_positioning_history(s, args.days + 2) if use_intel else {}
        data[s] = {
            "candles": c,
            "funding": interpolate_funding([x.timestamp for x in c], fmap),
            "htf": resample_to_htf(c, 4),
            "pos": pos,
        }
        n_pos = len((pos or {}).get("oi") or {})
        console.print(f"[green]  {s:20s} {len(c)} candles, {n_pos} positioning rows[/]")

    if use_intel:
        thin = [s for s, d in data.items() if len((d["pos"] or {}).get("oi") or {}) < args.days * 20]
        if thin:
            console.print(
                f"[yellow]Positioning history is capped at ~30d by Binance. "
                f"Points older than that will fall back to TA-only.[/]"
            )

    if not data:
        console.print("[red]no data[/]")
        sys.exit(1)

    n = min(len(d["candles"]) for d in data.values())
    # Sample --days of history ending --offset days ago (leaving MAX_HOLD_HOURS so
    # every trade can resolve). --offset is what makes out-of-sample validation
    # possible: tune on one window, confirm on one you never looked at.
    end = n - MAX_HOLD_HOURS - args.offset * 24
    start = max(LOOKBACK, end - args.days * 24)
    if end <= start:
        console.print("[red]window is empty — reduce --days or --offset[/]")
        sys.exit(1)
    points = list(range(start, end, args.every))
    span_from = data[list(data)[0]]["candles"][points[0]].timestamp
    span_to = data[list(data)[0]]["candles"][points[-1]].timestamp
    console.print(
        f"\n[cyan]{len(points)} decision points, {len(data)} symbols[/]\n"
        f"[dim]window: {span_from:%Y-%m-%d} -> {span_to:%Y-%m-%d}[/]\n"
    )

    def one_point(i):
        market, atrs, prices = [], {}, {}
        for s, d in data.items():
            hist = d["candles"][: i + 1]          # <-- everything up to now, nothing after
            if len(hist) < LOOKBACK:
                continue
            win = hist[-LOOKBACK:]
            now = win[-1].timestamp
            htf = [h for h in d["htf"] if h.timestamp <= now][-80:]
            fr = d["funding"][i] if i < len(d["funding"]) else 0.0

            highs = np.array([c.high for c in win])
            lows = np.array([c.low for c in win])
            closes = np.array([c.close for c in win])
            atrs[s] = float(calculate_atr(highs, lows, closes)[-1])
            prices[s] = float(closes[-1])
            e = edges.analyze_all_edges(win, fr, higher_tf_candles=htf or None) if use_osc else None

            # Point-in-time positioning: _at() only ever looks backwards from `now`.
            posn = None
            if use_intel and d.get("pos"):
                at_ms = int(now.replace(tzinfo=timezone.utc).timestamp() * 1000)
                chg24 = (
                    (float(closes[-1]) / float(closes[-24]) - 1) * 100
                    if len(closes) > 24 else None
                )
                posn = positioning_snapshot(d["pos"], at_ms, chg24)
            market.append(symbol_snapshot(s, win, fr, htf or None, e, posn,
                                          include_oscillators=use_osc))

        if not market:
            return []

        _ts = data[list(data)[0]]["candles"][i].timestamp
        day = _ts.strftime("%Y-%m-%d")
        at_ms_point = int(_ts.replace(tzinfo=timezone.utc).timestamp() * 1000)
        account = AccountState(
            balance=equity, equity=equity, unrealized_pnl=0,
            margin_used=0, available_margin=equity, positions=[],
        )
        ctx = build_context(
            symbols_data=market, account=account, risk=risk, recent_trades=[],
            competition={"ranking_metric": "cumulative PnL", "trades_executed": 0,
                         "minimum_trades_required": 10},
            fear_greed=fng.get(day) if use_intel else None,
            macro=macro_snapshot(macro_hist, at_ms_point) if use_macro else None,
        )
        decisions, _, _ = trader.decide(ctx)

        out = []
        for d in decisions:
            sym = str(d.get("symbol") or "")
            act = str(d.get("action", "hold")).lower()
            if sym not in prices:
                continue
            if act not in ("long", "short"):
                out.append({"symbol": sym, "traded": False})
                continue

            sig, why = trader.to_signal(d, sym, prices[sym], atrs.get(sym, 0), set(data))
            if sig is None:
                out.append({"symbol": sym, "traded": False, "rejected": why})
                continue

            entry = prices[sym] * (1 + slippage if sig.side == Side.LONG else 1 - slippage)
            stop_dist = abs(entry - sig.stop_loss)
            if stop_dist <= 0:
                continue
            # Same sizing rule the live engine uses: risk budget scaled by conviction.
            size = (equity * max_risk * sig.strength) / stop_dist

            exit_px, reason, bars = resolve(
                data[sym]["candles"], i, sig.side, entry,
                sig.stop_loss, sig.take_profit, commission, slippage,
            )
            exit_fill = exit_px * (1 - slippage if sig.side == Side.LONG else 1 + slippage)
            gross = (
                (exit_fill - entry) * size if sig.side == Side.LONG
                else (entry - exit_fill) * size
            )
            fees = size * (entry + exit_fill) * commission
            pnl = gross - fees

            # Capture the regime at entry so we can test *why* a call worked,
            # not just whether it did.
            snap = next((m for m in market if m["symbol"] == sym), {})
            adx = (snap.get("trend") or {}).get("adx")
            htf = (snap.get("trend") or {}).get("htf_4h_direction")
            regime = (snap.get("trend") or {}).get("regime")
            with_trend = (
                (sig.side == Side.LONG and htf == "up")
                or (sig.side == Side.SHORT and htf == "down")
            )
            out.append({
                "symbol": sym, "traded": True, "side": sig.side.value,
                "conviction": sig.strength, "pnl": pnl,
                "r_multiple": pnl / (equity * max_risk * sig.strength),
                "reason": reason, "bars": bars,
                "adx": adx, "htf": htf, "regime": regime,
                "with_htf_trend": with_trend,
                "at": str(data[sym]["candles"][i].timestamp),
            })
        return out

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(one_point, i): i for i in points}
        done = 0
        for f in as_completed(futs):
            done += 1
            try:
                r = f.result()
                results.extend(r)
                took = [x for x in r if x.get("traded")]
                if took:
                    console.print(
                        f"[dim]{done}/{len(points)}[/] "
                        + "  ".join(
                            f"[{'green' if t['pnl'] > 0 else 'red'}]"
                            f"{t['symbol'].split('/')[0]} {t['side'][:1].upper()} "
                            f"{t['pnl']:+.2f}[/]"
                            for t in took
                        )
                    )
                else:
                    console.print(f"[dim]{done}/{len(points)}  all hold[/]")
            except Exception as e:
                console.print(f"[red]point failed: {e}[/]")

    trades = [r for r in results if r.get("traded")]
    holds = [r for r in results if not r.get("traded")]
    rejects = [r for r in holds if r.get("rejected")]

    console.print()
    if not trades:
        console.print(Panel(
            "[bold red]The model took ZERO trades across the whole replay.[/]\n\n"
            "It cannot reach the 10-trade minimum, so it would be disqualified "
            "regardless of how good its reasoning reads.",
            title="Verdict",
        ))
        sys.exit(1)

    pnls = [t["pnl"] for t in trades]
    wins = [p for p in pnls if p > 0]
    longs = [t for t in trades if t["side"] == "long"]
    shorts = [t for t in trades if t["side"] == "short"]
    gross_win = sum(wins)
    gross_loss = abs(sum(p for p in pnls if p < 0))

    t = Table(title="AI Replay Results")
    t.add_column("Metric"); t.add_column("Value", justify="right")
    t.add_row("Decision points", str(len(points)))
    t.add_row("Trades taken", str(len(trades)))
    t.add_row("Trades / 14d (competition pace)",
              f"{len(trades) / args.days * 14:.1f}")
    t.add_row("Hold rate", f"{len(holds) / max(len(results), 1) * 100:.0f}%")
    t.add_row("Rejected by guardrails", str(len(rejects)))
    t.add_row("", "")
    t.add_row("Win rate", f"{len(wins) / len(trades) * 100:.1f}%")
    t.add_row("Total PnL", f"${sum(pnls):+.2f}")
    t.add_row("PnL as % of equity", f"{sum(pnls) / equity * 100:+.2f}%")
    t.add_row("Avg R-multiple", f"{np.mean([t['r_multiple'] for t in trades]):+.2f}R")
    t.add_row("Profit factor",
              f"{gross_win / gross_loss:.2f}" if gross_loss > 0 else "inf (no losses)")
    t.add_row("Best / Worst", f"${max(pnls):+.2f} / ${min(pnls):+.2f}")
    t.add_row("", "")
    t.add_row("Longs", f"{len(longs)} ({sum(x['pnl'] for x in longs):+.2f})")
    t.add_row("Shorts", f"{len(shorts)} ({sum(x['pnl'] for x in shorts):+.2f})")
    console.print(t)

    if len(trades) / args.days * 14 < 10:
        console.print(
            "[yellow]WARNING: pace is below the 10-trade minimum "
            "— this would be disqualified.[/]"
        )

    # --- Does conviction predict anything? If not, there is no signal to select on. ---
    ct = Table(title="Does conviction predict outcome?")
    for c in ("Conviction", "Trades", "Win rate", "Total PnL", "Avg R"):
        ct.add_column(c, justify="right")
    for lo, hi in [(0.0, 0.5), (0.5, 0.65), (0.65, 0.8), (0.8, 1.01)]:
        b = [t for t in trades if lo <= t["conviction"] < hi]
        if not b:
            continue
        bw = [t for t in b if t["pnl"] > 0]
        ct.add_row(
            f"{lo:.2f}-{hi if hi <= 1 else 1.0:.2f}", str(len(b)),
            f"{len(bw) / len(b) * 100:.0f}%",
            f"${sum(t['pnl'] for t in b):+.2f}",
            f"{np.mean([t['r_multiple'] for t in b]):+.2f}R",
        )
    console.print(ct)

    # --- Is it getting killed by fading trends? ---
    tt = Table(title="Trading with vs against the 4h trend")
    for c in ("Alignment", "Trades", "Win rate", "Total PnL", "Avg R"):
        tt.add_column(c, justify="right")
    for label, sel in [
        ("WITH 4h trend", [t for t in trades if t["with_htf_trend"]]),
        ("AGAINST / flat", [t for t in trades if not t["with_htf_trend"]]),
    ]:
        if not sel:
            continue
        w = [t for t in sel if t["pnl"] > 0]
        tt.add_row(
            label, str(len(sel)), f"{len(w) / len(sel) * 100:.0f}%",
            f"${sum(t['pnl'] for t in sel):+.2f}",
            f"{np.mean([t['r_multiple'] for t in sel]):+.2f}R",
        )
    for label, sel in [
        ("  ...of which ADX>25 (strong)",
         [t for t in trades if not t["with_htf_trend"] and (t["adx"] or 0) > 25]),
        ("  ...of which ADX<=25 (weak)",
         [t for t in trades if not t["with_htf_trend"] and (t["adx"] or 0) <= 25]),
    ]:
        if not sel:
            continue
        w = [t for t in sel if t["pnl"] > 0]
        tt.add_row(
            label, str(len(sel)), f"{len(w) / len(sel) * 100:.0f}%",
            f"${sum(t['pnl'] for t in sel):+.2f}",
            f"{np.mean([t['r_multiple'] for t in sel]):+.2f}R",
        )
    console.print(tt)

    import json
    tag = args.tag or f"{'pos' if use_intel else 'nopos'}_{'macro' if use_macro else 'nomacro'}"
    out_path = f"logs/ai_replay_{tag}_{args.days}d_{datetime.now(timezone.utc):%Y%m%d_%H%M}.json"
    with open(out_path, "w") as f:
        json.dump(trades, f, indent=2, default=str)
    console.print(f"[dim]trades -> {out_path}[/]")

    edge = "POSITIVE" if sum(pnls) > 0 else "NEGATIVE"
    colour = "green" if sum(pnls) > 0 else "red"
    console.print(
        f"\n[{colour}]Measured edge over {args.days}d: {edge} "
        f"({sum(pnls) / equity * 100:+.2f}% on {len(trades)} trades)[/]"
    )
    console.print(
        "[dim]Small samples lie. And tuning the prompt until THIS window turns "
        "green is just overfitting in English — validate on a window you did not "
        "tune on.[/]"
    )


if __name__ == "__main__":
    main()
