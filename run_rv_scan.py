"""Cross-sectional relative-value scan — the direction no previous test touched.

Every prior scan bet each coin against its own history, so the shared market
beta dominated: when BTC moves 3%, all 8 pairs move with it and any per-coin
signal drowns. Here the bet is coin A minus the basket of the other seven —
long the strongest, short the weakest. Beta cancels by construction, and what
is left (dispersion between the 8) is the part a signal could actually own.

Method, same discipline as run_macro_validate (measure -> OOS -> cost):
1. Two years of DAILY bars for the 8 competition pairs (independent samples,
   no overlapping-window inflation at horizon 1).
2. Features per coin per day, computed from data at or before that day only.
3. Per-date cross-sectional Spearman IC: rank of feature across the 8 coins vs
   rank of NEXT-period coin-minus-basket return (Fama-MacBeth style), then
   aggregated over ~700 independent dates: mean IC and its t-stat.
4. Halves: a real signal keeps its sign and |IC| >= 0.03 in BOTH halves.
5. Long-short simulation for every feature, oriented on FIRST-half IC only —
   the second half is genuinely out-of-sample. Funding accrual included.
   Costs charged on turnover at market (0.22% round trip) and maker (0.10%).

Multiple-testing note printed with results: testing ~10 features x 3 horizons
means a lone marginal PASS is expected by chance; trust magnitude + both
halves + net-of-cost, not a single green cell.

    python run_rv_scan.py                 # 730 days, horizons 1/3/7, top-2/bottom-2
    python run_rv_scan.py --days 365 --k 1
"""

import argparse
import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import ccxt
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from scipy import stats

sys.path.insert(0, ".")

from src.data.fetcher import fetch_ohlcv

console = Console()

SYMBOLS = [
    "BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT", "BNB/USDT:USDT",
    "XRP/USDT:USDT", "DOGE/USDT:USDT", "ADA/USDT:USDT", "LTC/USDT:USDT",
]
CACHE_DIR = Path("data/cache/rv")

# Round-trip cost per unit of notional turned over. Market = taker + slippage
# both ways (measured 0.22%). Maker = the realistic resting-order figure from
# the execution work, not the raw fee: fills carry adverse selection.
COST_MARKET = 0.0022
COST_MAKER = 0.0010


def fetch_funding_daily(symbol: str, days: int) -> dict[str, float]:
    """Full paginated funding history -> {YYYY-MM-DD: last print of that day}.

    The shared fetch_funding_map stops at one 1000-row page (~333 days of 8h
    prints), which silently truncates a 2-year study — so paginate here.
    The LAST print of day t is known before day t+1 starts: point-in-time safe.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / f"funding_{symbol.split('/')[0]}_{days}d.json"
    if cache.exists() and (time.time() - cache.stat().st_mtime) < 86400:
        try:
            return json.loads(cache.read_text())
        except Exception:
            pass

    ex = ccxt.binance({"enableRateLimit": True})
    since = int((datetime.utcnow() - timedelta(days=days)).timestamp() * 1000)
    per_day: dict[str, tuple[int, float]] = {}
    while True:
        try:
            rows = ex.fetch_funding_rate_history(symbol, since=since, limit=1000)
        except Exception as e:
            console.print(f"[yellow]funding fetch {symbol}: {e}[/]")
            break
        if not rows:
            break
        for fr in rows:
            ts = fr.get("timestamp")
            rate = fr.get("fundingRate")
            if ts is None or rate is None:
                continue
            day = datetime.utcfromtimestamp(ts / 1000).strftime("%Y-%m-%d")
            prev = per_day.get(day)
            if prev is None or ts > prev[0]:
                per_day[day] = (ts, float(rate))
        since = rows[-1]["timestamp"] + 1
        if len(rows) < 1000:
            break

    out = {day: rate for day, (_, rate) in per_day.items()}
    if out:
        cache.write_text(json.dumps(out))
    return out


def load_panel(days: int):
    """Aligned per-day panel: closes, volumes, funding — {symbol: {day: value}}."""
    closes, volumes, funding = {}, {}, {}
    for sym in SYMBOLS:
        candles = fetch_ohlcv(sym, "1d", days=days)
        if len(candles) < 200:
            console.print(f"[red]{sym}: only {len(candles)} daily bars — skipped[/]")
            continue
        c_map, v_map = {}, {}
        for c in candles:
            day = c.timestamp.strftime("%Y-%m-%d")
            c_map[day] = float(c.close)
            v_map[day] = float(c.volume)
        closes[sym] = c_map
        volumes[sym] = v_map
        funding[sym] = fetch_funding_daily(sym, days)
        console.print(f"[green]  {sym:16s} {len(c_map)} days, "
                      f"{len(funding[sym])} funding days[/]")
    return closes, volumes, funding


def build_features(closes, volumes, funding):
    """{feature: {day: {symbol: value}}} + {day: {symbol: next-1d relative ret}}.

    Everything at day t uses data at or before t. The target at t is the
    NEXT day's coin return minus the equal-weight basket's next-day return.
    """
    all_days = sorted(set().union(*[set(m) for m in closes.values()]))
    feats: dict[str, dict[str, dict[str, float]]] = {}
    fwd_rel: dict[str, dict[str, float]] = {}

    def put(name, day, sym, val):
        if val is not None and np.isfinite(val):
            feats.setdefault(name, {}).setdefault(day, {})[sym] = float(val)

    series = {
        sym: (
            [d for d in all_days if d in closes[sym]],
            np.array([closes[sym][d] for d in all_days if d in closes[sym]]),
        )
        for sym in closes
    }

    for sym, (days_s, px) in series.items():
        vols = np.array([volumes[sym].get(d, np.nan) for d in days_s])
        rets = np.diff(px) / px[:-1]
        for i, day in enumerate(days_s):
            if i < 30:
                continue
            put("mom_1d", day, sym, px[i] / px[i - 1] - 1)
            put("mom_3d", day, sym, px[i] / px[i - 3] - 1)
            put("mom_7d", day, sym, px[i] / px[i - 7] - 1)
            put("mom_14d", day, sym, px[i] / px[i - 14] - 1)
            put("mom_30d", day, sym, px[i] / px[i - 30] - 1)
            put("vol_14d", day, sym, float(np.std(rets[max(0, i - 14):i])))
            vmean = np.nanmean(vols[i - 30:i])
            if vmean and np.isfinite(vmean) and vmean > 0:
                put("volume_anom", day, sym, vols[i] / vmean)
            put("dist_high_30", day, sym, px[i] / float(np.max(px[i - 30:i + 1])) - 1)
            f = funding.get(sym, {}).get(day)
            if f is not None:
                put("funding", day, sym, f)

    # Composite momentum: mean of cross-sectional z-scores of 7/14/30d legs.
    # One extra test, disclosed — the classic formulation, not a tuned one.
    for day in all_days:
        legs = []
        for name in ("mom_7d", "mom_14d", "mom_30d"):
            m = feats.get(name, {}).get(day, {})
            if len(m) >= 6:
                vals = np.array(list(m.values()))
                if np.std(vals) > 0:
                    legs.append({s: (v - np.mean(vals)) / np.std(vals) for s, v in m.items()})
        if len(legs) == 3:
            syms = set(legs[0]) & set(legs[1]) & set(legs[2])
            for s in syms:
                put("mom_composite", day, s, np.mean([leg[s] for leg in legs]))

    # Forward 1d relative returns
    for i, day in enumerate(all_days[:-1]):
        nxt = all_days[i + 1]
        rets = {}
        for sym, (days_s, px) in series.items():
            if day in closes[sym] and nxt in closes[sym]:
                rets[sym] = closes[sym][nxt] / closes[sym][day] - 1
        if len(rets) >= 6:
            basket = np.mean(list(rets.values()))
            fwd_rel[day] = {s: r - basket for s, r in rets.items()}

    return feats, fwd_rel, all_days


def cs_ic_series(feat_by_day, fwd_by_day, days_list, horizon=1):
    """Per-date cross-sectional Spearman ICs at the given horizon (sampled every
    `horizon` days so multi-day forward windows never overlap)."""
    out = []
    sampled = days_list[::horizon]
    for day in sampled:
        f = feat_by_day.get(day)
        if not f:
            continue
        # Horizon-h relative return = sum of the next h daily relative returns
        # (approximation is fine at these magnitudes).
        rel = {}
        base_idx = days_list.index(day)
        window = days_list[base_idx: base_idx + horizon]
        legs = [fwd_by_day.get(d) for d in window]
        if len(legs) < horizon or any(l is None for l in legs):
            continue
        syms = set(f)
        for leg in legs:
            syms &= set(leg)
        if len(syms) < 6:
            continue
        for s in syms:
            rel[s] = sum(leg[s] for leg in legs)
        x = np.array([f[s] for s in sorted(syms)])
        y = np.array([rel[s] for s in sorted(syms)])
        if np.std(x) == 0 or np.std(y) == 0:
            continue
        ic, _ = stats.spearmanr(x, y)
        if np.isfinite(ic):
            out.append((day, float(ic)))
    return out


def ls_sim(feat_by_day, fwd_by_day, funding, days_list, k, orient, hold=1):
    """Long-short: long top-k, short bottom-k by feature, re-ranked every `hold`
    days and held in between. Slower holds amortise the entry/exit over more
    edge — turnover, not the signal, is what killed the daily version.

    orient=+1 buys high feature values, -1 buys low. Returns per-day net series
    at both cost models, per unit of GROSS notional (0.5 long + 0.5 short).
    Funding accrues 3x daily: longs pay it, shorts receive it.
    """
    daily = {"gross": [], "market": [], "maker": [], "days": []}
    prev_book: dict[str, float] = {}
    since_rebalance = hold  # force a ranking on the first tradeable day
    for day in days_list:
        f = feat_by_day.get(day)
        fwd = fwd_by_day.get(day)
        if not fwd:
            continue

        turnover = 0.0
        if since_rebalance >= hold and f:
            syms = sorted(set(f) & set(fwd))
            if len(syms) >= 2 * k + 2:
                ranked = sorted(syms, key=lambda s: orient * f[s])
                shorts, longs = ranked[:k], ranked[-k:]
                w = {s: 0.5 / k for s in longs}
                w.update({s: -0.5 / k for s in shorts})
                turnover = sum(abs(w.get(s, 0) - prev_book.get(s, 0))
                               for s in set(w) | set(prev_book))
                prev_book = w
                since_rebalance = 0

        if not prev_book:
            continue
        since_rebalance += 1

        gross = sum(wt * fwd.get(s, 0.0) for s, wt in prev_book.items())
        for s, wt in prev_book.items():
            rate = funding.get(s, {}).get(day, 0.0)
            gross -= wt * 3 * rate  # longs pay positive funding, shorts earn it

        daily["days"].append(day)
        daily["gross"].append(gross)
        daily["market"].append(gross - turnover * COST_MARKET / 2)
        daily["maker"].append(gross - turnover * COST_MAKER / 2)
    return daily


def perf(rets):
    r = np.array(rets)
    if len(r) < 30:
        return None
    eq = np.cumprod(1 + r)
    dd = float(np.max(1 - eq / np.maximum.accumulate(eq)))
    sharpe = float(np.mean(r) / np.std(r) * np.sqrt(365)) if np.std(r) > 0 else 0.0
    return {
        "total_pct": float((eq[-1] - 1) * 100),
        "sharpe": sharpe,
        "max_dd_pct": dd * 100,
        "mean_daily_bps": float(np.mean(r) * 1e4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=730)
    ap.add_argument("--k", type=int, default=2, help="names per side in the L/S sim")
    ap.add_argument("--horizons", default="1,3,7")
    ap.add_argument("--hold", type=int, default=1,
                    help="re-rank every N days in the L/S sim (cuts turnover ~Nx)")
    args = ap.parse_args()
    horizons = [int(h) for h in args.horizons.split(",")]

    console.print(Panel.fit(
        "[bold]CROSS-SECTIONAL RV SCAN — does anything predict coin-vs-basket?[/]\n"
        "[dim]Beta removed by construction. IC per date across the 8 pairs,\n"
        "halves must agree, L/S sim oriented on first half only.[/]",
        title="run_rv_scan",
    ))

    console.print(f"\n[cyan]Loading {args.days}d of daily bars + funding for "
                  f"{len(SYMBOLS)} pairs...[/]")
    closes, volumes, funding = load_panel(args.days)
    if len(closes) < 6:
        console.print("[red]Not enough pairs with data.[/]")
        return

    feats, fwd_rel, all_days = build_features(closes, volumes, funding)
    scorable = sorted(set(fwd_rel) & set().union(*[set(v) for v in feats.values()]))
    console.print(f"[green]{len(scorable)} scorable days, "
                  f"{len(feats)} features, horizons {horizons}[/]\n")

    results = {}
    for h in horizons:
        table = Table(title=f"Cross-sectional Spearman IC — horizon {h}d "
                            f"(n = independent {h}d windows)")
        for col in ("feature", "IC", "t", "n", "IC 1st half", "IC 2nd half", "verdict"):
            table.add_column(col)

        for name in sorted(feats):
            ics = cs_ic_series(feats[name], fwd_rel, scorable, horizon=h)
            if len(ics) < 60:
                continue
            vals = np.array([v for _, v in ics])
            mid = len(vals) // 2
            a, b = vals[:mid], vals[mid:]
            mean, sd = float(np.mean(vals)), float(np.std(vals))
            tstat = mean / sd * np.sqrt(len(vals)) if sd > 0 else 0.0
            ia, ib = float(np.mean(a)), float(np.mean(b))

            passed = (np.sign(ia) == np.sign(ib)
                      and abs(ia) >= 0.03 and abs(ib) >= 0.03
                      and abs(tstat) >= 2.5)
            verdict = ("[bold green]PASS[/]" if passed
                       else "[yellow]halves agree[/]" if np.sign(ia) == np.sign(ib)
                            and min(abs(ia), abs(ib)) >= 0.02
                       else "[dim]noise[/]")
            table.add_row(name, f"{mean:+.3f}", f"{tstat:+.1f}", str(len(vals)),
                          f"{ia:+.3f}", f"{ib:+.3f}", verdict,
                          style="bold green" if passed else None)
            results[f"{name}_h{h}"] = {
                "ic": mean, "t": tstat, "n": len(vals),
                "ic_first_half": ia, "ic_second_half": ib, "pass": bool(passed),
            }
        console.print(table)

    # --- L/S simulation at horizon 1, orientation fixed on the FIRST half ---
    console.print(Panel.fit(
        "[bold]Long-short simulation — daily rebalance, top-%d vs bottom-%d[/]\n"
        "[dim]Orientation chosen on the first half; the second half is honest OOS.\n"
        "Costs charged on turnover: market %.2f%%, maker %.2f%% round trip.\n"
        "Funding accrual included (longs pay, shorts receive).[/]"
        % (args.k, args.k, COST_MARKET * 100, COST_MAKER * 100),
        title="edge vs cost",
    ))

    sim_table = Table()
    for col in ("feature", "orient", "gross %", "net mkt %", "net maker %",
                "maker OOS half %", "Sharpe (maker)", "maxDD %", "turnover/day"):
        sim_table.add_column(col)

    sim_results = {}
    for name in sorted(feats):
        ics = cs_ic_series(feats[name], fwd_rel, scorable, horizon=1)
        if len(ics) < 120:
            continue
        vals = np.array([v for _, v in ics])
        first_half_ic = float(np.mean(vals[:len(vals) // 2]))
        orient = 1 if first_half_ic > 0 else -1

        sim = ls_sim(feats[name], fwd_rel, funding, scorable, args.k, orient,
                     hold=args.hold)
        p_gross = perf(sim["gross"])
        p_mkt = perf(sim["market"])
        p_mkr = perf(sim["maker"])
        if not (p_gross and p_mkt and p_mkr):
            continue
        half = len(sim["maker"]) // 2
        p_oos = perf(sim["maker"][half:])
        # Average daily turnover, backed out of the cost drag it caused.
        drag = (np.mean(sim["gross"]) - np.mean(sim["market"]))
        turn_per_day = drag / (COST_MARKET / 2) if COST_MARKET else 0.0

        good = p_mkr["total_pct"] > 0 and (p_oos or {}).get("total_pct", -1) > 0
        style = "bold green" if good else None
        sim_table.add_row(
            name, "high" if orient > 0 else "low",
            f"{p_gross['total_pct']:+.1f}", f"{p_mkt['total_pct']:+.1f}",
            f"{p_mkr['total_pct']:+.1f}",
            f"{(p_oos or {}).get('total_pct', 0):+.1f}",
            f"{p_mkr['sharpe']:+.2f}", f"{p_mkr['max_dd_pct']:.1f}",
            f"{turn_per_day:.2f}",
            style=style,
        )
        sim_results[name] = {
            "orient": orient, "gross": p_gross, "net_market": p_mkt,
            "net_maker": p_mkr, "net_maker_oos_half": p_oos,
            "turnover_per_day": turn_per_day,
        }
    console.print(sim_table)

    n_tests = len(results)
    console.print(Panel.fit(
        f"[bold]How to read this[/]\n"
        f"- {n_tests} feature x horizon cells were tested. At the PASS threshold a\n"
        f"  couple of marginal greens are expected BY CHANCE across this many tests.\n"
        f"  Believe a signal only if IC magnitude, both halves, AND the net-of-cost\n"
        f"  OOS half all point the same way.\n"
        f"- Validation data is Binance perps; execution is WEEX. Prices arbitrage\n"
        f"  tight; funding differs somewhat between venues.\n"
        f"- 'maker OOS half %' is the number that matters: costs at the resting-order\n"
        f"  rate, on the half the orientation never saw.",
        title="caveats",
    ))

    out = Path("logs") / f"rv_scan_{args.days}d_{datetime.now():%Y%m%d_%H%M}.json"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w") as fh:
        json.dump({"ic": results, "sim": sim_results,
                   "params": {"days": args.days, "k": args.k,
                              "horizons": horizons,
                              "cost_market": COST_MARKET, "cost_maker": COST_MAKER}},
                  fh, indent=2)
    console.print(f"\n[dim]results -> {out}[/]")


if __name__ == "__main__":
    main()
