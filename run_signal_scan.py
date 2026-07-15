"""Do our signals predict anything? Measured WITHOUT any AI.

This separates the two hypotheses we have been conflating:
  (a) the signals are worthless -> no model can help, we need better inputs
  (b) the signals are good and the model ignores them -> swap the model

For every candidate feature we compute the Information Coefficient: the Spearman
rank correlation between the feature's value now and the forward return over the
next N hours, pooled across symbols. This is the standard quant test for whether a
feature carries alpha, and no language model is anywhere near it.

Rules of thumb for crypto/equity IC:
  |IC| < 0.02   noise
  0.02 - 0.05   weak but potentially tradeable with low costs
  > 0.05        genuinely strong

Also reports the decile spread: mean forward return of the top 10% of readings
minus the bottom 10%. That is the money question — if you traded only the extremes,
would there be anything there?

Caveat stated up front: overlapping forward windows autocorrelate, which inflates
apparent significance. Treat the t-stats as optimistic and the IC as the real number.

    python run_signal_scan.py --days 28
"""

import argparse
import sys
import warnings
from datetime import timezone

import numpy as np
import yaml
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from scipy import stats

sys.path.insert(0, ".")
warnings.filterwarnings("ignore")

from src.data.fetcher import fetch_funding_map, fetch_ohlcv, interpolate_funding
from src.data.intel import fetch_positioning_history, _at as pos_at
from src.data.macro import fetch_macro_history, _at as macro_at
from src.indicators.technical import (
    calculate_adx, calculate_atr, calculate_bollinger_bands,
    calculate_rsi, calculate_stochastic_rsi, calculate_vwap,
)

console = Console()
HORIZONS = [4, 12, 24, 72]  # hours ahead


def build_features(symbol, candles, funding, pos, macro):
    """One row per hour: every feature, plus the forward returns we score it against."""
    closes = np.array([c.close for c in candles], float)
    highs = np.array([c.high for c in candles], float)
    lows = np.array([c.low for c in candles], float)
    vols = np.array([c.volume for c in candles], float)

    rsi = calculate_rsi(closes, 14)
    upper, mid, lower = calculate_bollinger_bands(closes, 20, 2.0)
    atr = calculate_atr(highs, lows, closes, 14)
    adx = calculate_adx(highs, lows, closes, 14)
    vwap = calculate_vwap(highs, lows, closes, vols, 20)
    k, _ = calculate_stochastic_rsi(closes)

    rows = []
    start = 200
    end = len(candles) - max(HORIZONS) - 1
    for i in range(start, end):
        ts = candles[i].timestamp
        at_ms = int(ts.replace(tzinfo=timezone.utc).timestamp() * 1000)
        px = closes[i]
        bw = float(upper[i] - lower[i])

        f = {}
        # --- technical ---
        f["ta_rsi"] = float(rsi[i])
        f["ta_stoch_rsi"] = float(k[i])
        f["ta_bb_zscore"] = (px - float(mid[i])) / (bw / 4) if bw > 0 else 0.0
        f["ta_adx"] = float(adx[i])
        f["ta_atr_pct"] = float(atr[i]) / px * 100 if px else 0.0
        f["ta_vwap_dev_pct"] = (px / float(vwap[i]) - 1) * 100 if vwap[i] else 0.0
        f["ta_ret_24h"] = (px / closes[i - 24] - 1) * 100 if i >= 24 else np.nan
        f["ta_ret_72h"] = (px / closes[i - 72] - 1) * 100 if i >= 72 else np.nan
        f["ta_vol_ratio"] = (
            float(vols[i] / np.mean(vols[i - 20:i])) if i >= 20 and np.mean(vols[i - 20:i]) > 0 else np.nan
        )

        # --- funding ---
        f["funding_rate"] = float(funding[i]) if i < len(funding) else np.nan

        # --- positioning (point-in-time) ---
        if pos:
            oi = pos_at(pos.get("oi") or {}, at_ms)
            oi24 = pos_at(pos.get("oi") or {}, at_ms - 24 * 3600 * 1000)
            retail = pos_at(pos.get("retail_long") or {}, at_ms)
            top = pos_at(pos.get("top_long") or {}, at_ms)
            taker = pos_at(pos.get("taker_buy_sell") or {}, at_ms)
            f["pos_oi_chg_24h"] = ((oi / oi24 - 1) * 100) if (oi and oi24) else np.nan
            f["pos_retail_long"] = retail * 100 if retail is not None else np.nan
            f["pos_top_long"] = top * 100 if top is not None else np.nan
            f["pos_retail_minus_top"] = (
                (retail - top) * 100 if (retail is not None and top is not None) else np.nan
            )
            f["pos_taker_ratio"] = taker if taker is not None else np.nan

        # --- macro (point-in-time, global) ---
        if macro:
            for name in ("dxy", "us_short_yield", "us_10y_yield", "sp500",
                         "nasdaq", "vix", "gold", "oil", "nikkei", "usdjpy"):
                s = macro.get(name) or {}
                now = macro_at(s, at_ms)
                d1 = macro_at(s, at_ms - 24 * 3600 * 1000)
                f[f"macro_{name}_chg24h"] = (
                    (now / d1 - 1) * 100 if (now and d1 and d1 != 0) else np.nan
                )
            vix_now = macro_at(macro.get("vix") or {}, at_ms)
            f["macro_vix_level"] = vix_now if vix_now else np.nan

        # --- forward returns: what we are trying to predict ---
        for h in HORIZONS:
            f[f"fwd_{h}h"] = (closes[i + h] / px - 1) * 100
        rows.append(f)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=28)
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    symbols = cfg["trading"]["symbols"]

    console.print(Panel.fit(
        "[bold]SIGNAL SCAN — do our inputs predict forward returns?[/]\n"
        "[dim]No AI involved. Spearman IC of each feature vs forward return.[/]",
        title="Signal Scan",
    ))

    macro = fetch_macro_history(args.days + 20)
    all_rows = []
    for s in symbols:
        c = fetch_ohlcv(s, "1h", args.days + 20, use_cache=True)
        if len(c) < 300:
            continue
        fmap = fetch_funding_map(s, args.days + 20, use_cache=True)
        funding = interpolate_funding([x.timestamp for x in c], fmap)
        pos = fetch_positioning_history(s, args.days)
        rows = build_features(s, c, funding, pos, macro)
        all_rows.extend(rows)
        console.print(f"[green]  {s:20s} {len(rows)} hourly observations[/]")

    if not all_rows:
        console.print("[red]no data[/]")
        sys.exit(1)

    features = sorted({k for r in all_rows for k in r if not k.startswith("fwd_")})
    console.print(f"\n[cyan]{len(all_rows)} observations, {len(features)} features[/]\n")

    results = []
    for feat in features:
        row = {"feature": feat}
        best_abs = 0.0
        for h in HORIZONS:
            x = np.array([r.get(feat, np.nan) for r in all_rows], float)
            y = np.array([r.get(f"fwd_{h}h", np.nan) for r in all_rows], float)
            m = np.isfinite(x) & np.isfinite(y)
            if m.sum() < 200 or np.std(x[m]) == 0:
                row[h] = None
                continue
            ic, p = stats.spearmanr(x[m], y[m])
            row[h] = (ic, p, m.sum())
            best_abs = max(best_abs, abs(ic))
        row["best"] = best_abs
        results.append(row)

    results.sort(key=lambda r: -r["best"])

    t = Table(title="Information Coefficient (Spearman) vs forward return")
    t.add_column("Feature", style="cyan")
    for h in HORIZONS:
        t.add_column(f"{h}h", justify="right")
    t.add_column("verdict", justify="left")

    for r in results:
        cells = []
        for h in HORIZONS:
            v = r.get(h)
            if not v:
                cells.append("-")
                continue
            ic, p, n = v
            colour = "green" if abs(ic) >= 0.05 else "yellow" if abs(ic) >= 0.02 else "dim"
            star = "*" if p < 0.01 else " "
            cells.append(f"[{colour}]{ic:+.3f}{star}[/]")
        b = r["best"]
        verdict = (
            "[green]STRONG[/]" if b >= 0.05
            else "[yellow]weak[/]" if b >= 0.02
            else "[dim]noise[/]"
        )
        t.add_row(r["feature"], *cells, verdict)
    console.print(t)

    # The money question: trade only the extremes — is anything there?
    console.print()
    dt = Table(title="Decile spread: mean fwd return of top 10% minus bottom 10% of readings")
    dt.add_column("Feature", style="cyan")
    for h in HORIZONS:
        dt.add_column(f"{h}h", justify="right")

    for r in results[:12]:
        feat = r["feature"]
        cells = []
        for h in HORIZONS:
            x = np.array([rr.get(feat, np.nan) for rr in all_rows], float)
            y = np.array([rr.get(f"fwd_{h}h", np.nan) for rr in all_rows], float)
            m = np.isfinite(x) & np.isfinite(y)
            if m.sum() < 200:
                cells.append("-")
                continue
            xv, yv = x[m], y[m]
            lo, hi = np.percentile(xv, 10), np.percentile(xv, 90)
            top = yv[xv >= hi].mean() if (xv >= hi).sum() else np.nan
            bot = yv[xv <= lo].mean() if (xv <= lo).sum() else np.nan
            spread = top - bot
            colour = "green" if abs(spread) > 1.0 else "yellow" if abs(spread) > 0.4 else "dim"
            cells.append(f"[{colour}]{spread:+.2f}%[/]")
        dt.add_row(feat, *cells)
    console.print(dt)

    strong = [r["feature"] for r in results if r["best"] >= 0.05]
    weak = [r["feature"] for r in results if 0.02 <= r["best"] < 0.05]
    console.print()
    if strong:
        console.print(f"[green]STRONG signals (|IC| >= 0.05): {', '.join(strong)}[/]")
    if weak:
        console.print(f"[yellow]Weak signals (|IC| 0.02-0.05): {', '.join(weak)}[/]")
    if not strong and not weak:
        console.print(
            "[red]NOTHING predicts forward returns. Every feature is noise.\n"
            "That means the model was never the bottleneck — we have been asking it "
            "to find signal in data that contains none.[/]"
        )
    console.print(
        "\n[dim]Costs are ~0.22% of notional round-trip. A signal must beat that "
        "to be worth trading at all.[/]"
    )


if __name__ == "__main__":
    main()
