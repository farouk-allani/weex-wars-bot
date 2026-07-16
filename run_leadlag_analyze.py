"""Lead-lag analyzer — does Binance lead WEEX, and by enough to trade?

Input: JSONL from run_leadlag_record.py. Method (same discipline as
run_rv_scan: measure -> split-half OOS -> cost bar):

1. Build mid-quote series per venue per symbol on a fixed grid
   (last-observation-carried-forward). One local clock for both feeds,
   so cross-venue ordering is valid.
2. LAG: cross-correlate 1-grid returns at shifts of -20s..+20s.
   Positive peak shift = Binance leads WEEX by that many seconds.
3. SIGNAL: gap g_t = ln(binance_mid / weex_mid) minus its rolling mean
   (rolling mean removes the constant basis; what's left is the part
   WEEX hasn't caught up to yet).
   Predict WEEX forward return over horizons 1s/2s/5s/10s/30s.
   IC (Spearman) on NON-OVERLAPPING windows only, split-half check.
4. TRADE TEST: when |g| > q75 of first half (threshold fixed in-sample),
   go in gap direction, exit after horizon. Report mean net move vs
   maker (0.02%/side) and taker (0.06%/side) cost on the SECOND half only.

    python run_leadlag_analyze.py data/leadlag/leadlag_XXXX.jsonl
    python run_leadlag_analyze.py data/leadlag/*.jsonl --grid-ms 250
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.table import Table
from scipy import stats as sstats

console = Console()

MAKER_SIDE = 0.0002  # 0.02% per side
TAKER_SIDE = 0.0006  # 0.06% per side


def load(paths):
    """Binance price = bookTicker mid (ch='ba'); WEEX price = trade prints
    (ch='trade') because WEEX swaps expose no quote stream. Both stamped
    with the same local clock at receive time."""
    rows = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue  # torn last line from a killed recorder
                if r.get("meta"):
                    continue
                ch, ex = r.get("ch"), r.get("ex")
                if ex == "binance" and ch == "ba" and r.get("bid") and r.get("ask"):
                    rows.append((r["rt"], ex, r["sym"], (r["bid"] + r["ask"]) / 2))
                elif ex == "weex" and ch == "trade" and r.get("px"):
                    rows.append((r["rt"], ex, r["sym"], r["px"]))
    rows.sort(key=lambda x: x[0])
    return rows


def to_grid(rows, ex, sym, grid_ms):
    pts = [(t, px) for t, e, s, px in rows if e == ex and s == sym]
    if len(pts) < 50:
        return None, None, None
    t0, t1 = pts[0][0], pts[-1][0]
    n = int((t1 - t0) // grid_ms) + 1
    grid = np.full(n, np.nan)
    for t, px in pts:
        grid[int((t - t0) // grid_ms)] = px
    # forward-fill
    idx = np.where(~np.isnan(grid))[0]
    if len(idx) == 0:
        return None, None, None
    filled = grid.copy()
    last = filled[idx[0]]
    for i in range(idx[0], n):
        if np.isnan(filled[i]):
            filled[i] = last
        else:
            last = filled[i]
    filled[: idx[0]] = filled[idx[0]]
    med_gap = np.median(np.diff([t for t, _ in pts])) if len(pts) > 1 else float("nan")
    return filled, t0, med_gap


def xcorr_lag(b, w, grid_ms, max_shift_s=20):
    """Correlate binance returns vs weex returns shifted; >0 = binance leads."""
    rb = np.diff(np.log(b))
    rw = np.diff(np.log(w))
    max_k = int(max_shift_s * 1000 / grid_ms)
    out = []
    for k in range(-max_k, max_k + 1):
        if k >= 0:
            a, c = rb[: len(rb) - k or None], rw[k:]
        else:
            a, c = rb[-k:], rw[: len(rw) + k]
        m = min(len(a), len(c))
        a, c = a[:m], c[:m]
        mask = (a != 0) | (c != 0)
        if mask.sum() < 100:
            out.append((k * grid_ms / 1000, np.nan))
            continue
        out.append((k * grid_ms / 1000, float(np.corrcoef(a[mask], c[mask])[0, 1])))
    return out


def gap_signal(b, w, grid_ms, roll_s=300):
    """Gap minus TRAILING rolling mean (no lookahead): the constant basis is
    removed, what's left is the part WEEX hasn't caught up to yet."""
    g = np.log(b / w)
    k = max(int(roll_s * 1000 / grid_ms), 10)
    k = min(k, max(len(g) // 4, 2))
    csum = np.cumsum(g)
    mean = np.empty_like(g)
    mean[:k] = csum[:k] / np.arange(1, k + 1)  # expanding mean warm-up
    mean[k:] = (csum[k:] - csum[:-k]) / k  # trailing k-window mean
    return g - mean


def ic_at_horizon(sig, w, grid_ms, h_s, half):
    """Spearman IC of gap vs weex forward return, non-overlapping windows."""
    step = max(int(h_s * 1000 / grid_ms), 1)
    fwd = np.full(len(w), np.nan)
    fwd[:-step] = np.log(w[step:] / w[:-step])
    i0, i1 = (0, len(w) // 2) if half == 1 else (len(w) // 2, len(w))
    idx = np.arange(i0, i1 - step, step)  # non-overlapping
    s, f = sig[idx], fwd[idx]
    ok = ~np.isnan(f)
    s, f = s[ok], f[ok]
    if len(s) < 30:
        return np.nan, np.nan, 0
    ic, _ = sstats.spearmanr(s, f)
    t = ic * np.sqrt(len(s) - 2) / np.sqrt(max(1 - ic * ic, 1e-9))
    return float(ic), float(t), len(s)


def trade_test(sig, w, grid_ms, h_s):
    """Threshold from first half (q75 of |gap|), trade only second half."""
    step = max(int(h_s * 1000 / grid_ms), 1)
    half = len(w) // 2
    thr = np.quantile(np.abs(sig[:half]), 0.75)
    fwd = np.full(len(w), np.nan)
    fwd[:-step] = np.log(w[step:] / w[:-step])
    rets, i = [], half
    while i < len(w) - step:
        if abs(sig[i]) > thr and not np.isnan(fwd[i]):
            rets.append(np.sign(sig[i]) * fwd[i])
            i += step  # no overlapping trades
        else:
            i += 1
    if len(rets) < 20:
        return None
    r = np.array(rets)
    t = r.mean() / (r.std(ddof=1) / np.sqrt(len(r))) if r.std() > 0 else 0
    return {"n": len(r), "gross": float(r.mean()), "t": float(t)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("files", nargs="+")
    p.add_argument("--grid-ms", type=int, default=250)
    args = p.parse_args()

    rows = load([Path(f) for f in args.files])
    if not rows:
        console.print("[red]No ticker rows found.[/red]")
        sys.exit(1)
    symbols = sorted({s for _, _, s, _ in rows})
    dur_min = (rows[-1][0] - rows[0][0]) / 60000
    console.print(f"[bold]{len(rows)} ticker updates, {dur_min:.1f} min, symbols={symbols}[/bold]\n")

    for sym in symbols:
        b, bt0, bgap = to_grid(rows, "binance", sym, args.grid_ms)
        w, wt0, wgap = to_grid(rows, "weex", sym, args.grid_ms)
        if b is None or w is None:
            console.print(f"[yellow]{sym}: not enough data on one venue, skipped[/yellow]")
            continue
        n = min(len(b), len(w))
        b, w = b[:n], w[:n]
        console.print(
            f"[bold cyan]{sym}[/bold cyan]  median update gap: "
            f"binance {bgap:.0f}ms, weex {wgap:.0f}ms  (grid {args.grid_ms}ms)"
        )

        # 1. who leads
        xc = [(lag, c) for lag, c in xcorr_lag(b, w, args.grid_ms) if not np.isnan(c)]
        if xc:
            best = max(xc, key=lambda x: x[1])
            zero = next((c for lag, c in xc if abs(lag) < 1e-9), np.nan)
            who = "BINANCE LEADS" if best[0] > 0 else ("WEEX leads(?)" if best[0] < 0 else "synchronous")
            console.print(
                f"  peak xcorr {best[1]:.3f} at shift {best[0]:+.2f}s ({who}); corr at 0s = {zero:.3f}"
            )

        # 2. gap signal IC
        sig = gap_signal(b, w, args.grid_ms)
        tbl = Table(title=f"{sym} — gap -> WEEX forward return")
        for col in ["horizon", "IC 1st half", "t", "IC 2nd half", "t", "n/half", "verdict"]:
            tbl.add_column(col, justify="right")
        for h in [1, 2, 5, 10, 30]:
            ic1, t1, n1 = ic_at_horizon(sig, w, args.grid_ms, h, half=1)
            ic2, t2, n2 = ic_at_horizon(sig, w, args.grid_ms, h, half=2)
            passed = (
                not np.isnan(ic1) and not np.isnan(ic2)
                and np.sign(ic1) == np.sign(ic2) and abs(ic1) >= 0.05 and abs(ic2) >= 0.05
                and abs(t1) >= 2 and abs(t2) >= 2
            )
            tbl.add_row(
                f"{h}s", f"{ic1:.3f}", f"{t1:.1f}", f"{ic2:.3f}", f"{t2:.1f}",
                f"{min(n1, n2)}", "[green]PASS[/green]" if passed else "-",
            )
        console.print(tbl)

        # 3. cost bar, OOS half only
        tbl2 = Table(title=f"{sym} — trade test (|gap|>q75 in-sample thr, 2nd half only)")
        for col in ["horizon", "trades", "gross/trade", "t", "vs maker rt 0.04%", "vs taker rt 0.12%"]:
            tbl2.add_column(col, justify="right")
        for h in [1, 2, 5, 10, 30]:
            r = trade_test(sig, w, args.grid_ms, h)
            if r is None:
                tbl2.add_row(f"{h}s", "<20", "-", "-", "-", "-")
                continue
            net_m = r["gross"] - 2 * MAKER_SIDE
            net_t = r["gross"] - 2 * TAKER_SIDE
            tbl2.add_row(
                f"{h}s", str(r["n"]), f"{r['gross']*100:.4f}%", f"{r['t']:.1f}",
                f"[{'green' if net_m > 0 else 'red'}]{net_m*100:+.4f}%[/]",
                f"[{'green' if net_t > 0 else 'red'}]{net_t*100:+.4f}%[/]",
            )
        console.print(tbl2)
        console.print()

    console.print(
        "[dim]Honesty notes: one clock, but venue latency asymmetry adds a constant offset "
        "(~10-100ms) — trust lags > 0.5s. 5 horizons x N symbols tested: demand PASS in both "
        "halves, not one green cell. A PASS here still needs a second recording session on a "
        "different day before wiring anything into the bot.[/dim]"
    )


if __name__ == "__main__":
    main()
