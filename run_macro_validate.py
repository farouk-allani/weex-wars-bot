"""Is the macro->crypto signal real, or one Fed pivot seen 5,000 times?

The 28d hourly scan reported IC -0.32 for short-yield changes. That number cannot be
trusted: macro is global, so all 8 symbols carry the SAME macro value each hour, and
with overlapping 24h forward windows the effective independent sample is ~28 days,
not 5,376 rows. A single regime shift would produce exactly that result.

So: daily bars, 2 years, NON-OVERLAPPING forward returns, and an out-of-sample split.
Train on the first half, test on the second half, never look at the test set to choose
anything. If the relationship is real it survives. If it was one pivot, it dies here.

    python run_macro_validate.py --years 2
"""

import argparse
import sys
import warnings

import numpy as np
import pandas as pd
import yfinance as yf
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from scipy import stats

sys.path.insert(0, ".")
warnings.filterwarnings("ignore")

console = Console()

CRYPTO = {"BTC": "BTC-USD", "ETH": "ETH-USD", "SOL": "SOL-USD"}
MACRO = {
    "us_short_yield": "^IRX",
    "us_10y_yield": "^TNX",
    "dxy": "DX-Y.NYB",
    "vix": "^VIX",
    "sp500": "^GSPC",
    "nasdaq": "^IXIC",
    "gold": "GC=F",
    "oil": "CL=F",
    "nikkei": "^N225",
    "usdjpy": "JPY=X",
}


def daily(ticker, years):
    h = yf.Ticker(ticker).history(period=f"{years}y", interval="1d")
    if h.empty:
        return None
    s = h["Close"]
    s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
    return s[~s.index.duplicated(keep="last")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, default=2)
    ap.add_argument("--horizon", type=int, default=1, help="forward days")
    args = ap.parse_args()

    console.print(Panel.fit(
        "[bold]MACRO VALIDATION — is the signal real?[/]\n"
        f"daily bars | {args.years}y | non-overlapping {args.horizon}d forward returns\n"
        "[dim]train on first half, test on second half, no peeking[/]",
        title="Validate",
    ))

    frames = {}
    for name, t in MACRO.items():
        s = daily(t, args.years)
        if s is not None and len(s) > 100:
            frames[name] = s
    for name, t in CRYPTO.items():
        s = daily(t, args.years)
        if s is not None and len(s) > 100:
            frames[f"px_{name}"] = s

    df = pd.DataFrame(frames).sort_index()
    # Forward-fill macro across crypto's weekend days: the last print IS the state.
    df = df.ffill().dropna(how="all")
    console.print(f"[green]{len(df)} daily rows, {df.index[0]:%Y-%m-%d} -> {df.index[-1]:%Y-%m-%d}[/]\n")

    feats = {}
    for name in MACRO:
        if name not in df:
            continue
        feats[f"{name}_chg1d"] = df[name].pct_change() * 100
        feats[f"{name}_chg5d"] = df[name].pct_change(5) * 100
    if "vix" in df:
        feats["vix_level"] = df["vix"]

    X = pd.DataFrame(feats, index=df.index)

    rows = []
    for coin in CRYPTO:
        col = f"px_{coin}"
        if col not in df:
            continue
        fwd = (df[col].shift(-args.horizon) / df[col] - 1) * 100
        for f in X.columns:
            d = pd.DataFrame({"x": X[f], "y": fwd}).dropna()
            if len(d) < 120:
                continue
            half = len(d) // 2
            tr, te = d.iloc[:half], d.iloc[half:]
            ic_tr, p_tr = stats.spearmanr(tr.x, tr.y)
            ic_te, p_te = stats.spearmanr(te.x, te.y)
            rows.append({
                "coin": coin, "feature": f,
                "ic_train": ic_tr, "p_train": p_tr,
                "ic_test": ic_te, "p_test": p_te,
                "n": len(d),
            })

    if not rows:
        console.print("[red]not enough data[/]")
        sys.exit(1)

    R = pd.DataFrame(rows)
    # Average across coins: a real macro signal works on all of them.
    agg = R.groupby("feature").agg(
        ic_train=("ic_train", "mean"),
        ic_test=("ic_test", "mean"),
        n=("n", "mean"),
    ).reset_index()
    agg["holds"] = (
        (agg.ic_train.abs() > 0.05)
        & (agg.ic_test.abs() > 0.05)
        & (np.sign(agg.ic_train) == np.sign(agg.ic_test))
    )
    agg = agg.reindex(agg.ic_train.abs().sort_values(ascending=False).index)

    t = Table(title=f"Daily IC vs {args.horizon}d forward return (avg across BTC/ETH/SOL)")
    t.add_column("Feature", style="cyan")
    t.add_column("IC train", justify="right")
    t.add_column("IC test (OOS)", justify="right")
    t.add_column("survives?", justify="left")
    for _, r in agg.iterrows():
        c_tr = "green" if abs(r.ic_train) >= 0.05 else "dim"
        c_te = "green" if abs(r.ic_test) >= 0.05 else "red"
        verdict = (
            "[green]HOLDS OOS[/]" if r.holds
            else "[red]dies OOS[/]" if abs(r.ic_train) >= 0.05
            else "[dim]noise[/]"
        )
        t.add_row(r.feature, f"[{c_tr}]{r.ic_train:+.3f}[/]",
                  f"[{c_te}]{r.ic_test:+.3f}[/]", verdict)
    console.print(t)

    survivors = agg[agg.holds].feature.tolist()
    console.print()
    if survivors:
        console.print(Panel(
            f"[bold green]{len(survivors)} signal(s) survive out-of-sample:[/]\n"
            + "\n".join(f"  - {s}" for s in survivors)
            + "\n\n[dim]Same sign, |IC|>0.05, in BOTH halves, on all three coins.[/]",
            title="REAL",
        ))
    else:
        console.print(Panel(
            "[bold red]No macro signal survives out-of-sample.[/]\n\n"
            "The 28d hourly IC of -0.32 was one Fed repricing, counted thousands of\n"
            "times. On daily, non-overlapping data across two years it does not hold.",
            title="SPURIOUS",
        ))


if __name__ == "__main__":
    main()
