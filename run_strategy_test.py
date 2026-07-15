"""Are the two surviving signals actually tradeable after costs?

Signals that survived 2y out-of-sample validation:
    sp500_chg1d   IC -0.05   (equities up today  -> crypto down tomorrow)
    vix_chg1d     IC +0.08   (VIX up today       -> crypto up tomorrow)

Both say the same thing: after a risk-off shock, crypto bounces the next day.

This trades that thesis mechanically. No LLM. Daily rebalance, one position, costs
charged on every switch. Train/test split with the rule FIXED on the training half —
the test half is never used to choose a threshold.

An IC of 0.06 is thin. The question is whether it clears 0.22% round-trip costs.

    python run_strategy_test.py --years 2
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

sys.path.insert(0, ".")
warnings.filterwarnings("ignore")

console = Console()
COST = 0.0022  # round-trip: commission 0.06%x2 + slippage 0.05%x2


def daily(t, years):
    h = yf.Ticker(t).history(period=f"{years}y", interval="1d")
    s = h["Close"]
    s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
    return s[~s.index.duplicated(keep="last")]


def stats_of(rets: pd.Series, n_trades: int, label: str) -> dict:
    if rets.empty:
        return {}
    total = (1 + rets).prod() - 1
    days = len(rets)
    sharpe = (rets.mean() / rets.std() * np.sqrt(365)) if rets.std() > 0 else 0
    eq = (1 + rets).cumprod()
    dd = ((eq - eq.cummax()) / eq.cummax()).min()
    return {
        "label": label, "days": days, "trades": n_trades,
        "total_pct": total * 100,
        "sharpe": sharpe,
        "max_dd_pct": dd * 100,
        "win_rate": (rets > 0).mean() * 100,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, default=2)
    ap.add_argument("--coin", default="BTC-USD")
    ap.add_argument("--z", type=float, default=0.5,
                    help="signal threshold in std devs (fixed on train, not tuned on test)")
    args = ap.parse_args()

    console.print(Panel.fit(
        "[bold]STRATEGY TEST — risk-off reversal[/]\n"
        "long crypto after equities drop / VIX spikes; short after the reverse\n"
        f"[dim]{args.coin} | daily | costs {COST*100:.2f}% round-trip | no AI[/]",
        title="Test",
    ))

    px = daily(args.coin, args.years)
    sp = daily("^GSPC", args.years)
    vix = daily("^VIX", args.years)

    df = pd.DataFrame({"px": px, "sp": sp, "vix": vix}).sort_index().ffill().dropna()
    df["sp_chg"] = df.sp.pct_change() * 100
    df["vix_chg"] = df.vix.pct_change() * 100
    df["fwd"] = df.px.shift(-1) / df.px - 1
    df = df.dropna()

    # Composite: both signals point the same way (risk-off today -> long tomorrow).
    # Standardised on the TRAINING half only; the test half never informs the scaling.
    half = len(df) // 2
    tr = df.iloc[:half]
    mu_sp, sd_sp = tr.sp_chg.mean(), tr.sp_chg.std()
    mu_vx, sd_vx = tr.vix_chg.mean(), tr.vix_chg.std()

    z_sp = -(df.sp_chg - mu_sp) / sd_sp     # sp down  -> positive
    z_vx = (df.vix_chg - mu_vx) / sd_vx     # vix up   -> positive
    df["score"] = (z_sp + z_vx) / 2

    df["pos"] = np.where(df.score > args.z, 1, np.where(df.score < -args.z, -1, 0))
    switches = (df.pos != df.pos.shift(1)).astype(int)
    df["ret"] = df.pos * df.fwd - switches * COST

    console.print(f"[green]{len(df)} daily bars, {df.index[0]:%Y-%m-%d} -> {df.index[-1]:%Y-%m-%d}[/]")
    console.print(f"[dim]threshold z={args.z} fixed on train half[/]\n")

    parts = {
        "TRAIN (in-sample)": df.iloc[:half],
        "TEST  (out-of-sample)": df.iloc[half:],
        "FULL": df,
    }

    t = Table(title=f"Risk-off reversal on {args.coin}")
    for c in ("Period", "Days", "Trades", "Return", "Sharpe", "MaxDD", "Win%"):
        t.add_column(c, justify="right")
    t.columns[0].justify = "left"

    oos = None
    for label, d in parts.items():
        sw = int((d.pos != d.pos.shift(1)).sum())
        s = stats_of(d.ret, sw, label)
        if not s:
            continue
        colour = "green" if s["total_pct"] > 0 else "red"
        t.add_row(
            label, str(s["days"]), str(s["trades"]),
            f"[{colour}]{s['total_pct']:+.1f}%[/]",
            f"{s['sharpe']:.2f}", f"{s['max_dd_pct']:.1f}%", f"{s['win_rate']:.0f}%",
        )
        if label.startswith("TEST"):
            oos = s

    # Benchmark: what did just holding do?
    bh = df.fwd
    b = stats_of(bh.iloc[half:], 1, "buy & hold (OOS)")
    t.add_row(
        "[dim]buy & hold (OOS)[/]", str(b["days"]), "1",
        f"[dim]{b['total_pct']:+.1f}%[/]", f"[dim]{b['sharpe']:.2f}[/]",
        f"[dim]{b['max_dd_pct']:.1f}%[/]", f"[dim]{b['win_rate']:.0f}%[/]",
    )
    console.print(t)

    console.print()
    if oos and oos["total_pct"] > 0 and oos["sharpe"] > 0.5:
        console.print(Panel(
            f"[bold green]The signal is tradeable out-of-sample.[/]\n\n"
            f"OOS return {oos['total_pct']:+.1f}% over {oos['days']} days, "
            f"Sharpe {oos['sharpe']:.2f}, max drawdown {oos['max_dd_pct']:.1f}%,\n"
            f"across {oos['trades']} trades — after costs.",
            title="REAL EDGE",
        ))
    elif oos and oos["total_pct"] > 0:
        console.print(Panel(
            f"[yellow]Positive out-of-sample ({oos['total_pct']:+.1f}%) but thin "
            f"(Sharpe {oos['sharpe']:.2f}).[/]\n"
            "Real but marginal. Costs eat most of it.",
            title="MARGINAL",
        ))
    else:
        console.print(Panel(
            "[red]Does not survive costs out-of-sample.[/]\n"
            "The IC was real but too small to pay for the spread.",
            title="NOT TRADEABLE",
        ))


if __name__ == "__main__":
    main()
