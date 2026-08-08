"""Score a run_ai_replay trade dump against the bar pre-declared in RESEARCH.md §2f.

The point of this file is that the verdict is arithmetic, not judgement. The bar was
registered (commit 88074bf) before the primary run existed; this only applies it.

    python run_replay_score.py logs/ai_replay_primary_predeclared_90d_*.json --days 90

Reports every clause of the bar as PASS/FAIL and prints the pre-declared consequence.
It deliberately does NOT offer a "close enough" verdict — near-misses are failures,
because the whole reason for declaring numbers in advance is to remove the negotiation.
"""

import argparse
import glob
import json
import math
import sys
from collections import defaultdict

import numpy as np
from rich.console import Console
from rich.table import Table

console = Console()

N_FLOOR = 120          # bar #1
T_BAR = 2.0            # bar #2
T_BAR_DROP = 1.5       # bar #3
SYMS_POSITIVE = 5      # bar #4
PF_BAR = 1.25          # bar #5
PACE_BAR = 10.0        # bar #6 — trades per WEEKLY round (risk.min_trades is per round)
T_BAR_OOS = 1.0        # bar #7


def tstat(xs):
    """t on the mean of per-trade PnL. Returns 0.0 when it is not defined."""
    if len(xs) < 2:
        return 0.0
    sd = float(np.std(xs, ddof=1))
    if sd == 0:
        return 0.0
    return float(np.mean(xs)) / (sd / math.sqrt(len(xs)))


def profit_factor(xs):
    gw = sum(x for x in xs if x > 0)
    gl = abs(sum(x for x in xs if x < 0))
    return (gw / gl) if gl > 0 else float("inf")


def load(patterns):
    paths = []
    for p in patterns:
        paths.extend(sorted(glob.glob(p)))
    if not paths:
        console.print(f"[red]no files matched {patterns}[/]")
        sys.exit(1)
    trades = []
    for p in paths:
        with open(p) as f:
            trades.extend(json.load(f))
        console.print(f"[dim]loaded {p}[/]")
    return [t for t in trades if t.get("traded")], paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--days", type=int, required=True, help="window length of the run")
    ap.add_argument("--oos", action="store_true",
                    help="score as the held-out window (bar #7: sign agreement + t>=1.0)")
    ap.add_argument("--primary-sign", type=int, default=None,
                    help="with --oos: sign of the primary window's total PnL (+1/-1)")
    args = ap.parse_args()

    trades, _ = load(args.files)
    if not trades:
        console.print("[red]zero trades in the dump — nothing to score[/]")
        sys.exit(1)

    pnls = [float(t["pnl"]) for t in trades]
    n = len(pnls)
    total = sum(pnls)
    t_all = tstat(pnls)
    pf = profit_factor(pnls)
    # Per WEEKLY round, not per 14d. The bar as originally written in RESEARCH.md §2f
    # said "10 per 14d", which inherited a 2x-too-lenient unit bug from the replay
    # script. Corrected here and in §2j; it does not change any verdict already
    # recorded, because every run so far cleared the stricter reading too.
    pace = n / args.days * 7

    by_sym = defaultdict(list)
    for tr in trades:
        by_sym[str(tr["symbol"]).split("/")[0]].append(float(tr["pnl"]))

    # Bar #3: drop the symbol contributing the most PnL, not the most trades.
    best_sym = max(by_sym, key=lambda s: sum(by_sym[s]))
    rest = [p for s, v in by_sym.items() if s != best_sym for p in v]
    total_drop = sum(rest)
    t_drop = tstat(rest)
    n_positive = sum(1 for v in by_sym.values() if sum(v) > 0)

    longs = [float(t["pnl"]) for t in trades if t["side"] == "long"]
    shorts = [float(t["pnl"]) for t in trades if t["side"] == "short"]

    st = Table(title="Per-symbol contribution")
    for c in ("Symbol", "n", "Total PnL", "Avg", "t"):
        st.add_column(c, justify="right")
    for s in sorted(by_sym, key=lambda x: -sum(by_sym[x])):
        v = by_sym[s]
        st.add_row(s, str(len(v)), f"${sum(v):+.2f}", f"${np.mean(v):+.2f}",
                   f"{tstat(v):+.2f}")
    console.print(st)

    dt = Table(title="Direction")
    for c in ("Side", "n", "Total PnL", "t"):
        dt.add_column(c, justify="right")
    for label, v in (("long", longs), ("short", shorts)):
        if v:
            dt.add_row(label, str(len(v)), f"${sum(v):+.2f}", f"{tstat(v):+.2f}")
    console.print(dt)

    # ---- the bar ----
    rows = []
    if args.oos:
        sign_ok = (args.primary_sign is not None
                   and math.copysign(1, total) == math.copysign(1, args.primary_sign))
        rows.append(("#7 OOS sign agrees with primary",
                     f"{'+' if total > 0 else '-'} vs "
                     f"{'+' if (args.primary_sign or 0) > 0 else '-'}", sign_ok))
        rows.append((f"#7 OOS t >= {T_BAR_OOS}", f"{t_all:+.2f}", t_all >= T_BAR_OOS))
    else:
        rows = [
            (f"#1 n >= {N_FLOOR}", str(n), n >= N_FLOOR),
            (f"#2 t >= {T_BAR}", f"{t_all:+.2f}", t_all >= T_BAR),
            (f"#3 drop best symbol ({best_sym}): PnL > 0",
             f"${total_drop:+.2f}", total_drop > 0),
            (f"#3 drop best symbol ({best_sym}): t >= {T_BAR_DROP}",
             f"{t_drop:+.2f}", t_drop >= T_BAR_DROP),
            # Denominator is the 8-pair universe, not the traded count: a symbol the
            # model never touched cannot count as net-positive, and showing /len()
            # would quietly shrink the bar to whatever it happened to trade.
            (f"#4 >= {SYMS_POSITIVE}/8 symbols net-positive",
             f"{n_positive}/8 ({len(by_sym)} traded)", n_positive >= SYMS_POSITIVE),
            (f"#5 profit factor >= {PF_BAR}", f"{pf:.2f}", pf >= PF_BAR),
            (f"#6 pace >= {PACE_BAR}/weekly round", f"{pace:.1f}", pace >= PACE_BAR),
        ]

    bt = Table(title="PRE-DECLARED BAR (RESEARCH.md §2f, registered before the run)")
    bt.add_column("Clause"); bt.add_column("Measured", justify="right")
    bt.add_column("Verdict", justify="right")
    for label, val, ok in rows:
        bt.add_row(label, val, "[green]PASS[/]" if ok else "[red]FAIL[/]")
    console.print(bt)

    console.print(f"\nTotal PnL [bold]${total:+.2f}[/] on {n} trades  |  "
                  f"t = [bold]{t_all:+.2f}[/]  |  PF {pf:.2f}  |  pace {pace:.1f}/14d")

    passed = all(ok for _, _, ok in rows)

    # Pre-declared shorts action — a diagnostic, evaluated regardless of the headline.
    # n floor first: at n=2 a t of -6.6 is arithmetic, not evidence, and gating a
    # whole direction off the back of it would be exactly the mistake this file exists
    # to prevent.
    if len(shorts) >= 30:
        t_short = tstat(shorts)
        if sum(shorts) < 0 and t_short <= -2.0:
            console.print("[yellow]PRE-DECLARED ACTION: shorts are a structural leak "
                          f"(n={len(shorts)}, ${sum(shorts):+.2f}, t={t_short:+.2f}) "
                          "-> gate shorts off in config.[/]")

    if passed:
        console.print("\n[green][bold]BAR MET.[/] Suggestive, not conclusive — the "
                      "model may know these windows from training (§2f limit 1). "
                      "Confirm on the held-out --offset window before believing it.[/]")
    else:
        failed = [label for label, _, ok in rows if not ok]
        console.print("\n[red][bold]BAR NOT MET.[/] Failed: " + "; ".join(failed) + "[/]")
        if not args.oos and not any(c.startswith("#2") and ok
                                    for c, _, ok in rows if c.startswith("#2")):
            console.print("[red]#2 is the significance clause. Pre-declared "
                          "consequence: the decision layer has NO demonstrated edge. "
                          "Do not tune the prompt until this window turns green — "
                          "that is overfitting in English. Shift the round strategy to "
                          "drawdown-minimisation + clearing the 10-trade minimum.[/]")
        elif not args.oos:
            console.print("[yellow]#2 held but a generality clause failed: the edge is "
                          "symbol-specific. Pre-declared consequence: restrict the "
                          "traded universe, do not claim a general edge.[/]")

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
