"""Walk-forward + mode comparison for WEEX bot v8.5

Compares:
  A) competition (KA on, BTC/ETH/SOL)
  B) pure_edge (KA off, optional BTC-only)

Usage:
  python run_walk_forward.py --days 120
  python run_walk_forward.py --days 90 --train 45 --test 15
"""

from __future__ import annotations

import argparse
import copy
import sys
from datetime import datetime

import yaml
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

sys.path.insert(0, ".")

from src.backtest.engine import resample_to_htf, print_backtest_results
from src.backtest.portfolio import PortfolioBacktester
from src.data.fetcher import fetch_ohlcv, fetch_funding_map, interpolate_funding

console = Console()


def load_market(symbols: list[str], days: int) -> dict:
    market = {}
    for sym in symbols:
        console.print(f"[cyan]Data {sym} ({days}d)...[/]")
        candles = fetch_ohlcv(sym, "1h", days, use_cache=True)
        if len(candles) < 200:
            console.print(f"[yellow]  skip {sym}: {len(candles)} bars[/]")
            continue
        fr = fetch_funding_map(sym, days, use_cache=True)
        funding = interpolate_funding([c.timestamp for c in candles], fr)
        market[sym] = {
            "candles": candles,
            "funding": funding,
            "htf": resample_to_htf(candles, 4),
        }
        console.print(f"[green]  {len(candles)} bars[/]")
    return market


def slice_market(market: dict, start_ts: datetime, end_ts: datetime) -> dict:
    out = {}
    for sym, data in market.items():
        c = [x for x in data["candles"] if start_ts <= x.timestamp < end_ts]
        if len(c) < 130:
            continue
        # Align funding by original indices
        full = data["candles"]
        idx_map = {x.timestamp: i for i, x in enumerate(full)}
        funding = []
        for candle in c:
            i = idx_map.get(candle.timestamp, 0)
            fr = data["funding"][i] if i < len(data["funding"]) else 0.0
            funding.append(fr)
        out[sym] = {
            "candles": c,
            "funding": funding,
            "htf": resample_to_htf(c, 4),
        }
    return out


def mode_configs(base: dict) -> list[tuple[str, dict]]:
    modes = []

    # Competition (activity + multi pair)
    a = copy.deepcopy(base)
    a.setdefault("strategy", {}).setdefault("keepalive", {})["enabled"] = True
    a.setdefault("competition", {})["pure_edge"] = False
    a.setdefault("competition", {})["disabled_pairs"] = []
    modes.append(("competition_v85", a))

    # Pure edge multi-pair no KA
    b = copy.deepcopy(base)
    b.setdefault("strategy", {}).setdefault("keepalive", {})["enabled"] = False
    b.setdefault("competition", {})["pure_edge"] = True
    b.setdefault("competition", {})["disabled_pairs"] = []
    modes.append(("pure_edge_multi", b))

    # Pure edge BTC-only (strongest pocket historically)
    c = copy.deepcopy(base)
    c.setdefault("trading", {})["symbols"] = ["BTC/USDT:USDT"]
    c.setdefault("strategy", {}).setdefault("keepalive", {})["enabled"] = False
    c.setdefault("competition", {})["pure_edge"] = True
    c.setdefault("competition", {})["disabled_pairs"] = ["ETH", "SOL"]
    modes.append(("pure_edge_btc", c))

    # Competition without ETH (ETH was drag)
    d = copy.deepcopy(base)
    d.setdefault("strategy", {}).setdefault("keepalive", {})["enabled"] = True
    d.setdefault("strategy", {}).setdefault("keepalive", {})["max_per_week"] = 2
    d.setdefault("competition", {})["disabled_pairs"] = ["ETH"]
    d.setdefault("competition", {})["pure_edge"] = False
    modes.append(("comp_no_eth", d))

    return modes


def walk_forward(market: dict, cfg: dict, train_days: int, test_days: int) -> dict:
    """Rolling windows on shared timeline of first symbol."""
    any_sym = next(iter(market.values()))
    candles = any_sym["candles"]
    t0, t1 = candles[0].timestamp, candles[-1].timestamp
    total_hours = (t1 - t0).total_seconds() / 3600
    total_days = total_hours / 24

    results = []
    cursor_days = 0.0
    fold = 0
    # Use hour steps for simplicity
    hour_start = 0
    train_h = train_days * 24
    test_h = test_days * 24

    while True:
        train_start_i = hour_start
        train_end_i = train_start_i + train_h
        test_end_i = train_end_i + test_h
        if test_end_i >= len(candles) - 10:
            break
        fold += 1
        tr_s = candles[train_start_i].timestamp
        tr_e = candles[min(train_end_i, len(candles) - 1)].timestamp
        te_s = tr_e
        te_e = candles[min(test_end_i, len(candles) - 1)].timestamp

        # OOS test only (train is for future parameter search; here we validate stability)
        test_m = slice_market(market, te_s, te_e)
        if len(test_m) < 1:
            hour_start += test_h
            continue

        # Filter symbols to config
        syms = cfg.get("trading", {}).get("symbols") or list(test_m.keys())
        test_m = {k: v for k, v in test_m.items() if k in syms}
        if not test_m:
            hour_start += test_h
            continue

        bt = PortfolioBacktester(cfg)
        res = bt.run(test_m)
        results.append({
            "fold": fold,
            "start": te_s.isoformat(),
            "end": te_e.isoformat(),
            "pnl": res.total_pnl,
            "trades": res.total_trades,
            "wr": res.win_rate,
            "dd": res.max_drawdown,
            "pf": res.profit_factor,
            "final": res.final_capital,
        })
        hour_start += test_h  # non-overlapping OOS

    if not results:
        return {"folds": [], "net_pnl": 0, "avg_wr": 0, "worst_dd": 0, "positive_folds": 0}

    net = sum(r["pnl"] for r in results)
    avg_wr = sum(r["wr"] for r in results) / len(results)
    worst_dd = max(r["dd"] for r in results)
    pos = sum(1 for r in results if r["pnl"] > 0)
    return {
        "folds": results,
        "net_pnl": net,
        "avg_wr": avg_wr,
        "worst_dd": worst_dd,
        "positive_folds": pos,
        "n_folds": len(results),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=120)
    parser.add_argument("--train", type=int, default=45)
    parser.add_argument("--test", type=int, default=15)
    parser.add_argument("--apply-best", action="store_true")
    args = parser.parse_args()

    with open("config.yaml") as f:
        base = yaml.safe_load(f)

    symbols = base.get("trading", {}).get("symbols", [
        "BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT"
    ])
    market = load_market(symbols, args.days)
    if not market:
        console.print("[red]No market data[/]")
        return

    # Full-period portfolio for each mode
    full_table = Table(title=f"Full {args.days}d Portfolio Comparison", show_header=True)
    full_table.add_column("Mode", style="cyan")
    full_table.add_column("PnL")
    full_table.add_column("Trades")
    full_table.add_column("WR")
    full_table.add_column("DD")
    full_table.add_column("PF")
    full_table.add_column("Final $")

    wfo_table = Table(title=f"Walk-Forward ({args.train}d train / {args.test}d test)", show_header=True)
    wfo_table.add_column("Mode", style="cyan")
    wfo_table.add_column("Folds")
    wfo_table.add_column("+Folds")
    wfo_table.add_column("OOS PnL")
    wfo_table.add_column("Avg WR")
    wfo_table.add_column("Worst DD")

    ranked = []

    for name, cfg in mode_configs(base):
        console.print(Panel.fit(f"[bold]Mode: {name}[/]"))
        syms = cfg.get("trading", {}).get("symbols") or list(market.keys())
        m = {k: v for k, v in market.items() if k in syms}
        if not m:
            continue

        bt = PortfolioBacktester(cfg)
        full = bt.run(m)
        col = "green" if full.total_pnl >= 0 else "red"
        full_table.add_row(
            name,
            f"[{col}]${full.total_pnl:+.2f}[/]",
            str(full.total_trades),
            f"{full.win_rate:.0%}",
            f"{full.max_drawdown:.1%}",
            f"{full.profit_factor:.2f}",
            f"${full.final_capital:.0f}",
        )

        wfo = walk_forward(m, cfg, args.train, args.test)
        wfo_table.add_row(
            name,
            str(wfo.get("n_folds", 0)),
            f"{wfo.get('positive_folds', 0)}/{wfo.get('n_folds', 0)}",
            f"${wfo.get('net_pnl', 0):+.2f}",
            f"{wfo.get('avg_wr', 0):.0%}",
            f"{wfo.get('worst_dd', 0):.1%}",
        )

        # Score: full pnl + OOS pnl - dd penalty
        score = (
            full.total_pnl
            + wfo.get("net_pnl", 0)
            - max(0, full.max_drawdown - 0.05) * 2000
            + wfo.get("positive_folds", 0) * 5
        )
        ranked.append((score, name, cfg, full, wfo))

        console.print(
            f"  Full: PnL=${full.total_pnl:+.2f} n={full.total_trades} "
            f"strats={full.strategy_stats}"
        )
        for fr in wfo.get("folds", [])[:6]:
            console.print(
                f"  fold{fr['fold']}: ${fr['pnl']:+.1f} n={fr['trades']} "
                f"WR={fr['wr']:.0%} DD={fr['dd']:.1%}"
            )

    console.print(full_table)
    console.print(wfo_table)

    ranked.sort(key=lambda x: x[0], reverse=True)
    best = ranked[0]
    console.print(Panel.fit(
        f"[bold green]BEST MODE: {best[1]}[/]\n"
        f"Score: {best[0]:.1f}\n"
        f"Full PnL: ${best[3].total_pnl:+.2f} | OOS: ${best[4].get('net_pnl', 0):+.2f}\n"
        f"DD: {best[3].max_drawdown:.1%} | Folds+: {best[4].get('positive_folds')}/{best[4].get('n_folds')}",
        title="Winner",
    ))

    if args.apply_best or best[3].total_pnl > 0:
        out = best[2]
        out["_tune"] = {
            "variant": best[1],
            "full_pnl": float(round(best[3].total_pnl, 2)),
            "oos_pnl": float(round(best[4].get("net_pnl", 0), 2)),
            "days": args.days,
            "version": "v8.5",
        }
        # numpy-safe dump
        def _py(o):
            if isinstance(o, dict):
                return {k: _py(v) for k, v in o.items()}
            if isinstance(o, list):
                return [_py(v) for v in o]
            if hasattr(o, "item"):
                return o.item()
            return o

        with open("config.yaml", "w") as f:
            yaml.safe_dump(_py(out), f, default_flow_style=False, sort_keys=False)
        console.print(f"[green]Wrote {best[1]} → config.yaml[/]")
        with open("config.edge.yaml", "w") as f:
            # Always save pure_edge_btc as alternate
            pe = next((x for x in ranked if x[1] == "pure_edge_btc"), None)
            if pe:
                yaml.safe_dump(_py(pe[2]), f, default_flow_style=False, sort_keys=False)
                console.print("[cyan]Also saved pure_edge_btc → config.edge.yaml[/]")


if __name__ == "__main__":
    main()
