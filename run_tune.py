"""WEEX AI Wars II — Quick parameter sweep + multi-symbol backtest

Runs baseline + a few competition-oriented variants on BTC/ETH/SOL.
Picks the best net PnL config with DD < 25% and enough trades.

Usage:
  python run_tune.py
  python run_tune.py --days 60
"""

import argparse
import copy
import sys
import yaml
import ccxt
from datetime import datetime, timedelta
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

sys.path.insert(0, ".")

from src.core.models import Candle
from src.backtest.engine import Backtester, print_backtest_results, resample_to_htf

console = Console()


def fetch_ohlcv(symbol: str, timeframe: str, days: int) -> list[Candle]:
    console.print(f"[cyan]Fetching {days}d {timeframe} {symbol}...[/]")
    exchange = ccxt.binance({"enableRateLimit": True})
    all_rows = []
    since = int((datetime.utcnow() - timedelta(days=days)).timestamp() * 1000)
    while True:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
            if not ohlcv:
                break
            all_rows.extend(ohlcv)
            since = ohlcv[-1][0] + 1
            if len(ohlcv) < 1000:
                break
        except Exception as e:
            console.print(f"[yellow]Fetch error: {e}[/]")
            break
    candles = [
        Candle(
            timestamp=datetime.utcfromtimestamp(r[0] / 1000),
            open=r[1], high=r[2], low=r[3], close=r[4], volume=r[5],
        )
        for r in all_rows
    ]
    console.print(f"[green]  → {len(candles)} bars[/]")
    return candles


def fetch_funding(symbol: str, days: int) -> dict:
    exchange = ccxt.binance({"enableRateLimit": True})
    try:
        since = int((datetime.utcnow() - timedelta(days=days)).timestamp() * 1000)
        data = exchange.fetch_funding_rate_history(symbol, since=since, limit=1000)
        return {
            fr["timestamp"]: fr.get("fundingRate", 0.0)
            for fr in (data or [])
            if fr.get("timestamp")
        }
    except Exception:
        return {}


def interpolate_funding(timestamps, rates: dict) -> list[float]:
    if not rates:
        return [0.0] * len(timestamps)
    sorted_ts = sorted(rates.keys())
    out, last = [], 0.0
    for ts in timestamps:
        ms = int(ts.timestamp() * 1000)
        for t in reversed(sorted_ts):
            if t <= ms:
                last = rates[t]
                break
        out.append(last)
    return out


def make_variants(base: dict) -> list[tuple[str, dict]]:
    """v8.3 competition variants."""
    variants = []
    variants.append(("v8.3_baseline", copy.deepcopy(base)))

    # MR only + HTF bias
    b = copy.deepcopy(base)
    b.setdefault("strategy", {}).setdefault("trend_follow", {})["enabled"] = False
    b.setdefault("strategy", {}).setdefault("breakout", {})["enabled"] = False
    b.setdefault("strategy", {}).setdefault("mean_reversion", {})["strength"] = 0.72
    b.setdefault("strategy", {}).setdefault("mean_reversion", {})["max_adx"] = 32
    b.setdefault("strategy", {}).setdefault("mean_reversion", {})["rsi_oversold"] = 32
    b.setdefault("strategy", {}).setdefault("mean_reversion", {})["rsi_overbought"] = 68
    b["competition"] = {
        "min_edges": 1, "min_rr": 1.3, "allow_asia_mr": True,
        "htf_directional_bias": True, "long_only": False,
    }
    variants.append(("mr_only_htf", b))

    # Long-only (crypto beta)
    c = copy.deepcopy(base)
    c["competition"] = {
        "min_edges": 1, "min_rr": 1.3, "allow_asia_mr": True,
        "htf_directional_bias": True, "long_only": True, "skip_asia_trend": True,
    }
    c.setdefault("strategy", {}).setdefault("mean_reversion", {})["strength"] = 0.7
    variants.append(("long_only", c))

    # Pullback only + long bias when HTF long
    d = copy.deepcopy(base)
    d.setdefault("strategy", {}).setdefault("mean_reversion", {})["enabled"] = False
    d.setdefault("strategy", {}).setdefault("trend_follow", {})["adx_threshold"] = 22
    d.setdefault("strategy", {}).setdefault("trend_follow", {})["max_extension_atr"] = 0.7
    d["competition"] = {
        "min_edges": 1, "min_rr": 1.3, "htf_directional_bias": True, "long_only": False,
    }
    variants.append(("pullback_htf", d))

    # Balanced lower risk
    e = copy.deepcopy(base)
    e.setdefault("risk", {})["max_risk_per_trade"] = 0.01
    e.setdefault("strategy", {}).setdefault("mean_reversion", {})["rsi_oversold"] = 28
    e.setdefault("strategy", {}).setdefault("mean_reversion", {})["rsi_overbought"] = 72
    e["competition"] = {
        "min_edges": 1, "min_rr": 1.4, "allow_asia_mr": True,
        "htf_directional_bias": True, "long_only": False,
    }
    variants.append(("balanced_tight", e))

    return variants


def score_result(pnl: float, max_dd: float, trades: int, sharpe: float) -> float:
    """Higher is better. Penalize high DD and too few trades."""
    if trades < 3:
        return -1e9
    dd_pen = max(0, max_dd - 0.20) * 5000  # hard penalty above 20%
    trade_bonus = min(trades, 40) * 2
    return pnl - dd_pen + trade_bonus + sharpe * 20


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--apply-best", action="store_true", help="Write best config to config.yaml")
    args = parser.parse_args()

    with open("config.yaml") as f:
        base = yaml.safe_load(f)

    symbols = base.get("trading", {}).get("symbols", [
        "BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT"
    ])

    # Fetch all market data once
    market = {}
    for sym in symbols:
        c1h = fetch_ohlcv(sym, "1h", args.days)
        if len(c1h) < 120:
            console.print(f"[red]Skip {sym}: not enough data[/]")
            continue
        fr = fetch_funding(sym, args.days)
        funding = interpolate_funding([c.timestamp for c in c1h], fr)
        htf = resample_to_htf(c1h, 4)
        market[sym] = {"candles": c1h, "funding": funding, "htf": htf}

    if not market:
        console.print("[red]No market data — check network / ccxt[/]")
        return

    variants = make_variants(base)
    summary_rows = []

    for name, cfg in variants:
        console.print(Panel.fit(f"[bold]Running variant: {name}[/]"))
        net_pnl = 0.0
        net_trades = 0
        worst_dd = 0.0
        sharpes = []
        per_sym = []

        for sym, data in market.items():
            # Inject competition flags into strategy via config
            bt = Backtester(cfg)
            # Wire min_edges into strategy if present
            min_edges = cfg.get("competition", {}).get("min_edges")
            if min_edges is not None:
                bt.strategy._min_edges = min_edges
            allow_asia_mr = cfg.get("competition", {}).get("allow_asia_mr", False)
            bt.strategy._allow_asia_mr = allow_asia_mr

            result = bt.run(
                data["candles"], sym, data["funding"], higher_tf_candles=data["htf"]
            )
            net_pnl += result.total_pnl
            net_trades += result.total_trades
            worst_dd = max(worst_dd, result.max_drawdown)
            sharpes.append(result.sharpe_ratio)
            per_sym.append((sym.split("/")[0], result.total_pnl, result.total_trades, result.win_rate, result.max_drawdown))

            console.print(
                f"  {sym.split('/')[0]}: PnL=${result.total_pnl:+.0f} "
                f"trades={result.total_trades} WR={result.win_rate:.0%} "
                f"DD={result.max_drawdown:.1%} PF={result.profit_factor:.2f}"
            )
            if result.strategy_stats:
                for sn, ss in result.strategy_stats.items():
                    console.print(f"      {sn}: n={ss['trades']} pnl=${ss['pnl']:+.0f}")

        avg_sharpe = sum(sharpes) / len(sharpes) if sharpes else 0
        sc = score_result(net_pnl, worst_dd, net_trades, avg_sharpe)
        summary_rows.append({
            "name": name,
            "net_pnl": net_pnl,
            "trades": net_trades,
            "worst_dd": worst_dd,
            "avg_sharpe": avg_sharpe,
            "score": sc,
            "config": cfg,
            "per_sym": per_sym,
        })

    summary_rows.sort(key=lambda r: r["score"], reverse=True)

    table = Table(title=f"{args.days}-Day Multi-Pair Tune Results", show_header=True)
    table.add_column("Rank")
    table.add_column("Variant", style="cyan")
    table.add_column("Net PnL")
    table.add_column("Trades")
    table.add_column("Worst DD")
    table.add_column("Avg Sharpe")
    table.add_column("Score")

    for i, r in enumerate(summary_rows, 1):
        col = "green" if r["net_pnl"] > 0 else "red"
        table.add_row(
            str(i),
            r["name"],
            f"[{col}]${r['net_pnl']:+,.0f}[/]",
            str(r["trades"]),
            f"{r['worst_dd']:.1%}",
            f"{r['avg_sharpe']:.2f}",
            f"{r['score']:.0f}",
        )
    console.print(table)

    best = summary_rows[0]
    console.print(Panel.fit(
        f"[bold green]BEST: {best['name']}[/]\n"
        f"Net PnL: ${best['net_pnl']:+,.2f}\n"
        f"Trades: {best['trades']} | Worst DD: {best['worst_dd']:.1%}\n"
        f"Avg Sharpe: {best['avg_sharpe']:.2f}",
        title="Winner",
    ))

    # Always write best into config if better, or if --apply-best
    out_path = Path("config.yaml")
    best_cfg = best["config"]
    # Stamp metadata
    def _py(obj):
        """Convert numpy scalars so yaml.safe_dump works."""
        if isinstance(obj, dict):
            return {k: _py(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_py(v) for v in obj]
        if hasattr(obj, "item"):
            try:
                return obj.item()
            except Exception:
                return float(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        return obj

    import numpy as np

    best_cfg["_tune"] = {
        "variant": best["name"],
        "days": int(args.days),
        "net_pnl": float(round(float(best["net_pnl"]), 2)),
        "trades": int(best["trades"]),
        "worst_dd": float(round(float(best["worst_dd"]), 4)),
        "tuned_at": datetime.now(tz=None).isoformat() + "Z",
    }
    best_cfg = _py(best_cfg)

    # Always persist best for iteration; only overwrite config.yaml if not worse than -2%
    tuned_path = Path("config.tuned.yaml")
    with open(tuned_path, "w") as f:
        yaml.safe_dump(best_cfg, f, default_flow_style=False, sort_keys=False)
    console.print(f"[cyan]Saved {tuned_path} ({best['name']})[/]")

    if args.apply_best or best["net_pnl"] > -50:
        with open(out_path, "w") as f:
            yaml.safe_dump(best_cfg, f, default_flow_style=False, sort_keys=False)
        console.print(f"[green]Wrote best config ({best['name']}) → config.yaml[/]")
    else:
        console.print("[yellow]Large loss on all variants — kept config.yaml; see config.tuned.yaml[/]")

    # Paper/live checklist
    console.print(Panel.fit(
        "[bold]Paper / Live Checklist[/]\n"
        "1. config trading.mode = paper\n"
        "2. python test_bot.py\n"
        "3. python -m src.main  (watch SL/TP on fills)\n"
        "4. Confirm WEEX keys in .env\n"
        "5. Flip mode: live only after 24h paper looks clean\n"
        "6. Start with default_leverage 3–5, max 8",
        title="Next Steps",
    ))


if __name__ == "__main__":
    main()
