"""WEEX AI Wars II — Multi-Symbol Backtest

Run: python run_multi_backtest.py
"""

import sys
import yaml
import ccxt
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from rich.console import Console
from rich.table import Table

sys.path.insert(0, ".")

from src.core.models import Candle
from src.backtest.engine import Backtester, print_backtest_results
from src.strategies.edges import EdgeStrategies

console = Console()


def fetch_historical_data(symbol: str, timeframe: str = "1h", days: int = 90) -> list:
    console.print(f"[cyan]Fetching {days} days of {timeframe} data for {symbol}...[/]")
    exchange = ccxt.binance({"enableRateLimit": True})
    all_candles = []
    since = int((datetime.utcnow() - timedelta(days=days)).timestamp() * 1000)

    while True:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
            if not ohlcv:
                break
            all_candles.extend(ohlcv)
            since = ohlcv[-1][0] + 1
            if len(ohlcv) < 1000:
                break
        except Exception as e:
            console.print(f"[yellow]Error: {e}[/]")
            break

    candles = [
        Candle(
            timestamp=datetime.utcfromtimestamp(row[0] / 1000),
            open=row[1], high=row[2], low=row[3], close=row[4], volume=row[5],
        )
        for row in all_candles
    ]
    console.print(f"[green]Fetched {len(candles)} candles[/]")
    return candles


def fetch_funding_rates(symbol: str, days: int = 90) -> dict:
    console.print(f"[cyan]Fetching funding rates for {symbol}...[/]")
    exchange = ccxt.binance({"enableRateLimit": True})
    try:
        since = int((datetime.utcnow() - timedelta(days=days)).timestamp() * 1000)
        funding_data = exchange.fetch_funding_rate_history(symbol, since=since, limit=1000)
        if funding_data:
            rates = {}
            for fr in funding_data:
                ts = fr.get("timestamp", 0)
                rate = fr.get("fundingRate", 0.0)
                if ts and rate is not None:
                    rates[ts] = rate
            console.print(f"[green]Fetched {len(rates)} funding rates[/]")
            return rates
        return {}
    except Exception as e:
        console.print(f"[yellow]Could not fetch funding rates: {e}[/]")
        return {}


def interpolate_funding_rates(candle_timestamps, funding_rates):
    if not funding_rates:
        return [0.0] * len(candle_timestamps)
    sorted_ts = sorted(funding_rates.keys())
    result = []
    last_rate = 0.0
    for candle_ts in candle_timestamps:
        candle_ms = int(candle_ts.timestamp() * 1000) if hasattr(candle_ts, 'timestamp') else candle_ts
        for ts in reversed(sorted_ts):
            if ts <= candle_ms:
                last_rate = funding_rates[ts]
                break
        result.append(last_rate)
    return result


def analyze_edges(config, candles, funding_list):
    edges = EdgeStrategies(config)
    stats = {"liq": 0, "fund": 0, "vol": 0, "mtf": 0, "sess": 0, "total": 0}
    lookback = 100
    for i in range(lookback, len(candles)):
        window = candles[i - lookback + 1 : i + 1]
        fr = funding_list[i] if i < len(funding_list) else 0.0
        result = edges.analyze_all_edges(window, fr)
        stats["total"] += 1
        if result.get("liquidation", {}).get("detected"): stats["liq"] += 1
        if result.get("funding", {}).get("signal"): stats["fund"] += 1
        if result.get("volume", {}).get("anomaly"): stats["vol"] += 1
        if result.get("mtf", {}).get("aligned"): stats["mtf"] += 1
        if result.get("session", {}).get("favorable"): stats["sess"] += 1
    return stats


def run_symbol(symbol: str, config: dict, days: int = 90):
    console.print(f"\n{'='*60}")
    console.print(f"[bold white on blue]  {symbol}  —  {days}-Day Backtest  [/]")
    console.print(f"{'='*60}\n")

    candles = fetch_historical_data(symbol, "1h", days)
    funding_rates = fetch_funding_rates(symbol, days)

    if len(candles) < 100:
        console.print(f"[red]Not enough data for {symbol}[/]")
        return None

    funding_list = interpolate_funding_rates(
        [c.timestamp for c in candles], funding_rates
    )

    backtester = Backtester(config)
    result = backtester.run(candles, symbol, funding_list)
    print_backtest_results(result)

    edge_stats = analyze_edges(config, candles, funding_list)

    console.print(f"\n[bold cyan]🔍 Edge Stats — {symbol}[/]")
    total = edge_stats["total"]
    table = Table(show_header=True)
    table.add_column("Edge", style="cyan")
    table.add_column("Count", style="white")
    table.add_column("Freq", style="yellow")
    table.add_row("🔴 Liquidation", str(edge_stats["liq"]), f"{edge_stats['liq']/total*100:.1f}%")
    table.add_row("💰 Funding", str(edge_stats["fund"]), f"{edge_stats['fund']/total*100:.1f}%")
    table.add_row("📊 Volume", str(edge_stats["vol"]), f"{edge_stats['vol']/total*100:.1f}%")
    table.add_row("📐 MTF Aligned", str(edge_stats["mtf"]), f"{edge_stats['mtf']/total*100:.1f}%")
    table.add_row("🕐 Session", str(edge_stats["sess"]), f"{edge_stats['sess']/total*100:.1f}%")
    console.print(table)

    return {
        "symbol": symbol,
        "total_pnl": result.total_pnl,
        "win_rate": result.win_rate,
        "total_trades": result.total_trades,
        "sharpe": result.sharpe_ratio,
        "max_dd": result.max_drawdown,
        "profit_factor": result.profit_factor,
        "avg_win": result.avg_win,
        "avg_loss": result.avg_loss,
        "edge_stats": edge_stats,
    }


def main():
    config_path = Path("config.yaml")
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
    else:
        config = {}

    symbols = [
        "BTC/USDT:USDT",
        "ETH/USDT:USDT",
        "SOL/USDT:USDT",
    ]

    results = []
    for symbol in symbols:
        r = run_symbol(symbol, config)
        if r:
            results.append(r)

    # Summary comparison
    if results:
        console.print(f"\n{'='*60}")
        console.print("[bold white on green]  📊 CROSS-SYMBOL COMPARISON  [/]")
        console.print(f"{'='*60}\n")

        table = Table(title="90-Day Backtest Comparison", show_header=True)
        table.add_column("Metric", style="cyan")
        for r in results:
            table.add_column(r["symbol"].split("/")[0], style="white")

        table.add_row("Total PnL",
            *[f"${r['total_pnl']:+,.0f}" for r in results])
        table.add_row("Win Rate",
            *[f"{r['win_rate']:.1f}%" for r in results])
        table.add_row("Total Trades",
            *[str(r['total_trades']) for r in results])
        table.add_row("Sharpe Ratio",
            *[f"{r['sharpe']:.2f}" for r in results])
        table.add_row("Max Drawdown",
            *[f"{r['max_dd']:.1f}%" for r in results])
        table.add_row("Profit Factor",
            *[f"{r['profit_factor']:.2f}" for r in results])
        table.add_row("Avg Win",
            *[f"${r['avg_win']:.0f}" for r in results])
        table.add_row("Avg Loss",
            *[f"${r['avg_loss']:.0f}" for r in results])
        table.add_row("MTF Aligned %",
            *[f"{r['edge_stats']['mtf']/r['edge_stats']['total']*100:.0f}%" for r in results])
        table.add_row("Funding Extremes",
            *[str(r['edge_stats']['fund']) for r in results])

        console.print(table)

        # Best pair recommendation
        best = max(results, key=lambda r: r["sharpe"])
        console.print(f"\n[bold green]🏆 Best pair by Sharpe: {best['symbol']} ({best['sharpe']:.2f})[/]")

        safest = min(results, key=lambda r: r["max_dd"])
        console.print(f"[bold blue]🛡️ Safest pair (lowest DD): {safest['symbol']} ({safest['max_dd']:.1f}%)[/]")

        most_trades = max(results, key=lambda r: r["total_trades"])
        console.print(f"[bold yellow]📈 Most active: {most_trades['symbol']} ({most_trades['total_trades']} trades)[/]")


if __name__ == "__main__":
    main()
