"""WEEX AI Wars II — Multi-Symbol Backtest v5

Tests: Dynamic allocation, funding filter, chandelier exit, time filter.
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

from src.core.models import Candle, Side, Position, AccountState
from src.backtest.engine import Backtester, BacktestResult, print_backtest_results
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


def run_symbol(symbol: str, config: dict, days: int = 90):
    console.print(f"\n{'='*60}")
    console.print(f"[bold white on blue]  {symbol}  —  {days}-Day Backtest v5  [/]")
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

    # Edge analysis
    edges = EdgeStrategies(config)
    stats = {"liq": 0, "fund": 0, "vol": 0, "mtf": 0, "sess": 0, "total": 0}
    lookback = 100
    for i in range(lookback, len(candles)):
        window = candles[i - lookback + 1 : i + 1]
        fr = funding_list[i] if i < len(funding_list) else 0.0
        result_edges = edges.analyze_all_edges(window, fr)
        stats["total"] += 1
        if result_edges.get("liquidation", {}).get("detected"): stats["liq"] += 1
        if result_edges.get("funding", {}).get("signal"): stats["fund"] += 1
        if result_edges.get("volume", {}).get("anomaly"): stats["vol"] += 1
        if result_edges.get("mtf", {}).get("aligned"): stats["mtf"] += 1
        if result_edges.get("session", {}).get("favorable"): stats["sess"] += 1

    console.print(f"\n[bold cyan]🔍 Edge Stats — {symbol}[/]")
    total = stats["total"]
    edge_table = Table(show_header=True)
    edge_table.add_column("Edge", style="cyan")
    edge_table.add_column("Count", style="white")
    edge_table.add_column("Freq", style="yellow")
    edge_table.add_row("🔴 Liquidation", str(stats["liq"]), f"{stats['liq']/total*100:.1f}%")
    edge_table.add_row("💰 Funding", str(stats["fund"]), f"{stats['fund']/total*100:.1f}%")
    edge_table.add_row("📊 Volume", str(stats["vol"]), f"{stats['vol']/total*100:.1f}%")
    edge_table.add_row("📐 MTF Aligned", str(stats["mtf"]), f"{stats['mtf']/total*100:.1f}%")
    edge_table.add_row("🕐 Session", str(stats["sess"]), f"{stats['sess']/total*100:.1f}%")
    console.print(edge_table)

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
    }


def main():
    config_path = Path("config.yaml")
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
    else:
        config = {}

    symbols = ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT"]
    results = []

    for symbol in symbols:
        r = run_symbol(symbol, config)
        if r:
            results.append(r)

    if results:
        console.print(f"\n{'='*60}")
        console.print("[bold white on green]  📊 CROSS-SYMBOL COMPARISON v5  [/]")
        console.print(f"{'='*60}\n")

        table = Table(title="90-Day Backtest Comparison", show_header=True)
        table.add_column("Metric", style="cyan")
        for r in results:
            table.add_column(r["symbol"].split("/")[0], style="white")

        table.add_row("Total PnL", *[f"${r['total_pnl']:+,.0f}" for r in results])
        table.add_row("Win Rate", *[f"{r['win_rate']:.1%}" for r in results])
        table.add_row("Total Trades", *[str(r['total_trades']) for r in results])
        table.add_row("Sharpe Ratio", *[f"{r['sharpe']:.2f}" for r in results])
        table.add_row("Max Drawdown", *[f"{r['max_dd']:.1%}" for r in results])
        table.add_row("Profit Factor", *[f"{r['profit_factor']:.2f}" for r in results])
        table.add_row("Avg Win", *[f"${r['avg_win']:.0f}" for r in results])
        table.add_row("Avg Loss", *[f"${r['avg_loss']:.0f}" for r in results])

        console.print(table)

        # Dynamic allocation recommendation
        console.print("\n[bold cyan]📊 Dynamic Allocation Recommendation[/]")
        total_sharpe = sum(max(0, r["sharpe"]) for r in results)
        if total_sharpe > 0:
            for r in results:
                weight = max(0.2, r["sharpe"] / total_sharpe) if r["sharpe"] > 0 else 0.2
                name = r["symbol"].split("/")[0]
                console.print(f"  {name}: {weight:.0%} allocation (Sharpe: {r['sharpe']:.2f})")

        net_pnl = sum(r["total_pnl"] for r in results)
        best = max(results, key=lambda r: r["sharpe"])
        console.print(f"\n[bold green]Net PnL: ${net_pnl:+,.0f}[/]")
        console.print(f"[bold green]🏆 Best: {best['symbol']} (Sharpe {best['sharpe']:.2f})[/]")


if __name__ == "__main__":
    main()
