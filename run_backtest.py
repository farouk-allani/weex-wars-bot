"""WEEX AI Wars II — Backtest Runner with Funding Rate Data

Run: python run_backtest.py
"""

import sys
import yaml
import ccxt
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from rich.console import Console

sys.path.insert(0, ".")

from src.core.models import Candle
from src.backtest.engine import Backtester, print_backtest_results

console = Console()


def fetch_historical_data(
    symbol: str = "BTC/USDT:USDT",
    timeframe: str = "1h",
    days: int = 90,
) -> list[Candle]:
    """Fetch historical candles from exchange."""
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
            console.print(f"[yellow]Error fetching candles: {e}[/]")
            break

    candles = [
        Candle(
            timestamp=datetime.utcfromtimestamp(row[0] / 1000),
            open=row[1],
            high=row[2],
            low=row[3],
            close=row[4],
            volume=row[5],
        )
        for row in all_candles
    ]

    console.print(f"[green]Fetched {len(candles)} candles[/]")
    return candles


def fetch_funding_rates(
    symbol: str = "BTC/USDT:USDT",
    days: int = 90,
) -> dict:
    """Fetch historical funding rates and return as timestamp->rate mapping."""
    console.print(f"[cyan]Fetching funding rates for {symbol}...[/]")

    exchange = ccxt.binance({"enableRateLimit": True})

    try:
        since = int((datetime.utcnow() - timedelta(days=days)).timestamp() * 1000)
        funding_data = exchange.fetch_funding_rate_history(symbol, since=since, limit=1000)

        if funding_data:
            # Map: timestamp_ms -> funding_rate
            rates = {}
            for fr in funding_data:
                ts = fr.get("timestamp", 0)
                rate = fr.get("fundingRate", 0.0)
                if ts and rate is not None:
                    rates[ts] = rate

            console.print(f"[green]Fetched {len(rates)} funding rates (every 8h)[/]")
            return rates
        else:
            console.print("[yellow]No funding rate data available[/]")
            return {}

    except Exception as e:
        console.print(f"[yellow]Could not fetch funding rates: {e}[/]")
        return {}


def interpolate_funding_rates(
    candle_timestamps: list,
    funding_rates: dict,
) -> list[float]:
    """
    Interpolate 8h funding rates to match 1h candle timestamps.
    Each candle gets the most recent funding rate before it.
    """
    if not funding_rates:
        return [0.0] * len(candle_timestamps)

    # Sort funding timestamps
    sorted_ts = sorted(funding_rates.keys())

    result = []
    last_rate = 0.0

    for candle_ts in candle_timestamps:
        # Find the most recent funding rate before this candle
        candle_ms = int(candle_ts.timestamp() * 1000) if hasattr(candle_ts, 'timestamp') else candle_ts

        for ts in reversed(sorted_ts):
            if ts <= candle_ms:
                last_rate = funding_rates[ts]
                break

        result.append(last_rate)

    return result


def run_backtest_with_edges(config: dict, candles: list[Candle], funding_rates_list: list[float]):
    """Run backtest and show edge strategy impact."""
    from src.strategies.edges import EdgeStrategies

    edges = EdgeStrategies(config)

    edge_stats = {
        "liquidation_cascades": 0,
        "funding_extremes": 0,
        "volume_anomalies": 0,
        "mtf_aligned": 0,
        "favorable_sessions": 0,
        "total_candles": 0,
    }

    lookback = 100
    for i in range(lookback, len(candles)):
        window = candles[i - lookback + 1 : i + 1]
        fr = funding_rates_list[i] if i < len(funding_rates_list) else 0.0

        edge_result = edges.analyze_all_edges(window, fr)

        edge_stats["total_candles"] += 1

        if edge_result.get("liquidation", {}).get("detected"):
            edge_stats["liquidation_cascades"] += 1
        if edge_result.get("funding", {}).get("signal"):
            edge_stats["funding_extremes"] += 1
        if edge_result.get("volume", {}).get("anomaly"):
            edge_stats["volume_anomalies"] += 1
        if edge_result.get("mtf", {}).get("aligned"):
            edge_stats["mtf_aligned"] += 1
        if edge_result.get("session", {}).get("favorable"):
            edge_stats["favorable_sessions"] += 1

    return edge_stats


def main():
    """Run backtest on historical data with edge analysis."""
    config_path = Path("config.yaml")
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
    else:
        config = {}

    symbols = config.get("trading", {}).get("symbols", ["BTC/USDT:USDT"])
    symbol = symbols[0]

    # Fetch data
    candles = fetch_historical_data(symbol, "1h", days=90)
    funding_rates = fetch_funding_rates(symbol, days=90)

    if len(candles) < 100:
        console.print("[red]Not enough data for backtest[/]")
        return

    # Interpolate funding rates to match candle timestamps
    funding_list = interpolate_funding_rates(
        [c.timestamp for c in candles],
        funding_rates,
    )

    # Run backtest
    console.print(f"\n[cyan]Running backtest on {symbol}...[/]")
    backtester = Backtester(config)
    result = backtester.run(candles, symbol, funding_list)

    # Display results
    print_backtest_results(result)

    # Edge strategy analysis
    console.print("\n[bold cyan]🔍 Edge Strategy Analysis (90 days)[/]")
    edge_stats = run_backtest_with_edges(config, candles, funding_list)

    from rich.table import Table
    table = Table(title="Edge Detection Stats", show_header=True)
    table.add_column("Edge", style="cyan")
    table.add_column("Detections", style="white")
    table.add_column("Frequency", style="yellow")
    table.add_column("Impact", style="green")

    total = edge_stats["total_candles"]

    # Calculate what % of trades each edge could have boosted
    liq_freq = edge_stats["liquidation_cascades"] / total * 100
    fund_freq = edge_stats["funding_extremes"] / total * 100
    vol_freq = edge_stats["volume_anomalies"] / total * 100
    mtf_freq = edge_stats["mtf_aligned"] / total * 100
    sess_freq = edge_stats["favorable_sessions"] / total * 100

    table.add_row("🔴 Liquidation Cascades", str(edge_stats["liquidation_cascades"]),
                   f"{liq_freq:.1f}%", "HIGH — trade with forced moves")
    table.add_row("💰 Funding Rate Extremes", str(edge_stats["funding_extremes"]),
                   f"{fund_freq:.1f}%", "MED — contrarian signal")
    table.add_row("📊 Volume Anomalies", str(edge_stats["volume_anomalies"]),
                   f"{vol_freq:.1f}%", "MED — smart money detection")
    table.add_row("📐 Multi-TF Aligned", str(edge_stats["mtf_aligned"]),
                   f"{mtf_freq:.1f}%", "HIGH — high conviction trades")
    table.add_row("🕐 Favorable Sessions", str(edge_stats["favorable_sessions"]),
                   f"{sess_freq:.1f}%", "LOW — timing filter")
    table.add_row("Total Candles", str(total), "100%", "")

    console.print(table)

    # Edge impact summary
    console.print("\n[bold cyan]📈 Edge Impact Summary[/]")
    if edge_stats["liquidation_cascades"] > 0:
        console.print(f"  🔴 {edge_stats['liquidation_cascades']} liquidation cascades detected — each is a 1.3x signal boost")
    if edge_stats["mtf_aligned"] > total * 0.2:
        console.print(f"  📐 MTF aligned {mtf_freq:.0f}% of the time — strong trend confirmation")
    if edge_stats["funding_extremes"] > 0:
        console.print(f"  💰 {edge_stats['funding_extremes']} funding extremes — contrarian opportunities")
    if edge_stats["volume_anomalies"] > 0:
        console.print(f"  📊 {edge_stats['volume_anomalies']} volume anomalies — smart money detected")

    # Tips
    console.print("\n[bold yellow]💡 Optimization Tips:[/]")
    if result.win_rate < 0.50:
        console.print("  - Win rate < 50%: Increase RSI thresholds (try 80/20)")
    if result.max_drawdown > 0.15:
        console.print("  - Drawdown > 15%: Reduce risk per trade")
    if result.profit_factor < 1.5:
        console.print("  - Profit factor < 1.5: Tighten stop-losses")
    if mtf_freq > 30:
        console.print(f"  - MTF aligned {mtf_freq:.0f}%: Consider increasing trend_follow weight to 0.8")
    if liq_freq > 0:
        console.print(f"  - Liquidation cascades found: these are your highest-edge trades")


if __name__ == "__main__":
    main()
