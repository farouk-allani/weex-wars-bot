"""Lead-lag recorder — Binance vs WEEX, side by side, timestamped locally.

Hypothesis: price discovery happens on Binance; WEEX perps follow with a lag.
If the Binance-minus-WEEX gap predicts WEEX's next move by more than cost,
that is a structural informational edge (no indicator involved).

This tool only RECORDS. It streams ticker + trades websockets from both
venues for the same symbols and writes every update to one JSONL file with
a local receive timestamp (single clock -> cross-venue ordering is valid;
network latency asymmetry adds a small constant offset, noted in analysis).

    python run_leadlag_record.py                          # 60 min, BTC/ETH/SOL
    python run_leadlag_record.py --minutes 180 --symbols BTC/USDT:USDT

Analyze afterwards with run_leadlag_analyze.py.
"""

import argparse
import asyncio
import json
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import aiohttp
import ccxt.pro as ccxtpro
from aiohttp.resolver import ThreadedResolver
from rich.console import Console

console = Console()

DEFAULT_SYMBOLS = ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT"]


def make_exchange(ex_cls, cfg=None):
    """aiodns/c-ares can't read this machine's DNS config -> force the OS
    resolver via a custom session, otherwise every REST call fails."""
    ex = ex_cls(cfg or {})
    ex.session = aiohttp.ClientSession(
        connector=aiohttp.TCPConnector(resolver=ThreadedResolver(), enable_cleanup_closed=True)
    )
    return ex

stats = {}  # (exchange, symbol, channel) -> update count
stop_event: asyncio.Event | None = None


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


async def stream_bidsasks(ex, ex_name: str, symbol: str, out, lock: asyncio.Lock):
    """Binance bookTicker: ~10ms quote updates — this is the LEADER signal."""
    key = (ex_name, symbol, "ba")
    stats[key] = 0
    while not stop_event.is_set():
        try:
            q = await ex.watch_bids_asks([symbol])
            t = q[symbol]
            rec = {
                "rt": time.time_ns() // 1_000_000,  # local receive, ms epoch
                "ex": ex_name,
                "sym": symbol,
                "ch": "ba",
                "bid": t.get("bid"),
                "ask": t.get("ask"),
                "ets": t.get("timestamp"),
            }
            async with lock:
                out.write(json.dumps(rec) + "\n")
            stats[key] += 1
        except Exception as e:  # reconnect on any stream hiccup
            console.print(f"[yellow]{ex_name} {symbol} ba error: {e} — retrying in 3s[/yellow]")
            await asyncio.sleep(3)


async def stream_trades(ex, ex_name: str, symbol: str, out, lock: asyncio.Lock):
    key = (ex_name, symbol, "trades")
    stats[key] = 0
    while not stop_event.is_set():
        try:
            trades = await ex.watch_trades(symbol)
            rt = time.time_ns() // 1_000_000
            async with lock:
                for tr in trades:
                    rec = {
                        "rt": rt,
                        "ex": ex_name,
                        "sym": symbol,
                        "ch": "trade",
                        "px": tr.get("price"),
                        "qty": tr.get("amount"),
                        "side": tr.get("side"),
                        "ets": tr.get("timestamp"),
                    }
                    out.write(json.dumps(rec) + "\n")
            stats[key] += len(trades)
        except Exception as e:
            console.print(f"[yellow]{ex_name} {symbol} trades error: {e} — retrying in 3s[/yellow]")
            await asyncio.sleep(3)


async def heartbeat(out, minutes: float, started: float):
    last_counts = {}
    while not stop_event.is_set():
        await asyncio.sleep(30)
        out.flush()
        elapsed = (time.time() - started) / 60
        lines = []
        for key, n in sorted(stats.items()):
            rate = (n - last_counts.get(key, 0)) / 30.0
            last_counts[key] = n
            lines.append(f"{key[0]:>8} {key[1].split('/')[0]:>5} {key[2]:>6}: {n:>7} total, {rate:5.1f}/s")
        console.print(f"[cyan]{elapsed:5.1f}/{minutes:.0f} min[/cyan]\n" + "\n".join(lines))
        if elapsed >= minutes:
            stop_event.set()


async def main(args):
    global stop_event
    stop_event = asyncio.Event()

    symbols = args.symbols
    out_dir = Path("data/leadlag")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"leadlag_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    console.print(f"[bold green]Recording Binance vs WEEX -> {out_path}[/bold green]")
    console.print(f"symbols={symbols} duration={args.minutes} min  (Ctrl+C stops early, file stays valid)")

    binance = make_exchange(ccxtpro.binanceusdm)
    weex = make_exchange(ccxtpro.weex, {"options": {"defaultType": "swap"}})

    out = open(out_path, "w", encoding="utf-8")
    lock = asyncio.Lock()
    out.write(json.dumps({"meta": True, "started": now_iso(), "symbols": symbols}) + "\n")

    tasks = []
    for sym in symbols:
        # Binance quotes = leader signal; trades on both; WEEX has no swap
        # bids/asks channel so its trade prints are the follower series.
        tasks.append(asyncio.create_task(stream_bidsasks(binance, "binance", sym, out, lock)))
        tasks.append(asyncio.create_task(stream_trades(binance, "binance", sym, out, lock)))
        tasks.append(asyncio.create_task(stream_trades(weex, "weex", sym, out, lock)))
    hb = asyncio.create_task(heartbeat(out, args.minutes, time.time()))

    loop = asyncio.get_running_loop()
    try:
        loop.add_signal_handler(signal.SIGINT, stop_event.set)
    except NotImplementedError:
        pass  # Windows: KeyboardInterrupt still lands in the wait below

    try:
        await stop_event.wait()
    except (KeyboardInterrupt, asyncio.CancelledError):
        stop_event.set()
    finally:
        for t in tasks + [hb]:
            t.cancel()
        await asyncio.gather(*tasks, hb, return_exceptions=True)
        out.flush()
        out.close()
        await binance.close()
        await weex.close()

    total = sum(stats.values())
    console.print(f"[bold green]Done. {total} updates -> {out_path}[/bold green]")
    console.print(f"Next: python run_leadlag_analyze.py {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--minutes", type=float, default=60)
    p.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    args = p.parse_args()
    try:
        asyncio.run(main(args))
    except KeyboardInterrupt:
        console.print("[yellow]Stopped by user — recording saved.[/yellow]")
