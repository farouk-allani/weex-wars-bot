"""Forward liquidation recorder — ground truth for the cascade-bounce edge.

The 30d OI-proxy scan (run_liq_scan.py) showed a mechanism-consistent
dose-response: deeper OI flushes -> bigger 15m bounce, ~+0.2%/event net maker
at the pre-declared thresholds, +1%+ on severe flushes — but t-stats are
underpowered and the proxy can't tell forced liquidations from voluntary
de-risking. This recorder fixes both by collecting REAL forced orders from
Binance's liquidation stream, so every day of running grows a clean,
liquidation-confirmed forward sample (the second period the validation bar
demands).

Records every forced order (symbol, side, price, qty, notional) plus a 1s
mid-price tape for the 8 competition pairs (to measure the bounce around
events later). JSONL, one file per launch.

    python run_liq_record.py                    # run until Ctrl+C
    python run_liq_record.py --hours 24

Analyze later: group forced orders into cascades (rolling notional per
symbol), measure forward WEEX/Binance returns at 15m/1h, compare severe vs
mild flushes against the scan's dose-response.
"""

import argparse
import asyncio
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import aiohttp
import ccxt.pro as ccxtpro
from aiohttp.resolver import ThreadedResolver
from rich.console import Console

console = Console()

PRICE_SYMBOLS = [
    "BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT", "BNB/USDT:USDT",
    "XRP/USDT:USDT", "DOGE/USDT:USDT", "ADA/USDT:USDT", "LTC/USDT:USDT",
]

# hand-verified crypto-only universe for liquidation collection (the ccxt
# all-market path is broken in this build; per-symbol works). No tokenized
# equities — AMD/TSM/CRCL etc. contaminated the volume-ranked list.
LIQ_SYMBOLS = PRICE_SYMBOLS + [
    "BCH/USDT:USDT", "LINK/USDT:USDT", "AVAX/USDT:USDT", "UNI/USDT:USDT",
    "AAVE/USDT:USDT", "XLM/USDT:USDT", "NEAR/USDT:USDT", "SUI/USDT:USDT",
    "ENA/USDT:USDT", "LDO/USDT:USDT", "TAO/USDT:USDT", "WLD/USDT:USDT",
    "ZEC/USDT:USDT", "1000PEPE/USDT:USDT", "HYPE/USDT:USDT", "ONDO/USDT:USDT",
]

stats = {"liqs": 0, "liq_notional": 0.0, "prices": 0}
stop_event: asyncio.Event | None = None


def make_exchange(ex_cls, cfg=None):
    """Windows dev box needs two fixes: aiodns can't read Windows DNS (-> OS
    resolver) and Python 3.13 strict SSL rejects some exchange CA chains
    (-> OS trust store via truststore). On Linux/Docker neither problem
    exists; fall back to defaults gracefully."""
    import ssl

    ex = ex_cls(cfg or {})
    kwargs = {"resolver": ThreadedResolver(), "enable_cleanup_closed": True}
    try:
        import truststore

        ctx = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        kwargs["ssl"] = ctx
        ex.ssl_context = ctx  # for the websocket side as well
    except ImportError:
        pass
    ex.session = aiohttp.ClientSession(connector=aiohttp.TCPConnector(**kwargs))
    return ex


async def stream_liquidations(ex, out, lock):
    while not stop_event.is_set():
        try:
            liqs = await ex.watch_liquidations_for_symbols(LIQ_SYMBOLS)
            rt = time.time_ns() // 1_000_000
            async with lock:
                for q in liqs:
                    # ccxt leaves `amount` empty for binance forced orders —
                    # size lives in contracts / raw info (o.q = qty, o.ap = avg px)
                    o = (q.get("info") or {}).get("o", {})
                    qty = q.get("amount") or q.get("contracts") or float(o.get("q") or 0)
                    px = q.get("price") or float(o.get("ap") or o.get("p") or 0)
                    notional = (px or 0) * (qty or 0)
                    out.write(json.dumps({
                        "rt": rt, "ch": "liq", "sym": q.get("symbol"),
                        "side": q.get("side"), "px": px,
                        "qty": qty, "usd": notional,
                        "ets": q.get("timestamp"),
                    }) + "\n")
                    stats["liqs"] += 1
                    stats["liq_notional"] += notional
        except Exception as e:
            console.print(f"[yellow]liq stream: {str(e)[:100]} — retry 3s[/yellow]")
            await asyncio.sleep(3)


async def stream_prices(ex, out, lock):
    """1s-throttled bookTicker mids for the competition pairs."""
    last_write = {}
    while not stop_event.is_set():
        try:
            q = await ex.watch_bids_asks(PRICE_SYMBOLS)
            rt = time.time_ns() // 1_000_000
            async with lock:
                for sym, t in q.items():
                    if rt - last_write.get(sym, 0) < 1000:
                        continue
                    if t.get("bid") and t.get("ask"):
                        out.write(json.dumps({
                            "rt": rt, "ch": "px", "sym": sym,
                            "mid": (t["bid"] + t["ask"]) / 2,
                        }) + "\n")
                        last_write[sym] = rt
                        stats["prices"] += 1
        except Exception as e:
            console.print(f"[yellow]price stream: {str(e)[:100]} — retry 3s[/yellow]")
            await asyncio.sleep(3)


async def heartbeat(out, hours, started):
    while not stop_event.is_set():
        await asyncio.sleep(60)
        out.flush()
        el = (time.time() - started) / 3600
        console.print(
            f"[cyan]{el:5.2f}/{hours}h[/cyan]  forced orders: {stats['liqs']} "
            f"(${stats['liq_notional']:,.0f})  price ticks: {stats['prices']}"
        )
        if el >= hours:
            stop_event.set()


async def main(args):
    global stop_event
    stop_event = asyncio.Event()
    out_dir = Path("data/liq_forward")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"liq_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    console.print(f"[bold green]Recording Binance forced orders -> {path}[/bold green]")

    ex = make_exchange(ccxtpro.binanceusdm)
    out = open(path, "w", encoding="utf-8")
    lock = asyncio.Lock()
    out.write(json.dumps({"meta": True, "started": datetime.now(timezone.utc).isoformat()}) + "\n")

    tasks = [
        asyncio.create_task(stream_liquidations(ex, out, lock)),
        asyncio.create_task(stream_prices(ex, out, lock)),
        asyncio.create_task(heartbeat(out, args.hours, time.time())),
    ]
    try:
        await stop_event.wait()
    except (KeyboardInterrupt, asyncio.CancelledError):
        stop_event.set()
    finally:
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        out.flush(); out.close()
        await ex.close()
    console.print(f"[bold green]Done: {stats['liqs']} forced orders -> {path}[/bold green]")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=12)
    a = ap.parse_args()
    try:
        asyncio.run(main(a))
    except KeyboardInterrupt:
        console.print("[yellow]Stopped — recording saved.[/yellow]")
