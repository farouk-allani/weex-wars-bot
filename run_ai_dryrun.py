"""Watch the AI make one decision cycle. Places no orders, ever.

Builds the real context from live market data, calls DeepSeek, prints the reasoning
and every decision, then runs each one through the same validation the live engine
uses — so you can see what would have been taken, what would have been rejected, and
why, without a cent at risk.

    python run_ai_dryrun.py
    python run_ai_dryrun.py --model deepseek-reasoner   # full chain-of-thought
"""

import argparse
import sys

import yaml
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

sys.path.insert(0, ".")

from src.ai import AITrader, DecisionLog, DeepSeekClient, build_context
from src.ai.context import symbol_snapshot
from src.core.exchange import ExchangeClient
from src.indicators.technical import calculate_atr
from src.risk.manager import RiskManager
from src.strategies.edges import EdgeStrategies
import numpy as np

console = Console()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--model", default=None, help="override ai.model")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config)) or {}
    cfg.setdefault("ai", {})["enabled"] = True
    if args.model:
        cfg["ai"]["model"] = args.model

    exchange = ExchangeClient(cfg)
    risk = RiskManager(cfg)
    edges = EdgeStrategies(cfg)

    try:
        client = DeepSeekClient(cfg)
    except Exception as e:
        console.print(f"[red]{e}[/]")
        console.print("\n[yellow]Add to .env:  DEEPSEEK_API_KEY=sk-...[/]")
        sys.exit(1)

    trader = AITrader(cfg, client, DecisionLog("logs/ai_dryrun.jsonl"))

    symbols = cfg["trading"]["symbols"]
    tf = cfg["trading"].get("timeframe", "1h")
    htf = cfg["trading"].get("higher_timeframe", "4h")

    console.print(Panel.fit(
        f"[bold]AI DRY RUN — no orders will be placed[/]\n"
        f"model: [cyan]{client.model}[/]\nsymbols: {len(symbols)}",
        title="Dry Run",
    ))

    market, atrs, prices = [], {}, {}
    for sym in symbols:
        candles = exchange.fetch_candles(sym, tf, cfg["trading"].get("lookback_periods", 120))
        if len(candles) < 100:
            console.print(f"[yellow]skip {sym}: {len(candles)} candles[/]")
            continue
        htf_c = exchange.fetch_candles(sym, htf, 80)
        funding = exchange.fetch_funding_rate(sym)
        highs = np.array([c.high for c in candles])
        lows = np.array([c.low for c in candles])
        closes = np.array([c.close for c in candles])
        atrs[sym] = float(calculate_atr(highs, lows, closes)[-1])
        prices[sym] = float(closes[-1])
        e = edges.analyze_all_edges(sym, candles, funding, higher_tf_candles=htf_c or None)
        market.append(symbol_snapshot(sym, candles, funding, htf_c or None, e))
        console.print(f"[green]  {sym:20s} ${prices[sym]:>10,.4f}  ATR={atrs[sym]:.4f}[/]")

    account = exchange.get_account_state()
    context = build_context(
        symbols_data=market, account=account, risk=risk,
        recent_trades=risk.trade_history,
        competition={"ranking_metric": "cumulative PnL", "trades_executed": 0,
                     "minimum_trades_required": 10},
    )

    console.print(f"\n[cyan]Calling {client.model}...[/]")
    decisions, assessment, decision_id = trader.decide(context)

    if not decisions and not assessment:
        console.print("[red]No decision returned — check logs/ai_dryrun.jsonl for the error.[/]")
        sys.exit(1)

    console.print(Panel(assessment or "(none)", title="Market Assessment"))

    t = Table(title="Decisions (validated exactly as the live engine would)")
    for c in ("Symbol", "Action", "Conv.", "Stop", "Target", "R:R", "Verdict"):
        t.add_column(c)

    allowed = set(symbols)
    accepted = 0
    for d in decisions:
        sym = str(d.get("symbol") or "")
        action = str(d.get("action", "hold")).lower()
        price = prices.get(sym, 0)

        if action in ("long", "short"):
            sig, why = trader.to_signal(d, sym, price, atrs.get(sym, 0), allowed)
            if sig:
                accepted += 1
                rr = abs(sig.take_profit - price) / abs(price - sig.stop_loss)
                verdict = "[green]ACCEPTED[/]"
                t.add_row(sym.split("/")[0], action.upper(), f"{sig.strength:.2f}",
                          f"{sig.stop_loss:,.4f}", f"{sig.take_profit:,.4f}",
                          f"{rr:.2f}", verdict)
            else:
                t.add_row(sym.split("/")[0], action.upper(),
                          f"{float(d.get('conviction') or 0):.2f}",
                          f"{float(d.get('stop_loss') or 0):,.4f}",
                          f"{float(d.get('take_profit') or 0):,.4f}",
                          "-", f"[red]REJECTED[/] {why}")
        else:
            t.add_row(sym.split("/")[0], action.upper(), "-", "-", "-", "-", "[dim]-[/]")

    console.print(t)

    console.print("\n[bold]Rationale per symbol:[/]")
    for d in decisions:
        sym = str(d.get("symbol") or "?").split("/")[0]
        act = str(d.get("action", "?")).upper()
        console.print(f"  [cyan]{sym:6s}[/] [bold]{act:6s}[/] {d.get('rationale', '')}")

    console.print(
        f"\n[bold]{accepted}[/] of {len(decisions)} decisions would have been executed."
    )
    console.print(f"[dim]Logged to logs/ai_dryrun.jsonl (decision_id={decision_id})[/]")


if __name__ == "__main__":
    main()
