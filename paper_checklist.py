"""Interactive paper-session checklist for v8.5.

Run before and during paper trading:
  python paper_checklist.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import yaml
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

sys.path.insert(0, ".")
console = Console()


def main():
    rows = []
    ok = fail = 0

    def check(name, cond, detail=""):
        nonlocal ok, fail
        if cond:
            ok += 1
            rows.append((name, "[green]OK[/]", detail))
        else:
            fail += 1
            rows.append((name, "[red]FIX[/]", detail))

    cfg = {}
    if Path("config.yaml").exists():
        with open("config.yaml") as f:
            cfg = yaml.safe_load(f) or {}

    mode = cfg.get("trading", {}).get("mode")
    check("Paper mode", mode == "paper", f"mode={mode}")
    symbols = cfg.get("trading", {}).get("symbols") or []
    check("Symbols set", len(symbols) >= 1, str(symbols))
    check("ETH disabled (WFO)", "ETH" in (cfg.get("competition", {}).get("disabled_pairs") or [])
          or not any("ETH" in s for s in symbols), "ETH was a drag")
    check("Breakouts off", not cfg.get("strategy", {}).get("breakout", {}).get("enabled", False))
    check("Partial TP on", cfg.get("risk", {}).get("partial_tp_enabled", False))
    check("KA capped", cfg.get("strategy", {}).get("keepalive", {}).get("max_per_week", 99) <= 3)

    # Modules
    try:
        from src.core.models import Position, Side
        from src.risk.manager import RiskManager
        from src.strategies.composite import CompositeStrategy
        check("Imports", True)
        p = Position("X", Side.SHORT, 1, 1, 5, 0, 0)
        check("Zero SL short safe", p.should_stop_loss(1) is False)
    except Exception as e:
        check("Imports", False, str(e))

    # State / logs dirs
    Path("logs").mkdir(exist_ok=True)
    Path("data").mkdir(exist_ok=True)
    check("logs/ ready", Path("logs").is_dir())
    check("data/ ready", Path("data").is_dir())

    from dotenv import load_dotenv
    load_dotenv()
    # Paper can run without keys if only public data — still warn
    has_key = bool(os.getenv("WEEX_API_KEY")) and os.getenv("WEEX_API_KEY") != "your_api_key_here"
    check("API key (optional for public data)", True, "SET" if has_key else "missing — OK for paper if exchange public")

    table = Table(title="Paper Session Checklist v8.5", show_header=True)
    table.add_column("Item")
    table.add_column("Status")
    table.add_column("Detail")
    for r in rows:
        table.add_row(*r)
    console.print(table)

    console.print(Panel.fit(
        "[bold]Session plan[/]\n"
        "1. python test_bot.py\n"
        "2. python run_portfolio_backtest.py --days 90\n"
        "3. python -m src.main\n"
        "4. On first signal verify: Stop + TP + Partial TP logged\n"
        "5. After 24h: review logs/trading.log + data/bot_state.json\n"
        "6. Live only if paper equity stable and no order errors\n\n"
        f"[cyan]Profile:[/] {'pure_edge' if cfg.get('competition', {}).get('pure_edge') else 'competition'}\n"
        f"[cyan]Pairs:[/] {', '.join(symbols)}\n"
        f"Checks: {ok} ok, {fail} need attention",
        title="GO",
    ))
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
