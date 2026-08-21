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
    # These three used to assert a config we deliberately stopped running — ETH
    # disabled, partial TP on, keepalive capped — so the preflight cried FIX on
    # every green run and stopped being read. A checklist that is wrong by default
    # is worse than no checklist: it trains you to ignore it. Each is replaced by
    # the invariant the current config actually holds.

    # Historical Season 1 used exactly these 8. The current set is unverified; this
    # only keeps the rehearsal configuration inside its declared universe.
    WEEX_PAIRS = {"BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "LTC"}
    bases = {s.split("/")[0] for s in symbols}
    check("Pairs stay inside the S1 rehearsal set", bases <= WEEX_PAIRS,
          f"not permitted: {sorted(bases - WEEX_PAIRS)}" if bases - WEEX_PAIRS else f"{len(bases)} pairs")
    check("Breakouts off", not cfg.get("strategy", {}).get("breakout", {}).get("enabled", False))

    # Keepalive must stay OFF while the AI decides: it only ever ran on the rules
    # path, and a code-generated order would not carry AI-log linkage.
    ai_on = cfg.get("ai", {}).get("enabled", False)
    ka_on = cfg.get("strategy", {}).get("keepalive", {}).get("enabled", False)
    check("Keepalive off under AI", not (ai_on and ka_on),
          "a heartbeat order has no ai-log" if (ai_on and ka_on) else "")

    # Paper realism (added 2026-08-03). Off means the paper book is being flattered:
    # free carry and fills at touches a real queue never gives. Only legitimate when
    # deliberately reproducing a pre-2026-08-03 run.
    ex = cfg.get("execution", {}) or {}
    funding_on = ex.get("paper_funding", True)
    check("Paper funding charged", funding_on,
          "" if funding_on else "OFF = positions are free to hold, unlike live")
    ticks = float(ex.get("paper_fill_through_ticks", 1))
    check("Fills need trade-through", ticks >= 1,
          f"{ticks} tick(s)" if ticks >= 1 else "0 = a touch fills, inflating the fill rate")
    policy_id = str((cfg.get("evaluation", {}) or {}).get("policy_id") or "").strip()
    check(
        "Forward-test policy cohort set",
        bool(policy_id),
        policy_id or "set evaluation.policy_id and bump it after behavior changes",
    )

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

    readiness = (
        "[green]Paper-runtime readiness checks passed.[/]"
        if fail == 0
        else "[red]Paper runtime still needs attention.[/]"
    )
    console.print(Panel.fit(
        f"{readiness}\n\n"
        "[bold]Session plan[/]\n"
        "1. python test_bot.py\n"
        "2. python run_portfolio_backtest.py --days 90\n"
        "3. python -m src.main\n"
        "4. On first fill verify: Stop + TP + AI decision link logged\n"
        "5. After 24h: review logs/trading.log + data/bot_state.json\n"
        "6. Review saved paper evidence separately before considering live mode\n\n"
        "[bold]Scope:[/] This checklist covers paper-runtime readiness only.\n"
        "It is not a live GO, alpha validation, or competition GO.\n\n"
        f"[cyan]Profile:[/] {'pure_edge' if cfg.get('competition', {}).get('pure_edge') else 'competition'}\n"
        f"[cyan]Pairs:[/] {', '.join(symbols)}\n"
        f"Checks: {ok} ok, {fail} need attention",
        title="PAPER RUNTIME READY" if fail == 0 else "PAPER RUNTIME NOT READY",
    ))
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
