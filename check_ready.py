"""Paper/Live readiness checklist for WEEX AI Wars bot v8.3"""

import os
import sys
from pathlib import Path

sys.path.insert(0, ".")

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
import yaml

console = Console()


def main():
    ok = 0
    fail = 0
    rows = []

    def check(name, cond, detail=""):
        nonlocal ok, fail
        if cond:
            ok += 1
            rows.append((name, "[green]PASS[/]", detail))
        else:
            fail += 1
            rows.append((name, "[red]FAIL[/]", detail))

    # Config
    cfg_path = Path("config.yaml")
    check("config.yaml exists", cfg_path.exists())
    cfg = {}
    if cfg_path.exists():
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f) or {}
        mode = cfg.get("trading", {}).get("mode", "?")
        check("trading.mode set", mode in ("paper", "live"), f"mode={mode}")
        check("has symbols", bool(cfg.get("trading", {}).get("symbols")), str(cfg.get("trading", {}).get("symbols")))
        check("breakout disabled (v8.3)", not cfg.get("strategy", {}).get("breakout", {}).get("enabled", False))
        check("HTF bias on", cfg.get("competition", {}).get("htf_directional_bias", False))
        check("risk/trade <= 2%", cfg.get("risk", {}).get("max_risk_per_trade", 1) <= 0.02)
        check("max DD <= 20%", cfg.get("risk", {}).get("max_drawdown", 1) <= 0.20)

    # Env
    from dotenv import load_dotenv
    load_dotenv()
    key = os.getenv("WEEX_API_KEY", "")
    secret = os.getenv("WEEX_API_SECRET", "")
    phrase = os.getenv("WEEX_API_PASSPHRASE") or os.getenv("WEEX_PASSPHRASE") or ""
    check(".env API key present", bool(key) and key != "your_api_key_here", "set WEEX_API_KEY")
    check(".env secret present", bool(secret) and secret != "your_api_secret_here")
    check("passphrase present", bool(phrase) and "your_passphrase" not in phrase)

    # Imports
    try:
        from src.strategies.composite import CompositeStrategy
        from src.risk.manager import RiskManager
        from src.core.models import Position, Side, Signal
        check("core imports", True)
    except Exception as e:
        check("core imports", False, str(e))
        CompositeStrategy = None

    # SL guard
    try:
        p = Position("X", Side.SHORT, 100, 1, 5, 0, 0)
        check("zero SL safe on shorts", p.should_stop_loss(100) is False)
    except Exception as e:
        check("zero SL safe on shorts", False, str(e))

    # Strength sizing
    try:
        rm = RiskManager(cfg)

        class A:
            equity = 10000
            available_margin = 10000
            positions = []
            balance = 10000

        full = Signal("BTC/USDT:USDT", Side.LONG, 0.8, "t", 50000, 49000, 52000, 5, "x")
        tiny = Signal("SOL/USDT:USDT", Side.LONG, 0.1, "ka", 150, 148, 155, 5, "x")
        sf = rm.calculate_position_size(full, A())
        st = rm.calculate_position_size(tiny, A())
        check("strength scales size", st * 150 < sf * 50000 * 0.5, f"full=${sf*50000:.0f} ka=${st*150:.0f}")
    except Exception as e:
        check("strength scales size", False, str(e))

    # Exchange class loads (optional network)
    try:
        import ccxt
        has_weex = hasattr(ccxt, "weex")
        check("ccxt has weex", has_weex, "upgrade ccxt if missing")
    except Exception as e:
        check("ccxt import", False, str(e))

    table = Table(title="Readiness Checklist", show_header=True)
    table.add_column("Check")
    table.add_column("Status")
    table.add_column("Detail")
    for r in rows:
        table.add_row(*r)
    console.print(table)

    color = "green" if fail == 0 else "yellow" if fail <= 2 else "red"
    console.print(Panel.fit(
        f"[{color}]{ok} passed, {fail} failed[/]\n\n"
        "Paper path:\n"
        "  1. Keep trading.mode: paper\n"
        "  2. python test_bot.py\n"
        "  3. python -m src.main\n"
        "  4. Confirm every fill logs Stop + TP\n\n"
        "Live path (only after paper is clean):\n"
        "  1. Fill .env keys\n"
        "  2. trading.mode: live\n"
        "  3. Start leverage 3–5\n"
        "  4. Watch first 3 orders manually",
        title="Next Steps",
    ))
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
