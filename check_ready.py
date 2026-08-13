"""Non-mutating readiness gate for the WEEX AI Wars bot."""

import argparse
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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target", choices=("paper", "live"), default="live",
        help="assess this target without changing config or placing an order",
    )
    args = parser.parse_args()
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
        if args.target == "live":
            check(
                "live remains disarmed for rehearsal",
                mode == "paper",
                "keep trading.mode=paper until the supervised minimum-size test",
            )
            check("AI decision layer enabled", bool((cfg.get("ai") or {}).get("enabled")))
            check(
                "maker entries enabled",
                bool((cfg.get("execution") or {}).get("maker_entries")),
            )
            check(
                "venue leverage ceiling <= 20x",
                cfg.get("trading", {}).get("max_leverage", 99) <= 20,
            )

    # Env
    from dotenv import load_dotenv
    load_dotenv()
    key = os.getenv("WEEX_API_KEY", "")
    secret = os.getenv("WEEX_API_SECRET", "")
    phrase = os.getenv("WEEX_API_PASSPHRASE") or os.getenv("WEEX_PASSPHRASE") or ""
    check(".env API key present", bool(key) and key != "your_api_key_here", "set WEEX_API_KEY")
    check(".env secret present", bool(secret) and secret != "your_api_secret_here")
    check("passphrase present", bool(phrase) and "your_passphrase" not in phrase)
    if args.target == "live":
        check("DeepSeek key present", bool(os.getenv("DEEPSEEK_API_KEY", "")))
        try:
            from src.ai.wars_log import WeexAILogUploader

            uploader = WeexAILogUploader(enabled=True)
            ready, why = uploader.readiness()
            check("official WEEX AI-log delivery ready", ready, why)
        except Exception as e:
            check("official WEEX AI-log delivery ready", False, str(e))

        # Credentials being non-empty proves almost nothing.  These are private,
        # read-only requests: they cannot place/cancel an order or move funds, but
        # they distinguish a configured bot from an account that can actually run.
        if key and secret and phrase:
            try:
                import ccxt

                venue = ccxt.weex({
                    "apiKey": key,
                    "secret": secret,
                    "password": phrase,
                    "enableRateLimit": True,
                    "options": {"defaultType": "swap"},
                })
                balance = venue.fetch_balance({"type": "swap"})
                usdt = balance.get("USDT") or {}
                total = float(usdt.get("total") or 0)
                check(
                    "authenticated WEEX futures balance",
                    True,
                    f"USDT total={total:.2f}",
                )
                check(
                    "positive WEEX futures balance",
                    total > 0,
                    f"USDT total={total:.2f}; fund/transfer into USDT futures",
                )

                account_cfg = venue.contractprivate_get_capi_v3_account_accountconfig()
                check(
                    "WEEX account can trade futures",
                    account_cfg.get("canTrade") is True,
                    f"canTrade={account_cfg.get('canTrade')}",
                )

                response = venue.contract_get_capi_v3_market_apitradingsymbols()
                raw_symbols = response.get("symbols") if isinstance(response, dict) else response
                tradable = set()
                for row in raw_symbols or []:
                    if isinstance(row, str):
                        tradable.add(row)
                    elif isinstance(row, dict):
                        tradable.add(str(row.get("symbol") or ""))
                configured = {
                    str(s).split("/")[0] + "USDT"
                    for s in (cfg.get("trading", {}).get("symbols") or [])
                }
                missing = sorted(configured - tradable)
                check(
                    "configured symbols allow API trading",
                    not missing,
                    "missing=" + str(missing) if missing else f"{len(configured)} checked",
                )
            except Exception as e:
                check("authenticated WEEX futures balance", False, str(e))
                check("positive WEEX futures balance", False, "private probe failed")
                check("WEEX account can trade futures", False, "private probe failed")
                check("configured symbols allow API trading", False, "private probe failed")
        else:
            for name in (
                "authenticated WEEX futures balance",
                "positive WEEX futures balance",
                "WEEX account can trade futures",
                "configured symbols allow API trading",
            ):
                check(name, False, "credentials missing; probe not attempted")

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
        "  1. Keep trading.mode: paper while this gate is red\n"
        "  2. Verify the account is AI-Wars allowlisted\n"
        "  3. Start leverage 3–5\n"
        "  4. Confirm UploadAiLog says upload success before another entry\n"
        "  5. Only then arm live mode and watch the first three round trips",
        title="Next Steps",
    ))
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
