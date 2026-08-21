"""Non-mutating readiness gate for the WEEX competition rehearsal bot."""

import argparse
import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, ".")

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
import yaml

console = Console()

LIVE_MIN_CLOSED_TRADES = 40
LIVE_MIN_PROFIT_FACTOR = 1.25
LIVE_RECENT_WINDOW = 20


def configured_policy_id(cfg):
    """Explicit forward-test cohort; operators bump it when behavior changes."""
    evaluation = cfg.get("evaluation") if isinstance(cfg, dict) else None
    return str((evaluation or {}).get("policy_id") or "").strip()


def paper_evidence_from_state(state, policy_id=None):
    """Extract finite closed-trade evidence, optionally for one policy cohort."""
    risk = state.get("risk") if isinstance(state, dict) else None
    history = risk.get("trade_history") if isinstance(risk, dict) else None
    rows = history if isinstance(history, list) else []
    pnls = []
    invalid_rows = 0
    other_policy_rows = 0
    for row in rows:
        if not isinstance(row, dict):
            invalid_rows += 1
            continue
        if policy_id is not None and str(row.get("policy_id") or "") != policy_id:
            other_policy_rows += 1
            continue
        try:
            pnl = float(row.get("pnl"))
        except (TypeError, ValueError, OverflowError):
            invalid_rows += 1
            continue
        if not math.isfinite(pnl):
            invalid_rows += 1
            continue
        pnls.append(pnl)

    gross_profit = sum(pnl for pnl in pnls if pnl > 0)
    gross_loss = -sum(pnl for pnl in pnls if pnl < 0)
    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss
    elif gross_profit > 0:
        profit_factor = math.inf
    else:
        profit_factor = None

    recent = pnls[-LIVE_RECENT_WINDOW:]
    return {
        "history_present": isinstance(history, list),
        "closed_trades": len(pnls),
        "invalid_rows": invalid_rows,
        "other_policy_rows": other_policy_rows,
        "policy_id": policy_id,
        "profit_factor": profit_factor,
        "recent_count": len(recent),
        "recent_net_pnl": sum(recent),
    }


def live_paper_evidence_gates(evidence):
    """Minimum saved-paper thresholds; these do not establish trading alpha."""
    profit_factor = evidence.get("profit_factor")
    return {
        "closed_trades": evidence.get("closed_trades", 0) >= LIVE_MIN_CLOSED_TRADES,
        "profit_factor": (
            isinstance(profit_factor, (int, float))
            and not math.isnan(profit_factor)
            and profit_factor >= LIVE_MIN_PROFIT_FACTOR
        ),
        "recent_net_pnl": (
            evidence.get("recent_count", 0) >= LIVE_RECENT_WINDOW
            and evidence.get("recent_net_pnl", 0) > 0
        ),
    }


def competition_rules_verified(cfg):
    """Require an explicit acknowledgement of the official current rulebook."""
    competition = cfg.get("competition") if isinstance(cfg, dict) else None
    return isinstance(competition, dict) and competition.get("rules_verified") is True


def load_saved_paper_evidence(cfg, base_dir=Path(".")):
    """Load configured state without mutating it; return evidence, path, error."""
    configured = (cfg.get("logging") or {}).get("state_file") if isinstance(cfg, dict) else None
    path = Path(configured or "data/bot_state.json")
    if not path.is_absolute():
        path = base_dir / path
    try:
        with open(path, encoding="utf-8") as handle:
            state = json.load(handle)
        if not isinstance(state, dict):
            raise ValueError("state root is not an object")
        policy_id = configured_policy_id(cfg)
        # A missing ID deliberately matches no real cohort. It must fail readiness
        # instead of silently pooling every historical implementation.
        selected = policy_id or "__missing_policy_id__"
        return paper_evidence_from_state(state, policy_id=selected), path, None
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return paper_evidence_from_state({}, policy_id=configured_policy_id(cfg)), path, str(exc)


def _format_profit_factor(value):
    if value is None:
        return "N/A"
    if math.isinf(value):
        return "inf (no losing trades)"
    return f"{value:.3f}"


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
        check(
            "forward-test policy cohort set",
            bool(configured_policy_id(cfg)),
            configured_policy_id(cfg) or "set evaluation.policy_id and bump it after behavior changes",
        )
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
                "configured rehearsal leverage <= 20x",
                cfg.get("trading", {}).get("max_leverage", 99) <= 20,
            )
            check(
                "official current competition rules verified",
                competition_rules_verified(cfg),
                "set competition.rules_verified=true only after reviewing the official current rulebook",
            )

    if args.target == "live":
        evidence, state_path, state_error = load_saved_paper_evidence(cfg)
        gates = live_paper_evidence_gates(evidence)
        source = str(state_path)
        if state_error:
            source += f"; unreadable: {state_error}"
        invalid = evidence["invalid_rows"]
        invalid_note = f"; skipped invalid rows={invalid}" if invalid else ""
        cohort_note = (
            f"; policy={evidence.get('policy_id') or '<missing>'}; "
            f"excluded other-policy rows={evidence.get('other_policy_rows', 0)}"
        )
        check(
            "saved paper: >=40 closed trades",
            gates["closed_trades"],
            f"finite closed trades={evidence['closed_trades']}; {source}"
            f"{cohort_note}{invalid_note}",
        )
        check(
            "saved paper: full-sample PF >=1.25",
            gates["profit_factor"],
            f"PF={_format_profit_factor(evidence['profit_factor'])}; "
            f"n={evidence['closed_trades']}; {source}{cohort_note}",
        )
        check(
            "saved paper: recent-20 net PnL >0",
            gates["recent_net_pnl"],
            f"recent sample={evidence['recent_count']}/20; "
            f"net=${evidence['recent_net_pnl']:.2f}; {source}",
        )

    # Env
    from dotenv import load_dotenv
    load_dotenv()
    key = os.getenv("WEEX_API_KEY", "")
    secret = os.getenv("WEEX_API_SECRET", "")
    phrase = os.getenv("WEEX_API_PASSPHRASE") or os.getenv("WEEX_PASSPHRASE") or ""
    if args.target == "live":
        check(".env API key present", bool(key) and key != "your_api_key_here", "set WEEX_API_KEY")
        check(".env secret present", bool(secret) and secret != "your_api_secret_here")
        check("passphrase present", bool(phrase) and "your_passphrase" not in phrase)
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
    else:
        check(
            "WEEX credentials optional in paper mode",
            True,
            "public market data works without private trading credentials",
        )

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
        "Saved-paper gates are minimum runtime/live-readiness thresholds only.\n"
        "They do not prove alpha or competition readiness.\n\n"
        "Live path (only after paper is clean):\n"
        "  1. Keep trading.mode: paper while this gate is red\n"
        "  2. Verify the current event's UID, API key and VPS IP are allowlisted\n"
        "  3. Start leverage 3–5\n"
        "  4. Confirm UploadAiLog says upload success before another entry\n"
        "  5. Only then arm live mode and watch the first three round trips",
        title="Next Steps",
    ))
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
