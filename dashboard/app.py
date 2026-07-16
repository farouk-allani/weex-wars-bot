"""WEEX AI Wars — Monitoring Dashboard API

Run:
  python -m dashboard.app
  # → http://127.0.0.1:8787
"""

from __future__ import annotations

import io
import json
import re
import sys
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal

import yaml
from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# Project root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

app = FastAPI(title="WEEX AI Wars Dashboard", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")


@app.get("/sw.js")
def service_worker():
    """Serve SW from root so scope covers the whole app (required for PWA)."""
    path = STATIC / "sw.js"
    return FileResponse(
        path,
        media_type="application/javascript",
        headers={
            "Service-Worker-Allowed": "/",
            "Cache-Control": "no-cache",
        },
    )


@app.get("/manifest.webmanifest")
def manifest_alias():
    return FileResponse(
        STATIC / "manifest.webmanifest",
        media_type="application/manifest+json",
        headers={"Cache-Control": "no-cache"},
    )


def _config_path() -> Path:
    return ROOT / "config.yaml"


def _state_path(cfg: dict | None = None) -> Path:
    if cfg is None:
        cfg = load_config()
    rel = cfg.get("logging", {}).get("state_file", "data/bot_state.json")
    p = Path(rel)
    return p if p.is_absolute() else ROOT / p


def _log_path(cfg: dict | None = None) -> Path:
    if cfg is None:
        cfg = load_config()
    rel = cfg.get("logging", {}).get("file", "logs/trading.log")
    p = Path(rel)
    return p if p.is_absolute() else ROOT / p


def load_config() -> dict:
    path = _config_path()
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_state() -> dict:
    path = _state_path()
    if not path.exists():
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception:
        return {}


def _data_dir() -> Path:
    return _state_path().parent


# incremental tail cache for the (large) liquidation tape: path -> progress
_liq_cache: dict[str, dict] = {}


def _scan_liq_file(path: Path) -> dict:
    key = str(path)
    st = path.stat()
    c = _liq_cache.get(key)
    if c is None or st.st_size < c["offset"]:
        c = {"offset": 0, "liqs": 0, "usd": 0.0, "last": None, "biggest": None}
    if st.st_size > c["offset"]:
        with open(path, encoding="utf-8", errors="ignore") as f:
            f.seek(c["offset"])
            for line in f:
                if '"liq"' not in line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                c["liqs"] += 1
                c["usd"] += r.get("usd") or 0
                c["last"] = r
                if c["biggest"] is None or (r.get("usd") or 0) > (c["biggest"].get("usd") or 0):
                    c["biggest"] = r
            c["offset"] = f.tell()
    _liq_cache[key] = c
    return c


def build_edges(cfg: dict) -> dict[str, Any]:
    d = _data_dir()
    out: dict[str, Any] = {"carry": None, "carry_paper": None, "liq": None}

    fwd = d / "carry_forward.jsonl"
    if fwd.exists():
        lines = fwd.read_text(encoding="utf-8").strip().splitlines()
        if lines:
            try:
                snap = json.loads(lines[-1])
                snap["snapshots"] = len(lines)
                out["carry"] = snap
            except json.JSONDecodeError:
                pass

    pstate = d / "carry_paper_state.json"
    paper: dict[str, Any] = {}
    if pstate.exists():
        try:
            paper = json.loads(pstate.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            paper = {}
    trades_f = d / "carry_paper_trades.jsonl"
    trades = []
    if trades_f.exists():
        for line in trades_f.read_text(encoding="utf-8").strip().splitlines():
            try:
                trades.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if paper or trades:
        out["carry_paper"] = {
            "equity": paper.get("equity"),
            "book": paper.get("book"),
            "closed": len(trades),
            "net_total": sum(t.get("net", 0) for t in trades),
            "last_trades": trades[-5:],
        }

    liq_dir = d / "liq_forward"
    if liq_dir.exists():
        files = sorted(liq_dir.glob("*.jsonl"))
        total, usd, last, biggest, newest_age = 0, 0.0, None, None, None
        for f in files:
            c = _scan_liq_file(f)
            total += c["liqs"]
            usd += c["usd"]
            if c["last"]:
                last = c["last"]
            if c["biggest"] and (biggest is None or c["biggest"]["usd"] > biggest["usd"]):
                biggest = c["biggest"]
        if files:
            newest_age = max(0, datetime.now(timezone.utc).timestamp() - files[-1].stat().st_mtime)
        out["liq"] = {
            "files": len(files),
            "forced_orders": total,
            "notional_usd": usd,
            "last": last,
            "biggest": biggest,
            "recorder_age_sec": newest_age,
        }
    return out


def parse_log_lines(text: str, limit: int = 150) -> list[dict]:
    lines = text.splitlines()
    lines = lines[-limit:]
    out = []
    # 2026-07-13 12:00:00 | INFO    | message
    pat = re.compile(
        r"^(?P<ts>\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})\s*\|\s*(?P<level>\w+)\s*\|\s*(?P<msg>.*)$"
    )
    for line in lines:
        line = line.rstrip()
        if not line:
            continue
        m = pat.match(line)
        if m:
            level = m.group("level").upper()
            msg = m.group("msg")
            out.append({"ts": m.group("ts"), "level": level, "message": msg, "raw": line})
        else:
            out.append({"ts": "", "level": "INFO", "message": line, "raw": line})
    return out


def build_metrics(state: dict, cfg: dict) -> dict[str, Any]:
    risk = state.get("risk") or {}
    history = risk.get("trade_history") or []
    pair_sharpes = risk.get("pair_sharpes") or {}
    strategy_pnls = risk.get("strategy_pnls") or {}
    account = state.get("account") or {}

    pnls = [float(t.get("pnl") or 0) for t in history]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    total_pnl = sum(pnls)
    win_rate = len(wins) / len(pnls) if pnls else 0.0
    avg_win = sum(wins) / len(wins) if wins else 0.0
    avg_loss = sum(losses) / len(losses) if losses else 0.0
    best = max(pnls) if pnls else 0.0
    worst = min(pnls) if pnls else 0.0

    # Equity curve: prefer live ticks (open pos mark-to-market), else closed trades
    initial = float(cfg.get("backtest", {}).get("initial_capital", 10000))
    ticks = state.get("equity_ticks") or []
    if ticks:
        curve = []
        peak = initial
        max_dd = 0.0
        for i, t in enumerate(ticks):
            eq = float(t.get("equity") or initial)
            peak = max(peak, eq)
            dd = (peak - eq) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)
            curve.append({
                "i": i,
                "equity": round(eq, 2),
                "pnl": round(float(t.get("unrealized") or 0), 2),
                "t": t.get("t"),
            })
        equity = float(ticks[-1].get("equity") or initial)
    else:
        equity = initial
        curve = [{"i": 0, "equity": equity, "pnl": 0.0}]
        peak = equity
        max_dd = 0.0
        for i, p in enumerate(pnls, 1):
            equity += p
            peak = max(peak, equity)
            dd = (peak - equity) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)
            curve.append({"i": i, "equity": round(equity, 2), "pnl": round(p, 2)})

    # Live account overrides equity when bot is writing paper/live snapshots
    if account.get("equity") is not None:
        equity = float(account["equity"])
    balance = float(account.get("balance") if account.get("balance") is not None else equity)
    unrealized = float(account.get("unrealized_pnl") or 0)
    open_positions = account.get("positions") or []
    # Closed + unrealized for "session" PnL feel
    session_pnl = (equity - initial) if account else total_pnl

    # Strategy stats
    strat_stats = []
    for name, arr in strategy_pnls.items():
        arr = [float(x) for x in arr]
        strat_stats.append({
            "name": name,
            "trades": len(arr),
            "pnl": round(sum(arr), 2),
            "wins": sum(1 for x in arr if x > 0),
            "win_rate": (sum(1 for x in arr if x > 0) / len(arr)) if arr else 0,
        })
    strat_stats.sort(key=lambda x: x["pnl"], reverse=True)

    # Pair stats
    pair_stats = []
    for sym, arr in pair_sharpes.items():
        arr = [float(x) for x in arr]
        name = sym.split("/")[0] if "/" in sym else sym
        pair_stats.append({
            "symbol": sym,
            "name": name,
            "trades": len(arr),
            "pnl": round(sum(arr), 2),
            "wins": sum(1 for x in arr if x > 0),
        })
    pair_stats.sort(key=lambda x: x["pnl"], reverse=True)

    # Recent trades (newest first)
    trades = []
    for t in reversed(history[-50:]):
        trades.append({
            "symbol": t.get("symbol"),
            "side": t.get("side"),
            "entry": t.get("entry_price"),
            "exit": t.get("exit_price"),
            "size": t.get("size"),
            "pnl": t.get("pnl"),
            "pnl_pct": t.get("pnl_pct"),
            "strategy": t.get("strategy"),
            "exit_reason": t.get("exit_reason"),
            "leverage": t.get("leverage"),
        })

    # Exit reason breakdown
    reasons: dict[str, int] = {}
    for t in history:
        r = t.get("exit_reason") or "unknown"
        reasons[r] = reasons.get(r, 0) + 1

    saved_at = state.get("saved_at")
    age_sec = None
    bot_alive = False
    if saved_at:
        try:
            ts = datetime.fromisoformat(str(saved_at).replace("Z", "+00:00"))
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            age_sec = (datetime.now(timezone.utc) - ts).total_seconds()
            bot_alive = age_sec < 300  # state written within 5 min
        except Exception:
            pass

    cooldown = risk.get("cooldown_until")
    is_killed = bool(risk.get("is_killed"))

    return {
        "initial_capital": initial,
        "equity": round(equity, 2),
        "balance": round(balance, 2),
        "unrealized_pnl": round(unrealized, 2),
        "total_pnl": round(session_pnl if account else total_pnl, 2),
        "closed_pnl": round(total_pnl, 2),
        "pnl_pct": round((equity - initial) / initial * 100, 3) if initial else 0,
        "win_rate": win_rate,
        "total_trades": len(pnls),
        "wins": len(wins),
        "losses": len(losses),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "best_trade": round(best, 2),
        "worst_trade": round(worst, 2),
        "max_drawdown": round(max_dd, 4),
        "peak_equity": round(float(risk.get("peak_equity") or peak or equity), 2),
        "daily_pnl": round(float(risk.get("daily_pnl") or 0), 2),
        "consecutive_losses": int(risk.get("consecutive_losses") or 0),
        "consecutive_wins": int(risk.get("consecutive_wins") or 0),
        "is_killed": is_killed,
        "cooldown_until": cooldown,
        "cycle_count": state.get("cycle_count"),
        "saved_at": saved_at,
        "state_age_sec": age_sec,
        "bot_alive": bot_alive,
        "has_state": bool(state),
        "open_positions": open_positions,
        "open_count": len(open_positions),
        "equity_curve": curve,
        "trades": trades,
        "strategy_stats": strat_stats,
        "pair_stats": pair_stats,
        "exit_reasons": reasons,
        "demo": False,
    }


def demo_metrics(cfg: dict) -> dict[str, Any]:
    """Pretty sample data when bot hasn't written state yet."""
    initial = float(cfg.get("backtest", {}).get("initial_capital", 10000))
    # synthetic mild up curve
    import random
    random.seed(42)
    equity = initial
    curve = [{"i": 0, "equity": equity, "pnl": 0.0}]
    pnls = []
    for i in range(1, 29):
        p = random.choice([1.2, -0.8, 0.9, -1.1, 2.5, -0.5, 1.8, -1.4, 0.6, 3.1])
        pnls.append(p)
        equity += p
        curve.append({"i": i, "equity": round(equity, 2), "pnl": p})

    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    return {
        "initial_capital": initial,
        "equity": round(equity, 2),
        "total_pnl": round(sum(pnls), 2),
        "pnl_pct": round(sum(pnls) / initial * 100, 3),
        "win_rate": len(wins) / len(pnls),
        "total_trades": len(pnls),
        "wins": len(wins),
        "losses": len(losses),
        "avg_win": round(sum(wins) / len(wins), 2),
        "avg_loss": round(sum(losses) / len(losses), 2),
        "best_trade": round(max(pnls), 2),
        "worst_trade": round(min(pnls), 2),
        "max_drawdown": 0.012,
        "peak_equity": round(max(c["equity"] for c in curve), 2),
        "daily_pnl": 1.2,
        "consecutive_losses": 0,
        "consecutive_wins": 2,
        "is_killed": False,
        "cooldown_until": None,
        "cycle_count": 0,
        "saved_at": None,
        "state_age_sec": None,
        "bot_alive": False,
        "has_state": False,
        "equity_curve": curve,
        "trades": [
            {
                "symbol": "BTC/USDT:USDT", "side": "long", "entry": 97500,
                "exit": 98100, "size": 0.02, "pnl": 12.0, "pnl_pct": 1.2,
                "strategy": "mean_reversion", "exit_reason": "take_profit", "leverage": 5,
            },
            {
                "symbol": "SOL/USDT:USDT", "side": "short", "entry": 148.2,
                "exit": 149.1, "size": 2.0, "pnl": -1.8, "pnl_pct": -0.6,
                "strategy": "keepalive_vwap", "exit_reason": "stop_loss", "leverage": 5,
            },
        ],
        "strategy_stats": [
            {"name": "mean_reversion", "trades": 12, "pnl": 18.4, "wins": 7, "win_rate": 0.58},
            {"name": "keepalive_vwap", "trades": 10, "pnl": -4.2, "wins": 4, "win_rate": 0.4},
            {"name": "keepalive_bb", "trades": 6, "pnl": -1.1, "wins": 2, "win_rate": 0.33},
        ],
        "pair_stats": [
            {"symbol": "BTC/USDT:USDT", "name": "BTC", "trades": 14, "pnl": 22.1, "wins": 8},
            {"symbol": "SOL/USDT:USDT", "name": "SOL", "trades": 14, "pnl": -9.0, "wins": 5},
        ],
        "exit_reasons": {"stop_loss": 16, "take_profit": 8, "trailing_stop": 4},
        "balance": initial,
        "unrealized_pnl": 0.0,
        "closed_pnl": round(sum(pnls), 2),
        "open_positions": [],
        "open_count": 0,
        "demo": True,
    }


@app.get("/", response_class=HTMLResponse)
def index():
    html = STATIC / "index.html"
    return FileResponse(html)


@app.get("/api/overview")
def overview():
    cfg = load_config()
    state = load_state()
    # Use real state if bot has written anything (risk and/or live account)
    metrics = (
        build_metrics(state, cfg)
        if (state.get("risk") or state.get("account") or state.get("equity_ticks"))
        else demo_metrics(cfg)
    )

    trading = cfg.get("trading", {})
    risk = cfg.get("risk", {})
    strat = cfg.get("strategy", {})
    comp = cfg.get("competition", {})

    return {
        "bot": {
            "name": "WEEX AI Wars Bot",
            "version": "v8.5",
            "mode": trading.get("mode", "paper"),
            "profile": "pure_edge" if comp.get("pure_edge") else "competition",
            "symbols": trading.get("symbols", []),
            "timeframe": trading.get("timeframe", "1h"),
            "higher_timeframe": trading.get("higher_timeframe", "4h"),
            "leverage": trading.get("default_leverage", 5),
            "alive": metrics.get("bot_alive"),
            "has_state": metrics.get("has_state"),
            "demo": metrics.get("demo"),
            "saved_at": metrics.get("saved_at"),
            "state_age_sec": metrics.get("state_age_sec"),
            "cycle_count": metrics.get("cycle_count"),
        },
        "risk_config": {
            "max_risk_per_trade": risk.get("max_risk_per_trade"),
            "max_drawdown": risk.get("max_drawdown"),
            "daily_loss_limit": risk.get("daily_loss_limit"),
            "max_open_positions": risk.get("max_open_positions"),
            "partial_tp": risk.get("partial_tp_enabled"),
            "cooldown_hours": risk.get("cooldown_hours"),
        },
        "strategy_config": {
            "mean_reversion": strat.get("mean_reversion", {}).get("enabled"),
            "breakout": strat.get("breakout", {}).get("enabled"),
            "keepalive": strat.get("keepalive", {}).get("enabled"),
            "disabled_pairs": comp.get("disabled_pairs", []),
        },
        "metrics": metrics,
        "server_time": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/logs")
def logs(lines: int = Query(120, ge=10, le=500)):
    path = _log_path()
    if not path.exists():
        return {
            "lines": [
                {
                    "ts": "",
                    "level": "INFO",
                    "message": "No log file yet. Start the bot with: python -m src.main",
                    "raw": "",
                }
            ],
            "path": str(path),
            "exists": False,
        }
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return {"lines": [], "path": str(path), "exists": True, "error": str(e)}
    return {"lines": parse_log_lines(text, lines), "path": str(path), "exists": True}


@app.get("/api/config")
def config_view():
    return load_config()


@app.get("/api/health")
def health():
    cfg = load_config()
    state = load_state()
    log = _log_path(cfg)
    st = _state_path(cfg)
    return {
        "ok": True,
        "config_exists": _config_path().exists(),
        "state_exists": st.exists(),
        "log_exists": log.exists(),
        "state_path": str(st),
        "log_path": str(log),
        "has_trades": bool((state.get("risk") or {}).get("trade_history")),
    }


@app.get("/api/edges")
def edges():
    """Edge Lab: carry paper book + funding gate + liquidation collector."""
    cfg = load_config()
    return build_edges(cfg)


# ---------- Exports (for iteration / hand logs to AI) ----------

Period = Literal["all", "today", "24h", "7d", "30d"]

_LOG_TS = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})\s*\|"
)


def _period_cutoff(period: str) -> datetime | None:
    """Naive local timestamps in log file are treated as-is; cutoff uses local now."""
    now = datetime.now()
    if period in ("all", "", None):
        return None
    if period == "today":
        return now.replace(hour=0, minute=0, second=0, microsecond=0)
    if period == "24h":
        return now - timedelta(hours=24)
    if period == "7d":
        return now - timedelta(days=7)
    if period == "30d":
        return now - timedelta(days=30)
    return None


def _filter_log_text(text: str, period: str) -> str:
    cutoff = _period_cutoff(period)
    if cutoff is None:
        return text
    out: list[str] = []
    last_kept = False
    for line in text.splitlines(keepends=True):
        m = _LOG_TS.match(line)
        if m:
            try:
                ts = datetime.strptime(m.group("ts"), "%Y-%m-%d %H:%M:%S")
                last_kept = ts >= cutoff
            except ValueError:
                last_kept = True
            if last_kept:
                out.append(line)
        else:
            # continuation / plain lines stick to previous decision
            if last_kept or not out:
                if last_kept:
                    out.append(line)
    return "".join(out) if out else f"# No log lines for period={period}\n"


def _export_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


@app.get("/api/export/state")
def export_state():
    """Download current bot_state.json."""
    path = _state_path()
    if not path.exists():
        raise HTTPException(status_code=404, detail="bot_state.json not found yet")
    return FileResponse(
        path,
        media_type="application/json",
        filename=f"bot_state_{_export_stamp()}.json",
        headers={"Cache-Control": "no-store"},
    )


@app.get("/api/export/log")
def export_log(period: Period = Query("all", description="all | today | 24h | 7d | 30d")):
    """Download trading.log (optionally filtered by period)."""
    path = _log_path()
    if not path.exists():
        raise HTTPException(status_code=404, detail="trading.log not found yet")
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    filtered = _filter_log_text(text, period)
    name = f"trading_{period}_{_export_stamp()}.log"
    return Response(
        content=filtered,
        media_type="text/plain; charset=utf-8",
        headers={
            "Content-Disposition": f'attachment; filename="{name}"',
            "Cache-Control": "no-store",
        },
    )


@app.get("/api/export/bundle")
def export_bundle(period: Period = Query("7d", description="Log period in the zip")):
    """ZIP: bot_state.json + filtered trading.log + short README for AI review."""
    log_path = _log_path()
    state_path = _state_path()
    if not log_path.exists() and not state_path.exists():
        raise HTTPException(status_code=404, detail="No state or log files yet")

    buf = io.BytesIO()
    stamp = _export_stamp()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        if state_path.exists():
            zf.write(state_path, arcname="bot_state.json")
        if log_path.exists():
            text = log_path.read_text(encoding="utf-8", errors="replace")
            filtered = _filter_log_text(text, period)
            zf.writestr(f"trading_{period}.log", filtered)
        cfg = load_config()
        meta = {
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "period": period,
            "mode": cfg.get("trading", {}).get("mode"),
            "symbols": cfg.get("trading", {}).get("symbols"),
            "note": "Share this zip for the next bot iteration review.",
        }
        zf.writestr("export_meta.json", json.dumps(meta, indent=2))
        zf.writestr(
            "README.txt",
            "WEEX Wars bot export\n"
            f"period={period}\n"
            "Files:\n"
            "  bot_state.json  — trades, risk, account snapshot\n"
            f"  trading_{period}.log — filtered app log\n"
            "  export_meta.json — export context\n",
        )
    buf.seek(0)
    filename = f"weex_bot_export_{period}_{stamp}.zip"
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Cache-Control": "no-store",
        },
    )


@app.get("/api/export/info")
def export_info():
    """Sizes / existence for the download UI."""
    log_path = _log_path()
    state_path = _state_path()
    log_size = log_path.stat().st_size if log_path.exists() else 0
    state_size = state_path.stat().st_size if state_path.exists() else 0
    return {
        "log_exists": log_path.exists(),
        "state_exists": state_path.exists(),
        "log_size": log_size,
        "state_size": state_size,
        "log_path": str(log_path),
        "state_path": str(state_path),
        "periods": ["all", "today", "24h", "7d", "30d"],
    }


def _port_free(host: str, port: int) -> bool:
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((host, port))
            return True
        except OSError:
            return False


def pick_port(host: str = "127.0.0.1", preferred: int = 8787, tries: int = 20) -> int:
    """Return preferred port if free, else first free port in range."""
    for p in range(preferred, preferred + tries):
        if _port_free(host, p):
            return p
    raise RuntimeError(
        f"No free port found in {preferred}-{preferred + tries - 1}. "
        f"Kill the old process or pass --port."
    )


def main(argv: list[str] | None = None):
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="WEEX AI Wars monitoring dashboard")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default 127.0.0.1)")
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port (default: 8787, auto-picks next free if busy)",
    )
    parser.add_argument(
        "--no-auto-port",
        action="store_true",
        help="Fail if preferred port is busy instead of trying another",
    )
    args = parser.parse_args(argv)

    preferred = args.port if args.port is not None else 8787
    if args.no_auto_port:
        if not _port_free(args.host, preferred):
            print(f"ERROR: port {preferred} already in use.")
            print("  Fix options:")
            print(f"    1) Open existing dashboard:  http://{args.host}:{preferred}")
            print(f"    2) Free the port (PowerShell):")
            print(f"         netstat -ano | findstr :{preferred}")
            print(f"         taskkill /PID <pid> /F")
            print(f"    3) Use another port:  python run_dashboard.py --port 8788")
            raise SystemExit(1)
        port = preferred
    else:
        port = pick_port(args.host, preferred)
        if port != preferred:
            print(f"NOTE: port {preferred} busy — using {port} instead")
            print(f"      (old instance may still be at http://{args.host}:{preferred})")

    print("=" * 56)
    print("  WEEX AI Wars — Dashboard")
    print(f"  http://{args.host}:{port}")
    print("  Start bot separately: python -m src.main")
    print("=" * 56)
    uvicorn.run(
        "dashboard.app:app",
        host=args.host,
        port=port,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
