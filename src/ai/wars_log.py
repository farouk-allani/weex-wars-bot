"""WEEX AI Wars ai-log emitter.

The competition verifies Team AI trades by requiring an AI decision log per
AI-driven order, uploaded via the official trader skill's `--ai-log @file.json`
flow. Schema (weex-agent-skills-ai-wars/skills/weex-trader-skill/references/
ai-log-schema.md):

  stage        non-empty string
  model        EXACT provider-returned model id (no aliases/marketing names)
  input        complete original request: message array + market context,
               unsummarized, unflattened, unredacted
  output       ONLY the concrete action with the parameters that must match
               the final trade request (symbol/side/positionSide/type/
               quantity/price)
  explanation  <=1000 chars, tied to specific facts in input

This module builds and saves those files at order time (data/ai_logs/), so
when live rounds start the upload step just points at ready-made files.
Emission must never break trading: callers wrap in try/except.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

AI_LOGS_DIR = Path("data/ai_logs")


def build_ai_log(entry: dict, order: dict) -> dict:
    """entry = the logbook decision record; order = the linked order params."""
    side = str(order.get("side", "")).lower()
    return {
        "stage": "Strategy Generation",
        "model": entry.get("model") or "",
        "input": {
            # the literal request messages, preserved verbatim
            "messages": entry.get("messages") or [],
            "market_context": entry.get("context") or {},
        },
        "output": {
            "symbol": str(order.get("symbol", "")).replace("/", "").replace(":USDT", ""),
            "side": "BUY" if side in ("long", "buy") else "SELL",
            "positionSide": "LONG" if side in ("long", "buy") else "SHORT",
            "type": "LIMIT" if order.get("entry_price") else "MARKET",
            "quantity": order.get("size"),
            "price": order.get("entry_price"),
            "stopLoss": order.get("stop_loss"),
            "takeProfit": order.get("take_profit"),
        },
        "explanation": _explanation(entry, order),
    }


def _explanation(entry: dict, order: dict) -> str:
    """The model's own reasoning for THIS symbol, capped at the schema's 1000."""
    sym = order.get("symbol", "")
    reason = ""
    for d in entry.get("decisions") or []:
        if str(d.get("symbol", "")) in sym or sym in str(d.get("symbol", "")):
            reason = d.get("reason") or d.get("reasoning") or ""
            break
    if not reason:
        reason = entry.get("reasoning") or "Decision per attached model output."
    return reason[:1000]


def emit(entry: dict, order: dict, out_dir: Path | None = None) -> Path:
    """Write the ai-log JSON next to the data volume. Returns the file path."""
    out = Path(out_dir) if out_dir else AI_LOGS_DIR
    out.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = out / f"ailog_{entry.get('decision_id','unknown')}_{order.get('order_id','x')}_{ts}.json"
    path.write_text(
        json.dumps(build_ai_log(entry, order), ensure_ascii=False, indent=1, default=str),
        encoding="utf-8",
    )
    return path
