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
    """The model's own reasoning for THIS symbol, capped at the schema's 1000.

    The per-symbol key the model actually emits is `rationale` (see the schema in
    ai/trader.py). This used to look only for `reason`/`reasoning`, never matched,
    and fell through to the whole cycle's raw chain-of-thought truncated at exactly
    1000 chars — a mid-sentence CoT dump, when the schema asks for an explanation
    tied to specific facts in the input. Prefer the symbol's own rationale.
    """
    sym = str(order.get("symbol") or "")
    reason = ""
    for d in entry.get("decisions") or []:
        if not isinstance(d, dict):
            continue
        dsym = str(d.get("symbol") or "")
        # Both sides must be non-empty, otherwise "" matches the first decision.
        if not dsym or not sym:
            continue
        if dsym == sym or dsym in sym or sym in dsym:
            reason = str(
                d.get("rationale") or d.get("reason") or d.get("reasoning") or ""
            ).strip()
            if reason:
                break
    if not reason:
        reason = str(entry.get("reasoning") or "").strip()
    if not reason:
        reason = "Decision per attached model output."
    return _truncate(reason, 1000)


def _truncate(text: str, limit: int) -> str:
    """Cap at `limit` on a sentence, then word, boundary — never mid-word."""
    if len(text) <= limit:
        return text
    cut = text[: limit - 1]
    for sep in (". ", "; ", ", ", " "):
        i = cut.rfind(sep)
        # Only honour a boundary that keeps most of the budget, else a stray early
        # period would throw away the explanation.
        if i > limit * 0.6:
            return cut[: i + (1 if sep.startswith(".") else 0)].rstrip() or cut
    return cut.rstrip() + "…"


def validate(payload: dict) -> list[str]:
    """Schema problems in a built ai-log, worst first. Empty list = usable.

    Presence of a file is a weaker guarantee than a usable log, and only the
    stricter one is worth anything at review time: a log whose `input` has no
    message array does not carry "the complete original request" the schema asks
    for, however well-formed the rest of it looks.
    """
    problems: list[str] = []
    if not str(payload.get("stage") or "").strip():
        problems.append("stage is empty")
    model = str(payload.get("model") or "").strip()
    if not model:
        problems.append("model is empty")

    src = payload.get("input") or {}
    if not isinstance(src, dict):
        problems.append("input is not an object")
    else:
        if not (src.get("messages") or []):
            problems.append("input.messages is empty (no verbatim prompt)")
        if not (src.get("market_context") or {}):
            problems.append("input.market_context is empty")

    out = payload.get("output") or {}
    if not isinstance(out, dict):
        problems.append("output is not an object")
    else:
        for k in ("symbol", "side", "quantity"):
            if out.get(k) in (None, "", 0):
                problems.append(f"output.{k} is missing")

    expl = str(payload.get("explanation") or "")
    if not expl.strip():
        problems.append("explanation is empty")
    elif len(expl) > 1000:
        problems.append(f"explanation is {len(expl)} chars (limit 1000)")
    return problems


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
