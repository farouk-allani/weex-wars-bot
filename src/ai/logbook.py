"""Append-only AI decision log — the competition submission artifact.

WEEX requires "complete AI decision logs including OrderId matching, decision
reasoning, and strategy documentation", and treats >8h of inactivity *without
valid AI logs* as non-compliant. So every cycle is logged, including the cycles
where the model decides to do nothing — a reasoned HOLD is a valid log entry and
is what keeps the heartbeat alive between trades.

JSONL, one decision per line, fsync'd on write: a crash must never cost us the
record of a decision the exchange already acted on.
"""

import json
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


class DecisionLog:
    def __init__(self, path: str | Path = "logs/ai_decisions.jsonl"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._recent: dict[str, dict] = {}  # decision_id -> entry, for ai-log emission

    def record(
        self,
        *,
        model: str,
        context: dict[str, Any],
        decisions: list[dict[str, Any]],
        raw_response: str,
        reasoning: str = "",
        usage: Optional[dict] = None,
        latency_ms: Optional[int] = None,
        error: Optional[str] = None,
        messages: Optional[list] = None,
    ) -> str:
        """Log one AI decision cycle. Returns the decision_id used for OrderId matching."""
        decision_id = f"dec_{uuid.uuid4().hex[:16]}"
        entry = {
            "decision_id": decision_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": model,
            # The exact inputs the model saw. Without these the reasoning is
            # unauditable and the log is worthless for compliance review.
            "context": context,
            # Literal request message array — the WEEX ai-log schema requires
            # the complete prompt in its original form.
            "messages": messages or [],
            "reasoning": reasoning,
            "decisions": decisions,
            "raw_response": raw_response,
            "usage": usage or {},
            "latency_ms": latency_ms,
            "error": error,
            # Filled in by link_order() once the exchange confirms a fill.
            "orders": [],
        }
        self._append(entry)
        # In-memory tail so link_order can emit a WEEX ai-log without a file
        # re-read. Capped: decisions older than the cap can't get new orders.
        self._recent[decision_id] = entry
        while len(self._recent) > 100:
            self._recent.pop(next(iter(self._recent)))
        return decision_id

    def link_order(
        self,
        decision_id: str,
        *,
        symbol: str,
        order_id: str,
        side: str,
        size: float,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
    ) -> None:
        """Bind an exchange OrderId back to the decision that produced it.

        Written as a separate linkage record rather than by rewriting the original
        line: the log stays append-only, so a fill can never corrupt the decision
        that preceded it. Readers fold these into the parent by decision_id.
        """
        order = {
            "symbol": symbol,
            "order_id": str(order_id),
            "side": side,
            "size": size,
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
        }
        self._append({
            "type": "order_link",
            "decision_id": decision_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "order": order,
        })
        # Emit the WEEX-schema ai-log file for this order. Must never break
        # trading — the trade is already live when this runs.
        try:
            from . import wars_log

            entry = self._recent.get(decision_id)
            if entry:
                wars_log.emit(entry, order)
        except Exception:
            pass

    def record_outcome(
        self,
        decision_id: str,
        *,
        symbol: str,
        order_id: str,
        pnl: float,
        exit_price: float,
        exit_reason: str,
    ) -> None:
        """Close the loop so the log shows what each decision actually earned."""
        self._append({
            "type": "outcome",
            "decision_id": decision_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "outcome": {
                "symbol": symbol,
                "order_id": str(order_id),
                "pnl": pnl,
                "exit_price": exit_price,
                "exit_reason": exit_reason,
            },
        })

    def _append(self, entry: dict) -> None:
        line = json.dumps(entry, default=str, ensure_ascii=False)
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
                f.flush()
                os.fsync(f.fileno())

    def last_decision_at(self) -> Optional[datetime]:
        """Most recent logged decision — used to enforce the 8h activity rule."""
        if not self.path.exists():
            return None
        last = None
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                if row.get("type"):  # linkage/outcome records, not decisions
                    continue
                ts = row.get("timestamp")
                if ts:
                    try:
                        last = datetime.fromisoformat(ts)
                    except Exception:
                        pass
        return last
