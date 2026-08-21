"""Append-only AI decision log for competition rehearsal.

The historical WEEX Season 1 workflow required complete AI decision logs with
matching OrderIds and continued no-trade logs during inactivity. Every cycle is
therefore retained locally as a conservative rehearsal artifact. This module does
not claim that local HOLD records satisfy a current competition's upload rules;
the current rulebook and enrollment must be verified separately.

JSONL, one decision per line, fsync'd on write: a crash must never cost us the
record of a decision the exchange already acted on.
"""

import copy
import json
import logging
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class DecisionLog:
    def __init__(
        self,
        path: str | Path = "logs/ai_decisions.jsonl",
        *,
        live_upload: bool = False,
        uploader=None,
    ):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._recent: dict[str, dict] = {}  # decision_id -> entry, for ai-log emission
        # ai-log emission counters. An order without an ai-log is a compliance
        # failure, so the tally is state the engine reports, not a debug detail.
        self.ailog_emitted = 0
        self.ailog_failed = 0
        self.last_ailog_error: Optional[str] = None
        self.live_upload = bool(live_upload)
        if uploader is None:
            from .wars_log import WeexAILogUploader

            uploader = WeexAILogUploader(enabled=self.live_upload)
        self.uploader = uploader

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
        order_type: str = "LIMIT",
        intent: str = "entry",
        position_side: Optional[str] = None,
        trigger_price: Optional[float] = None,
    ) -> Optional[dict]:
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
            "order_type": str(order_type).upper(),
            "intent": intent,
            "position_side": position_side,
            "trigger_price": trigger_price,
        }
        self._append({
            "type": "order_link",
            "decision_id": decision_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "order": order,
        })
        return self._emit_ai_log(decision_id, order)

    def _lookup(self, decision_id: str) -> Optional[dict]:
        """The decision behind an order, from memory or from disk."""
        entry = self._recent.get(decision_id)
        if entry is not None:
            return entry
        return self._read_decision(decision_id)

    def get_decision(self, decision_id: str) -> Optional[dict]:
        """Return a read-only copy of the decision behind an open position."""
        entry = self._lookup(str(decision_id or ""))
        return copy.deepcopy(entry) if entry is not None else None

    def _read_decision(self, decision_id: str) -> Optional[dict]:
        """Recover a decision record from the log file.

        `_recent` is process memory, but a maker entry rests for up to
        execution.entry_ttl_minutes and can therefore fill *after* a restart or a
        deploy. Without this fallback that order's ai-log was never written —
        silently, because the emit path swallowed the miss. Measured 2026-07-27:
        7 of 18 orders had no ai-log file.
        """
        if not self.path.exists():
            return None
        found = None
        try:
            with open(self.path, encoding="utf-8") as f:
                for line in f:
                    # Cheap pre-filter: skip the JSON parse for non-matching lines.
                    if decision_id not in line:
                        continue
                    try:
                        row = json.loads(line)
                    except Exception:
                        continue
                    if row.get("type"):  # linkage/outcome record, not the decision
                        continue
                    if row.get("decision_id") == decision_id:
                        found = row
        except Exception as e:
            logger.error("ai-log: could not read decision log for %s: %s", decision_id, e)
        return found

    def _emit_ai_log(self, decision_id: str, order: dict) -> Optional[dict]:
        """Write the WEEX-schema ai-log for an order.

        Never raises — the trade is already live by the time this runs. But a
        failure is recorded and logged at ERROR: a missing order log failed the
        historical rehearsal workflow, and the previous silent `except: pass` is
        exactly why that went unnoticed for six days.
        """
        try:
            from . import wars_log

            entry = self._lookup(decision_id)
            if entry is None:
                raise LookupError("decision not found in memory or on disk")
            path = wars_log.emit(entry, order)
            upload = self.uploader.register(path, required=self.live_upload)
            self.ailog_emitted += 1
            if self.live_upload and not upload.get("uploaded"):
                self.ailog_failed += 1
                self.last_ailog_error = str(upload.get("last_error") or "upload pending")
                logger.error(
                    "AILOG_UPLOAD_PENDING (COMPLIANCE) %s order=%s: %s",
                    order.get("symbol"), order.get("order_id"), self.last_ailog_error,
                )
            else:
                self.last_ailog_error = None
                logger.info(
                    "AILOG_OK %s order=%s -> %s uploaded=%s",
                    order.get("symbol"), order.get("order_id"), path.name,
                    bool(upload.get("uploaded")),
                )
            return {"path": str(path), "upload": upload}
        except Exception as e:
            self.ailog_failed += 1
            self.last_ailog_error = (
                f"{order.get('symbol')} order {order.get('order_id')}: {e}"
            )
            logger.error(
                "AILOG_FAILED (COMPLIANCE) %s order=%s decision=%s: %s",
                order.get("symbol"), order.get("order_id"), decision_id, e,
            )
            return None

    def compliance_status(self, ai_logs_dir: str | Path = "data/ai_logs") -> dict:
        """Every linked order vs every ai-log file on disk.

        Read off the artifacts themselves rather than the in-process counters, so
        it stays true across restarts and reports the same thing a competition
        reviewer would see.
        """
        links: list[dict] = []
        try:
            if self.path.exists():
                with open(self.path, encoding="utf-8") as f:
                    for line in f:
                        try:
                            row = json.loads(line)
                        except Exception:
                            continue
                        if row.get("type") == "order_link":
                            links.append(row)
        except Exception as e:
            return {"error": str(e)}

        # A file that exists but carries no verbatim prompt is not a usable log, so
        # presence alone is not the invariant worth reporting.
        from . import wars_log

        d = Path(ai_logs_dir)
        files = sorted(d.glob("*.json")) if d.exists() else []
        names = [p.name for p in files]

        missing: list[dict] = []
        broken: list[dict] = []
        unrepairable: list[dict] = []
        for r in links:
            did = r.get("decision_id") or ""
            order = r.get("order") or {}
            oid = str(order.get("order_id") or "")
            row = {
                "timestamp": r.get("timestamp"),
                "symbol": order.get("symbol"),
                "order_id": oid,
                "decision_id": did,
            }
            prefix = f"ailog_{did}_{oid}_"
            match = next((p for p in files if p.name.startswith(prefix)), None)
            if match is None:
                missing.append(row)
                continue
            try:
                problems = wars_log.validate(
                    json.loads(match.read_text(encoding="utf-8"))
                )
            except Exception as e:
                problems = [f"unreadable: {e}"]
            if not problems:
                continue
            row = dict(row, file=match.name, problems=problems)
            # Split "we have a bug" from "the source record never held this".
            # Decisions logged before the logbook captured verbatim messages cannot
            # be completed from anything on disk, and inventing a prompt to close
            # the count would be worse than reporting the hole. Those are recorded
            # once as a known historical fact so they cannot mask a NEW failure.
            src = self._read_decision(did) or {}
            if any("input.messages" in p for p in problems) and not (
                src.get("messages") or []
            ):
                unrepairable.append(row)
            else:
                broken.append(row)

        # `compliant` is about what is still actionable: every order has a log, and
        # every log that COULD be complete is. A permanently-false flag is a monitor
        # nobody reads.
        upload = self.uploader.status()
        return {
            "orders_linked": len(links),
            "ai_logs_on_disk": len(names),
            "orders_without_ai_log": len(missing),
            "ai_logs_repairable_incomplete": len(broken),
            "ai_logs_unrepairable_historical": len(unrepairable),
            # Bounded: the lists are for diagnosis, the counts are the alarm.
            "missing": missing[-20:],
            "incomplete": broken[-20:],
            "unrepairable": unrepairable[-20:],
            "official_upload": upload,
            "official_upload_pending": upload.get("pending", 0),
            "compliant": not missing and not broken and (
                not self.live_upload or bool(upload.get("ready"))
            ),
            "note": (
                f"{len(unrepairable)} historical log(s) predate verbatim-prompt "
                "capture and cannot be completed from the record"
                if unrepairable else ""
            ),
            "emitted_this_process": self.ailog_emitted,
            "failed_this_process": self.ailog_failed,
            "last_error": self.last_ailog_error,
        }

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
        """Most recent local decision, used for rehearsal cadence health."""
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
