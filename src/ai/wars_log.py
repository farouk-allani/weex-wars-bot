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

import base64
import hashlib
import hmac
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from urllib import error, request
from urllib.parse import urlparse

AI_LOGS_DIR = Path("data/ai_logs")
UPLOAD_STATUS_DIR = Path("data/ai_log_upload_status")
ALLOWLIST_STATUS_PATH = Path("data/ai_log_allowlist_status.json")
DEFAULT_UPLOAD_BASE = "https://api-contract.weex.com"
UPLOAD_PATH = "/capi/v3/order/uploadAiLog"
SUCCESS_CODES = {"0", "00000"}


def build_ai_log(entry: dict, order: dict) -> dict:
    """entry = the logbook decision record; order = the linked order params."""
    side = str(order.get("side", "")).lower()
    intent = str(order.get("intent") or "entry").lower()
    symbol = str(order.get("symbol", "")).replace("/", "").replace(":USDT", "")
    position_side = str(order.get("position_side") or "").upper() or (
        "LONG" if side in ("long", "buy") else "SHORT"
    )

    if intent == "close":
        output = {
            "symbol": symbol,
            "action": "CLOSE",
            "positionSide": position_side,
            "type": "MARKET",
            "quantity": order.get("size"),
        }
    elif intent in ("stop_loss", "take_profit_trigger"):
        output = {
            "symbol": symbol,
            "side": "BUY" if side in ("long", "buy") else "SELL",
            "positionSide": position_side,
            "type": "STOP_MARKET" if intent == "stop_loss" else "TAKE_PROFIT_MARKET",
            "quantity": order.get("size"),
            "triggerPrice": order.get("trigger_price"),
        }
    else:
        order_type = str(order.get("order_type") or "LIMIT").upper()
        output = {
            "symbol": symbol,
            "side": "BUY" if side in ("long", "buy") else "SELL",
            "positionSide": position_side,
            "type": order_type,
            "quantity": order.get("size"),
        }
        if order_type == "LIMIT":
            output["price"] = order.get("entry_price")
            output["timeInForce"] = str(
                order.get("time_in_force") or "POST_ONLY"
            ).upper()
        if intent == "reduce":
            output["reduceOnly"] = True
        elif intent == "entry" and order.get("stop_loss"):
            output["slTriggerPrice"] = order.get("stop_loss")

    payload = {
        "stage": "Strategy Generation",
        "model": entry.get("model") or "",
        "input": {
            # the literal request messages, preserved verbatim
            "messages": entry.get("messages") or [],
            "market_context": entry.get("context") or {},
        },
        "output": output,
        "explanation": _explanation(entry, order),
    }
    order_id = str(order.get("order_id") or "")
    if order_id.isdigit():
        payload["orderId"] = int(order_id)
    return payload


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
        if out.get("action") == "CLOSE":
            required = ("symbol", "quantity")
        elif out.get("triggerPrice") is not None:
            required = ("symbol", "side", "positionSide", "type", "quantity", "triggerPrice")
        else:
            required = ("symbol", "side", "quantity")
        for k in required:
            if out.get(k) in (None, "", 0):
                problems.append(f"output.{k} is missing")
        if out.get("type") == "LIMIT" and out.get("price") in (None, "", 0):
            problems.append("output.price is missing for LIMIT")

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


class WeexAILogUploader:
    """Durable UploadAiLog delivery after a successful live order.

    The order is already real when this runs, so a failed upload is persisted and
    retried; it must never cause the trading engine to retry the order itself.
    New entries can use ``readiness()`` to fail closed while any upload is pending.
    """

    def __init__(
        self,
        *,
        enabled: bool,
        status_dir: str | Path = UPLOAD_STATUS_DIR,
        allowlist_status_path: str | Path = ALLOWLIST_STATUS_PATH,
        base_url: Optional[str] = None,
        timeout: float = 15.0,
    ) -> None:
        self.enabled = bool(enabled)
        self.status_dir = Path(status_dir)
        self.allowlist_status_path = Path(allowlist_status_path)
        self.base_url = (base_url or os.getenv("WEEX_AI_API_BASE") or DEFAULT_UPLOAD_BASE).rstrip("/")
        self.timeout = float(timeout)
        self.api_key = os.getenv("WEEX_API_KEY", "")
        self.api_secret = os.getenv("WEEX_API_SECRET", "")
        self.passphrase = os.getenv("WEEX_API_PASSPHRASE") or os.getenv("WEEX_PASSPHRASE") or ""

    def _configuration_error(self) -> Optional[str]:
        if not self.enabled:
            return None
        parsed = urlparse(self.base_url)
        host = (parsed.hostname or "").lower()
        if parsed.scheme != "https" or not (
            host == "weex.com" or host.endswith(".weex.com")
            or host == "weex.tech" or host.endswith(".weex.tech")
        ):
            return "WEEX AI log base URL must be HTTPS on weex.com or weex.tech"
        missing = []
        if not self.api_key:
            missing.append("WEEX_API_KEY")
        if not self.api_secret:
            missing.append("WEEX_API_SECRET")
        if not self.passphrase:
            missing.append("WEEX_API_PASSPHRASE")
        return f"missing {', '.join(missing)}" if missing else None

    def _status_path(self, ai_log_path: Path) -> Path:
        return self.status_dir / f"{ai_log_path.stem}.json"

    @staticmethod
    def _read_json(path: Path) -> dict:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    @staticmethod
    def _atomic_write(path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, path)

    def register(self, ai_log_path: Path, *, required: bool) -> dict:
        status = {
            "file": str(ai_log_path),
            "required": bool(required),
            "uploaded": False,
            "attempts": 0,
            "last_attempt": None,
            "last_error": None,
            "http_status": None,
        }
        self._atomic_write(self._status_path(ai_log_path), status)
        if required:
            return self.upload(ai_log_path)
        return status

    def _post_payload(self, payload: dict) -> tuple[bool, Optional[int], Optional[str]]:
        """Sign and submit one payload, returning only non-secret delivery state."""
        body = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
        timestamp = str(int(time.time() * 1000))
        message = f"{timestamp}POST{UPLOAD_PATH}{body}"
        signature = base64.b64encode(
            hmac.new(
                self.api_secret.encode("utf-8"),
                message.encode("utf-8"),
                hashlib.sha256,
            ).digest()
        ).decode("ascii")
        req = request.Request(
            self.base_url + UPLOAD_PATH,
            data=body.encode("utf-8"),
            method="POST",
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
                "locale": "en-US",
                "ACCESS-KEY": self.api_key,
                "ACCESS-PASSPHRASE": self.passphrase,
                "ACCESS-TIMESTAMP": timestamp,
                "ACCESS-SIGN": signature,
                "User-Agent": "weex-ai-wars-bot/1.0",
            },
        )
        try:
            with request.urlopen(req, timeout=self.timeout) as response:
                http_status = response.status
                raw = response.read().decode("utf-8", errors="replace")
                response_payload = json.loads(raw)
        except error.HTTPError as exc:
            raw = exc.read().decode("utf-8", errors="replace")
            return False, exc.code, f"HTTP {exc.code}: {raw[:500]}"
        except Exception as exc:
            return False, None, str(exc)

        raw_code = response_payload.get("code")
        code = str(raw_code) if raw_code is not None else ""
        if code not in SUCCESS_CODES:
            return False, http_status, (
                f"WEEX code {code or '<missing>'}: "
                f"{response_payload.get('msg') or response_payload.get('message') or 'unknown error'}"
            )
        return True, http_status, None

    def probe_allowlist(self, decision: dict) -> dict:
        """Prove AI-Wars allowlisting with an authentic no-order decision log.

        WEEX explicitly permits ``orderId: null``.  This writes a compliance log
        but cannot create/cancel an order or move funds.  A successful probe is
        persisted and becomes a prerequisite for live uploader readiness, so the
        first real order is never used to discover a missing UID allowlist entry.
        """
        status = {
            "verified": False,
            "last_attempt": datetime.now(timezone.utc).isoformat(),
            "http_status": None,
            "last_error": None,
        }
        config_error = self._configuration_error()
        if config_error:
            status["last_error"] = config_error
            self._atomic_write(self.allowlist_status_path, status)
            return status

        try:
            output = json.loads(decision.get("raw_response") or "{}")
        except Exception:
            output = {"decisions": decision.get("decisions") or []}
        explanation = str(
            output.get("market_assessment")
            or decision.get("reasoning")
            or "No-order AI decision compliance probe."
        )[:1000]
        payload = {
            "orderId": None,
            "stage": "Decision Making",
            "model": decision.get("model") or "",
            "input": {
                "messages": decision.get("messages") or [],
                "market_context": decision.get("context") or {},
            },
            "output": output,
            "explanation": explanation,
        }
        missing = []
        if not payload["model"]:
            missing.append("model")
        if not payload["input"]["messages"]:
            missing.append("input.messages")
        if not payload["input"]["market_context"]:
            missing.append("input.market_context")
        if not payload["output"]:
            missing.append("output")
        if missing:
            status["last_error"] = "probe decision missing " + ", ".join(missing)
        else:
            ok, http_status, last_error = self._post_payload(payload)
            status.update(
                verified=ok,
                http_status=http_status,
                last_error=last_error,
            )
            if ok:
                status["verified_at"] = datetime.now(timezone.utc).isoformat()
        self._atomic_write(self.allowlist_status_path, status)
        return status

    def allowlist_status(self) -> dict:
        if not self.enabled:
            return {"required": False, "verified": True, "last_error": None}
        row = self._read_json(self.allowlist_status_path)
        return {
            "required": True,
            "verified": bool(row.get("verified")),
            "last_attempt": row.get("last_attempt"),
            "verified_at": row.get("verified_at"),
            "http_status": row.get("http_status"),
            "last_error": row.get("last_error") or (
                None if row.get("verified") else "AI-Wars allowlist has not been verified"
            ),
        }

    def upload(self, ai_log_path: str | Path) -> dict:
        path = Path(ai_log_path)
        status_path = self._status_path(path)
        status = self._read_json(status_path) or {
            "file": str(path), "required": True, "uploaded": False, "attempts": 0
        }
        if status.get("uploaded"):
            return status

        status["attempts"] = int(status.get("attempts") or 0) + 1
        status["last_attempt"] = datetime.now(timezone.utc).isoformat()
        config_error = self._configuration_error()
        if config_error:
            status["last_error"] = config_error
            self._atomic_write(status_path, status)
            return status

        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            status["last_error"] = f"could not read AI log: {exc}"
            self._atomic_write(status_path, status)
            return status
        problems = validate(payload)
        if problems:
            status["last_error"] = "; ".join(problems)
            self._atomic_write(status_path, status)
            return status

        try:
            uploaded, http_status, last_error = self._post_payload(payload)
            status["http_status"] = http_status
            if uploaded:
                status["uploaded"] = True
                status["uploaded_at"] = datetime.now(timezone.utc).isoformat()
                status["last_error"] = None
            else:
                status["last_error"] = last_error
        except Exception as exc:
            status["last_error"] = str(exc)
        self._atomic_write(status_path, status)
        return status

    def retry_pending(self, *, max_attempts: int = 10) -> dict:
        if not self.status_dir.exists():
            return self.status()
        for status_path in sorted(self.status_dir.glob("*.json")):
            status = self._read_json(status_path)
            if not status.get("required") or status.get("uploaded"):
                continue
            if int(status.get("attempts") or 0) >= max_attempts:
                continue
            file_path = status.get("file")
            if file_path:
                self.upload(file_path)
        return self.status()

    def status(self) -> dict:
        rows = []
        if self.status_dir.exists():
            rows = [self._read_json(p) for p in sorted(self.status_dir.glob("*.json"))]
        required = [r for r in rows if r.get("required")]
        pending = [r for r in required if not r.get("uploaded")]
        config_error = self._configuration_error()
        allowlist = self.allowlist_status()
        return {
            "enabled": self.enabled,
            "configured": config_error is None,
            "configuration_error": config_error,
            "required": len(required),
            "uploaded": sum(1 for r in required if r.get("uploaded")),
            "pending": len(pending),
            "pending_items": pending[-10:],
            "allowlist": allowlist,
            "allowlist_verified": bool(allowlist.get("verified")),
            "ready": (not self.enabled) or (
                config_error is None
                and not pending
                and bool(allowlist.get("verified"))
            ),
        }

    def readiness(self) -> tuple[bool, str]:
        status = self.status()
        if status["ready"]:
            return True, "ready"
        if status.get("configuration_error"):
            return False, str(status["configuration_error"])
        if not status.get("allowlist_verified"):
            return False, str(
                (status.get("allowlist") or {}).get("last_error")
                or "AI-Wars allowlist has not been verified"
            )
        return False, f"{status['pending']} WEEX AI-log upload(s) pending"
