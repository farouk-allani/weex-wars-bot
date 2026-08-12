"""DeepSeek client (OpenAI-compatible endpoint).

Current models (verified against /models 2026-07-25):
  deepseek-v4-pro   — the default. Emits `reasoning_content` (its actual chain of
                      thought) AND accepts response_format/temperature.
  deepseek-v4-flash — same contract, faster and cheaper, less capable.

Both v4 models reason, so the chain of thought — the richest possible artifact for
the WEEX compliance review ("decision reasoning") — now comes for free on every
call. It is also charged and budgeted against max_tokens: the model thinks BEFORE
it emits the answer, so an output budget sized only for the JSON will be spent on
reasoning and return empty content. Hence MIN_REASONING_TOKENS below.

Legacy `deepseek-reasoner` (R1) is still handled because it rejects
response_format/temperature. `deepseek-chat` (V3) was retired by the provider on
2026-07-24; pinning it cost us 16h of silent outage, which is why
`validate_model()` exists.

Fails closed. Any error returns no decision, and no decision means HOLD. A trading
bot that guesses when its brain is unreachable is worse than one that sits still.
"""

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any, Optional

from dotenv import load_dotenv
from openai import OpenAI

# Load here rather than relying on some other module having imported the exchange
# first: any entrypoint that needs a decision model needs the key.
load_dotenv()

DEFAULT_BASE_URL = "https://api.deepseek.com"
# R1-style models reject response_format/temperature. The v4 family accepts both.
STRICT_REASONING_MODELS = {"deepseek-reasoner"}
# Any model that spends output budget on chain-of-thought before answering.
REASONING_MODELS = STRICT_REASONING_MODELS | {"deepseek-v4-pro", "deepseek-v4-flash"}
# Floor for reasoning models: enough for the thinking AND the decision JSON. An
# 8-symbol decision object is ~700 tokens; the rest is the model's reasoning.
MIN_REASONING_TOKENS = 8000

# Error classes that RETRYING CANNOT FIX. Measured 2026-08-12: the account ran out
# of credit on 08-07 22:19 and every hourly call 402'd for 65 hours until someone
# happened to top it up. Each of those cycles burned its full retry ladder first.
# Retries exist for a flaky network; a billing wall, a revoked key or a retired
# model id are all human-intervention states, and treating them as transient
# costs latency, hides the diagnosis behind a generic "failed after N attempts",
# and — worst — makes them wait for the consecutive-failure counter before anyone
# is told. They are separated here so the engine can alarm on the FIRST one.
TERMINAL_ERROR_KINDS = {"billing", "auth", "model"}


def classify_error(exc: Exception) -> str:
    """Map a provider exception to billing / auth / model / transient.

    Matches on HTTP status where the SDK exposes it and falls back to message
    text, because the OpenAI-compatible shim does not guarantee a typed error for
    every provider-specific condition (DeepSeek returns its billing wall as a bare
    402 with `code: invalid_request_error`, which is indistinguishable from a bad
    request by type alone).
    """
    status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    text = str(exc).lower()

    if status == 402 or "insufficient balance" in text or "quota" in text:
        return "billing"
    if status in (401, 403) or "invalid api key" in text or "authentication" in text:
        return "auth"
    if status == 404 or "does not exist" in text or "model not found" in text:
        return "model"
    return "transient"


class AIError(Exception):
    """Provider failure. `kind` is one of billing / auth / model / transient."""

    def __init__(self, message: str, kind: str = "transient"):
        super().__init__(message)
        self.kind = kind


class DeepSeekClient:
    def __init__(self, config: dict):
        ai = config.get("ai", {}) or {}
        self.model = ai.get("model", "deepseek-v4-pro")
        self.temperature = float(ai.get("temperature", 0.3))
        self.max_tokens = int(ai.get("max_tokens", 2000))
        self.timeout = float(ai.get("timeout_seconds", 90))
        self.max_retries = int(ai.get("max_retries", 2))

        api_key = os.getenv("DEEPSEEK_API_KEY", "")
        if not api_key:
            raise AIError(
                "DEEPSEEK_API_KEY is not set. Add it to .env — the bot will not "
                "trade without a reachable decision model.",
                "auth",
            )
        # Kept for check_balance(), which talks to a REST path the chat SDK does
        # not model.
        self.api_key = api_key
        self.base_url = ai.get("base_url", DEFAULT_BASE_URL)
        self.client = OpenAI(
            api_key=api_key,
            base_url=self.base_url,
            timeout=self.timeout,
        )

    @property
    def is_reasoner(self) -> bool:
        """Spends output budget on chain-of-thought before answering."""
        return self.model in REASONING_MODELS

    @property
    def is_strict_reasoner(self) -> bool:
        """Additionally rejects response_format and temperature."""
        return self.model in STRICT_REASONING_MODELS

    def available_models(self) -> list[str]:
        return sorted(m.id for m in self.client.models.list().data)

    def validate_model(self) -> list[str]:
        """Assert the configured model still exists at the provider.

        A pinned model id is a dependency that expires. When `deepseek-chat` was
        retired every hourly decision failed with a 400 for 16 hours while every
        healthcheck stayed green — the bot looked perfectly alive and was not
        thinking at all. Failing loudly at startup is the cheapest possible place
        to catch that, so this raises rather than warns.
        """
        try:
            models = self.available_models()
        except Exception as e:
            # A listing outage must not stop a bot whose model is probably fine;
            # the consecutive-failure alarm is the backstop for that case.
            raise AIError(f"could not list provider models: {e}") from e
        if self.model not in models:
            raise AIError(
                f"configured ai.model={self.model!r} no longer exists at the "
                f"provider. Available: {', '.join(models)}. "
                f"Update ai.model in config.yaml.",
                "model",
            )
        return models

    def decide(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        """Returns {content, reasoning, usage, latency_ms, raw}. Raises AIError."""
        # Reasoning models spend their budget on chain-of-thought BEFORE emitting
        # the answer. At 2k tokens they reason until the budget is gone and return
        # empty content — which raises below, fails closed, and looks exactly like
        # an outage. Give them room for the thinking plus the JSON.
        max_tokens = (
            max(self.max_tokens, MIN_REASONING_TOKENS)
            if self.is_reasoner
            else self.max_tokens
        )

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": max_tokens,
        }
        # Only R1-style models reject these. The v4 family reasons *and* honours
        # JSON mode, so keep both — enforced JSON removes a whole class of
        # parse-failure holds.
        if not self.is_strict_reasoner:
            kwargs["temperature"] = self.temperature
            kwargs["response_format"] = {"type": "json_object"}

        last_err: Optional[Exception] = None
        last_kind = "transient"
        for attempt in range(self.max_retries + 1):
            started = time.time()
            try:
                resp = self.client.chat.completions.create(**kwargs)
                msg = resp.choices[0].message
                content = (msg.content or "").strip()
                if not content:
                    raise AIError("empty response from model")
                return {
                    "content": content,
                    # The model's chain of thought. Logged verbatim: it is the single
                    # most valuable thing we can hand the compliance reviewers.
                    "reasoning": getattr(msg, "reasoning_content", "") or "",
                    "usage": resp.usage.model_dump() if resp.usage else {},
                    "latency_ms": int((time.time() - started) * 1000),
                    "raw": content,
                    # Provider-RESOLVED model id. The WEEX ai-log schema requires
                    # the exact raw id, not the requested alias.
                    "model": getattr(resp, "model", None) or self.model,
                }
            except Exception as e:
                last_err = e
                last_kind = classify_error(e)
                # A billing wall, a dead key or a retired model will still be there
                # in two seconds. Stop immediately so the engine gets the diagnosis
                # on the first cycle instead of after the full ladder.
                if last_kind in TERMINAL_ERROR_KINDS:
                    raise AIError(f"DeepSeek {last_kind} error: {last_err}", last_kind) from e
                if attempt < self.max_retries:
                    time.sleep(2 ** attempt)

        raise AIError(
            f"DeepSeek call failed after {self.max_retries + 1} attempts: {last_err}",
            last_kind,
        )

    def check_balance(self) -> dict[str, Any]:
        """Query remaining provider credit. Free — no tokens are billed for this.

        This is the probe that would have caught the 65h outage on its first hour.
        `validate_model()` already guards a retired model id, but a listing call
        succeeds perfectly well on a zero balance, so the existing preflight was
        blind to the one failure that actually happened. Money is a dependency that
        expires exactly like a model id, and it expires with no warning at all.

        Returns {available, balance, currency, checked_at} or {error} — never
        raises, because a telemetry probe must not be able to stop a trading loop.
        """
        url = self.base_url.rstrip("/") + "/user/balance"
        req = urllib.request.Request(
            url, headers={"Authorization": f"Bearer {self.api_key}", "Accept": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except Exception as e:  # network, auth, schema — all just "unknown balance"
            return {"error": str(e), "checked_at": time.time()}

        infos = payload.get("balance_infos") or []
        primary = infos[0] if infos else {}
        try:
            balance = float(primary.get("total_balance"))
        except (TypeError, ValueError):
            balance = None
        return {
            "available": bool(payload.get("is_available")),
            "balance": balance,
            "currency": primary.get("currency"),
            "checked_at": time.time(),
        }

    @staticmethod
    def parse_json(content: str) -> dict:
        """Tolerate fenced or prose-wrapped JSON — reasoner output isn't always clean."""
        text = content.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start, end = text.find("{"), text.rfind("}")
            if start != -1 and end > start:
                return json.loads(text[start : end + 1])
            raise
