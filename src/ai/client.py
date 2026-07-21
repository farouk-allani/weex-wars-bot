"""DeepSeek client (OpenAI-compatible endpoint).

Two models, deliberately both supported:
  deepseek-chat     — V3. Fast, cheap, supports JSON mode. The default.
  deepseek-reasoner — R1. Emits `reasoning_content`, its actual chain of thought,
                      which is the richest possible artifact for the compliance
                      review ("decision reasoning"). It rejects response_format and
                      temperature, so those are stripped for it.

Fails closed. Any error returns no decision, and no decision means HOLD. A trading
bot that guesses when its brain is unreachable is worse than one that sits still.
"""

import json
import os
import time
from typing import Any, Optional

from dotenv import load_dotenv
from openai import OpenAI

# Load here rather than relying on some other module having imported the exchange
# first: any entrypoint that needs a decision model needs the key.
load_dotenv()

DEFAULT_BASE_URL = "https://api.deepseek.com"
# R1 rejects response_format/temperature; chat supports both.
REASONING_MODELS = {"deepseek-reasoner"}


class AIError(Exception):
    pass


class DeepSeekClient:
    def __init__(self, config: dict):
        ai = config.get("ai", {}) or {}
        self.model = ai.get("model", "deepseek-chat")
        self.temperature = float(ai.get("temperature", 0.3))
        self.max_tokens = int(ai.get("max_tokens", 2000))
        self.timeout = float(ai.get("timeout_seconds", 90))
        self.max_retries = int(ai.get("max_retries", 2))

        api_key = os.getenv("DEEPSEEK_API_KEY", "")
        if not api_key:
            raise AIError(
                "DEEPSEEK_API_KEY is not set. Add it to .env — the bot will not "
                "trade without a reachable decision model."
            )
        self.client = OpenAI(
            api_key=api_key,
            base_url=ai.get("base_url", DEFAULT_BASE_URL),
            timeout=self.timeout,
        )

    @property
    def is_reasoner(self) -> bool:
        return self.model in REASONING_MODELS

    def decide(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        """Returns {content, reasoning, usage, latency_ms, raw}. Raises AIError."""
        # R1 spends its budget on chain-of-thought BEFORE emitting the answer. At
        # 2k tokens it reasons until the budget is gone and returns empty content —
        # so give it room for the thinking plus the JSON.
        max_tokens = max(self.max_tokens, 8000) if self.is_reasoner else self.max_tokens

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": max_tokens,
        }
        if not self.is_reasoner:
            kwargs["temperature"] = self.temperature
            kwargs["response_format"] = {"type": "json_object"}

        last_err: Optional[Exception] = None
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
                    # R1's chain of thought. Logged verbatim: it is the single most
                    # valuable thing we can hand the compliance reviewers.
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
                if attempt < self.max_retries:
                    time.sleep(2 ** attempt)

        raise AIError(f"DeepSeek call failed after {self.max_retries + 1} attempts: {last_err}")

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
