"""The decision layer: prompt DeepSeek, then refuse anything unsafe it says.

Every field the model returns is treated as untrusted input. Stops are checked for
side and sanity, R:R is enforced, conviction is clamped, and *size is never taken
from the model* — it is computed by the risk engine from conviction and stop
distance. The model's influence on execution is real and material (it picks the
instrument, the direction, the levels and the conviction), which is what the rules
require; what it cannot do is exceed the risk envelope.
"""

import json
from datetime import datetime, timezone
from typing import Any, Optional

from ..core.models import Candle, Side, Signal

SYSTEM_PROMPT = """You are the decision engine of a crypto perpetual-futures trading bot competing in the WEEX AI Wars hackathon.

OBJECTIVE
Maximise cumulative PnL over the competition. Ranking is by absolute PnL, so a flat account does not place — but a blown account is worse than a flat one, and you cannot recover from a drawdown you did not survive. Trade when you have an edge; hold when you do not. Do not manufacture trades out of boredom.

WHAT YOU CONTROL
- Which instrument to trade, and which direction.
- Your conviction (0.0-1.0), which scales position size.
- Where the stop-loss and take-profit sit.
- Whether to close an existing position early.

WHAT YOU DO NOT CONTROL
- Position size. The risk engine derives it from your conviction and stop distance.
- Stop enforcement. Once set, stops execute in code. You cannot widen or remove one.
- Risk limits. Proposals that breach them are rejected, so reason within them.

HOW TO DECIDE
Weigh the evidence you are given: trend regime and ADX, mean-reversion stretch (RSI, Bollinger z-score, VWAP deviation), momentum, volatility (ATR%), volume anomalies, funding (crowded positioning is a contrarian tell), higher-timeframe direction, and the structure of recent highs/lows.

Nothing here is a mechanical rule. A z-score of 2 in a strong trend is a continuation signal, not a fade — context decides. Conflicting evidence is a reason to hold or to lower conviction, not to pick a side and hope.

Look at `recent_closed_trades`. If your recent calls in a regime are failing, adapt — do not repeat a thesis the market has just rejected.

STOPS AND TARGETS
- Anchor the stop to volatility (ATR) and to structure (beyond the swing that invalidates your thesis), not to a round number.
- A stop tighter than ~0.5x ATR will be noise-stopped. Wider than ~4x ATR is not a trade, it is a donation.
- Required reward:risk is stated in hard_limits. Below it, the trade is rejected — so if the nearest sensible target does not clear it, hold instead.

CONVICTION
- 0.8-1.0: multiple independent factors align, clean structure, clear invalidation.
- 0.4-0.7: a real but contested setup.
- 0.1-0.3: marginal. Prefer holding.
Be honest. Inflated conviction inflates size, and inflated size is how accounts die.

OUTPUT
Return ONLY valid JSON, no prose outside it:
{
  "market_assessment": "2-4 sentences on the overall tape and what regime you think we are in",
  "decisions": [
    {
      "symbol": "<exact symbol string from the context>",
      "action": "long" | "short" | "hold" | "close",
      "conviction": 0.0,
      "stop_loss": 0.0,
      "take_profit": 0.0,
      "rationale": "the specific evidence for this call, and what would prove you wrong"
    }
  ]
}
Include an entry for every tradeable symbol. Use "hold" (with stop_loss and take_profit as 0) when there is no trade. Use "close" only for a symbol you already hold.
A well-reasoned "hold" on every symbol is a valid, and often correct, answer."""


class AITrader:
    def __init__(self, config: dict, client, logbook):
        self.config = config
        self.client = client
        self.log = logbook
        comp = config.get("competition", {}) or {}
        self.min_rr = float(comp.get("min_rr", 1.35))
        ai = config.get("ai", {}) or {}
        self.min_conviction = float(ai.get("min_conviction", 0.35))
        self.leverage = int(config.get("trading", {}).get("default_leverage", 5))
        # Guard against a hallucinated stop that is either noise-tight or absurd.
        self.min_stop_atr = float(ai.get("min_stop_atr", 0.5))
        self.max_stop_atr = float(ai.get("max_stop_atr", 4.0))

    def decide(self, context: dict) -> tuple[list[dict], str, str]:
        """Call the model. Returns (raw_decisions, assessment, decision_id).

        Always logs, including on failure — an unreachable model must still leave an
        audit trail, both for compliance and so a silent outage is visible later.
        """
        user_prompt = (
            "Current market and account state:\n\n"
            + json.dumps(context, indent=2, default=str)
            + "\n\nDecide. Return only the JSON object."
        )
        try:
            result = self.client.decide(SYSTEM_PROMPT, user_prompt)
        except Exception as e:
            decision_id = self.log.record(
                model=self.client.model,
                context=context,
                decisions=[],
                raw_response="",
                reasoning="",
                error=str(e),
            )
            return [], "", decision_id

        try:
            parsed = self.client.parse_json(result["content"])
            decisions = parsed.get("decisions") or []
            assessment = parsed.get("market_assessment", "")
        except Exception as e:
            decisions, assessment = [], ""
            result["content"] += f"\n\n[PARSE ERROR: {e}]"

        decision_id = self.log.record(
            model=self.client.model,
            context=context,
            decisions=decisions,
            raw_response=result["content"],
            # R1's chain of thought if present; otherwise the model's own assessment.
            reasoning=result.get("reasoning") or assessment,
            usage=result.get("usage"),
            latency_ms=result.get("latency_ms"),
        )
        return decisions, assessment, decision_id

    def to_signal(
        self,
        decision: dict,
        symbol: str,
        price: float,
        atr: float,
        allowed_symbols: set[str],
    ) -> tuple[Optional[Signal], str]:
        """Validate one model decision into a Signal, or explain the rejection.

        Rejections are returned rather than raised so they can be logged: a model
        that keeps proposing invalid stops is telling us something.
        """
        sym = str(decision.get("symbol") or symbol)
        if sym not in allowed_symbols:
            return None, f"symbol {sym!r} not in the permitted competition set"

        action = str(decision.get("action", "hold")).lower()
        if action not in ("long", "short"):
            return None, f"action={action}"

        try:
            conviction = float(decision.get("conviction") or 0)
            sl = float(decision.get("stop_loss") or 0)
            tp = float(decision.get("take_profit") or 0)
        except (TypeError, ValueError):
            return None, "non-numeric conviction/stop_loss/take_profit"

        conviction = max(0.0, min(1.0, conviction))
        if conviction < self.min_conviction:
            return None, f"conviction {conviction:.2f} below floor {self.min_conviction:.2f}"

        if sl <= 0 or tp <= 0:
            return None, "missing stop_loss or take_profit"

        side = Side.LONG if action == "long" else Side.SHORT

        # Stops on the correct side of entry. A model that inverts these would
        # otherwise open a position whose "stop" is a take-profit.
        if side == Side.LONG and not (sl < price < tp):
            return None, f"long needs sl({sl}) < price({price}) < tp({tp})"
        if side == Side.SHORT and not (tp < price < sl):
            return None, f"short needs tp({tp}) < price({price}) < sl({sl})"

        stop_dist = abs(price - sl)
        if atr > 0:
            lo, hi = self.min_stop_atr * atr, self.max_stop_atr * atr
            if stop_dist < lo:
                return None, f"stop {stop_dist:.4f} tighter than {self.min_stop_atr}x ATR — noise"
            if stop_dist > hi:
                return None, f"stop {stop_dist:.4f} wider than {self.max_stop_atr}x ATR"

        rr = abs(tp - price) / stop_dist if stop_dist > 0 else 0
        if rr < self.min_rr:
            return None, f"R:R {rr:.2f} below required {self.min_rr:.2f}"

        signal = Signal(
            symbol=sym,
            side=side,
            strength=conviction,  # drives size via the risk engine, not directly
            strategy="ai_deepseek",
            entry_price=price,
            stop_loss=sl,
            take_profit=tp,
            leverage=self.leverage,
            reason=str(decision.get("rationale") or "")[:400],
            timestamp=datetime.now(timezone.utc).replace(tzinfo=None),
            # Bank half at 1R, trail the rest — the runner logic already in the engine.
            partial_take_profit=price + (tp - price) * 0.5,
            partial_fraction=0.5,
        )
        return signal, "ok"
