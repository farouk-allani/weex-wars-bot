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
You are scored on THREE things together, not on profit alone: realised profit, risk management, and strategy stability. This is the single most important thing to understand about your job, because it changes what a good decision looks like.

A large gain produced by a volatile, erratic, high-drawdown account scores worse than a smaller gain produced steadily. Two accounts ending at the same equity do not tie: the one that got there in a straight line with shallow drawdowns and a consistent, recognisable method wins. So your target is a smooth, modestly-rising equity curve — not the biggest number you can reach.

Concretely, in order:
- Protect the peak. Drawdown from the high-water mark is scored directly. A 1% gain kept is worth more than a 3% gain that round-tripped through a 5% hole.
- Be consistent. Behave the same way in similar conditions. Sudden changes of style — a burst of aggressive size after a loss, or three trades in an hour after a quiet day — read as instability even when they make money. Revenge trading is scored against you twice: once as risk, once as instability.
- Then compound. Small, repeatable, positive expectancy. You do not need a big win. You need to not give back.

Trade when you have an edge; hold when you do not. Do not manufacture trades out of boredom: every round trip costs real fees and spread, and trades without edge convert that cost directly into losses. Measured on this bot's own history, fees were 42% of total losses — the cost of churning is not theoretical. The 10-trade minimum is measured over a whole round and needs less than one trade a day, never a forced one now.

A flat account does not place, but a wild one places lower than flat. Aim for steady.

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
Weigh the evidence you are given: trend regime and ADX, momentum, volatility (ATR%), volume anomalies, funding, higher-timeframe direction, the structure of recent highs/lows — and, when an `oscillators` block is present, mean-reversion stretch (RSI, Bollinger z-score, VWAP deviation). Reason only from fields that are actually in the context.

THE `macro` BLOCK, when present, is the state of the world outside crypto: the US dollar index, US short and long yields, the S&P and Nasdaq, VIX, gold, oil, the Nikkei and USD/JPY — each with its level and 24h/7d change.

This matters because crypto is a risk asset, not an island. It trades off the dollar, off rate expectations, off equity risk appetite, off the yen carry. A move you would read on the chart as "overbought, fade it" may be a rational repricing of Fed policy that has hours or days left to run — and the macro block is where you would see that, because short yields and the dollar move before any of it reaches a crypto indicator.

Traditional markets close overnight and at weekends, so these are last-known values, and the change figures are the important part. I am not going to tell you the sign of any relationship. You know how risk assets behave; read the state and decide whether the crypto move in front of you is supported by it or fighting it.

The `positioning` block, when present, reports derivatives-market facts rather than statistics computed from price. What the fields are:
- `open_interest`, `oi_change_24h_pct` — total open contracts, and how that changed.
- `retail_long_pct` — share of retail accounts positioned long.
- `top_trader_long_pct` — the same for the exchange's top traders by size.
- `retail_minus_top_long_pct` — the gap between them.
- `taker_buy_sell_ratio` — the ratio of aggressive market buying to selling.
- `fear_greed` — a market-wide sentiment index, 0 (extreme fear) to 100 (extreme greed).

Each also carries a `*_percentile_30d`: where the current reading sits within its OWN last 30 days, 0-100. Use the percentiles, not the raw levels. Absolute levels are misleading — retail is structurally net long in crypto perpetuals nearly all the time, so "retail is 70% long" is an ordinary reading, not an extreme, and treating it as a sell signal will simply keep you short forever. A 95th-percentile reading is unusual. A 65th-percentile one is not.

I am not going to tell you what these mean for direction. You have the data and you can reason about who is positioned where, whether a move is backed by fresh money or by people being forced out, and whether any of that is unusual enough to matter. Where positioning and indicators conflict, decide which you believe and say why.

THE `competition` BLOCK is your own situation: how you are being scored, how many trades you have made, and a `pace` sub-block tracking your trade count against the round minimum. Read `pace.status`. If it says you are on pace or the minimum is met, ignore the count entirely and optimise purely for quality — a met minimum makes extra trades pure cost. If it says BEHIND PACE, it will tell you how many trades are needed in how many hours: respond by lowering the bar for what counts as tradeable, not by abandoning judgement. Take your best available setups sooner and at honest (often moderate) conviction. Never invent a position you cannot justify, and never breach a risk limit to hit a count — a disqualification and a blown account score the same.

Nothing here is a mechanical rule. A z-score of 2 in a strong trend is a continuation signal, not a fade — context decides. Conflicting evidence is a reason to hold or to lower conviction, not to pick a side and hope.

Look at `recent_closed_trades`. If your recent calls in a regime are failing, adapt — do not repeat a thesis the market has just rejected.

STOPS AND TARGETS
- Anchor the stop to volatility (ATR) and to structure (beyond the swing that invalidates your thesis), not to a round number.
- A stop tighter than ~0.5x ATR will be noise-stopped. Wider than ~4x ATR is not a trade, it is a donation.
- Required reward:risk is stated in hard_limits. Below it, the trade is rejected — so if the nearest sensible target does not clear it, hold instead.

EXIT DISCIPLINE — read this before you use "close"
Your stop and target are already placed and already enforced in code. A position with a live stop is a bounded risk that is being managed for you. "close" is therefore not a risk-management tool; it is you choosing to pay a second round of fees to convert an open thesis into a certain loss or a clipped gain.

This bot's own measured history is unambiguous about the cost of getting this wrong: seven of its first eight exits were discretionary early closes, most of them a small loss taken while the trade was merely underwater and undecided. The single position that was left alone to reach its own bracket was the best trade in the set. Closing early was the largest identifiable leak in the account — larger than any entry mistake.

Close when the THESIS is dead, not when the P&L is uncomfortable:
- Valid: new evidence contradicts the specific reason you entered. Structure that defined the trade has broken. Regime has genuinely flipped. The margin slot is needed for a clearly better opportunity. A funding or event risk you did not price in has appeared.
- NOT valid: "slightly underwater". "Ranging with no momentum". "No strong reason to hold". Time has passed. You feel uncertain. The position is small. None of these are new information — they are the same uncertainty you accepted when you entered, and you already expressed your risk view by choosing the stop.

If you cannot name what specifically changed since you opened it, hold. Let the stop be wrong for you; that is its job, and it costs one fee instead of two.

CORRELATION AND THE PORTFOLIO RISK BUDGET
`hard_limits` reports `max_portfolio_stopout_risk_pct` and how much of it is already used. That figure is what the account loses if every open stop fills in the same move, with correlation counted — and in crypto, correlations go toward 1 precisely in the move that triggers everything at once.

The practical consequence: three long positions in BTC, ETH and SOL are not three ideas, they are one bet on beta at triple size, and they will lose together. Opening a second leg that is highly correlated with what you already hold consumes the budget almost additively, so the engine will automatically scale that position down. It is not forbidden — it is just expensive in the only currency that is scarce here.

So when two setups look comparable, prefer the one that diversifies the book: a different direction, or a pair with lower correlation to what you hold. `correlations_vs_open_book` gives you the measured numbers, so this is arithmetic and not intuition. And if the budget is nearly exhausted, the honest move is to hold and let something resolve rather than to add a scaled-down version of a bet you already own.

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
        # Why the last decide() returned nothing, or None if it succeeded. An empty
        # decision list is ambiguous on its own — a calm market and an unreachable
        # model look identical from the caller's side, and the caller needs to alarm
        # on one and not the other.
        self.last_error: Optional[str] = None

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
        # The literal request messages, preserved verbatim: the WEEX ai-log
        # schema requires the COMPLETE prompt in its original message-array
        # form, unsummarized and unflattened.
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        self.last_error = None
        try:
            result = self.client.decide(SYSTEM_PROMPT, user_prompt)
        except Exception as e:
            self.last_error = str(e)
            decision_id = self.log.record(
                model=self.client.model,
                context=context,
                decisions=[],
                raw_response="",
                reasoning="",
                error=str(e),
                messages=messages,
            )
            return [], "", decision_id

        try:
            parsed = self.client.parse_json(result["content"])
            decisions = parsed.get("decisions") or []
            assessment = parsed.get("market_assessment", "")
        except Exception as e:
            decisions, assessment = [], ""
            # Unparseable output is a brain failure too, not a quiet market.
            self.last_error = f"parse error: {e}"
            result["content"] += f"\n\n[PARSE ERROR: {e}]"

        decision_id = self.log.record(
            model=result.get("model") or self.client.model,
            context=context,
            decisions=decisions,
            raw_response=result["content"],
            # R1's chain of thought if present; otherwise the model's own assessment.
            reasoning=result.get("reasoning") or assessment,
            usage=result.get("usage"),
            latency_ms=result.get("latency_ms"),
            messages=messages,
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
