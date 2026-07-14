"""AI decision layer.

WEEX AI Wars requires AI to be the primary decision-making component:
"Strategies relying solely on predefined conditions, manual rules, or historical
data curve fitting will not be recognized as compliant AI solutions."

So the model makes the call. The rule engine keeps exactly one job: refusing calls
that break risk limits. Proposal and veto stay separate on purpose — an LLM that
can widen its own stop is a liquidation waiting for a bad night.
"""

from .logbook import DecisionLog
from .context import build_context
from .client import DeepSeekClient, AIError
from .trader import AITrader

__all__ = ["DecisionLog", "build_context", "DeepSeekClient", "AIError", "AITrader"]
