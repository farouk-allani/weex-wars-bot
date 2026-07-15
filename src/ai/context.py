"""Assemble the market/account state the model reasons over.

Design notes:
- Everything is derived, not raw. 120 raw candles is ~4k tokens of noise the model
  reads poorly; the same information as RSI/z-score/ADX/ATR% is a few hundred tokens
  it reads well. We hand it the *interpretation-ready* numbers and let it interpret.
- Recent closed trades go in deliberately. The rules demand "adaptive reasoning"
  rather than static automation, and a model that can see its last five outcomes can
  actually adapt. This is the loop a fixed rule engine structurally cannot have.
- Risk limits are included so the model reasons *within* them instead of proposing
  trades the risk manager will silently veto.
"""

from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np

from ..core.models import AccountState, Candle
from ..indicators.technical import (
    calculate_adx,
    calculate_atr,
    calculate_bollinger_bands,
    calculate_ema,
    calculate_macd,
    calculate_rsi,
    calculate_stochastic_rsi,
    calculate_vwap,
    detect_regime,
)


def _r(x: Any, n: int = 4) -> Any:
    try:
        v = float(x)
        if not np.isfinite(v):
            return None
        return round(v, n)
    except Exception:
        return None


def _clean(obj: Any) -> Any:
    """Coerce numpy scalars to native Python types.

    The indicator and edge code returns np.int64/np.bool_/np.float64. json.dumps
    would fall back to default=str on those and hand the model the *string* "28" or
    "True" instead of a number or a boolean — which it reads, and reasons over,
    noticeably worse. Fix the types rather than the symptom.
    """
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        v = float(obj)
        return round(v, 6) if np.isfinite(v) else None
    if isinstance(obj, np.ndarray):
        return _clean(obj.tolist())
    return obj


def symbol_snapshot(
    symbol: str,
    candles: list[Candle],
    funding_rate: float = 0.0,
    htf_candles: Optional[list[Candle]] = None,
    edges: Optional[dict] = None,
    positioning: Optional[dict] = None,
    include_oscillators: bool = False,
) -> dict:
    """
    include_oscillators: RSI / StochRSI / Bollinger z-score / VWAP deviation.

    Off by default, and that is not an aesthetic choice.
    Feeding these to the model triggers a memorised playbook — "price at upper BB +
    overbought stoch => short to VWAP" — which it recited on ~80% of decisions,
    verbatim, while ignoring macro and positioning entirely. It fades every rally,
    including macro-driven ones, and it loses. The oscillators have no demonstrated
    edge here and they actively crowd out the inputs that might.
    """
    closes = np.array([c.close for c in candles], dtype=float)
    highs = np.array([c.high for c in candles], dtype=float)
    lows = np.array([c.low for c in candles], dtype=float)
    vols = np.array([c.volume for c in candles], dtype=float)

    price = float(closes[-1])
    rsi = calculate_rsi(closes, 14)
    macd_line, signal_line, hist = calculate_macd(closes)
    upper, mid, lower = calculate_bollinger_bands(closes, 20, 2.0)
    atr = calculate_atr(highs, lows, closes, 14)
    adx = calculate_adx(highs, lows, closes, 14)
    vwap = calculate_vwap(highs, lows, closes, vols, 20)
    k, d = calculate_stochastic_rsi(closes)

    band_width = float(upper[-1] - lower[-1])
    # Z-score of price within the Bollinger channel: the single most decision-relevant
    # number for a mean-reversion call, so give it to the model directly.
    z = (price - float(mid[-1])) / (band_width / 4) if band_width > 0 else 0.0
    atr_val = float(atr[-1]) if len(atr) else 0.0

    vol_ratio = (
        float(vols[-1] / np.mean(vols[-20:])) if len(vols) >= 20 and np.mean(vols[-20:]) > 0 else 1.0
    )

    def pct_change(n: int) -> Optional[float]:
        return _r((price / float(closes[-n]) - 1) * 100, 2) if len(closes) > n else None

    snap = {
        "symbol": symbol,
        "price": _r(price),
        "change_pct": {
            "1h": pct_change(1),
            "4h": pct_change(4),
            "24h": pct_change(24),
            "7d": pct_change(168),
        },
        "trend": {
            "regime": str(detect_regime(closes, highs, lows)).split(".")[-1].lower(),
            "adx": _r(adx[-1], 1) if len(adx) else None,
            "ema_fast": _r(calculate_ema(closes, 9)[-1]),
            "ema_slow": _r(calculate_ema(closes, 21)[-1]),
            "htf_4h_direction": _htf_direction(htf_candles),
        },
        "momentum": {
            "macd_hist": _r(hist[-1], 5) if len(hist) else None,
            "macd_cross": (
                "bullish" if len(hist) > 1 and hist[-1] > 0 >= hist[-2]
                else "bearish" if len(hist) > 1 and hist[-1] < 0 <= hist[-2]
                else "none"
            ),
        },
        "volatility": {
            # ATR as a % of price is what actually sizes the stop, so surface it
            # pre-computed rather than making the model divide.
            "atr": _r(atr_val),
            "atr_pct": _r(atr_val / price * 100, 2) if price else None,
        },
        "volume": {
            "ratio_vs_20": _r(vol_ratio, 2),
            "anomaly": vol_ratio > 2.0,
        },
        "funding": {
            "rate_8h": _r(funding_rate, 6),
            "annualised_pct": _r(funding_rate * 3 * 365 * 100, 2),
            "crowded": (
                "longs_pay_shorts" if funding_rate > 0.0001
                else "shorts_pay_longs" if funding_rate < -0.0001
                else "neutral"
            ),
        },
        "levels": {
            "recent_high_24": _r(float(np.max(highs[-24:]))) if len(highs) >= 24 else None,
            "recent_low_24": _r(float(np.min(lows[-24:]))) if len(lows) >= 24 else None,
            "recent_high_72": _r(float(np.max(highs[-72:]))) if len(highs) >= 72 else None,
            "recent_low_72": _r(float(np.min(lows[-72:]))) if len(lows) >= 72 else None,
        },
    }
    if include_oscillators:
        snap["oscillators"] = {
            "rsi_14": _r(rsi[-1], 1) if len(rsi) else None,
            "stoch_rsi_k": _r(k[-1], 1) if len(k) else None,
            "bb_upper": _r(upper[-1]),
            "bb_mid": _r(mid[-1]),
            "bb_lower": _r(lower[-1]),
            "bb_zscore": _r(z, 2),
            "vwap_20": _r(vwap[-1]) if len(vwap) else None,
            "vwap_deviation_pct": _r((price / float(vwap[-1]) - 1) * 100, 2)
            if len(vwap) and vwap[-1]
            else None,
        }
    if edges:
        # Ships with the oscillators: the "volume divergence" detector fires on
        # almost every bar and the model quotes it as corroboration for the fade.
        snap["edge_signals"] = edges
    if positioning:
        # Who is on the wrong side of this move. The one input here that is
        # information rather than interpretation of price.
        snap["positioning"] = positioning
    return _clean(snap)


def _htf_direction(htf: Optional[list[Candle]]) -> Optional[str]:
    if not htf or len(htf) < 21:
        return None
    c = np.array([x.close for x in htf], dtype=float)
    fast, slow = calculate_ema(c, 9)[-1], calculate_ema(c, 21)[-1]
    if fast > slow * 1.001:
        return "up"
    if fast < slow * 0.999:
        return "down"
    return "flat"


def build_context(
    *,
    symbols_data: list[dict],
    account: AccountState,
    risk,
    recent_trades: list,
    competition: Optional[dict] = None,
    fear_greed: Optional[int] = None,
    macro: Optional[dict] = None,
) -> dict:
    """Full decision context: market + book + risk envelope + own recent results."""
    drawdown = 0.0
    if risk.peak_equity > 0:
        drawdown = max(0.0, (risk.peak_equity - account.equity) / risk.peak_equity)

    positions = []
    for p in account.positions:
        age_h = (datetime.utcnow() - p.opened_at).total_seconds() / 3600
        positions.append({
            "symbol": p.symbol,
            "side": p.side.value,
            "entry_price": _r(p.entry_price),
            "size": _r(p.size, 6),
            "leverage": p.leverage,
            "stop_loss": _r(p.stop_loss),
            "take_profit": _r(p.take_profit),
            "unrealized_pnl": _r(p.unrealized_pnl, 2),
            "age_hours": _r(age_h, 1),
            "partial_taken": p.partial_taken,
        })

    # The model's own recent track record. This is the input that makes adaptation
    # possible — it can see that, say, its last three shorts all stopped out.
    history = []
    for t in recent_trades[-8:]:
        history.append({
            "symbol": t.symbol,
            "side": t.side.value if hasattr(t.side, "value") else str(t.side),
            "pnl": _r(t.pnl, 2),
            "pnl_pct": _r(t.pnl_pct, 2),
            "exit_reason": t.exit_reason,
            "duration_hours": _r((t.duration_seconds or 0) / 3600, 1),
        })

    ctx = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "account": {
            "equity": _r(account.equity, 2),
            "balance": _r(account.balance, 2),
            "available_margin": _r(account.available_margin, 2),
            "open_positions": len(account.positions),
            "unrealized_pnl": _r(account.unrealized_pnl, 2),
        },
        "risk_state": {
            "drawdown_from_peak_pct": _r(drawdown * 100, 2),
            "daily_pnl": _r(risk.daily_pnl, 2),
            "consecutive_losses": risk.consecutive_losses,
            "consecutive_wins": risk.consecutive_wins,
            "kill_switch_active": risk.is_killed,
            "in_cooldown": risk.cooldown_until is not None,
        },
        # Stated up front so the model reasons inside the envelope rather than
        # proposing trades that will be silently vetoed.
        "hard_limits": {
            "max_risk_per_trade_pct": _r(risk.max_risk_per_trade * 100, 2),
            "max_open_positions": risk.max_open_positions,
            "max_same_side_positions": risk.max_same_side_positions,
            "max_drawdown_pct": _r(risk.max_drawdown * 100, 1),
            "daily_loss_limit_pct": _r(risk.daily_loss_limit * 100, 2),
            "note": (
                "Position size is computed by the risk engine from your conviction "
                "and stop distance. You do not set size. Stops are enforced in code "
                "and cannot be widened after entry."
            ),
        },
        "open_positions": positions,
        "recent_closed_trades": history,
        "markets": symbols_data,
    }
    if macro:
        # Why the tape is moving, not just that it moved.
        ctx["macro"] = macro
    if fear_greed is not None:
        ctx["market_sentiment"] = {
            "fear_greed_index": fear_greed,
            "label": (
                "extreme fear" if fear_greed <= 25 else
                "fear" if fear_greed <= 45 else
                "neutral" if fear_greed <= 55 else
                "greed" if fear_greed <= 75 else "extreme greed"
            ),
            # Scale only. What an extreme means for direction is the model's call —
            # the last time a "note" here editorialised, the model traded the label.
            "note": "0 = extreme fear, 100 = extreme greed.",
        }
    if competition:
        ctx["competition"] = competition
    return _clean(ctx)
