"""WEEX AI Wars II — Composite Strategy v8

Modes:
1. High-conviction trend rider (BTC/ETH/SOL) — primary PnL engine
2. Mean-reversion in ranging markets (when enabled)
3. SOL keep-alive VWAP bounce — minimal size for activity rules

Fixes vs v7:
- Dead BB fallbacks fixed
- Per-symbol trade timers
- Real HTF candles wired into edges
- Mean reversion actually used
- Keep-alive stricter + small strength (risk manager scales size)
"""

import numpy as np
from datetime import datetime, timezone, timedelta

from ..core.models import Candle, Signal, Side, MarketRegime
from ..indicators.technical import (
    calculate_rsi, calculate_macd, calculate_bollinger_bands,
    calculate_atr, calculate_adx, calculate_ema, calculate_vwap,
    detect_regime, calculate_stochastic_rsi,
)
from .edges import EdgeStrategies


ATR_PROFILES = {
    "BTC": {"stop": 1.25, "target": 3.8, "breakeven": 0.75},
    "ETH": {"stop": 1.4, "target": 4.2, "breakeven": 0.9},
    "SOL": {"stop": 1.7, "target": 5.0, "breakeven": 1.1},
}
DEFAULT_PROFILE = {"stop": 1.5, "target": 4.5, "breakeven": 1.0}

KEEPALIVE_STOP = 0.9
KEEPALIVE_TARGET = 1.8


class CompositeStrategy:
    """v8 competition composite: conviction + MR + smart keep-alive."""

    def __init__(self, config: dict):
        self.config = config
        self.tf_config = config.get("strategy", {}).get("trend_follow", {})
        self.mr_config = config.get("strategy", {}).get("mean_reversion", {})
        self.ind_config = config.get("indicators", {})
        self.ka_config = config.get("strategy", {}).get("keepalive", {})
        self.edges = EdgeStrategies(config)

        self.consecutive_losses = 0
        self.circuit_breaker_until = None
        # Per-symbol last trade time (fixes shared timer bug)
        self.last_trade_time: dict[str, datetime] = {}
        self.mandatory_interval = timedelta(
            hours=self.ka_config.get("interval_hours", 8)
        )
        self.keepalive_symbol = self.ka_config.get("symbol_base", "SOL")

    def analyze(
        self,
        symbol: str,
        candles: list[Candle],
        funding_rate: float = 0.0,
        existing_positions=None,
        higher_tf_candles: list[Candle] | None = None,
    ):
        if len(candles) < 100:
            return None

        candle_time = candles[-1].timestamp

        signal = self._high_conviction(
            symbol, candles, funding_rate, existing_positions, higher_tf_candles
        )
        if signal:
            self.last_trade_time[symbol] = candle_time
            return signal

        if self.mr_config.get("enabled", True):
            signal = self._mean_reversion(
                symbol, candles, funding_rate, existing_positions, higher_tf_candles
            )
            if signal:
                self.last_trade_time[symbol] = candle_time
                return signal

        if self._needs_keepalive(candle_time, symbol):
            signal = self._keepalive_vwap(
                symbol, candles, funding_rate, existing_positions
            )
            if signal:
                self.last_trade_time[symbol] = candle_time
                return signal

        return None

    def _needs_keepalive(self, candle_time, symbol):
        if not self.ka_config.get("enabled", True):
            return False
        base = symbol.split("/")[0]
        if base != self.keepalive_symbol:
            return False
        last = self.last_trade_time.get(symbol)
        if last is None:
            # Also check any SOL key variants
            for k, v in self.last_trade_time.items():
                if k.startswith(self.keepalive_symbol):
                    last = v
                    break
        if last is None:
            return True
        return (candle_time - last) >= self.mandatory_interval

    # ---- High-Conviction Trend ----

    def _high_conviction(
        self, symbol, candles, funding_rate, existing_positions, higher_tf_candles
    ):
        base = symbol.split("/")[0]
        profile = ATR_PROFILES.get(base, DEFAULT_PROFILE)

        closes = np.array([c.close for c in candles])
        highs = np.array([c.high for c in candles])
        lows = np.array([c.low for c in candles])
        candle_time = candles[-1].timestamp

        if self._is_asia_session(candle_time):
            return None
        if self.circuit_breaker_until and candle_time < self.circuit_breaker_until:
            return None
        self.circuit_breaker_until = None

        adx = calculate_adx(highs, lows, closes, 14)
        adx_thresh = self.tf_config.get("adx_threshold", 28)
        if adx[-1] < adx_thresh:
            return None

        ema_9 = calculate_ema(closes, 9)
        ema_21 = calculate_ema(closes, 21)
        ema_50 = calculate_ema(closes, 50)
        slope = (
            (ema_21[-1] - ema_21[-5]) / ema_21[-5] * 100 if ema_21[-5] > 0 else 0
        )

        bullish = ema_9[-1] > ema_21[-1] > ema_50[-1]
        bearish = ema_9[-1] < ema_21[-1] < ema_50[-1]
        if not bullish and not bearish:
            return None

        edges = self.edges.analyze_all_edges(
            candles, funding_rate, higher_tf_candles
        )
        mult, direction = self.edges.get_combined_modifier(edges)
        want = "long" if bullish else "short"
        if self._count_directional_edges(edges, want) < 2:
            return None

        crowded = self.edges.funding_crowded
        if bullish and funding_rate > crowded:
            return None
        if bearish and funding_rate < -crowded:
            return None

        if existing_positions:
            for ps, pp in existing_positions:
                if ps == symbol:
                    return None
                if self._is_correlated(symbol, ps) and pp == want:
                    return None

        atr = calculate_atr(highs, lows, closes, 14)
        rsi = calculate_rsi(closes, 10)
        stoch_k, stoch_d = calculate_stochastic_rsi(closes)
        bb_u, bb_m, bb_l = calculate_bollinger_bands(closes, 20, 1.8)

        signal = None
        if bullish:
            signal = self._long_signal(
                symbol, closes, closes[-1], atr[-1], rsi[-1], rsi[-2],
                stoch_k, stoch_d, ema_21, slope, adx[-1], bb_l, profile,
            )
        if signal is None and bearish:
            signal = self._short_signal(
                symbol, closes, closes[-1], atr[-1], rsi[-1], rsi[-2],
                stoch_k, stoch_d, ema_21, slope, adx[-1], bb_u, profile,
            )
        if signal is None:
            return None

        if signal.risk_reward_ratio < 2.0:
            return None

        signal.strength = min(1.0, signal.strength * mult)
        if direction == "long" and signal.side == Side.SHORT and mult > 1.2:
            return None
        if direction == "short" and signal.side == Side.LONG and mult > 1.2:
            return None
        if signal.strength < 0.45:
            return None
        return signal

    # ---- Mean Reversion (ranging markets) ----

    def _mean_reversion(
        self, symbol, candles, funding_rate, existing_positions, higher_tf_candles
    ):
        closes = np.array([c.close for c in candles])
        highs = np.array([c.high for c in candles])
        lows = np.array([c.low for c in candles])
        candle_time = candles[-1].timestamp

        if self._is_asia_session(candle_time):
            return None
        if self.circuit_breaker_until and candle_time < self.circuit_breaker_until:
            return None

        regime = detect_regime(
            highs, lows, closes,
            adx_threshold=self.tf_config.get("adx_threshold", 28),
        )
        if regime not in (MarketRegime.RANGING, MarketRegime.HIGH_VOLATILITY):
            return None

        # Avoid mean-reverting hard against HTF trend
        if higher_tf_candles and len(higher_tf_candles) >= 50:
            h = np.array([c.close for c in higher_tf_candles])
            h9, h21 = calculate_ema(h, 9), calculate_ema(h, 21)
            htf_strong_bull = h9[-1] > h21[-1] * 1.004
            htf_strong_bear = h9[-1] < h21[-1] * 0.996
        else:
            htf_strong_bull = htf_strong_bear = False

        if existing_positions:
            for ps, pp in existing_positions:
                if ps == symbol:
                    return None

        rsi_period = self.mr_config.get("rsi_period", 10)
        ob = self.mr_config.get("rsi_overbought", 75)
        os_ = self.mr_config.get("rsi_oversold", 25)
        bb_period = self.mr_config.get("bb_period", 20)
        bb_std = self.mr_config.get("bb_std", 2.0)

        rsi = calculate_rsi(closes, rsi_period)
        bb_u, bb_m, bb_l = calculate_bollinger_bands(closes, bb_period, bb_std)
        atr = calculate_atr(highs, lows, closes, 14)
        price = closes[-1]
        atr_v = atr[-1]
        if atr_v <= 0 or np.isnan(bb_l[-1]):
            return None

        base = symbol.split("/")[0]
        profile = ATR_PROFILES.get(base, DEFAULT_PROFILE)
        lev = self.config.get("trading", {}).get("default_leverage", 5)

        # Long mean reversion: oversold at lower band
        if (
            price <= bb_l[-1] * 1.002
            and rsi[-1] <= os_
            and rsi[-1] > rsi[-2]
            and not htf_strong_bear
        ):
            stop = price - atr_v * profile["stop"] * 0.9
            target = min(bb_m[-1], price + atr_v * 2.2)
            if abs(target - price) / abs(price - stop) < 1.5:
                return None
            return Signal(
                symbol=symbol, side=Side.LONG, strength=0.55,
                strategy="mean_reversion",
                entry_price=price, stop_loss=stop, take_profit=target,
                leverage=lev,
                reason=f"MR LONG: RSI={rsi[-1]:.0f} at lower BB",
            )

        # Short mean reversion
        if (
            price >= bb_u[-1] * 0.998
            and rsi[-1] >= ob
            and rsi[-1] < rsi[-2]
            and not htf_strong_bull
        ):
            stop = price + atr_v * profile["stop"] * 0.9
            target = max(bb_m[-1], price - atr_v * 2.2)
            if abs(price - target) / abs(stop - price) < 1.5:
                return None
            return Signal(
                symbol=symbol, side=Side.SHORT, strength=0.55,
                strategy="mean_reversion",
                entry_price=price, stop_loss=stop, take_profit=target,
                leverage=lev,
                reason=f"MR SHORT: RSI={rsi[-1]:.0f} at upper BB",
            )

        return None

    # ---- Keep-Alive VWAP ----

    def _keepalive_vwap(self, symbol, candles, funding_rate, existing_positions):
        closes = np.array([c.close for c in candles])
        highs = np.array([c.high for c in candles])
        lows = np.array([c.low for c in candles])
        volumes = np.array([c.volume for c in candles])
        candle_time = candles[-1].timestamp

        if self.circuit_breaker_until and candle_time < self.circuit_breaker_until:
            return None
        if existing_positions:
            for ps, _ in existing_positions:
                if ps == symbol:
                    return None

        adx = calculate_adx(highs, lows, closes, 14)
        if adx[-1] < 14:
            return None

        ema_9 = calculate_ema(closes, 9)
        ema_21 = calculate_ema(closes, 21)
        is_bullish = ema_9[-1] > ema_21[-1]
        is_bearish = ema_9[-1] < ema_21[-1]
        if not is_bullish and not is_bearish:
            return None

        vwap = calculate_vwap(highs, lows, closes, volumes, 20)
        if np.isnan(vwap[-1]):
            return None

        atr = calculate_atr(highs, lows, closes, 14)
        atr_v = atr[-1]
        price = closes[-1]
        vwap_val = vwap[-1]
        if atr_v <= 0:
            return None

        avg_vol = np.mean(volumes[-20:])
        vol_ratio = volumes[-1] / avg_vol if avg_vol > 0 else 1
        rsi = calculate_rsi(closes, 10)
        lev = self.config.get("trading", {}).get("default_leverage", 5)
        ka_strength = self.ka_config.get("strength", 0.18)

        if is_bullish:
            near = 0 <= (price - vwap_val) / atr_v <= 0.9
            bounce = closes[-1] > closes[-2] and closes[-1] > opens_safe(candles[-1])
            rsi_ok = 32 < rsi[-1] < 62
            if near and bounce and rsi_ok and vol_ratio >= 1.0:
                return Signal(
                    symbol=symbol, side=Side.LONG, strength=ka_strength,
                    strategy="keepalive_vwap",
                    entry_price=price,
                    stop_loss=price - atr_v * KEEPALIVE_STOP,
                    take_profit=price + atr_v * KEEPALIVE_TARGET,
                    leverage=lev,
                    reason=f"Keep-alive VWAP LONG ADX={adx[-1]:.0f} vol={vol_ratio:.1f}x",
                )

        if is_bearish:
            near = 0 <= (vwap_val - price) / atr_v <= 0.9
            reject = closes[-1] < closes[-2] and closes[-1] < opens_safe(candles[-1])
            rsi_ok = 38 < rsi[-1] < 68
            if near and reject and rsi_ok and vol_ratio >= 1.0:
                return Signal(
                    symbol=symbol, side=Side.SHORT, strength=ka_strength,
                    strategy="keepalive_vwap",
                    entry_price=price,
                    stop_loss=price + atr_v * KEEPALIVE_STOP,
                    take_profit=price - atr_v * KEEPALIVE_TARGET,
                    leverage=lev,
                    reason=f"Keep-alive VWAP SHORT ADX={adx[-1]:.0f} vol={vol_ratio:.1f}x",
                )

        # BB fallback (reachable)
        bb_u, bb_m, bb_l = calculate_bollinger_bands(closes, 20, 1.8)
        bb_strength = max(0.1, ka_strength * 0.7)

        if is_bullish and price <= bb_l[-1] * 1.004 and rsi[-1] < 42 and rsi[-1] > rsi[-2]:
            return Signal(
                symbol=symbol, side=Side.LONG, strength=bb_strength,
                strategy="keepalive_bb",
                entry_price=price,
                stop_loss=price - atr_v * KEEPALIVE_STOP,
                take_profit=price + atr_v * KEEPALIVE_TARGET,
                leverage=lev,
                reason=f"Keep-alive BB bounce ADX={adx[-1]:.0f}",
            )

        if is_bearish and price >= bb_u[-1] * 0.996 and rsi[-1] > 58 and rsi[-1] < rsi[-2]:
            return Signal(
                symbol=symbol, side=Side.SHORT, strength=bb_strength,
                strategy="keepalive_bb",
                entry_price=price,
                stop_loss=price + atr_v * KEEPALIVE_STOP,
                take_profit=price - atr_v * KEEPALIVE_TARGET,
                leverage=lev,
                reason=f"Keep-alive BB reject ADX={adx[-1]:.0f}",
            )

        return None

    # ---- Signal builders (BB fallback reachable) ----

    def _long_signal(
        self, symbol, closes, price, atr_val, rsi, prev_rsi, sk, sd,
        ema21, slope, adx_val, bb_lower, profile,
    ):
        lev = self.config.get("trading", {}).get("default_leverage", 5)
        ema_val = ema21[-1]
        dist = (price - ema_val) / atr_val if atr_val > 0 else 999

        # Primary: pullback to EMA with momentum reclaim
        if (
            -0.5 <= dist <= 1.5
            and closes[-1] > closes[-2]
            and rsi > prev_rsi
            and 30 < rsi < 65
            and sk[-1] > sd[-1]
            and price >= ema_val * 0.997
            and slope > 0
        ):
            strength = 0.68 + (0.12 if adx_val > 32 else 0)
            stop = min(ema_val - atr_val * profile["breakeven"], price - atr_val * profile["stop"])
            return Signal(
                symbol=symbol, side=Side.LONG, strength=min(1.0, strength),
                strategy="trend_rider", entry_price=price,
                stop_loss=stop, take_profit=price + atr_val * profile["target"],
                leverage=lev, reason=f"Trend LONG ADX={adx_val:.0f}",
            )

        # Secondary: BB lower touch in uptrend
        if (
            price <= bb_lower[-1] * 1.005
            and rsi > prev_rsi
            and slope > 0
            and rsi < 55
        ):
            return Signal(
                symbol=symbol, side=Side.LONG, strength=0.58,
                strategy="trend_rider_bb", entry_price=price,
                stop_loss=price - atr_val * profile["stop"],
                take_profit=price + atr_val * profile["target"],
                leverage=lev, reason=f"BB trend LONG ADX={adx_val:.0f}",
            )
        return None

    def _short_signal(
        self, symbol, closes, price, atr_val, rsi, prev_rsi, sk, sd,
        ema21, slope, adx_val, bb_upper, profile,
    ):
        lev = self.config.get("trading", {}).get("default_leverage", 5)
        ema_val = ema21[-1]
        dist = (ema_val - price) / atr_val if atr_val > 0 else 999

        if (
            -0.5 <= dist <= 1.5
            and closes[-1] < closes[-2]
            and rsi < prev_rsi
            and 35 < rsi < 70
            and sk[-1] < sd[-1]
            and price <= ema_val * 1.003
            and slope < 0
        ):
            strength = 0.68 + (0.12 if adx_val > 32 else 0)
            stop = max(ema_val + atr_val * profile["breakeven"], price + atr_val * profile["stop"])
            return Signal(
                symbol=symbol, side=Side.SHORT, strength=min(1.0, strength),
                strategy="trend_rider", entry_price=price,
                stop_loss=stop, take_profit=price - atr_val * profile["target"],
                leverage=lev, reason=f"Trend SHORT ADX={adx_val:.0f}",
            )

        if (
            price >= bb_upper[-1] * 0.995
            and rsi < prev_rsi
            and slope < 0
            and rsi > 45
        ):
            return Signal(
                symbol=symbol, side=Side.SHORT, strength=0.58,
                strategy="trend_rider_bb", entry_price=price,
                stop_loss=price + atr_val * profile["stop"],
                take_profit=price - atr_val * profile["target"],
                leverage=lev, reason=f"BB trend SHORT ADX={adx_val:.0f}",
            )
        return None

    # ---- Utilities ----

    def record_loss(self, t):
        self.consecutive_losses += 1
        if self.consecutive_losses >= 3:
            self.circuit_breaker_until = t + timedelta(hours=6)
            self.consecutive_losses = 0

    def record_win(self):
        self.consecutive_losses = 0

    def _is_asia_session(self, t):
        if t is None:
            return False
        if t.tzinfo is None:
            t = t.replace(tzinfo=timezone.utc)
        return 0 <= t.hour < 8

    def _is_correlated(self, s1, s2):
        b1, b2 = s1.split("/")[0], s2.split("/")[0]
        majors = {"BTC", "ETH"}
        return b1 in majors and b2 in majors

    def _count_directional_edges(self, edges, d):
        c = 0
        if edges.get("liquidation", {}).get("detected") and edges["liquidation"].get("direction") == d:
            c += 1
        if edges.get("funding", {}).get("signal") and edges["funding"].get("direction") == d:
            c += 1
        v = edges.get("volume", {})
        if v.get("anomaly") and (
            (d == "long" and v.get("direction") == "accumulation")
            or (d == "short" and v.get("direction") == "distribution")
        ):
            c += 1
        if edges.get("mtf", {}).get("aligned") and edges["mtf"].get("direction") == d:
            c += 1
        mom = edges.get("momentum", {})
        if mom.get("quality") and mom.get("direction") == d:
            c += 1
        return c


def opens_safe(candle: Candle) -> float:
    return candle.open if candle else 0.0
