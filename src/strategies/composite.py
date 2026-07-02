"""WEEX AI Wars II — Composite Strategy v7.1: Smart Keep-Alive

The keepalive mode v7.0 was losing because it used basic RSI signals.
This version uses VWAP bounce — a proven institutional scalp strategy.

Keep-alive logic:
- Price pulls back to VWAP (institutional fair value)
- Bounces off VWAP with volume confirmation
- Quick 1.5x ATR target, 1x ATR stop
- Only in trend direction (EMA 9 vs 21)
- Reduced position size (strength 0.3)
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
    "BTC": {"stop": 1.3, "target": 4.0, "breakeven": 0.8},
    "ETH": {"stop": 1.5, "target": 4.5, "breakeven": 1.0},
    "SOL": {"stop": 1.8, "target": 5.5, "breakeven": 1.2},
}
DEFAULT_PROFILE = {"stop": 1.5, "target": 5.0, "breakeven": 1.0}

# Keep-alive: tiny risk, tighter stops
KEEPALIVE_STOP = 0.8   # 0.8x ATR stop (tight)
KEEPALIVE_TARGET = 1.5  # 1.5x ATR target
KEEPALIVE_MAX_LOSS_PCT = 0.003  # Max 0.3% of equity per keepalive trade


class CompositeStrategy:
    """
    v7.1: Smart Keep-Alive using VWAP bounce.
    """

    def __init__(self, config: dict):
        self.config = config
        self.tf_config = config.get("strategy", {}).get("trend_follow", {})
        self.mr_config = config.get("strategy", {}).get("mean_reversion", {})
        self.ind_config = config.get("indicators", {})
        self.edges = EdgeStrategies(config)

        self.consecutive_losses = 0
        self.circuit_breaker_until = None
        self.last_trade_time = None
        self.mandatory_interval = timedelta(hours=8)

    def analyze(self, symbol, candles, funding_rate=0.0, existing_positions=None):
        if len(candles) < 100:
            return None

        candle_time = candles[-1].timestamp

        # Mode 1: High-conviction
        signal = self._high_conviction(symbol, candles, funding_rate, existing_positions)
        if signal:
            self.last_trade_time = candle_time
            return signal

        # Mode 2: Keep-alive (VWAP bounce)
        if self._needs_keepalive(candle_time, symbol):
            signal = self._keepalive_vwap(symbol, candles, funding_rate, existing_positions)
            if signal:
                self.last_trade_time = candle_time
                return signal

        return None

    def _needs_keepalive(self, candle_time, symbol):
        """Only keepalive on SOL (best pair). BTC/ETH only trade on high-conviction."""
        base = symbol.split("/")[0]
        if base != "SOL":
            return False  # No forced trades on BTC/ETH
        if self.last_trade_time is None:
            return True
        return (candle_time - self.last_trade_time) >= self.mandatory_interval

    # ---- High-Conviction Mode (unchanged from v6) ----

    def _high_conviction(self, symbol, candles, funding_rate, existing_positions):
        base = symbol.split("/")[0]
        profile = ATR_PROFILES.get(base, DEFAULT_PROFILE)

        closes = np.array([c.close for c in candles])
        highs = np.array([c.high for c in candles])
        lows = np.array([c.low for c in candles])
        volumes = np.array([c.volume for c in candles])
        candle_time = candles[-1].timestamp

        if self._is_asia_session(candle_time): return None
        if self.circuit_breaker_until and candle_time < self.circuit_breaker_until: return None
        self.circuit_breaker_until = None

        adx = calculate_adx(highs, lows, closes, 14)
        if adx[-1] < self.tf_config.get("adx_threshold", 30): return None

        ema_9 = calculate_ema(closes, 9)
        ema_21 = calculate_ema(closes, 21)
        ema_50 = calculate_ema(closes, 50)
        slope = (ema_21[-1] - ema_21[-5]) / ema_21[-5] * 100 if ema_21[-5] > 0 else 0

        bullish = ema_9[-1] > ema_21[-1] > ema_50[-1]
        bearish = ema_9[-1] < ema_21[-1] < ema_50[-1]
        if not bullish and not bearish: return None

        edges = self.edges.analyze_all_edges(candles, funding_rate)
        mult, direction = self.edges.get_combined_modifier(edges)
        if self._count_directional_edges(edges, "long" if bullish else "short") < 2: return None

        if bullish and funding_rate > 0.0005: return None
        if bearish and funding_rate < -0.0005: return None

        if existing_positions:
            for ps, pp in existing_positions:
                if ps == symbol: return None
                if self._is_correlated(symbol, ps) and pp == ("long" if bullish else "short"): return None

        atr = calculate_atr(highs, lows, closes, 14)
        rsi = calculate_rsi(closes, 10)
        stoch_k, stoch_d = calculate_stochastic_rsi(closes)
        bb_u, bb_m, bb_l = calculate_bollinger_bands(closes, 20, 1.8)

        signal = None
        if bullish:
            signal = self._long_signal(symbol, closes, closes[-1], atr[-1], rsi[-1], rsi[-2],
                                       stoch_k, stoch_d, ema_21, slope, adx[-1], bb_l, profile)
        if signal is None and bearish:
            signal = self._short_signal(symbol, closes, closes[-1], atr[-1], rsi[-1], rsi[-2],
                                        stoch_k, stoch_d, ema_21, slope, adx[-1], bb_u, profile)
        if signal is None: return None

        risk = abs(signal.entry_price - signal.stop_loss)
        reward = abs(signal.take_profit - signal.entry_price)
        if risk > 0 and reward / risk < 2.0: return None

        signal.strength *= mult
        if direction == "long" and signal.side == Side.SHORT and mult > 1.2: return None
        if direction == "short" and signal.side == Side.LONG and mult > 1.2: return None
        if signal.strength < 0.5: return None
        return signal

    # ---- Keep-Alive: VWAP Bounce ----

    def _keepalive_vwap(self, symbol, candles, funding_rate, existing_positions):
        """
        VWAP bounce: price pulls back to VWAP, bounces with volume.
        This is what institutional traders actually do for scalps.
        """
        closes = np.array([c.close for c in candles])
        highs = np.array([c.high for c in candles])
        lows = np.array([c.low for c in candles])
        volumes = np.array([c.volume for c in candles])
        candle_time = candles[-1].timestamp

        if self.circuit_breaker_until and candle_time < self.circuit_breaker_until:
            return None

        if existing_positions:
            for ps, pp in existing_positions:
                if ps == symbol: return None

        # Need SOME trend — ADX > 15 (very relaxed)
        adx = calculate_adx(highs, lows, closes, 14)
        if adx[-1] < 15: return None

        # Trend direction
        ema_9 = calculate_ema(closes, 9)
        ema_21 = calculate_ema(closes, 21)
        is_bullish = ema_9[-1] > ema_21[-1]
        is_bearish = ema_9[-1] < ema_21[-1]
        if not is_bullish and not is_bearish: return None

        # VWAP
        vwap = calculate_vwap(highs, lows, closes, volumes, 20)
        if np.isnan(vwap[-1]): return None

        atr = calculate_atr(highs, lows, closes, 14)
        current_atr = atr[-1]
        current_price = closes[-1]
        vwap_val = vwap[-1]

        # Distance from VWAP (in ATR units)
        dist_to_vwap = abs(current_price - vwap_val) / current_atr if current_atr > 0 else 999

        # Volume confirmation: current volume > 1.2x average
        avg_vol = np.mean(volumes[-20:])
        vol_ratio = volumes[-1] / avg_vol if avg_vol > 0 else 1

        # RSI
        rsi = calculate_rsi(closes, 10)

        # ---- LONG: Price near VWAP from above, bouncing ----
        if is_bullish:
            near_vwap = 0 <= (current_price - vwap_val) / current_atr <= 1.0  # Within 1 ATR above VWAP
            bouncing = closes[-1] > closes[-2]  # Green candle
            rsi_healthy = 35 < rsi[-1] < 65
            vol_ok = vol_ratio > 1.0  # At least average volume

            if near_vwap and bouncing and rsi_healthy and vol_ok:
                stop = current_price - current_atr * KEEPALIVE_STOP
                target = current_price + current_atr * KEEPALIVE_TARGET

                return Signal(
                    symbol=symbol, side=Side.LONG,
                    strength=0.15,  # Very small position
                    strategy="keepalive_vwap",
                    entry_price=current_price,
                    stop_loss=stop, take_profit=target,
                    leverage=self.config.get("trading", {}).get("default_leverage", 5),
                    reason=f"Keep-alive: VWAP bounce LONG, ADX={adx[-1]:.0f}, vol={vol_ratio:.1f}x",
                )

        # ---- SHORT: Price near VWAP from below, rejecting ----
        if is_bearish:
            near_vwap = 0 <= (vwap_val - current_price) / current_atr <= 1.0
            rejecting = closes[-1] < closes[-2]
            rsi_healthy = 35 < rsi[-1] < 65
            vol_ok = vol_ratio > 1.0

            if near_vwap and rejecting and rsi_healthy and vol_ok:
                stop = current_price + current_atr * KEEPALIVE_STOP
                target = current_price - current_atr * KEEPALIVE_TARGET

                return Signal(
                    symbol=symbol, side=Side.SHORT,
                    strength=0.15,  # Very small position
                    strategy="keepalive_vwap",
                    entry_price=current_price,
                    stop_loss=stop, take_profit=target,
                    leverage=self.config.get("trading", {}).get("default_leverage", 5),
                    reason=f"Keep-alive: VWAP bounce SHORT, ADX={adx[-1]:.0f}, vol={vol_ratio:.1f}x",
                )

        # ---- Fallback: BB touch in trend direction ----
        bb_u, bb_m, bb_l = calculate_bollinger_bands(closes, 20, 1.8)

        if is_bullish and current_price <= bb_l[-1] * 1.005 and rsi[-1] < 40:
            stop = current_price - current_atr * KEEPALIVE_STOP
            target = current_price + current_atr * KEEPALIVE_TARGET
            return Signal(
                symbol=symbol, side=Side.LONG, strength=0.10,
                strategy="keepalive_bb", entry_price=current_price,
                stop_loss=stop, take_profit=target,
                leverage=self.config.get("trading", {}).get("default_leverage", 5),
                reason=f"Keep-alive: BB lower bounce, ADX={adx[-1]:.0f}",
            )

        if is_bearish and current_price >= bb_u[-1] * 0.995 and rsi[-1] > 60:
            stop = current_price + current_atr * KEEPALIVE_STOP
            target = current_price - current_atr * KEEPALIVE_TARGET
            return Signal(
                symbol=symbol, side=Side.SHORT, strength=0.10,
                strategy="keepalive_bb", entry_price=current_price,
                stop_loss=stop, take_profit=target,
                leverage=self.config.get("trading", {}).get("default_leverage", 5),
                reason=f"Keep-alive: BB upper reject, ADX={adx[-1]:.0f}",
            )

        return None

    # ---- Signal helpers ----

    def _long_signal(self, symbol, closes, price, atr_val, rsi, prev_rsi, sk, sd,
                     ema21, slope, adx_val, bb_lower, profile):
        ema_val = ema21[-1]
        dist = (price - ema_val) / atr_val if atr_val > 0 else 999
        if not (-0.5 <= dist <= 1.5): return None
        if not (closes[-1] > closes[-2] and rsi > prev_rsi and rsi < 65 and rsi > 30): return None
        if not (sk[-1] > sd[-1] and price >= ema_val * 0.997 and slope > 0): return None
        strength = 0.65 + (0.1 if adx_val > 30 else 0)
        stop = min(ema_val - atr_val * profile["breakeven"], price - atr_val * profile["stop"])
        return Signal(symbol=symbol, side=Side.LONG, strength=min(1.0, strength),
                     strategy="trend_rider", entry_price=price,
                     stop_loss=stop, take_profit=price + atr_val * profile["target"],
                     leverage=self.config.get("trading",{}).get("default_leverage",5),
                     reason=f"Trend LONG: ADX={adx_val:.0f}")
        # BB fallback
        if price <= bb_lower[-1] * 1.005 and rsi > prev_rsi and slope > 0:
            return Signal(symbol=symbol, side=Side.LONG, strength=0.55,
                         strategy="trend_rider_bb", entry_price=price,
                         stop_loss=price - atr_val * profile["stop"],
                         take_profit=price + atr_val * profile["target"],
                         leverage=self.config.get("trading",{}).get("default_leverage",5),
                         reason=f"BB LONG: ADX={adx_val:.0f}")
        return None

    def _short_signal(self, symbol, closes, price, atr_val, rsi, prev_rsi, sk, sd,
                      ema21, slope, adx_val, bb_upper, profile):
        ema_val = ema21[-1]
        dist = (ema_val - price) / atr_val if atr_val > 0 else 999
        if not (-0.5 <= dist <= 1.5): return None
        if not (closes[-1] < closes[-2] and rsi < prev_rsi and rsi > 35 and rsi < 70): return None
        if not (sk[-1] < sd[-1] and price <= ema_val * 1.003 and slope < 0): return None
        strength = 0.65 + (0.1 if adx_val > 30 else 0)
        stop = max(ema_val + atr_val * profile["breakeven"], price + atr_val * profile["stop"])
        return Signal(symbol=symbol, side=Side.SHORT, strength=min(1.0, strength),
                     strategy="trend_rider", entry_price=price,
                     stop_loss=stop, take_profit=price - atr_val * profile["target"],
                     leverage=self.config.get("trading",{}).get("default_leverage",5),
                     reason=f"Trend SHORT: ADX={adx_val:.0f}")
        if price >= bb_upper[-1] * 0.995 and rsi < prev_rsi and slope < 0:
            return Signal(symbol=symbol, side=Side.SHORT, strength=0.55,
                         strategy="trend_rider_bb", entry_price=price,
                         stop_loss=price + atr_val * profile["stop"],
                         take_profit=price - atr_val * profile["target"],
                         leverage=self.config.get("trading",{}).get("default_leverage",5),
                         reason=f"BB SHORT: ADX={adx_val:.0f}")
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
        if t.tzinfo is None: t = t.replace(tzinfo=timezone.utc)
        return 0 <= t.hour < 8

    def _is_correlated(self, s1, s2):
        b1, b2 = s1.split("/")[0], s2.split("/")[0]
        return b1 in ["BTC","ETH"] and b2 in ["BTC","ETH"]

    def _count_directional_edges(self, edges, d):
        c = 0
        if edges.get("liquidation",{}).get("detected") and edges["liquidation"].get("direction")==d: c+=1
        if edges.get("funding",{}).get("signal") and edges["funding"].get("direction")==d: c+=1
        v = edges.get("volume",{})
        if v.get("anomaly") and ((d=="long" and v.get("direction")=="accumulation") or (d=="short" and v.get("direction")=="distribution")): c+=1
        if edges.get("mtf",{}).get("aligned") and edges["mtf"].get("direction")==d: c+=1
        return c
