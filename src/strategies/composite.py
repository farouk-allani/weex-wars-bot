"""WEEX AI Wars II — Composite Strategy v8.5

v8.5:
- Wick-rejection quality filter on MR entries
- Disabled-pairs support (pure-edge / skip toxic pairs)
- Stronger mid-band targets with optional runner extension
- Keep-alive book-level activity gate + weekly cap
"""

import numpy as np
from datetime import datetime, timezone, timedelta
from collections import defaultdict
from typing import Optional

from ..core.models import Candle, Signal, Side
from ..indicators.technical import (
    calculate_rsi, calculate_macd, calculate_bollinger_bands,
    calculate_atr, calculate_adx, calculate_ema, calculate_vwap,
    calculate_stochastic_rsi,
)
from .edges import EdgeStrategies


ATR_PROFILES = {
    "BTC": {"stop": 1.35, "target": 2.5, "breakeven": 0.8},
    "ETH": {"stop": 1.45, "target": 2.4, "breakeven": 0.85},
    "SOL": {"stop": 1.65, "target": 2.7, "breakeven": 0.95},
}
DEFAULT_PROFILE = {"stop": 1.45, "target": 2.5, "breakeven": 0.85}

KEEPALIVE_STOP = 1.0
KEEPALIVE_TARGET = 1.7


class CompositeStrategy:
    def __init__(self, config: dict):
        self.config = config
        self.tf_config = config.get("strategy", {}).get("trend_follow", {})
        self.mr_config = config.get("strategy", {}).get("mean_reversion", {})
        self.bo_config = config.get("strategy", {}).get("breakout", {})
        self.ka_config = config.get("strategy", {}).get("keepalive", {})
        self.edges = EdgeStrategies(config)

        self.consecutive_losses = 0
        self.circuit_breaker_until = None
        self.last_trade_time: dict[str, datetime] = {}
        self.last_any_trade_time: Optional[datetime] = None
        self.mandatory_interval = timedelta(
            hours=self.ka_config.get("interval_hours", 12)
        )
        self.keepalive_symbol = self.ka_config.get("symbol_base", "SOL")
        self._ka_max_per_week = int(self.ka_config.get("max_per_week", 3))
        self._ka_timestamps: list[datetime] = []

        comp = config.get("competition", {})
        self._min_edges = int(comp.get("min_edges", 1))
        self._allow_asia_mr = bool(comp.get("allow_asia_mr", True))
        self._min_rr = float(comp.get("min_rr", 1.35))
        self._skip_asia_trend = bool(comp.get("skip_asia_trend", True))
        self._htf_bias = bool(comp.get("htf_directional_bias", True))
        self._long_only = bool(comp.get("long_only", False))
        self._use_partial_tp = bool(comp.get("partial_tp", True))
        self._disabled_pairs = set(
            p.upper() for p in (comp.get("disabled_pairs") or self.mr_config.get("disabled_pairs") or [])
        )
        self._pure_edge = bool(comp.get("pure_edge", False))
        if self._pure_edge:
            # Pure edge: no keep-alive tax
            self.ka_config = {**self.ka_config, "enabled": False}

        # Adaptive: strategy name -> recent pnls (fed by engine/risk)
        self._strategy_scores: dict[str, float] = defaultdict(lambda: 1.0)

    def set_strategy_score(self, strategy: str, score: float):
        self._strategy_scores[strategy] = max(0.5, min(1.35, score))

    def sync_scores_from_risk(self, risk_manager):
        if not risk_manager:
            return
        for name in ("mean_reversion", "trend_pullback", "keepalive_vwap",
                     "keepalive_bb", "momentum_breakout"):
            self._strategy_scores[name] = risk_manager.get_strategy_weight(name)

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
        htf_dir = self._htf_direction(higher_tf_candles)
        base = symbol.split("/")[0]

        trend_pairs = set(self.tf_config.get("enabled_pairs", []) or [])
        trend_on = self.tf_config.get("enabled", True) or base in trend_pairs
        if trend_on:
            signal = self._trend_pullback(
                symbol, candles, funding_rate, existing_positions,
                higher_tf_candles, htf_dir,
            )
            if signal and self._passes_bias(signal, htf_dir):
                self._mark_trade(symbol, candle_time)
                return self._finalize(signal, candle_time)

        if self.bo_config.get("enabled", False):
            signal = self._momentum_breakout(
                symbol, candles, funding_rate, existing_positions,
                higher_tf_candles, htf_dir,
            )
            if signal and self._passes_bias(signal, htf_dir):
                self._mark_trade(symbol, candle_time)
                return self._finalize(signal, candle_time)

        if self.mr_config.get("enabled", True):
            signal = self._mean_reversion(
                symbol, candles, funding_rate, existing_positions,
                higher_tf_candles, htf_dir,
            )
            if signal and self._passes_bias(signal, htf_dir):
                self._mark_trade(symbol, candle_time)
                return self._finalize(signal, candle_time)

        if self._needs_keepalive(candle_time, symbol):
            signal = self._keepalive_vwap(symbol, candles, existing_positions, htf_dir)
            if signal and self._passes_bias(signal, htf_dir):
                self._mark_trade(symbol, candle_time)
                self._ka_timestamps.append(candle_time)
                return self._finalize(signal, candle_time)

        return None

    def _mark_trade(self, symbol: str, candle_time):
        self.last_trade_time[symbol] = candle_time
        self.last_any_trade_time = candle_time

    def _finalize(self, signal: Signal, candle_time) -> Signal:
        # Adaptive strength
        score = self._strategy_scores.get(signal.strategy, 1.0)
        signal.strength = min(1.0, max(0.05, signal.strength * score))

        # Session quality
        sess = self._session_mult(candle_time)
        signal.strength = min(1.0, signal.strength * sess)

        # Partial TP at ~1R
        if self._use_partial_tp and signal.stop_loss > 0:
            risk = abs(signal.entry_price - signal.stop_loss)
            if signal.side == Side.LONG:
                ptp = signal.entry_price + risk * 1.0
                # Don't set partial beyond full TP
                if ptp < signal.take_profit:
                    signal.partial_take_profit = ptp
                    signal.partial_fraction = 0.5
            else:
                ptp = signal.entry_price - risk * 1.0
                if ptp > signal.take_profit:
                    signal.partial_take_profit = ptp
                    signal.partial_fraction = 0.5
        return signal

    def _session_mult(self, t) -> float:
        if t is None:
            return 1.0
        if t.tzinfo is None:
            t = t.replace(tzinfo=timezone.utc)
        h = t.hour
        if 13 <= h <= 16:
            return 1.12
        if 8 <= h <= 21:
            return 1.0
        return 0.85  # Asia — smaller

    def _htf_direction(self, higher_tf_candles) -> str:
        if not higher_tf_candles or len(higher_tf_candles) < 55:
            return "neutral"
        h = np.array([c.close for c in higher_tf_candles], dtype=float)
        e9, e21, e50 = calculate_ema(h, 9), calculate_ema(h, 21), calculate_ema(h, 50)
        if e9[-1] > e21[-1] > e50[-1]:
            return "long"
        if e9[-1] < e21[-1] < e50[-1]:
            return "short"
        if e9[-1] > e21[-1]:
            return "long"
        if e9[-1] < e21[-1]:
            return "short"
        return "neutral"

    def _passes_bias(self, signal: Signal, htf_dir: str) -> bool:
        if self._long_only and signal.side == Side.SHORT:
            return False
        if not self._htf_bias or htf_dir == "neutral":
            return True
        if htf_dir == "long" and signal.side == Side.SHORT:
            return False
        if htf_dir == "short" and signal.side == Side.LONG:
            return False
        return True

    def _needs_keepalive(self, candle_time, symbol):
        """Only force activity if NOTHING traded recently across the book."""
        if not self.ka_config.get("enabled", True):
            return False
        base = symbol.split("/")[0]
        if base != self.keepalive_symbol:
            return False
        # Weekly cap — keep-alive is a tax, not an edge
        week_ago = candle_time - timedelta(days=7)
        recent = [t for t in self._ka_timestamps if t >= week_ago]
        self._ka_timestamps = recent
        if len(recent) >= self._ka_max_per_week:
            return False
        last = self.last_any_trade_time
        if last is None:
            last = self.last_trade_time.get(symbol)
        if last is None:
            return True
        return (candle_time - last) >= self.mandatory_interval

    def _blocked(self, symbol, existing_positions, want_side: str) -> bool:
        if not existing_positions:
            return False
        for ps, pp in existing_positions:
            if ps == symbol:
                return True
            if self._is_correlated(symbol, ps) and pp == want_side:
                return True
        return False

    # ---- Trend pullback ----

    def _trend_pullback(
        self, symbol, candles, funding_rate, existing_positions,
        higher_tf_candles, htf_dir,
    ):
        base = symbol.split("/")[0]
        profile = ATR_PROFILES.get(base, DEFAULT_PROFILE)
        closes = np.array([c.close for c in candles], dtype=float)
        highs = np.array([c.high for c in candles], dtype=float)
        lows = np.array([c.low for c in candles], dtype=float)
        volumes = np.array([c.volume for c in candles], dtype=float)
        candle_time = candles[-1].timestamp

        if self._skip_asia_trend and self._is_asia_session(candle_time):
            return None
        if self.circuit_breaker_until and candle_time < self.circuit_breaker_until:
            return None
        self.circuit_breaker_until = None

        adx = calculate_adx(highs, lows, closes, 14)
        adx_thresh = self.tf_config.get("adx_threshold", 25)
        if adx[-1] < adx_thresh:
            return None

        ema_9 = calculate_ema(closes, 9)
        ema_21 = calculate_ema(closes, 21)
        ema_50 = calculate_ema(closes, 50)
        macd_line, macd_sig, macd_hist = calculate_macd(closes, 8, 21, 5)

        bullish = ema_9[-1] > ema_21[-1] > ema_50[-1]
        bearish = ema_9[-1] < ema_21[-1] < ema_50[-1]
        if not bullish and not bearish:
            return None

        edges = self.edges.analyze_all_edges(candles, funding_rate, higher_tf_candles)
        mult, direction = self.edges.get_combined_modifier(edges)
        want = "long" if bullish else "short"
        if self._count_directional_edges(edges, want) < self._min_edges:
            return None

        crowded = self.edges.funding_crowded
        if bullish and funding_rate > crowded:
            return None
        if bearish and funding_rate < -crowded:
            return None
        if self._blocked(symbol, existing_positions, want):
            return None

        atr = calculate_atr(highs, lows, closes, 14)
        atr_v = float(atr[-1])
        if atr_v <= 0:
            return None
        price = float(closes[-1])
        ema_val = float(ema_21[-1])
        rsi = calculate_rsi(closes, 10)
        stoch_k, stoch_d = calculate_stochastic_rsi(closes)
        lev = self.config.get("trading", {}).get("default_leverage", 5)
        max_ext = float(self.tf_config.get("max_extension_atr", 0.55))
        dist = (price - ema_val) / atr_v
        avg_vol = float(np.mean(volumes[-20:]))
        vol_ok = volumes[-1] >= avg_vol * 0.75

        if bullish:
            near = -0.35 <= dist <= max_ext
            reclaim = closes[-1] > closes[-2] and price >= ema_val * 0.997
            macd_ok = macd_hist[-1] >= macd_hist[-2]
            stoch_ok = stoch_k[-1] > stoch_d[-1] or stoch_k[-1] > stoch_k[-2]
            rsi_ok = 32 < rsi[-1] < 58
            if not (near and reclaim and macd_ok and stoch_ok and rsi_ok and vol_ok):
                return None
            stop = min(price - atr_v * profile["stop"], ema_val - atr_v * 0.5)
            target = price + atr_v * profile["target"]
            strength = 0.7 + (0.1 if adx[-1] > 30 else 0) + (0.08 if htf_dir == "long" else 0)
            signal = Signal(
                symbol=symbol, side=Side.LONG, strength=min(1.0, strength * mult),
                strategy="trend_pullback", entry_price=price,
                stop_loss=stop, take_profit=target, leverage=lev,
                reason=f"Pullback LONG ADX={adx[-1]:.0f} dist={dist:.2f}",
            )
        else:
            near = -max_ext <= dist <= 0.35
            reclaim = closes[-1] < closes[-2] and price <= ema_val * 1.003
            macd_ok = macd_hist[-1] <= macd_hist[-2]
            stoch_ok = stoch_k[-1] < stoch_d[-1] or stoch_k[-1] < stoch_k[-2]
            rsi_ok = 42 < rsi[-1] < 68
            if not (near and reclaim and macd_ok and stoch_ok and rsi_ok and vol_ok):
                return None
            stop = max(price + atr_v * profile["stop"], ema_val + atr_v * 0.5)
            target = price - atr_v * profile["target"]
            strength = 0.7 + (0.1 if adx[-1] > 30 else 0) + (0.08 if htf_dir == "short" else 0)
            signal = Signal(
                symbol=symbol, side=Side.SHORT, strength=min(1.0, strength * mult),
                strategy="trend_pullback", entry_price=price,
                stop_loss=stop, take_profit=target, leverage=lev,
                reason=f"Pullback SHORT ADX={adx[-1]:.0f} dist={dist:.2f}",
            )

        if signal.risk_reward_ratio < self._min_rr:
            return None
        if direction == "long" and signal.side == Side.SHORT and mult > 1.3:
            return None
        if direction == "short" and signal.side == Side.LONG and mult > 1.3:
            return None
        return signal if signal.strength >= 0.5 else None

    # ---- Breakout (off by default) ----

    def _momentum_breakout(
        self, symbol, candles, funding_rate, existing_positions,
        higher_tf_candles, htf_dir,
    ):
        base = symbol.split("/")[0]
        profile = ATR_PROFILES.get(base, DEFAULT_PROFILE)
        closes = np.array([c.close for c in candles], dtype=float)
        highs = np.array([c.high for c in candles], dtype=float)
        lows = np.array([c.low for c in candles], dtype=float)
        volumes = np.array([c.volume for c in candles], dtype=float)
        candle_time = candles[-1].timestamp

        if self._skip_asia_trend and self._is_asia_session(candle_time):
            return None
        if self.circuit_breaker_until and candle_time < self.circuit_breaker_until:
            return None

        look = int(self.bo_config.get("lookback", 20))
        if len(closes) < look + 5:
            return None

        atr = calculate_atr(highs, lows, closes, 14)
        atr_v = float(atr[-1])
        price = float(closes[-1])
        if atr_v <= 0:
            return None

        adx = calculate_adx(highs, lows, closes, 14)
        if adx[-1] < self.bo_config.get("adx_min", 22):
            return None

        avg_vol = float(np.mean(volumes[-look:]))
        vol_ok = volumes[-1] >= avg_vol * self.bo_config.get("volume_mult", 1.5)
        prior_high = float(np.max(highs[-look - 1 : -1]))
        prior_low = float(np.min(lows[-look - 1 : -1]))
        ema_21 = calculate_ema(closes, 21)
        ema_50 = calculate_ema(closes, 50)
        lev = self.config.get("trading", {}).get("default_leverage", 5)
        mult, _ = self.edges.get_combined_modifier(
            self.edges.analyze_all_edges(candles, funding_rate, higher_tf_candles)
        )
        crowded = self.edges.funding_crowded

        if (
            price > prior_high
            and closes[-1] > opens_safe(candles[-1])
            and ema_21[-1] > ema_50[-1]
            and vol_ok
            and funding_rate <= crowded
            and not self._blocked(symbol, existing_positions, "long")
        ):
            return Signal(
                symbol=symbol, side=Side.LONG, strength=min(1.0, 0.65 * mult),
                strategy="momentum_breakout", entry_price=price,
                stop_loss=price - atr_v * profile["stop"],
                take_profit=price + atr_v * profile["target"],
                leverage=lev, reason=f"Breakout LONG ADX={adx[-1]:.0f}",
            )
        if (
            price < prior_low
            and closes[-1] < opens_safe(candles[-1])
            and ema_21[-1] < ema_50[-1]
            and vol_ok
            and funding_rate >= -crowded
            and not self._blocked(symbol, existing_positions, "short")
        ):
            return Signal(
                symbol=symbol, side=Side.SHORT, strength=min(1.0, 0.65 * mult),
                strategy="momentum_breakout", entry_price=price,
                stop_loss=price + atr_v * profile["stop"],
                take_profit=price - atr_v * profile["target"],
                leverage=lev, reason=f"Breakout SHORT ADX={adx[-1]:.0f}",
            )
        return None

    # ---- Mean reversion (primary edge) ----

    def _mean_reversion(
        self, symbol, candles, funding_rate, existing_positions,
        higher_tf_candles, htf_dir,
    ):
        closes = np.array([c.close for c in candles], dtype=float)
        highs = np.array([c.high for c in candles], dtype=float)
        lows = np.array([c.low for c in candles], dtype=float)
        volumes = np.array([c.volume for c in candles], dtype=float)
        candle_time = candles[-1].timestamp

        if self._is_asia_session(candle_time) and not self._allow_asia_mr:
            return None
        if self.circuit_breaker_until and candle_time < self.circuit_breaker_until:
            return None

        adx = calculate_adx(highs, lows, closes, 14)
        max_adx = float(self.mr_config.get("max_adx", 30))
        if adx[-1] > max_adx:
            return None

        if existing_positions:
            for ps, _ in existing_positions:
                if ps == symbol:
                    return None

        base = symbol.split("/")[0]
        if base.upper() in self._disabled_pairs:
            return None

        # ETH: stricter — require stronger extreme
        rsi_period = self.mr_config.get("rsi_period", 10)
        ob = self.mr_config.get("rsi_overbought", 68)
        os_ = self.mr_config.get("rsi_oversold", 32)
        if base == "ETH":
            ob = min(ob, 65)
            os_ = max(os_, 35)
            if adx[-1] > max_adx - 5:
                return None

        bb_period = self.mr_config.get("bb_period", 20)
        bb_std = self.mr_config.get("bb_std", 2.0)

        rsi = calculate_rsi(closes, rsi_period)
        stoch_k, stoch_d = calculate_stochastic_rsi(closes)
        bb_u, bb_m, bb_l = calculate_bollinger_bands(closes, bb_period, bb_std)
        atr = calculate_atr(highs, lows, closes, 14)
        price = float(closes[-1])
        atr_v = float(atr[-1])
        if atr_v <= 0 or np.isnan(bb_l[-1]) or np.isnan(bb_m[-1]):
            return None

        # Z-score stretch: how many std from mid
        band_half = float(bb_u[-1] - bb_m[-1])
        if band_half <= 0:
            return None
        z = (price - float(bb_m[-1])) / band_half  # ~±1 at bands
        min_stretch = float(self.mr_config.get("min_z", 0.92))

        avg_vol = float(np.mean(volumes[-20:]))
        vol_ratio = float(volumes[-1] / avg_vol) if avg_vol > 0 else 1.0
        if vol_ratio < float(self.mr_config.get("min_volume_ratio", 0.7)):
            return None

        # Wick rejection quality (hammer / shooting star style)
        last = candles[-1]
        rng = max(last.high - last.low, 1e-12)
        body = abs(last.close - last.open)
        lower_wick = min(last.open, last.close) - last.low
        upper_wick = last.high - max(last.open, last.close)
        close_loc = (last.close - last.low) / rng  # 0=low, 1=high

        profile = ATR_PROFILES.get(base, DEFAULT_PROFILE)
        lev = self.config.get("trading", {}).get("default_leverage", 5)
        strength = float(self.mr_config.get("strength", 0.72))
        pair_mult = {"BTC": 1.2, "ETH": 0.45, "SOL": 0.75}.get(base, 0.8)
        strength *= pair_mult
        strength *= min(1.18, 0.88 + abs(z) * 0.18)
        # Favor EU/US already applied in finalize; bonus for clean rejection
        runner_ext = float(self.mr_config.get("runner_extension_atr", 0.35))

        # Wick quality is a BONUS (not hard gate) — hard gate hurt 120d OOS
        long_reject = lower_wick >= body * 0.5 and close_loc >= 0.5
        short_reject = upper_wick >= body * 0.5 and close_loc <= 0.5
        # Extreme stretch can enter without perfect wick
        extreme = abs(z) >= min_stretch + 0.15

        # LONG: stretch down + reclaim
        if (
            z <= -min_stretch
            and price <= float(bb_l[-1]) * 1.006
            and rsi[-1] <= os_
            and rsi[-1] > rsi[-2]
            and closes[-1] > closes[-2]
            and (stoch_k[-1] > stoch_d[-1] or stoch_k[-1] > stoch_k[-2])
            and (long_reject or extreme)
        ):
            stop = min(float(bb_l[-1]) - atr_v * 0.35, price - atr_v * profile["stop"])
            target = float(bb_m[-1]) + atr_v * runner_ext
            if target <= price:
                target = price + atr_v * 1.8
            target = min(target, price + atr_v * (profile["target"] + 0.3))
            if abs(target - price) / max(abs(price - stop), 1e-9) < self._min_rr:
                return None
            s = strength
            if long_reject:
                s = min(1.0, s * 1.1)
            elif extreme:
                s = min(1.0, s * 0.95)  # size down if no wick confirmation
            return Signal(
                symbol=symbol, side=Side.LONG, strength=min(1.0, s),
                strategy="mean_reversion",
                entry_price=price, stop_loss=stop, take_profit=target,
                leverage=lev,
                reason=f"MR LONG z={z:.2f} RSI={rsi[-1]:.0f} wick={long_reject} vol={vol_ratio:.1f}x",
            )

        # SHORT
        if (
            z >= min_stretch
            and price >= float(bb_u[-1]) * 0.994
            and rsi[-1] >= ob
            and rsi[-1] < rsi[-2]
            and closes[-1] < closes[-2]
            and (stoch_k[-1] < stoch_d[-1] or stoch_k[-1] < stoch_k[-2])
            and (short_reject or extreme)
        ):
            stop = max(float(bb_u[-1]) + atr_v * 0.35, price + atr_v * profile["stop"])
            target = float(bb_m[-1]) - atr_v * runner_ext
            if target >= price:
                target = price - atr_v * 1.8
            target = max(target, price - atr_v * (profile["target"] + 0.3))
            if abs(price - target) / max(abs(stop - price), 1e-9) < self._min_rr:
                return None
            s = strength
            if short_reject:
                s = min(1.0, s * 1.1)
            elif extreme:
                s = min(1.0, s * 0.95)
            return Signal(
                symbol=symbol, side=Side.SHORT, strength=min(1.0, s),
                strategy="mean_reversion",
                entry_price=price, stop_loss=stop, take_profit=target,
                leverage=lev,
                reason=f"MR SHORT z={z:.2f} RSI={rsi[-1]:.0f} wick={short_reject} vol={vol_ratio:.1f}x",
            )

        return None

    # ---- Keep-alive ----

    def _keepalive_vwap(self, symbol, candles, existing_positions, htf_dir):
        closes = np.array([c.close for c in candles], dtype=float)
        highs = np.array([c.high for c in candles], dtype=float)
        lows = np.array([c.low for c in candles], dtype=float)
        volumes = np.array([c.volume for c in candles], dtype=float)

        if self.circuit_breaker_until and candles[-1].timestamp < self.circuit_breaker_until:
            return None
        if existing_positions:
            for ps, _ in existing_positions:
                if ps == symbol:
                    return None

        adx = calculate_adx(highs, lows, closes, 14)
        if adx[-1] < 12:
            return None

        ema_9 = calculate_ema(closes, 9)
        ema_21 = calculate_ema(closes, 21)
        is_bullish = ema_9[-1] > ema_21[-1]
        is_bearish = ema_9[-1] < ema_21[-1]
        if htf_dir == "long":
            is_bearish = False
        elif htf_dir == "short":
            is_bullish = False
        if not is_bullish and not is_bearish:
            return None

        vwap = calculate_vwap(highs, lows, closes, volumes, 20)
        if np.isnan(vwap[-1]):
            return None

        atr = calculate_atr(highs, lows, closes, 14)
        atr_v = float(atr[-1])
        price = float(closes[-1])
        vwap_val = float(vwap[-1])
        if atr_v <= 0:
            return None

        rsi = calculate_rsi(closes, 10)
        lev = self.config.get("trading", {}).get("default_leverage", 5)
        ka_strength = float(self.ka_config.get("strength", 0.10))

        if is_bullish and 0 <= (price - vwap_val) / atr_v <= 0.85:
            if closes[-1] > closes[-2] and 35 < rsi[-1] < 58:
                return Signal(
                    symbol=symbol, side=Side.LONG, strength=ka_strength,
                    strategy="keepalive_vwap", entry_price=price,
                    stop_loss=price - atr_v * KEEPALIVE_STOP,
                    take_profit=price + atr_v * KEEPALIVE_TARGET,
                    leverage=lev, reason=f"KA LONG ADX={adx[-1]:.0f}",
                )
        if is_bearish and 0 <= (vwap_val - price) / atr_v <= 0.85:
            if closes[-1] < closes[-2] and 42 < rsi[-1] < 65:
                return Signal(
                    symbol=symbol, side=Side.SHORT, strength=ka_strength,
                    strategy="keepalive_vwap", entry_price=price,
                    stop_loss=price + atr_v * KEEPALIVE_STOP,
                    take_profit=price - atr_v * KEEPALIVE_TARGET,
                    leverage=lev, reason=f"KA SHORT ADX={adx[-1]:.0f}",
                )

        bb_u, _, bb_l = calculate_bollinger_bands(closes, 20, 1.8)
        bb_s = max(0.07, ka_strength * 0.7)
        if is_bullish and price <= float(bb_l[-1]) * 1.004 and rsi[-1] < 40:
            return Signal(
                symbol=symbol, side=Side.LONG, strength=bb_s,
                strategy="keepalive_bb", entry_price=price,
                stop_loss=price - atr_v * KEEPALIVE_STOP,
                take_profit=price + atr_v * KEEPALIVE_TARGET,
                leverage=lev, reason="KA BB LONG",
            )
        if is_bearish and price >= float(bb_u[-1]) * 0.996 and rsi[-1] > 60:
            return Signal(
                symbol=symbol, side=Side.SHORT, strength=bb_s,
                strategy="keepalive_bb", entry_price=price,
                stop_loss=price + atr_v * KEEPALIVE_STOP,
                take_profit=price - atr_v * KEEPALIVE_TARGET,
                leverage=lev, reason="KA BB SHORT",
            )
        return None

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
        return b1 in {"BTC", "ETH"} and b2 in {"BTC", "ETH"}

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
        if edges.get("session", {}).get("favorable"):
            c += 1
        return c


def opens_safe(candle: Candle) -> float:
    return candle.open if candle else 0.0
