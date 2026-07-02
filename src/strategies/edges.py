"""WEEX AI Wars II — Edge Strategies v2

Fixes:
1. session_time_filter now takes candle timestamp, not datetime.now()
2. All time-based logic uses candle time for backtest accuracy
"""

import numpy as np
from datetime import datetime, timezone
from ..core.models import Candle, Signal, Side, MarketRegime
from ..indicators.technical import (
    calculate_rsi, calculate_macd, calculate_bollinger_bands,
    calculate_atr, calculate_adx, calculate_ema, calculate_vwap,
    detect_regime, calculate_stochastic_rsi,
)


class EdgeStrategies:
    """
    Edge strategies v2 — all time-based logic uses candle timestamp.
    """

    def __init__(self, config: dict):
        self.config = config

    def analyze_all_edges(
        self,
        candles: list[Candle],
        funding_rate: float = 0.0,
        open_interest_change: float = 0.0,
    ) -> dict:
        """Run all edge detectors. Uses LAST candle's timestamp for time filter."""
        closes = np.array([c.close for c in candles])
        highs = np.array([c.high for c in candles])
        lows = np.array([c.low for c in candles])
        volumes = np.array([c.volume for c in candles])

        # Use the last candle's timestamp for time-based filters
        candle_time = candles[-1].timestamp if candles else datetime.now(timezone.utc)

        edges = {}
        edges["liquidation"] = self.detect_liquidation_cascade(closes, highs, lows, volumes)
        edges["funding"] = self.funding_rate_signal(funding_rate)
        edges["session"] = self.session_time_filter(candle_time)  # FIXED: pass candle time
        edges["volume"] = self.volume_anomaly_detector(closes, volumes)
        edges["mtf"] = self.multi_timeframe_confluence(closes, highs, lows, volumes)

        return edges

    def get_combined_modifier(self, edges: dict) -> tuple[float, str]:
        """Combine all edge signals into a single strength multiplier and bias."""
        multiplier = 1.0
        long_score = 0
        short_score = 0

        liq = edges.get("liquidation", {})
        if liq.get("detected"):
            if liq.get("direction") == "long":
                long_score += 2; multiplier *= 1.3
            else:
                short_score += 2; multiplier *= 1.3

        fund = edges.get("funding", {})
        if fund.get("signal"):
            if fund.get("direction") == "long":
                long_score += 1; multiplier *= 1.15
            else:
                short_score += 1; multiplier *= 1.15

        session = edges.get("session", {})
        if session.get("favorable"):
            multiplier *= 1.1

        vol = edges.get("volume", {})
        if vol.get("anomaly"):
            if vol.get("direction") == "accumulation":
                long_score += 1; multiplier *= 1.2
            elif vol.get("direction") == "distribution":
                short_score += 1; multiplier *= 1.2

        mtf = edges.get("mtf", {})
        if mtf.get("aligned"):
            if mtf.get("direction") == "long":
                long_score += 2; multiplier *= 1.25
            else:
                short_score += 2; multiplier *= 1.25

        if long_score > short_score + 1:
            direction = "long"
        elif short_score > long_score + 1:
            direction = "short"
        else:
            direction = "neutral"

        multiplier = max(0.5, min(2.0, multiplier))
        return multiplier, direction

    # ---- Edge 1: Liquidation Cascade Detector ----

    def detect_liquidation_cascade(self, closes, highs, lows, volumes) -> dict:
        if len(closes) < 20:
            return {"detected": False}

        price_change_3 = (closes[-1] - closes[-4]) / closes[-4]
        avg_volume = np.mean(volumes[-20:])
        current_volume = volumes[-1]
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1

        is_cascade = abs(price_change_3) > 0.015 and volume_ratio > 2.0

        if is_cascade:
            direction = "long" if price_change_3 > 0 else "short"
            return {
                "detected": True,
                "direction": direction,
                "magnitude": abs(price_change_3),
                "volume_ratio": volume_ratio,
                "confidence": min(1.0, abs(price_change_3) / 0.03 * volume_ratio / 3),
            }
        return {"detected": False}

    # ---- Edge 2: Funding Rate Extreme ----

    def funding_rate_signal(self, funding_rate: float) -> dict:
        extreme_threshold = 0.00005

        if funding_rate > extreme_threshold:
            return {
                "signal": True, "direction": "short",
                "strength": min(1.0, funding_rate / 0.003),
                "reason": f"Funding {funding_rate:.4%} positive — longs crowded",
            }
        elif funding_rate < -extreme_threshold:
            return {
                "signal": True, "direction": "long",
                "strength": min(1.0, abs(funding_rate) / 0.003),
                "reason": f"Funding {funding_rate:.4%} negative — shorts crowded",
            }
        return {"signal": False}

    # ---- Edge 3: Session Time Filter (FIXED) ----

    def session_time_filter(self, candle_time: datetime = None) -> dict:
        """
        Trade during high-liquidity sessions.
        FIXED: Uses candle timestamp, not datetime.now().
        """
        if candle_time is None:
            candle_time = datetime.now(timezone.utc)

        # Ensure UTC
        if candle_time.tzinfo is None:
            candle_time = candle_time.replace(tzinfo=timezone.utc)

        hour = candle_time.hour

        if 13 <= hour <= 16:
            return {
                "favorable": True, "session": "US-EU Overlap",
                "multiplier": 1.2,
                "reason": "Peak liquidity window",
            }
        elif 13 <= hour <= 21:
            return {
                "favorable": True, "session": "US",
                "multiplier": 1.1,
                "reason": "US session — high volatility",
            }
        elif 8 <= hour <= 16:
            return {
                "favorable": True, "session": "Europe",
                "multiplier": 1.0,
                "reason": "European session — moderate",
            }
        else:
            return {
                "favorable": False, "session": "Asia",
                "multiplier": 0.9,
                "reason": "Asian session — lower volatility",
            }

    # ---- Edge 4: Volume Anomaly Detector ----

    def volume_anomaly_detector(self, closes, volumes) -> dict:
        if len(closes) < 20:
            return {"anomaly": False}

        avg_vol = np.mean(volumes[-20:])
        current_vol = volumes[-1]
        vol_ratio = current_vol / avg_vol if avg_vol > 0 else 1
        price_change = abs(closes[-1] - closes[-2]) / closes[-2]

        if vol_ratio > 2.5 and price_change < 0.005:
            direction = "accumulation" if closes[-1] > closes[-2] else "distribution"
            return {
                "anomaly": True, "direction": direction,
                "volume_ratio": vol_ratio, "price_change": price_change,
                "reason": f"Smart money {direction}: {vol_ratio:.1f}x volume, {price_change:.2%} move",
            }

        if len(closes) >= 10:
            price_higher = closes[-1] > closes[-5]
            vol_lower = np.mean(volumes[-3:]) < np.mean(volumes[-8:-3])
            if price_higher and vol_lower:
                return {
                    "anomaly": True, "direction": "distribution",
                    "reason": "Bearish volume divergence",
                }

        return {"anomaly": False}

    # ---- Edge 5: Multi-Timeframe Confluence ----

    def multi_timeframe_confluence(self, closes, highs, lows, volumes) -> dict:
        if len(closes) < 100:
            return {"aligned": False}

        ema_9 = calculate_ema(closes, 9)
        ema_21 = calculate_ema(closes, 21)
        ema_50 = calculate_ema(closes, 50)
        ema_100 = calculate_ema(closes, 100)

        short_bullish = ema_9[-1] > ema_21[-1]
        medium_bullish = ema_21[-1] > ema_50[-1]
        long_bullish = ema_50[-1] > ema_100[-1]

        all_bullish = short_bullish and medium_bullish and long_bullish
        all_bearish = (not short_bullish) and (not medium_bullish) and (not long_bullish)

        if all_bullish:
            return {
                "aligned": True, "direction": "long",
                "timeframes_aligned": 3,
                "reason": "All timeframes bullish: 9>21>50>100 EMA",
            }
        elif all_bearish:
            return {
                "aligned": True, "direction": "short",
                "timeframes_aligned": 3,
                "reason": "All timeframes bearish: 9<21<50<100 EMA",
            }

        bullish_count = sum([short_bullish, medium_bullish, long_bullish])
        if bullish_count >= 2:
            return {
                "aligned": False, "direction": "long",
                "timeframes_aligned": bullish_count,
                "reason": f"Partial bullish ({bullish_count}/3)",
            }
        elif bullish_count <= 1:
            return {
                "aligned": False, "direction": "short",
                "timeframes_aligned": 3 - bullish_count,
                "reason": f"Partial bearish ({3-bullish_count}/3)",
            }

        return {"aligned": False, "direction": "neutral", "timeframes_aligned": 0}
