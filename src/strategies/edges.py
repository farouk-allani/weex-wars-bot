"""WEEX AI Wars II — Edge Strategies v8

Improvements:
1. Realistic funding extremes (configurable, default ~0.03%)
2. Real multi-timeframe when higher-TF candles are provided
3. Session filter uses candle timestamps
4. Cleaner edge scoring for strategy confluence
"""

import numpy as np
from datetime import datetime, timezone
from ..core.models import Candle
from ..indicators.technical import calculate_ema, calculate_adx


class EdgeStrategies:
    def __init__(self, config: dict):
        self.config = config
        edge_cfg = config.get("edges", {})
        # ~0.03% is a real crowded-funding zone for many perps
        self.funding_extreme = edge_cfg.get("funding_extreme", 0.0003)
        self.funding_crowded = edge_cfg.get("funding_crowded", 0.0005)

    def analyze_all_edges(
        self,
        candles: list[Candle],
        funding_rate: float = 0.0,
        higher_tf_candles: list[Candle] | None = None,
    ) -> dict:
        closes = np.array([c.close for c in candles])
        highs = np.array([c.high for c in candles])
        lows = np.array([c.low for c in candles])
        volumes = np.array([c.volume for c in candles])
        candle_time = candles[-1].timestamp if candles else datetime.now(timezone.utc)

        edges = {
            "liquidation": self.detect_liquidation_cascade(closes, highs, lows, volumes),
            "funding": self.funding_rate_signal(funding_rate),
            "session": self.session_time_filter(candle_time),
            "volume": self.volume_anomaly_detector(closes, volumes),
            "mtf": self.multi_timeframe_confluence(
                closes, highs, lows, volumes, higher_tf_candles
            ),
            "momentum": self.momentum_quality(closes, highs, lows),
        }
        return edges

    def get_combined_modifier(self, edges: dict) -> tuple[float, str]:
        multiplier = 1.0
        long_score = 0
        short_score = 0

        liq = edges.get("liquidation", {})
        if liq.get("detected"):
            if liq.get("direction") == "long":
                long_score += 2
                multiplier *= 1.25
            else:
                short_score += 2
                multiplier *= 1.25

        fund = edges.get("funding", {})
        if fund.get("signal"):
            if fund.get("direction") == "long":
                long_score += 1
                multiplier *= 1.12
            else:
                short_score += 1
                multiplier *= 1.12

        session = edges.get("session", {})
        if session.get("favorable"):
            multiplier *= session.get("multiplier", 1.1)

        vol = edges.get("volume", {})
        if vol.get("anomaly"):
            if vol.get("direction") == "accumulation":
                long_score += 1
                multiplier *= 1.15
            elif vol.get("direction") == "distribution":
                short_score += 1
                multiplier *= 1.15

        mtf = edges.get("mtf", {})
        if mtf.get("aligned"):
            if mtf.get("direction") == "long":
                long_score += 2
                multiplier *= 1.2
            else:
                short_score += 2
                multiplier *= 1.2

        mom = edges.get("momentum", {})
        if mom.get("quality"):
            if mom.get("direction") == "long":
                long_score += 1
                multiplier *= 1.08
            elif mom.get("direction") == "short":
                short_score += 1
                multiplier *= 1.08

        if long_score > short_score + 1:
            direction = "long"
        elif short_score > long_score + 1:
            direction = "short"
        else:
            direction = "neutral"

        return max(0.5, min(2.0, multiplier)), direction

    # ---- Edge 1: Volatile cascade proxy ----

    def detect_liquidation_cascade(self, closes, highs, lows, volumes) -> dict:
        if len(closes) < 20:
            return {"detected": False}

        price_change_3 = (closes[-1] - closes[-4]) / closes[-4]
        avg_volume = np.mean(volumes[-20:])
        volume_ratio = volumes[-1] / avg_volume if avg_volume > 0 else 1

        # Strong directional move + volume spike
        is_cascade = abs(price_change_3) > 0.018 and volume_ratio > 2.2
        if not is_cascade:
            return {"detected": False}

        # Fade exhausted moves: after cascade, bias opposite of wick extreme
        # If sharp drop, bounce long; sharp pump → short mean-revert bias
        direction = "long" if price_change_3 < 0 else "short"
        return {
            "detected": True,
            "direction": direction,
            "magnitude": abs(price_change_3),
            "volume_ratio": float(volume_ratio),
            "confidence": min(1.0, abs(price_change_3) / 0.03 * volume_ratio / 3),
        }

    # ---- Edge 2: Funding ----

    def funding_rate_signal(self, funding_rate: float) -> dict:
        if funding_rate > self.funding_extreme:
            return {
                "signal": True,
                "direction": "short",
                "strength": min(1.0, funding_rate / 0.003),
                "reason": f"Funding {funding_rate:.4%} crowded long",
            }
        if funding_rate < -self.funding_extreme:
            return {
                "signal": True,
                "direction": "long",
                "strength": min(1.0, abs(funding_rate) / 0.003),
                "reason": f"Funding {funding_rate:.4%} crowded short",
            }
        return {"signal": False}

    # ---- Edge 3: Session ----

    def session_time_filter(self, candle_time: datetime = None) -> dict:
        if candle_time is None:
            candle_time = datetime.now(timezone.utc)
        if candle_time.tzinfo is None:
            candle_time = candle_time.replace(tzinfo=timezone.utc)

        hour = candle_time.hour
        if 13 <= hour <= 16:
            return {
                "favorable": True, "session": "US-EU Overlap",
                "multiplier": 1.15, "reason": "Peak liquidity",
            }
        if 13 <= hour <= 21:
            return {
                "favorable": True, "session": "US",
                "multiplier": 1.08, "reason": "US session",
            }
        if 8 <= hour <= 16:
            return {
                "favorable": True, "session": "Europe",
                "multiplier": 1.0, "reason": "European session",
            }
        return {
            "favorable": False, "session": "Asia",
            "multiplier": 0.85, "reason": "Asian session — lower quality",
        }

    # ---- Edge 4: Volume ----

    def volume_anomaly_detector(self, closes, volumes) -> dict:
        if len(closes) < 20:
            return {"anomaly": False}

        avg_vol = np.mean(volumes[-20:])
        vol_ratio = volumes[-1] / avg_vol if avg_vol > 0 else 1
        price_change = abs(closes[-1] - closes[-2]) / closes[-2] if closes[-2] else 0

        # High volume, small range → absorption
        if vol_ratio > 2.3 and price_change < 0.004:
            direction = "accumulation" if closes[-1] > closes[-2] else "distribution"
            return {
                "anomaly": True, "direction": direction,
                "volume_ratio": float(vol_ratio),
                "reason": f"Absorption {direction}: {vol_ratio:.1f}x vol",
            }

        # Price up on declining volume → weak
        if len(closes) >= 10:
            price_higher = closes[-1] > closes[-5]
            vol_lower = np.mean(volumes[-3:]) < np.mean(volumes[-8:-3])
            if price_higher and vol_lower:
                return {
                    "anomaly": True, "direction": "distribution",
                    "reason": "Bearish volume divergence",
                }
            price_lower = closes[-1] < closes[-5]
            if price_lower and vol_lower:
                return {
                    "anomaly": True, "direction": "accumulation",
                    "reason": "Bullish volume divergence",
                }

        return {"anomaly": False}

    # ---- Edge 5: Real multi-timeframe + same-TF stack ----

    def multi_timeframe_confluence(
        self,
        closes,
        highs,
        lows,
        volumes,
        higher_tf_candles: list[Candle] | None = None,
    ) -> dict:
        if len(closes) < 100:
            return {"aligned": False, "direction": "neutral", "timeframes_aligned": 0}

        ema_9 = calculate_ema(closes, 9)
        ema_21 = calculate_ema(closes, 21)
        ema_50 = calculate_ema(closes, 50)
        ema_100 = calculate_ema(closes, 100)

        short_bull = ema_9[-1] > ema_21[-1]
        med_bull = ema_21[-1] > ema_50[-1]
        long_bull = ema_50[-1] > ema_100[-1]
        htf_bull = None

        # Real higher timeframe (e.g. 4h) when available
        if higher_tf_candles and len(higher_tf_candles) >= 50:
            h_closes = np.array([c.close for c in higher_tf_candles])
            h_ema9 = calculate_ema(h_closes, 9)
            h_ema21 = calculate_ema(h_closes, 21)
            h_ema50 = calculate_ema(h_closes, 50)
            htf_bull = h_ema9[-1] > h_ema21[-1] > h_ema50[-1]
            htf_bear = h_ema9[-1] < h_ema21[-1] < h_ema50[-1]
            if not htf_bull and not htf_bear:
                htf_bull = h_ema9[-1] > h_ema21[-1]

        votes_long = sum([short_bull, med_bull, long_bull])
        if htf_bull is True:
            votes_long += 2
        elif htf_bull is False:
            votes_long -= 2

        if votes_long >= 4 or (votes_long >= 3 and htf_bull is True):
            return {
                "aligned": True, "direction": "long",
                "timeframes_aligned": votes_long,
                "reason": "HTF+LTF bullish alignment",
                "htf": htf_bull,
            }
        if votes_long <= 0 or (votes_long <= 1 and htf_bull is False):
            return {
                "aligned": True, "direction": "short",
                "timeframes_aligned": 3 - votes_long,
                "reason": "HTF+LTF bearish alignment",
                "htf": htf_bull,
            }

        direction = "long" if votes_long >= 2 else "short"
        return {
            "aligned": False,
            "direction": direction,
            "timeframes_aligned": max(0, votes_long),
            "reason": f"Partial alignment ({votes_long})",
            "htf": htf_bull,
        }

    # ---- Edge 6: Momentum quality ----

    def momentum_quality(self, closes, highs, lows) -> dict:
        if len(closes) < 30:
            return {"quality": False}

        ema_21 = calculate_ema(closes, 21)
        slope = (ema_21[-1] - ema_21[-6]) / ema_21[-6] if ema_21[-6] else 0
        adx = calculate_adx(highs, lows, closes, 14)
        strong = adx[-1] >= 22 and abs(slope) > 0.002

        if not strong:
            return {"quality": False}

        direction = "long" if slope > 0 else "short"
        return {
            "quality": True,
            "direction": direction,
            "adx": float(adx[-1]),
            "slope": float(slope),
        }
