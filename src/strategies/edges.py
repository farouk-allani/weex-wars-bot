"""WEEX AI Wars II — Secret Sauce: Edge Strategies

These are the unfair advantages:
1. Liquidation Cascade Detector — front-run forced liquidations
2. Funding Rate Extremes — trade against crowded positions
3. Session Time Filter — trade with institutional flow
4. Volume Anomaly Detector — detect smart money accumulation
5. Multi-Timeframe Confluence — only trade when all timeframes agree
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
    Collection of edge strategies that provide alpha over retail bots.
    Each returns a signal modifier (strength multiplier + direction bias).
    """

    def __init__(self, config: dict):
        self.config = config

    def analyze_all_edges(
        self,
        candles: list[Candle],
        funding_rate: float = 0.0,
        open_interest_change: float = 0.0,
    ) -> dict:
        """Run all edge detectors and return combined signal modifiers."""
        closes = np.array([c.close for c in candles])
        highs = np.array([c.high for c in candles])
        lows = np.array([c.low for c in candles])
        volumes = np.array([c.volume for c in candles])

        edges = {}

        # 1. Liquidation cascade detection
        edges["liquidation"] = self.detect_liquidation_cascade(
            closes, highs, lows, volumes
        )

        # 2. Funding rate extreme
        edges["funding"] = self.funding_rate_signal(funding_rate)

        # 3. Session time filter
        edges["session"] = self.session_time_filter()

        # 4. Volume anomaly
        edges["volume"] = self.volume_anomaly_detector(closes, volumes)

        # 5. Multi-timeframe confluence
        edges["mtf"] = self.multi_timeframe_confluence(
            closes, highs, lows, volumes
        )

        return edges

    def get_combined_modifier(self, edges: dict) -> tuple[float, str]:
        """
        Combine all edge signals into a single strength multiplier and bias.
        Returns (multiplier, direction_hint)
        multiplier: 0.5 to 2.0 (scales signal strength)
        direction_hint: 'long', 'short', or 'neutral'
        """
        multiplier = 1.0
        long_score = 0
        short_score = 0

        # Liquidation cascade (high impact)
        liq = edges.get("liquidation", {})
        if liq.get("detected"):
            if liq.get("direction") == "long":
                long_score += 2
                multiplier *= 1.3
            else:
                short_score += 2
                multiplier *= 1.3

        # Funding rate (medium impact)
        fund = edges.get("funding", {})
        if fund.get("signal"):
            if fund.get("direction") == "long":
                long_score += 1
                multiplier *= 1.15
            else:
                short_score += 1
                multiplier *= 1.15

        # Session time (low impact, consistency)
        session = edges.get("session", {})
        if session.get("favorable"):
            multiplier *= 1.1

        # Volume anomaly (medium impact)
        vol = edges.get("volume", {})
        if vol.get("anomaly"):
            if vol.get("direction") == "accumulation":
                long_score += 1
                multiplier *= 1.2
            elif vol.get("direction") == "distribution":
                short_score += 1
                multiplier *= 1.2

        # Multi-timeframe (high impact)
        mtf = edges.get("mtf", {})
        if mtf.get("aligned"):
            if mtf.get("direction") == "long":
                long_score += 2
                multiplier *= 1.25
            else:
                short_score += 2
                multiplier *= 1.25

        # Determine direction hint
        if long_score > short_score + 1:
            direction = "long"
        elif short_score > long_score + 1:
            direction = "short"
        else:
            direction = "neutral"

        # Clamp multiplier
        multiplier = max(0.5, min(2.0, multiplier))

        return multiplier, direction

    # ---- Edge 1: Liquidation Cascade Detector ----

    def detect_liquidation_cascade(
        self,
        closes: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        volumes: np.ndarray,
    ) -> dict:
        """
        Detect when a price move will trigger cascading liquidations.

        Logic:
        - Sharp price move (>1.5% in 3 candles) + volume spike (>2x avg)
        - Indicates forced liquidations are happening
        - The move will likely continue as more stops get hit
        - Trade WITH the cascade, not against it
        """
        if len(closes) < 20:
            return {"detected": False}

        # Price change over last 3 candles
        price_change_3 = (closes[-1] - closes[-4]) / closes[-4]

        # Volume spike (current vs 20-period average)
        avg_volume = np.mean(volumes[-20:])
        current_volume = volumes[-1]
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1

        # Detect cascade conditions
        is_cascade = (
            abs(price_change_3) > 0.015 and  # >1.5% move in 3 candles
            volume_ratio > 2.0  # >2x average volume
        )

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
        """
        Trade against extreme funding rates.

        Logic:
        - Very high positive funding (>0.1%) = too many longs → short bias
        - Very negative funding (<-0.1%) = too many shorts → long bias
        - This is a contrarian indicator — crowded trades get squeezed
        """
        extreme_threshold = 0.001  # 0.1%

        if funding_rate > extreme_threshold:
            return {
                "signal": True,
                "direction": "short",
                "strength": min(1.0, funding_rate / 0.003),
                "reason": f"Funding rate {funding_rate:.4%} is extremely positive — longs are crowded",
            }
        elif funding_rate < -extreme_threshold:
            return {
                "signal": True,
                "direction": "long",
                "strength": min(1.0, abs(funding_rate) / 0.003),
                "reason": f"Funding rate {funding_rate:.4%} is extremely negative — shorts are crowded",
            }

        return {"signal": False}

    # ---- Edge 3: Session Time Filter ----

    def session_time_filter(self) -> dict:
        """
        Trade during high-liquidity sessions.

        Crypto session patterns:
        - Asia (00:00-08:00 UTC): Lower volatility, range-bound
        - Europe (08:00-16:00 UTC): Medium volatility, trend continuation
        - US (13:00-21:00 UTC): Highest volatility, trend initiation
        - US-EU overlap (13:00-16:00 UTC): Best trading window
        """
        now = datetime.now(timezone.utc)
        hour = now.hour

        # US-EU overlap (best window)
        if 13 <= hour <= 16:
            return {
                "favorable": True,
                "session": "US-EU Overlap",
                "multiplier": 1.2,
                "reason": "Peak liquidity window — institutional flow active",
            }
        # US session
        elif 13 <= hour <= 21:
            return {
                "favorable": True,
                "session": "US",
                "multiplier": 1.1,
                "reason": "US session — high volatility expected",
            }
        # Europe session
        elif 8 <= hour <= 16:
            return {
                "favorable": True,
                "session": "Europe",
                "multiplier": 1.0,
                "reason": "European session — moderate activity",
            }
        # Asia session (lower edge)
        else:
            return {
                "favorable": False,
                "session": "Asia",
                "multiplier": 0.9,
                "reason": "Asian session — lower volatility, tighter ranges",
            }

    # ---- Edge 4: Volume Anomaly Detector ----

    def volume_anomaly_detector(
        self, closes: np.ndarray, volumes: np.ndarray
    ) -> dict:
        """
        Detect smart money accumulation/distribution.

        Logic:
        - High volume + small price change = accumulation (smart money buying quietly)
        - High volume + large price change = distribution (smart money selling into strength)
        - Volume divergence: price makes new high but volume declining = weakness
        """
        if len(closes) < 20:
            return {"anomaly": False}

        # Current vs average volume
        avg_vol = np.mean(volumes[-20:])
        current_vol = volumes[-1]
        vol_ratio = current_vol / avg_vol if avg_vol > 0 else 1

        # Price change
        price_change = abs(closes[-1] - closes[-2]) / closes[-2]

        # Volume anomaly: high volume + small price = accumulation
        if vol_ratio > 1.8 and price_change < 0.005:
            direction = "accumulation" if closes[-1] > closes[-2] else "distribution"
            return {
                "anomaly": True,
                "direction": direction,
                "volume_ratio": vol_ratio,
                "price_change": price_change,
                "reason": f"Smart money {direction}: {vol_ratio:.1f}x volume with only {price_change:.2%} price move",
            }

        # Volume divergence: price high but volume declining
        if len(closes) >= 10:
            price_higher = closes[-1] > closes[-5]
            vol_lower = np.mean(volumes[-3:]) < np.mean(volumes[-8:-3])

            if price_higher and vol_lower:
                return {
                    "anomaly": True,
                    "direction": "distribution",
                    "reason": "Bearish volume divergence: price rising but volume declining",
                }

        return {"anomaly": False}

    # ---- Edge 5: Multi-Timeframe Confluence ----

    def multi_timeframe_confluence(
        self,
        closes: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        volumes: np.ndarray,
    ) -> dict:
        """
        Check if multiple timeframes agree on direction.

        Uses EMA crossovers on different lookback periods as proxies:
        - Short-term (9/21 EMA) = "15m timeframe"
        - Medium-term (21/50 EMA) = "1H timeframe"
        - Long-term (50/100 EMA) = "4H timeframe"

        When all three align = high conviction trade
        """
        if len(closes) < 100:
            return {"aligned": False}

        # Short-term trend (proxy for 15m)
        ema_9 = calculate_ema(closes, 9)
        ema_21 = calculate_ema(closes, 21)
        short_bullish = ema_9[-1] > ema_21[-1]

        # Medium-term trend (proxy for 1H)
        ema_50 = calculate_ema(closes, 50)
        medium_bullish = ema_21[-1] > ema_50[-1]

        # Long-term trend (proxy for 4H)
        ema_100 = calculate_ema(closes, 100)
        long_bullish = ema_50[-1] > ema_100[-1]

        # Check confluence
        all_bullish = short_bullish and medium_bullish and long_bullish
        all_bearish = (not short_bullish) and (not medium_bullish) and (not long_bullish)

        if all_bullish:
            return {
                "aligned": True,
                "direction": "long",
                "timeframes_aligned": 3,
                "reason": "All timeframes bullish: 9>21>50>100 EMA",
            }
        elif all_bearish:
            return {
                "aligned": True,
                "direction": "short",
                "timeframes_aligned": 3,
                "reason": "All timeframes bearish: 9<21<50<100 EMA",
            }

        # Partial alignment (2 of 3)
        bullish_count = sum([short_bullish, medium_bullish, long_bullish])
        if bullish_count >= 2:
            return {
                "aligned": False,
                "direction": "long",
                "timeframes_aligned": 2,
                "reason": f"Partial bullish alignment ({bullish_count}/3)",
            }
        elif bullish_count <= 1:
            return {
                "aligned": False,
                "direction": "short",
                "timeframes_aligned": 3 - bullish_count,
                "reason": f"Partial bearish alignment ({3 - bullish_count}/3)",
            }

        return {"aligned": False, "direction": "neutral", "timeframes_aligned": 0}
