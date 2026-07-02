"""WEEX AI Wars II — Composite Strategy Orchestrator"""

import numpy as np
from datetime import datetime

from ..core.models import Candle, Signal, Side, MarketRegime
from ..indicators.technical import (
    calculate_rsi, calculate_macd, calculate_bollinger_bands,
    calculate_atr, calculate_adx, calculate_ema, calculate_vwap,
    detect_regime, calculate_stochastic_rsi,
)
from .edges import EdgeStrategies


class CompositeStrategy:
    """
    Multi-strategy adaptive system:
    - Trend following (MACD + BB breakout) in trending markets
    - Mean reversion (RSI + BB bounce) in ranging markets
    - Regime detection switches between strategies
    - Edge strategies provide alpha over retail bots
    """

    def __init__(self, config: dict):
        self.config = config
        self.tf_config = config.get("strategy", {}).get("trend_follow", {})
        self.mr_config = config.get("strategy", {}).get("mean_reversion", {})
        self.ind_config = config.get("indicators", {})
        self.edges = EdgeStrategies(config)

    def analyze(self, symbol: str, candles: list[Candle], funding_rate: float = 0.0) -> Signal | None:
        """Analyze candles and generate a signal with edge enhancements."""
        if len(candles) < 50:
            return None

        # Extract OHLCV arrays
        opens = np.array([c.open for c in candles])
        highs = np.array([c.high for c in candles])
        lows = np.array([c.low for c in candles])
        closes = np.array([c.close for c in candles])
        volumes = np.array([c.volume for c in candles])

        # Detect market regime
        regime = detect_regime(
            highs, lows, closes,
            self.ind_config.get("adx_period", 14),
            self.tf_config.get("adx_threshold", 25),
        )

        # Get signals from both strategies
        trend_signal = self._trend_follow_signal(symbol, closes, highs, lows, volumes, regime)
        reversion_signal = self._mean_reversion_signal(symbol, closes, highs, lows, volumes, regime)

        # Weight and combine signals
        signals = []
        if trend_signal and self.tf_config.get("enabled", True):
            signals.append((trend_signal, self.tf_config.get("weight", 0.6)))
        if reversion_signal and self.mr_config.get("enabled", True):
            signals.append((reversion_signal, self.mr_config.get("weight", 0.4)))

        if not signals:
            return None

        # Pick strongest weighted signal
        best = max(signals, key=lambda x: x[0].strength * x[1])
        signal = best[0]

        # ---- Apply Edge Strategies ----
        edges = self.edges.analyze_all_edges(candles, funding_rate)
        edge_multiplier, edge_direction = self.edges.get_combined_modifier(edges)

        # Apply edge multiplier to signal strength
        signal.strength *= edge_multiplier

        # If edges strongly suggest opposite direction, skip
        if edge_direction == "long" and signal.side == Side.SHORT and edge_multiplier > 1.3:
            return None
        if edge_direction == "short" and signal.side == Side.LONG and edge_multiplier > 1.3:
            return None

        # Legacy funding rate filter (superseded by edges but kept as safety)
        if funding_rate > 0.001 and signal.side == Side.LONG:
            signal.strength *= 0.7
        elif funding_rate < -0.001 and signal.side == Side.SHORT:
            signal.strength *= 0.7

        # Minimum strength threshold
        if signal.strength < 0.4:
            return None

        return signal

    def _trend_follow_signal(
        self, symbol: str, closes: np.ndarray, highs: np.ndarray,
        lows: np.ndarray, volumes: np.ndarray, regime: MarketRegime,
    ) -> Signal | None:
        """Trend-following strategy: MACD + Bollinger Band breakout."""
        # Only trade in trending markets
        if regime not in (MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN):
            return None

        # Calculate indicators
        macd, signal_line, histogram = calculate_macd(
            closes,
            self.tf_config.get("macd_fast", 12),
            self.tf_config.get("macd_slow", 26),
            self.tf_config.get("macd_signal", 9),
        )

        bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(
            closes,
            self.tf_config.get("bb_period", 20),
            self.tf_config.get("bb_std", 2.0),
        )

        atr = calculate_atr(highs, lows, closes, 14)
        ema_fast = calculate_ema(closes, 9)
        ema_slow = calculate_ema(closes, 21)
        vwap = calculate_vwap(highs, lows, closes, volumes, 20)

        current_price = closes[-1]
        current_atr = atr[-1]

        # ---- Long Signal ----
        if (regime == MarketRegime.TRENDING_UP and
            histogram[-1] > 0 and histogram[-2] <= 0 and  # MACD histogram turning positive
            current_price > bb_middle[-1] and  # Above BB middle
            ema_fast[-1] > ema_slow[-1] and  # Fast EMA above slow
            not np.isnan(vwap[-1]) and current_price > vwap[-1]):  # Above VWAP

            strength = 0.5
            # Stronger signal if MACD is accelerating
            if histogram[-1] > histogram[-2]:
                strength += 0.15
            # Stronger if price is near BB upper (momentum)
            if current_price > bb_middle[-1] + (bb_upper[-1] - bb_middle[-1]) * 0.3:
                strength += 0.1

            return Signal(
                symbol=symbol,
                side=Side.LONG,
                strength=min(1.0, strength),
                strategy="trend_follow",
                entry_price=current_price,
                stop_loss=current_price - current_atr * 2,
                take_profit=current_price + current_atr * 3,
                leverage=self.config.get("trading", {}).get("default_leverage", 5),
                reason=f"Trend UP: MACD cross + above BB mid + VWAP confirm",
            )

        # ---- Short Signal ----
        if (regime == MarketRegime.TRENDING_DOWN and
            histogram[-1] < 0 and histogram[-2] >= 0 and  # MACD histogram turning negative
            current_price < bb_middle[-1] and  # Below BB middle
            ema_fast[-1] < ema_slow[-1] and  # Fast EMA below slow
            not np.isnan(vwap[-1]) and current_price < vwap[-1]):  # Below VWAP

            strength = 0.5
            if histogram[-1] < histogram[-2]:
                strength += 0.15
            if current_price < bb_middle[-1] - (bb_middle[-1] - bb_lower[-1]) * 0.3:
                strength += 0.1

            return Signal(
                symbol=symbol,
                side=Side.SHORT,
                strength=min(1.0, strength),
                strategy="trend_follow",
                entry_price=current_price,
                stop_loss=current_price + current_atr * 2,
                take_profit=current_price - current_atr * 3,
                leverage=self.config.get("trading", {}).get("default_leverage", 5),
                reason=f"Trend DOWN: MACD cross + below BB mid + VWAP confirm",
            )

        return None

    def _mean_reversion_signal(
        self, symbol: str, closes: np.ndarray, highs: np.ndarray,
        lows: np.ndarray, volumes: np.ndarray, regime: MarketRegime,
    ) -> Signal | None:
        """Mean-reversion strategy: RSI extremes + BB bounce."""
        # Only trade in ranging markets
        if regime != MarketRegime.RANGING:
            return None

        rsi = calculate_rsi(closes, self.mr_config.get("rsi_period", 14))
        bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(
            closes,
            self.mr_config.get("bb_period", 20),
            self.mr_config.get("bb_std", 2.0),
        )
        atr = calculate_atr(highs, lows, closes, 14)
        stoch_k, stoch_d = calculate_stochastic_rsi(closes)

        current_price = closes[-1]
        current_atr = atr[-1]
        current_rsi = rsi[-1]

        # ---- Long Signal (Oversold bounce) ----
        if (current_rsi < self.mr_config.get("rsi_oversold", 25) and
            current_price <= bb_lower[-1] * 1.01 and  # Near or below lower BB
            stoch_k[-1] < 20 and  # Stoch RSI oversold
            stoch_k[-1] > stoch_d[-1]):  # Stoch RSI crossing up

            strength = 0.5
            # Stronger signal at extreme RSI
            if current_rsi < 20:
                strength += 0.2
            # BB squeeze adds confidence
            bb_width = (bb_upper[-1] - bb_lower[-1]) / bb_middle[-1]
            if bb_width < 0.03:  # Tight squeeze
                strength += 0.1

            return Signal(
                symbol=symbol,
                side=Side.LONG,
                strength=min(1.0, strength),
                strategy="mean_reversion",
                entry_price=current_price,
                stop_loss=current_price - current_atr * 1.5,
                take_profit=bb_middle[-1],  # Target: BB middle
                leverage=self.config.get("trading", {}).get("default_leverage", 5),
                reason=f"Reversion LONG: RSI={current_rsi:.0f}, BB lower touch",
            )

        # ---- Short Signal (Overbought rejection) ----
        if (current_rsi > self.mr_config.get("rsi_overbought", 75) and
            current_price >= bb_upper[-1] * 0.99 and  # Near or above upper BB
            stoch_k[-1] > 80 and  # Stoch RSI overbought
            stoch_k[-1] < stoch_d[-1]):  # Stoch RSI crossing down

            strength = 0.5
            if current_rsi > 80:
                strength += 0.2
            bb_width = (bb_upper[-1] - bb_lower[-1]) / bb_middle[-1]
            if bb_width < 0.03:
                strength += 0.1

            return Signal(
                symbol=symbol,
                side=Side.SHORT,
                strength=min(1.0, strength),
                strategy="mean_reversion",
                entry_price=current_price,
                stop_loss=current_price + current_atr * 1.5,
                take_profit=bb_middle[-1],
                leverage=self.config.get("trading", {}).get("default_leverage", 5),
                reason=f"Reversion SHORT: RSI={current_rsi:.0f}, BB upper touch",
            )

        return None
