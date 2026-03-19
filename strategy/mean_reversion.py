"""
Mean reversion strategy.

Signal logic: compute a rolling z-score of the price vs. its mean.
If z-score > threshold, price is overbought → go short.
If z-score < -threshold, price is oversold → go long.

This is the complement to momentum — it captures the idea that extreme
price moves tend to revert to the mean (contrarian).
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from config.settings import StrategySettings
from core.events import SignalEvent
from strategy.base import Strategy


class MeanReversionStrategy(Strategy):
    strategy_id = "mean_reversion"

    def __init__(self, settings: StrategySettings) -> None:
        self.lookback_window = settings.mean_reversion_lookback
        self.zscore_threshold = settings.mean_reversion_zscore
        super().__init__()

    def generate_signal(self) -> Optional[SignalEvent]:
        closes = np.array([b.close for b in self.bars])
        symbol = list(self.bars)[-1].symbol

        mean = closes.mean()
        std = closes.std()

        if std == 0:
            return None  # no volatility — no signal

        zscore = (closes[-1] - mean) / std

        if zscore < -self.zscore_threshold:
            # Oversold: expect reversion upward → long
            confidence = min(1.0, abs(zscore) / (self.zscore_threshold * 2))
            return SignalEvent(
                strategy_id=self.strategy_id,
                symbol=symbol,
                direction=min(1.0, abs(zscore) / (self.zscore_threshold * 2)),
                confidence=confidence,
                reason=f"zscore={zscore:.2f} oversold (threshold={self.zscore_threshold})",
            )
        elif zscore > self.zscore_threshold:
            # Overbought: expect reversion downward → short
            confidence = min(1.0, abs(zscore) / (self.zscore_threshold * 2))
            return SignalEvent(
                strategy_id=self.strategy_id,
                symbol=symbol,
                direction=-min(1.0, abs(zscore) / (self.zscore_threshold * 2)),
                confidence=confidence,
                reason=f"zscore={zscore:.2f} overbought (threshold={self.zscore_threshold})",
            )

        return None
