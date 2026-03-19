"""
Momentum strategy.

Signal logic: if the price return over the lookback window exceeds
the threshold, go long. If it falls below -threshold, go short.
Otherwise, go flat (direction=0).

This is the simplest momentum strategy — it captures the idea that
assets that have been rising tend to continue rising (trend-following).
"""
from __future__ import annotations

from typing import Optional

from config.settings import StrategySettings
from core.events import MarketDataEvent, SignalEvent
from strategy.base import Strategy


class MomentumStrategy(Strategy):
    strategy_id = "momentum"

    def __init__(self, settings: StrategySettings) -> None:
        self.lookback_window = settings.momentum_lookback
        self.threshold = settings.momentum_threshold
        super().__init__()

    def generate_signal(self) -> Optional[SignalEvent]:
        bars = list(self.bars)
        oldest_close = bars[0].close
        newest_close = bars[-1].close
        symbol = bars[-1].symbol

        if oldest_close == 0:
            return None

        returns = (newest_close - oldest_close) / oldest_close

        if returns > self.threshold:
            direction = min(1.0, returns / (self.threshold * 5))  # scale up to 1.0
            confidence = min(1.0, abs(returns) / (self.threshold * 3))
            return SignalEvent(
                strategy_id=self.strategy_id,
                symbol=symbol,
                direction=direction,
                confidence=confidence,
                reason=f"momentum={returns:.3f} > threshold={self.threshold}",
            )
        elif returns < -self.threshold:
            direction = max(-1.0, returns / (self.threshold * 5))
            confidence = min(1.0, abs(returns) / (self.threshold * 3))
            return SignalEvent(
                strategy_id=self.strategy_id,
                symbol=symbol,
                direction=direction,
                confidence=confidence,
                reason=f"momentum={returns:.3f} < -{self.threshold}",
            )

        # No signal — stay flat
        return None
