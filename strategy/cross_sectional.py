"""
Cross-sectional momentum strategy.

Ranks symbols by their N-bar return. Long the top third, short the
bottom third, no signal for the middle third.

Unlike single-asset momentum, cross-sectional rank is market-neutral
in expectation: the longs and shorts offset directional beta.

Requires at least 3 symbols to produce any signals.
"""
from __future__ import annotations

from core.events import SignalEvent
from core.models import Bar
from config.settings import StrategySettings
from strategy.multi_asset_base import MultiAssetStrategy


class CrossSectionalMomentumStrategy(MultiAssetStrategy):
    """
    Long top-tercile symbols, short bottom-tercile, flat on middle.
    Lookback: 20 bars (configurable via self.lookback).
    """

    def __init__(self, settings: StrategySettings, symbols: list[str]) -> None:
        super().__init__(settings, symbols)
        self.strategy_id = "cross_sectional_momentum"
        self.lookback = 20
        self._warmup_bars = self.lookback + 1  # need lookback + current bar

    def on_all_bars(self, bars: dict[str, list[Bar]]) -> list[SignalEvent]:
        returns: dict[str, float] = {}
        for symbol, bar_list in bars.items():
            if len(bar_list) < self.lookback:
                continue
            oldest_close = bar_list[-self.lookback].close
            newest_close = bar_list[-1].close
            if oldest_close > 0:
                returns[symbol] = (newest_close - oldest_close) / oldest_close

        if len(returns) < 3:
            return []  # need at least 3 symbols to meaningfully rank

        ranked = sorted(returns.items(), key=lambda x: x[1], reverse=True)
        n = len(ranked)
        tercile = max(1, n // 3)

        signals = []
        for i, (symbol, ret) in enumerate(ranked):
            if i < tercile:
                direction = 1.0
            elif i >= n - tercile:
                direction = -1.0
            else:
                continue  # middle tercile: no signal

            # Confidence scales with magnitude of the return differential
            confidence = min(abs(ret) * 20, 1.0)

            signals.append(SignalEvent(
                strategy_id=self.strategy_id,
                symbol=symbol,
                direction=direction,
                confidence=confidence,
                reason=f"xs_momentum rank={i+1}/{n} ret={ret:.4f}",
            ))

        return signals

    def reset(self) -> None:
        super().reset()
