"""
Multi-asset strategy base class.

Extends the single-asset Strategy pattern to handle multiple symbols
simultaneously. Each symbol gets its own bar deque and warmup state.
The strategy fires when ALL symbols have warmed up.

Usage:
    class MyPairsStrategy(MultiAssetStrategy):
        def __init__(self, settings, symbols):
            super().__init__(settings, symbols)
            self.strategy_id = "my_pairs"

        def on_all_bars(self, bars: dict[str, list[Bar]]) -> list[SignalEvent]:
            ...
"""
from __future__ import annotations

from abc import abstractmethod
from collections import deque
from typing import Optional

from config.settings import StrategySettings
from core.events import MarketDataEvent, SignalEvent
from core.models import Bar


class MultiAssetStrategy:
    def __init__(self, settings: StrategySettings, symbols: list[str]) -> None:
        self.settings = settings
        self.symbols = symbols
        self.strategy_id: str = "multi_asset"  # override in subclass
        self._bars: dict[str, deque[Bar]] = {
            s: deque(maxlen=settings.lookback_bars if hasattr(settings, "lookback_bars") else 100)
            for s in symbols
        }
        self._warmed_up: dict[str, bool] = {s: False for s in symbols}
        self._warmup_bars = 20  # default; subclass can override

    def on_market_data(self, event: MarketDataEvent) -> list[SignalEvent]:
        """Called for each bar of any subscribed symbol."""
        symbol = event.bar.symbol
        if symbol not in self._bars:
            return []

        self._bars[symbol].append(event.bar)

        if not self._warmed_up[symbol] and len(self._bars[symbol]) >= self._warmup_bars:
            self._warmed_up[symbol] = True

        if not all(self._warmed_up.values()):
            return []

        current_bars = {s: list(self._bars[s]) for s in self.symbols}
        return self.on_all_bars(current_bars)

    @abstractmethod
    def on_all_bars(self, bars: dict[str, list[Bar]]) -> list[SignalEvent]:
        """
        Override this. Called when all symbols have enough bars.
        bars: {symbol: [Bar, ...]} in chronological order (oldest first).
        """
        ...

    def reset(self) -> None:
        """Called before every backtest run."""
        for s in self.symbols:
            self._bars[s].clear()
            self._warmed_up[s] = False
