"""
Abstract base class for all trading strategies.

All strategies must subclass Strategy and implement generate_signal().
The base class handles:
  - Rolling bar window (deque with maxlen=lookback_window)
  - NaN guard: validates each bar before appending
  - Warmup period: returns None until lookback_window bars are available
  - reset(): clears state between backtest runs

Implementing a strategy:
    class MyStrategy(Strategy):
        strategy_id = "my_strategy"
        lookback_window = 50

        def generate_signal(self) -> Optional[SignalEvent]:
            # self.bars is always fully populated (lookback_window bars)
            closes = [b.close for b in self.bars]
            ...

Data flow for each bar:
    on_market_data(event)
        │
        ├── validate bar (NaN guard)
        │     └── invalid → log warning, return None
        │
        ├── append to self.bars
        │
        ├── check warmup
        │     └── not enough bars → return None
        │
        └── call generate_signal()
              └── return SignalEvent or None
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections import deque
from typing import Optional

import structlog

from core.events import MarketDataEvent, SignalEvent
from core.models import Bar

logger = structlog.get_logger(__name__)


class Strategy(ABC):
    """
    Abstract base for all strategies.

    Subclasses must set:
        strategy_id: str   — unique identifier (used in events, logs, DB)
        lookback_window: int — number of bars needed before signals are generated
    """
    strategy_id: str
    lookback_window: int = 200

    def __init__(self) -> None:
        self.bars: deque[Bar] = deque(maxlen=self.lookback_window)
        self._bar_count: int = 0   # total bars seen (including pre-warmup)
        self._log = logger.bind(strategy=self.strategy_id)

    # ------------------------------------------------------------------
    # Public interface — called by the system
    # ------------------------------------------------------------------

    def on_market_data(self, event: MarketDataEvent) -> Optional[SignalEvent]:
        """
        Process one market data bar.
        Returns a SignalEvent or None (no action).
        Never raises — all exceptions are caught and logged.
        """
        bar = event.bar

        # NaN guard: reject invalid bars before they contaminate the window
        if not bar.is_valid():
            self._log.warning(
                "strategy.invalid_bar_rejected",
                symbol=bar.symbol,
                close=bar.close,
            )
            return None

        self.bars.append(bar)
        self._bar_count += 1

        # Warmup: don't generate signals until we have enough history
        if len(self.bars) < self.lookback_window:
            return None

        try:
            signal = self.generate_signal()
        except Exception:
            self._log.exception("strategy.generate_signal_error")
            return None

        if signal is not None:
            # Validate signal before returning
            if math.isnan(signal.direction) or math.isnan(signal.confidence):
                self._log.warning(
                    "strategy.nan_signal_rejected",
                    direction=signal.direction,
                    confidence=signal.confidence,
                )
                return None

        return signal

    def reset(self) -> None:
        """
        Clear all state. MUST be called by BacktestRunner before each run.
        Without this, state from a previous backtest contaminates the next.
        """
        self.bars.clear()
        self._bar_count = 0
        self._reset_state()
        self._log.debug("strategy.reset")

    # ------------------------------------------------------------------
    # Subclass interface — must implement
    # ------------------------------------------------------------------

    @abstractmethod
    def generate_signal(self) -> Optional[SignalEvent]:
        """
        Called when self.bars has exactly lookback_window bars.
        Return a SignalEvent to act, or None to do nothing.
        Do NOT return NaN — the base class validates your return value.
        """
        ...

    def _reset_state(self) -> None:
        """
        Override to reset any additional state (e.g. trained ML models,
        indicator accumulators). Called by reset().
        """
        pass
