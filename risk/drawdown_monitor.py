"""
Portfolio drawdown-from-peak monitor.

The daily circuit breaker resets each day — it cannot catch slow multi-day
capital erosion. This monitor tracks the rolling peak portfolio value and
halts trading when cumulative drawdown from peak exceeds the threshold.
"""
from __future__ import annotations
from datetime import datetime, timezone
import structlog

logger = structlog.get_logger(__name__)


class DrawdownMonitor:
    def __init__(self, max_drawdown_pct: float = 0.10) -> None:
        self.max_drawdown_pct = max_drawdown_pct
        self._peak_value: float = 0.0
        self._current_value: float = 0.0
        self._halted: bool = False

    def update(self, portfolio_value: float) -> bool:
        """
        Update portfolio value. Returns True if trading should halt.
        Call this after every fill or on a periodic heartbeat.
        """
        self._current_value = portfolio_value
        if portfolio_value > self._peak_value:
            self._peak_value = portfolio_value
            if self._halted:
                logger.info("drawdown_monitor.recovered", peak=self._peak_value)
                self._halted = False

        if self._peak_value > 0:
            drawdown = (self._peak_value - portfolio_value) / self._peak_value
            if drawdown >= self.max_drawdown_pct and not self._halted:
                self._halted = True
                logger.critical(
                    "drawdown_monitor.halt_triggered",
                    peak=self._peak_value,
                    current=portfolio_value,
                    drawdown_pct=round(drawdown * 100, 2),
                    threshold_pct=round(self.max_drawdown_pct * 100, 2),
                )
        return self._halted

    @property
    def is_halted(self) -> bool:
        return self._halted

    @property
    def current_drawdown_pct(self) -> float:
        if self._peak_value <= 0:
            return 0.0
        return (self._peak_value - self._current_value) / self._peak_value

    def reset(self) -> None:
        """Manual reset — requires human intervention to acknowledge halt."""
        self._halted = False
        logger.info("drawdown_monitor.manually_reset")
