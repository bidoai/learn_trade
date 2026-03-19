"""
Circuit breaker — halts all trading when daily loss exceeds the threshold.

State machine:
  NORMAL ──daily_loss > threshold──▶ TRIGGERED ──new_trading_day──▶ NORMAL

When TRIGGERED:
  - All OrderRequestEvents are blocked (OMS checks before submitting)
  - RiskAlertEvent('circuit_breaker_triggered') is emitted
  - Dashboard shows a red alert banner

The circuit breaker check is called from inside RiskEngine.check(),
which is itself called synchronously by OMS before every order.
It lives in a try/finally in OMS to guarantee it cannot be bypassed
by exceptions in other risk checks.
"""
from __future__ import annotations

from datetime import date, datetime
from enum import Enum, auto

import structlog

logger = structlog.get_logger(__name__)


class CircuitBreakerState(Enum):
    NORMAL = auto()
    TRIGGERED = auto()


class CircuitBreaker:
    def __init__(self, max_daily_loss_pct: float, initial_capital: float) -> None:
        self.max_daily_loss_pct = max_daily_loss_pct
        self.initial_capital = initial_capital
        self.max_daily_loss_abs = initial_capital * max_daily_loss_pct

        self.state = CircuitBreakerState.NORMAL
        self._daily_loss: float = 0.0
        self._trading_day: date = date.today()

    # ------------------------------------------------------------------
    # Called by RiskEngine.check() — synchronous, always
    # ------------------------------------------------------------------

    def is_triggered(self) -> bool:
        """Returns True if all trading should be halted."""
        self._check_day_rollover()
        return self.state == CircuitBreakerState.TRIGGERED

    def record_pnl(self, realized_pnl: float) -> bool:
        """
        Update daily P&L after a fill. Returns True if circuit breaker
        was just triggered (caller should emit RiskAlertEvent).
        """
        self._check_day_rollover()

        if realized_pnl < 0:
            self._daily_loss += abs(realized_pnl)

        if (
            self.state == CircuitBreakerState.NORMAL
            and self._daily_loss >= self.max_daily_loss_abs
        ):
            self.state = CircuitBreakerState.TRIGGERED
            logger.critical(
                "circuit_breaker.triggered",
                daily_loss=self._daily_loss,
                threshold=self.max_daily_loss_abs,
                threshold_pct=self.max_daily_loss_pct,
            )
            return True  # caller should emit RiskAlertEvent

        return False

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _check_day_rollover(self) -> None:
        """Reset daily loss at the start of a new trading day."""
        today = date.today()
        if today != self._trading_day:
            if self.state == CircuitBreakerState.TRIGGERED:
                logger.info(
                    "circuit_breaker.reset",
                    old_day=self._trading_day.isoformat(),
                    new_day=today.isoformat(),
                )
                self.state = CircuitBreakerState.NORMAL
            self._daily_loss = 0.0
            self._trading_day = today

    @property
    def daily_loss(self) -> float:
        return self._daily_loss

    @property
    def daily_loss_pct(self) -> float:
        if self.initial_capital == 0:
            return 0.0
        return self._daily_loss / self.initial_capital
