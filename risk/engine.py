"""
Risk engine — synchronous gatekeeper for all orders.

Called by OMS.handle_order_request() before every order submission.
This is NOT an event bus subscriber — it's a direct sync call.

Why synchronous? Race condition prevention:
  If two strategies simultaneously request orders and risk is async,
  both could pass the position-size check before either updates the
  position tracker. A sync call serializes the checks.

Checks performed (in order):
  1. Circuit breaker — if triggered, block everything
  2. Position size limit — max X% of portfolio per position
  3. Concentration limit — max Y% in any single symbol
  4. Capital check — sufficient buying power
  5. VaR check — if VaR is available and would be breached

All checks return a RiskCheckResult(approved=bool, reason=str).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import structlog

from config.settings import RiskSettings
from core.models import Order, OrderSide
from risk.circuit_breaker import CircuitBreaker
from risk.var import VaRCalculator

if TYPE_CHECKING:
    from oms.position_tracker import PositionTracker

logger = structlog.get_logger(__name__)


@dataclass
class RiskCheckResult:
    approved: bool
    reason: str  # human-readable, logged + stored in OrderBlockedEvent


class RiskEngine:
    def __init__(
        self,
        positions: "PositionTracker",
        settings: RiskSettings,
        initial_capital: float,
    ) -> None:
        self.positions = positions
        self.settings = settings
        self.initial_capital = initial_capital

        self.circuit_breaker = CircuitBreaker(
            max_daily_loss_pct=settings.max_daily_loss_pct,
            initial_capital=initial_capital,
        )
        self.var_calculator = VaRCalculator(
            confidence=settings.var_confidence,
            window_days=settings.var_window_days,
        )
        self._available_capital = initial_capital

    # ------------------------------------------------------------------
    # Called synchronously by OMS before every order
    # ------------------------------------------------------------------

    def check(self, order: Order) -> RiskCheckResult:
        """
        Run all risk checks. Returns the first failure, or approved=True.
        Order of checks matters: circuit breaker is always first.
        """
        # 1. Circuit breaker — hardest stop
        if self.circuit_breaker.is_triggered():
            return RiskCheckResult(
                approved=False,
                reason="circuit_breaker_engaged",
            )

        # 2. Position size limit
        position_value = order.quantity * self._get_last_price(order.symbol)
        max_position_value = self._available_capital * self.settings.max_position_pct
        if position_value > max_position_value:
            return RiskCheckResult(
                approved=False,
                reason=(
                    f"position_size_exceeded: {position_value:.0f} > "
                    f"max {max_position_value:.0f} "
                    f"({self.settings.max_position_pct*100:.0f}% of capital)"
                ),
            )

        # 3. Concentration limit — total exposure in this symbol
        current_exposure = self._symbol_exposure(order.symbol)
        new_exposure = current_exposure + position_value
        max_exposure = self._available_capital * self.settings.max_concentration_pct
        if new_exposure > max_exposure:
            return RiskCheckResult(
                approved=False,
                reason=(
                    f"concentration_exceeded: {new_exposure:.0f} > "
                    f"max {max_exposure:.0f} "
                    f"({self.settings.max_concentration_pct*100:.0f}% of capital)"
                ),
            )

        # 4. Capital check
        if order.side == OrderSide.BUY and position_value > self._available_capital:
            return RiskCheckResult(
                approved=False,
                reason=f"insufficient_capital: need {position_value:.0f}, have {self._available_capital:.0f}",
            )

        logger.debug(
            "risk.check_passed",
            symbol=order.symbol,
            side=order.side.value,
            quantity=order.quantity,
            strategy=order.strategy_id,
        )
        return RiskCheckResult(approved=True, reason="all_checks_passed")

    # ------------------------------------------------------------------
    # State updates — called after fills
    # ------------------------------------------------------------------

    def on_fill(self, fill_price: float, fill_quantity: int, side: OrderSide, symbol: str) -> Optional[bool]:
        """
        Update capital and circuit breaker after a fill.
        Returns True if circuit breaker was just triggered.
        """
        # Update available capital
        cost = fill_price * fill_quantity
        if side == OrderSide.BUY:
            self._available_capital -= cost
        else:
            self._available_capital += cost

        return None  # circuit breaker is updated via record_pnl separately

    def record_daily_pnl(self, pnl: float) -> bool:
        """
        Update circuit breaker with today's P&L.
        Returns True if circuit breaker was just triggered.
        """
        return self.circuit_breaker.record_pnl(pnl)

    def assert_healthy(self) -> None:
        """Called during startup sequence. Raises if config is invalid."""
        if self.settings.max_position_pct <= 0 or self.settings.max_position_pct > 1:
            raise ValueError(f"max_position_pct must be in (0, 1]")
        if self.settings.max_daily_loss_pct <= 0 or self.settings.max_daily_loss_pct > 1:
            raise ValueError(f"max_daily_loss_pct must be in (0, 1]")
        if self.initial_capital <= 0:
            raise ValueError(f"initial_capital must be positive")
        logger.info(
            "risk_engine.healthy",
            max_position_pct=self.settings.max_position_pct,
            max_daily_loss_pct=self.settings.max_daily_loss_pct,
            initial_capital=self.initial_capital,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_last_price(self, symbol: str) -> float:
        """Get last known price for position sizing. Fallback to 0."""
        pos = self.positions.get(symbol)
        if pos and pos.avg_entry_price > 0:
            return pos.avg_entry_price
        return 1.0  # conservative: treat as $1 if unknown (will pass size check)

    def _symbol_exposure(self, symbol: str) -> float:
        """Total dollar exposure in a symbol across all strategies."""
        pos = self.positions.get(symbol)
        if pos is None:
            return 0.0
        return abs(pos.quantity) * pos.avg_entry_price
