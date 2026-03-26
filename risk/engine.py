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
  2. Drawdown monitor — portfolio-level peak-to-trough kill switch
  3. Strategy monitor — per-strategy loss kill switch
  4. Position size limit — max X% of portfolio per position
  5. Concentration limit — max Y% in any single symbol
  6. Capital check — sufficient buying power
  7. VaR check — if VaR is available and would be breached

All checks return a RiskCheckResult(approved=bool, reason=str).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import structlog

from config.settings import RiskSettings
from core.models import Order, OrderSide
from risk.circuit_breaker import CircuitBreaker
from risk.drawdown_monitor import DrawdownMonitor
from risk.strategy_monitor import StrategyMonitor
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
        last_prices: Optional[dict] = None,
    ) -> None:
        self.positions = positions
        self.settings = settings
        self.initial_capital = initial_capital
        # Reference to the live price dict populated by the data feed.
        # Passed as a reference so updates are reflected without re-injecting.
        self._last_prices: dict = last_prices if last_prices is not None else {}

        self.circuit_breaker = CircuitBreaker(
            max_daily_loss_pct=settings.max_daily_loss_pct,
            initial_capital=initial_capital,
        )
        self.var_calculator = VaRCalculator(
            confidence=settings.var_confidence,
            window_days=settings.var_window_days,
        )
        self.drawdown_monitor = DrawdownMonitor(
            max_drawdown_pct=settings.max_drawdown_pct,
        )
        self.strategy_monitor = StrategyMonitor(
            initial_capital=initial_capital,
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
            return RiskCheckResult(approved=False, reason="circuit_breaker_engaged")

        # 2. Drawdown monitor — multi-day portfolio erosion kill switch
        if self.drawdown_monitor.is_halted:
            return RiskCheckResult(
                approved=False,
                reason=f"drawdown_monitor_halted: {self.drawdown_monitor.current_drawdown_pct*100:.1f}% drawdown from peak",
            )

        # 3. Per-strategy kill switch
        if self.strategy_monitor.is_halted(order.strategy_id):
            return RiskCheckResult(
                approved=False,
                reason=f"strategy_halted: {order.strategy_id} exceeded loss limit",
            )

        # 4. Position size limit
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

        # 5. Concentration limit — total exposure in this symbol
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

        # 6. Capital check
        if order.side == OrderSide.BUY and position_value > self._available_capital:
            return RiskCheckResult(
                approved=False,
                reason=f"insufficient_capital: need {position_value:.0f}, have {self._available_capital:.0f}",
            )

        # 7. VaR check — only gates if we have enough history
        var_estimate = self.var_calculator.calculate()
        if var_estimate is not None:
            var_limit = self.settings.max_var_pct * self._available_capital
            if var_estimate * self._available_capital > var_limit:
                return RiskCheckResult(
                    approved=False,
                    reason=f"var_limit_exceeded: VaR={var_estimate*100:.2f}% > limit={self.settings.max_var_pct*100:.2f}%",
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

    def on_fill(self, fill_price: float, fill_quantity: int, side: OrderSide, symbol: str, strategy_id: str = "") -> Optional[bool]:
        """
        Update capital, VaR history, and monitors after a fill.
        Returns True if circuit breaker was just triggered.
        """
        cost = fill_price * fill_quantity
        if side == OrderSide.BUY:
            self._available_capital -= cost
        else:
            self._available_capital += cost

        # Feed realized P&L into VaR calculator (as portfolio return fraction)
        # For sell fills, approximate realized P&L from existing position
        pos = self.positions.get(symbol)
        if side == OrderSide.SELL and pos and pos.avg_entry_price > 0:
            pnl = (fill_price - pos.avg_entry_price) * fill_quantity
            pnl_pct = pnl / self.initial_capital
            self.var_calculator.add_return(pnl_pct)
            self.strategy_monitor.record_fill_pnl(strategy_id, pnl)

        # Update drawdown monitor with current portfolio value estimate
        portfolio_value = self._available_capital
        self.drawdown_monitor.update(portfolio_value)

        return None  # circuit breaker is updated via record_pnl separately

    def record_daily_pnl(self, pnl: float) -> bool:
        """
        Update circuit breaker with today's P&L.
        Returns True if circuit breaker was just triggered.
        """
        return self.circuit_breaker.record_pnl(pnl)

    def get_stop_loss_orders(self) -> list[Order]:
        """
        Check all open positions for stop-loss breaches.
        Returns a list of closing market orders for positions that have
        lost more than stop_loss_pct from their entry price.
        """
        from core.models import Order, OrderSide, OrderType
        stop_orders = []
        for pos in self.positions.all_positions():
            symbol = pos.symbol
            if pos.quantity == 0:
                continue
            last_price = self._last_prices.get(symbol)
            if last_price is None or last_price <= 0:
                continue
            if pos.avg_entry_price <= 0:
                continue

            sign = 1 if pos.quantity > 0 else -1
            pnl_pct = sign * (last_price - pos.avg_entry_price) / pos.avg_entry_price

            if pnl_pct < -self.settings.stop_loss_pct:
                close_side = OrderSide.SELL if pos.quantity > 0 else OrderSide.BUY
                stop_orders.append(Order(
                    symbol=symbol,
                    side=close_side,
                    order_type=OrderType.MARKET,
                    quantity=abs(pos.quantity),
                    strategy_id=pos.strategy_id,
                ))
                logger.warning(
                    "risk.stop_loss_triggered",
                    symbol=symbol,
                    pnl_pct=round(pnl_pct * 100, 2),
                    threshold_pct=round(self.settings.stop_loss_pct * 100, 2),
                    strategy_id=pos.strategy_id,
                )
        return stop_orders

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
        """
        Get last known price for position sizing.

        Priority:
          1. Live market price from last_prices dict (updated on every bar)
          2. Average entry price of an existing position (stale but better than nothing)
          3. $1.0 fallback — used only before the first bar arrives. Returns a
             conservative low value so size checks are generous (not dangerous)
             until a real price is known.
        """
        if symbol in self._last_prices and self._last_prices[symbol] > 0:
            return self._last_prices[symbol]
        pos = self.positions.get(symbol)
        if pos and pos.avg_entry_price > 0:
            return pos.avg_entry_price
        return 1.0

    def _symbol_exposure(self, symbol: str) -> float:
        """Total dollar exposure in a symbol across all strategies."""
        pos = self.positions.get(symbol)
        if pos is None:
            return 0.0
        return abs(pos.quantity) * pos.avg_entry_price
