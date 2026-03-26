"""
Simulated execution engine for backtesting.

Assumes fills at the next bar's open price (realistic: you can't
fill at the same bar that generated the signal).

Cost model:
  Market orders: filled at mid ± half-spread + sqrt market impact.
  Limit  orders: filled only when the bar's high/low crosses the limit.

Used by BacktestRunner. Identical interface to AlpacaExecutor so
the backtest wiring requires no mode flags in OMS or strategies.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Optional

import structlog

from core.event_bus import EventBus
from core.events import FillEvent
from core.models import Bar, Fill, Order, OrderSide, OrderStatus, OrderType
from execution.base import ExecutionEngine
from oms.state_machine import OrderStateMachine

logger = structlog.get_logger(__name__)

# Conservative assumed ADV (average daily volume) for market impact sizing.
# At $1M ADV, a 100-share order in a $200 stock is ~2% participation → ~2 bps impact.
_ASSUMED_ADV_SHARES = 1_000_000


class SimulatedExecutor(ExecutionEngine):
    """
    Fills orders at the price provided via set_current_price() / set_current_bar().

    BacktestRunner calls set_current_bar() after each bar with the next bar's data.
    set_current_price() remains for backward compatibility (tests use it).

    Cost model parameters:
      spread_bps               : round-trip spread in basis points (default 2 bps)
      market_impact_bps_per_pct_adv : additional impact per 1% of ADV traded (default 10 bps)
    """

    def __init__(
        self,
        event_bus: EventBus,
        spread_bps: float = 2.0,
        market_impact_bps_per_pct_adv: float = 10.0,
    ) -> None:
        self.bus = event_bus
        self.spread_bps = spread_bps
        self.market_impact_bps_per_pct_adv = market_impact_bps_per_pct_adv
        self._pending_orders: list[Order] = []
        self._current_prices: dict[str, float] = {}
        self._current_bars: dict[str, Bar] = {}

    async def connect(self) -> None:
        logger.info("simulator.connected")

    async def disconnect(self) -> None:
        self._pending_orders.clear()
        logger.info("simulator.disconnected")

    async def submit(self, order: Order) -> None:
        """Queue the order. It fills on the next set_current_bar() call."""
        order.status = OrderStateMachine.transition(order.status, OrderStatus.OPEN)
        self._pending_orders.append(order)
        logger.debug("simulator.order_queued", order_id=order.order_id, symbol=order.symbol)

    async def cancel(self, order_id: str) -> None:
        self._pending_orders = [o for o in self._pending_orders if o.order_id != order_id]

    async def set_current_bar(self, symbol: str, bar: Bar) -> None:
        """
        Called by BacktestRunner after each bar.
        Market orders fill with realistic spread + impact.
        Limit orders fill only if the bar's range crosses the limit price.
        """
        self._current_bars[symbol] = bar
        self._current_prices[symbol] = bar.close
        to_fill = [o for o in self._pending_orders if o.symbol == symbol]

        for order in to_fill:
            fill_price = self._compute_fill_price(order, bar)
            if fill_price is None:
                continue  # limit order: bar didn't cross the limit

            self._pending_orders.remove(order)
            fill = Fill(
                order_id=order.order_id,
                symbol=order.symbol,
                side=order.side,
                fill_price=fill_price,
                fill_quantity=order.quantity,
                strategy_id=order.strategy_id,
                timestamp=datetime.now(timezone.utc),
            )
            await self.bus.publish(FillEvent(fill=fill))

            spread_cost = self.spread_bps / 2 / 10_000
            impact_cost = self._impact_bps(order.quantity, bar.close) / 10_000
            logger.debug(
                "simulator.fill",
                order_id=order.order_id,
                symbol=symbol,
                mid_price=bar.close,
                fill_price=round(fill_price, 4),
                quantity=order.quantity,
                spread_cost_bps=round(self.spread_bps / 2, 2),
                impact_cost_bps=round(self._impact_bps(order.quantity, bar.close), 4),
            )

    async def set_current_price(self, symbol: str, price: float) -> None:
        """
        Backward-compatible shim. Constructs a synthetic bar (high=low=close=price)
        so limit fill logic can run safely. Spread/impact still applies to market orders.
        """
        synthetic_bar = Bar(
            symbol=symbol,
            open=price,
            high=price,
            low=price,
            close=price,
            volume=0,
            timestamp=datetime.now(timezone.utc),
        )
        await self.set_current_bar(symbol, synthetic_bar)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_fill_price(self, order: Order, bar: Bar) -> Optional[float]:
        """
        Returns the fill price for an order given the current bar, or None
        if the order cannot fill (limit order not crossed).
        """
        mid = bar.close
        half_spread = self.spread_bps / 2 / 10_000
        impact = self._impact_bps(order.quantity, mid) / 10_000

        if order.order_type == OrderType.MARKET or order.order_type is None:
            if order.side == OrderSide.BUY:
                return mid * (1 + half_spread + impact)
            else:
                return mid * (1 - half_spread - impact)

        elif order.order_type == OrderType.LIMIT:
            limit = order.limit_price
            if limit is None:
                # Malformed limit order — fill at mid as fallback
                return mid
            if order.side == OrderSide.BUY and bar.low <= limit:
                return limit  # filled passively, no spread cost
            if order.side == OrderSide.SELL and bar.high >= limit:
                return limit
            return None  # limit not crossed this bar

        return mid  # unknown order type: fill at mid

    def _impact_bps(self, quantity: int, price: float) -> float:
        """
        Sqrt market impact model (simplified Almgren-Chriss).
        Impact in basis points = impact_coeff * sqrt(participation_rate)
        where participation_rate = quantity / ADV.
        """
        if price <= 0 or _ASSUMED_ADV_SHARES <= 0:
            return 0.0
        participation = quantity / _ASSUMED_ADV_SHARES
        import math
        return self.market_impact_bps_per_pct_adv * math.sqrt(participation * 100)
