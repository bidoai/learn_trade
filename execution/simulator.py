"""
Simulated execution engine for backtesting.

Assumes fills at the next bar's open price (realistic: you can't
fill at the same bar that generated the signal).

Used by BacktestRunner. Identical interface to AlpacaExecutor so
the backtest wiring requires no mode flags in OMS or strategies.
"""
from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Optional

import structlog

from core.event_bus import EventBus
from core.events import FillEvent
from core.models import Fill, Order, OrderSide, OrderStatus
from execution.base import ExecutionEngine
from oms.state_machine import OrderStateMachine

logger = structlog.get_logger(__name__)


class SimulatedExecutor(ExecutionEngine):
    """
    Fills orders at the price provided via set_current_price().
    BacktestRunner calls this after each bar with the next bar's open.
    """

    def __init__(self, event_bus: EventBus) -> None:
        self.bus = event_bus
        self._pending_orders: list[Order] = []
        self._current_prices: dict[str, float] = {}

    async def connect(self) -> None:
        logger.info("simulator.connected")

    async def disconnect(self) -> None:
        self._pending_orders.clear()
        logger.info("simulator.disconnected")

    async def submit(self, order: Order) -> None:
        """Queue the order. It fills on the next set_current_price() call."""
        order.status = OrderStateMachine.transition(order.status, OrderStatus.OPEN)
        self._pending_orders.append(order)
        logger.debug("simulator.order_queued", order_id=order.order_id, symbol=order.symbol)

    async def cancel(self, order_id: str) -> None:
        self._pending_orders = [o for o in self._pending_orders if o.order_id != order_id]

    async def set_current_price(self, symbol: str, price: float) -> None:
        """
        Called by BacktestRunner after each bar.
        Fills all pending orders for this symbol at the given price.
        """
        self._current_prices[symbol] = price
        to_fill = [o for o in self._pending_orders if o.symbol == symbol]

        for order in to_fill:
            self._pending_orders.remove(order)
            fill = Fill(
                order_id=order.order_id,
                symbol=order.symbol,
                side=order.side,
                fill_price=price,
                fill_quantity=order.quantity,
                strategy_id=order.strategy_id,
                timestamp=datetime.utcnow(),
            )
            await self.bus.publish(FillEvent(fill=fill))
            logger.debug(
                "simulator.fill",
                order_id=order.order_id,
                symbol=symbol,
                price=price,
                quantity=order.quantity,
            )
