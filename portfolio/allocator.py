"""
Portfolio allocator — converts strategy signals into sized orders.

Takes SignalEvents from multiple strategies and determines:
  1. How much capital to allocate to each strategy
  2. What position size to request for each signal
  3. Whether to send a new OrderRequestEvent or ignore the signal

Capital allocation is fixed-weight: each strategy gets a predefined
percentage of total capital (from StrategySettings.strategy_weights).
This is a simple but realistic starting point — real funds use
risk-parity, mean-variance optimization, etc.

Signal to order conversion:
  signal.direction > 0  → BUY (quantity based on available capital)
  signal.direction < 0  → SELL (close/short)
  signal.direction == 0 → close any existing position
"""
from __future__ import annotations

import math
from typing import Optional

import structlog

from config.settings import StrategySettings
from core.event_bus import EventBus
from core.events import OrderRequestEvent, SignalEvent
from core.models import Order, OrderSide, OrderType, Position
from oms.position_tracker import PositionTracker

logger = structlog.get_logger(__name__)


class PortfolioAllocator:
    def __init__(
        self,
        settings: StrategySettings,
        positions: PositionTracker,
        event_bus: EventBus,
        last_prices: dict[str, float],
    ) -> None:
        self.settings = settings
        self.positions = positions
        self.bus = event_bus
        self.last_prices = last_prices  # shared dict updated by data feed
        self._signal_queue = event_bus.subscribe(SignalEvent)

    async def run(self) -> None:
        """Process signals and emit order requests."""
        while True:
            event: SignalEvent = await self._signal_queue.get()
            order = self._signal_to_order(event)
            if order:
                await self.bus.publish(OrderRequestEvent(order=order))

    def _signal_to_order(self, signal: SignalEvent) -> Optional[Order]:
        """Convert a signal to a sized order request, or None if no action needed."""
        price = self.last_prices.get(signal.symbol)
        if price is None or price <= 0:
            logger.warning(
                "allocator.no_price",
                symbol=signal.symbol,
                strategy=signal.strategy_id,
            )
            return None

        # Capital allocated to this strategy
        weight = self.settings.strategy_weights.get(signal.strategy_id, 0.0)
        allocated_capital = self.settings.initial_capital * weight

        current_pos = self.positions.get(signal.symbol)
        current_qty = current_pos.quantity if current_pos else 0

        if signal.direction > 0:
            # Target: long position
            target_value = allocated_capital * signal.direction * signal.confidence
            target_qty = math.floor(target_value / price)
            needed_qty = target_qty - max(0, current_qty)
            if needed_qty <= 0:
                return None
            return Order(
                symbol=signal.symbol,
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=needed_qty,
                strategy_id=signal.strategy_id,
            )

        elif signal.direction < 0:
            # Target: short (or close long and go short)
            target_value = allocated_capital * abs(signal.direction) * signal.confidence
            target_qty = math.floor(target_value / price)
            # If long, close first then short
            if current_qty > 0:
                return Order(
                    symbol=signal.symbol,
                    side=OrderSide.SELL,
                    order_type=OrderType.MARKET,
                    quantity=current_qty,
                    strategy_id=signal.strategy_id,
                )
            needed_qty = target_qty - abs(min(0, current_qty))
            if needed_qty <= 0:
                return None
            return Order(
                symbol=signal.symbol,
                side=OrderSide.SELL,
                order_type=OrderType.MARKET,
                quantity=needed_qty,
                strategy_id=signal.strategy_id,
            )

        else:
            # direction == 0: close any existing position
            if current_qty > 0:
                return Order(
                    symbol=signal.symbol,
                    side=OrderSide.SELL,
                    order_type=OrderType.MARKET,
                    quantity=current_qty,
                    strategy_id=signal.strategy_id,
                )
            elif current_qty < 0:
                return Order(
                    symbol=signal.symbol,
                    side=OrderSide.BUY,
                    order_type=OrderType.MARKET,
                    quantity=abs(current_qty),
                    strategy_id=signal.strategy_id,
                )
            return None
