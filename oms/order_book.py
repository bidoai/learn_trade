"""Order book — tracks all active and completed orders in memory."""
from __future__ import annotations

from typing import Optional

from core.models import Order, OrderStatus


class OrderBook:
    def __init__(self) -> None:
        self._orders: dict[str, Order] = {}  # order_id → Order

    def add(self, order: Order) -> None:
        self._orders[order.order_id] = order

    def get(self, order_id: str) -> Optional[Order]:
        return self._orders.get(order_id)

    def open_orders(self) -> list[Order]:
        return [
            o for o in self._orders.values()
            if o.status in (OrderStatus.PENDING_SUBMIT, OrderStatus.OPEN, OrderStatus.PARTIAL)
        ]

    def all_orders(self) -> list[Order]:
        return list(self._orders.values())
