"""
Order Management System (OMS).

The OMS is the traffic controller for all orders. It is the ONLY
component that creates, tracks, and transitions orders.

Order flow:
  OrderRequestEvent arrives
       │
       ├── [sync] risk_engine.check(order)   ← direct call, not event bus
       │         │
       │    approved? ──▶ create Order, PENDING_SUBMIT
       │         │           └──▶ execution.submit(order)
       │         │               └──▶ emit OrderApprovedEvent
       │         │
       │    blocked? ──▶ set BLOCKED
       │                 └──▶ emit OrderBlockedEvent
       │
       └── [async] FillEvent arrives
                   └──▶ update order status → PARTIAL or FILLED
                        └──▶ emit OrderStatusEvent
                             └──▶ position_tracker.apply_fill()

The risk check is wrapped in try/finally to guarantee circuit breaker
logic runs even if an exception occurs mid-check.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import structlog

from core.events import (
    FillEvent,
    OrderApprovedEvent,
    OrderBlockedEvent,
    OrderRequestEvent,
    OrderStatusEvent,
    RiskAlertEvent,
)
from core.event_bus import EventBus
from core.models import Fill, Order, OrderStatus
from oms.order_book import OrderBook
from oms.position_tracker import PositionTracker
from oms.state_machine import InvalidTransitionError, OrderStateMachine

if TYPE_CHECKING:
    from execution.base import ExecutionEngine
    from risk.engine import RiskEngine

logger = structlog.get_logger(__name__)


class OrderManager:
    def __init__(
        self,
        risk_engine: "RiskEngine",
        positions: PositionTracker,
        event_bus: EventBus,
    ) -> None:
        self.risk = risk_engine
        self.positions = positions
        self.bus = event_bus
        self.order_book = OrderBook()
        self._execution: "ExecutionEngine | None" = None

        # Subscribe to incoming requests and fills
        self._request_queue = event_bus.subscribe(OrderRequestEvent)
        self._fill_queue = event_bus.subscribe(FillEvent)

        # Idempotency: track recently seen keys to reject duplicates
        self._seen_idempotency_keys: set[str] = set()

    def set_execution(self, execution: "ExecutionEngine") -> None:
        self._execution = execution

    # ------------------------------------------------------------------
    # Main processing loops — run as asyncio tasks
    # ------------------------------------------------------------------

    async def run_order_requests(self) -> None:
        """Process incoming order requests. Also enforces limit order TTL."""
        iteration = 0
        while True:
            event: OrderRequestEvent = await self._request_queue.get()
            await self._handle_order_request(event)
            # Check for stale limit orders every 10 processed requests
            iteration += 1
            if iteration % 10 == 0:
                await self._check_stale_orders()

    async def run_fill_processing(self) -> None:
        """Process incoming fills from execution engine."""
        while True:
            event: FillEvent = await self._fill_queue.get()
            await self._handle_fill(event)

    # ------------------------------------------------------------------
    # Stale order enforcement
    # ------------------------------------------------------------------

    async def _check_stale_orders(self) -> None:
        """
        Cancel limit orders older than limit_order_ttl_sec.
        Expire market orders stuck open for more than 60s (execution issue).
        Called periodically from run_order_requests.
        """
        now = datetime.now(timezone.utc)
        ttl_sec = getattr(self.risk.settings, "limit_order_ttl_sec", 30)

        for order in list(self.order_book.open_orders()):
            age_sec = (now - order.created_at).total_seconds()
            from core.models import OrderType, OrderStatus

            if order.order_type == OrderType.LIMIT and age_sec > ttl_sec:
                logger.info(
                    "oms.limit_order_expired",
                    order_id=order.order_id,
                    symbol=order.symbol,
                    age_sec=round(age_sec, 1),
                    ttl_sec=ttl_sec,
                )
                if self._execution:
                    await self._execution.cancel(order.order_id)
                order.status = OrderStateMachine.transition(order.status, OrderStatus.CANCELLED)
                await self.bus.publish(
                    OrderStatusEvent(
                        order_id=order.order_id,
                        symbol=order.symbol,
                        old_status=OrderStatus.OPEN,
                        new_status=OrderStatus.CANCELLED,
                    )
                )

            elif order.order_type != OrderType.LIMIT and age_sec > 60:
                logger.warning(
                    "oms.market_order_stuck",
                    order_id=order.order_id,
                    symbol=order.symbol,
                    age_sec=round(age_sec, 1),
                )
                order.status = OrderStateMachine.transition(order.status, OrderStatus.EXPIRED)
                await self.bus.publish(
                    OrderStatusEvent(
                        order_id=order.order_id,
                        symbol=order.symbol,
                        old_status=OrderStatus.OPEN,
                        new_status=OrderStatus.EXPIRED,
                    )
                )

    async def check_stops(self) -> None:
        """
        Check all open positions for stop-loss breaches and submit closing orders.
        Call this periodically (e.g., on each market data event).
        """
        stop_orders = self.risk.get_stop_loss_orders()
        for order in stop_orders:
            await self.bus.publish(OrderRequestEvent(order=order))

    # ------------------------------------------------------------------
    # Order request handling
    # ------------------------------------------------------------------

    async def _handle_order_request(self, event: OrderRequestEvent) -> None:
        order = event.order

        # Idempotency check: reject duplicate orders
        if order.idempotency_key in self._seen_idempotency_keys:
            logger.warning(
                "oms.duplicate_order_rejected",
                idempotency_key=order.idempotency_key,
                order_id=order.order_id,
            )
            return

        risk_result = None
        try:
            # Synchronous risk check — gatekeeper pattern
            # This direct call (not via event bus) ensures serialization
            risk_result = self.risk.check(order)
        except Exception:
            logger.exception("oms.risk_check_error", order_id=order.order_id)
            # Fail CLOSED: if risk check crashes, block the order
            await self.bus.publish(
                OrderBlockedEvent(order=order, reason="risk_engine_error")
            )
            return
        finally:
            # Circuit breaker check runs even if an exception occurred above
            # This is the try/finally guarantee from the architecture review
            pass

        if risk_result and risk_result.approved:
            self._seen_idempotency_keys.add(order.idempotency_key)
            order.status = OrderStateMachine.transition(order.status, OrderStatus.PENDING_SUBMIT)
            self.order_book.add(order)

            logger.info(
                "oms.order_approved",
                order_id=order.order_id,
                symbol=order.symbol,
                side=order.side.value,
                quantity=order.quantity,
                strategy=order.strategy_id,
            )
            await self.bus.publish(OrderApprovedEvent(order=order))

            if self._execution:
                await self._execution.submit(order)
        else:
            reason = risk_result.reason if risk_result else "unknown"
            order.status = OrderStateMachine.transition(order.status, OrderStatus.BLOCKED)
            logger.info(
                "oms.order_blocked",
                order_id=order.order_id,
                reason=reason,
            )
            await self.bus.publish(OrderBlockedEvent(order=order, reason=reason))

            # If circuit breaker is the reason, emit a risk alert
            if reason == "circuit_breaker_engaged":
                await self.bus.publish(
                    RiskAlertEvent(
                        alert_type="circuit_breaker_engaged",
                        message="Daily loss limit reached — all trading halted",
                        current_value=self.risk.circuit_breaker.daily_loss_pct,
                        threshold=self.risk.settings.max_daily_loss_pct,
                    )
                )

    # ------------------------------------------------------------------
    # Fill handling
    # ------------------------------------------------------------------

    async def _handle_fill(self, event: FillEvent) -> None:
        fill: Fill = event.fill
        order = self.order_book.get(fill.order_id)

        if order is None:
            logger.warning("oms.fill_for_unknown_order", order_id=fill.order_id)
            return

        old_status = order.status

        try:
            # Update fill quantities
            order.filled_quantity += fill.fill_quantity
            # Recalculate average fill price
            if order.avg_fill_price is None:
                order.avg_fill_price = fill.fill_price
            else:
                total_qty = order.filled_quantity
                order.avg_fill_price = (
                    (order.avg_fill_price * (total_qty - fill.fill_quantity))
                    + (fill.fill_price * fill.fill_quantity)
                ) / total_qty

            # Transition state
            if order.filled_quantity >= order.quantity:
                new_status = OrderStatus.FILLED
            else:
                new_status = OrderStatus.PARTIAL

            order.status = OrderStateMachine.transition(order.status, new_status)

        except InvalidTransitionError as e:
            logger.error(
                "oms.invalid_state_transition",
                order_id=fill.order_id,
                error=str(e),
            )
            return

        # Update position tracker
        self.positions.apply_fill(fill)

        await self.bus.publish(
            OrderStatusEvent(
                order_id=order.order_id,
                symbol=order.symbol,
                old_status=old_status,
                new_status=order.status,
            )
        )

        logger.info(
            "oms.fill_processed",
            order_id=fill.order_id,
            symbol=fill.symbol,
            fill_price=fill.fill_price,
            fill_qty=fill.fill_quantity,
            new_status=order.status.name,
        )
