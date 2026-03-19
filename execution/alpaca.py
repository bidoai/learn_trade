"""
Alpaca paper trading executor.

Connects to Alpaca's paper trading API to submit real orders
with fake money. This gives realistic fill behavior, real market
prices, and real order book dynamics.

Why Alpaca for a learning system:
  - Free paper trading account
  - WebSocket market data (real-time)
  - REST API for orders (simple, well-documented)
  - Python SDK (alpaca-py)

Order submission flow:
  OMS calls submit(order)
    └─▶ Alpaca REST: POST /v2/orders
        └─▶ Alpaca WebSocket: order update event arrives
            └─▶ emit FillEvent → OMS processes fill
"""
from __future__ import annotations

import asyncio
import json
from typing import Optional

import structlog

from config.settings import AlpacaSettings
from core.event_bus import EventBus
from core.events import FillEvent, SystemEvent
from core.models import Fill, Order, OrderSide, OrderStatus, OrderType
from execution.base import ExecutionEngine
from oms.state_machine import OrderStateMachine

logger = structlog.get_logger(__name__)

_RECONNECT_DELAYS = [1, 2, 4, 8, 16, 30]  # exponential backoff (seconds)


class AlpacaExecutor(ExecutionEngine):
    def __init__(self, settings: AlpacaSettings, event_bus: EventBus) -> None:
        self.settings = settings
        self.bus = event_bus
        self._trading_client = None
        self._stream_client = None
        self._connected = False

    async def connect(self) -> None:
        """
        Connect to Alpaca paper trading API.
        Raises on auth failure (fail-fast during startup).
        """
        try:
            from alpaca.trading.client import TradingClient
            from alpaca.trading.stream import TradingStream

            self._trading_client = TradingClient(
                api_key=self.settings.api_key,
                secret_key=self.settings.secret_key,
                paper=True,
            )
            # Verify credentials by fetching account
            account = self._trading_client.get_account()
            logger.info(
                "alpaca.connected",
                buying_power=account.buying_power,
                portfolio_value=account.portfolio_value,
            )

            # Subscribe to order updates via WebSocket
            self._stream_client = TradingStream(
                api_key=self.settings.api_key,
                secret_key=self.settings.secret_key,
                paper=True,
            )
            self._stream_client.subscribe_trade_updates(self._on_trade_update)
            asyncio.create_task(self._run_stream())

            self._connected = True
            await self.bus.publish(SystemEvent(event_type="feed_connected", message="Alpaca executor connected"))

        except Exception as e:
            logger.error("alpaca.connect_failed", error=str(e))
            raise

    async def submit(self, order: Order) -> None:
        """Submit order to Alpaca REST API."""
        if not self._connected or self._trading_client is None:
            logger.error("alpaca.submit_without_connection", order_id=order.order_id)
            return

        try:
            from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
            from alpaca.trading.enums import OrderSide as AlpacaSide, TimeInForce

            side = AlpacaSide.BUY if order.side == OrderSide.BUY else AlpacaSide.SELL

            if order.order_type == OrderType.MARKET:
                request = MarketOrderRequest(
                    symbol=order.symbol,
                    qty=order.quantity,
                    side=side,
                    time_in_force=TimeInForce.DAY,
                    client_order_id=order.order_id,  # our ID for tracking
                )
            else:
                request = LimitOrderRequest(
                    symbol=order.symbol,
                    qty=order.quantity,
                    side=side,
                    time_in_force=TimeInForce.DAY,
                    limit_price=order.limit_price,
                    client_order_id=order.order_id,
                )

            alpaca_order = self._trading_client.submit_order(request)
            order.broker_order_id = str(alpaca_order.id)
            order.status = OrderStateMachine.transition(order.status, OrderStatus.OPEN)

            logger.info(
                "alpaca.order_submitted",
                order_id=order.order_id,
                broker_id=order.broker_order_id,
                symbol=order.symbol,
                side=order.side.value,
                quantity=order.quantity,
            )

        except Exception as e:
            logger.error(
                "alpaca.submit_failed",
                order_id=order.order_id,
                error=str(e),
            )
            order.status = OrderStateMachine.transition(order.status, OrderStatus.REJECTED)

    async def cancel(self, order_id: str) -> None:
        if self._trading_client and order_id:
            try:
                self._trading_client.cancel_order_by_id(order_id)
            except Exception as e:
                logger.warning("alpaca.cancel_failed", order_id=order_id, error=str(e))

    async def disconnect(self) -> None:
        self._connected = False
        if self._stream_client:
            await self._stream_client.stop_ws()
        logger.info("alpaca.disconnected")

    # ------------------------------------------------------------------
    # Internal: handle trade updates from Alpaca WebSocket
    # ------------------------------------------------------------------

    async def _on_trade_update(self, data) -> None:
        """
        Called by Alpaca WebSocket stream on order updates.
        Emits FillEvent when a fill arrives.
        """
        try:
            event = data.event
            order_data = data.order

            if event not in ("fill", "partial_fill"):
                return

            fill = Fill(
                order_id=str(order_data.client_order_id),  # our ID
                symbol=str(order_data.symbol),
                side=OrderSide.BUY if str(order_data.side) == "buy" else OrderSide.SELL,
                fill_price=float(data.price),
                fill_quantity=int(data.qty),
                strategy_id="",  # OMS will look up from order_id
                broker_fill_id=str(order_data.id),
            )
            await self.bus.publish(FillEvent(fill=fill))

        except Exception:
            logger.exception("alpaca.trade_update_error")

    async def _run_stream(self) -> None:
        """Run the Alpaca WebSocket stream with reconnect logic."""
        attempt = 0
        while True:
            try:
                await self._stream_client._run_forever()
                attempt = 0
            except Exception as e:
                delay = _RECONNECT_DELAYS[min(attempt, len(_RECONNECT_DELAYS) - 1)]
                logger.warning(
                    "alpaca.stream_disconnected",
                    error=str(e),
                    reconnect_in_sec=delay,
                )
                await self.bus.publish(SystemEvent(
                    event_type="reconnecting",
                    message=f"Alpaca stream reconnecting in {delay}s",
                ))
                await asyncio.sleep(delay)
                attempt += 1
