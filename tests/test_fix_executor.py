"""
Tests for the FIX 4.2 execution engine.

Tests:
  - Logon/Logout handshake completes correctly
  - Market order submit → FillEvent received
  - Limit order submit → FillEvent with correct price
  - Partial fill → two FillEvents, correct fill quantities
  - Cancel → order status transitions to CANCELLED
  - Rejected order → order status transitions to REJECTED, no FillEvent
  - Submit without connection → no crash
  - Cancel of unknown order → no crash
  - Multiple independent orders → each gets its own fill

All tests use FIXSimulator (no real FIX counterparty required).
"""
from __future__ import annotations

import asyncio
import pytest

from config.settings import FIXSettings, AlpacaSettings, Settings, StrategySettings
from core.event_bus import EventBus
from core.events import FillEvent
from core.models import Order, OrderSide, OrderStatus, OrderType
from execution.fix import FIXExecutor
from tests.fixtures.fix_simulator import FIXSimulator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def event_bus():
    return EventBus()


@pytest.fixture
def settings():
    return Settings(
        alpaca=AlpacaSettings(api_key="test", secret_key="test"),
        strategy=StrategySettings(initial_capital=100_000),
    )


def make_order(
    symbol: str = "AAPL",
    side: OrderSide = OrderSide.BUY,
    quantity: int = 100,
    order_type: OrderType = OrderType.MARKET,
    limit_price: float | None = None,
) -> Order:
    return Order(
        symbol=symbol,
        side=side,
        order_type=order_type,
        quantity=quantity,
        strategy_id="momentum",
        limit_price=limit_price,
        status=OrderStatus.PENDING_SUBMIT,  # OMS sets this before calling submit()
    )


async def start_executor(
    event_bus: EventBus,
    sim: FIXSimulator,
    heartbeat_interval: int = 30,
) -> FIXExecutor:
    fix_settings = FIXSettings(
        host="127.0.0.1",
        port=sim.port,
        sender_comp_id="CLIENT",
        target_comp_id="SERVER",
        heartbeat_interval=heartbeat_interval,
    )
    executor = FIXExecutor(settings=fix_settings, event_bus=event_bus)
    await executor.connect()
    return executor


# ---------------------------------------------------------------------------
# Session tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_logon_sets_connected_flag(event_bus):
    sim = FIXSimulator()
    await sim.start()
    executor = None
    try:
        executor = await start_executor(event_bus, sim)
        assert executor._connected is True
    finally:
        if executor:
            await executor.disconnect()
        await sim.stop()
    assert executor._connected is False


@pytest.mark.asyncio
async def test_connect_timeout_raises(event_bus):
    """Connecting to a port with no server should raise."""
    fix_settings = FIXSettings(host="127.0.0.1", port=19999)
    executor = FIXExecutor(settings=fix_settings, event_bus=event_bus)
    with pytest.raises((TimeoutError, ConnectionRefusedError, OSError)):
        await executor.connect()


# ---------------------------------------------------------------------------
# Market orders
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_market_order_produces_fill_event(event_bus):
    sim = FIXSimulator(fill_price=150.0)
    await sim.start()
    fill_queue = event_bus.subscribe(FillEvent)
    executor = None
    try:
        executor = await start_executor(event_bus, sim)
        order = make_order(quantity=50)
        await executor.submit(order)

        fill_event = await asyncio.wait_for(fill_queue.get(), timeout=2.0)

        assert fill_event.fill.order_id == order.order_id
        assert fill_event.fill.symbol == "AAPL"
        assert fill_event.fill.fill_quantity == 50
        assert fill_event.fill.fill_price == pytest.approx(150.0)
        assert fill_event.fill.side == OrderSide.BUY
    finally:
        if executor:
            await executor.disconnect()
        await sim.stop()


@pytest.mark.asyncio
async def test_market_order_transitions_to_filled(event_bus):
    sim = FIXSimulator(fill_price=200.0)
    await sim.start()
    fill_queue = event_bus.subscribe(FillEvent)
    executor = None
    try:
        executor = await start_executor(event_bus, sim)
        order = make_order(quantity=10)
        await executor.submit(order)

        await asyncio.wait_for(fill_queue.get(), timeout=2.0)
        assert order.status == OrderStatus.FILLED
    finally:
        if executor:
            await executor.disconnect()
        await sim.stop()


@pytest.mark.asyncio
async def test_sell_order_produces_fill_event(event_bus):
    sim = FIXSimulator(fill_price=175.0)
    await sim.start()
    fill_queue = event_bus.subscribe(FillEvent)
    executor = None
    try:
        executor = await start_executor(event_bus, sim)
        order = make_order(side=OrderSide.SELL, quantity=25)
        await executor.submit(order)

        fill_event = await asyncio.wait_for(fill_queue.get(), timeout=2.0)
        assert fill_event.fill.side == OrderSide.SELL
        assert fill_event.fill.fill_quantity == 25
    finally:
        if executor:
            await executor.disconnect()
        await sim.stop()


# ---------------------------------------------------------------------------
# Limit orders
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_limit_order_produces_fill_event(event_bus):
    sim = FIXSimulator(fill_price=149.99)
    await sim.start()
    fill_queue = event_bus.subscribe(FillEvent)
    executor = None
    try:
        executor = await start_executor(event_bus, sim)
        order = make_order(order_type=OrderType.LIMIT, limit_price=150.00, quantity=30)
        await executor.submit(order)

        fill_event = await asyncio.wait_for(fill_queue.get(), timeout=2.0)
        assert fill_event.fill.fill_price == pytest.approx(149.99)
        assert fill_event.fill.fill_quantity == 30
    finally:
        if executor:
            await executor.disconnect()
        await sim.stop()


# ---------------------------------------------------------------------------
# Partial fills
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_partial_fill_emits_two_fill_events(event_bus):
    """Simulator splits 100 into two fills of 50."""
    sim = FIXSimulator(fill_price=100.0, partial_fill=True)
    await sim.start()
    fill_queue = event_bus.subscribe(FillEvent)
    executor = None
    try:
        executor = await start_executor(event_bus, sim)
        order = make_order(quantity=100)
        await executor.submit(order)

        first  = await asyncio.wait_for(fill_queue.get(), timeout=2.0)
        second = await asyncio.wait_for(fill_queue.get(), timeout=2.0)

        assert first.fill.fill_quantity == 50
        assert second.fill.fill_quantity == 50
        assert order.status == OrderStatus.FILLED
    finally:
        if executor:
            await executor.disconnect()
        await sim.stop()


@pytest.mark.asyncio
async def test_partial_fill_total_quantity_correct(event_bus):
    """Total filled quantity across all fills equals order quantity."""
    sim = FIXSimulator(fill_price=100.0, partial_fill=True)
    await sim.start()
    fill_queue = event_bus.subscribe(FillEvent)
    executor = None
    try:
        executor = await start_executor(event_bus, sim)
        order = make_order(quantity=100)
        await executor.submit(order)

        fills = []
        for _ in range(2):
            fe = await asyncio.wait_for(fill_queue.get(), timeout=2.0)
            fills.append(fe.fill)

        assert sum(f.fill_quantity for f in fills) == 100
        assert all(f.order_id == order.order_id for f in fills)
    finally:
        if executor:
            await executor.disconnect()
        await sim.stop()


# ---------------------------------------------------------------------------
# Cancellation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_cancel_transitions_order_to_cancelled(event_bus):
    sim = FIXSimulator(fill_price=100.0)
    await sim.start()
    fill_queue = event_bus.subscribe(FillEvent)
    executor = None
    try:
        executor = await start_executor(event_bus, sim)

        # Confirm the session works first
        order = make_order(quantity=10)
        await executor.submit(order)
        await asyncio.wait_for(fill_queue.get(), timeout=2.0)

        # Inject a second order in OPEN state (skip submit — we're testing cancel path)
        order2 = make_order(quantity=20)
        order2.status = OrderStatus.OPEN
        executor._pending[order2.order_id] = order2

        await executor.cancel(order2.order_id)

        for _ in range(30):
            if order2.status == OrderStatus.CANCELLED:
                break
            await asyncio.sleep(0.1)

        assert order2.status == OrderStatus.CANCELLED
    finally:
        if executor:
            await executor.disconnect()
        await sim.stop()


@pytest.mark.asyncio
async def test_cancel_unknown_order_does_not_raise(event_bus):
    sim = FIXSimulator()
    await sim.start()
    executor = None
    try:
        executor = await start_executor(event_bus, sim)
        await executor.cancel("nonexistent-order-id")  # should log warning, not raise
    finally:
        if executor:
            await executor.disconnect()
        await sim.stop()


# ---------------------------------------------------------------------------
# Rejected orders
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_rejected_order_transitions_to_rejected(event_bus):
    sim = FIXSimulator(reject_next=True)
    await sim.start()
    fill_queue = event_bus.subscribe(FillEvent)
    executor = None
    try:
        executor = await start_executor(event_bus, sim)
        order = make_order(quantity=10)
        await executor.submit(order)

        for _ in range(20):
            if order.status == OrderStatus.REJECTED:
                break
            await asyncio.sleep(0.1)

        assert order.status == OrderStatus.REJECTED
        assert fill_queue.empty(), "No FillEvent should be emitted for rejected orders"
    finally:
        if executor:
            await executor.disconnect()
        await sim.stop()


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_submit_without_connect_does_not_raise(event_bus):
    fix_settings = FIXSettings(host="127.0.0.1", port=9999)
    executor = FIXExecutor(settings=fix_settings, event_bus=event_bus)
    order = make_order()
    await executor.submit(order)  # not connected — should log and return, not raise


@pytest.mark.asyncio
async def test_multiple_orders_independent(event_bus):
    """Three orders each get their own FillEvent with the correct order_id."""
    sim = FIXSimulator(fill_price=50.0)
    await sim.start()
    fill_queue = event_bus.subscribe(FillEvent)
    executor = None
    try:
        executor = await start_executor(event_bus, sim)

        orders = [make_order(quantity=10 * (i + 1)) for i in range(3)]
        for o in orders:
            await executor.submit(o)

        fills = []
        for _ in range(3):
            fe = await asyncio.wait_for(fill_queue.get(), timeout=3.0)
            fills.append(fe.fill)

        filled_order_ids = {f.order_id for f in fills}
        expected_order_ids = {o.order_id for o in orders}
        assert filled_order_ids == expected_order_ids
    finally:
        if executor:
            await executor.disconnect()
        await sim.stop()
