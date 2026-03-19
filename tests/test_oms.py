"""
Tests for OMS, order state machine, and position tracker.

Critical tests:
  - All valid state machine transitions succeed
  - All invalid transitions raise InvalidTransitionError
  - Duplicate orders rejected via idempotency key
  - Position tracker correctly averages entry price on multiple fills
"""
from __future__ import annotations

import asyncio
import pytest
from datetime import datetime

from core.event_bus import EventBus
from core.events import FillEvent, OrderBlockedEvent, OrderApprovedEvent, OrderRequestEvent
from core.models import Fill, Order, OrderSide, OrderStatus, OrderType, Position
from oms.order_manager import OrderManager
from oms.position_tracker import PositionTracker
from oms.state_machine import InvalidTransitionError, OrderStateMachine
from risk.engine import RiskEngine
from config.settings import RiskSettings


@pytest.fixture
def positions():
    return PositionTracker()


@pytest.fixture
def risk_engine(positions):
    return RiskEngine(
        positions=positions,
        settings=RiskSettings(max_position_pct=0.50),
        initial_capital=100_000,
    )


@pytest.fixture
def event_bus():
    return EventBus()


@pytest.fixture
def oms(risk_engine, positions, event_bus):
    return OrderManager(
        risk_engine=risk_engine,
        positions=positions,
        event_bus=event_bus,
    )


def make_order(**kwargs):
    defaults = dict(
        symbol="AAPL",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=10,
        strategy_id="test_strategy",
    )
    defaults.update(kwargs)
    return Order(**defaults)


class TestOrderStateMachine:
    def test_valid_transitions(self):
        """All allowed transitions should succeed."""
        valid_cases = [
            (OrderStatus.NEW, OrderStatus.PENDING_SUBMIT),
            (OrderStatus.NEW, OrderStatus.BLOCKED),
            (OrderStatus.PENDING_SUBMIT, OrderStatus.OPEN),
            (OrderStatus.PENDING_SUBMIT, OrderStatus.REJECTED),
            (OrderStatus.OPEN, OrderStatus.PARTIAL),
            (OrderStatus.OPEN, OrderStatus.FILLED),
            (OrderStatus.OPEN, OrderStatus.PENDING_CANCEL),
            (OrderStatus.OPEN, OrderStatus.EXPIRED),
            (OrderStatus.PARTIAL, OrderStatus.FILLED),
            (OrderStatus.PENDING_CANCEL, OrderStatus.CANCELLED),
            (OrderStatus.PENDING_CANCEL, OrderStatus.FILLED),
        ]
        for current, next_s in valid_cases:
            result = OrderStateMachine.transition(current, next_s)
            assert result == next_s, f"Transition {current} → {next_s} failed"

    def test_invalid_transitions_raise(self):
        """Invalid transitions must raise InvalidTransitionError immediately."""
        invalid_cases = [
            (OrderStatus.FILLED, OrderStatus.OPEN),
            (OrderStatus.FILLED, OrderStatus.PENDING_SUBMIT),
            (OrderStatus.BLOCKED, OrderStatus.OPEN),
            (OrderStatus.CANCELLED, OrderStatus.FILLED),
            (OrderStatus.NEW, OrderStatus.FILLED),
            (OrderStatus.EXPIRED, OrderStatus.OPEN),
        ]
        for current, next_s in invalid_cases:
            with pytest.raises(InvalidTransitionError):
                OrderStateMachine.transition(current, next_s)

    def test_terminal_states_have_no_transitions(self):
        terminal = [
            OrderStatus.FILLED, OrderStatus.BLOCKED, OrderStatus.CANCELLED,
            OrderStatus.EXPIRED, OrderStatus.REJECTED,
        ]
        for status in terminal:
            assert OrderStateMachine.is_terminal(status)


class TestPositionTracker:
    def test_new_position_from_fill(self, positions):
        fill = Fill(
            order_id="o1", symbol="AAPL", side=OrderSide.BUY,
            fill_price=150.0, fill_quantity=100, strategy_id="test",
        )
        pos = positions.apply_fill(fill)
        assert pos.symbol == "AAPL"
        assert pos.quantity == 100
        assert pos.avg_entry_price == 150.0

    def test_average_price_on_add_to_position(self, positions):
        """Buying more of a position should update avg price correctly."""
        fill1 = Fill(order_id="o1", symbol="AAPL", side=OrderSide.BUY,
                     fill_price=100.0, fill_quantity=100, strategy_id="test")
        fill2 = Fill(order_id="o2", symbol="AAPL", side=OrderSide.BUY,
                     fill_price=120.0, fill_quantity=100, strategy_id="test")

        positions.apply_fill(fill1)
        pos = positions.apply_fill(fill2)

        # avg = (100*100 + 120*100) / 200 = 110
        assert pos.avg_entry_price == 110.0
        assert pos.quantity == 200

    def test_position_closes_to_flat(self, positions):
        fill_buy = Fill(order_id="o1", symbol="AAPL", side=OrderSide.BUY,
                        fill_price=100.0, fill_quantity=100, strategy_id="test")
        fill_sell = Fill(order_id="o2", symbol="AAPL", side=OrderSide.SELL,
                         fill_price=110.0, fill_quantity=100, strategy_id="test")
        positions.apply_fill(fill_buy)
        pos = positions.apply_fill(fill_sell)

        assert pos.quantity == 0
        assert positions.is_flat("AAPL")

    def test_all_positions_excludes_flat(self, positions):
        fill = Fill(order_id="o1", symbol="AAPL", side=OrderSide.BUY,
                    fill_price=100.0, fill_quantity=100, strategy_id="test")
        positions.apply_fill(fill)
        fill_close = Fill(order_id="o2", symbol="AAPL", side=OrderSide.SELL,
                          fill_price=100.0, fill_quantity=100, strategy_id="test")
        positions.apply_fill(fill_close)

        assert positions.all_positions() == []


class TestIdempotency:
    @pytest.mark.asyncio
    async def test_duplicate_order_rejected(self, oms):
        """Same idempotency key should not result in two orders."""
        order = make_order()
        approved_count = 0

        approved_queue = oms.bus.subscribe(OrderApprovedEvent)
        blocked_queue = oms.bus.subscribe(OrderBlockedEvent)

        event = OrderRequestEvent(order=order)
        await oms._handle_order_request(event)

        # Try submitting the same order again (same idempotency key)
        await oms._handle_order_request(event)

        # Only one should have been approved
        approved = []
        while not approved_queue.empty():
            approved.append(approved_queue.get_nowait())

        assert len(approved) == 1, "Duplicate order should have been rejected"
