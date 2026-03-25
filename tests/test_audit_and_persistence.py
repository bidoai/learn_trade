"""
Tests for:
  1. EventStore position persistence (save_snapshot / load_snapshot)
  2. AuditSubscriber — events flow from bus → EventStore
  3. RiskEngine price lookup — uses last_prices before falling back to position avg
"""
from __future__ import annotations

import asyncio
import tempfile
import os
from datetime import datetime

import pytest

from audit.event_store import EventStore
from audit.subscriber import AuditSubscriber
from config.settings import RiskSettings
from core.event_bus import EventBus
from core.events import FillEvent, SignalEvent, OrderBlockedEvent
from core.models import Fill, Order, OrderSide, OrderType, Position
from oms.position_tracker import PositionTracker
from risk.engine import RiskEngine
# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_position(symbol="AAPL", quantity=100, avg_entry_price=150.0, strategy_id="test"):
    return Position(
        symbol=symbol,
        quantity=quantity,
        avg_entry_price=avg_entry_price,
        strategy_id=strategy_id,
    )


def make_order(symbol="AAPL", side=OrderSide.BUY, quantity=10):
    return Order(
        symbol=symbol,
        side=side,
        order_type=OrderType.MARKET,
        quantity=quantity,
        strategy_id="test",
    )


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "test_trading.db")


@pytest.fixture
def store(db_path):
    s = EventStore(db_path)
    s.initialize()
    yield s
    s.close()


# ---------------------------------------------------------------------------
# 1. Position persistence
# ---------------------------------------------------------------------------

class TestPositionPersistence:
    def test_save_and_load_roundtrip(self, store):
        positions = [
            make_position("AAPL", 100, 150.0),
            make_position("GOOGL", 50, 2800.0),
        ]
        store.save_snapshot(positions)

        loaded = store.load_snapshot()
        assert len(loaded) == 2

        symbols = {p.symbol for p in loaded}
        assert symbols == {"AAPL", "GOOGL"}

        aapl = next(p for p in loaded if p.symbol == "AAPL")
        assert aapl.quantity == 100
        assert aapl.avg_entry_price == 150.0

    def test_flat_positions_excluded_from_load(self, store):
        # Save a flat position (quantity=0) — should not be loaded back
        positions = [make_position("AAPL", 0, 0.0)]
        store.save_snapshot(positions)

        loaded = store.load_snapshot()
        assert loaded == []

    def test_upsert_updates_existing_position(self, store):
        store.save_snapshot([make_position("AAPL", 100, 150.0)])
        store.save_snapshot([make_position("AAPL", 200, 155.0)])

        loaded = store.load_snapshot()
        assert len(loaded) == 1
        assert loaded[0].quantity == 200
        assert loaded[0].avg_entry_price == 155.0

    def test_save_removes_flat_positions(self, store):
        # Position opens, then closes — DB should reflect flat state
        store.save_snapshot([make_position("AAPL", 100, 150.0)])
        store.save_snapshot([make_position("AAPL", 0, 0.0)])

        loaded = store.load_snapshot()
        assert loaded == []

    def test_load_empty_db_returns_empty_list(self, store):
        assert store.load_snapshot() == []

    def test_positions_restored_into_tracker(self, store):
        store.save_snapshot([make_position("AAPL", 100, 150.0)])

        tracker = PositionTracker()
        tracker.load_from_snapshot(store.load_snapshot())

        pos = tracker.get("AAPL")
        assert pos is not None
        assert pos.quantity == 100
        assert pos.avg_entry_price == 150.0


# ---------------------------------------------------------------------------
# 2. AuditSubscriber
# ---------------------------------------------------------------------------

class TestAuditSubscriber:
    @pytest.mark.asyncio
    async def test_fill_event_persisted(self, store):
        bus = EventBus()
        subscriber = AuditSubscriber(store, bus)
        task = asyncio.create_task(subscriber.run())

        fill = Fill(
            order_id="ord-1",
            symbol="AAPL",
            side=OrderSide.BUY,
            fill_price=150.0,
            fill_quantity=10,
            strategy_id="test",
            timestamp=datetime.utcnow(),
        )
        await bus.publish(FillEvent(fill=fill, timestamp=datetime.utcnow()))
        await asyncio.sleep(0.05)  # let drain tasks process

        events = store.replay(event_types=["FillEvent"])
        assert len(events) == 1
        assert events[0]["event_type"] == "FillEvent"

        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_signal_event_persisted(self, store):
        from core.events import SignalEvent
        bus = EventBus()
        subscriber = AuditSubscriber(store, bus)
        task = asyncio.create_task(subscriber.run())

        await bus.publish(SignalEvent(
            strategy_id="momentum",
            symbol="AAPL",
            direction=1.0,
            confidence=0.8,
            timestamp=datetime.utcnow(),
        ))
        await asyncio.sleep(0.05)

        events = store.replay(event_types=["SignalEvent"])
        assert len(events) == 1

        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_multiple_event_types_persisted(self, store):
        bus = EventBus()
        subscriber = AuditSubscriber(store, bus)
        task = asyncio.create_task(subscriber.run())

        fill = Fill(
            order_id="ord-2",
            symbol="MSFT",
            side=OrderSide.SELL,
            fill_price=300.0,
            fill_quantity=5,
            strategy_id="test",
            timestamp=datetime.utcnow(),
        )
        await bus.publish(FillEvent(fill=fill, timestamp=datetime.utcnow()))
        await bus.publish(SignalEvent(
            strategy_id="ml",
            symbol="MSFT",
            direction=-1.0,
            confidence=0.6,
            timestamp=datetime.utcnow(),
        ))
        await asyncio.sleep(0.05)

        all_events = store.replay()
        assert len(all_events) == 2

        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


# ---------------------------------------------------------------------------
# 3. RiskEngine price lookup
# ---------------------------------------------------------------------------

class TestRiskEnginePriceLookup:
    def _make_engine(self, last_prices=None):
        positions = PositionTracker()
        return RiskEngine(
            positions=positions,
            settings=RiskSettings(max_position_pct=0.10),
            initial_capital=100_000,
            last_prices=last_prices,
        ), positions

    def test_uses_last_prices_dict_when_available(self):
        last_prices = {"AAPL": 200.0}
        engine, _ = self._make_engine(last_prices=last_prices)
        assert engine._get_last_price("AAPL") == 200.0

    def test_falls_back_to_position_avg_when_no_market_price(self):
        engine, positions = self._make_engine(last_prices={})

        # Simulate an existing position with known avg entry price
        from core.models import Fill
        fill = Fill(
            order_id="o1",
            symbol="AAPL",
            side=OrderSide.BUY,
            fill_price=180.0,
            fill_quantity=10,
            strategy_id="test",
            timestamp=datetime.utcnow(),
        )
        positions.apply_fill(fill)

        assert engine._get_last_price("AAPL") == 180.0

    def test_last_prices_takes_priority_over_position_avg(self):
        last_prices = {"AAPL": 195.0}
        engine, positions = self._make_engine(last_prices=last_prices)

        from core.models import Fill
        fill = Fill(
            order_id="o2",
            symbol="AAPL",
            side=OrderSide.BUY,
            fill_price=150.0,
            fill_quantity=10,
            strategy_id="test",
            timestamp=datetime.utcnow(),
        )
        positions.apply_fill(fill)

        # Should use live market price, not stale entry price
        assert engine._get_last_price("AAPL") == 195.0

    def test_falls_back_to_one_dollar_when_no_data(self):
        engine, _ = self._make_engine(last_prices={})
        assert engine._get_last_price("UNKNOWN") == 1.0

    def test_risk_check_uses_real_price(self):
        """Order for 1000 shares of a $200 stock should breach position size limit."""
        last_prices = {"AAPL": 200.0}
        engine, _ = self._make_engine(last_prices=last_prices)

        order = make_order(symbol="AAPL", quantity=1000)
        result = engine.check(order)

        # 1000 * $200 = $200k > 10% of $100k ($10k limit) → blocked
        assert not result.approved
        assert "position_size_exceeded" in result.reason

    def test_without_last_prices_large_order_passes_incorrectly(self):
        """Demonstrates the old bug: without last_prices, $1 fallback lets large orders through."""
        engine, _ = self._make_engine(last_prices={})  # no price data

        order = make_order(symbol="AAPL", quantity=1000)
        result = engine.check(order)

        # 1000 * $1 (fallback) = $1k < $10k limit → incorrectly passes
        assert result.approved

    def test_last_prices_dict_is_live_reference(self):
        """last_prices updates are seen by RiskEngine without re-injection."""
        last_prices: dict = {}
        engine, _ = self._make_engine(last_prices=last_prices)

        assert engine._get_last_price("AAPL") == 1.0  # no price yet

        last_prices["AAPL"] = 175.0  # price arrives mid-session
        assert engine._get_last_price("AAPL") == 175.0
