"""Tests for strategy/delta_hedge.py"""
from __future__ import annotations

import asyncio
from datetime import date, datetime, timedelta

import pytest

from config.settings import StrategySettings
from core.event_bus import EventBus
from core.events import GreeksEvent, MarketDataEvent
from core.models import OptionContract, OptionType
from strategy.delta_hedge import DeltaHedgeStrategy
from tests.fixtures.market_data import make_trending_bars


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def settings():
    return StrategySettings()


@pytest.fixture
def expiry():
    # 90 days out
    return date.today() + timedelta(days=90)


@pytest.fixture
def option(expiry):
    return OptionContract(
        symbol="AAPL",
        option_type=OptionType.CALL,
        strike=100.0,
        expiry=expiry,
        contracts=1,
        entry_price=5.0,
    )


@pytest.fixture
def event_bus():
    return EventBus()


def _make_events(n: int, start_price: float = 100.0, symbol: str = "AAPL") -> list[MarketDataEvent]:
    bars = make_trending_bars(n=n, start_price=start_price, symbol=symbol, seed=42)
    return [MarketDataEvent(bar=bar) for bar in bars]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_no_signal_before_warmup(settings, option):
    vol_lookback = 5
    strategy = DeltaHedgeStrategy(option, settings, vol_lookback=vol_lookback)
    events = _make_events(vol_lookback)  # exactly vol_lookback bars — deque fills at this point

    # Send vol_lookback - 1 bars: should all return None (not enough for warmup)
    for event in events[: vol_lookback - 1]:
        result = strategy.on_market_data(event)
        assert result is None


def test_generates_signal_after_warmup(settings, option):
    """Strategy should generate a signal once vol_lookback bars are accumulated."""
    vol_lookback = 5
    strategy = DeltaHedgeStrategy(option, settings, vol_lookback=vol_lookback)
    events = _make_events(vol_lookback + 5)

    signal = None
    for event in events:
        result = strategy.on_market_data(event)
        if result is not None:
            signal = result
            break

    assert signal is not None
    assert signal.strategy_id == "delta_hedge"
    assert signal.symbol == "AAPL"
    assert -1.0 <= signal.direction <= 1.0
    assert 0.0 <= signal.confidence <= 1.0


def test_signal_direction_buy_when_need_more_shares(settings, expiry):
    """
    For a long call, delta increases as price rises.
    The hedge is -delta × contracts × 100. When delta increases (price rises),
    target_shares becomes more negative (sell). But starting from 0, the first
    signal after warmup adjusts from 0 to target, so direction depends on whether
    target > 0 or < 0. For a long call, delta > 0 so target = -delta × n < 0,
    meaning we need to short the underlying (direction = -1).
    """
    option = OptionContract(
        symbol="AAPL",
        option_type=OptionType.CALL,
        strike=100.0,
        expiry=expiry,
        contracts=1,
        entry_price=5.0,
    )
    vol_lookback = 5
    strategy = DeltaHedgeStrategy(option, settings, vol_lookback=vol_lookback)

    # Rising prices → call delta > 0 → hedge target < 0 → SELL signal
    events = _make_events(vol_lookback + 5, start_price=100.0)

    signal = None
    for event in events:
        result = strategy.on_market_data(event)
        if result is not None:
            signal = result
            break

    assert signal is not None
    # For a call with positive delta, we need to short → direction should be -1
    assert signal.direction == -1.0


def test_reset_clears_bar_history(settings, option):
    """reset() should clear accumulated bars and reset hedge."""
    vol_lookback = 5
    strategy = DeltaHedgeStrategy(option, settings, vol_lookback=vol_lookback)
    events = _make_events(vol_lookback + 2)

    for event in events:
        strategy.on_market_data(event)

    assert len(strategy.bars) > 0

    strategy.reset()
    assert len(strategy.bars) == 0
    assert strategy._current_hedge == 0.0


def test_greeks_event_published(settings, option):
    """After warmup, a GreeksEvent should be published on each bar."""
    vol_lookback = 5
    bus = EventBus()
    strategy = DeltaHedgeStrategy(option, settings, vol_lookback=vol_lookback, event_bus=bus)

    events = _make_events(vol_lookback + 3)

    async def run():
        queue = bus.subscribe(GreeksEvent)
        for event in events:
            strategy.on_market_data(event)
        # Give tasks a chance to execute
        await asyncio.sleep(0)
        published = []
        while not queue.empty():
            published.append(queue.get_nowait())
        return published

    published = asyncio.get_event_loop().run_until_complete(run())
    assert len(published) > 0
    greeks = published[0]
    assert isinstance(greeks, GreeksEvent)
    assert greeks.symbol == "AAPL"
    assert isinstance(greeks.delta, float)
    assert isinstance(greeks.gamma, float)
    assert isinstance(greeks.vega, float)
    assert isinstance(greeks.theta, float)
    assert isinstance(greeks.rho, float)


def test_option_contract_properties(expiry):
    """OptionContract multiplier and notional computed correctly."""
    option = OptionContract(
        symbol="AAPL",
        option_type=OptionType.CALL,
        strike=150.0,
        expiry=expiry,
        contracts=2,
        entry_price=3.50,
    )
    assert option.multiplier == 100
    assert abs(option.notional - 700.0) < 1e-8  # 3.50 × 2 × 100


def test_put_option_delta_hedges_opposite(settings, expiry):
    """For a long put, delta < 0, so hedge target = -delta × n > 0 (buy)."""
    option = OptionContract(
        symbol="AAPL",
        option_type=OptionType.PUT,
        strike=100.0,
        expiry=expiry,
        contracts=1,
        entry_price=5.0,
    )
    vol_lookback = 5
    strategy = DeltaHedgeStrategy(option, settings, vol_lookback=vol_lookback)
    events = _make_events(vol_lookback + 5, start_price=100.0)

    signal = None
    for event in events:
        result = strategy.on_market_data(event)
        if result is not None:
            signal = result
            break

    assert signal is not None
    # Put delta < 0 → target = -delta × n > 0 → direction = +1 (BUY)
    assert signal.direction == 1.0
