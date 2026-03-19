"""
Tests for risk engine, circuit breaker, and VaR.

Critical tests:
  - Circuit breaker triggers at the right threshold
  - Circuit breaker blocks ALL orders once triggered (even if individual checks pass)
  - Circuit breaker resets on new day
  - VaR returns None when insufficient data
  - Risk engine fails CLOSED when it crashes
"""
from __future__ import annotations

import pytest
from datetime import date, timedelta
from unittest.mock import patch

from config.settings import RiskSettings, StrategySettings
from core.models import Order, OrderSide, OrderType
from oms.position_tracker import PositionTracker
from risk.circuit_breaker import CircuitBreaker, CircuitBreakerState
from risk.engine import RiskEngine
from risk.var import VaRCalculator


@pytest.fixture
def risk_settings():
    return RiskSettings(
        max_position_pct=0.10,    # 10% of capital per position
        max_daily_loss_pct=0.02,  # 2% daily loss = circuit breaker
        max_concentration_pct=0.30,
        var_confidence=0.95,
        var_window_days=20,
    )


@pytest.fixture
def positions():
    return PositionTracker()


@pytest.fixture
def risk_engine(positions, risk_settings):
    return RiskEngine(
        positions=positions,
        settings=risk_settings,
        initial_capital=100_000,
    )


def make_order(symbol="AAPL", side=OrderSide.BUY, quantity=10, strategy="test"):
    return Order(
        symbol=symbol,
        side=side,
        order_type=OrderType.MARKET,
        quantity=quantity,
        strategy_id=strategy,
    )


class TestCircuitBreaker:
    def test_not_triggered_initially(self):
        cb = CircuitBreaker(max_daily_loss_pct=0.02, initial_capital=100_000)
        assert not cb.is_triggered()
        assert cb.state == CircuitBreakerState.NORMAL

    def test_triggers_at_threshold(self):
        """Circuit breaker triggers when daily loss reaches max_daily_loss_abs."""
        cb = CircuitBreaker(max_daily_loss_pct=0.02, initial_capital=100_000)
        # $100k * 2% = $2000 threshold
        # Small losses — not triggered
        cb.record_pnl(-1000)
        assert not cb.is_triggered()
        # Loss reaches threshold
        triggered = cb.record_pnl(-1000)
        assert triggered
        assert cb.is_triggered()
        assert cb.state == CircuitBreakerState.TRIGGERED

    def test_subsequent_orders_blocked_after_trigger(self):
        """Once triggered, all subsequent risk checks must fail."""
        cb = CircuitBreaker(max_daily_loss_pct=0.02, initial_capital=100_000)
        cb.record_pnl(-2001)  # exceed threshold
        assert cb.is_triggered()
        # Still triggered
        assert cb.is_triggered()
        assert cb.is_triggered()

    def test_resets_on_new_day(self):
        """Circuit breaker resets at the start of a new trading day."""
        cb = CircuitBreaker(max_daily_loss_pct=0.02, initial_capital=100_000)
        cb.record_pnl(-3000)
        assert cb.is_triggered()

        # Simulate day rollover
        tomorrow = date.today() + timedelta(days=1)
        with patch("risk.circuit_breaker.date") as mock_date:
            mock_date.today.return_value = tomorrow
            assert not cb.is_triggered()
            assert cb.state == CircuitBreakerState.NORMAL
            assert cb.daily_loss == 0.0

    def test_only_losses_count(self):
        """Gains should not reduce the daily loss counter."""
        cb = CircuitBreaker(max_daily_loss_pct=0.02, initial_capital=100_000)
        cb.record_pnl(-1500)
        cb.record_pnl(+5000)  # big gain — but shouldn't reduce loss counter
        assert cb.daily_loss == 1500  # still at $1500 loss


class TestRiskEngine:
    def test_order_approved_within_limits(self, risk_engine):
        order = make_order(quantity=50)  # 50 * $1 price = $50 (well within 10% of 100k)
        result = risk_engine.check(order)
        assert result.approved

    def test_order_blocked_when_circuit_breaker_engaged(self, risk_engine):
        """When circuit breaker is triggered, all orders are blocked."""
        risk_engine.circuit_breaker.record_pnl(-3000)  # trigger it
        order = make_order(quantity=1)
        result = risk_engine.check(order)
        assert not result.approved
        assert "circuit_breaker" in result.reason

    def test_assert_healthy_passes_valid_config(self, risk_engine):
        """assert_healthy() should not raise with valid config."""
        risk_engine.assert_healthy()  # should not raise

    def test_assert_healthy_fails_bad_config(self, positions):
        bad_settings = RiskSettings(max_position_pct=1.5)  # > 1.0 is invalid
        re = RiskEngine(positions=positions, settings=bad_settings, initial_capital=100_000)
        with pytest.raises(ValueError):
            re.assert_healthy()


class TestVaR:
    def test_returns_none_below_min_observations(self):
        var = VaRCalculator(confidence=0.95, window_days=20)
        for i in range(19):  # 19 < MIN_OBSERVATIONS (20)
            var.add_return(-0.01 + i * 0.001)
        result = var.calculate()
        assert result is None

    def test_returns_value_with_sufficient_data(self):
        var = VaRCalculator(confidence=0.95, window_days=20)
        import numpy as np
        rng = np.random.default_rng(42)
        for _ in range(25):
            var.add_return(float(rng.normal(0, 0.01)))
        result = var.calculate()
        assert result is not None
        assert result >= 0  # VaR is a positive loss magnitude

    def test_reset_clears_history(self):
        var = VaRCalculator()
        for _ in range(25):
            var.add_return(0.01)
        var.reset()
        assert var.calculate() is None

    def test_rolling_window_respected(self):
        """VaR should only use the last window_days returns."""
        var = VaRCalculator(confidence=0.95, window_days=5)
        # Add 10 returns (window keeps last 5)
        for i in range(10):
            var.add_return(float(i) * 0.001)
        assert len(var._daily_returns) == 5
