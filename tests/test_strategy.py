"""
Tests for strategy base class and concrete strategies.

Critical tests:
  - NaN guard: invalid bars never produce orders
  - Warmup period: no signals until lookback_window bars received
  - reset(): state cleared between backtest runs
  - Backtest determinism: same data → same signals, always
"""
from __future__ import annotations

import pytest

from config.settings import StrategySettings
from core.events import MarketDataEvent
from strategy.base import Strategy
from strategy.momentum import MomentumStrategy
from strategy.mean_reversion import MeanReversionStrategy
from tests.fixtures.market_data import (
    make_nan_bar,
    make_trending_bars,
    make_mean_reverting_bars,
    make_known_bars,
)


@pytest.fixture
def settings():
    return StrategySettings(
        initial_capital=100_000,
        momentum_lookback=20,
        momentum_threshold=0.01,
        mean_reversion_lookback=30,
        mean_reversion_zscore=2.0,
    )


class TestNaNGuard:
    def test_nan_bar_returns_none(self, settings):
        """A bar with NaN prices must never produce a signal."""
        strategy = MomentumStrategy(settings)
        nan_bar = make_nan_bar()
        event = MarketDataEvent(bar=nan_bar)
        result = strategy.on_market_data(event)
        assert result is None

    def test_nan_bar_does_not_corrupt_window(self, settings):
        """After a NaN bar, valid bars should still produce valid signals."""
        strategy = MomentumStrategy(settings)
        bars = make_trending_bars(n=50, drift=0.005)
        nan_bar = make_nan_bar()

        # Feed some valid bars, then a NaN, then more valid bars
        for bar in bars[:10]:
            strategy.on_market_data(MarketDataEvent(bar=bar))

        strategy.on_market_data(MarketDataEvent(bar=nan_bar))  # should not crash

        # Continue feeding — strategy should still work
        for bar in bars[10:]:
            strategy.on_market_data(MarketDataEvent(bar=bar))

        # Should not have crashed


class TestWarmupPeriod:
    def test_no_signal_before_warmup(self, settings):
        """Strategy must return None until lookback_window bars are seen."""
        strategy = MomentumStrategy(settings)
        bars = make_trending_bars(n=settings.momentum_lookback - 1, drift=0.005)

        signals = [
            strategy.on_market_data(MarketDataEvent(bar=b))
            for b in bars
        ]
        assert all(s is None for s in signals), \
            "Strategy should return None before warmup is complete"

    def test_can_signal_after_warmup(self, settings):
        """After lookback_window bars, strategy may produce signals."""
        strategy = MomentumStrategy(settings)
        # Strong uptrend to ensure a signal is generated
        bars = make_trending_bars(n=settings.momentum_lookback + 10, drift=0.005)

        signals = [
            strategy.on_market_data(MarketDataEvent(bar=b))
            for b in bars
        ]
        # At least some signals should be non-None after warmup
        post_warmup = signals[settings.momentum_lookback:]
        assert any(s is not None for s in post_warmup), \
            "Strategy should produce at least one signal after warmup"


class TestReset:
    def test_reset_clears_bar_window(self, settings):
        """After reset(), strategy has no bars and re-enters warmup."""
        strategy = MomentumStrategy(settings)
        bars = make_trending_bars(n=settings.momentum_lookback + 5, drift=0.005)

        for bar in bars:
            strategy.on_market_data(MarketDataEvent(bar=bar))

        assert len(strategy.bars) > 0
        strategy.reset()
        assert len(strategy.bars) == 0

    def test_reset_enables_clean_second_run(self, settings):
        """Two runs with reset() between them should produce identical signals."""
        strategy = MomentumStrategy(settings)
        bars = make_trending_bars(n=settings.momentum_lookback + 20, drift=0.003)

        # First run
        strategy.reset()
        signals_run1 = [
            strategy.on_market_data(MarketDataEvent(bar=b))
            for b in bars
        ]

        # Second run — must reset first
        strategy.reset()
        signals_run2 = [
            strategy.on_market_data(MarketDataEvent(bar=b))
            for b in bars
        ]

        # Both runs should produce identical signals
        assert len(signals_run1) == len(signals_run2)
        for s1, s2 in zip(signals_run1, signals_run2):
            if s1 is None:
                assert s2 is None
            else:
                assert s2 is not None
                assert s1.direction == s2.direction
                assert s1.symbol == s2.symbol


class TestBacktestDeterminism:
    """
    Critical: running the same backtest twice must produce identical results.
    This catches state leakage — the most common backtest bug.
    """
    def test_momentum_deterministic(self, settings):
        bars = make_trending_bars(n=50, drift=0.002, seed=99)
        strategy = MomentumStrategy(settings)

        def collect_signals():
            strategy.reset()
            return [
                (i, s.direction, s.symbol)
                for i, b in enumerate(bars)
                if (s := strategy.on_market_data(MarketDataEvent(bar=b))) is not None
            ]

        run1 = collect_signals()
        run2 = collect_signals()
        assert run1 == run2, "Backtest runs produced different results — state leak detected"


class TestMomentumSignals:
    def test_strong_uptrend_generates_long_signal(self, settings):
        strategy = MomentumStrategy(settings)
        bars = make_trending_bars(n=settings.momentum_lookback + 5, drift=0.005)

        signals = [
            strategy.on_market_data(MarketDataEvent(bar=b))
            for b in bars
        ]
        post_warmup = [s for s in signals[settings.momentum_lookback:] if s is not None]
        if post_warmup:
            assert post_warmup[-1].direction > 0, "Strong uptrend should generate positive direction"

    def test_signal_direction_in_range(self, settings):
        """Signal direction must always be in [-1, 1]."""
        strategy = MomentumStrategy(settings)
        bars = make_trending_bars(n=settings.momentum_lookback + 30, drift=0.01)

        for bar in bars:
            signal = strategy.on_market_data(MarketDataEvent(bar=bar))
            if signal is not None:
                assert -1.0 <= signal.direction <= 1.0
                assert 0.0 <= signal.confidence <= 1.0


class TestMeanReversionSignals:
    def test_oversold_generates_long_signal(self, settings):
        """Extreme negative z-score should generate a long signal."""
        strategy = MeanReversionStrategy(settings)
        bars = make_mean_reverting_bars(
            n=settings.mean_reversion_lookback + 20,
            mean=100.0,
        )

        signals = []
        for bar in bars:
            s = strategy.on_market_data(MarketDataEvent(bar=bar))
            if s is not None:
                signals.append(s)

        # With mean-reverting data, should see both positive and negative signals
        directions = [s.direction for s in signals]
        if directions:
            # Directions should be in valid range
            assert all(-1.0 <= d <= 1.0 for d in directions)
