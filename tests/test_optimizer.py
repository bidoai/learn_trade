"""
Tests for the strategy parameter optimizer.

Tests:
  - _run_backtest_sync: produces valid Sharpe/return for trending data
  - _split_bars: correct segment counts and coverage
  - StrategyOptimizer.run: walk-forward produces n_splits folds with valid results
  - Overfitting detection: IS Sharpe >= OOS Sharpe is at least representable
  - No data leakage: train bars never appear in test bars
"""
from __future__ import annotations

import pytest
import pytest_asyncio

from config.settings import Settings, StrategySettings, AlpacaSettings
from backtest.optimizer import (
    StrategyOptimizer,
    _run_backtest_sync,
    _split_bars,
)
from strategy.momentum import MomentumStrategy
from tests.fixtures.market_data import make_trending_bars, make_mean_reverting_bars


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def settings():
    return Settings(
        alpaca=AlpacaSettings(api_key="test", secret_key="test"),
        strategy=StrategySettings(initial_capital=100_000),
    )


@pytest.fixture
def trending_bars():
    return make_trending_bars(n=300, drift=0.002, seed=42)


@pytest.fixture
def flat_bars():
    return make_trending_bars(n=300, drift=0.0, seed=99)


def make_momentum(params: dict) -> MomentumStrategy:
    s = StrategySettings(
        momentum_lookback=params["lookback"],
        momentum_threshold=params["threshold"],
    )
    return MomentumStrategy(s)


PARAM_SPACE = {
    "lookback":  {"type": "int",   "low": 5,     "high": 30},
    "threshold": {"type": "float", "low": 0.005, "high": 0.04},
}


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

class TestRunBacktestSync:
    def test_returns_sharpe_and_return(self, trending_bars, settings):
        strategy = MomentumStrategy(StrategySettings(momentum_lookback=20, momentum_threshold=0.01))
        sharpe, ret = _run_backtest_sync(strategy, trending_bars, settings)
        assert sharpe is not None
        assert ret is not None
        assert isinstance(sharpe, float)
        assert isinstance(ret, float)

    def test_trending_data_positive_return(self, trending_bars, settings):
        strategy = MomentumStrategy(StrategySettings(momentum_lookback=10, momentum_threshold=0.005))
        _, ret = _run_backtest_sync(strategy, trending_bars, settings)
        # Strong uptrend → momentum should produce positive return
        assert ret > 0

    def test_resets_strategy_between_calls(self, trending_bars, settings):
        strategy = MomentumStrategy(StrategySettings(momentum_lookback=20, momentum_threshold=0.01))
        sharpe1, ret1 = _run_backtest_sync(strategy, trending_bars, settings)
        sharpe2, ret2 = _run_backtest_sync(strategy, trending_bars, settings)
        assert abs(sharpe1 - sharpe2) < 1e-9, "Results not deterministic — reset() not called"

    def test_too_few_bars_returns_none(self, settings):
        strategy = MomentumStrategy(StrategySettings(momentum_lookback=20, momentum_threshold=0.01))
        sharpe, ret = _run_backtest_sync(strategy, [], settings)
        assert sharpe is None
        assert ret is None


class TestSplitBars:
    def test_correct_number_of_segments(self, trending_bars):
        segments = _split_bars(trending_bars, 5)
        assert len(segments) == 5

    def test_all_bars_covered(self, trending_bars):
        segments = _split_bars(trending_bars, 4)
        total = sum(len(s) for s in segments)
        assert total == len(trending_bars)

    def test_no_overlap(self, trending_bars):
        segments = _split_bars(trending_bars, 4)
        all_bars = [b for s in segments for b in s]
        # Timestamps should be strictly increasing (no duplication)
        timestamps = [b.timestamp for b in all_bars]
        assert timestamps == sorted(timestamps)
        assert len(set(timestamps)) == len(timestamps)

    def test_segments_not_empty(self, trending_bars):
        segments = _split_bars(trending_bars, 5)
        for seg in segments:
            assert len(seg) > 0


# ---------------------------------------------------------------------------
# Integration test — runs a real (but fast) optimization
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_walk_forward_produces_correct_fold_count(trending_bars, settings, monkeypatch):
    """
    Full optimizer run with mocked data loader (no network).
    Verifies fold structure, no data leakage, and result integrity.
    Uses few trials (5) to keep the test fast.
    """
    from datetime import date
    import backtest.optimizer as opt_module

    # Patch HistoricalLoader.load to return our synthetic bars
    class MockLoader:
        def __init__(self, **kwargs): pass
        def load(self, symbol, start, end, timeframe="1Day"):
            return trending_bars

    monkeypatch.setattr(opt_module, "HistoricalLoader", MockLoader)

    optimizer = StrategyOptimizer(
        strategy_factory=make_momentum,
        param_space=PARAM_SPACE,
        symbol="AAPL",
        start=date(2020, 1, 1),
        end=date(2023, 12, 31),
        settings=settings,
        n_splits=3,
        n_trials=5,   # fast for tests
    )

    result = await optimizer.run()

    assert len(result.folds) == 3
    assert result.symbol == "AAPL"

    for fold in result.folds:
        assert fold.best_params, "best_params should not be empty"
        assert "lookback" in fold.best_params
        assert "threshold" in fold.best_params
        assert fold.in_sample_sharpe is not None
        # OOS dates should be after train dates
        assert fold.test_start > fold.train_end


@pytest.mark.asyncio
async def test_no_data_leakage_between_folds(trending_bars, settings, monkeypatch):
    """Train and test bars for each fold must not overlap."""
    from datetime import date
    import backtest.optimizer as opt_module

    class MockLoader:
        def __init__(self, **kwargs): pass
        def load(self, *args, **kwargs):
            return trending_bars

    monkeypatch.setattr(opt_module, "HistoricalLoader", MockLoader)

    optimizer = StrategyOptimizer(
        strategy_factory=make_momentum,
        param_space=PARAM_SPACE,
        symbol="AAPL",
        start=date(2020, 1, 1),
        end=date(2023, 12, 31),
        settings=settings,
        n_splits=3,
        n_trials=3,
    )

    result = await optimizer.run()

    for fold in result.folds:
        assert fold.train_end < fold.test_start, (
            f"Fold {fold.fold}: train_end={fold.train_end} >= test_start={fold.test_start}"
        )


@pytest.mark.asyncio
async def test_mean_oos_sharpe_is_aggregate(trending_bars, settings, monkeypatch):
    """mean_oos_sharpe should equal the mean of per-fold OOS Sharpes."""
    from datetime import date
    import numpy as np
    import backtest.optimizer as opt_module

    class MockLoader:
        def __init__(self, **kwargs): pass
        def load(self, *args, **kwargs):
            return trending_bars

    monkeypatch.setattr(opt_module, "HistoricalLoader", MockLoader)

    optimizer = StrategyOptimizer(
        strategy_factory=make_momentum,
        param_space=PARAM_SPACE,
        symbol="AAPL",
        start=date(2020, 1, 1),
        end=date(2023, 12, 31),
        settings=settings,
        n_splits=2,
        n_trials=3,
    )

    result = await optimizer.run()
    manual_mean = np.mean([f.out_of_sample_sharpe for f in result.folds if f.out_of_sample_sharpe is not None])
    assert abs(result.mean_oos_sharpe - manual_mean) < 1e-9
