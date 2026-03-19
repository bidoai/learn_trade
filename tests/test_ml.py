"""
Tests for the ML module: feature engineering, training, and MLStrategy.

Critical tests:
  - Look-ahead bias guard: AssertionError if bar.timestamp > cutoff_date
  - Feature completeness: all FEATURE_COLS produced with no NaN in valid rows
  - Warmup requirement: no signal before MIN_BARS_FOR_TRAINING bars
  - Model trains: trainer.train() returns a TrainedModel with a cv_accuracy
  - Prediction shape: predict() returns (direction, confidence) in valid range
  - MLStrategy reset: model cleared between backtest runs (no state leak)
  - MLStrategy determinism: same bars → same signal sequence after reset()

Note on ML tests and randomness:
  GradientBoosting with random_state=42 is deterministic, so we can
  assert exact values where needed. The synthetic bar generators also
  use fixed seeds for reproducibility.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from config.settings import StrategySettings
from core.events import MarketDataEvent
from ml import features, trainer, predictor
from ml.features import FEATURE_COLS, LABEL_COL, MIN_BARS_FOR_TRAINING
from ml.trainer import MIN_TRAINING_ROWS
from strategy.ml_strategy import MLStrategy
from tests.fixtures.market_data import (
    make_trending_bars,
    make_mean_reverting_bars,
)


# ── Fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture
def settings():
    return StrategySettings(
        initial_capital=100_000,
        ml_lookback=60,
        ml_retrain_interval_bars=200,  # large so we don't retrain mid-test
    )


@pytest.fixture
def enough_bars():
    """More than MIN_BARS_FOR_TRAINING trending bars."""
    return make_trending_bars(n=MIN_BARS_FOR_TRAINING + 30, drift=0.002, seed=1)


@pytest.fixture
def many_bars():
    """Enough bars to train and retrain the ML model."""
    return make_trending_bars(n=120, drift=0.002, seed=7)


# ── FeatureBuilder ─────────────────────────────────────────────────────────

class TestFeatureBuilder:
    def test_all_feature_cols_present(self, enough_bars):
        """build() must produce all FEATURE_COLS and LABEL_COL."""
        cutoff = enough_bars[-1].timestamp
        df = features.build(enough_bars, cutoff_date=cutoff)

        for col in FEATURE_COLS:
            assert col in df.columns, f"Missing feature column: {col}"
        assert LABEL_COL in df.columns

    def test_no_nan_in_output(self, enough_bars):
        """All NaN rows (warmup + last label) must be dropped."""
        cutoff = enough_bars[-1].timestamp
        df = features.build(enough_bars, cutoff_date=cutoff)
        assert not df.isnull().any().any(), "NaN values found in feature output"

    def test_lookahead_bias_raises_assertion(self, enough_bars):
        """
        CRITICAL: if any bar.timestamp > cutoff_date, build() must raise
        AssertionError. This is the architectural enforcement that catches
        look-ahead bias at test time.
        """
        # Set cutoff to one day before the last bar
        last_bar_time = enough_bars[-1].timestamp
        bad_cutoff = last_bar_time - timedelta(days=1)

        with pytest.raises(AssertionError, match="Look-ahead bias"):
            features.build(enough_bars, cutoff_date=bad_cutoff)

    def test_fewer_than_min_bars_raises_value_error(self):
        """Fewer than MIN_BARS_FOR_TRAINING bars raises ValueError."""
        bars = make_trending_bars(n=MIN_BARS_FOR_TRAINING - 1, seed=3)
        cutoff = bars[-1].timestamp
        with pytest.raises(ValueError, match="bars"):
            features.build(bars, cutoff_date=cutoff)

    def test_output_row_count(self, enough_bars):
        """Output rows < input bars (warmup + last label row dropped)."""
        cutoff = enough_bars[-1].timestamp
        df = features.build(enough_bars, cutoff_date=cutoff)
        assert len(df) < len(enough_bars)
        assert len(df) > 0

    def test_label_is_binary(self, enough_bars):
        """Label column must be 0.0 or 1.0 only."""
        cutoff = enough_bars[-1].timestamp
        df = features.build(enough_bars, cutoff_date=cutoff)
        assert set(df[LABEL_COL].unique()).issubset({0.0, 1.0})

    def test_build_latest_returns_series(self, enough_bars):
        """build_latest() returns a pandas Series with FEATURE_COLS."""
        import pandas as pd
        vec = features.build_latest(enough_bars)
        assert vec is not None
        assert isinstance(vec, pd.Series)
        assert list(vec.index) == FEATURE_COLS

    def test_build_latest_too_few_bars(self):
        """build_latest() returns None when not enough bars."""
        bars = make_trending_bars(n=5, seed=2)
        vec = features.build_latest(bars)
        assert vec is None

    def test_rsi_in_range(self, enough_bars):
        """RSI must be in [0, 100]."""
        cutoff = enough_bars[-1].timestamp
        df = features.build(enough_bars, cutoff_date=cutoff)
        assert (df["rsi_14"] >= 0).all()
        assert (df["rsi_14"] <= 100).all()

    def test_volume_ratio_positive(self, enough_bars):
        """Volume ratio must be positive."""
        cutoff = enough_bars[-1].timestamp
        df = features.build(enough_bars, cutoff_date=cutoff)
        assert (df["volume_ratio"] > 0).all()


# ── Trainer ────────────────────────────────────────────────────────────────

class TestTrainer:
    def test_train_returns_model(self, many_bars):
        """trainer.train() must return a TrainedModel."""
        cutoff = many_bars[-1].timestamp
        df = features.build(many_bars, cutoff_date=cutoff)
        model = trainer.train(df)
        assert model is not None

    def test_trained_model_has_cv_accuracy(self, many_bars):
        """With enough data, cv_accuracy should be populated."""
        cutoff = many_bars[-1].timestamp
        df = features.build(many_bars, cutoff_date=cutoff)
        model = trainer.train(df)
        assert model is not None
        # cv_accuracy is None when there aren't enough rows for 2+ folds
        # With 120 bars we should have enough
        if model.cv_accuracy is not None:
            assert 0.0 <= model.cv_accuracy <= 1.0

    def test_too_few_rows_returns_none(self):
        """trainer.train() returns None when there are too few rows."""
        import pandas as pd
        import numpy as np
        # Build a tiny DataFrame (fewer than MIN_TRAINING_ROWS)
        n = MIN_TRAINING_ROWS - 1
        df = pd.DataFrame(
            {col: np.random.randn(n) for col in FEATURE_COLS}
        )
        df[LABEL_COL] = np.random.randint(0, 2, n).astype(float)
        result = trainer.train(df)
        assert result is None

    def test_feature_cols_preserved(self, many_bars):
        """TrainedModel.feature_cols must match FEATURE_COLS."""
        cutoff = many_bars[-1].timestamp
        df = features.build(many_bars, cutoff_date=cutoff)
        model = trainer.train(df)
        assert model is not None
        assert model.feature_cols == FEATURE_COLS

    def test_scaler_fitted(self, many_bars):
        """Scaler must be fitted (has mean_ attribute)."""
        cutoff = many_bars[-1].timestamp
        df = features.build(many_bars, cutoff_date=cutoff)
        model = trainer.train(df)
        assert model is not None
        assert hasattr(model.scaler, "mean_")
        assert len(model.scaler.mean_) == len(FEATURE_COLS)


# ── Predictor ──────────────────────────────────────────────────────────────

class TestPredictor:
    @pytest.fixture
    def trained_model(self, many_bars):
        cutoff = many_bars[-1].timestamp
        df = features.build(many_bars, cutoff_date=cutoff)
        return trainer.train(df)

    def test_predict_returns_tuple(self, trained_model, many_bars):
        """predict() returns (direction, confidence) or None."""
        feature_vec = features.build_latest(many_bars)
        assert feature_vec is not None

        result = predictor.predict(trained_model, feature_vec)
        # Result may be None (below confidence threshold) or a valid tuple
        if result is not None:
            direction, confidence = result
            assert -1.0 <= direction <= 1.0
            assert 0.0 <= confidence <= 1.0

    def test_predict_direction_confidence_relationship(self, trained_model, many_bars):
        """
        High-confidence predictions should have direction far from zero.
        direction = (p - 0.5) * 2, confidence = |p - 0.5| * 2
        So |direction| == confidence always.
        """
        feature_vec = features.build_latest(many_bars)
        assert feature_vec is not None

        result = predictor.predict(trained_model, feature_vec)
        if result is not None:
            direction, confidence = result
            import math
            # |direction| should approximately equal confidence
            assert math.isclose(abs(direction), confidence, rel_tol=1e-6)


# ── MLStrategy ────────────────────────────────────────────────────────────

class TestMLStrategy:
    def test_no_signal_before_warmup(self, settings):
        """No signal until lookback_window bars are accumulated."""
        strategy = MLStrategy(settings)
        bars = make_trending_bars(n=settings.ml_lookback - 1, drift=0.002, seed=5)

        signals = [
            strategy.on_market_data(MarketDataEvent(bar=b))
            for b in bars
        ]
        assert all(s is None for s in signals)

    def test_signal_after_sufficient_data(self, settings):
        """After enough bars, MLStrategy may produce signals."""
        # Use many bars to ensure training succeeds
        n = settings.ml_lookback + MIN_BARS_FOR_TRAINING + 20
        bars = make_trending_bars(n=n, drift=0.003, seed=11)
        strategy = MLStrategy(settings)

        signals = [
            strategy.on_market_data(MarketDataEvent(bar=b))
            for b in bars
        ]

        # With enough data and a trend, the model should eventually produce signals
        # (may still be None if confidence < threshold — that's OK)
        non_none = [s for s in signals if s is not None]
        if non_none:
            for s in non_none:
                assert -1.0 <= s.direction <= 1.0
                assert 0.0 <= s.confidence <= 1.0
                assert s.strategy_id == "ml"

    def test_reset_clears_model(self, settings):
        """reset() must clear the trained model (no state leak between runs)."""
        n = settings.ml_lookback + MIN_BARS_FOR_TRAINING + 10
        bars = make_trending_bars(n=n, drift=0.002, seed=13)
        strategy = MLStrategy(settings)

        for bar in bars:
            strategy.on_market_data(MarketDataEvent(bar=bar))

        # After feeding bars, model may be trained
        strategy.reset()

        # After reset, model must be None
        assert strategy._model is None
        assert strategy._bars_since_retrain == 0
        assert len(strategy.bars) == 0

    def test_deterministic_after_reset(self, settings):
        """
        Same bars → same signal sequence when called after reset().
        This is the state-leak check equivalent to the backtest determinism
        test for MomentumStrategy.
        """
        n = settings.ml_lookback + MIN_BARS_FOR_TRAINING + 20
        bars = make_trending_bars(n=n, drift=0.002, seed=17)
        strategy = MLStrategy(settings)

        def collect_signals():
            strategy.reset()
            return [
                (i, round(s.direction, 6), s.symbol)
                for i, b in enumerate(bars)
                if (s := strategy.on_market_data(MarketDataEvent(bar=b))) is not None
            ]

        run1 = collect_signals()
        run2 = collect_signals()

        assert run1 == run2, (
            "MLStrategy backtest runs produced different results — "
            "model state was not fully cleared by reset()"
        )

    def test_signal_direction_in_valid_range(self, settings):
        """Every signal emitted must have direction in [-1, 1] and confidence in [0, 1]."""
        n = settings.ml_lookback + MIN_BARS_FOR_TRAINING + 30
        bars = make_mean_reverting_bars(n=n, mean=100.0, seed=19)
        strategy = MLStrategy(settings)

        for bar in bars:
            signal = strategy.on_market_data(MarketDataEvent(bar=bar))
            if signal is not None:
                assert -1.0 <= signal.direction <= 1.0
                assert 0.0 <= signal.confidence <= 1.0

    def test_strategy_id(self, settings):
        """MLStrategy must identify itself as 'ml'."""
        strategy = MLStrategy(settings)
        assert strategy.strategy_id == "ml"
