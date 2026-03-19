"""
ML-based trading strategy.

Architecture
────────────
MLStrategy inherits from Strategy (rolling deque, NaN guard, warmup period)
and adds a train/predict layer on top:

  on_market_data(event)          [inherited from Strategy base]
      │
      ├── NaN guard + warmup (returns None until lookback_window bars seen)
      │
      └── generate_signal()      [implemented here]
              │
              ├── retrain?  (_bar_count % retrain_interval == 0 AND enough data)
              │       └── FeatureBuilder.build(bars, cutoff=latest_bar.timestamp)
              │           → trainer.train(df)
              │           → self._model = TrainedModel
              │
              ├── model not yet trained → None
              │
              └── features.build_latest(bars)
                      → predictor.predict(model, features)
                      → SignalEvent or None

Look-ahead bias prevention
──────────────────────────
Training uses only bars in self.bars (past data up to current bar).
build() is called with cutoff_date = current_bar.timestamp.
The assertion in features.build() catches any violation at test time.

Retrain schedule
────────────────
The model is trained:
  1. First time: when lookback_window bars are accumulated AND
     training data contains >= MIN_TRAINING_ROWS usable rows.
  2. Subsequently: every ml_retrain_interval_bars bars.

Retraining is synchronous and happens in the event loop — acceptable
for 1-minute bars (retrain takes ~100ms for 100 bars). For tick data
or large windows, move to run_in_executor().

State reset
───────────
_reset_state() clears self._model and self._bars_since_retrain.
BacktestRunner calls strategy.reset() between runs, which calls
_reset_state(). Without this, a model trained in run N would
contaminate run N+1.
"""
from __future__ import annotations

import structlog
from typing import Optional

from config.settings import StrategySettings
from core.events import SignalEvent
from strategy.base import Strategy
import ml.features as features
import ml.trainer as trainer
import ml.predictor as predictor
from ml.trainer import TrainedModel

logger = structlog.get_logger(__name__)


class MLStrategy(Strategy):
    """
    Gradient-boosted classifier strategy.

    Trains on the rolling bar window and predicts the direction of the
    next bar's return. Retrains every `retrain_interval` bars.
    """
    strategy_id = "ml"

    def __init__(self, settings: StrategySettings) -> None:
        self.lookback_window = settings.ml_lookback
        self.retrain_interval = settings.ml_retrain_interval_bars
        self._model: Optional[TrainedModel] = None
        self._bars_since_retrain: int = 0
        super().__init__()
        self._log = logger.bind(strategy=self.strategy_id)

    # ── Core signal generation ─────────────────────────────────────────────

    def generate_signal(self) -> Optional[SignalEvent]:
        """
        Called by base class when self.bars has lookback_window bars.

        Decision tree:
          1. Should we retrain? (first time, or interval elapsed)
          2. If no model yet, wait.
          3. Build features for latest bar.
          4. Predict → emit SignalEvent or None.
        """
        bars = list(self.bars)
        current_bar = bars[-1]
        symbol = current_bar.symbol

        self._bars_since_retrain += 1

        # ── Retrain trigger ────────────────────────────────────────────────
        should_retrain = (
            self._model is None
            or self._bars_since_retrain >= self.retrain_interval
        )

        if should_retrain:
            self._retrain(bars, cutoff=current_bar.timestamp)
            self._bars_since_retrain = 0

        if self._model is None:
            # Not enough data to train yet
            return None

        # ── Predict ────────────────────────────────────────────────────────
        feature_vec = features.build_latest(bars)
        if feature_vec is None:
            return None

        result = predictor.predict(self._model, feature_vec)
        if result is None:
            return None

        direction, confidence = result

        return SignalEvent(
            strategy_id=self.strategy_id,
            symbol=symbol,
            direction=direction,
            confidence=confidence,
            reason=(
                f"ml_prediction: direction={direction:.3f} "
                f"confidence={confidence:.3f} "
                f"cv_acc={self._model.cv_accuracy:.3f}"
                if self._model.cv_accuracy is not None
                else f"ml_prediction: direction={direction:.3f} confidence={confidence:.3f}"
            ),
        )

    # ── Private ────────────────────────────────────────────────────────────

    def _retrain(self, bars: list, cutoff) -> None:
        """
        Build features and train a new model.
        Sets self._model = None if training fails (not enough data).

        The cutoff_date=cutoff argument is the look-ahead bias guard:
        no bar beyond the current bar's timestamp can appear in training.
        """
        try:
            df = features.build(bars, cutoff_date=cutoff)
        except (ValueError, AssertionError) as e:
            self._log.warning("ml_strategy.feature_build_failed", error=str(e))
            self._model = None
            return

        model = trainer.train(df)
        if model is not None:
            self._log.info(
                "ml_strategy.retrained",
                n_bars=len(bars),
                n_training_rows=model.n_training_rows,
                cv_accuracy=model.cv_accuracy,
            )
        self._model = model

    def _reset_state(self) -> None:
        """Clear trained model. Called by Strategy.reset() between backtest runs."""
        self._model = None
        self._bars_since_retrain = 0
        self._log.debug("ml_strategy.state_reset")
