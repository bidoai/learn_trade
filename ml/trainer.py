"""
ML model training with TimeSeriesSplit.

WHY TimeSeriesSplit, NOT train_test_split
─────────────────────────────────────────
Random splitting shuffles the time axis. For time series, that means
you might train on 2024 data and test on 2023 data — effectively telling
the model the future to validate on the past. Results look great but
are completely fictional.

TimeSeriesSplit always tests on data AFTER the training window:

  Fold 1: train=[t0..t16]  test=[t17..t33]
  Fold 2: train=[t0..t33]  test=[t34..t50]
  Fold 3: train=[t0..t50]  test=[t51..t67]
  Fold 4: train=[t0..t67]  test=[t68..t83]
  Fold 5: train=[t0..t83]  test=[t84..t100]

This mirrors real-world conditions: you only know the past.

Training flow:
  DataFrame (features + label)
      │
      ▼ extract X, y; drop NaN label (last row)
      │
      ▼ TimeSeriesSplit cross-validation → log accuracy per fold
      │
      ▼ train final model on ALL data
      │
      ▼ return TrainedModel

Model choice: GradientBoostingClassifier
  - Handles nonlinear relationships well
  - Less prone to overfitting than raw decision trees
  - predict_proba() gives calibrated probabilities for confidence scores
  - Fast enough to retrain every 500 bars (~minutes of live data)

In a real fund, you'd use LightGBM or XGBoost with more careful
hyperparameter tuning and regime detection.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import structlog
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from ml.features import FEATURE_COLS, LABEL_COL

logger = structlog.get_logger(__name__)

# Minimum training rows after feature warmup rows are dropped
MIN_TRAINING_ROWS = 50


@dataclass
class TrainedModel:
    """
    Wraps the sklearn model + scaler as a unit.

    Scaler note: StandardScaler is fit on training data only.
    The same scaler is applied to live features at prediction time.
    Fitting the scaler on all data (including test) is another form
    of look-ahead bias — avoided here by pairing scaler + model.
    """
    classifier: GradientBoostingClassifier
    scaler: StandardScaler
    n_training_rows: int
    cv_accuracy: Optional[float]  # mean CV accuracy across folds
    feature_cols: list[str]       # column order the model expects


def train(df) -> Optional[TrainedModel]:
    """
    Train a GradientBoostingClassifier on the feature DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Output of features.build() — must contain FEATURE_COLS and LABEL_COL.
        Rows with NaN should already be dropped.

    Returns
    -------
    TrainedModel or None
        None if there is insufficient data to train.
    """
    if len(df) < MIN_TRAINING_ROWS:
        logger.warning(
            "trainer.insufficient_data",
            rows=len(df),
            min_required=MIN_TRAINING_ROWS,
        )
        return None

    X = df[FEATURE_COLS].values
    y = df[LABEL_COL].values.astype(int)

    # ── Cross-validation (for logging/monitoring, not for model selection) ──
    cv_accuracy = _cross_validate(X, y)

    # ── Train final model on ALL available data ────────────────────────────
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=3,          # shallow trees → less overfitting
        learning_rate=0.05,   # smaller steps → more robust
        subsample=0.8,        # row sampling → more robust
        random_state=42,
    )
    clf.fit(X_scaled, y)

    logger.info(
        "trainer.model_trained",
        n_training_rows=len(df),
        cv_accuracy=round(cv_accuracy, 4) if cv_accuracy is not None else None,
        n_estimators=clf.n_estimators_,
    )

    return TrainedModel(
        classifier=clf,
        scaler=scaler,
        n_training_rows=len(df),
        cv_accuracy=cv_accuracy,
        feature_cols=FEATURE_COLS,
    )


def _cross_validate(X: np.ndarray, y: np.ndarray) -> Optional[float]:
    """
    Run TimeSeriesSplit cross-validation and return mean accuracy.
    Returns None if there aren't enough rows for even one fold.
    """
    n_splits = min(5, len(X) // 20)   # at least 20 rows per fold
    if n_splits < 2:
        return None

    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_accuracies = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        clf = GradientBoostingClassifier(
            n_estimators=50,      # fewer for CV speed
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
        )
        clf.fit(X_train_scaled, y_train)
        acc = clf.score(X_test_scaled, y_test)
        fold_accuracies.append(acc)

        logger.debug(
            "trainer.cv_fold",
            fold=fold + 1,
            train_rows=len(X_train),
            test_rows=len(X_test),
            accuracy=round(acc, 4),
        )

    mean_acc = float(np.mean(fold_accuracies))
    logger.info(
        "trainer.cv_complete",
        n_splits=n_splits,
        mean_accuracy=round(mean_acc, 4),
        fold_accuracies=[round(a, 4) for a in fold_accuracies],
    )
    return mean_acc
