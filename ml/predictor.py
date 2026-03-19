"""
Model inference: converts a feature vector into (direction, confidence).

Prediction math
───────────────
The classifier outputs p = probability that next close > current close.

We map p → (direction, confidence):

  p in [0, 1]

  direction  = (p - 0.5) * 2          maps [0, 1] → [-1, +1]
             = 0.0  when p = 0.5  (model is uncertain)
             = +1.0 when p = 1.0  (model is fully long)
             = -1.0 when p = 0.0  (model is fully short)

  confidence = abs(p - 0.5) * 2       maps [0, 0.5] → [0, 1]
             = 0.0  when p = 0.5  (no conviction)
             = 1.0  when p = 0.0 or 1.0 (max conviction)

  threshold  = 0.1  — minimum confidence below which we emit no signal.
               This filters out weak predictions near p=0.5.

Real-world note:
  Calibration matters. A model that says p=0.9 should actually be right
  ~90% of the time. GradientBoosting tends to produce uncalibrated
  probabilities; CalibratedClassifierCV from sklearn can fix this.
  We skip that here but it's a meaningful TODO for production.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd
import structlog

from ml.trainer import TrainedModel

logger = structlog.get_logger(__name__)

# Suppress signals when abs(p - 0.5) * 2 < MIN_CONFIDENCE
MIN_CONFIDENCE = 0.1


def predict(
    model: TrainedModel,
    features: pd.Series,
) -> Optional[Tuple[float, float]]:
    """
    Run inference and return (direction, confidence) or None.

    Parameters
    ----------
    model : TrainedModel
        Trained model from trainer.train().
    features : pd.Series
        Feature vector from features.build_latest(), indexed by FEATURE_COLS.

    Returns
    -------
    (direction, confidence) or None
        direction  : float in [-1.0, +1.0]
        confidence : float in [ 0.0,  1.0]
        None if confidence is below threshold or inference fails.
    """
    try:
        # Align feature order to match training
        x = features[model.feature_cols].values.reshape(1, -1)

        # Apply the same scaler fitted during training
        x_scaled = model.scaler.transform(x)

        # predict_proba returns [[p_down, p_up]]
        proba = model.classifier.predict_proba(x_scaled)[0]
        p_up = float(proba[1])   # probability that next close > current close

    except Exception:
        logger.exception("predictor.inference_error")
        return None

    direction  = (p_up - 0.5) * 2
    confidence = abs(p_up - 0.5) * 2

    if confidence < MIN_CONFIDENCE:
        logger.debug(
            "predictor.low_confidence_suppressed",
            p_up=round(p_up, 4),
            confidence=round(confidence, 4),
            threshold=MIN_CONFIDENCE,
        )
        return None

    # Clamp to [-1, 1] and [0, 1] defensively (math should guarantee this)
    direction  = max(-1.0, min(1.0, direction))
    confidence = max(0.0,  min(1.0, confidence))

    # Guard against NaN (shouldn't happen with sklearn, but be explicit)
    if np.isnan(direction) or np.isnan(confidence):
        logger.warning("predictor.nan_output", p_up=p_up)
        return None

    logger.debug(
        "predictor.signal",
        p_up=round(p_up, 4),
        direction=round(direction, 4),
        confidence=round(confidence, 4),
    )

    return direction, confidence
