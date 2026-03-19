"""
Value at Risk (VaR) calculator.

VaR answers: "What is the maximum loss I'd expect with X% confidence
over the next trading day?"

Example: VaR(95%, $100k portfolio) = $2,000 means there's a 95% chance
of not losing more than $2,000 tomorrow.

Method: Historical simulation VaR.
  1. Collect the last N days of portfolio returns
  2. Sort them
  3. Take the (1 - confidence) percentile as the loss estimate

Limitations for a learning system:
  - Assumes returns are i.i.d. (not always true)
  - Doesn't capture fat tails as well as CVaR would
  - Requires at least 20 data points (returns None if fewer)

Real funds use more sophisticated methods (CVaR, Monte Carlo, parametric)
but historical simulation is the most intuitive starting point.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import structlog

logger = structlog.get_logger(__name__)

MIN_OBSERVATIONS = 20  # minimum data points for a meaningful VaR estimate


class VaRCalculator:
    def __init__(self, confidence: float = 0.95, window_days: int = 20) -> None:
        if not 0 < confidence < 1:
            raise ValueError(f"confidence must be in (0, 1), got {confidence}")
        self.confidence = confidence
        self.window_days = window_days
        self._daily_returns: list[float] = []

    def add_return(self, daily_return: float) -> None:
        """Record today's portfolio return. Called by PerformanceTracker."""
        self._daily_returns.append(daily_return)
        # Keep only the rolling window
        if len(self._daily_returns) > self.window_days:
            self._daily_returns.pop(0)

    def calculate(self) -> Optional[float]:
        """
        Returns the VaR as a positive number representing the loss threshold.
        Returns None if insufficient data (< MIN_OBSERVATIONS).

        Example: returns 0.018 means VaR = 1.8% of portfolio value.
        """
        if len(self._daily_returns) < MIN_OBSERVATIONS:
            logger.debug(
                "var.insufficient_data",
                observations=len(self._daily_returns),
                required=MIN_OBSERVATIONS,
            )
            return None

        try:
            returns = np.array(self._daily_returns)
            # Historical VaR: the loss at the (1 - confidence) percentile
            # Negative returns are losses, so we negate for the loss measure
            var = float(np.percentile(returns, (1 - self.confidence) * 100))
            return abs(var)  # return as positive loss magnitude
        except np.linalg.LinAlgError as e:
            # Fail CLOSED: if calculation fails, return a conservative estimate
            logger.error("var.calculation_failed", error=str(e))
            return None

    def reset(self) -> None:
        """Clear history. Called by BacktestRunner between runs."""
        self._daily_returns.clear()
