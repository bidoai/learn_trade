"""
Strategy performance tracker for volatility-scaled capital allocation.

Tracks rolling realized P&L per strategy and computes volatility-scaled
weights so each strategy contributes equal volatility dollars to the book.

Replaces the fixed strategy_weights dict in PortfolioAllocator when wired in.

Usage:
    tracker = StrategyPerformanceTracker(
        strategy_ids=["momentum", "mean_reversion", "ml"],
        base_weights=settings.strategy_weights,
    )
    # On each fill:
    tracker.record_return("momentum", pnl_pct)
    # In allocator:
    weights = tracker.get_scaled_weights()
"""
from __future__ import annotations

from collections import deque
from math import sqrt
from typing import Optional

import structlog

logger = structlog.get_logger(__name__)


class StrategyPerformanceTracker:
    """
    Computes inverse-volatility weights across strategies.

    When a strategy has high realized volatility, its weight is reduced
    so that all strategies contribute equal volatility-dollars to the
    combined portfolio. Falls back to base_weights when any strategy
    lacks sufficient return history.
    """

    def __init__(
        self,
        strategy_ids: list[str],
        base_weights: dict[str, float],
        vol_window: int = 20,
        min_observations: int = 10,
    ) -> None:
        self.base_weights = base_weights
        self.vol_window = vol_window
        self.min_observations = min_observations
        self._returns: dict[str, deque[float]] = {
            sid: deque(maxlen=vol_window) for sid in strategy_ids
        }

    def record_return(self, strategy_id: str, pnl_pct: float) -> None:
        """Record a realized return fraction for a strategy. Call on each fill."""
        if strategy_id in self._returns:
            self._returns[strategy_id].append(pnl_pct)

    def get_scaled_weights(self) -> dict[str, float]:
        """
        Return volatility-scaled weights. Falls back to base_weights if any
        strategy has fewer than min_observations returns recorded.
        """
        vols: dict[str, Optional[float]] = {}
        for sid, returns in self._returns.items():
            if len(returns) >= self.min_observations:
                mean = sum(returns) / len(returns)
                variance = sum((r - mean) ** 2 for r in returns) / len(returns)
                vols[sid] = sqrt(variance) if variance > 1e-12 else None
            else:
                vols[sid] = None

        if any(v is None for v in vols.values()):
            return dict(self.base_weights)

        inv_vols = {sid: 1.0 / v if v and v > 1e-9 else 1.0 for sid, v in vols.items()}
        total_inv_vol = sum(inv_vols.values())

        if total_inv_vol < 1e-9:
            return dict(self.base_weights)

        scaled = {sid: inv_v / total_inv_vol for sid, inv_v in inv_vols.items()}
        logger.debug("performance_tracker.weights_computed", weights=scaled)
        return scaled

    def strategy_volatilities(self) -> dict[str, Optional[float]]:
        """Return current realized volatility per strategy (for monitoring)."""
        result: dict[str, Optional[float]] = {}
        for sid, returns in self._returns.items():
            if len(returns) >= self.min_observations:
                mean = sum(returns) / len(returns)
                variance = sum((r - mean) ** 2 for r in returns) / len(returns)
                result[sid] = round(sqrt(variance), 6) if variance > 1e-12 else 0.0
            else:
                result[sid] = None
        return result
