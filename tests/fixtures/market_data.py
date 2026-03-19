"""
Synthetic market data generators for tests.

Functions produce lists of Bar objects with controllable statistical
properties. Tests use these instead of real market data so they:
  - Run without network access
  - Can test specific conditions on demand
  - Are deterministic (set numpy seed for reproducibility)

Usage:
    bars = make_trending_bars(n=100, drift=0.002)
    bars = make_mean_reverting_bars(n=50, mean=100.0)
    bars = make_bars_with_gaps(n=100, gap_indices=[20, 50])
    bar  = make_nan_bar()
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

import numpy as np

from core.models import Bar

_BASE_TIME = datetime(2024, 1, 2, 9, 30, 0)  # market open Jan 2 2024


def make_trending_bars(
    n: int = 100,
    drift: float = 0.001,
    symbol: str = "AAPL",
    start_price: float = 100.0,
    seed: Optional[int] = 42,
) -> list[Bar]:
    """
    Upward-trending price series.
    Use to test momentum strategy signal generation.
    drift > 0 = uptrend, drift < 0 = downtrend.
    """
    rng = np.random.default_rng(seed)
    returns = rng.normal(drift, 0.01, n)
    prices = start_price * np.cumprod(1 + returns)
    return _prices_to_bars(prices, symbol)


def make_mean_reverting_bars(
    n: int = 100,
    mean: float = 100.0,
    std: float = 5.0,
    symbol: str = "AAPL",
    seed: Optional[int] = 42,
) -> list[Bar]:
    """
    Price series that oscillates around a mean.
    Use to test mean reversion strategy signal generation.
    """
    rng = np.random.default_rng(seed)
    # Ornstein-Uhlenbeck process: dX = theta*(mu - X)*dt + sigma*dW
    theta = 0.3
    prices = np.zeros(n)
    prices[0] = mean
    for i in range(1, n):
        prices[i] = prices[i-1] + theta * (mean - prices[i-1]) + rng.normal(0, std * 0.1)
    return _prices_to_bars(prices, symbol)


def make_bars_with_gaps(
    n: int = 100,
    gap_indices: tuple = (20, 50),
    symbol: str = "AAPL",
    seed: Optional[int] = 42,
) -> list[Bar]:
    """
    Price series with missing bars at specified indices.
    Use to test stale data detection and gap handling.
    """
    bars = make_trending_bars(n=n, symbol=symbol, seed=seed)
    return [b for i, b in enumerate(bars) if i not in gap_indices]


def make_nan_bar(symbol: str = "AAPL") -> Bar:
    """
    A bar with NaN close price.
    Use to test NaN guard in Strategy base class.
    Expected: strategy returns None, no order created.
    """
    return Bar(
        symbol=symbol,
        timestamp=_BASE_TIME,
        open=float("nan"),
        high=float("nan"),
        low=float("nan"),
        close=float("nan"),
        volume=0,
    )


def make_known_bars(prices: list[float], symbol: str = "AAPL") -> list[Bar]:
    """
    Create bars from an explicit list of close prices.
    Use for known-outcome backtest tests where you need exact P&L.
    OHLC set to close (simplification — fine for known-outcome tests).
    """
    return _prices_to_bars(np.array(prices), symbol)


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------

def _prices_to_bars(prices: np.ndarray, symbol: str) -> list[Bar]:
    bars = []
    for i, close in enumerate(prices):
        # Simulate realistic OHLC from close
        noise = abs(close) * 0.002
        rng = np.random.default_rng(i)
        spread = rng.uniform(0, noise)
        bars.append(Bar(
            symbol=symbol,
            timestamp=_BASE_TIME + timedelta(days=i),
            open=float(close - spread / 2),
            high=float(close + spread),
            low=float(close - spread),
            close=float(close),
            volume=int(rng.integers(100_000, 1_000_000)),
        ))
    return bars
