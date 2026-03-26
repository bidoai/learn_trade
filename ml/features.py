"""
Feature engineering for the ML strategy.

LOOK-AHEAD BIAS GUARD
─────────────────────
build() takes a mandatory cutoff_date parameter and asserts that every
bar in the input has timestamp <= cutoff_date. This is the architectural
enforcement that prevents leaking future prices into the training set.

When called during live prediction, cutoff_date = current bar's timestamp.
When called during backtest training, cutoff_date = the last bar seen so far.

Features (all derived from OHLCV bars, no external data):
  return_1      : log return over 1 bar
  return_5      : log return over 5 bars
  return_10     : log return over 10 bars
  return_20     : log return over 20 bars
  volatility_10 : rolling std of 1-bar returns, window=10
  volatility_20 : rolling std of 1-bar returns, window=20
  vol_regime    : volatility_5 / volatility_20  — <1 = compression, >1 = expansion
  rsi_14        : Relative Strength Index, period=14
  volume_ratio  : current volume / 20-bar average volume
  hl_ratio      : (high - low) / close  — intrabar range as % of price
  close_pos     : (close - low) / (high - low)  — where close falls in the bar
  obv_trend     : sign of OBV slope over 10 bars — buying/selling pressure
  gap_open      : (open - prev_close) / prev_close — overnight gap
  body_size     : abs(close - open) / (high - low) — candle conviction
  stoch_k       : (close - min_low_14) / (max_high_14 - min_low_14) — stochastic %K
  vpt_5         : volume-price trend over 5 bars — normalized

Label (for training only):
  Ternary: 2 = strong up (> +threshold), 1 = flat, 0 = strong down (< -threshold)
  threshold = ml_label_threshold (default 0.001 = 0.1%)
  The last row always has NaN label (no next bar yet) — drop before training.

Data flow:
  List[Bar] + cutoff_date
      │
      ▼ assert all timestamps <= cutoff_date
      │
      ▼ build raw DataFrame (symbol, timestamp, OHLCV)
      │
      ▼ compute features (pandas vectorized ops)
      │
      ▼ compute label (shift(-1)) — last row NaN
      │
      ▼ drop rows with any NaN (warmup period rows)
      │
      ▼ return DataFrame with FEATURE_COLS + 'label'
"""
from __future__ import annotations

import math
from datetime import datetime
from typing import List

import numpy as np
import pandas as pd
import structlog

from core.models import Bar

logger = structlog.get_logger(__name__)

# Columns produced by build() that are used as model inputs
FEATURE_COLS = [
    "return_1",
    "return_5",
    "return_10",
    "return_20",
    "volatility_10",
    "volatility_20",
    "vol_regime",
    "rsi_14",
    "volume_ratio",
    "hl_ratio",
    "close_pos",
    "obv_trend",
    "gap_open",
    "body_size",
    "stoch_k",
    "vpt_5",
]

LABEL_COL = "label"

# Default label threshold: moves < ±0.1% are labelled "flat" (class 1)
DEFAULT_LABEL_THRESHOLD = 0.001

# Minimum bars required before any feature row is valid
# (max lookback is 20, plus we need 1 extra bar for the label)
MIN_BARS_FOR_TRAINING = 22


def build(bars: List[Bar], cutoff_date: datetime, label_threshold: float = DEFAULT_LABEL_THRESHOLD) -> pd.DataFrame:
    """
    Build a feature DataFrame from a list of OHLCV bars.

    Parameters
    ----------
    bars : List[Bar]
        Bars in chronological order (oldest first).
    cutoff_date : datetime
        LOOK-AHEAD BIAS GUARD — every bar must have timestamp <= cutoff_date.
        This is asserted, not just checked, to make violations immediately
        visible during testing and backtest runs.

    Returns
    -------
    pd.DataFrame
        Columns: FEATURE_COLS + [LABEL_COL].
        All rows with NaN dropped (warmup period + last row missing label).
        Index is the bar timestamps.

    Raises
    ------
    AssertionError
        If any bar's timestamp exceeds cutoff_date (look-ahead bias detected).
    ValueError
        If fewer than MIN_BARS_FOR_TRAINING bars are provided.
    """
    if len(bars) < MIN_BARS_FOR_TRAINING:
        raise ValueError(
            f"Need at least {MIN_BARS_FOR_TRAINING} bars, got {len(bars)}"
        )

    # ── LOOK-AHEAD BIAS GUARD ──────────────────────────────────────────────
    for bar in bars:
        assert bar.timestamp <= cutoff_date, (
            f"Look-ahead bias detected: bar.timestamp={bar.timestamp} "
            f"> cutoff_date={cutoff_date}. "
            f"Training set contains data from the future."
        )

    # ── Build raw DataFrame ────────────────────────────────────────────────
    df = pd.DataFrame(
        {
            "timestamp": [b.timestamp for b in bars],
            "open":      [b.open for b in bars],
            "high":      [b.high for b in bars],
            "low":       [b.low for b in bars],
            "close":     [b.close for b in bars],
            "volume":    [b.volume for b in bars],
        }
    )
    df = df.set_index("timestamp").sort_index()

    # ── Features ───────────────────────────────────────────────────────────
    log_close = np.log(df["close"])

    df["return_1"]  = log_close.diff(1)
    df["return_5"]  = log_close.diff(5)
    df["return_10"] = log_close.diff(10)
    df["return_20"] = log_close.diff(20)

    df["volatility_10"] = df["return_1"].rolling(10).std()
    df["volatility_20"] = df["return_1"].rolling(20).std()

    vol_5 = df["return_1"].rolling(5).std()
    df["vol_regime"] = (vol_5 / df["volatility_20"].replace(0, np.nan)).clip(0.1, 5.0)

    df["rsi_14"] = _rsi(df["close"], period=14)

    vol_ma = df["volume"].rolling(20).mean()
    df["volume_ratio"] = df["volume"] / vol_ma.replace(0, np.nan)

    hl = df["high"] - df["low"]
    df["hl_ratio"]  = hl / df["close"]
    df["close_pos"] = (df["close"] - df["low"]) / hl.replace(0, np.nan)

    # OBV trend: sign of OBV slope over 10 bars
    obv = (np.sign(df["return_1"]) * df["volume"]).cumsum()
    df["obv_trend"] = np.sign(obv.diff(10))

    # Overnight gap: (open - prev_close) / prev_close
    df["gap_open"] = (df["open"] - df["close"].shift(1)) / df["close"].shift(1).replace(0, np.nan)
    df["gap_open"] = df["gap_open"].fillna(0.0)

    # Candle body ratio: conviction of the directional move
    df["body_size"] = (df["close"] - df["open"]).abs() / hl.replace(0, np.nan)
    df["body_size"] = df["body_size"].clip(0.0, 1.0)

    # Stochastic %K: where close falls within the 14-bar high-low range
    low_14  = df["low"].rolling(14).min()
    high_14 = df["high"].rolling(14).max()
    df["stoch_k"] = ((df["close"] - low_14) / (high_14 - low_14).replace(0, np.nan)).clip(0.0, 1.0)

    # Volume-price trend over 5 bars, normalized by mean volume
    vpt = (df["return_1"] * df["volume"]).rolling(5).sum()
    mean_vol = df["volume"].rolling(20).mean().replace(0, np.nan)
    df["vpt_5"] = vpt / mean_vol

    # ── Ternary label: 2=up, 1=flat, 0=down ───────────────────────────────
    # shift(-1) looks at the NEXT bar's close — valid for training because
    # we only label rows where the next bar is in our training window.
    # The last row gets NaN (no next bar) and is dropped below.
    next_return = df["close"].shift(-1) / df["close"] - 1
    label = pd.Series(1.0, index=df.index)  # default: flat
    label[next_return > label_threshold]  = 2.0
    label[next_return < -label_threshold] = 0.0
    label[next_return.isna()] = np.nan
    df[LABEL_COL] = label

    # Drop NaN label (last row) and NaN features (warmup rows)
    result = df[FEATURE_COLS + [LABEL_COL]].dropna()

    logger.debug(
        "features.built",
        input_bars=len(bars),
        output_rows=len(result),
        cutoff_date=cutoff_date.isoformat(),
    )

    return result


def build_latest(bars: List[Bar]) -> pd.Series:
    """
    Build features for the most recent bar only (for live prediction).

    Uses the last bar's timestamp as the cutoff_date — no look-ahead bias
    possible since we're predicting from the current bar forward.

    Returns
    -------
    pd.Series
        Feature values for the latest bar (FEATURE_COLS only, no label).
        Returns None if there aren't enough bars to compute all features.
    """
    if len(bars) < MIN_BARS_FOR_TRAINING:
        return None

    cutoff = bars[-1].timestamp

    try:
        df = build(bars, cutoff_date=cutoff)
    except (ValueError, AssertionError):
        return None

    if df.empty:
        return None

    return df[FEATURE_COLS].iloc[-1]


# ── Private helpers ────────────────────────────────────────────────────────

def _rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """
    Wilder's RSI.

    RSI = 100 - (100 / (1 + RS))
    RS  = avg_gain / avg_loss  over `period` bars

    Returns values in [0, 100]. Returns NaN for first `period` rows.
    """
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    # Wilder's smoothing = exponential with alpha = 1/period
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi
