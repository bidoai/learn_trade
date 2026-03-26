"""
Information Coefficient (IC) tracker.

Queries the EventStore to measure whether SignalEvent.direction
correlated with actual subsequent returns. A decaying IC indicates
the strategy's edge is being arbitraged away.

IC = Pearson correlation(signal_direction, realized_outcome)
Run this offline as a post-session diagnostic, not in the hot path.

Usage:
    tracker = ICTracker("trading.db")
    ics = tracker.compute_ic(lookback_days=30)
    # {"momentum": 0.12, "ml": 0.08, ...}
"""
from __future__ import annotations

import json
import sqlite3
from typing import Optional

import structlog

logger = structlog.get_logger(__name__)


class ICTracker:
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path

    def compute_ic(
        self,
        strategy_id: Optional[str] = None,
        lookback_days: int = 30,
    ) -> dict[str, float]:
        """
        Compute IC per strategy over the last N days.

        IC is approximated as: Pearson correlation between signal direction
        and the sign of the realized fill outcome (fill direction vs. prior
        signal direction). A positive IC means signals predicted direction
        correctly more than randomly.

        Returns {strategy_id: ic_value}. Empty dict if insufficient data.
        """
        conn = sqlite3.connect(self.db_path)
        try:
            signals_query = """
                SELECT timestamp, payload
                FROM events
                WHERE event_type = 'SignalEvent'
                AND timestamp >= datetime('now', ?)
                ORDER BY timestamp ASC
            """
            rows = conn.execute(signals_query, (f"-{lookback_days} days",)).fetchall()
        except sqlite3.OperationalError as e:
            logger.warning("ic_tracker.query_failed", error=str(e))
            return {}
        finally:
            conn.close()

        if not rows:
            logger.info("ic_tracker.no_signals_found", lookback_days=lookback_days)
            return {}

        # Parse signals
        signals: list[dict] = []
        for ts, payload_str in rows:
            try:
                payload = json.loads(payload_str)
            except (json.JSONDecodeError, TypeError):
                continue
            sid = payload.get("strategy_id", "")
            if strategy_id and sid != strategy_id:
                continue
            signals.append({
                "strategy_id": sid,
                "symbol": payload.get("symbol", ""),
                "direction": float(payload.get("direction", 0)),
                "timestamp": ts,
            })

        if len(signals) < 10:
            logger.info("ic_tracker.insufficient_signals", count=len(signals))
            return {}

        # Re-open for fills query
        conn = sqlite3.connect(self.db_path)
        try:
            fill_rows = conn.execute(
                "SELECT timestamp, payload FROM events WHERE event_type = 'FillEvent' ORDER BY timestamp ASC"
            ).fetchall()
        except sqlite3.OperationalError:
            return {}
        finally:
            conn.close()

        fills_by_symbol: dict[str, list[dict]] = {}
        for ts, payload_str in fill_rows:
            try:
                payload = json.loads(payload_str)
            except (json.JSONDecodeError, TypeError):
                continue
            fill = payload.get("fill", payload)
            sym = fill.get("symbol", "")
            if sym not in fills_by_symbol:
                fills_by_symbol[sym] = []
            fills_by_symbol[sym].append({
                "timestamp": ts,
                "side": fill.get("side", ""),
                "fill_price": float(fill.get("fill_price", 0)),
            })

        # Pair each signal with the next fill for the same symbol
        pairs: dict[str, list[tuple[float, float]]] = {}
        for sig in signals:
            sym = sig["symbol"]
            sig_ts = sig["timestamp"]
            candidates = [
                f for f in fills_by_symbol.get(sym, [])
                if f["timestamp"] > sig_ts
            ]
            if not candidates:
                continue
            next_fill = candidates[0]
            # Outcome: did fill direction match signal direction?
            fill_direction = 1.0 if next_fill["side"] == "buy" else -1.0
            outcome = 1.0 if fill_direction == sig["direction"] else -1.0
            sid = sig["strategy_id"]
            if sid not in pairs:
                pairs[sid] = []
            pairs[sid].append((sig["direction"], outcome))

        # Pearson IC per strategy
        result: dict[str, float] = {}
        for sid, data in pairs.items():
            if len(data) < 5:
                continue
            directions = [d for d, _ in data]
            outcomes = [o for _, o in data]
            ic = _pearson(directions, outcomes)
            if ic is not None:
                result[sid] = round(ic, 4)
                logger.info("ic_tracker.ic_computed", strategy_id=sid, ic=result[sid], n=len(data))

        return result


def _pearson(xs: list[float], ys: list[float]) -> Optional[float]:
    n = len(xs)
    if n < 2:
        return None
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / n
    std_x = (sum((x - mean_x) ** 2 for x in xs) / n) ** 0.5
    std_y = (sum((y - mean_y) ** 2 for y in ys) / n) ** 0.5
    if std_x < 1e-9 or std_y < 1e-9:
        return None
    return cov / (std_x * std_y)
