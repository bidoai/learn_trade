"""
Position tracker — single source of truth for current positions.

All position reads and writes go through here.
No other component calculates or modifies positions directly.

Positions are updated on fills (FillEvent). On startup, positions
are loaded from SQLite to restore state across restarts.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

import structlog

from core.models import Fill, OrderSide, Position

logger = structlog.get_logger(__name__)


class PositionTracker:
    def __init__(self) -> None:
        # symbol → Position (one aggregate position per symbol, across strategies)
        self._positions: dict[str, Position] = {}

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def get(self, symbol: str) -> Optional[Position]:
        return self._positions.get(symbol)

    def all_positions(self) -> list[Position]:
        return [p for p in self._positions.values() if not p.is_flat]

    def is_flat(self, symbol: str) -> bool:
        pos = self._positions.get(symbol)
        return pos is None or pos.is_flat

    # ------------------------------------------------------------------
    # Writes — only called by OMS after fills
    # ------------------------------------------------------------------

    def apply_fill(self, fill: Fill) -> Position:
        """
        Update position based on a fill.
        Handles: new position, adding to position, partial close, full close, flip.
        Returns the updated Position.
        """
        pos = self._positions.get(fill.symbol)

        if pos is None:
            # New position
            quantity = fill.fill_quantity if fill.side == OrderSide.BUY else -fill.fill_quantity
            pos = Position(
                symbol=fill.symbol,
                quantity=quantity,
                avg_entry_price=fill.fill_price,
                strategy_id=fill.strategy_id,
            )
            self._positions[fill.symbol] = pos
        else:
            fill_qty = fill.fill_quantity if fill.side == OrderSide.BUY else -fill.fill_quantity
            new_quantity = pos.quantity + fill_qty

            if new_quantity == 0:
                # Position closed
                pos.quantity = 0
                pos.avg_entry_price = 0.0
            elif (pos.quantity > 0) == (new_quantity > 0):
                # Same direction: update average entry price
                total_cost = (pos.avg_entry_price * abs(pos.quantity)) + (fill.fill_price * abs(fill_qty))
                pos.avg_entry_price = total_cost / abs(new_quantity)
                pos.quantity = new_quantity
            else:
                # Position flipped direction
                pos.quantity = new_quantity
                pos.avg_entry_price = fill.fill_price

            pos.updated_at = datetime.utcnow()

        logger.info(
            "position.updated",
            symbol=fill.symbol,
            quantity=pos.quantity,
            avg_price=pos.avg_entry_price,
            strategy=fill.strategy_id,
        )
        return pos

    def load_from_snapshot(self, positions: list[Position]) -> None:
        """Restore positions from DB on startup."""
        for p in positions:
            if not p.is_flat:
                self._positions[p.symbol] = p
        logger.info("position_tracker.loaded", count=len(self._positions))

    def snapshot(self) -> list[Position]:
        """Return all non-flat positions for persistence."""
        return self.all_positions()
