"""
Per-strategy P&L attribution and kill switch.

The circuit breaker sees only net portfolio P&L. If one strategy is
bleeding badly, only this monitor can identify and halt it specifically
without stopping all strategies.
"""
from __future__ import annotations
from collections import defaultdict
from datetime import datetime, timezone
import structlog

logger = structlog.get_logger(__name__)


class StrategyMonitor:
    def __init__(self, max_strategy_loss_pct: float = 0.05, initial_capital: float = 100_000) -> None:
        self.max_strategy_loss_pct = max_strategy_loss_pct
        self.initial_capital = initial_capital
        self._pnl: dict[str, float] = defaultdict(float)
        self._halted: set[str] = set()

    def record_fill_pnl(self, strategy_id: str, pnl: float) -> None:
        """Call this with realized P&L on each fill."""
        self._pnl[strategy_id] += pnl
        loss_pct = -self._pnl[strategy_id] / self.initial_capital
        if loss_pct >= self.max_strategy_loss_pct and strategy_id not in self._halted:
            self._halted.add(strategy_id)
            logger.critical(
                "strategy_monitor.strategy_halted",
                strategy_id=strategy_id,
                cumulative_pnl=self._pnl[strategy_id],
                loss_pct=round(loss_pct * 100, 2),
            )

    def is_halted(self, strategy_id: str) -> bool:
        return strategy_id in self._halted

    def resume(self, strategy_id: str) -> None:
        """Manual resume after human review."""
        self._halted.discard(strategy_id)
        logger.info("strategy_monitor.strategy_resumed", strategy_id=strategy_id)

    def summary(self) -> dict:
        return {sid: {"pnl": pnl, "halted": sid in self._halted}
                for sid, pnl in self._pnl.items()}
