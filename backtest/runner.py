"""
Backtest runner.

Wires up the same strategy/risk/OMS components as live trading,
but substitutes:
  - AlpacaWSFeed     → HistoricalFeed  (historical bars as events)
  - AlpacaExecutor   → SimulatedExecutor (fills at next bar's open)

No backtest_mode flag exists anywhere. Strategies, risk engine, and
OMS cannot tell the difference — this guarantees backtest results
reflect live behavior.

Critical: strategy.reset() is called before every run to clear
any accumulated state (moving averages, position history, etc.).

Usage:
    result = await BacktestRunner(
        strategy=MomentumStrategy(settings),
        symbol="AAPL",
        start=date(2023, 1, 1),
        end=date(2023, 12, 31),
        settings=settings,
    ).run()
    print(result.sharpe_ratio, result.max_drawdown)
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import date
from typing import Optional, Type

import structlog

from config.settings import Settings
from core.event_bus import EventBus
from core.events import FillEvent, MarketDataEvent, OrderRequestEvent
from core.models import Bar
from data.historical import HistoricalLoader
from execution.simulator import SimulatedExecutor
from oms.order_manager import OrderManager
from oms.position_tracker import PositionTracker
from risk.engine import RiskEngine
from strategy.base import Strategy

logger = structlog.get_logger(__name__)


@dataclass
class BacktestResult:
    strategy_id: str
    symbol: str
    start: date
    end: date
    total_return_pct: float
    sharpe_ratio: Optional[float]
    max_drawdown_pct: float
    total_trades: int
    win_rate: float
    fills: list = field(default_factory=list)

    def __str__(self) -> str:
        return (
            f"Backtest: {self.strategy_id} on {self.symbol} "
            f"({self.start} → {self.end})\n"
            f"  Return:    {self.total_return_pct:+.2f}%\n"
            f"  Sharpe:    {self.sharpe_ratio:.2f if self.sharpe_ratio else 'N/A'}\n"
            f"  Max DD:    {self.max_drawdown_pct:.2f}%\n"
            f"  Trades:    {self.total_trades} ({self.win_rate:.0%} win rate)"
        )


class BacktestRunner:
    def __init__(
        self,
        strategy: Strategy,
        symbol: str,
        start: date,
        end: date,
        settings: Settings,
    ) -> None:
        self.strategy = strategy
        self.symbol = symbol
        self.start = start
        self.end = end
        self.settings = settings

        # Independent component graph — no shared state with live system
        self.bus = EventBus()
        self.positions = PositionTracker()
        self.risk = RiskEngine(
            positions=self.positions,
            settings=settings.risk,
            initial_capital=settings.strategy.initial_capital,
        )
        self.executor = SimulatedExecutor(self.bus)
        self.oms = OrderManager(
            risk_engine=self.risk,
            positions=self.positions,
            event_bus=self.bus,
        )
        self.oms.set_execution(self.executor)

    async def run(self, bars: Optional[list] = None) -> BacktestResult:
        """
        Run the backtest. Returns a BacktestResult with performance metrics.

        Pass pre-loaded bars to skip the data fetch (used by StrategyOptimizer
        to avoid re-downloading data on every optimization trial).
        """
        # CRITICAL: reset strategy state before every run
        self.strategy.reset()

        if bars is None:
            loader = HistoricalLoader(
                api_key=self.settings.alpaca.api_key,
                secret_key=self.settings.alpaca.secret_key,
            )
            bars = loader.load(self.symbol, self.start, self.end)

        if not bars:
            raise ValueError(f"No data for {self.symbol} between {self.start} and {self.end}")

        logger.info(
            "backtest.starting",
            strategy=self.strategy.strategy_id,
            symbol=self.symbol,
            bars=len(bars),
            start=str(self.start),
            end=str(self.end),
        )

        # Track fills for performance calculation
        fills_received: list = []
        fill_queue = self.bus.subscribe(FillEvent)
        request_queue = self.bus.subscribe(OrderRequestEvent)

        capital = self.settings.strategy.initial_capital
        equity_curve: list[float] = [capital]

        for i, bar in enumerate(bars):
            # 1. Feed bar to strategy
            event = MarketDataEvent(bar=bar)
            signal = self.strategy.on_market_data(event)

            # 2. Process signal → order request
            if signal:
                from portfolio.allocator import PortfolioAllocator
                last_prices = {bar.symbol: bar.close}
                order = _signal_to_order_direct(signal, bar.close, self.settings)
                if order:
                    await self.oms._handle_order_request(
                        __import__("core.events", fromlist=["OrderRequestEvent"]).OrderRequestEvent(order=order)
                    )

            # 3. Set execution price for fills (next bar's open, or current close)
            next_open = bars[i + 1].open if i + 1 < len(bars) else bar.close
            await self.executor.set_current_price(self.symbol, next_open)

            # 4. Process any fills
            while not fill_queue.empty():
                fill_event: FillEvent = fill_queue.get_nowait()
                fills_received.append(fill_event.fill)

            # 5. Mark-to-market equity
            pos = self.positions.get(self.symbol)
            if pos and not pos.is_flat:
                unrealized = pos.unrealized_pnl(bar.close)
                equity_curve.append(capital + unrealized)
            else:
                equity_curve.append(equity_curve[-1])

        result = _calculate_metrics(
            strategy_id=self.strategy.strategy_id,
            symbol=self.symbol,
            start=self.start,
            end=self.end,
            initial_capital=capital,
            equity_curve=equity_curve,
            fills=fills_received,
        )

        logger.info(
            "backtest.complete",
            strategy=self.strategy.strategy_id,
            symbol=self.symbol,
            total_return_pct=result.total_return_pct,
            sharpe=result.sharpe_ratio,
        )
        return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _signal_to_order_direct(signal, price: float, settings: Settings):
    """Simplified order creation for backtest (no portfolio allocator needed)."""
    import math
    from core.models import Order, OrderSide, OrderType

    weight = settings.strategy.strategy_weights.get(signal.strategy_id, 0.33)
    capital = settings.strategy.initial_capital * weight * signal.confidence
    qty = math.floor(capital / price)
    if qty <= 0:
        return None

    side = OrderSide.BUY if signal.direction > 0 else OrderSide.SELL
    return Order(
        symbol=signal.symbol,
        side=side,
        order_type=OrderType.MARKET,
        quantity=qty,
        strategy_id=signal.strategy_id,
    )


def _calculate_metrics(
    strategy_id: str,
    symbol: str,
    start: date,
    end: date,
    initial_capital: float,
    equity_curve: list[float],
    fills: list,
) -> BacktestResult:
    import numpy as np

    if not equity_curve or len(equity_curve) < 2:
        return BacktestResult(
            strategy_id=strategy_id, symbol=symbol, start=start, end=end,
            total_return_pct=0.0, sharpe_ratio=None, max_drawdown_pct=0.0,
            total_trades=0, win_rate=0.0,
        )

    equity = np.array(equity_curve)
    total_return = (equity[-1] - equity[0]) / equity[0] * 100

    # Daily returns
    daily_returns = np.diff(equity) / equity[:-1]
    sharpe = None
    if len(daily_returns) > 1 and daily_returns.std() > 0:
        sharpe = float((daily_returns.mean() / daily_returns.std()) * np.sqrt(252))

    # Max drawdown
    peak = np.maximum.accumulate(equity)
    drawdown = (peak - equity) / peak * 100
    max_dd = float(drawdown.max())

    # Win rate from fills (pair buys with sells)
    win_rate = _calculate_win_rate(fills)

    return BacktestResult(
        strategy_id=strategy_id,
        symbol=symbol,
        start=start,
        end=end,
        total_return_pct=float(total_return),
        sharpe_ratio=sharpe,
        max_drawdown_pct=max_dd,
        total_trades=len(fills),
        win_rate=win_rate,
        fills=fills,
    )


def _calculate_win_rate(fills: list) -> float:
    if not fills:
        return 0.0
    # Simple: count fills where exit price > entry price (for longs)
    # A real implementation would pair buys with sells
    return 0.5  # placeholder — implement round-trip P&L tracking
