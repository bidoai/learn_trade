"""
All events in the trading system.

Events flow through the EventBus. Every component that needs to react
to something subscribes to the relevant event type.

Design rules:
- All events are frozen dataclasses (immutable — subscribers cannot mutate shared state)
- All events carry a timestamp
- Events are keyed by type in the EventBus (no inheritance hierarchy needed)

Event flow:
  AlpacaWSFeed ──▶ MarketDataEvent ──▶ Strategy
  Strategy ──▶ SignalEvent ──▶ PortfolioManager
  PortfolioManager ──▶ OrderRequestEvent ──▶ OMS
  OMS ──▶ OrderApprovedEvent / OrderBlockedEvent ──▶ Execution / AuditLog
  Execution ──▶ FillEvent ──▶ PositionTracker, PerformanceTracker, Dashboard
  RiskEngine ──▶ RiskAlertEvent ──▶ Dashboard, AuditLog
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from core.models import Bar, Fill, Order, OrderStatus


# ---------------------------------------------------------------------------
# Market Data Events
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MarketDataEvent:
    """Emitted by AlpacaWSFeed (live) or HistoricalFeed (backtest) for each bar."""
    bar: Bar
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @property
    def symbol(self) -> str:
        return self.bar.symbol


@dataclass(frozen=True)
class StaleDataEvent:
    """
    Emitted when no bars have been received for a symbol within the
    stale_data_timeout_sec threshold. Signals the strategy engine to
    halt signal generation for this symbol.
    """
    symbol: str
    last_bar_timestamp: datetime
    seconds_since_last_bar: float
    timestamp: datetime = field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Strategy / Signal Events
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SignalEvent:
    """
    Emitted by a Strategy when it wants to take a position.

    direction: -1.0 = full short, 0.0 = flat/close, +1.0 = full long
    confidence: 0.0 to 1.0 (used by PortfolioManager for sizing)
    """
    strategy_id: str
    symbol: str
    direction: float    # -1.0 to +1.0
    confidence: float   # 0.0 to 1.0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    reason: Optional[str] = None  # human-readable explanation

    def __post_init__(self):
        if not -1.0 <= self.direction <= 1.0:
            raise ValueError(f"direction must be in [-1, 1], got {self.direction}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")


# ---------------------------------------------------------------------------
# Order Events
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OrderRequestEvent:
    """
    Emitted by PortfolioManager to request an order.
    OMS consumes this, runs the sync risk check, then emits
    OrderApprovedEvent or OrderBlockedEvent.
    """
    order: Order
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass(frozen=True)
class OrderApprovedEvent:
    """Risk check passed. OMS has submitted the order to execution."""
    order: Order
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass(frozen=True)
class OrderBlockedEvent:
    """
    Risk check failed. Order was NOT submitted to execution.
    reason explains which check failed (e.g. 'circuit_breaker_engaged',
    'max_position_exceeded', 'insufficient_capital').
    """
    order: Order
    reason: str
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass(frozen=True)
class OrderStatusEvent:
    """
    Emitted by OMS whenever an order transitions state.
    Consumed by: AuditLog, Dashboard.
    """
    order_id: str
    symbol: str
    old_status: OrderStatus
    new_status: OrderStatus
    timestamp: datetime = field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Fill Events
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FillEvent:
    """
    Emitted by AlpacaExecutor or SimulatedExecutor when a fill arrives.
    Consumed by: PositionTracker, PerformanceTracker, AuditLog, Dashboard.
    """
    fill: Fill
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @property
    def symbol(self) -> str:
        return self.fill.symbol

    @property
    def strategy_id(self) -> str:
        return self.fill.strategy_id


# ---------------------------------------------------------------------------
# Risk Events
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RiskAlertEvent:
    """
    Emitted by RiskEngine when a risk threshold is breached.
    alert_type: 'circuit_breaker_triggered', 'var_breach', 'concentration_limit',
                'circuit_breaker_reset'
    """
    alert_type: str
    message: str
    current_value: float
    threshold: float
    timestamp: datetime = field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Options Events
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GreeksEvent:
    """Published when Greeks are computed for an options position."""
    symbol: str
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float
    timestamp: datetime = field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# System Events
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SystemEvent:
    """
    Lifecycle events: 'startup_complete', 'shutdown_initiated', 'reconnecting',
    'feed_connected', 'feed_disconnected'.
    """
    event_type: str
    message: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
