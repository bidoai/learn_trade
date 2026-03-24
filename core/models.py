"""
Domain models for the trading system.

These are the core data structures shared across all components.
All models are dataclasses or enums — no business logic here.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum, auto
from typing import Optional


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"


class OrderStatus(Enum):
    """
    Order lifecycle state machine:

    NEW ──risk_pass──▶ PENDING_SUBMIT ──sent──▶ OPEN
     │                                           │        │
    risk_fail                                 partial   full_fill
     │                                          fill       │
     ▼                                           │         ▼
    BLOCKED                                  PARTIAL    FILLED
                                                │
                                            full_fill
                                                │
                                                ▼
                                             FILLED

    OPEN/PARTIAL ──cancel_req──▶ PENDING_CANCEL ──confirmed──▶ CANCELLED
    OPEN/PARTIAL ──timeout──▶ EXPIRED
    """
    NEW = auto()
    PENDING_SUBMIT = auto()
    OPEN = auto()
    PARTIAL = auto()
    FILLED = auto()
    BLOCKED = auto()
    PENDING_CANCEL = auto()
    CANCELLED = auto()
    EXPIRED = auto()
    REJECTED = auto()


@dataclass
class Bar:
    """A single OHLCV price bar."""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int

    def is_valid(self) -> bool:
        """Returns False if any price field is NaN or non-positive."""
        import math
        return all(
            not math.isnan(v) and v > 0
            for v in (self.open, self.high, self.low, self.close)
        ) and self.volume >= 0


@dataclass
class Order:
    """
    Represents a single order throughout its lifecycle.
    Created by OMS, updated as status changes.
    """
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: int
    strategy_id: str
    order_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    status: OrderStatus = OrderStatus.NEW
    limit_price: Optional[float] = None
    filled_quantity: int = 0
    avg_fill_price: Optional[float] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    broker_order_id: Optional[str] = None  # Alpaca's order ID

    # Idempotency key: strategy_id + symbol + side + timestamp bucket (1-min)
    # Prevents duplicate submissions on retry storms
    idempotency_key: str = field(init=False)

    def __post_init__(self):
        minute_bucket = self.created_at.strftime("%Y%m%d%H%M")
        self.idempotency_key = f"{self.strategy_id}:{self.symbol}:{self.side.value}:{minute_bucket}"

    @property
    def is_terminal(self) -> bool:
        return self.status in (
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
            OrderStatus.REJECTED,
            OrderStatus.EXPIRED,
            OrderStatus.BLOCKED,
        )

    @property
    def remaining_quantity(self) -> int:
        return self.quantity - self.filled_quantity


@dataclass
class Position:
    """Current open position for a symbol."""
    symbol: str
    quantity: int          # positive = long, negative = short
    avg_entry_price: float
    strategy_id: str
    opened_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def is_flat(self) -> bool:
        return self.quantity == 0

    @property
    def market_value(self, current_price: float = 0.0) -> float:
        return self.quantity * current_price

    def unrealized_pnl(self, current_price: float) -> float:
        return (current_price - self.avg_entry_price) * self.quantity


@dataclass
class Fill:
    """A single execution fill (may be partial)."""
    order_id: str
    symbol: str
    side: OrderSide
    fill_price: float
    fill_quantity: int
    strategy_id: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    broker_fill_id: Optional[str] = None


class OptionType(Enum):
    CALL = "call"
    PUT = "put"


@dataclass
class OptionContract:
    """Defines an option position."""
    symbol: str          # underlying symbol (e.g. "AAPL")
    option_type: OptionType
    strike: float        # strike price K
    expiry: date         # expiration date
    contracts: int       # number of contracts (1 contract = 100 shares)
    entry_price: float   # premium paid/received per share

    @property
    def multiplier(self) -> int:
        return 100

    @property
    def notional(self) -> float:
        return self.entry_price * self.contracts * self.multiplier
