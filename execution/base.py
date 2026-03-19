"""Abstract base for execution engines."""
from __future__ import annotations

from abc import ABC, abstractmethod

from core.models import Order


class ExecutionEngine(ABC):
    @abstractmethod
    async def submit(self, order: Order) -> None:
        """Submit an order. Emits FillEvent(s) when filled."""
        ...

    @abstractmethod
    async def cancel(self, order_id: str) -> None:
        """Request cancellation of an open order."""
        ...

    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to broker. Called during startup."""
        ...

    @abstractmethod
    async def disconnect(self) -> None:
        """Graceful shutdown — cancel open orders first."""
        ...
