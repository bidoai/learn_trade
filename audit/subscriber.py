"""
Audit subscriber — drains the event bus into the EventStore.

Subscribes to all significant event types and calls event_store.append()
for each one. Runs as a background asyncio task.

Events logged:
  FillEvent           — every fill (core audit trail)
  SignalEvent         — strategy decisions
  OrderApprovedEvent  — risk passed
  OrderBlockedEvent   — risk rejected (why orders were blocked)
  OrderStatusEvent    — order lifecycle transitions
  RiskAlertEvent      — risk limit warnings
  SystemEvent         — startup/shutdown lifecycle
"""
from __future__ import annotations

import asyncio

import structlog

from audit.event_store import EventStore
from core.event_bus import EventBus
from core.events import (
    FillEvent,
    OrderApprovedEvent,
    OrderBlockedEvent,
    OrderStatusEvent,
    RiskAlertEvent,
    SignalEvent,
    SystemEvent,
)

logger = structlog.get_logger(__name__)

_AUDIT_EVENT_TYPES = [
    FillEvent,
    SignalEvent,
    OrderApprovedEvent,
    OrderBlockedEvent,
    OrderStatusEvent,
    RiskAlertEvent,
    SystemEvent,
]


class AuditSubscriber:
    """
    Subscribes to all significant event types and persists them to EventStore.

    Usage:
        subscriber = AuditSubscriber(event_store, event_bus)
        task = asyncio.create_task(subscriber.run())
    """

    def __init__(self, event_store: EventStore, event_bus: EventBus) -> None:
        self._store = event_store
        self._queues = [
            (event_type, event_bus.subscribe(event_type))
            for event_type in _AUDIT_EVENT_TYPES
        ]

    async def run(self) -> None:
        """
        Fan-in: drain all subscribed queues and append each event to the store.
        One asyncio.Task per event type so a slow DB write on one type
        doesn't delay another.
        """
        drain_tasks = [
            asyncio.create_task(self._drain(event_type.__name__, q))
            for event_type, q in self._queues
        ]
        logger.info(
            "audit_subscriber.started",
            event_types=[et.__name__ for et in _AUDIT_EVENT_TYPES],
        )
        await asyncio.gather(*drain_tasks)

    async def _drain(self, event_type_name: str, queue: asyncio.Queue) -> None:
        while True:
            event = await queue.get()
            self._store.append(event)
            logger.debug("audit_subscriber.persisted", event_type=event_type_name)
