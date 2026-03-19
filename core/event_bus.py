"""
Internal async event bus — fan-out pub/sub using asyncio.Queue.

Architecture:
  Publisher ──publish(event)──▶ EventBus ──put_nowait──▶ Queue A (subscriber 1)
                                                    ──▶ Queue B (subscriber 2)
                                                    ──▶ Queue C (subscriber 3)

Design decisions:
  - Each subscriber gets its own Queue. A slow or crashed subscriber
    does NOT block or affect other subscribers.
  - Queues are bounded (maxsize=1000). If a subscriber falls behind,
    events are dropped with a warning rather than blocking the publisher.
    This matches how real trading systems handle backpressure.
  - Events are keyed by type (type(event)). Subscribers register for
    a specific event class and only receive that type.
  - The bus is not thread-safe — all usage must be from the asyncio event loop.

Usage:
  bus = EventBus()

  # Subscriber (e.g. strategy):
  queue = bus.subscribe(MarketDataEvent)
  async def run():
      while True:
          event = await queue.get()
          handle(event)

  # Publisher (e.g. data feed):
  await bus.publish(MarketDataEvent(bar=bar))
"""
from __future__ import annotations

import asyncio
from typing import Type, TypeVar

import structlog

logger = structlog.get_logger(__name__)

T = TypeVar("T")


class EventBus:
    def __init__(self) -> None:
        # Maps event type → list of subscriber queues
        self._subscribers: dict[type, list[asyncio.Queue]] = {}
        self._publish_count: dict[str, int] = {}   # telemetry
        self._drop_count: dict[str, int] = {}       # dropped events per type

    def subscribe(self, event_type: Type[T], maxsize: int = 1000) -> asyncio.Queue:
        """
        Register a subscriber for event_type.
        Returns a Queue — the subscriber reads from it in its own coroutine.
        """
        q: asyncio.Queue = asyncio.Queue(maxsize=maxsize)
        self._subscribers.setdefault(event_type, []).append(q)
        logger.debug(
            "bus.subscribed",
            event_type=event_type.__name__,
            total_subscribers=len(self._subscribers[event_type]),
        )
        return q

    async def publish(self, event: object) -> None:
        """
        Fan out event to all registered subscribers.
        Non-blocking: if a subscriber queue is full, the event is dropped
        and a warning is logged (never blocks the publisher).
        """
        event_type = type(event)
        type_name = event_type.__name__
        queues = self._subscribers.get(event_type, [])

        self._publish_count[type_name] = self._publish_count.get(type_name, 0) + 1

        for q in queues:
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                self._drop_count[type_name] = self._drop_count.get(type_name, 0) + 1
                logger.warning(
                    "bus.event_dropped",
                    event_type=type_name,
                    queue_size=q.maxsize,
                    total_drops=self._drop_count[type_name],
                )

    def subscriber_count(self, event_type: type) -> int:
        return len(self._subscribers.get(event_type, []))

    def stats(self) -> dict:
        """Returns publish and drop counts per event type (for observability)."""
        return {
            "published": dict(self._publish_count),
            "dropped": dict(self._drop_count),
        }
