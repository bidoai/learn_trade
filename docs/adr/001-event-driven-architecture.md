# ADR 001: Event-Driven Architecture with Async Fan-Out Bus

**Status:** Accepted
**Date:** 2026-03-19

## Context

The trading system has multiple components that need to react to the same events:
- A market data bar arrives → strategy, audit log, and dashboard all need it
- A fill arrives → position tracker, performance tracker, and dashboard all need it

The naive approach (direct function calls between components) tightly couples them.
If the dashboard is slow, it blocks the strategy engine.

## Decision

Use an internal event bus with fan-out: each subscriber gets its own asyncio.Queue.
Publisher calls `bus.publish(event)` — non-blocking. Each subscriber runs its own
`async for` loop reading from its queue.

Key design choices:
- `asyncio.Queue(maxsize=1000)`: bounded. If a subscriber falls behind, events are
  dropped with a warning (not silently) rather than blocking the publisher.
- `put_nowait()`: publisher never waits for subscribers.
- `@dataclass(frozen=True)` on all events: subscribers cannot mutate shared state.

**Exception**: Risk check is NOT via the event bus. OMS calls `risk_engine.check()`
directly (synchronous) to prevent the race condition where two simultaneous orders
both pass risk before either updates position state. See ADR 002.

## Consequences

**Good:**
- Components are decoupled — each can be developed/tested independently
- Adding a new subscriber (e.g. a Slack alerter) requires no changes to existing code
- The audit log comes for free (EventStore subscribes to all event types)
- Slow subscribers (dashboard) cannot block the trading engine

**Bad:**
- Dropped events (QueueFull) are possible under extreme load
- Harder to trace than direct calls (require structured logging)
- asyncio single-threaded — CPU-intensive work (ML) must use run_in_executor
