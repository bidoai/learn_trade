# ADR 002: Risk Engine as Synchronous Gatekeeper

**Status:** Accepted
**Date:** 2026-03-19

## Context

The risk engine must approve every order before it reaches the broker.
The question: should it subscribe to order events on the bus (async)
or be called directly by the OMS (sync)?

## The Race Condition

If risk were async:
1. Strategy A submits an order for AAPL (50 shares, $5k)
2. Strategy B simultaneously submits an order for AAPL (50 shares, $5k)
3. Risk receives both events. Checks A: position is 0, $5k < max. Approves.
4. Risk checks B: position is still 0 (fill from A hasn't arrived yet). Approves.
5. Result: $10k in AAPL, exceeding the $5k limit.

## Decision

OMS calls `risk_engine.check(order)` synchronously before every submission.
Risk is not an event bus subscriber for pre-trade checks.

The call is wrapped in `try/except` with a fail-CLOSED default: if the risk
engine throws any exception, the order is blocked (not approved).

Post-trade risk updates (circuit breaker, P&L tracking) can remain event-driven
since they don't gate order submission.

## Consequences

**Good:**
- Race condition eliminated
- Risk is always consistent with position state
- Simple to reason about — one order processed at a time

**Bad:**
- Slightly less decoupled (OMS knows about RiskEngine directly)
- Pre-trade risk is a latency bottleneck (acceptable for a learning system)
