# Trading System — TODOS

Items explicitly deferred from the initial CEO plan review (2026-03-18).
Each item has been reviewed and consciously deferred — not forgotten.

---

## P2 — High Value, Phase 2

### ~~FIX Protocol Implementation~~ ✓ DONE
**Shipped:** `execution/fix.py` — `FIXExecutor(ExecutionEngine)` using FIX 4.2 over TCP.
`FIXSession` manages sequence numbers and header building. Full message handling:
Logon/Logout, Heartbeat, NewOrderSingle, OrderCancelRequest, ExecutionReport (fills/rejects).
`tests/fixtures/fix_simulator.py` — in-process simulator (immediate fills, partial fills,
rejects). 13 passing tests covering all order lifecycle paths.

---

### ~~Strategy Optimization Framework~~ ✓ DONE
**Shipped:** `backtest/optimizer.py` — `StrategyOptimizer` with Optuna Bayesian search,
walk-forward expanding-window validation, IS vs OOS Sharpe overfitting detection.
`scripts/optimize.py` — CLI entry point for momentum and mean-reversion strategies.
11 passing tests in `tests/test_optimizer.py`.

---

## System Fixes — ✓ DONE (2026-03-24)

### ~~Audit subscriber wiring~~ ✓ DONE
**Shipped:** `audit/subscriber.py` — `AuditSubscriber` subscribes to FillEvent, SignalEvent,
OrderApprovedEvent, OrderBlockedEvent, OrderStatusEvent, RiskAlertEvent, SystemEvent and
persists each to SQLite via EventStore. Wired in `run_live.py` as a background task.

### ~~Position persistence~~ ✓ DONE
**Shipped:** EventStore now has a `positions` table with `save_snapshot()` / `load_snapshot()`.
Positions are saved after every fill and restored on startup — state survives restarts.

### ~~Risk engine price lookup fix~~ ✓ DONE
**Shipped:** RiskEngine now accepts `last_prices: dict` and checks it first in `_get_last_price()`.
Real market prices (from the live data feed) are used for position sizing. Falls back to
avg_entry_price if no live price is available, then to $1.0. Fixes the bug where a 1000-share
order of a $200 stock would pass size checks as if the stock cost $1.

16 new tests in `tests/test_audit_and_persistence.py`. All 117 tests passing.

---

## P3 — High Value, Phase 3

### Options / Greeks Support
**What:** Extend the system to trade options — Black-Scholes pricing, Greeks (delta, gamma,
vega, theta), delta hedging strategy, and volatility surface modeling.
**Why:** Many hedge funds run options books. Understanding Greeks is foundational quant
knowledge. Delta hedging teaches the relationship between options and their underlyings.
**Pros:** Major learning milestone; opens up a whole new class of strategies.
**Cons:** Significant complexity increase. Options OMS is more complex than equity OMS
(expiry dates, strikes, multiple legs). QuantLib can accelerate pricing.
**Context:** The event bus architecture supports options — just add new event types
(OptionsQuoteEvent, GreeksEvent). The OMS needs extension for multi-leg orders.
Start with single-leg calls/puts before spreads.
**Effort:** XL (human: 1-2 months / CC: ~3 hours)
**Priority:** P3
**Depends on:** Equity system complete and stable

---

## Out of Scope (explicitly decided)
- Co-location / HFT latency engineering (different problem domain)
- Real (live-money) trading (learning system only)
- Cloud deployment / Docker Compose (local development only)
- Multi-asset class (crypto, futures, FX) — extend after equities proven
