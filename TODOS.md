# Trading System — TODOS

Items explicitly deferred from the initial CEO plan review (2026-03-18).
Each item has been reviewed and consciously deferred — not forgotten.

---

## P2 — High Value, Phase 2

### FIX Protocol Implementation
**What:** Implement a FIX (Financial Information eXchange) session layer and order router
as an alternative execution transport to Alpaca REST.
**Why:** FIX is the industry-standard messaging protocol used by every professional broker
and exchange. Understanding it is career-critical for anyone working in hedge fund technology.
**Pros:** Deep dive into the actual wire protocol; makes the system closer to production-grade.
**Cons:** Significant complexity; requires a FIX engine library (e.g., QuickFIX/Python);
needs a FIX-capable counterparty to test against (can use a simulator).
**Context:** The execution layer is already abstracted behind an `ExecutionEngine` interface.
Adding FIX means implementing a new `FIXExecutor` class. Start by reading the FIX 4.2/4.4 spec.
**Effort:** L (human: 2 weeks / CC: ~2 hours)
**Priority:** P2
**Depends on:** Core system complete and stable

---

### Strategy Optimization Framework
**What:** A system to automatically tune strategy parameters (moving average windows,
ML hyperparameters, etc.) using grid search or Bayesian optimization over historical data.
**Why:** Teaches the quant research workflow of parameter discovery. Walk-forward validation
is a critical concept — prevents in-sample overfitting that makes backtests look great but
live trading fail.
**Pros:** Significant learning value; teaches the full quant research cycle.
**Cons:** Must implement walk-forward validation correctly or results will be meaningless.
Risk of overfitting is real and instructive when you experience it.
**Context:** The backtest runner is the foundation. `StrategyOptimizer` wraps it with a
parameter grid, runs N backtests, and aggregates Sharpe ratios. Use Optuna for Bayesian
optimization. Always use out-of-sample test sets.
**Effort:** M (human: 1 week / CC: ~30 min)
**Priority:** P2
**Depends on:** Backtest engine complete

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
