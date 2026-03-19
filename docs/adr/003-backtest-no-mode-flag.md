# ADR 003: Backtest Uses Identical Components — No Mode Flag

**Status:** Accepted
**Date:** 2026-03-19

## Context

The #1 mistake in trading system design: separate code paths for backtest and live.
When they diverge, backtest results don't predict live behavior.

## Decision

`BacktestRunner` wires the same Strategy, RiskEngine, and OrderManager as the live
system, but substitutes:
- `AlpacaWSFeed` → `HistoricalFeed` (historical bars replayed as MarketDataEvents)
- `AlpacaExecutor` → `SimulatedExecutor` (fills at next bar's open price)

There is **no `backtest_mode: bool` flag** anywhere in Strategy, RiskEngine, or
OrderManager. These components cannot tell whether they're in backtest or live mode.
This is intentional.

`strategy.reset()` is called by `BacktestRunner` before every run. This clears
the rolling bar window and any accumulated indicator state. If `reset()` is not
called between runs, state from run N contaminates run N+1.

## Consequences

**Good:**
- Backtest results are meaningful predictors of live behavior
- Strategies and risk logic can be tested in backtest mode with confidence
- Adding a new component (e.g. cost model for commissions) affects both modes automatically

**Bad:**
- `SimulatedExecutor` fills at next bar's open (not the signal bar) — this is correct
  (you can't fill at the same bar that generated the signal) but requires understanding
- Backtest doesn't model slippage or market impact (acceptable for a learning system)
