# Trading System — CLAUDE.md

Hedge fund trading system learning platform. Python/asyncio, paper trading via Alpaca.

## Architecture Decisions

All key decisions are in `docs/adr/`. Summary:

1. **Event bus**: custom asyncio fan-out (`core/event_bus.py`). Queue per subscriber, `maxsize=1000`.
2. **Risk gating**: sync call from OMS, not async subscriber (`oms/order_manager.py`).
3. **No mode flag**: same code for backtest and live. BacktestRunner swaps I/O only (`backtest/runner.py`).
4. **Config**: central Pydantic settings in `config/settings.py`. Env vars use `ALPACA__API_KEY` format.
5. **Events**: all `@dataclass(frozen=True)`. See `core/events.py`.

## Critical Invariants — Never Break These

- `strategy.reset()` MUST be called before every backtest run
- `RiskEngine.check()` is synchronous — never make it async
- `EventBus.publish()` is non-blocking — never use `await queue.put()` in the publisher
- `FeatureBuilder.build()` asserts `cutoff_date` — never bypass this in ML training
- Dashboard binds to `127.0.0.1` only — never `0.0.0.0`
- API keys live in `.env` only — never hardcode credentials

## Running

```bash
# Install
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Configure
cp .env.example .env
# Edit .env with your Alpaca paper trading credentials

# Live paper trading
python scripts/run_live.py

# Backtest
python scripts/run_backtest.py --strategy momentum --symbol AAPL --start 2023-01-01 --end 2023-12-31

# Tests
pytest tests/ -v
```

## Project Structure

```
core/           Events, event bus, domain models
config/         Pydantic settings (reads .env)
data/           Market data feeds (Alpaca WS + historical loader)
strategy/       Abstract base + concrete strategies
portfolio/      Capital allocation and position sizing
risk/           Pre-trade checks, circuit breaker, VaR
oms/            Order lifecycle, position tracking, order book
execution/      Alpaca paper trading + simulated executor
backtest/       Backtesting runner (identical code to live)
audit/          Append-only SQLite event log
api/            FastAPI dashboard backend
dashboard/      Single-page HTML dashboard
ml/             ML strategy pipeline (feature eng, training, inference)
tests/          Test suite (synthetic fixtures, no real API calls)
docs/adr/       Architecture Decision Records
scripts/        Entry points (run_live.py, run_backtest.py)
```

## Testing Notes

- Tests use synthetic market data from `tests/fixtures/market_data.py` — no real API calls
- `pytest-asyncio` required for async tests
- Run `pytest tests/test_strategy.py -v` for strategy-specific tests
- The backtest determinism test (`test_backtest_deterministic`) is the most important test — it catches state leakage

## TODO / Next Steps

See `TODOS.md` for explicitly deferred items.
Immediate next:
- `ml/` module: ML strategy with FeatureBuilder + TimeSeriesSplit
- `audit/` subscriber wiring: EventStore should subscribe to all event types in run_live.py
- Position persistence: load/save positions on restart
