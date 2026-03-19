"""
Start the live paper trading system.

Startup sequence (order matters — see architecture decision in docs/adr/):
  1. Database (EventStore) initialized
  2. PositionTracker loaded from DB
  3. RiskEngine validated
  4. OrderManager created
  5. AlpacaExecutor connected (fails fast on bad API key)
  6. Strategies initialized
  7. PortfolioAllocator + strategy event routing
  8. AlpacaWSFeed connected — event flow begins
  9. Dashboard (non-blocking background task)

Shutdown:
  Ctrl+C → cancel open orders → flush event store → exit

Usage:
  python scripts/run_live.py
"""
from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import structlog
import uvicorn

from api.main import app, configure
from api.websocket import DashboardBroadcaster
from audit.event_store import EventStore
from config.settings import Settings
from core.event_bus import EventBus
from core.events import FillEvent, MarketDataEvent, SignalEvent
from data.alpaca_feed import AlpacaWSFeed
from execution.alpaca import AlpacaExecutor
from oms.order_manager import OrderManager
from oms.position_tracker import PositionTracker
from risk.engine import RiskEngine
from strategy.momentum import MomentumStrategy
from strategy.mean_reversion import MeanReversionStrategy
from strategy.ml_strategy import MLStrategy

logger = structlog.get_logger(__name__)


async def startup() -> None:
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
    )

    settings = Settings()  # Fails fast if .env is missing required fields
    last_prices: dict[str, float] = {}

    # ----------------------------------------------------------------
    # Step 1: Storage
    # ----------------------------------------------------------------
    event_store = EventStore(settings.db_path)
    event_store.initialize()

    bus = EventBus()

    # ----------------------------------------------------------------
    # Step 2: State — restore positions from last session
    # ----------------------------------------------------------------
    positions = PositionTracker()
    # TODO: load persisted positions from event_store on restart
    logger.info("startup.positions_ready")

    # ----------------------------------------------------------------
    # Step 3: Risk engine
    # ----------------------------------------------------------------
    risk = RiskEngine(
        positions=positions,
        settings=settings.risk,
        initial_capital=settings.strategy.initial_capital,
    )
    risk.assert_healthy()  # Raises if config is invalid

    # ----------------------------------------------------------------
    # Step 4: OMS
    # ----------------------------------------------------------------
    oms = OrderManager(risk_engine=risk, positions=positions, event_bus=bus)

    # ----------------------------------------------------------------
    # Step 5: Execution
    # ----------------------------------------------------------------
    execution = AlpacaExecutor(settings=settings.alpaca, event_bus=bus)
    await execution.connect()  # Fails fast if API key is invalid
    oms.set_execution(execution)

    # ----------------------------------------------------------------
    # Step 6: Strategies
    # ----------------------------------------------------------------
    strategies = [
        MomentumStrategy(settings.strategy),
        MeanReversionStrategy(settings.strategy),
        MLStrategy(settings.strategy),
    ]

    # Wire strategy data feed: each strategy subscribes to MarketDataEvent
    strategy_by_id = {s.strategy_id: s for s in strategies}
    market_queues = {s.strategy_id: bus.subscribe(MarketDataEvent) for s in strategies}

    async def run_strategy(strategy_id: str) -> None:
        q = market_queues[strategy_id]
        strategy = strategy_by_id[strategy_id]
        while True:
            event: MarketDataEvent = await q.get()
            last_prices[event.symbol] = event.bar.close
            signal = strategy.on_market_data(event)
            if signal:
                await bus.publish(signal)

    # ----------------------------------------------------------------
    # Step 7: Portfolio allocator
    # ----------------------------------------------------------------
    from portfolio.allocator import PortfolioAllocator
    allocator = PortfolioAllocator(
        settings=settings.strategy,
        positions=positions,
        event_bus=bus,
        last_prices=last_prices,
    )

    # ----------------------------------------------------------------
    # Step 8: Data feed (LAST — starts event flow)
    # ----------------------------------------------------------------
    feed = AlpacaWSFeed(
        settings=settings.alpaca,
        risk_settings=settings.risk,
        event_bus=bus,
    )
    await feed.connect()

    # ----------------------------------------------------------------
    # Step 9: Dashboard (non-blocking — doesn't gate trading)
    # ----------------------------------------------------------------
    broadcaster = DashboardBroadcaster(
        event_bus=bus,
        positions=positions,
        risk=risk,
        last_prices=last_prices,
        update_interval_sec=settings.dashboard.update_interval_sec,
    )
    configure(
        positions=positions,
        order_book=oms.order_book,
        risk=risk,
        broadcaster=broadcaster,
        last_prices=last_prices,
    )

    # ----------------------------------------------------------------
    # Start all background tasks
    # ----------------------------------------------------------------
    tasks = [
        asyncio.create_task(oms.run_order_requests()),
        asyncio.create_task(oms.run_fill_processing()),
        asyncio.create_task(allocator.run()),
        asyncio.create_task(broadcaster.run()),
        *[asyncio.create_task(run_strategy(s.strategy_id)) for s in strategies],
    ]

    logger.info("startup.complete", symbols=settings.alpaca.symbols)
    print(f"\nTrading system running. Dashboard: http://{settings.dashboard.host}:{settings.dashboard.port}\n")

    # Start uvicorn in background
    config = uvicorn.Config(
        app,
        host=settings.dashboard.host,
        port=settings.dashboard.port,
        log_level="warning",
    )
    server = uvicorn.Server(config)

    try:
        await asyncio.gather(server.serve(), *tasks)
    except asyncio.CancelledError:
        pass
    finally:
        logger.info("shutdown.started")
        await execution.disconnect()
        await feed.disconnect()
        event_store.close()
        logger.info("shutdown.complete")


if __name__ == "__main__":
    try:
        asyncio.run(startup())
    except KeyboardInterrupt:
        print("\nShutting down...")
