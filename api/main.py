"""
FastAPI dashboard backend.

Serves:
  GET  /api/positions     — current open positions
  GET  /api/orders        — recent orders (last 100)
  GET  /api/performance   — P&L and Sharpe
  GET  /api/risk          — risk metrics and circuit breaker state
  WS   /ws/dashboard      — real-time event stream (1 update/sec)
  GET  /                  — dashboard HTML

Bound to 127.0.0.1 only — not accessible from the network.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any

import structlog
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse

from api.websocket import DashboardBroadcaster
from oms.order_book import OrderBook
from oms.position_tracker import PositionTracker
from risk.engine import RiskEngine

logger = structlog.get_logger(__name__)

app = FastAPI(title="Trading System Dashboard", docs_url=None, redoc_url=None)

# These are injected at startup by run_live.py
_positions: PositionTracker = None
_order_book: OrderBook = None
_risk: RiskEngine = None
_broadcaster: DashboardBroadcaster = None
_last_prices: dict[str, float] = {}


def configure(
    positions: PositionTracker,
    order_book: OrderBook,
    risk: RiskEngine,
    broadcaster: DashboardBroadcaster,
    last_prices: dict[str, float],
) -> None:
    """Inject dependencies. Called by run_live.py during startup."""
    global _positions, _order_book, _risk, _broadcaster, _last_prices
    _positions = positions
    _order_book = order_book
    _risk = risk
    _broadcaster = broadcaster
    _last_prices = last_prices


# ---------------------------------------------------------------------------
# REST endpoints
# ---------------------------------------------------------------------------

@app.get("/api/positions")
async def get_positions() -> JSONResponse:
    if _positions is None:
        return JSONResponse({"error": "not_initialized"}, status_code=503)
    positions = _positions.all_positions()
    return JSONResponse([
        {
            "symbol": p.symbol,
            "quantity": p.quantity,
            "avg_entry_price": p.avg_entry_price,
            "strategy": p.strategy_id,
            "current_price": _last_prices.get(p.symbol, p.avg_entry_price),
            "unrealized_pnl": p.unrealized_pnl(_last_prices.get(p.symbol, p.avg_entry_price)),
        }
        for p in positions
    ])


@app.get("/api/orders")
async def get_orders() -> JSONResponse:
    if _order_book is None:
        return JSONResponse({"error": "not_initialized"}, status_code=503)
    orders = _order_book.all_orders()[-100:]  # last 100
    return JSONResponse([
        {
            "order_id": o.order_id,
            "symbol": o.symbol,
            "side": o.side.value,
            "quantity": o.quantity,
            "filled_quantity": o.filled_quantity,
            "avg_fill_price": o.avg_fill_price,
            "status": o.status.name,
            "strategy": o.strategy_id,
            "created_at": o.created_at.isoformat(),
        }
        for o in orders
    ])


@app.get("/api/risk")
async def get_risk() -> JSONResponse:
    if _risk is None:
        return JSONResponse({"error": "not_initialized"}, status_code=503)
    cb = _risk.circuit_breaker
    return JSONResponse({
        "circuit_breaker": cb.state.name,
        "daily_loss": cb.daily_loss,
        "daily_loss_pct": cb.daily_loss_pct,
        "max_daily_loss_pct": _risk.settings.max_daily_loss_pct,
        "available_capital": _risk._available_capital,
    })


@app.get("/")
async def dashboard() -> FileResponse:
    return FileResponse("dashboard/index.html")


# ---------------------------------------------------------------------------
# WebSocket
# ---------------------------------------------------------------------------

@app.websocket("/ws/dashboard")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    if _broadcaster:
        _broadcaster.clients.add(websocket)
    try:
        while True:
            # Keep connection alive — broadcaster pushes data, we just wait
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        if _broadcaster:
            _broadcaster.clients.discard(websocket)
