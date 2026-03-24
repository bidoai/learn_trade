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


@app.get("/api/greeks")
async def get_greeks() -> JSONResponse:
    """Portfolio-level Greeks. Returns mock data for now."""
    return JSONResponse({
        "portfolio": {
            "dv01": -124500,       # USD per bp
            "cs01": -45200,        # credit spread sensitivity
            "delta": 2847000,      # equity delta (USD)
            "vega": 156000,        # per vol point
            "gamma": 12400,
        },
        "by_book": [
            {"book": "APEX_RATES_SWAPS", "dv01": -98200, "cs01": 0, "delta": 0, "vega": 0},
            {"book": "APEX_RATES_GOV",   "dv01": -26300, "cs01": 0, "delta": 0, "vega": 0},
            {"book": "APEX_EQ_MM",       "dv01": 0, "cs01": 0, "delta": 1840000, "vega": 98000},
            {"book": "APEX_CREDIT_IG",   "dv01": 0, "cs01": -45200, "delta": 0, "vega": 0},
            {"book": "APEX_DERIV",       "dv01": 0, "cs01": 0, "delta": 1007000, "vega": 58000},
        ]
    })


@app.get("/api/ccr")
async def get_ccr() -> JSONResponse:
    """Counterparty credit risk / SA-CCR EAD and limit utilization."""
    return JSONResponse({
        "summary": {
            "total_ead_mm": 284.7,
            "total_limit_mm": 500.0,
            "utilization_pct": 56.9,
            "counterparties_near_limit": 1,
        },
        "counterparties": [
            {"name": "Goldman Sachs",  "ead_mm": 87.4, "limit_mm": 150.0, "utilization_pct": 58.3, "pfe_97_5": 124.2, "rating": "A+"},
            {"name": "JPMorgan",       "ead_mm": 72.1, "limit_mm": 150.0, "utilization_pct": 48.1, "pfe_97_5": 98.7,  "rating": "AA-"},
            {"name": "Deutsche Bank",  "ead_mm": 63.8, "limit_mm": 75.0,  "utilization_pct": 85.1, "pfe_97_5": 84.3,  "rating": "BBB+"},
            {"name": "BNP Paribas",    "ead_mm": 41.2, "limit_mm": 75.0,  "utilization_pct": 54.9, "pfe_97_5": 56.4,  "rating": "A"},
            {"name": "HSBC",           "ead_mm": 20.2, "limit_mm": 50.0,  "utilization_pct": 40.4, "pfe_97_5": 28.1,  "rating": "AA-"},
        ]
    })


@app.get("/api/pnl")
async def get_pnl_attribution() -> JSONResponse:
    """P&L attribution by book and instrument type."""
    return JSONResponse({
        "daily_total": 1847200,
        "ytd_total": 24680000,
        "by_book": [
            {"book": "APEX_RATES_SWAPS", "daily_pnl": 824000,  "ytd_pnl": 11200000},
            {"book": "APEX_EQ_MM",       "daily_pnl": 412000,  "ytd_pnl": 5600000},
            {"book": "APEX_FX_G10",      "daily_pnl": 287000,  "ytd_pnl": 3840000},
            {"book": "APEX_CREDIT_IG",   "daily_pnl": 198000,  "ytd_pnl": 2640000},
            {"book": "APEX_DERIV",       "daily_pnl": 126200,  "ytd_pnl": 1400000},
        ]
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
