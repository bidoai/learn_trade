"""
Dashboard broadcaster — pushes live system state to WebSocket clients.

Subscribes to the internal event bus and fans out to all connected
dashboard clients at a maximum of 1 update per second.

This throttle prevents the dashboard from overwhelming browsers with
updates during high-frequency trading periods.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any

import structlog
from fastapi import WebSocket

from core.event_bus import EventBus
from core.events import FillEvent, OrderStatusEvent, RiskAlertEvent, SystemEvent
from oms.position_tracker import PositionTracker
from risk.engine import RiskEngine

logger = structlog.get_logger(__name__)


class DashboardBroadcaster:
    def __init__(
        self,
        event_bus: EventBus,
        positions: PositionTracker,
        risk: RiskEngine,
        last_prices: dict[str, float],
        update_interval_sec: float = 1.0,
    ) -> None:
        self.positions = positions
        self.risk = risk
        self.last_prices = last_prices
        self.update_interval = update_interval_sec
        self.clients: set[WebSocket] = set()

        # Subscribe to relevant events
        self._fill_queue = event_bus.subscribe(FillEvent)
        self._order_queue = event_bus.subscribe(OrderStatusEvent)
        self._risk_queue = event_bus.subscribe(RiskAlertEvent)
        self._system_queue = event_bus.subscribe(SystemEvent)

        # Recent events buffer for initial state on connect
        self._recent_events: list[dict] = []

    async def run(self) -> None:
        """
        Main loop: collect pending events, build snapshot, push to all clients.
        Runs at update_interval_sec cadence regardless of event volume.
        """
        while True:
            await asyncio.sleep(self.update_interval)

            if not self.clients:
                # Drain queues to prevent buildup when no clients are connected
                self._drain_queues()
                continue

            snapshot = self._build_snapshot()
            message = json.dumps(snapshot, default=str)

            dead_clients: set[WebSocket] = set()
            for ws in self.clients:
                try:
                    await ws.send_text(message)
                except Exception:
                    dead_clients.add(ws)

            self.clients -= dead_clients
            if dead_clients:
                logger.debug("broadcaster.clients_removed", count=len(dead_clients))

    def _build_snapshot(self) -> dict[str, Any]:
        """Build current system state snapshot."""
        # Drain event queues
        recent_fills = []
        while not self._fill_queue.empty():
            event: FillEvent = self._fill_queue.get_nowait()
            recent_fills.append({
                "symbol": event.fill.symbol,
                "side": event.fill.side.value,
                "price": event.fill.fill_price,
                "quantity": event.fill.fill_quantity,
                "strategy": event.fill.strategy_id,
                "timestamp": event.timestamp.isoformat(),
            })

        risk_alerts = []
        while not self._risk_queue.empty():
            alert: RiskAlertEvent = self._risk_queue.get_nowait()
            risk_alerts.append({
                "type": alert.alert_type,
                "message": alert.message,
                "timestamp": alert.timestamp.isoformat(),
            })

        system_events = []
        while not self._system_queue.empty():
            ev: SystemEvent = self._system_queue.get_nowait()
            system_events.append({"type": ev.event_type, "message": ev.message})

        # Drain order queue (not sent to dashboard, just consumed)
        while not self._order_queue.empty():
            self._order_queue.get_nowait()

        cb = self.risk.circuit_breaker
        return {
            "positions": [
                {
                    "symbol": p.symbol,
                    "quantity": p.quantity,
                    "avg_price": p.avg_entry_price,
                    "current_price": self.last_prices.get(p.symbol, p.avg_entry_price),
                    "unrealized_pnl": p.unrealized_pnl(
                        self.last_prices.get(p.symbol, p.avg_entry_price)
                    ),
                }
                for p in self.positions.all_positions()
            ],
            "risk": {
                "circuit_breaker": cb.state.name,
                "daily_loss_pct": cb.daily_loss_pct,
                "max_daily_loss_pct": self.risk.settings.max_daily_loss_pct,
                "available_capital": self.risk._available_capital,
            },
            "recent_fills": recent_fills,
            "risk_alerts": risk_alerts,
            "system_events": system_events,
        }

    def _drain_queues(self) -> None:
        for q in (self._fill_queue, self._order_queue, self._risk_queue, self._system_queue):
            while not q.empty():
                q.get_nowait()
