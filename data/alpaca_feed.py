"""
Alpaca WebSocket market data feed.

Connects to Alpaca's real-time data stream and emits MarketDataEvent
for each bar received. Handles reconnection with exponential backoff.

Stale data detection: if no bars are received for a symbol within
stale_data_timeout_sec, a StaleDataEvent is emitted to halt signals.

Data flow:
  Alpaca WebSocket ──bar──▶ _on_bar() ──validate──▶ EventBus ──▶ Strategy
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Optional

import structlog

from config.settings import AlpacaSettings, RiskSettings
from core.event_bus import EventBus
from core.events import MarketDataEvent, StaleDataEvent, SystemEvent
from core.models import Bar

logger = structlog.get_logger(__name__)

_RECONNECT_DELAYS = [1, 2, 4, 8, 16, 30]


class AlpacaWSFeed:
    def __init__(
        self,
        settings: AlpacaSettings,
        risk_settings: RiskSettings,
        event_bus: EventBus,
    ) -> None:
        self.settings = settings
        self.stale_timeout = risk_settings.stale_data_timeout_sec
        self.bus = event_bus
        self._stream = None
        self._last_bar_time: dict[str, datetime] = {}
        self._stale_monitor_task: Optional[asyncio.Task] = None

    async def connect(self) -> None:
        """
        Connect to Alpaca data stream and subscribe to bar updates.
        Called during startup (step 7 — data feed starts LAST).
        """
        try:
            from alpaca.data.live import StockDataStream

            self._stream = StockDataStream(
                api_key=self.settings.api_key,
                secret_key=self.settings.secret_key,
                feed=self.settings.data_feed,
            )
            self._stream.subscribe_bars(self._on_bar, *self.settings.symbols)

            self._stale_monitor_task = asyncio.create_task(self._monitor_stale_data())
            asyncio.create_task(self._run_stream())

            logger.info(
                "alpaca_feed.connected",
                symbols=self.settings.symbols,
                feed=self.settings.data_feed,
            )
            await self.bus.publish(SystemEvent(
                event_type="feed_connected",
                message=f"Alpaca data feed connected: {self.settings.symbols}",
            ))

        except Exception as e:
            logger.error("alpaca_feed.connect_failed", error=str(e))
            raise

    async def disconnect(self) -> None:
        if self._stale_monitor_task:
            self._stale_monitor_task.cancel()
        if self._stream:
            await self._stream.stop_ws()
        logger.info("alpaca_feed.disconnected")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _on_bar(self, bar) -> None:
        """
        Called by Alpaca WebSocket for each new bar.
        Validates the bar, updates stale tracking, publishes event.
        """
        try:
            b = Bar(
                symbol=str(bar.symbol),
                timestamp=bar.timestamp.replace(tzinfo=None),
                open=float(bar.open),
                high=float(bar.high),
                low=float(bar.low),
                close=float(bar.close),
                volume=int(bar.volume),
            )

            if not b.is_valid():
                logger.warning(
                    "alpaca_feed.invalid_bar_dropped",
                    symbol=b.symbol,
                    close=b.close,
                )
                return

            self._last_bar_time[b.symbol] = datetime.utcnow()
            await self.bus.publish(MarketDataEvent(bar=b))

        except Exception:
            # json.JSONDecodeError, AttributeError, etc. — skip bar, keep going
            logger.exception("alpaca_feed.bar_parse_error")

    async def _monitor_stale_data(self) -> None:
        """
        Checks every 10 seconds if any symbol has gone stale.
        Emits StaleDataEvent if a symbol hasn't received a bar
        within stale_data_timeout_sec.
        """
        while True:
            await asyncio.sleep(10)
            now = datetime.utcnow()
            for symbol in self.settings.symbols:
                last = self._last_bar_time.get(symbol)
                if last is None:
                    continue
                elapsed = (now - last).total_seconds()
                if elapsed > self.stale_timeout:
                    logger.warning(
                        "alpaca_feed.stale_data",
                        symbol=symbol,
                        seconds_since_last_bar=elapsed,
                    )
                    await self.bus.publish(StaleDataEvent(
                        symbol=symbol,
                        last_bar_timestamp=last,
                        seconds_since_last_bar=elapsed,
                    ))

    async def _run_stream(self) -> None:
        """Run WebSocket stream with exponential backoff reconnect."""
        attempt = 0
        while True:
            try:
                await self._stream._run_forever()
                attempt = 0
            except Exception as e:
                delay = _RECONNECT_DELAYS[min(attempt, len(_RECONNECT_DELAYS) - 1)]
                logger.warning(
                    "alpaca_feed.disconnected",
                    error=str(e),
                    reconnect_in_sec=delay,
                )
                await self.bus.publish(SystemEvent(
                    event_type="feed_disconnected",
                    message=f"Feed disconnected, reconnecting in {delay}s",
                ))
                await asyncio.sleep(delay)
                # Re-subscribe after reconnect
                if self._stream:
                    self._stream.subscribe_bars(self._on_bar, *self.settings.symbols)
                attempt += 1
