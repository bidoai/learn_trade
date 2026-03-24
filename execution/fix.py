"""
FIX 4.2 execution engine.

Implements the ExecutionEngine interface over a TCP FIX session.
Sends orders via NewOrderSingle (D) and processes ExecutionReport (8)
messages to emit FillEvents into the event bus.

FIX 4.2 message types used:
    0  = Heartbeat
    1  = TestRequest
    3  = Reject (session-level)
    5  = Logout
    8  = ExecutionReport
    9  = OrderCancelReject
    A  = Logon
    D  = NewOrderSingle
    F  = OrderCancelRequest

Order flow:
    submit(order)
        └─▶ NewOrderSingle (D) ──TCP──▶ [FIX counterparty]
                                                │
                                    ExecutionReport (8)
                                                │
                                    _handle_execution_report()
                                                │
                                    FillEvent ──▶ EventBus ──▶ OMS

Session state machine:
    DISCONNECTED → (TCP connect) → LOGON_SENT → (Logon ack) → ACTIVE
    ACTIVE → (disconnect/Logout) → DISCONNECTED

Sequence numbers are per-session (reset on reconnect). A production
implementation would persist them to disk for gap fill recovery.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Dict, Optional

import simplefix
import structlog

from config.settings import FIXSettings
from core.event_bus import EventBus
from core.events import FillEvent, SystemEvent
from core.models import Fill, Order, OrderSide, OrderStatus, OrderType
from execution.base import ExecutionEngine
from oms.state_machine import OrderStateMachine

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# FIX 4.2 constants
# ---------------------------------------------------------------------------

MSG_HEARTBEAT        = b"0"
MSG_TEST_REQUEST     = b"1"
MSG_REJECT           = b"3"
MSG_LOGOUT           = b"5"
MSG_EXECUTION_REPORT = b"8"
MSG_ORDER_CANCEL_REJ = b"9"
MSG_LOGON            = b"A"
MSG_NEW_ORDER        = b"D"
MSG_CANCEL_REQUEST   = b"F"

# ExecType (tag 150)
EXEC_NEW          = b"0"
EXEC_PARTIAL_FILL = b"1"
EXEC_FILL         = b"2"
EXEC_CANCELLED    = b"4"
EXEC_REJECTED     = b"8"


def _utcnow() -> bytes:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H:%M:%S").encode()


# ---------------------------------------------------------------------------
# Session — sequence numbers and header building
# ---------------------------------------------------------------------------

class FIXSession:
    """
    Manages per-session state: sequence numbers and comp IDs.

    Outgoing MsgSeqNum increments on every send.
    Both counters reset to 1 on reconnect (simplified — no gap fill).
    """

    def __init__(
        self,
        sender_comp_id: str,
        target_comp_id: str,
        heartbeat_interval: int = 30,
    ) -> None:
        self.sender = sender_comp_id.encode()
        self.target = target_comp_id.encode()
        self.heartbeat_interval = heartbeat_interval
        self._out_seq = 1

    def next_seq(self) -> int:
        seq = self._out_seq
        self._out_seq += 1
        return seq

    def build_header(self, msg: simplefix.FixMessage, msg_type: bytes) -> None:
        msg.append_pair(8, b"FIX.4.2")          # BeginString
        msg.append_pair(35, msg_type)             # MsgType
        msg.append_pair(49, self.sender)          # SenderCompID
        msg.append_pair(56, self.target)          # TargetCompID
        msg.append_pair(34, str(self.next_seq()).encode())  # MsgSeqNum
        msg.append_pair(52, _utcnow())            # SendingTime

    def reset(self) -> None:
        self._out_seq = 1


# ---------------------------------------------------------------------------
# FIXExecutor
# ---------------------------------------------------------------------------

class FIXExecutor(ExecutionEngine):
    """
    FIX 4.2 execution engine.

    Args:
        settings: FIXSettings with host, port, comp IDs, heartbeat interval.
        event_bus: Shared event bus — FillEvents are published here.
    """

    def __init__(self, settings: FIXSettings, event_bus: EventBus) -> None:
        self.settings = settings
        self.bus = event_bus
        self._session = FIXSession(
            sender_comp_id=settings.sender_comp_id,
            target_comp_id=settings.target_comp_id,
            heartbeat_interval=settings.heartbeat_interval,
        )
        self._reader: Optional[asyncio.StreamReader] = None
        self._writer: Optional[asyncio.StreamWriter] = None
        self._connected = False
        self._pending: Dict[str, Order] = {}       # ClOrdID → Order
        self._hb_task: Optional[asyncio.Task] = None
        self._rx_task: Optional[asyncio.Task] = None

    # ------------------------------------------------------------------
    # ExecutionEngine interface
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Open TCP connection and complete FIX Logon handshake."""
        self._session.reset()
        self._reader, self._writer = await asyncio.open_connection(
            self.settings.host, self.settings.port
        )

        self._rx_task = asyncio.create_task(self._read_loop())
        self._hb_task = asyncio.create_task(self._heartbeat_loop())

        await self._send_logon()

        # Wait up to 5 s for Logon ack
        for _ in range(50):
            if self._connected:
                break
            await asyncio.sleep(0.1)

        if not self._connected:
            await self._cleanup()
            raise TimeoutError(
                f"FIX Logon timed out connecting to "
                f"{self.settings.host}:{self.settings.port}"
            )

        logger.info(
            "fix.connected",
            host=self.settings.host,
            port=self.settings.port,
            sender=self.settings.sender_comp_id,
            target=self.settings.target_comp_id,
        )
        await self.bus.publish(
            SystemEvent(event_type="feed_connected", message="FIX executor connected")
        )

    async def disconnect(self) -> None:
        """Send FIX Logout and close the TCP connection."""
        if self._writer and self._connected:
            await self._send_logout()
            await asyncio.sleep(0.1)   # brief grace period for logout ack

        self._connected = False
        await self._cleanup()
        logger.info("fix.disconnected")

    async def submit(self, order: Order) -> None:
        """
        Send NewOrderSingle (D).

        The order should already be in PENDING_SUBMIT state (set by OMS).
        On send, transitions to OPEN. Fills arrive asynchronously via
        ExecutionReport and emit FillEvents.
        """
        if not self._connected:
            logger.error("fix.submit_without_connection", order_id=order.order_id)
            return

        self._pending[order.order_id] = order

        msg = simplefix.FixMessage()
        self._session.build_header(msg, MSG_NEW_ORDER)
        msg.append_pair(11, order.order_id.encode())        # ClOrdID
        msg.append_pair(55, order.symbol.encode())           # Symbol
        msg.append_pair(54, b"1" if order.side == OrderSide.BUY else b"2")  # Side
        msg.append_pair(38, str(order.quantity).encode())    # OrderQty
        msg.append_pair(60, _utcnow())                       # TransactTime

        if order.order_type == OrderType.MARKET:
            msg.append_pair(40, b"1")                        # OrdType = Market
        else:
            msg.append_pair(40, b"2")                        # OrdType = Limit
            msg.append_pair(44, f"{order.limit_price:.4f}".encode())  # Price

        await self._send(msg)
        order.status = OrderStateMachine.transition(order.status, OrderStatus.OPEN)

        logger.info(
            "fix.order_submitted",
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side.value,
            quantity=order.quantity,
            order_type=order.order_type.value,
        )

    async def cancel(self, order_id: str) -> None:
        """Send OrderCancelRequest (F) for a previously submitted order."""
        if not self._connected:
            return

        original = self._pending.get(order_id)
        if original is None:
            logger.warning("fix.cancel_unknown_order", order_id=order_id)
            return

        msg = simplefix.FixMessage()
        self._session.build_header(msg, MSG_CANCEL_REQUEST)
        msg.append_pair(41, order_id.encode())                    # OrigClOrdID
        msg.append_pair(11, f"{order_id}_CXL".encode())           # ClOrdID (new cancel req)
        msg.append_pair(55, original.symbol.encode())              # Symbol
        msg.append_pair(54, b"1" if original.side == OrderSide.BUY else b"2")
        msg.append_pair(38, str(original.quantity).encode())       # OrderQty
        msg.append_pair(60, _utcnow())                             # TransactTime

        await self._send(msg)
        logger.info("fix.cancel_sent", order_id=order_id)

    # ------------------------------------------------------------------
    # Send helpers
    # ------------------------------------------------------------------

    async def _send(self, msg: simplefix.FixMessage) -> None:
        if self._writer is None:
            return
        self._writer.write(msg.encode())
        await self._writer.drain()

    async def _send_logon(self) -> None:
        msg = simplefix.FixMessage()
        self._session.build_header(msg, MSG_LOGON)
        msg.append_pair(98, b"0")    # EncryptMethod = None
        msg.append_pair(108, str(self._session.heartbeat_interval).encode())  # HeartBtInt
        await self._send(msg)

    async def _send_logout(self) -> None:
        msg = simplefix.FixMessage()
        self._session.build_header(msg, MSG_LOGOUT)
        await self._send(msg)

    async def _send_heartbeat(self, test_req_id: Optional[bytes] = None) -> None:
        msg = simplefix.FixMessage()
        self._session.build_header(msg, MSG_HEARTBEAT)
        if test_req_id:
            msg.append_pair(112, test_req_id)  # TestReqID echo
        await self._send(msg)

    # ------------------------------------------------------------------
    # Receive loop
    # ------------------------------------------------------------------

    async def _read_loop(self) -> None:
        """Read bytes from the TCP stream and dispatch complete FIX messages."""
        parser = simplefix.FixParser()
        try:
            while True:
                data = await self._reader.read(4096)
                if not data:
                    logger.warning("fix.connection_closed_by_peer")
                    self._connected = False
                    break
                parser.append_buffer(data)
                while True:
                    fix_msg = parser.get_message()
                    if fix_msg is None:
                        break
                    await self._dispatch(fix_msg)
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("fix.read_loop_error")
            self._connected = False

    async def _dispatch(self, msg: simplefix.FixMessage) -> None:
        """Route an incoming FIX message to the appropriate handler."""
        msg_type = msg.get(35)

        if msg_type == MSG_LOGON:
            self._connected = True
            logger.debug("fix.logon_ack_received")

        elif msg_type == MSG_LOGOUT:
            self._connected = False
            logger.info("fix.logout_received")

        elif msg_type == MSG_HEARTBEAT:
            pass  # no action required

        elif msg_type == MSG_TEST_REQUEST:
            # Counterparty is checking we're alive — reply with Heartbeat
            await self._send_heartbeat(test_req_id=msg.get(112))

        elif msg_type == MSG_EXECUTION_REPORT:
            await self._handle_execution_report(msg)

        elif msg_type == MSG_ORDER_CANCEL_REJ:
            cl_ord_id = msg.get(11)
            logger.warning(
                "fix.cancel_rejected",
                order_id=cl_ord_id.decode() if cl_ord_id else "?",
            )

        elif msg_type == MSG_REJECT:
            logger.warning("fix.session_reject", ref_msg_type=msg.get(372))

        else:
            logger.debug("fix.unhandled_msg_type", msg_type=msg_type)

    async def _handle_execution_report(self, msg: simplefix.FixMessage) -> None:
        """
        Parse ExecutionReport (8) and emit FillEvent on fills.

        ExecType values that matter:
            0 = New          (acknowledged, no fill yet)
            1 = PartialFill  → emit FillEvent
            2 = Fill         → emit FillEvent, remove from pending
            4 = Cancelled    → update order status
            8 = Rejected     → update order status
        """
        exec_type = msg.get(150)
        cl_ord_id_raw = msg.get(11)

        if cl_ord_id_raw is None:
            return

        cl_ord_id = cl_ord_id_raw.decode()
        order = self._pending.get(cl_ord_id)

        if exec_type == EXEC_REJECTED:
            if order:
                order.status = OrderStateMachine.transition(order.status, OrderStatus.REJECTED)
                self._pending.pop(cl_ord_id, None)
            logger.warning("fix.order_rejected", order_id=cl_ord_id)
            return

        if exec_type == EXEC_CANCELLED:
            if order:
                # OPEN → PENDING_CANCEL → CANCELLED (state machine requires the hop)
                if order.status == OrderStatus.OPEN:
                    order.status = OrderStateMachine.transition(
                        order.status, OrderStatus.PENDING_CANCEL
                    )
                order.status = OrderStateMachine.transition(
                    order.status, OrderStatus.CANCELLED
                )
                self._pending.pop(cl_ord_id, None)
            return

        if exec_type not in (EXEC_PARTIAL_FILL, EXEC_FILL):
            return  # New / Replaced / etc. — no fill to emit

        if order is None:
            logger.warning("fix.fill_for_unknown_order", order_id=cl_ord_id)
            return

        last_px_raw  = msg.get(31)   # LastPx  — this fill's price
        last_qty_raw = msg.get(32)   # LastQty — this fill's quantity
        exec_id_raw  = msg.get(17)   # ExecID  — broker's fill ID

        if last_px_raw is None or last_qty_raw is None:
            logger.error("fix.execution_report_missing_price_qty", order_id=cl_ord_id)
            return

        fill = Fill(
            order_id=cl_ord_id,
            symbol=order.symbol,
            side=order.side,
            fill_price=float(last_px_raw),
            fill_quantity=int(last_qty_raw),
            strategy_id=order.strategy_id,
            broker_fill_id=exec_id_raw.decode() if exec_id_raw else None,
        )

        if exec_type == EXEC_FILL:
            order.status = OrderStateMachine.transition(order.status, OrderStatus.FILLED)
            self._pending.pop(cl_ord_id, None)
        else:
            order.status = OrderStatus.PARTIAL

        await self.bus.publish(FillEvent(fill=fill))

        logger.info(
            "fix.fill_received",
            order_id=cl_ord_id,
            symbol=fill.symbol,
            fill_price=fill.fill_price,
            fill_quantity=fill.fill_quantity,
            exec_type="fill" if exec_type == EXEC_FILL else "partial_fill",
        )

    # ------------------------------------------------------------------
    # Heartbeat loop
    # ------------------------------------------------------------------

    async def _heartbeat_loop(self) -> None:
        """Send FIX Heartbeat every heartbeat_interval seconds."""
        try:
            while True:
                await asyncio.sleep(self._session.heartbeat_interval)
                if self._connected:
                    await self._send_heartbeat()
        except asyncio.CancelledError:
            pass

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    async def _cleanup(self) -> None:
        if self._hb_task and not self._hb_task.done():
            self._hb_task.cancel()
        if self._rx_task and not self._rx_task.done():
            self._rx_task.cancel()
        if self._writer:
            try:
                self._writer.close()
                await self._writer.wait_closed()
            except Exception:
                pass
        self._reader = None
        self._writer = None
