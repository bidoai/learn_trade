"""
Minimal FIX 4.2 simulator for testing FIXExecutor.

Listens on a TCP port, handles one client connection, and auto-responds
to all FIX session and application messages.

Behaviors:
  - Logon (A)             → Logon response
  - Heartbeat (0)         → Heartbeat response
  - TestRequest (1)       → Heartbeat with TestReqID echoed
  - NewOrderSingle (D)    → ExecutionReport: New, then immediate Fill
                            (or partial-fill + fill if sim.partial_fill=True)
  - OrderCancelRequest (F)→ ExecutionReport: Cancelled
  - Logout (5)            → Logout response, then close

Usage in tests:
    sim = FIXSimulator()
    await sim.start()              # binds to OS-assigned port
    port = sim.port

    executor = FIXExecutor(FIXSettings(port=port, ...), event_bus)
    await executor.connect()
    ...
    await executor.disconnect()
    await sim.stop()
"""
from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone
from typing import Optional

import simplefix


def _utcnow() -> bytes:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H:%M:%S").encode()


class FIXSimulator:
    """
    In-process FIX 4.2 simulator.

    Args:
        host: Bind address (default 127.0.0.1).
        fill_price: Price reported in ExecutionReport fills (default 100.0).
        partial_fill: If True, send a partial fill (half qty) before full fill.
        reject_next: If True, reject the next NewOrderSingle received.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        fill_price: float = 100.0,
        partial_fill: bool = False,
        reject_next: bool = False,
    ) -> None:
        self.host = host
        self.fill_price = fill_price
        self.partial_fill = partial_fill
        self.reject_next = reject_next
        self._server: Optional[asyncio.AbstractServer] = None
        self._out_seq = 1
        self.port: int = 0            # set after start()
        self.connected = False
        self._writer: Optional[asyncio.StreamWriter] = None

    async def start(self) -> None:
        """Bind to an OS-assigned port and start accepting connections."""
        self._server = await asyncio.start_server(
            self._handle_client, self.host, 0   # port 0 → OS picks
        )
        self.port = self._server.sockets[0].getsockname()[1]

    async def stop(self) -> None:
        if self._server:
            self._server.close()
            await self._server.wait_closed()
        if self._writer:
            try:
                self._writer.close()
                await self._writer.wait_closed()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _next_seq(self) -> int:
        seq = self._out_seq
        self._out_seq += 1
        return seq

    def _build_header(
        self, msg: simplefix.FixMessage, msg_type: bytes
    ) -> None:
        msg.append_pair(8, b"FIX.4.2")
        msg.append_pair(35, msg_type)
        msg.append_pair(49, b"SERVER")
        msg.append_pair(56, b"CLIENT")
        msg.append_pair(34, str(self._next_seq()).encode())
        msg.append_pair(52, _utcnow())

    async def _send(self, msg: simplefix.FixMessage) -> None:
        if self._writer:
            self._writer.write(msg.encode())
            await self._writer.drain()

    async def _handle_client(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        self._writer = writer
        self.connected = True
        parser = simplefix.FixParser()

        try:
            while True:
                data = await reader.read(4096)
                if not data:
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
            pass
        finally:
            self.connected = False
            writer.close()

    async def _dispatch(self, msg: simplefix.FixMessage) -> None:
        msg_type = msg.get(35)

        if msg_type == b"A":   # Logon
            await self._reply_logon()

        elif msg_type == b"0":  # Heartbeat
            await self._reply_heartbeat()

        elif msg_type == b"1":  # TestRequest
            await self._reply_heartbeat(test_req_id=msg.get(112))

        elif msg_type == b"5":  # Logout
            await self._reply_logout()

        elif msg_type == b"D":  # NewOrderSingle
            await self._handle_new_order(msg)

        elif msg_type == b"F":  # OrderCancelRequest
            await self._handle_cancel(msg)

    # ------------------------------------------------------------------
    # Session messages
    # ------------------------------------------------------------------

    async def _reply_logon(self) -> None:
        msg = simplefix.FixMessage()
        self._build_header(msg, b"A")
        msg.append_pair(98, b"0")
        msg.append_pair(108, b"30")
        await self._send(msg)

    async def _reply_heartbeat(
        self, test_req_id: Optional[bytes] = None
    ) -> None:
        msg = simplefix.FixMessage()
        self._build_header(msg, b"0")
        if test_req_id:
            msg.append_pair(112, test_req_id)
        await self._send(msg)

    async def _reply_logout(self) -> None:
        msg = simplefix.FixMessage()
        self._build_header(msg, b"5")
        await self._send(msg)

    # ------------------------------------------------------------------
    # Application messages
    # ------------------------------------------------------------------

    async def _handle_new_order(self, msg: simplefix.FixMessage) -> None:
        cl_ord_id = msg.get(11)
        symbol    = msg.get(55) or b"AAPL"
        side      = msg.get(54) or b"1"
        qty_raw   = msg.get(38) or b"100"
        qty       = int(qty_raw)
        broker_id = str(uuid.uuid4())[:8].encode()

        if self.reject_next:
            self.reject_next = False
            await self._send_exec_report(
                cl_ord_id, symbol, side, broker_id,
                exec_type=b"8",   # Rejected
                ord_status=b"8",
                last_px=b"0",
                last_qty=b"0",
                cum_qty=b"0",
                leaves_qty=qty_raw,
            )
            return

        # Acknowledge (ExecType=0, OrdStatus=0)
        await self._send_exec_report(
            cl_ord_id, symbol, side, broker_id,
            exec_type=b"0",
            ord_status=b"0",
            last_px=b"0",
            last_qty=b"0",
            cum_qty=b"0",
            leaves_qty=qty_raw,
        )

        fill_px = f"{self.fill_price:.2f}".encode()

        if self.partial_fill:
            half = qty // 2
            await self._send_exec_report(
                cl_ord_id, symbol, side, broker_id,
                exec_type=b"1",   # Partial fill
                ord_status=b"1",
                last_px=fill_px,
                last_qty=str(half).encode(),
                cum_qty=str(half).encode(),
                leaves_qty=str(qty - half).encode(),
            )
            # Full fill on the rest
            await self._send_exec_report(
                cl_ord_id, symbol, side, broker_id,
                exec_type=b"2",   # Fill
                ord_status=b"2",
                last_px=fill_px,
                last_qty=str(qty - half).encode(),
                cum_qty=qty_raw,
                leaves_qty=b"0",
            )
        else:
            await self._send_exec_report(
                cl_ord_id, symbol, side, broker_id,
                exec_type=b"2",   # Fill
                ord_status=b"2",
                last_px=fill_px,
                last_qty=qty_raw,
                cum_qty=qty_raw,
                leaves_qty=b"0",
            )

    async def _handle_cancel(self, msg: simplefix.FixMessage) -> None:
        orig_cl_ord_id = msg.get(41)  # OrigClOrdID
        cl_ord_id      = msg.get(11)  # ClOrdID of cancel request
        symbol         = msg.get(55) or b"AAPL"
        side           = msg.get(54) or b"1"
        qty_raw        = msg.get(38) or b"0"
        broker_id      = str(uuid.uuid4())[:8].encode()

        await self._send_exec_report(
            orig_cl_ord_id, symbol, side, broker_id,
            exec_type=b"4",   # Cancelled
            ord_status=b"4",
            last_px=b"0",
            last_qty=b"0",
            cum_qty=b"0",
            leaves_qty=b"0",
        )

    async def _send_exec_report(
        self,
        cl_ord_id: Optional[bytes],
        symbol: bytes,
        side: bytes,
        broker_id: bytes,
        exec_type: bytes,
        ord_status: bytes,
        last_px: bytes,
        last_qty: bytes,
        cum_qty: bytes,
        leaves_qty: bytes,
    ) -> None:
        exec_id = str(uuid.uuid4())[:8].encode()
        msg = simplefix.FixMessage()
        self._build_header(msg, b"8")                        # ExecutionReport
        msg.append_pair(37, broker_id)                        # OrderID
        msg.append_pair(17, exec_id)                          # ExecID
        msg.append_pair(150, exec_type)                       # ExecType
        msg.append_pair(39, ord_status)                       # OrdStatus
        msg.append_pair(11, cl_ord_id or b"UNKNOWN")          # ClOrdID
        msg.append_pair(55, symbol)                           # Symbol
        msg.append_pair(54, side)                             # Side
        msg.append_pair(31, last_px)                          # LastPx
        msg.append_pair(32, last_qty)                         # LastQty
        msg.append_pair(14, cum_qty)                          # CumQty
        msg.append_pair(151, leaves_qty)                      # LeavesQty
        msg.append_pair(60, _utcnow())                        # TransactTime
        await self._send(msg)
