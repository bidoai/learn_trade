"""
Order state machine — enforces valid transitions.

Invalid transitions raise InvalidTransitionError immediately.
This makes bugs loud and obvious rather than silently corrupting state.

Valid transitions:
  NEW           → PENDING_SUBMIT (risk passed, being sent to broker)
  NEW           → BLOCKED        (risk failed)
  PENDING_SUBMIT → OPEN          (broker acknowledged)
  PENDING_SUBMIT → REJECTED      (broker rejected)
  OPEN          → PARTIAL        (partial fill received)
  OPEN          → FILLED         (full fill received)
  OPEN          → PENDING_CANCEL (cancel requested)
  OPEN          → EXPIRED        (timeout, no fill)
  OPEN          → REJECTED       (broker rejected post-acknowledgement)
  PARTIAL       → PARTIAL        (another partial fill)
  PARTIAL       → FILLED         (remaining quantity filled)
  PARTIAL       → PENDING_CANCEL (cancel rest of order)
  PENDING_CANCEL → CANCELLED     (cancel confirmed)
  PENDING_CANCEL → FILLED        (fill arrived before cancel processed)
"""
from __future__ import annotations

from core.models import OrderStatus


class InvalidTransitionError(Exception):
    def __init__(self, from_status: OrderStatus, to_status: OrderStatus) -> None:
        super().__init__(
            f"Invalid order state transition: {from_status.name} → {to_status.name}"
        )
        self.from_status = from_status
        self.to_status = to_status


# Maps each state to the set of valid next states
_ALLOWED_TRANSITIONS: dict[OrderStatus, set[OrderStatus]] = {
    OrderStatus.NEW: {
        OrderStatus.PENDING_SUBMIT,
        OrderStatus.BLOCKED,
    },
    OrderStatus.PENDING_SUBMIT: {
        OrderStatus.OPEN,
        OrderStatus.REJECTED,
    },
    OrderStatus.OPEN: {
        OrderStatus.PARTIAL,
        OrderStatus.FILLED,
        OrderStatus.PENDING_CANCEL,
        OrderStatus.EXPIRED,
        OrderStatus.REJECTED,
    },
    OrderStatus.PARTIAL: {
        OrderStatus.PARTIAL,
        OrderStatus.FILLED,
        OrderStatus.PENDING_CANCEL,
    },
    OrderStatus.PENDING_CANCEL: {
        OrderStatus.CANCELLED,
        OrderStatus.FILLED,   # fill can race with cancel
    },
    # Terminal states — no transitions out
    OrderStatus.FILLED: set(),
    OrderStatus.BLOCKED: set(),
    OrderStatus.CANCELLED: set(),
    OrderStatus.EXPIRED: set(),
    OrderStatus.REJECTED: set(),
}


class OrderStateMachine:
    @staticmethod
    def transition(current: OrderStatus, next_status: OrderStatus) -> OrderStatus:
        """
        Validate and apply a state transition.
        Returns next_status on success, raises InvalidTransitionError on failure.
        """
        allowed = _ALLOWED_TRANSITIONS.get(current, set())
        if next_status not in allowed:
            raise InvalidTransitionError(current, next_status)
        return next_status

    @staticmethod
    def is_terminal(status: OrderStatus) -> bool:
        return not _ALLOWED_TRANSITIONS.get(status, set())
