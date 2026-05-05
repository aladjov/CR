"""Public surface of the `cr` runtime namespace.

Users import this module via `from customer_retention.runtime import cr`
(the `cr` attribute on the package `__init__` is this module). Any symbol
not re-exported here is private to `customer_retention.runtime.*`.
"""
from __future__ import annotations

from .decorator import register
from .logging import in_notebook, log, log_table
from .registry import registry
from .replay import replay_registered_landing_steps


def mark_lane2_executed(name: str) -> None:
    """Public shortcut for `registry.mark_lane2_executed(name)` (FW-15a).

    Call this at the bottom of every imperative ``@cr.register`` cell after
    the in-cell Lane-2 action has mutated exploration state. The FW-15a
    NB00 guard inspects this flag to verify exploration parity before
    moving on to NB01-09.
    """
    registry.mark_lane2_executed(name)


__all__ = [
    "register", "log", "log_table", "in_notebook", "registry",
    "replay_registered_landing_steps", "mark_lane2_executed",
]
