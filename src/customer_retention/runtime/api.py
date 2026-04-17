"""Public surface of the `cr` runtime namespace.

Users import this module via `from customer_retention.runtime import cr`
(the `cr` attribute on the package `__init__` is this module). Any symbol
not re-exported here is private to `customer_retention.runtime.*`.
"""
from __future__ import annotations

from .decorator import register
from .logging import in_notebook, log, log_table
from .registry import registry

__all__ = ["register", "log", "log_table", "in_notebook", "registry"]
