"""Context-safe cr.log, cr.log_table, cr.in_notebook.

In notebook: `display(Markdown(...))` + `display(df)`.
In generated pipeline: stdlib logger at INFO + df.show() for tables.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

_logger = logging.getLogger("customer_retention.runtime")
_in_notebook_cache: Optional[bool] = None


def in_notebook() -> bool:
    global _in_notebook_cache
    if _in_notebook_cache is not None:
        return _in_notebook_cache
    _in_notebook_cache = _detect_notebook()
    return _in_notebook_cache


def _detect_notebook() -> bool:
    try:
        from IPython import get_ipython
    except ImportError:
        return False
    ip = get_ipython()
    if ip is None:
        return False
    shell_name = type(ip).__name__
    return shell_name in ("ZMQInteractiveShell", "Shell")


def _reset_cache_for_tests() -> None:
    global _in_notebook_cache
    _in_notebook_cache = None


def log(message: str, **kw: Any) -> None:
    if in_notebook():
        try:
            from IPython.display import Markdown, display
        except ImportError:
            _logger.info(message, extra=kw)
            return
        display(Markdown(message))
    else:
        _logger.info(message, extra=kw)


def log_table(df: Any, message: Optional[str] = None, max_rows: int = 20) -> None:
    if in_notebook():
        try:
            from IPython.display import Markdown, display
        except ImportError:
            _emit_log_table_cli(df, message, max_rows)
            return
        if message:
            display(Markdown(message))
        display(df)
    else:
        _emit_log_table_cli(df, message, max_rows)


def _emit_log_table_cli(df: Any, message: Optional[str], max_rows: int) -> None:
    if message:
        _logger.info(message)
    if hasattr(df, "show"):
        try:
            df.show(max_rows)
            return
        except (AttributeError, TypeError, RuntimeError):
            pass
    try:
        head = df.head(max_rows)
    except (AttributeError, TypeError):
        head = df
    _logger.info("%s", head)
