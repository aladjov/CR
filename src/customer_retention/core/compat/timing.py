from __future__ import annotations

import functools
import logging
import time
from contextlib import contextmanager
from typing import Any, Callable, Optional

_ENABLED = True


def set_timing_enabled(enabled: bool) -> None:
    global _ENABLED  # noqa: PLW0603
    _ENABLED = enabled


def is_timing_enabled() -> bool:
    return _ENABLED


@contextmanager
def log_timing(
    label: str,
    logger: Optional[logging.Logger] = None,
    level: int = logging.INFO,
    **extra_fields: Any,
):
    if not _ENABLED:
        yield
        return
    t0 = time.monotonic()
    yield
    elapsed = time.monotonic() - t0
    _log = logger or logging.getLogger(__name__)
    suffix = "".join(f", {k}={v}" for k, v in extra_fields.items())
    _log.log(level, "[TIMING] %s: %.2fs%s", label, elapsed, suffix)


def timed(
    label: Optional[str] = None,
    logger_name: Optional[str] = None,
    level: int = logging.INFO,
) -> Callable:
    def decorator(fn: Callable) -> Callable:
        _label = label or fn.__qualname__

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if not _ENABLED:
                return fn(*args, **kwargs)
            _log = logging.getLogger(logger_name or fn.__module__)
            t0 = time.monotonic()
            result = fn(*args, **kwargs)
            _log.log(level, "[TIMING] %s: %.2fs", _label, time.monotonic() - t0)
            return result

        return wrapper

    return decorator
