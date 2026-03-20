from __future__ import annotations

import functools
import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

_ENABLED = True
_collector: TimingCollector | None = None


@dataclass
class TimingEntry:
    label: str
    elapsed: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)


class TimingCollector:
    def __init__(self) -> None:
        self.entries: list[TimingEntry] = []

    def record(self, label: str, elapsed: float, **extra: Any) -> None:
        self.entries.append(TimingEntry(label=label, elapsed=elapsed, extra=extra))

    def clear(self) -> None:
        self.entries.clear()


def set_timing_enabled(enabled: bool) -> None:
    global _ENABLED  # noqa: PLW0603
    _ENABLED = enabled


def is_timing_enabled() -> bool:
    return _ENABLED


def start_collecting() -> TimingCollector:
    global _collector  # noqa: PLW0603
    _collector = TimingCollector()
    return _collector


def stop_collecting() -> list[TimingEntry]:
    global _collector  # noqa: PLW0603
    entries = _collector.entries if _collector else []
    _collector = None
    return entries


def get_collector() -> TimingCollector | None:
    return _collector


@contextmanager
def log_timing(
    label: str,
    logger: Optional[logging.Logger] = None,
    level: int = logging.INFO,
    **extra_fields: Any,
):
    entry = TimingEntry(label=label, extra=extra_fields)
    if not _ENABLED:
        yield entry
        return
    t0 = time.monotonic()
    yield entry
    entry.elapsed = time.monotonic() - t0
    _log = logger or logging.getLogger(__name__)
    suffix = "".join(f", {k}={v}" for k, v in extra_fields.items())
    _log.log(level, "[TIMING] %s: %.2fs%s", label, entry.elapsed, suffix)
    if _collector is not None:
        _collector.record(label, entry.elapsed, **extra_fields)


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
            elapsed = time.monotonic() - t0
            _log.log(level, "[TIMING] %s: %.2fs", _label, elapsed)
            if _collector is not None:
                _collector.record(_label, elapsed)
            return result

        return wrapper

    return decorator
