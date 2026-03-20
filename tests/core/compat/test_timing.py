from __future__ import annotations

import logging

from customer_retention.core.compat.timing import (
    get_collector,
    is_timing_enabled,
    log_timing,
    set_timing_enabled,
    start_collecting,
    stop_collecting,
    timed,
)


def test_log_timing_emits_message(caplog):
    with caplog.at_level(logging.INFO):
        with log_timing("test_block", cols=5):
            pass
    assert any("[TIMING] test_block:" in r.message and "cols=5" in r.message for r in caplog.records)


def test_log_timing_disabled_emits_nothing(caplog):
    set_timing_enabled(False)
    try:
        with caplog.at_level(logging.INFO):
            with log_timing("should_not_appear"):
                pass
        assert not any("[TIMING]" in r.message for r in caplog.records)
    finally:
        set_timing_enabled(True)


def test_timed_decorator(caplog):
    @timed(label="my_func")
    def my_func(x):
        return x + 1

    with caplog.at_level(logging.INFO):
        result = my_func(41)
    assert result == 42
    assert any("[TIMING] my_func:" in r.message for r in caplog.records)


def test_timed_decorator_disabled(caplog):
    @timed(label="no_log")
    def my_func():
        return "ok"

    set_timing_enabled(False)
    try:
        with caplog.at_level(logging.INFO):
            result = my_func()
        assert result == "ok"
        assert not any("[TIMING]" in r.message for r in caplog.records)
    finally:
        set_timing_enabled(True)


def test_is_timing_enabled():
    assert is_timing_enabled() is True
    set_timing_enabled(False)
    assert is_timing_enabled() is False
    set_timing_enabled(True)
    assert is_timing_enabled() is True


def test_collector_accumulates_log_timing_entries():
    start_collecting()
    with log_timing("batch_one", cols=3):
        pass
    with log_timing("batch_two"):
        pass
    entries = stop_collecting()
    assert len(entries) == 2
    assert entries[0].label == "batch_one"
    assert entries[0].extra == {"cols": 3}
    assert entries[1].label == "batch_two"
    assert entries[1].elapsed >= 0
    assert get_collector() is None


def test_collector_accumulates_timed_decorator():
    @timed(label="decorated")
    def add(a, b):
        return a + b

    start_collecting()
    assert add(1, 2) == 3
    entries = stop_collecting()
    assert len(entries) == 1
    assert entries[0].label == "decorated"


def test_collector_not_active_by_default():
    assert get_collector() is None
    with log_timing("orphan"):
        pass


def test_stop_collecting_without_start_returns_empty():
    entries = stop_collecting()
    assert entries == []


def test_log_timing_with_explicit_logger(caplog):
    custom_logger = logging.getLogger("test.custom")
    with caplog.at_level(logging.INFO, logger="test.custom"):
        with log_timing("explicit_logger_block", custom_logger):
            pass
    assert any(
        "[TIMING] explicit_logger_block:" in r.message and r.name == "test.custom"
        for r in caplog.records
    )


def test_log_timing_rejects_non_logger_as_logger():
    """Passing a string where a Logger is expected must raise AttributeError."""
    import pytest

    with pytest.raises(AttributeError, match="log"):
        with log_timing("label", "not_a_logger"):
            pass


def test_log_timing_yields_timing_entry():
    import time

    from customer_retention.core.compat.timing import TimingEntry

    with log_timing("yielded_block") as entry:
        assert isinstance(entry, TimingEntry)
        assert entry.label == "yielded_block"
        time.sleep(0.01)
    assert entry.elapsed >= 0.01


def test_log_timing_yields_entry_when_disabled():
    from customer_retention.core.compat.timing import TimingEntry

    set_timing_enabled(False)
    try:
        with log_timing("disabled_block") as entry:
            assert isinstance(entry, TimingEntry)
        assert entry.elapsed == 0.0
    finally:
        set_timing_enabled(True)


def test_log_timing_entry_extra_fields():
    with log_timing("extra_block", cols=10) as entry:
        pass
    assert entry.extra == {"cols": 10}
    assert entry.elapsed >= 0
