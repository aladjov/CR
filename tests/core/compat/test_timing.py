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
