from __future__ import annotations

import logging

from customer_retention.core.compat.timing import (
    is_timing_enabled,
    log_timing,
    set_timing_enabled,
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
