from __future__ import annotations

import os
import threading

import pytest

from customer_retention.parity.decorator import (
    APPLY_REGISTRY,
    ApplyOpDescriptor,
    active_dataset_hint,
    apply_context,
    apply_op,
    trace_active,
)
from customer_retention.parity.kinds import ApplyOpKind


@pytest.fixture(autouse=True)
def _clear_registry():
    APPLY_REGISTRY.clear()
    yield
    APPLY_REGISTRY.clear()


@pytest.fixture(autouse=True)
def _clear_trace_env(monkeypatch):
    monkeypatch.delenv("CR_PARITY_TRACE", raising=False)
    yield


class TestDecoratorRegistration:
    def test_decoration_adds_to_registry(self):
        @apply_op(kind=ApplyOpKind.LIFECYCLE_ENRICH)
        def fn(df):
            return df

        keys = [k for k in APPLY_REGISTRY if k.endswith(".fn")]
        assert len(keys) == 1
        desc = APPLY_REGISTRY[keys[0]]
        assert isinstance(desc, ApplyOpDescriptor)
        assert desc.kind is ApplyOpKind.LIFECYCLE_ENRICH

    def test_decoration_is_noop_on_input(self):
        @apply_op(kind=ApplyOpKind.LIFECYCLE_ENRICH)
        def fn(df):
            return df + 1

        assert fn(1) == 2

    def test_decorated_keeps_wraps_metadata(self):
        @apply_op(kind=ApplyOpKind.LIFECYCLE_ENRICH)
        def my_special_function(df):
            """docstring is preserved."""
            return df

        assert my_special_function.__name__ == "my_special_function"
        assert my_special_function.__doc__ == "docstring is preserved."

    def test_double_decoration_same_qualname_raises(self):
        @apply_op(kind=ApplyOpKind.LIFECYCLE_ENRICH)
        def fn(df):
            return df

        with pytest.raises(TypeError, match="already registered"):
            @apply_op(kind=ApplyOpKind.LIFECYCLE_ENRICH)  # noqa: F811
            def fn(df):  # noqa: F811
                return df

    def test_dataset_kwarg_recorded(self):
        @apply_op(
            kind=ApplyOpKind.LIFECYCLE_ENRICH,
            dataset_kwarg="dataset",
        )
        def fn(df, dataset=None):
            return df

        desc = next(iter(APPLY_REGISTRY.values()))
        assert desc.dataset_kwarg == "dataset"

    def test_capture_kwargs_recorded(self):
        @apply_op(
            kind=ApplyOpKind.TEMPORAL_LOOKBACK,
            capture_kwargs={"time_col", "intent"},
        )
        def fn(df, time_col, intent):
            return df

        desc = next(iter(APPLY_REGISTRY.values()))
        assert desc.capture_kwargs == {"time_col", "intent"}


class TestApplyContext:
    def test_outside_context_hint_is_none(self):
        assert active_dataset_hint() is None

    def test_basic_context_sets_hint(self):
        with apply_context(dataset="contract"):
            assert active_dataset_hint() == "contract"
        assert active_dataset_hint() is None

    def test_nested_contexts_inner_wins(self):
        with apply_context(dataset="contract"):
            with apply_context(dataset="account"):
                assert active_dataset_hint() == "account"
            assert active_dataset_hint() == "contract"
        assert active_dataset_hint() is None

    def test_exception_in_body_pops_context(self):
        with pytest.raises(RuntimeError):
            with apply_context(dataset="contract"):
                raise RuntimeError("boom")
        assert active_dataset_hint() is None

    def test_isolated_across_threads(self):
        seen = {}

        def worker(name):
            with apply_context(dataset=name):
                seen[name] = active_dataset_hint()

        threads = [threading.Thread(target=worker, args=(n,)) for n in ("a", "b", "c")]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert seen == {"a": "a", "b": "b", "c": "c"}


class TestTracingFlag:
    def test_inactive_by_default(self):
        assert trace_active() is False

    def test_active_when_env_set(self, monkeypatch):
        monkeypatch.setenv("CR_PARITY_TRACE", "1")
        assert trace_active() is True

    def test_other_values_count_as_off(self, monkeypatch):
        monkeypatch.setenv("CR_PARITY_TRACE", "0")
        assert trace_active() is False
        monkeypatch.setenv("CR_PARITY_TRACE", "")
        assert trace_active() is False


class TestZeroOverheadWhenTracingOff:
    def test_decorated_call_does_not_consult_trace(self, monkeypatch):
        sentinel = {"called": False}

        @apply_op(kind=ApplyOpKind.LIFECYCLE_ENRICH)
        def fn(df):
            sentinel["called"] = True
            return df

        # Trace inactive — function should call through directly
        assert os.environ.get("CR_PARITY_TRACE") is None
        result = fn("data")
        assert sentinel["called"] is True
        assert result == "data"


class TestRegistryKey:
    def test_key_is_module_plus_qualname(self):
        @apply_op(kind=ApplyOpKind.LIFECYCLE_ENRICH)
        def fn(df):
            return df

        # Module is tests.parity.test_decorator
        key = next(iter(APPLY_REGISTRY))
        assert "test_decorator" in key
        assert key.endswith(".fn")

    def test_methods_register_under_class_qualname(self):
        class Aggregator:
            @apply_op(kind=ApplyOpKind.BRONZE_AGGREGATE)
            def aggregate(self, df):
                return df

        keys = list(APPLY_REGISTRY.keys())
        assert any("Aggregator.aggregate" in k for k in keys)
