"""Unit tests for `replay_registered_landing_steps`.

The helper is invoked from NB00 / NB01a between operator-side
`registry.add_landing_*` calls and downstream consumers (fingerprinting,
codegen). The behavioral surface this file pins down:

  - no-op when no registry / no landing recommendations are present;
  - idempotency — datasets that already point at a `global_temp.*` view
    are left untouched (an operator Lane-2 cell already mutated them);
  - filter-only datasets get a deterministic replay-view name
    (`<name>__landing_replay`).

The Spark / lifecycle execution paths are out of scope here because they
require a live Spark session and a real Snowflake-shaped DataFrame.
"""
from __future__ import annotations

import sys
import types
from typing import Any, Dict


class _FakeRec:
    def __init__(self, **params: Any) -> None:
        self.parameters = dict(params)


class _FakeLanding:
    def __init__(self) -> None:
        self.filters: list = []
        self.lifecycle_enrichments: list = []


class _FakeRegistry:
    def __init__(self) -> None:
        self.landing = _FakeLanding()


class _FakeNamespace:
    def __init__(self, originals: Dict[str, str]) -> None:
        self.original_datasets = dict(originals)


def test_no_op_when_registry_is_none() -> None:
    from customer_retention.runtime.replay import replay_registered_landing_steps
    datasets = {"a": "catalog.schema.a"}
    out = replay_registered_landing_steps(
        datasets=datasets, registry=None, namespace=_FakeNamespace({}),
    )
    assert out is datasets
    assert datasets == {"a": "catalog.schema.a"}


def test_no_op_when_no_landing_recommendations() -> None:
    from customer_retention.runtime.replay import replay_registered_landing_steps
    reg = _FakeRegistry()
    datasets = {"a": "catalog.schema.a"}
    out = replay_registered_landing_steps(
        datasets=datasets, registry=reg, namespace=_FakeNamespace({"a": "catalog.schema.a"}),
    )
    assert out == {"a": "catalog.schema.a"}


def test_skips_dataset_already_mutated_to_global_temp() -> None:
    """Operator Lane-2 already produced the temp view — helper must not double-apply."""
    from customer_retention.runtime.replay import replay_registered_landing_steps

    reg = _FakeRegistry()
    reg.landing.filters.append(_FakeRec(dataset="a", predicate="x IS NOT NULL"))
    datasets = {"a": "global_temp.a__lane2"}
    out = replay_registered_landing_steps(
        datasets=datasets, registry=reg,
        namespace=_FakeNamespace({"a": "catalog.schema.a"}),
    )
    assert out["a"] == "global_temp.a__lane2"


def test_filter_only_replay_uses_default_view_name(monkeypatch) -> None:
    """When only `add_landing_filter` is registered, helper picks
    `<name>__landing_replay` as the temp view name."""
    from customer_retention.runtime import replay as replay_mod

    captured: Dict[str, Any] = {}

    class _FakeDF:
        def filter(self, predicate: str) -> "_FakeDF":
            captured.setdefault("filters", []).append(predicate)
            return self

    monkeypatch.setattr(replay_mod, "__name__", replay_mod.__name__)
    fake_compat = types.ModuleType("customer_retention.core.compat")

    def _load(src: str) -> _FakeDF:
        captured["loaded_from"] = src
        return _FakeDF()

    def _register(df: _FakeDF, view_name: str, *, purpose: str) -> str:
        captured["view_name"] = view_name
        captured["purpose"] = purpose
        return f"global_temp.{view_name}"

    fake_compat.load_spark_table = _load
    fake_compat.register_temp_view = _register
    monkeypatch.setitem(sys.modules, "customer_retention.core.compat", fake_compat)

    reg = _FakeRegistry()
    reg.landing.filters.append(_FakeRec(dataset="request", predicate="ACCOUNT_ID IS NOT NULL"))

    datasets: Dict[str, Any] = {"request": "prod.salesforce.request"}
    out = replay_mod.replay_registered_landing_steps(
        datasets=datasets, registry=reg,
        namespace=_FakeNamespace({"request": "prod.salesforce.request"}),
    )
    assert out["request"] == "global_temp.request__landing_replay"
    assert captured["loaded_from"] == "prod.salesforce.request"
    assert captured["filters"] == ["ACCOUNT_ID IS NOT NULL"]
    assert captured["purpose"] == "exploration_landing_replay"
