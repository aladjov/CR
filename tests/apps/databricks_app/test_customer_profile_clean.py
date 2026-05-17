"""Regression tests for ``customer_profile._clean``.

The L4 customer card reads via ``cur.fetchall_arrow().to_pandas()``, which
lands Spark ``ARRAY<STRUCT<...>>`` columns (``account_top_shap_features``,
``alternate_playbooks``, etc.) as ``numpy.ndarray`` of dicts. Pybars
evaluates ``{{#if x}}`` by calling ``bool(x)``, and a numpy ndarray with
more than one element raises ``ValueError: The truth value of an array
with more than one element is ambiguous`` -- which kills the whole render
and surfaces the "Showing raw data instead." fallback.

These tests pin the cleaner's contract: every nested array becomes a
plain Python list, every nested struct becomes a plain dict, NaN scalars
become None, and the result feeds pybars without raising.
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _load_clean():
    """Import ``customer_profile._clean`` without dragging in streamlit /
    databricks-sdk (the live app surface), which the test venv intentionally
    omits. Strips the module's runtime-only top-level imports first.
    """
    src_path = (
        Path(__file__).resolve().parents[3]
        / "apps"
        / "databricks_app"
        / "src"
        / "customer_profile.py"
    )
    raw = src_path.read_text(encoding="utf-8")

    # Stub heavy deps so importing the module doesn't fail in this venv.
    for name in ("streamlit", "databricks", "databricks.sql", "databricks.sdk", "databricks.sdk.core"):
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.modules["databricks.sdk.core"].Config = lambda *a, **kw: types.SimpleNamespace(  # type: ignore[attr-defined]
        host="", authenticate=lambda: None,
    )

    # Strip leading runtime-only imports so the module loads in isolation.
    sanitised = []
    for line in raw.splitlines(keepends=True):
        if line.startswith("import streamlit") or line.startswith("from databricks"):
            continue
        if line.startswith("from . import data, state"):
            continue
        if line.startswith("from .config import load_config"):
            continue
        if line.startswith("from .template import "):
            continue
        sanitised.append(line)
    code = "".join(sanitised)

    spec = importlib.util.spec_from_loader("customer_profile_under_test", loader=None)
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    mod.__dict__["pd"] = pd
    mod.__dict__["Any"] = object
    mod.__dict__["Optional"] = object
    # No-op ``st.cache_data`` so module-level decorations (e.g. the
    # connection-shared ``_fetch_data_source_cached``) can evaluate
    # without a real streamlit install. The decorator passes the
    # function through unchanged, which is fine -- _clean and the other
    # tested helpers don't depend on caching.
    class _StStub:
        @staticmethod
        def cache_data(*_args, **_kwargs):
            def _wrap(fn):
                return fn
            return _wrap
    mod.__dict__["st"] = _StStub()
    # ``data`` is referenced by the connection-shared fetcher.
    mod.__dict__["data"] = types.SimpleNamespace(
        fetch_template_data_source=lambda *a, **k: None,
    )
    exec(compile(code, str(src_path), "exec"), mod.__dict__)
    return mod._clean


_clean = _load_clean()


class TestCleanScalars:
    def test_none_passes_through(self):
        assert _clean(None) is None

    def test_nan_float_becomes_none(self):
        assert _clean(float("nan")) is None

    def test_nat_becomes_none(self):
        assert _clean(pd.NaT) is None

    def test_pd_na_becomes_none(self):
        assert _clean(pd.NA) is None

    def test_finite_number_passes_through(self):
        assert _clean(0.42) == 0.42
        assert _clean(7) == 7

    def test_string_passes_through(self):
        assert _clean("Active") == "Active"

    def test_bool_passes_through(self):
        assert _clean(True) is True
        assert _clean(False) is False


class TestCleanArrays:
    def test_ndarray_with_one_element_returns_python_list(self):
        # Even a 1-element ndarray must be normalised so downstream
        # ``{{#each x}}`` iterates a stable Python list.
        arr = np.array([{"feature": "f0", "shap_contribution": 0.4}], dtype=object)
        out = _clean(arr)
        assert isinstance(out, list)
        assert out == [{"feature": "f0", "shap_contribution": 0.4}]

    def test_ndarray_with_many_elements_does_not_raise(self):
        # Reproduces the production failure mode: bool(ndarray) on a >1
        # element array raises ``ValueError`` that kills the template
        # render. ``_clean`` must coerce it to list before pybars ever
        # sees it.
        arr = np.array(
            [
                {"feature": "f0", "shap_contribution": 0.4},
                {"feature": "f1", "shap_contribution": -0.2},
                {"feature": "f2", "shap_contribution": 0.1},
            ],
            dtype=object,
        )
        out = _clean(arr)
        assert isinstance(out, list)
        assert len(out) == 3
        assert out[0]["feature"] == "f0"
        # Smoke: ``bool(out)`` must not raise.
        assert bool(out) is True

    def test_empty_ndarray_returns_empty_list(self):
        # Empty array -> empty list -> falsy -> ``{{#if x}}`` skips block.
        arr = np.array([], dtype=object)
        out = _clean(arr)
        assert out == []
        assert bool(out) is False

    def test_python_list_passed_through(self):
        out = _clean([1, 2, 3])
        assert out == [1, 2, 3]

    def test_tuple_becomes_list(self):
        out = _clean((1, 2))
        assert out == [1, 2]


class TestCleanNested:
    def test_dict_values_are_cleaned(self):
        nan_val = float("nan")
        out = _clean({"a": 1, "b": nan_val, "c": "x"})
        assert out == {"a": 1, "b": None, "c": "x"}

    def test_ndarray_of_dicts_each_cleaned(self):
        # Each struct's NaN fields collapse to None so {{#if v}} works
        # inside the per-row block.
        arr = np.array(
            [{"feature": "f0", "value": float("nan"), "shap_contribution": 0.4}],
            dtype=object,
        )
        out = _clean(arr)
        assert out == [{"feature": "f0", "value": None, "shap_contribution": 0.4}]

    def test_list_of_dicts_with_nested_arrays(self):
        # Mirrors ``alternate_playbooks`` shape: list of structs with
        # number-typed scoring fields that arrive as numpy scalars.
        arr = np.array(
            [
                {"playbook_id": "a", "fit_score": np.float64(0.82)},
                {"playbook_id": "b", "fit_score": np.float64(0.71)},
            ],
            dtype=object,
        )
        out = _clean(arr)
        assert out[0]["playbook_id"] == "a"
        assert out[1]["fit_score"] == pytest.approx(0.71)
        # Numpy scalars survive cleaning -- they remain truthy / formattable.
        assert bool(out) is True


class TestCleanDoesNotRaiseOnRowDicts:
    def test_full_row_with_array_fields_renders_to_jsonable_dict(self):
        # End-to-end: a fake row in the same shape as
        # ``v_account_explanation`` -- including the array fields that
        # caused the production failure -- must clean to something pybars
        # can iterate without raising.
        row = pd.Series({
            "entity_id": "C53DF2",
            "playbook_name": "Onboarding Recovery",
            "archetype_name": "Stalled Onboarders",
            "churn_probability": 0.42,
            "value_at_risk": float("nan"),
            "account_top_shap_features": np.array(
                [
                    {"feature": "active_span_days", "value": 4.0, "shap_contribution": 0.40, "direction": "positive"},
                    {"feature": "open_rate",        "value": 0.1, "shap_contribution": -0.20, "direction": "negative"},
                ],
                dtype=object,
            ),
            "alternate_playbooks": np.array(
                [{"playbook_id": "x", "playbook_name": "X", "fit_score": 0.7, "expected_uplift_pct": 0.1}],
                dtype=object,
            ),
        })
        cleaned = {k: _clean(v) for k, v in row.items()}

        # Field-by-field assertions
        assert cleaned["value_at_risk"] is None
        assert isinstance(cleaned["account_top_shap_features"], list)
        assert len(cleaned["account_top_shap_features"]) == 2
        assert cleaned["account_top_shap_features"][0]["feature"] == "active_span_days"
        assert isinstance(cleaned["alternate_playbooks"], list)
        # ``bool(...)`` must not raise on any cleaned array.
        assert bool(cleaned["account_top_shap_features"]) is True
        assert bool(cleaned["alternate_playbooks"]) is True
