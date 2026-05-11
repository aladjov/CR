from __future__ import annotations

from pathlib import Path

import pytest

from customer_retention.parity import ApplyOpKind, apply_context, apply_op
from customer_retention.parity.decorator import APPLY_REGISTRY
from customer_retention.parity.gaps import GapKind
from customer_retention.parity.trace import (
    DEFAULT_TOLERANCE,
    TOLERANCE_BY_KIND,
    TraceRecord,
    audit_trace,
    clear_records,
    flush_to_yaml,
    get_records,
    load_from_yaml,
)


@pytest.fixture(autouse=True)
def _isolated_state(monkeypatch):
    snapshot = dict(APPLY_REGISTRY)
    APPLY_REGISTRY.clear()
    clear_records()
    monkeypatch.setenv("CR_PARITY_TRACE", "1")
    yield
    APPLY_REGISTRY.clear()
    APPLY_REGISTRY.update(snapshot)
    clear_records()


def _df(n: int):
    """A trivial pandas DataFrame stand-in."""
    import pandas as pd
    return pd.DataFrame({"x": range(n)})


class TestRecordCall:
    def test_invocation_appends_record(self):
        @apply_op(kind=ApplyOpKind.LIFECYCLE_ENRICH)
        def fn(df):
            return df

        with apply_context(dataset="contract"):
            fn(_df(5))

        records = get_records()
        assert len(records) == 1
        r = records[0]
        assert r.kind is ApplyOpKind.LIFECYCLE_ENRICH
        assert r.dataset == "contract"
        assert r.input_rows == 5
        assert r.output_rows == 5

    def test_dataset_hint_falls_back_to_kwarg(self):
        @apply_op(kind=ApplyOpKind.SAMPLE_FILTER, dataset_kwarg="dataset_name")
        def fn(df, dataset_name=None):
            return df

        fn(_df(10), dataset_name="account")
        records = get_records()
        assert len(records) == 1
        assert records[0].dataset == "account"

    def test_dataset_unknown_when_no_hint(self):
        @apply_op(kind=ApplyOpKind.LIFECYCLE_ENRICH)
        def fn(df):
            return df

        fn(_df(3))
        assert get_records()[0].dataset == "<unknown>"

    def test_call_order_increments(self):
        @apply_op(kind=ApplyOpKind.LIFECYCLE_ENRICH)
        def a(df):
            return df

        @apply_op(kind=ApplyOpKind.TEMPORAL_LOOKBACK)
        def b(df):
            return df

        with apply_context(dataset="x"):
            a(_df(2))
            b(_df(2))
        records = get_records()
        assert records[0].call_order == 1
        assert records[1].call_order == 2

    def test_returns_function_value(self):
        @apply_op(kind=ApplyOpKind.LIFECYCLE_ENRICH)
        def fn(df):
            df["new_col"] = 1
            return df

        result = fn(_df(3))
        assert "new_col" in result.columns

    def test_kwargs_fingerprint_filtered_by_capture_kwargs(self):
        @apply_op(
            kind=ApplyOpKind.DATETIME_DERIVE,
            capture_kwargs={"time_column"},
        )
        def fn(df, time_column=None, datetime_columns=None):
            return df

        fn(_df(1), time_column="ts", datetime_columns=["a", "b"])
        fp = get_records()[0].kwargs_fingerprint
        assert "time_column" in fp
        assert "datetime_columns" not in fp

    def test_inactive_trace_does_not_record(self, monkeypatch):
        monkeypatch.delenv("CR_PARITY_TRACE", raising=False)

        @apply_op(kind=ApplyOpKind.LIFECYCLE_ENRICH)
        def fn(df):
            return df

        fn(_df(5))
        assert get_records() == ()


class TestYamlRoundTrip:
    def test_flush_writes_yaml(self, tmp_path):
        @apply_op(kind=ApplyOpKind.LIFECYCLE_ENRICH)
        def fn(df):
            return df

        with apply_context(dataset="contract"):
            fn(_df(7))

        out = tmp_path / "trace.yaml"
        flush_to_yaml(out)
        assert out.exists()
        # Buffer is cleared after flush
        assert get_records() == ()

    def test_load_reconstructs_records(self, tmp_path):
        @apply_op(kind=ApplyOpKind.LIFECYCLE_ENRICH)
        def fn(df):
            return df

        with apply_context(dataset="contract"):
            fn(_df(10))

        out = tmp_path / "trace.yaml"
        flush_to_yaml(out)
        records = load_from_yaml(out)
        assert len(records) == 1
        r = records[0]
        assert r.kind is ApplyOpKind.LIFECYCLE_ENRICH
        assert r.dataset == "contract"
        assert r.input_rows == 10
        assert r.output_rows == 10

    def test_load_from_missing_file_returns_empty(self, tmp_path):
        assert load_from_yaml(tmp_path / "nope.yaml") == ()


class TestAuditTrace:
    def _write_trace(self, path: Path, records: list[TraceRecord]):
        from customer_retention.parity.trace import _buffer_records
        clear_records()
        buf = _buffer_records()
        for r in records:
            buf.append(r)
        flush_to_yaml(path)

    def test_matching_traces_yield_no_gaps(self, tmp_path):
        rec = TraceRecord(
            kind=ApplyOpKind.LIFECYCLE_ENRICH,
            qualified_name="x.fn",
            dataset="contract",
            kwargs_fingerprint={},
            input_rows=100,
            output_rows=160,
            call_order=1,
        )
        exp = tmp_path / "exploration.yaml"
        prod = tmp_path / "production.yaml"
        self._write_trace(exp, [rec])
        self._write_trace(prod, [rec])
        outcome = audit_trace(exp, prod)
        assert outcome.has_gaps is False

    def test_row_count_drift_exceeds_tolerance(self, tmp_path):
        exp_rec = TraceRecord(
            kind=ApplyOpKind.TEMPORAL_LOOKBACK,
            qualified_name="x.fn",
            dataset="contract",
            kwargs_fingerprint={},
            input_rows=100,
            output_rows=80,
            call_order=1,
        )
        prod_rec = TraceRecord(
            kind=ApplyOpKind.TEMPORAL_LOOKBACK,
            qualified_name="x.fn",
            dataset="contract",
            kwargs_fingerprint={},
            input_rows=100,
            output_rows=2,  # collapsed by 97.5% — the sentinel bug shape
            call_order=1,
        )
        exp = tmp_path / "exploration.yaml"
        prod = tmp_path / "production.yaml"
        self._write_trace(exp, [exp_rec])
        self._write_trace(prod, [prod_rec])
        outcome = audit_trace(exp, prod)
        assert outcome.has_gaps is True
        assert any(g.gap_kind is GapKind.RUNTIME_DRIFT for g in outcome.gaps)

    def test_per_kind_tolerance_lifecycle_enrich_5_percent(self):
        assert TOLERANCE_BY_KIND.get(ApplyOpKind.LIFECYCLE_ENRICH) == 0.05
        assert DEFAULT_TOLERANCE == 0.005

    def test_lifecycle_enrich_allows_5_percent_drift(self, tmp_path):
        # Lifecycle enrichment doubles row counts due to terminate events;
        # the rate varies by churn rate. Tolerance is 5%.
        exp_rec = TraceRecord(
            kind=ApplyOpKind.LIFECYCLE_ENRICH,
            qualified_name="x.fn",
            dataset="contract",
            kwargs_fingerprint={},
            input_rows=100,
            output_rows=160,
            call_order=1,
        )
        prod_rec = TraceRecord(
            kind=ApplyOpKind.LIFECYCLE_ENRICH,
            qualified_name="x.fn",
            dataset="contract",
            kwargs_fingerprint={},
            input_rows=100,
            output_rows=164,  # 2.5% drift, below 5% threshold
            call_order=1,
        )
        exp = tmp_path / "exploration.yaml"
        prod = tmp_path / "production.yaml"
        self._write_trace(exp, [exp_rec])
        self._write_trace(prod, [prod_rec])
        outcome = audit_trace(exp, prod)
        assert outcome.has_gaps is False

    def test_missing_key_in_one_side_is_a_gap(self, tmp_path):
        rec = TraceRecord(
            kind=ApplyOpKind.TEMPORAL_LOOKBACK,
            qualified_name="x.fn",
            dataset="contract",
            kwargs_fingerprint={},
            input_rows=100,
            output_rows=80,
            call_order=1,
        )
        exp = tmp_path / "exploration.yaml"
        prod = tmp_path / "production.yaml"
        self._write_trace(exp, [])
        self._write_trace(prod, [rec])
        outcome = audit_trace(exp, prod)
        assert outcome.has_gaps is True
        # Production has an op exploration didn't — production-only at runtime
        kinds = {g.gap_kind for g in outcome.gaps}
        assert GapKind.PRODUCTION_ONLY in kinds or GapKind.RUNTIME_DRIFT in kinds


class TestSafeRowCount:
    def test_pandas_dataframe_counted(self):
        from customer_retention.parity.trace import _safe_row_count
        assert _safe_row_count(_df(7)) == 7

    def test_none_returns_none(self):
        from customer_retention.parity.trace import _safe_row_count
        assert _safe_row_count(None) is None

    def test_unsized_object_returns_none(self):
        from customer_retention.parity.trace import _safe_row_count
        assert _safe_row_count(object()) is None

    def test_tuple_with_dataframe_first_member_unwrapped(self):
        from customer_retention.parity.trace import _safe_row_count
        df = _df(4)
        # Some apply ops return (df, extras); the helper should unwrap to
        # the leading DataFrame so the row count is meaningful.
        assert _safe_row_count((df, ["a", "b"])) == 4
