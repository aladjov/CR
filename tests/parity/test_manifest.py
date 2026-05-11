from __future__ import annotations

from pathlib import Path

import pytest

from customer_retention.parity.kinds import ApplyOpKind
from customer_retention.parity.manifest import (
    Manifest,
    ManifestEntry,
    SourceLocation,
    fingerprint_kwargs,
)


def _entry(dataset: str, kind: ApplyOpKind, order: int = 0, **kwargs) -> ManifestEntry:
    return ManifestEntry(
        dataset=dataset,
        kind=kind,
        kwargs_fingerprint=fingerprint_kwargs(kwargs),
        call_order=order,
        source_location=SourceLocation(file=Path("x.py"), line=1),
    )


class TestSourceLocation:
    def test_minimal_fields(self):
        loc = SourceLocation(file=Path("a/b.py"), line=42)
        assert loc.cell_id is None
        assert loc.component is None

    def test_format_compact(self):
        loc = SourceLocation(file=Path("a/b.py"), line=42)
        assert loc.format() == "a/b.py:42"

    def test_format_with_cell(self):
        loc = SourceLocation(
            file=Path("00_start_here.ipynb"), line=5, cell_id="abcd1234"
        )
        assert loc.format() == "00_start_here.ipynb:5 (cell=abcd1234)"

    def test_format_with_component(self):
        loc = SourceLocation(
            file=Path("renderer.py"), line=100, component="apply_history_window template"
        )
        assert "apply_history_window template" in loc.format()

    def test_is_frozen(self):
        loc = SourceLocation(file=Path("x.py"), line=1)
        with pytest.raises((AttributeError, Exception)):
            loc.line = 2  # type: ignore[misc]


class TestFingerprintKwargs:
    def test_orders_keys_for_stability(self):
        a = fingerprint_kwargs({"b": 1, "a": 2})
        b = fingerprint_kwargs({"a": 2, "b": 1})
        assert a == b
        assert list(a.keys()) == ["a", "b"]

    def test_drops_none_values(self):
        fp = fingerprint_kwargs({"x": 1, "y": None})
        assert fp == {"x": 1}

    def test_preserves_nested_dicts(self):
        fp = fingerprint_kwargs({"cfg": {"z": 1, "a": 2}})
        assert fp["cfg"] == {"z": 1, "a": 2}

    def test_lists_become_tuples_for_hashability(self):
        fp = fingerprint_kwargs({"cols": ["c", "b", "a"]})
        assert fp["cols"] == ("c", "b", "a")

    def test_drops_dataframe_like_args(self):
        class _DfStub:
            _internal = None

        fp = fingerprint_kwargs({"df": _DfStub(), "x": 5})
        assert "df" not in fp
        assert fp == {"x": 5}

    def test_drops_dataframe_by_to_spark_attr(self):
        class _PandasOnSparkStub:
            def to_spark(self):
                return None

        fp = fingerprint_kwargs({"raw_df": _PandasOnSparkStub(), "window": "30d"})
        assert fp == {"window": "30d"}


class TestManifestEntry:
    def test_is_frozen(self):
        entry = _entry("contract", ApplyOpKind.TEMPORAL_LOOKBACK)
        with pytest.raises((AttributeError, Exception)):
            entry.dataset = "other"  # type: ignore[misc]

    def test_equality_by_value(self):
        a = _entry("c", ApplyOpKind.LIFECYCLE_ENRICH)
        b = _entry("c", ApplyOpKind.LIFECYCLE_ENRICH)
        assert a == b

    def test_diff_key_excludes_source_and_order(self):
        a = _entry("c", ApplyOpKind.LIFECYCLE_ENRICH, order=0, x=1)
        b = ManifestEntry(
            dataset="c",
            kind=ApplyOpKind.LIFECYCLE_ENRICH,
            kwargs_fingerprint=fingerprint_kwargs({"x": 1}),
            call_order=99,
            source_location=SourceLocation(file=Path("other.py"), line=999),
        )
        assert a.diff_key() == b.diff_key()

    def test_diff_key_includes_dataset_kind_kwargs(self):
        a = _entry("c", ApplyOpKind.LIFECYCLE_ENRICH, x=1)
        b = _entry("c", ApplyOpKind.LIFECYCLE_ENRICH, x=2)
        assert a.diff_key() != b.diff_key()


class TestManifest:
    def test_empty(self):
        m = Manifest(entries=())
        assert m.by_dataset("any") == ()
        assert m.kinds_for("any") == frozenset()
        assert m.datasets() == frozenset()

    def test_by_dataset_filters(self):
        m = Manifest(
            entries=(
                _entry("a", ApplyOpKind.LIFECYCLE_ENRICH),
                _entry("b", ApplyOpKind.LIFECYCLE_ENRICH),
                _entry("a", ApplyOpKind.TEMPORAL_LOOKBACK),
            )
        )
        assert len(m.by_dataset("a")) == 2
        assert m.kinds_for("a") == {
            ApplyOpKind.LIFECYCLE_ENRICH,
            ApplyOpKind.TEMPORAL_LOOKBACK,
        }
        assert m.kinds_for("b") == {ApplyOpKind.LIFECYCLE_ENRICH}
        assert m.datasets() == {"a", "b"}

    def test_by_dataset_preserves_order(self):
        m = Manifest(
            entries=(
                _entry("a", ApplyOpKind.LIFECYCLE_ENRICH, order=0),
                _entry("a", ApplyOpKind.TEMPORAL_LOOKBACK, order=1),
                _entry("a", ApplyOpKind.DATETIME_DERIVE, order=2),
            )
        )
        kinds_in_order = [e.kind for e in m.by_dataset("a")]
        assert kinds_in_order == [
            ApplyOpKind.LIFECYCLE_ENRICH,
            ApplyOpKind.TEMPORAL_LOOKBACK,
            ApplyOpKind.DATETIME_DERIVE,
        ]

    def test_extend_returns_new_manifest(self):
        a = Manifest(entries=(_entry("a", ApplyOpKind.LIFECYCLE_ENRICH),))
        b = a.extend([_entry("a", ApplyOpKind.TEMPORAL_LOOKBACK)])
        assert len(a.entries) == 1
        assert len(b.entries) == 2

    def test_is_frozen(self):
        m = Manifest(entries=())
        with pytest.raises((AttributeError, Exception)):
            m.entries = (1, 2)  # type: ignore[misc]
