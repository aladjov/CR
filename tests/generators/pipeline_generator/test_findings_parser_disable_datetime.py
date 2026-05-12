"""V4.3 — DISABLE_DATETIME_DERIVATIONS plumbed through findings_parser.

When `ExplorationFindings.disable_datetime_derivations=True` for a dataset:
- `_build_datetime_derivation_config` must return None (no per-event
  `*_delta_hours/_hour/_dow/_is_weekend` derivations rendered).
- `_build_aggregation_config` must skip the suffix-expansion loop that
  feeds derived columns into `value_columns` (no `*_dow_max_30d` etc.).
- The spec/silver-rec reconcilers must fail-fast if any selected feature
  or silver-derived rec source references a derived column for a dataset
  that has the flag set.

The flag round-trips through YAML so a single NB01 toggle propagates to
NB10 codegen for both entity-level and event-level datasets.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import pytest
import yaml

from customer_retention.analysis.auto_explorer.findings import ExplorationFindings
from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser
from customer_retention.stages.modeling.feature_spec import (
    FeatureSpec,
    FittedTransform,
)


def _fake_namespace(tmp_path: Path):
    from customer_retention.analysis.auto_explorer.run_namespace import RunNamespace
    ns = RunNamespace(root=tmp_path, run_id="test-run")
    ns.merged_dir.mkdir(parents=True, exist_ok=True)
    return ns


def _build_spec(selected: List[str]) -> FeatureSpec:
    return FeatureSpec(
        spec_version=1, exploration_run_id="test-run",
        target_column="churn", entity_column="entity_id",
        timestamp_column="as_of_date", horizon_days=90,
        selected_features=list(selected),
        fitted_transforms=[
            FittedTransform(column=c, action="impute", method="median") for c in selected
        ],
    )


def _write_spec(ns, selected: List[str]) -> None:
    _build_spec(selected).save(ns.feature_spec_path)


def _write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        yaml.safe_dump(data, f)


def _case_findings_doc(disable_dt: bool) -> dict:
    return {
        "source_path": "/data/case.csv", "source_format": "csv",
        "row_count": 1200, "column_count": 5,
        "target_column": "churn",
        "identifier_columns": ["ACCOUNT_ID"],
        "datetime_columns": ["CREATED_DATE", "FIRST_ASSIGNED_DATE_TIME"],
        "datetime_derivation_sources": ["FIRST_ASSIGNED_DATE_TIME"],
        "disable_datetime_derivations": disable_dt,
        "columns": {
            "ACCOUNT_ID": {"name": "ACCOUNT_ID", "inferred_type": "identifier",
                           "confidence": 1.0, "evidence": []},
            "CREATED_DATE": {"name": "CREATED_DATE", "inferred_type": "datetime",
                             "confidence": 1.0, "evidence": []},
            "FIRST_ASSIGNED_DATE_TIME": {"name": "FIRST_ASSIGNED_DATE_TIME",
                                         "inferred_type": "datetime",
                                         "confidence": 1.0, "evidence": []},
            "amount": {"name": "amount", "inferred_type": "numeric_continuous",
                       "confidence": 1.0, "evidence": []},
        },
        "time_series_metadata": {
            "entity_column": "ACCOUNT_ID", "time_column": "feature_timestamp",
            "avg_events_per_entity": 1.2, "time_span_days": 730,
            "aggregation_windows_used": ["30d", "90d", "all_time"],
            "suggested_aggregations": ["30d", "90d", "all_time"],
            "aggregation_executed": True,
        },
    }


def _multi_doc() -> dict:
    return {
        "datasets": {
            "case": {
                "name": "case",
                "source_path": "/data/case.csv",
                "granularity": "event_level",
                "row_count": 1200, "column_count": 5,
                "entity_column": "ACCOUNT_ID", "time_column": "feature_timestamp",
                "target_column": "churn",
            },
        },
        "primary_entity_dataset": "case",
        "event_datasets": ["case"],
        "excluded_datasets": [],
        "aggregation_windows": ["30d", "90d", "all_time"],
    }


class TestExplorationFindingsRoundtrip:
    """The flag must survive `to_dict`/`from_dict` so NB01-written YAMLs are
    read identically by NB10 on a different host/run."""

    def test_default_is_false(self):
        f = ExplorationFindings(source_path="/data/case.csv", source_format="csv")
        assert f.disable_datetime_derivations is False

    def test_round_trips_true(self):
        f = ExplorationFindings(source_path="/data/case.csv", source_format="csv")
        f.disable_datetime_derivations = True
        restored = ExplorationFindings.from_dict(f.to_dict())
        assert restored.disable_datetime_derivations is True

    def test_round_trips_false(self):
        f = ExplorationFindings(source_path="/data/case.csv", source_format="csv")
        f.disable_datetime_derivations = False
        restored = ExplorationFindings.from_dict(f.to_dict())
        assert restored.disable_datetime_derivations is False

    def test_missing_field_in_legacy_yaml_defaults_to_false(self):
        f = ExplorationFindings(source_path="/data/case.csv", source_format="csv")
        d = f.to_dict()
        d.pop("disable_datetime_derivations", None)
        restored = ExplorationFindings.from_dict(d)
        assert restored.disable_datetime_derivations is False


class TestBuildDatetimeDerivationConfigGate:

    def test_returns_none_when_flag_set(self):
        f = ExplorationFindings(source_path="/data/case.csv", source_format="csv")
        f.datetime_derivation_sources = ["CREATED_DATE", "ACTIVATED_DATE"]
        f.disable_datetime_derivations = True
        out = FindingsParser._build_datetime_derivation_config(
            f, reference_column="feature_timestamp", mask_future=False,
        )
        assert out is None

    def test_returns_config_when_flag_unset_and_sources_present(self):
        f = ExplorationFindings(source_path="/data/case.csv", source_format="csv")
        f.datetime_derivation_sources = ["CREATED_DATE"]
        f.disable_datetime_derivations = False
        out = FindingsParser._build_datetime_derivation_config(
            f, reference_column="feature_timestamp", mask_future=False,
        )
        assert out is not None
        assert out.source_columns == ["CREATED_DATE"]


class TestAggregationConfigSkipsSuffixExpansion:
    """When the flag is set, the per-event suffix expansion at
    `_build_aggregation_config` must NOT inject `_delta_hours/_hour/_dow/
    _is_weekend` into `value_columns`. The bronze aggregator therefore
    cannot compute `sum/mean/max/count_{window}` on those derived
    columns — which is the leakage class V3 surfaced."""

    def test_disabled_dataset_omits_derived_columns_from_value_columns(self, tmp_path):
        ns = _fake_namespace(tmp_path)
        case_path = ns.dataset_findings_dir("case") / "case_findings.yaml"
        _write_yaml(case_path, _case_findings_doc(disable_dt=True))
        multi = _multi_doc()
        multi["datasets"]["case"]["findings_path"] = str(case_path)
        _write_yaml(ns.multi_dataset_findings_path, multi)
        _write_spec(ns, ["amount_max_30d", "event_count_all_time"])

        parser = FindingsParser(findings_dir=str(ns.merged_dir), namespace=ns)
        config = parser.parse()

        agg = config.bronze_event["case"].aggregation
        for col in (agg.value_columns or []):
            assert not col.endswith(("_delta_hours", "_hour", "_dow", "_is_weekend")), (
                f"value_columns contains derived datetime column {col!r} despite "
                "DISABLE_DATETIME_DERIVATIONS=True on the dataset's findings"
            )

    def test_enabled_dataset_keeps_derived_columns_in_value_columns(self, tmp_path):
        ns = _fake_namespace(tmp_path)
        case_path = ns.dataset_findings_dir("case") / "case_findings.yaml"
        _write_yaml(case_path, _case_findings_doc(disable_dt=False))
        multi = _multi_doc()
        multi["datasets"]["case"]["findings_path"] = str(case_path)
        _write_yaml(ns.multi_dataset_findings_path, multi)
        _write_spec(ns, ["FIRST_ASSIGNED_DATE_TIME_hour_max_30d", "event_count_all_time"])

        parser = FindingsParser(findings_dir=str(ns.merged_dir), namespace=ns)
        config = parser.parse()

        agg = config.bronze_event["case"].aggregation
        derived = [c for c in (agg.value_columns or [])
                   if c.endswith(("_delta_hours", "_hour", "_dow", "_is_weekend"))]
        assert derived, (
            "value_columns should include derived datetime columns when "
            "DISABLE_DATETIME_DERIVATIONS is False"
        )

    def test_disabled_dataset_emits_no_landing_or_bronze_datetime_derivation(self, tmp_path):
        ns = _fake_namespace(tmp_path)
        case_path = ns.dataset_findings_dir("case") / "case_findings.yaml"
        _write_yaml(case_path, _case_findings_doc(disable_dt=True))
        multi = _multi_doc()
        multi["datasets"]["case"]["findings_path"] = str(case_path)
        _write_yaml(ns.multi_dataset_findings_path, multi)
        _write_spec(ns, ["amount_max_30d"])

        parser = FindingsParser(findings_dir=str(ns.merged_dir), namespace=ns)
        config = parser.parse()

        assert config.bronze_event["case"].datetime_derivation is None


class TestReconcilerFailFast:
    """When the spec references a datetime-derived feature for a disabled
    dataset, the reconciler must raise rather than silently auto-extend.
    Operator must either flip the knob off or remove the recommendation."""

    def test_raises_when_spec_references_derived_for_disabled_dataset(self, tmp_path):
        ns = _fake_namespace(tmp_path)
        case_path = ns.dataset_findings_dir("case") / "case_findings.yaml"
        # Disable, but leave `datetime_derivation_sources` empty so the
        # reconciler would normally try to auto-extend.
        doc = _case_findings_doc(disable_dt=True)
        doc["datetime_derivation_sources"] = []
        _write_yaml(case_path, doc)
        multi = _multi_doc()
        multi["datasets"]["case"]["findings_path"] = str(case_path)
        _write_yaml(ns.multi_dataset_findings_path, multi)
        _write_spec(ns, [
            "FIRST_ASSIGNED_DATE_TIME_hour_max_30d",
            "amount_max_30d",
        ])

        parser = FindingsParser(findings_dir=str(ns.merged_dir), namespace=ns)
        with pytest.raises(ValueError, match="DISABLE_DATETIME_DERIVATIONS=True"):
            parser.parse()

    def test_does_not_raise_when_spec_has_no_derived_features(self, tmp_path):
        ns = _fake_namespace(tmp_path)
        case_path = ns.dataset_findings_dir("case") / "case_findings.yaml"
        doc = _case_findings_doc(disable_dt=True)
        doc["datetime_derivation_sources"] = []
        _write_yaml(case_path, doc)
        multi = _multi_doc()
        multi["datasets"]["case"]["findings_path"] = str(case_path)
        _write_yaml(ns.multi_dataset_findings_path, multi)
        _write_spec(ns, ["amount_max_30d", "event_count_all_time"])

        parser = FindingsParser(findings_dir=str(ns.merged_dir), namespace=ns)
        config = parser.parse()
        assert config.bronze_event["case"].datetime_derivation is None
