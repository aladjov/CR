"""FeatureSpec-driven reconciliation of BronzeEventConfig.

Parity contract: every column declared in `FeatureSpec.selected_features` must
be reachable through the generated pipeline. When NB01d produced a feature and
NB08 selected it, NB10 cannot silently drop it via a density heuristic, a
missing lag flag, or an un-enabled cyclical/temporal block. These tests
exercise each path by constructing the exact gap observed in the user-reported
failures (SPS churn and email churn).
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import yaml

from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser
from customer_retention.generators.pipeline_generator.models import (
    AggregationWindowConfig,
    BronzeEventConfig,
    LifecycleConfig,
    SourceConfig,
    TemporalFeatureConfig,
)
from customer_retention.stages.modeling.feature_spec import (
    FeatureSpec,
    FittedTransform,
)


def _event_source() -> SourceConfig:
    return SourceConfig(
        name="case", path="/data/case.csv", format="csv",
        entity_key="ACCOUNT_ID", time_column="CREATED_DATE", is_event_level=True,
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


class TestReconcileSparseAggregatePrune:
    """NB01d emits mean/max unconditionally; spec selects them. NB10's density
    heuristic must not win over explicit selection.
    """

    def _agg(self, blocked: dict) -> AggregationWindowConfig:
        return AggregationWindowConfig(
            windows=["30d", "90d", "all_time"],
            value_columns=[
                "FIRST_ASSIGNED_DATE_TIME_hour",
                "FIRST_ASSIGNED_DATE_TIME_dow",
                "FIRST_ASSIGNED_DATE_TIME_is_weekend",
                "RESPONSE_DATE_TIME_delta_hours",
                "amount",
            ],
            agg_funcs=["sum", "mean", "max", "count"],
            column_blocked_funcs=dict(blocked),
        )

    def test_unblocks_mean_max_when_spec_declares_them(self):
        blocked = {
            "FIRST_ASSIGNED_DATE_TIME_hour": ["max", "mean"],
            "RESPONSE_DATE_TIME_delta_hours": ["max", "mean"],
            "amount": ["max", "mean"],
        }
        event_cfg = BronzeEventConfig(
            source=_event_source(), entity_column="ACCOUNT_ID",
            time_column="CREATED_DATE", aggregation=self._agg(blocked),
        )
        spec = _build_spec([
            "FIRST_ASSIGNED_DATE_TIME_hour_max_30d",
            "RESPONSE_DATE_TIME_delta_hours_max_30d",
        ])
        FindingsParser._reconcile_event_config_with_spec(event_cfg, spec)
        remaining = event_cfg.aggregation.column_blocked_funcs
        assert "max" not in remaining.get("FIRST_ASSIGNED_DATE_TIME_hour", [])
        assert "max" not in remaining.get("RESPONSE_DATE_TIME_delta_hours", [])
        assert set(remaining.get("amount", [])) == {"max", "mean"}

    def test_unblocks_only_the_func_that_spec_requires(self):
        blocked = {"amount": ["max", "mean"]}
        event_cfg = BronzeEventConfig(
            source=_event_source(), entity_column="ACCOUNT_ID",
            time_column="CREATED_DATE", aggregation=self._agg(blocked),
        )
        spec = _build_spec(["amount_mean_30d"])
        FindingsParser._reconcile_event_config_with_spec(event_cfg, spec)
        remaining = set(event_cfg.aggregation.column_blocked_funcs.get("amount", []))
        assert remaining == {"max"}

    def test_any_window_in_spec_unblocks_func(self):
        blocked = {"amount": ["max"]}
        event_cfg = BronzeEventConfig(
            source=_event_source(), entity_column="ACCOUNT_ID",
            time_column="CREATED_DATE", aggregation=self._agg(blocked),
        )
        spec = _build_spec(["amount_max_all_time"])
        FindingsParser._reconcile_event_config_with_spec(event_cfg, spec)
        assert "max" not in event_cfg.aggregation.column_blocked_funcs.get("amount", [])


class TestReconcileAggregationWindows:
    """event_count_365d / event_count_all_time failures trace to windows the
    spec needs but the resolved aggregation windows omit.
    """

    def test_adds_missing_event_count_windows(self):
        agg = AggregationWindowConfig(
            windows=["7d", "30d", "90d", "180d"],
            value_columns=["amount"],
            agg_funcs=["sum", "mean", "max", "count"],
        )
        event_cfg = BronzeEventConfig(
            source=_event_source(), entity_column="ACCOUNT_ID",
            time_column="CREATED_DATE", aggregation=agg,
        )
        spec = _build_spec(["event_count_365d", "event_count_all_time"])
        FindingsParser._reconcile_event_config_with_spec(event_cfg, spec)
        assert "365d" in event_cfg.aggregation.windows
        assert "all_time" in event_cfg.aggregation.windows

    def test_existing_windows_are_preserved_in_original_order(self):
        agg = AggregationWindowConfig(
            windows=["7d", "30d"], value_columns=["amount"],
            agg_funcs=["sum", "mean", "max", "count"],
        )
        event_cfg = BronzeEventConfig(
            source=_event_source(), entity_column="ACCOUNT_ID",
            time_column="CREATED_DATE", aggregation=agg,
        )
        spec = _build_spec(["event_count_all_time"])
        FindingsParser._reconcile_event_config_with_spec(event_cfg, spec)
        assert event_cfg.aggregation.windows[:2] == ["7d", "30d"]
        assert "all_time" in event_cfg.aggregation.windows

    def test_adds_windows_required_by_value_column_aggregates(self):
        agg = AggregationWindowConfig(
            windows=["30d"], value_columns=["amount"],
            agg_funcs=["sum", "mean", "max", "count"],
        )
        event_cfg = BronzeEventConfig(
            source=_event_source(), entity_column="ACCOUNT_ID",
            time_column="CREATED_DATE", aggregation=agg,
        )
        spec = _build_spec(["amount_sum_365d"])
        FindingsParser._reconcile_event_config_with_spec(event_cfg, spec)
        assert "365d" in event_cfg.aggregation.windows


class TestReconcileLifecycleCyclical:
    def test_enables_cyclical_when_dow_cos_in_spec(self):
        event_cfg = BronzeEventConfig(
            source=_event_source(), entity_column="ACCOUNT_ID",
            time_column="CREATED_DATE",
            lifecycle=LifecycleConfig(include_cyclical_features=False),
        )
        spec = _build_spec(["dow_cos", "dow_sin"])
        FindingsParser._reconcile_event_config_with_spec(event_cfg, spec)
        assert event_cfg.lifecycle.include_cyclical_features is True

    def test_enables_month_cyclical_when_month_sin_selected(self):
        event_cfg = BronzeEventConfig(
            source=_event_source(), entity_column="ACCOUNT_ID",
            time_column="CREATED_DATE",
            lifecycle=LifecycleConfig(include_month_cyclical=False),
        )
        spec = _build_spec(["month_sin"])
        FindingsParser._reconcile_event_config_with_spec(event_cfg, spec)
        assert event_cfg.lifecycle.include_month_cyclical is True

    def test_enables_quadrant_when_selected(self):
        event_cfg = BronzeEventConfig(
            source=_event_source(), entity_column="ACCOUNT_ID",
            time_column="CREATED_DATE",
            lifecycle=LifecycleConfig(include_lifecycle_quadrant=False),
        )
        spec = _build_spec(["lifecycle_quadrant"])
        FindingsParser._reconcile_event_config_with_spec(event_cfg, spec)
        assert event_cfg.lifecycle.include_lifecycle_quadrant is True

    def test_enables_recency_bucket_when_selected(self):
        event_cfg = BronzeEventConfig(
            source=_event_source(), entity_column="ACCOUNT_ID",
            time_column="CREATED_DATE",
            lifecycle=LifecycleConfig(include_recency_bucket=False),
        )
        spec = _build_spec(["recency_bucket"])
        FindingsParser._reconcile_event_config_with_spec(event_cfg, spec)
        assert event_cfg.lifecycle.include_recency_bucket is True

    def test_creates_lifecycle_config_when_none_and_cyclical_needed(self):
        event_cfg = BronzeEventConfig(
            source=_event_source(), entity_column="ACCOUNT_ID",
            time_column="CREATED_DATE", lifecycle=None,
        )
        spec = _build_spec(["dow_cos"])
        FindingsParser._reconcile_event_config_with_spec(event_cfg, spec)
        assert event_cfg.lifecycle is not None
        assert event_cfg.lifecycle.include_cyclical_features is True


class TestReconcileTemporalFeatures:
    def test_recency_group_enabled_from_spec(self):
        event_cfg = BronzeEventConfig(
            source=_event_source(), entity_column="ACCOUNT_ID",
            time_column="CREATED_DATE", temporal_features=None,
        )
        spec = _build_spec(["active_span_days", "days_since_last_event"])
        FindingsParser._reconcile_event_config_with_spec(event_cfg, spec)
        assert event_cfg.temporal_features is not None
        assert "recency" in event_cfg.temporal_features.feature_groups

    def test_regularity_group_enabled_from_spec(self):
        event_cfg = BronzeEventConfig(
            source=_event_source(), entity_column="ACCOUNT_ID",
            time_column="CREATED_DATE", temporal_features=None,
        )
        spec = _build_spec([
            "event_frequency", "inter_event_gap_max", "regularity_score",
        ])
        FindingsParser._reconcile_event_config_with_spec(event_cfg, spec)
        assert event_cfg.temporal_features is not None
        assert "regularity" in event_cfg.temporal_features.feature_groups

    def test_existing_temporal_features_augmented_not_overwritten(self):
        tf = TemporalFeatureConfig(lag_columns=["amount"], feature_groups=["lagged_windows"])
        event_cfg = BronzeEventConfig(
            source=_event_source(), entity_column="ACCOUNT_ID",
            time_column="CREATED_DATE", temporal_features=tf,
        )
        spec = _build_spec(["active_span_days"])
        FindingsParser._reconcile_event_config_with_spec(event_cfg, spec)
        groups = set(event_cfg.temporal_features.feature_groups)
        assert "lagged_windows" in groups
        assert "recency" in groups
        assert event_cfg.temporal_features.lag_columns == ["amount"]

    def test_no_temporal_config_created_when_not_needed(self):
        event_cfg = BronzeEventConfig(
            source=_event_source(), entity_column="ACCOUNT_ID",
            time_column="CREATED_DATE", temporal_features=None,
        )
        spec = _build_spec(["amount_mean_30d"])
        FindingsParser._reconcile_event_config_with_spec(event_cfg, spec)
        assert event_cfg.temporal_features is None


class TestReconcileIntegrationWithParser:
    """Drive a sparse event dataset through parse() with a FeatureSpec that
    includes mean/max aggregates. The violation must not fire.
    """

    def test_sparse_dataset_with_spec_passes_parity(self, tmp_path):
        ns = _fake_namespace(tmp_path)

        ts = {
            "entity_column": "ACCOUNT_ID", "time_column": "feature_timestamp",
            "avg_events_per_entity": 1.2, "time_span_days": 730,
            "aggregation_windows_used": ["30d", "90d", "365d", "all_time"],
            "suggested_aggregations": ["30d", "90d", "365d", "all_time"],
            "aggregation_executed": True,
        }
        case_findings = {
            "source_path": "/data/case.csv", "source_format": "csv",
            "row_count": 1200, "column_count": 5,
            "target_column": "churn",
            "identifier_columns": ["ACCOUNT_ID"],
            "datetime_columns": ["CREATED_DATE", "FIRST_ASSIGNED_DATE_TIME"],
            "datetime_derivation_sources": ["FIRST_ASSIGNED_DATE_TIME"],
            "columns": {
                "ACCOUNT_ID": {"name": "ACCOUNT_ID", "inferred_type": "identifier",
                               "confidence": 1.0, "evidence": []},
                "CREATED_DATE": {"name": "CREATED_DATE", "inferred_type": "datetime",
                                 "confidence": 1.0, "evidence": []},
                "FIRST_ASSIGNED_DATE_TIME": {"name": "FIRST_ASSIGNED_DATE_TIME",
                                             "inferred_type": "datetime",
                                             "confidence": 1.0, "evidence": []},
                "FIRST_ASSIGNED_DATE_TIME_hour": {
                    "name": "FIRST_ASSIGNED_DATE_TIME_hour",
                    "inferred_type": "numeric_continuous",
                    "confidence": 1.0, "evidence": ["derived:datetime_features"],
                },
                "amount": {"name": "amount", "inferred_type": "numeric_continuous",
                           "confidence": 1.0, "evidence": []},
            },
            "time_series_metadata": ts,
        }
        case_path = ns.dataset_findings_dir("case") / "case_findings.yaml"
        _write_yaml(case_path, case_findings)

        multi_doc = {
            "datasets": {
                "case": {
                    "name": "case",
                    "findings_path": str(case_path),
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
            "aggregation_windows": ["30d", "90d", "365d", "all_time"],
        }
        _write_yaml(ns.multi_dataset_findings_path, multi_doc)

        _write_spec(ns, [
            "FIRST_ASSIGNED_DATE_TIME_hour_max_30d",
            "FIRST_ASSIGNED_DATE_TIME_hour_mean_30d",
            "amount_max_365d",
            "event_count_all_time",
            "dow_cos",
            "active_span_days",
            "event_frequency",
        ])

        parser = FindingsParser(findings_dir=str(ns.merged_dir), namespace=ns)
        config = parser.parse()

        event_cfg = config.bronze_event["case"]
        assert "all_time" in event_cfg.aggregation.windows
        assert event_cfg.lifecycle is not None
        assert event_cfg.lifecycle.include_cyclical_features is True
        assert event_cfg.temporal_features is not None
        assert "recency" in event_cfg.temporal_features.feature_groups
        assert "regularity" in event_cfg.temporal_features.feature_groups
        blocked = event_cfg.aggregation.column_blocked_funcs or {}
        assert "max" not in blocked.get("FIRST_ASSIGNED_DATE_TIME_hour", [])
        assert "mean" not in blocked.get("FIRST_ASSIGNED_DATE_TIME_hour", [])
        assert "max" not in blocked.get("amount", [])


class TestReconcileRunsBeforeSilverRecommendations:
    """Silver ratio recs gate on predicted pipeline columns. If the reconcile
    pass ran after recommendations, ratios over reconciled columns (e.g.
    `event_count_180d_to_active_span_days_ratio`) would be permanently dropped.
    """

    def test_ratio_rec_over_reconciled_columns_survives(self, tmp_path):
        ns = _fake_namespace(tmp_path)

        ts = {
            "entity_column": "customer_id", "time_column": "feature_timestamp",
            "avg_events_per_entity": 8.0, "time_span_days": 730,
            "aggregation_windows_used": ["30d", "180d"],
            "suggested_aggregations": ["30d", "180d"],
            "aggregation_executed": True,
        }
        email_findings = {
            "source_path": "/data/email.csv", "source_format": "csv",
            "row_count": 12000, "column_count": 4,
            "target_column": "churn",
            "identifier_columns": ["customer_id"],
            "datetime_columns": ["sent_date"],
            "columns": {
                "customer_id": {"name": "customer_id", "inferred_type": "identifier",
                                "confidence": 1.0, "evidence": []},
                "sent_date": {"name": "sent_date", "inferred_type": "datetime",
                              "confidence": 1.0, "evidence": []},
                "amount": {"name": "amount", "inferred_type": "numeric_continuous",
                           "confidence": 1.0, "evidence": []},
            },
            "time_series_metadata": ts,
        }
        email_path = ns.dataset_findings_dir("email") / "email_findings.yaml"
        _write_yaml(email_path, email_findings)

        multi_doc = {
            "datasets": {
                "email": {
                    "name": "email",
                    "findings_path": str(email_path),
                    "source_path": "/data/email.csv",
                    "granularity": "event_level",
                    "row_count": 12000, "column_count": 4,
                    "entity_column": "customer_id", "time_column": "feature_timestamp",
                    "target_column": "churn",
                },
            },
            "primary_entity_dataset": "email",
            "event_datasets": ["email"],
            "excluded_datasets": [],
            "aggregation_windows": ["30d", "180d"],
        }
        _write_yaml(ns.multi_dataset_findings_path, multi_doc)

        recommendations = {
            "silver": {
                "entity_column": "customer_id",
                "time_column": "feature_timestamp",
                "derived_columns": [
                    {
                        "id": "silver.ratio.001",
                        "layer": "silver",
                        "category": "derived_column",
                        "target_column": "event_count_180d_to_event_count_all_time_ratio",
                        "action": "ratio",
                        "parameters": {"numerator": "event_count_180d",
                                       "denominator": "event_count_all_time"},
                        "rationale": "engagement recency vs lifetime",
                        "source_notebook": "01e",
                    },
                    {
                        "id": "silver.ratio.002",
                        "layer": "silver",
                        "category": "derived_column",
                        "target_column": "event_count_180d_to_active_span_days_ratio",
                        "action": "ratio",
                        "parameters": {"numerator": "event_count_180d",
                                       "denominator": "active_span_days"},
                        "rationale": "events per active span",
                        "source_notebook": "01e",
                    },
                ],
            },
        }
        _write_yaml(ns.merged_recommendations_path, recommendations)

        _write_spec(ns, [
            "event_count_180d",
            "event_count_all_time",
            "active_span_days",
            "event_count_180d_to_event_count_all_time_ratio",
            "event_count_180d_to_active_span_days_ratio",
            "dow_cos",
            "regularity_score",
        ])

        parser = FindingsParser(findings_dir=str(ns.merged_dir), namespace=ns)
        config = parser.parse()

        silver_columns = {s.column for s in config.silver.derived_columns}
        assert "event_count_180d_to_event_count_all_time_ratio" in silver_columns
        assert "event_count_180d_to_active_span_days_ratio" in silver_columns
        event_cfg = config.bronze_event["email"]
        assert "all_time" in event_cfg.aggregation.windows
        assert event_cfg.temporal_features is not None
        assert "recency" in event_cfg.temporal_features.feature_groups


class TestReconcileCreatesAggregationWhenAbsent:
    """If findings produced no aggregation config but spec needs event_count_*,
    reconcile must create a minimal AggregationWindowConfig so parity holds.
    """

    def test_creates_minimal_aggregation_when_spec_requires_event_count(self):
        event_cfg = BronzeEventConfig(
            source=_event_source(), entity_column="ACCOUNT_ID",
            time_column="CREATED_DATE", aggregation=None,
        )
        spec = _build_spec(["event_count_365d", "event_count_all_time"])
        FindingsParser._reconcile_event_config_with_spec(event_cfg, spec)
        assert event_cfg.aggregation is not None
        assert "365d" in event_cfg.aggregation.windows
        assert "all_time" in event_cfg.aggregation.windows


class TestReconcileIsNoOpWithoutSpec:
    def test_no_spec_means_no_reconciliation_side_effect(self, tmp_path):
        parser = FindingsParser(findings_dir=str(tmp_path))
        event_cfg = BronzeEventConfig(
            source=_event_source(), entity_column="ACCOUNT_ID",
            time_column="CREATED_DATE",
            aggregation=AggregationWindowConfig(
                windows=["30d"], value_columns=["amount"],
                agg_funcs=["sum", "mean", "max", "count"],
                column_blocked_funcs={"amount": ["mean", "max"]},
            ),
        )
        parser._reconcile_config_with_spec(
            type("Cfg", (), {"bronze_event": {"case": event_cfg}})(),
        )
        assert set(event_cfg.aggregation.column_blocked_funcs["amount"]) == {"mean", "max"}
