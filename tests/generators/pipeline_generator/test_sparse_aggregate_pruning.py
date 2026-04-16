"""Sparse-aggregate pruning: block mean/max on value columns when the dataset
has too few events/entity for those aggregates to be statistically meaningful.
"""
from __future__ import annotations

from customer_retention.analysis.auto_explorer.exploration_manager import (
    DatasetInfo,
    MultiDatasetFindings,
)
from customer_retention.analysis.auto_explorer.findings import (
    ColumnFinding,
    ColumnType,
    ExplorationFindings,
    TimeSeriesMetadata,
)
from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser


def _make_findings(*, avg_events_per_entity: float, value_col: str = "amount"):
    ts = TimeSeriesMetadata(
        entity_column="entity_id", time_column="event_ts",
        avg_events_per_entity=avg_events_per_entity, time_span_days=730,
        unique_entities=1000, suggested_aggregations=["365d", "all_time"],
    )
    ts.aggregation_windows_used = ["365d", "all_time"]
    findings = ExplorationFindings(
        source_path="unused", source_format="csv",
        columns={
            "entity_id": ColumnFinding(name="entity_id", inferred_type=ColumnType.IDENTIFIER, confidence=1.0, evidence=[]),
            "event_ts": ColumnFinding(name="event_ts", inferred_type=ColumnType.DATETIME, confidence=1.0, evidence=[]),
            value_col: ColumnFinding(name=value_col, inferred_type=ColumnType.NUMERIC_CONTINUOUS, confidence=1.0, evidence=[]),
            "cat_status": ColumnFinding(name="cat_status", inferred_type=ColumnType.CATEGORICAL_NOMINAL, confidence=1.0, evidence=[]),
            "is_flag": ColumnFinding(name="is_flag", inferred_type=ColumnType.BINARY, confidence=1.0, evidence=[]),
        },
        target_column="churned",
        time_series_metadata=ts,
    )
    return findings


def _make_multi(findings):
    return MultiDatasetFindings(
        datasets={"my_events": DatasetInfo(
            name="my_events", findings_path="unused", source_path="",
            granularity=None, row_count=0, column_count=0,
        )},
    )


class TestSparsePruneThreshold:
    def test_dense_dataset_all_aggregates_emitted(self):
        findings = _make_findings(avg_events_per_entity=10.0)
        multi = _make_multi(findings)
        cfg = FindingsParser(findings_dir="unused")._build_aggregation_config(multi, findings, dataset_name="my_events")
        assert cfg.column_blocked_funcs.get("amount", []) == []

    def test_sparse_dataset_blocks_mean_max(self):
        findings = _make_findings(avg_events_per_entity=1.2)
        multi = _make_multi(findings)
        cfg = FindingsParser(findings_dir="unused")._build_aggregation_config(multi, findings, dataset_name="my_events")
        assert set(cfg.column_blocked_funcs["amount"]) == {"mean", "max"}

    def test_borderline_density_at_threshold_kept(self):
        findings = _make_findings(avg_events_per_entity=2.0)
        multi = _make_multi(findings)
        cfg = FindingsParser(findings_dir="unused")._build_aggregation_config(multi, findings, dataset_name="my_events")
        assert cfg.column_blocked_funcs.get("amount", []) == []

    def test_borderline_density_below_threshold_pruned(self):
        findings = _make_findings(avg_events_per_entity=1.99)
        multi = _make_multi(findings)
        cfg = FindingsParser(findings_dir="unused")._build_aggregation_config(multi, findings, dataset_name="my_events")
        assert "mean" in cfg.column_blocked_funcs["amount"]
        assert "max" in cfg.column_blocked_funcs["amount"]

    def test_count_and_sum_never_blocked(self):
        findings = _make_findings(avg_events_per_entity=0.5)
        multi = _make_multi(findings)
        cfg = FindingsParser(findings_dir="unused")._build_aggregation_config(multi, findings, dataset_name="my_events")
        blocked = set(cfg.column_blocked_funcs.get("amount", []))
        assert "count" not in blocked
        assert "sum" not in blocked

    def test_categorical_and_binary_unaffected(self):
        findings = _make_findings(avg_events_per_entity=0.5)
        multi = _make_multi(findings)
        cfg = FindingsParser(findings_dir="unused")._build_aggregation_config(multi, findings, dataset_name="my_events")
        assert cfg.column_blocked_funcs.get("cat_status", []) == []
        assert cfg.column_blocked_funcs.get("is_flag", []) == []

    def test_missing_metadata_does_not_prune(self):
        findings = _make_findings(avg_events_per_entity=1.0)
        findings.time_series_metadata.avg_events_per_entity = None
        multi = _make_multi(findings)
        cfg = FindingsParser(findings_dir="unused")._build_aggregation_config(multi, findings, dataset_name="my_events")
        assert cfg.column_blocked_funcs.get("amount", []) == []

    def test_multiple_value_columns_all_pruned(self):
        findings = _make_findings(avg_events_per_entity=1.0)
        findings.columns["quantity"] = ColumnFinding(
            name="quantity", inferred_type=ColumnType.NUMERIC_DISCRETE,
            confidence=1.0, evidence=[],
        )
        multi = _make_multi(findings)
        cfg = FindingsParser(findings_dir="unused")._build_aggregation_config(multi, findings, dataset_name="my_events")
        assert set(cfg.column_blocked_funcs["amount"]) >= {"mean", "max"}
        assert set(cfg.column_blocked_funcs["quantity"]) >= {"mean", "max"}

    def test_existing_blocked_funcs_preserved(self):
        findings = _make_findings(avg_events_per_entity=1.0)
        multi = MultiDatasetFindings(datasets={
            "my_events": DatasetInfo(
                name="my_events", findings_path="unused", source_path="",
                granularity=None, row_count=0, column_count=0,
                feature_exclusions=[
                    type("Exc", (), {
                        "column": "amount", "blocked_categories": [], "blocked_funcs": ["sum"],
                    })(),
                ],
            ),
        })
        cfg = FindingsParser(findings_dir="unused")._build_aggregation_config(multi, findings, dataset_name="my_events")
        assert set(cfg.column_blocked_funcs["amount"]) >= {"mean", "max", "sum"}


class TestExpectedEventsInWindow:
    """Document the scaling heuristic for potential per-window use (not wired in today)."""

    def test_all_time_returns_lifetime_average(self):
        from customer_retention.generators.pipeline_generator.findings_parser import (
            _expected_events_in_window,
        )
        ts = TimeSeriesMetadata(avg_events_per_entity=3.0, time_span_days=730)
        assert _expected_events_in_window(ts, "all_time") == 3.0

    def test_short_window_scales_proportionally(self):
        from customer_retention.generators.pipeline_generator.findings_parser import (
            _expected_events_in_window,
        )
        ts = TimeSeriesMetadata(avg_events_per_entity=3.0, time_span_days=730)
        assert abs(_expected_events_in_window(ts, "365d") - 1.5) < 1e-6

    def test_window_larger_than_span_caps_at_lifetime(self):
        from customer_retention.generators.pipeline_generator.findings_parser import (
            _expected_events_in_window,
        )
        ts = TimeSeriesMetadata(avg_events_per_entity=3.0, time_span_days=100)
        assert _expected_events_in_window(ts, "365d") == 3.0

    def test_missing_metadata_returns_infinity(self):
        from customer_retention.generators.pipeline_generator.findings_parser import (
            _expected_events_in_window,
        )
        ts = TimeSeriesMetadata(avg_events_per_entity=None, time_span_days=100)
        import math
        assert math.isinf(_expected_events_in_window(ts, "365d"))


class TestCategoricalValueCountsConfig:
    """Schema tests for the new categorical_value_counts field."""

    def test_default_is_empty_dict(self):
        from customer_retention.generators.pipeline_generator.models import (
            AggregationWindowConfig,
        )
        cfg = AggregationWindowConfig()
        assert cfg.categorical_value_counts == {}

    def test_accepts_single_col_with_values(self):
        from customer_retention.generators.pipeline_generator.models import (
            AggregationWindowConfig,
        )
        cfg = AggregationWindowConfig(
            categorical_value_counts={"event_type": ["start", "terminate"]},
        )
        assert cfg.categorical_value_counts == {"event_type": ["start", "terminate"]}

    def test_accepts_multiple_cols(self):
        from customer_retention.generators.pipeline_generator.models import (
            AggregationWindowConfig,
        )
        cfg = AggregationWindowConfig(
            categorical_value_counts={
                "event_type": ["start", "terminate"],
                "status": ["open", "closed"],
            },
        )
        assert len(cfg.categorical_value_counts) == 2


class TestBronzeAggregationOverridesPopulateValueCounts:
    """bronze_aggregation_overrides → AggregationWindowConfig.categorical_value_counts."""

    def test_override_propagates(self):
        from customer_retention.analysis.auto_explorer.exploration_manager import (
            DatasetInfo,
            MultiDatasetFindings,
        )
        from customer_retention.analysis.auto_explorer.findings import (
            ColumnFinding,
            ColumnType,
            ExplorationFindings,
            TimeSeriesMetadata,
        )
        from customer_retention.generators.pipeline_generator.findings_parser import (
            FindingsParser,
        )
        ts = TimeSeriesMetadata(
            entity_column="entity_id", time_column="event_ts",
            avg_events_per_entity=10.0, time_span_days=730,
        )
        ts.aggregation_windows_used = ["365d", "all_time"]
        findings = ExplorationFindings(
            source_path="unused", source_format="csv",
            columns={
                "entity_id": ColumnFinding(
                    name="entity_id", inferred_type=ColumnType.IDENTIFIER,
                    confidence=1.0, evidence=[],
                ),
                "event_ts": ColumnFinding(
                    name="event_ts", inferred_type=ColumnType.DATETIME,
                    confidence=1.0, evidence=[],
                ),
                "amount": ColumnFinding(
                    name="amount", inferred_type=ColumnType.NUMERIC_CONTINUOUS,
                    confidence=1.0, evidence=[],
                ),
            },
            target_column="churned",
            time_series_metadata=ts,
        )
        multi = MultiDatasetFindings(datasets={
            "contract": DatasetInfo(
                name="contract", findings_path="unused", source_path="",
                granularity=None, row_count=0, column_count=0,
            ),
        })
        parser = FindingsParser(
            findings_dir="unused",
            bronze_aggregation_overrides={
                "contract": {
                    "categorical_value_counts": {"event_type": ["start", "terminate"]},
                },
            },
        )
        cfg = parser._build_aggregation_config(multi, findings, dataset_name="contract")
        assert cfg.categorical_value_counts == {"event_type": ["start", "terminate"]}

    def test_no_override_yields_empty_dict(self):
        from customer_retention.analysis.auto_explorer.exploration_manager import (
            DatasetInfo,
            MultiDatasetFindings,
        )
        from customer_retention.analysis.auto_explorer.findings import (
            ColumnFinding,
            ColumnType,
            ExplorationFindings,
            TimeSeriesMetadata,
        )
        from customer_retention.generators.pipeline_generator.findings_parser import (
            FindingsParser,
        )
        ts = TimeSeriesMetadata(
            entity_column="entity_id", time_column="event_ts",
            avg_events_per_entity=10.0, time_span_days=730,
        )
        ts.aggregation_windows_used = ["365d", "all_time"]
        findings = ExplorationFindings(
            source_path="unused", source_format="csv",
            columns={
                "entity_id": ColumnFinding(
                    name="entity_id", inferred_type=ColumnType.IDENTIFIER,
                    confidence=1.0, evidence=[],
                ),
                "event_ts": ColumnFinding(
                    name="event_ts", inferred_type=ColumnType.DATETIME,
                    confidence=1.0, evidence=[],
                ),
                "amount": ColumnFinding(
                    name="amount", inferred_type=ColumnType.NUMERIC_CONTINUOUS,
                    confidence=1.0, evidence=[],
                ),
            },
            target_column="churned",
            time_series_metadata=ts,
        )
        multi = MultiDatasetFindings(datasets={
            "contract": DatasetInfo(
                name="contract", findings_path="unused", source_path="",
                granularity=None, row_count=0, column_count=0,
            ),
        })
        cfg = FindingsParser(findings_dir="unused")._build_aggregation_config(
            multi, findings, dataset_name="contract",
        )
        assert cfg.categorical_value_counts == {}
