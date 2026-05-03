"""Tests for FindingsParser.bronze_aggregation_overrides hook."""
from __future__ import annotations

import pytest

from customer_retention.generators.pipeline_generator.databricks_generator import (
    DatabricksPipelineGenerator,
)
from customer_retention.generators.pipeline_generator.findings_parser import (
    FindingsParser,
)
from customer_retention.generators.pipeline_generator.generator import PipelineGenerator
from customer_retention.generators.pipeline_generator.models import BronzeEventConfig

# Reuse the existing rich fixture from test_findings_parser.py — it sets up
# a multi-dataset findings dir with one entity dataset (customers) and one
# event dataset (orders).
pytest_plugins = ["tests.generators.pipeline_generator.test_findings_parser"]


class TestBronzeEventConfigDefaults:
    def test_per_grid_date_mode_default_false(self):
        cfg = BronzeEventConfig(
            source=None,  # type: ignore[arg-type]
            entity_column="customer_id",
            time_column="order_date",
        )
        assert cfg.per_grid_date_mode is False
        assert cfg.value_counts_columns == ()


class TestNoOverridesRegression:
    def test_no_overrides_produces_default_configs(self, sample_findings_dir):
        parser = FindingsParser(str(sample_findings_dir))
        config = parser.parse()
        assert "orders" in config.bronze_event
        orders_cfg = config.bronze_event["orders"]
        assert orders_cfg.per_grid_date_mode is False
        assert orders_cfg.value_counts_columns == ()

    def test_explicit_none_overrides_equivalent_to_no_overrides(self, sample_findings_dir):
        parser_a = FindingsParser(str(sample_findings_dir))
        parser_b = FindingsParser(
            str(sample_findings_dir), bronze_aggregation_overrides=None
        )
        a = parser_a.parse().bronze_event["orders"]
        b = parser_b.parse().bronze_event["orders"]
        assert a.per_grid_date_mode == b.per_grid_date_mode
        assert a.value_counts_columns == b.value_counts_columns


class TestBronzeAggregationOverrides:
    def test_per_grid_date_mode_override_applied(self, sample_findings_dir):
        parser = FindingsParser(
            str(sample_findings_dir),
            bronze_aggregation_overrides={"orders": {"per_grid_date_mode": True}},
        )
        config = parser.parse()
        orders_cfg = config.bronze_event["orders"]
        assert orders_cfg.per_grid_date_mode is True
        assert orders_cfg.value_counts_columns == ()

    def test_value_counts_columns_override_applied(self, sample_findings_dir):
        parser = FindingsParser(
            str(sample_findings_dir),
            bronze_aggregation_overrides={
                "orders": {"value_counts_columns": ("event_type",)}
            },
        )
        config = parser.parse()
        orders_cfg = config.bronze_event["orders"]
        assert orders_cfg.value_counts_columns == ("event_type",)

    def test_value_counts_columns_accepts_list(self, sample_findings_dir):
        # Users may write a list literal in their notebook cell — accept and
        # coerce to tuple for immutability.
        parser = FindingsParser(
            str(sample_findings_dir),
            bronze_aggregation_overrides={
                "orders": {"value_counts_columns": ["event_type", "status"]}
            },
        )
        orders_cfg = parser.parse().bronze_event["orders"]
        assert orders_cfg.value_counts_columns == ("event_type", "status")

    def test_windows_override_widens_to_intent_when_subset(self, sample_findings_dir):
        # Cycle 4 / G6 widening rule: when override is a subset of intent
        # (multi.aggregation_windows default is the 7-window list), the
        # override is treated as a redundant restatement and intent breadth
        # wins. Closes the silent-narrow case where an SPS operator restated
        # the default 6-window list and the 24h window dropped.
        parser = FindingsParser(
            str(sample_findings_dir),
            bronze_aggregation_overrides={
                "orders": {"windows": ["7d", "30d", "all_time"]}
            },
        )
        orders_cfg = parser.parse().bronze_event["orders"]
        assert orders_cfg.aggregation is not None
        assert orders_cfg.aggregation.windows == [
            "24h", "7d", "30d", "90d", "180d", "365d", "all_time",
        ]

    def test_windows_override_replaces_when_genuine_narrowing(self, sample_findings_dir):
        # Genuine narrowing: override contains a window NOT in intent
        # (multi.aggregation_windows default has no "1h"), so the override
        # is preserved verbatim — operator declared deliberate scope change.
        parser = FindingsParser(
            str(sample_findings_dir),
            bronze_aggregation_overrides={
                "orders": {"windows": ["1h", "7d"]}
            },
        )
        orders_cfg = parser.parse().bronze_event["orders"]
        assert orders_cfg.aggregation is not None
        assert orders_cfg.aggregation.windows == ["1h", "7d"]

    def test_combined_overrides(self, sample_findings_dir):
        # ["30d", "all_time"] IS a subset of the default 7-window intent,
        # so it widens to intent under the cycle-4 rule.
        parser = FindingsParser(
            str(sample_findings_dir),
            bronze_aggregation_overrides={
                "orders": {
                    "per_grid_date_mode": True,
                    "value_counts_columns": ("event_type",),
                    "windows": ["30d", "all_time"],
                }
            },
        )
        cfg = parser.parse().bronze_event["orders"]
        assert cfg.per_grid_date_mode is True
        assert cfg.value_counts_columns == ("event_type",)
        assert cfg.aggregation.windows == [
            "24h", "7d", "30d", "90d", "180d", "365d", "all_time",
        ]


class TestOverrideValidation:
    def test_unknown_key_raises(self, sample_findings_dir):
        parser = FindingsParser(
            str(sample_findings_dir),
            bronze_aggregation_overrides={"orders": {"bogus_setting": True}},
        )
        with pytest.raises(ValueError, match="unknown bronze aggregation override key"):
            parser.parse()

    def test_override_targeting_non_event_dataset_raises(self, sample_findings_dir):
        parser = FindingsParser(
            str(sample_findings_dir),
            bronze_aggregation_overrides={"customers": {"per_grid_date_mode": True}},
        )
        with pytest.raises(ValueError, match="not an event source"):
            parser.parse()


class TestGeneratorForwarding:
    def test_pipeline_generator_forwards_overrides(self, sample_findings_dir, tmp_path):
        gen = PipelineGenerator(
            findings_dir=str(sample_findings_dir),
            output_dir=str(tmp_path / "out"),
            pipeline_name="test_pipeline",
            bronze_aggregation_overrides={"orders": {"per_grid_date_mode": True}},
        )
        # Internal parser must carry the overrides through.
        assert gen._parser._bronze_aggregation_overrides == {
            "orders": {"per_grid_date_mode": True}
        }

    def test_databricks_generator_forwards_overrides(self, sample_findings_dir, tmp_path):
        gen = DatabricksPipelineGenerator(
            findings_dir=str(sample_findings_dir),
            output_dir=str(tmp_path / "out"),
            pipeline_name="test_pipeline",
            bronze_aggregation_overrides={"orders": {"per_grid_date_mode": True}},
        )
        assert gen._parser._bronze_aggregation_overrides == {
            "orders": {"per_grid_date_mode": True}
        }
