"""Landing-sibling findings resolver (Cycle 002).

`<name>_findings.yaml` is the landing-stage profile emitted by NB01 —
post-NB00/NB01 enrichments (datetime derivation, lifecycle enrichment,
type coercion) but pre-aggregation. It is preferred over the raw CSV
because landing enrichments may introduce columns the bronze aggregator
needs (e.g. `event_timestamp`, `event_type`, `{col}_hour`).

When `exploration_manager._source_event_level_metadata` (5ebc83a) preserves
EVENT_LEVEL granularity for a dataset whose `_aggregated_findings.yaml`
lacks `time_series_metadata`, the generator must still see the richer
landing-stage findings to route numeric columns to `value_columns` and
to populate `lag_columns`. Without the resolver below,
`_build_aggregation_config` iterates the 18-col aggregated profile and
emits `value_columns=[]`, which crashes downstream with
`AssertionError: exprs should not be empty` in Spark.

This test file asserts the three supported shapes pass through the
resolver correctly:

1. Single-dataset event_level with `_aggregated_findings.yaml` + sibling
   landing findings (engagement_03 shape). value/lag columns come from
   the landing profile.
2. Multi-dataset event_level mirror (engagement_02 `case` shape). Same
   behavior — landing sibling drives column routing.
3. Entity-level with landing-only findings. Resolver is a no-op;
   aggregation proceeds from the single findings object as before.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import pytest
import yaml


def _write_findings_yaml(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload))


def _write_multi_dataset(path: Path, payload: Dict) -> None:
    path.write_text(yaml.safe_dump(payload))


@pytest.fixture
def engagement_03_shape(tmp_path: Path) -> Path:
    """Recreate the engagement_03 `email-779d2d18` artifact layout.

    Landing findings carry EVENT_LEVEL and numeric columns; aggregated
    findings carry the 18-col post-aggregation profile with
    `time_series_metadata: null`. `multi_dataset_findings` points at the
    aggregated path (NB03 behavior post-5ebc83a).
    """
    ds_dir = tmp_path / "customer_emails" / "findings"
    raw = ds_dir / "customer_emails_findings.yaml"
    agg = ds_dir / "customer_emails_aggregated_findings.yaml"
    _write_findings_yaml(raw, {
        "source_path": "/data/customer_emails.csv",
        "source_format": "csv",
        "row_count": 83000, "column_count": 13,
        "target_column": "unsubscribed",
        "identifier_columns": ["customer_id"],
        "datetime_columns": ["sent_date"],
        "columns": {
            "customer_id": {"name": "customer_id", "inferred_type": "identifier", "confidence": 1.0, "evidence": []},
            "sent_date":  {"name": "sent_date", "inferred_type": "datetime", "confidence": 1.0, "evidence": []},
            "send_hour":  {"name": "send_hour", "inferred_type": "numeric_discrete", "confidence": 0.95, "evidence": []},
            "time_to_open_hours": {"name": "time_to_open_hours", "inferred_type": "numeric_continuous", "confidence": 0.95, "evidence": []},
            "campaign_type": {"name": "campaign_type", "inferred_type": "categorical_nominal", "confidence": 0.95, "evidence": []},
            "opened":    {"name": "opened", "inferred_type": "binary", "confidence": 1.0, "evidence": []},
            "clicked":   {"name": "clicked", "inferred_type": "binary", "confidence": 1.0, "evidence": []},
            "bounced":   {"name": "bounced", "inferred_type": "binary", "confidence": 1.0, "evidence": []},
            "unsubscribed": {"name": "unsubscribed", "inferred_type": "target", "confidence": 1.0, "evidence": []},
        },
        "time_series_metadata": {
            "granularity": "event_level",
            "entity_column": "customer_id",
            "time_column": "sent_date",
            "avg_events_per_entity": 16.5,
            "time_span_days": 3285,
            "aggregation_windows_used": ["180d", "365d", "all_time"],
        },
        "metadata": {
            "temporal_patterns": {"lag_features_computed": True, "lag_columns": []},
        },
    })
    _write_findings_yaml(agg, {
        "source_path": "/data/bronze/customer_emails_aggregated",
        "source_format": "delta",
        "row_count": 1000, "column_count": 18,
        "target_column": "unsubscribed",
        "identifier_columns": ["customer_id"],
        "datetime_columns": [],
        "columns": {
            "customer_id": {"name": "customer_id", "inferred_type": "identifier", "confidence": 1.0, "evidence": []},
            "event_count_180d":     {"name": "event_count_180d", "inferred_type": "numeric_discrete", "confidence": 1.0, "evidence": []},
            "event_count_365d":     {"name": "event_count_365d", "inferred_type": "numeric_discrete", "confidence": 1.0, "evidence": []},
            "event_count_all_time": {"name": "event_count_all_time", "inferred_type": "numeric_continuous", "confidence": 1.0, "evidence": []},
            "lifecycle_quadrant":   {"name": "lifecycle_quadrant", "inferred_type": "categorical_nominal", "confidence": 1.0, "evidence": []},
            "recency_bucket":       {"name": "recency_bucket", "inferred_type": "categorical_nominal", "confidence": 1.0, "evidence": []},
            "unsubscribed":         {"name": "unsubscribed", "inferred_type": "target", "confidence": 1.0, "evidence": []},
        },
        "time_series_metadata": None,
    })
    _write_multi_dataset(tmp_path / "multi_dataset_findings.yaml", {
        "datasets": {
            "customer_emails": {
                "name": "customer_emails",
                "findings_path": str(agg),
                "source_path": "/data/bronze/customer_emails_aggregated",
                "granularity": "event_level",
                "row_count": 1000, "column_count": 18,
                "entity_column": "customer_id",
                "time_column": "sent_date",
                "target_column": "unsubscribed",
            },
        },
        "primary_entity_dataset": None,
        "event_datasets": ["customer_emails"],
        "excluded_datasets": [],
        "aggregation_windows": ["24h", "7d", "30d", "90d", "180d", "365d", "all_time"],
    })
    return tmp_path


class TestLandingSiblingResolverEventLevel:
    def test_value_columns_resolved_from_landing_sibling(self, engagement_03_shape: Path):
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser
        parser = FindingsParser(str(engagement_03_shape))
        config = parser.parse()

        event_cfg = config.bronze_event["customer_emails"]
        assert event_cfg.aggregation is not None
        value_cols = set(event_cfg.aggregation.value_columns)

        assert {"send_hour", "time_to_open_hours"}.issubset(value_cols), (
            f"landing-sibling numeric columns must populate value_columns — got {value_cols}"
        )
        assert "customer_id" not in value_cols
        assert "unsubscribed" not in value_cols
        assert "sent_date" not in value_cols

    def test_lag_columns_resolved_from_landing_sibling(self, engagement_03_shape: Path):
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser
        parser = FindingsParser(str(engagement_03_shape))
        config = parser.parse()

        event_cfg = config.bronze_event["customer_emails"]
        assert event_cfg.temporal_features is not None
        lag_cols = set(event_cfg.temporal_features.lag_columns)
        assert {"send_hour", "time_to_open_hours"}.issubset(lag_cols), (
            f"lag_columns must be derived from landing-sibling numerics — got {lag_cols}"
        )

    def test_binary_and_categorical_columns_split_correctly(self, engagement_03_shape: Path):
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser
        parser = FindingsParser(str(engagement_03_shape))
        config = parser.parse()

        agg = config.bronze_event["customer_emails"].aggregation
        assert {"opened", "clicked", "bounced"} == set(agg.binary_columns)
        assert "campaign_type" in agg.categorical_columns
        for binary_col in ("opened", "clicked", "bounced"):
            assert binary_col in agg.categorical_columns


class TestLandingSiblingResolverEntityLevelUnchanged:
    """NR1: entity-level datasets stay on the pre-existing code path."""

    def test_entity_level_without_aggregated_sibling_uses_landing_directly(self, tmp_path: Path):
        ds_dir = tmp_path / "accounts" / "findings"
        raw = ds_dir / "accounts_findings.yaml"
        _write_findings_yaml(raw, {
            "source_path": "/data/accounts.csv",
            "source_format": "csv",
            "row_count": 4998, "column_count": 5,
            "target_column": "churned",
            "identifier_columns": ["account_id"],
            "datetime_columns": [],
            "columns": {
                "account_id": {"name": "account_id", "inferred_type": "identifier", "confidence": 1.0, "evidence": []},
                "mrr":        {"name": "mrr", "inferred_type": "numeric_continuous", "confidence": 1.0, "evidence": []},
                "tier":       {"name": "tier", "inferred_type": "categorical_nominal", "confidence": 1.0, "evidence": []},
                "churned":    {"name": "churned", "inferred_type": "target", "confidence": 1.0, "evidence": []},
            },
            "time_series_metadata": None,
        })
        _write_multi_dataset(tmp_path / "multi_dataset_findings.yaml", {
            "datasets": {
                "accounts": {
                    "name": "accounts",
                    "findings_path": str(raw),
                    "source_path": "/data/accounts.csv",
                    "granularity": "entity_level",
                    "row_count": 4998, "column_count": 5,
                    "entity_column": "account_id",
                    "target_column": "churned",
                },
            },
            "primary_entity_dataset": "accounts",
            "event_datasets": [],
        })

        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser
        parser = FindingsParser(str(tmp_path))
        config = parser.parse()

        assert "accounts" not in (config.bronze_event or {})
        assert parser._landing_sibling_findings == {}


class TestLandingSiblingIgnoredWhenSiblingEntityLevel:
    """Hypothetical: aggregated findings exist but the sibling is entity-level,
    meaning the aggregator somehow emitted aggregated output for an
    entity-level dataset. In that case the resolver should stay out of the
    way — iterating the aggregated columns is the right choice."""

    def test_entity_level_sibling_is_not_registered(self, tmp_path: Path):
        ds_dir = tmp_path / "snapshots" / "findings"
        raw = ds_dir / "snapshots_findings.yaml"
        agg = ds_dir / "snapshots_aggregated_findings.yaml"
        _write_findings_yaml(raw, {
            "source_path": "/data/snapshots.csv",
            "source_format": "csv",
            "row_count": 100, "column_count": 3,
            "target_column": "churned",
            "identifier_columns": ["account_id"],
            "datetime_columns": [],
            "columns": {
                "account_id": {"name": "account_id", "inferred_type": "identifier", "confidence": 1.0, "evidence": []},
                "churned":    {"name": "churned", "inferred_type": "target", "confidence": 1.0, "evidence": []},
            },
            "time_series_metadata": {
                "granularity": "entity_level",
                "entity_column": "account_id",
                "time_column": None,
            },
        })
        _write_findings_yaml(agg, {
            "source_path": "/data/bronze/snapshots_aggregated",
            "source_format": "delta",
            "row_count": 100, "column_count": 3,
            "target_column": "churned",
            "identifier_columns": ["account_id"],
            "datetime_columns": [],
            "columns": {
                "account_id": {"name": "account_id", "inferred_type": "identifier", "confidence": 1.0, "evidence": []},
                "churned":    {"name": "churned", "inferred_type": "target", "confidence": 1.0, "evidence": []},
            },
            "time_series_metadata": None,
        })
        _write_multi_dataset(tmp_path / "multi_dataset_findings.yaml", {
            "datasets": {
                "snapshots": {
                    "name": "snapshots",
                    "findings_path": str(agg),
                    "source_path": "/data/bronze/snapshots_aggregated",
                    "granularity": "entity_level",
                    "row_count": 100, "column_count": 3,
                    "entity_column": "account_id",
                    "target_column": "churned",
                },
            },
            "primary_entity_dataset": "snapshots",
            "event_datasets": [],
        })

        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser
        parser = FindingsParser(str(tmp_path))
        _ = parser.parse()
        assert parser._landing_sibling_findings == {}
