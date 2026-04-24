"""Tests for the interpretation-layer JSON sidecars."""
from __future__ import annotations

from datetime import datetime, timezone

from customer_retention.analysis.auto_explorer.run_namespace import RunNamespace
from customer_retention.stages.causal.column_descriptions_writer import ColumnDescriptionRow
from customer_retention.stages.causal.feature_meta_writer import FeatureMetaRow
from customer_retention.stages.causal.interpretation.sidecars import (
    load_column_descriptions_sidecar,
    load_feature_meta_sidecar,
    load_population_stats_sidecar,
    write_column_descriptions_sidecar,
    write_feature_meta_sidecar,
    write_population_stats_sidecar,
)
from customer_retention.stages.causal.population_stats import PopulationStatsRow, TopCategory


def _namespace(tmp_path):
    return RunNamespace(root=tmp_path, run_id="proj-abc")


class TestFeatureMetaSidecar:
    def test_roundtrip(self, tmp_path):
        ns = _namespace(tmp_path)
        rows = [
            FeatureMetaRow(
                composite_name="cn1", feature_name="nps_mean_90d",
                source_columns=["nps"], aggregation_kind="avg",
                window_days=90, polarity="high_is_good",
                business_phrase="average NPS score over last 90 days",
            ),
            FeatureMetaRow(composite_name="cn1", feature_name="event_count_30d"),
        ]
        write_feature_meta_sidecar(ns, "cn1", rows)
        loaded = load_feature_meta_sidecar(ns, composite_name="cn1")
        assert set(loaded.keys()) == {"nps_mean_90d", "event_count_30d"}
        assert loaded["nps_mean_90d"].business_phrase == "average NPS score over last 90 days"
        assert loaded["nps_mean_90d"].aggregation_kind == "avg"
        assert loaded["nps_mean_90d"].source_columns == ["nps"]

    def test_missing_file_returns_empty_dict(self, tmp_path):
        assert load_feature_meta_sidecar(_namespace(tmp_path)) == {}

    def test_composite_name_mismatch_returns_empty(self, tmp_path):
        ns = _namespace(tmp_path)
        write_feature_meta_sidecar(ns, "cn1", [
            FeatureMetaRow(composite_name="cn1", feature_name="x"),
        ])
        assert load_feature_meta_sidecar(ns, composite_name="cn2") == {}

    def test_no_composite_name_filter_loads_all(self, tmp_path):
        ns = _namespace(tmp_path)
        write_feature_meta_sidecar(ns, "cn1", [
            FeatureMetaRow(composite_name="cn1", feature_name="x"),
        ])
        loaded = load_feature_meta_sidecar(ns)
        assert "x" in loaded

    def test_malformed_json_returns_empty(self, tmp_path):
        ns = _namespace(tmp_path)
        directory = ns.feature_meta_dir
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "feature_meta.json").write_text("{not json")
        assert load_feature_meta_sidecar(ns) == {}

    def test_rows_without_feature_name_skipped(self, tmp_path):
        ns = _namespace(tmp_path)
        directory = ns.feature_meta_dir
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "feature_meta.json").write_text(
            '{"composite_name": "cn1", "rows": [{"composite_name": "cn1"}]}'
        )
        assert load_feature_meta_sidecar(ns) == {}


class TestPopulationStatsSidecar:
    def test_roundtrip_numeric(self, tmp_path):
        ns = _namespace(tmp_path)
        rows = [
            PopulationStatsRow(
                run_id="r1", feature_name="nps", dtype="numeric",
                q05=3.0, q25=5.0, q50=7.0, q75=9.0, q95=10.0,
            ),
        ]
        write_population_stats_sidecar(ns, rows)
        loaded = load_population_stats_sidecar(ns)
        assert "nps" in loaded
        stats = loaded["nps"]
        assert stats.q05 == 3.0
        assert stats.q75 == 9.0

    def test_roundtrip_categorical(self, tmp_path):
        ns = _namespace(tmp_path)
        rows = [
            PopulationStatsRow(
                run_id="r1", feature_name="segment", dtype="categorical",
                top_categories=[TopCategory(value="SMB", count=10, share=0.5)],
            ),
        ]
        write_population_stats_sidecar(ns, rows)
        loaded = load_population_stats_sidecar(ns)
        assert "segment" in loaded

    def test_missing_file_empty(self, tmp_path):
        assert load_population_stats_sidecar(_namespace(tmp_path)) == {}


class TestColumnDescriptionsSidecar:
    def test_roundtrip_with_datetime(self, tmp_path):
        ns = _namespace(tmp_path)
        verified = datetime(2026, 1, 1, tzinfo=timezone.utc)
        rows = [
            ColumnDescriptionRow(
                table="account", column_name="nps",
                business_name="Net Promoter Score",
                polarity="high_is_good", last_verified_at=verified,
            ),
        ]
        write_column_descriptions_sidecar(ns, rows)
        loaded = load_column_descriptions_sidecar(ns)
        entry = loaded["nps"]
        assert entry.business_name == "Net Promoter Score"
        assert entry.polarity == "high_is_good"

    def test_missing_file_empty(self, tmp_path):
        assert load_column_descriptions_sidecar(_namespace(tmp_path)) == {}

    def test_is_root_scoped_not_run_scoped(self, tmp_path):
        ns_a = RunNamespace(root=tmp_path, run_id="run-a")
        ns_b = RunNamespace(root=tmp_path, run_id="run-b")
        write_column_descriptions_sidecar(ns_a, [
            ColumnDescriptionRow(table="t", column_name="x", business_name="X"),
        ])
        # different run_id, same root → same sidecar visible
        loaded = load_column_descriptions_sidecar(ns_b)
        assert "x" in loaded
