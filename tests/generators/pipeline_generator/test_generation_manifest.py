from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

import pytest

from customer_retention.generators.pipeline_generator.generation_manifest import (
    BASELINE_TAG,
    MANIFEST_FILENAME,
    RELEASE,
    build_generation_manifest,
    template_versions_for,
    write_generation_manifest,
)
from customer_retention.generators.pipeline_generator.models import (
    GoldLayerConfig,
    LandingLayerConfig,
    PipelineConfig,
    PipelineTransformationType,
    SilverLayerConfig,
    SourceConfig,
    TransformationStep,
)


def _src():
    return SourceConfig(
        name="orders",
        path="/d/orders.csv",
        format="csv",
        entity_key="customer_id",
        time_column="order_date",
        is_event_level=True,
    )


def _landing_with_filter(predicate="amount > 0"):
    return LandingLayerConfig(
        source=_src(),
        raw_source_path="/d/orders.csv",
        raw_source_format="csv",
        entity_column="customer_id",
        time_column="order_date",
        target_column="churn",
        filters=[
            TransformationStep(
                type=PipelineTransformationType.LANDING_FILTER,
                column="orders",
                parameters={"predicate": predicate},
                rationale="r",
                source_notebook="nb",
            )
        ],
    )


def _pipeline_config(landing=None) -> PipelineConfig:
    return PipelineConfig(
        name="test_pipeline",
        target_column="churn",
        sources=[_src()],
        bronze={},
        silver=SilverLayerConfig(),
        gold=GoldLayerConfig(),
        output_dir=".",
        landing=landing or {},
    )


class TestTemplateVersions:
    def test_hash_is_deterministic(self):
        templates = {"landing": "abc", "bronze": "def"}
        a = template_versions_for(templates)
        b = template_versions_for(templates)
        assert a == b

    def test_different_templates_produce_different_hashes(self):
        v1 = template_versions_for({"landing": "abc"})
        v2 = template_versions_for({"landing": "def"})
        assert v1["landing"] != v2["landing"]

    def test_hash_prefix_is_sha256(self):
        v = template_versions_for({"x": "content"})
        assert v["x"].startswith("sha256:")
        assert len(v["x"]) == len("sha256:") + 16


class TestBuildGenerationManifest:
    def test_empty_config_produces_minimum_keys(self, tmp_path):
        config = _pipeline_config()
        manifest = build_generation_manifest(
            config, [], tmp_path,
            template_versions={}, kill_switch_active=False,
        )
        for key in [
            "generated_at", "baseline_tag", "release", "template_versions",
            "harvested_functions", "landing_filters", "lifecycle_enrichments",
            "kill_switch_active", "file_checksums",
        ]:
            assert key in manifest
        assert manifest["baseline_tag"] == BASELINE_TAG
        assert manifest["release"] == RELEASE
        assert manifest["harvested_functions"] == []
        assert manifest["landing_filters"] == []
        assert manifest["kill_switch_active"] is False

    def test_generated_at_is_iso_utc(self, tmp_path):
        fixed = datetime(2026, 4, 17, 10, 23, 14, tzinfo=timezone.utc)
        manifest = build_generation_manifest(
            _pipeline_config(), [], tmp_path,
            template_versions={}, kill_switch_active=False, now=fixed,
        )
        assert manifest["generated_at"] == "2026-04-17T10:23:14Z"

    def test_landing_filters_captured_with_dataset_and_predicate(self, tmp_path):
        config = _pipeline_config(landing={"orders": _landing_with_filter("x IS NOT NULL")})
        manifest = build_generation_manifest(
            config, [], tmp_path,
            template_versions={}, kill_switch_active=False,
        )
        assert manifest["landing_filters"] == [
            {"dataset": "orders", "predicate": "x IS NOT NULL"}
        ]

    def test_lifecycle_enrichments_captured_with_config_payload(self, tmp_path):
        cfg_payload = {
            "enriched_view_name": "v",
            "parent_entity_key": "pk",
            "valid_from_column": "from",
            "valid_to_columns": ["to"],
        }
        landing = _landing_with_filter()
        landing.filters = []
        landing.lifecycle_enrichments = [
            TransformationStep(
                type=PipelineTransformationType.LANDING_LIFECYCLE_ENRICHMENT,
                column="subscription",
                parameters={"config": cfg_payload},
                rationale="r",
                source_notebook="nb",
            )
        ]
        config = _pipeline_config(landing={"subscription": landing})
        manifest = build_generation_manifest(
            config, [], tmp_path,
            template_versions={}, kill_switch_active=False,
        )
        assert manifest["lifecycle_enrichments"] == [
            {"dataset": "subscription", "config": cfg_payload}
        ]

    def test_kill_switch_active_flag_propagates(self, tmp_path):
        manifest = build_generation_manifest(
            _pipeline_config(), [], tmp_path,
            template_versions={}, kill_switch_active=True,
        )
        assert manifest["kill_switch_active"] is True

    def test_file_checksums_are_sha256_of_content(self, tmp_path):
        f1 = tmp_path / "a.txt"
        f1.write_text("hello")
        f2 = tmp_path / "sub" / "b.txt"
        f2.parent.mkdir()
        f2.write_text("world")

        manifest = build_generation_manifest(
            _pipeline_config(), [f1, f2], tmp_path,
            template_versions={}, kill_switch_active=False,
        )
        expected_a = "sha256:" + hashlib.sha256(b"hello").hexdigest()
        expected_b = "sha256:" + hashlib.sha256(b"world").hexdigest()
        assert manifest["file_checksums"]["a.txt"] == expected_a
        assert manifest["file_checksums"]["sub/b.txt"] == expected_b

    def test_missing_files_are_skipped_silently(self, tmp_path):
        manifest = build_generation_manifest(
            _pipeline_config(), [tmp_path / "nope.txt"], tmp_path,
            template_versions={}, kill_switch_active=False,
        )
        assert manifest["file_checksums"] == {}


class TestWriteGenerationManifest:
    def test_writes_readable_json_at_expected_path(self, tmp_path):
        manifest = build_generation_manifest(
            _pipeline_config(), [], tmp_path,
            template_versions={"landing": "sha256:abc"}, kill_switch_active=False,
        )
        path = write_generation_manifest(manifest, tmp_path)
        assert path == tmp_path / MANIFEST_FILENAME
        loaded = json.loads(path.read_text())
        assert loaded["template_versions"] == {"landing": "sha256:abc"}


class TestPipelineGeneratorIntegration:
    """End-to-end: PipelineGenerator.generate() writes a valid manifest."""

    @pytest.fixture
    def generator(self, aggregated_event_setup, tmp_path):
        from customer_retention.generators.pipeline_generator.generator import PipelineGenerator
        out = tmp_path / "generated"
        return PipelineGenerator(
            findings_dir=str(aggregated_event_setup),
            output_dir=str(out),
            pipeline_name="test",
        )

    def test_generate_writes_generation_manifest(self, generator, tmp_path):
        generator.generate()
        manifest_path = tmp_path / "generated" / MANIFEST_FILENAME
        assert manifest_path.exists()
        data = json.loads(manifest_path.read_text())
        assert data["baseline_tag"] == BASELINE_TAG
        assert data["release"] == RELEASE
        assert data["kill_switch_active"] is False
        assert "landing.py.j2" in data["template_versions"]
        assert data["template_versions"]["landing.py.j2"].startswith("sha256:")
        assert len(data["file_checksums"]) > 0

    def test_landing_filter_shows_up_in_manifest(self, aggregated_event_setup, tmp_path):
        import yaml

        from customer_retention.generators.pipeline_generator.generator import PipelineGenerator

        (aggregated_event_setup / "recommendations.yaml").write_text(yaml.dump({
            "landing": {
                "filters": [{
                    "id": "landing_landing_filtering_orders_agg",
                    "layer": "landing",
                    "category": "landing_filtering",
                    "action": "filter",
                    "target_column": "orders_agg",
                    "parameters": {"dataset": "orders_agg", "predicate": "customer_id IS NOT NULL"},
                    "rationale": "r",
                    "source_notebook": "nb",
                    "priority": 1,
                    "dependencies": [],
                }],
                "lifecycle_enrichments": [],
            },
        }))

        out = tmp_path / "gen"
        PipelineGenerator(
            findings_dir=str(aggregated_event_setup),
            output_dir=str(out),
            pipeline_name="test",
        ).generate()
        data = json.loads((out / MANIFEST_FILENAME).read_text())
        assert data["landing_filters"] == [
            {"dataset": "orders_agg", "predicate": "customer_id IS NOT NULL"}
        ]


@pytest.fixture
def aggregated_event_setup(tmp_path):
    """Copy of the same fixture used in test_findings_parser to keep this
    suite self-contained."""
    import yaml

    findings_dir = tmp_path / "findings"
    findings_dir.mkdir()
    agg_parquet = str(findings_dir / "orders_agg.parquet")
    raw_csv = "/data/raw/orders.csv"

    multi_dataset = {
        "datasets": {
            "customers": {
                "name": "customers",
                "findings_path": str(findings_dir / "customers_findings.yaml"),
                "source_path": "/data/customers.csv",
                "granularity": "entity_level",
                "row_count": 1000,
                "column_count": 3,
                "excluded": False,
            },
            "orders_agg": {
                "name": "orders_agg",
                "findings_path": str(findings_dir / "orders_agg_findings.yaml"),
                "source_path": agg_parquet,
                "granularity": "entity_level",
                "row_count": 500,
                "column_count": 6,
                "excluded": False,
            },
        },
        "relationships": [{
            "left_dataset": "customers",
            "right_dataset": "orders_agg",
            "left_column": "customer_id",
            "right_column": "customer_id",
            "relationship_type": "one_to_one",
            "confidence": 1.0,
        }],
        "primary_entity_dataset": "customers",
        "event_datasets": [],
        "excluded_datasets": [],
    }
    (findings_dir / "multi_dataset_findings.yaml").write_text(yaml.dump(multi_dataset))

    (findings_dir / "customers_findings.yaml").write_text(yaml.dump({
        "source_path": "/data/customers.csv",
        "source_format": "csv",
        "row_count": 1000,
        "column_count": 3,
        "columns": {
            "customer_id": {"name": "customer_id", "inferred_type": "identifier",
                            "confidence": 0.95, "evidence": [], "quality_score": 100,
                            "cleaning_needed": False, "cleaning_recommendations": []},
            "churn": {"name": "churn", "inferred_type": "binary",
                      "confidence": 0.99, "evidence": [], "quality_score": 100,
                      "cleaning_needed": False, "cleaning_recommendations": []},
        },
        "target_column": "churn",
        "identifier_columns": ["customer_id"],
    }))
    (findings_dir / "orders_agg_findings.yaml").write_text(yaml.dump({
        "source_path": agg_parquet,
        "source_format": "parquet",
        "row_count": 500,
        "column_count": 6,
        "columns": {
            "customer_id": {"name": "customer_id", "inferred_type": "identifier",
                            "confidence": 0.95, "evidence": [], "quality_score": 100,
                            "cleaning_needed": False, "cleaning_recommendations": []},
            "total_amount": {"name": "total_amount", "inferred_type": "numeric_continuous",
                             "confidence": 0.9, "evidence": [], "quality_score": 100,
                             "cleaning_needed": True,
                             "cleaning_recommendations": ["impute_null:0"]},
        },
        "identifier_columns": ["customer_id"],
    }))
    (findings_dir / "orders_raw_findings.yaml").write_text(yaml.dump({
        "source_path": raw_csv,
        "source_format": "csv",
        "row_count": 5000,
        "column_count": 4,
        "columns": {
            "order_id": {"name": "order_id", "inferred_type": "identifier",
                         "confidence": 0.95, "evidence": [], "quality_score": 100,
                         "cleaning_needed": False, "cleaning_recommendations": []},
            "customer_id": {"name": "customer_id", "inferred_type": "identifier",
                            "confidence": 0.95, "evidence": [], "quality_score": 100,
                            "cleaning_needed": False, "cleaning_recommendations": []},
            "amount": {"name": "amount", "inferred_type": "numeric_continuous",
                       "confidence": 0.9, "evidence": [], "quality_score": 90,
                       "cleaning_needed": True,
                       "cleaning_recommendations": ["cap_outlier:iqr"]},
            "order_date": {"name": "order_date", "inferred_type": "datetime",
                           "confidence": 0.95, "evidence": [], "quality_score": 100,
                           "cleaning_needed": False, "cleaning_recommendations": []},
        },
        "identifier_columns": ["order_id"],
        "datetime_columns": ["order_date"],
        "time_series_metadata": {
            "granularity": "event_level",
            "entity_column": "customer_id",
            "time_column": "order_date",
            "aggregation_executed": True,
            "aggregated_findings_path": str(findings_dir / "orders_agg_findings.yaml"),
            "suggested_aggregations": ["7d", "30d", "90d"],
            "aggregation_windows_used": ["7d", "30d", "90d"],
        },
    }))
    return findings_dir
