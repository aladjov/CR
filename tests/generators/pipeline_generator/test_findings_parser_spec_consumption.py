from pathlib import Path

import pytest
import yaml

from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser
from customer_retention.stages.modeling.feature_spec import (
    FeatureSpec,
    FittedTransform,
)


def _write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        yaml.safe_dump(data, f)


def _fake_namespace(tmp_path: Path):
    from customer_retention.analysis.auto_explorer.run_namespace import RunNamespace
    ns = RunNamespace(root=tmp_path, run_id="test-run")
    ns.merged_dir.mkdir(parents=True, exist_ok=True)
    return ns


def _build_spec(selected_features):
    return FeatureSpec(
        spec_version=1, exploration_run_id="test-run",
        target_column="churn", entity_column="entity_id",
        timestamp_column="as_of_date", horizon_days=90,
        selected_features=list(selected_features),
        fitted_transforms=[
            FittedTransform(column=c, action="impute", method="median")
            for c in selected_features
        ],
    )


class TestFeatureSpecLoad:
    def test_returns_none_when_no_namespace_and_no_file(self, tmp_path):
        parser = FindingsParser(findings_dir=str(tmp_path))
        assert parser._load_feature_spec() is None

    def test_loads_from_namespace_path(self, tmp_path):
        ns = _fake_namespace(tmp_path)
        spec = _build_spec(["a", "b"])
        spec.save(ns.feature_spec_path)
        parser = FindingsParser(findings_dir=str(tmp_path), namespace=ns)
        loaded = parser._load_feature_spec()
        assert loaded is not None
        assert loaded.selected_features == ["a", "b"]

    def test_loads_from_findings_dir_fallback(self, tmp_path):
        spec = _build_spec(["c"])
        spec.save(tmp_path / "feature_spec.yaml")
        parser = FindingsParser(findings_dir=str(tmp_path))
        loaded = parser._load_feature_spec()
        assert loaded is not None
        assert loaded.selected_features == ["c"]


class TestAllowlistDrops:
    def test_drops_are_pipeline_columns_minus_selected_minus_metadata(self, tmp_path):
        parser = FindingsParser(findings_dir=str(tmp_path))
        spec = _build_spec(["feature_a", "feature_b"])
        pipeline = {
            "entity_id", "as_of_date", "churn",
            "feature_a", "feature_b",
            "unselected_c", "unselected_d",
            "original_churn",
        }
        drops = parser._collect_allowlist_drops(spec, pipeline, target_column="churn")
        assert drops == {"unselected_c", "unselected_d"}

    def test_keeps_entity_timestamp_target_and_originals(self, tmp_path):
        parser = FindingsParser(findings_dir=str(tmp_path))
        spec = _build_spec(["x"])
        pipeline = {
            "entity_id", "as_of_date", "churn",
            "feature_timestamp", "label_timestamp", "label_available_flag",
            "original_churn", "x", "drop_me",
        }
        drops = parser._collect_allowlist_drops(spec, pipeline, target_column="churn")
        assert drops == {"drop_me"}

    def test_raises_when_selected_feature_missing_from_pipeline(self, tmp_path):
        parser = FindingsParser(findings_dir=str(tmp_path))
        spec = _build_spec(["missing_feature", "exists"])
        pipeline = {"entity_id", "as_of_date", "churn", "exists"}
        with pytest.raises(ValueError, match="parity violation"):
            parser._collect_allowlist_drops(spec, pipeline, target_column="churn")

    def test_custom_entity_and_timestamp_columns_respected(self, tmp_path):
        parser = FindingsParser(findings_dir=str(tmp_path))
        spec = FeatureSpec(
            exploration_run_id="r", target_column="churn",
            entity_column="customer_id", timestamp_column="snapshot_date",
            horizon_days=30, selected_features=["x"],
            fitted_transforms=[FittedTransform(column="x", action="impute", method="median")],
        )
        pipeline = {"customer_id", "snapshot_date", "churn", "x", "drop_me"}
        drops = parser._collect_allowlist_drops(spec, pipeline, target_column="churn")
        assert drops == {"drop_me"}


class TestParserSpecIntegration:
    def _write_minimal_fixture(self, tmp_path: Path, spec: FeatureSpec):
        ns = _fake_namespace(tmp_path)
        spec.save(ns.feature_spec_path)

        columns_dict = {
            feat: {
                "name": feat, "inferred_type": "numeric_continuous", "confidence": 0.9,
                "evidence": [], "alternatives": [], "universal_metrics": {}, "type_metrics": {},
                "quality_issues": [], "quality_score": 100.0, "cleaning_needed": False,
                "cleaning_recommendations": [], "transformation_recommendations": [],
            }
            for feat in spec.selected_features
        }
        findings_doc = {
            "source_path": str(tmp_path / "data.csv"),
            "source_format": "csv",
            "row_count": 1000, "column_count": 3 + len(spec.selected_features),
            "target_column": "churn",
            "identifier_columns": ["entity_id"],
            "datetime_columns": ["as_of_date"],
            "columns": columns_dict,
        }
        findings_path = ns.dataset_findings_dir("src") / "src_findings.yaml"
        _write_yaml(findings_path, findings_doc)

        multi_doc = {
            "datasets": {
                "src": {
                    "name": "src",
                    "findings_path": str(findings_path),
                    "source_path": str(tmp_path / "data.csv"),
                    "granularity": "entity_level",
                    "row_count": 1000, "column_count": 3 + len(spec.selected_features),
                    "entity_column": "entity_id",
                    "target_column": "churn",
                },
            },
            "primary_entity_dataset": "src",
            "event_datasets": [],
            "excluded_datasets": [],
        }
        _write_yaml(ns.multi_dataset_findings_path, multi_doc)
        return ns

    def test_parse_stores_spec_path_on_config_when_present(self, tmp_path):
        spec = _build_spec(["a", "b"])
        ns = self._write_minimal_fixture(tmp_path, spec)
        parser = FindingsParser(findings_dir=str(ns.merged_dir), namespace=ns)
        config = parser.parse()
        assert config.feature_spec_path == str(ns.feature_spec_path)
        if config.training is not None:
            assert config.training.feature_spec_path == str(ns.feature_spec_path)

    def test_parse_without_spec_leaves_spec_path_none(self, tmp_path):
        ns = self._write_minimal_fixture(tmp_path, _build_spec(["a"]))
        ns.feature_spec_path.unlink()
        parser = FindingsParser(findings_dir=str(ns.merged_dir), namespace=ns)
        config = parser.parse()
        assert config.feature_spec_path is None
