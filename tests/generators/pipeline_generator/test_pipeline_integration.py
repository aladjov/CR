import ast
import json

import pytest
import yaml

from customer_retention.generators.pipeline_generator.generator import PipelineGenerator


@pytest.fixture
def full_findings_setup(tmp_path):
    findings_dir = tmp_path / "findings"
    findings_dir.mkdir()

    multi_dataset = {
        "datasets": {
            "customers": {
                "name": "customers",
                "findings_path": str(findings_dir / "customers_findings.yaml"),
                "source_path": str(tmp_path / "customers.csv"),
                "granularity": "entity_level",
                "row_count": 1000, "column_count": 5,
                "entity_column": "customer_id",
                "target_column": "churn",
                "excluded": False,
                # Opt `age` into the gold zero-inflation handling so the
                # recommendation in this fixture survives the NB05 opt-in gate.
                # Without this, the framework default suppresses _is_zero/_log.
                "zero_inflation_opt_in": ["age"],
            },
            "orders": {
                "name": "orders",
                "findings_path": str(findings_dir / "orders_findings.yaml"),
                "source_path": str(tmp_path / "orders.csv"),
                "granularity": "event_level",
                "row_count": 5000, "column_count": 4,
                "entity_column": "customer_id",
                "time_column": "order_date",
                "excluded": False,
            },
            "inactive": {
                "name": "inactive",
                "findings_path": str(findings_dir / "inactive_findings.yaml"),
                "source_path": str(tmp_path / "inactive.csv"),
                "granularity": "entity_level",
                "row_count": 200, "column_count": 3,
                "excluded": True,
            }
        },
        "relationships": [
            {"left_dataset": "customers", "right_dataset": "orders",
             "left_column": "customer_id", "right_column": "customer_id",
             "relationship_type": "one_to_many", "confidence": 1.0}
        ],
        "primary_entity_dataset": "customers",
        "event_datasets": ["orders"],
        "excluded_datasets": ["inactive"],
        "aggregation_windows": ["7d", "30d", "90d"],
        "notes": {
            "temporal_config": {
                "feature_groups": ["lifecycle", "recency", "regularity"],
            }
        }
    }
    (findings_dir / "multi_dataset_findings.yaml").write_text(yaml.dump(multi_dataset))

    for name, cols in [
        ("customers", {
            "customer_id": {"name": "customer_id", "inferred_type": "identifier", "confidence": 0.95,
                           "evidence": [], "quality_score": 100, "cleaning_needed": False, "cleaning_recommendations": []},
            "age": {"name": "age", "inferred_type": "numeric_continuous", "confidence": 0.9, "evidence": [],
                   "quality_score": 85, "cleaning_needed": True,
                   "cleaning_recommendations": ["impute_null:median"]},
            "status": {"name": "status", "inferred_type": "categorical_nominal", "confidence": 0.9, "evidence": [],
                      "quality_score": 100, "cleaning_needed": False, "cleaning_recommendations": []},
            "churn": {"name": "churn", "inferred_type": "binary", "confidence": 0.99, "evidence": [],
                     "quality_score": 100, "cleaning_needed": False, "cleaning_recommendations": []},
        }),
        ("orders", {
            "order_id": {"name": "order_id", "inferred_type": "identifier", "confidence": 0.95, "evidence": [],
                        "quality_score": 100, "cleaning_needed": False, "cleaning_recommendations": []},
            "customer_id": {"name": "customer_id", "inferred_type": "identifier", "confidence": 0.95, "evidence": [],
                          "quality_score": 100, "cleaning_needed": False, "cleaning_recommendations": []},
            "amount": {"name": "amount", "inferred_type": "numeric_continuous", "confidence": 0.9, "evidence": [],
                      "quality_score": 90, "cleaning_needed": True,
                      "cleaning_recommendations": ["cap_outlier:iqr"]},
            "order_date": {"name": "order_date", "inferred_type": "datetime", "confidence": 0.95, "evidence": [],
                          "quality_score": 100, "cleaning_needed": False, "cleaning_recommendations": []},
        }),
        ("inactive", {
            "customer_id": {"name": "customer_id", "inferred_type": "identifier", "confidence": 0.95, "evidence": [],
                          "quality_score": 100, "cleaning_needed": False, "cleaning_recommendations": []},
            "status": {"name": "status", "inferred_type": "categorical_nominal", "confidence": 0.9, "evidence": [],
                      "quality_score": 100, "cleaning_needed": False, "cleaning_recommendations": []},
            "churn": {"name": "churn", "inferred_type": "binary", "confidence": 0.99, "evidence": [],
                     "quality_score": 100, "cleaning_needed": False, "cleaning_recommendations": []},
        }),
    ]:
        findings = {
            "source_path": str(tmp_path / f"{name}.csv"),
            "source_format": "csv",
            "row_count": multi_dataset["datasets"][name]["row_count"],
            "column_count": multi_dataset["datasets"][name]["column_count"],
            "columns": cols,
            "target_column": "churn",
            "identifier_columns": ["customer_id"] if "customer_id" in cols else ["order_id"],
        }
        if name == "orders":
            findings["datetime_columns"] = ["order_date"]
            findings["time_series_metadata"] = {
                "granularity": "event_level",
                "entity_column": "customer_id",
                "time_column": "order_date",
                "aggregation_windows_used": ["7d", "30d", "90d"],
            }
        (findings_dir / f"{name}_findings.yaml").write_text(yaml.dump(findings))

    recommendations = {
        "bronze": {
            "source_file": str(tmp_path / "customers.csv"),
            "null_handling": [
                {"id": "bronze_null_age", "layer": "bronze", "category": "null", "action": "impute",
                 "target_column": "age", "parameters": {"strategy": "median"},
                 "rationale": "15% missing", "source_notebook": "03", "priority": 1, "dependencies": [], "fit_artifact_id": None}
            ],
            "outlier_handling": [
                {"id": "bronze_outlier_amount", "layer": "bronze", "category": "outlier",
                 "action": "winsorize", "target_column": "amount",
                 "parameters": {"method": "iqr", "lower_bound": 0, "upper_bound": 500},
                 "rationale": "5% outliers", "source_notebook": "03", "priority": 1, "dependencies": [], "fit_artifact_id": None},
                {"id": "bronze_outlier_age_seg", "layer": "bronze", "category": "outlier",
                 "action": "segment_aware_cap", "target_column": "age",
                 "parameters": {"method": "segment_iqr", "n_segments": 2},
                 "rationale": "Bimodal", "source_notebook": "03", "priority": 1, "dependencies": [], "fit_artifact_id": None},
            ],
            "type_conversions": [], "deduplication": [], "filtering": [], "text_processing": [], "modeling_strategy": [],
        },
        "sources": {},
        "silver": {
            "entity_column": "customer_id",
            "time_column": None,
            "joins": [], "aggregations": [],
            "derived_columns": [
                {"id": "silver_ratio", "layer": "silver", "category": "derived", "action": "ratio",
                 "target_column": "age_to_amount_sum_7d_ratio",
                 "parameters": {"feature_type": "ratio", "numerator": "age", "denominator": "amount_sum_7d",
                                "expression": "age / amount_sum_7d"},
                 "rationale": "Ratio feature", "source_notebook": "04", "priority": 1, "dependencies": [], "fit_artifact_id": None},
                {"id": "silver_interaction", "layer": "silver", "category": "derived", "action": "interaction",
                 "target_column": "age_x_amount_sum_7d",
                 "parameters": {"feature_type": "interaction", "features": ["age", "amount_sum_7d"],
                                "expression": "age * amount_sum_7d"},
                 "rationale": "Interaction", "source_notebook": "04", "priority": 1, "dependencies": [], "fit_artifact_id": None},
            ],
        },
        "gold": {
            "target_column": "churn",
            "encoding": [
                {"id": "gold_enc_status", "layer": "gold", "category": "encoding", "action": "one_hot",
                 "target_column": "status", "parameters": {"method": "one_hot"},
                 "rationale": "Low card", "source_notebook": "02", "priority": 1, "dependencies": [], "fit_artifact_id": None},
            ],
            "scaling": [
                {"id": "gold_scale_age", "layer": "gold", "category": "scaling", "action": "standard",
                 "target_column": "age", "parameters": {"method": "standard"},
                 "rationale": "Normal dist", "source_notebook": "06", "priority": 1, "dependencies": [], "fit_artifact_id": None},
            ],
            "feature_selection": [
                {"id": "gold_drop_amount", "layer": "gold", "category": "feature_selection",
                 "action": "drop_multicollinear", "target_column": "amount_sum_7d",
                 "parameters": {"correlated_with": "age", "correlation": 0.85},
                 "rationale": "High corr", "source_notebook": "04", "priority": 1, "dependencies": [], "fit_artifact_id": None},
            ],
            "transformations": [
                {"id": "gold_transform_age_ctl", "layer": "gold", "category": "transformation",
                 "action": "cap_then_log", "target_column": "age",
                 "parameters": {"cap_method": "iqr", "cap_multiplier": 1.5},
                 "rationale": "Skewed", "source_notebook": "02", "priority": 1, "dependencies": [], "fit_artifact_id": None},
                {"id": "gold_transform_age_zi", "layer": "gold", "category": "transformation",
                 "action": "zero_inflation_handling", "target_column": "age",
                 "parameters": {"strategy": "separate_indicator", "transform_non_zero": "log"},
                 "rationale": "Zero-inflated", "source_notebook": "02", "priority": 1, "dependencies": [], "fit_artifact_id": None},
            ],
        },
    }
    (findings_dir / "test_recommendations.yaml").write_text(yaml.dump(recommendations))

    return tmp_path, findings_dir


@pytest.fixture
def generated_output(full_findings_setup):
    tmp_path, findings_dir = full_findings_setup
    output_dir = tmp_path / "output"
    gen = PipelineGenerator(str(findings_dir), str(output_dir), "test_pipeline")
    gen.generate()
    return output_dir


class TestRecommendationCoverage:
    def test_parser_loads_recommendations(self, full_findings_setup):
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser
        _, findings_dir = full_findings_setup
        parser = FindingsParser(str(findings_dir))
        config = parser.parse()
        assert config.recommendations_hash is not None

    def test_bronze_has_recommendation_transformations(self, full_findings_setup):
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser
        from customer_retention.generators.pipeline_generator.models import PipelineTransformationType
        _, findings_dir = full_findings_setup
        parser = FindingsParser(str(findings_dir))
        config = parser.parse()
        all_bronze_types = set()
        for bronze in config.bronze.values():
            for t in bronze.transformations:
                all_bronze_types.add(t.type)
        assert PipelineTransformationType.WINSORIZE in all_bronze_types or PipelineTransformationType.SEGMENT_AWARE_CAP in all_bronze_types

    def test_silver_has_derived_columns(self, full_findings_setup):
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser
        _, findings_dir = full_findings_setup
        parser = FindingsParser(str(findings_dir))
        config = parser.parse()
        assert len(config.silver.derived_columns) >= 2
        actions = {dc.parameters.get("action") for dc in config.silver.derived_columns}
        assert "ratio" in actions
        assert "interaction" in actions

    def test_gold_has_transformations(self, full_findings_setup):
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser
        from customer_retention.generators.pipeline_generator.models import PipelineTransformationType
        _, findings_dir = full_findings_setup
        parser = FindingsParser(str(findings_dir))
        config = parser.parse()
        assert len(config.gold.transformations) >= 2
        gold_types = {t.type for t in config.gold.transformations}
        assert PipelineTransformationType.CAP_THEN_LOG in gold_types
        assert PipelineTransformationType.ZERO_INFLATION_HANDLING in gold_types

    def test_gold_has_feature_selections(self, full_findings_setup):
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser
        _, findings_dir = full_findings_setup
        parser = FindingsParser(str(findings_dir))
        config = parser.parse()
        assert len(config.gold.feature_selections) >= 1
        assert "amount_sum_7d" in config.gold.feature_selections

    def test_gold_has_recommendation_encodings(self, full_findings_setup):
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser
        _, findings_dir = full_findings_setup
        parser = FindingsParser(str(findings_dir))
        config = parser.parse()
        encoding_cols = {e.column for e in config.gold.encodings}
        assert "status" in encoding_cols

    def test_gold_has_recommendation_scalings(self, full_findings_setup):
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser
        _, findings_dir = full_findings_setup
        parser = FindingsParser(str(findings_dir))
        config = parser.parse()
        scaling_cols = {s.column for s in config.gold.scalings}
        assert "age" in scaling_cols


class TestMultiDatasetGeneration:
    def test_generates_full_file_structure(self, generated_output):
        file_names = [f.name for f in generated_output.rglob("*") if f.is_file()]
        assert "config.py" in file_names
        assert "pipeline_runner.py" in file_names
        assert "workflow.json" in file_names
        assert "run_all.py" in file_names

    def test_all_python_files_valid(self, generated_output):
        for f in generated_output.rglob("*.py"):
            if f.stat().st_size > 0:
                content = f.read_text()
                ast.parse(content)

    def test_no_framework_imports(self, generated_output):
        _ALLOWED_IMPORTS = {
            "from customer_retention.transforms",
            "from customer_retention.generators.pipeline_generator.models",
            "from customer_retention.integrations.adapters.factory",
            "from customer_retention.core.compat",
            "from customer_retention.core.config",
            "from customer_retention.stages.modeling.data_splitter",
            "from customer_retention.stages.modeling.cross_validator",
            "from customer_retention.stages.modeling.feature_profile",
            "from customer_retention.analysis.auto_explorer.run_namespace",
            "from customer_retention.analysis.auto_explorer.layered_recommendations",
        }
        for f in generated_output.rglob("*.py"):
            if f.stat().st_size > 0:
                content = f.read_text()
                for line in content.splitlines():
                    stripped = line.strip()
                    if "from customer_retention" in stripped:
                        allowed = any(stripped.startswith(prefix) for prefix in _ALLOWED_IMPORTS)
                        assert allowed, (
                            f"Disallowed framework import in {f.name}: {stripped}"
                        )

    def test_generates_validation_directory(self, generated_output):
        assert (generated_output / "validation").exists()
        assert (generated_output / "validation" / "validate_pipeline.py").exists()
        assert (generated_output / "validation" / "run_validation.py").exists()
        assert (generated_output / "validation" / "__init__.py").exists()

    def test_generates_landing_directory_for_event_sources(self, generated_output):
        assert (generated_output / "landing").exists()
        landing_files = list((generated_output / "landing").glob("*.py"))
        assert len(landing_files) >= 1


class TestSourcePathPreservation:
    def test_source_configs_have_raw_paths(self, full_findings_setup):
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser
        tmp_path, findings_dir = full_findings_setup
        parser = FindingsParser(str(findings_dir))
        config = parser.parse()
        for source in config.sources:
            assert source.raw_source_path, f"Source {source.name} missing raw_source_path"

    def test_excluded_sources_marked(self, full_findings_setup):
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser
        _, findings_dir = full_findings_setup
        parser = FindingsParser(str(findings_dir))
        config = parser.parse()
        inactive = next((s for s in config.sources if s.name == "inactive"), None)
        assert inactive is not None
        assert inactive.excluded is True

    def test_config_contains_source_paths(self, generated_output):
        config_content = (generated_output / "config.py").read_text()
        assert "SOURCES" in config_content
        assert "customers" in config_content


class TestValidationScriptGeneration:
    def test_validation_scripts_are_valid_python(self, generated_output):
        for py_file in (generated_output / "validation").glob("*.py"):
            if py_file.stat().st_size > 0:
                ast.parse(py_file.read_text())

    def test_validation_has_stage_functions(self, generated_output):
        validate_content = (generated_output / "validation" / "validate_pipeline.py").read_text()
        assert "validate_bronze" in validate_content
        assert "validate_silver" in validate_content
        assert "validate_gold" in validate_content

    def test_run_validation_is_standalone(self, generated_output):
        run_content = (generated_output / "validation" / "run_validation.py").read_text()
        assert "from customer_retention" not in run_content
        assert "validate" in run_content.lower()


class TestLandingConfiguration:
    def test_landing_config_built_for_event_datasets(self, full_findings_setup):
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser
        _, findings_dir = full_findings_setup
        parser = FindingsParser(str(findings_dir))
        config = parser.parse()
        assert "orders" in config.landing
        landing = config.landing["orders"]
        assert landing.entity_column == "customer_id"
        assert landing.time_column == "order_date"

    def test_bronze_event_has_aggregation_config(self, full_findings_setup):
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser
        _, findings_dir = full_findings_setup
        parser = FindingsParser(str(findings_dir))
        config = parser.parse()
        assert "orders" in config.bronze_event
        bronze_event = config.bronze_event["orders"]
        assert bronze_event.aggregation is not None
        assert len(bronze_event.aggregation.windows) >= 3

    def test_landing_is_slim(self, full_findings_setup):
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser
        _, findings_dir = full_findings_setup
        parser = FindingsParser(str(findings_dir))
        config = parser.parse()
        landing = config.landing["orders"]
        assert not hasattr(landing, "lifecycle")
        assert not hasattr(landing, "aggregation")

    def test_bronze_event_source_has_lifecycle_config(self, full_findings_setup):
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser
        _, findings_dir = full_findings_setup
        parser = FindingsParser(str(findings_dir))
        config = parser.parse()
        bronze_event = config.bronze_event["orders"]
        assert bronze_event.lifecycle is not None
        assert bronze_event.lifecycle.include_lifecycle_quadrant is True
        assert bronze_event.lifecycle.include_recency_bucket is True

    def test_no_landing_for_entity_level_only(self, tmp_path):
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser
        findings_dir = tmp_path / "findings"
        findings_dir.mkdir()
        multi = {
            "datasets": {
                "test": {"name": "test", "findings_path": str(findings_dir / "test_findings.yaml"),
                        "source_path": "/test.csv", "granularity": "entity_level",
                        "row_count": 100, "column_count": 2, "excluded": False}
            },
            "relationships": [], "primary_entity_dataset": "test",
            "event_datasets": [], "excluded_datasets": []
        }
        (findings_dir / "multi_dataset_findings.yaml").write_text(yaml.dump(multi))
        (findings_dir / "test_findings.yaml").write_text(yaml.dump({
            "source_path": "/test.csv", "source_format": "csv",
            "row_count": 100, "column_count": 2,
            "columns": {
                "id": {"name": "id", "inferred_type": "identifier", "confidence": 0.95,
                      "evidence": [], "quality_score": 100, "cleaning_needed": False, "cleaning_recommendations": []},
                "target": {"name": "target", "inferred_type": "binary", "confidence": 0.9,
                          "evidence": [], "quality_score": 100, "cleaning_needed": False, "cleaning_recommendations": []}
            },
            "target_column": "target", "identifier_columns": ["id"]
        }))
        parser = FindingsParser(str(findings_dir))
        config = parser.parse()
        assert len(config.landing) == 0


class TestWorkflowDAG:
    def test_workflow_has_landing_tasks(self, generated_output):
        workflow = json.loads((generated_output / "workflow.json").read_text())
        task_keys = [t["task_key"] for t in workflow["tasks"]]
        has_landing = any("landing" in k for k in task_keys)
        assert has_landing or len([s for s in workflow["tasks"] if "bronze" in s["task_key"]]) > 0
