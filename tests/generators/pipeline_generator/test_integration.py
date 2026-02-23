import json

import pandas as pd
import pytest
import yaml

from customer_retention.core.naming import composite_name
from customer_retention.generators.pipeline_generator import PipelineGenerator


@pytest.fixture
def full_findings_setup(tmp_path):
    findings_dir = tmp_path / "findings"
    findings_dir.mkdir()
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    customers_df = pd.DataFrame({
        "customer_id": [1, 2, 3, 4, 5],
        "age": [25, None, 35, 40, 30],
        "income": [50000, 60000, 70000, 80000, 55000],
        "churn": [0, 1, 0, 1, 0]
    })
    customers_df.to_csv(data_dir / "customers.csv", index=False)

    orders_df = pd.DataFrame({
        "order_id": [101, 102, 103, 104, 105, 106],
        "customer_id": [1, 1, 2, 3, 4, 5],
        "amount": [100, 200, 150, 300, 50, 10000],
        "order_date": pd.date_range("2024-01-01", periods=6)
    })
    orders_df.to_parquet(data_dir / "orders.parquet", index=False)

    multi_dataset = {
        "datasets": {
            "customers": {"name": "customers", "findings_path": str(findings_dir / "customers_findings.yaml"),
                         "source_path": str(data_dir / "customers.csv"), "granularity": "entity_level",
                         "row_count": 5, "column_count": 4, "excluded": False},
            "orders": {"name": "orders", "findings_path": str(findings_dir / "orders_findings.yaml"),
                      "source_path": str(data_dir / "orders.parquet"), "granularity": "event_level",
                      "row_count": 6, "column_count": 4, "excluded": False,
                      "entity_column": "customer_id", "time_column": "order_date"}
        },
        "relationships": [
            {"left_dataset": "customers", "right_dataset": "orders", "left_column": "customer_id",
             "right_column": "customer_id", "relationship_type": "one_to_many", "confidence": 1.0}
        ],
        "primary_entity_dataset": "customers",
        "event_datasets": ["orders"],
        "excluded_datasets": []
    }
    (findings_dir / "multi_dataset_findings.yaml").write_text(yaml.dump(multi_dataset))

    customers_findings = {
        "source_path": str(data_dir / "customers.csv"),
        "source_format": "csv",
        "row_count": 5,
        "column_count": 4,
        "columns": {
            "customer_id": {"name": "customer_id", "inferred_type": "identifier", "confidence": 0.95,
                          "evidence": [], "quality_score": 100, "cleaning_needed": False, "cleaning_recommendations": []},
            "age": {"name": "age", "inferred_type": "numeric_continuous", "confidence": 0.9, "evidence": [],
                   "quality_score": 80, "cleaning_needed": True,
                   "cleaning_recommendations": ["impute_null:0"], "type_metrics": {}},
            "income": {"name": "income", "inferred_type": "numeric_continuous", "confidence": 0.9, "evidence": [],
                      "quality_score": 100, "cleaning_needed": False, "cleaning_recommendations": []},
            "churn": {"name": "churn", "inferred_type": "binary", "confidence": 0.99, "evidence": [],
                     "quality_score": 100, "cleaning_needed": False, "cleaning_recommendations": []}
        },
        "target_column": "churn",
        "identifier_columns": ["customer_id"]
    }
    (findings_dir / "customers_findings.yaml").write_text(yaml.dump(customers_findings))

    orders_findings = {
        "source_path": str(data_dir / "orders.parquet"),
        "source_format": "parquet",
        "row_count": 6,
        "column_count": 4,
        "columns": {
            "order_id": {"name": "order_id", "inferred_type": "identifier", "confidence": 0.95, "evidence": [],
                        "quality_score": 100, "cleaning_needed": False, "cleaning_recommendations": []},
            "customer_id": {"name": "customer_id", "inferred_type": "identifier", "confidence": 0.95, "evidence": [],
                          "quality_score": 100, "cleaning_needed": False, "cleaning_recommendations": []},
            "amount": {"name": "amount", "inferred_type": "numeric_continuous", "confidence": 0.9, "evidence": [],
                      "quality_score": 90, "cleaning_needed": True,
                      "cleaning_recommendations": ["cap_outlier:iqr"], "type_metrics": {}},
            "order_date": {"name": "order_date", "inferred_type": "datetime", "confidence": 0.95, "evidence": [],
                          "quality_score": 100, "cleaning_needed": False, "cleaning_recommendations": []}
        },
        "identifier_columns": ["order_id"],
        "datetime_columns": ["order_date"],
        "time_series_metadata": {"granularity": "event_level", "entity_column": "customer_id", "time_column": "order_date"}
    }
    (findings_dir / "orders_findings.yaml").write_text(yaml.dump(orders_findings))

    return {"findings_dir": findings_dir, "data_dir": data_dir, "tmp_path": tmp_path}


@pytest.fixture
def generated_output(full_findings_setup):
    findings_dir = full_findings_setup["findings_dir"]
    output_dir = full_findings_setup["tmp_path"] / "output"
    generator = PipelineGenerator(str(findings_dir), str(output_dir), "test_pipeline")
    generator.generate()
    return output_dir


class TestEndToEndPipelineGeneration:
    def test_generate_complete_pipeline(self, full_findings_setup):
        findings_dir = full_findings_setup["findings_dir"]
        output_dir = full_findings_setup["tmp_path"] / "output"
        generator = PipelineGenerator(str(findings_dir), str(output_dir), "churn_pipeline")
        files = generator.generate()
        assert len(files) >= 7
        assert output_dir.exists()
        assert (output_dir / "config.py").exists()
        assert (output_dir / "bronze").exists()
        assert (output_dir / "silver").exists()
        assert (output_dir / "gold").exists()
        assert (output_dir / "training").exists()
        assert (output_dir / "pipeline_runner.py").exists()
        assert (output_dir / "workflow.json").exists()

    def test_generated_config_has_correct_sources(self, full_findings_setup):
        findings_dir = full_findings_setup["findings_dir"]
        output_dir = full_findings_setup["tmp_path"] / "output"
        data_dir = full_findings_setup["data_dir"]
        generator = PipelineGenerator(str(findings_dir), str(output_dir), "test_config")
        generator.generate()
        config_content = (output_dir / "config.py").read_text()
        assert "customers" in config_content
        assert "orders" in config_content
        assert str(data_dir) in config_content

    def test_workflow_has_correct_task_dependencies(self, full_findings_setup):
        findings_dir = full_findings_setup["findings_dir"]
        output_dir = full_findings_setup["tmp_path"] / "output"
        generator = PipelineGenerator(str(findings_dir), str(output_dir), "workflow_test")
        generator.generate()
        workflow_content = (output_dir / "workflow.json").read_text()
        workflow = json.loads(workflow_content)
        task_keys = [t["task_key"] for t in workflow["tasks"]]
        cn = composite_name(["customers", "orders_aggregated"])
        assert "bronze_entity_customers" in task_keys
        assert "bronze_event_orders" in task_keys
        assert "bronze_entity_orders_aggregated" in task_keys
        assert f"silver_featureset_{cn}" in task_keys
        assert f"gold_features_{cn}" in task_keys
        assert "ml_experiment" in task_keys
        silver_task = next(t for t in workflow["tasks"] if t["task_key"] == f"silver_featureset_{cn}")
        depends_on_keys = [d["task_key"] for d in silver_task["depends_on"]]
        assert "bronze_entity_customers" in depends_on_keys
        assert "bronze_entity_orders_aggregated" in depends_on_keys


class TestBronzeLayerGeneration:
    def test_bronze_entity_applies_impute_null(self, generated_output):
        bronze_content = (generated_output / "bronze" / "bronze_entity_customers.py").read_text()
        assert "apply_impute_null" in bronze_content

    def test_bronze_event_has_pre_shaping_cap_outlier(self, generated_output):
        bronze_content = (generated_output / "bronze" / "bronze_event_orders.py").read_text()
        assert "apply_pre_shaping" in bronze_content
        assert "apply_cap_outlier" in bronze_content
        assert "apply_event_aggregation" in bronze_content

    def test_landing_is_slim(self, generated_output):
        landing_content = (generated_output / "landing" / "landing_orders.py").read_text()
        assert "load_raw_data" in landing_content
        assert "apply_pre_agg_transformations" not in landing_content
        assert "aggregate_time_windows" not in landing_content


class TestSilverLayerGeneration:
    def test_silver_includes_merge_logic(self, generated_output):
        cn = composite_name(["customers", "orders_aggregated"])
        silver_content = (generated_output / "silver" / f"silver_featureset_{cn}.py").read_text()
        assert "merge" in silver_content
        assert "customer_id" in silver_content


class TestGoldLayerGeneration:
    def test_gold_includes_feature_engineering(self, generated_output):
        cn = composite_name(["customers", "orders_aggregated"])
        gold_content = (generated_output / "gold" / f"gold_features_{cn}.py").read_text()
        assert "apply_encodings" in gold_content or "encoding" in gold_content.lower()
        assert "apply_scaling" in gold_content or "scal" in gold_content.lower()


class TestTrainingGeneration:
    def test_training_includes_model_and_metrics(self, generated_output):
        training_content = (generated_output / "training" / "ml_experiment.py").read_text()
        assert "RandomForestClassifier" in training_content
        assert "roc_auc_score" in training_content
        assert "DataSplitter" in training_content
        assert "SplitStrategy.TEMPORAL" in training_content
