from pathlib import Path

import pytest
import yaml

from customer_retention.analysis.auto_explorer.run_namespace import RunNamespace
from customer_retention.generators.pipeline_generator.llm_docs_generator import LLMDocsGenerator


def _make_findings(source_path="/data/src.csv", target="churn", columns=None, **extra):
    cols = columns or {
        "customer_id": {"name": "customer_id", "inferred_type": "identifier", "confidence": 0.95,
                        "evidence": [], "quality_score": 100, "cleaning_needed": False,
                        "cleaning_recommendations": [],
                        "universal_metrics": {"null_percentage": 0, "distinct_count": 500}},
        "age": {"name": "age", "inferred_type": "numeric_continuous", "confidence": 0.9,
                "evidence": [], "quality_score": 80, "cleaning_needed": True,
                "cleaning_recommendations": ["impute_null:median"],
                "transformation_recommendations": ["bin into age groups"],
                "universal_metrics": {"null_percentage": 5.2, "distinct_count": 60}},
        "churn": {"name": "churn", "inferred_type": "binary", "confidence": 0.99,
                  "evidence": [], "quality_score": 100, "cleaning_needed": False,
                  "cleaning_recommendations": [],
                  "universal_metrics": {"null_percentage": 0, "distinct_count": 2}},
    }
    base = {
        "source_path": source_path, "source_format": "csv",
        "row_count": 1000, "column_count": len(cols),
        "columns": cols, "target_column": target,
        "identifier_columns": ["customer_id"],
        "overall_quality_score": 85.0,
    }
    base.update(extra)
    return base


def _make_event_findings(source_path="/data/orders.parquet"):
    return _make_findings(
        source_path=source_path, target=None,
        columns={
            "order_id": {"name": "order_id", "inferred_type": "identifier", "confidence": 0.95,
                         "evidence": [], "quality_score": 100, "cleaning_needed": False,
                         "cleaning_recommendations": [],
                         "universal_metrics": {"null_percentage": 0, "distinct_count": 5000}},
            "customer_id": {"name": "customer_id", "inferred_type": "identifier", "confidence": 0.95,
                            "evidence": [], "quality_score": 100, "cleaning_needed": False,
                            "cleaning_recommendations": [],
                            "universal_metrics": {"null_percentage": 0, "distinct_count": 500}},
            "amount": {"name": "amount", "inferred_type": "numeric_continuous", "confidence": 0.9,
                       "evidence": [], "quality_score": 90, "cleaning_needed": True,
                       "cleaning_recommendations": ["cap_outlier:iqr"],
                       "universal_metrics": {"null_percentage": 1.5, "distinct_count": 3200}},
            "order_date": {"name": "order_date", "inferred_type": "datetime", "confidence": 0.95,
                           "evidence": [], "quality_score": 100, "cleaning_needed": False,
                           "cleaning_recommendations": [],
                           "universal_metrics": {"null_percentage": 0, "distinct_count": 365}},
        },
        time_series_metadata={"granularity": "event_level", "entity_column": "customer_id",
                              "time_column": "order_date", "unique_entities": 500,
                              "avg_events_per_entity": 10.0, "time_span_days": 365,
                              "aggregation_windows_used": ["7d", "30d", "90d"]},
        source_format="parquet", row_count=5000, column_count=4,
    )


def _make_multi_dataset(findings_dir, *, with_events=True):
    ds = {
        "customers": {
            "name": "customers",
            "findings_path": str(findings_dir / "customers_findings.yaml"),
            "source_path": "/data/customers.csv", "granularity": "entity_level",
            "row_count": 1000, "column_count": 3, "excluded": False,
            "entity_column": "customer_id",
        },
    }
    events, rels = [], []
    if with_events:
        ds["orders"] = {
            "name": "orders",
            "findings_path": str(findings_dir / "orders_findings.yaml"),
            "source_path": "/data/orders.parquet", "granularity": "event_level",
            "row_count": 5000, "column_count": 4, "excluded": False,
            "entity_column": "customer_id", "time_column": "order_date",
        }
        events = ["orders"]
        rels = [{"left_dataset": "customers", "right_dataset": "orders",
                 "left_columns": ["customer_id"], "right_columns": ["customer_id"],
                 "relationship_type": "one_to_many", "confidence": 1.0}]
    return {
        "datasets": ds, "relationships": rels,
        "primary_entity_dataset": "customers",
        "event_datasets": events, "excluded_datasets": [],
        "aggregation_windows": ["7d", "30d", "90d"],
    }


def _make_recommendations(source_name="customers"):
    return {
        "sources": {
            source_name: {
                "source_file": "/data/customers.csv",
                "null_handling": [
                    {"id": "bronze_null_age", "layer": "bronze", "category": "null",
                     "action": "impute", "target_column": "age",
                     "parameters": {"strategy": "median"}, "rationale": "Fill age nulls",
                     "source_notebook": "02_source_integrity"},
                ],
                "outlier_handling": [], "type_conversions": [],
                "deduplication": [], "filtering": [],
                "text_processing": [], "modeling_strategy": [],
            },
        },
        "silver": {
            "entity_column": "customer_id", "time_column": None,
            "joins": [], "aggregations": [], "derived_columns": [],
        },
        "gold": {
            "target_column": "churn",
            "encoding": [
                {"id": "gold_enc_status", "layer": "gold", "category": "encoding",
                 "action": "one_hot", "target_column": "status",
                 "parameters": {"method": "one_hot"}, "rationale": "Encode status",
                 "source_notebook": "07_modeling"},
            ],
            "scaling": [], "feature_selection": [], "transformations": [],
        },
    }


def _make_project_context(*, with_intent=True):
    pc = {
        "project_name": "test_project", "run_id": "abc123",
        "timezone": "UTC", "storage_backend": "local",
        "datasets": {},
        "objectives": [
            {"objective": "immediate_risk", "priority": "primary",
             "anchor": "now", "parameters": {"prediction_horizon": 30}},
        ],
        "temporal_posture": "stable",
    }
    if with_intent:
        pc["intent"] = {
            "prediction_horizons": [30, 60, 90],
            "observation_window_days": 270, "label_window_days": 90,
            "purge_gap_days": 104, "recent_window_days": 270,
            "cadence_interval": "monthly", "split_strategy": "temporal",
            "temporal_split": True,
        }
    return pc


@pytest.fixture
def namespace_dir(tmp_path):
    """Build a RunNamespace on disk with findings, recs, multi_dataset, project_context."""
    root = tmp_path / "experiments"
    ns = RunNamespace(root=root, run_id="run-001")

    # per-dataset findings
    cust_findings_dir = ns.datasets_dir / "customers" / "findings"
    cust_findings_dir.mkdir(parents=True)
    (cust_findings_dir / "customers_findings.yaml").write_text(yaml.dump(_make_findings()))

    orders_findings_dir = ns.datasets_dir / "orders" / "findings"
    orders_findings_dir.mkdir(parents=True)
    (orders_findings_dir / "orders_findings.yaml").write_text(yaml.dump(_make_event_findings()))

    # multi_dataset + recommendations in merged dir
    findings_dir = ns.merged_dir
    findings_dir.mkdir(parents=True)
    (ns.multi_dataset_findings_path).write_text(yaml.dump(_make_multi_dataset(cust_findings_dir.parent)))
    (ns.merged_recommendations_path).write_text(yaml.dump(_make_recommendations()))

    # project context
    (ns.project_context_path).write_text(yaml.dump(_make_project_context()))

    # FindingsParser also needs findings files in its findings_dir
    parser_dir = tmp_path / "findings_for_parser"
    parser_dir.mkdir()
    (parser_dir / "customers_findings.yaml").write_text(yaml.dump(_make_findings()))
    (parser_dir / "orders_findings.yaml").write_text(yaml.dump(_make_event_findings()))
    (parser_dir / "multi_dataset_findings.yaml").write_text(
        yaml.dump(_make_multi_dataset(parser_dir)))

    return ns, parser_dir


@pytest.fixture
def minimal_namespace(tmp_path):
    """Namespace with single dataset, no recommendations, no project context."""
    root = tmp_path / "experiments"
    ns = RunNamespace(root=root, run_id="run-min")

    cust_dir = ns.datasets_dir / "customers" / "findings"
    cust_dir.mkdir(parents=True)
    (cust_dir / "customers_findings.yaml").write_text(yaml.dump(_make_findings()))

    parser_dir = tmp_path / "findings_for_parser"
    parser_dir.mkdir()
    (parser_dir / "customers_findings.yaml").write_text(yaml.dump(_make_findings()))

    return ns, parser_dir


# =========================================================================
# Directory structure
# =========================================================================

class TestDirectoryStructure:
    def test_creates_medallion_subdirectories(self, namespace_dir, tmp_path):
        ns, parser_dir = namespace_dir
        gen = LLMDocsGenerator(str(parser_dir), str(tmp_path / "out"), "test", namespace=ns)
        gen.generate()
        docs = tmp_path / "out" / "docs"
        for subdir in ("landing", "bronze", "silver", "gold", "training"):
            assert (docs / subdir).is_dir()

    def test_returns_list_of_existing_paths(self, namespace_dir, tmp_path):
        ns, parser_dir = namespace_dir
        gen = LLMDocsGenerator(str(parser_dir), str(tmp_path / "out"), "test", namespace=ns)
        files = gen.generate()
        assert isinstance(files, list)
        for f in files:
            assert isinstance(f, Path)
            assert f.exists()

    def test_all_files_are_markdown(self, namespace_dir, tmp_path):
        ns, parser_dir = namespace_dir
        gen = LLMDocsGenerator(str(parser_dir), str(tmp_path / "out"), "test", namespace=ns)
        for f in gen.generate():
            assert f.suffix == ".md"


# =========================================================================
# Overview
# =========================================================================

class TestOverview:
    def test_contains_pipeline_name(self, namespace_dir, tmp_path):
        ns, parser_dir = namespace_dir
        gen = LLMDocsGenerator(str(parser_dir), str(tmp_path / "out"), "my_pipeline", namespace=ns)
        gen.generate()
        text = (tmp_path / "out" / "docs" / "overview.md").read_text()
        assert "my_pipeline" in text

    def test_contains_composite_name(self, namespace_dir, tmp_path):
        ns, parser_dir = namespace_dir
        gen = LLMDocsGenerator(str(parser_dir), str(tmp_path / "out"), "p", namespace=ns)
        gen.generate()
        text = (tmp_path / "out" / "docs" / "overview.md").read_text()
        assert "Composite name:" in text

    def test_contains_target_column(self, namespace_dir, tmp_path):
        ns, parser_dir = namespace_dir
        gen = LLMDocsGenerator(str(parser_dir), str(tmp_path / "out"), "p", namespace=ns)
        gen.generate()
        text = (tmp_path / "out" / "docs" / "overview.md").read_text()
        assert "churn" in text

    def test_contains_datasets_table(self, namespace_dir, tmp_path):
        ns, parser_dir = namespace_dir
        gen = LLMDocsGenerator(str(parser_dir), str(tmp_path / "out"), "p", namespace=ns)
        gen.generate()
        text = (tmp_path / "out" / "docs" / "overview.md").read_text()
        assert "| Dataset |" in text
        assert "customers" in text

    def test_contains_relationships(self, namespace_dir, tmp_path):
        ns, parser_dir = namespace_dir
        gen = LLMDocsGenerator(str(parser_dir), str(tmp_path / "out"), "p", namespace=ns)
        gen.generate()
        text = (tmp_path / "out" / "docs" / "overview.md").read_text()
        assert "Relationships" in text
        assert "one_to_many" in text

    def test_contains_objectives_from_project_context(self, namespace_dir, tmp_path):
        ns, parser_dir = namespace_dir
        gen = LLMDocsGenerator(str(parser_dir), str(tmp_path / "out"), "p", namespace=ns)
        gen.generate()
        text = (tmp_path / "out" / "docs" / "overview.md").read_text()
        assert "immediate_risk" in text

    def test_no_objectives_without_project_context(self, minimal_namespace, tmp_path):
        ns, parser_dir = minimal_namespace
        gen = LLMDocsGenerator(str(parser_dir), str(tmp_path / "out"), "p", namespace=ns)
        gen.generate()
        text = (tmp_path / "out" / "docs" / "overview.md").read_text()
        assert "Prediction Objective" not in text


# =========================================================================
# Config
# =========================================================================

class TestConfig:
    def test_contains_naming_section(self, namespace_dir, tmp_path):
        ns, parser_dir = namespace_dir
        gen = LLMDocsGenerator(str(parser_dir), str(tmp_path / "out"), "p", namespace=ns)
        gen.generate()
        text = (tmp_path / "out" / "docs" / "config.md").read_text()
        assert "silver_featureset_" in text
        assert "gold_features_" in text

    def test_contains_temporal_config_from_intent(self, namespace_dir, tmp_path):
        ns, parser_dir = namespace_dir
        gen = LLMDocsGenerator(str(parser_dir), str(tmp_path / "out"), "p", namespace=ns)
        gen.generate()
        text = (tmp_path / "out" / "docs" / "config.md").read_text()
        assert "270 days" in text
        assert "temporal" in text.lower()

    def test_contains_aggregation_windows(self, namespace_dir, tmp_path):
        ns, parser_dir = namespace_dir
        gen = LLMDocsGenerator(str(parser_dir), str(tmp_path / "out"), "p", namespace=ns)
        gen.generate()
        text = (tmp_path / "out" / "docs" / "config.md").read_text()
        assert "`7d`" in text
        assert "`30d`" in text

    def test_contains_source_classification(self, namespace_dir, tmp_path):
        ns, parser_dir = namespace_dir
        gen = LLMDocsGenerator(str(parser_dir), str(tmp_path / "out"), "p", namespace=ns)
        gen.generate()
        text = (tmp_path / "out" / "docs" / "config.md").read_text()
        assert "Entity sources" in text
        assert "Event sources" in text

    def test_no_temporal_config_without_intent(self, minimal_namespace, tmp_path):
        ns, parser_dir = minimal_namespace
        gen = LLMDocsGenerator(str(parser_dir), str(tmp_path / "out"), "p", namespace=ns)
        gen.generate()
        text = (tmp_path / "out" / "docs" / "config.md").read_text()
        assert "Temporal Configuration" not in text


# =========================================================================
# Landing
# =========================================================================

class TestLanding:
    def test_creates_per_dataset_file(self, namespace_dir, tmp_path):
        ns, parser_dir = namespace_dir
        gen = LLMDocsGenerator(str(parser_dir), str(tmp_path / "out"), "p", namespace=ns)
        gen.generate()
        landing = tmp_path / "out" / "docs" / "landing"
        files = sorted(f.name for f in landing.glob("*.md"))
        assert "landing_customers.md" in files
        assert "landing_orders.md" in files

    def test_contains_column_schema_table(self, namespace_dir, tmp_path):
        ns, parser_dir = namespace_dir
        gen = LLMDocsGenerator(str(parser_dir), str(tmp_path / "out"), "p", namespace=ns)
        gen.generate()
        text = (tmp_path / "out" / "docs" / "landing" / "landing_customers.md").read_text()
        assert "| Column |" in text
        assert "`customer_id`" in text

    def test_contains_quality_score(self, namespace_dir, tmp_path):
        ns, parser_dir = namespace_dir
        gen = LLMDocsGenerator(str(parser_dir), str(tmp_path / "out"), "p", namespace=ns)
        gen.generate()
        text = (tmp_path / "out" / "docs" / "landing" / "landing_customers.md").read_text()
        assert "Quality score:" in text

    def test_event_source_has_temporal_profile(self, namespace_dir, tmp_path):
        ns, parser_dir = namespace_dir
        gen = LLMDocsGenerator(str(parser_dir), str(tmp_path / "out"), "p", namespace=ns)
        gen.generate()
        text = (tmp_path / "out" / "docs" / "landing" / "landing_orders.md").read_text()
        assert "Temporal Profile" in text
        assert "event_level" in text

    def test_null_percentage_in_schema(self, namespace_dir, tmp_path):
        ns, parser_dir = namespace_dir
        gen = LLMDocsGenerator(str(parser_dir), str(tmp_path / "out"), "p", namespace=ns)
        gen.generate()
        text = (tmp_path / "out" / "docs" / "landing" / "landing_customers.md").read_text()
        assert "5.2%" in text


# =========================================================================
# Bronze
# =========================================================================

class TestBronze:
    def test_entity_source_labelled_correctly(self, namespace_dir, tmp_path):
        ns, parser_dir = namespace_dir
        gen = LLMDocsGenerator(str(parser_dir), str(tmp_path / "out"), "p", namespace=ns)
        gen.generate()
        bronze = tmp_path / "out" / "docs" / "bronze"
        assert (bronze / "bronze_entity_customers.md").exists()

    def test_event_source_labelled_correctly(self, namespace_dir, tmp_path):
        ns, parser_dir = namespace_dir
        gen = LLMDocsGenerator(str(parser_dir), str(tmp_path / "out"), "p", namespace=ns)
        gen.generate()
        bronze = tmp_path / "out" / "docs" / "bronze"
        assert (bronze / "bronze_event_orders.md").exists()

    def test_event_source_mentions_two_scripts(self, namespace_dir, tmp_path):
        ns, parser_dir = namespace_dir
        gen = LLMDocsGenerator(str(parser_dir), str(tmp_path / "out"), "p", namespace=ns)
        gen.generate()
        text = (tmp_path / "out" / "docs" / "bronze" / "bronze_event_orders.md").read_text()
        assert "two scripts" in text.lower()

    def test_includes_registry_recommendations(self, namespace_dir, tmp_path):
        ns, parser_dir = namespace_dir
        gen = LLMDocsGenerator(str(parser_dir), str(tmp_path / "out"), "p", namespace=ns)
        gen.generate()
        text = (tmp_path / "out" / "docs" / "bronze" / "bronze_entity_customers.md").read_text()
        assert "Null Handling" in text
        assert "median" in text

    def test_includes_column_level_cleaning(self, namespace_dir, tmp_path):
        ns, parser_dir = namespace_dir
        gen = LLMDocsGenerator(str(parser_dir), str(tmp_path / "out"), "p", namespace=ns)
        gen.generate()
        text = (tmp_path / "out" / "docs" / "bronze" / "bronze_entity_customers.md").read_text()
        assert "Column-Level Cleaning" in text
        assert "`age`" in text


# =========================================================================
# Silver
# =========================================================================

class TestSilver:
    def test_file_named_with_composite_name(self, namespace_dir, tmp_path):
        ns, parser_dir = namespace_dir
        gen = LLMDocsGenerator(str(parser_dir), str(tmp_path / "out"), "p", namespace=ns)
        gen.generate()
        silver = tmp_path / "out" / "docs" / "silver"
        files = list(silver.glob("silver_featureset_*.md"))
        assert len(files) == 1

    def test_mentions_entity_id_key(self, namespace_dir, tmp_path):
        ns, parser_dir = namespace_dir
        gen = LLMDocsGenerator(str(parser_dir), str(tmp_path / "out"), "p", namespace=ns)
        gen.generate()
        silver = tmp_path / "out" / "docs" / "silver"
        text = list(silver.glob("*.md"))[0].read_text()
        assert "entity_id" in text

    def test_contains_relationships_from_multi_dataset(self, namespace_dir, tmp_path):
        ns, parser_dir = namespace_dir
        gen = LLMDocsGenerator(str(parser_dir), str(tmp_path / "out"), "p", namespace=ns)
        gen.generate()
        silver = tmp_path / "out" / "docs" / "silver"
        text = list(silver.glob("*.md"))[0].read_text()
        assert "Dataset Relationships" in text


# =========================================================================
# Gold
# =========================================================================

class TestGold:
    def test_file_named_with_composite_name(self, namespace_dir, tmp_path):
        ns, parser_dir = namespace_dir
        gen = LLMDocsGenerator(str(parser_dir), str(tmp_path / "out"), "p", namespace=ns)
        gen.generate()
        gold = tmp_path / "out" / "docs" / "gold"
        files = list(gold.glob("gold_features_*.md"))
        assert len(files) == 1

    def test_contains_target_column(self, namespace_dir, tmp_path):
        ns, parser_dir = namespace_dir
        gen = LLMDocsGenerator(str(parser_dir), str(tmp_path / "out"), "p", namespace=ns)
        gen.generate()
        gold = tmp_path / "out" / "docs" / "gold"
        text = list(gold.glob("*.md"))[0].read_text()
        assert "churn" in text

    def test_contains_encoding_recommendations(self, namespace_dir, tmp_path):
        ns, parser_dir = namespace_dir
        gen = LLMDocsGenerator(str(parser_dir), str(tmp_path / "out"), "p", namespace=ns)
        gen.generate()
        gold = tmp_path / "out" / "docs" / "gold"
        text = list(gold.glob("*.md"))[0].read_text()
        assert "Encoding" in text
        assert "one_hot" in text

    def test_contains_feature_opportunities(self, namespace_dir, tmp_path):
        ns, parser_dir = namespace_dir
        gen = LLMDocsGenerator(str(parser_dir), str(tmp_path / "out"), "p", namespace=ns)
        gen.generate()
        gold = tmp_path / "out" / "docs" / "gold"
        text = list(gold.glob("*.md"))[0].read_text()
        assert "Feature Engineering Opportunities" in text
        assert "age groups" in text


# =========================================================================
# Training
# =========================================================================

class TestTraining:
    def test_file_exists(self, namespace_dir, tmp_path):
        ns, parser_dir = namespace_dir
        gen = LLMDocsGenerator(str(parser_dir), str(tmp_path / "out"), "p", namespace=ns)
        gen.generate()
        assert (tmp_path / "out" / "docs" / "training" / "ml_experiment.md").exists()

    def test_contains_temporal_split_config(self, namespace_dir, tmp_path):
        ns, parser_dir = namespace_dir
        gen = LLMDocsGenerator(str(parser_dir), str(tmp_path / "out"), "p", namespace=ns)
        gen.generate()
        text = (tmp_path / "out" / "docs" / "training" / "ml_experiment.md").read_text()
        assert "Temporal Split Configuration" in text
        assert "entity-aware" in text

    def test_contains_evaluation_metrics(self, namespace_dir, tmp_path):
        ns, parser_dir = namespace_dir
        gen = LLMDocsGenerator(str(parser_dir), str(tmp_path / "out"), "p", namespace=ns)
        gen.generate()
        text = (tmp_path / "out" / "docs" / "training" / "ml_experiment.md").read_text()
        assert "ROC-AUC" in text
        assert "PR-AUC" in text

    def test_no_temporal_section_without_intent(self, minimal_namespace, tmp_path):
        ns, parser_dir = minimal_namespace
        gen = LLMDocsGenerator(str(parser_dir), str(tmp_path / "out"), "p", namespace=ns)
        gen.generate()
        text = (tmp_path / "out" / "docs" / "training" / "ml_experiment.md").read_text()
        assert "Temporal Split Configuration" not in text


# =========================================================================
# Minimal / edge cases
# =========================================================================

class TestMinimalInput:
    def test_works_without_namespace(self, tmp_path):
        findings_dir = tmp_path / "findings"
        findings_dir.mkdir()
        (findings_dir / "test_findings.yaml").write_text(yaml.dump(_make_findings()))
        gen = LLMDocsGenerator(str(findings_dir), str(tmp_path / "out"), "bare")
        files = gen.generate()
        assert len(files) > 0
        for f in files:
            assert f.exists()

    def test_single_dataset_no_recs(self, minimal_namespace, tmp_path):
        ns, parser_dir = minimal_namespace
        gen = LLMDocsGenerator(str(parser_dir), str(tmp_path / "out"), "p", namespace=ns)
        files = gen.generate()
        assert any("overview" in f.name for f in files)
        assert any("config" in f.name for f in files)
        assert any("landing_customers" in f.name for f in files)
        assert any("ml_experiment" in f.name for f in files)

    def test_bronze_without_registry_still_shows_cleaning(self, minimal_namespace, tmp_path):
        ns, parser_dir = minimal_namespace
        gen = LLMDocsGenerator(str(parser_dir), str(tmp_path / "out"), "p", namespace=ns)
        gen.generate()
        bronze = tmp_path / "out" / "docs" / "bronze"
        text = (bronze / "bronze_entity_customers.md").read_text()
        assert "Column-Level Cleaning" in text


class TestFileCount:
    def test_multi_source_file_count(self, namespace_dir, tmp_path):
        ns, parser_dir = namespace_dir
        gen = LLMDocsGenerator(str(parser_dir), str(tmp_path / "out"), "p", namespace=ns)
        files = gen.generate()
        # overview + config + 2 landing + 2 bronze + 1 silver + 1 gold + 1 training = 9
        assert len(files) == 9

    def test_single_source_file_count(self, minimal_namespace, tmp_path):
        ns, parser_dir = minimal_namespace
        gen = LLMDocsGenerator(str(parser_dir), str(tmp_path / "out"), "p", namespace=ns)
        files = gen.generate()
        # overview + config + 1 landing + 1 bronze + 1 silver + 1 gold + 1 training = 7
        assert len(files) == 7
