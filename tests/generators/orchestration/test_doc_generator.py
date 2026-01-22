"""Tests for PipelineDocGenerator that produces pipeline_spec.md for LLM context."""
import pytest

from customer_retention.analysis.auto_explorer.findings import ColumnFinding, ExplorationFindings
from customer_retention.analysis.auto_explorer.layered_recommendations import (
    LayeredRecommendation,
    RecommendationRegistry,
)
from customer_retention.core.config.column_config import ColumnType
from customer_retention.generators.orchestration.doc_generator import PipelineDocGenerator


@pytest.fixture
def sample_findings():
    columns = {
        "customer_id": ColumnFinding(
            name="customer_id", inferred_type=ColumnType.IDENTIFIER,
            confidence=0.95, evidence=["Unique"],
            universal_metrics={"null_count": 0, "distinct_count": 1000}
        ),
        "age": ColumnFinding(
            name="age", inferred_type=ColumnType.NUMERIC_CONTINUOUS,
            confidence=0.9, evidence=["Numeric"],
            universal_metrics={"null_count": 50, "null_percentage": 5.0},
            type_metrics={"mean": 35.5, "std": 12.3}
        ),
        "revenue": ColumnFinding(
            name="revenue", inferred_type=ColumnType.NUMERIC_CONTINUOUS,
            confidence=0.9, evidence=["Numeric"],
            universal_metrics={"null_count": 0},
            type_metrics={"skewness": 2.5}
        ),
    }
    return ExplorationFindings(
        source_path="customers.csv", source_format="csv",
        row_count=1000, column_count=3, columns=columns,
        target_column="churned", identifier_columns=["customer_id"]
    )


@pytest.fixture
def full_registry():
    registry = RecommendationRegistry()
    registry.init_bronze("customers.csv")
    registry.init_silver("customer_id", "event_date")
    registry.init_gold("churned")

    registry.bronze.null_handling.append(LayeredRecommendation(
        id="bronze_null_age", layer="bronze", category="null", action="impute",
        target_column="age", parameters={"strategy": "median"},
        rationale="5% missing values", source_notebook="03_quality"
    ))
    registry.bronze.outlier_handling.append(LayeredRecommendation(
        id="bronze_outlier_revenue", layer="bronze", category="outlier", action="cap",
        target_column="revenue", parameters={"method": "iqr", "factor": 1.5},
        rationale="12% outliers detected", source_notebook="03_quality"
    ))
    registry.silver.aggregations.append(LayeredRecommendation(
        id="silver_agg_revenue", layer="silver", category="aggregation", action="sum",
        target_column="revenue", parameters={"aggregation": "sum", "windows": ["7d", "30d"]},
        rationale="Revenue trends over time", source_notebook="04TS"
    ))
    registry.gold.encoding.append(LayeredRecommendation(
        id="gold_encode_contract", layer="gold", category="encoding", action="one_hot",
        target_column="contract_type", parameters={"method": "one_hot"},
        rationale="Low cardinality (3 values)", source_notebook="06_modeling"
    ))
    registry.gold.scaling.append(LayeredRecommendation(
        id="gold_scale_revenue", layer="gold", category="scaling", action="standard",
        target_column="revenue", parameters={"method": "standard"},
        rationale="Normalize for model", source_notebook="06_modeling"
    ))
    return registry


class TestDocGeneratorInit:
    def test_creates_with_registry_and_findings(self, full_registry, sample_findings):
        gen = PipelineDocGenerator(full_registry, sample_findings)
        assert gen.registry == full_registry
        assert gen.findings == sample_findings


class TestDocGeneratorOutput:
    def test_generates_markdown(self, full_registry, sample_findings):
        gen = PipelineDocGenerator(full_registry, sample_findings)
        doc = gen.generate()
        assert isinstance(doc, str)
        assert len(doc) > 100

    def test_includes_header(self, full_registry, sample_findings):
        gen = PipelineDocGenerator(full_registry, sample_findings)
        doc = gen.generate()
        assert "# Pipeline Specification" in doc

    def test_includes_data_overview(self, full_registry, sample_findings):
        gen = PipelineDocGenerator(full_registry, sample_findings)
        doc = gen.generate()
        assert "Data Overview" in doc
        assert "customers.csv" in doc
        assert "1000" in doc or "1,000" in doc


class TestBronzeSection:
    def test_includes_bronze_section(self, full_registry, sample_findings):
        gen = PipelineDocGenerator(full_registry, sample_findings)
        doc = gen.generate()
        assert "Bronze" in doc

    def test_includes_null_handling_details(self, full_registry, sample_findings):
        gen = PipelineDocGenerator(full_registry, sample_findings)
        doc = gen.generate()
        assert "age" in doc
        assert "median" in doc or "impute" in doc.lower()

    def test_includes_outlier_handling_details(self, full_registry, sample_findings):
        gen = PipelineDocGenerator(full_registry, sample_findings)
        doc = gen.generate()
        assert "revenue" in doc
        assert "outlier" in doc.lower()


class TestSilverSection:
    def test_includes_silver_section(self, full_registry, sample_findings):
        gen = PipelineDocGenerator(full_registry, sample_findings)
        doc = gen.generate()
        assert "Silver" in doc

    def test_includes_aggregation_details(self, full_registry, sample_findings):
        gen = PipelineDocGenerator(full_registry, sample_findings)
        doc = gen.generate()
        assert "aggregation" in doc.lower() or "sum" in doc.lower()
        assert "7d" in doc or "30d" in doc


class TestGoldSection:
    def test_includes_gold_section(self, full_registry, sample_findings):
        gen = PipelineDocGenerator(full_registry, sample_findings)
        doc = gen.generate()
        assert "Gold" in doc

    def test_includes_encoding_details(self, full_registry, sample_findings):
        gen = PipelineDocGenerator(full_registry, sample_findings)
        doc = gen.generate()
        assert "contract_type" in doc
        assert "one_hot" in doc.lower() or "encoding" in doc.lower()

    def test_includes_scaling_details(self, full_registry, sample_findings):
        gen = PipelineDocGenerator(full_registry, sample_findings)
        doc = gen.generate()
        assert "standard" in doc.lower() or "scaling" in doc.lower()


class TestEmptyRecommendations:
    def test_handles_empty_registry(self, sample_findings):
        registry = RecommendationRegistry()
        gen = PipelineDocGenerator(registry, sample_findings)
        doc = gen.generate()
        assert "# Pipeline Specification" in doc
        assert "No recommendations" in doc.lower() or len(doc) > 50


class TestDocStructure:
    def test_uses_markdown_headers(self, full_registry, sample_findings):
        gen = PipelineDocGenerator(full_registry, sample_findings)
        doc = gen.generate()
        assert "##" in doc

    def test_uses_bullet_points_or_tables(self, full_registry, sample_findings):
        gen = PipelineDocGenerator(full_registry, sample_findings)
        doc = gen.generate()
        has_bullets = "-" in doc or "*" in doc
        has_tables = "|" in doc
        assert has_bullets or has_tables

    def test_includes_rationale_for_recommendations(self, full_registry, sample_findings):
        gen = PipelineDocGenerator(full_registry, sample_findings)
        doc = gen.generate()
        assert "5% missing" in doc or "missing" in doc.lower()


class TestLLMAssistantFormat:
    def test_includes_column_schema_section(self, full_registry, sample_findings):
        gen = PipelineDocGenerator(full_registry, sample_findings)
        doc = gen.generate()
        assert "Column Schema" in doc or "Schema" in doc

    def test_includes_column_types(self, full_registry, sample_findings):
        gen = PipelineDocGenerator(full_registry, sample_findings)
        doc = gen.generate()
        assert "IDENTIFIER" in doc or "identifier" in doc.lower()
        assert "NUMERIC" in doc or "numeric" in doc.lower()

    def test_includes_column_statistics(self, full_registry, sample_findings):
        gen = PipelineDocGenerator(full_registry, sample_findings)
        doc = gen.generate()
        assert "mean" in doc.lower() or "35.5" in doc
        assert "null" in doc.lower() or "missing" in doc.lower()

    def test_includes_implementation_hints(self, full_registry, sample_findings):
        gen = PipelineDocGenerator(full_registry, sample_findings)
        doc = gen.generate()
        assert "Implementation" in doc or "How to" in doc or "pyspark" in doc.lower()

    def test_bronze_has_code_pattern(self, full_registry, sample_findings):
        gen = PipelineDocGenerator(full_registry, sample_findings)
        doc = gen.generate()
        assert "fillna" in doc.lower() or "na.fill" in doc.lower() or "impute" in doc.lower()

    def test_gold_has_encoding_pattern(self, full_registry, sample_findings):
        gen = PipelineDocGenerator(full_registry, sample_findings)
        doc = gen.generate()
        assert "StringIndexer" in doc or "OneHotEncoder" in doc or "get_dummies" in doc.lower()

    def test_includes_delta_table_structure(self, full_registry, sample_findings):
        gen = PipelineDocGenerator(full_registry, sample_findings)
        doc = gen.generate()
        assert "delta" in doc.lower() or "table" in doc.lower()

    def test_includes_target_variable_context(self, full_registry, sample_findings):
        gen = PipelineDocGenerator(full_registry, sample_findings)
        doc = gen.generate()
        assert "churned" in doc
        assert "target" in doc.lower() or "predict" in doc.lower()


class TestAssistantPromptSection:
    def test_includes_assistant_section(self, full_registry, sample_findings):
        gen = PipelineDocGenerator(full_registry, sample_findings)
        doc = gen.generate()
        assert "Assistant" in doc or "LLM" in doc or "Generate" in doc

    def test_assistant_section_describes_task(self, full_registry, sample_findings):
        gen = PipelineDocGenerator(full_registry, sample_findings)
        doc = gen.generate()
        has_task = "generate" in doc.lower() or "create" in doc.lower() or "implement" in doc.lower()
        assert has_task

    def test_assistant_section_mentions_pyspark(self, full_registry, sample_findings):
        gen = PipelineDocGenerator(full_registry, sample_findings)
        doc = gen.generate()
        assert "pyspark" in doc.lower() or "spark" in doc.lower()


class TestOutputQuality:
    def test_no_framework_references(self, full_registry, sample_findings):
        gen = PipelineDocGenerator(full_registry, sample_findings)
        doc = gen.generate()
        assert "customer_retention" not in doc
        assert "from customer_retention" not in doc

    def test_operations_are_explicit(self, full_registry, sample_findings):
        gen = PipelineDocGenerator(full_registry, sample_findings)
        doc = gen.generate()
        assert "median" in doc.lower()
        assert "iqr" in doc.lower() or "outlier" in doc.lower()
        assert "one_hot" in doc.lower() or "one-hot" in doc.lower()
        assert "standard" in doc.lower()


@pytest.fixture
def multi_source_registry():
    registry = RecommendationRegistry()
    registry.add_source("customers", "/mnt/landing/customers.csv")
    registry.add_source("events", "/mnt/landing/events.csv")
    registry.init_silver("customer_id", "event_date")
    registry.init_gold("churned")
    registry.add_bronze_null("age", "median", "5% missing", "03", source="customers")
    registry.add_bronze_null("event_value", "zero", "3% missing", "03", source="events")
    registry.add_silver_join("customers", "events", ["customer_id"], "left", "Enrich with events")
    registry.gold.encoding.append(LayeredRecommendation(
        id="gold_encode_contract", layer="gold", category="encoding", action="one_hot",
        target_column="contract_type", parameters={"method": "one_hot"},
        rationale="Low cardinality", source_notebook="06"
    ))
    return registry


class TestMultiSourceDocs:
    def test_lists_all_sources(self, multi_source_registry):
        gen = PipelineDocGenerator(multi_source_registry)
        doc = gen.generate()
        assert "customers" in doc.lower()
        assert "events" in doc.lower()

    def test_shows_source_paths(self, multi_source_registry):
        gen = PipelineDocGenerator(multi_source_registry)
        doc = gen.generate()
        assert "customers.csv" in doc
        assert "events.csv" in doc

    def test_shows_per_source_bronze_recommendations(self, multi_source_registry):
        gen = PipelineDocGenerator(multi_source_registry)
        doc = gen.generate()
        assert "age" in doc
        assert "event_value" in doc

    def test_shows_join_specification(self, multi_source_registry):
        gen = PipelineDocGenerator(multi_source_registry)
        doc = gen.generate()
        assert "join" in doc.lower()
        assert "customer_id" in doc

    def test_indicates_parallel_execution(self, multi_source_registry):
        gen = PipelineDocGenerator(multi_source_registry)
        doc = gen.generate()
        assert "parallel" in doc.lower() or "independent" in doc.lower()


class TestNotebookStructureDoc:
    def test_describes_folder_structure(self, multi_source_registry):
        gen = PipelineDocGenerator(multi_source_registry)
        doc = gen.generate()
        has_structure = "bronze/" in doc or "silver/" in doc or "structure" in doc.lower()
        assert has_structure

    def test_describes_execution_order(self, multi_source_registry):
        gen = PipelineDocGenerator(multi_source_registry)
        doc = gen.generate()
        has_order = ("then" in doc.lower() or "after" in doc.lower() or
                     "1." in doc or "step" in doc.lower())
        assert has_order
