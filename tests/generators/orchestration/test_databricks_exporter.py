"""Tests for DatabricksExporter that generates standalone PySpark notebooks."""
import pytest

from customer_retention.analysis.auto_explorer.findings import ColumnFinding, ExplorationFindings
from customer_retention.analysis.auto_explorer.layered_recommendations import (
    LayeredRecommendation,
    RecommendationRegistry,
)
from customer_retention.core.config.column_config import ColumnType
from customer_retention.generators.orchestration.databricks_exporter import DatabricksExporter


@pytest.fixture
def sample_findings():
    columns = {
        "customer_id": ColumnFinding(
            name="customer_id", inferred_type=ColumnType.IDENTIFIER,
            confidence=0.95, evidence=["Unique"],
            universal_metrics={"null_count": 0}
        ),
        "age": ColumnFinding(
            name="age", inferred_type=ColumnType.NUMERIC_CONTINUOUS,
            confidence=0.9, evidence=["Numeric"],
            universal_metrics={"null_count": 50, "null_percentage": 5.0}
        ),
    }
    return ExplorationFindings(
        source_path="/mnt/landing/customers.csv", source_format="csv",
        row_count=10000, column_count=2, columns=columns,
        target_column="churned", identifier_columns=["customer_id"]
    )


@pytest.fixture
def full_registry():
    registry = RecommendationRegistry()
    registry.init_bronze("/mnt/landing/customers.csv")
    registry.init_silver("customer_id", "event_date")
    registry.init_gold("churned")
    registry.bronze.null_handling.append(LayeredRecommendation(
        id="bronze_null_age", layer="bronze", category="null", action="impute",
        target_column="age", parameters={"strategy": "median"},
        rationale="5% missing", source_notebook="03"
    ))
    registry.bronze.outlier_handling.append(LayeredRecommendation(
        id="bronze_outlier_revenue", layer="bronze", category="outlier", action="cap",
        target_column="revenue", parameters={"method": "iqr", "factor": 1.5},
        rationale="12% outliers", source_notebook="03"
    ))
    registry.silver.aggregations.append(LayeredRecommendation(
        id="silver_agg_revenue", layer="silver", category="aggregation", action="sum",
        target_column="revenue", parameters={"aggregation": "sum", "windows": ["7d", "30d"]},
        rationale="Revenue over time", source_notebook="04TS"
    ))
    registry.gold.encoding.append(LayeredRecommendation(
        id="gold_encode_contract", layer="gold", category="encoding", action="one_hot",
        target_column="contract_type", parameters={"method": "one_hot"},
        rationale="Low cardinality", source_notebook="06"
    ))
    registry.gold.scaling.append(LayeredRecommendation(
        id="gold_scale_revenue", layer="gold", category="scaling", action="standard",
        target_column="revenue", parameters={"method": "standard"},
        rationale="Normalize", source_notebook="06"
    ))
    return registry


class TestDatabricksExporterInit:
    def test_creates_with_registry_and_findings(self, full_registry, sample_findings):
        exporter = DatabricksExporter(full_registry, sample_findings)
        assert exporter.registry == full_registry
        assert exporter.findings == sample_findings


class TestStandaloneCodeGeneration:
    def test_generates_no_framework_imports(self, full_registry, sample_findings):
        exporter = DatabricksExporter(full_registry, sample_findings)
        code = exporter.generate_notebook()
        assert "customer_retention" not in code
        assert "from customer_retention" not in code

    def test_uses_pyspark_imports(self, full_registry, sample_findings):
        exporter = DatabricksExporter(full_registry, sample_findings)
        code = exporter.generate_notebook()
        assert "pyspark" in code or "spark" in code

    def test_uses_pyspark_functions(self, full_registry, sample_findings):
        exporter = DatabricksExporter(full_registry, sample_findings)
        code = exporter.generate_notebook()
        assert "from pyspark.sql import" in code or "spark.read" in code


class TestBronzeNotebook:
    def test_generates_bronze_cell(self, full_registry, sample_findings):
        exporter = DatabricksExporter(full_registry, sample_findings)
        code = exporter.generate_bronze_notebook()
        assert "BRONZE" in code.upper() or "bronze" in code

    def test_bronze_uses_spark_fillna(self, full_registry, sample_findings):
        exporter = DatabricksExporter(full_registry, sample_findings)
        code = exporter.generate_bronze_notebook()
        assert "fillna" in code or "fill" in code or "na.fill" in code

    def test_bronze_reads_from_landing(self, full_registry, sample_findings):
        exporter = DatabricksExporter(full_registry, sample_findings)
        code = exporter.generate_bronze_notebook()
        assert "landing" in code.lower() or "read" in code

    def test_bronze_writes_to_delta(self, full_registry, sample_findings):
        exporter = DatabricksExporter(full_registry, sample_findings)
        code = exporter.generate_bronze_notebook()
        assert "delta" in code.lower() or "write" in code


class TestSilverNotebook:
    def test_generates_silver_cell(self, full_registry, sample_findings):
        exporter = DatabricksExporter(full_registry, sample_findings)
        code = exporter.generate_silver_notebook()
        assert "SILVER" in code.upper() or "silver" in code

    def test_silver_uses_groupby(self, full_registry, sample_findings):
        exporter = DatabricksExporter(full_registry, sample_findings)
        code = exporter.generate_silver_notebook()
        assert "groupBy" in code or "groupby" in code or "agg" in code


class TestGoldNotebook:
    def test_generates_gold_cell(self, full_registry, sample_findings):
        exporter = DatabricksExporter(full_registry, sample_findings)
        code = exporter.generate_gold_notebook()
        assert "GOLD" in code.upper() or "gold" in code

    def test_gold_uses_stringindexer_or_onehot(self, full_registry, sample_findings):
        exporter = DatabricksExporter(full_registry, sample_findings)
        code = exporter.generate_gold_notebook()
        has_encoding = "StringIndexer" in code or "OneHotEncoder" in code or "get_dummies" in code.lower()
        assert has_encoding or "when(" in code

    def test_gold_uses_standardscaler(self, full_registry, sample_findings):
        exporter = DatabricksExporter(full_registry, sample_findings)
        code = exporter.generate_gold_notebook()
        assert "StandardScaler" in code or "scale" in code.lower()


class TestFullNotebookExport:
    def test_generates_complete_notebook(self, full_registry, sample_findings):
        exporter = DatabricksExporter(full_registry, sample_findings)
        code = exporter.generate_notebook()
        assert "bronze" in code.lower()
        assert "silver" in code.lower()
        assert "gold" in code.lower()

    def test_notebook_has_markdown_cells(self, full_registry, sample_findings):
        exporter = DatabricksExporter(full_registry, sample_findings)
        code = exporter.generate_notebook()
        assert "# COMMAND" in code or "# MAGIC" in code or "##" in code

    def test_notebook_is_valid_python_syntax(self, full_registry, sample_findings):
        exporter = DatabricksExporter(full_registry, sample_findings)
        code = exporter.generate_notebook()
        python_code = "\n".join(
            line for line in code.split("\n")
            if not line.strip().startswith("# MAGIC") and not line.strip().startswith("# COMMAND")
        )
        compile(python_code, "<string>", "exec")


class TestDeltaTablePaths:
    def test_uses_configurable_catalog(self, full_registry, sample_findings):
        exporter = DatabricksExporter(full_registry, sample_findings, catalog="main", schema="churn")
        code = exporter.generate_notebook()
        assert "main" in code or "churn" in code

    def test_default_paths_are_relative(self, full_registry, sample_findings):
        exporter = DatabricksExporter(full_registry, sample_findings)
        code = exporter.generate_notebook()
        assert "bronze" in code.lower()


class TestNotebookFormat:
    def test_exports_as_python_string(self, full_registry, sample_findings):
        exporter = DatabricksExporter(full_registry, sample_findings)
        code = exporter.generate_notebook()
        assert isinstance(code, str)

    def test_exports_as_dbc_dict(self, full_registry, sample_findings):
        exporter = DatabricksExporter(full_registry, sample_findings)
        cells = exporter.to_notebook_cells()
        assert isinstance(cells, list)
        assert len(cells) > 0
        assert all("content" in cell for cell in cells)


@pytest.fixture
def multi_source_registry():
    registry = RecommendationRegistry()
    registry.add_source("customers", "/mnt/landing/customers.csv")
    registry.add_source("events", "/mnt/landing/events.csv")
    registry.init_silver("customer_id", "event_date")
    registry.init_gold("churned")
    registry.add_bronze_null("age", "median", "5% missing", "03", source="customers")
    registry.add_bronze_outlier("revenue", "cap", {"method": "iqr"}, "12% outliers", "03", source="customers")
    registry.add_bronze_null("event_value", "zero", "3% missing", "03", source="events")
    registry.add_silver_join("customers", "events", ["customer_id"], "left", "Enrich with events")
    registry.gold.encoding.append(LayeredRecommendation(
        id="gold_encode_contract", layer="gold", category="encoding", action="one_hot",
        target_column="contract_type", parameters={"method": "one_hot"},
        rationale="Low cardinality", source_notebook="06"
    ))
    return registry


class TestMultiSourceNotebooks:
    def test_generates_per_source_bronze_notebooks(self, multi_source_registry):
        exporter = DatabricksExporter(multi_source_registry)
        notebooks = exporter.generate_source_notebooks()
        assert "customers" in notebooks
        assert "events" in notebooks
        assert len(notebooks) == 2

    def test_source_notebook_contains_source_path(self, multi_source_registry):
        exporter = DatabricksExporter(multi_source_registry)
        notebooks = exporter.generate_source_notebooks()
        assert "customers.csv" in notebooks["customers"]
        assert "events.csv" in notebooks["events"]

    def test_source_notebook_has_bronze_transformations(self, multi_source_registry):
        exporter = DatabricksExporter(multi_source_registry)
        notebooks = exporter.generate_source_notebooks()
        assert "age" in notebooks["customers"]
        assert "median" in notebooks["customers"].lower() or "fill" in notebooks["customers"].lower()
        assert "event_value" in notebooks["events"]

    def test_generates_silver_merge_notebook(self, multi_source_registry):
        exporter = DatabricksExporter(multi_source_registry)
        silver_code = exporter.generate_silver_merge_notebook()
        assert "join" in silver_code.lower()
        assert "customers" in silver_code.lower()
        assert "events" in silver_code.lower()

    def test_silver_merge_reads_from_bronze_tables(self, multi_source_registry):
        exporter = DatabricksExporter(multi_source_registry)
        silver_code = exporter.generate_silver_merge_notebook()
        assert "bronze_customers" in silver_code or "bronze" in silver_code.lower()

    def test_generates_gold_features_notebook(self, multi_source_registry):
        exporter = DatabricksExporter(multi_source_registry)
        gold_code = exporter.generate_gold_features_notebook()
        assert "gold" in gold_code.lower()
        assert "contract_type" in gold_code

    def test_exports_notebook_structure(self, multi_source_registry):
        exporter = DatabricksExporter(multi_source_registry)
        structure = exporter.export_notebook_structure()
        assert "bronze" in structure
        assert "silver" in structure
        assert "gold" in structure
        assert isinstance(structure["bronze"], dict)
        assert "customers" in structure["bronze"]
        assert "events" in structure["bronze"]


class TestParallelExecutionHints:
    def test_bronze_notebooks_are_independent(self, multi_source_registry):
        exporter = DatabricksExporter(multi_source_registry)
        notebooks = exporter.generate_source_notebooks()
        for name, code in notebooks.items():
            other_sources = [n for n in notebooks.keys() if n != name]
            for other in other_sources:
                assert other not in code.lower() or "join" not in code.lower()

    def test_structure_indicates_parallelism(self, multi_source_registry):
        exporter = DatabricksExporter(multi_source_registry)
        structure = exporter.export_notebook_structure()
        bronze_notebooks = structure.get("bronze", {})
        assert len(bronze_notebooks) >= 2
