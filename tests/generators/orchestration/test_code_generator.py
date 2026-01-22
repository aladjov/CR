"""Tests for PipelineCodeGenerator that produces adapter-pattern Python code."""
import pytest

from customer_retention.generators.orchestration.code_generator import PipelineCodeGenerator
from customer_retention.analysis.auto_explorer.layered_recommendations import (
    RecommendationRegistry,
    LayeredRecommendation,
)


@pytest.fixture
def bronze_recommendations():
    registry = RecommendationRegistry()
    registry.init_bronze("customers.csv")
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
    return registry


@pytest.fixture
def silver_recommendations():
    registry = RecommendationRegistry()
    registry.init_silver("customer_id", "event_date")
    registry.silver.aggregations.append(LayeredRecommendation(
        id="silver_agg_revenue", layer="silver", category="aggregation", action="sum",
        target_column="revenue", parameters={"aggregation": "sum", "windows": ["7d", "30d"]},
        rationale="Revenue trends", source_notebook="04TS"
    ))
    registry.silver.derived_columns.append(LayeredRecommendation(
        id="silver_derive_avg", layer="silver", category="derived", action="compute",
        target_column="avg_order", parameters={"formula": "revenue / order_count"},
        rationale="Average order value", source_notebook="05"
    ))
    return registry


@pytest.fixture
def gold_recommendations():
    registry = RecommendationRegistry()
    registry.init_gold("churned")
    registry.gold.encoding.append(LayeredRecommendation(
        id="gold_encode_contract", layer="gold", category="encoding", action="one_hot",
        target_column="contract_type", parameters={"method": "one_hot", "drop_first": True},
        rationale="Low cardinality", source_notebook="06"
    ))
    registry.gold.scaling.append(LayeredRecommendation(
        id="gold_scale_revenue", layer="gold", category="scaling", action="standard",
        target_column="revenue", parameters={"method": "standard"},
        rationale="Normalize", source_notebook="06"
    ))
    registry.gold.transformations.append(LayeredRecommendation(
        id="gold_transform_revenue", layer="gold", category="transformation", action="log",
        target_column="revenue", parameters={"method": "log"},
        rationale="High skewness", source_notebook="06"
    ))
    return registry


@pytest.fixture
def full_registry(bronze_recommendations, silver_recommendations, gold_recommendations):
    registry = RecommendationRegistry()
    registry.bronze = bronze_recommendations.bronze
    registry.silver = silver_recommendations.silver
    registry.gold = gold_recommendations.gold
    return registry


class TestCodeGeneratorInit:
    def test_creates_generator(self, bronze_recommendations):
        gen = PipelineCodeGenerator(bronze_recommendations)
        assert gen.registry == bronze_recommendations


class TestBronzeCodeGeneration:
    def test_generates_bronze_function(self, bronze_recommendations):
        gen = PipelineCodeGenerator(bronze_recommendations)
        code = gen.generate_bronze_code()
        assert "def bronze_transform(df)" in code

    def test_includes_imports(self, bronze_recommendations):
        gen = PipelineCodeGenerator(bronze_recommendations)
        code = gen.generate_bronze_code()
        assert "from customer_retention.stages.cleaning import" in code

    def test_generates_null_imputation(self, bronze_recommendations):
        gen = PipelineCodeGenerator(bronze_recommendations)
        code = gen.generate_bronze_code()
        assert "MissingValueHandler" in code
        assert "median" in code
        assert "age" in code

    def test_generates_outlier_capping(self, bronze_recommendations):
        gen = PipelineCodeGenerator(bronze_recommendations)
        code = gen.generate_bronze_code()
        assert "OutlierHandler" in code
        assert "iqr" in code
        assert "revenue" in code

    def test_bronze_code_is_executable_syntax(self, bronze_recommendations):
        gen = PipelineCodeGenerator(bronze_recommendations)
        code = gen.generate_bronze_code()
        compile(code, "<string>", "exec")


class TestSilverCodeGeneration:
    def test_generates_silver_function(self, silver_recommendations):
        gen = PipelineCodeGenerator(silver_recommendations)
        code = gen.generate_silver_code()
        assert "def silver_transform(df)" in code

    def test_includes_aggregation_logic(self, silver_recommendations):
        gen = PipelineCodeGenerator(silver_recommendations)
        code = gen.generate_silver_code()
        assert "groupby" in code or "agg" in code
        assert "revenue" in code

    def test_includes_time_windows(self, silver_recommendations):
        gen = PipelineCodeGenerator(silver_recommendations)
        code = gen.generate_silver_code()
        assert "7d" in code or "7" in code

    def test_includes_derived_columns(self, silver_recommendations):
        gen = PipelineCodeGenerator(silver_recommendations)
        code = gen.generate_silver_code()
        assert "avg_order" in code

    def test_silver_code_is_executable_syntax(self, silver_recommendations):
        gen = PipelineCodeGenerator(silver_recommendations)
        code = gen.generate_silver_code()
        compile(code, "<string>", "exec")


class TestGoldCodeGeneration:
    def test_generates_gold_function(self, gold_recommendations):
        gen = PipelineCodeGenerator(gold_recommendations)
        code = gen.generate_gold_code()
        assert "def gold_transform(df)" in code

    def test_includes_encoding(self, gold_recommendations):
        gen = PipelineCodeGenerator(gold_recommendations)
        code = gen.generate_gold_code()
        assert "one_hot" in code.lower() or "get_dummies" in code
        assert "contract_type" in code

    def test_includes_scaling(self, gold_recommendations):
        gen = PipelineCodeGenerator(gold_recommendations)
        code = gen.generate_gold_code()
        assert "StandardScaler" in code or "scale" in code.lower()

    def test_includes_transformations(self, gold_recommendations):
        gen = PipelineCodeGenerator(gold_recommendations)
        code = gen.generate_gold_code()
        assert "log" in code.lower()

    def test_gold_code_is_executable_syntax(self, gold_recommendations):
        gen = PipelineCodeGenerator(gold_recommendations)
        code = gen.generate_gold_code()
        compile(code, "<string>", "exec")


class TestFullPipelineGeneration:
    def test_generates_full_pipeline(self, full_registry):
        gen = PipelineCodeGenerator(full_registry)
        code = gen.generate_full_pipeline()
        assert "bronze_transform" in code
        assert "silver_transform" in code
        assert "gold_transform" in code

    def test_full_pipeline_has_main_function(self, full_registry):
        gen = PipelineCodeGenerator(full_registry)
        code = gen.generate_full_pipeline()
        assert "def run_pipeline" in code or "def transform_pipeline" in code

    def test_full_pipeline_chains_transforms(self, full_registry):
        gen = PipelineCodeGenerator(full_registry)
        code = gen.generate_full_pipeline()
        assert "bronze_transform" in code
        assert "silver_transform" in code
        assert "gold_transform" in code

    def test_full_pipeline_is_executable_syntax(self, full_registry):
        gen = PipelineCodeGenerator(full_registry)
        code = gen.generate_full_pipeline()
        compile(code, "<string>", "exec")


class TestEmptyRecommendations:
    def test_handles_empty_bronze(self):
        registry = RecommendationRegistry()
        registry.init_bronze("data.csv")
        gen = PipelineCodeGenerator(registry)
        code = gen.generate_bronze_code()
        assert "def bronze_transform(df)" in code
        assert "return df" in code

    def test_handles_no_silver(self):
        registry = RecommendationRegistry()
        gen = PipelineCodeGenerator(registry)
        code = gen.generate_silver_code()
        assert "def silver_transform(df)" in code
        assert "return df" in code

    def test_handles_no_gold(self):
        registry = RecommendationRegistry()
        gen = PipelineCodeGenerator(registry)
        code = gen.generate_gold_code()
        assert "def gold_transform(df)" in code
        assert "return df" in code
