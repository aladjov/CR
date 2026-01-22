"""Tests for fluent RecommendationBuilder used by exploration notebooks."""

import pytest

from customer_retention.analysis.auto_explorer.findings import ColumnFinding, ExplorationFindings
from customer_retention.analysis.auto_explorer.recommendation_builder import (
    BronzeBuilder,
    GoldBuilder,
    RecommendationBuilder,
    SilverBuilder,
)
from customer_retention.core.config.column_config import ColumnType


@pytest.fixture
def sample_findings():
    columns = {
        "customer_id": ColumnFinding(
            name="customer_id", inferred_type=ColumnType.IDENTIFIER,
            confidence=0.95, evidence=["Unique"], universal_metrics={"null_count": 0}
        ),
        "age": ColumnFinding(
            name="age", inferred_type=ColumnType.NUMERIC_CONTINUOUS,
            confidence=0.9, evidence=["Numeric"],
            universal_metrics={"null_count": 50, "null_percentage": 5.0},
            type_metrics={"outlier_percentage": 8.0, "skewness": 0.5}
        ),
        "revenue": ColumnFinding(
            name="revenue", inferred_type=ColumnType.NUMERIC_CONTINUOUS,
            confidence=0.9, evidence=["Numeric"],
            universal_metrics={"null_count": 0},
            type_metrics={"outlier_percentage": 12.0, "skewness": 2.5}
        ),
        "contract": ColumnFinding(
            name="contract", inferred_type=ColumnType.CATEGORICAL_NOMINAL,
            confidence=0.9, evidence=["Categorical"],
            universal_metrics={"null_count": 0},
            type_metrics={"cardinality": 3}
        ),
        "churned": ColumnFinding(
            name="churned", inferred_type=ColumnType.BINARY,
            confidence=0.9, evidence=["Binary"],
            universal_metrics={"null_count": 0, "distinct_count": 2}
        )
    }
    return ExplorationFindings(
        source_path="data.csv", source_format="csv",
        row_count=1000, column_count=5, columns=columns,
        target_column="churned", identifier_columns=["customer_id"]
    )


class TestRecommendationBuilderInit:
    def test_creates_builder_with_findings(self, sample_findings):
        builder = RecommendationBuilder(sample_findings, "03_quality")
        assert builder.findings == sample_findings
        assert builder.notebook == "03_quality"

    def test_initializes_registry(self, sample_findings):
        builder = RecommendationBuilder(sample_findings, "03_quality")
        assert builder.registry is not None


class TestBronzeBuilder:
    def test_returns_bronze_builder(self, sample_findings):
        builder = RecommendationBuilder(sample_findings, "03_quality")
        bronze = builder.bronze()
        assert isinstance(bronze, BronzeBuilder)

    def test_bronze_impute_nulls(self, sample_findings):
        builder = RecommendationBuilder(sample_findings, "03_quality")
        builder.bronze().impute_nulls("age", "median", "5% missing")
        recs = builder.registry.get_by_layer("bronze")
        assert len(recs) == 1
        assert recs[0].target_column == "age"
        assert recs[0].parameters["strategy"] == "median"

    def test_bronze_cap_outliers(self, sample_findings):
        builder = RecommendationBuilder(sample_findings, "03_quality")
        builder.bronze().cap_outliers("revenue", "iqr", factor=1.5, reason="12% outliers")
        recs = builder.registry.get_by_layer("bronze")
        assert len(recs) == 1
        assert recs[0].action == "cap"
        assert recs[0].parameters["factor"] == 1.5

    def test_bronze_drop_column(self, sample_findings):
        builder = RecommendationBuilder(sample_findings, "03_quality")
        builder.bronze().drop_column("unused_col", "Not needed")
        recs = builder.registry.get_by_layer("bronze")
        assert len(recs) == 1
        assert recs[0].action == "drop"

    def test_bronze_convert_type(self, sample_findings):
        builder = RecommendationBuilder(sample_findings, "03_quality")
        builder.bronze().convert_type("date_col", "datetime", "Parse date string")
        recs = builder.registry.get_by_layer("bronze")
        assert len(recs) == 1
        assert recs[0].category == "type"

    def test_bronze_fluent_chaining(self, sample_findings):
        builder = RecommendationBuilder(sample_findings, "03_quality")
        (builder.bronze()
            .impute_nulls("age", "median", "5% missing")
            .cap_outliers("revenue", "iqr", reason="high outliers")
            .drop_column("unused", "not needed"))
        recs = builder.registry.get_by_layer("bronze")
        assert len(recs) == 3


class TestSilverBuilder:
    def test_returns_silver_builder(self, sample_findings):
        builder = RecommendationBuilder(sample_findings, "04_relationship")
        silver = builder.silver()
        assert isinstance(silver, SilverBuilder)

    def test_silver_aggregate_timeseries(self, sample_findings):
        builder = RecommendationBuilder(sample_findings, "04TS")
        builder.silver().aggregate("revenue", "sum", windows=["7d", "30d"], reason="Revenue trends")
        recs = builder.registry.get_by_layer("silver")
        assert len(recs) == 1
        assert recs[0].parameters["windows"] == ["7d", "30d"]

    def test_silver_join_dataset(self, sample_findings):
        builder = RecommendationBuilder(sample_findings, "04_relationship")
        builder.silver().join("orders.csv", join_key="customer_id", join_type="left", reason="Add order data")
        recs = builder.registry.get_by_layer("silver")
        assert len(recs) == 1
        assert recs[0].action == "join"

    def test_silver_derive_column(self, sample_findings):
        builder = RecommendationBuilder(sample_findings, "05_features")
        builder.silver().derive("tenure_months", "days_active / 30", reason="Convert to months")
        recs = builder.registry.get_by_layer("silver")
        assert len(recs) == 1
        assert recs[0].category == "derived"

    def test_silver_fluent_chaining(self, sample_findings):
        builder = RecommendationBuilder(sample_findings, "04TS")
        (builder.silver()
            .aggregate("revenue", "sum", windows=["7d"], reason="")
            .aggregate("purchases", "count", windows=["30d"], reason="")
            .derive("avg_order", "revenue / purchases", reason=""))
        recs = builder.registry.get_by_layer("silver")
        assert len(recs) == 3


class TestGoldBuilder:
    def test_returns_gold_builder(self, sample_findings):
        builder = RecommendationBuilder(sample_findings, "06_modeling")
        gold = builder.gold()
        assert isinstance(gold, GoldBuilder)

    def test_gold_encode_onehot(self, sample_findings):
        builder = RecommendationBuilder(sample_findings, "06_modeling")
        builder.gold().encode("contract", "one_hot", "Low cardinality")
        recs = builder.registry.get_by_layer("gold")
        assert len(recs) == 1
        assert recs[0].action == "one_hot"

    def test_gold_encode_target(self, sample_findings):
        builder = RecommendationBuilder(sample_findings, "06_modeling")
        builder.gold().encode("category", "target", "High cardinality", smoothing=1.0)
        recs = builder.registry.get_by_layer("gold")
        assert len(recs) == 1
        assert recs[0].parameters.get("smoothing") == 1.0

    def test_gold_scale_standard(self, sample_findings):
        builder = RecommendationBuilder(sample_findings, "06_modeling")
        builder.gold().scale("age", "standard", "Normalize for model")
        recs = builder.registry.get_by_layer("gold")
        assert len(recs) == 1
        assert recs[0].action == "standard"

    def test_gold_scale_robust(self, sample_findings):
        builder = RecommendationBuilder(sample_findings, "06_modeling")
        builder.gold().scale("revenue", "robust", "Has outliers")
        recs = builder.registry.get_by_layer("gold")
        assert len(recs) == 1
        assert recs[0].parameters["method"] == "robust"

    def test_gold_select_feature(self, sample_findings):
        builder = RecommendationBuilder(sample_findings, "06_modeling")
        builder.gold().select("age", include=True, reason="High IV")
        recs = builder.registry.get_by_layer("gold")
        assert len(recs) == 1
        assert recs[0].category == "selection"

    def test_gold_drop_feature(self, sample_findings):
        builder = RecommendationBuilder(sample_findings, "06_modeling")
        builder.gold().select("low_iv_col", include=False, reason="IV < 0.02")
        recs = builder.registry.get_by_layer("gold")
        assert len(recs) == 1
        assert recs[0].action == "exclude"

    def test_gold_transform_log(self, sample_findings):
        builder = RecommendationBuilder(sample_findings, "06_modeling")
        builder.gold().transform("revenue", "log", "High skewness")
        recs = builder.registry.get_by_layer("gold")
        assert len(recs) == 1
        assert recs[0].action == "log"

    def test_gold_fluent_chaining(self, sample_findings):
        builder = RecommendationBuilder(sample_findings, "06_modeling")
        (builder.gold()
            .encode("contract", "one_hot", "")
            .scale("age", "standard", "")
            .transform("revenue", "log", "")
            .select("tenure", include=True, reason=""))
        recs = builder.registry.get_by_layer("gold")
        assert len(recs) == 4


class TestBuilderPersistence:
    def test_get_all_recommendations(self, sample_findings):
        builder = RecommendationBuilder(sample_findings, "test")
        builder.bronze().impute_nulls("age", "median", "")
        builder.silver().aggregate("revenue", "sum", windows=["7d"], reason="")
        builder.gold().encode("contract", "one_hot", "")
        all_recs = builder.all_recommendations
        assert len(all_recs) == 3

    def test_to_dict(self, sample_findings):
        builder = RecommendationBuilder(sample_findings, "test")
        builder.bronze().impute_nulls("age", "median", "5% missing")
        d = builder.to_dict()
        assert "bronze" in d
        assert len(d["bronze"]["null_handling"]) == 1


class TestBuilderIntegration:
    def test_full_workflow(self, sample_findings):
        builder = RecommendationBuilder(sample_findings, "03_quality")
        (builder.bronze()
            .impute_nulls("age", "median", "5% missing")
            .cap_outliers("revenue", "iqr", reason="12% outliers"))
        builder.gold().encode("contract", "one_hot", "3 categories")
        bronze_recs = builder.registry.get_by_layer("bronze")
        gold_recs = builder.registry.get_by_layer("gold")
        assert len(bronze_recs) == 2
        assert len(gold_recs) == 1
