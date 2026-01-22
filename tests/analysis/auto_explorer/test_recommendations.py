import pytest

from customer_retention.analysis.auto_explorer.findings import ColumnFinding, ExplorationFindings
from customer_retention.analysis.auto_explorer.recommendations import (
    CleaningRecommendation,
    FeatureRecommendation,
    RecommendationEngine,
    TargetRecommendation,
    TransformRecommendation,
)
from customer_retention.core.config.column_config import ColumnType


@pytest.fixture
def sample_findings() -> ExplorationFindings:
    columns = {
        "customer_id": ColumnFinding(
            name="customer_id",
            inferred_type=ColumnType.IDENTIFIER,
            confidence=0.95,
            evidence=["All unique"],
            universal_metrics={"null_count": 0, "distinct_count": 1000}
        ),
        "age": ColumnFinding(
            name="age",
            inferred_type=ColumnType.NUMERIC_CONTINUOUS,
            confidence=0.85,
            evidence=["Numeric with many values"],
            universal_metrics={"null_count": 50, "null_percentage": 5.0},
            type_metrics={"skewness": 0.5, "outlier_percentage": 2.0}
        ),
        "monthly_charges": ColumnFinding(
            name="monthly_charges",
            inferred_type=ColumnType.NUMERIC_CONTINUOUS,
            confidence=0.9,
            evidence=["Numeric continuous"],
            universal_metrics={"null_count": 0},
            type_metrics={"skewness": 2.5, "outlier_percentage": 8.0}
        ),
        "tenure": ColumnFinding(
            name="tenure",
            inferred_type=ColumnType.NUMERIC_DISCRETE,
            confidence=0.8,
            evidence=["Few unique numeric values"],
            universal_metrics={"null_count": 0, "distinct_count": 72}
        ),
        "contract_type": ColumnFinding(
            name="contract_type",
            inferred_type=ColumnType.CATEGORICAL_NOMINAL,
            confidence=0.9,
            evidence=["String with few categories"],
            universal_metrics={"null_count": 0},
            type_metrics={"cardinality": 3, "rare_category_count": 0}
        ),
        "payment_method": ColumnFinding(
            name="payment_method",
            inferred_type=ColumnType.CATEGORICAL_NOMINAL,
            confidence=0.85,
            evidence=["String categorical"],
            universal_metrics={"null_count": 100, "null_percentage": 10.0},
            type_metrics={"cardinality": 4, "rare_category_count": 1}
        ),
        "signup_date": ColumnFinding(
            name="signup_date",
            inferred_type=ColumnType.DATETIME,
            confidence=0.95,
            evidence=["Datetime column"],
            universal_metrics={"null_count": 0}
        ),
        "churned": ColumnFinding(
            name="churned",
            inferred_type=ColumnType.TARGET,
            confidence=0.9,
            evidence=["Binary target"],
            universal_metrics={"null_count": 0, "distinct_count": 2},
            type_metrics={"imbalance_ratio": 3.5}
        )
    }
    return ExplorationFindings(
        source_path="test_data.csv",
        source_format="csv",
        row_count=1000,
        column_count=8,
        columns=columns,
        target_column="churned",
        target_type="binary",
        identifier_columns=["customer_id"],
        datetime_columns=["signup_date"]
    )


@pytest.fixture
def findings_without_target() -> ExplorationFindings:
    columns = {
        "id": ColumnFinding(
            name="id",
            inferred_type=ColumnType.IDENTIFIER,
            confidence=0.95,
            evidence=["All unique"],
            universal_metrics={"null_count": 0}
        ),
        "churn_flag": ColumnFinding(
            name="churn_flag",
            inferred_type=ColumnType.BINARY,
            confidence=0.85,
            evidence=["Two unique values: 0, 1"],
            universal_metrics={"null_count": 0, "distinct_count": 2}
        ),
        "is_active": ColumnFinding(
            name="is_active",
            inferred_type=ColumnType.BINARY,
            confidence=0.8,
            evidence=["Boolean values"],
            universal_metrics={"null_count": 0, "distinct_count": 2}
        ),
        "score": ColumnFinding(
            name="score",
            inferred_type=ColumnType.NUMERIC_CONTINUOUS,
            confidence=0.9,
            evidence=["Numeric"],
            universal_metrics={"null_count": 0}
        )
    }
    return ExplorationFindings(
        source_path="test_data.csv",
        source_format="csv",
        row_count=100,
        column_count=4,
        columns=columns,
        target_column=None,
        identifier_columns=["id"]
    )


class TestRecommendationEngineInit:
    def test_default_init(self):
        engine = RecommendationEngine()
        assert engine is not None

    def test_custom_init(self):
        engine = RecommendationEngine(min_confidence=0.8)
        assert engine.min_confidence == 0.8


class TestTargetRecommendation:
    def test_recommend_target_when_already_detected(self, sample_findings):
        engine = RecommendationEngine()
        rec = engine.recommend_target(sample_findings)

        assert rec is not None
        assert rec.column_name == "churned"
        assert rec.confidence > 0.8
        assert "already detected" in rec.rationale.lower() or len(rec.rationale) > 0

    def test_recommend_target_from_binary_columns(self, findings_without_target):
        engine = RecommendationEngine()
        rec = engine.recommend_target(findings_without_target)

        assert rec is not None
        assert rec.column_name in ["churn_flag", "is_active"]
        assert "churn" in rec.column_name.lower() or rec.confidence > 0

    def test_target_recommendation_has_alternatives(self, findings_without_target):
        engine = RecommendationEngine()
        rec = engine.recommend_target(findings_without_target)

        assert hasattr(rec, "alternatives")
        assert isinstance(rec.alternatives, list)

    def test_target_recommendation_dataclass(self):
        rec = TargetRecommendation(
            column_name="target",
            confidence=0.9,
            rationale="Binary column with target pattern",
            alternatives=["other_col"],
            target_type="binary"
        )
        assert rec.column_name == "target"
        assert rec.target_type == "binary"


class TestFeatureRecommendations:
    def test_recommend_features_returns_list(self, sample_findings):
        engine = RecommendationEngine()
        recs = engine.recommend_features(sample_findings)

        assert isinstance(recs, list)
        assert len(recs) > 0

    def test_datetime_feature_recommendations(self, sample_findings):
        engine = RecommendationEngine()
        recs = engine.recommend_features(sample_findings)

        datetime_recs = [r for r in recs if "datetime" in r.feature_type.lower() or "temporal" in r.feature_type.lower()]
        assert len(datetime_recs) > 0

    def test_numeric_feature_recommendations(self, sample_findings):
        engine = RecommendationEngine()
        recs = engine.recommend_features(sample_findings)

        numeric_recs = [r for r in recs if r.source_column in ["age", "monthly_charges", "tenure"]]
        assert len(numeric_recs) > 0

    def test_categorical_feature_recommendations(self, sample_findings):
        engine = RecommendationEngine()
        recs = engine.recommend_features(sample_findings)

        cat_recs = [r for r in recs if r.source_column in ["contract_type", "payment_method"]]
        assert len(cat_recs) > 0

    def test_feature_recommendation_dataclass(self):
        rec = FeatureRecommendation(
            source_column="signup_date",
            feature_name="days_since_signup",
            feature_type="temporal",
            description="Days since customer signed up",
            priority="high",
            implementation_hint="Use DatetimeTransformer"
        )
        assert rec.source_column == "signup_date"
        assert rec.priority == "high"


class TestCleaningRecommendations:
    def test_recommend_cleaning_returns_list(self, sample_findings):
        engine = RecommendationEngine()
        recs = engine.recommend_cleaning(sample_findings)

        assert isinstance(recs, list)

    def test_cleaning_for_missing_values(self, sample_findings):
        engine = RecommendationEngine()
        recs = engine.recommend_cleaning(sample_findings)

        missing_recs = [r for r in recs if "missing" in r.issue_type.lower() or "null" in r.issue_type.lower()]
        assert len(missing_recs) > 0

    def test_cleaning_for_outliers(self, sample_findings):
        engine = RecommendationEngine()
        recs = engine.recommend_cleaning(sample_findings)

        outlier_recs = [r for r in recs if "outlier" in r.issue_type.lower()]
        assert len(outlier_recs) > 0

    def test_cleaning_recommendation_has_strategy(self, sample_findings):
        engine = RecommendationEngine()
        recs = engine.recommend_cleaning(sample_findings)

        for rec in recs:
            assert hasattr(rec, "strategy")
            assert rec.strategy is not None

    def test_cleaning_recommendation_dataclass(self):
        rec = CleaningRecommendation(
            column_name="age",
            issue_type="missing_values",
            severity="medium",
            strategy="impute_median",
            description="5% missing values in age column",
            affected_rows=50
        )
        assert rec.column_name == "age"
        assert rec.severity == "medium"


class TestTransformRecommendations:
    def test_recommend_transformations_returns_list(self, sample_findings):
        engine = RecommendationEngine()
        recs = engine.recommend_transformations(sample_findings)

        assert isinstance(recs, list)
        assert len(recs) > 0

    def test_numeric_scaling_recommendation(self, sample_findings):
        engine = RecommendationEngine()
        recs = engine.recommend_transformations(sample_findings)

        scaling_recs = [r for r in recs if "scal" in r.transform_type.lower()]
        assert len(scaling_recs) > 0

    def test_categorical_encoding_recommendation(self, sample_findings):
        engine = RecommendationEngine()
        recs = engine.recommend_transformations(sample_findings)

        encoding_recs = [r for r in recs if "encod" in r.transform_type.lower()]
        assert len(encoding_recs) > 0

    def test_skewed_data_log_transform(self, sample_findings):
        engine = RecommendationEngine()
        recs = engine.recommend_transformations(sample_findings)

        log_recs = [r for r in recs if "log" in r.transform_type.lower() and r.column_name == "monthly_charges"]
        assert len(log_recs) > 0

    def test_transform_recommendation_dataclass(self):
        rec = TransformRecommendation(
            column_name="monthly_charges",
            transform_type="log_transform",
            reason="High skewness (2.5)",
            parameters={"base": "natural"},
            priority="high"
        )
        assert rec.column_name == "monthly_charges"
        assert rec.parameters["base"] == "natural"


class TestRecommendationEngineIntegration:
    def test_get_all_recommendations(self, sample_findings):
        engine = RecommendationEngine()

        target_rec = engine.recommend_target(sample_findings)
        feature_recs = engine.recommend_features(sample_findings)
        cleaning_recs = engine.recommend_cleaning(sample_findings)
        transform_recs = engine.recommend_transformations(sample_findings)

        assert target_rec is not None
        assert len(feature_recs) > 0
        assert len(cleaning_recs) >= 0
        assert len(transform_recs) > 0

    def test_generate_summary(self, sample_findings):
        engine = RecommendationEngine()
        summary = engine.generate_summary(sample_findings)

        assert isinstance(summary, dict)
        assert "target" in summary
        assert "features" in summary
        assert "cleaning" in summary
        assert "transformations" in summary

    def test_to_markdown(self, sample_findings):
        engine = RecommendationEngine()
        markdown = engine.to_markdown(sample_findings)

        assert isinstance(markdown, str)
        assert "# Recommendations" in markdown or "##" in markdown
        assert len(markdown) > 100
