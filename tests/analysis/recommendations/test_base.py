
import pandas as pd
import pytest

from customer_retention.analysis.recommendations.base import (
    BaseRecommendation,
    CleaningRecommendation,
    DatetimeRecommendation,
    EncodingRecommendation,
    Platform,
    RecommendationResult,
    TransformRecommendation,
)


class TestPlatformEnum:
    def test_local_value(self):
        assert Platform.LOCAL.value == "local"

    def test_databricks_value(self):
        assert Platform.DATABRICKS.value == "databricks"


class TestRecommendationResult:
    def test_creation_with_required_fields(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        result = RecommendationResult(
            data=df, columns_affected=["a"], rows_before=3, rows_after=3
        )
        assert len(result.data) == 3
        assert result.columns_affected == ["a"]
        assert result.rows_before == 3
        assert result.rows_after == 3
        assert result.metadata == {}
        assert result.warnings == []

    def test_creation_with_optional_fields(self):
        df = pd.DataFrame({"a": [1, 2]})
        result = RecommendationResult(
            data=df, columns_affected=["a"], rows_before=3, rows_after=2,
            metadata={"nulls": 1}, warnings=["dropped row"]
        )
        assert result.metadata == {"nulls": 1}
        assert result.warnings == ["dropped row"]


class ConcreteRecommendation(BaseRecommendation):
    @property
    def recommendation_type(self) -> str:
        return "test_recommendation"

    @property
    def category(self) -> str:
        return "test"

    def _fit_impl(self, df: pd.DataFrame) -> None:
        self._fit_params["test_param"] = df.shape[0]

    def _transform_local(self, df: pd.DataFrame) -> RecommendationResult:
        return RecommendationResult(
            data=df.copy(), columns_affected=self.columns,
            rows_before=len(df), rows_after=len(df), metadata={"local": True}
        )

    def _transform_databricks(self, df: pd.DataFrame) -> RecommendationResult:
        return RecommendationResult(
            data=df.copy(), columns_affected=self.columns,
            rows_before=len(df), rows_after=len(df), metadata={"databricks": True}
        )

    def _generate_local_code(self) -> str:
        return f"# local code for {self.columns}"

    def _generate_databricks_code(self) -> str:
        return f"# databricks code for {self.columns}"


class TestBaseRecommendation:
    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})

    @pytest.fixture
    def recommendation(self):
        return ConcreteRecommendation(
            columns=["col1"], rationale="Test rationale", evidence=["ev1", "ev2"]
        )

    def test_initialization_with_all_params(self):
        rec = ConcreteRecommendation(
            columns=["a", "b"], rationale="reason", evidence=["e1"],
            priority="high", source_finding=None
        )
        assert rec.columns == ["a", "b"]
        assert rec.rationale == "reason"
        assert rec.evidence == ["e1"]
        assert rec.priority == "high"
        assert rec._is_fitted is False
        assert rec._fit_params == {}

    def test_initialization_defaults(self):
        rec = ConcreteRecommendation(columns=["x"], rationale="test")
        assert rec.evidence == []
        assert rec.priority == "medium"
        assert rec.source_finding is None

    def test_recommendation_type_property(self, recommendation):
        assert recommendation.recommendation_type == "test_recommendation"

    def test_category_property(self, recommendation):
        assert recommendation.category == "test"

    def test_fit_returns_self(self, recommendation, sample_df):
        result = recommendation.fit(sample_df)
        assert result is recommendation

    def test_fit_sets_is_fitted(self, recommendation, sample_df):
        assert recommendation._is_fitted is False
        recommendation.fit(sample_df)
        assert recommendation._is_fitted is True

    def test_fit_populates_fit_params(self, recommendation, sample_df):
        recommendation.fit(sample_df)
        assert "test_param" in recommendation._fit_params

    def test_transform_raises_when_not_fitted(self, recommendation, sample_df):
        with pytest.raises(ValueError, match="not fitted"):
            recommendation.transform(sample_df)

    def test_transform_local_default(self, recommendation, sample_df):
        recommendation.fit(sample_df)
        result = recommendation.transform(sample_df)
        assert result.metadata.get("local") is True

    def test_transform_with_local_platform(self, recommendation, sample_df):
        recommendation.fit(sample_df)
        result = recommendation.transform(sample_df, Platform.LOCAL)
        assert result.metadata.get("local") is True

    def test_transform_with_databricks_platform(self, recommendation, sample_df):
        recommendation.fit(sample_df)
        result = recommendation.transform(sample_df, Platform.DATABRICKS)
        assert result.metadata.get("databricks") is True

    def test_fit_transform(self, recommendation, sample_df):
        result = recommendation.fit_transform(sample_df)
        assert recommendation._is_fitted is True
        assert isinstance(result, RecommendationResult)

    def test_generate_code_local(self, recommendation):
        code = recommendation.generate_code(Platform.LOCAL)
        assert "local code" in code

    def test_generate_code_databricks(self, recommendation):
        code = recommendation.generate_code(Platform.DATABRICKS)
        assert "databricks code" in code

    def test_to_dict(self, recommendation, sample_df):
        recommendation.fit(sample_df)
        d = recommendation.to_dict()
        assert d["type"] == "test_recommendation"
        assert d["category"] == "test"
        assert d["columns"] == ["col1"]
        assert d["rationale"] == "Test rationale"
        assert d["evidence"] == ["ev1", "ev2"]
        assert d["priority"] == "medium"
        assert d["is_fitted"] is True
        assert "test_param" in d["fit_params"]

    def test_describe(self, recommendation):
        desc = recommendation.describe()
        assert "test_recommendation" in desc
        assert "col1" in desc
        assert "Test rationale" in desc


class ConcreteCleaningRecommendation(CleaningRecommendation):
    @property
    def recommendation_type(self) -> str:
        return "clean_test"

    def _fit_impl(self, df: pd.DataFrame) -> None:
        pass

    def _transform_local(self, df: pd.DataFrame) -> RecommendationResult:
        return RecommendationResult(data=df.copy(), columns_affected=self.columns, rows_before=len(df), rows_after=len(df))

    def _transform_databricks(self, df: pd.DataFrame) -> RecommendationResult:
        return self._transform_local(df)

    def _generate_local_code(self) -> str:
        return ""

    def _generate_databricks_code(self) -> str:
        return ""


class ConcreteTransformRecommendation(TransformRecommendation):
    @property
    def recommendation_type(self) -> str:
        return "transform_test"

    def _fit_impl(self, df: pd.DataFrame) -> None:
        pass

    def _transform_local(self, df: pd.DataFrame) -> RecommendationResult:
        return RecommendationResult(data=df.copy(), columns_affected=self.columns, rows_before=len(df), rows_after=len(df))

    def _transform_databricks(self, df: pd.DataFrame) -> RecommendationResult:
        return self._transform_local(df)

    def _generate_local_code(self) -> str:
        return ""

    def _generate_databricks_code(self) -> str:
        return ""


class ConcreteEncodingRecommendation(EncodingRecommendation):
    @property
    def recommendation_type(self) -> str:
        return "encode_test"

    def _fit_impl(self, df: pd.DataFrame) -> None:
        pass

    def _transform_local(self, df: pd.DataFrame) -> RecommendationResult:
        return RecommendationResult(data=df.copy(), columns_affected=self.columns, rows_before=len(df), rows_after=len(df))

    def _transform_databricks(self, df: pd.DataFrame) -> RecommendationResult:
        return self._transform_local(df)

    def _generate_local_code(self) -> str:
        return ""

    def _generate_databricks_code(self) -> str:
        return ""


class ConcreteDatetimeRecommendation(DatetimeRecommendation):
    @property
    def recommendation_type(self) -> str:
        return "datetime_test"

    def _fit_impl(self, df: pd.DataFrame) -> None:
        pass

    def _transform_local(self, df: pd.DataFrame) -> RecommendationResult:
        return RecommendationResult(data=df.copy(), columns_affected=self.columns, rows_before=len(df), rows_after=len(df))

    def _transform_databricks(self, df: pd.DataFrame) -> RecommendationResult:
        return self._transform_local(df)

    def _generate_local_code(self) -> str:
        return ""

    def _generate_databricks_code(self) -> str:
        return ""


class TestCategoryRecommendations:
    def test_cleaning_recommendation_category(self):
        rec = ConcreteCleaningRecommendation(columns=["x"], rationale="clean")
        assert rec.category == "cleaning"

    def test_transform_recommendation_category(self):
        rec = ConcreteTransformRecommendation(columns=["x"], rationale="transform")
        assert rec.category == "transform"

    def test_encoding_recommendation_category(self):
        rec = ConcreteEncodingRecommendation(columns=["x"], rationale="encode")
        assert rec.category == "encoding"

    def test_datetime_recommendation_category(self):
        rec = ConcreteDatetimeRecommendation(columns=["x"], rationale="datetime")
        assert rec.category == "datetime"


class TestMLflowIntegration:
    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({"col1": [1, 2, 3]})

    def test_transform_with_mlflow_adapter(self, sample_df):
        class MockMLflowAdapter:
            def __init__(self):
                self.logged_params = {}
                self.logged_metrics = {}

            def log_params(self, params):
                self.logged_params.update(params)

            def log_metrics(self, metrics):
                self.logged_metrics.update(metrics)

        rec = ConcreteRecommendation(columns=["col1"], rationale="test")
        rec.fit(sample_df)
        adapter = MockMLflowAdapter()
        result = rec.transform(sample_df, Platform.LOCAL, mlflow_adapter=adapter)
        assert "test_param" in adapter.logged_params
        assert isinstance(result, RecommendationResult)

    def test_transform_without_mlflow_adapter(self, sample_df):
        rec = ConcreteRecommendation(columns=["col1"], rationale="test")
        result = rec.fit_transform(sample_df)
        assert isinstance(result, RecommendationResult)


class TestDatabricksFallback:
    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({"col1": [1, 2, 3]})

    def test_databricks_falls_back_to_local_when_no_spark(self, sample_df):
        class FallbackRecommendation(BaseRecommendation):
            @property
            def recommendation_type(self) -> str:
                return "fallback_test"

            @property
            def category(self) -> str:
                return "test"

            def _fit_impl(self, df: pd.DataFrame) -> None:
                pass

            def _transform_local(self, df: pd.DataFrame) -> RecommendationResult:
                return RecommendationResult(
                    data=df.copy(), columns_affected=self.columns,
                    rows_before=len(df), rows_after=len(df), metadata={"used_local": True}
                )

            def _transform_databricks(self, df: pd.DataFrame) -> RecommendationResult:
                from customer_retention.core.compat import is_spark_available
                if not is_spark_available():
                    return self._transform_local(df)
                return RecommendationResult(
                    data=df.copy(), columns_affected=self.columns,
                    rows_before=len(df), rows_after=len(df), metadata={"used_spark": True}
                )

            def _generate_local_code(self) -> str:
                return ""

            def _generate_databricks_code(self) -> str:
                return ""

        rec = FallbackRecommendation(columns=["col1"], rationale="test")
        rec.fit(sample_df)
        result = rec.transform(sample_df, Platform.DATABRICKS)
        assert result.metadata.get("used_local") is True


class TestToFeatureDefinition:
    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({"col1": [1, 2, 3]})

    def test_returns_feature_definition(self, sample_df):
        from customer_retention.stages.features.feature_definitions import FeatureDefinition
        rec = ConcreteRecommendation(columns=["col1"], rationale="Test rationale")
        rec.fit(sample_df)
        feature = rec.to_feature_definition()
        assert isinstance(feature, FeatureDefinition)

    def test_feature_name_format(self, sample_df):
        rec = ConcreteRecommendation(columns=["col1"], rationale="Test rationale")
        rec.fit(sample_df)
        feature = rec.to_feature_definition()
        assert feature.name == "col1_test_recommendation"

    def test_feature_description_is_rationale(self, sample_df):
        rec = ConcreteRecommendation(columns=["col1"], rationale="My detailed rationale")
        rec.fit(sample_df)
        feature = rec.to_feature_definition()
        assert feature.description == "My detailed rationale"

    def test_feature_source_columns(self, sample_df):
        rec = ConcreteRecommendation(columns=["col1", "col2"], rationale="Test")
        rec.fit(sample_df)
        feature = rec.to_feature_definition()
        assert feature.source_columns == ["col1", "col2"]

    def test_feature_derivation_contains_code(self, sample_df):
        rec = ConcreteRecommendation(columns=["col1"], rationale="Test")
        rec.fit(sample_df)
        feature = rec.to_feature_definition()
        assert "local code" in feature.derivation

    def test_feature_business_meaning_is_rationale(self, sample_df):
        rec = ConcreteRecommendation(columns=["col1"], rationale="Business meaning here")
        rec.fit(sample_df)
        feature = rec.to_feature_definition()
        assert feature.business_meaning == "Business meaning here"

    def test_cleaning_recommendation_category(self, sample_df):
        from customer_retention.stages.features.feature_definitions import FeatureCategory
        rec = ConcreteCleaningRecommendation(columns=["col1"], rationale="Clean")
        rec.fit(sample_df)
        feature = rec.to_feature_definition()
        assert feature.category == FeatureCategory.AGGREGATE

    def test_datetime_recommendation_category(self, sample_df):
        from customer_retention.stages.features.feature_definitions import FeatureCategory
        rec = ConcreteDatetimeRecommendation(columns=["col1"], rationale="Datetime")
        rec.fit(sample_df)
        feature = rec.to_feature_definition()
        assert feature.category == FeatureCategory.TEMPORAL

    def test_feature_has_low_leakage_risk(self, sample_df):
        from customer_retention.stages.features.feature_definitions import LeakageRisk
        rec = ConcreteRecommendation(columns=["col1"], rationale="Test")
        rec.fit(sample_df)
        feature = rec.to_feature_definition()
        assert feature.leakage_risk == LeakageRisk.LOW

    def test_feature_data_type(self, sample_df):
        rec = ConcreteRecommendation(columns=["col1"], rationale="Test")
        rec.fit(sample_df)
        feature = rec.to_feature_definition()
        assert feature.data_type == "float64"

    def test_multiple_columns_uses_first_for_name(self, sample_df):
        rec = ConcreteRecommendation(columns=["col1", "col2"], rationale="Test")
        rec.fit(sample_df)
        feature = rec.to_feature_definition()
        assert feature.name.startswith("col1_")

    def test_transform_recommendation_category(self, sample_df):
        from customer_retention.stages.features.feature_definitions import FeatureCategory
        rec = ConcreteTransformRecommendation(columns=["col1"], rationale="Transform")
        rec.fit(sample_df)
        feature = rec.to_feature_definition()
        assert feature.category == FeatureCategory.AGGREGATE

    def test_encoding_recommendation_category(self, sample_df):
        from customer_retention.stages.features.feature_definitions import FeatureCategory
        rec = ConcreteEncodingRecommendation(columns=["col1"], rationale="Encode")
        rec.fit(sample_df)
        feature = rec.to_feature_definition()
        assert feature.category == FeatureCategory.AGGREGATE
