import pandas as pd
import pytest

from customer_retention.analysis.recommendations.base import Platform, RecommendationResult
from customer_retention.analysis.recommendations.cleaning.outlier import OutlierCapRecommendation


class TestOutlierCapRecommendationInit:
    def test_default_percentile(self):
        rec = OutlierCapRecommendation(columns=["a"], rationale="test")
        assert rec.percentile == 99

    def test_custom_percentile(self):
        rec = OutlierCapRecommendation(columns=["a"], percentile=95, rationale="test")
        assert rec.percentile == 95

    def test_recommendation_type(self):
        rec = OutlierCapRecommendation(columns=["a"], percentile=99, rationale="test")
        assert rec.recommendation_type == "cap_outliers_99"

    def test_category_is_cleaning(self):
        rec = OutlierCapRecommendation(columns=["a"], rationale="test")
        assert rec.category == "cleaning"


class TestOutlierCapRecommendationFit:
    @pytest.fixture
    def df_with_outliers(self):
        return pd.DataFrame({"a": [1, 2, 3, 4, 5, 100]})

    def test_fit_computes_bounds(self, df_with_outliers):
        rec = OutlierCapRecommendation(columns=["a"], percentile=95, rationale="test")
        rec.fit(df_with_outliers)
        assert "a" in rec._bounds
        assert "lower" in rec._bounds["a"]
        assert "upper" in rec._bounds["a"]

    def test_fit_stores_in_fit_params(self, df_with_outliers):
        rec = OutlierCapRecommendation(columns=["a"], percentile=95, rationale="test")
        rec.fit(df_with_outliers)
        assert "bounds" in rec._fit_params


class TestOutlierCapRecommendationTransform:
    @pytest.fixture
    def df_with_outliers(self):
        return pd.DataFrame({"a": [1, 2, 3, 4, 5, 100, -50]})

    def test_transform_caps_outliers(self, df_with_outliers):
        rec = OutlierCapRecommendation(columns=["a"], percentile=90, rationale="test")
        rec.fit(df_with_outliers)
        result = rec.transform(df_with_outliers)
        assert result.data["a"].max() < 100
        assert result.data["a"].min() > -50

    def test_transform_returns_recommendation_result(self, df_with_outliers):
        rec = OutlierCapRecommendation(columns=["a"], percentile=95, rationale="test")
        result = rec.fit_transform(df_with_outliers)
        assert isinstance(result, RecommendationResult)
        assert result.columns_affected == ["a"]

    def test_transform_metadata_contains_outliers_capped(self, df_with_outliers):
        rec = OutlierCapRecommendation(columns=["a"], percentile=90, rationale="test")
        result = rec.fit_transform(df_with_outliers)
        assert "outliers_capped" in result.metadata

    def test_transform_does_not_modify_original(self, df_with_outliers):
        rec = OutlierCapRecommendation(columns=["a"], percentile=90, rationale="test")
        original_max = df_with_outliers["a"].max()
        rec.fit_transform(df_with_outliers)
        assert df_with_outliers["a"].max() == original_max


class TestOutlierCapRecommendationDatabricks:
    @pytest.fixture
    def df_with_outliers(self):
        return pd.DataFrame({"a": [1, 2, 3, 100]})

    def test_databricks_falls_back_to_local(self, df_with_outliers):
        rec = OutlierCapRecommendation(columns=["a"], percentile=90, rationale="test")
        rec.fit(df_with_outliers)
        result = rec.transform(df_with_outliers, Platform.DATABRICKS)
        assert result.data["a"].max() < 100


class TestOutlierCapRecommendationCodeGeneration:
    def test_generate_local_code(self):
        rec = OutlierCapRecommendation(columns=["a"], percentile=99, rationale="cap outliers")
        rec._bounds = {"a": {"lower": 0, "upper": 100}}
        code = rec.generate_code(Platform.LOCAL)
        assert "clip" in code

    def test_generate_databricks_code(self):
        rec = OutlierCapRecommendation(columns=["a"], percentile=99, rationale="cap outliers")
        rec._bounds = {"a": {"lower": 0, "upper": 100}}
        code = rec.generate_code(Platform.DATABRICKS)
        assert "clip" in code.lower() or "when" in code


class TestOutlierCapRecommendationEdgeCases:
    def test_column_not_in_dataframe(self):
        df = pd.DataFrame({"b": [1, 2, 3]})
        rec = OutlierCapRecommendation(columns=["a"], percentile=99, rationale="test")
        rec.fit(df)
        result = rec.transform(df)
        assert "b" in result.data.columns

    def test_values_within_bounds_unchanged(self):
        df = pd.DataFrame({"a": list(range(1, 101))})
        rec = OutlierCapRecommendation(columns=["a"], percentile=99, rationale="test")
        result = rec.fit_transform(df)
        assert result.data["a"].iloc[50] == 51

    def test_multiple_columns(self):
        df = pd.DataFrame({"a": [1, 2, 100], "b": [10, 20, 1000]})
        rec = OutlierCapRecommendation(columns=["a", "b"], percentile=90, rationale="test")
        result = rec.fit_transform(df)
        assert result.data["a"].max() < 100
        assert result.data["b"].max() < 1000
