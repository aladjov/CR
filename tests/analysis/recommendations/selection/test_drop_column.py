import pytest
import pandas as pd

from customer_retention.analysis.recommendations.base import Platform, RecommendationResult
from customer_retention.analysis.recommendations.selection.drop_column import DropColumnRecommendation


class TestDropColumnRecommendationInit:
    def test_default_reason_is_not_specified(self):
        rec = DropColumnRecommendation(columns=["a"], rationale="test")
        assert rec.reason == "not_specified"

    def test_custom_reason(self):
        rec = DropColumnRecommendation(columns=["a"], reason="multicollinear", rationale="test")
        assert rec.reason == "multicollinear"

    def test_recommendation_type(self):
        rec = DropColumnRecommendation(columns=["a"], reason="weak", rationale="test")
        assert rec.recommendation_type == "drop_weak"

    def test_category_is_feature_selection(self):
        rec = DropColumnRecommendation(columns=["a"], rationale="test")
        assert rec.category == "feature_selection"


class TestDropColumnRecommendationFit:
    @pytest.fixture
    def df(self):
        return pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})

    def test_fit_stores_columns_to_drop(self, df):
        rec = DropColumnRecommendation(columns=["a", "b"], rationale="test")
        rec.fit(df)
        assert rec._columns_to_drop == ["a", "b"]

    def test_fit_ignores_missing_columns(self, df):
        rec = DropColumnRecommendation(columns=["a", "missing"], rationale="test")
        rec.fit(df)
        assert rec._columns_to_drop == ["a"]


class TestDropColumnRecommendationTransform:
    @pytest.fixture
    def df(self):
        return pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})

    def test_drops_specified_columns(self, df):
        rec = DropColumnRecommendation(columns=["a"], rationale="test")
        result = rec.fit_transform(df)
        assert "a" not in result.data.columns
        assert "b" in result.data.columns
        assert "c" in result.data.columns

    def test_drops_multiple_columns(self, df):
        rec = DropColumnRecommendation(columns=["a", "b"], rationale="test")
        result = rec.fit_transform(df)
        assert list(result.data.columns) == ["c"]

    def test_returns_recommendation_result(self, df):
        rec = DropColumnRecommendation(columns=["a"], rationale="test")
        result = rec.fit_transform(df)
        assert isinstance(result, RecommendationResult)
        assert result.rows_before == 3
        assert result.rows_after == 3

    def test_result_metadata_contains_dropped_columns(self, df):
        rec = DropColumnRecommendation(columns=["a", "b"], rationale="test")
        result = rec.fit_transform(df)
        assert result.metadata["dropped_columns"] == ["a", "b"]

    def test_transform_does_not_modify_original(self, df):
        rec = DropColumnRecommendation(columns=["a"], rationale="test")
        original_cols = list(df.columns)
        rec.fit_transform(df)
        assert list(df.columns) == original_cols


class TestDropColumnRecommendationCodeGeneration:
    def test_generate_local_code_single_column(self):
        rec = DropColumnRecommendation(columns=["a"], rationale="multicollinear")
        rec._columns_to_drop = ["a"]
        rec._is_fitted = True
        code = rec.generate_code(Platform.LOCAL)
        assert "drop" in code
        assert "'a'" in code

    def test_generate_local_code_multiple_columns(self):
        rec = DropColumnRecommendation(columns=["a", "b"], rationale="weak")
        rec._columns_to_drop = ["a", "b"]
        rec._is_fitted = True
        code = rec.generate_code(Platform.LOCAL)
        assert "'a'" in code
        assert "'b'" in code

    def test_generate_databricks_code(self):
        rec = DropColumnRecommendation(columns=["a"], rationale="test")
        rec._columns_to_drop = ["a"]
        rec._is_fitted = True
        code = rec.generate_code(Platform.DATABRICKS)
        assert "drop" in code


class TestDropColumnRecommendationEdgeCases:
    def test_no_columns_to_drop(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        rec = DropColumnRecommendation(columns=["missing"], rationale="test")
        result = rec.fit_transform(df)
        assert list(result.data.columns) == ["a"]
        assert result.metadata["dropped_columns"] == []

    def test_empty_dataframe(self):
        df = pd.DataFrame({"a": [], "b": []})
        rec = DropColumnRecommendation(columns=["a"], rationale="test")
        result = rec.fit_transform(df)
        assert "a" not in result.data.columns
