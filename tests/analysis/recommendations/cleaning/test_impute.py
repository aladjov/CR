import pytest
import pandas as pd
import numpy as np

from customer_retention.analysis.recommendations.base import Platform, RecommendationResult
from customer_retention.analysis.recommendations.cleaning.impute import ImputeRecommendation


class TestImputeRecommendationInit:
    def test_default_strategy_is_median(self):
        rec = ImputeRecommendation(columns=["a"], rationale="test")
        assert rec.strategy == "median"

    def test_custom_strategy(self):
        rec = ImputeRecommendation(columns=["a"], strategy="mean", rationale="test")
        assert rec.strategy == "mean"

    def test_constant_strategy_with_fill_value(self):
        rec = ImputeRecommendation(columns=["a"], strategy="constant", fill_value=0, rationale="test")
        assert rec.strategy == "constant"
        assert rec.fill_value == 0

    def test_recommendation_type(self):
        rec = ImputeRecommendation(columns=["a"], strategy="mean", rationale="test")
        assert rec.recommendation_type == "impute_mean"

    def test_category_is_cleaning(self):
        rec = ImputeRecommendation(columns=["a"], rationale="test")
        assert rec.category == "cleaning"


class TestImputeRecommendationFit:
    @pytest.fixture
    def df_with_nulls(self):
        return pd.DataFrame({"a": [1.0, 2.0, None, 4.0, 5.0], "b": [10.0, None, 30.0, None, 50.0]})

    def test_fit_median_computes_median(self, df_with_nulls):
        rec = ImputeRecommendation(columns=["a"], strategy="median", rationale="test")
        rec.fit(df_with_nulls)
        assert rec._impute_values["a"] == 3.0

    def test_fit_mean_computes_mean(self, df_with_nulls):
        rec = ImputeRecommendation(columns=["a"], strategy="mean", rationale="test")
        rec.fit(df_with_nulls)
        assert rec._impute_values["a"] == 3.0

    def test_fit_mode_computes_mode(self):
        df = pd.DataFrame({"a": [1, 1, 2, None, 3]})
        rec = ImputeRecommendation(columns=["a"], strategy="mode", rationale="test")
        rec.fit(df)
        assert rec._impute_values["a"] == 1

    def test_fit_constant_uses_fill_value(self, df_with_nulls):
        rec = ImputeRecommendation(columns=["a"], strategy="constant", fill_value=-1, rationale="test")
        rec.fit(df_with_nulls)
        assert rec._impute_values["a"] == -1

    def test_fit_multiple_columns(self, df_with_nulls):
        rec = ImputeRecommendation(columns=["a", "b"], strategy="median", rationale="test")
        rec.fit(df_with_nulls)
        assert "a" in rec._impute_values
        assert "b" in rec._impute_values

    def test_fit_stores_in_fit_params(self, df_with_nulls):
        rec = ImputeRecommendation(columns=["a"], strategy="median", rationale="test")
        rec.fit(df_with_nulls)
        assert "impute_values" in rec._fit_params


class TestImputeRecommendationTransform:
    @pytest.fixture
    def df_with_nulls(self):
        return pd.DataFrame({"a": [1.0, 2.0, None, 4.0, 5.0]})

    def test_transform_fills_nulls_with_fitted_value(self, df_with_nulls):
        rec = ImputeRecommendation(columns=["a"], strategy="median", rationale="test")
        rec.fit(df_with_nulls)
        result = rec.transform(df_with_nulls)
        assert result.data["a"].isna().sum() == 0
        assert result.data["a"].iloc[2] == 3.0

    def test_transform_returns_recommendation_result(self, df_with_nulls):
        rec = ImputeRecommendation(columns=["a"], strategy="median", rationale="test")
        result = rec.fit_transform(df_with_nulls)
        assert isinstance(result, RecommendationResult)
        assert result.columns_affected == ["a"]
        assert result.rows_before == 5
        assert result.rows_after == 5

    def test_transform_metadata_contains_nulls_imputed(self, df_with_nulls):
        rec = ImputeRecommendation(columns=["a"], strategy="median", rationale="test")
        result = rec.fit_transform(df_with_nulls)
        assert "nulls_imputed" in result.metadata
        assert result.metadata["nulls_imputed"]["a"] == 1

    def test_transform_does_not_modify_original(self, df_with_nulls):
        rec = ImputeRecommendation(columns=["a"], strategy="median", rationale="test")
        original_null_count = df_with_nulls["a"].isna().sum()
        rec.fit_transform(df_with_nulls)
        assert df_with_nulls["a"].isna().sum() == original_null_count

    def test_transform_with_constant_zero(self):
        df = pd.DataFrame({"a": [1.0, None, 3.0]})
        rec = ImputeRecommendation(columns=["a"], strategy="constant", fill_value=0, rationale="test")
        result = rec.fit_transform(df)
        assert result.data["a"].iloc[1] == 0


class TestImputeRecommendationDatabricks:
    @pytest.fixture
    def df_with_nulls(self):
        return pd.DataFrame({"a": [1.0, None, 3.0]})

    def test_databricks_falls_back_to_local(self, df_with_nulls):
        rec = ImputeRecommendation(columns=["a"], strategy="median", rationale="test")
        rec.fit(df_with_nulls)
        result = rec.transform(df_with_nulls, Platform.DATABRICKS)
        assert result.data["a"].isna().sum() == 0


class TestImputeRecommendationCodeGeneration:
    def test_generate_local_code(self):
        rec = ImputeRecommendation(columns=["a"], strategy="median", rationale="test imputation")
        rec._impute_values = {"a": 5.0}
        code = rec.generate_code(Platform.LOCAL)
        assert "fillna" in code
        assert "5.0" in code

    def test_generate_databricks_code(self):
        rec = ImputeRecommendation(columns=["a"], strategy="median", rationale="test imputation")
        rec._impute_values = {"a": 5.0}
        code = rec.generate_code(Platform.DATABRICKS)
        assert "fillna" in code


class TestImputeRecommendationStrategies:
    def test_median_strategy(self):
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5, None]})
        rec = ImputeRecommendation(columns=["a"], strategy="median", rationale="test")
        result = rec.fit_transform(df)
        assert result.data["a"].iloc[5] == 3.0

    def test_mean_strategy(self):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, None]})
        rec = ImputeRecommendation(columns=["a"], strategy="mean", rationale="test")
        result = rec.fit_transform(df)
        assert result.data["a"].iloc[4] == 2.5

    def test_mode_strategy(self):
        df = pd.DataFrame({"a": ["x", "x", "y", None]})
        rec = ImputeRecommendation(columns=["a"], strategy="mode", rationale="test")
        result = rec.fit_transform(df)
        assert result.data["a"].iloc[3] == "x"

    def test_constant_strategy(self):
        df = pd.DataFrame({"a": [1, 2, None]})
        rec = ImputeRecommendation(columns=["a"], strategy="constant", fill_value=999, rationale="test")
        result = rec.fit_transform(df)
        assert result.data["a"].iloc[2] == 999


class TestImputeRecommendationEdgeCases:
    def test_column_not_in_dataframe(self):
        df = pd.DataFrame({"b": [1, 2, 3]})
        rec = ImputeRecommendation(columns=["a"], strategy="median", rationale="test")
        rec.fit(df)
        result = rec.transform(df)
        assert "b" in result.data.columns

    def test_no_nulls_to_impute(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        rec = ImputeRecommendation(columns=["a"], strategy="median", rationale="test")
        result = rec.fit_transform(df)
        assert result.metadata["nulls_imputed"]["a"] == 0

    def test_all_nulls_uses_fill_value(self):
        df = pd.DataFrame({"a": [None, None, None]})
        rec = ImputeRecommendation(columns=["a"], strategy="constant", fill_value=0, rationale="test")
        result = rec.fit_transform(df)
        assert list(result.data["a"]) == [0, 0, 0]
