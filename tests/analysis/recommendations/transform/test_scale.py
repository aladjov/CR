import pytest
import pandas as pd
import numpy as np

from customer_retention.analysis.recommendations.base import Platform, RecommendationResult
from customer_retention.analysis.recommendations.transform.scale import (
    StandardScaleRecommendation,
    MinMaxScaleRecommendation,
)


class TestStandardScaleRecommendationInit:
    def test_default_rationale(self):
        rec = StandardScaleRecommendation(columns=["a"])
        assert "zero mean" in rec.rationale.lower()

    def test_custom_rationale(self):
        rec = StandardScaleRecommendation(columns=["a"], rationale="custom")
        assert rec.rationale == "custom"

    def test_recommendation_type(self):
        rec = StandardScaleRecommendation(columns=["a"])
        assert rec.recommendation_type == "standard_scale"

    def test_category_is_transform(self):
        rec = StandardScaleRecommendation(columns=["a"])
        assert rec.category == "transform"


class TestStandardScaleRecommendationFit:
    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0], "b": [10.0, 20.0, 30.0, 40.0, 50.0]})

    def test_fit_computes_mean_and_std(self, sample_df):
        rec = StandardScaleRecommendation(columns=["a"])
        rec.fit(sample_df)
        assert "a" in rec._means
        assert "a" in rec._stds
        assert rec._means["a"] == 3.0

    def test_fit_multiple_columns(self, sample_df):
        rec = StandardScaleRecommendation(columns=["a", "b"])
        rec.fit(sample_df)
        assert "a" in rec._means
        assert "b" in rec._means

    def test_fit_stores_in_fit_params(self, sample_df):
        rec = StandardScaleRecommendation(columns=["a"])
        rec.fit(sample_df)
        assert "means" in rec._fit_params
        assert "stds" in rec._fit_params


class TestStandardScaleRecommendationTransform:
    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0]})

    def test_transform_produces_zero_mean(self, sample_df):
        rec = StandardScaleRecommendation(columns=["a"])
        result = rec.fit_transform(sample_df)
        assert abs(result.data["a"].mean()) < 1e-10

    def test_transform_produces_unit_variance(self, sample_df):
        rec = StandardScaleRecommendation(columns=["a"])
        result = rec.fit_transform(sample_df)
        assert abs(result.data["a"].std(ddof=0) - 1.0) < 1e-10

    def test_transform_returns_recommendation_result(self, sample_df):
        rec = StandardScaleRecommendation(columns=["a"])
        result = rec.fit_transform(sample_df)
        assert isinstance(result, RecommendationResult)
        assert result.columns_affected == ["a"]

    def test_transform_metadata_contains_statistics(self, sample_df):
        rec = StandardScaleRecommendation(columns=["a"])
        result = rec.fit_transform(sample_df)
        assert "means" in result.metadata
        assert "stds" in result.metadata


class TestStandardScaleRecommendationCodeGeneration:
    def test_generate_local_code(self):
        rec = StandardScaleRecommendation(columns=["a", "b"])
        code = rec.generate_code(Platform.LOCAL)
        assert "StandardScaler" in code

    def test_generate_databricks_code(self):
        rec = StandardScaleRecommendation(columns=["a", "b"])
        code = rec.generate_code(Platform.DATABRICKS)
        assert "StandardScaler" in code


class TestMinMaxScaleRecommendationInit:
    def test_default_range(self):
        rec = MinMaxScaleRecommendation(columns=["a"])
        assert rec.feature_range == (0, 1)

    def test_custom_range(self):
        rec = MinMaxScaleRecommendation(columns=["a"], feature_range=(-1, 1))
        assert rec.feature_range == (-1, 1)

    def test_recommendation_type(self):
        rec = MinMaxScaleRecommendation(columns=["a"])
        assert rec.recommendation_type == "minmax_scale"

    def test_category_is_transform(self):
        rec = MinMaxScaleRecommendation(columns=["a"])
        assert rec.category == "transform"


class TestMinMaxScaleRecommendationFit:
    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0]})

    def test_fit_computes_min_max(self, sample_df):
        rec = MinMaxScaleRecommendation(columns=["a"])
        rec.fit(sample_df)
        assert "a" in rec._mins
        assert "a" in rec._maxs
        assert rec._mins["a"] == 1.0
        assert rec._maxs["a"] == 5.0


class TestMinMaxScaleRecommendationTransform:
    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0]})

    def test_transform_produces_values_in_range(self, sample_df):
        rec = MinMaxScaleRecommendation(columns=["a"])
        result = rec.fit_transform(sample_df)
        assert result.data["a"].min() == 0.0
        assert result.data["a"].max() == 1.0

    def test_transform_with_custom_range(self, sample_df):
        rec = MinMaxScaleRecommendation(columns=["a"], feature_range=(-1, 1))
        result = rec.fit_transform(sample_df)
        assert result.data["a"].min() == -1.0
        assert result.data["a"].max() == 1.0

    def test_transform_returns_recommendation_result(self, sample_df):
        rec = MinMaxScaleRecommendation(columns=["a"])
        result = rec.fit_transform(sample_df)
        assert isinstance(result, RecommendationResult)

    def test_transform_metadata_contains_statistics(self, sample_df):
        rec = MinMaxScaleRecommendation(columns=["a"])
        result = rec.fit_transform(sample_df)
        assert "mins" in result.metadata
        assert "maxs" in result.metadata


class TestMinMaxScaleRecommendationCodeGeneration:
    def test_generate_local_code(self):
        rec = MinMaxScaleRecommendation(columns=["a"])
        code = rec.generate_code(Platform.LOCAL)
        assert "MinMaxScaler" in code

    def test_generate_databricks_code(self):
        rec = MinMaxScaleRecommendation(columns=["a"])
        code = rec.generate_code(Platform.DATABRICKS)
        assert "MinMaxScaler" in code


class TestScaleRecommendationEdgeCases:
    def test_column_not_in_dataframe(self):
        df = pd.DataFrame({"b": [1, 2, 3]})
        rec = StandardScaleRecommendation(columns=["a"])
        rec.fit(df)
        result = rec.transform(df)
        assert "b" in result.data.columns

    def test_constant_column(self):
        df = pd.DataFrame({"a": [5.0, 5.0, 5.0]})
        rec = StandardScaleRecommendation(columns=["a"])
        result = rec.fit_transform(df)
        assert not result.data["a"].isna().any()
