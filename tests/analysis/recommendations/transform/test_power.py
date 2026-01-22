import pytest
import pandas as pd
import numpy as np

from customer_retention.analysis.recommendations.base import Platform, RecommendationResult
from customer_retention.analysis.recommendations.transform.power import LogTransformRecommendation, SqrtTransformRecommendation


class TestLogTransformRecommendationInit:
    def test_default_rationale(self):
        rec = LogTransformRecommendation(columns=["a"])
        assert "log" in rec.rationale.lower()

    def test_recommendation_type(self):
        rec = LogTransformRecommendation(columns=["a"])
        assert rec.recommendation_type == "log_transform"

    def test_category_is_transform(self):
        rec = LogTransformRecommendation(columns=["a"])
        assert rec.category == "transform"


class TestLogTransformRecommendationFit:
    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({"a": [1.0, 10.0, 100.0, 1000.0]})

    def test_fit_stores_columns(self, sample_df):
        rec = LogTransformRecommendation(columns=["a"])
        rec.fit(sample_df)
        assert rec._is_fitted


class TestLogTransformRecommendationTransform:
    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({"a": [1.0, 10.0, 100.0]})

    def test_transform_applies_log1p(self, sample_df):
        rec = LogTransformRecommendation(columns=["a"])
        result = rec.fit_transform(sample_df)
        expected = np.log1p(sample_df["a"])
        np.testing.assert_array_almost_equal(result.data["a"], expected)

    def test_transform_returns_recommendation_result(self, sample_df):
        rec = LogTransformRecommendation(columns=["a"])
        result = rec.fit_transform(sample_df)
        assert isinstance(result, RecommendationResult)
        assert result.columns_affected == ["a"]

    def test_transform_handles_zeros(self):
        df = pd.DataFrame({"a": [0.0, 1.0, 10.0]})
        rec = LogTransformRecommendation(columns=["a"])
        result = rec.fit_transform(df)
        assert result.data["a"].iloc[0] == 0.0

    def test_transform_does_not_modify_original(self, sample_df):
        rec = LogTransformRecommendation(columns=["a"])
        original_values = sample_df["a"].copy()
        rec.fit_transform(sample_df)
        pd.testing.assert_series_equal(sample_df["a"], original_values)


class TestLogTransformRecommendationDatabricks:
    def test_databricks_falls_back_to_local(self):
        df = pd.DataFrame({"a": [1.0, 10.0, 100.0]})
        rec = LogTransformRecommendation(columns=["a"])
        rec.fit(df)
        result = rec.transform(df, Platform.DATABRICKS)
        assert isinstance(result, RecommendationResult)


class TestLogTransformRecommendationCodeGeneration:
    def test_generate_local_code(self):
        rec = LogTransformRecommendation(columns=["a"])
        code = rec.generate_code(Platform.LOCAL)
        assert "log1p" in code

    def test_generate_databricks_code(self):
        rec = LogTransformRecommendation(columns=["a"])
        code = rec.generate_code(Platform.DATABRICKS)
        assert "log1p" in code


class TestLogTransformRecommendationEdgeCases:
    def test_column_not_in_dataframe(self):
        df = pd.DataFrame({"b": [1, 2, 3]})
        rec = LogTransformRecommendation(columns=["a"])
        rec.fit(df)
        result = rec.transform(df)
        assert "b" in result.data.columns

    def test_multiple_columns(self):
        df = pd.DataFrame({"a": [1.0, 10.0], "b": [100.0, 1000.0]})
        rec = LogTransformRecommendation(columns=["a", "b"])
        result = rec.fit_transform(df)
        assert result.data["a"].iloc[0] == np.log1p(1.0)
        assert result.data["b"].iloc[0] == np.log1p(100.0)


class TestSqrtTransformRecommendationInit:
    def test_default_rationale(self):
        rec = SqrtTransformRecommendation(columns=["a"])
        assert "sqrt" in rec.rationale.lower()

    def test_recommendation_type(self):
        rec = SqrtTransformRecommendation(columns=["a"])
        assert rec.recommendation_type == "sqrt_transform"

    def test_category_is_transform(self):
        rec = SqrtTransformRecommendation(columns=["a"])
        assert rec.category == "transform"


class TestSqrtTransformRecommendationFit:
    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({"a": [1.0, 4.0, 9.0, 16.0]})

    def test_fit_stores_columns(self, sample_df):
        rec = SqrtTransformRecommendation(columns=["a"])
        rec.fit(sample_df)
        assert rec._is_fitted


class TestSqrtTransformRecommendationTransform:
    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({"a": [1.0, 4.0, 9.0]})

    def test_transform_applies_sqrt(self, sample_df):
        rec = SqrtTransformRecommendation(columns=["a"])
        result = rec.fit_transform(sample_df)
        expected = np.sqrt(sample_df["a"])
        np.testing.assert_array_almost_equal(result.data["a"], expected)

    def test_transform_returns_recommendation_result(self, sample_df):
        rec = SqrtTransformRecommendation(columns=["a"])
        result = rec.fit_transform(sample_df)
        assert isinstance(result, RecommendationResult)
        assert result.columns_affected == ["a"]

    def test_transform_handles_zeros(self):
        df = pd.DataFrame({"a": [0.0, 1.0, 4.0]})
        rec = SqrtTransformRecommendation(columns=["a"])
        result = rec.fit_transform(df)
        assert result.data["a"].iloc[0] == 0.0

    def test_transform_does_not_modify_original(self, sample_df):
        rec = SqrtTransformRecommendation(columns=["a"])
        original_values = sample_df["a"].copy()
        rec.fit_transform(sample_df)
        pd.testing.assert_series_equal(sample_df["a"], original_values)


class TestSqrtTransformRecommendationDatabricks:
    def test_databricks_falls_back_to_local(self):
        df = pd.DataFrame({"a": [1.0, 4.0, 9.0]})
        rec = SqrtTransformRecommendation(columns=["a"])
        rec.fit(df)
        result = rec.transform(df, Platform.DATABRICKS)
        assert isinstance(result, RecommendationResult)


class TestSqrtTransformRecommendationCodeGeneration:
    def test_generate_local_code(self):
        rec = SqrtTransformRecommendation(columns=["a"])
        code = rec.generate_code(Platform.LOCAL)
        assert "sqrt" in code

    def test_generate_databricks_code(self):
        rec = SqrtTransformRecommendation(columns=["a"])
        code = rec.generate_code(Platform.DATABRICKS)
        assert "sqrt" in code


class TestSqrtTransformRecommendationEdgeCases:
    def test_column_not_in_dataframe(self):
        df = pd.DataFrame({"b": [1, 2, 3]})
        rec = SqrtTransformRecommendation(columns=["a"])
        rec.fit(df)
        result = rec.transform(df)
        assert "b" in result.data.columns

    def test_multiple_columns(self):
        df = pd.DataFrame({"a": [1.0, 4.0], "b": [9.0, 16.0]})
        rec = SqrtTransformRecommendation(columns=["a", "b"])
        result = rec.fit_transform(df)
        assert result.data["a"].iloc[0] == np.sqrt(1.0)
        assert result.data["b"].iloc[0] == np.sqrt(9.0)
