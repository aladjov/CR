import numpy as np
import pandas as pd
import pytest

from customer_retention.analysis.recommendations.base import Platform, RecommendationResult
from customer_retention.analysis.recommendations.encoding.categorical import (
    LabelEncodeRecommendation,
    OneHotEncodeRecommendation,
)


class TestOneHotEncodeRecommendationInit:
    def test_default_rationale(self):
        rec = OneHotEncodeRecommendation(columns=["a"])
        assert "one-hot" in rec.rationale.lower()

    def test_recommendation_type(self):
        rec = OneHotEncodeRecommendation(columns=["a"])
        assert rec.recommendation_type == "onehot_encode"

    def test_category_is_encoding(self):
        rec = OneHotEncodeRecommendation(columns=["a"])
        assert rec.category == "encoding"


class TestOneHotEncodeRecommendationFit:
    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({"color": ["red", "blue", "green", "red", "blue"]})

    def test_fit_stores_categories(self, sample_df):
        rec = OneHotEncodeRecommendation(columns=["color"])
        rec.fit(sample_df)
        assert "color" in rec._categories

    def test_fit_stores_in_fit_params(self, sample_df):
        rec = OneHotEncodeRecommendation(columns=["color"])
        rec.fit(sample_df)
        assert "categories" in rec._fit_params


class TestOneHotEncodeRecommendationTransform:
    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({"color": ["red", "blue", "green"]})

    def test_transform_creates_dummy_columns(self, sample_df):
        rec = OneHotEncodeRecommendation(columns=["color"])
        result = rec.fit_transform(sample_df)
        assert any("color_" in col or "red" in col or "blue" in col for col in result.data.columns)

    def test_transform_returns_recommendation_result(self, sample_df):
        rec = OneHotEncodeRecommendation(columns=["color"])
        result = rec.fit_transform(sample_df)
        assert isinstance(result, RecommendationResult)

    def test_transform_produces_binary_values(self, sample_df):
        rec = OneHotEncodeRecommendation(columns=["color"])
        result = rec.fit_transform(sample_df)
        encoded_cols = [c for c in result.data.columns if c != "color"]
        for col in encoded_cols:
            assert set(result.data[col].unique()).issubset({0, 1, True, False})


class TestOneHotEncodeRecommendationCodeGeneration:
    def test_generate_local_code(self):
        rec = OneHotEncodeRecommendation(columns=["color"])
        code = rec.generate_code(Platform.LOCAL)
        assert "get_dummies" in code or "OneHotEncoder" in code

    def test_generate_databricks_code(self):
        rec = OneHotEncodeRecommendation(columns=["color"])
        code = rec.generate_code(Platform.DATABRICKS)
        assert "OneHotEncoder" in code or "StringIndexer" in code


class TestLabelEncodeRecommendationInit:
    def test_default_rationale(self):
        rec = LabelEncodeRecommendation(columns=["a"])
        assert "label" in rec.rationale.lower()

    def test_recommendation_type(self):
        rec = LabelEncodeRecommendation(columns=["a"])
        assert rec.recommendation_type == "label_encode"

    def test_category_is_encoding(self):
        rec = LabelEncodeRecommendation(columns=["a"])
        assert rec.category == "encoding"


class TestLabelEncodeRecommendationFit:
    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({"color": ["red", "blue", "green", "red", "blue"]})

    def test_fit_stores_mappings(self, sample_df):
        rec = LabelEncodeRecommendation(columns=["color"])
        rec.fit(sample_df)
        assert "color" in rec._mappings

    def test_fit_stores_in_fit_params(self, sample_df):
        rec = LabelEncodeRecommendation(columns=["color"])
        rec.fit(sample_df)
        assert "mappings" in rec._fit_params


class TestLabelEncodeRecommendationTransform:
    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({"color": ["red", "blue", "green", "red"]})

    def test_transform_produces_integers(self, sample_df):
        rec = LabelEncodeRecommendation(columns=["color"])
        result = rec.fit_transform(sample_df)
        assert result.data["color"].dtype in [np.int64, np.int32, int, "int64", "int32"]

    def test_transform_returns_recommendation_result(self, sample_df):
        rec = LabelEncodeRecommendation(columns=["color"])
        result = rec.fit_transform(sample_df)
        assert isinstance(result, RecommendationResult)

    def test_transform_same_value_gets_same_code(self, sample_df):
        rec = LabelEncodeRecommendation(columns=["color"])
        result = rec.fit_transform(sample_df)
        red_values = result.data.loc[sample_df["color"] == "red", "color"]
        assert red_values.iloc[0] == red_values.iloc[1]

    def test_transform_metadata_contains_mappings(self, sample_df):
        rec = LabelEncodeRecommendation(columns=["color"])
        result = rec.fit_transform(sample_df)
        assert "mappings" in result.metadata


class TestLabelEncodeRecommendationCodeGeneration:
    def test_generate_local_code(self):
        rec = LabelEncodeRecommendation(columns=["color"])
        code = rec.generate_code(Platform.LOCAL)
        assert "LabelEncoder" in code

    def test_generate_databricks_code(self):
        rec = LabelEncodeRecommendation(columns=["color"])
        code = rec.generate_code(Platform.DATABRICKS)
        assert "StringIndexer" in code


class TestEncodingRecommendationEdgeCases:
    def test_column_not_in_dataframe(self):
        df = pd.DataFrame({"b": ["x", "y", "z"]})
        rec = LabelEncodeRecommendation(columns=["a"])
        rec.fit(df)
        result = rec.transform(df)
        assert "b" in result.data.columns

    def test_multiple_columns(self):
        df = pd.DataFrame({"color": ["red", "blue"], "size": ["S", "M"]})
        rec = LabelEncodeRecommendation(columns=["color", "size"])
        result = rec.fit_transform(df)
        assert result.data["color"].dtype in [np.int64, np.int32, int, "int64", "int32"]
        assert result.data["size"].dtype in [np.int64, np.int32, int, "int64", "int32"]
