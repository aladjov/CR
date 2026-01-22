import pandas as pd
import pytest

from customer_retention.analysis.recommendations.base import Platform, RecommendationResult
from customer_retention.analysis.recommendations.cleaning.consistency import ConsistencyNormalizeRecommendation


class TestConsistencyNormalizeRecommendationInit:
    def test_default_normalization_is_lowercase(self):
        rec = ConsistencyNormalizeRecommendation(columns=["city"], rationale="test")
        assert rec.normalization == "lowercase"

    def test_custom_normalization(self):
        rec = ConsistencyNormalizeRecommendation(columns=["city"], normalization="strip_whitespace", rationale="test")
        assert rec.normalization == "strip_whitespace"

    def test_recommendation_type(self):
        rec = ConsistencyNormalizeRecommendation(columns=["city"], normalization="lowercase", rationale="test")
        assert rec.recommendation_type == "normalize_lowercase"

    def test_category_is_cleaning(self):
        rec = ConsistencyNormalizeRecommendation(columns=["city"], rationale="test")
        assert rec.category == "cleaning"


class TestConsistencyNormalizeRecommendationFit:
    @pytest.fixture
    def df_with_variants(self):
        return pd.DataFrame({
            "city": ["NYC", "nyc", "Nyc", "New York", "new york"],
            "country": ["USA", "usa", "USA", "US", "us"]
        })

    def test_fit_detects_variants(self, df_with_variants):
        rec = ConsistencyNormalizeRecommendation(columns=["city"], rationale="test")
        rec.fit(df_with_variants)
        assert "city" in rec._fit_params["variants"]
        assert len(rec._fit_params["variants"]["city"]) > 0

    def test_fit_counts_unique_before_normalization(self, df_with_variants):
        rec = ConsistencyNormalizeRecommendation(columns=["city"], rationale="test")
        rec.fit(df_with_variants)
        assert rec._fit_params["unique_before"]["city"] == 5


class TestConsistencyNormalizeLowercase:
    def test_lowercase_normalizes_case_variants(self):
        df = pd.DataFrame({"city": ["NYC", "nyc", "Nyc", "New York"]})
        rec = ConsistencyNormalizeRecommendation(columns=["city"], normalization="lowercase", rationale="test")
        result = rec.fit_transform(df)
        assert list(result.data["city"]) == ["nyc", "nyc", "nyc", "new york"]

    def test_lowercase_preserves_non_string_types(self):
        df = pd.DataFrame({"value": [1, 2, 3]})
        rec = ConsistencyNormalizeRecommendation(columns=["value"], normalization="lowercase", rationale="test")
        result = rec.fit_transform(df)
        assert list(result.data["value"]) == [1, 2, 3]


class TestConsistencyNormalizeStripWhitespace:
    def test_strip_whitespace_removes_leading_trailing(self):
        df = pd.DataFrame({"name": ["  Alice  ", "Bob ", " Carol"]})
        rec = ConsistencyNormalizeRecommendation(columns=["name"], normalization="strip_whitespace", rationale="test")
        result = rec.fit_transform(df)
        assert list(result.data["name"]) == ["Alice", "Bob", "Carol"]

    def test_strip_whitespace_collapses_internal_spaces(self):
        df = pd.DataFrame({"name": ["New  York", "Los   Angeles"]})
        rec = ConsistencyNormalizeRecommendation(columns=["name"], normalization="collapse_whitespace", rationale="test")
        result = rec.fit_transform(df)
        assert list(result.data["name"]) == ["New York", "Los Angeles"]


class TestConsistencyNormalizeUppercase:
    def test_uppercase_normalizes_to_upper(self):
        df = pd.DataFrame({"code": ["abc", "ABC", "Abc"]})
        rec = ConsistencyNormalizeRecommendation(columns=["code"], normalization="uppercase", rationale="test")
        result = rec.fit_transform(df)
        assert list(result.data["code"]) == ["ABC", "ABC", "ABC"]


class TestConsistencyNormalizeTitleCase:
    def test_titlecase_normalizes_names(self):
        df = pd.DataFrame({"name": ["john doe", "JANE SMITH", "Bob Jones"]})
        rec = ConsistencyNormalizeRecommendation(columns=["name"], normalization="titlecase", rationale="test")
        result = rec.fit_transform(df)
        assert list(result.data["name"]) == ["John Doe", "Jane Smith", "Bob Jones"]


class TestConsistencyNormalizeResult:
    @pytest.fixture
    def df_with_variants(self):
        return pd.DataFrame({"city": ["NYC", "nyc", "Nyc"]})

    def test_returns_recommendation_result(self, df_with_variants):
        rec = ConsistencyNormalizeRecommendation(columns=["city"], rationale="test")
        result = rec.fit_transform(df_with_variants)
        assert isinstance(result, RecommendationResult)

    def test_result_has_correct_row_counts(self, df_with_variants):
        rec = ConsistencyNormalizeRecommendation(columns=["city"], rationale="test")
        result = rec.fit_transform(df_with_variants)
        assert result.rows_before == 3
        assert result.rows_after == 3

    def test_result_metadata_contains_values_changed(self, df_with_variants):
        rec = ConsistencyNormalizeRecommendation(columns=["city"], rationale="test")
        result = rec.fit_transform(df_with_variants)
        assert "values_changed" in result.metadata
        assert result.metadata["values_changed"]["city"] == 2

    def test_result_metadata_contains_unique_after(self, df_with_variants):
        rec = ConsistencyNormalizeRecommendation(columns=["city"], rationale="test")
        result = rec.fit_transform(df_with_variants)
        assert "unique_after" in result.metadata
        assert result.metadata["unique_after"]["city"] == 1

    def test_transform_does_not_modify_original(self, df_with_variants):
        rec = ConsistencyNormalizeRecommendation(columns=["city"], rationale="test")
        original = df_with_variants["city"].tolist()
        rec.fit_transform(df_with_variants)
        assert df_with_variants["city"].tolist() == original


class TestConsistencyNormalizeCodeGeneration:
    def test_generate_local_code_lowercase(self):
        rec = ConsistencyNormalizeRecommendation(columns=["city"], normalization="lowercase", rationale="normalize case")
        rec._is_fitted = True
        code = rec.generate_code(Platform.LOCAL)
        assert "str.lower()" in code
        assert "'city'" in code

    def test_generate_local_code_strip(self):
        rec = ConsistencyNormalizeRecommendation(columns=["name"], normalization="strip_whitespace", rationale="strip")
        rec._is_fitted = True
        code = rec.generate_code(Platform.LOCAL)
        assert "str.strip()" in code

    def test_generate_databricks_code_lowercase(self):
        rec = ConsistencyNormalizeRecommendation(columns=["city"], normalization="lowercase", rationale="normalize")
        rec._is_fitted = True
        code = rec.generate_code(Platform.DATABRICKS)
        assert "lower" in code


class TestConsistencyNormalizeMultipleColumns:
    def test_normalizes_multiple_columns(self):
        df = pd.DataFrame({
            "city": ["NYC", "nyc"],
            "country": ["USA", "usa"]
        })
        rec = ConsistencyNormalizeRecommendation(columns=["city", "country"], normalization="lowercase", rationale="test")
        result = rec.fit_transform(df)
        assert result.data["city"].tolist() == ["nyc", "nyc"]
        assert result.data["country"].tolist() == ["usa", "usa"]


class TestConsistencyNormalizeEdgeCases:
    def test_handles_null_values(self):
        df = pd.DataFrame({"city": ["NYC", None, "nyc"]})
        rec = ConsistencyNormalizeRecommendation(columns=["city"], normalization="lowercase", rationale="test")
        result = rec.fit_transform(df)
        assert result.data["city"].tolist()[0] == "nyc"
        assert pd.isna(result.data["city"].tolist()[1])
        assert result.data["city"].tolist()[2] == "nyc"

    def test_handles_empty_strings(self):
        df = pd.DataFrame({"city": ["NYC", "", "nyc"]})
        rec = ConsistencyNormalizeRecommendation(columns=["city"], normalization="lowercase", rationale="test")
        result = rec.fit_transform(df)
        assert result.data["city"].tolist() == ["nyc", "", "nyc"]

    def test_column_not_in_dataframe(self):
        df = pd.DataFrame({"other": ["a", "b"]})
        rec = ConsistencyNormalizeRecommendation(columns=["city"], rationale="test")
        rec.fit(df)
        result = rec.transform(df)
        assert "other" in result.data.columns

    def test_already_normalized_data(self):
        df = pd.DataFrame({"city": ["nyc", "nyc", "nyc"]})
        rec = ConsistencyNormalizeRecommendation(columns=["city"], normalization="lowercase", rationale="test")
        result = rec.fit_transform(df)
        assert result.metadata["values_changed"]["city"] == 0
