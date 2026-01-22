import pandas as pd
import pytest

from customer_retention.analysis.recommendations.base import Platform, RecommendationResult
from customer_retention.analysis.recommendations.cleaning.deduplicate import DeduplicateRecommendation


class TestDeduplicateRecommendationInit:
    def test_default_strategy_is_keep_first(self):
        rec = DeduplicateRecommendation(key_columns=["id"], rationale="test")
        assert rec.strategy == "keep_first"

    def test_custom_strategy(self):
        rec = DeduplicateRecommendation(key_columns=["id"], strategy="keep_last", rationale="test")
        assert rec.strategy == "keep_last"

    def test_recommendation_type(self):
        rec = DeduplicateRecommendation(key_columns=["id"], strategy="keep_first", rationale="test")
        assert rec.recommendation_type == "deduplicate_keep_first"

    def test_category_is_cleaning(self):
        rec = DeduplicateRecommendation(key_columns=["id"], rationale="test")
        assert rec.category == "cleaning"

    def test_stores_key_columns(self):
        rec = DeduplicateRecommendation(key_columns=["id", "date"], rationale="test")
        assert rec.key_columns == ["id", "date"]

    def test_stores_timestamp_column(self):
        rec = DeduplicateRecommendation(key_columns=["id"], timestamp_column="updated_at", rationale="test")
        assert rec.timestamp_column == "updated_at"


class TestDeduplicateRecommendationFit:
    @pytest.fixture
    def df_with_duplicates(self):
        return pd.DataFrame({
            "id": [1, 2, 2, 3, 3, 3],
            "value": ["a", "b", "c", "d", "e", "f"],
            "timestamp": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-01", "2024-01-02", "2024-01-03"])
        })

    def test_fit_counts_duplicates(self, df_with_duplicates):
        rec = DeduplicateRecommendation(key_columns=["id"], rationale="test")
        rec.fit(df_with_duplicates)
        assert rec._fit_params["duplicate_count"] == 3

    def test_fit_identifies_duplicate_keys(self, df_with_duplicates):
        rec = DeduplicateRecommendation(key_columns=["id"], rationale="test")
        rec.fit(df_with_duplicates)
        assert set(rec._fit_params["duplicate_keys"]) == {2, 3}


class TestDeduplicateRecommendationTransformKeepFirst:
    @pytest.fixture
    def df_with_duplicates(self):
        return pd.DataFrame({
            "id": [1, 2, 2, 3, 3, 3],
            "value": ["a", "b", "c", "d", "e", "f"]
        })

    def test_keep_first_removes_duplicates(self, df_with_duplicates):
        rec = DeduplicateRecommendation(key_columns=["id"], strategy="keep_first", rationale="test")
        result = rec.fit_transform(df_with_duplicates)
        assert len(result.data) == 3
        assert list(result.data["id"]) == [1, 2, 3]

    def test_keep_first_preserves_first_value(self, df_with_duplicates):
        rec = DeduplicateRecommendation(key_columns=["id"], strategy="keep_first", rationale="test")
        result = rec.fit_transform(df_with_duplicates)
        assert result.data[result.data["id"] == 2]["value"].values[0] == "b"
        assert result.data[result.data["id"] == 3]["value"].values[0] == "d"


class TestDeduplicateRecommendationTransformKeepLast:
    @pytest.fixture
    def df_with_duplicates(self):
        return pd.DataFrame({
            "id": [1, 2, 2, 3, 3, 3],
            "value": ["a", "b", "c", "d", "e", "f"]
        })

    def test_keep_last_removes_duplicates(self, df_with_duplicates):
        rec = DeduplicateRecommendation(key_columns=["id"], strategy="keep_last", rationale="test")
        result = rec.fit_transform(df_with_duplicates)
        assert len(result.data) == 3

    def test_keep_last_preserves_last_value(self, df_with_duplicates):
        rec = DeduplicateRecommendation(key_columns=["id"], strategy="keep_last", rationale="test")
        result = rec.fit_transform(df_with_duplicates)
        assert result.data[result.data["id"] == 2]["value"].values[0] == "c"
        assert result.data[result.data["id"] == 3]["value"].values[0] == "f"


class TestDeduplicateRecommendationTransformKeepMostRecent:
    @pytest.fixture
    def df_with_timestamps(self):
        return pd.DataFrame({
            "id": [1, 2, 2, 3, 3, 3],
            "value": ["a", "b", "c", "d", "e", "f"],
            "updated_at": pd.to_datetime([
                "2024-01-01", "2024-01-01", "2024-01-05",
                "2024-01-03", "2024-01-02", "2024-01-01"
            ])
        })

    def test_keep_most_recent_uses_timestamp(self, df_with_timestamps):
        rec = DeduplicateRecommendation(
            key_columns=["id"], strategy="keep_most_recent",
            timestamp_column="updated_at", rationale="test"
        )
        result = rec.fit_transform(df_with_timestamps)
        assert len(result.data) == 3
        assert result.data[result.data["id"] == 2]["value"].values[0] == "c"
        assert result.data[result.data["id"] == 3]["value"].values[0] == "d"


class TestDeduplicateRecommendationTransformDropExact:
    def test_drop_exact_only_removes_identical_rows(self):
        df = pd.DataFrame({
            "id": [1, 2, 2, 2],
            "value": ["a", "b", "b", "c"]
        })
        rec = DeduplicateRecommendation(key_columns=["id", "value"], strategy="drop_exact", rationale="test")
        result = rec.fit_transform(df)
        assert len(result.data) == 3
        assert list(result.data["value"]) == ["a", "b", "c"]


class TestDeduplicateRecommendationResult:
    @pytest.fixture
    def df_with_duplicates(self):
        return pd.DataFrame({"id": [1, 2, 2, 3], "value": ["a", "b", "c", "d"]})

    def test_returns_recommendation_result(self, df_with_duplicates):
        rec = DeduplicateRecommendation(key_columns=["id"], rationale="test")
        result = rec.fit_transform(df_with_duplicates)
        assert isinstance(result, RecommendationResult)

    def test_result_has_correct_row_counts(self, df_with_duplicates):
        rec = DeduplicateRecommendation(key_columns=["id"], rationale="test")
        result = rec.fit_transform(df_with_duplicates)
        assert result.rows_before == 4
        assert result.rows_after == 3

    def test_result_metadata_contains_duplicates_removed(self, df_with_duplicates):
        rec = DeduplicateRecommendation(key_columns=["id"], rationale="test")
        result = rec.fit_transform(df_with_duplicates)
        assert result.metadata["duplicates_removed"] == 1

    def test_transform_does_not_modify_original(self, df_with_duplicates):
        rec = DeduplicateRecommendation(key_columns=["id"], rationale="test")
        original_len = len(df_with_duplicates)
        rec.fit_transform(df_with_duplicates)
        assert len(df_with_duplicates) == original_len


class TestDeduplicateRecommendationCodeGeneration:
    def test_generate_local_code_keep_first(self):
        rec = DeduplicateRecommendation(key_columns=["id"], strategy="keep_first", rationale="remove dups")
        rec._is_fitted = True
        code = rec.generate_code(Platform.LOCAL)
        assert "drop_duplicates" in code
        assert "'id'" in code
        assert "keep='first'" in code

    def test_generate_local_code_keep_last(self):
        rec = DeduplicateRecommendation(key_columns=["id"], strategy="keep_last", rationale="remove dups")
        rec._is_fitted = True
        code = rec.generate_code(Platform.LOCAL)
        assert "keep='last'" in code

    def test_generate_local_code_keep_most_recent(self):
        rec = DeduplicateRecommendation(
            key_columns=["id"], strategy="keep_most_recent",
            timestamp_column="updated_at", rationale="remove dups"
        )
        rec._is_fitted = True
        code = rec.generate_code(Platform.LOCAL)
        assert "sort_values" in code
        assert "updated_at" in code

    def test_generate_databricks_code(self):
        rec = DeduplicateRecommendation(key_columns=["id"], strategy="keep_first", rationale="remove dups")
        rec._is_fitted = True
        code = rec.generate_code(Platform.DATABRICKS)
        assert "dropDuplicates" in code or "drop_duplicates" in code


class TestDeduplicateRecommendationMultipleKeyColumns:
    def test_multiple_key_columns(self):
        df = pd.DataFrame({
            "customer_id": [1, 1, 1, 2],
            "order_date": ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-01"],
            "value": ["a", "b", "c", "d"]
        })
        rec = DeduplicateRecommendation(key_columns=["customer_id", "order_date"], rationale="test")
        result = rec.fit_transform(df)
        assert len(result.data) == 3


class TestDeduplicateRecommendationEdgeCases:
    def test_no_duplicates(self):
        df = pd.DataFrame({"id": [1, 2, 3], "value": ["a", "b", "c"]})
        rec = DeduplicateRecommendation(key_columns=["id"], rationale="test")
        result = rec.fit_transform(df)
        assert len(result.data) == 3
        assert result.metadata["duplicates_removed"] == 0

    def test_empty_dataframe(self):
        df = pd.DataFrame({"id": [], "value": []})
        rec = DeduplicateRecommendation(key_columns=["id"], rationale="test")
        result = rec.fit_transform(df)
        assert len(result.data) == 0
