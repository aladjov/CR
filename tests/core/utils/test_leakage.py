"""Tests for the centralized feature column filtering utilities."""
import pandas as pd
import pytest

from customer_retention.core.utils.leakage import (
    TEMPORAL_METADATA_COLUMNS,
    get_valid_feature_columns,
)


class TestTemporalMetadataColumns:
    def test_contains_expected_columns(self):
        expected = {"feature_timestamp", "label_timestamp", "label_available_flag", "event_timestamp"}
        assert TEMPORAL_METADATA_COLUMNS == expected

    def test_is_frozenset(self):
        assert isinstance(TEMPORAL_METADATA_COLUMNS, frozenset)


class TestGetValidFeatureColumns:
    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({
            "customer_id": [1, 2, 3],
            "feature1": [1.0, 2.0, 3.0],
            "feature2": [4.0, 5.0, 6.0],
            "feature_timestamp": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "label_timestamp": ["2024-02-01", "2024-02-02", "2024-02-03"],
            "label_available_flag": [1, 1, 0],
            "event_timestamp": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "target": [0, 1, 0],
            "original_target": [0, 1, None],
        })

    def test_excludes_temporal_metadata_columns(self, sample_df):
        result = get_valid_feature_columns(sample_df)
        for col in TEMPORAL_METADATA_COLUMNS:
            assert col not in result

    def test_excludes_entity_column(self, sample_df):
        result = get_valid_feature_columns(sample_df, entity_column="customer_id")
        assert "customer_id" not in result

    def test_excludes_target_column(self, sample_df):
        result = get_valid_feature_columns(sample_df, target_column="target")
        assert "target" not in result

    def test_excludes_original_prefix_columns(self, sample_df):
        result = get_valid_feature_columns(sample_df)
        assert "original_target" not in result

    def test_excludes_additional_columns(self, sample_df):
        result = get_valid_feature_columns(sample_df, additional_exclude={"feature1"})
        assert "feature1" not in result
        assert "feature2" in result

    def test_includes_valid_features(self, sample_df):
        result = get_valid_feature_columns(
            sample_df,
            entity_column="customer_id",
            target_column="target",
        )
        assert "feature1" in result
        assert "feature2" in result

    def test_returns_list(self, sample_df):
        result = get_valid_feature_columns(sample_df)
        assert isinstance(result, list)

    def test_combined_exclusions(self, sample_df):
        result = get_valid_feature_columns(
            sample_df,
            entity_column="customer_id",
            target_column="target",
            additional_exclude={"feature2"},
        )
        assert "feature1" in result
        assert "feature2" not in result
        assert "customer_id" not in result
        assert "target" not in result
        for col in TEMPORAL_METADATA_COLUMNS:
            assert col not in result
        assert "original_target" not in result

    def test_empty_dataframe(self):
        df = pd.DataFrame()
        result = get_valid_feature_columns(df)
        assert result == []

    def test_none_entity_column(self, sample_df):
        result = get_valid_feature_columns(sample_df, entity_column=None)
        assert "customer_id" in result

    def test_none_target_column(self, sample_df):
        result = get_valid_feature_columns(sample_df, target_column=None)
        assert "target" in result

    def test_none_additional_exclude(self, sample_df):
        result = get_valid_feature_columns(sample_df, additional_exclude=None)
        assert "feature1" in result
        assert "feature2" in result
