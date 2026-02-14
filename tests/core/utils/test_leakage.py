"""Tests for the centralized feature column filtering utilities."""
import numpy as np
import pandas as pd
import pytest

from customer_retention.core.utils.leakage import (
    TEMPORAL_METADATA_COLUMNS,
    detect_leaking_features,
    detect_target_leaking_datetime_columns,
    get_valid_feature_columns,
)


class TestTemporalMetadataColumns:
    def test_contains_pipeline_derived_columns_only(self):
        expected = {"feature_timestamp", "label_timestamp", "label_available_flag"}
        assert TEMPORAL_METADATA_COLUMNS == expected

    def test_event_timestamp_not_in_metadata(self):
        assert "event_timestamp" not in TEMPORAL_METADATA_COLUMNS

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

    def test_event_timestamp_excluded_from_features(self, sample_df):
        result = get_valid_feature_columns(sample_df)
        assert "event_timestamp" not in result


class TestDetectTargetLeakingDatetimeColumns:
    def test_perfect_null_correlation_detected(self):
        df = pd.DataFrame({
            "churn_end": [pd.NaT, pd.NaT, pd.NaT, pd.Timestamp("2024-03-01"),
                          pd.Timestamp("2024-03-02"), pd.Timestamp("2024-03-03")],
            "churned": [0, 0, 0, 1, 1, 1],
        })
        result = detect_target_leaking_datetime_columns(df, ["churn_end"], "churned")
        assert result == ["churn_end"]

    def test_no_correlation_returns_empty(self):
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            "signup_date": pd.date_range("2024-01-01", periods=100),
            "churned": rng.integers(0, 2, size=100),
        })
        result = detect_target_leaking_datetime_columns(df, ["signup_date"], "churned")
        assert result == []

    def test_empty_datetime_columns_returns_empty(self):
        df = pd.DataFrame({"churned": [0, 1, 0, 1]})
        result = detect_target_leaking_datetime_columns(df, [], "churned")
        assert result == []

    def test_no_nulls_not_flagged(self):
        df = pd.DataFrame({
            "created_at": pd.date_range("2024-01-01", periods=50),
            "churned": [0, 1] * 25,
        })
        result = detect_target_leaking_datetime_columns(df, ["created_at"], "churned")
        assert result == []

    def test_partial_correlation_below_threshold_not_flagged(self):
        df = pd.DataFrame({
            "end_date": [pd.NaT, pd.NaT, pd.Timestamp("2024-03-01"),
                         pd.Timestamp("2024-03-02"), pd.Timestamp("2024-03-03"),
                         pd.Timestamp("2024-03-04")],
            "churned": [0, 0, 1, 1, 0, 1],
        })
        result = detect_target_leaking_datetime_columns(
            df, ["end_date"], "churned", null_correlation_threshold=0.9,
        )
        assert result == []

    def test_partial_correlation_above_threshold_flagged(self):
        df = pd.DataFrame({
            "end_date": [pd.NaT, pd.NaT, pd.NaT, pd.NaT,
                         pd.Timestamp("2024-03-01"), pd.Timestamp("2024-03-02"),
                         pd.Timestamp("2024-03-03"), pd.Timestamp("2024-03-04")],
            "churned": [0, 0, 0, 0, 1, 1, 1, 1],
        })
        result = detect_target_leaking_datetime_columns(
            df, ["end_date"], "churned", null_correlation_threshold=0.8,
        )
        assert result == ["end_date"]

    def test_target_column_none_returns_empty(self):
        df = pd.DataFrame({
            "churn_end": [pd.NaT, pd.Timestamp("2024-01-01")],
            "value": [0, 1],
        })
        result = detect_target_leaking_datetime_columns(df, ["churn_end"], None)
        assert result == []

    def test_all_null_datetime_column_not_flagged(self):
        df = pd.DataFrame({
            "empty_date": [pd.NaT, pd.NaT, pd.NaT, pd.NaT],
            "churned": [0, 1, 0, 1],
        })
        result = detect_target_leaking_datetime_columns(df, ["empty_date"], "churned")
        assert result == []

    def test_multiclass_target(self):
        df = pd.DataFrame({
            "cancel_date": [pd.NaT, pd.NaT, pd.NaT,
                            pd.Timestamp("2024-01-01"), pd.Timestamp("2024-02-01"),
                            pd.Timestamp("2024-03-01")],
            "status": [0, 0, 0, 1, 2, 3],
        })
        result = detect_target_leaking_datetime_columns(df, ["cancel_date"], "status")
        assert result == ["cancel_date"]

    def test_multiple_columns_mixed(self):
        df = pd.DataFrame({
            "churn_end": [pd.NaT, pd.NaT, pd.Timestamp("2024-03-01"), pd.Timestamp("2024-03-02")],
            "signup_date": pd.date_range("2024-01-01", periods=4),
            "churned": [0, 0, 1, 1],
        })
        result = detect_target_leaking_datetime_columns(
            df, ["churn_end", "signup_date"], "churned",
        )
        assert result == ["churn_end"]
        assert "signup_date" not in result

    def test_inverse_correlation_detected(self):
        df = pd.DataFrame({
            "active_until": [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02"),
                             pd.NaT, pd.NaT],
            "churned": [0, 0, 1, 1],
        })
        result = detect_target_leaking_datetime_columns(df, ["active_until"], "churned")
        assert result == ["active_until"]


class TestDetectLeakingFeatures:
    def test_detects_null_pattern_leaker(self):
        df = pd.DataFrame({
            "is_missing_end": [1, 1, 1, 0, 0, 0],
            "churned": [0, 0, 0, 1, 1, 1],
        })
        result = detect_leaking_features(df, ["is_missing_end"], "churned")
        assert result == ["is_missing_end"]

    def test_detects_value_correlation_leaker(self):
        df = pd.DataFrame({
            "is_missing_churn_end": [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
            "churned": [1, 1, 1, 0, 0, 0],
        })
        result = detect_leaking_features(df, ["is_missing_churn_end"], "churned")
        assert result == ["is_missing_churn_end"]

    def test_returns_empty_for_clean_features(self):
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            "age": rng.integers(18, 65, size=100),
            "income": rng.normal(50000, 10000, size=100),
            "churned": rng.integers(0, 2, size=100),
        })
        result = detect_leaking_features(df, ["age", "income"], "churned")
        assert result == []

    def test_returns_empty_for_none_target(self):
        df = pd.DataFrame({
            "feature": [1, 2, 3],
            "target": [0, 1, 0],
        })
        result = detect_leaking_features(df, ["feature"], None)
        assert result == []

    def test_returns_empty_for_empty_columns(self):
        df = pd.DataFrame({
            "feature": [1, 2, 3],
            "target": [0, 1, 0],
        })
        result = detect_leaking_features(df, [], "target")
        assert result == []

    def test_preserves_input_order(self):
        df = pd.DataFrame({
            "z_leaker": [1.0, 1.0, 0.0, 0.0],
            "a_leaker": [0.0, 0.0, 1.0, 1.0],
            "m_clean": [0.5, 0.3, 0.5, 0.3],
            "churned": [1, 1, 0, 0],
        })
        result = detect_leaking_features(df, ["z_leaker", "a_leaker", "m_clean"], "churned")
        assert result == ["z_leaker", "a_leaker"]

    def test_skips_non_numeric_columns(self):
        df = pd.DataFrame({
            "text_col": ["hello", "world", "foo", "bar"],
            "numeric_leaker": [1.0, 1.0, 0.0, 0.0],
            "churned": [1, 1, 0, 0],
        })
        result = detect_leaking_features(df, ["text_col", "numeric_leaker"], "churned")
        assert "text_col" not in result
        assert "numeric_leaker" in result

    def test_custom_thresholds(self):
        df = pd.DataFrame({
            "moderate_corr": [1, 1, 1, 0, 0, 1, 0, 0],
            "churned": [1, 1, 1, 0, 0, 0, 0, 0],
        })
        result_strict = detect_leaking_features(
            df, ["moderate_corr"], "churned",
            value_correlation_threshold=0.99,
        )
        assert result_strict == []

        result_relaxed = detect_leaking_features(
            df, ["moderate_corr"], "churned",
            value_correlation_threshold=0.5,
        )
        assert result_relaxed == ["moderate_corr"]

    def test_both_null_and_value_leakers_detected(self):
        df = pd.DataFrame({
            "null_leaker": [np.nan, np.nan, np.nan, 1.0, 2.0, 3.0],
            "value_leaker": [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
            "clean": [10, 20, 30, 10, 20, 30],
            "churned": [0, 0, 0, 1, 1, 1],
        })
        result = detect_leaking_features(
            df, ["null_leaker", "value_leaker", "clean"], "churned",
        )
        assert "null_leaker" in result
        assert "value_leaker" in result
        assert "clean" not in result

    def test_constant_column_not_flagged(self):
        df = pd.DataFrame({
            "constant": [1.0, 1.0, 1.0, 1.0],
            "churned": [0, 0, 1, 1],
        })
        result = detect_leaking_features(df, ["constant"], "churned")
        assert result == []

    def test_all_null_column_not_flagged(self):
        df = pd.DataFrame({
            "all_null": [np.nan, np.nan, np.nan, np.nan],
            "churned": [0, 0, 1, 1],
        })
        result = detect_leaking_features(df, ["all_null"], "churned")
        assert result == []

    def test_missing_columns_skipped(self):
        df = pd.DataFrame({
            "existing": [1.0, 0.0, 1.0, 0.0],
            "churned": [1, 0, 1, 0],
        })
        result = detect_leaking_features(
            df, ["existing", "nonexistent_col"], "churned",
        )
        assert "nonexistent_col" not in result

    def test_missing_target_column_returns_empty(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        result = detect_leaking_features(df, ["a"], "missing_target")
        assert result == []
