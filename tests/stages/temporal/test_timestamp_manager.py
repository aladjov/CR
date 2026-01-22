from datetime import timedelta

import pandas as pd
import pytest

from customer_retention.stages.temporal.timestamp_manager import TimestampConfig, TimestampManager, TimestampStrategy


class TestTimestampConfig:
    def test_default_config(self):
        config = TimestampConfig(strategy=TimestampStrategy.PRODUCTION)
        assert config.observation_window_days == 90
        assert config.synthetic_base_date == "2024-01-01"
        assert config.derive_label_from_feature is False

    def test_custom_config(self):
        config = TimestampConfig(
            strategy=TimestampStrategy.SYNTHETIC_FIXED,
            observation_window_days=30,
            synthetic_base_date="2023-01-01"
        )
        assert config.observation_window_days == 30
        assert config.synthetic_base_date == "2023-01-01"


class TestTimestampManagerProduction:
    @pytest.fixture
    def production_df(self):
        return pd.DataFrame({
            "customer_id": ["A", "B", "C"],
            "feature_date": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01"]),
            "label_date": pd.to_datetime(["2024-04-01", "2024-05-01", "2024-06-01"]),
            "value": [100, 200, 300]
        })

    def test_validate_production_timestamps(self, production_df):
        config = TimestampConfig(
            strategy=TimestampStrategy.PRODUCTION,
            feature_timestamp_column="feature_date",
            label_timestamp_column="label_date"
        )
        manager = TimestampManager(config)
        result = manager.ensure_timestamps(production_df)

        assert "feature_timestamp" in result.columns
        assert "label_timestamp" in result.columns
        assert "label_available_flag" in result.columns

    def test_missing_column_raises_error(self, production_df):
        config = TimestampConfig(
            strategy=TimestampStrategy.PRODUCTION,
            feature_timestamp_column="nonexistent_column",
            label_timestamp_column="label_date"
        )
        manager = TimestampManager(config)

        with pytest.raises(ValueError, match="Missing required timestamp columns"):
            manager.ensure_timestamps(production_df)

    def test_derive_label_from_feature(self, production_df):
        config = TimestampConfig(
            strategy=TimestampStrategy.PRODUCTION,
            feature_timestamp_column="feature_date",
            derive_label_from_feature=True,
            observation_window_days=90
        )
        manager = TimestampManager(config)
        result = manager.ensure_timestamps(production_df)

        expected_label = pd.to_datetime("2024-01-01") + timedelta(days=90)
        assert result["label_timestamp"].iloc[0] == expected_label


class TestTimestampManagerSynthetic:
    @pytest.fixture
    def raw_df(self):
        return pd.DataFrame({
            "customer_id": ["A", "B", "C", "D", "E"],
            "value": [100, 200, 300, 400, 500]
        })

    def test_synthetic_fixed_timestamps(self, raw_df):
        config = TimestampConfig(
            strategy=TimestampStrategy.SYNTHETIC_FIXED,
            synthetic_base_date="2024-01-01",
            observation_window_days=90
        )
        manager = TimestampManager(config)
        result = manager.ensure_timestamps(raw_df)

        assert (result["feature_timestamp"] == pd.to_datetime("2024-01-01")).all()
        assert (result["label_timestamp"] == pd.to_datetime("2024-03-31")).all()  # 90 days from Jan 1
        assert result["label_available_flag"].all()

    def test_synthetic_index_timestamps(self, raw_df):
        config = TimestampConfig(
            strategy=TimestampStrategy.SYNTHETIC_INDEX,
            synthetic_base_date="2024-01-01",
            observation_window_days=90
        )
        manager = TimestampManager(config)
        result = manager.ensure_timestamps(raw_df)

        base = pd.to_datetime("2024-01-01")
        expected_feature_ts = [base + timedelta(days=i) for i in range(5)]
        assert list(result["feature_timestamp"]) == expected_feature_ts

    def test_synthetic_random_timestamps(self, raw_df):
        config = TimestampConfig(
            strategy=TimestampStrategy.SYNTHETIC_RANDOM,
            synthetic_base_date="2024-01-01",
            synthetic_range_days=365,
            observation_window_days=90
        )
        manager = TimestampManager(config)
        result = manager.ensure_timestamps(raw_df)

        base = pd.to_datetime("2024-01-01")
        assert (result["feature_timestamp"] >= base).all()
        assert (result["feature_timestamp"] < base + timedelta(days=365)).all()


class TestPointInTimeValidation:
    def test_valid_point_in_time(self):
        df = pd.DataFrame({
            "feature_timestamp": pd.to_datetime(["2024-01-01", "2024-02-01"]),
            "label_timestamp": pd.to_datetime(["2024-04-01", "2024-05-01"]),
        })
        config = TimestampConfig(strategy=TimestampStrategy.PRODUCTION)
        manager = TimestampManager(config)

        assert manager.validate_point_in_time(df) is True

    def test_invalid_point_in_time_raises_error(self):
        df = pd.DataFrame({
            "feature_timestamp": pd.to_datetime(["2024-06-01", "2024-02-01"]),
            "label_timestamp": pd.to_datetime(["2024-04-01", "2024-05-01"]),
        })
        config = TimestampConfig(strategy=TimestampStrategy.PRODUCTION)
        manager = TimestampManager(config)

        with pytest.raises(ValueError, match="Point-in-time violation"):
            manager.validate_point_in_time(df)

    def test_missing_columns_raises_error(self):
        df = pd.DataFrame({"value": [1, 2, 3]})
        config = TimestampConfig(strategy=TimestampStrategy.PRODUCTION)
        manager = TimestampManager(config)

        with pytest.raises(ValueError, match="Missing timestamp columns"):
            manager.validate_point_in_time(df)


class TestTimestampSummary:
    def test_get_timestamp_summary(self):
        df = pd.DataFrame({
            "feature_timestamp": pd.to_datetime(["2024-01-01", "2024-02-01", None]),
            "label_timestamp": pd.to_datetime(["2024-04-01", "2024-05-01", "2024-06-01"]),
            "label_available_flag": [True, True, False]
        })
        config = TimestampConfig(strategy=TimestampStrategy.PRODUCTION)
        manager = TimestampManager(config)
        summary = manager.get_timestamp_summary(df)

        assert summary["strategy"] == "production"
        assert "feature_timestamp_min" in summary
        assert "feature_timestamp_max" in summary
        assert summary["feature_timestamp_null_pct"] == pytest.approx(1/3)
        assert summary["label_available_pct"] == pytest.approx(2/3)
