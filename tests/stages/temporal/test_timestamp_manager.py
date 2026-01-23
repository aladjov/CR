from datetime import datetime, timedelta

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


class TestTimestampManagerDerived:
    def test_derived_without_config_raises(self):
        config = TimestampConfig(strategy=TimestampStrategy.DERIVED)
        manager = TimestampManager(config)
        df = pd.DataFrame({"tenure_months": [12, 24], "value": [1, 2]})

        with pytest.raises(ValueError, match="derivation_config required"):
            manager.ensure_timestamps(df)

    def test_derived_feature_from_tenure(self):
        config = TimestampConfig(
            strategy=TimestampStrategy.DERIVED,
            observation_window_days=90,
            derivation_config={
                "feature_derivation": {
                    "sources": ["tenure_months"],
                    "formula": "reference_date - tenure * 30 days",
                }
            },
        )
        manager = TimestampManager(config)
        df = pd.DataFrame({"tenure_months": [6, 12], "value": [1, 2]})

        result = manager.ensure_timestamps(df)

        assert "feature_timestamp" in result.columns
        assert "label_timestamp" in result.columns
        assert result["label_available_flag"].all()
        # Longer tenure → earlier feature_timestamp
        assert result["feature_timestamp"].iloc[0] > result["feature_timestamp"].iloc[1]

    def test_derived_label_derivation(self):
        config = TimestampConfig(
            strategy=TimestampStrategy.DERIVED,
            derivation_config={
                "feature_derivation": {
                    "sources": ["tenure_months"],
                    "formula": "reference_date - tenure * 30 days",
                },
                "label_derivation": {
                    "sources": ["tenure_months"],
                    "formula": "reference_date - tenure * 30 days",
                },
            },
        )
        manager = TimestampManager(config)
        df = pd.DataFrame({"tenure_months": [6, 12], "value": [1, 2]})

        result = manager.ensure_timestamps(df)

        assert "label_timestamp" in result.columns
        assert result["label_available_flag"].all()

    def test_derived_with_empty_formula_no_op(self):
        config = TimestampConfig(
            strategy=TimestampStrategy.DERIVED,
            derivation_config={
                "feature_derivation": {"sources": [], "formula": ""},
            },
        )
        manager = TimestampManager(config)
        df = pd.DataFrame({"value": [1, 2]})

        result = manager.ensure_timestamps(df)

        assert "label_available_flag" in result.columns

    def test_derived_label_defaults_to_feature_plus_window(self):
        config = TimestampConfig(
            strategy=TimestampStrategy.DERIVED,
            observation_window_days=60,
            derivation_config={
                "feature_derivation": {
                    "sources": ["tenure_months"],
                    "formula": "reference_date - tenure * 30 days",
                }
            },
        )
        manager = TimestampManager(config)
        df = pd.DataFrame({"tenure_months": [3], "value": [1]})

        result = manager.ensure_timestamps(df)

        diff = result["label_timestamp"].iloc[0] - result["feature_timestamp"].iloc[0]
        assert diff == timedelta(days=60)


class TestLabelAvailableFlag:
    def test_sparse_label_timestamps_observation_complete(self):
        config = TimestampConfig(
            strategy=TimestampStrategy.PRODUCTION,
            feature_timestamp_column="feature_date",
            label_timestamp_column="event_date",
            observation_window_days=90,
        )
        manager = TimestampManager(config)
        df = pd.DataFrame({
            "feature_date": pd.to_datetime(["2022-01-01", "2022-02-01", "2022-03-01"]),
            "event_date": pd.to_datetime([None, "2022-05-01", None]),
            "value": [1, 2, 3],
        })

        result = manager.ensure_timestamps(df)

        # All rows: feature_date + 90 days << now, so observation complete
        assert result["label_available_flag"].all()

    def test_sparse_label_within_observation_window(self):
        config = TimestampConfig(
            strategy=TimestampStrategy.PRODUCTION,
            feature_timestamp_column="feature_date",
            label_timestamp_column="event_date",
            observation_window_days=90,
        )
        manager = TimestampManager(config)
        now = datetime.now()
        recent = now - timedelta(days=30)  # Only 30 days ago, window is 90
        df = pd.DataFrame({
            "feature_date": [recent, recent, recent],
            "event_date": pd.to_datetime([None, None, None]),
            "value": [1, 2, 3],
        })

        result = manager.ensure_timestamps(df)

        # Observation window not complete AND no event → label NOT available
        assert not result["label_available_flag"].any()

    def test_event_happened_within_window_still_available(self):
        config = TimestampConfig(
            strategy=TimestampStrategy.PRODUCTION,
            feature_timestamp_column="feature_date",
            label_timestamp_column="event_date",
            observation_window_days=90,
        )
        manager = TimestampManager(config)
        now = datetime.now()
        recent = now - timedelta(days=30)
        event_in_past = now - timedelta(days=10)
        df = pd.DataFrame({
            "feature_date": [recent, recent],
            "event_date": [event_in_past, None],
            "value": [1, 2],
        })

        result = manager.ensure_timestamps(df)

        # Row 0: event happened in past → available
        # Row 1: no event and window not complete → not available
        assert result["label_available_flag"].iloc[0] is True or result["label_available_flag"].iloc[0] == True
        assert result["label_available_flag"].iloc[1] is False or result["label_available_flag"].iloc[1] == False

    def test_all_labels_present_and_past(self):
        config = TimestampConfig(
            strategy=TimestampStrategy.PRODUCTION,
            feature_timestamp_column="feature_date",
            label_timestamp_column="event_date",
        )
        manager = TimestampManager(config)
        df = pd.DataFrame({
            "feature_date": pd.to_datetime(["2022-01-01", "2022-02-01"]),
            "event_date": pd.to_datetime(["2022-04-01", "2022-05-01"]),
        })

        result = manager.ensure_timestamps(df)

        assert result["label_available_flag"].all()


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
