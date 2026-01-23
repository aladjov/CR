from datetime import datetime

import pandas as pd
import pytest

from customer_retention.stages.temporal.scenario_detector import ScenarioDetector, auto_detect_and_configure
from customer_retention.stages.temporal.timestamp_manager import TimestampStrategy


class TestScenarioDetector:
    @pytest.fixture
    def detector(self):
        return ScenarioDetector(reference_date=datetime(2024, 6, 1))

    @pytest.fixture
    def production_df(self):
        return pd.DataFrame({
            "customer_id": ["A", "B", "C"],
            "last_activity_date": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01"]),
            "churn_date": pd.to_datetime(["2024-04-01", "2024-05-01", "2024-06-01"]),
            "churned": [1, 0, 1]
        })

    @pytest.fixture
    def partial_df(self):
        return pd.DataFrame({
            "customer_id": ["A", "B", "C"],
            "last_activity_date": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01"]),
            "churned": [1, 0, 1]
        })

    @pytest.fixture
    def kaggle_df(self):
        return pd.DataFrame({
            "customer_id": ["A", "B", "C"],
            "tenure_months": [12, 24, 6],
            "churned": [1, 0, 1]
        })

    @pytest.fixture
    def synthetic_df(self):
        return pd.DataFrame({
            "customer_id": ["A", "B", "C"],
            "value": [100, 200, 300],
            "category": ["X", "Y", "Z"],
            "churned": [1, 0, 1]
        })


class TestDetectScenario(TestScenarioDetector):
    def test_detect_production_scenario(self, detector, production_df):
        scenario, config, result = detector.detect(production_df, "churned")

        assert scenario in ["production", "production_derived"]
        assert config.strategy == TimestampStrategy.PRODUCTION
        assert not result.requires_synthetic

    def test_detect_partial_scenario(self, detector, partial_df):
        scenario, config, result = detector.detect(partial_df, "churned")

        assert scenario == "partial"
        assert config.derive_label_from_feature is True

    def test_detect_derivable_scenario(self, detector, kaggle_df):
        scenario, config, result = detector.detect(kaggle_df, "churned")

        assert scenario in ["derived", "synthetic"]
        if scenario == "derived":
            assert config.strategy == TimestampStrategy.DERIVED
            assert config.derivation_config is not None

    def test_detect_synthetic_scenario(self, detector, synthetic_df):
        scenario, config, result = detector.detect(synthetic_df, "churned")

        assert scenario == "synthetic"
        assert config.strategy == TimestampStrategy.SYNTHETIC_INDEX
        assert result.requires_synthetic


class TestConfigureScenarios(TestScenarioDetector):
    def test_production_config_sets_columns(self, detector, production_df):
        scenario, config, _ = detector.detect(production_df, "churned")

        if not config.derivation_config:
            assert config.feature_timestamp_column is not None or config.derive_label_from_feature

    def test_partial_config_derives_label(self, detector, partial_df):
        _, config, _ = detector.detect(partial_df, "churned")

        assert config.derive_label_from_feature is True
        assert config.observation_window_days == 180

    def test_custom_label_window_days(self, partial_df):
        detector = ScenarioDetector(label_window_days=90)
        _, config, _ = detector.detect(partial_df, "churned")

        assert config.observation_window_days == 90

    def test_synthetic_config_sets_base_date(self, detector, synthetic_df):
        _, config, _ = detector.detect(synthetic_df, "churned")

        assert config.synthetic_base_date == "2024-01-01"


class TestGetScenarioSummary(TestScenarioDetector):
    def test_summary_contains_required_fields(self, detector, production_df):
        scenario, config, result = detector.detect(production_df, "churned")
        summary = detector.get_scenario_summary(scenario, config, result)

        assert "scenario" in summary
        assert "strategy" in summary
        assert "feature_timestamp_column" in summary
        assert "observation_window_days" in summary
        assert "recommendation" in summary

    def test_summary_shows_synthetic_requirement(self, detector, synthetic_df):
        scenario, config, result = detector.detect(synthetic_df, "churned")
        summary = detector.get_scenario_summary(scenario, config, result)

        assert summary["requires_synthetic"] is True


class TestAutoDetectAndConfigure:
    def test_auto_detect_returns_scenario_and_config(self):
        df = pd.DataFrame({
            "customer_id": ["A", "B"],
            "last_activity_date": pd.to_datetime(["2024-01-01", "2024-02-01"]),
            "churned": [1, 0]
        })

        scenario, config = auto_detect_and_configure(df, "churned")

        assert scenario is not None
        assert config is not None
        assert hasattr(config, "strategy")

    def test_auto_detect_with_no_timestamps(self):
        df = pd.DataFrame({
            "customer_id": ["A", "B"],
            "value": [100, 200],
            "churned": [1, 0]
        })

        scenario, config = auto_detect_and_configure(df, "churned")

        assert scenario == "synthetic"
        assert config.strategy == TimestampStrategy.SYNTHETIC_INDEX


class TestEdgeCases(TestScenarioDetector):
    def test_empty_dataframe(self, detector):
        df = pd.DataFrame(columns=["customer_id", "churned"])
        scenario, config, result = detector.detect(df, "churned")

        assert scenario == "synthetic"

    def test_dataframe_with_mixed_date_formats(self, detector):
        df = pd.DataFrame({
            "customer_id": ["A", "B", "C"],
            "last_login": ["2024-01-01", "2024/02/01", "01-Mar-2024"],
            "churned": [1, 0, 1]
        })

        scenario, config, result = detector.detect(df, "churned")
        assert result.discovery_report["datetime_columns_found"] >= 1
