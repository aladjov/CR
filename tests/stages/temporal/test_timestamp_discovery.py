import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from customer_retention.stages.temporal.timestamp_discovery import (
    TimestampRole, TimestampCandidate, TimestampDiscoveryResult, TimestampDiscoveryEngine
)


class TestTimestampRole:
    def test_role_values(self):
        assert TimestampRole.FEATURE_TIMESTAMP.value == "feature_timestamp"
        assert TimestampRole.LABEL_TIMESTAMP.value == "label_timestamp"
        assert TimestampRole.ENTITY_CREATED.value == "entity_created"


class TestTimestampCandidate:
    def test_create_candidate(self):
        candidate = TimestampCandidate(
            column_name="last_activity_date",
            role=TimestampRole.FEATURE_TIMESTAMP,
            confidence=0.9,
            coverage=0.95,
            date_range=(datetime(2024, 1, 1), datetime(2024, 12, 31)),
            is_derived=False
        )
        assert candidate.column_name == "last_activity_date"
        assert candidate.role == TimestampRole.FEATURE_TIMESTAMP
        assert candidate.confidence == 0.9


class TestTimestampDiscoveryEngine:
    @pytest.fixture
    def engine(self):
        return TimestampDiscoveryEngine(reference_date=datetime(2024, 6, 1))

    @pytest.fixture
    def df_with_datetime_columns(self):
        return pd.DataFrame({
            "customer_id": ["A", "B", "C"],
            "last_activity_date": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01"]),
            "signup_date": pd.to_datetime(["2023-01-01", "2023-02-01", "2023-03-01"]),
            "value": [100, 200, 300]
        })

    @pytest.fixture
    def df_with_tenure_column(self):
        return pd.DataFrame({
            "customer_id": ["A", "B", "C"],
            "tenure_months": [12, 24, 6],
            "value": [100, 200, 300]
        })

    @pytest.fixture
    def df_without_timestamps(self):
        return pd.DataFrame({
            "customer_id": ["A", "B", "C"],
            "value": [100, 200, 300],  # Small numbers won't be detected as unix timestamps
            "category": ["X", "Y", "Z"],
            "score": [0.5, 0.7, 0.9]  # Floats won't be timestamps
        })

    def test_discover_datetime_columns(self, engine, df_with_datetime_columns):
        result = engine.discover(df_with_datetime_columns)

        assert result.discovery_report["datetime_columns_found"] >= 2
        assert not result.requires_synthetic

    def test_identify_feature_timestamp_by_name(self, engine, df_with_datetime_columns):
        result = engine.discover(df_with_datetime_columns)

        assert result.feature_timestamp is not None
        assert result.feature_timestamp.column_name == "last_activity_date"
        assert result.feature_timestamp.role == TimestampRole.FEATURE_TIMESTAMP

    def test_identify_entity_created(self, engine, df_with_datetime_columns):
        result = engine.discover(df_with_datetime_columns)

        entity_created = [c for c in result.all_candidates if c.role == TimestampRole.ENTITY_CREATED]
        assert len(entity_created) >= 1
        assert entity_created[0].column_name == "signup_date"

    def test_discover_derivable_from_tenure(self, engine, df_with_tenure_column):
        result = engine.discover(df_with_tenure_column)

        assert result.discovery_report["derivable_timestamps_found"] >= 1
        derivable = [c for c in result.derivable_options if "tenure" in c.column_name.lower()]
        assert len(derivable) >= 1

    def test_no_timestamps_returns_synthetic_recommendation(self, engine, df_without_timestamps):
        result = engine.discover(df_without_timestamps)

        assert result.requires_synthetic
        assert "FALLBACK" in result.recommendation or "synthetic" in result.recommendation.lower()

    def test_label_timestamp_derived_from_feature(self, engine, df_with_datetime_columns):
        result = engine.discover(df_with_datetime_columns)

        assert result.label_timestamp is not None
        if result.label_timestamp.is_derived:
            assert "observation window" in result.label_timestamp.notes.lower()

    def test_discovery_report_structure(self, engine, df_with_datetime_columns):
        result = engine.discover(df_with_datetime_columns)
        report = result.discovery_report

        assert "total_columns" in report
        assert "datetime_columns_found" in report
        assert "derivable_timestamps_found" in report
        assert "candidates_by_role" in report
        assert "all_candidates" in report


class TestTimestampDiscoveryEdgeCases:
    @pytest.fixture
    def engine(self):
        return TimestampDiscoveryEngine()

    def test_unix_timestamp_detection(self, engine):
        df = pd.DataFrame({
            "customer_id": ["A", "B", "C"],
            "event_time": [1704067200, 1706745600, 1709251200],
            "value": [100, 200, 300]
        })
        result = engine.discover(df)

        datetime_candidates = [c for c in result.all_candidates if not c.is_derived]
        assert len(datetime_candidates) >= 1

    def test_string_date_detection(self, engine):
        df = pd.DataFrame({
            "customer_id": ["A", "B", "C"],
            "last_login": ["2024-01-01", "2024-02-01", "2024-03-01"],
            "value": [100, 200, 300]
        })
        result = engine.discover(df)

        assert result.discovery_report["datetime_columns_found"] >= 1

    def test_partial_null_datetime_column(self, engine):
        df = pd.DataFrame({
            "customer_id": ["A", "B", "C", "D"],
            "last_activity_date": pd.to_datetime(["2024-01-01", "2024-02-01", None, None]),
            "value": [100, 200, 300, 400]
        })
        result = engine.discover(df)

        candidate = result.feature_timestamp
        assert candidate is not None
        assert candidate.coverage == 0.5


class TestTimestampDiscoveryConfidence:
    @pytest.fixture
    def engine(self):
        return TimestampDiscoveryEngine()

    def test_high_confidence_for_named_feature_timestamp(self, engine):
        df = pd.DataFrame({
            "last_activity_date": pd.to_datetime(["2024-01-01", "2024-02-01"]),
        })
        result = engine.discover(df)

        assert result.feature_timestamp is not None
        assert result.feature_timestamp.confidence >= 0.7

    def test_lower_confidence_for_unknown_role(self, engine):
        df = pd.DataFrame({
            "some_date": pd.to_datetime(["2024-01-01", "2024-02-01"]),
        })
        result = engine.discover(df)

        if result.all_candidates:
            unknown_candidates = [c for c in result.all_candidates if c.role == TimestampRole.UNKNOWN]
            if unknown_candidates:
                assert unknown_candidates[0].confidence < 0.8


class TestLabelTimestampPatterns:
    @pytest.fixture
    def engine(self):
        return TimestampDiscoveryEngine()

    @pytest.mark.parametrize("column_name", [
        "churn_date", "churned_date", "customer_churn_date", "churn_timestamp",
    ])
    def test_churn_date_patterns(self, engine, column_name):
        df = pd.DataFrame({
            "customer_id": ["A", "B", "C"],
            column_name: pd.to_datetime(["2024-01-01", "2024-02-01", None]),
            "last_activity": pd.to_datetime(["2023-12-01", "2024-01-01", "2024-01-15"]),
        })
        result = engine.discover(df)
        label_candidates = [c for c in result.all_candidates if c.role == TimestampRole.LABEL_TIMESTAMP]
        assert any(c.column_name == column_name for c in label_candidates)

    @pytest.mark.parametrize("column_name", [
        "unsubscribe_date", "unsubscribed_date", "unsub_date",
    ])
    def test_unsubscribe_date_patterns(self, engine, column_name):
        df = pd.DataFrame({
            "customer_id": ["A", "B", "C"],
            column_name: pd.to_datetime(["2024-01-01", None, "2024-03-01"]),
            "sent_date": pd.to_datetime(["2023-12-01", "2024-01-01", "2024-01-15"]),
        })
        result = engine.discover(df)
        label_candidates = [c for c in result.all_candidates if c.role == TimestampRole.LABEL_TIMESTAMP]
        assert any(c.column_name == column_name for c in label_candidates)

    @pytest.mark.parametrize("column_name", [
        "cancellation_date", "cancel_date", "cancelled_date",
        "termination_date", "terminate_date", "terminated_date",
    ])
    def test_cancellation_termination_patterns(self, engine, column_name):
        df = pd.DataFrame({
            "customer_id": ["A", "B", "C"],
            column_name: pd.to_datetime(["2024-01-01", "2024-02-01", None]),
        })
        result = engine.discover(df)
        label_candidates = [c for c in result.all_candidates if c.role == TimestampRole.LABEL_TIMESTAMP]
        assert any(c.column_name == column_name for c in label_candidates)

    @pytest.mark.parametrize("column_name", [
        "close_date", "closed_date", "account_close_date", "closure_date",
        "discontinue_date", "discontinued_date", "discontinuation_date",
    ])
    def test_close_discontinue_patterns(self, engine, column_name):
        df = pd.DataFrame({
            "customer_id": ["A", "B", "C"],
            column_name: pd.to_datetime(["2024-01-01", None, "2024-03-01"]),
        })
        result = engine.discover(df)
        label_candidates = [c for c in result.all_candidates if c.role == TimestampRole.LABEL_TIMESTAMP]
        assert any(c.column_name == column_name for c in label_candidates)

    @pytest.mark.parametrize("column_name", [
        "exit_date", "leave_date", "left_date", "end_date",
        "expiry_date", "expiration_date", "expired_date",
    ])
    def test_exit_expiry_patterns(self, engine, column_name):
        df = pd.DataFrame({
            "customer_id": ["A", "B", "C"],
            column_name: pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01"]),
        })
        result = engine.discover(df)
        label_candidates = [c for c in result.all_candidates if c.role == TimestampRole.LABEL_TIMESTAMP]
        assert any(c.column_name == column_name for c in label_candidates)

    @pytest.mark.parametrize("column_name", [
        "outcome_date", "event_date", "target_date", "label_date", "prediction_date",
    ])
    def test_generic_label_patterns(self, engine, column_name):
        df = pd.DataFrame({
            "customer_id": ["A", "B", "C"],
            column_name: pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01"]),
        })
        result = engine.discover(df)
        label_candidates = [c for c in result.all_candidates if c.role == TimestampRole.LABEL_TIMESTAMP]
        assert any(c.column_name == column_name for c in label_candidates)
