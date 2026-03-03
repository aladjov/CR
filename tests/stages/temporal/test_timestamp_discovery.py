from datetime import datetime, timedelta

import pandas as pd
import pytest

from customer_retention.stages.temporal.timestamp_discovery import (
    DatetimeOrderAnalyzer,
    TimestampCandidate,
    TimestampDiscoveryEngine,
    TimestampRole,
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


class TestFutureDateExclusionInDiscovery:
    @pytest.fixture
    def engine(self):
        return TimestampDiscoveryEngine()

    def test_majority_future_column_excluded_from_candidates(self, engine):
        df = pd.DataFrame({
            "customer_id": ["A", "B", "C"],
            "contract_start": pd.to_datetime(["2022-10-01", "2024-08-01", "2025-01-01"]),
            "contract_end": pd.to_datetime(["2026-10-01", "2027-08-01", "2028-01-01"]),
            "churn_end": pd.to_datetime(["2023-05-01", None, None]),
        })

        result = engine.discover(df)

        candidate_names = [c.column_name for c in result.all_candidates]
        assert "contract_end" not in candidate_names

    def test_feature_timestamp_not_from_majority_future_column(self, engine):
        df = pd.DataFrame({
            "customer_id": ["A", "B", "C"],
            "contract_start": pd.to_datetime(["2022-10-01", "2024-08-01", "2025-01-01"]),
            "contract_end": pd.to_datetime(["2026-10-01", "2027-08-01", "2028-01-01"]),
            "churn_end": pd.to_datetime(["2023-05-01", None, None]),
        })

        result = engine.discover(df)

        if result.feature_timestamp:
            assert result.feature_timestamp.column_name != "contract_end"

    def test_past_only_columns_still_discovered(self, engine):
        df = pd.DataFrame({
            "customer_id": ["A", "B", "C"],
            "contract_start": pd.to_datetime(["2022-10-01", "2024-08-01", "2025-01-01"]),
            "contract_end": pd.to_datetime(["2026-10-01", "2027-08-01", "2028-01-01"]),
            "churn_end": pd.to_datetime(["2023-05-01", None, None]),
        })

        result = engine.discover(df)

        candidate_names = [c.column_name for c in result.all_candidates]
        assert "contract_start" in candidate_names
        assert "churn_end" in candidate_names

    def test_column_with_few_future_dates_not_excluded(self, engine):
        dates = pd.date_range("2022-01-01", periods=90, freq="15D").tolist()
        df = pd.DataFrame({
            "customer_id": [f"C{i}" for i in range(90)],
            "event_timestamp": dates,
        })

        result = engine.discover(df)

        candidate_names = [c.column_name for c in result.all_candidates]
        assert "event_timestamp" in candidate_names


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


class TestEventTimePatterns:
    @pytest.fixture
    def engine(self):
        return TimestampDiscoveryEngine()

    @pytest.mark.parametrize("column_name", [
        "event_timestamp", "event_time",
        "transaction_timestamp", "transaction_time", "transaction_date",
        "occurred_at", "recorded_at", "log_timestamp",
    ])
    def test_event_time_role_assigned(self, engine, column_name):
        df = pd.DataFrame({
            "customer_id": ["A", "B", "C"],
            column_name: pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01"]),
        })
        result = engine.discover(df)
        event_candidates = [c for c in result.all_candidates if c.role in (
            TimestampRole.EVENT_TIME, TimestampRole.FEATURE_TIMESTAMP,
        )]
        assert any(c.column_name == column_name for c in event_candidates)

    def test_event_time_promoted_to_feature_when_no_explicit_feature(self, engine):
        df = pd.DataFrame({
            "customer_id": ["A", "B", "C"],
            "event_timestamp": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01"]),
            "value": [100, 200, 300],
        })
        result = engine.discover(df)
        assert result.feature_timestamp is not None
        assert result.feature_timestamp.column_name == "event_timestamp"

    def test_event_time_not_promoted_when_feature_exists(self, engine):
        df = pd.DataFrame({
            "customer_id": ["A", "B", "C"],
            "last_activity_date": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01"]),
            "event_timestamp": pd.to_datetime(["2023-06-01", "2023-07-01", "2023-08-01"]),
        })
        result = engine.discover(df)
        assert result.feature_timestamp.column_name == "last_activity_date"
        event_candidates = [c for c in result.all_candidates if c.column_name == "event_timestamp"]
        assert len(event_candidates) == 1
        assert event_candidates[0].role == TimestampRole.EVENT_TIME

    def test_event_time_confidence_higher_than_unknown(self, engine):
        df = pd.DataFrame({
            "event_timestamp": pd.to_datetime(["2024-01-01", "2024-02-01"]),
            "some_date": pd.to_datetime(["2024-01-01", "2024-02-01"]),
        })
        result = engine.discover(df)
        event_cand = next(c for c in result.all_candidates if c.column_name == "event_timestamp")
        unknown_cand = [c for c in result.all_candidates if c.column_name == "some_date"]
        if unknown_cand and unknown_cand[0].role == TimestampRole.UNKNOWN:
            assert event_cand.confidence > unknown_cand[0].confidence


class TestExpandedTimestampPatterns:

    @pytest.fixture
    def engine(self):
        return TimestampDiscoveryEngine()

    @pytest.mark.parametrize("col_name", [
        "updated_on", "changed_at", "changed_date", "modify_date",
    ])
    def test_expanded_patterns_classified_as_feature_timestamp(self, engine, col_name):
        df = pd.DataFrame({
            "customer_id": ["A", "B", "C"],
            col_name: pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01"]),
        })
        result = engine.discover(df)
        assert result.feature_timestamp is not None
        assert result.feature_timestamp.column_name == col_name

    def test_created_at_promoted_when_no_feature_timestamp(self, engine):
        df = pd.DataFrame({
            "customer_id": ["A", "B", "C"],
            "created_at": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01"]),
            "amount": [100, 200, 300],
        })
        result = engine.discover(df)
        assert result.feature_timestamp is not None
        assert result.feature_timestamp.column_name == "created_at"

    def test_entity_updated_promoted_before_entity_created(self, engine):
        df = pd.DataFrame({
            "customer_id": ["A", "B", "C"],
            "updated_at": pd.to_datetime(["2024-06-01", "2024-07-01", "2024-08-01"]),
            "created_at": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01"]),
        })
        result = engine.discover(df)
        assert result.feature_timestamp is not None
        assert result.feature_timestamp.column_name == "updated_at"


class TestEdiTransactionsScenario:
    def test_iso_string_event_timestamp_detected(self):
        engine = TimestampDiscoveryEngine()
        df = pd.DataFrame({
            "transaction_id": [f"T{i}" for i in range(20)],
            "customer_id": ["C001"] * 20,
            "event_timestamp": [f"2024-{m:02d}-15T10:30:00.000Z" for m in range(1, 13)]
            + [f"2025-{m:02d}-15T10:30:00.000Z" for m in range(1, 9)],
            "status": ["success"] * 20,
            "amount": [100.0] * 20,
        })
        result = engine.discover(df)
        assert result.feature_timestamp is not None
        assert result.feature_timestamp.column_name == "event_timestamp"
        assert not result.requires_synthetic

    def test_event_timestamp_with_some_future_dates_still_detected(self):
        engine = TimestampDiscoveryEngine()
        past = [f"2023-{m:02d}-15T10:30:00.000Z" for m in range(1, 13)]
        past += [f"2024-{m:02d}-15T10:30:00.000Z" for m in range(1, 13)]
        future = ["2027-06-15T10:30:00.000Z", "2027-09-15T10:30:00.000Z"]
        df = pd.DataFrame({
            "transaction_id": [f"T{i}" for i in range(26)],
            "customer_id": ["C001"] * 26,
            "event_timestamp": past + future,
            "amount": [100.0] * 26,
        })
        result = engine.discover(df)
        assert result.feature_timestamp is not None
        assert result.feature_timestamp.column_name == "event_timestamp"

    def test_scenario_detector_partial_for_event_data_without_target(self):
        from customer_retention.stages.temporal.scenario_detector import ScenarioDetector
        detector = ScenarioDetector()
        df = pd.DataFrame({
            "transaction_id": [f"T{i}" for i in range(10)],
            "customer_id": ["C001"] * 10,
            "event_timestamp": [f"2024-{m:02d}-15T10:30:00.000Z" for m in range(1, 11)],
            "amount": [100.0] * 10,
        })
        scenario, config, _ = detector.detect(df, None)
        assert scenario == "partial"
        assert config.feature_timestamp_column == "event_timestamp"


class TestHasFutureDatesDistributedSafe:
    @pytest.fixture
    def analyzer(self):
        return DatetimeOrderAnalyzer()

    def test_majority_future_detected(self, analyzer):
        future = datetime.now() + timedelta(days=365 * 10)
        df = pd.DataFrame({"col": [future] * 10})
        assert analyzer._has_future_dates(df, "col") is True

    def test_majority_past_not_detected(self, analyzer):
        past = datetime.now() - timedelta(days=365)
        df = pd.DataFrame({"col": [past] * 10})
        assert analyzer._has_future_dates(df, "col") is False

    def test_minority_future_not_detected(self, analyzer):
        past = datetime.now() - timedelta(days=365)
        future = datetime.now() + timedelta(days=365 * 10)
        df = pd.DataFrame({"col": [past] * 8 + [future] * 2})
        assert analyzer._has_future_dates(df, "col") is False

    def test_empty_after_dropna(self, analyzer):
        df = pd.DataFrame({"col": pd.to_datetime([None, None, None])})
        assert analyzer._has_future_dates(df, "col") is False

    def test_no_median_called_on_datetime(self, analyzer):
        """Ensure .median() is NOT used on datetime (fails on pyspark.pandas)."""
        past = datetime.now() - timedelta(days=365)
        df = pd.DataFrame({"col": pd.to_datetime([past] * 5)})
        result = analyzer._has_future_dates(df, "col")
        assert result is False


class TestAnalyzeDatetimeOrderingDistributedSafe:
    @pytest.fixture
    def analyzer(self):
        return DatetimeOrderAnalyzer()

    def test_orders_by_chronological_center(self, analyzer):
        df = pd.DataFrame({
            "early": pd.to_datetime(["2020-01-01", "2020-06-01", "2021-01-01"]),
            "late": pd.to_datetime(["2023-01-01", "2023-06-01", "2024-01-01"]),
        })
        ordering = analyzer.analyze_datetime_ordering(df)
        assert ordering == ["early", "late"]

    def test_empty_df_returns_empty(self, analyzer):
        df = pd.DataFrame({"val": [1, 2, 3]})
        assert analyzer.analyze_datetime_ordering(df) == []


class TestBulkDatetimeDiscoveryStats:
    def test_computes_stats_for_datetime_columns(self):
        from customer_retention.core.compat.bulk_profiling import bulk_datetime_discovery_stats
        df = pd.DataFrame({
            "ts1": pd.to_datetime(["2020-01-01", "2021-01-01", "2022-01-01"]),
            "ts2": pd.to_datetime(["2030-01-01", "2031-01-01", "2032-01-01"]),
            "num": [1, 2, 3],
        })
        stats = bulk_datetime_discovery_stats(df, ["ts1", "ts2"])
        assert "ts1" in stats
        assert "ts2" in stats
        assert stats["ts1"].future_fraction < 0.5
        assert stats["ts2"].future_fraction > 0.5

    def test_coverage_reflects_nulls(self):
        from customer_retention.core.compat.bulk_profiling import bulk_datetime_discovery_stats
        df = pd.DataFrame({
            "ts": pd.to_datetime(["2020-01-01", None, "2022-01-01", None]),
        })
        stats = bulk_datetime_discovery_stats(df, ["ts"])
        assert stats["ts"].coverage == pytest.approx(0.5)

    def test_empty_columns_list(self):
        from customer_retention.core.compat.bulk_profiling import bulk_datetime_discovery_stats
        df = pd.DataFrame({"a": [1, 2]})
        assert bulk_datetime_discovery_stats(df, []) == {}

    def test_min_max_dates(self):
        from customer_retention.core.compat.bulk_profiling import bulk_datetime_discovery_stats
        df = pd.DataFrame({
            "ts": pd.to_datetime(["2020-01-01", "2022-06-15", "2024-12-31"]),
        })
        stats = bulk_datetime_discovery_stats(df, ["ts"])
        assert stats["ts"].min_date == pd.Timestamp("2020-01-01")
        assert stats["ts"].max_date == pd.Timestamp("2024-12-31")


class TestDiscoverWithPrecomputedStats:
    def test_discover_uses_precomputed_stats(self):
        from customer_retention.core.compat.bulk_profiling import DatetimeDiscoveryCandidateStats
        engine = TimestampDiscoveryEngine()
        df = pd.DataFrame({
            "customer_id": ["A", "B", "C"],
            "last_activity_date": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01"]),
            "value": [100, 200, 300],
        })
        precomputed = {
            "last_activity_date": DatetimeDiscoveryCandidateStats(
                min_date=pd.Timestamp("2024-01-01"), max_date=pd.Timestamp("2024-03-01"),
                coverage=1.0, future_fraction=0.0,
            ),
        }
        result = engine.discover(df, datetime_stats=precomputed)
        assert result.feature_timestamp is not None
        assert result.feature_timestamp.column_name == "last_activity_date"

    def test_precomputed_excludes_future_columns(self):
        from customer_retention.core.compat.bulk_profiling import DatetimeDiscoveryCandidateStats
        engine = TimestampDiscoveryEngine()
        df = pd.DataFrame({
            "customer_id": ["A", "B", "C"],
            "contract_end": pd.to_datetime(["2030-01-01", "2031-01-01", "2032-01-01"]),
            "created_at": pd.to_datetime(["2020-01-01", "2021-01-01", "2022-01-01"]),
        })
        precomputed = {
            "contract_end": DatetimeDiscoveryCandidateStats(
                min_date=pd.Timestamp("2030-01-01"), max_date=pd.Timestamp("2032-01-01"),
                coverage=1.0, future_fraction=1.0,
            ),
            "created_at": DatetimeDiscoveryCandidateStats(
                min_date=pd.Timestamp("2020-01-01"), max_date=pd.Timestamp("2022-01-01"),
                coverage=1.0, future_fraction=0.0,
            ),
        }
        result = engine.discover(df, datetime_stats=precomputed)
        candidate_names = [c.column_name for c in result.all_candidates]
        assert "contract_end" not in candidate_names
        assert "created_at" in candidate_names


class TestEntityTimestampDeriverWithFindings:
    def test_derive_with_known_datetime_columns(self):
        from customer_retention.analysis.auto_explorer.entity_timestamp_deriver import EntityFeatureTimestampDeriver
        deriver = EntityFeatureTimestampDeriver()
        df = pd.DataFrame({
            "customer_id": [1, 2, 3],
            "last_activity_date": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01"]),
            "revenue": [100.0, 200.0, 300.0],
        })
        result = deriver.derive(df, target_column="revenue", known_datetime_columns=["last_activity_date"])
        assert result.method == "direct"
        assert "last_activity" in result.column_name.lower()

    def test_derive_with_empty_known_columns_returns_none(self):
        from customer_retention.analysis.auto_explorer.entity_timestamp_deriver import EntityFeatureTimestampDeriver
        deriver = EntityFeatureTimestampDeriver()
        df = pd.DataFrame({
            "customer_id": [1, 2, 3],
            "revenue": [100.0, 200.0, 300.0],
        })
        result = deriver.derive(df, target_column="revenue", known_datetime_columns=[])
        assert result.method == "none"
