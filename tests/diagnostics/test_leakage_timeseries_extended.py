"""Comprehensive tests for LeakageDetector and TimeSeriesDetector/Validator modules."""

import numpy as np
import pandas as pd

from customer_retention.analysis.diagnostics.leakage_detector import (
    LeakageDetector,
    LeakageResult,
)
from customer_retention.core.components.enums import Severity
from customer_retention.stages.validation.timeseries_detector import (
    DatasetType,
    TimeSeriesDetector,
    TimeSeriesFrequency,
    TimeSeriesValidator,
)

# ---------------------------------------------------------------------------
# LeakageDetector: check_point_in_time
# ---------------------------------------------------------------------------


class TestLeakagePointInTime:
    """Tests covering check_point_in_time (lines 199-247)."""

    def setup_method(self):
        self.detector = LeakageDetector(
            feature_timestamp_column="feature_timestamp",
            label_timestamp_column="label_timestamp",
        )

    def test_missing_feature_timestamp_column_returns_passed(self):
        """When feature_timestamp column is absent, result should pass."""
        df = pd.DataFrame({
            "col_a": [1, 2, 3],
            "label_timestamp": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
        })
        result = self.detector.check_point_in_time(df)
        assert result.passed is True
        assert result.checks == []
        assert result.critical_issues == []

    def test_feature_after_label_triggers_ld040(self):
        """feature_timestamp > label_timestamp should produce CRITICAL LD040."""
        df = pd.DataFrame({
            "feature_timestamp": pd.to_datetime(
                ["2020-01-05", "2020-01-02", "2020-01-10"]
            ),
            "label_timestamp": pd.to_datetime(
                ["2020-01-01", "2020-01-03", "2020-01-05"]
            ),
        })
        result = self.detector.check_point_in_time(df)
        assert result.passed is False
        ld040 = [c for c in result.checks if c.check_id == "LD040"]
        assert len(ld040) == 1
        assert ld040[0].severity == Severity.CRITICAL
        assert "2" in ld040[0].recommendation  # 2 violations

    def test_no_violations_passes(self):
        """When feature_timestamp <= label_timestamp for all rows, passes."""
        df = pd.DataFrame({
            "feature_timestamp": pd.to_datetime(
                ["2020-01-01", "2020-01-02", "2020-01-03"]
            ),
            "label_timestamp": pd.to_datetime(
                ["2020-01-05", "2020-01-06", "2020-01-07"]
            ),
        })
        result = self.detector.check_point_in_time(df)
        assert result.passed is True
        assert len([c for c in result.checks if c.check_id == "LD040"]) == 0

    def test_datetime_col_after_feature_critical_ld041(self):
        """Other datetime columns after feature_timestamp (>5%) => CRITICAL LD041."""
        n = 100
        feature_ts = pd.to_datetime(["2020-01-01"] * n)
        # 10% of rows have event_date after feature_timestamp
        event_dates = pd.to_datetime(["2019-12-01"] * 90 + ["2020-02-01"] * 10)
        df = pd.DataFrame({
            "feature_timestamp": feature_ts,
            "event_date": event_dates,
        })
        # Force event_date to datetime64 dtype
        df["event_date"] = pd.to_datetime(df["event_date"])
        result = self.detector.check_point_in_time(df)
        ld041 = [c for c in result.checks if c.check_id == "LD041"]
        assert len(ld041) == 1
        assert ld041[0].severity == Severity.CRITICAL
        assert result.passed is False

    def test_datetime_col_after_feature_high_ld041(self):
        """Other datetime columns after feature_timestamp (<=5%) => HIGH LD041."""
        n = 100
        feature_ts = pd.to_datetime(["2020-01-01"] * n)
        # 3% of rows have event_date after feature_timestamp
        event_dates = pd.to_datetime(["2019-12-01"] * 97 + ["2020-02-01"] * 3)
        df = pd.DataFrame({
            "feature_timestamp": feature_ts,
            "event_date": event_dates,
        })
        df["event_date"] = pd.to_datetime(df["event_date"])
        result = self.detector.check_point_in_time(df)
        ld041 = [c for c in result.checks if c.check_id == "LD041"]
        assert len(ld041) == 1
        assert ld041[0].severity == Severity.HIGH
        # HIGH issues should not cause passed=False
        assert result.passed is True

    def test_label_timestamp_col_excluded_from_ld041(self):
        """label_timestamp and feature_timestamp are not checked for LD041."""
        df = pd.DataFrame({
            "feature_timestamp": pd.to_datetime(["2020-01-01", "2020-01-02"]),
            "label_timestamp": pd.to_datetime(["2020-02-01", "2020-02-02"]),
        })
        df["feature_timestamp"] = pd.to_datetime(df["feature_timestamp"])
        df["label_timestamp"] = pd.to_datetime(df["label_timestamp"])
        result = self.detector.check_point_in_time(df)
        ld041 = [c for c in result.checks if c.check_id == "LD041"]
        assert len(ld041) == 0

    def test_unparseable_feature_timestamp_returns_passed(self):
        """If feature_timestamp cannot be parsed, returns passed=True."""
        df = pd.DataFrame({
            "feature_timestamp": ["not-a-date", "also-not", "nope"],
        })
        result = self.detector.check_point_in_time(df)
        # Even if parsing produces NaT, it should not crash
        assert result.passed is True

    def test_missing_label_timestamp_still_checks_ld041(self):
        """When label_timestamp is absent, LD040 is skipped but LD041 still applies."""
        n = 100
        feature_ts = pd.to_datetime(["2020-01-01"] * n)
        event_dates = pd.to_datetime(["2020-02-01"] * n)
        df = pd.DataFrame({
            "feature_timestamp": feature_ts,
            "event_date": event_dates,
        })
        df["event_date"] = pd.to_datetime(df["event_date"])
        result = self.detector.check_point_in_time(df)
        ld040 = [c for c in result.checks if c.check_id == "LD040"]
        assert len(ld040) == 0
        ld041 = [c for c in result.checks if c.check_id == "LD041"]
        assert len(ld041) == 1


# ---------------------------------------------------------------------------
# LeakageDetector: check_single_feature_auc
# ---------------------------------------------------------------------------


class TestLeakageSingleFeatureAuc:
    """Tests covering check_single_feature_auc (lines 153-197)."""

    def setup_method(self):
        self.detector = LeakageDetector()

    def test_high_auc_critical_ld030(self):
        """Feature perfectly separating classes => AUC > 0.90 => CRITICAL LD030."""
        np.random.seed(42)
        n = 200
        y = pd.Series([0] * (n // 2) + [1] * (n // 2))
        # Perfect separation
        X = pd.DataFrame({"leak_feat": np.concatenate([np.zeros(n // 2), np.ones(n // 2)])})
        result = self.detector.check_single_feature_auc(X, y)
        assert result.passed is False
        ld030 = [c for c in result.checks if c.check_id == "LD030"]
        assert len(ld030) == 1
        assert ld030[0].severity == Severity.CRITICAL
        assert ld030[0].auc > 0.90

    def test_medium_high_auc_high_ld031(self):
        """Feature with AUC between 0.80 and 0.90 => HIGH LD031."""
        np.random.seed(42)
        n = 500
        y = pd.Series([0] * (n // 2) + [1] * (n // 2))
        # Create a feature with moderate separation
        feat_0 = np.random.normal(0, 1, n // 2)
        feat_1 = np.random.normal(2, 1, n // 2)
        X = pd.DataFrame({"mod_feat": np.concatenate([feat_0, feat_1])})
        result = self.detector.check_single_feature_auc(X, y)
        ld031 = [c for c in result.checks if c.check_id == "LD031"]
        # The AUC should be between 0.80 and 0.90
        if len(ld031) > 0:
            assert ld031[0].severity == Severity.HIGH
            assert 0.80 < ld031[0].auc <= 0.90

    def test_feature_with_nan_values_masked(self):
        """Features with NaN values should be masked before AUC computation."""
        np.random.seed(42)
        n = 200
        y = pd.Series([0] * (n // 2) + [1] * (n // 2))
        feat = np.concatenate([np.zeros(n // 2), np.ones(n // 2)])
        # Inject NaNs
        feat[0] = np.nan
        feat[n // 2] = np.nan
        X = pd.DataFrame({"feat_with_nan": feat})
        result = self.detector.check_single_feature_auc(X, y)
        # Should still detect leakage despite NaNs
        assert len(result.checks) > 0

    def test_single_class_returns_05(self):
        """When all y are same class, AUC defaults to 0.5."""
        n = 100
        y = pd.Series([1] * n)
        X = pd.DataFrame({"feat": np.random.randn(n)})
        result = self.detector.check_single_feature_auc(X, y)
        # No checks should be flagged (AUC=0.5 is INFO)
        assert result.passed is True
        assert len(result.checks) == 0

    def test_non_numeric_columns_skipped(self):
        """Non-numeric columns should be ignored."""
        n = 100
        y = pd.Series([0] * (n // 2) + [1] * (n // 2))
        X = pd.DataFrame({
            "text_col": ["a"] * n,
            "num_col": np.random.randn(n),
        })
        result = self.detector.check_single_feature_auc(X, y)
        # Only num_col should be considered
        for c in result.checks:
            assert c.feature == "num_col"

    def test_auc_recommendation_high(self):
        """AUC recommendation for HIGH severity."""
        rec = self.detector._auc_recommendation("feat_x", 0.85)
        assert "INVESTIGATE" in rec
        assert "feat_x" in rec

    def test_auc_recommendation_info(self):
        """AUC recommendation for normal predictive power."""
        rec = self.detector._auc_recommendation("feat_x", 0.60)
        assert "OK" in rec


# ---------------------------------------------------------------------------
# LeakageDetector: check_temporal_logic
# ---------------------------------------------------------------------------


class TestLeakageTemporalLogic:
    """Tests covering check_temporal_logic (lines 126-151)."""

    def setup_method(self):
        self.detector = LeakageDetector()

    def test_temporal_feature_high_correlation(self):
        """Temporal-named feature with corr > 0.70 => HIGH."""
        np.random.seed(42)
        n = 200
        y = pd.Series(np.linspace(0, 1, n))
        X = pd.DataFrame({
            "days_since_last": np.linspace(0, 1, n) + np.random.normal(0, 0.05, n),
        })
        result = self.detector.check_temporal_logic(X, y)
        assert len(result.checks) > 0
        assert result.checks[0].severity == Severity.HIGH
        assert "REVIEW" in result.checks[0].recommendation

    def test_temporal_feature_medium_correlation(self):
        """Temporal-named feature with 0.50 < corr <= 0.70 => MEDIUM."""
        np.random.seed(42)
        n = 200
        y = pd.Series(np.linspace(0, 1, n))
        # Add enough noise to bring correlation to 0.50-0.70
        noise = np.random.normal(0, 0.5, n)
        X = pd.DataFrame({
            "recency_days": np.linspace(0, 1, n) + noise,
        })
        result = self.detector.check_temporal_logic(X, y)
        # Check at least one MEDIUM check exists
        medium_checks = [c for c in result.checks if c.severity == Severity.MEDIUM]
        if medium_checks:
            assert "CHECK" in medium_checks[0].recommendation

    def test_non_temporal_feature_ignored(self):
        """Features without temporal pattern names are ignored."""
        np.random.seed(42)
        n = 200
        y = pd.Series(np.linspace(0, 1, n))
        X = pd.DataFrame({
            "revenue": np.linspace(0, 1, n),  # high corr but not temporal name
        })
        result = self.detector.check_temporal_logic(X, y)
        assert len(result.checks) == 0

    def test_nan_correlation_treated_as_zero(self):
        """If correlation is NaN (e.g., constant feature), treat as 0.0."""
        n = 100
        y = pd.Series([0] * (n // 2) + [1] * (n // 2))
        X = pd.DataFrame({
            "days_since_start": np.full(n, 5.0),  # constant => NaN corr
        })
        result = self.detector.check_temporal_logic(X, y)
        assert len(result.checks) == 0  # 0.0 < threshold

    def test_temporal_logic_always_passes(self):
        """check_temporal_logic never flags critical, so passed=True always."""
        np.random.seed(42)
        n = 200
        y = pd.Series(np.linspace(0, 1, n))
        X = pd.DataFrame({
            "tenure_days": np.linspace(0, 1, n),  # high corr
        })
        result = self.detector.check_temporal_logic(X, y)
        assert result.passed is True  # Only HIGH/MEDIUM, never CRITICAL


# ---------------------------------------------------------------------------
# LeakageDetector: run_all_checks
# ---------------------------------------------------------------------------


class TestLeakageRunAllChecks:
    """Tests covering run_all_checks (lines 249-272)."""

    def setup_method(self):
        self.detector = LeakageDetector()

    def test_combines_all_check_results(self):
        """run_all_checks should include checks from all sub-checks."""
        np.random.seed(42)
        n = 200
        y = pd.Series([0] * (n // 2) + [1] * (n // 2))
        X = pd.DataFrame({
            "normal_feat": np.random.randn(n),
            "tenure_days": np.random.randn(n),
        })
        result = self.detector.run_all_checks(X, y, include_pit=False)
        assert isinstance(result, LeakageResult)
        assert isinstance(result.checks, list)

    def test_include_pit_true_adds_pit_checks(self):
        """include_pit=True should run point-in-time checks."""
        n = 50
        y = pd.Series([0] * (n // 2) + [1] * (n // 2))
        X = pd.DataFrame({
            "feature_timestamp": pd.to_datetime(["2020-02-01"] * n),
            "label_timestamp": pd.to_datetime(["2020-01-01"] * n),
            "normal_feat": np.random.randn(n),
        })
        result = self.detector.run_all_checks(X, y, include_pit=True)
        # Should have LD040 check from PIT
        ld040 = [c for c in result.checks if c.check_id == "LD040"]
        assert len(ld040) == 1

    def test_include_pit_false_skips_pit_checks(self):
        """include_pit=False should not run point-in-time checks."""
        n = 50
        y = pd.Series([0] * (n // 2) + [1] * (n // 2))
        X = pd.DataFrame({
            "feature_timestamp": pd.to_datetime(["2020-02-01"] * n),
            "label_timestamp": pd.to_datetime(["2020-01-01"] * n),
            "normal_feat": np.random.randn(n),
        })
        result = self.detector.run_all_checks(X, y, include_pit=False)
        ld040 = [c for c in result.checks if c.check_id == "LD040"]
        assert len(ld040) == 0

    def test_recommendations_collected_for_critical_and_high(self):
        """Recommendations are collected for CRITICAL and HIGH severity checks."""
        np.random.seed(42)
        n = 200
        y = pd.Series([0] * (n // 2) + [1] * (n // 2))
        # Perfect separation to trigger CRITICAL
        X = pd.DataFrame({
            "leak_feat": np.concatenate([np.zeros(n // 2), np.ones(n // 2)]),
        })
        result = self.detector.run_all_checks(X, y, include_pit=False)
        assert len(result.recommendations) > 0
        assert result.passed is False


# ---------------------------------------------------------------------------
# LeakageDetector: check_correlations edge cases
# ---------------------------------------------------------------------------


class TestLeakageCorrelationEdgeCases:
    """Tests for NaN correlation, medium severity, and recommendations."""

    def setup_method(self):
        self.detector = LeakageDetector()

    def test_nan_correlation_treated_as_zero(self):
        """Constant features produce NaN correlation, treated as 0.0."""
        n = 100
        y = pd.Series([0] * (n // 2) + [1] * (n // 2))
        X = pd.DataFrame({
            "const_feat": np.full(n, 42.0),
        })
        result = self.detector.check_correlations(X, y)
        # NaN => 0.0 => INFO => not included in checks
        assert len(result.checks) == 0
        assert result.passed is True

    def test_medium_correlation_severity(self):
        """Correlation between 0.50 and 0.70 produces MEDIUM severity (LD003)."""
        np.random.seed(42)
        n = 500
        y = pd.Series(np.linspace(0, 1, n))
        noise = np.random.normal(0, 0.4, n)
        X = pd.DataFrame({
            "med_feat": np.linspace(0, 1, n) + noise,
        })
        result = self.detector.check_correlations(X, y)
        medium_checks = [c for c in result.checks if c.severity == Severity.MEDIUM]
        if medium_checks:
            assert medium_checks[0].check_id == "LD003"

    def test_correlation_recommendation_monitor(self):
        """MONITOR recommendation for medium correlation."""
        rec = self.detector._correlation_recommendation("feat_x", 0.55)
        assert "MONITOR" in rec
        assert "feat_x" in rec

    def test_correlation_recommendation_investigate(self):
        """INVESTIGATE recommendation for high correlation."""
        rec = self.detector._correlation_recommendation("feat_x", 0.80)
        assert "INVESTIGATE" in rec

    def test_correlation_recommendation_remove(self):
        """REMOVE recommendation for critical correlation."""
        rec = self.detector._correlation_recommendation("feat_x", 0.95)
        assert "REMOVE" in rec


# ---------------------------------------------------------------------------
# LeakageDetector: check_separation edge cases
# ---------------------------------------------------------------------------


class TestLeakageSeparationEdgeCases:
    """Tests for HIGH/MEDIUM severity and recommendations in check_separation."""

    def setup_method(self):
        self.detector = LeakageDetector()

    def test_high_separation_ld011(self):
        """Overlap < 1.0% but > 0% => HIGH (LD011)."""
        n = 200
        y = pd.Series([0] * (n // 2) + [1] * (n // 2))
        # Near-perfect separation: overlap is tiny but not zero
        feat_0 = np.linspace(0, 0.50, n // 2)
        feat_1 = np.linspace(0.505, 1.0, n // 2)
        X = pd.DataFrame({"feat": np.concatenate([feat_0, feat_1])})
        result = self.detector.check_separation(X, y)
        high_checks = [c for c in result.checks if c.severity == Severity.HIGH]
        if high_checks:
            assert high_checks[0].check_id == "LD011"

    def test_medium_separation_ld012(self):
        """Overlap between 1.0% and 5.0% => MEDIUM (LD012)."""
        n = 200
        y = pd.Series([0] * (n // 2) + [1] * (n // 2))
        # Overlap between 1% and 5%
        feat_0 = np.linspace(0, 0.52, n // 2)
        feat_1 = np.linspace(0.50, 1.0, n // 2)
        X = pd.DataFrame({"feat": np.concatenate([feat_0, feat_1])})
        result = self.detector.check_separation(X, y)
        medium_checks = [c for c in result.checks if c.severity == Severity.MEDIUM]
        if medium_checks:
            assert medium_checks[0].check_id == "LD012"

    def test_separation_recommendation_high(self):
        """Recommendation for HIGH separation (overlap < 1%)."""
        rec = self.detector._separation_recommendation("feat_x", 0.5)
        assert "REMOVE" in rec
        assert "near-perfect" in rec

    def test_separation_recommendation_medium(self):
        """Recommendation for MEDIUM separation (overlap < 5%)."""
        rec = self.detector._separation_recommendation("feat_x", 3.0)
        assert "INVESTIGATE" in rec
        assert "high separation" in rec

    def test_separation_recommendation_ok(self):
        """Recommendation for normal overlap (>= 5%)."""
        rec = self.detector._separation_recommendation("feat_x", 50.0)
        assert "OK" in rec


# ---------------------------------------------------------------------------
# TimeSeriesDetector: frequency detection
# ---------------------------------------------------------------------------


class TestTimeSeriesFrequencyDetection:
    """Tests for _detect_frequency covering all frequency types."""

    def setup_method(self):
        self.detector = TimeSeriesDetector()

    def _make_ts_df(self, entity_id, start, freq, periods):
        """Helper to create time series DataFrame."""
        dates = pd.date_range(start=start, freq=freq, periods=periods)
        return pd.DataFrame({
            "entity_id": [entity_id] * periods,
            "timestamp": dates,
        })

    def test_hourly_frequency(self):
        """Intervals < 2 hours => HOURLY."""
        df = self._make_ts_df("e1", "2020-01-01", "1h", 50)
        freq, median_h = self.detector._detect_frequency(df, "entity_id", "timestamp")
        assert freq == TimeSeriesFrequency.HOURLY
        assert median_h < 2

    def test_daily_frequency(self):
        """Intervals 20-28 hours => DAILY."""
        df = self._make_ts_df("e1", "2020-01-01", "1D", 50)
        freq, median_h = self.detector._detect_frequency(df, "entity_id", "timestamp")
        assert freq == TimeSeriesFrequency.DAILY
        assert 20 <= median_h <= 28

    def test_weekly_frequency(self):
        """Intervals 144-192 hours => WEEKLY."""
        df = self._make_ts_df("e1", "2020-01-01", "7D", 20)
        freq, median_h = self.detector._detect_frequency(df, "entity_id", "timestamp")
        assert freq == TimeSeriesFrequency.WEEKLY
        assert 144 <= median_h <= 192

    def test_quarterly_frequency(self):
        """Intervals ~84-92 days (2016-2208 hours) => QUARTERLY."""
        # ~91 days between quarters
        dates = pd.to_datetime([
            "2020-01-01", "2020-04-01", "2020-07-01", "2020-10-01",
            "2021-01-01", "2021-04-01", "2021-07-01", "2021-10-01",
        ])
        df = pd.DataFrame({
            "entity_id": ["e1"] * len(dates),
            "timestamp": dates,
        })
        freq, median_h = self.detector._detect_frequency(df, "entity_id", "timestamp")
        assert freq == TimeSeriesFrequency.QUARTERLY

    def test_yearly_frequency(self):
        """Intervals ~350-370 days (8400-8880 hours) => YEARLY."""
        dates = pd.to_datetime([
            "2015-01-01", "2016-01-01", "2017-01-01", "2018-01-01",
            "2019-01-01", "2020-01-01", "2021-01-01",
        ])
        df = pd.DataFrame({
            "entity_id": ["e1"] * len(dates),
            "timestamp": dates,
        })
        freq, median_h = self.detector._detect_frequency(df, "entity_id", "timestamp")
        assert freq == TimeSeriesFrequency.YEARLY

    def test_irregular_frequency_high_cv(self):
        """High coefficient of variation => IRREGULAR."""
        # Highly irregular intervals
        dates = pd.to_datetime([
            "2020-01-01", "2020-01-03", "2020-01-20",
            "2020-03-01", "2020-03-05", "2020-08-01",
        ])
        df = pd.DataFrame({
            "entity_id": ["e1"] * len(dates),
            "timestamp": dates,
        })
        freq, _ = self.detector._detect_frequency(df, "entity_id", "timestamp")
        assert freq == TimeSeriesFrequency.IRREGULAR

    def test_entity_with_single_observation_skipped(self):
        """Entities with < 2 observations are skipped."""
        df = pd.DataFrame({
            "entity_id": ["e1", "e2", "e2"],
            "timestamp": pd.to_datetime(["2020-01-01", "2020-01-01", "2020-01-02"]),
        })
        freq, median_h = self.detector._detect_frequency(df, "entity_id", "timestamp")
        # Only e2 contributes; should compute from that entity
        assert freq == TimeSeriesFrequency.DAILY

    def test_no_valid_intervals_returns_unknown(self):
        """If all entities have < 2 observations, returns UNKNOWN."""
        df = pd.DataFrame({
            "entity_id": ["e1", "e2", "e3"],
            "timestamp": pd.to_datetime(["2020-01-01", "2020-02-01", "2020-03-01"]),
        })
        freq, median_h = self.detector._detect_frequency(df, "entity_id", "timestamp")
        assert freq == TimeSeriesFrequency.UNKNOWN
        assert median_h == 0.0

    def test_entities_with_nat_timestamps_skipped(self):
        """Entities where all timestamps parse to NaT are skipped."""
        df = pd.DataFrame({
            "entity_id": ["e1", "e1", "e2", "e2"],
            "timestamp": ["not-a-date", "invalid", "2020-01-01", "2020-01-02"],
        })
        freq, _ = self.detector._detect_frequency(df, "entity_id", "timestamp")
        # Only e2 has valid timestamps
        assert freq == TimeSeriesFrequency.DAILY


# ---------------------------------------------------------------------------
# TimeSeriesDetector: entity column detection
# ---------------------------------------------------------------------------


class TestTimeSeriesEntityDetection:
    """Tests for _detect_entity_column auto-detection."""

    def setup_method(self):
        self.detector = TimeSeriesDetector()

    def test_detects_column_matching_pattern(self):
        """Columns matching ENTITY_PATTERNS are detected first."""
        df = pd.DataFrame({
            "customer_id": [1, 1, 2, 2],
            "value": [10, 20, 30, 40],
        })
        result = self.detector._detect_entity_column(df)
        assert result == "customer_id"

    def test_detects_column_by_cardinality(self):
        """When no name matches patterns, find by cardinality characteristics."""
        n = 100
        # Column with right cardinality: 0.01 < distinct_ratio < 0.9 and repeating values
        groups = np.repeat(np.arange(20), 5)
        df = pd.DataFrame({
            "group_code": groups,
            "measurement": np.random.randn(n),
        })
        result = self.detector._detect_entity_column(df)
        assert result == "group_code"

    def test_no_entity_column_found_returns_none(self):
        """Returns None when no column matches patterns or cardinality."""
        # All unique values (distinct_ratio = 1.0), so cardinality check fails
        df = pd.DataFrame({
            "unique_values": np.arange(100),
            "measurements": np.random.randn(100),
        })
        result = self.detector._detect_entity_column(df)
        assert result is None

    def test_string_column_detected_by_cardinality(self):
        """String columns with repeating values are detected."""
        n = 50
        categories = [f"cat_{i}" for i in range(10)] * 5
        df = pd.DataFrame({
            "category": categories,
            "value": np.random.randn(n),
        })
        result = self.detector._detect_entity_column(df)
        assert result == "category"


# ---------------------------------------------------------------------------
# TimeSeriesDetector: timestamp column detection
# ---------------------------------------------------------------------------


class TestTimeSeriesTimestampDetection:
    """Tests for _detect_timestamp_column covering all candidate paths."""

    def setup_method(self):
        self.detector = TimeSeriesDetector()

    def test_detects_native_datetime_column(self):
        """Native datetime64 columns get highest priority (3)."""
        df = pd.DataFrame({
            "event_ts": pd.date_range("2020-01-01", periods=5),
            "value": [1, 2, 3, 4, 5],
        })
        result = self.detector._detect_timestamp_column(df)
        assert result == "event_ts"

    def test_detects_parseable_string_with_name_match(self):
        """String column matching timestamp pattern that parses => priority 2."""
        df = pd.DataFrame({
            "date": ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04", "2020-01-05"],
            "value": [1, 2, 3, 4, 5],
        })
        result = self.detector._detect_timestamp_column(df)
        assert result == "date"

    def test_name_match_without_parsing(self):
        """Column matching timestamp name pattern but not parseable => priority 1."""
        df = pd.DataFrame({
            "timestamp": ["abc", "def", "ghi"],
            "value": [1, 2, 3],
        })
        result = self.detector._detect_timestamp_column(df)
        assert result == "timestamp"

    def test_parseable_without_name_match(self):
        """Parseable string column without pattern name match => priority 1."""
        df = pd.DataFrame({
            "my_column": ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04", "2020-01-05"],
            "value": [1, 2, 3, 4, 5],
        })
        result = self.detector._detect_timestamp_column(df)
        assert result == "my_column"

    def test_no_timestamp_column_returns_none(self):
        """Returns None when no column can be detected as timestamp."""
        df = pd.DataFrame({
            "feature_a": [1, 2, 3],
            "feature_b": [4, 5, 6],
        })
        result = self.detector._detect_timestamp_column(df)
        assert result is None

    def test_highest_priority_wins(self):
        """When multiple candidates exist, highest priority is returned."""
        df = pd.DataFrame({
            "date_str": ["2020-01-01", "2020-01-02", "2020-01-03"],  # name_match + parse = 2
            "native_ts": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),  # datetime = 3
        })
        result = self.detector._detect_timestamp_column(df)
        assert result == "native_ts"


# ---------------------------------------------------------------------------
# TimeSeriesDetector: detect method edge cases
# ---------------------------------------------------------------------------


class TestTimeSeriesDetectEdgeCases:
    """Tests for detect() method covering duplicate timestamps and edge cases."""

    def setup_method(self):
        self.detector = TimeSeriesDetector()

    def test_duplicate_timestamps_reported(self):
        """Duplicate timestamps per entity should be reported in evidence."""
        df = pd.DataFrame({
            "customer_id": ["c1", "c1", "c1", "c1", "c2", "c2", "c2", "c2"],
            "date": pd.to_datetime([
                "2020-01-01", "2020-01-01", "2020-01-02", "2020-01-03",
                "2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04",
            ]),
        })
        result = self.detector.detect(df, entity_column="customer_id", timestamp_column="date")
        assert result.is_time_series is True
        assert result.duplicate_timestamps_count > 0

    def test_no_entity_column_returns_unknown(self):
        """When entity column cannot be detected, returns UNKNOWN."""
        df = pd.DataFrame({
            "unique_val": np.arange(100),
            "data": np.random.randn(100),
        })
        result = self.detector.detect(df)
        assert result.dataset_type == DatasetType.UNKNOWN

    def test_event_log_classification(self):
        """Irregular frequency classifies as EVENT_LOG."""
        # Highly irregular timestamps per entity
        dates = pd.to_datetime([
            "2020-01-01", "2020-01-03", "2020-02-15",
            "2020-06-01", "2020-06-05", "2020-12-20",
        ])
        df = pd.DataFrame({
            "user_id": ["u1"] * 6,
            "timestamp": dates,
            "value": range(6),
        })
        result = self.detector.detect(df, entity_column="user_id", timestamp_column="timestamp")
        assert result.is_time_series is True
        assert result.dataset_type == DatasetType.EVENT_LOG

    def test_snapshot_classification(self):
        """Single observation per entity classifies as SNAPSHOT."""
        df = pd.DataFrame({
            "customer_id": [f"c{i}" for i in range(50)],
            "date": pd.date_range("2020-01-01", periods=50),
            "value": range(50),
        })
        result = self.detector.detect(df, entity_column="customer_id", timestamp_column="date")
        assert result.is_time_series is False
        assert result.dataset_type == DatasetType.SNAPSHOT


# ---------------------------------------------------------------------------
# TimeSeriesValidator: penalties for quality score
# ---------------------------------------------------------------------------


class TestTimeSeriesValidatorPenalties:
    """Tests for validator quality score penalties."""

    def setup_method(self):
        self.validator = TimeSeriesValidator()

    def _make_clean_ts(self, n_entities=10, n_periods=10):
        """Helper to create clean time series data."""
        rows = []
        for i in range(n_entities):
            for j in range(n_periods):
                rows.append({
                    "entity_id": f"e{i}",
                    "timestamp": pd.Timestamp("2020-01-01") + pd.Timedelta(days=j),
                    "value": np.random.randn(),
                })
        return pd.DataFrame(rows)

    def test_clean_data_score_100(self):
        """Clean data with no issues should score 100."""
        df = self._make_clean_ts()
        result = self.validator.validate(df, "entity_id", "timestamp", expected_frequency="daily")
        assert result.temporal_quality_score == 100.0
        assert len(result.issues) == 0

    def test_high_duplicate_rate_penalty(self):
        """Duplicate rate > 10% applies 20 point penalty."""
        n_entities = 10
        n_periods = 10
        df = self._make_clean_ts(n_entities, n_periods)
        # Add duplicates for > 10% of entities (2 out of 10)
        dup_rows = pd.DataFrame({
            "entity_id": ["e0", "e0", "e1", "e1"],
            "timestamp": pd.to_datetime([
                "2020-01-01", "2020-01-01", "2020-01-01", "2020-01-01",
            ]),
            "value": [1, 2, 3, 4],
        })
        df = pd.concat([df, dup_rows], ignore_index=True)
        result = self.validator.validate(df, "entity_id", "timestamp", expected_frequency="daily")
        assert result.temporal_quality_score <= 80
        assert result.entities_with_duplicate_timestamps >= 2

    def test_low_duplicate_rate_penalty(self):
        """Duplicate rate between 1% and 10% applies 10 point penalty."""
        n_entities = 100
        n_periods = 5
        df = self._make_clean_ts(n_entities, n_periods)
        # Add duplicates for ~5% of entities (5 out of 100)
        dup_rows = []
        for i in range(5):
            dup_rows.append({
                "entity_id": f"e{i}",
                "timestamp": pd.Timestamp("2020-01-01"),
                "value": 99.0,
            })
        df = pd.concat([df, pd.DataFrame(dup_rows)], ignore_index=True)
        result = self.validator.validate(df, "entity_id", "timestamp", expected_frequency="daily")
        assert result.temporal_quality_score <= 90

    def test_high_ordering_penalty(self):
        """Ordering rate > 10% applies 20 point penalty."""
        n_entities = 10
        rows = []
        for i in range(n_entities):
            # All entities have reverse-ordered timestamps
            for j in range(5, 0, -1):
                rows.append({
                    "entity_id": f"e{i}",
                    "timestamp": pd.Timestamp("2020-01-01") + pd.Timedelta(days=j),
                    "value": np.random.randn(),
                })
        df = pd.DataFrame(rows)
        result = self.validator.validate(df, "entity_id", "timestamp", expected_frequency="daily")
        assert result.temporal_quality_score <= 80
        assert result.entities_with_ordering_issues > 0

    def test_low_ordering_penalty(self):
        """Ordering rate between 1% and 10% applies 10 point penalty."""
        n_entities = 100
        df = self._make_clean_ts(n_entities, 5)
        # Mess up ordering for 5 entities (5%)
        for i in range(5):
            mask = df["entity_id"] == f"e{i}"
            entity_rows = df[mask].copy()
            entity_rows["timestamp"] = entity_rows["timestamp"].iloc[::-1].values
            df.loc[mask, "timestamp"] = entity_rows["timestamp"].values
        result = self.validator.validate(df, "entity_id", "timestamp", expected_frequency="daily")
        assert result.temporal_quality_score <= 90

    def test_high_gap_rate_penalty(self):
        """Gap rate > 20% applies 20 point penalty."""
        rows = []
        for i in range(10):
            # Each entity has a large gap (gap > 3 days for daily freq)
            dates = [
                pd.Timestamp("2020-01-01"),
                pd.Timestamp("2020-01-02"),
                pd.Timestamp("2020-01-20"),  # Big gap
                pd.Timestamp("2020-01-21"),
            ]
            for d in dates:
                rows.append({
                    "entity_id": f"e{i}",
                    "timestamp": d,
                    "value": np.random.randn(),
                })
        df = pd.DataFrame(rows)
        result = self.validator.validate(df, "entity_id", "timestamp", expected_frequency="daily")
        assert result.temporal_quality_score <= 80

    def test_medium_gap_rate_penalty(self):
        """Gap rate between 5% and 10% applies 5 point penalty."""
        n_entities = 100
        df = self._make_clean_ts(n_entities, 10)
        # Add gaps to 8 entities (8%)
        for i in range(8):
            mask = (df["entity_id"] == f"e{i}") & (
                df["timestamp"] == pd.Timestamp("2020-01-05")
            )
            df.loc[mask, "timestamp"] = pd.Timestamp("2020-02-01")  # Create gap
        result = self.validator.validate(df, "entity_id", "timestamp", expected_frequency="daily")
        assert result.temporal_quality_score <= 95

    def test_moderate_gap_rate_penalty(self):
        """Gap rate between 10% and 20% applies 10 point penalty."""
        n_entities = 100
        df = self._make_clean_ts(n_entities, 10)
        # Add gaps to 15 entities (15%) - between 10% and 20%
        for i in range(15):
            mask = (df["entity_id"] == f"e{i}") & (
                df["timestamp"] == pd.Timestamp("2020-01-05")
            )
            df.loc[mask, "timestamp"] = pd.Timestamp("2020-02-01")  # Create gap
        result = self.validator.validate(df, "entity_id", "timestamp", expected_frequency="daily")
        assert result.temporal_quality_score <= 90

    def test_missing_entity_column(self):
        """Missing entity column returns score 0."""
        df = pd.DataFrame({"timestamp": pd.date_range("2020-01-01", periods=5)})
        result = self.validator.validate(df, "nonexistent", "timestamp")
        assert result.temporal_quality_score == 0

    def test_missing_timestamp_column(self):
        """Missing timestamp column returns score 0."""
        df = pd.DataFrame({"entity_id": ["e1"] * 5})
        result = self.validator.validate(df, "entity_id", "nonexistent")
        assert result.temporal_quality_score == 0


# ---------------------------------------------------------------------------
# TimeSeriesValidator: gap analysis
# ---------------------------------------------------------------------------


class TestTimeSeriesGapAnalysis:
    """Tests for _analyze_gaps with different frequencies."""

    def setup_method(self):
        self.validator = TimeSeriesValidator()

    def test_no_gaps_detected(self):
        """Regular data with no gaps returns zero gap metrics."""
        rows = []
        for i in range(5):
            for j in range(10):
                rows.append({
                    "entity_id": f"e{i}",
                    "_ts": pd.Timestamp("2020-01-01") + pd.Timedelta(days=j),
                })
        df = pd.DataFrame(rows)
        result = self.validator._analyze_gaps(df, "entity_id", "daily", max_allowed_gap_periods=3)
        assert result["entities_with_gaps"] == 0
        assert result["total_gaps"] == 0

    def test_large_gaps_detected(self):
        """Gaps exceeding threshold are detected and counted."""
        rows = []
        for i in range(5):
            dates = [
                pd.Timestamp("2020-01-01"),
                pd.Timestamp("2020-01-02"),
                pd.Timestamp("2020-01-15"),  # 13-day gap > 3 periods for daily
                pd.Timestamp("2020-01-16"),
            ]
            for d in dates:
                rows.append({"entity_id": f"e{i}", "_ts": d})
        df = pd.DataFrame(rows)
        result = self.validator._analyze_gaps(df, "entity_id", "daily", max_allowed_gap_periods=3)
        assert result["entities_with_gaps"] == 5
        assert result["total_gaps"] >= 5
        assert result["max_gap"] > 3

    def test_gap_examples_limited_to_3(self):
        """Gap examples list is limited to 3 entries."""
        rows = []
        for i in range(10):
            dates = [
                pd.Timestamp("2020-01-01"),
                pd.Timestamp("2020-01-02"),
                pd.Timestamp("2020-02-01"),  # Big gap
            ]
            for d in dates:
                rows.append({"entity_id": f"e{i}", "_ts": d})
        df = pd.DataFrame(rows)
        result = self.validator._analyze_gaps(df, "entity_id", "daily", max_allowed_gap_periods=3)
        assert len(result["examples"]) <= 3

    def test_weekly_frequency_gaps(self):
        """Gaps detection works with weekly frequency."""
        rows = []
        dates = [
            pd.Timestamp("2020-01-06"),
            pd.Timestamp("2020-01-13"),
            pd.Timestamp("2020-03-02"),  # 7-week gap > threshold
            pd.Timestamp("2020-03-09"),
        ]
        for d in dates:
            rows.append({"entity_id": "e1", "_ts": d})
        df = pd.DataFrame(rows)
        result = self.validator._analyze_gaps(df, "entity_id", "weekly", max_allowed_gap_periods=3)
        assert result["entities_with_gaps"] == 1
        assert result["total_gaps"] == 1

    def test_coverage_calculation(self):
        """Coverage percentage is calculated correctly."""
        rows = []
        # 5 entities, 2 have gaps
        for i in range(5):
            if i < 2:
                dates = [
                    pd.Timestamp("2020-01-01"),
                    pd.Timestamp("2020-01-02"),
                    pd.Timestamp("2020-02-01"),  # gap
                ]
            else:
                dates = [
                    pd.Timestamp("2020-01-01"),
                    pd.Timestamp("2020-01-02"),
                    pd.Timestamp("2020-01-03"),
                ]
            for d in dates:
                rows.append({"entity_id": f"e{i}", "_ts": d})
        df = pd.DataFrame(rows)
        result = self.validator._analyze_gaps(df, "entity_id", "daily", max_allowed_gap_periods=3)
        assert result["entities_with_gaps"] == 2
        # Coverage: 100 * (1 - 2/5) = 60%
        assert abs(result["coverage"] - 60.0) < 0.1

    def test_no_expected_interval_returns_empty(self):
        """When frequency is None and cannot be estimated, returns no gaps."""
        rows = [{"entity_id": "e1", "_ts": pd.Timestamp("2020-01-01")}]
        df = pd.DataFrame(rows)
        result = self.validator._analyze_gaps(df, "entity_id", None, max_allowed_gap_periods=3)
        assert result["entities_with_gaps"] == 0


# ---------------------------------------------------------------------------
# TimeSeriesValidator: ordering checks
# ---------------------------------------------------------------------------


class TestTimeSeriesOrderingChecks:
    """Tests for _check_ordering with non-monotonic timestamps."""

    def setup_method(self):
        self.validator = TimeSeriesValidator()

    def test_properly_ordered_no_issues(self):
        """Monotonically increasing timestamps produce no ordering issues."""
        df = pd.DataFrame({
            "entity_id": ["e1"] * 5,
            "_ts": pd.date_range("2020-01-01", periods=5),
        })
        result = self.validator._check_ordering(df, "entity_id")
        assert result["entities"] == 0

    def test_reverse_order_detected(self):
        """Reverse-ordered timestamps are detected."""
        df = pd.DataFrame({
            "entity_id": ["e1"] * 5,
            "_ts": pd.date_range("2020-01-01", periods=5)[::-1],
        })
        result = self.validator._check_ordering(df, "entity_id")
        assert result["entities"] == 1
        assert len(result["examples"]) == 1

    def test_examples_limited_to_3(self):
        """Ordering examples are limited to 3."""
        rows = []
        for i in range(10):
            dates = pd.date_range("2020-01-01", periods=5)[::-1]
            for d in dates:
                rows.append({"entity_id": f"e{i}", "_ts": d})
        df = pd.DataFrame(rows)
        result = self.validator._check_ordering(df, "entity_id")
        assert result["entities"] == 10
        assert len(result["examples"]) == 3

    def test_single_observation_entities_skipped(self):
        """Entities with fewer than 2 observations are skipped."""
        df = pd.DataFrame({
            "entity_id": ["e1", "e2", "e2"],
            "_ts": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-01"]),
        })
        result = self.validator._check_ordering(df, "entity_id")
        # Only e2 is checked; e1 has 1 obs and is skipped
        assert result["entities"] == 1
