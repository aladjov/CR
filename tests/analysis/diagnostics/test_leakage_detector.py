import numpy as np
import pandas as pd
import pytest

from customer_retention.analysis.diagnostics import LeakageDetector
from customer_retention.core.components.enums import Severity
from customer_retention.core.utils.leakage import TEMPORAL_METADATA_COLUMNS


@pytest.fixture
def sample_data():
    np.random.seed(42)
    n = 500
    target = np.random.choice([0, 1], n, p=[0.3, 0.7])
    return pd.DataFrame({
        "feature1": np.random.randn(n),
        "feature2": np.random.randn(n),
        "feature3": np.random.randn(n),
        "target": target,
    })


@pytest.fixture
def leaky_data():
    np.random.seed(42)
    n = 500
    target = np.random.choice([0, 1], n, p=[0.3, 0.7])
    leaky_feature = target + np.random.randn(n) * 0.01
    return pd.DataFrame({
        "normal_feature": np.random.randn(n),
        "leaky_feature": leaky_feature,
        "target": target,
    })


class TestCorrelationCheck:
    def test_ld001_detects_high_correlation(self, leaky_data):
        detector = LeakageDetector()
        X = leaky_data.drop(columns=["target"])
        y = leaky_data["target"]

        result = detector.check_correlations(X, y)

        critical_issues = [c for c in result.checks if c.severity == Severity.CRITICAL]
        assert len(critical_issues) > 0
        assert any("leaky_feature" in c.feature for c in critical_issues)

    def test_ld002_detects_suspicious_correlation(self):
        np.random.seed(42)
        n = 500
        target = np.random.choice([0, 1], n, p=[0.3, 0.7])
        suspicious = target * 2.0 + np.random.randn(n) * 0.3
        df = pd.DataFrame({"suspicious": suspicious, "normal": np.random.randn(n)})

        detector = LeakageDetector()
        result = detector.check_correlations(df, pd.Series(target))

        high_issues = [c for c in result.checks if c.severity in [Severity.CRITICAL, Severity.HIGH]]
        assert len(high_issues) > 0

    def test_no_false_positives_for_normal_features(self, sample_data):
        detector = LeakageDetector()
        X = sample_data.drop(columns=["target"])
        y = sample_data["target"]

        result = detector.check_correlations(X, y)

        critical_issues = [c for c in result.checks if c.severity == Severity.CRITICAL]
        assert len(critical_issues) == 0

    def test_correlation_values_in_result(self, leaky_data):
        detector = LeakageDetector()
        X = leaky_data.drop(columns=["target"])
        y = leaky_data["target"]

        result = detector.check_correlations(X, y)

        for check in result.checks:
            assert hasattr(check, "correlation")
            assert -1 <= check.correlation <= 1


class TestPerfectSeparationCheck:
    def test_ld010_detects_perfect_separation(self):
        np.random.seed(42)
        n = 500
        target = np.array([0] * 250 + [1] * 250)
        separating = np.array([0.0] * 250 + [100.0] * 250)
        df = pd.DataFrame({"separating": separating, "normal": np.random.randn(n)})

        detector = LeakageDetector()
        result = detector.check_separation(df, pd.Series(target))

        critical_issues = [c for c in result.checks if c.severity == Severity.CRITICAL]
        assert len(critical_issues) > 0
        assert any("separating" in c.feature for c in critical_issues)

    def test_ld011_detects_near_perfect_separation(self):
        np.random.seed(42)
        n = 500
        target = np.array([0] * 250 + [1] * 250)
        near_sep = np.concatenate([np.random.uniform(0, 10, 250), np.random.uniform(10.1, 20, 250)])
        df = pd.DataFrame({"near_sep": near_sep, "normal": np.random.randn(n)})

        detector = LeakageDetector()
        result = detector.check_separation(df, pd.Series(target))

        high_issues = [c for c in result.checks if c.severity in [Severity.CRITICAL, Severity.HIGH]]
        assert len(high_issues) > 0

    def test_no_separation_for_normal_features(self, sample_data):
        detector = LeakageDetector()
        X = sample_data.drop(columns=["target"])
        y = sample_data["target"]

        result = detector.check_separation(X, y)

        critical_issues = [c for c in result.checks if c.severity == Severity.CRITICAL]
        assert len(critical_issues) == 0

    def test_overlap_percentage_in_result(self, sample_data):
        detector = LeakageDetector()
        X = sample_data.drop(columns=["target"])
        y = sample_data["target"]

        result = detector.check_separation(X, y)

        for check in result.checks:
            assert hasattr(check, "overlap_pct")
            assert 0 <= check.overlap_pct <= 100


class TestTemporalLogicCheck:
    def test_ld020_detects_temporal_feature_leakage(self):
        np.random.seed(42)
        n = 500
        target = np.random.choice([0, 1], n, p=[0.3, 0.7])
        days_since = target * -10 + np.random.randn(n) * 0.1
        df = pd.DataFrame({
            "days_since_churn": days_since,
            "normal": np.random.randn(n),
        })

        detector = LeakageDetector()
        result = detector.check_temporal_logic(df, pd.Series(target))

        flagged = [c for c in result.checks if c.severity in [Severity.CRITICAL, Severity.HIGH]]
        assert len(flagged) > 0

    def test_identifies_temporal_features_by_name(self):
        np.random.seed(42)
        n = 500
        target = np.random.choice([0, 1], n, p=[0.3, 0.7])
        # Create temporal features with correlation to target so they get flagged
        days_since = target * 10 + np.random.randn(n) * 0.5
        df = pd.DataFrame({
            "days_since_last_order": days_since,
            "tenure_days": np.random.randn(n),
            "recency_score": np.random.randn(n),
            "normal_feature": np.random.randn(n),
        })

        detector = LeakageDetector()
        result = detector.check_temporal_logic(df, pd.Series(target))

        temporal_features = [c.feature for c in result.checks if c.check_id.startswith("LD02")]
        assert len(temporal_features) > 0
        assert "days_since_last_order" in temporal_features


class TestSingleFeatureAUC:
    def test_ld030_detects_single_feature_too_predictive(self, leaky_data):
        detector = LeakageDetector()
        X = leaky_data.drop(columns=["target"])
        y = leaky_data["target"]

        result = detector.check_single_feature_auc(X, y)

        critical_issues = [c for c in result.checks if c.severity == Severity.CRITICAL]
        assert len(critical_issues) > 0
        assert any("leaky_feature" in c.feature for c in critical_issues)

    def test_ld031_detects_very_predictive_feature(self):
        np.random.seed(42)
        n = 500
        target = np.random.choice([0, 1], n, p=[0.3, 0.7])
        predictive = target * 0.8 + np.random.randn(n) * 0.3
        df = pd.DataFrame({"predictive": predictive, "normal": np.random.randn(n)})

        detector = LeakageDetector()
        result = detector.check_single_feature_auc(df, pd.Series(target))

        high_issues = [c for c in result.checks if c.severity in [Severity.CRITICAL, Severity.HIGH]]
        assert len(high_issues) > 0

    def test_auc_values_in_result(self, sample_data):
        detector = LeakageDetector()
        X = sample_data.drop(columns=["target"])
        y = sample_data["target"]

        result = detector.check_single_feature_auc(X, y)

        for check in result.checks:
            assert hasattr(check, "auc")
            assert 0 <= check.auc <= 1


class TestLeakageResult:
    def test_result_contains_required_fields(self, sample_data):
        detector = LeakageDetector()
        X = sample_data.drop(columns=["target"])
        y = sample_data["target"]

        result = detector.run_all_checks(X, y)

        assert hasattr(result, "passed")
        assert hasattr(result, "checks")
        assert hasattr(result, "critical_issues")
        assert hasattr(result, "recommendations")

    def test_passed_is_false_with_critical_issues(self, leaky_data):
        detector = LeakageDetector()
        X = leaky_data.drop(columns=["target"])
        y = leaky_data["target"]

        result = detector.run_all_checks(X, y)

        assert result.passed is False
        assert len(result.critical_issues) > 0

    def test_passed_is_true_without_critical_issues(self, sample_data):
        detector = LeakageDetector()
        X = sample_data.drop(columns=["target"])
        y = sample_data["target"]

        result = detector.run_all_checks(X, y)

        assert result.passed is True


class TestLeakageCheck:
    def test_check_has_required_fields(self, leaky_data):
        detector = LeakageDetector()
        X = leaky_data.drop(columns=["target"])
        y = leaky_data["target"]

        result = detector.check_correlations(X, y)

        for check in result.checks:
            assert hasattr(check, "check_id")
            assert hasattr(check, "feature")
            assert hasattr(check, "severity")
            assert hasattr(check, "recommendation")


class TestRecommendations:
    def test_provides_actionable_recommendations(self, leaky_data):
        detector = LeakageDetector()
        X = leaky_data.drop(columns=["target"])
        y = leaky_data["target"]

        result = detector.run_all_checks(X, y)

        assert len(result.recommendations) > 0
        for rec in result.recommendations:
            assert len(rec) > 10


class TestTemporalMetadataExclusion:
    """Tests ensuring temporal metadata columns are never analyzed as features."""

    @pytest.fixture
    def data_with_temporal_metadata(self):
        """Create data that includes temporal metadata columns with high correlation to target."""
        np.random.seed(42)
        n = 500
        target = np.random.choice([0, 1], n, p=[0.3, 0.7])
        return pd.DataFrame({
            "normal_feature": np.random.randn(n),
            "feature_timestamp": target + np.random.randn(n) * 0.01,  # would be flagged if analyzed
            "label_timestamp": target + np.random.randn(n) * 0.01,
            "label_available_flag": target,  # perfectly correlated
            "event_timestamp": target + np.random.randn(n) * 0.01,
            "target": target,
        })

    def test_correlation_check_excludes_temporal_metadata(self, data_with_temporal_metadata):
        detector = LeakageDetector()
        X = data_with_temporal_metadata.drop(columns=["target"])
        y = data_with_temporal_metadata["target"]

        result = detector.check_correlations(X, y)

        flagged_features = {c.feature for c in result.checks}
        for col in TEMPORAL_METADATA_COLUMNS:
            assert col not in flagged_features, f"{col} should be excluded from correlation analysis"

    def test_separation_check_excludes_temporal_metadata(self, data_with_temporal_metadata):
        detector = LeakageDetector()
        X = data_with_temporal_metadata.drop(columns=["target"])
        y = data_with_temporal_metadata["target"]

        result = detector.check_separation(X, y)

        flagged_features = {c.feature for c in result.checks}
        for col in TEMPORAL_METADATA_COLUMNS:
            assert col not in flagged_features, f"{col} should be excluded from separation analysis"

    def test_temporal_logic_check_excludes_temporal_metadata(self, data_with_temporal_metadata):
        detector = LeakageDetector()
        X = data_with_temporal_metadata.drop(columns=["target"])
        y = data_with_temporal_metadata["target"]

        result = detector.check_temporal_logic(X, y)

        flagged_features = {c.feature for c in result.checks}
        for col in TEMPORAL_METADATA_COLUMNS:
            assert col not in flagged_features, f"{col} should be excluded from temporal logic analysis"

    def test_single_feature_auc_excludes_temporal_metadata(self, data_with_temporal_metadata):
        detector = LeakageDetector()
        X = data_with_temporal_metadata.drop(columns=["target"])
        y = data_with_temporal_metadata["target"]

        result = detector.check_single_feature_auc(X, y)

        flagged_features = {c.feature for c in result.checks}
        for col in TEMPORAL_METADATA_COLUMNS:
            assert col not in flagged_features, f"{col} should be excluded from AUC analysis"

    def test_run_all_checks_excludes_temporal_metadata_from_feature_analysis(self, data_with_temporal_metadata):
        """Verify temporal metadata is excluded from feature-based checks (correlation, separation, AUC).

        Note: check_uniform_timestamps and check_point_in_time are allowed to flag
        temporal metadata columns because they validate timestamp correctness, not feature analysis.
        """
        detector = LeakageDetector()
        X = data_with_temporal_metadata.drop(columns=["target"])
        y = data_with_temporal_metadata["target"]

        result = detector.run_all_checks(X, y, include_pit=False)

        flagged_features = {c.feature for c in result.checks}
        for col in TEMPORAL_METADATA_COLUMNS:
            assert col not in flagged_features, f"{col} should be excluded from feature analysis checks"

    def test_get_analyzable_columns_filters_temporal_metadata(self):
        detector = LeakageDetector()
        df = pd.DataFrame({
            "feature1": [1, 2, 3],
            "feature_timestamp": [1, 2, 3],
            "label_timestamp": [1, 2, 3],
            "label_available_flag": [1, 0, 1],
            "event_timestamp": [1, 2, 3],
        })

        analyzable = detector._get_analyzable_columns(df)

        assert "feature1" in analyzable
        for col in TEMPORAL_METADATA_COLUMNS:
            assert col not in analyzable


class TestWindowOverlapsHorizonLD062LD063:
    """LD062 / LD063: aggregation windows that span the label horizon."""

    def _make_x(self, columns):
        n = 50
        rng = np.random.default_rng(0)
        return pd.DataFrame({col: rng.standard_normal(n) for col in columns})

    def test_is_zero_flag_on_window_equal_horizon_is_critical_ld062(self):
        X = self._make_x(["NET_PRICE_count_90d_is_zero"])
        detector = LeakageDetector(label_horizon_days=90)
        result = detector.check_window_overlaps_horizon(X)
        ld062 = [c for c in result.checks if c.check_id == "LD062"]
        assert len(ld062) == 1
        assert ld062[0].feature == "NET_PRICE_count_90d_is_zero"
        assert ld062[0].severity == Severity.CRITICAL

    def test_is_zero_flag_on_window_above_horizon_is_critical_ld062(self):
        X = self._make_x(["NET_PRICE_sum_180d_is_zero"])
        detector = LeakageDetector(label_horizon_days=90)
        result = detector.check_window_overlaps_horizon(X)
        ld062 = [c for c in result.checks if c.check_id == "LD062"]
        assert len(ld062) == 1

    def test_is_zero_flag_on_all_time_window_is_critical_ld062(self):
        X = self._make_x(["EVENT_count_all_time_is_zero"])
        detector = LeakageDetector(label_horizon_days=30)
        result = detector.check_window_overlaps_horizon(X)
        ld062 = [c for c in result.checks if c.check_id == "LD062"]
        assert len(ld062) == 1

    def test_is_zero_flag_on_window_below_horizon_is_clean(self):
        X = self._make_x(["NET_PRICE_count_30d_is_zero"])
        detector = LeakageDetector(label_horizon_days=90)
        result = detector.check_window_overlaps_horizon(X)
        ld062 = [c for c in result.checks if c.check_id == "LD062"]
        assert len(ld062) == 0
        assert result.passed

    def test_count_aggregation_over_horizon_is_high_ld063(self):
        X = self._make_x(["NET_PRICE_count_180d"])
        detector = LeakageDetector(label_horizon_days=90)
        result = detector.check_window_overlaps_horizon(X)
        ld063 = [c for c in result.checks if c.check_id == "LD063"]
        assert len(ld063) == 1
        assert ld063[0].feature == "NET_PRICE_count_180d"
        assert ld063[0].severity == Severity.HIGH

    def test_sum_aggregation_at_horizon_is_high_ld063(self):
        X = self._make_x(["NET_PRICE_sum_90d"])
        detector = LeakageDetector(label_horizon_days=90)
        result = detector.check_window_overlaps_horizon(X)
        ld063 = [c for c in result.checks if c.check_id == "LD063"]
        assert len(ld063) == 1

    def test_mean_aggregation_not_flagged(self):
        # mean describes the shape of activity, not presence/absence
        X = self._make_x(["NET_PRICE_mean_180d"])
        detector = LeakageDetector(label_horizon_days=90)
        result = detector.check_window_overlaps_horizon(X)
        flagged = [c for c in result.checks if c.check_id in ("LD062", "LD063")]
        assert len(flagged) == 0

    def test_max_min_aggregation_not_flagged(self):
        X = self._make_x(["NET_PRICE_max_180d", "NET_PRICE_min_180d"])
        detector = LeakageDetector(label_horizon_days=90)
        result = detector.check_window_overlaps_horizon(X)
        flagged = [c for c in result.checks if c.check_id in ("LD062", "LD063")]
        assert len(flagged) == 0

    def test_count_below_horizon_is_clean(self):
        X = self._make_x(["NET_PRICE_count_30d", "NET_PRICE_sum_30d"])
        detector = LeakageDetector(label_horizon_days=90)
        result = detector.check_window_overlaps_horizon(X)
        flagged = [c for c in result.checks if c.check_id in ("LD062", "LD063")]
        assert len(flagged) == 0

    def test_check_disabled_when_no_horizon(self):
        X = self._make_x(["NET_PRICE_count_180d_is_zero", "NET_PRICE_sum_180d"])
        detector = LeakageDetector()  # no label_horizon_days
        result = detector.check_window_overlaps_horizon(X)
        flagged = [c for c in result.checks if c.check_id in ("LD062", "LD063")]
        assert len(flagged) == 0

    def test_non_aggregation_columns_are_ignored(self):
        X = self._make_x(["customer_age", "tenure_months", "annual_revenue"])
        detector = LeakageDetector(label_horizon_days=90)
        result = detector.check_window_overlaps_horizon(X)
        assert len(result.checks) == 0

    def test_week_unit_is_converted_correctly(self):
        # 13w = 91 days >= 90d horizon → flagged
        X = self._make_x(["EVENT_count_13w_is_zero"])
        detector = LeakageDetector(label_horizon_days=90)
        result = detector.check_window_overlaps_horizon(X)
        ld062 = [c for c in result.checks if c.check_id == "LD062"]
        assert len(ld062) == 1

    def test_week_unit_below_horizon_is_clean(self):
        # 12w = 84 days < 90d horizon → clean
        X = self._make_x(["EVENT_count_12w_is_zero"])
        detector = LeakageDetector(label_horizon_days=90)
        result = detector.check_window_overlaps_horizon(X)
        ld062 = [c for c in result.checks if c.check_id == "LD062"]
        assert len(ld062) == 0

    def test_temporal_metadata_columns_are_excluded(self):
        # Even if the column name happened to match the pattern, temporal
        # metadata should be filtered before parsing.
        X = self._make_x(["feature_timestamp", "label_timestamp", "label_available_flag"])
        detector = LeakageDetector(label_horizon_days=90)
        result = detector.check_window_overlaps_horizon(X)
        assert len(result.checks) == 0

    def test_full_sps_leakage_pattern_detected(self):
        # Reproduces the SPS NET_PRICE incident: 90d horizon, the full set of
        # leaderboard features dominating the chart.
        X = self._make_x([
            "NET_PRICE_count_180d_is_zero",
            "NET_PRICE_sum_180d_is_zero",
            "NET_PRICE_count_90d_is_zero",
            "NET_PRICE_sum_90d_is_zero",
            "NET_PRICE_count_180d",
            "NET_PRICE_count_all_time",
            "NET_PRICE_sum_180d",
            "NET_PRICE_mean_180d",  # not flagged
            "BLENDED_DOC_RATE_STANDARD_DOCUMENT_sum_180d",
            "QUANTITY_count_30d_is_zero",  # not flagged: 30d < 90d
        ])
        detector = LeakageDetector(label_horizon_days=90)
        result = detector.check_window_overlaps_horizon(X)

        critical_features = {c.feature for c in result.checks if c.check_id == "LD062"}
        high_features = {c.feature for c in result.checks if c.check_id == "LD063"}

        assert critical_features == {
            "NET_PRICE_count_180d_is_zero",
            "NET_PRICE_sum_180d_is_zero",
            "NET_PRICE_count_90d_is_zero",
            "NET_PRICE_sum_90d_is_zero",
        }
        assert high_features == {
            "NET_PRICE_count_180d",
            "NET_PRICE_count_all_time",
            "NET_PRICE_sum_180d",
            "BLENDED_DOC_RATE_STANDARD_DOCUMENT_sum_180d",
        }
        assert not result.passed

    def test_run_all_checks_includes_window_overlap(self):
        # Verify the integration into run_all_checks fires LD062/LD063 too.
        n = 100
        rng = np.random.default_rng(42)
        X = pd.DataFrame({
            "NET_PRICE_count_180d_is_zero": rng.integers(0, 2, n).astype(float),
            "normal_feature": rng.standard_normal(n),
        })
        y = pd.Series(rng.integers(0, 2, n))

        detector = LeakageDetector(label_horizon_days=90)
        result = detector.run_all_checks(X, y, include_pit=False)

        ld062 = [c for c in result.checks if c.check_id == "LD062"]
        assert len(ld062) == 1
        assert ld062[0].feature == "NET_PRICE_count_180d_is_zero"
        assert not result.passed

    def test_run_all_checks_no_horizon_does_not_fire_window_check(self):
        n = 100
        rng = np.random.default_rng(42)
        X = pd.DataFrame({
            "NET_PRICE_count_180d_is_zero": rng.integers(0, 2, n).astype(float),
        })
        y = pd.Series(rng.integers(0, 2, n))

        detector = LeakageDetector()  # no horizon
        result = detector.run_all_checks(X, y, include_pit=False)

        ld062 = [c for c in result.checks if c.check_id == "LD062"]
        assert len(ld062) == 0
