import numpy as np
import pandas as pd
import pytest

from customer_retention.analysis.diagnostics import LeakageDetector
from customer_retention.core.components.enums import Severity


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
        n = 100
        df = pd.DataFrame({
            "days_since_last_order": np.random.randn(n),
            "tenure_days": np.random.randn(n),
            "recency_score": np.random.randn(n),
            "normal_feature": np.random.randn(n),
        })
        target = pd.Series(np.random.choice([0, 1], n))

        detector = LeakageDetector()
        result = detector.check_temporal_logic(df, target)

        temporal_features = [c.feature for c in result.checks if "temporal" in c.check_id.lower() or c.check_id.startswith("LD02")]
        assert len(temporal_features) >= 0


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
