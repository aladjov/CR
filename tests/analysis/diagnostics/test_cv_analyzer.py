import numpy as np
import pandas as pd
import pytest

from customer_retention.analysis.diagnostics import CVAnalyzer
from customer_retention.core.components.enums import Severity


@pytest.fixture
def sample_data():
    np.random.seed(42)
    n = 500
    X = pd.DataFrame({
        "feature1": np.random.randn(n),
        "feature2": np.random.randn(n),
        "feature3": np.random.randn(n),
    })
    y = pd.Series(np.random.choice([0, 1], n, p=[0.3, 0.7]))
    return X, y


class TestCVVarianceAnalysis:
    def test_cv001_detects_very_high_variance(self):
        cv_scores = [0.5, 0.9, 0.6, 0.85, 0.55]
        analyzer = CVAnalyzer()
        result = analyzer.analyze_variance(cv_scores)
        critical_issues = [c for c in result.checks if c.severity == Severity.CRITICAL]
        assert len(critical_issues) > 0

    def test_cv002_detects_high_variance(self):
        cv_scores = [0.60, 0.90, 0.62, 0.88, 0.65]
        analyzer = CVAnalyzer()
        result = analyzer.analyze_variance(cv_scores)
        high_issues = [c for c in result.checks if c.severity in [Severity.CRITICAL, Severity.HIGH]]
        assert len(high_issues) > 0

    def test_cv003_detects_moderate_variance(self):
        cv_scores = [0.70, 0.85, 0.72, 0.83, 0.75]
        analyzer = CVAnalyzer()
        result = analyzer.analyze_variance(cv_scores)
        medium_issues = [c for c in result.checks if c.severity == Severity.MEDIUM]
        assert len(medium_issues) > 0

    def test_cv004_stable_cv(self):
        cv_scores = [0.78, 0.79, 0.78, 0.80, 0.79]
        analyzer = CVAnalyzer()
        result = analyzer.analyze_variance(cv_scores)
        critical_issues = [c for c in result.checks if c.severity == Severity.CRITICAL]
        assert len(critical_issues) == 0


class TestV3VarianceThresholds:
    """V3 8.5: STD_CRITICAL lowered from 0.15 to 0.10. cv_std > 0.10 must
    now produce a CRITICAL CV check (and via _compute_verdict, an
    'unstable' aggregate verdict). Motivated by spschurn-cd9565f9 LR
    cv_std=0.196 ('UNSTABLE' verdict was correct only because cv_std
    happened to exceed 0.15; a hypothetical cv_std=0.12 would have
    silently produced 'solid' under pre-V3 thresholds)."""

    def test_threshold_constants_match_v3_spec(self):
        assert CVAnalyzer.STD_CRITICAL == 0.10
        assert CVAnalyzer.STD_HIGH == 0.075
        assert CVAnalyzer.STD_MEDIUM == 0.05

    def test_cv_std_just_above_010_is_critical(self):
        # Synthetic scores giving cv_std ~ 0.141 (in 0.10-0.15 band, was HIGH pre-V3)
        scores = [0.40, 0.80, 0.50, 0.70, 0.60]
        analyzer = CVAnalyzer()
        result = analyzer.analyze_variance(scores)
        assert 0.10 < result.cv_std < 0.15, f"setup error: cv_std={result.cv_std}"
        critical = [c for c in result.checks if c.severity == Severity.CRITICAL]
        assert len(critical) == 1, "cv_std > 0.10 should produce CRITICAL"
        assert critical[0].check_id == "CV001"

    def test_cv_std_between_075_and_010_is_high(self):
        # Synthetic scores giving cv_std ~ 0.084 (between 0.075 and 0.10)
        scores = [0.50, 0.70, 0.55, 0.70, 0.55]
        analyzer = CVAnalyzer()
        result = analyzer.analyze_variance(scores)
        assert 0.075 < result.cv_std < 0.10, f"setup error: cv_std={result.cv_std}"
        criticals = [c for c in result.checks if c.severity == Severity.CRITICAL]
        highs = [c for c in result.checks if c.severity == Severity.HIGH]
        assert len(criticals) == 0
        assert len(highs) == 1
        assert highs[0].check_id == "CV002"

    def test_recommendation_message_quotes_threshold(self):
        scores = [0.5, 0.9, 0.6, 0.85, 0.55]   # cv_std ~ 0.16
        analyzer = CVAnalyzer()
        result = analyzer.analyze_variance(scores)
        rec = result.checks[0].recommendation
        assert "0.10" in rec, f"recommendation should cite the V3 threshold: {rec}"
        assert "Model is unstable" in rec


class TestFoldByFoldAnalysis:
    def test_analyzes_each_fold(self):
        cv_scores = [0.75, 0.78, 0.72, 0.80, 0.76]
        analyzer = CVAnalyzer()
        result = analyzer.analyze_folds(cv_scores)
        assert hasattr(result, "fold_analysis")
        assert len(result.fold_analysis) == 5

    def test_identifies_best_worst_gap(self):
        cv_scores = [0.75, 0.78, 0.72, 0.80, 0.76]
        analyzer = CVAnalyzer()
        result = analyzer.analyze_folds(cv_scores)
        assert hasattr(result, "best_worst_gap")
        assert result.best_worst_gap == pytest.approx(0.08, abs=0.01)

    def test_identifies_outlier_folds(self):
        cv_scores = [0.80, 0.80, 0.80, 0.80, 0.80, 0.80, 0.80, 0.80, 0.80, 0.10]
        analyzer = CVAnalyzer()
        result = analyzer.analyze_folds(cv_scores)
        assert hasattr(result, "outlier_folds")
        assert len(result.outlier_folds) > 0


class TestCVTestComparison:
    def test_cv010_cv_optimistic(self):
        cv_mean = 0.85
        test_score = 0.70
        analyzer = CVAnalyzer()
        result = analyzer.compare_cv_test(cv_mean, test_score)
        high_issues = [c for c in result.checks if c.severity == Severity.HIGH]
        assert len(high_issues) > 0

    def test_cv011_cv_pessimistic(self):
        cv_mean = 0.70
        test_score = 0.85
        analyzer = CVAnalyzer()
        result = analyzer.compare_cv_test(cv_mean, test_score)
        medium_issues = [c for c in result.checks if c.severity == Severity.MEDIUM]
        assert len(medium_issues) > 0

    def test_cv012_cv_consistent(self):
        cv_mean = 0.78
        test_score = 0.76
        analyzer = CVAnalyzer()
        result = analyzer.compare_cv_test(cv_mean, test_score)
        info_issues = [c for c in result.checks if c.severity == Severity.INFO]
        assert len(info_issues) > 0


class TestCVAnalysisResult:
    def test_result_contains_required_fields(self):
        cv_scores = [0.75, 0.78, 0.72, 0.80, 0.76]
        analyzer = CVAnalyzer()
        result = analyzer.analyze_variance(cv_scores)
        assert hasattr(result, "passed")
        assert hasattr(result, "checks")
        assert hasattr(result, "cv_mean")
        assert hasattr(result, "cv_std")


class TestRunAllAnalysis:
    def test_run_all_combines_results(self):
        cv_scores = [0.75, 0.78, 0.72, 0.80, 0.76]
        test_score = 0.74
        analyzer = CVAnalyzer()
        result = analyzer.run_all(cv_scores, test_score)
        assert hasattr(result, "passed")
        assert hasattr(result, "checks")
        assert len(result.checks) > 0
