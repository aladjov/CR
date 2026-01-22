import pytest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from customer_retention.analysis.diagnostics import (
    OverfittingAnalyzer, OverfittingResult, OverfittingCheck
)
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


@pytest.fixture
def simple_model(sample_data):
    X, y = sample_data
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X, y)
    return model


class TestTrainTestGap:
    def test_of001_detects_severe_overfitting(self, sample_data):
        X, y = sample_data
        model = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42)
        model.fit(X, y)

        train_metrics = {"pr_auc": 0.99, "roc_auc": 0.99}
        test_metrics = {"pr_auc": 0.60, "roc_auc": 0.65}

        analyzer = OverfittingAnalyzer()
        result = analyzer.analyze_train_test_gap(train_metrics, test_metrics)

        critical_issues = [c for c in result.checks if c.severity == Severity.CRITICAL]
        assert len(critical_issues) > 0

    def test_of002_detects_moderate_overfitting(self):
        train_metrics = {"pr_auc": 0.85, "roc_auc": 0.88}
        test_metrics = {"pr_auc": 0.73, "roc_auc": 0.76}

        analyzer = OverfittingAnalyzer()
        result = analyzer.analyze_train_test_gap(train_metrics, test_metrics)

        high_issues = [c for c in result.checks if c.severity == Severity.HIGH]
        assert len(high_issues) > 0

    def test_of003_detects_mild_overfitting(self):
        train_metrics = {"pr_auc": 0.80, "roc_auc": 0.82}
        test_metrics = {"pr_auc": 0.74, "roc_auc": 0.76}

        analyzer = OverfittingAnalyzer()
        result = analyzer.analyze_train_test_gap(train_metrics, test_metrics)

        medium_issues = [c for c in result.checks if c.severity == Severity.MEDIUM]
        assert len(medium_issues) > 0

    def test_of004_excellent_generalization(self):
        train_metrics = {"pr_auc": 0.78, "roc_auc": 0.80}
        test_metrics = {"pr_auc": 0.75, "roc_auc": 0.78}

        analyzer = OverfittingAnalyzer()
        result = analyzer.analyze_train_test_gap(train_metrics, test_metrics)

        critical_issues = [c for c in result.checks if c.severity == Severity.CRITICAL]
        assert len(critical_issues) == 0

    def test_gap_calculation_accurate(self):
        train_metrics = {"pr_auc": 0.90}
        test_metrics = {"pr_auc": 0.75}

        analyzer = OverfittingAnalyzer()
        result = analyzer.analyze_train_test_gap(train_metrics, test_metrics)

        assert any(c.gap == pytest.approx(0.15, abs=0.01) for c in result.checks)


class TestLearningCurve:
    def test_generates_learning_curve_data(self, sample_data):
        X, y = sample_data
        model = LogisticRegression(max_iter=1000, random_state=42)

        analyzer = OverfittingAnalyzer()
        result = analyzer.analyze_learning_curve(model, X, y)

        assert hasattr(result, "learning_curve")
        assert len(result.learning_curve) > 0

    def test_learning_curve_has_train_and_val_scores(self, sample_data):
        X, y = sample_data
        model = LogisticRegression(max_iter=1000, random_state=42)

        analyzer = OverfittingAnalyzer()
        result = analyzer.analyze_learning_curve(model, X, y)

        for point in result.learning_curve:
            assert "train_size" in point
            assert "train_score" in point
            assert "val_score" in point

    def test_learning_curve_diagnosis_provided(self, sample_data):
        X, y = sample_data
        model = LogisticRegression(max_iter=1000, random_state=42)

        analyzer = OverfittingAnalyzer()
        result = analyzer.analyze_learning_curve(model, X, y)

        assert hasattr(result, "diagnosis")
        assert result.diagnosis is not None


class TestComplexityAnalysis:
    def test_of010_detects_low_sample_to_feature_ratio(self):
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(50, 10))
        y = pd.Series(np.random.choice([0, 1], 50))

        analyzer = OverfittingAnalyzer()
        result = analyzer.analyze_complexity(X, y)

        critical_issues = [c for c in result.checks if c.severity == Severity.CRITICAL]
        assert len(critical_issues) > 0

    def test_of011_detects_concerning_ratio(self):
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(200, 10))
        y = pd.Series(np.random.choice([0, 1], 200))

        analyzer = OverfittingAnalyzer()
        result = analyzer.analyze_complexity(X, y)

        high_issues = [c for c in result.checks if c.severity == Severity.HIGH]
        assert len(high_issues) > 0

    def test_of012_detects_deep_model(self):
        model_params = {"max_depth": 20}

        analyzer = OverfittingAnalyzer()
        result = analyzer.analyze_model_complexity(model_params)

        high_issues = [c for c in result.checks if c.severity == Severity.HIGH]
        assert len(high_issues) > 0

    def test_sample_to_feature_ratio_calculated(self, sample_data):
        X, y = sample_data

        analyzer = OverfittingAnalyzer()
        result = analyzer.analyze_complexity(X, y)

        assert hasattr(result, "sample_to_feature_ratio")
        assert result.sample_to_feature_ratio > 0


class TestOverfittingResult:
    def test_result_contains_required_fields(self):
        train_metrics = {"pr_auc": 0.80}
        test_metrics = {"pr_auc": 0.75}

        analyzer = OverfittingAnalyzer()
        result = analyzer.analyze_train_test_gap(train_metrics, test_metrics)

        assert hasattr(result, "passed")
        assert hasattr(result, "checks")
        assert hasattr(result, "recommendations")

    def test_passed_is_false_with_critical_issues(self):
        train_metrics = {"pr_auc": 0.99}
        test_metrics = {"pr_auc": 0.60}

        analyzer = OverfittingAnalyzer()
        result = analyzer.analyze_train_test_gap(train_metrics, test_metrics)

        assert result.passed is False


class TestOverfittingCheck:
    def test_check_has_required_fields(self):
        train_metrics = {"pr_auc": 0.90}
        test_metrics = {"pr_auc": 0.70}

        analyzer = OverfittingAnalyzer()
        result = analyzer.analyze_train_test_gap(train_metrics, test_metrics)

        for check in result.checks:
            assert hasattr(check, "check_id")
            assert hasattr(check, "metric")
            assert hasattr(check, "severity")
            assert hasattr(check, "recommendation")


class TestRecommendations:
    def test_provides_actionable_recommendations(self):
        train_metrics = {"pr_auc": 0.95}
        test_metrics = {"pr_auc": 0.65}

        analyzer = OverfittingAnalyzer()
        result = analyzer.analyze_train_test_gap(train_metrics, test_metrics)

        assert len(result.recommendations) > 0
        for rec in result.recommendations:
            assert len(rec) > 10


class TestRunAllAnalysis:
    def test_run_all_combines_results(self, sample_data):
        X, y = sample_data
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X, y)

        train_metrics = {"pr_auc": 0.80, "roc_auc": 0.82}
        test_metrics = {"pr_auc": 0.75, "roc_auc": 0.78}

        analyzer = OverfittingAnalyzer()
        result = analyzer.run_all(model, X, y, train_metrics, test_metrics)

        assert hasattr(result, "passed")
        assert hasattr(result, "checks")
        assert len(result.checks) > 0
