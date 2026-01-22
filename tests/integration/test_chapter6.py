import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from customer_retention.analysis.diagnostics import (
    LeakageDetector,
    OverfittingAnalyzer,
    CVAnalyzer,
    SegmentPerformanceAnalyzer,
    CalibrationAnalyzer,
    ErrorAnalyzer,
    NoiseTester,
)
from customer_retention.stages.validation import BusinessSenseGate
from customer_retention.core.components.enums import Severity


@pytest.fixture
def retail_data():
    retail_path = Path(__file__).parent.parent / "fixtures" / "customer_retention_retail.csv"
    return pd.read_csv(retail_path)


@pytest.fixture
def feature_columns():
    return ["avgorder", "ordfreq", "eopenrate", "eclickrate", "paperless", "refill", "doorstep"]


@pytest.fixture
def trained_model(retail_data, feature_columns):
    X = retail_data[feature_columns].fillna(0)
    y = retail_data["retained"]
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X, y)
    return model


class TestLeakageDetection:
    def test_ac6_1_correlation_check_identifies_high_correlation(self):
        np.random.seed(42)
        n = 500
        target = np.random.choice([0, 1], n, p=[0.3, 0.7])
        leaky_feature = target + np.random.randn(n) * 0.01
        df = pd.DataFrame({"leaky": leaky_feature, "normal": np.random.randn(n)})

        detector = LeakageDetector()
        result = detector.check_correlations(df, pd.Series(target))

        critical_issues = [c for c in result.checks if c.severity == Severity.CRITICAL]
        assert len(critical_issues) > 0

    def test_ac6_2_separation_check_identifies_perfect_separation(self):
        np.random.seed(42)
        n = 500
        target = np.array([0] * 250 + [1] * 250)
        separating = np.array([0.0] * 250 + [100.0] * 250)
        df = pd.DataFrame({"separating": separating, "normal": np.random.randn(n)})

        detector = LeakageDetector()
        result = detector.check_separation(df, pd.Series(target))

        critical_issues = [c for c in result.checks if c.severity == Severity.CRITICAL]
        assert len(critical_issues) > 0

    def test_ac6_4_single_feature_auc_calculated(self, retail_data, feature_columns):
        X = retail_data[feature_columns].fillna(0)
        y = retail_data["retained"]

        detector = LeakageDetector()
        result = detector.check_single_feature_auc(X, y)

        for check in result.checks:
            assert hasattr(check, "auc")
            assert 0 <= check.auc <= 1


class TestOverfittingDiagnosis:
    def test_ac6_5_train_test_gap_calculated_correctly(self):
        train_metrics = {"pr_auc": 0.85, "roc_auc": 0.88}
        test_metrics = {"pr_auc": 0.70, "roc_auc": 0.75}

        analyzer = OverfittingAnalyzer()
        result = analyzer.analyze_train_test_gap(train_metrics, test_metrics)

        gap_check = [c for c in result.checks if c.metric == "pr_auc"][0]
        assert gap_check.gap == pytest.approx(0.15, abs=0.01)

    def test_ac6_6_learning_curve_generated(self, retail_data, feature_columns):
        X = retail_data[feature_columns].fillna(0)
        y = retail_data["retained"]
        model = LogisticRegression(max_iter=1000, random_state=42)

        analyzer = OverfittingAnalyzer()
        result = analyzer.analyze_learning_curve(model, X, y)

        assert len(result.learning_curve) > 0
        for point in result.learning_curve:
            assert "train_size" in point
            assert "train_score" in point
            assert "val_score" in point

    def test_ac6_7_complexity_metrics_computed(self, retail_data, feature_columns):
        X = retail_data[feature_columns].fillna(0)
        y = retail_data["retained"]

        analyzer = OverfittingAnalyzer()
        result = analyzer.analyze_complexity(X, y)

        assert result.sample_to_feature_ratio > 0

    def test_ac6_8_recommendations_provided(self):
        train_metrics = {"pr_auc": 0.95}
        test_metrics = {"pr_auc": 0.65}

        analyzer = OverfittingAnalyzer()
        result = analyzer.analyze_train_test_gap(train_metrics, test_metrics)

        assert len(result.recommendations) > 0


class TestStabilityAnalysis:
    def test_ac6_9_cv_variance_calculated(self):
        cv_scores = [0.75, 0.78, 0.72, 0.80, 0.76]

        analyzer = CVAnalyzer()
        result = analyzer.analyze_variance(cv_scores)

        assert result.cv_std > 0

    def test_ac6_10_fold_by_fold_breakdown(self):
        cv_scores = [0.75, 0.78, 0.72, 0.80, 0.76]

        analyzer = CVAnalyzer()
        result = analyzer.analyze_folds(cv_scores)

        assert len(result.fold_analysis) == 5
        for fold in result.fold_analysis:
            assert "score" in fold

    def test_ac6_11_outlier_folds_identified(self):
        cv_scores = [0.80, 0.80, 0.80, 0.80, 0.80, 0.80, 0.80, 0.80, 0.80, 0.10]

        analyzer = CVAnalyzer()
        result = analyzer.analyze_folds(cv_scores)

        assert hasattr(result, "outlier_folds")


class TestSegmentAnalysis:
    def test_ac6_12_segments_defined_correctly(self, retail_data, feature_columns, trained_model):
        X = retail_data[feature_columns].fillna(0)
        y = retail_data["retained"]
        segments = pd.Series(["A"] * (len(y) // 2) + ["B"] * (len(y) - len(y) // 2))

        analyzer = SegmentPerformanceAnalyzer()
        result = analyzer.analyze_performance(trained_model, X, y, segments)

        assert len(result.segment_metrics) > 0

    def test_ac6_13_per_segment_metrics_calculated(self, retail_data, feature_columns, trained_model):
        X = retail_data[feature_columns].fillna(0)
        y = retail_data["retained"]
        segments = pd.Series(["A"] * (len(y) // 2) + ["B"] * (len(y) - len(y) // 2))

        analyzer = SegmentPerformanceAnalyzer()
        result = analyzer.analyze_performance(trained_model, X, y, segments)

        for segment, metrics in result.segment_metrics.items():
            assert "pr_auc" in metrics or "roc_auc" in metrics

    def test_ac6_14_underperforming_segments_flagged(self, retail_data, feature_columns, trained_model):
        X = retail_data[feature_columns].fillna(0)
        y = retail_data["retained"]
        segments = pd.Series(["A"] * (len(y) // 2) + ["B"] * (len(y) - len(y) // 2))

        analyzer = SegmentPerformanceAnalyzer()
        result = analyzer.analyze_performance(trained_model, X, y, segments)

        assert hasattr(result, "checks")


class TestBusinessSenseGate:
    def test_ac6_15_all_checks_in_gate(self):
        metrics = {"pr_auc_test": 0.65, "roc_auc_test": 0.75}
        feature_importance = {"days_since_last_order": 0.30, "email_engagement": 0.25}

        gate = BusinessSenseGate()
        result = gate.run(metrics, feature_importance)

        assert hasattr(result, "checks")
        assert len(result.checks) > 0

    def test_ac6_16_gate_blocks_on_critical(self):
        metrics = {"pr_auc_test": 0.98}
        feature_importance = {"unknown_suspicious_feature": 0.90}

        gate = BusinessSenseGate()
        result = gate.run(metrics, feature_importance)

        assert result.passed is False
        assert len(result.critical_issues) > 0

    def test_ac6_17_report_generated(self):
        metrics = {"pr_auc_test": 0.65}
        feature_importance = {"ordfreq": 0.30}

        gate = BusinessSenseGate()
        result = gate.run(metrics, feature_importance)

        assert hasattr(result, "recommendation")
        assert len(result.recommendation) > 0

    def test_ac6_18_sign_off_tracked(self):
        gate = BusinessSenseGate(required_sign_offs=["Data Engineer", "Domain Expert"])
        gate.add_sign_off("Data Engineer", "Approved")
        gate.add_sign_off("Domain Expert", "Approved")

        result = gate.check_sign_offs()
        assert result.passed is True


class TestCalibration:
    def test_brier_score_calculated(self, retail_data, feature_columns, trained_model):
        X = retail_data[feature_columns].fillna(0)
        y = retail_data["retained"]
        y_proba = trained_model.predict_proba(X)[:, 1]

        analyzer = CalibrationAnalyzer()
        result = analyzer.analyze_calibration(y.values, y_proba)

        assert result.brier_score >= 0
        assert result.ece >= 0

    def test_reliability_diagram_generated(self, retail_data, feature_columns, trained_model):
        X = retail_data[feature_columns].fillna(0)
        y = retail_data["retained"]
        y_proba = trained_model.predict_proba(X)[:, 1]

        analyzer = CalibrationAnalyzer()
        result = analyzer.analyze_calibration(y.values, y_proba)

        assert len(result.reliability_data) > 0


class TestErrorAnalysis:
    def test_error_patterns_identified(self, retail_data, feature_columns, trained_model):
        X = retail_data[feature_columns].fillna(0)
        y = retail_data["retained"]

        analyzer = ErrorAnalyzer()
        result = analyzer.analyze_errors(trained_model, X, y)

        assert result.total_errors >= 0
        assert 0 <= result.error_rate <= 1


class TestNoiseRobustness:
    def test_gaussian_noise_test_runs(self, retail_data, feature_columns, trained_model):
        X = retail_data[feature_columns].fillna(0)
        y = retail_data["retained"]

        tester = NoiseTester()
        result = tester.test_gaussian_noise(trained_model, X, y)

        assert len(result.degradation_curve) > 0
        assert 0 <= result.robustness_score <= 1


class TestFullDiagnosticPipeline:
    def test_end_to_end_diagnostics(self, retail_data, feature_columns):
        X = retail_data[feature_columns].fillna(0)
        y = retail_data["retained"]

        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X, y)

        leakage_detector = LeakageDetector()
        leakage_result = leakage_detector.run_all_checks(X, y)

        train_metrics = {"pr_auc": 0.75, "roc_auc": 0.78}
        test_metrics = {"pr_auc": 0.70, "roc_auc": 0.73}
        overfitting_analyzer = OverfittingAnalyzer()
        overfitting_result = overfitting_analyzer.analyze_train_test_gap(train_metrics, test_metrics)

        cv_scores = [0.72, 0.75, 0.71, 0.74, 0.73]
        cv_analyzer = CVAnalyzer()
        cv_result = cv_analyzer.run_all(cv_scores, test_score=0.70)

        y_proba = model.predict_proba(X)[:, 1]
        calibration_analyzer = CalibrationAnalyzer()
        calibration_result = calibration_analyzer.analyze_calibration(y.values, y_proba)

        feature_importance = {"ordfreq": 0.30, "avgorder": 0.25, "eopenrate": 0.20}
        gate = BusinessSenseGate()
        gate_result = gate.run(test_metrics, feature_importance)

        all_passed = (
            leakage_result.passed and
            overfitting_result.passed and
            cv_result.passed and
            gate_result.passed
        )

        assert isinstance(all_passed, bool)
