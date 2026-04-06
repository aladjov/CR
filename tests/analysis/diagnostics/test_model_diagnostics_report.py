from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from customer_retention.analysis.diagnostics.feature_stability import FeatureStabilityResult
from customer_retention.analysis.diagnostics.model_diagnostics_report import (
    CrossModelAgreement,
    ModelDiagnosticsReport,
    ModelDiagnosticsReportGenerator,
    ModelDiagnosticsSummary,
)
from customer_retention.core.components.enums import Severity
from customer_retention.stages.modeling.spark_classifier_wrapper import SparkClassifierWrapper


def _mock_cv_analysis(passed=True):
    m = MagicMock()
    m.passed = passed
    m.checks = []
    m.cv_mean = 0.80
    m.cv_std = 0.03
    m.recommendations = []
    return m


def _mock_overfitting(passed=True, has_critical=False):
    m = MagicMock()
    m.passed = passed
    check = MagicMock()
    check.severity = Severity.CRITICAL if has_critical else Severity.LOW
    m.checks = [check] if has_critical else []
    m.recommendations = []
    m.learning_curve = []
    return m


def _mock_calibration(passed=True):
    m = MagicMock()
    m.passed = passed
    m.checks = []
    m.brier_score = 0.15
    m.ece = 0.05
    return m


def _mock_validity(passed=True):
    m = MagicMock()
    m.passed = passed
    m.critical_issues = []
    m.high_issues = []
    m.warnings = []
    m.recommendation = "Model looks good"
    m.diagnostic_hints = []
    return m


def _mock_leakage(has_critical=False):
    m = MagicMock()
    check = MagicMock()
    check.severity = Severity.CRITICAL if has_critical else Severity.LOW
    check.check_id = "LD001"
    check.feature = "feat_x"
    check.recommendation = "Investigate"
    m.checks = [check] if has_critical else []
    m.critical_issues = [check] if has_critical else []
    return m


def _mock_error_analysis():
    m = MagicMock()
    m.error_patterns = []
    m.hypotheses = []
    return m


def _make_model(importances):
    m = MagicMock()
    m.feature_importances_ = np.array(importances)
    m.predict_proba = MagicMock(return_value=np.column_stack([
        np.random.rand(50), np.random.rand(50)]))
    return m


def _make_cv_results(n_folds=3, with_importance=False, feature_names=None):
    result = {
        "cv_scores": [0.78, 0.80, 0.82][:n_folds],
        "cv_mean": 0.80,
        "cv_std": 0.02,
        "fold_details": [],
        "scoring": "roc_auc",
    }
    for i in range(n_folds):
        detail = {"fold": i + 1, "score": 0.78 + i * 0.02}
        if with_importance and feature_names:
            detail["feature_importance"] = {f: np.random.rand() for f in feature_names}
        result["fold_details"].append(detail)
    return result


@pytest.fixture
def feature_names():
    return ["feat_a", "feat_b", "feat_c"]


@pytest.fixture
def generator():
    return ModelDiagnosticsReportGenerator()


@pytest.fixture
def sample_data():
    np.random.seed(42)
    n = 50
    X_train = pd.DataFrame({"feat_a": np.random.randn(n), "feat_b": np.random.randn(n), "feat_c": np.random.randn(n)})
    X_test = pd.DataFrame({"feat_a": np.random.randn(20), "feat_b": np.random.randn(20), "feat_c": np.random.randn(20)})
    y_train = pd.Series(np.random.choice([0, 1], n))
    y_test = pd.Series(np.random.choice([0, 1], 20))
    return X_train, X_test, y_train, y_test


class TestCrossModelAgreement:
    def test_identical_importances_perfect_jaccard(self, generator, feature_names):
        models = {
            "A": _make_model([0.5, 0.3, 0.2]),
            "B": _make_model([0.5, 0.3, 0.2]),
        }
        agreement = generator._compute_cross_model_agreement(models, feature_names, top_n=3)

        assert agreement.agreement_score == 1.0
        assert set(agreement.consensus_features) == set(feature_names)

    def test_disjoint_importances_zero_jaccard(self, generator):
        models = {
            "A": _make_model([0.5, 0.3, 0.0, 0.0]),
            "B": _make_model([0.0, 0.0, 0.5, 0.3]),
        }
        names = ["a", "b", "c", "d"]
        agreement = generator._compute_cross_model_agreement(models, names, top_n=2)

        assert agreement.agreement_score == 0.0
        assert agreement.consensus_features == []

    def test_model_specific_features_detected(self, generator):
        models = {
            "A": _make_model([0.5, 0.3, 0.0, 0.0]),
            "B": _make_model([0.0, 0.3, 0.5, 0.0]),
        }
        names = ["a", "b", "c", "d"]
        agreement = generator._compute_cross_model_agreement(models, names, top_n=2)

        assert "a" in agreement.model_specific_features["A"]
        assert "c" in agreement.model_specific_features["B"]

    def test_single_model_perfect_agreement(self, generator, feature_names):
        models = {"A": _make_model([0.5, 0.3, 0.2])}
        agreement = generator._compute_cross_model_agreement(models, feature_names, top_n=3)

        assert agreement.agreement_score == 1.0

    def test_lr_model_uses_coef(self, generator):
        model = MagicMock(spec=[])
        model.coef_ = np.array([[0.5, -0.3, 0.1]])
        models = {"LR": model}
        names = ["a", "b", "c"]
        agreement = generator._compute_cross_model_agreement(models, names, top_n=2)

        assert "a" in agreement.consensus_features or "b" in agreement.consensus_features


class TestVerdictLogic:
    def test_verdict_solid(self, generator):
        summaries = {"A": MagicMock(
            cv_analysis=_mock_cv_analysis(True),
            overfitting=_mock_overfitting(True),
            calibration=_mock_calibration(True),
            validity=_mock_validity(True),
            feature_stability=None,
        )}
        leakage = _mock_leakage(has_critical=False)

        verdict, issues, recs = generator._compute_verdict(summaries, leakage)
        assert verdict == "solid"
        assert len(issues) == 0

    def test_verdict_leaky(self, generator):
        summaries = {"A": MagicMock(
            cv_analysis=_mock_cv_analysis(True),
            overfitting=_mock_overfitting(True),
            calibration=_mock_calibration(True),
            validity=_mock_validity(True),
            feature_stability=None,
        )}
        leakage = _mock_leakage(has_critical=True)

        verdict, issues, _ = generator._compute_verdict(summaries, leakage)
        assert verdict == "leaky"
        assert len(issues) > 0

    def test_verdict_overfit(self, generator):
        summaries = {"A": MagicMock(
            cv_analysis=_mock_cv_analysis(True),
            overfitting=_mock_overfitting(False, has_critical=True),
            calibration=_mock_calibration(True),
            validity=_mock_validity(True),
            feature_stability=None,
        )}
        leakage = _mock_leakage(has_critical=False)

        verdict, issues, _ = generator._compute_verdict(summaries, leakage)
        assert verdict == "overfit"

    def test_verdict_unstable(self, generator):
        cv = _mock_cv_analysis(False)
        check = MagicMock()
        check.severity = Severity.CRITICAL
        check.metric = "cv_std"
        check.recommendation = "High CV variance"
        cv.checks = [check]

        summaries = {"A": MagicMock(
            cv_analysis=cv,
            overfitting=_mock_overfitting(True),
            calibration=_mock_calibration(True),
            validity=_mock_validity(True),
            feature_stability=None,
        )}
        leakage = _mock_leakage(has_critical=False)

        verdict, _, _ = generator._compute_verdict(summaries, leakage)
        assert verdict == "unstable"

    def test_verdict_caution_on_high_issues(self, generator):
        validity = _mock_validity(False)
        issue = MagicMock()
        issue.severity = Severity.HIGH
        issue.description = "Moderate overfitting"
        validity.high_issues = [issue]

        summaries = {"A": MagicMock(
            cv_analysis=_mock_cv_analysis(True),
            overfitting=_mock_overfitting(True),
            calibration=_mock_calibration(True),
            validity=validity,
            feature_stability=None,
        )}
        leakage = _mock_leakage(has_critical=False)

        verdict, _, _ = generator._compute_verdict(summaries, leakage)
        assert verdict == "caution"


class TestGenerateOrchestration:
    @patch("customer_retention.analysis.diagnostics.model_diagnostics_report.LeakageDetector")
    @patch("customer_retention.analysis.diagnostics.model_diagnostics_report.CVAnalyzer")
    @patch("customer_retention.analysis.diagnostics.model_diagnostics_report.OverfittingAnalyzer")
    @patch("customer_retention.analysis.diagnostics.model_diagnostics_report.CalibrationAnalyzer")
    @patch("customer_retention.analysis.diagnostics.model_diagnostics_report.ModelValidityGate")
    def test_leakage_called_once(self, mock_gate, mock_cal, mock_of, mock_cv, mock_leak,
                                  generator, sample_data, feature_names):
        X_train, X_test, y_train, y_test = sample_data
        mock_leak.return_value.run_all_checks.return_value = _mock_leakage()
        mock_cv.return_value.run_all.return_value = _mock_cv_analysis()
        mock_of.return_value.analyze_train_test_gap.return_value = _mock_overfitting()
        mock_of.return_value.analyze_complexity.return_value = _mock_overfitting()
        mock_of.return_value.analyze_learning_curve.return_value = _mock_overfitting()
        mock_cal.return_value.analyze_calibration.return_value = _mock_calibration()
        mock_gate.return_value.run.return_value = _mock_validity()

        models = {"RF": _make_model([0.5, 0.3, 0.2]), "LR": _make_model([0.4, 0.4, 0.2])}
        cv_results = {n: _make_cv_results() for n in models}
        train_mets = {n: {"roc_auc": 0.85} for n in models}
        test_mets = {n: {"roc_auc": 0.80} for n in models}

        generator.generate(models, X_train, X_test, y_train, y_test,
                           cv_results, train_mets, test_mets, feature_names, "RF", 0.25)

        assert mock_leak.return_value.run_all_checks.call_count == 1

    @patch("customer_retention.analysis.diagnostics.model_diagnostics_report.LeakageDetector")
    @patch("customer_retention.analysis.diagnostics.model_diagnostics_report.CVAnalyzer")
    @patch("customer_retention.analysis.diagnostics.model_diagnostics_report.OverfittingAnalyzer")
    @patch("customer_retention.analysis.diagnostics.model_diagnostics_report.CalibrationAnalyzer")
    @patch("customer_retention.analysis.diagnostics.model_diagnostics_report.ModelValidityGate")
    def test_learning_curve_only_for_best(self, mock_gate, mock_cal, mock_of, mock_cv, mock_leak,
                                           generator, sample_data, feature_names):
        X_train, X_test, y_train, y_test = sample_data
        mock_leak.return_value.run_all_checks.return_value = _mock_leakage()
        mock_cv.return_value.run_all.return_value = _mock_cv_analysis()
        mock_of.return_value.analyze_train_test_gap.return_value = _mock_overfitting()
        mock_of.return_value.analyze_complexity.return_value = _mock_overfitting()
        mock_of.return_value.analyze_learning_curve.return_value = _mock_overfitting()
        mock_cal.return_value.analyze_calibration.return_value = _mock_calibration()
        mock_gate.return_value.run.return_value = _mock_validity()

        models = {"RF": _make_model([0.5, 0.3, 0.2]), "LR": _make_model([0.4, 0.4, 0.2])}
        cv_results = {n: _make_cv_results() for n in models}
        train_mets = {n: {"roc_auc": 0.85} for n in models}
        test_mets = {n: {"roc_auc": 0.80} for n in models}

        report = generator.generate(models, X_train, X_test, y_train, y_test,
                                    cv_results, train_mets, test_mets, feature_names, "RF", 0.25)

        assert mock_of.return_value.analyze_learning_curve.call_count == 1
        assert report.best_model_learning_curve is not None


class TestSerialize:
    def test_serialize_produces_dict(self, generator, feature_names):
        report = ModelDiagnosticsReport(
            summaries={"A": ModelDiagnosticsSummary(
                model_name="A", cv_analysis=_mock_cv_analysis(),
                overfitting=_mock_overfitting(), calibration=_mock_calibration(),
                validity=_mock_validity(), feature_stability=None,
            )},
            leakage=_mock_leakage(),
            cross_model_agreement=CrossModelAgreement({}, ["a"], {}, 1.0),
            best_model_learning_curve=None,
            verdict="solid", critical_issues=[], recommendations=[],
        )
        serialized = ModelDiagnosticsReportGenerator.serialize(report)

        assert isinstance(serialized, dict)
        assert serialized["verdict"] == "solid"
        assert "summaries" in serialized
        assert "cross_model_agreement" in serialized


def _make_spark_model(feature_names, importances=None):
    wrapper = SparkClassifierWrapper(
        spark_model_class="LogisticRegression",
        spark_model_params={"maxIter": 10},
        feature_names=feature_names,
    )
    mock_fitted = MagicMock()
    if importances is not None:
        mock_fitted.coefficients.toArray.return_value = np.array(importances)
        del mock_fitted.featureImportances
    wrapper._fitted_model = mock_fitted
    wrapper._classes = np.array([0, 1])
    return wrapper


class TestDistributedSafety:
    @patch("customer_retention.analysis.diagnostics.model_diagnostics_report.LeakageDetector")
    @patch("customer_retention.analysis.diagnostics.model_diagnostics_report.CVAnalyzer")
    @patch("customer_retention.analysis.diagnostics.model_diagnostics_report.OverfittingAnalyzer")
    @patch("customer_retention.analysis.diagnostics.model_diagnostics_report.CalibrationAnalyzer")
    @patch("customer_retention.analysis.diagnostics.model_diagnostics_report.ModelValidityGate")
    def test_spark_models_skip_learning_curve(self, mock_gate, mock_cal, mock_of, mock_cv, mock_leak,
                                               generator, sample_data, feature_names):
        X_train, X_test, y_train, y_test = sample_data
        mock_leak.return_value.run_all_checks.return_value = _mock_leakage()
        mock_cv.return_value.run_all.return_value = _mock_cv_analysis()
        mock_of.return_value.analyze_train_test_gap.return_value = _mock_overfitting()
        mock_cal.return_value.analyze_calibration.return_value = _mock_calibration()
        mock_gate.return_value.run.return_value = _mock_validity()

        spark_model = _make_spark_model(feature_names, [0.5, 0.3, 0.2])
        spark_model.predict_proba = MagicMock(return_value=np.column_stack([np.random.rand(50), np.random.rand(50)]))
        models = {"RF": spark_model}
        cv_results = {"RF": _make_cv_results()}
        train_mets = {"RF": {"roc_auc": 0.85}}
        test_mets = {"RF": {"roc_auc": 0.80}}

        report = generator.generate(models, X_train, X_test, y_train, y_test,
                                    cv_results, train_mets, test_mets, feature_names, "RF", 0.25)

        mock_of.return_value.analyze_learning_curve.assert_not_called()
        assert report.best_model_learning_curve is None

    @patch("customer_retention.analysis.diagnostics.model_diagnostics_report.LeakageDetector")
    @patch("customer_retention.analysis.diagnostics.model_diagnostics_report.CVAnalyzer")
    @patch("customer_retention.analysis.diagnostics.model_diagnostics_report.OverfittingAnalyzer")
    @patch("customer_retention.analysis.diagnostics.model_diagnostics_report.CalibrationAnalyzer")
    @patch("customer_retention.analysis.diagnostics.model_diagnostics_report.ModelValidityGate")
    def test_spark_models_run_sequentially(self, mock_gate, mock_cal, mock_of, mock_cv, mock_leak,
                                            generator, sample_data, feature_names):
        X_train, X_test, y_train, y_test = sample_data
        mock_leak.return_value.run_all_checks.return_value = _mock_leakage()
        mock_cv.return_value.run_all.return_value = _mock_cv_analysis()
        mock_of.return_value.analyze_train_test_gap.return_value = _mock_overfitting()
        mock_cal.return_value.analyze_calibration.return_value = _mock_calibration()
        mock_gate.return_value.run.return_value = _mock_validity()

        spark_a = _make_spark_model(feature_names, [0.5, 0.3, 0.2])
        spark_a.predict_proba = MagicMock(return_value=np.column_stack([np.random.rand(50), np.random.rand(50)]))
        spark_b = _make_spark_model(feature_names, [0.4, 0.4, 0.2])
        spark_b.predict_proba = MagicMock(return_value=np.column_stack([np.random.rand(50), np.random.rand(50)]))
        models = {"A": spark_a, "B": spark_b}
        cv_results = {n: _make_cv_results() for n in models}
        train_mets = {n: {"roc_auc": 0.85} for n in models}
        test_mets = {n: {"roc_auc": 0.80} for n in models}

        with patch("customer_retention.analysis.diagnostics.model_diagnostics_report.ThreadPoolExecutor") as mock_tpe:
            report = generator.generate(models, X_train, X_test, y_train, y_test,
                                        cv_results, train_mets, test_mets, feature_names, "A", 0.25)
            mock_tpe.assert_not_called()

        assert len(report.summaries) == 2


class TestFeatureStabilityIntegration:
    @patch("customer_retention.analysis.diagnostics.model_diagnostics_report.LeakageDetector")
    @patch("customer_retention.analysis.diagnostics.model_diagnostics_report.CVAnalyzer")
    @patch("customer_retention.analysis.diagnostics.model_diagnostics_report.OverfittingAnalyzer")
    @patch("customer_retention.analysis.diagnostics.model_diagnostics_report.CalibrationAnalyzer")
    @patch("customer_retention.analysis.diagnostics.model_diagnostics_report.ModelValidityGate")
    def test_fold_importances_produce_stability(self, mock_gate, mock_cal, mock_of, mock_cv, mock_leak,
                                                 generator, sample_data, feature_names):
        X_train, X_test, y_train, y_test = sample_data
        mock_leak.return_value.run_all_checks.return_value = _mock_leakage()
        mock_cv.return_value.run_all.return_value = _mock_cv_analysis()
        mock_of.return_value.analyze_train_test_gap.return_value = _mock_overfitting()
        mock_of.return_value.analyze_complexity.return_value = _mock_overfitting()
        mock_of.return_value.analyze_learning_curve.return_value = _mock_overfitting()
        mock_cal.return_value.analyze_calibration.return_value = _mock_calibration()
        mock_gate.return_value.run.return_value = _mock_validity()

        models = {"RF": _make_model([0.5, 0.3, 0.2])}
        cv_results = {"RF": _make_cv_results(with_importance=True, feature_names=feature_names)}
        train_mets = {"RF": {"roc_auc": 0.85}}
        test_mets = {"RF": {"roc_auc": 0.80}}

        report = generator.generate(models, X_train, X_test, y_train, y_test,
                                    cv_results, train_mets, test_mets, feature_names, "RF", 0.25)

        assert report.summaries["RF"].feature_stability is not None
        assert isinstance(report.summaries["RF"].feature_stability, FeatureStabilityResult)
