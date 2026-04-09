from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from customer_retention.analysis.diagnostics.feature_stability import FeatureStabilityResult
from customer_retention.analysis.diagnostics.leakage_detector import LeakageCheck, LeakageResult
from customer_retention.analysis.diagnostics.model_diagnostics_report import (
    CrossModelAgreement,
    ModelDiagnosticsReport,
    ModelDiagnosticsReportGenerator,
    ModelDiagnosticsSummary,
    compute_and_persist_diagnostics,
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


def _make_leakage_with(critical=False):
    check = LeakageCheck(
        check_id="LD001", feature="feat_x", severity=Severity.CRITICAL if critical else Severity.LOW,
        recommendation="...",
    )
    return LeakageResult(passed=not critical, checks=[check] if critical else [])


def _make_model(importances):
    m = MagicMock()
    m.feature_importances_ = np.array(importances)
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
            detail["feature_importance"] = {f: float(np.random.rand()) for f in feature_names}
        result["fold_details"].append(detail)
    return result


def _make_recommendations(prioritize: list = None, target_column: str = "target", analyzed: list = None):
    """Build a minimal RecommendationRegistry-like stub with prioritized features."""
    from customer_retention.analysis.auto_explorer.layered_recommendations import (
        GoldRecommendations,
        LayeredRecommendation,
        RecommendationRegistry,
    )
    registry = RecommendationRegistry()
    registry.gold = GoldRecommendations(target_column=target_column)
    for i, (col, corr) in enumerate(prioritize or []):
        registry.gold.feature_selection.append(LayeredRecommendation(
            id=f"rec_{i}", layer="gold", category="feature_selection", action="prioritize",
            target_column=col, parameters={"correlation": corr, "effect_size": 0.0},
            rationale="test", source_notebook="05_relationship_analysis",
        ))
    if analyzed is not None:
        registry.set_feature_selection_config(
            variance_threshold=0.01, correlation_threshold=0.95,
            analyzed_features=analyzed,
        )
    return registry


@pytest.fixture
def feature_names():
    return ["feat_a", "feat_b", "feat_c"]


@pytest.fixture
def generator():
    return ModelDiagnosticsReportGenerator()


@pytest.fixture
def y_test_binary():
    np.random.seed(42)
    return np.random.choice([0, 1], 30)


@pytest.fixture
def models_dict():
    return {
        "RF": _make_model([0.5, 0.3, 0.2]),
        "LR": _make_model([0.4, 0.4, 0.2]),
    }


@pytest.fixture
def cv_results_dict(models_dict):
    return {n: _make_cv_results() for n in models_dict}


@pytest.fixture
def train_metrics_dict(models_dict):
    return {n: {"roc_auc": 0.85, "pr_auc": 0.70} for n in models_dict}


@pytest.fixture
def test_metrics_dict(models_dict):
    return {n: {"roc_auc": 0.80, "pr_auc": 0.65, "f1": 0.55, "precision": 0.5, "recall": 0.6} for n in models_dict}


@pytest.fixture
def predictions_dict(y_test_binary):
    n = len(y_test_binary)
    proba = np.column_stack([1 - np.linspace(0, 1, n), np.linspace(0, 1, n)])
    y_pred = np.argmax(proba, axis=1)
    return {
        "RF": {"y_proba_test": proba, "y_pred": y_pred},
        "LR": {"y_proba_test": proba, "y_pred": y_pred},
    }


# ---------------------------------------------------------------------------
# Cached leakage from NB05 recommendations
# ---------------------------------------------------------------------------


class TestLeakageFromCache:
    def test_no_recommendations_returns_only_pattern_checks(self, generator, feature_names):
        result = generator._build_leakage_from_recommendations(None, feature_names, "target")
        assert isinstance(result, LeakageResult)
        assert result.checks == []

    def test_target_column_in_features_is_critical(self, generator):
        result = generator._build_leakage_from_recommendations(None, ["target", "feat_a"], "target")
        criticals = [c for c in result.checks if c.severity == Severity.CRITICAL]
        assert any(c.feature == "target" for c in criticals)

    def test_target_derived_name_pattern_is_critical(self, generator):
        result = generator._build_leakage_from_recommendations(None, ["target_proxy", "feat_a"], "target")
        assert any(c.feature == "target_proxy" and c.severity == Severity.CRITICAL for c in result.checks)

    def test_high_cached_correlation_becomes_high_leakage(self, generator, feature_names):
        recs = _make_recommendations([("feat_a", 0.85), ("feat_b", 0.40)])
        result = generator._build_leakage_from_recommendations(recs, feature_names, "target")
        a_check = next(c for c in result.checks if c.feature == "feat_a")
        assert a_check.severity == Severity.HIGH
        # 0.40 < 0.50 medium threshold, so feat_b shouldn't appear
        assert not any(c.feature == "feat_b" for c in result.checks)

    def test_critical_cached_correlation_becomes_critical_leakage(self, generator, feature_names):
        recs = _make_recommendations([("feat_a", 0.95)])
        result = generator._build_leakage_from_recommendations(recs, feature_names, "target")
        a_check = next(c for c in result.checks if c.feature == "feat_a")
        assert a_check.severity == Severity.CRITICAL
        assert "0.95" in a_check.recommendation

    def test_medium_cached_correlation_becomes_medium_leakage(self, generator, feature_names):
        recs = _make_recommendations([("feat_a", 0.55)])
        result = generator._build_leakage_from_recommendations(recs, feature_names, "target")
        a_check = next(c for c in result.checks if c.feature == "feat_a")
        assert a_check.severity == Severity.MEDIUM

    def test_temporal_named_feature_appends_temporal_note(self, generator, feature_names):
        recs = _make_recommendations([("days_since_event", 0.92)])
        result = generator._build_leakage_from_recommendations(recs, feature_names, "target")
        check = next(c for c in result.checks if c.feature == "days_since_event")
        assert "Temporal" in check.recommendation or "reference date" in check.recommendation.lower()

    def test_domain_target_pattern_appends_domain_note(self, generator, feature_names):
        recs = _make_recommendations([("churn_score", 0.88)])
        result = generator._build_leakage_from_recommendations(recs, feature_names, "target")
        check = next(c for c in result.checks if c.feature == "churn_score")
        assert "Domain" in check.recommendation or "semantic" in check.recommendation.lower()

    def test_safe_leakage_swallows_failures(self, generator, feature_names, capsys):
        with patch.object(generator, "_build_leakage_from_recommendations", side_effect=RuntimeError("explode")):
            result = generator._safe_leakage_from_cache(None, feature_names, "target")
        assert isinstance(result, LeakageResult)
        assert result.checks == []
        captured = capsys.readouterr()
        assert "[diagnostics] cached leakage build failed" in captured.out


class TestWindowOverlapGoldOutputGate:
    """LD062/LD063 second-pass scan against the actual model feature set.

    Catches gold-derived features (e.g. `_is_zero` flags, windowed
    counts/sums) that NB05's cached correlation view does not see.
    """

    def test_zero_inflation_flag_is_critical(self, generator):
        feature_names = ["NET_PRICE_count_180d_is_zero", "safe_feature"]
        result = generator._build_leakage_from_recommendations(
            None, feature_names, "target", label_horizon_days=30,
        )
        assert any(
            c.check_id == "LD062"
            and c.feature == "NET_PRICE_count_180d_is_zero"
            and c.severity == Severity.CRITICAL
            for c in result.checks
        )

    def test_windowed_count_above_horizon_is_high(self, generator):
        feature_names = ["NET_PRICE_count_180d", "safe_feature"]
        result = generator._build_leakage_from_recommendations(
            None, feature_names, "target", label_horizon_days=30,
        )
        assert any(
            c.check_id == "LD063"
            and c.feature == "NET_PRICE_count_180d"
            and c.severity == Severity.HIGH
            for c in result.checks
        )

    def test_window_below_horizon_is_safe(self, generator):
        feature_names = ["NET_PRICE_count_7d", "NET_PRICE_count_24h"]
        result = generator._build_leakage_from_recommendations(
            None, feature_names, "target", label_horizon_days=30,
        )
        assert not any(c.check_id in ("LD062", "LD063") for c in result.checks)

    def test_no_horizon_is_noop(self, generator):
        feature_names = ["NET_PRICE_count_180d_is_zero", "NET_PRICE_count_365d"]
        # Without label_horizon_days the gate must not fire — preserves
        # backwards compatibility for callers that don't yet pass it through.
        result = generator._build_leakage_from_recommendations(
            None, feature_names, "target", label_horizon_days=None,
        )
        assert not any(c.check_id in ("LD062", "LD063") for c in result.checks)

    def test_other_aggregations_are_not_flagged(self, generator):
        # mean/max/min/std on a long window are not flagged: they describe
        # the SHAPE of activity, not its presence/absence. Only count/sum
        # (LD063) and `_is_zero` (LD062) are leak-shaped.
        feature_names = ["NET_PRICE_mean_180d", "NET_PRICE_max_365d"]
        result = generator._build_leakage_from_recommendations(
            None, feature_names, "target", label_horizon_days=30,
        )
        assert not any(c.check_id in ("LD062", "LD063") for c in result.checks)

    def test_window_overlap_runs_alongside_cached_correlation_checks(self, generator):
        # End-to-end: a feature with a cached HIGH correlation AND another
        # feature that triggers LD062 should both surface in the same result.
        recs = _make_recommendations([("feat_a", 0.85)])
        feature_names = ["feat_a", "feat_b", "TXN_count_180d_is_zero"]
        result = generator._build_leakage_from_recommendations(
            recs, feature_names, "target", label_horizon_days=30,
        )
        assert any(c.feature == "feat_a" and c.severity == Severity.HIGH for c in result.checks)
        assert any(c.check_id == "LD062" and c.feature == "TXN_count_180d_is_zero" for c in result.checks)


# ---------------------------------------------------------------------------
# Cross-model agreement (unchanged from prior — feature_importances only)
# ---------------------------------------------------------------------------


class TestCrossModelAgreement:
    def test_identical_importances_perfect_jaccard(self, generator, feature_names):
        models = {"A": _make_model([0.5, 0.3, 0.2]), "B": _make_model([0.5, 0.3, 0.2])}
        agreement = generator._compute_cross_model_agreement(models, feature_names, top_n=3)
        assert agreement.agreement_score == 1.0
        assert set(agreement.consensus_features) == set(feature_names)

    def test_disjoint_importances_zero_jaccard(self, generator):
        models = {"A": _make_model([0.5, 0.3, 0.0, 0.0]), "B": _make_model([0.0, 0.0, 0.5, 0.3])}
        agreement = generator._compute_cross_model_agreement(models, ["a", "b", "c", "d"], top_n=2)
        assert agreement.agreement_score == 0.0
        assert agreement.consensus_features == []

    def test_single_model_perfect_agreement(self, generator, feature_names):
        models = {"A": _make_model([0.5, 0.3, 0.2])}
        agreement = generator._compute_cross_model_agreement(models, feature_names, top_n=3)
        assert agreement.agreement_score == 1.0

    def test_lr_model_uses_coef(self, generator):
        model = MagicMock(spec=[])
        model.coef_ = np.array([[0.5, -0.3, 0.1]])
        agreement = generator._compute_cross_model_agreement({"LR": model}, ["a", "b", "c"], top_n=2)
        assert "a" in agreement.consensus_features or "b" in agreement.consensus_features


# ---------------------------------------------------------------------------
# Verdict logic (unchanged)
# ---------------------------------------------------------------------------


class TestVerdictLogic:
    def test_verdict_solid(self, generator):
        summaries = {"A": MagicMock(
            cv_analysis=_mock_cv_analysis(True), overfitting=_mock_overfitting(True),
            calibration=_mock_calibration(True), validity=_mock_validity(True), feature_stability=None,
        )}
        verdict, issues, _ = generator._compute_verdict(summaries, _make_leakage_with(critical=False))
        assert verdict == "solid"
        assert issues == []

    def test_verdict_leaky(self, generator):
        summaries = {"A": MagicMock(
            cv_analysis=_mock_cv_analysis(True), overfitting=_mock_overfitting(True),
            calibration=_mock_calibration(True), validity=_mock_validity(True), feature_stability=None,
        )}
        verdict, issues, _ = generator._compute_verdict(summaries, _make_leakage_with(critical=True))
        assert verdict == "leaky"
        assert issues

    def test_verdict_overfit(self, generator):
        summaries = {"A": MagicMock(
            cv_analysis=_mock_cv_analysis(True),
            overfitting=_mock_overfitting(False, has_critical=True),
            calibration=_mock_calibration(True), validity=_mock_validity(True), feature_stability=None,
        )}
        verdict, _, _ = generator._compute_verdict(summaries, _make_leakage_with(critical=False))
        assert verdict == "overfit"

    def test_verdict_unstable(self, generator):
        cv = _mock_cv_analysis(False)
        check = MagicMock()
        check.severity = Severity.CRITICAL
        cv.checks = [check]
        summaries = {"A": MagicMock(
            cv_analysis=cv, overfitting=_mock_overfitting(True),
            calibration=_mock_calibration(True), validity=_mock_validity(True), feature_stability=None,
        )}
        verdict, _, _ = generator._compute_verdict(summaries, _make_leakage_with(critical=False))
        assert verdict == "unstable"

    def test_verdict_caution_on_high_issues(self, generator):
        validity = _mock_validity(False)
        issue = MagicMock()
        issue.severity = Severity.HIGH
        validity.high_issues = [issue]
        summaries = {"A": MagicMock(
            cv_analysis=_mock_cv_analysis(True), overfitting=_mock_overfitting(True),
            calibration=_mock_calibration(True), validity=validity, feature_stability=None,
        )}
        verdict, _, _ = generator._compute_verdict(summaries, _make_leakage_with(critical=False))
        assert verdict == "caution"


# ---------------------------------------------------------------------------
# Generate orchestration — new kwargs API, no fresh fits, no learning curve
# ---------------------------------------------------------------------------


class TestGenerate:
    def test_no_threadpoolexecutor_imported_or_used(self):
        import customer_retention.analysis.diagnostics.model_diagnostics_report as mod
        assert not hasattr(mod, "ThreadPoolExecutor"), (
            "ThreadPoolExecutor must not be imported — it is banned on Databricks shared clusters"
        )

    def test_learning_curve_always_none(
        self, generator, models_dict, cv_results_dict, train_metrics_dict,
        test_metrics_dict, predictions_dict, feature_names, y_test_binary,
    ):
        report = generator.generate(
            models=models_dict, cv_results=cv_results_dict,
            train_metrics=train_metrics_dict, test_metrics=test_metrics_dict,
            feature_names=feature_names, best_model_name="RF",
            class_proportion=0.5, predictions=predictions_dict, y_test=y_test_binary,
        )
        assert report.best_model_learning_curve is None

    def test_per_model_summaries_built_for_each_model(
        self, generator, models_dict, cv_results_dict, train_metrics_dict,
        test_metrics_dict, predictions_dict, feature_names, y_test_binary,
    ):
        report = generator.generate(
            models=models_dict, cv_results=cv_results_dict,
            train_metrics=train_metrics_dict, test_metrics=test_metrics_dict,
            feature_names=feature_names, best_model_name="RF",
            class_proportion=0.5, predictions=predictions_dict, y_test=y_test_binary,
        )
        assert set(report.summaries) == {"RF", "LR"}

    def test_calibration_uses_precomputed_test_proba_with_test_labels(
        self, generator, models_dict, cv_results_dict, train_metrics_dict,
        test_metrics_dict, predictions_dict, feature_names, y_test_binary,
    ):
        with patch(
            "customer_retention.analysis.diagnostics.model_diagnostics_report.CalibrationAnalyzer"
        ) as mock_cal:
            mock_cal.return_value.analyze_calibration.return_value = _mock_calibration()
            generator.generate(
                models=models_dict, cv_results=cv_results_dict,
                train_metrics=train_metrics_dict, test_metrics=test_metrics_dict,
                feature_names=feature_names, best_model_name="RF",
                class_proportion=0.5, predictions=predictions_dict, y_test=y_test_binary,
            )
            assert mock_cal.return_value.analyze_calibration.called
            call_args = mock_cal.return_value.analyze_calibration.call_args
            passed_y_true, passed_y_proba = call_args[0]
            assert len(passed_y_true) == len(y_test_binary)
            assert len(passed_y_proba) == len(y_test_binary)
            np.testing.assert_array_equal(passed_y_true, np.asarray(y_test_binary))
            np.testing.assert_array_equal(passed_y_proba, predictions_dict["RF"]["y_proba_test"][:, 1])

    def test_calibration_skipped_when_predictions_missing(
        self, generator, models_dict, cv_results_dict, train_metrics_dict,
        test_metrics_dict, feature_names, y_test_binary,
    ):
        with patch(
            "customer_retention.analysis.diagnostics.model_diagnostics_report.CalibrationAnalyzer"
        ) as mock_cal:
            generator.generate(
                models=models_dict, cv_results=cv_results_dict,
                train_metrics=train_metrics_dict, test_metrics=test_metrics_dict,
                feature_names=feature_names, best_model_name="RF",
                class_proportion=0.5, predictions={}, y_test=y_test_binary,
            )
            mock_cal.return_value.analyze_calibration.assert_not_called()

    def test_per_model_failure_skips_only_that_model(
        self, generator, models_dict, cv_results_dict, train_metrics_dict,
        test_metrics_dict, predictions_dict, feature_names, y_test_binary,
    ):
        with patch.object(generator, "_run_per_model") as run_per_model:
            def fake(name, *args, **kwargs):
                if name == "RF":
                    raise ValueError("synthetic per-model failure")
                return ModelDiagnosticsSummary(
                    model_name=name, cv_analysis=_mock_cv_analysis(),
                    overfitting=_mock_overfitting(), calibration=_mock_calibration(),
                    validity=_mock_validity(), feature_stability=None,
                )
            run_per_model.side_effect = fake
            report = generator.generate(
                models=models_dict, cv_results=cv_results_dict,
                train_metrics=train_metrics_dict, test_metrics=test_metrics_dict,
                feature_names=feature_names, best_model_name="LR",
                class_proportion=0.5, predictions=predictions_dict, y_test=y_test_binary,
            )
        assert "LR" in report.summaries
        assert "RF" not in report.summaries

    def test_leakage_built_from_cached_recommendations(
        self, generator, models_dict, cv_results_dict, train_metrics_dict,
        test_metrics_dict, predictions_dict, feature_names, y_test_binary,
    ):
        recs = _make_recommendations([("feat_a", 0.95)])
        report = generator.generate(
            models=models_dict, cv_results=cv_results_dict,
            train_metrics=train_metrics_dict, test_metrics=test_metrics_dict,
            feature_names=feature_names, best_model_name="RF",
            class_proportion=0.5, predictions=predictions_dict, y_test=y_test_binary,
            recommendations=recs, target_column="target",
        )
        feats_with_critical = [c.feature for c in report.leakage.checks if c.severity == Severity.CRITICAL]
        assert "feat_a" in feats_with_critical
        assert report.verdict == "leaky"


class TestLeakageCoverage:
    def test_no_recommendations_marks_all_features_as_unanalyzed(
        self, generator, models_dict, cv_results_dict, train_metrics_dict,
        test_metrics_dict, predictions_dict, feature_names, y_test_binary,
    ):
        report = generator.generate(
            models=models_dict, cv_results=cv_results_dict,
            train_metrics=train_metrics_dict, test_metrics=test_metrics_dict,
            feature_names=feature_names, best_model_name="RF",
            class_proportion=0.5, predictions=predictions_dict, y_test=y_test_binary,
            recommendations=None,
        )
        assert report.leakage_coverage is not None
        assert report.leakage_coverage.total_features == len(feature_names)
        assert report.leakage_coverage.analyzed_in_nb05 == 0
        assert set(report.leakage_coverage.unanalyzed) == set(feature_names)

    def test_features_in_nb05_analyzed_set_count_as_covered(
        self, generator, models_dict, cv_results_dict, train_metrics_dict,
        test_metrics_dict, predictions_dict, feature_names, y_test_binary,
    ):
        # NB05 saw feat_a + feat_b but not feat_c (added by gold transforms)
        recs = _make_recommendations(prioritize=[], analyzed=["feat_a", "feat_b"])
        report = generator.generate(
            models=models_dict, cv_results=cv_results_dict,
            train_metrics=train_metrics_dict, test_metrics=test_metrics_dict,
            feature_names=feature_names, best_model_name="RF",
            class_proportion=0.5, predictions=predictions_dict, y_test=y_test_binary,
            recommendations=recs,
        )
        cov = report.leakage_coverage
        assert cov.total_features == 3
        assert cov.analyzed_in_nb05 == 2
        assert cov.unanalyzed == ["feat_c"]

    def test_low_correlation_features_in_analyzed_set_are_silently_safe(
        self, generator, models_dict, cv_results_dict, train_metrics_dict,
        test_metrics_dict, predictions_dict, feature_names, y_test_binary,
    ):
        # NB05 analyzed all 3 features but only feat_a was strong enough to be prioritized
        recs = _make_recommendations(
            prioritize=[("feat_a", 0.4)],
            analyzed=feature_names,
        )
        report = generator.generate(
            models=models_dict, cv_results=cv_results_dict,
            train_metrics=train_metrics_dict, test_metrics=test_metrics_dict,
            feature_names=feature_names, best_model_name="RF",
            class_proportion=0.5, predictions=predictions_dict, y_test=y_test_binary,
            recommendations=recs,
        )
        # Coverage = full, no leakage warnings
        assert report.leakage_coverage.analyzed_in_nb05 == len(feature_names)
        assert report.leakage_coverage.unanalyzed == []
        # 0.4 < MEDIUM (0.5) → no leakage check
        assert not any(c.feature == "feat_a" for c in report.leakage.checks)

    def test_unanalyzed_sample_capped(self, generator, models_dict, cv_results_dict,
                                       train_metrics_dict, test_metrics_dict,
                                       predictions_dict, y_test_binary):
        many_features = [f"feat_{i}" for i in range(50)]
        # cv/metrics dicts only need to cover the model names, not feature counts
        report = generator.generate(
            models=models_dict, cv_results=cv_results_dict,
            train_metrics=train_metrics_dict, test_metrics=test_metrics_dict,
            feature_names=many_features, best_model_name="RF",
            class_proportion=0.5, predictions=predictions_dict, y_test=y_test_binary,
            recommendations=None,
        )
        assert report.leakage_coverage.total_features == 50
        # Sample is capped to UNANALYZED_SAMPLE_SIZE (10)
        assert len(report.leakage_coverage.unanalyzed) == ModelDiagnosticsReportGenerator.UNANALYZED_SAMPLE_SIZE


class TestSkippedAnalyses:
    def test_skipped_analyses_populated(
        self, generator, models_dict, cv_results_dict, train_metrics_dict,
        test_metrics_dict, predictions_dict, feature_names, y_test_binary,
    ):
        report = generator.generate(
            models=models_dict, cv_results=cv_results_dict,
            train_metrics=train_metrics_dict, test_metrics=test_metrics_dict,
            feature_names=feature_names, best_model_name="RF",
            class_proportion=0.5, predictions=predictions_dict, y_test=y_test_binary,
        )
        assert isinstance(report.skipped_analyses, list)
        joined = " | ".join(report.skipped_analyses)
        assert "learning_curve" in joined
        assert "per_feature_single_auc" in joined
        assert "per_feature_class_separation" in joined

    def test_skipped_analyses_explain_why(
        self, generator, models_dict, cv_results_dict, train_metrics_dict,
        test_metrics_dict, predictions_dict, feature_names, y_test_binary,
    ):
        report = generator.generate(
            models=models_dict, cv_results=cv_results_dict,
            train_metrics=train_metrics_dict, test_metrics=test_metrics_dict,
            feature_names=feature_names, best_model_name="RF",
            class_proportion=0.5, predictions=predictions_dict, y_test=y_test_binary,
        )
        # Each entry must include a colon-separated reason, not just the name
        for entry in report.skipped_analyses:
            assert ":" in entry, f"skipped_analyses entry missing reason: {entry}"
            label, _, reason = entry.partition(":")
            assert reason.strip(), f"empty reason for {label}"

    def test_skipped_analyses_round_trip_through_jsonable(
        self, generator, models_dict, cv_results_dict, train_metrics_dict,
        test_metrics_dict, predictions_dict, feature_names, y_test_binary,
    ):
        import json
        report = generator.generate(
            models=models_dict, cv_results=cv_results_dict,
            train_metrics=train_metrics_dict, test_metrics=test_metrics_dict,
            feature_names=feature_names, best_model_name="RF",
            class_proportion=0.5, predictions=predictions_dict, y_test=y_test_binary,
        )
        payload = ModelDiagnosticsReportGenerator.to_jsonable(report)
        decoded = json.loads(json.dumps(payload, default=str))
        assert decoded["skipped_analyses"] == report.skipped_analyses
        assert decoded["leakage_coverage"]["total_features"] == len(feature_names)


class TestFeatureStabilityIntegration:
    def test_fold_importances_produce_stability(
        self, generator, models_dict, train_metrics_dict, test_metrics_dict,
        predictions_dict, feature_names, y_test_binary,
    ):
        cv = {n: _make_cv_results(with_importance=True, feature_names=feature_names) for n in models_dict}
        report = generator.generate(
            models=models_dict, cv_results=cv,
            train_metrics=train_metrics_dict, test_metrics=test_metrics_dict,
            feature_names=feature_names, best_model_name="RF",
            class_proportion=0.5, predictions=predictions_dict, y_test=y_test_binary,
        )
        assert report.summaries["RF"].feature_stability is not None
        assert isinstance(report.summaries["RF"].feature_stability, FeatureStabilityResult)


# ---------------------------------------------------------------------------
# JSON serialization
# ---------------------------------------------------------------------------


class TestToJsonable:
    def test_round_trip_via_json_module(self):
        import json
        report = ModelDiagnosticsReport(
            summaries={
                "A": ModelDiagnosticsSummary(
                    model_name="A", cv_analysis=_mock_cv_analysis(),
                    overfitting=_mock_overfitting(), calibration=_mock_calibration(),
                    validity=_mock_validity(), feature_stability=None,
                ),
            },
            leakage=_make_leakage_with(critical=False),
            cross_model_agreement=CrossModelAgreement({"A vs B": 0.5}, ["a"], {"A": []}, 0.5),
            best_model_learning_curve=None,
            verdict="solid", critical_issues=[], recommendations=["rec1"],
        )
        payload = ModelDiagnosticsReportGenerator.to_jsonable(report)
        decoded = json.loads(json.dumps(payload, default=str))
        assert decoded["verdict"] == "solid"
        assert decoded["recommendations"] == ["rec1"]
        assert decoded["cross_model_agreement"]["agreement_score"] == 0.5
        assert "summaries" in decoded
        assert decoded["best_model_learning_curve"] is None

    def test_severity_enum_serialized_with_name(self):
        from customer_retention.analysis.diagnostics.model_diagnostics_report import _to_jsonable

        result = _to_jsonable(Severity.CRITICAL)
        assert isinstance(result, dict)
        assert result["name"] == "CRITICAL"

    def test_numpy_arrays_serialized_as_lists(self):
        from customer_retention.analysis.diagnostics.model_diagnostics_report import _to_jsonable

        assert _to_jsonable(np.array([1.5, 2.5, 3.5])) == [1.5, 2.5, 3.5]


# ---------------------------------------------------------------------------
# compute_and_persist_diagnostics
# ---------------------------------------------------------------------------


class TestComputeAndPersistDiagnostics:
    def _build_kwargs(self, tmp_path, feature_names, recs=None):
        np.random.seed(0)
        n_test = 30
        y_test = pd.Series(np.random.choice([0, 1], n_test))
        proba_test = np.column_stack([np.linspace(1, 0, n_test), np.linspace(0, 1, n_test)])
        y_pred = np.argmax(proba_test, axis=1)
        return dict(
            diagnostics_path=tmp_path / "exploration_diagnostics.json",
            models={"RF": _make_model([0.5, 0.3, 0.2])},
            predictions={"RF": {"y_proba_test": proba_test, "y_pred": y_pred}},
            feature_names=feature_names,
            cv_results={"RF": _make_cv_results()},
            train_metrics={"RF": {"roc_auc": 0.82, "pr_auc": 0.65}},
            test_metrics={"RF": {"roc_auc": 0.78, "pr_auc": 0.62, "f1": 0.55, "precision": 0.5, "recall": 0.6}},
            best_model_name="RF",
            y_test=y_test,
            class_proportion=0.5,
            recommendations=recs,
        )

    def test_writes_report_and_holdout_into_existing_file(self, tmp_path, feature_names, capsys):
        import json

        diag_path = tmp_path / "exploration_diagnostics.json"
        diag_path.write_text(json.dumps({"feature_names": feature_names, "best_model_name": "RF"}))

        kwargs = self._build_kwargs(tmp_path, feature_names)
        ok = compute_and_persist_diagnostics(**kwargs)

        assert ok is True
        on_disk = json.loads(diag_path.read_text())
        assert on_disk["feature_names"] == feature_names  # preserved
        assert "model_diagnostics_report" in on_disk
        assert on_disk["model_diagnostics_report"]["verdict"] in {"solid", "caution", "overfit", "leaky", "unstable"}
        assert "best_model_holdout_metrics" in on_disk
        holdout = on_disk["best_model_holdout_metrics"]
        assert holdout["model_name"] == "RF"
        cm = holdout["confusion_matrix"]
        assert cm["tn"] + cm["fp"] + cm["fn"] + cm["tp"] == len(kwargs["y_test"])
        assert "probability_stats" in holdout
        captured = capsys.readouterr()
        assert "[diagnostics] building report" in captured.out
        assert "[diagnostics] complete" in captured.out

    def test_missing_predictions_returns_false_without_raising(self, tmp_path, feature_names):
        ok = compute_and_persist_diagnostics(
            diagnostics_path=tmp_path / "exploration_diagnostics.json",
            models={"RF": _make_model([0.5, 0.3, 0.2])},
            predictions={},
            feature_names=feature_names,
            cv_results={"RF": _make_cv_results()},
            train_metrics={"RF": {"roc_auc": 0.85}},
            test_metrics={"RF": {"roc_auc": 0.80}},
            best_model_name="RF",
            y_test=pd.Series([0, 1, 0, 1]),
            class_proportion=0.5,
        )
        assert ok is False  # holdout absent → partial → False

    def test_holdout_uses_cached_y_pred_when_present(self, tmp_path, feature_names):
        import json

        kwargs = self._build_kwargs(tmp_path, feature_names)
        # Inject a deliberately-wrong y_pred so we can detect whether the helper reused it
        wrong_pred = np.zeros(len(kwargs["y_test"]), dtype=int)
        kwargs["predictions"]["RF"]["y_pred"] = wrong_pred
        compute_and_persist_diagnostics(**kwargs)
        on_disk = json.loads(kwargs["diagnostics_path"].read_text())
        cm = on_disk["best_model_holdout_metrics"]["confusion_matrix"]
        # All preds == 0 → no positive predictions → fp == 0 and tp == 0
        assert cm["fp"] == 0
        assert cm["tp"] == 0

    def test_holdout_falls_back_to_threshold_when_no_cached_pred(self, tmp_path, feature_names):
        import json

        kwargs = self._build_kwargs(tmp_path, feature_names)
        kwargs["predictions"]["RF"].pop("y_pred", None)
        compute_and_persist_diagnostics(**kwargs)
        on_disk = json.loads(kwargs["diagnostics_path"].read_text())
        # Threshold-derived y_pred exists and confusion matrix sums to n
        cm = on_disk["best_model_holdout_metrics"]["confusion_matrix"]
        assert cm["tn"] + cm["fp"] + cm["fn"] + cm["tp"] == len(kwargs["y_test"])

    def test_report_failure_does_not_lose_holdout(self, tmp_path, feature_names):
        import json

        kwargs = self._build_kwargs(tmp_path, feature_names)
        with patch.object(ModelDiagnosticsReportGenerator, "generate", side_effect=RuntimeError("explode")):
            ok = compute_and_persist_diagnostics(**kwargs)

        assert ok is False
        on_disk = json.loads(kwargs["diagnostics_path"].read_text())
        assert "best_model_holdout_metrics" in on_disk
        assert "model_diagnostics_report" not in on_disk

    def test_total_failure_returns_false_without_raising(self, tmp_path, feature_names):
        kwargs = self._build_kwargs(tmp_path, feature_names)
        with patch.object(ModelDiagnosticsReportGenerator, "generate", side_effect=RuntimeError("boom")), \
             patch("customer_retention.analysis.diagnostics.model_diagnostics_report._compute_best_model_holdout",
                   side_effect=RuntimeError("also boom")):
            ok = compute_and_persist_diagnostics(**kwargs)
        assert ok is False
        if kwargs["diagnostics_path"].exists():
            import json
            on_disk = json.loads(kwargs["diagnostics_path"].read_text())
            assert "model_diagnostics_report" not in on_disk
            assert "best_model_holdout_metrics" not in on_disk

    def test_recommendations_propagate_into_leakage_section(self, tmp_path, feature_names):
        import json

        recs = _make_recommendations([("feat_a", 0.96)])
        kwargs = self._build_kwargs(tmp_path, feature_names, recs=recs)
        ok = compute_and_persist_diagnostics(**kwargs)
        assert ok is True
        on_disk = json.loads(kwargs["diagnostics_path"].read_text())
        leakage_checks = on_disk["model_diagnostics_report"]["leakage"]["checks"]
        assert any(c["feature"] == "feat_a" and c["severity"]["name"] == "CRITICAL" for c in leakage_checks)
        assert on_disk["model_diagnostics_report"]["verdict"] == "leaky"


# ---------------------------------------------------------------------------
# Spark wrapper safety
# ---------------------------------------------------------------------------


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


class TestSparkSafety:
    def test_spark_models_do_not_trigger_predict_proba(self, generator, feature_names, y_test_binary):
        spark_model = _make_spark_model(feature_names, [0.5, 0.3, 0.2])
        spark_model.predict_proba = MagicMock(side_effect=AssertionError("must not call predict_proba"))
        n = len(y_test_binary)
        proba = np.column_stack([1 - np.linspace(0, 1, n), np.linspace(0, 1, n)])
        report = generator.generate(
            models={"RF": spark_model},
            cv_results={"RF": _make_cv_results()},
            train_metrics={"RF": {"roc_auc": 0.85}},
            test_metrics={"RF": {"roc_auc": 0.80}},
            feature_names=feature_names, best_model_name="RF",
            class_proportion=0.5,
            predictions={"RF": {"y_proba_test": proba, "y_pred": np.argmax(proba, axis=1)}},
            y_test=y_test_binary,
        )
        assert "RF" in report.summaries
        assert report.best_model_learning_curve is None
