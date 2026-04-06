import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Set

import numpy as np

from customer_retention.analysis.diagnostics.calibration_analyzer import CalibrationAnalyzer, CalibrationResult
from customer_retention.analysis.diagnostics.cv_analyzer import CVAnalysisResult, CVAnalyzer
from customer_retention.analysis.diagnostics.feature_stability import FeatureStabilityAnalyzer, FeatureStabilityResult
from customer_retention.analysis.diagnostics.leakage_detector import LeakageDetector, LeakageResult
from customer_retention.analysis.diagnostics.overfitting_analyzer import OverfittingAnalyzer, OverfittingResult
from customer_retention.core.components.enums import Severity
from customer_retention.stages.validation.model_validity_gate import ModelValidityGate, ModelValidityResult


def _is_spark_wrapper(model: Any) -> bool:
    return type(model).__name__ == "SparkClassifierWrapper"


@dataclass
class CrossModelAgreement:
    pairwise_jaccard: Dict[str, float]
    consensus_features: List[str]
    model_specific_features: Dict[str, List[str]]
    agreement_score: float


@dataclass
class ModelDiagnosticsSummary:
    model_name: str
    cv_analysis: CVAnalysisResult
    overfitting: OverfittingResult
    calibration: CalibrationResult
    validity: ModelValidityResult
    feature_stability: Optional[FeatureStabilityResult]


@dataclass
class ModelDiagnosticsReport:
    summaries: Dict[str, ModelDiagnosticsSummary]
    leakage: LeakageResult
    cross_model_agreement: CrossModelAgreement
    best_model_learning_curve: Optional[OverfittingResult]
    verdict: str
    critical_issues: List[str]
    recommendations: List[str]


class ModelDiagnosticsReportGenerator:

    def generate(
        self, models: Dict[str, Any],
        X_train: Any, X_test: Any, y_train: Any, y_test: Any,
        cv_results: Dict[str, Dict], train_metrics: Dict[str, Dict[str, float]],
        test_metrics: Dict[str, Dict[str, float]], feature_names: List[str],
        best_model_name: str, class_proportion: float,
    ) -> ModelDiagnosticsReport:
        y_test_np = np.asarray(y_test)

        leakage = LeakageDetector().run_all_checks(X_train, y_train, include_pit=False)

        has_spark = any(_is_spark_wrapper(m) for m in models.values())
        summaries: Dict[str, ModelDiagnosticsSummary] = {}
        if has_spark:
            for name, model in models.items():
                summaries[name] = self._run_per_model(
                    name, model, X_train, y_test_np,
                    cv_results.get(name, {}), train_metrics.get(name, {}),
                    test_metrics.get(name, {}), feature_names,
                )
        else:
            max_workers = min(len(models), os.cpu_count() or 4)
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = {
                    pool.submit(
                        self._run_per_model, name, model, X_train, y_test_np,
                        cv_results.get(name, {}), train_metrics.get(name, {}),
                        test_metrics.get(name, {}), feature_names,
                    ): name
                    for name, model in models.items()
                }
                for future in as_completed(futures):
                    summaries[futures[future]] = future.result()

        learning_curve = None
        best = models.get(best_model_name)
        if best is not None and not _is_spark_wrapper(best):
            learning_curve = OverfittingAnalyzer().analyze_learning_curve(best, X_train, y_train, cv=3)

        agreement = self._compute_cross_model_agreement(models, feature_names)
        verdict, critical_issues, recs = self._compute_verdict(summaries, leakage)

        return ModelDiagnosticsReport(
            summaries=summaries, leakage=leakage,
            cross_model_agreement=agreement,
            best_model_learning_curve=learning_curve,
            verdict=verdict, critical_issues=critical_issues, recommendations=recs,
        )

    def _run_per_model(
        self, name: str, model: Any, X_train: Any, y_test_np: np.ndarray,
        cv_data: Dict, train_mets: Dict[str, float], test_mets: Dict[str, float],
        feature_names: List[str],
    ) -> ModelDiagnosticsSummary:
        cv_scores = cv_data.get("cv_scores", [])
        test_score = test_mets.get("roc_auc") or test_mets.get("pr_auc")

        cv_analysis = CVAnalyzer().run_all(cv_scores, test_score)
        overfitting = OverfittingAnalyzer().analyze_train_test_gap(train_mets, test_mets)

        y_proba = model.predict_proba(X_train)
        if y_proba.ndim == 2:
            y_proba_pos = y_proba[:, 1]
        else:
            y_proba_pos = y_proba
        calibration = CalibrationAnalyzer().analyze_calibration(y_test_np, y_proba_pos[:len(y_test_np)])

        validity = ModelValidityGate().run(test_mets)

        fold_details = cv_data.get("fold_details", [])
        fold_importances = [d["feature_importance"] for d in fold_details if "feature_importance" in d]
        feature_stability = None
        if fold_importances:
            feature_stability = FeatureStabilityAnalyzer().analyze(fold_importances)

        return ModelDiagnosticsSummary(
            model_name=name, cv_analysis=cv_analysis, overfitting=overfitting,
            calibration=calibration, validity=validity,
            feature_stability=feature_stability,
        )

    def _compute_cross_model_agreement(
        self, models: Dict[str, Any], feature_names: List[str], top_n: int = 20,
    ) -> CrossModelAgreement:
        model_tops = {name: self._get_model_top_features(m, feature_names, top_n)
                      for name, m in models.items()}

        if len(model_tops) < 2:
            all_consensus = sorted(set.union(*model_tops.values())) if model_tops else []
            return CrossModelAgreement({}, all_consensus, {n: [] for n in model_tops}, 1.0)

        names = list(model_tops.keys())
        pairwise: Dict[str, float] = {}
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                inter = len(model_tops[a] & model_tops[b])
                union = len(model_tops[a] | model_tops[b])
                pairwise[f"{a} vs {b}"] = inter / union if union > 0 else 0.0

        consensus = sorted(set.intersection(*model_tops.values()))
        model_specific: Dict[str, List[str]] = {}
        for name, tops in model_tops.items():
            others = set.union(*(t for n, t in model_tops.items() if n != name))
            model_specific[name] = sorted(tops - others)

        avg_jaccard = sum(pairwise.values()) / len(pairwise) if pairwise else 1.0
        return CrossModelAgreement(pairwise, consensus, model_specific, avg_jaccard)

    @staticmethod
    def _get_model_top_features(model: Any, feature_names: List[str], top_n: int) -> Set[str]:
        importances = None
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_).flatten()
        if importances is None:
            return set(feature_names[:top_n])
        indices = np.argsort(importances)[::-1][:top_n]
        return {feature_names[i] for i in indices if i < len(feature_names)}

    def _compute_verdict(
        self, summaries: Dict[str, ModelDiagnosticsSummary], leakage: LeakageResult,
    ) -> tuple:
        critical_issues: List[str] = []
        recommendations: List[str] = []

        has_critical_leakage = any(
            getattr(c, 'severity', None) == Severity.CRITICAL for c in getattr(leakage, 'checks', [])
        )
        if has_critical_leakage:
            critical_issues.append("Critical data leakage detected")

        has_critical_overfit = False
        has_critical_cv = False
        has_high = False

        for name, s in summaries.items():
            for check in getattr(s.overfitting, 'checks', []):
                if getattr(check, 'severity', None) == Severity.CRITICAL:
                    has_critical_overfit = True
                    critical_issues.append(f"{name}: critical overfitting detected")

            for check in getattr(s.cv_analysis, 'checks', []):
                if getattr(check, 'severity', None) == Severity.CRITICAL:
                    has_critical_cv = True
                    critical_issues.append(f"{name}: CV stability critical")

            if getattr(s.validity, 'high_issues', []):
                has_high = True

            for rec in getattr(s.cv_analysis, 'recommendations', []):
                if rec not in recommendations:
                    recommendations.append(rec)
            for rec in getattr(s.overfitting, 'recommendations', []):
                if rec not in recommendations:
                    recommendations.append(rec)

        if has_critical_leakage:
            return "leaky", critical_issues, recommendations
        if has_critical_overfit:
            return "overfit", critical_issues, recommendations
        if has_critical_cv:
            return "unstable", critical_issues, recommendations
        if has_high:
            return "caution", critical_issues, recommendations
        return "solid", critical_issues, recommendations

    @staticmethod
    def serialize(report: ModelDiagnosticsReport) -> Dict:
        def _convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.integer):
                return int(obj)
            if hasattr(obj, '__dataclass_fields__'):
                return asdict(obj)
            if hasattr(obj, '__dict__') and not callable(obj):
                return {k: _convert(v) for k, v in obj.__dict__.items() if not k.startswith('_')}
            return obj

        return {
            "verdict": report.verdict,
            "critical_issues": report.critical_issues,
            "recommendations": report.recommendations,
            "cross_model_agreement": asdict(report.cross_model_agreement),
            "summaries": {
                name: {
                    "model_name": s.model_name,
                    "cv_mean": getattr(s.cv_analysis, 'cv_mean', None),
                    "cv_std": getattr(s.cv_analysis, 'cv_std', None),
                    "validity_passed": getattr(s.validity, 'passed', None),
                    "calibration_brier": getattr(s.calibration, 'brier_score', None),
                    "feature_stability": asdict(s.feature_stability) if s.feature_stability else None,
                }
                for name, s in report.summaries.items()
            },
        }
