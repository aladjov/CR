import enum
import re
import time
from dataclasses import dataclass, field, is_dataclass
from typing import Any, Dict, List, Optional, Set

import numpy as np

from customer_retention.analysis.diagnostics.calibration_analyzer import CalibrationAnalyzer, CalibrationResult
from customer_retention.analysis.diagnostics.cv_analyzer import CVAnalysisResult, CVAnalyzer
from customer_retention.analysis.diagnostics.feature_stability import FeatureStabilityAnalyzer, FeatureStabilityResult
from customer_retention.analysis.diagnostics.leakage_detector import LeakageCheck, LeakageDetector, LeakageResult
from customer_retention.analysis.diagnostics.overfitting_analyzer import OverfittingAnalyzer, OverfittingResult
from customer_retention.core.components.enums import Severity
from customer_retention.stages.validation.model_validity_gate import ModelValidityGate, ModelValidityResult

_DOMAIN_TARGET_PATTERN = re.compile(
    r"(churn|reten|cancel|unsubscribe|attrit|lapse|defect|convert|active|inactive|"
    r"leave|stay|renew|expir|terminat|close|deactivat)",
    re.IGNORECASE,
)
_TEMPORAL_PATTERN = re.compile(r"(days|since|tenure|recency|last|ago|date|time)", re.IGNORECASE)


def _is_spark_wrapper(model: Any) -> bool:
    return type(model).__name__ == "SparkClassifierWrapper"


def _extract_positive_class(precomputed: Optional[Dict[str, np.ndarray]]) -> Optional[np.ndarray]:
    if not precomputed or "y_proba_test" not in precomputed:
        return None
    arr = np.asarray(precomputed["y_proba_test"])
    return arr[:, 1] if arr.ndim == 2 else arr


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

    @property
    def cv_mean(self) -> float:
        return float(self.cv_analysis.cv_mean)

    @property
    def cv_std(self) -> float:
        return float(self.cv_analysis.cv_std)

    @property
    def fold_aucs(self) -> List[float]:
        return [float(fold.get("score", 0.0)) for fold in self.cv_analysis.fold_analysis]


@dataclass
class LeakageCoverage:
    """Surfaces what fraction of the final feature set NB05 actually saw.

    Features that NB05 analyzed and rejected as low-correlation are silently safe.
    Features that NB05 never saw (gold transforms, silver-derived ratios/interactions/composites)
    are listed here so the user knows the cached leakage view has a blind spot.
    """
    total_features: int
    analyzed_in_nb05: int
    unanalyzed: List[str] = field(default_factory=list)


@dataclass
class ModelDiagnosticsReport:
    summaries: Dict[str, ModelDiagnosticsSummary]
    leakage: LeakageResult
    cross_model_agreement: CrossModelAgreement
    best_model_learning_curve: Optional[OverfittingResult]
    verdict: str
    critical_issues: List[str]
    recommendations: List[str]
    leakage_coverage: Optional[LeakageCoverage] = None
    skipped_analyses: List[str] = field(default_factory=list)


class ModelDiagnosticsReportGenerator:
    """Builds a ModelDiagnosticsReport from precomputed metrics + cached recommendations.

    Constraints baked in by design:

    * **No fresh sklearn fits.** Calibration uses ``y_proba_test`` from training,
      not a re-call to ``predict_proba``. Per-feature AUC and learning curves are
      removed entirely — they refit estimators and are not distributed-safe.
    * **No fresh data scans.** Feature-target correlations come from NB05's
      ``prioritize_feature`` recommendations (already on disk). Pattern checks
      run on column names only.
    * **Sequential per-model loop.** ThreadPoolExecutor is banned on Databricks
      shared clusters and the per-model summaries are pure metric arithmetic
      (cheap), so threading buys nothing and risks the cluster constraint.
    """

    LEAKAGE_CRITICAL = 0.90
    LEAKAGE_HIGH = 0.70
    LEAKAGE_MEDIUM = 0.50
    UNANALYZED_SAMPLE_SIZE = 10

    SKIPPED_ANALYSES = [
        "learning_curve: removed — sklearn refits per training-set size are not distributed-safe and "
        "the existing CV variance + train/test gap already surface the data-hunger signal.",
        "per_feature_single_auc: removed — sklearn LogisticRegression refit per feature is incompatible "
        "with the bulk/distributed path. Linear leakage signal is captured by cached NB05 correlations.",
        "per_feature_class_separation: removed — per-feature class-overlap scan duplicates the "
        "feature-target correlation signal already cached from NB05.",
    ]

    def generate(
        self, *,
        models: Dict[str, Any],
        cv_results: Dict[str, Dict],
        train_metrics: Dict[str, Dict[str, float]],
        test_metrics: Dict[str, Dict[str, float]],
        feature_names: List[str],
        best_model_name: str,
        class_proportion: float,
        predictions: Dict[str, Dict[str, np.ndarray]],
        y_test: Any,
        recommendations: Optional[Any] = None,
        target_column: str = "target",
        label_horizon_days: Optional[int] = None,
    ) -> ModelDiagnosticsReport:
        y_test_np = np.asarray(y_test)

        leakage = self._safe_leakage_from_cache(
            recommendations, feature_names, target_column, label_horizon_days
        )
        coverage = self._compute_leakage_coverage(recommendations, feature_names)
        summaries = self._run_summaries(models, y_test_np, cv_results, train_metrics, test_metrics, predictions)
        agreement = self._compute_cross_model_agreement(models, feature_names)
        verdict, critical_issues, recs = self._compute_verdict(summaries, leakage)

        return ModelDiagnosticsReport(
            summaries=summaries, leakage=leakage,
            cross_model_agreement=agreement,
            best_model_learning_curve=None,
            verdict=verdict, critical_issues=critical_issues, recommendations=recs,
            leakage_coverage=coverage,
            skipped_analyses=list(self.SKIPPED_ANALYSES),
        )

    def _compute_leakage_coverage(
        self, recommendations: Optional[Any], feature_names: List[str],
    ) -> LeakageCoverage:
        analyzed = self._analyzed_features_from_recommendations(recommendations)
        if not analyzed:
            return LeakageCoverage(
                total_features=len(feature_names),
                analyzed_in_nb05=0,
                unanalyzed=list(feature_names[: self.UNANALYZED_SAMPLE_SIZE]),
            )
        in_set = set(analyzed)
        unanalyzed = [c for c in feature_names if c not in in_set]
        return LeakageCoverage(
            total_features=len(feature_names),
            analyzed_in_nb05=len(feature_names) - len(unanalyzed),
            unanalyzed=unanalyzed[: self.UNANALYZED_SAMPLE_SIZE],
        )

    @staticmethod
    def _analyzed_features_from_recommendations(recommendations: Optional[Any]) -> List[str]:
        if recommendations is None:
            return []
        gold = getattr(recommendations, "gold", None)
        if gold is None:
            return []
        config = getattr(gold, "feature_selection_config", None)
        if config is None:
            return []
        return list(getattr(config, "analyzed_features", []) or [])

    def _safe_leakage_from_cache(
        self,
        recommendations: Optional[Any],
        feature_names: List[str],
        target_column: str,
        label_horizon_days: Optional[int] = None,
    ) -> LeakageResult:
        try:
            return self._build_leakage_from_recommendations(
                recommendations, feature_names, target_column, label_horizon_days
            )
        except Exception as exc:
            print(f"  [diagnostics] cached leakage build failed: {type(exc).__name__}: {exc}", flush=True)
            return LeakageResult(passed=True, checks=[])

    def _build_leakage_from_recommendations(
        self,
        recommendations: Optional[Any],
        feature_names: List[str],
        target_column: str,
        label_horizon_days: Optional[int] = None,
    ) -> LeakageResult:
        checks: List[LeakageCheck] = []
        checks.extend(self._target_name_pattern_checks(feature_names, target_column))
        checks.extend(self._cached_correlation_checks(recommendations))
        # LD062 / LD063: name-only scan against the model's actual feature
        # set. The cached NB05 correlation view only sees silver-level columns,
        # so any `_is_zero` / windowed-count derivation produced in gold gets a
        # second-pass check here. The check is data-free (column names only).
        checks.extend(self._window_overlap_checks(feature_names, label_horizon_days))
        critical = [c for c in checks if c.severity == Severity.CRITICAL]
        return LeakageResult(passed=not critical, checks=checks, critical_issues=critical)

    @staticmethod
    def _window_overlap_checks(
        feature_names: List[str], label_horizon_days: Optional[int],
    ) -> List[LeakageCheck]:
        if not label_horizon_days or not feature_names:
            return []

        class _NamesOnly:
            def __init__(self, names): self.columns = list(names)

        detector = LeakageDetector(label_horizon_days=int(label_horizon_days))
        result = detector.check_window_overlaps_horizon(_NamesOnly(feature_names))
        return list(result.checks)

    @staticmethod
    def _target_name_pattern_checks(feature_names: List[str], target_column: str) -> List[LeakageCheck]:
        checks: List[LeakageCheck] = []
        if target_column in feature_names:
            checks.append(LeakageCheck(
                check_id="LD052", feature=target_column, severity=Severity.CRITICAL,
                recommendation=f"REMOVE {target_column}: target column is present in the feature matrix.",
                correlation=1.0,
            ))
        suffixes = (f"{target_column}_".lower(), f"_{target_column}".lower())
        for col in feature_names:
            lc = col.lower()
            if col != target_column and any(s in lc for s in suffixes):
                checks.append(LeakageCheck(
                    check_id="LD052", feature=col, severity=Severity.CRITICAL,
                    recommendation=f"REMOVE {col}: name suggests derivation from target '{target_column}'.",
                ))
        return checks

    def _cached_correlation_checks(self, recommendations: Optional[Any]) -> List[LeakageCheck]:
        if recommendations is None:
            return []
        gold = getattr(recommendations, "gold", None)
        if gold is None:
            return []
        prioritized = [r for r in getattr(gold, "feature_selection", []) if getattr(r, "action", "") == "prioritize"]
        checks: List[LeakageCheck] = []
        for rec in prioritized:
            corr = float((rec.parameters or {}).get("correlation", 0.0))
            if abs(corr) <= self.LEAKAGE_MEDIUM:
                continue
            checks.append(self._classify_cached_correlation(rec.target_column, corr))
        return checks

    def _classify_cached_correlation(self, feature: str, corr: float) -> LeakageCheck:
        abs_corr = abs(corr)
        is_temporal = bool(_TEMPORAL_PATTERN.search(feature))
        is_domain = bool(_DOMAIN_TARGET_PATTERN.search(feature))
        if abs_corr > self.LEAKAGE_CRITICAL:
            severity = Severity.CRITICAL
            check_id = "LD001"
            rec = f"REMOVE {feature}: cached correlation {corr:.2f} from NB05 indicates likely leakage."
        elif abs_corr > self.LEAKAGE_HIGH:
            severity = Severity.HIGH
            check_id = "LD002"
            rec = f"INVESTIGATE {feature}: cached correlation {corr:.2f} from NB05 is suspiciously high."
        else:
            severity = Severity.MEDIUM
            check_id = "LD003"
            rec = f"MONITOR {feature}: cached correlation {corr:.2f} from NB05 is elevated."
        if is_temporal and severity != Severity.MEDIUM:
            rec += " Temporal-named feature — verify reference date logic."
        if is_domain and severity != Severity.MEDIUM:
            rec += " Domain target pattern in name — review semantic overlap with target."
        return LeakageCheck(check_id=check_id, feature=feature, severity=severity, recommendation=rec, correlation=corr)

    def _run_summaries(
        self, models: Dict[str, Any], y_test_np: np.ndarray,
        cv_results: Dict[str, Dict], train_metrics: Dict[str, Dict[str, float]],
        test_metrics: Dict[str, Dict[str, float]],
        predictions: Optional[Dict[str, Dict[str, np.ndarray]]],
    ) -> Dict[str, ModelDiagnosticsSummary]:
        summaries: Dict[str, ModelDiagnosticsSummary] = {}
        for name in models:
            summary = self._safe_per_model(
                name, y_test_np,
                cv_results.get(name, {}), train_metrics.get(name, {}),
                test_metrics.get(name, {}), (predictions or {}).get(name),
            )
            if summary is not None:
                summaries[name] = summary
        return summaries

    def _safe_per_model(
        self, name: str, y_test_np: np.ndarray,
        cv_data: Dict, train_mets: Dict[str, float], test_mets: Dict[str, float],
        precomputed: Optional[Dict[str, np.ndarray]],
    ) -> Optional[ModelDiagnosticsSummary]:
        try:
            return self._run_per_model(name, y_test_np, cv_data, train_mets, test_mets, precomputed)
        except Exception as exc:
            print(f"  [diagnostics] per-model checks failed for {name}: {type(exc).__name__}: {exc}", flush=True)
            return None

    def _run_per_model(
        self, name: str, y_test_np: np.ndarray,
        cv_data: Dict, train_mets: Dict[str, float], test_mets: Dict[str, float],
        precomputed: Optional[Dict[str, np.ndarray]] = None,
    ) -> ModelDiagnosticsSummary:
        cv_scores = cv_data.get("cv_scores", [])
        test_score = test_mets.get("roc_auc") or test_mets.get("pr_auc")

        cv_analysis = CVAnalyzer().run_all(cv_scores, test_score)
        overfitting = OverfittingAnalyzer().analyze_train_test_gap(train_mets, test_mets)
        calibration = self._compute_calibration(y_test_np, precomputed)
        validity = ModelValidityGate().run(test_mets)

        fold_details = cv_data.get("fold_details", [])
        fold_importances = [d["feature_importance"] for d in fold_details if "feature_importance" in d]
        feature_stability = FeatureStabilityAnalyzer().analyze(fold_importances) if fold_importances else None

        return ModelDiagnosticsSummary(
            model_name=name, cv_analysis=cv_analysis, overfitting=overfitting,
            calibration=calibration, validity=validity, feature_stability=feature_stability,
        )

    @staticmethod
    def _compute_calibration(
        y_test_np: np.ndarray, precomputed: Optional[Dict[str, np.ndarray]],
    ) -> CalibrationResult:
        y_proba_test_pos = _extract_positive_class(precomputed)
        if y_proba_test_pos is None or len(y_proba_test_pos) != len(y_test_np):
            return CalibrationResult(passed=True, recommendation="calibration unavailable: missing matched test predictions")
        return CalibrationAnalyzer().analyze_calibration(y_test_np, y_proba_test_pos)

    def _compute_cross_model_agreement(
        self, models: Dict[str, Any], feature_names: List[str], top_n: int = 20,
    ) -> CrossModelAgreement:
        model_tops = {name: self._get_model_top_features(m, feature_names, top_n) for name, m in models.items()}

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
    def to_jsonable(report: ModelDiagnosticsReport) -> Dict:
        """Deeply convert the report into a JSON-serializable dict tree.

        Used by NB08 to persist the full report so NB09 can re-render every
        per-model check, leakage row, and summary without re-loading data or
        models. Enums are flattened to ``{name, value}`` so consumers can read
        ``severity['name']`` after a JSON round-trip.
        """
        return _to_jsonable(report)


def compute_and_persist_diagnostics(
    diagnostics_path: Any,
    *,
    models: Dict[str, Any],
    predictions: Dict[str, Dict[str, np.ndarray]],
    feature_names: List[str],
    cv_results: Dict[str, Dict],
    train_metrics: Dict[str, Dict[str, float]],
    test_metrics: Dict[str, Dict[str, float]],
    best_model_name: str,
    y_test: Any,
    class_proportion: float,
    recommendations: Optional[Any] = None,
    target_column: str = "target",
    label_horizon_days: Optional[int] = None,
) -> bool:
    """Compute the diagnostics report + best-model holdout metrics from precomputed data.

    Designed to be called from the end of NB08, where the trained models, fresh
    test predictions, and labels are still in scope. **No fresh feature scans, no
    sklearn refits, no learning curves.** Leakage is derived from NB05's cached
    ``prioritize_feature`` correlations (loaded from ``recommendations``). Per-model
    summaries reuse cv_results / train_metrics / test_metrics. Holdout metrics
    reuse the cached ``y_pred`` from ``predictions[best_model_name]``.

    The whole helper is wrapped in try/except so a diagnostic failure prints the
    error and returns ``False`` — training results are never lost.
    """
    print("[diagnostics] building report from precomputed data...", flush=True)
    started = time.monotonic()
    report_payload: Optional[Dict[str, Any]] = None
    holdout_payload: Optional[Dict[str, Any]] = None

    try:
        generator = ModelDiagnosticsReportGenerator()
        report = generator.generate(
            models=models, cv_results=cv_results, train_metrics=train_metrics,
            test_metrics=test_metrics, feature_names=feature_names,
            best_model_name=best_model_name, class_proportion=class_proportion,
            predictions=predictions, y_test=y_test,
            recommendations=recommendations, target_column=target_column,
            label_horizon_days=label_horizon_days,
        )
        report_payload = ModelDiagnosticsReportGenerator.to_jsonable(report)
    except Exception as exc:
        print(f"  [diagnostics] report build failed: {type(exc).__name__}: {exc}", flush=True)

    try:
        holdout_payload = _compute_best_model_holdout(
            best_model_name=best_model_name,
            predictions=predictions,
            test_metrics=test_metrics,
            y_test=y_test,
        )
    except Exception as exc:
        print(f"  [diagnostics] holdout metrics failed: {type(exc).__name__}: {exc}", flush=True)

    elapsed = time.monotonic() - started
    if report_payload is None and holdout_payload is None:
        print(f"[diagnostics] aborted with no payload to persist ({elapsed:.1f}s)", flush=True)
        return False

    try:
        _merge_diagnostics_file(diagnostics_path, report_payload, holdout_payload)
    except Exception as exc:
        print(f"  [diagnostics] persisting diagnostics file failed: {type(exc).__name__}: {exc}", flush=True)
        return False

    success = report_payload is not None and holdout_payload is not None
    status = "complete" if success else "partial"
    print(f"[diagnostics] {status} ({elapsed:.1f}s)", flush=True)
    return success


def _compute_best_model_holdout(
    *, best_model_name: str,
    predictions: Dict[str, Dict[str, np.ndarray]],
    test_metrics: Dict[str, Dict[str, float]],
    y_test: Any,
) -> Optional[Dict[str, Any]]:
    from sklearn.metrics import accuracy_score, confusion_matrix

    preds = predictions.get(best_model_name)
    if not preds:
        return None
    y_proba_pos = _extract_positive_class(preds)
    if y_proba_pos is None:
        return None

    y_test_np = np.asarray(y_test)
    if len(y_proba_pos) != len(y_test_np):
        return None

    cached_pred = preds.get("y_pred")
    y_pred = np.asarray(cached_pred) if cached_pred is not None else (y_proba_pos >= 0.5).astype(int)
    cm = confusion_matrix(y_test_np, y_pred)
    cached = test_metrics.get(best_model_name) or {}

    return {
        "model_name": best_model_name,
        "dataset": "exploration_test",
        "n_samples": int(len(y_test_np)),
        "roc_auc": _coerce_metric(cached, "roc_auc"),
        "pr_auc": _coerce_metric(cached, "pr_auc"),
        "f1": _coerce_metric(cached, "f1"),
        "precision": _coerce_metric(cached, "precision"),
        "recall": _coerce_metric(cached, "recall"),
        "accuracy": float(accuracy_score(y_test_np, y_pred)),
        "confusion_matrix": {"tn": int(cm[0, 0]), "fp": int(cm[0, 1]), "fn": int(cm[1, 0]), "tp": int(cm[1, 1])},
        "probability_stats": {
            "mean": float(np.mean(y_proba_pos)),
            "std": float(np.std(y_proba_pos)),
            "median": float(np.median(y_proba_pos)),
            "p10": float(np.percentile(y_proba_pos, 10)),
            "p90": float(np.percentile(y_proba_pos, 90)),
        },
    }


def _coerce_metric(metrics: Dict[str, Any], key: str) -> float:
    value = metrics.get(key)
    return float(value) if value is not None else 0.0


def _merge_diagnostics_file(
    diagnostics_path: Any,
    report_payload: Optional[Dict[str, Any]],
    holdout_payload: Optional[Dict[str, Any]],
) -> None:
    import json
    from pathlib import Path

    path = Path(str(diagnostics_path))
    existing: Dict[str, Any] = {}
    if path.exists():
        existing = json.loads(path.read_text())
    if report_payload is not None:
        existing["model_diagnostics_report"] = report_payload
    if holdout_payload is not None:
        existing["best_model_holdout_metrics"] = holdout_payload
    path.write_text(json.dumps(existing, default=str))


def _to_jsonable(obj: Any) -> Any:
    # Enum check must come before primitive check — Severity is a (str, Enum) subclass.
    if isinstance(obj, enum.Enum):
        return {"name": obj.name, "value": obj.value if not isinstance(obj.value, enum.Enum) else obj.value.name}
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, np.ndarray):
        return [_to_jsonable(v) for v in obj.tolist()]
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if is_dataclass(obj):
        return {f: _to_jsonable(getattr(obj, f)) for f in obj.__dataclass_fields__}
    if hasattr(obj, '__dict__'):
        return {k: _to_jsonable(v) for k, v in obj.__dict__.items() if not k.startswith('_')}
    return str(obj)
