"""Segment performance analysis probes."""

from dataclasses import dataclass, field
from typing import List, Dict, Optional

import numpy as np

from customer_retention.core.compat import pd, DataFrame, Series
from customer_retention.core.components.enums import Severity
from sklearn.metrics import precision_score, recall_score, roc_auc_score, average_precision_score


@dataclass
class SegmentCheck:
    check_id: str
    segment: str
    severity: Severity
    recommendation: str
    metric: str = ""
    value: float = 0.0


@dataclass
class SegmentResult:
    passed: bool
    checks: List[SegmentCheck] = field(default_factory=list)
    segment_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    recommendation: str = ""


class SegmentPerformanceAnalyzer:
    UNDERPERFORMANCE_THRESHOLD = 0.20
    LOW_RECALL_THRESHOLD = 0.20
    SMALL_SEGMENT_THRESHOLD = 0.05

    def define_segments(self, X: DataFrame, segment_column: str, segment_type: str = "quantile") -> Series:
        if segment_column not in X.columns:
            return pd.Series(["all"] * len(X))
        values = X[segment_column]
        if segment_type == "tenure":
            return pd.cut(values, bins=[0, 90, 365, np.inf], labels=["new", "established", "mature"])
        if segment_type == "quantile":
            return pd.qcut(values, q=3, labels=["low", "medium", "high"], duplicates="drop")
        return Series(["all"] * len(X))

    def analyze_performance(self, model, X: DataFrame, y: Series, segments: Series) -> SegmentResult:
        checks = []
        segment_metrics = {}
        global_metrics = self._compute_metrics(model, X, y)
        unique_segments = segments.unique()
        for seg in unique_segments:
            mask = segments == seg
            if mask.sum() < 10:
                continue
            X_seg = X[mask]
            y_seg = y[mask]
            seg_size_pct = mask.sum() / len(y)
            metrics = self._compute_metrics(model, X_seg, y_seg)
            segment_metrics[str(seg)] = metrics
            if seg_size_pct < self.SMALL_SEGMENT_THRESHOLD:
                checks.append(SegmentCheck(
                    check_id="SG003",
                    segment=str(seg),
                    severity=Severity.MEDIUM,
                    recommendation=f"MEDIUM: Segment '{seg}' is small ({seg_size_pct:.1%}). Results may be unreliable.",
                    metric="size",
                    value=seg_size_pct,
                ))
            if "pr_auc" in metrics and "pr_auc" in global_metrics:
                gap = global_metrics["pr_auc"] - metrics["pr_auc"]
                if gap > self.UNDERPERFORMANCE_THRESHOLD:
                    checks.append(SegmentCheck(
                        check_id="SG001",
                        segment=str(seg),
                        severity=Severity.HIGH,
                        recommendation=f"HIGH: Segment '{seg}' underperforms by {gap:.1%}. Consider segment-specific model.",
                        metric="pr_auc",
                        value=metrics["pr_auc"],
                    ))
            if "recall" in metrics and metrics["recall"] < self.LOW_RECALL_THRESHOLD:
                checks.append(SegmentCheck(
                    check_id="SG002",
                    segment=str(seg),
                    severity=Severity.HIGH,
                    recommendation=f"HIGH: Segment '{seg}' has low recall ({metrics['recall']:.1%}). Adjust threshold or add features.",
                    metric="recall",
                    value=metrics["recall"],
                ))
        critical = [c for c in checks if c.severity == Severity.CRITICAL]
        recommendations = [c.recommendation for c in checks if c.severity in [Severity.CRITICAL, Severity.HIGH]]
        recommendation = self._global_recommendation(checks, unique_segments)
        return SegmentResult(
            passed=len(critical) == 0,
            checks=checks,
            segment_metrics=segment_metrics,
            recommendations=recommendations,
            recommendation=recommendation,
        )

    def _compute_metrics(self, model, X: DataFrame, y: Series) -> Dict[str, float]:
        try:
            y_pred = model.predict(X)
            y_proba = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else y_pred
            return {
                "precision": precision_score(y, y_pred, zero_division=0),
                "recall": recall_score(y, y_pred, zero_division=0),
                "roc_auc": roc_auc_score(y, y_proba) if len(np.unique(y)) > 1 else 0.5,
                "pr_auc": average_precision_score(y, y_proba) if len(np.unique(y)) > 1 else 0.5,
                "churn_rate": y.mean(),
                "sample_size": len(y),
            }
        except Exception:
            return {}

    def _global_recommendation(self, checks: List[SegmentCheck], segments) -> str:
        high_issues = [c for c in checks if c.severity == Severity.HIGH]
        if not high_issues:
            return "No significant segment gaps. Continue with global model."
        if len(high_issues) == 1:
            return f"One segment underperforms. Consider adding segment as feature."
        return "Multiple segments underperform. Consider segment-specific models."
