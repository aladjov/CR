from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
from datetime import datetime
import numpy as np
from sklearn.metrics import (
    precision_recall_curve, auc, roc_auc_score,
    precision_score, recall_score, brier_score_loss
)
from sklearn.calibration import calibration_curve

from customer_retention.core.compat import pd, DataFrame, Series


class PerformanceStatus(Enum):
    OK = "OK"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


@dataclass
class MonitoringConfig:
    pr_auc_warning_drop: float = 0.10
    pr_auc_critical_drop: float = 0.15
    roc_auc_warning_drop: float = 0.08
    roc_auc_critical_drop: float = 0.10
    precision_warning_drop: float = 0.20
    recall_warning_drop: float = 0.20
    brier_warning_increase: float = 0.05
    brier_critical_increase: float = 0.10


@dataclass
class PerformanceResult:
    current_metrics: Dict[str, float]
    baseline_metrics: Dict[str, float]
    comparison: Dict[str, float]
    status: PerformanceStatus
    labels_available: int
    proxy_metrics: Optional[Dict] = None
    monitoring_date: datetime = field(default_factory=datetime.now)


@dataclass
class CalibrationCurve:
    bin_means: List[float]
    actual_rates: List[float]
    counts: List[int]


@dataclass
class DistributionAnalysis:
    mean: float
    std: float
    min_val: float
    max_val: float
    percentiles: Dict[str, float]


@dataclass
class ProportionAnalysis:
    proportions: Dict[str, float]


@dataclass
class DistributionComparison:
    distribution_shift_detected: bool
    ks_statistic: float
    mean_diff: float


@dataclass
class TrendReport:
    pr_auc_trend: List[float]
    dates: List[datetime]
    trend_direction: str


class PerformanceMonitor:
    def __init__(self, baseline_metrics: Dict[str, float],
                 config: Optional[MonitoringConfig] = None):
        self.baseline_metrics = baseline_metrics
        self.config = config or MonitoringConfig()
        self._history: List[PerformanceResult] = []

    def evaluate(self, y_true: Series, y_prob: Series,
                 y_pred: Optional[Series] = None) -> PerformanceResult:
        y_true_clean = y_true.dropna()
        y_prob_clean = y_prob[y_true.notna()]
        current_metrics = {}
        precision, recall, _ = precision_recall_curve(y_true_clean, y_prob_clean)
        current_metrics["pr_auc"] = auc(recall, precision)
        current_metrics["roc_auc"] = roc_auc_score(y_true_clean, y_prob_clean)
        current_metrics["brier_score"] = brier_score_loss(y_true_clean, y_prob_clean)
        if y_pred is not None:
            y_pred_clean = y_pred[y_true.notna()]
            current_metrics["precision"] = precision_score(y_true_clean, y_pred_clean)
            current_metrics["recall"] = recall_score(y_true_clean, y_pred_clean)
        else:
            y_pred_binary = (y_prob_clean >= 0.5).astype(int)
            current_metrics["precision"] = precision_score(y_true_clean, y_pred_binary)
            current_metrics["recall"] = recall_score(y_true_clean, y_pred_binary)
        comparison = self._compare_to_baseline(current_metrics)
        status = self._determine_status(current_metrics)
        result = PerformanceResult(
            current_metrics=current_metrics,
            baseline_metrics=self.baseline_metrics,
            comparison=comparison,
            status=status,
            labels_available=len(y_true_clean)
        )
        self._history.append(result)
        return result

    def evaluate_without_labels(self, y_prob: Series) -> PerformanceResult:
        proxy = ProxyMetrics()
        dist_analysis = proxy.analyze_prediction_distribution(y_prob)
        proxy_metrics = {
            "mean_prediction": dist_analysis.mean,
            "std_prediction": dist_analysis.std,
            "percentile_25": dist_analysis.percentiles["25"],
            "percentile_50": dist_analysis.percentiles["50"],
            "percentile_75": dist_analysis.percentiles["75"]
        }
        return PerformanceResult(
            current_metrics={},
            baseline_metrics=self.baseline_metrics,
            comparison={},
            status=PerformanceStatus.OK,
            labels_available=0,
            proxy_metrics=proxy_metrics
        )

    def compare_metrics(self, metrics: Dict[str, float]) -> PerformanceResult:
        comparison = self._compare_to_baseline(metrics)
        status = self._determine_status(metrics)
        return PerformanceResult(
            current_metrics=metrics,
            baseline_metrics=self.baseline_metrics,
            comparison=comparison,
            status=status,
            labels_available=0
        )

    def _compare_to_baseline(self, current: Dict[str, float]) -> Dict[str, float]:
        comparison = {}
        for metric, value in current.items():
            if metric in self.baseline_metrics:
                baseline = self.baseline_metrics[metric]
                if baseline != 0:
                    change_pct = (value - baseline) / baseline * 100
                else:
                    change_pct = 0
                comparison[f"{metric}_change_pct"] = change_pct
                comparison[f"{metric}_diff"] = value - baseline
        return comparison

    def _determine_status(self, current: Dict[str, float]) -> PerformanceStatus:
        if "pr_auc" in current and "pr_auc" in self.baseline_metrics:
            drop = self.baseline_metrics["pr_auc"] - current["pr_auc"]
            if drop >= self.config.pr_auc_critical_drop:
                return PerformanceStatus.CRITICAL
            elif drop >= self.config.pr_auc_warning_drop:
                return PerformanceStatus.WARNING
        if "roc_auc" in current and "roc_auc" in self.baseline_metrics:
            drop = self.baseline_metrics["roc_auc"] - current["roc_auc"]
            if drop >= self.config.roc_auc_critical_drop:
                return PerformanceStatus.CRITICAL
            elif drop >= self.config.roc_auc_warning_drop:
                return PerformanceStatus.WARNING
        if "brier_score" in current and "brier_score" in self.baseline_metrics:
            increase = current["brier_score"] - self.baseline_metrics["brier_score"]
            if increase >= self.config.brier_critical_increase:
                return PerformanceStatus.CRITICAL
            elif increase >= self.config.brier_warning_increase:
                return PerformanceStatus.WARNING
        return PerformanceStatus.OK

    def get_history(self) -> List[PerformanceResult]:
        return self._history.copy()

    def get_trend_report(self) -> Dict:
        if len(self._history) < 2:
            return {"pr_auc_trend": [], "dates": [], "trend_direction": "insufficient_data"}
        pr_auc_values = [h.current_metrics.get("pr_auc", 0) for h in self._history]
        dates = [h.monitoring_date for h in self._history]
        if pr_auc_values[-1] > pr_auc_values[0]:
            direction = "improving"
        elif pr_auc_values[-1] < pr_auc_values[0]:
            direction = "declining"
        else:
            direction = "stable"
        return {
            "pr_auc_trend": pr_auc_values,
            "dates": dates,
            "trend_direction": direction
        }

    def get_calibration_curve(self, y_true: Series, y_prob: Series,
                              n_bins: int = 10) -> CalibrationCurve:
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
        bin_counts = []
        bins = np.linspace(0, 1, n_bins + 1)
        for i in range(n_bins):
            mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
            bin_counts.append(mask.sum())
        return CalibrationCurve(
            bin_means=prob_pred.tolist(),
            actual_rates=prob_true.tolist(),
            counts=bin_counts
        )


class ProxyMetrics:
    def analyze_prediction_distribution(self, y_prob: Series) -> DistributionAnalysis:
        return DistributionAnalysis(
            mean=y_prob.mean(),
            std=y_prob.std(),
            min_val=y_prob.min(),
            max_val=y_prob.max(),
            percentiles={
                "10": y_prob.quantile(0.10),
                "25": y_prob.quantile(0.25),
                "50": y_prob.quantile(0.50),
                "75": y_prob.quantile(0.75),
                "90": y_prob.quantile(0.90)
            }
        )

    def analyze_segment_proportions(self, segments: Series) -> ProportionAnalysis:
        proportions = segments.value_counts(normalize=True).to_dict()
        return ProportionAnalysis(proportions=proportions)

    def compare_distributions(self, reference: Series,
                              current: Series) -> DistributionComparison:
        from scipy import stats
        ks_stat, _ = stats.ks_2samp(reference, current)
        mean_diff = abs(current.mean() - reference.mean())
        shift_detected = ks_stat > 0.1 or mean_diff > reference.std() * 0.5
        return DistributionComparison(
            distribution_shift_detected=shift_detected,
            ks_statistic=ks_stat,
            mean_diff=mean_diff
        )
