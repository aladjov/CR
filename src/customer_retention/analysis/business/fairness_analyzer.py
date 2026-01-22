"""Fairness analysis for model predictions."""

from dataclasses import dataclass
from typing import Dict, List, Optional

from customer_retention.core.compat import Series, pd


@dataclass
class GroupMetrics:
    group_name: str
    size: int
    positive_rate: float
    true_positive_rate: Optional[float] = None
    false_positive_rate: Optional[float] = None
    accuracy: Optional[float] = None


@dataclass
class FairnessMetric:
    name: str
    values: Dict[str, float]
    ratio: float
    passed: bool
    threshold: float


@dataclass
class FairnessResult:
    passed: bool
    metrics: List[FairnessMetric]
    group_metrics: Dict[str, GroupMetrics]
    recommendations: List[str]


class FairnessAnalyzer:
    def __init__(self, threshold: float = 0.8):
        self.threshold = threshold

    def analyze(self, y_true: Series, y_pred: Series,
                protected: Series) -> FairnessResult:
        groups = protected.unique()
        group_metrics = {}
        metrics = []
        for group in groups:
            mask = protected == group
            y_t = y_true[mask]
            y_p = y_pred[mask]
            positive_rate = y_p.mean()
            accuracy = (y_t == y_p).mean()
            tp = ((y_t == 1) & (y_p == 1)).sum()
            fn = ((y_t == 1) & (y_p == 0)).sum()
            fp = ((y_t == 0) & (y_p == 1)).sum()
            tn = ((y_t == 0) & (y_p == 0)).sum()
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            group_metrics[group] = GroupMetrics(
                group_name=group,
                size=int(mask.sum()),
                positive_rate=float(positive_rate),
                true_positive_rate=float(tpr),
                false_positive_rate=float(fpr),
                accuracy=float(accuracy)
            )
        positive_rates = {g: m.positive_rate for g, m in group_metrics.items()}
        if positive_rates:
            min_rate = min(positive_rates.values())
            max_rate = max(positive_rates.values())
            dp_ratio = min_rate / max_rate if max_rate > 0 else 1.0
            metrics.append(FairnessMetric(
                name="demographic_parity",
                values=positive_rates,
                ratio=dp_ratio,
                passed=dp_ratio >= self.threshold,
                threshold=self.threshold
            ))
            metrics.append(FairnessMetric(
                name="disparate_impact",
                values=positive_rates,
                ratio=dp_ratio,
                passed=dp_ratio >= self.threshold,
                threshold=self.threshold
            ))
        tprs = {g: m.true_positive_rate for g, m in group_metrics.items()}
        fprs = {g: m.false_positive_rate for g, m in group_metrics.items()}
        if tprs:
            min_tpr = min(tprs.values())
            max_tpr = max(tprs.values())
            tpr_ratio = min_tpr / max_tpr if max_tpr > 0 else 1.0
            min_fpr = min(fprs.values())
            max_fpr = max(fprs.values())
            fpr_ratio = min_fpr / max_fpr if max_fpr > 0 else 1.0
            eo_ratio = min(tpr_ratio, fpr_ratio)
            metrics.append(FairnessMetric(
                name="equalized_odds",
                values={"tpr_ratio": tpr_ratio, "fpr_ratio": fpr_ratio},
                ratio=eo_ratio,
                passed=eo_ratio >= self.threshold,
                threshold=self.threshold
            ))
        overall_passed = all(m.passed for m in metrics)
        recommendations = self._generate_recommendations(metrics, group_metrics)
        return FairnessResult(
            passed=overall_passed,
            metrics=metrics,
            group_metrics=group_metrics,
            recommendations=recommendations
        )

    def _generate_recommendations(self, metrics: List[FairnessMetric],
                                  group_metrics: Dict[str, GroupMetrics]) -> List[str]:
        recommendations = []
        for metric in metrics:
            if not metric.passed:
                recommendations.append(
                    f"Metric '{metric.name}' failed with ratio {metric.ratio:.2f} "
                    f"(threshold: {metric.threshold}). Consider rebalancing training data."
                )
        accuracies = {g: m.accuracy for g, m in group_metrics.items()}
        if accuracies:
            max_acc = max(accuracies.values())
            min_acc = min(accuracies.values())
            if max_acc - min_acc > 0.1:
                worst_group = min(accuracies, key=accuracies.get)
                recommendations.append(
                    f"Accuracy differs significantly across groups. "
                    f"Consider additional features for {worst_group}."
                )
        if not recommendations:
            recommendations.append("No significant bias detected. Model passes fairness checks.")
        return recommendations

    def analyze_calibration(self, y_true: Series, y_proba: Series,
                            protected: Series) -> FairnessResult:
        groups = protected.unique()
        group_metrics = {}
        for group in groups:
            mask = protected == group
            y_t = y_true[mask]
            y_p = y_proba[mask]
            bins = pd.cut(y_p, bins=10, labels=False)
            calibration_error = 0
            for b in range(10):
                bin_mask = bins == b
                if bin_mask.sum() > 0:
                    predicted_prob = y_p[bin_mask].mean()
                    actual_prob = y_t[bin_mask].mean()
                    calibration_error += abs(predicted_prob - actual_prob) * bin_mask.sum()
            calibration_error /= len(y_t) if len(y_t) > 0 else 1
            group_metrics[group] = GroupMetrics(
                group_name=group,
                size=int(mask.sum()),
                positive_rate=float(y_t.mean()),
                accuracy=1 - calibration_error
            )
        return FairnessResult(
            passed=True,
            metrics=[],
            group_metrics=group_metrics,
            recommendations=[]
        )

    def analyze_multiple(self, y_true: Series, y_pred: Series,
                        protected_attributes: Dict[str, Series]) -> Dict[str, FairnessResult]:
        return {name: self.analyze(y_true, y_pred, protected)
                for name, protected in protected_attributes.items()}
