"""Cohort-level interpretability analysis."""

from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np
import shap

from customer_retention.core.compat import DataFrame, Series, pd


@dataclass
class CohortInsight:
    cohort_name: str
    cohort_size: int
    cohort_percentage: float
    churn_rate: float
    top_features: List[Dict[str, float]]
    key_differentiators: List[str] = field(default_factory=list)
    recommended_strategy: str = ""


@dataclass
class CohortComparison:
    cohort_a: str
    cohort_b: str
    feature_differences: Dict[str, float]
    churn_rate_difference: float
    key_differences: List[str] = field(default_factory=list)


@dataclass
class CohortAnalysisResult:
    cohort_insights: List[CohortInsight]
    key_differences: List[str]
    overall_summary: str = ""


class CohortAnalyzer:
    def __init__(self, model: Any, background_data: DataFrame, max_samples: int = 100):
        self.model = model
        self.background_data = background_data.head(max_samples)
        self._explainer = self._create_explainer()

    def _create_explainer(self) -> shap.Explainer:
        model_type = type(self.model).__name__
        if model_type in ["RandomForestClassifier", "GradientBoostingClassifier"]:
            return shap.TreeExplainer(self.model)
        return shap.KernelExplainer(self.model.predict_proba, self.background_data)

    def analyze(self, X: DataFrame, y: Series, cohorts: Series) -> CohortAnalysisResult:
        unique_cohorts = cohorts.unique()
        insights = []
        all_features_by_cohort = {}
        for cohort in unique_cohorts:
            mask = cohorts == cohort
            cohort_X = X[mask]
            cohort_y = y[mask]
            churn_rate = float(1 - cohort_y.mean())
            top_features = self._get_cohort_feature_importance(cohort_X)
            all_features_by_cohort[cohort] = top_features
            strategy = self._generate_strategy(cohort, churn_rate, top_features)
            insights.append(CohortInsight(
                cohort_name=cohort,
                cohort_size=len(cohort_X),
                cohort_percentage=len(cohort_X) / len(X),
                churn_rate=churn_rate,
                top_features=top_features,
                recommended_strategy=strategy
            ))
        key_differences = self._identify_key_differences(all_features_by_cohort, insights)
        for insight in insights:
            insight.key_differentiators = self._get_differentiators(insight.cohort_name, all_features_by_cohort)
        return CohortAnalysisResult(
            cohort_insights=insights,
            key_differences=key_differences
        )

    def _get_cohort_feature_importance(self, cohort_X: DataFrame) -> List[Dict[str, float]]:
        if len(cohort_X) == 0:
            return []
        sample = cohort_X.head(min(50, len(cohort_X)))
        shap_values = self._extract_shap_values(sample)
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        sorted_indices = np.argsort(mean_abs_shap)[::-1][:5]
        result = []
        for idx in sorted_indices:
            importance_val = mean_abs_shap[idx]
            if hasattr(importance_val, '__len__') and len(importance_val) == 1:
                importance_val = importance_val[0]
            result.append({"feature": cohort_X.columns[idx], "importance": float(importance_val)})
        return result

    def _extract_shap_values(self, X: DataFrame) -> np.ndarray:
        shap_values = self._explainer.shap_values(X)
        if hasattr(shap_values, 'values'):
            shap_values = shap_values.values
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        if len(shap_values.shape) == 3:
            shap_values = shap_values[:, :, 1]
        return shap_values

    def _generate_strategy(self, cohort: str, churn_rate: float,
                           top_features: List[Dict[str, float]]) -> str:
        if churn_rate > 0.5:
            priority = "urgent intervention"
        elif churn_rate > 0.3:
            priority = "proactive engagement"
        else:
            priority = "standard nurturing"
        top_feature = top_features[0]["feature"] if top_features else "engagement"
        return f"Focus on {top_feature} with {priority} for {cohort} cohort"

    def _identify_key_differences(self, features_by_cohort: Dict[str, List[Dict[str, float]]],
                                  insights: List[CohortInsight]) -> List[str]:
        differences = []
        churn_rates = {i.cohort_name: i.churn_rate for i in insights}
        if churn_rates:
            max_cohort = max(churn_rates, key=churn_rates.get)
            min_cohort = min(churn_rates, key=churn_rates.get)
            diff = churn_rates[max_cohort] - churn_rates[min_cohort]
            differences.append(f"{max_cohort} has {diff:.1%} higher churn than {min_cohort}")
        for cohort, features in features_by_cohort.items():
            if features:
                top = features[0]["feature"]
                differences.append(f"{cohort}: top driver is {top}")
        return differences

    def _get_differentiators(self, cohort: str,
                             features_by_cohort: Dict[str, List[Dict[str, float]]]) -> List[str]:
        cohort_features = features_by_cohort.get(cohort, [])
        cohort_top = set(f["feature"] for f in cohort_features[:3])
        other_tops = set()
        for other, features in features_by_cohort.items():
            if other != cohort:
                other_tops.update(f["feature"] for f in features[:3])
        unique = cohort_top - other_tops
        return [f"{cohort} uniquely driven by {f}" for f in unique]

    def compare_cohorts(self, X: DataFrame, y: Series, cohorts: Series,
                        cohort_a: str, cohort_b: str) -> CohortComparison:
        mask_a = cohorts == cohort_a
        mask_b = cohorts == cohort_b
        churn_a = 1 - y[mask_a].mean()
        churn_b = 1 - y[mask_b].mean()
        feature_diffs = {}
        for col in X.columns:
            mean_a = X.loc[mask_a, col].mean()
            mean_b = X.loc[mask_b, col].mean()
            feature_diffs[col] = float(mean_a - mean_b)
        key_diffs = []
        sorted_diffs = sorted(feature_diffs.items(), key=lambda x: abs(x[1]), reverse=True)
        for feature, diff in sorted_diffs[:3]:
            direction = "higher" if diff > 0 else "lower"
            key_diffs.append(f"{cohort_a} has {direction} {feature} than {cohort_b}")
        return CohortComparison(
            cohort_a=cohort_a,
            cohort_b=cohort_b,
            feature_differences=feature_diffs,
            churn_rate_difference=float(churn_a - churn_b),
            key_differences=key_diffs
        )

    @staticmethod
    def create_tenure_cohorts(tenure: Series,
                              bins: List[float] = None) -> Series:
        bins = bins or [0, 90, 365, float("inf")]
        labels = ["New", "Established", "Mature"]
        return pd.cut(tenure, bins=bins, labels=labels)

    @staticmethod
    def create_value_cohorts(value: Series,
                             quantiles: List[float] = None) -> Series:
        quantiles = quantiles or [0.33, 0.66]
        q1, q2 = value.quantile(quantiles[0]), value.quantile(quantiles[1])
        return pd.cut(value, bins=[-float("inf"), q1, q2, float("inf")],
                      labels=["Low", "Medium", "High"])

    @staticmethod
    def create_activity_cohorts(activity: Series,
                                thresholds: List[float] = None) -> Series:
        thresholds = thresholds or [5, 15]
        return pd.cut(activity, bins=[-float("inf"), thresholds[0], thresholds[1], float("inf")],
                      labels=["Dormant", "Moderate", "Active"])
