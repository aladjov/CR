"""A/B test design for retention interventions."""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional

import numpy as np
from scipy import stats

from customer_retention.core.compat import DataFrame, concat


@dataclass
class SampleSizeResult:
    sample_size_per_group: int
    total_sample_size: int
    baseline_rate: float
    min_detectable_effect: float
    alpha: float
    power: float


@dataclass
class MeasurementPlan:
    primary_metric: str
    secondary_metrics: List[str]
    tracking_events: List[str] = field(default_factory=list)


@dataclass
class ABTestDesign:
    test_name: str
    control_name: str
    treatment_groups: List[str]
    recommended_sample_size: int
    total_required: int
    available_customers: int
    feasible: bool
    duration_days: int
    expected_completion_date: datetime
    stratification_variable: Optional[str] = None
    measurement_plan: Optional[MeasurementPlan] = None


class ABTestDesigner:
    def calculate_sample_size(self, baseline_rate: float = 0.25,
                              min_detectable_effect: float = 0.05,
                              alpha: float = 0.05, power: float = 0.80) -> SampleSizeResult:
        z_alpha = stats.norm.ppf(1 - alpha / 2)
        z_beta = stats.norm.ppf(power)
        p1 = baseline_rate
        p2 = baseline_rate - min_detectable_effect
        p_pooled = (p1 + p2) / 2
        numerator = (z_alpha * np.sqrt(2 * p_pooled * (1 - p_pooled)) +
                     z_beta * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
        denominator = (p1 - p2) ** 2
        n = int(np.ceil(numerator / denominator))
        return SampleSizeResult(
            sample_size_per_group=n,
            total_sample_size=n * 2,
            baseline_rate=baseline_rate,
            min_detectable_effect=min_detectable_effect,
            alpha=alpha,
            power=power
        )

    def calculate_power(self, sample_size_per_group: int, baseline_rate: float = 0.25,
                        effect_size: float = 0.05, alpha: float = 0.05) -> float:
        z_alpha = stats.norm.ppf(1 - alpha / 2)
        p1 = baseline_rate
        p2 = baseline_rate - effect_size
        (p1 + p2) / 2
        se = np.sqrt(p1 * (1 - p1) / sample_size_per_group + p2 * (1 - p2) / sample_size_per_group)
        z = abs(p1 - p2) / se
        power = stats.norm.cdf(z - z_alpha) + stats.norm.cdf(-z - z_alpha)
        return float(np.clip(power, 0, 1))

    def design_test(self, test_name: str, customer_pool: DataFrame,
                    control_name: str, treatment_names: List[str],
                    baseline_rate: float = 0.25, min_detectable_effect: float = 0.05,
                    alpha: float = 0.05, power: float = 0.80,
                    stratify_by: Optional[str] = None, duration_days: int = 30,
                    primary_metric: str = "churn_rate",
                    secondary_metrics: Optional[List[str]] = None) -> ABTestDesign:
        sample_result = self.calculate_sample_size(
            baseline_rate=baseline_rate,
            min_detectable_effect=min_detectable_effect,
            alpha=alpha,
            power=power
        )
        n_groups = 1 + len(treatment_names)
        total_required = sample_result.sample_size_per_group * n_groups
        available = len(customer_pool)
        feasible = available >= total_required
        measurement_plan = MeasurementPlan(
            primary_metric=primary_metric,
            secondary_metrics=secondary_metrics or [],
            tracking_events=["assignment", "intervention_delivered", "outcome_measured"]
        )
        return ABTestDesign(
            test_name=test_name,
            control_name=control_name,
            treatment_groups=treatment_names,
            recommended_sample_size=sample_result.sample_size_per_group,
            total_required=total_required,
            available_customers=available,
            feasible=feasible,
            duration_days=duration_days,
            expected_completion_date=datetime.now() + timedelta(days=duration_days),
            stratification_variable=stratify_by,
            measurement_plan=measurement_plan
        )

    def generate_assignments(self, customer_pool: DataFrame, groups: List[str],
                             sample_size_per_group: int,
                             stratify_by: Optional[str] = None) -> DataFrame:
        total_needed = sample_size_per_group * len(groups)
        if len(customer_pool) < total_needed:
            sample = customer_pool.copy()
        else:
            sample = customer_pool.sample(n=total_needed, random_state=42)
        if stratify_by and stratify_by in sample.columns:
            assignments = []
            for stratum in sample[stratify_by].unique():
                stratum_data = sample[sample[stratify_by] == stratum]
                n_per_group = len(stratum_data) // len(groups)
                shuffled = stratum_data.sample(frac=1, random_state=42)
                for i, group in enumerate(groups):
                    start = i * n_per_group
                    end = start + n_per_group if i < len(groups) - 1 else len(shuffled)
                    group_data = shuffled.iloc[start:end].copy()
                    group_data["group"] = group
                    assignments.append(group_data)
            return concat(assignments, ignore_index=True)
        else:
            shuffled = sample.sample(frac=1, random_state=42).reset_index(drop=True)
            assignments = []
            for i, group in enumerate(groups):
                start = i * sample_size_per_group
                end = start + sample_size_per_group
                group_data = shuffled.iloc[start:end].copy()
                group_data["group"] = group
                assignments.append(group_data)
            return concat(assignments, ignore_index=True)
