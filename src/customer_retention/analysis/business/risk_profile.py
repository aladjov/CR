"""Customer risk profiling."""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Any
import numpy as np
import shap

from customer_retention.core.compat import pd, DataFrame, Series
from customer_retention.core.components.enums import RiskSegment


class Urgency(Enum):
    IMMEDIATE = "Immediate"
    THIS_WEEK = "This Week"
    THIS_MONTH = "This Month"
    MONITOR = "Monitor"


@dataclass
class RiskFactor:
    factor_name: str
    current_value: str
    comparison: str
    impact: str
    actionable: bool
    suggested_action: str = ""


@dataclass
class Intervention:
    intervention_type: str
    description: str
    estimated_cost: float
    estimated_success_rate: float
    expected_roi: float
    priority: int
    reasoning: str
    channel: str
    timing: str


@dataclass
class CustomerRiskProfile:
    customer_id: Optional[str]
    churn_probability: float
    risk_segment: RiskSegment
    confidence: str
    risk_factors: List[RiskFactor]
    recommended_interventions: List[Intervention]
    expected_ltv_if_retained: float
    expected_ltv_if_churned: float
    intervention_roi_estimate: float
    urgency: Urgency
    days_until_likely_churn: Optional[int] = None


class RiskProfiler:
    SEGMENT_THRESHOLDS = [(0.80, RiskSegment.CRITICAL), (0.60, RiskSegment.HIGH),
                         (0.40, RiskSegment.MEDIUM), (0.20, RiskSegment.LOW)]
    INTERVENTION_CATALOG = [
        {"name": "email_campaign", "cost": 2, "success_rate": 0.10, "channel": "email",
         "segments": [RiskSegment.LOW, RiskSegment.MEDIUM]},
        {"name": "phone_call", "cost": 15, "success_rate": 0.25, "channel": "phone",
         "segments": [RiskSegment.MEDIUM, RiskSegment.HIGH]},
        {"name": "discount_offer", "cost": 25, "success_rate": 0.35, "channel": "email",
         "segments": [RiskSegment.HIGH, RiskSegment.CRITICAL]},
        {"name": "account_manager", "cost": 150, "success_rate": 0.60, "channel": "personal",
         "segments": [RiskSegment.CRITICAL]},
    ]

    def __init__(self, model: Any, background_data: DataFrame,
                 actionable_features: Optional[List[str]] = None,
                 avg_customer_ltv: float = 500, max_samples: int = 100):
        self.model = model
        self.background_data = background_data.head(max_samples)
        self.actionable_features = actionable_features or []
        self.avg_ltv = avg_customer_ltv
        self.feature_names = list(background_data.columns)
        self._explainer = self._create_explainer()

    def _create_explainer(self) -> shap.Explainer:
        model_type = type(self.model).__name__
        if model_type in ["RandomForestClassifier", "GradientBoostingClassifier"]:
            return shap.TreeExplainer(self.model)
        return shap.KernelExplainer(self.model.predict_proba, self.background_data)

    def generate_profile(self, instance: Series,
                         customer_id: Optional[str] = None) -> CustomerRiskProfile:
        instance_df = instance.to_frame().T
        churn_prob = float(self.model.predict_proba(instance_df)[0, 1])
        segment = self._assign_segment(churn_prob)
        confidence = self._assess_confidence(churn_prob)
        risk_factors = self._extract_risk_factors(instance)
        interventions = self._match_interventions(segment, churn_prob)
        ltv_retained = self.avg_ltv
        ltv_churned = self.avg_ltv * 0.1
        best_intervention = interventions[0] if interventions else None
        roi = best_intervention.expected_roi if best_intervention else 0
        urgency = self._assign_urgency(segment)
        return CustomerRiskProfile(
            customer_id=customer_id,
            churn_probability=churn_prob,
            risk_segment=segment,
            confidence=confidence,
            risk_factors=risk_factors,
            recommended_interventions=interventions,
            expected_ltv_if_retained=ltv_retained,
            expected_ltv_if_churned=ltv_churned,
            intervention_roi_estimate=roi,
            urgency=urgency
        )

    def _assign_segment(self, probability: float) -> RiskSegment:
        for threshold, segment in self.SEGMENT_THRESHOLDS:
            if probability >= threshold:
                return segment
        return RiskSegment.VERY_LOW

    def _assess_confidence(self, probability: float) -> str:
        if probability < 0.2 or probability > 0.8:
            return "High"
        if 0.4 < probability < 0.6:
            return "Low"
        return "Medium"

    def _extract_risk_factors(self, instance: Series) -> List[RiskFactor]:
        instance_df = instance.to_frame().T
        shap_values = self._extract_shap_values(instance_df)
        sorted_indices = np.argsort(np.abs(shap_values))[::-1]
        factors = []
        for idx in sorted_indices[:5]:
            feature = self.feature_names[idx]
            value = instance[feature]
            impact_pct = abs(shap_values[idx]) * 100
            direction = "increases" if shap_values[idx] > 0 else "decreases"
            factors.append(RiskFactor(
                factor_name=feature,
                current_value=f"{value:.2f}" if isinstance(value, float) else str(value),
                comparison=f"vs avg {self.background_data[feature].mean():.2f}",
                impact=f"{direction} risk by {impact_pct:.1f}%",
                actionable=feature in self.actionable_features,
                suggested_action=f"Improve {feature}" if feature in self.actionable_features else ""
            ))
        return factors

    def _extract_shap_values(self, X: DataFrame) -> np.ndarray:
        shap_values = self._explainer.shap_values(X)
        if hasattr(shap_values, 'values'):
            shap_values = shap_values.values
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        if len(shap_values.shape) == 3:
            shap_values = shap_values[:, :, 1]
        return shap_values.flatten()

    def _match_interventions(self, segment: RiskSegment, churn_prob: float) -> List[Intervention]:
        applicable = [i for i in self.INTERVENTION_CATALOG if segment in i["segments"]]
        interventions = []
        for item in applicable:
            expected_saves = churn_prob * item["success_rate"]
            revenue_saved = expected_saves * self.avg_ltv
            roi = (revenue_saved - item["cost"]) / item["cost"] if item["cost"] > 0 else 0
            interventions.append(Intervention(
                intervention_type=item["name"],
                description=f"{item['name'].replace('_', ' ').title()} via {item['channel']}",
                estimated_cost=item["cost"],
                estimated_success_rate=item["success_rate"],
                expected_roi=roi,
                priority=self._get_priority(segment),
                reasoning=f"Recommended for {segment.value} risk customers",
                channel=item["channel"],
                timing="Within 24 hours" if segment == RiskSegment.CRITICAL else "Within 1 week"
            ))
        return sorted(interventions, key=lambda x: x.expected_roi, reverse=True)

    def _get_priority(self, segment: RiskSegment) -> int:
        priorities = {RiskSegment.CRITICAL: 1, RiskSegment.HIGH: 2,
                      RiskSegment.MEDIUM: 3, RiskSegment.LOW: 4, RiskSegment.VERY_LOW: 5}
        return priorities.get(segment, 5)

    def _assign_urgency(self, segment: RiskSegment) -> Urgency:
        urgency_map = {
            RiskSegment.CRITICAL: Urgency.IMMEDIATE,
            RiskSegment.HIGH: Urgency.THIS_WEEK,
            RiskSegment.MEDIUM: Urgency.THIS_MONTH,
            RiskSegment.LOW: Urgency.MONITOR,
            RiskSegment.VERY_LOW: Urgency.MONITOR
        }
        return urgency_map.get(segment, Urgency.MONITOR)

    def generate_batch(self, X: DataFrame, customer_ids: Optional[List[str]] = None,
                       sort_by_risk: bool = False) -> List[CustomerRiskProfile]:
        customer_ids = customer_ids or [None] * len(X)
        profiles = [self.generate_profile(X.iloc[i], customer_ids[i]) for i in range(len(X))]
        if sort_by_risk:
            profiles.sort(key=lambda p: p.churn_probability, reverse=True)
        return profiles
