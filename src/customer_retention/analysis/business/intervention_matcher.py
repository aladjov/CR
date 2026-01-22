"""Intervention matching and recommendation."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from customer_retention.core.components.enums import RiskSegment


@dataclass
class Intervention:
    name: str
    cost: float
    success_rate: float
    channel: str
    min_ltv: float = 0
    applicable_segments: List[RiskSegment] = field(default_factory=list)
    timing: str = "Within 1 week"


@dataclass
class InterventionRecommendation:
    intervention: Optional[Intervention]
    reasoning: str
    expected_roi: Optional[float] = None
    timing: str = "Within 1 week"
    priority: int = 5


class InterventionCatalog:
    def __init__(self, interventions: List[Intervention]):
        self.interventions = interventions
        self._by_name = {i.name: i for i in interventions}

    def get(self, name: str) -> Optional[Intervention]:
        return self._by_name.get(name)

    def filter_by_segment(self, segment: RiskSegment) -> List[Intervention]:
        return [i for i in self.interventions if segment in i.applicable_segments]

    def filter_by_ltv(self, min_ltv: float) -> List[Intervention]:
        return [i for i in self.interventions if i.min_ltv <= min_ltv]


class InterventionMatcher:
    PRIORITY_MAP = {RiskSegment.CRITICAL: 1, RiskSegment.HIGH: 2,
                    RiskSegment.MEDIUM: 3, RiskSegment.LOW: 4, RiskSegment.VERY_LOW: 5}
    TIMING_MAP = {
        RiskSegment.CRITICAL: "Within 24 hours",
        RiskSegment.HIGH: "Within 3 days",
        RiskSegment.MEDIUM: "Within 1 week",
        RiskSegment.LOW: "Within 2 weeks",
        RiskSegment.VERY_LOW: "Standard schedule"
    }

    def __init__(self, catalog: InterventionCatalog, avg_ltv: float = 500):
        self.catalog = catalog
        self.avg_ltv = avg_ltv

    def match(self, risk_segment: RiskSegment, customer_ltv: float,
              churn_probability: float = 0.5) -> InterventionRecommendation:
        if risk_segment == RiskSegment.VERY_LOW:
            return InterventionRecommendation(
                intervention=Intervention(name="none", cost=0, success_rate=0, channel="none"),
                reasoning="Customer is low risk, no intervention needed",
                expected_roi=0,
                timing=self.TIMING_MAP[risk_segment],
                priority=self.PRIORITY_MAP[risk_segment]
            )
        applicable = self.catalog.filter_by_segment(risk_segment)
        affordable = [i for i in applicable if i.min_ltv <= customer_ltv]
        if not affordable:
            affordable = [i for i in applicable if i.cost <= customer_ltv * 0.1]
        if not affordable and applicable:
            affordable = [min(applicable, key=lambda x: x.min_ltv)]
        if not affordable:
            return InterventionRecommendation(
                intervention=None,
                reasoning="No suitable intervention found",
                timing=self.TIMING_MAP.get(risk_segment, "Within 1 week"),
                priority=self.PRIORITY_MAP.get(risk_segment, 5)
            )
        best = max(affordable, key=lambda i: self._calculate_roi(i, churn_probability, customer_ltv))
        roi = self._calculate_roi(best, churn_probability, customer_ltv)
        return InterventionRecommendation(
            intervention=best,
            reasoning=f"Best ROI option for {risk_segment.value} risk with LTV ${customer_ltv:.0f}",
            expected_roi=roi,
            timing=self.TIMING_MAP.get(risk_segment, "Within 1 week"),
            priority=self.PRIORITY_MAP.get(risk_segment, 5)
        )

    def _calculate_roi(self, intervention: Intervention, churn_prob: float, ltv: float) -> float:
        expected_saves = churn_prob * intervention.success_rate
        revenue_saved = expected_saves * ltv
        if intervention.cost == 0:
            return float("inf") if revenue_saved > 0 else 0
        return (revenue_saved - intervention.cost) / intervention.cost

    def match_multiple(self, risk_segment: RiskSegment, customer_ltv: float,
                       churn_probability: float = 0.5, n: int = 3) -> List[InterventionRecommendation]:
        applicable = self.catalog.filter_by_segment(risk_segment)
        affordable = [i for i in applicable if i.min_ltv <= customer_ltv]
        recommendations = []
        for intervention in affordable:
            roi = self._calculate_roi(intervention, churn_probability, customer_ltv)
            recommendations.append(InterventionRecommendation(
                intervention=intervention,
                reasoning=f"Option: {intervention.name} via {intervention.channel}",
                expected_roi=roi,
                timing=self.TIMING_MAP.get(risk_segment, "Within 1 week"),
                priority=self.PRIORITY_MAP.get(risk_segment, 5)
            ))
        recommendations.sort(key=lambda r: r.expected_roi or 0, reverse=True)
        return recommendations[:n]

    def match_batch(self, customers: List[Dict]) -> List[InterventionRecommendation]:
        return [self.match(
            risk_segment=c["risk_segment"],
            customer_ltv=c.get("customer_ltv", self.avg_ltv),
            churn_probability=c.get("churn_probability", 0.5)
        ) for c in customers]
