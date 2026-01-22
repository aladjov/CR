"""Business report generation."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from customer_retention.core.compat import DataFrame, Series, pd


@dataclass
class ExecutiveDashboard:
    total_customers: int
    churn_rate: float
    revenue_at_risk: float
    risk_distribution: Dict[str, int]
    expected_saves: Optional[int] = None
    expected_roi: Optional[float] = None
    trend: Optional[Dict[str, float]] = None
    top_actions: List[str] = field(default_factory=list)


@dataclass
class CampaignList:
    customers: List[Dict]
    total_count: int
    segment_breakdown: Dict[str, int]

    def to_dict_list(self) -> List[Dict]:
        return self.customers

    def to_dataframe(self) -> DataFrame:
        return pd.DataFrame(self.customers)


@dataclass
class CustomerServiceReport:
    customer_id: str
    risk_segment: str
    churn_probability: float
    risk_factors: List[Dict]
    talking_points: List[str]
    offer_eligibility: List[str]
    intervention_history: List[Dict] = field(default_factory=list)


@dataclass
class ProductInsights:
    top_churn_drivers: List[Dict[str, Any]]
    segment_risk_profiles: Dict[str, Dict]
    product_gaps: List[str]
    competitive_indicators: List[str]
    improvement_recommendations: List[str]


@dataclass
class GovernanceReport:
    model_performance: Dict[str, float]
    data_quality_summary: Dict[str, float]
    drift_status: Optional[Dict[str, bool]] = None
    fairness_summary: Optional[Dict[str, float]] = None
    retraining_recommendation: str = "No retraining needed"


class ReportGenerator:
    def generate_executive_dashboard(self, customer_data: DataFrame,
                                     model_metrics: Dict[str, float],
                                     intervention_data: Optional[Dict] = None) -> ExecutiveDashboard:
        total = len(customer_data)
        churn_rate = customer_data["churn_probability"].mean()
        if "ltv" in customer_data.columns:
            revenue_at_risk = (customer_data["churn_probability"] * customer_data["ltv"]).sum()
        else:
            revenue_at_risk = churn_rate * total * 500
        if "risk_segment" in customer_data.columns:
            risk_dist = customer_data["risk_segment"].value_counts().to_dict()
        else:
            risk_dist = {"Unknown": total}
        expected_saves = intervention_data.get("expected_saves") if intervention_data else None
        expected_roi = intervention_data.get("expected_roi") if intervention_data else None
        top_actions = self._generate_top_actions(customer_data, risk_dist)
        return ExecutiveDashboard(
            total_customers=total,
            churn_rate=churn_rate,
            revenue_at_risk=revenue_at_risk,
            risk_distribution=risk_dist,
            expected_saves=expected_saves,
            expected_roi=expected_roi,
            top_actions=top_actions
        )

    def _generate_top_actions(self, data: DataFrame, risk_dist: Dict) -> List[str]:
        actions = []
        critical = risk_dist.get("Critical", 0)
        high = risk_dist.get("High", 0)
        if critical > 0:
            actions.append(f"Prioritize outreach to {critical} critical-risk customers")
        if high > 0:
            actions.append(f"Schedule engagement campaigns for {high} high-risk customers")
        actions.append("Review top churn drivers for product improvements")
        return actions[:5]

    def generate_campaign_list(self, customer_data: DataFrame,
                               risk_segments: List[str]) -> CampaignList:
        filtered = customer_data[customer_data["risk_segment"].isin(risk_segments)]
        customers = []
        for _, row in filtered.iterrows():
            customers.append({
                "customer_id": row.get("customer_id", ""),
                "risk_segment": row["risk_segment"],
                "churn_probability": row["churn_probability"],
                "ltv": row.get("ltv", 500),
                "recommended_intervention": self._get_intervention(row["risk_segment"])
            })
        segment_breakdown = filtered["risk_segment"].value_counts().to_dict()
        return CampaignList(
            customers=customers,
            total_count=len(customers),
            segment_breakdown=segment_breakdown
        )

    def _get_intervention(self, segment: str) -> str:
        interventions = {
            "Critical": "Account manager call",
            "High": "Phone call + discount",
            "Medium": "Personalized email",
            "Low": "Standard nurturing"
        }
        return interventions.get(segment, "Standard communication")

    def generate_customer_service_report(self, customer_id: str,
                                         customer_data: Series,
                                         risk_factors: List[Dict]) -> CustomerServiceReport:
        risk_segment = customer_data.get("risk_segment", "Unknown")
        churn_prob = customer_data.get("churn_probability", 0.5)
        talking_points = self._generate_talking_points(risk_factors, risk_segment)
        offer_eligibility = self._determine_offers(risk_segment, customer_data.get("ltv", 500))
        return CustomerServiceReport(
            customer_id=customer_id,
            risk_segment=risk_segment,
            churn_probability=churn_prob,
            risk_factors=risk_factors,
            talking_points=talking_points,
            offer_eligibility=offer_eligibility
        )

    def _generate_talking_points(self, risk_factors: List[Dict], segment: str) -> List[str]:
        points = [f"Customer is in {segment} risk category"]
        for factor in risk_factors[:3]:
            name = factor.get("name", "Unknown factor")
            points.append(f"Address concern about {name}")
        points.append("Express appreciation for their business")
        return points

    def _determine_offers(self, segment: str, ltv: float) -> List[str]:
        offers = ["Standard loyalty points"]
        if segment in ["Critical", "High"]:
            offers.append("10% discount on next order")
            if ltv > 500:
                offers.append("Free premium upgrade for 1 month")
        if segment == "Critical":
            offers.append("Dedicated account manager")
        return offers

    def generate_product_insights(self, customer_data: DataFrame,
                                  feature_importance: Dict[str, float]) -> ProductInsights:
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        top_drivers = [{"feature": f, "importance": i} for f, i in sorted_features[:5]]
        segment_profiles = {}
        if "risk_segment" in customer_data.columns:
            for segment in customer_data["risk_segment"].unique():
                seg_data = customer_data[customer_data["risk_segment"] == segment]
                segment_profiles[segment] = {
                    "count": len(seg_data),
                    "avg_churn_prob": seg_data["churn_probability"].mean()
                }
        gaps = self._identify_product_gaps(feature_importance)
        indicators = self._identify_competitive_indicators(feature_importance)
        recommendations = self._generate_improvement_recommendations(top_drivers)
        return ProductInsights(
            top_churn_drivers=top_drivers,
            segment_risk_profiles=segment_profiles,
            product_gaps=gaps,
            competitive_indicators=indicators,
            improvement_recommendations=recommendations
        )

    def _identify_product_gaps(self, importance: Dict[str, float]) -> List[str]:
        gaps = []
        if importance.get("engagement", 0) > 0.15:
            gaps.append("Low engagement indicates need for better onboarding")
        if importance.get("recency", 0) > 0.15:
            gaps.append("High recency impact suggests need for re-engagement features")
        if not gaps:
            gaps.append("No critical product gaps identified")
        return gaps

    def _identify_competitive_indicators(self, importance: Dict[str, float]) -> List[str]:
        return ["Monitor competitor pricing", "Track feature parity"]

    def _generate_improvement_recommendations(self, drivers: List[Dict]) -> List[str]:
        recommendations = []
        for driver in drivers[:3]:
            feature = driver["feature"]
            recommendations.append(f"Improve {feature} experience to reduce churn")
        return recommendations

    def generate_governance_report(self, model_metrics: Dict[str, float],
                                   data_quality_summary: Dict[str, float],
                                   drift_status: Optional[Dict] = None,
                                   fairness_summary: Optional[Dict] = None) -> GovernanceReport:
        retraining_rec = "No retraining needed"
        if drift_status:
            if drift_status.get("feature_drift", False) or drift_status.get("target_drift", False):
                retraining_rec = "Retraining recommended due to detected drift"
        if model_metrics.get("pr_auc", 1) < 0.6:
            retraining_rec = "Retraining recommended due to performance degradation"
        return GovernanceReport(
            model_performance=model_metrics,
            data_quality_summary=data_quality_summary,
            drift_status=drift_status,
            fairness_summary=fairness_summary,
            retraining_recommendation=retraining_rec
        )
