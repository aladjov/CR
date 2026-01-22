"""ROI analysis for retention interventions."""

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class InterventionROI:
    intervention: str
    targeted_customers: int
    actual_churners: float
    customers_saved: float
    total_cost: float
    revenue_saved: float
    net_benefit: float
    roi_pct: float


@dataclass
class OptimizationResult:
    allocations: Dict[str, Dict[str, any]]
    total_cost: float
    total_saves: float
    total_revenue: float
    overall_roi: float


@dataclass
class ROIResult:
    intervention_rois: List[InterventionROI]
    best_intervention: str
    best_roi: float


class ROIAnalyzer:
    def __init__(self, avg_ltv: float, intervention_costs: Dict[str, float],
                 success_rates: Dict[str, float]):
        self.avg_ltv = avg_ltv
        self.intervention_costs = intervention_costs
        self.success_rates = success_rates

    def calculate_roi(self, intervention: str, targeted_customers: int,
                      actual_churn_rate: float) -> InterventionROI:
        cost = self.intervention_costs.get(intervention, 0)
        success_rate = self.success_rates.get(intervention, 0)
        actual_churners = targeted_customers * actual_churn_rate
        customers_saved = actual_churners * success_rate
        revenue_saved = customers_saved * self.avg_ltv
        total_cost = targeted_customers * cost
        net_benefit = revenue_saved - total_cost
        roi_pct = (net_benefit / total_cost * 100) if total_cost > 0 else 0
        return InterventionROI(
            intervention=intervention,
            targeted_customers=targeted_customers,
            actual_churners=actual_churners,
            customers_saved=customers_saved,
            total_cost=total_cost,
            revenue_saved=revenue_saved,
            net_benefit=net_benefit,
            roi_pct=roi_pct
        )

    def analyze_all_interventions(self, targeted_customers: int,
                                  actual_churn_rate: float) -> List[InterventionROI]:
        results = [self.calculate_roi(intervention, targeted_customers, actual_churn_rate)
                   for intervention in self.intervention_costs.keys()]
        return sorted(results, key=lambda r: r.roi_pct, reverse=True)

    def compare_interventions(self, targeted_customers: int,
                              actual_churn_rate: float) -> List[InterventionROI]:
        return self.analyze_all_interventions(targeted_customers, actual_churn_rate)

    def analyze_by_segment(self, segment_data: Dict[str, Dict]) -> Dict[str, List[InterventionROI]]:
        results = {}
        for segment, data in segment_data.items():
            customers = data["customers"]
            churn_rate = data["churn_rate"]
            results[segment] = self.analyze_all_interventions(customers, churn_rate)
        return results

    def optimize_budget(self, segment_data: Dict[str, Dict], total_budget: float,
                        objective: str = "maximize_roi") -> OptimizationResult:
        all_options = []
        for segment, data in segment_data.items():
            customers = data["customers"]
            churn_rate = data["churn_rate"]
            for intervention in self.intervention_costs.keys():
                cost_per = self.intervention_costs[intervention]
                success_rate = self.success_rates[intervention]
                all_options.append({
                    "segment": segment,
                    "intervention": intervention,
                    "customers": customers,
                    "churn_rate": churn_rate,
                    "cost_per": cost_per,
                    "success_rate": success_rate,
                    "total_cost": customers * cost_per,
                    "expected_saves": customers * churn_rate * success_rate,
                    "expected_revenue": customers * churn_rate * success_rate * self.avg_ltv
                })
        for opt in all_options:
            if opt["total_cost"] > 0:
                opt["roi"] = (opt["expected_revenue"] - opt["total_cost"]) / opt["total_cost"]
            else:
                opt["roi"] = 0
        if objective == "maximize_roi":
            all_options.sort(key=lambda x: x["roi"], reverse=True)
        else:
            all_options.sort(key=lambda x: x["expected_saves"], reverse=True)
        allocations = {}
        remaining_budget = total_budget
        total_saves = 0
        total_revenue = 0
        total_cost = 0
        for opt in all_options:
            if opt["total_cost"] <= remaining_budget and opt["segment"] not in allocations:
                allocations[opt["segment"]] = {
                    "intervention": opt["intervention"],
                    "customers": opt["customers"],
                    "cost": opt["total_cost"],
                    "expected_saves": opt["expected_saves"],
                    "expected_revenue": opt["expected_revenue"]
                }
                remaining_budget -= opt["total_cost"]
                total_saves += opt["expected_saves"]
                total_revenue += opt["expected_revenue"]
                total_cost += opt["total_cost"]
        overall_roi = (total_revenue - total_cost) / total_cost if total_cost > 0 else 0
        return OptimizationResult(
            allocations=allocations,
            total_cost=total_cost,
            total_saves=total_saves,
            total_revenue=total_revenue,
            overall_roi=overall_roi
        )

    def run_scenarios(self, intervention: str, targeted_customers: int,
                      churn_rates: List[float]) -> List[InterventionROI]:
        return [self.calculate_roi(intervention, targeted_customers, rate) for rate in churn_rates]
