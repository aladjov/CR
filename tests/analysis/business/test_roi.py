import pytest

from customer_retention.analysis.business import InterventionROI, OptimizationResult, ROIAnalyzer


@pytest.fixture
def business_params():
    return {
        "avg_customer_ltv": 500,
        "monthly_revenue": 50,
        "customer_lifespan": 24,
    }


@pytest.fixture
def intervention_costs():
    return {
        "email_campaign": 2,
        "phone_call": 15,
        "discount_10pct": 25,
        "account_manager": 150,
    }


@pytest.fixture
def success_rates():
    return {
        "email_campaign": 0.10,
        "phone_call": 0.25,
        "discount_10pct": 0.30,
        "account_manager": 0.60,
    }


class TestROICalculation:
    def test_ac7_17_roi_calculated(self, business_params, intervention_costs, success_rates):
        analyzer = ROIAnalyzer(
            avg_ltv=business_params["avg_customer_ltv"],
            intervention_costs=intervention_costs,
            success_rates=success_rates
        )
        result = analyzer.calculate_roi(
            intervention="phone_call",
            targeted_customers=100,
            actual_churn_rate=0.25
        )
        assert isinstance(result, InterventionROI)
        assert result.roi_pct is not None

    def test_roi_formula_correct(self, business_params, intervention_costs, success_rates):
        analyzer = ROIAnalyzer(
            avg_ltv=500,
            intervention_costs=intervention_costs,
            success_rates=success_rates
        )
        result = analyzer.calculate_roi(
            intervention="phone_call",
            targeted_customers=100,
            actual_churn_rate=0.25
        )
        expected_churners = 100 * 0.25
        expected_saves = expected_churners * 0.25
        expected_revenue = expected_saves * 500
        expected_cost = 100 * 15
        expected_net = expected_revenue - expected_cost
        expected_roi = (expected_net / expected_cost) * 100
        assert result.roi_pct == pytest.approx(expected_roi, rel=0.01)

    def test_roi_contains_all_components(self, business_params, intervention_costs, success_rates):
        analyzer = ROIAnalyzer(
            avg_ltv=business_params["avg_customer_ltv"],
            intervention_costs=intervention_costs,
            success_rates=success_rates
        )
        result = analyzer.calculate_roi(
            intervention="phone_call",
            targeted_customers=100,
            actual_churn_rate=0.25
        )
        assert result.targeted_customers == 100
        assert result.actual_churners is not None
        assert result.customers_saved is not None
        assert result.total_cost is not None
        assert result.revenue_saved is not None
        assert result.net_benefit is not None


class TestPositiveROI:
    def test_ac7_18_positive_roi_strategies_exist(self, business_params, intervention_costs, success_rates):
        analyzer = ROIAnalyzer(
            avg_ltv=business_params["avg_customer_ltv"],
            intervention_costs=intervention_costs,
            success_rates=success_rates
        )
        results = analyzer.analyze_all_interventions(
            targeted_customers=100,
            actual_churn_rate=0.30
        )
        positive_roi = [r for r in results if r.roi_pct > 0]
        assert len(positive_roi) > 0


class TestSegmentROI:
    def test_roi_by_segment(self, business_params, intervention_costs, success_rates):
        analyzer = ROIAnalyzer(
            avg_ltv=business_params["avg_customer_ltv"],
            intervention_costs=intervention_costs,
            success_rates=success_rates
        )
        segment_data = {
            "Critical": {"customers": 50, "churn_rate": 0.80},
            "High": {"customers": 100, "churn_rate": 0.65},
            "Medium": {"customers": 200, "churn_rate": 0.45},
        }
        results = analyzer.analyze_by_segment(segment_data)
        assert "Critical" in results
        assert "High" in results
        assert "Medium" in results

    def test_best_intervention_per_segment(self, business_params, intervention_costs, success_rates):
        analyzer = ROIAnalyzer(
            avg_ltv=business_params["avg_customer_ltv"],
            intervention_costs=intervention_costs,
            success_rates=success_rates
        )
        segment_data = {
            "Critical": {"customers": 50, "churn_rate": 0.80},
            "Medium": {"customers": 200, "churn_rate": 0.45},
        }
        results = analyzer.analyze_by_segment(segment_data)
        for segment, segment_results in results.items():
            best = max(segment_results, key=lambda r: r.roi_pct)
            assert best is not None


class TestBudgetOptimization:
    def test_ac7_19_optimization_works(self, business_params, intervention_costs, success_rates):
        analyzer = ROIAnalyzer(
            avg_ltv=business_params["avg_customer_ltv"],
            intervention_costs=intervention_costs,
            success_rates=success_rates
        )
        segment_data = {
            "Critical": {"customers": 50, "churn_rate": 0.80},
            "High": {"customers": 100, "churn_rate": 0.65},
            "Medium": {"customers": 200, "churn_rate": 0.45},
        }
        result = analyzer.optimize_budget(
            segment_data=segment_data,
            total_budget=5000
        )
        assert isinstance(result, OptimizationResult)
        assert result.total_cost <= 5000

    def test_optimization_maximizes_roi(self, business_params, intervention_costs, success_rates):
        analyzer = ROIAnalyzer(
            avg_ltv=business_params["avg_customer_ltv"],
            intervention_costs=intervention_costs,
            success_rates=success_rates
        )
        segment_data = {
            "Critical": {"customers": 50, "churn_rate": 0.80},
            "High": {"customers": 100, "churn_rate": 0.65},
        }
        result = analyzer.optimize_budget(
            segment_data=segment_data,
            total_budget=3000,
            objective="maximize_roi"
        )
        assert result.overall_roi is not None

    def test_optimization_maximizes_saves(self, business_params, intervention_costs, success_rates):
        analyzer = ROIAnalyzer(
            avg_ltv=business_params["avg_customer_ltv"],
            intervention_costs=intervention_costs,
            success_rates=success_rates
        )
        segment_data = {
            "Critical": {"customers": 50, "churn_rate": 0.80},
            "High": {"customers": 100, "churn_rate": 0.65},
        }
        result = analyzer.optimize_budget(
            segment_data=segment_data,
            total_budget=3000,
            objective="maximize_saves"
        )
        assert result.total_saves is not None


class TestROIComparison:
    def test_compares_interventions(self, business_params, intervention_costs, success_rates):
        analyzer = ROIAnalyzer(
            avg_ltv=business_params["avg_customer_ltv"],
            intervention_costs=intervention_costs,
            success_rates=success_rates
        )
        comparison = analyzer.compare_interventions(
            targeted_customers=100,
            actual_churn_rate=0.30
        )
        assert len(comparison) == len(intervention_costs)

    def test_comparison_sorted_by_roi(self, business_params, intervention_costs, success_rates):
        analyzer = ROIAnalyzer(
            avg_ltv=business_params["avg_customer_ltv"],
            intervention_costs=intervention_costs,
            success_rates=success_rates
        )
        comparison = analyzer.compare_interventions(
            targeted_customers=100,
            actual_churn_rate=0.30
        )
        rois = [r.roi_pct for r in comparison]
        assert rois == sorted(rois, reverse=True)


class TestScenarioAnalysis:
    def test_scenario_analysis(self, business_params, intervention_costs, success_rates):
        analyzer = ROIAnalyzer(
            avg_ltv=business_params["avg_customer_ltv"],
            intervention_costs=intervention_costs,
            success_rates=success_rates
        )
        scenarios = analyzer.run_scenarios(
            intervention="phone_call",
            targeted_customers=100,
            churn_rates=[0.20, 0.30, 0.40, 0.50]
        )
        assert len(scenarios) == 4
