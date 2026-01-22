import numpy as np
import pandas as pd
import pytest

from customer_retention.analysis.business import ABTestDesign, ABTestDesigner, MeasurementPlan, SampleSizeResult


@pytest.fixture
def sample_customer_pool():
    np.random.seed(42)
    n = 5000
    return pd.DataFrame({
        "customer_id": [f"CUST{i:05d}" for i in range(n)],
        "risk_segment": np.random.choice(["Critical", "High", "Medium", "Low"], n, p=[0.1, 0.2, 0.3, 0.4]),
        "churn_probability": np.random.uniform(0, 1, n),
        "ltv": np.random.uniform(100, 1000, n),
    })


class TestSampleSizeCalculation:
    def test_calculates_sample_size(self):
        designer = ABTestDesigner()
        result = designer.calculate_sample_size(
            baseline_rate=0.25,
            min_detectable_effect=0.05,
            alpha=0.05,
            power=0.80
        )
        assert isinstance(result, SampleSizeResult)
        assert result.sample_size_per_group > 0

    def test_sample_size_increases_for_smaller_effect(self):
        designer = ABTestDesigner()
        large_effect = designer.calculate_sample_size(
            baseline_rate=0.25,
            min_detectable_effect=0.10
        )
        small_effect = designer.calculate_sample_size(
            baseline_rate=0.25,
            min_detectable_effect=0.02
        )
        assert small_effect.sample_size_per_group > large_effect.sample_size_per_group

    def test_sample_size_increases_for_higher_power(self):
        designer = ABTestDesigner()
        low_power = designer.calculate_sample_size(
            baseline_rate=0.25,
            min_detectable_effect=0.05,
            power=0.70
        )
        high_power = designer.calculate_sample_size(
            baseline_rate=0.25,
            min_detectable_effect=0.05,
            power=0.95
        )
        assert high_power.sample_size_per_group > low_power.sample_size_per_group


class TestABTestDesign:
    def test_creates_test_design(self, sample_customer_pool):
        designer = ABTestDesigner()
        design = designer.design_test(
            test_name="retention_campaign_v1",
            customer_pool=sample_customer_pool,
            control_name="no_intervention",
            treatment_names=["email_campaign", "phone_call"],
            baseline_rate=0.25,
            min_detectable_effect=0.05
        )
        assert isinstance(design, ABTestDesign)
        assert design.test_name == "retention_campaign_v1"

    def test_design_contains_sample_size(self, sample_customer_pool):
        designer = ABTestDesigner()
        design = designer.design_test(
            test_name="test1",
            customer_pool=sample_customer_pool,
            control_name="control",
            treatment_names=["treatment"],
            baseline_rate=0.25,
            min_detectable_effect=0.05
        )
        assert design.recommended_sample_size > 0

    def test_design_checks_feasibility(self, sample_customer_pool):
        designer = ABTestDesigner()
        design = designer.design_test(
            test_name="test1",
            customer_pool=sample_customer_pool,
            control_name="control",
            treatment_names=["treatment"],
            baseline_rate=0.25,
            min_detectable_effect=0.05
        )
        assert design.feasible is not None
        assert design.available_customers == len(sample_customer_pool)


class TestStratification:
    def test_stratifies_by_risk_segment(self, sample_customer_pool):
        designer = ABTestDesigner()
        design = designer.design_test(
            test_name="test1",
            customer_pool=sample_customer_pool,
            control_name="control",
            treatment_names=["treatment"],
            baseline_rate=0.25,
            min_detectable_effect=0.05,
            stratify_by="risk_segment"
        )
        assert design.stratification_variable == "risk_segment"

    def test_generates_stratified_assignment(self, sample_customer_pool):
        designer = ABTestDesigner()
        assignments = designer.generate_assignments(
            customer_pool=sample_customer_pool,
            groups=["control", "treatment"],
            sample_size_per_group=500,
            stratify_by="risk_segment"
        )
        assert len(assignments) == 1000
        control_segments = assignments[assignments["group"] == "control"]["risk_segment"]
        treatment_segments = assignments[assignments["group"] == "treatment"]["risk_segment"]
        control_dist = control_segments.value_counts(normalize=True)
        treatment_dist = treatment_segments.value_counts(normalize=True)
        for segment in control_dist.index:
            assert abs(control_dist[segment] - treatment_dist[segment]) < 0.1


class TestMeasurementPlan:
    def test_generates_measurement_plan(self, sample_customer_pool):
        designer = ABTestDesigner()
        design = designer.design_test(
            test_name="test1",
            customer_pool=sample_customer_pool,
            control_name="control",
            treatment_names=["treatment"],
            baseline_rate=0.25,
            min_detectable_effect=0.05,
            primary_metric="churn_rate",
            secondary_metrics=["engagement", "satisfaction"]
        )
        assert isinstance(design.measurement_plan, MeasurementPlan)
        assert design.measurement_plan.primary_metric == "churn_rate"
        assert "engagement" in design.measurement_plan.secondary_metrics


class TestTestDuration:
    def test_estimates_duration(self, sample_customer_pool):
        designer = ABTestDesigner()
        design = designer.design_test(
            test_name="test1",
            customer_pool=sample_customer_pool,
            control_name="control",
            treatment_names=["treatment"],
            baseline_rate=0.25,
            min_detectable_effect=0.05,
            duration_days=30
        )
        assert design.duration_days == 30

    def test_calculates_expected_completion(self, sample_customer_pool):
        designer = ABTestDesigner()
        design = designer.design_test(
            test_name="test1",
            customer_pool=sample_customer_pool,
            control_name="control",
            treatment_names=["treatment"],
            baseline_rate=0.25,
            min_detectable_effect=0.05,
            duration_days=30
        )
        assert design.expected_completion_date is not None


class TestMultipleTreatments:
    def test_supports_multiple_treatments(self, sample_customer_pool):
        designer = ABTestDesigner()
        design = designer.design_test(
            test_name="test1",
            customer_pool=sample_customer_pool,
            control_name="control",
            treatment_names=["treatment_a", "treatment_b", "treatment_c"],
            baseline_rate=0.25,
            min_detectable_effect=0.05
        )
        assert len(design.treatment_groups) == 3
        assert design.total_required == design.recommended_sample_size * 4


class TestPowerAnalysis:
    def test_power_analysis_for_given_sample(self):
        designer = ABTestDesigner()
        power = designer.calculate_power(
            sample_size_per_group=500,
            baseline_rate=0.25,
            effect_size=0.05,
            alpha=0.05
        )
        assert 0 <= power <= 1

    def test_power_increases_with_sample_size(self):
        designer = ABTestDesigner()
        power_small = designer.calculate_power(
            sample_size_per_group=100,
            baseline_rate=0.25,
            effect_size=0.05
        )
        power_large = designer.calculate_power(
            sample_size_per_group=1000,
            baseline_rate=0.25,
            effect_size=0.05
        )
        assert power_large > power_small
