import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from customer_retention.analysis.business import CustomerRiskProfile, RiskFactor, RiskProfiler, RiskSegment, Urgency


@pytest.fixture
def sample_data():
    np.random.seed(42)
    n = 200
    X = pd.DataFrame({
        "recency": np.random.randint(1, 365, n),
        "frequency": np.random.randint(1, 50, n),
        "monetary": np.random.uniform(10, 500, n),
        "tenure": np.random.randint(30, 1000, n),
        "engagement": np.random.uniform(0, 1, n),
    })
    y = pd.Series(np.random.choice([0, 1], n, p=[0.3, 0.7]))
    return X, y


@pytest.fixture
def trained_model(sample_data):
    X, y = sample_data
    model = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)
    model.fit(X, y)
    return model


class TestRiskProfileGeneration:
    def test_ac7_14_profiles_complete(self, trained_model, sample_data):
        X, _ = sample_data
        profiler = RiskProfiler(trained_model, X)
        profile = profiler.generate_profile(X.iloc[0], customer_id="CUST001")
        assert isinstance(profile, CustomerRiskProfile)
        assert profile.customer_id == "CUST001"
        assert profile.churn_probability is not None
        assert profile.risk_segment is not None
        assert profile.confidence is not None

    def test_ac7_15_segments_assigned(self, trained_model, sample_data):
        X, _ = sample_data
        profiler = RiskProfiler(trained_model, X)
        segments_found = set()
        for i in range(len(X)):
            profile = profiler.generate_profile(X.iloc[i])
            segments_found.add(profile.risk_segment)
        assert len(segments_found) > 1

    def test_ac7_16_interventions_matched(self, trained_model, sample_data):
        X, _ = sample_data
        profiler = RiskProfiler(trained_model, X)
        profile = profiler.generate_profile(X.iloc[0])
        assert len(profile.recommended_interventions) > 0


class TestRiskSegments:
    def test_critical_segment_threshold(self, trained_model, sample_data):
        X, _ = sample_data
        profiler = RiskProfiler(trained_model, X)
        for i in range(len(X)):
            profile = profiler.generate_profile(X.iloc[i])
            if profile.churn_probability >= 0.80:
                assert profile.risk_segment == RiskSegment.CRITICAL

    def test_high_segment_threshold(self, trained_model, sample_data):
        X, _ = sample_data
        profiler = RiskProfiler(trained_model, X)
        for i in range(len(X)):
            profile = profiler.generate_profile(X.iloc[i])
            if 0.60 <= profile.churn_probability < 0.80:
                assert profile.risk_segment == RiskSegment.HIGH

    def test_low_segment_threshold(self, trained_model, sample_data):
        X, _ = sample_data
        profiler = RiskProfiler(trained_model, X)
        for i in range(len(X)):
            profile = profiler.generate_profile(X.iloc[i])
            if profile.churn_probability < 0.20:
                assert profile.risk_segment == RiskSegment.VERY_LOW


class TestRiskFactors:
    def test_risk_factors_extracted(self, trained_model, sample_data):
        X, _ = sample_data
        profiler = RiskProfiler(trained_model, X)
        profile = profiler.generate_profile(X.iloc[0])
        assert len(profile.risk_factors) > 0

    def test_risk_factors_contain_required_fields(self, trained_model, sample_data):
        X, _ = sample_data
        profiler = RiskProfiler(trained_model, X)
        profile = profiler.generate_profile(X.iloc[0])
        for factor in profile.risk_factors:
            assert isinstance(factor, RiskFactor)
            assert factor.factor_name is not None
            assert factor.current_value is not None
            assert factor.impact is not None

    def test_risk_factors_actionable_flag(self, trained_model, sample_data):
        X, _ = sample_data
        actionable = ["frequency", "engagement"]
        profiler = RiskProfiler(trained_model, X, actionable_features=actionable)
        profile = profiler.generate_profile(X.iloc[0])
        for factor in profile.risk_factors:
            expected_actionable = factor.factor_name in actionable
            assert factor.actionable == expected_actionable


class TestUrgency:
    def test_urgency_assigned(self, trained_model, sample_data):
        X, _ = sample_data
        profiler = RiskProfiler(trained_model, X)
        profile = profiler.generate_profile(X.iloc[0])
        assert profile.urgency in [Urgency.IMMEDIATE, Urgency.THIS_WEEK, Urgency.THIS_MONTH, Urgency.MONITOR]

    def test_immediate_urgency_for_critical(self, trained_model, sample_data):
        X, _ = sample_data
        profiler = RiskProfiler(trained_model, X)
        for i in range(len(X)):
            profile = profiler.generate_profile(X.iloc[i])
            if profile.risk_segment == RiskSegment.CRITICAL:
                assert profile.urgency == Urgency.IMMEDIATE


class TestLTVEstimation:
    def test_ltv_if_retained_calculated(self, trained_model, sample_data):
        X, _ = sample_data
        profiler = RiskProfiler(trained_model, X, avg_customer_ltv=500)
        profile = profiler.generate_profile(X.iloc[0])
        assert profile.expected_ltv_if_retained > 0

    def test_ltv_if_churned_calculated(self, trained_model, sample_data):
        X, _ = sample_data
        profiler = RiskProfiler(trained_model, X, avg_customer_ltv=500)
        profile = profiler.generate_profile(X.iloc[0])
        assert profile.expected_ltv_if_churned is not None

    def test_intervention_roi_estimated(self, trained_model, sample_data):
        X, _ = sample_data
        profiler = RiskProfiler(trained_model, X, avg_customer_ltv=500)
        profile = profiler.generate_profile(X.iloc[0])
        assert profile.intervention_roi_estimate is not None


class TestBatchProfiles:
    def test_generates_batch_profiles(self, trained_model, sample_data):
        X, _ = sample_data
        profiler = RiskProfiler(trained_model, X)
        profiles = profiler.generate_batch(X.head(20))
        assert len(profiles) == 20

    def test_batch_profiles_sorted_by_risk(self, trained_model, sample_data):
        X, _ = sample_data
        profiler = RiskProfiler(trained_model, X)
        profiles = profiler.generate_batch(X.head(20), sort_by_risk=True)
        probabilities = [p.churn_probability for p in profiles]
        assert probabilities == sorted(probabilities, reverse=True)
