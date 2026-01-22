import pytest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from customer_retention.analysis.interpretability import (
    CohortAnalyzer, CohortInsight, CohortComparison
)


@pytest.fixture
def sample_data():
    np.random.seed(42)
    n = 300
    X = pd.DataFrame({
        "recency": np.random.randint(1, 365, n),
        "frequency": np.random.randint(1, 50, n),
        "monetary": np.random.uniform(10, 500, n),
        "tenure": np.random.randint(30, 1000, n),
        "engagement": np.random.uniform(0, 1, n),
    })
    y = pd.Series(np.random.choice([0, 1], n, p=[0.3, 0.7]))
    cohorts = pd.Series(np.random.choice(["New", "Established", "Mature"], n))
    return X, y, cohorts


@pytest.fixture
def trained_model(sample_data):
    X, y, _ = sample_data
    model = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)
    model.fit(X, y)
    return model


class TestCohortAnalysis:
    def test_ac7_5_cohort_analysis_runs(self, trained_model, sample_data):
        X, y, cohorts = sample_data
        analyzer = CohortAnalyzer(trained_model, X)
        result = analyzer.analyze(X, y, cohorts)
        assert result is not None
        assert len(result.cohort_insights) > 0

    def test_ac7_6_differences_identified(self, trained_model, sample_data):
        X, y, cohorts = sample_data
        analyzer = CohortAnalyzer(trained_model, X)
        result = analyzer.analyze(X, y, cohorts)
        assert hasattr(result, "key_differences")
        assert len(result.key_differences) > 0


class TestCohortInsights:
    def test_insight_contains_cohort_name(self, trained_model, sample_data):
        X, y, cohorts = sample_data
        analyzer = CohortAnalyzer(trained_model, X)
        result = analyzer.analyze(X, y, cohorts)
        for insight in result.cohort_insights:
            assert isinstance(insight, CohortInsight)
            assert insight.cohort_name in ["New", "Established", "Mature"]

    def test_insight_contains_cohort_size(self, trained_model, sample_data):
        X, y, cohorts = sample_data
        analyzer = CohortAnalyzer(trained_model, X)
        result = analyzer.analyze(X, y, cohorts)
        total = sum(insight.cohort_size for insight in result.cohort_insights)
        assert total == len(X)

    def test_insight_contains_churn_rate(self, trained_model, sample_data):
        X, y, cohorts = sample_data
        analyzer = CohortAnalyzer(trained_model, X)
        result = analyzer.analyze(X, y, cohorts)
        for insight in result.cohort_insights:
            assert 0 <= insight.churn_rate <= 1

    def test_insight_contains_top_features(self, trained_model, sample_data):
        X, y, cohorts = sample_data
        analyzer = CohortAnalyzer(trained_model, X)
        result = analyzer.analyze(X, y, cohorts)
        for insight in result.cohort_insights:
            assert len(insight.top_features) > 0


class TestCohortComparison:
    def test_compares_two_cohorts(self, trained_model, sample_data):
        X, y, cohorts = sample_data
        analyzer = CohortAnalyzer(trained_model, X)
        comparison = analyzer.compare_cohorts(X, y, cohorts, "New", "Mature")
        assert isinstance(comparison, CohortComparison)
        assert comparison.cohort_a == "New"
        assert comparison.cohort_b == "Mature"

    def test_comparison_identifies_feature_differences(self, trained_model, sample_data):
        X, y, cohorts = sample_data
        analyzer = CohortAnalyzer(trained_model, X)
        comparison = analyzer.compare_cohorts(X, y, cohorts, "New", "Mature")
        assert len(comparison.feature_differences) > 0

    def test_comparison_identifies_churn_difference(self, trained_model, sample_data):
        X, y, cohorts = sample_data
        analyzer = CohortAnalyzer(trained_model, X)
        comparison = analyzer.compare_cohorts(X, y, cohorts, "New", "Mature")
        assert comparison.churn_rate_difference is not None


class TestCohortStrategies:
    def test_ac7_7_insights_actionable(self, trained_model, sample_data):
        X, y, cohorts = sample_data
        analyzer = CohortAnalyzer(trained_model, X)
        result = analyzer.analyze(X, y, cohorts)
        for insight in result.cohort_insights:
            assert insight.recommended_strategy is not None
            assert len(insight.recommended_strategy) > 0


class TestCohortTypes:
    def test_tenure_cohorts_creation(self, sample_data):
        X, y, _ = sample_data
        cohorts = CohortAnalyzer.create_tenure_cohorts(X["tenure"], bins=[0, 90, 365, float("inf")])
        assert all(c in ["New", "Established", "Mature"] for c in cohorts.unique())

    def test_value_cohorts_creation(self, sample_data):
        X, y, _ = sample_data
        cohorts = CohortAnalyzer.create_value_cohorts(X["monetary"], quantiles=[0.33, 0.66])
        assert all(c in ["Low", "Medium", "High"] for c in cohorts.unique())

    def test_activity_cohorts_creation(self, sample_data):
        X, y, _ = sample_data
        cohorts = CohortAnalyzer.create_activity_cohorts(X["frequency"], thresholds=[5, 15])
        assert all(c in ["Dormant", "Moderate", "Active"] for c in cohorts.unique())
