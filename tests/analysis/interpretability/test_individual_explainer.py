import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from customer_retention.analysis.interpretability import Confidence, IndividualExplainer, IndividualExplanation


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


class TestIndividualExplanation:
    def test_ac7_8_waterfall_data_generated(self, trained_model, sample_data):
        X, _ = sample_data
        explainer = IndividualExplainer(trained_model, X)
        result = explainer.explain(X.iloc[0])
        assert isinstance(result, IndividualExplanation)
        assert result.shap_values is not None

    def test_ac7_9_risk_factors_extracted(self, trained_model, sample_data):
        X, _ = sample_data
        explainer = IndividualExplainer(trained_model, X)
        result = explainer.explain(X.iloc[0])
        assert len(result.top_positive_factors) > 0
        assert len(result.top_negative_factors) > 0

    def test_ac7_10_confidence_assigned(self, trained_model, sample_data):
        X, _ = sample_data
        explainer = IndividualExplainer(trained_model, X)
        result = explainer.explain(X.iloc[0])
        assert result.confidence in [Confidence.HIGH, Confidence.MEDIUM, Confidence.LOW]


class TestExplanationDetails:
    def test_churn_probability_returned(self, trained_model, sample_data):
        X, _ = sample_data
        explainer = IndividualExplainer(trained_model, X)
        result = explainer.explain(X.iloc[0])
        assert 0 <= result.churn_probability <= 1

    def test_base_value_returned(self, trained_model, sample_data):
        X, _ = sample_data
        explainer = IndividualExplainer(trained_model, X)
        result = explainer.explain(X.iloc[0])
        assert result.base_value is not None

    def test_customer_id_attached(self, trained_model, sample_data):
        X, _ = sample_data
        explainer = IndividualExplainer(trained_model, X)
        result = explainer.explain(X.iloc[0], customer_id="CUST001")
        assert result.customer_id == "CUST001"


class TestRiskFactors:
    def test_top_3_positive_factors(self, trained_model, sample_data):
        X, _ = sample_data
        explainer = IndividualExplainer(trained_model, X)
        result = explainer.explain(X.iloc[0], top_n=3)
        assert len(result.top_positive_factors) <= 3

    def test_top_3_negative_factors(self, trained_model, sample_data):
        X, _ = sample_data
        explainer = IndividualExplainer(trained_model, X)
        result = explainer.explain(X.iloc[0], top_n=3)
        assert len(result.top_negative_factors) <= 3

    def test_factors_contain_feature_name(self, trained_model, sample_data):
        X, _ = sample_data
        explainer = IndividualExplainer(trained_model, X)
        result = explainer.explain(X.iloc[0])
        for factor in result.top_positive_factors:
            assert factor.feature_name in X.columns

    def test_factors_contain_contribution(self, trained_model, sample_data):
        X, _ = sample_data
        explainer = IndividualExplainer(trained_model, X)
        result = explainer.explain(X.iloc[0])
        for factor in result.top_positive_factors:
            assert factor.contribution is not None


class TestConfidenceAssessment:
    def test_high_confidence_for_extreme_probabilities(self, trained_model, sample_data):
        X, _ = sample_data
        explainer = IndividualExplainer(trained_model, X)
        for i in range(len(X)):
            result = explainer.explain(X.iloc[i])
            if result.churn_probability < 0.2 or result.churn_probability > 0.8:
                assert result.confidence == Confidence.HIGH
                break

    def test_low_confidence_for_middle_probabilities(self, trained_model, sample_data):
        X, _ = sample_data
        explainer = IndividualExplainer(trained_model, X)
        for i in range(len(X)):
            result = explainer.explain(X.iloc[i])
            if 0.4 < result.churn_probability < 0.6:
                assert result.confidence == Confidence.LOW
                break


class TestSimilarCustomers:
    def test_finds_similar_customers(self, trained_model, sample_data):
        X, y = sample_data
        explainer = IndividualExplainer(trained_model, X)
        similar = explainer.find_similar_customers(X.iloc[0], X, y, k=5)
        assert len(similar) == 5

    def test_similar_customers_contain_outcome(self, trained_model, sample_data):
        X, y = sample_data
        explainer = IndividualExplainer(trained_model, X)
        similar = explainer.find_similar_customers(X.iloc[0], X, y, k=5)
        for customer in similar:
            assert "outcome" in customer
            assert customer["outcome"] in [0, 1]

    def test_similar_customers_sorted_by_distance(self, trained_model, sample_data):
        X, y = sample_data
        explainer = IndividualExplainer(trained_model, X)
        similar = explainer.find_similar_customers(X.iloc[0], X, y, k=5)
        distances = [c["distance"] for c in similar]
        assert distances == sorted(distances)


class TestBatchExplanation:
    def test_explains_multiple_customers(self, trained_model, sample_data):
        X, _ = sample_data
        explainer = IndividualExplainer(trained_model, X)
        results = explainer.explain_batch(X.head(10))
        assert len(results) == 10
        for result in results:
            assert isinstance(result, IndividualExplanation)
