import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from customer_retention.analysis.interpretability import InteractionResult, PDPGenerator, PDPResult


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


class TestPDPGeneration:
    def test_generates_pdp_for_single_feature(self, trained_model, sample_data):
        X, _ = sample_data
        generator = PDPGenerator(trained_model)
        result = generator.generate(X, feature="recency")
        assert isinstance(result, PDPResult)
        assert result.feature_name == "recency"
        assert len(result.grid_values) > 0
        assert len(result.pdp_values) > 0

    def test_pdp_grid_resolution_configurable(self, trained_model, sample_data):
        X, _ = sample_data
        generator = PDPGenerator(trained_model)
        result = generator.generate(X, feature="recency", grid_resolution=20)
        assert len(result.grid_values) == 20

    def test_pdp_values_within_probability_range(self, trained_model, sample_data):
        X, _ = sample_data
        generator = PDPGenerator(trained_model)
        result = generator.generate(X, feature="recency")
        assert all(0 <= v <= 1 for v in result.pdp_values)


class TestICEPlots:
    def test_ice_lines_generated(self, trained_model, sample_data):
        X, _ = sample_data
        generator = PDPGenerator(trained_model)
        result = generator.generate(X, feature="recency", include_ice=True, ice_lines=50)
        assert result.ice_values is not None
        assert len(result.ice_values) == 50

    def test_ice_values_shape_matches_grid(self, trained_model, sample_data):
        X, _ = sample_data
        generator = PDPGenerator(trained_model)
        result = generator.generate(X, feature="recency", include_ice=True, ice_lines=10, grid_resolution=25)
        for ice_line in result.ice_values:
            assert len(ice_line) == 25


class TestMultipleFeaturePDP:
    def test_generates_pdp_for_multiple_features(self, trained_model, sample_data):
        X, _ = sample_data
        generator = PDPGenerator(trained_model)
        results = generator.generate_multiple(X, features=["recency", "frequency", "monetary"])
        assert len(results) == 3
        assert all(isinstance(r, PDPResult) for r in results)

    def test_top_features_auto_selection(self, trained_model, sample_data):
        X, _ = sample_data
        generator = PDPGenerator(trained_model)
        results = generator.generate_top_features(X, n_features=3)
        assert len(results) == 3


class TestFeatureInteraction:
    def test_2d_pdp_generated(self, trained_model, sample_data):
        X, _ = sample_data
        generator = PDPGenerator(trained_model)
        result = generator.generate_interaction(X, feature1="recency", feature2="frequency")
        assert isinstance(result, InteractionResult)
        assert result.feature1_name == "recency"
        assert result.feature2_name == "frequency"
        assert result.pdp_matrix is not None

    def test_interaction_matrix_shape(self, trained_model, sample_data):
        X, _ = sample_data
        generator = PDPGenerator(trained_model)
        result = generator.generate_interaction(X, feature1="recency", feature2="frequency", grid_resolution=10)
        assert result.pdp_matrix.shape == (10, 10)


class TestPDPMetadata:
    def test_pdp_contains_feature_range(self, trained_model, sample_data):
        X, _ = sample_data
        generator = PDPGenerator(trained_model)
        result = generator.generate(X, feature="recency")
        assert result.feature_min is not None
        assert result.feature_max is not None
        assert result.feature_min < result.feature_max

    def test_pdp_contains_average_prediction(self, trained_model, sample_data):
        X, _ = sample_data
        generator = PDPGenerator(trained_model)
        result = generator.generate(X, feature="recency")
        assert result.average_prediction is not None
        assert 0 <= result.average_prediction <= 1
