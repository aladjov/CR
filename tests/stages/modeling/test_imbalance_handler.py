import numpy as np
import pandas as pd
import pytest

from customer_retention.stages.modeling import (
    ClassWeightMethod,
    ImbalanceHandler,
    ImbalanceRecommender,
    ImbalanceStrategy,
)


@pytest.fixture
def imbalanced_data():
    np.random.seed(42)
    n_majority = 800
    n_minority = 200
    X = pd.DataFrame({
        "feature1": np.concatenate([np.random.randn(n_majority), np.random.randn(n_minority) + 2]),
        "feature2": np.concatenate([np.random.randn(n_majority), np.random.randn(n_minority) - 1]),
    })
    y = pd.Series([0] * n_majority + [1] * n_minority)
    return X, y


class TestImbalanceStrategy:
    def test_strategy_enum_has_required_values(self):
        assert hasattr(ImbalanceStrategy, "CLASS_WEIGHT")
        assert hasattr(ImbalanceStrategy, "SMOTE")
        assert hasattr(ImbalanceStrategy, "RANDOM_OVERSAMPLE")
        assert hasattr(ImbalanceStrategy, "RANDOM_UNDERSAMPLE")
        assert hasattr(ImbalanceStrategy, "SMOTEENN")
        assert hasattr(ImbalanceStrategy, "ADASYN")
        assert hasattr(ImbalanceStrategy, "NONE")


class TestClassWeightMethod:
    def test_weight_method_enum_has_required_values(self):
        assert hasattr(ClassWeightMethod, "BALANCED")
        assert hasattr(ClassWeightMethod, "CUSTOM")
        assert hasattr(ClassWeightMethod, "INVERSE")


class TestClassWeightCalculation:
    def test_balanced_weights_computation(self, imbalanced_data):
        X, y = imbalanced_data
        handler = ImbalanceHandler(strategy=ImbalanceStrategy.CLASS_WEIGHT, weight_method=ClassWeightMethod.BALANCED)
        result = handler.fit(X, y)

        assert result.class_weights is not None
        assert 0 in result.class_weights
        assert 1 in result.class_weights
        assert result.class_weights[1] > result.class_weights[0]

    def test_inverse_weights_computation(self, imbalanced_data):
        X, y = imbalanced_data
        handler = ImbalanceHandler(strategy=ImbalanceStrategy.CLASS_WEIGHT, weight_method=ClassWeightMethod.INVERSE)
        result = handler.fit(X, y)

        minority_ratio = y.sum() / len(y)
        majority_ratio = 1 - minority_ratio
        assert result.class_weights[1] > result.class_weights[0]

    def test_custom_weights(self, imbalanced_data):
        X, y = imbalanced_data
        custom = {0: 1.0, 1: 5.0}
        handler = ImbalanceHandler(
            strategy=ImbalanceStrategy.CLASS_WEIGHT,
            weight_method=ClassWeightMethod.CUSTOM,
            custom_weights=custom
        )
        result = handler.fit(X, y)

        assert result.class_weights == custom


class TestSMOTE:
    def test_smote_increases_minority_samples(self, imbalanced_data):
        X, y = imbalanced_data
        handler = ImbalanceHandler(strategy=ImbalanceStrategy.SMOTE, random_state=42)
        result = handler.fit_transform(X, y)

        original_minority = (y == 1).sum()
        resampled_minority = (result.y_resampled == 1).sum()
        assert resampled_minority > original_minority

    def test_smote_balances_classes(self, imbalanced_data):
        X, y = imbalanced_data
        handler = ImbalanceHandler(strategy=ImbalanceStrategy.SMOTE, random_state=42)
        result = handler.fit_transform(X, y)

        class_counts = result.y_resampled.value_counts()
        assert abs(class_counts[0] - class_counts[1]) < 10


class TestRandomOversampling:
    def test_oversampling_increases_minority(self, imbalanced_data):
        X, y = imbalanced_data
        handler = ImbalanceHandler(strategy=ImbalanceStrategy.RANDOM_OVERSAMPLE, random_state=42)
        result = handler.fit_transform(X, y)

        original_minority = (y == 1).sum()
        resampled_minority = (result.y_resampled == 1).sum()
        assert resampled_minority > original_minority


class TestRandomUndersampling:
    def test_undersampling_reduces_majority(self, imbalanced_data):
        X, y = imbalanced_data
        handler = ImbalanceHandler(strategy=ImbalanceStrategy.RANDOM_UNDERSAMPLE, random_state=42)
        result = handler.fit_transform(X, y)

        original_majority = (y == 0).sum()
        resampled_majority = (result.y_resampled == 0).sum()
        assert resampled_majority < original_majority


class TestSMOTEENN:
    def test_smoteenn_balances_and_cleans(self, imbalanced_data):
        X, y = imbalanced_data
        handler = ImbalanceHandler(strategy=ImbalanceStrategy.SMOTEENN, random_state=42)
        result = handler.fit_transform(X, y)

        assert result.X_resampled is not None
        assert result.y_resampled is not None
        assert result.strategy_used == ImbalanceStrategy.SMOTEENN


class TestADASYN:
    def test_adasyn_increases_minority(self, imbalanced_data):
        X, y = imbalanced_data
        handler = ImbalanceHandler(strategy=ImbalanceStrategy.ADASYN, random_state=42)
        result = handler.fit_transform(X, y)

        original_minority = (y == 1).sum()
        resampled_minority = (result.y_resampled == 1).sum()
        assert resampled_minority > original_minority


class TestNoResampling:
    def test_none_strategy_returns_unchanged(self, imbalanced_data):
        X, y = imbalanced_data
        handler = ImbalanceHandler(strategy=ImbalanceStrategy.NONE)
        result = handler.fit_transform(X, y)

        pd.testing.assert_frame_equal(result.X_resampled, X)
        pd.testing.assert_series_equal(result.y_resampled, y)


class TestImbalanceResult:
    def test_result_contains_required_fields(self, imbalanced_data):
        X, y = imbalanced_data
        handler = ImbalanceHandler(strategy=ImbalanceStrategy.SMOTE, random_state=42)
        result = handler.fit_transform(X, y)

        assert hasattr(result, "X_resampled")
        assert hasattr(result, "y_resampled")
        assert hasattr(result, "strategy_used")
        assert hasattr(result, "original_class_counts")
        assert hasattr(result, "resampled_class_counts")

    def test_result_tracks_class_counts(self, imbalanced_data):
        X, y = imbalanced_data
        handler = ImbalanceHandler(strategy=ImbalanceStrategy.SMOTE, random_state=42)
        result = handler.fit_transform(X, y)

        assert result.original_class_counts[0] == 800
        assert result.original_class_counts[1] == 200


class TestImbalanceRatioDetection:
    def test_detects_imbalance_ratio(self, imbalanced_data):
        X, y = imbalanced_data
        handler = ImbalanceHandler(strategy=ImbalanceStrategy.SMOTE)
        result = handler.fit(X, y)

        assert hasattr(result, "imbalance_ratio")
        assert result.imbalance_ratio == pytest.approx(4.0, rel=0.01)

    def test_recommends_strategy_for_moderate_imbalance(self):
        np.random.seed(42)
        X = pd.DataFrame({"f1": np.random.randn(1000)})
        y = pd.Series([0] * 750 + [1] * 250)

        handler = ImbalanceHandler(strategy=ImbalanceStrategy.CLASS_WEIGHT)
        result = handler.fit(X, y)

        assert result.imbalance_ratio == pytest.approx(3.0, rel=0.01)


class TestSamplingStrategyParameter:
    def test_custom_sampling_strategy(self, imbalanced_data):
        X, y = imbalanced_data
        handler = ImbalanceHandler(
            strategy=ImbalanceStrategy.SMOTE,
            sampling_strategy=0.5,
            random_state=42
        )
        result = handler.fit_transform(X, y)

        minority_count = (result.y_resampled == 1).sum()
        majority_count = (result.y_resampled == 0).sum()
        assert minority_count / majority_count >= 0.45


class TestFitTransformSeparation:
    def test_fit_only_computes_weights(self, imbalanced_data):
        X, y = imbalanced_data
        handler = ImbalanceHandler(strategy=ImbalanceStrategy.CLASS_WEIGHT)
        result = handler.fit(X, y)

        assert result.class_weights is not None
        assert result.X_resampled is None
        assert result.y_resampled is None

    def test_fit_transform_resamples_data(self, imbalanced_data):
        X, y = imbalanced_data
        handler = ImbalanceHandler(strategy=ImbalanceStrategy.SMOTE, random_state=42)
        result = handler.fit_transform(X, y)

        assert result.X_resampled is not None
        assert result.y_resampled is not None


class TestImbalanceRecommender:
    def test_low_imbalance_recommends_class_weights(self):
        y = pd.Series([0] * 600 + [1] * 400)  # 1.5:1 ratio
        recommender = ImbalanceRecommender()
        rec = recommender.recommend(y)

        assert rec.severity == "low"
        assert rec.primary_strategy == ImbalanceStrategy.CLASS_WEIGHT

    def test_moderate_imbalance_recommends_class_weights_or_smote(self):
        y = pd.Series([0] * 800 + [1] * 200)  # 4:1 ratio
        recommender = ImbalanceRecommender()
        rec = recommender.recommend(y)

        assert rec.severity == "moderate"
        assert rec.primary_strategy == ImbalanceStrategy.CLASS_WEIGHT
        assert any(s[0] == ImbalanceStrategy.SMOTE for s in rec.strategies)

    def test_high_imbalance_recommends_smote(self):
        y = pd.Series([0] * 950 + [1] * 50)  # 19:1 ratio
        recommender = ImbalanceRecommender()
        rec = recommender.recommend(y)

        assert rec.severity == "high"
        assert rec.primary_strategy == ImbalanceStrategy.SMOTE

    def test_severe_imbalance_large_dataset_recommends_undersample(self):
        y = pd.Series([0] * 100000 + [1] * 3000)  # 33:1 ratio, large dataset
        recommender = ImbalanceRecommender()
        rec = recommender.recommend(y, n_samples=103000)

        assert rec.severity == "severe"
        assert rec.primary_strategy == ImbalanceStrategy.RANDOM_UNDERSAMPLE

    def test_severe_imbalance_small_minority_recommends_oversample(self):
        y = pd.Series([0] * 995 + [1] * 5)  # 199:1 ratio, tiny minority
        recommender = ImbalanceRecommender()
        rec = recommender.recommend(y)

        assert rec.severity == "severe"
        assert rec.primary_strategy == ImbalanceStrategy.RANDOM_OVERSAMPLE

    def test_recommendation_includes_ratio(self):
        y = pd.Series([0] * 900 + [1] * 100)  # 9:1 ratio
        recommender = ImbalanceRecommender()
        rec = recommender.recommend(y)

        assert rec.ratio == pytest.approx(9.0, rel=0.01)

    def test_recommendation_includes_strategies_list(self):
        y = pd.Series([0] * 900 + [1] * 100)
        recommender = ImbalanceRecommender()
        rec = recommender.recommend(y)

        assert len(rec.strategies) >= 1
        assert all(isinstance(s[0], ImbalanceStrategy) for s in rec.strategies)

    def test_recommendation_includes_explanation(self):
        y = pd.Series([0] * 900 + [1] * 100)
        recommender = ImbalanceRecommender()
        rec = recommender.recommend(y)

        assert rec.explanation is not None
        assert len(rec.explanation) > 0

    def test_high_imbalance_small_minority_recommends_oversample(self):
        y = pd.Series([0] * 95 + [1] * 5)  # 19:1 ratio but only 5 minority samples
        recommender = ImbalanceRecommender()
        rec = recommender.recommend(y)

        assert rec.severity == "high"
        assert rec.primary_strategy == ImbalanceStrategy.RANDOM_OVERSAMPLE

    def test_severe_imbalance_default_recommends_smote(self):
        y = pd.Series([0] * 970 + [1] * 30)  # 32:1 ratio, enough minority for SMOTE
        recommender = ImbalanceRecommender()
        rec = recommender.recommend(y, n_samples=1000)

        assert rec.severity == "severe"
        assert rec.primary_strategy == ImbalanceStrategy.SMOTE

    def test_print_recommendation(self, capsys):
        y = pd.Series([0] * 900 + [1] * 100)
        recommender = ImbalanceRecommender()
        rec = recommender.recommend(y)
        rec.print_recommendation()

        captured = capsys.readouterr()
        assert "Class Imbalance" in captured.out
        assert "Recommended strategies" in captured.out
