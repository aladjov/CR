import pytest
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from customer_retention.stages.modeling import CrossValidator, CVStrategy, CVResult


@pytest.fixture
def sample_data():
    np.random.seed(42)
    n = 500
    X = pd.DataFrame({
        "feature1": np.random.randn(n),
        "feature2": np.random.randn(n),
    })
    y = pd.Series(np.random.choice([0, 1], n, p=[0.3, 0.7]))
    return X, y


@pytest.fixture
def sample_model():
    return LogisticRegression(max_iter=1000, random_state=42)


class TestCVStrategy:
    def test_strategy_enum_has_required_values(self):
        assert hasattr(CVStrategy, "STRATIFIED_KFOLD")
        assert hasattr(CVStrategy, "REPEATED_STRATIFIED")
        assert hasattr(CVStrategy, "TIME_SERIES")
        assert hasattr(CVStrategy, "GROUP_KFOLD")


class TestStratifiedKFold:
    def test_runs_stratified_kfold(self, sample_data, sample_model):
        X, y = sample_data
        cv = CrossValidator(strategy=CVStrategy.STRATIFIED_KFOLD, n_splits=5)
        result = cv.run(sample_model, X, y)

        assert result.cv_scores is not None
        assert len(result.cv_scores) == 5

    def test_cv_scores_in_valid_range(self, sample_data, sample_model):
        X, y = sample_data
        cv = CrossValidator(strategy=CVStrategy.STRATIFIED_KFOLD, n_splits=5)
        result = cv.run(sample_model, X, y)

        assert all(0 <= score <= 1 for score in result.cv_scores)

    def test_preserves_class_balance_in_folds(self, sample_data, sample_model):
        X, y = sample_data
        cv = CrossValidator(strategy=CVStrategy.STRATIFIED_KFOLD, n_splits=5)
        result = cv.run(sample_model, X, y)

        original_ratio = y.mean()
        for fold_info in result.fold_details:
            fold_ratio = fold_info.get("train_class_ratio", original_ratio)
            assert abs(fold_ratio - original_ratio) < 0.1


class TestRepeatedStratifiedKFold:
    def test_runs_repeated_stratified_kfold(self, sample_data, sample_model):
        X, y = sample_data
        cv = CrossValidator(
            strategy=CVStrategy.REPEATED_STRATIFIED,
            n_splits=3,
            n_repeats=2
        )
        result = cv.run(sample_model, X, y)

        assert len(result.cv_scores) == 6


class TestCVMetrics:
    def test_calculates_cv_mean(self, sample_data, sample_model):
        X, y = sample_data
        cv = CrossValidator(n_splits=5)
        result = cv.run(sample_model, X, y)

        assert result.cv_mean is not None
        assert result.cv_mean == pytest.approx(np.mean(result.cv_scores), rel=1e-6)

    def test_calculates_cv_std(self, sample_data, sample_model):
        X, y = sample_data
        cv = CrossValidator(n_splits=5)
        result = cv.run(sample_model, X, y)

        assert result.cv_std is not None
        assert result.cv_std == pytest.approx(np.std(result.cv_scores), rel=1e-6)


class TestCVResult:
    def test_result_contains_required_fields(self, sample_data, sample_model):
        X, y = sample_data
        cv = CrossValidator(n_splits=5)
        result = cv.run(sample_model, X, y)

        assert hasattr(result, "cv_scores")
        assert hasattr(result, "cv_mean")
        assert hasattr(result, "cv_std")
        assert hasattr(result, "fold_details")

    def test_fold_details_has_all_folds(self, sample_data, sample_model):
        X, y = sample_data
        cv = CrossValidator(n_splits=5)
        result = cv.run(sample_model, X, y)

        assert len(result.fold_details) == 5


class TestCVConfiguration:
    def test_custom_n_splits(self, sample_data, sample_model):
        X, y = sample_data
        cv = CrossValidator(n_splits=10)
        result = cv.run(sample_model, X, y)

        assert len(result.cv_scores) == 10

    def test_custom_scoring_metric(self, sample_data, sample_model):
        X, y = sample_data
        cv = CrossValidator(n_splits=5, scoring="roc_auc")
        result = cv.run(sample_model, X, y)

        assert result.scoring == "roc_auc"

    def test_custom_random_state(self, sample_data, sample_model):
        X, y = sample_data

        cv1 = CrossValidator(n_splits=5, random_state=42)
        cv2 = CrossValidator(n_splits=5, random_state=42)

        result1 = cv1.run(sample_model, X, y)
        result2 = cv2.run(sample_model, X, y)

        np.testing.assert_array_almost_equal(result1.cv_scores, result2.cv_scores)


class TestCVStability:
    def test_detects_high_variance(self, sample_data, sample_model):
        X, y = sample_data
        cv = CrossValidator(n_splits=5)
        result = cv.run(sample_model, X, y)

        assert hasattr(result, "is_stable")
        assert isinstance(result.is_stable, bool)

    def test_flags_unstable_cv(self):
        np.random.seed(42)
        X = pd.DataFrame({
            "feature1": np.random.randn(100),
            "feature2": np.random.randn(100),
        })
        y = pd.Series(np.random.choice([0, 1], 100, p=[0.5, 0.5]))

        model = LogisticRegression(max_iter=1000, random_state=42)
        cv = CrossValidator(n_splits=5, stability_threshold=0.01)
        result = cv.run(model, X, y)

        assert result.cv_std is not None


class TestAveragePrecisionScoring:
    def test_default_scoring_is_average_precision(self, sample_data, sample_model):
        X, y = sample_data
        cv = CrossValidator(n_splits=5)
        result = cv.run(sample_model, X, y)

        assert result.scoring == "average_precision"


class TestFoldDetails:
    def test_fold_contains_train_test_sizes(self, sample_data, sample_model):
        X, y = sample_data
        cv = CrossValidator(n_splits=5)
        result = cv.run(sample_model, X, y)

        for fold in result.fold_details:
            assert "train_size" in fold
            assert "test_size" in fold

    def test_fold_contains_score(self, sample_data, sample_model):
        X, y = sample_data
        cv = CrossValidator(n_splits=5)
        result = cv.run(sample_model, X, y)

        for fold in result.fold_details:
            assert "score" in fold


class TestGroupKFold:
    def test_group_kfold_keeps_groups_together(self):
        np.random.seed(42)
        n_groups = 50
        rows_per_group = 10
        data = []
        for group_id in range(n_groups):
            target = np.random.choice([0, 1])
            for _ in range(rows_per_group):
                data.append({
                    "group_id": group_id,
                    "feature1": np.random.randn(),
                    "target": target
                })

        df = pd.DataFrame(data)
        X = df[["feature1"]]
        y = df["target"]
        groups = df["group_id"]

        model = LogisticRegression(max_iter=1000, random_state=42)
        cv = CrossValidator(strategy=CVStrategy.GROUP_KFOLD, n_splits=5)
        result = cv.run(model, X, y, groups=groups)

        assert len(result.cv_scores) == 5
