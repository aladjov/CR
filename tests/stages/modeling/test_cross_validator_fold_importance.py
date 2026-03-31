import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from customer_retention.stages.modeling import CrossValidator, CVStrategy


@pytest.fixture
def binary_data():
    np.random.seed(42)
    n = 300
    X = pd.DataFrame({
        "feat_a": np.random.randn(n),
        "feat_b": np.random.randn(n),
        "feat_c": np.random.randn(n),
    })
    y = pd.Series(np.random.choice([0, 1], n, p=[0.4, 0.6]))
    return X, y


@pytest.fixture
def feature_names():
    return ["feat_a", "feat_b", "feat_c"]


class TestFoldImportanceCaptureRF:
    def test_fold_details_contain_importance(self, binary_data, feature_names):
        X, y = binary_data
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        cv = CrossValidator(strategy=CVStrategy.STRATIFIED_KFOLD, n_splits=3)
        result = cv.run(model, X, y, capture_fold_importance=True, feature_names=feature_names)

        for detail in result.fold_details:
            assert "feature_importance" in detail
            assert isinstance(detail["feature_importance"], dict)

    def test_importance_keys_match_feature_names(self, binary_data, feature_names):
        X, y = binary_data
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        cv = CrossValidator(strategy=CVStrategy.STRATIFIED_KFOLD, n_splits=3)
        result = cv.run(model, X, y, capture_fold_importance=True, feature_names=feature_names)

        for detail in result.fold_details:
            assert set(detail["feature_importance"].keys()) == set(feature_names)

    def test_importance_values_are_nonnegative(self, binary_data, feature_names):
        X, y = binary_data
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        cv = CrossValidator(strategy=CVStrategy.STRATIFIED_KFOLD, n_splits=3)
        result = cv.run(model, X, y, capture_fold_importance=True, feature_names=feature_names)

        for detail in result.fold_details:
            for v in detail["feature_importance"].values():
                assert v >= 0.0

    def test_importance_values_sum_to_one(self, binary_data, feature_names):
        X, y = binary_data
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        cv = CrossValidator(strategy=CVStrategy.STRATIFIED_KFOLD, n_splits=3)
        result = cv.run(model, X, y, capture_fold_importance=True, feature_names=feature_names)

        for detail in result.fold_details:
            total = sum(detail["feature_importance"].values())
            assert abs(total - 1.0) < 0.01


class TestFoldImportanceCaptureLR:
    def test_lr_uses_abs_coef(self, binary_data, feature_names):
        X, y = binary_data
        model = LogisticRegression(max_iter=1000, random_state=42)
        cv = CrossValidator(strategy=CVStrategy.STRATIFIED_KFOLD, n_splits=3)
        result = cv.run(model, X, y, capture_fold_importance=True, feature_names=feature_names)

        for detail in result.fold_details:
            assert "feature_importance" in detail
            for v in detail["feature_importance"].values():
                assert v >= 0.0


class TestFoldImportanceDefault:
    def test_not_captured_by_default(self, binary_data):
        X, y = binary_data
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        cv = CrossValidator(strategy=CVStrategy.STRATIFIED_KFOLD, n_splits=3)
        result = cv.run(model, X, y)

        for detail in result.fold_details:
            assert "feature_importance" not in detail

    def test_not_captured_when_false(self, binary_data, feature_names):
        X, y = binary_data
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        cv = CrossValidator(strategy=CVStrategy.STRATIFIED_KFOLD, n_splits=3)
        result = cv.run(model, X, y, capture_fold_importance=False, feature_names=feature_names)

        for detail in result.fold_details:
            assert "feature_importance" not in detail

    def test_no_feature_names_skips_capture(self, binary_data):
        X, y = binary_data
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        cv = CrossValidator(strategy=CVStrategy.STRATIFIED_KFOLD, n_splits=3)
        result = cv.run(model, X, y, capture_fold_importance=True, feature_names=None)

        for detail in result.fold_details:
            assert "feature_importance" not in detail


class TestFoldImportanceTemporalEntity:
    def test_temporal_entity_captures_importance(self, binary_data, feature_names):
        X, y = binary_data
        n = len(X)
        groups = pd.Series(np.arange(n) % 50)
        temporal = pd.Series(pd.date_range("2024-01-01", periods=n, freq="h"))

        model = RandomForestClassifier(n_estimators=10, random_state=42)
        cv = CrossValidator(strategy=CVStrategy.TEMPORAL_ENTITY, n_splits=3)
        result = cv.run(
            model, X, y, groups=groups, temporal_values=temporal,
            capture_fold_importance=True, feature_names=feature_names,
        )

        for detail in result.fold_details:
            assert "feature_importance" in detail
            assert set(detail["feature_importance"].keys()) == set(feature_names)


class TestFoldImportanceWithCallback:
    def test_importance_and_callback_coexist(self, binary_data, feature_names):
        X, y = binary_data
        callback_calls = []

        def on_fold(detail, fold_num, total):
            callback_calls.append(fold_num)

        model = RandomForestClassifier(n_estimators=10, random_state=42)
        cv = CrossValidator(strategy=CVStrategy.STRATIFIED_KFOLD, n_splits=3)
        result = cv.run(
            model, X, y, on_fold_complete=on_fold,
            capture_fold_importance=True, feature_names=feature_names,
        )

        assert len(callback_calls) == 3
        for detail in result.fold_details:
            assert "feature_importance" in detail


class TestExtractImportance:
    def test_extracts_from_feature_importances(self):
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        X = pd.DataFrame({"a": [1, 2, 3, 4], "b": [4, 3, 2, 1]})
        y = pd.Series([0, 1, 0, 1])
        model.fit(X, y)

        result = CrossValidator._extract_importance(model, ["a", "b"])
        assert result is not None
        assert set(result.keys()) == {"a", "b"}

    def test_extracts_from_coef(self):
        model = LogisticRegression(max_iter=1000, random_state=42)
        X = pd.DataFrame({"a": [1, 2, 3, 4], "b": [4, 3, 2, 1]})
        y = pd.Series([0, 1, 0, 1])
        model.fit(X, y)

        result = CrossValidator._extract_importance(model, ["a", "b"])
        assert result is not None
        assert all(v >= 0 for v in result.values())

    def test_returns_none_for_unknown_model(self):

        class DummyModel:
            pass

        result = CrossValidator._extract_importance(DummyModel(), ["a", "b"])
        assert result is None
