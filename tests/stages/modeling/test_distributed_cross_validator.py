from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from customer_retention.stages.modeling.cross_validator import (
    CrossValidator,
    CVResult,
    CVStrategy,
)


@pytest.fixture
def grouped_data():
    np.random.seed(42)
    n_entities = 50
    rows_per_entity = 10
    n = n_entities * rows_per_entity

    entity_ids = np.repeat(np.arange(n_entities), rows_per_entity)
    X = pd.DataFrame({
        "f1": np.random.randn(n),
        "f2": np.random.randn(n),
    })
    y = pd.Series(np.random.randint(0, 2, n), name="target")
    groups = pd.Series(entity_ids, name="entity_id")
    temporal = pd.Series(
        pd.date_range("2020-01-01", periods=n, freq="D"),
        name="as_of_date",
    )
    return X, y, groups, temporal


@pytest.fixture
def mock_distributed_model():
    model = MagicMock()
    model.clone.return_value = MagicMock()

    def mock_predict(X):
        return np.random.randint(0, 2, len(X))

    def mock_predict_proba(X):
        proba = np.random.rand(len(X), 2)
        proba = proba / proba.sum(axis=1, keepdims=True)
        return proba

    for m in [model, model.clone.return_value]:
        m.predict = mock_predict
        m.predict_proba = mock_predict_proba
        m.fit.return_value = m
        m.classes_ = np.array([0, 1])

    return model


class TestDistributedDetection:
    def test_pandas_data_uses_standard_path(self, grouped_data):
        X, y, groups, temporal = grouped_data
        cv = CrossValidator(
            strategy=CVStrategy.TEMPORAL_ENTITY,
            n_splits=3,
            scoring="roc_auc",
        )
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(max_iter=100)
        result = cv.run(model, X, y, groups=groups, temporal_values=temporal)
        assert isinstance(result, CVResult)
        assert len(result.cv_scores) == 3


class TestRunDistributed:
    def test_returns_cv_result(self, grouped_data, mock_distributed_model):
        X, y, groups, temporal = grouped_data
        cv = CrossValidator(
            strategy=CVStrategy.TEMPORAL_ENTITY,
            n_splits=3,
            scoring="roc_auc",
        )
        result = cv._run_distributed(mock_distributed_model, X, y, groups=groups, temporal_values=temporal)
        assert isinstance(result, CVResult)

    def test_correct_number_of_folds(self, grouped_data, mock_distributed_model):
        X, y, groups, temporal = grouped_data
        cv = CrossValidator(
            strategy=CVStrategy.TEMPORAL_ENTITY,
            n_splits=5,
            scoring="roc_auc",
        )
        result = cv._run_distributed(mock_distributed_model, X, y, groups=groups, temporal_values=temporal)
        assert len(result.cv_scores) == 5

    def test_fold_details_populated(self, grouped_data, mock_distributed_model):
        X, y, groups, temporal = grouped_data
        cv = CrossValidator(
            strategy=CVStrategy.TEMPORAL_ENTITY,
            n_splits=3,
            scoring="roc_auc",
        )
        result = cv._run_distributed(mock_distributed_model, X, y, groups=groups, temporal_values=temporal)
        assert len(result.fold_details) == 3
        for fold in result.fold_details:
            assert "fold" in fold
            assert "train_size" in fold
            assert "test_size" in fold

    def test_cv_mean_and_std(self, grouped_data, mock_distributed_model):
        X, y, groups, temporal = grouped_data
        cv = CrossValidator(
            strategy=CVStrategy.TEMPORAL_ENTITY,
            n_splits=3,
            scoring="roc_auc",
        )
        result = cv._run_distributed(mock_distributed_model, X, y, groups=groups, temporal_values=temporal)
        assert result.cv_mean == pytest.approx(np.mean(result.cv_scores), rel=1e-6)
        assert result.cv_std == pytest.approx(np.std(result.cv_scores), rel=1e-6)

    def test_clones_model_per_fold(self, grouped_data, mock_distributed_model):
        X, y, groups, temporal = grouped_data
        cv = CrossValidator(
            strategy=CVStrategy.TEMPORAL_ENTITY,
            n_splits=3,
            scoring="roc_auc",
        )
        cv._run_distributed(mock_distributed_model, X, y, groups=groups, temporal_values=temporal)
        assert mock_distributed_model.clone.call_count == 3

    def test_stability_flag(self, grouped_data, mock_distributed_model):
        X, y, groups, temporal = grouped_data
        cv = CrossValidator(
            strategy=CVStrategy.TEMPORAL_ENTITY,
            n_splits=3,
            scoring="roc_auc",
            stability_threshold=0.5,
        )
        result = cv._run_distributed(mock_distributed_model, X, y, groups=groups, temporal_values=temporal)
        assert isinstance(result.is_stable, bool)

    def test_scoring_stored(self, grouped_data, mock_distributed_model):
        X, y, groups, temporal = grouped_data
        cv = CrossValidator(
            strategy=CVStrategy.TEMPORAL_ENTITY,
            n_splits=3,
            scoring="average_precision",
        )
        result = cv._run_distributed(mock_distributed_model, X, y, groups=groups, temporal_values=temporal)
        assert result.scoring == "average_precision"

    def test_purge_gap_applied(self, grouped_data, mock_distributed_model):
        X, y, groups, temporal = grouped_data
        cv = CrossValidator(
            strategy=CVStrategy.TEMPORAL_ENTITY,
            n_splits=3,
            scoring="roc_auc",
            purge_gap_days=30,
        )
        result = cv._run_distributed(mock_distributed_model, X, y, groups=groups, temporal_values=temporal)
        assert isinstance(result, CVResult)
        for fold in result.fold_details:
            assert fold["train_size"] > 0

    def test_entity_aware_splits(self, grouped_data, mock_distributed_model):
        X, y, groups, temporal = grouped_data
        cv = CrossValidator(
            strategy=CVStrategy.TEMPORAL_ENTITY,
            n_splits=3,
            scoring="roc_auc",
        )
        result = cv._run_distributed(mock_distributed_model, X, y, groups=groups, temporal_values=temporal)
        for fold in result.fold_details:
            assert "train_entities" in fold
            assert "test_entities" in fold

    def test_no_temporal_values_still_works(self, grouped_data, mock_distributed_model):
        X, y, groups, _ = grouped_data
        cv = CrossValidator(
            strategy=CVStrategy.TEMPORAL_ENTITY,
            n_splits=3,
            scoring="roc_auc",
        )
        result = cv._run_distributed(mock_distributed_model, X, y, groups=groups, temporal_values=None)
        assert isinstance(result, CVResult)
        assert len(result.cv_scores) == 3


class TestDistributedNoCollect:
    def test_does_not_collect_features_to_pandas(self, grouped_data, mock_distributed_model):
        X, y, groups, temporal = grouped_data
        cv = CrossValidator(
            strategy=CVStrategy.TEMPORAL_ENTITY,
            n_splits=3,
            scoring="roc_auc",
        )
        original_to_pandas = __import__(
            "customer_retention.core.compat", fromlist=["to_pandas"]
        ).to_pandas

        def guarded_to_pandas(obj):
            if obj is X:
                raise AssertionError("to_pandas was called on X — must stay distributed")
            return original_to_pandas(obj)

        with patch(
            "customer_retention.stages.modeling.cross_validator.to_pandas",
            side_effect=guarded_to_pandas,
        ):
            result = cv._run_distributed(mock_distributed_model, X, y, groups=groups, temporal_values=temporal)
        assert isinstance(result, CVResult)
        assert len(result.cv_scores) == 3

    def test_does_not_collect_y_to_pandas(self, grouped_data, mock_distributed_model):
        X, y, groups, temporal = grouped_data
        cv = CrossValidator(
            strategy=CVStrategy.TEMPORAL_ENTITY,
            n_splits=3,
            scoring="roc_auc",
        )
        original_to_pandas = __import__(
            "customer_retention.core.compat", fromlist=["to_pandas"]
        ).to_pandas

        def guarded_to_pandas(obj):
            if obj is y:
                raise AssertionError("to_pandas was called on y — must stay distributed")
            return original_to_pandas(obj)

        with patch(
            "customer_retention.stages.modeling.cross_validator.to_pandas",
            side_effect=guarded_to_pandas,
        ):
            result = cv._run_distributed(mock_distributed_model, X, y, groups=groups, temporal_values=temporal)
        assert isinstance(result, CVResult)
        assert len(result.cv_scores) == 3

    def test_resets_index_for_spark_pandas(self, grouped_data, mock_distributed_model):
        X, y, groups, temporal = grouped_data
        X_mock = MagicMock(wraps=X)
        X_mock.spark = True  # Makes _is_spark_pandas return True
        X_mock.reset_index = MagicMock(return_value=X.reset_index(drop=True))
        X_mock.__len__ = lambda self: len(X)

        y_mock = MagicMock(wraps=y)
        y_mock.spark = True
        y_mock.reset_index = MagicMock(return_value=y.reset_index(drop=True))
        y_mock.__len__ = lambda self: len(y)

        cv = CrossValidator(
            strategy=CVStrategy.TEMPORAL_ENTITY,
            n_splits=3,
            scoring="roc_auc",
        )
        cv._run_distributed(mock_distributed_model, X_mock, y_mock, groups=groups, temporal_values=temporal)
        X_mock.reset_index.assert_called_once_with(drop=True)
        y_mock.reset_index.assert_called_once_with(drop=True)

    def test_backward_compatible_with_pandas(self, grouped_data, mock_distributed_model):
        X, y, groups, temporal = grouped_data
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        cv = CrossValidator(
            strategy=CVStrategy.TEMPORAL_ENTITY,
            n_splits=3,
            scoring="roc_auc",
        )
        result = cv._run_distributed(mock_distributed_model, X, y, groups=groups, temporal_values=temporal)
        assert isinstance(result, CVResult)
        assert len(result.cv_scores) == 3
        for fold in result.fold_details:
            assert fold["train_size"] > 0
            assert fold["test_size"] > 0
            assert "train_entities" in fold
            assert "test_entities" in fold


class TestDistributedGroupsAsDataFrame:
    def test_run_distributed_with_dataframe_groups(self, grouped_data, mock_distributed_model):
        X, y, groups, temporal = grouped_data
        groups_df = groups.to_frame()
        cv = CrossValidator(strategy=CVStrategy.TEMPORAL_ENTITY, n_splits=3, scoring="roc_auc")
        result = cv._run_distributed(mock_distributed_model, X, y, groups=groups_df, temporal_values=temporal)
        assert isinstance(result, CVResult)
        assert len(result.cv_scores) == 3

    def test_run_distributed_with_dataframe_temporal(self, grouped_data, mock_distributed_model):
        X, y, groups, temporal = grouped_data
        temporal_df = temporal.to_frame()
        cv = CrossValidator(
            strategy=CVStrategy.TEMPORAL_ENTITY, n_splits=3,
            scoring="roc_auc", purge_gap_days=30,
        )
        result = cv._run_distributed(mock_distributed_model, X, y, groups=groups, temporal_values=temporal_df)
        assert isinstance(result, CVResult)
        assert len(result.cv_scores) == 3
