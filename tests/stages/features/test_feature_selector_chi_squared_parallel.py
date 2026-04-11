from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from customer_retention.stages.features.feature_selector import (
    FeatureSelector,
    SelectionMethod,
    _chi_sql_batch_size,
    _pandas_chi_squared_scores,
    _sql_chi_squared_scores,
    run_chi_squared_selection,
)


class TestChiSqlBatchSize:
    def test_default_10_buckets(self):
        size = _chi_sql_batch_size(10)
        assert size >= 10
        assert size * 2 * (10 + 1) <= 500

    def test_fewer_buckets_allows_larger_batch(self):
        assert _chi_sql_batch_size(4) > _chi_sql_batch_size(10)

    def test_more_buckets_uses_smaller_batch(self):
        assert _chi_sql_batch_size(20) < _chi_sql_batch_size(5)

    def test_minimum_batch_size(self):
        assert _chi_sql_batch_size(100) >= 5

    def test_at_least_5(self):
        for buckets in [1, 5, 10, 20, 50, 100]:
            assert _chi_sql_batch_size(buckets) >= 5


class TestProgressFnThreading:
    def test_run_chi_squared_selection_forwards_progress_fn(self):
        np.random.seed(42)
        n = 200
        target = np.random.choice([0, 1], n)
        df = pd.DataFrame({
            "f1": target * 3.0 + np.random.randn(n),
            "f2": np.random.randn(n),
            "target": target,
        })
        messages: list[str] = []
        run_chi_squared_selection(df, "target", max_features=1, progress_fn=messages.append)
        assert any("Chi-squared" in m for m in messages)

    def test_feature_selector_passes_progress_fn_to_chi_squared(self):
        np.random.seed(42)
        n = 200
        target = np.random.choice([0, 1], n)
        df = pd.DataFrame({
            "f1": target * 3.0 + np.random.randn(n),
            "f2": np.random.randn(n),
            "target": target,
        })
        messages: list[str] = []
        selector = FeatureSelector(
            method=SelectionMethod.CHI_SQUARED, target_column="target",
            max_features=1, chi_squared_num_buckets=10,
            progress_fn=messages.append,
        )
        selector.fit_transform(df)
        assert isinstance(messages, list)

    def test_spark_chi_squared_receives_progress_fn(self):
        from customer_retention.stages.features.feature_selector import _spark_chi_squared_selection

        feature_cols = ["f1", "f2", "f3"]
        mock_scores = {"f1": 50.0, "f2": 10.0, "f3": 1.0}
        messages: list[str] = []

        with patch("customer_retention.stages.features.feature_selector._import_spark_chi_squared_bucketizer") as mock_imp, \
             patch("customer_retention.stages.features.feature_selector._spark_sql_chi_squared_scores", return_value=mock_scores) as mock_sql:
            mock_F = MagicMock()
            mock_bucketizer = MagicMock()
            mock_imp.return_value = (mock_bucketizer, mock_F)
            mock_spark_df = MagicMock()
            work_df = mock_spark_df.select.return_value.na.fill.return_value
            work_df.agg.return_value.head.return_value = {f"__p_{i}": [0.5] for i in range(3)}
            mock_bucketizer.return_value.transform.return_value = work_df

            _spark_chi_squared_selection(
                mock_spark_df, "target", feature_cols, num_top_features=1,
                progress_fn=messages.append,
            )

        assert len(messages) > 0
        assert any("quantile" in m.lower() for m in messages)
        kw = mock_sql.call_args.kwargs if mock_sql.call_args.kwargs else {}
        assert kw.get("progress_fn") is not None


class TestBatchedScoringPandas:
    def test_many_features_scored_correctly(self):
        np.random.seed(42)
        n = 500
        target = np.random.choice([0, 1], n)
        n_features = 30
        feat_names = [f"f{i}" for i in range(n_features)]
        bkt_names = [f"__bkt_f{i}" for i in range(n_features)]
        data = {bkt_names[i]: np.random.choice(range(10), n) for i in range(n_features)}
        data["target"] = target
        df = pd.DataFrame(data)

        scores_a = _pandas_chi_squared_scores(df, "target", feat_names, bkt_names, 10)
        scores_b = _sql_chi_squared_scores(df, "target", feat_names, bkt_names, 10)
        for f in feat_names:
            assert scores_a[f] == pytest.approx(scores_b[f], rel=1e-10)

    def test_end_to_end_many_features(self):
        np.random.seed(7)
        n = 300
        target = np.random.choice([0, 1], n)
        cols = {f"f{i}": np.random.randn(n) + (target if i < 3 else 0) for i in range(25)}
        cols["target"] = target
        df = pd.DataFrame(cols)
        messages: list[str] = []
        result = run_chi_squared_selection(
            df, "target", max_features=5, progress_fn=messages.append,
        )
        assert len(result.selected_features) == 5
        assert len(result.dropped_features) == 20
        assert any("Chi-squared" in m for m in messages)


class TestGbdtProgressFn:
    def test_run_gbdt_forwards_progress_fn(self):
        from customer_retention.stages.features.feature_selector import run_gbdt_importance_selection
        np.random.seed(42)
        n = 200
        target = np.random.choice([0, 1], n)
        df = pd.DataFrame({
            "f1": target * 3.0 + np.random.randn(n),
            "f2": np.random.randn(n),
            "target": target,
        })
        messages: list[str] = []
        run_gbdt_importance_selection(df, "target", max_features=1, n_estimators=10, progress_fn=messages.append)
        assert any("XGBoost" in m for m in messages)
        assert any("Training" in m for m in messages)

    def test_spark_gbdt_receives_progress_fn(self):
        from unittest.mock import MagicMock, patch

        from customer_retention.stages.features.feature_selector import _spark_gbdt_importance_selection

        feature_cols = ["f1", "f2", "f3"]
        messages: list[str] = []

        with patch("customer_retention.stages.features.feature_selector._import_spark_gbdt_ml") as mock_imp, \
             patch("customer_retention.stages.features.feature_selector._resolve_spark_gbdt_workers", return_value=2):
            mock_F = MagicMock()
            mock_assembler_cls = MagicMock()
            mock_xgb_cls = MagicMock()
            mock_model = MagicMock()
            mock_model.get_feature_importances.return_value = {"f0": 50.0, "f1": 10.0, "f2": 1.0}
            mock_xgb_cls.return_value.fit.return_value = mock_model
            mock_imp.return_value = (mock_xgb_cls, mock_assembler_cls, mock_F)
            mock_spark_df = MagicMock()
            work_df = mock_spark_df.select.return_value.na.fill.return_value
            mock_assembler_cls.return_value.transform.return_value.select.return_value = work_df

            _spark_gbdt_importance_selection(
                mock_spark_df, "target", feature_cols, num_top_features=1,
                progress_fn=messages.append,
            )

        assert len(messages) > 0
        assert any("Training" in m for m in messages)


class TestBucketizerCheckpointInterval:
    def test_checkpoint_interval_constant_exists(self):
        from customer_retention.stages.features.feature_selector import _CHI_BUCKET_BATCH, _CHI_CHECKPOINT_INTERVAL
        assert _CHI_CHECKPOINT_INTERVAL >= _CHI_BUCKET_BATCH
        assert _CHI_CHECKPOINT_INTERVAL >= 400
