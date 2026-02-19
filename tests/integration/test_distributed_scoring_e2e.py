from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from customer_retention.stages.scoring.config import ScoringConfig
from customer_retention.stages.scoring.data_loader import ScoringDataLoader


@pytest.fixture
def gold_features_df():
    np.random.seed(42)
    n_train, n_holdout = 80, 20
    n = n_train + n_holdout
    df = pd.DataFrame({
        "customer_id": [f"c{i}" for i in range(n)],
        "event_timestamp": pd.date_range("2024-01-01", periods=n),
        "feature_a": np.random.randn(n),
        "feature_b": np.random.randn(n) * 10 + 50,
        "feature_c": np.random.randint(0, 100, n).astype(float),
        "unsubscribed": np.concatenate([
            np.random.randint(0, 2, n_train).astype(float),
            [np.nan] * n_holdout,
        ]),
        "original_unsubscribed": np.concatenate([
            [np.nan] * n_train,
            np.random.randint(0, 2, n_holdout).astype(float),
        ]),
    })
    return df


@pytest.fixture
def scoring_config(tmp_path):
    return ScoringConfig(
        pipeline_name="test_pipeline",
        composite_name="test__abc1234",
        target_column="unsubscribed",
        entity_key="customer_id",
        timestamp_column="event_timestamp",
        recommendations_hash="hash123",
        experiments_dir=tmp_path / "experiments",
        artifacts_path=tmp_path / "artifacts",
        mlflow_tracking_uri="sqlite:///mlruns.db",
        production_dir=tmp_path / "production",
        pipeline_dir=tmp_path,
    )


@pytest.fixture
def databricks_scoring_config(tmp_path):
    return ScoringConfig(
        pipeline_name="test_pipeline",
        composite_name="test__abc1234",
        target_column="unsubscribed",
        entity_key="customer_id",
        timestamp_column="event_timestamp",
        recommendations_hash="hash123",
        experiments_dir=tmp_path / "experiments",
        artifacts_path=tmp_path / "artifacts",
        mlflow_tracking_uri="databricks",
        production_dir=tmp_path / "production",
        catalog="analytics",
        schema="churn",
    )


class TestDistributedScoringPipelineLocal:
    def test_distributed_load_filter_predict(self, scoring_config, gold_features_df, tmp_path):
        from sklearn.ensemble import RandomForestClassifier

        from customer_retention.integrations.adapters.factory import get_delta

        cn = scoring_config.composite_name
        gold_dir = tmp_path / "production" / "data" / "gold" / f"gold_features_{cn}"
        gold_dir.parent.mkdir(parents=True)
        storage = get_delta(force_local=True)
        storage.write(gold_features_df, str(gold_dir))

        loader = ScoringDataLoader(scoring_config)
        features_df = loader.load_gold_features_distributed()

        assert len(features_df) == len(gold_features_df)
        assert "original_unsubscribed" in features_df.columns

        scoring_mask = (
            features_df["unsubscribed"].isna()
            & features_df["original_unsubscribed"].notna()
        )
        scoring_subset = features_df[scoring_mask].copy()
        assert len(scoring_subset) == 20

        X = scoring_subset[["feature_a", "feature_b", "feature_c"]].fillna(0)
        y_true = scoring_subset["original_unsubscribed"].to_numpy()

        train_mask = features_df["original_unsubscribed"].isna()
        train_df = features_df[train_mask]
        X_train = train_df[["feature_a", "feature_b", "feature_c"]].fillna(0)
        y_train = train_df["unsubscribed"].to_numpy()

        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)

        y_proba = model.predict_proba(X)[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)

        assert len(y_pred) == 20
        assert all(p in [0, 1] for p in y_pred)
        assert all(0 <= p <= 1 for p in y_proba)

    def test_distributed_matches_local_output(self, scoring_config, gold_features_df, tmp_path):
        from customer_retention.integrations.adapters.factory import get_delta

        cn = scoring_config.composite_name
        gold_dir = tmp_path / "production" / "data" / "gold" / f"gold_features_{cn}"
        gold_dir.parent.mkdir(parents=True)
        storage = get_delta(force_local=True)
        storage.write(gold_features_df, str(gold_dir))

        loader = ScoringDataLoader(scoring_config)
        local_result = loader.load_gold_features()
        distributed_result = loader.load_gold_features_distributed()

        pd.testing.assert_frame_equal(local_result, distributed_result)


class TestDistributedScoringPipelineDatabricks:
    def test_distributed_load_returns_pyspark_pandas(self, databricks_scoring_config, gold_features_df):
        mock_spark = MagicMock()
        mock_psdf = MagicMock()
        mock_psdf.columns = gold_features_df.columns.tolist()
        mock_spark.table.return_value.pandas_api.return_value = mock_psdf

        cn = databricks_scoring_config.composite_name
        with patch(
            "customer_retention.stages.scoring.data_loader.get_spark_session",
            return_value=mock_spark,
        ):
            loader = ScoringDataLoader(databricks_scoring_config)
            result = loader.load_gold_features_distributed()

        expected_table = f"analytics.churn.gold_features_{cn}"
        mock_spark.table.assert_called_once_with(expected_table)
        mock_spark.table.return_value.toPandas.assert_not_called()
        assert result is mock_psdf

    def test_local_load_collects_to_pandas(self, databricks_scoring_config, gold_features_df):
        mock_spark = MagicMock()
        mock_spark.table.return_value.toPandas.return_value = gold_features_df

        with patch(
            "customer_retention.stages.scoring.data_loader.get_spark_session",
            return_value=mock_spark,
        ):
            loader = ScoringDataLoader(databricks_scoring_config)
            result = loader.load_gold_features()

        mock_spark.table.return_value.toPandas.assert_called_once()
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(gold_features_df)

    def test_holdout_filter_before_collect(self, databricks_scoring_config, gold_features_df):
        mock_spark = MagicMock()
        mock_psdf = MagicMock()
        mock_columns = gold_features_df.columns.tolist()
        mock_psdf.columns = mock_columns
        mock_psdf.__getitem__ = MagicMock()

        holdout_mask = MagicMock()
        target_isna = MagicMock()
        original_notna = MagicMock()
        mock_psdf.__getitem__.side_effect = lambda key: (
            target_isna if key == "unsubscribed" else
            original_notna if key == "original_unsubscribed" else
            MagicMock()
        )
        target_isna.isna.return_value = MagicMock()
        original_notna.notna.return_value = MagicMock()

        mock_spark.table.return_value.pandas_api.return_value = mock_psdf

        with patch(
            "customer_retention.stages.scoring.data_loader.get_spark_session",
            return_value=mock_spark,
        ):
            loader = ScoringDataLoader(databricks_scoring_config)
            result = loader.load_gold_features_distributed()

        mock_spark.table.return_value.toPandas.assert_not_called()

    def test_export_to_delta_uses_normalize(self, databricks_scoring_config, gold_features_df):
        from customer_retention.core.compat import normalize_timestamp_columns, pandas_dtype_to_spark_schema

        predictions_df = pd.DataFrame({
            "customer_id": ["c80", "c81", "c82"],
            "prediction": [1, 0, 1],
            "probability": [0.8, 0.3, 0.7],
            "actual": [1.0, 0.0, 1.0],
        })

        normalized = normalize_timestamp_columns(predictions_df)
        schema = pandas_dtype_to_spark_schema(normalized)

        assert len(schema.fields) == len(predictions_df.columns)
        pd.testing.assert_frame_equal(normalized, predictions_df)


class TestScoringConfigDistributed:
    def test_local_config_is_not_databricks(self, scoring_config):
        assert not scoring_config.is_databricks

    def test_databricks_config_is_databricks(self, databricks_scoring_config):
        assert databricks_scoring_config.is_databricks

    def test_scoring_output_dir(self, scoring_config):
        expected = scoring_config.experiments_dir / "data" / "scoring"
        assert scoring_config.scoring_output_dir == expected

    def test_original_column_derived_from_target(self, scoring_config):
        assert scoring_config.original_column == "original_unsubscribed"
