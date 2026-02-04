from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from customer_retention.stages.scoring.config import ScoringConfig
from customer_retention.stages.scoring.data_loader import ScoringDataLoader


@pytest.fixture
def local_config():
    return ScoringConfig(
        pipeline_name="customer_churn",
        target_column="unsubscribed",
        entity_key="customer_id",
        timestamp_column="event_timestamp",
        recommendations_hash="abc123",
        experiments_dir=Path("/tmp/experiments"),
        artifacts_path=Path("/tmp/artifacts"),
        mlflow_tracking_uri="sqlite:///mlruns.db",
        production_dir=Path("/tmp/production"),
        feast_repo_path="/tmp/feast_repo",
        feast_feature_view="customer_churn_features",
    )


@pytest.fixture
def databricks_config():
    return ScoringConfig(
        pipeline_name="customer_churn",
        target_column="unsubscribed",
        entity_key="customer_id",
        timestamp_column="event_timestamp",
        recommendations_hash="abc123",
        experiments_dir=Path("/Volumes/analytics/churn/experiments"),
        artifacts_path=Path("/Volumes/analytics/churn/experiments/artifacts/abc123"),
        mlflow_tracking_uri="databricks",
        production_dir=Path("/Volumes/analytics/churn/experiments"),
        catalog="analytics",
        schema="churn",
    )


@pytest.fixture
def sample_gold_df():
    np.random.seed(42)
    n = 50
    df = pd.DataFrame({
        "customer_id": [f"c{i}" for i in range(n)],
        "event_timestamp": pd.date_range("2024-01-01", periods=n),
        "feature_a": np.random.randn(n),
        "feature_b": np.random.randn(n) * 10 + 50,
        "unsubscribed": np.random.randint(0, 2, n).astype(float),
        "original_unsubscribed": [np.nan] * n,
    })
    holdout_idx = list(range(40, 50))
    df.loc[holdout_idx, "original_unsubscribed"] = df.loc[holdout_idx, "unsubscribed"]
    df.loc[holdout_idx, "unsubscribed"] = np.nan
    return df


@pytest.fixture
def mock_mlflow_client():
    client = MagicMock()
    experiment = MagicMock()
    experiment.experiment_id = "exp_1"
    client.get_experiment_by_name.return_value = experiment
    parent_run = MagicMock()
    parent_run.info.run_id = "parent_run_1"
    parent_run.data.tags = {"best_model": "random_forest", "recommendations_hash": "abc123"}
    client.search_runs.return_value = [parent_run]
    return client


class TestLoadGoldFeaturesLocal:
    def test_loads_from_parquet(self, local_config, sample_gold_df, tmp_path):
        gold_path = tmp_path / "data" / "gold" / "customer_churn" / "features.parquet"
        gold_path.parent.mkdir(parents=True)
        sample_gold_df.to_parquet(gold_path, index=False)
        local_config.production_dir = tmp_path
        loader = ScoringDataLoader(local_config)
        result = loader.load_gold_features()
        assert len(result) == len(sample_gold_df)
        assert "feature_a" in result.columns

    def test_missing_gold_raises(self, local_config, tmp_path):
        local_config.production_dir = tmp_path
        loader = ScoringDataLoader(local_config)
        with pytest.raises(FileNotFoundError):
            loader.load_gold_features()


class TestLoadGoldFeaturesDatabricks:
    def test_loads_from_spark_table(self, databricks_config):
        mock_spark = MagicMock()
        mock_pdf = pd.DataFrame({"customer_id": ["c1"], "feature_a": [1.0]})
        mock_spark.table.return_value.toPandas.return_value = mock_pdf
        with patch("customer_retention.stages.scoring.data_loader.get_spark_session", return_value=mock_spark):
            loader = ScoringDataLoader(databricks_config)
            result = loader.load_gold_features()
        mock_spark.table.assert_called_once_with("analytics.churn.gold_features")
        assert len(result) == 1

    def test_spark_unavailable_raises(self, databricks_config):
        with patch("customer_retention.stages.scoring.data_loader.get_spark_session", return_value=None):
            loader = ScoringDataLoader(databricks_config)
            with pytest.raises(RuntimeError, match="Spark"):
                loader.load_gold_features()


class TestLoadModel:
    def test_loads_sklearn_model(self, local_config, mock_mlflow_client):
        mock_model = MagicMock()
        with (
            patch("customer_retention.stages.scoring.data_loader.mlflow") as mock_mlflow,
            patch("customer_retention.stages.scoring.data_loader.MlflowClient", return_value=mock_mlflow_client),
        ):
            mock_mlflow.sklearn.load_model.return_value = mock_model
            loader = ScoringDataLoader(local_config)
            model, uri = loader.load_model()
        assert model is mock_model
        assert "runs:/" in uri

    def test_loads_xgboost_model(self, local_config, mock_mlflow_client):
        mock_mlflow_client.search_runs.return_value[0].data.tags["best_model"] = "xgboost"
        mock_model = MagicMock()
        with (
            patch("customer_retention.stages.scoring.data_loader.mlflow") as mock_mlflow,
            patch("customer_retention.stages.scoring.data_loader.MlflowClient", return_value=mock_mlflow_client),
        ):
            mock_mlflow.xgboost.load_model.return_value = mock_model
            loader = ScoringDataLoader(local_config)
            model, uri = loader.load_model()
        assert model is mock_model
        mock_mlflow.xgboost.load_model.assert_called_once()

    def test_no_experiment_raises(self, local_config):
        client = MagicMock()
        client.get_experiment_by_name.return_value = None
        with (
            patch("customer_retention.stages.scoring.data_loader.mlflow"),
            patch("customer_retention.stages.scoring.data_loader.MlflowClient", return_value=client),
        ):
            loader = ScoringDataLoader(local_config)
            with pytest.raises(ValueError, match="not found"):
                loader.load_model()

    def test_no_runs_raises(self, local_config):
        client = MagicMock()
        experiment = MagicMock()
        experiment.experiment_id = "exp_1"
        client.get_experiment_by_name.return_value = experiment
        client.search_runs.return_value = []
        with (
            patch("customer_retention.stages.scoring.data_loader.mlflow"),
            patch("customer_retention.stages.scoring.data_loader.MlflowClient", return_value=client),
        ):
            loader = ScoringDataLoader(local_config)
            with pytest.raises(ValueError, match="No runs"):
                loader.load_model()

    def test_child_run_used_when_available(self, local_config, mock_mlflow_client):
        child_run = MagicMock()
        child_run.info.run_id = "child_1"
        child_run.info.run_name = "random_forest"

        def side_effect(experiment_ids, filter_string="", order_by=None, max_results=None):
            if "parentRunId" in filter_string:
                return [child_run]
            return mock_mlflow_client.search_runs.return_value

        mock_mlflow_client.search_runs.side_effect = side_effect
        with (
            patch("customer_retention.stages.scoring.data_loader.mlflow") as mock_mlflow,
            patch("customer_retention.stages.scoring.data_loader.MlflowClient", return_value=mock_mlflow_client),
        ):
            mock_mlflow.sklearn.load_model.return_value = MagicMock()
            loader = ScoringDataLoader(local_config)
            _, uri = loader.load_model()
        assert "child_1" in uri


class TestLoadScoringFeatures:
    def test_feast_fallback_to_direct(self, local_config, sample_gold_df):
        local_config.feast_repo_path = "/nonexistent/feast"
        loader = ScoringDataLoader(local_config)
        holdout = sample_gold_df[sample_gold_df["unsubscribed"].isna()].copy()
        result = loader.load_scoring_features(holdout)
        assert len(result) == len(holdout)

    def test_feast_loads_when_available(self, local_config, sample_gold_df, tmp_path):
        feast_path = tmp_path / "feast"
        feast_path.mkdir()
        (feast_path / "feature_store.yaml").write_text("project: test")
        local_config.feast_repo_path = str(feast_path)
        holdout = sample_gold_df[sample_gold_df["unsubscribed"].isna()].copy()
        mock_store = MagicMock()
        feast_result = holdout.drop(columns=["original_unsubscribed", "unsubscribed"]).copy()
        mock_store.get_online_features.return_value.to_df.return_value = feast_result
        with patch("customer_retention.stages.scoring.data_loader.FeatureStore", return_value=mock_store):
            loader = ScoringDataLoader(local_config)
            result = loader.load_scoring_features(holdout)
        assert "original_unsubscribed" in result.columns

    def test_feast_exception_falls_back(self, local_config, sample_gold_df, tmp_path):
        feast_path = tmp_path / "feast"
        feast_path.mkdir()
        (feast_path / "feature_store.yaml").write_text("project: test")
        local_config.feast_repo_path = str(feast_path)
        holdout = sample_gold_df[sample_gold_df["unsubscribed"].isna()].copy()
        with patch("customer_retention.stages.scoring.data_loader.FeatureStore", side_effect=Exception("boom")):
            loader = ScoringDataLoader(local_config)
            result = loader.load_scoring_features(holdout)
        assert len(result) == len(holdout)

    def test_databricks_uses_direct(self, databricks_config, sample_gold_df):
        loader = ScoringDataLoader(databricks_config)
        holdout = sample_gold_df[sample_gold_df["unsubscribed"].isna()].copy()
        result = loader.load_scoring_features(holdout)
        assert len(result) == len(holdout)


class TestLoadTransforms:
    def test_loads_from_manifest(self, local_config, tmp_path):
        artifacts_dir = tmp_path / "artifacts"
        artifacts_dir.mkdir()
        manifest = {"col_a_scaler": {"type": "scaler", "column": "col_a", "path": str(artifacts_dir / "col_a.joblib")}}
        import yaml
        (artifacts_dir / "manifest.yaml").write_text(yaml.dump(manifest))
        local_config.artifacts_path = artifacts_dir
        mock_gold_module = MagicMock()
        mock_gold_module.ENCODINGS = [MagicMock()]
        mock_gold_module.SCALINGS = [MagicMock()]
        with patch("customer_retention.stages.scoring.data_loader.ScoringDataLoader._load_gold_module", return_value=mock_gold_module):
            loader = ScoringDataLoader(local_config)
            encodings, scalings = loader.load_transforms()
        assert len(encodings) == 1
        assert len(scalings) == 1

    def test_missing_manifest_raises(self, local_config, tmp_path):
        local_config.artifacts_path = tmp_path / "nonexistent"
        with patch(
            "customer_retention.stages.scoring.data_loader.ScoringDataLoader._load_gold_module",
            side_effect=FileNotFoundError("no gold module"),
        ):
            loader = ScoringDataLoader(local_config)
            with pytest.raises(FileNotFoundError):
                loader.load_transforms()


class TestPrepareFeatures:
    def test_drops_meta_columns(self, local_config, sample_gold_df):
        loader = ScoringDataLoader(local_config)
        mock_executor = MagicMock()
        mock_executor.apply_all.side_effect = lambda df, *a, **kw: df
        mock_registry = MagicMock()
        holdout = sample_gold_df[sample_gold_df["unsubscribed"].isna()].copy()
        result = loader.prepare_features(holdout, [], mock_executor, mock_registry)
        assert "customer_id" not in result.columns
        assert "event_timestamp" not in result.columns
        assert "original_unsubscribed" not in result.columns
        assert "unsubscribed" not in result.columns

    def test_selects_numeric_dtypes(self, local_config):
        loader = ScoringDataLoader(local_config)
        df = pd.DataFrame({
            "num_col": [1.0, 2.0],
            "str_col": ["a", "b"],
            "int_col": [1, 2],
        })
        mock_executor = MagicMock()
        mock_executor.apply_all.side_effect = lambda df, *a, **kw: df
        result = loader.prepare_features(df, [], mock_executor, MagicMock())
        assert "num_col" in result.columns
        assert "int_col" in result.columns
        assert "str_col" not in result.columns

    def test_fills_nan(self, local_config):
        loader = ScoringDataLoader(local_config)
        df = pd.DataFrame({"val": [1.0, np.nan, 3.0]})
        mock_executor = MagicMock()
        mock_executor.apply_all.side_effect = lambda df, *a, **kw: df
        result = loader.prepare_features(df, [], mock_executor, MagicMock())
        assert result["val"].isna().sum() == 0
