import textwrap
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
        composite_name="cust_emails_prof__a1b2c3d",
        target_column="unsubscribed",
        entity_key="customer_id",
        timestamp_column="event_timestamp",
        recommendations_hash="abc123",
        experiments_dir=Path("/tmp/experiments"),
        artifacts_path=Path("/tmp/artifacts"),
        mlflow_tracking_uri="sqlite:///mlruns.db",
        production_dir=Path("/tmp/production"),
        feast_repo_path="/tmp/feast_repo",
        feast_feature_view="featureset_cust_emails_prof__a1b2c3d",
    )


@pytest.fixture
def databricks_config():
    return ScoringConfig(
        pipeline_name="customer_churn",
        composite_name="cust_emails_prof__a1b2c3d",
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
    df = pd.DataFrame(
        {
            "customer_id": [f"c{i}" for i in range(n)],
            "event_timestamp": pd.date_range("2024-01-01", periods=n),
            "feature_a": np.random.randn(n),
            "feature_b": np.random.randn(n) * 10 + 50,
            "unsubscribed": np.random.randint(0, 2, n).astype(float),
            "original_unsubscribed": [np.nan] * n,
        }
    )
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
    def test_loads_from_delta_with_cn(self, local_config, sample_gold_df, tmp_path):
        from customer_retention.integrations.adapters.factory import get_delta

        cn = local_config.composite_name
        gold_dir = tmp_path / "data" / "gold" / f"gold_features_{cn}"
        gold_dir.parent.mkdir(parents=True)
        storage = get_delta(force_local=True)
        storage.write(sample_gold_df, str(gold_dir))
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
    def test_loads_from_spark_table_with_cn(self, databricks_config):
        mock_spark = MagicMock()
        mock_pdf = pd.DataFrame({"customer_id": ["c1"], "feature_a": [1.0]})
        mock_spark.table.return_value.toPandas.return_value = mock_pdf
        cn = databricks_config.composite_name
        with patch("customer_retention.stages.scoring.data_loader.get_spark_session", return_value=mock_spark):
            loader = ScoringDataLoader(databricks_config)
            result = loader.load_gold_features()
        mock_spark.table.assert_called_once_with(f"analytics.churn.gold_features_{cn}")
        assert len(result) == 1

    def test_spark_unavailable_raises(self, databricks_config):
        with patch("customer_retention.stages.scoring.data_loader.get_spark_session", return_value=None):
            loader = ScoringDataLoader(databricks_config)
            with pytest.raises(RuntimeError, match="Spark"):
                loader.load_gold_features()


class TestLoadGoldFeaturesDistributedLocal:
    def test_loads_from_delta_same_as_local(self, local_config, sample_gold_df, tmp_path):
        from customer_retention.integrations.adapters.factory import get_delta

        cn = local_config.composite_name
        gold_dir = tmp_path / "data" / "gold" / f"gold_features_{cn}"
        gold_dir.parent.mkdir(parents=True)
        storage = get_delta(force_local=True)
        storage.write(sample_gold_df, str(gold_dir))
        local_config.production_dir = tmp_path
        loader = ScoringDataLoader(local_config)
        result = loader.load_gold_features_distributed()
        assert len(result) == len(sample_gold_df)
        assert "feature_a" in result.columns

    def test_missing_gold_raises(self, local_config, tmp_path):
        local_config.production_dir = tmp_path
        loader = ScoringDataLoader(local_config)
        with pytest.raises(FileNotFoundError):
            loader.load_gold_features_distributed()


class TestLoadGoldFeaturesDistributedDatabricks:
    def test_returns_distributed_dataframe(self, databricks_config):
        mock_spark = MagicMock()
        mock_psdf = MagicMock()
        mock_spark.table.return_value.pandas_api.return_value = mock_psdf
        cn = databricks_config.composite_name
        with patch("customer_retention.stages.scoring.data_loader.get_spark_session", return_value=mock_spark):
            loader = ScoringDataLoader(databricks_config)
            result = loader.load_gold_features_distributed()
        mock_spark.table.assert_called_once_with(f"analytics.churn.gold_features_{cn}")
        mock_spark.table.return_value.toPandas.assert_not_called()
        assert result is mock_psdf

    def test_falls_back_to_to_pandas_on_spark(self, databricks_config):
        mock_spark = MagicMock()
        mock_spark_df = MagicMock(spec=[])
        mock_psdf = MagicMock()
        mock_spark_df.to_pandas_on_spark = MagicMock(return_value=mock_psdf)
        mock_spark.table.return_value = mock_spark_df
        with patch("customer_retention.stages.scoring.data_loader.get_spark_session", return_value=mock_spark):
            loader = ScoringDataLoader(databricks_config)
            result = loader.load_gold_features_distributed()
        mock_spark_df.to_pandas_on_spark.assert_called_once()
        assert result is mock_psdf

    def test_spark_unavailable_raises(self, databricks_config):
        with patch("customer_retention.stages.scoring.data_loader.get_spark_session", return_value=None):
            loader = ScoringDataLoader(databricks_config)
            with pytest.raises(RuntimeError, match="Spark"):
                loader.load_gold_features_distributed()

    def test_does_not_call_to_pandas(self, databricks_config):
        mock_spark = MagicMock()
        mock_spark.table.return_value.pandas_api.return_value = MagicMock()
        with patch("customer_retention.stages.scoring.data_loader.get_spark_session", return_value=mock_spark):
            loader = ScoringDataLoader(databricks_config)
            loader.load_gold_features_distributed()
        mock_spark.table.return_value.toPandas.assert_not_called()


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


    def test_loads_spark_model_on_databricks(self, databricks_config, mock_mlflow_client):
        mock_spark_model = MagicMock()
        with (
            patch("customer_retention.stages.scoring.data_loader.mlflow") as mock_mlflow,
            patch("customer_retention.stages.scoring.data_loader.MlflowClient", return_value=mock_mlflow_client),
        ):
            mock_mlflow.spark.load_model.return_value = mock_spark_model
            loader = ScoringDataLoader(databricks_config)
            model, uri = loader.load_model()
        assert model is mock_spark_model
        mock_mlflow.spark.load_model.assert_called_once()
        mock_mlflow.sklearn.load_model.assert_not_called()

    def test_databricks_model_name_has_no_hash(self, databricks_config, mock_mlflow_client):
        with (
            patch("customer_retention.stages.scoring.data_loader.mlflow") as mock_mlflow,
            patch("customer_retention.stages.scoring.data_loader.MlflowClient", return_value=mock_mlflow_client),
        ):
            mock_mlflow.spark.load_model.return_value = MagicMock()
            loader = ScoringDataLoader(databricks_config)
            _, uri = loader.load_model()
        assert "model_random_forest" in uri
        assert "abc123" not in uri

    def test_local_model_name_includes_hash(self, local_config, mock_mlflow_client):
        with (
            patch("customer_retention.stages.scoring.data_loader.mlflow") as mock_mlflow,
            patch("customer_retention.stages.scoring.data_loader.MlflowClient", return_value=mock_mlflow_client),
        ):
            mock_mlflow.sklearn.load_model.return_value = MagicMock()
            loader = ScoringDataLoader(local_config)
            _, uri = loader.load_model()
        assert "model_random_forest_abc123" in uri


class TestLoadScoringFeatures:
    def test_feast_missing_repo_falls_back_to_scoring_df(self, local_config, sample_gold_df):
        local_config.feast_repo_path = "/nonexistent/feast"
        loader = ScoringDataLoader(local_config)
        holdout = sample_gold_df[sample_gold_df["unsubscribed"].isna()].copy()
        result = loader.load_scoring_features(holdout)
        pd.testing.assert_frame_equal(result, holdout)

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

    def test_feast_exception_propagates(self, local_config, sample_gold_df, tmp_path):
        feast_path = tmp_path / "feast"
        feast_path.mkdir()
        (feast_path / "feature_store.yaml").write_text("project: test")
        local_config.feast_repo_path = str(feast_path)
        holdout = sample_gold_df[sample_gold_df["unsubscribed"].isna()].copy()
        with patch("customer_retention.stages.scoring.data_loader.FeatureStore", side_effect=Exception("boom")):
            loader = ScoringDataLoader(local_config)
            with pytest.raises(Exception, match="boom"):
                loader.load_scoring_features(holdout)

    def test_databricks_uses_direct(self, databricks_config, sample_gold_df):
        loader = ScoringDataLoader(databricks_config)
        holdout = sample_gold_df[sample_gold_df["unsubscribed"].isna()].copy()
        result = loader.load_scoring_features(holdout)
        assert len(result) == len(holdout)


class TestLoadArtifactStore:
    def test_loads_from_manifest_on_local(self, local_config, tmp_path):
        artifacts_dir = tmp_path / "artifacts"
        artifacts_dir.mkdir()
        import yaml

        manifest = {"col_a_scaler": {"type": "scaler", "column": "col_a", "path": str(artifacts_dir / "col_a.joblib")}}
        (artifacts_dir / "manifest.yaml").write_text(yaml.dump(manifest))
        local_config.artifacts_path = artifacts_dir
        loader = ScoringDataLoader(local_config)
        store = loader.load_artifact_store()
        assert store is not None
        assert store.has("col_a_scaler")

    def test_returns_none_on_databricks(self, databricks_config):
        loader = ScoringDataLoader(databricks_config)
        assert loader.load_artifact_store() is None

    def test_missing_manifest_raises_on_local(self, local_config, tmp_path):
        local_config.artifacts_path = tmp_path / "nonexistent"
        loader = ScoringDataLoader(local_config)
        with pytest.raises(FileNotFoundError):
            loader.load_artifact_store()


class TestLoadTransforms:
    def test_loads_from_gold_module_on_local(self, local_config, tmp_path):
        mock_gold_module = MagicMock()
        mock_gold_module.ENCODINGS = [MagicMock()]
        mock_gold_module.SCALINGS = [MagicMock()]
        with patch(
            "customer_retention.stages.scoring.data_loader.ScoringDataLoader._load_gold_module",
            return_value=mock_gold_module,
        ):
            loader = ScoringDataLoader(local_config)
            encodings, scalings = loader.load_transforms()
        assert len(encodings) == 1
        assert len(scalings) == 1

    def test_returns_empty_on_databricks(self, databricks_config):
        loader = ScoringDataLoader(databricks_config)
        encodings, scalings = loader.load_transforms()
        assert encodings == []
        assert scalings == []

    def test_missing_gold_module_raises_on_local(self, local_config, tmp_path):
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

    def test_works_with_none_artifact_store(self, local_config, sample_gold_df):
        loader = ScoringDataLoader(local_config)
        mock_executor = MagicMock()
        mock_executor.apply_all.side_effect = lambda df, *a, **kw: df
        holdout = sample_gold_df[sample_gold_df["unsubscribed"].isna()].copy()
        result = loader.prepare_features(holdout, [], mock_executor, None)
        assert "feature_a" in result.columns
        assert "feature_b" in result.columns

    def test_selects_numeric_dtypes(self, local_config):
        loader = ScoringDataLoader(local_config)
        df = pd.DataFrame(
            {
                "num_col": [1.0, 2.0],
                "str_col": ["a", "b"],
                "int_col": [1, 2],
            }
        )
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

    def test_excludes_datetime_columns(self, local_config):
        loader = ScoringDataLoader(local_config)
        df = pd.DataFrame({
            "num": [1.0, 2.0],
            "dt": pd.date_range("2024-01-01", periods=2),
        })
        mock_executor = MagicMock()
        mock_executor.apply_all.side_effect = lambda df, *a, **kw: df
        result = loader.prepare_features(df, [], mock_executor, MagicMock())
        assert "dt" not in result.columns
        assert "num" in result.columns

    def test_preserves_bool_columns(self, local_config):
        loader = ScoringDataLoader(local_config)
        df = pd.DataFrame({"flag": [True, False], "num": [1.0, 2.0]})
        mock_executor = MagicMock()
        mock_executor.apply_all.side_effect = lambda df, *a, **kw: df
        result = loader.prepare_features(df, [], mock_executor, MagicMock())
        assert "flag" in result.columns
        assert "num" in result.columns

    def test_preserves_nullable_int_columns(self, local_config):
        loader = ScoringDataLoader(local_config)
        df = pd.DataFrame({"val": pd.array([1, 2, None], dtype="Int64")})
        mock_executor = MagicMock()
        mock_executor.apply_all.side_effect = lambda df, *a, **kw: df
        result = loader.prepare_features(df, [], mock_executor, MagicMock())
        assert "val" in result.columns


class TestAlignFeaturesToModel:
    def test_returns_aligned_when_features_match(self, local_config):
        loader = ScoringDataLoader(local_config)
        X = pd.DataFrame({"a": [1.0], "b": [2.0]})
        model = MagicMock()
        model.feature_names_in_ = np.array(["a", "b"])
        aligned, missing, extra = loader.align_features_to_model(X, model)
        assert list(aligned.columns) == ["a", "b"]
        assert missing == []
        assert extra == []

    def test_adds_missing_columns_as_zero(self, local_config):
        loader = ScoringDataLoader(local_config)
        X = pd.DataFrame({"a": [1.0, 2.0]})
        model = MagicMock()
        model.feature_names_in_ = np.array(["a", "b", "c"])
        aligned, missing, extra = loader.align_features_to_model(X, model)
        assert list(aligned.columns) == ["a", "b", "c"]
        assert (aligned["b"] == 0.0).all()
        assert (aligned["c"] == 0.0).all()
        assert set(missing) == {"b", "c"}

    def test_drops_extra_columns(self, local_config):
        loader = ScoringDataLoader(local_config)
        X = pd.DataFrame({"a": [1.0], "b": [2.0], "c": [3.0]})
        model = MagicMock()
        model.feature_names_in_ = np.array(["a", "b"])
        aligned, missing, extra = loader.align_features_to_model(X, model)
        assert list(aligned.columns) == ["a", "b"]
        assert extra == ["c"]

    def test_reorders_columns_to_match_model(self, local_config):
        loader = ScoringDataLoader(local_config)
        X = pd.DataFrame({"b": [2.0], "a": [1.0]})
        model = MagicMock()
        model.feature_names_in_ = np.array(["a", "b"])
        aligned, _, _ = loader.align_features_to_model(X, model)
        assert list(aligned.columns) == ["a", "b"]

    def test_handles_xgboost_feature_names(self, local_config):
        loader = ScoringDataLoader(local_config)
        X = pd.DataFrame({"a": [1.0], "b": [2.0]})
        model = MagicMock(spec=[])
        model.feature_names = ["a", "b"]
        aligned, missing, extra = loader.align_features_to_model(X, model)
        assert list(aligned.columns) == ["a", "b"]

    def test_returns_unmodified_when_no_feature_names(self, local_config):
        loader = ScoringDataLoader(local_config)
        X = pd.DataFrame({"a": [1.0], "b": [2.0]})
        model = MagicMock(spec=[])
        aligned, missing, extra = loader.align_features_to_model(X, model)
        assert list(aligned.columns) == ["a", "b"]
        assert missing == []
        assert extra == []

    def test_combined_missing_and_extra(self, local_config):
        loader = ScoringDataLoader(local_config)
        X = pd.DataFrame({"a": [1.0], "new_feat": [5.0]})
        model = MagicMock()
        model.feature_names_in_ = np.array(["a", "old_feat"])
        aligned, missing, extra = loader.align_features_to_model(X, model)
        assert list(aligned.columns) == ["a", "old_feat"]
        assert missing == ["old_feat"]
        assert extra == ["new_feat"]
        assert (aligned["old_feat"] == 0.0).all()


@pytest.fixture
def local_pipeline_with_gold(tmp_path):
    pipeline_dir = tmp_path / "my_pipeline"
    pipeline_dir.mkdir()
    (pipeline_dir / "config.py").write_text(
        textwrap.dedent("""\
        ENCODINGS = [{"type": "label", "column": "cat_a"}]
        SCALINGS = [{"type": "standard", "column": "num_b"}]
    """)
    )
    gold_dir = pipeline_dir / "gold"
    gold_dir.mkdir()
    (gold_dir / "gold_features.py").write_text(
        textwrap.dedent("""\
        from config import ENCODINGS, SCALINGS
    """)
    )
    return pipeline_dir


class TestLoadGoldModule:
    def test_loads_gold_module_with_config_dependency(self, local_pipeline_with_gold):
        config = ScoringConfig(
            pipeline_name="my_pipeline",
            composite_name="my_pipe__abc1234",
            target_column="target",
            entity_key="customer_id",
            timestamp_column="event_timestamp",
            recommendations_hash="",
            experiments_dir=Path("/tmp/experiments"),
            artifacts_path=Path("/tmp/artifacts"),
            mlflow_tracking_uri="sqlite:///mlruns.db",
            production_dir=Path("/tmp/production"),
            pipeline_dir=local_pipeline_with_gold,
        )
        loader = ScoringDataLoader(config)
        module = loader._load_gold_module()
        assert module.ENCODINGS == [{"type": "label", "column": "cat_a"}]
        assert module.SCALINGS == [{"type": "standard", "column": "num_b"}]

    def test_missing_gold_features_raises(self, tmp_path):
        pipeline_dir = tmp_path / "no_gold"
        pipeline_dir.mkdir()
        (pipeline_dir / "config.py").write_text("X = 1\n")
        gold_dir = pipeline_dir / "gold"
        gold_dir.mkdir()
        config = ScoringConfig(
            pipeline_name="test",
            composite_name="test__abc1234",
            target_column="target",
            entity_key="customer_id",
            timestamp_column="event_timestamp",
            recommendations_hash="",
            experiments_dir=Path("/tmp/experiments"),
            artifacts_path=Path("/tmp/artifacts"),
            mlflow_tracking_uri="sqlite:///mlruns.db",
            production_dir=Path("/tmp/production"),
            pipeline_dir=pipeline_dir,
        )
        loader = ScoringDataLoader(config)
        with pytest.raises(FileNotFoundError, match="gold_features"):
            loader._load_gold_module()

    def test_missing_config_in_pipeline_dir_raises(self, tmp_path):
        pipeline_dir = tmp_path / "no_config"
        pipeline_dir.mkdir()
        gold_dir = pipeline_dir / "gold"
        gold_dir.mkdir()
        (gold_dir / "gold_features.py").write_text("X = 1\n")
        config = ScoringConfig(
            pipeline_name="test",
            composite_name="test__abc1234",
            target_column="target",
            entity_key="customer_id",
            timestamp_column="event_timestamp",
            recommendations_hash="",
            experiments_dir=Path("/tmp/experiments"),
            artifacts_path=Path("/tmp/artifacts"),
            mlflow_tracking_uri="sqlite:///mlruns.db",
            production_dir=Path("/tmp/production"),
            pipeline_dir=pipeline_dir,
        )
        loader = ScoringDataLoader(config)
        with pytest.raises(FileNotFoundError, match="config.py"):
            loader._load_gold_module()

    def test_databricks_raises_file_not_found(self, databricks_config):
        loader = ScoringDataLoader(databricks_config)
        with pytest.raises(FileNotFoundError, match="Databricks"):
            loader._load_gold_module()

    def test_empty_pipeline_dir_raises(self):
        config = ScoringConfig(
            pipeline_name="test",
            composite_name="test__abc1234",
            target_column="target",
            entity_key="customer_id",
            timestamp_column="event_timestamp",
            recommendations_hash="",
            experiments_dir=Path("/tmp/experiments"),
            artifacts_path=Path("/tmp/artifacts"),
            mlflow_tracking_uri="sqlite:///mlruns.db",
            production_dir=Path("/tmp/production"),
            pipeline_dir=Path(),
        )
        loader = ScoringDataLoader(config)
        with pytest.raises(FileNotFoundError, match="config.py"):
            loader._load_gold_module()


class TestPredictSparkMl:
    def test_assembles_features_and_returns_probabilities(self, databricks_config):
        pytest.importorskip("pyspark")
        mock_spark = MagicMock()
        mock_spark_df = MagicMock()
        mock_spark.createDataFrame.return_value = mock_spark_df

        mock_assembler = MagicMock()
        mock_assembled = MagicMock()
        mock_assembler.transform.return_value = mock_assembled

        mock_model = MagicMock()
        mock_predictions = MagicMock()
        mock_model.transform.return_value = mock_predictions

        proba_values = np.array([0.2, 0.8, 0.5])
        mock_predictions.select.return_value.toPandas.return_value = pd.DataFrame({"prob": proba_values})

        X = pd.DataFrame({"feat_a": [1.0, 2.0, 3.0], "feat_b": [4.0, 5.0, 6.0]})

        with (
            patch("customer_retention.stages.scoring.data_loader.get_spark_session", return_value=mock_spark),
            patch("customer_retention.stages.scoring.data_loader._VectorAssembler", return_value=mock_assembler),
            patch("customer_retention.stages.scoring.data_loader._vector_to_array") as mock_v2a,
        ):
            loader = ScoringDataLoader(databricks_config)
            result = loader.predict_spark_ml(mock_model, X)

        mock_assembler.transform.assert_called_once_with(mock_spark_df)
        mock_model.transform.assert_called_once_with(mock_assembled)
        np.testing.assert_array_equal(result, proba_values)

    def test_assembler_uses_feature_columns(self, databricks_config):
        pytest.importorskip("pyspark")
        mock_spark = MagicMock()
        mock_assembler = MagicMock()
        mock_assembled = MagicMock()
        mock_assembler.transform.return_value = mock_assembled

        mock_model = MagicMock()
        mock_predictions = MagicMock()
        mock_model.transform.return_value = mock_predictions
        mock_predictions.select.return_value.toPandas.return_value = pd.DataFrame({"prob": [0.5]})

        X = pd.DataFrame({"col_x": [1.0], "col_y": [2.0]})

        with (
            patch("customer_retention.stages.scoring.data_loader.get_spark_session", return_value=mock_spark),
            patch("customer_retention.stages.scoring.data_loader._VectorAssembler", return_value=mock_assembler) as mock_va_cls,
            patch("customer_retention.stages.scoring.data_loader._vector_to_array"),
        ):
            loader = ScoringDataLoader(databricks_config)
            loader.predict_spark_ml(mock_model, X)

        mock_va_cls.assert_called_once_with(inputCols=["col_x", "col_y"], outputCol="features", handleInvalid="keep")
