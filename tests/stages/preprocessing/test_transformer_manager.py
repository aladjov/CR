import pytest
import pandas as pd
import numpy as np
from pathlib import Path


@pytest.fixture
def sample_df():
    """Sample DataFrame for testing transformations."""
    np.random.seed(42)
    return pd.DataFrame({
        "customer_id": range(100),
        "age": np.random.randint(18, 80, 100),
        "income": np.random.uniform(20000, 150000, 100),
        "gender": np.random.choice(["M", "F", "O"], 100),
        "region": np.random.choice(["North", "South", "East", "West"], 100),
        "target": np.random.choice([0, 1], 100),
    })


@pytest.fixture
def scoring_df():
    """Sample scoring DataFrame (may have unseen categories)."""
    np.random.seed(123)
    return pd.DataFrame({
        "customer_id": range(100, 120),
        "age": np.random.randint(18, 80, 20),
        "income": np.random.uniform(20000, 150000, 20),
        "gender": np.random.choice(["M", "F", "X"], 20),  # 'X' is unseen
        "region": np.random.choice(["North", "South", "Central"], 20),  # 'Central' is unseen
    })


class TestTransformerManagerFitTransform:
    def test_fit_transform_numeric_columns(self, sample_df):
        from customer_retention.stages.preprocessing import TransformerManager

        manager = TransformerManager()
        result = manager.fit_transform(
            sample_df,
            numeric_columns=["age", "income"],
            categorical_columns=[],
            exclude_columns=["customer_id", "target"]
        )

        # Numeric columns should be standardized (mean ~0, std ~1)
        assert abs(result["age"].mean()) < 0.1
        assert abs(result["age"].std() - 1.0) < 0.1
        assert manager.is_fitted

    def test_fit_transform_categorical_columns(self, sample_df):
        from customer_retention.stages.preprocessing import TransformerManager

        manager = TransformerManager()
        result = manager.fit_transform(
            sample_df,
            numeric_columns=[],
            categorical_columns=["gender", "region"],
            exclude_columns=["customer_id", "target"]
        )

        # Categorical columns should be encoded as integers
        assert result["gender"].dtype in [np.int64, np.int32, int]
        assert result["region"].dtype in [np.int64, np.int32, int]
        assert set(result["gender"].unique()).issubset({0, 1, 2})

    def test_fit_transform_preserves_excluded_columns(self, sample_df):
        from customer_retention.stages.preprocessing import TransformerManager

        manager = TransformerManager()
        result = manager.fit_transform(
            sample_df,
            numeric_columns=["age"],
            categorical_columns=["gender"],
            exclude_columns=["customer_id", "target", "income", "region"]
        )

        # Excluded columns should not be in manifest
        assert "customer_id" not in manager.manifest.numeric_columns
        assert "target" not in manager.manifest.numeric_columns

    def test_manifest_populated(self, sample_df):
        from customer_retention.stages.preprocessing import TransformerManager

        manager = TransformerManager(scaler_type="robust")
        manager.fit_transform(
            sample_df,
            numeric_columns=["age", "income"],
            categorical_columns=["gender", "region"],
            exclude_columns=["customer_id", "target"]
        )

        manifest = manager.manifest
        assert manifest.scaler_type == "robust"
        assert set(manifest.numeric_columns) == {"age", "income"}
        assert set(manifest.categorical_columns) == {"gender", "region"}
        assert manifest.created_at is not None


class TestTransformerManagerTransform:
    def test_transform_applies_same_scaling(self, sample_df):
        from customer_retention.stages.preprocessing import TransformerManager

        manager = TransformerManager()
        manager.fit_transform(
            sample_df,
            numeric_columns=["age", "income"],
            categorical_columns=[],
            exclude_columns=["customer_id", "target", "gender", "region"]
        )

        # Create new data with known values
        new_df = pd.DataFrame({
            "customer_id": [999],
            "age": [sample_df["age"].mean()],  # Should transform to ~0
            "income": [sample_df["income"].mean()],
        })

        result = manager.transform(new_df, exclude_columns=["customer_id"])
        assert abs(result["age"].iloc[0]) < 0.2  # Should be close to 0

    def test_transform_handles_unseen_categories(self, sample_df, scoring_df):
        from customer_retention.stages.preprocessing import TransformerManager

        manager = TransformerManager()
        manager.fit_transform(
            sample_df,
            numeric_columns=["age", "income"],
            categorical_columns=["gender", "region"],
            exclude_columns=["customer_id", "target"]
        )

        # Scoring df has 'X' gender and 'Central' region which are unseen
        result = manager.transform(scoring_df, exclude_columns=["customer_id"])

        # Should not raise, unseen categories mapped to 0
        assert result["gender"].dtype in [np.int64, np.int32, int]
        assert result["region"].dtype in [np.int64, np.int32, int]

    def test_transform_raises_if_not_fitted(self, sample_df):
        from customer_retention.stages.preprocessing import TransformerManager

        manager = TransformerManager()
        with pytest.raises(ValueError, match="not fitted"):
            manager.transform(sample_df)


class TestTransformerManagerPersistence:
    def test_save_and_load(self, sample_df, tmp_path):
        from customer_retention.stages.preprocessing import TransformerManager

        manager = TransformerManager()
        original_result = manager.fit_transform(
            sample_df,
            numeric_columns=["age", "income"],
            categorical_columns=["gender"],
            exclude_columns=["customer_id", "target", "region"]
        )

        # Save
        save_path = tmp_path / "transformers.joblib"
        manager.save(save_path)
        assert save_path.exists()

        # Load
        loaded_manager = TransformerManager.load(save_path)
        assert loaded_manager.is_fitted
        assert loaded_manager.manifest.numeric_columns == manager.manifest.numeric_columns

        # Transform new data with loaded manager
        new_df = sample_df.head(10).copy()
        result = loaded_manager.transform(new_df, exclude_columns=["customer_id", "target", "region"])

        # Results should match original transformation
        expected = original_result.head(10)[["age", "income", "gender"]]
        pd.testing.assert_frame_equal(result, expected, check_dtype=False)

    def test_save_raises_if_not_fitted(self, tmp_path):
        from customer_retention.stages.preprocessing import TransformerManager

        manager = TransformerManager()
        with pytest.raises(ValueError, match="Cannot save unfitted"):
            manager.save(tmp_path / "test.joblib")


class TestTransformerManagerScalerTypes:
    @pytest.mark.parametrize("scaler_type", ["standard", "robust", "minmax"])
    def test_different_scaler_types(self, sample_df, scaler_type):
        from customer_retention.stages.preprocessing import TransformerManager

        manager = TransformerManager(scaler_type=scaler_type)
        result = manager.fit_transform(
            sample_df,
            numeric_columns=["age", "income"],
            categorical_columns=[],
            exclude_columns=["customer_id", "target", "gender", "region"]
        )

        assert manager.manifest.scaler_type == scaler_type
        # All scaler types should produce numeric output
        assert result["age"].dtype in [np.float64, np.float32]


class TestTransformerBundle:
    def test_to_dict_and_from_dict(self, sample_df):
        from customer_retention.stages.preprocessing import TransformerManager, TransformerBundle

        manager = TransformerManager()
        manager.fit_transform(
            sample_df,
            numeric_columns=["age"],
            categorical_columns=["gender"],
            exclude_columns=["customer_id", "target", "income", "region"]
        )

        # Convert to dict and back
        bundle_dict = manager._bundle.to_dict()
        assert "numeric_scaler" in bundle_dict
        assert "label_encoders" in bundle_dict
        assert "manifest" in bundle_dict

        # Reconstruct
        reconstructed = TransformerBundle.from_dict(bundle_dict)
        assert reconstructed.scaler is not None
        assert "gender" in reconstructed.encoders


class TestTransformerManifest:
    def test_to_dict_and_from_dict(self):
        from customer_retention.stages.preprocessing.transformer_manager import TransformerManifest

        manifest = TransformerManifest(
            numeric_columns=["a", "b"],
            categorical_columns=["c"],
            scaler_type="standard",
            encoder_type="label",
            feature_order=["a", "b", "c"],
            created_at="2024-01-01T00:00:00"
        )

        d = manifest.to_dict()
        reconstructed = TransformerManifest.from_dict(d)

        assert reconstructed.numeric_columns == ["a", "b"]
        assert reconstructed.categorical_columns == ["c"]
        assert reconstructed.scaler_type == "standard"


class TestTransformerManagerMLflowIntegration:
    def test_log_to_mlflow_creates_artifacts(self, sample_df, tmp_path, monkeypatch):
        from customer_retention.stages.preprocessing import TransformerManager
        import mlflow

        mlflow.set_tracking_uri(f"file://{tmp_path}/mlruns")

        manager = TransformerManager()
        manager.fit_transform(
            sample_df,
            numeric_columns=["age", "income"],
            categorical_columns=["gender"],
            exclude_columns=["customer_id", "target", "region"]
        )

        with mlflow.start_run() as run:
            manager.log_to_mlflow()
            run_id = run.info.run_id

        client = mlflow.tracking.MlflowClient()
        artifacts = client.list_artifacts(run_id, "transformers")
        artifact_names = [a.path for a in artifacts]
        assert any("transformers.joblib" in p for p in artifact_names)
        assert any("transformer_manifest.json" in p for p in artifact_names)

    def test_log_to_mlflow_with_run_id(self, sample_df, tmp_path):
        from customer_retention.stages.preprocessing import TransformerManager
        import mlflow

        mlflow.set_tracking_uri(f"file://{tmp_path}/mlruns")

        manager = TransformerManager()
        manager.fit_transform(
            sample_df,
            numeric_columns=["age"],
            categorical_columns=[],
            exclude_columns=["customer_id", "target", "income", "gender", "region"]
        )

        with mlflow.start_run() as run:
            run_id = run.info.run_id

        manager.log_to_mlflow(run_id=run_id)

        client = mlflow.tracking.MlflowClient()
        artifacts = client.list_artifacts(run_id, "transformers")
        assert len(artifacts) > 0

    def test_load_from_mlflow(self, sample_df, tmp_path):
        from customer_retention.stages.preprocessing import TransformerManager
        import mlflow

        mlflow.set_tracking_uri(f"file://{tmp_path}/mlruns")

        manager = TransformerManager()
        original = manager.fit_transform(
            sample_df,
            numeric_columns=["age", "income"],
            categorical_columns=["gender"],
            exclude_columns=["customer_id", "target", "region"]
        )

        with mlflow.start_run() as run:
            manager.log_to_mlflow()
            run_id = run.info.run_id

        loaded = TransformerManager.load_from_mlflow(run_id, tracking_uri=f"file://{tmp_path}/mlruns")

        assert loaded.is_fitted
        assert loaded.manifest.numeric_columns == manager.manifest.numeric_columns
        assert loaded.manifest.categorical_columns == manager.manifest.categorical_columns

    def test_load_from_mlflow_by_experiment(self, sample_df, tmp_path):
        from customer_retention.stages.preprocessing import TransformerManager
        import mlflow

        mlflow.set_tracking_uri(f"file://{tmp_path}/mlruns")
        experiment_name = "test_experiment"
        mlflow.set_experiment(experiment_name)

        manager = TransformerManager()
        manager.fit_transform(
            sample_df,
            numeric_columns=["age"],
            categorical_columns=["gender"],
            exclude_columns=["customer_id", "target", "income", "region"]
        )

        with mlflow.start_run(run_name="04_transformation") as run:
            manager.log_to_mlflow()

        loaded = TransformerManager.load_from_mlflow_by_experiment(
            experiment_name,
            run_name_filter="04_transformation",
            tracking_uri=f"file://{tmp_path}/mlruns"
        )

        assert loaded.is_fitted
        assert loaded.manifest.scaler_type == "standard"

    def test_load_from_mlflow_raises_if_experiment_not_found(self, tmp_path):
        from customer_retention.stages.preprocessing import TransformerManager
        import mlflow

        mlflow.set_tracking_uri(f"file://{tmp_path}/mlruns")

        with pytest.raises(ValueError, match="not found"):
            TransformerManager.load_from_mlflow_by_experiment(
                "nonexistent_experiment",
                tracking_uri=f"file://{tmp_path}/mlruns"
            )

    def test_log_to_mlflow_raises_if_not_fitted(self, tmp_path):
        from customer_retention.stages.preprocessing import TransformerManager
        import mlflow

        mlflow.set_tracking_uri(f"file://{tmp_path}/mlruns")
        mlflow.set_experiment("test_unfitted")

        manager = TransformerManager()
        with pytest.raises(ValueError, match="Cannot log unfitted"):
            with mlflow.start_run():
                manager.log_to_mlflow()


class TestTransformerManagerEdgeCases:
    def test_transform_with_missing_numeric_columns(self, sample_df):
        from customer_retention.stages.preprocessing import TransformerManager

        manager = TransformerManager()
        manager.fit_transform(
            sample_df,
            numeric_columns=["age", "income"],
            categorical_columns=[],
            exclude_columns=["customer_id", "target", "gender", "region"]
        )

        incomplete_df = pd.DataFrame({
            "age": [25, 30],
        })

        result = manager.transform(incomplete_df, exclude_columns=[])
        assert "income" in result.columns
        assert result["income"].iloc[0] == 0.0

    def test_safe_encode_handles_unseen_values(self, sample_df):
        from customer_retention.stages.preprocessing import TransformerManager

        manager = TransformerManager()
        manager.fit_transform(
            sample_df,
            numeric_columns=[],
            categorical_columns=["gender"],
            exclude_columns=["customer_id", "target", "age", "income", "region"]
        )

        encoder = manager._bundle.encoders["gender"]
        result = manager._safe_encode(encoder, "UNKNOWN_VALUE")
        assert result == 0

    def test_transform_without_scaler(self, sample_df):
        from customer_retention.stages.preprocessing import TransformerManager

        manager = TransformerManager()
        manager.fit_transform(
            sample_df,
            numeric_columns=[],
            categorical_columns=["gender", "region"],
            exclude_columns=["customer_id", "target", "age", "income"]
        )

        new_df = pd.DataFrame({"gender": ["M", "F"], "region": ["North", "South"]})
        result = manager.transform(new_df, exclude_columns=[])

        assert manager._bundle.scaler is None
        assert result["gender"].dtype in [np.int64, np.int32, int]
