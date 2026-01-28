import numpy as np
import pytest
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

from customer_retention.artifacts import FitArtifact, FitArtifactRegistry


@pytest.fixture
def tmp_artifacts_dir(tmp_path):
    return tmp_path / "artifacts" / "abc12345"


@pytest.fixture
def fitted_standard_scaler():
    scaler = StandardScaler()
    scaler.fit(np.array([[1, 2], [3, 4], [5, 6]]))
    return scaler


@pytest.fixture
def fitted_label_encoder():
    encoder = LabelEncoder()
    encoder.fit(["cat", "dog", "bird"])
    return encoder


class TestFitArtifact:
    def test_creates_with_required_fields(self):
        artifact = FitArtifact(
            artifact_id="income_standard",
            artifact_type="scaler",
            target_column="income",
            transformer_class="StandardScaler",
            fit_timestamp="2024-01-01T00:00:00",
            fit_data_hash="abc123",
            parameters={"mean_": [50000.0], "scale_": [20000.0]},
        )
        assert artifact.artifact_id == "income_standard"
        assert artifact.artifact_type == "scaler"
        assert artifact.transformer_class == "StandardScaler"

    def test_to_dict_serialization(self):
        artifact = FitArtifact(
            artifact_id="region_onehot",
            artifact_type="encoder",
            target_column="region",
            transformer_class="LabelEncoder",
            fit_timestamp="2024-01-01T00:00:00",
            fit_data_hash="def456",
            parameters={"classes_": ["East", "North", "South", "West"]},
        )
        d = artifact.to_dict()
        assert d["artifact_id"] == "region_onehot"
        assert d["parameters"]["classes_"] == ["East", "North", "South", "West"]

    def test_from_dict_deserialization(self):
        data = {
            "artifact_id": "amount_minmax",
            "artifact_type": "scaler",
            "target_column": "amount",
            "transformer_class": "MinMaxScaler",
            "fit_timestamp": "2024-01-01T00:00:00",
            "fit_data_hash": "ghi789",
            "parameters": {"data_min_": [0.0], "data_max_": [1000.0]},
            "file_path": "scalers/amount_minmax.pkl",
        }
        artifact = FitArtifact.from_dict(data)
        assert artifact.artifact_id == "amount_minmax"
        assert artifact.file_path == "scalers/amount_minmax.pkl"


class TestFitArtifactRegistryBasic:
    def test_creates_empty_registry(self, tmp_artifacts_dir):
        registry = FitArtifactRegistry(tmp_artifacts_dir)
        assert len(registry.get_manifest()) == 0

    def test_artifacts_dir_created_on_init(self, tmp_artifacts_dir):
        FitArtifactRegistry(tmp_artifacts_dir)
        assert tmp_artifacts_dir.exists()

    def test_subdirectories_created(self, tmp_artifacts_dir):
        FitArtifactRegistry(tmp_artifacts_dir)
        assert (tmp_artifacts_dir / "scalers").exists()
        assert (tmp_artifacts_dir / "encoders").exists()
        assert (tmp_artifacts_dir / "reducers").exists()


class TestFitArtifactRegistryRegister:
    def test_register_scaler_creates_artifact(self, tmp_artifacts_dir, fitted_standard_scaler):
        registry = FitArtifactRegistry(tmp_artifacts_dir)
        artifact_id = registry.register(
            artifact_type="scaler",
            target_column="income",
            transformer=fitted_standard_scaler,
        )
        assert artifact_id == "income_scaler"
        assert (tmp_artifacts_dir / "scalers" / "income_scaler.pkl").exists()
        assert "income_scaler" in registry.get_manifest()

    def test_register_encoder_creates_artifact(self, tmp_artifacts_dir, fitted_label_encoder):
        registry = FitArtifactRegistry(tmp_artifacts_dir)
        artifact_id = registry.register(
            artifact_type="encoder",
            target_column="region",
            transformer=fitted_label_encoder,
        )
        assert artifact_id == "region_encoder"
        assert (tmp_artifacts_dir / "encoders" / "region_encoder.pkl").exists()

    def test_register_pca_reducer_creates_artifact(self, tmp_artifacts_dir):
        from sklearn.decomposition import PCA
        pca = PCA(n_components=3)
        pca.fit(np.random.randn(100, 10))
        registry = FitArtifactRegistry(tmp_artifacts_dir)
        artifact_id = registry.register(
            artifact_type="reducer",
            target_column="description",
            transformer=pca,
        )
        assert artifact_id == "description_reducer"
        assert (tmp_artifacts_dir / "reducers" / "description_reducer.pkl").exists()

    def test_register_with_custom_artifact_id(self, tmp_artifacts_dir, fitted_standard_scaler):
        registry = FitArtifactRegistry(tmp_artifacts_dir)
        artifact_id = registry.register(
            artifact_type="scaler",
            target_column="income",
            transformer=fitted_standard_scaler,
            artifact_id="custom_income_scaler_v2",
        )
        assert artifact_id == "custom_income_scaler_v2"
        assert (tmp_artifacts_dir / "scalers" / "custom_income_scaler_v2.pkl").exists()

    def test_register_stores_transformer_parameters(self, tmp_artifacts_dir, fitted_standard_scaler):
        registry = FitArtifactRegistry(tmp_artifacts_dir)
        registry.register(
            artifact_type="scaler",
            target_column="income",
            transformer=fitted_standard_scaler,
        )
        manifest = registry.get_manifest()
        artifact = manifest["income_scaler"]
        assert "mean_" in artifact.parameters
        assert "scale_" in artifact.parameters

    def test_register_duplicate_artifact_id_raises(self, tmp_artifacts_dir, fitted_standard_scaler):
        registry = FitArtifactRegistry(tmp_artifacts_dir)
        registry.register(
            artifact_type="scaler",
            target_column="income",
            transformer=fitted_standard_scaler,
        )
        with pytest.raises(ValueError, match="already exists"):
            registry.register(
                artifact_type="scaler",
                target_column="income",
                transformer=fitted_standard_scaler,
            )

    def test_register_with_overwrite_replaces_artifact(self, tmp_artifacts_dir, fitted_standard_scaler):
        registry = FitArtifactRegistry(tmp_artifacts_dir)
        registry.register(
            artifact_type="scaler",
            target_column="income",
            transformer=fitted_standard_scaler,
        )
        scaler2 = MinMaxScaler()
        scaler2.fit(np.array([[0], [100]]))
        artifact_id = registry.register(
            artifact_type="scaler",
            target_column="income",
            transformer=scaler2,
            overwrite=True,
        )
        assert artifact_id == "income_scaler"
        loaded = registry.load("income_scaler")
        assert isinstance(loaded, MinMaxScaler)


class TestFitArtifactRegistryLoad:
    def test_load_returns_fitted_transformer(self, tmp_artifacts_dir, fitted_standard_scaler):
        registry = FitArtifactRegistry(tmp_artifacts_dir)
        registry.register(
            artifact_type="scaler",
            target_column="income",
            transformer=fitted_standard_scaler,
        )
        loaded = registry.load("income_scaler")
        assert isinstance(loaded, StandardScaler)
        np.testing.assert_array_almost_equal(loaded.mean_, fitted_standard_scaler.mean_)

    def test_load_encoder_preserves_categories(self, tmp_artifacts_dir, fitted_label_encoder):
        registry = FitArtifactRegistry(tmp_artifacts_dir)
        registry.register(
            artifact_type="encoder",
            target_column="category",
            transformer=fitted_label_encoder,
        )
        loaded = registry.load("category_encoder")
        assert list(loaded.classes_) == ["bird", "cat", "dog"]

    def test_load_missing_artifact_raises(self, tmp_artifacts_dir):
        registry = FitArtifactRegistry(tmp_artifacts_dir)
        with pytest.raises(KeyError, match="not found"):
            registry.load("nonexistent_artifact")

    def test_loaded_transformer_produces_same_output(self, tmp_artifacts_dir, fitted_standard_scaler):
        registry = FitArtifactRegistry(tmp_artifacts_dir)
        registry.register(
            artifact_type="scaler",
            target_column="value",
            transformer=fitted_standard_scaler,
        )
        test_data = np.array([[1, 2], [7, 8]])
        original_result = fitted_standard_scaler.transform(test_data)
        loaded = registry.load("value_scaler")
        loaded_result = loaded.transform(test_data)
        np.testing.assert_array_almost_equal(original_result, loaded_result)


class TestFitArtifactRegistryManifest:
    def test_get_manifest_returns_all_artifacts(self, tmp_artifacts_dir, fitted_standard_scaler, fitted_label_encoder):
        registry = FitArtifactRegistry(tmp_artifacts_dir)
        registry.register(artifact_type="scaler", target_column="income", transformer=fitted_standard_scaler)
        registry.register(artifact_type="encoder", target_column="region", transformer=fitted_label_encoder)
        manifest = registry.get_manifest()
        assert len(manifest) == 2
        assert "income_scaler" in manifest
        assert "region_encoder" in manifest

    def test_save_manifest_creates_yaml(self, tmp_artifacts_dir, fitted_standard_scaler):
        registry = FitArtifactRegistry(tmp_artifacts_dir)
        registry.register(artifact_type="scaler", target_column="income", transformer=fitted_standard_scaler)
        registry.save_manifest()
        assert (tmp_artifacts_dir / "manifest.yaml").exists()

    def test_manifest_roundtrip_yaml(self, tmp_artifacts_dir, fitted_standard_scaler, fitted_label_encoder):
        registry = FitArtifactRegistry(tmp_artifacts_dir)
        registry.register(artifact_type="scaler", target_column="income", transformer=fitted_standard_scaler)
        registry.register(artifact_type="encoder", target_column="region", transformer=fitted_label_encoder)
        registry.save_manifest()
        loaded_registry = FitArtifactRegistry.load_manifest(tmp_artifacts_dir / "manifest.yaml")
        assert len(loaded_registry.get_manifest()) == 2
        assert "income_scaler" in loaded_registry.get_manifest()
        assert "region_encoder" in loaded_registry.get_manifest()

    def test_loaded_registry_can_load_transformers(self, tmp_artifacts_dir, fitted_standard_scaler):
        registry = FitArtifactRegistry(tmp_artifacts_dir)
        registry.register(artifact_type="scaler", target_column="income", transformer=fitted_standard_scaler)
        registry.save_manifest()
        loaded_registry = FitArtifactRegistry.load_manifest(tmp_artifacts_dir / "manifest.yaml")
        scaler = loaded_registry.load("income_scaler")
        assert isinstance(scaler, StandardScaler)

    def test_manifest_includes_fit_timestamp(self, tmp_artifacts_dir, fitted_standard_scaler):
        registry = FitArtifactRegistry(tmp_artifacts_dir)
        registry.register(artifact_type="scaler", target_column="income", transformer=fitted_standard_scaler)
        manifest = registry.get_manifest()
        assert manifest["income_scaler"].fit_timestamp is not None


class TestFitArtifactRegistryEdgeCases:
    def test_empty_manifest_load_returns_empty_registry(self, tmp_artifacts_dir):
        registry = FitArtifactRegistry(tmp_artifacts_dir)
        registry.save_manifest()
        loaded = FitArtifactRegistry.load_manifest(tmp_artifacts_dir / "manifest.yaml")
        assert len(loaded.get_manifest()) == 0

    def test_load_manifest_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            FitArtifactRegistry.load_manifest(tmp_path / "nonexistent.yaml")

    def test_register_unknown_artifact_type_raises(self, tmp_artifacts_dir, fitted_standard_scaler):
        registry = FitArtifactRegistry(tmp_artifacts_dir)
        with pytest.raises(ValueError, match="Unknown artifact type"):
            registry.register(
                artifact_type="unknown",
                target_column="col",
                transformer=fitted_standard_scaler,
            )

    def test_multiple_columns_same_type(self, tmp_artifacts_dir):
        registry = FitArtifactRegistry(tmp_artifacts_dir)
        scaler1 = StandardScaler().fit(np.array([[1], [2], [3]]))
        scaler2 = StandardScaler().fit(np.array([[10], [20], [30]]))
        registry.register(artifact_type="scaler", target_column="col_a", transformer=scaler1)
        registry.register(artifact_type="scaler", target_column="col_b", transformer=scaler2)
        assert len(registry.get_manifest()) == 2
        loaded_a = registry.load("col_a_scaler")
        loaded_b = registry.load("col_b_scaler")
        assert loaded_a.mean_[0] != loaded_b.mean_[0]

    def test_has_artifact_returns_true_for_existing(self, tmp_artifacts_dir, fitted_standard_scaler):
        registry = FitArtifactRegistry(tmp_artifacts_dir)
        registry.register(artifact_type="scaler", target_column="income", transformer=fitted_standard_scaler)
        assert registry.has_artifact("income_scaler")
        assert not registry.has_artifact("nonexistent")

    def test_get_artifact_info_returns_artifact(self, tmp_artifacts_dir, fitted_standard_scaler):
        registry = FitArtifactRegistry(tmp_artifacts_dir)
        registry.register(artifact_type="scaler", target_column="income", transformer=fitted_standard_scaler)
        info = registry.get_artifact_info("income_scaler")
        assert info.artifact_id == "income_scaler"
        assert info.target_column == "income"


class TestFitArtifactRegistryDataHash:
    def test_register_computes_data_hash(self, tmp_artifacts_dir, fitted_standard_scaler):
        registry = FitArtifactRegistry(tmp_artifacts_dir)
        registry.register(
            artifact_type="scaler",
            target_column="income",
            transformer=fitted_standard_scaler,
        )
        manifest = registry.get_manifest()
        assert manifest["income_scaler"].fit_data_hash is not None
        assert len(manifest["income_scaler"].fit_data_hash) > 0

    def test_same_transformer_same_hash(self, tmp_artifacts_dir):
        scaler1 = StandardScaler().fit(np.array([[1], [2], [3]]))
        scaler2 = StandardScaler().fit(np.array([[1], [2], [3]]))
        registry = FitArtifactRegistry(tmp_artifacts_dir)
        registry.register(artifact_type="scaler", target_column="col1", transformer=scaler1)
        registry.register(artifact_type="scaler", target_column="col2", transformer=scaler2)
        manifest = registry.get_manifest()
        assert manifest["col1_scaler"].fit_data_hash == manifest["col2_scaler"].fit_data_hash


class TestFitArtifactRegistryIntegration:
    def test_training_scoring_consistency(self, tmp_artifacts_dir):
        train_data = np.array([[10], [20], [30], [40], [50]])
        scaler = StandardScaler().fit(train_data)
        registry = FitArtifactRegistry(tmp_artifacts_dir)
        registry.register(artifact_type="scaler", target_column="value", transformer=scaler)
        registry.save_manifest()
        loaded_registry = FitArtifactRegistry.load_manifest(tmp_artifacts_dir / "manifest.yaml")
        loaded_scaler = loaded_registry.load("value_scaler")
        score_data = np.array([[15], [25], [35]])
        expected = scaler.transform(score_data)
        actual = loaded_scaler.transform(score_data)
        np.testing.assert_array_almost_equal(expected, actual)

    def test_encoder_training_scoring_consistency(self, tmp_artifacts_dir):
        encoder = LabelEncoder().fit(["A", "B", "C"])
        registry = FitArtifactRegistry(tmp_artifacts_dir)
        registry.register(artifact_type="encoder", target_column="category", transformer=encoder)
        registry.save_manifest()
        loaded_registry = FitArtifactRegistry.load_manifest(tmp_artifacts_dir / "manifest.yaml")
        loaded_encoder = loaded_registry.load("category_encoder")
        assert list(encoder.transform(["A", "B", "C"])) == list(loaded_encoder.transform(["A", "B", "C"]))
