import pytest
import yaml

from customer_retention.transforms.artifact_store import ArtifactStore


@pytest.fixture
def store(tmp_path):
    return ArtifactStore(str(tmp_path / "artifacts"))


class TestArtifactStore:
    def test_register_and_load(self, store):
        store.register("scaler", "age", {"mean": 50, "std": 10})
        loaded = store.load("age_scaler")
        assert loaded == {"mean": 50, "std": 10}

    def test_has_artifact(self, store):
        assert not store.has("age_scaler")
        store.register("scaler", "age", {"mean": 50})
        assert store.has("age_scaler")

    def test_load_missing_raises(self, store):
        with pytest.raises(KeyError, match="not found"):
            store.load("nonexistent")

    def test_save_and_load_manifest(self, tmp_path):
        store = ArtifactStore(str(tmp_path / "artifacts"))
        store.register("scaler", "age", [1, 2, 3])
        store.register("encoder", "status", {"a": 0})
        store.save_manifest()

        manifest_path = tmp_path / "artifacts" / "manifest.yaml"
        assert manifest_path.exists()

        store2 = ArtifactStore.from_manifest(str(manifest_path))
        assert store2.has("age_scaler")
        assert store2.has("status_encoder")
        assert store2.load("age_scaler") == [1, 2, 3]

    def test_from_manifest_empty(self, tmp_path):
        manifest_path = tmp_path / "manifest.yaml"
        manifest_path.write_text(yaml.dump(None))
        store = ArtifactStore.from_manifest(str(manifest_path))
        assert not store.has("anything")
