from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from customer_retention.artifacts import FitArtifactRegistry
from customer_retention.stages.profiling.text_processor import TextColumnProcessor


@pytest.fixture
def tmp_artifacts_dir(tmp_path):
    return tmp_path / "artifacts" / "abc12345"


@pytest.fixture
def sample_text_df():
    return pd.DataFrame({
        "customer_id": range(50),
        "description": [
            "Great product, very satisfied with quality",
            "Bad experience, would not recommend",
            "Average service, nothing special",
            "Excellent customer support team",
            "Poor delivery time, product damaged",
        ] * 10,
        "feedback": [
            "positive feedback",
            "negative feedback",
            "neutral feedback",
            "excellent feedback",
            "poor feedback",
        ] * 10,
    })


@pytest.fixture
def mock_embedder():
    def mock_embed_column(df, column, batch_size=32):
        np.random.seed(42)
        return np.random.randn(len(df), 384)
    mock = MagicMock()
    mock.embed_column = mock_embed_column
    return mock


@pytest.fixture
def patched_processor(tmp_artifacts_dir, mock_embedder):
    def create_processor(registry=None):
        processor = TextColumnProcessor(registry=registry)
        processor._embedder = mock_embedder
        return processor
    return create_processor


class TestTextColumnProcessorWithRegistry:
    def test_processor_accepts_registry(self, tmp_artifacts_dir):
        registry = FitArtifactRegistry(tmp_artifacts_dir)
        processor = TextColumnProcessor(registry=registry)
        assert processor.registry is registry

    def test_processor_works_without_registry(self):
        processor = TextColumnProcessor()
        assert processor.registry is None

    def test_process_column_fit_registers_reducer(self, tmp_artifacts_dir, sample_text_df, mock_embedder):
        registry = FitArtifactRegistry(tmp_artifacts_dir)
        processor = TextColumnProcessor(registry=registry)
        processor._embedder = mock_embedder
        processor.process_column(sample_text_df, "description", fit=True)
        assert registry.has_artifact("description_reducer")

    def test_process_column_no_fit_skips_registration(self, tmp_artifacts_dir, sample_text_df, mock_embedder):
        registry = FitArtifactRegistry(tmp_artifacts_dir)
        processor = TextColumnProcessor(registry=registry)
        processor._embedder = mock_embedder
        processor.process_column(sample_text_df, "description", fit=True)
        initial_count = len(registry.get_manifest())
        processor.process_column(sample_text_df, "description", fit=False)
        assert len(registry.get_manifest()) == initial_count

    def test_process_column_transform_loads_from_registry(self, tmp_artifacts_dir, sample_text_df, mock_embedder):
        registry = FitArtifactRegistry(tmp_artifacts_dir)
        processor1 = TextColumnProcessor(registry=registry)
        processor1._embedder = mock_embedder
        df_train, result_train = processor1.process_column(sample_text_df, "description", fit=True)
        registry.save_manifest()
        loaded_registry = FitArtifactRegistry.load_manifest(tmp_artifacts_dir / "manifest.yaml")
        processor2 = TextColumnProcessor(registry=loaded_registry)
        processor2._embedder = mock_embedder
        df_score, result_score = processor2.process_column(sample_text_df, "description", fit=False)
        assert result_train.n_components == result_score.n_components

    def test_registry_none_skips_registration(self, sample_text_df, mock_embedder):
        processor = TextColumnProcessor(registry=None)
        processor._embedder = mock_embedder
        processor.process_column(sample_text_df, "description", fit=True)

    def test_explained_variance_stored_in_artifact(self, tmp_artifacts_dir, sample_text_df, mock_embedder):
        registry = FitArtifactRegistry(tmp_artifacts_dir)
        processor = TextColumnProcessor(registry=registry)
        processor._embedder = mock_embedder
        processor.process_column(sample_text_df, "description", fit=True)
        artifact = registry.get_artifact_info("description_reducer")
        assert "explained_variance_ratio_" in artifact.parameters

    def test_multiple_text_columns_registered(self, tmp_artifacts_dir, sample_text_df, mock_embedder):
        registry = FitArtifactRegistry(tmp_artifacts_dir)
        processor = TextColumnProcessor(registry=registry)
        processor._embedder = mock_embedder
        processor.process_all_text_columns(sample_text_df, ["description", "feedback"])
        assert registry.has_artifact("description_reducer")
        assert registry.has_artifact("feedback_reducer")


class TestTextColumnProcessorTrainingScoringConsistency:
    def test_scoring_produces_same_components_as_training(self, tmp_artifacts_dir, sample_text_df, mock_embedder):
        registry = FitArtifactRegistry(tmp_artifacts_dir)
        processor = TextColumnProcessor(registry=registry)
        processor._embedder = mock_embedder
        df_train, result_train = processor.process_column(sample_text_df, "description", fit=True)
        registry.save_manifest()
        loaded_registry = FitArtifactRegistry.load_manifest(tmp_artifacts_dir / "manifest.yaml")
        processor2 = TextColumnProcessor(registry=loaded_registry)
        processor2._embedder = mock_embedder
        df_score, result_score = processor2.process_column(sample_text_df, "description", fit=False)
        train_components = df_train[[c for c in df_train.columns if c.startswith("description_pc")]]
        score_components = df_score[[c for c in df_score.columns if c.startswith("description_pc")]]
        np.testing.assert_array_almost_equal(train_components.values, score_components.values, decimal=5)

    def test_new_data_uses_trained_reducer(self, tmp_artifacts_dir, sample_text_df, mock_embedder):
        registry = FitArtifactRegistry(tmp_artifacts_dir)
        processor = TextColumnProcessor(registry=registry)
        processor._embedder = mock_embedder
        processor.process_column(sample_text_df, "description", fit=True)
        registry.save_manifest()
        new_df = pd.DataFrame({
            "customer_id": [100, 101],
            "description": ["New test data", "Another test"],
        })
        loaded_registry = FitArtifactRegistry.load_manifest(tmp_artifacts_dir / "manifest.yaml")
        processor2 = TextColumnProcessor(registry=loaded_registry)
        def mock_embed_new(df, column, batch_size=32):
            np.random.seed(123)
            return np.random.randn(len(df), 384)
        mock_new = MagicMock()
        mock_new.embed_column = mock_embed_new
        processor2._embedder = mock_new
        df_score, result_score = processor2.process_column(new_df, "description", fit=False)
        component_cols = [c for c in df_score.columns if c.startswith("description_pc")]
        assert len(component_cols) > 0
        assert len(df_score) == 2
