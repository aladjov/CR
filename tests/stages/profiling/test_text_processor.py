"""Tests for TextColumnProcessor - TDD first."""
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from customer_retention.core.compat import pd


class TestTextProcessingConfig:
    def test_default_values(self):
        from customer_retention.stages.profiling.text_processor import TextProcessingConfig
        config = TextProcessingConfig()
        assert config.embedding_model == "all-MiniLM-L6-v2"
        assert config.variance_threshold == 0.95
        assert config.max_components is None
        assert config.min_components == 2
        assert config.batch_size == 32

    def test_custom_values(self):
        from customer_retention.stages.profiling.text_processor import TextProcessingConfig
        config = TextProcessingConfig(
            embedding_model="custom-model",
            variance_threshold=0.80,
            max_components=10,
            min_components=3,
            batch_size=64
        )
        assert config.embedding_model == "custom-model"
        assert config.variance_threshold == 0.80
        assert config.max_components == 10


class TestTextColumnResult:
    def test_dataclass_fields(self):
        from customer_retention.stages.profiling.text_processor import TextColumnResult
        result = TextColumnResult(
            column_name="description",
            n_components=5,
            explained_variance=0.92,
            component_columns=["description_pc1", "description_pc2", "description_pc3",
                             "description_pc4", "description_pc5"],
            embeddings_shape=(100, 384),
            sample_size=100
        )
        assert result.column_name == "description"
        assert result.n_components == 5
        assert len(result.component_columns) == 5


class TestTextColumnProcessorInit:
    def test_default_config(self):
        from customer_retention.stages.profiling.text_processor import TextColumnProcessor
        processor = TextColumnProcessor()
        assert processor.config.variance_threshold == 0.95

    def test_custom_config(self):
        from customer_retention.stages.profiling.text_processor import TextColumnProcessor, TextProcessingConfig
        config = TextProcessingConfig(variance_threshold=0.80)
        processor = TextColumnProcessor(config)
        assert processor.config.variance_threshold == 0.80


class TestTextColumnProcessorProcessColumn:
    @pytest.fixture
    def mock_processor(self):
        from customer_retention.stages.profiling.text_processor import TextColumnProcessor
        mock_st = MagicMock()
        mock_model = MagicMock()
        mock_model.encode.return_value = np.random.randn(10, 384)
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_st.SentenceTransformer.return_value = mock_model

        with patch.dict('sys.modules', {'sentence_transformers': mock_st}):
            processor = TextColumnProcessor()
            processor._embedder = MagicMock()
            processor._embedder.embed_column.return_value = np.random.randn(10, 384)
            processor._embedder.embedding_dim = 384
            yield processor

    def test_returns_dataframe_and_result(self, mock_processor):
        from customer_retention.stages.profiling.text_processor import TextColumnResult
        df = pd.DataFrame({"text": ["hello " * 20] * 10})
        result_df, result = mock_processor.process_column(df, "text")
        assert isinstance(result_df, pd.DataFrame)
        assert isinstance(result, TextColumnResult)

    def test_adds_pc_columns_to_dataframe(self, mock_processor):
        df = pd.DataFrame({"text": ["hello " * 20] * 10, "other": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
        result_df, result = mock_processor.process_column(df, "text")
        for col in result.component_columns:
            assert col in result_df.columns
        assert "other" in result_df.columns

    def test_result_has_correct_metadata(self, mock_processor):
        df = pd.DataFrame({"text": ["hello " * 20] * 10})
        _, result = mock_processor.process_column(df, "text")
        assert result.column_name == "text"
        assert result.n_components >= 2
        assert result.sample_size == 10


class TestTextColumnProcessorProcessAllColumns:
    @pytest.fixture
    def mock_processor(self):
        from customer_retention.stages.profiling.text_processor import TextColumnProcessor
        mock_st = MagicMock()
        mock_model = MagicMock()
        mock_model.encode.return_value = np.random.randn(10, 384)
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_st.SentenceTransformer.return_value = mock_model

        with patch.dict('sys.modules', {'sentence_transformers': mock_st}):
            processor = TextColumnProcessor()
            processor._embedder = MagicMock()
            processor._embedder.embed_column.return_value = np.random.randn(10, 384)
            processor._embedder.embedding_dim = 384
            yield processor

    def test_processes_multiple_columns(self, mock_processor):
        df = pd.DataFrame({
            "text1": ["hello " * 20] * 10,
            "text2": ["world " * 20] * 10,
            "numeric": range(10)
        })
        result_df, results = mock_processor.process_all_text_columns(df, ["text1", "text2"])
        assert len(results) == 2
        assert results[0].column_name == "text1"
        assert results[1].column_name == "text2"

    def test_each_column_gets_own_pcs(self, mock_processor):
        df = pd.DataFrame({
            "text1": ["hello " * 20] * 10,
            "text2": ["world " * 20] * 10
        })
        result_df, results = mock_processor.process_all_text_columns(df, ["text1", "text2"])
        text1_cols = [c for c in result_df.columns if c.startswith("text1_pc")]
        text2_cols = [c for c in result_df.columns if c.startswith("text2_pc")]
        assert len(text1_cols) >= 2
        assert len(text2_cols) >= 2


class TestTextColumnProcessorReducerReuse:
    @pytest.fixture
    def mock_processor(self):
        from customer_retention.stages.profiling.text_processor import TextColumnProcessor
        mock_st = MagicMock()
        mock_model = MagicMock()
        mock_model.encode.return_value = np.random.randn(10, 384)
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_st.SentenceTransformer.return_value = mock_model

        with patch.dict('sys.modules', {'sentence_transformers': mock_st}):
            processor = TextColumnProcessor()
            processor._embedder = MagicMock()
            processor._embedder.embed_column.return_value = np.random.randn(10, 384)
            processor._embedder.embedding_dim = 384
            yield processor

    def test_stores_fitted_reducer(self, mock_processor):
        df = pd.DataFrame({"text": ["hello " * 20] * 10})
        mock_processor.process_column(df, "text")
        assert "text" in mock_processor._reducers

    def test_reuses_reducer_when_fit_false(self, mock_processor):
        df = pd.DataFrame({"text": ["hello " * 20] * 10})
        mock_processor.process_column(df, "text", fit=True)
        original_reducer = mock_processor._reducers["text"]
        mock_processor.process_column(df, "text", fit=False)
        assert mock_processor._reducers["text"] is original_reducer
