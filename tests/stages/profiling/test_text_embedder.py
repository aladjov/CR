"""Tests for TextEmbedder - TDD first."""
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from customer_retention.core.compat import pd


class TestTextEmbedderLazyLoading:
    def test_model_not_loaded_on_init(self):
        from customer_retention.stages.profiling.text_embedder import TextEmbedder
        with patch.dict('sys.modules', {'sentence_transformers': MagicMock()}):
            embedder = TextEmbedder()
            assert embedder._model is None

    def test_model_loaded_on_first_embed(self):
        from customer_retention.stages.profiling.text_embedder import TextEmbedder
        mock_st = MagicMock()
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1] * 384])
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_st.SentenceTransformer.return_value = mock_model

        with patch.dict('sys.modules', {'sentence_transformers': mock_st}):
            embedder = TextEmbedder()
            embedder.embed(["test"])
            assert embedder._model is not None


class TestTextEmbedderEmbed:
    @pytest.fixture
    def mock_embedder(self):
        from customer_retention.stages.profiling.text_embedder import TextEmbedder
        mock_st = MagicMock()
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_st.SentenceTransformer.return_value = mock_model

        with patch.dict('sys.modules', {'sentence_transformers': mock_st}):
            embedder = TextEmbedder()
            embedder._model = mock_model
            yield embedder, mock_model

    def test_embeds_single_text(self, mock_embedder):
        embedder, mock_model = mock_embedder
        mock_model.encode.return_value = np.array([[0.1] * 384])
        result = embedder.embed(["hello world"])
        assert result.shape == (1, 384)

    def test_embeds_list_of_texts(self, mock_embedder):
        embedder, mock_model = mock_embedder
        mock_model.encode.return_value = np.array([[0.1] * 384, [0.2] * 384, [0.3] * 384])
        result = embedder.embed(["text1", "text2", "text3"])
        assert result.shape == (3, 384)

    def test_handles_empty_string(self, mock_embedder):
        embedder, mock_model = mock_embedder
        mock_model.encode.return_value = np.array([[0.0] * 384])
        result = embedder.embed([""])
        assert result.shape == (1, 384)

    def test_handles_none_values(self, mock_embedder):
        embedder, mock_model = mock_embedder
        mock_model.encode.return_value = np.array([[0.0] * 384, [0.1] * 384])
        result = embedder.embed([None, "text"])
        mock_model.encode.assert_called_once()
        call_args = mock_model.encode.call_args[0][0]
        assert call_args[0] == ""
        assert call_args[1] == "text"

    def test_handles_whitespace_only(self, mock_embedder):
        embedder, mock_model = mock_embedder
        mock_model.encode.return_value = np.array([[0.0] * 384])
        embedder.embed(["   "])
        call_args = mock_model.encode.call_args[0][0]
        assert call_args[0] == ""


class TestTextEmbedderEmbedColumn:
    def test_embeds_dataframe_column(self):
        from customer_retention.stages.profiling.text_embedder import TextEmbedder
        mock_st = MagicMock()
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1] * 384, [0.2] * 384, [0.3] * 384])
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_st.SentenceTransformer.return_value = mock_model

        with patch.dict('sys.modules', {'sentence_transformers': mock_st}):
            embedder = TextEmbedder()
            embedder._model = mock_model
            df = pd.DataFrame({"text": ["hello", "world", "test"]})
            result = embedder.embed_column(df, "text")
            assert result.shape == (3, 384)

    def test_handles_null_values_in_column(self):
        from customer_retention.stages.profiling.text_embedder import TextEmbedder
        mock_st = MagicMock()
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1] * 384, [0.0] * 384, [0.2] * 384])
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_st.SentenceTransformer.return_value = mock_model

        with patch.dict('sys.modules', {'sentence_transformers': mock_st}):
            embedder = TextEmbedder()
            embedder._model = mock_model
            df = pd.DataFrame({"text": ["hello", None, "test"]})
            result = embedder.embed_column(df, "text")
            assert result.shape == (3, 384)
            call_args = mock_model.encode.call_args[0][0]
            assert call_args[1] == ""


class TestTextEmbedderProperties:
    def test_embedding_dim_returns_model_dimension(self):
        from customer_retention.stages.profiling.text_embedder import TextEmbedder
        mock_st = MagicMock()
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_st.SentenceTransformer.return_value = mock_model

        with patch.dict('sys.modules', {'sentence_transformers': mock_st}):
            embedder = TextEmbedder()
            embedder._model = mock_model
            assert embedder.embedding_dim == 384

    def test_default_model_name(self):
        from customer_retention.stages.profiling.text_embedder import TextEmbedder
        embedder = TextEmbedder.__new__(TextEmbedder)
        embedder.model_name = TextEmbedder.DEFAULT_MODEL
        assert embedder.model_name == "all-MiniLM-L6-v2"

    def test_custom_model_name(self):
        from customer_retention.stages.profiling.text_embedder import TextEmbedder
        with patch.dict('sys.modules', {'sentence_transformers': MagicMock()}):
            embedder = TextEmbedder(model_name="paraphrase-MiniLM-L6-v2")
            assert embedder.model_name == "paraphrase-MiniLM-L6-v2"


class TestEmbeddingModelPresets:
    def test_presets_exist(self):
        from customer_retention.stages.profiling.text_embedder import EMBEDDING_MODELS
        assert "minilm" in EMBEDDING_MODELS
        assert "qwen3-0.6b" in EMBEDDING_MODELS

    def test_preset_has_required_fields(self):
        from customer_retention.stages.profiling.text_embedder import EMBEDDING_MODELS
        for name, preset in EMBEDDING_MODELS.items():
            assert "model_name" in preset
            assert "embedding_dim" in preset
            assert "size_mb" in preset
            assert "description" in preset

    def test_minilm_preset_values(self):
        from customer_retention.stages.profiling.text_embedder import EMBEDDING_MODELS
        minilm = EMBEDDING_MODELS["minilm"]
        assert minilm["model_name"] == "all-MiniLM-L6-v2"
        assert minilm["embedding_dim"] == 384
        assert minilm["size_mb"] == 90

    def test_qwen3_preset_values(self):
        from customer_retention.stages.profiling.text_embedder import EMBEDDING_MODELS
        qwen = EMBEDDING_MODELS["qwen3-0.6b"]
        assert qwen["model_name"] == "Qwen/Qwen3-Embedding-0.6B"
        assert qwen["embedding_dim"] == 1024
        assert qwen["size_mb"] == 1200


class TestTextEmbedderFromPreset:
    def test_creates_from_minilm_preset(self):
        from customer_retention.stages.profiling.text_embedder import TextEmbedder
        with patch.dict('sys.modules', {'sentence_transformers': MagicMock()}):
            embedder = TextEmbedder.from_preset("minilm")
            assert embedder.model_name == "all-MiniLM-L6-v2"

    def test_creates_from_qwen3_preset(self):
        from customer_retention.stages.profiling.text_embedder import TextEmbedder
        with patch.dict('sys.modules', {'sentence_transformers': MagicMock()}):
            embedder = TextEmbedder.from_preset("qwen3-0.6b")
            assert embedder.model_name == "Qwen/Qwen3-Embedding-0.6B"

    def test_raises_for_unknown_preset(self):
        from customer_retention.stages.profiling.text_embedder import TextEmbedder
        with pytest.raises(ValueError, match="Unknown preset"):
            TextEmbedder.from_preset("unknown-model")


class TestGetModelInfo:
    def test_get_model_info_without_loading(self):
        from customer_retention.stages.profiling.text_embedder import get_model_info
        info = get_model_info("minilm")
        assert info["model_name"] == "all-MiniLM-L6-v2"
        assert info["embedding_dim"] == 384
        assert info["size_mb"] == 90

    def test_get_model_info_for_qwen(self):
        from customer_retention.stages.profiling.text_embedder import get_model_info
        info = get_model_info("qwen3-0.6b")
        assert info["model_name"] == "Qwen/Qwen3-Embedding-0.6B"
        assert info["embedding_dim"] == 1024

    def test_list_available_models(self):
        from customer_retention.stages.profiling.text_embedder import list_available_models
        models = list_available_models()
        assert "minilm" in models
        assert "qwen3-0.6b" in models


class TestTextEmbedderWithQwen:
    def test_qwen_model_lazy_loading(self):
        from customer_retention.stages.profiling.text_embedder import TextEmbedder
        with patch.dict('sys.modules', {'sentence_transformers': MagicMock()}):
            embedder = TextEmbedder(model_name="Qwen/Qwen3-Embedding-0.6B")
            assert embedder._model is None
            assert embedder.model_name == "Qwen/Qwen3-Embedding-0.6B"

    def test_qwen_embed_returns_correct_shape(self):
        from customer_retention.stages.profiling.text_embedder import TextEmbedder
        mock_st = MagicMock()
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1] * 1024, [0.2] * 1024])
        mock_model.get_sentence_embedding_dimension.return_value = 1024
        mock_st.SentenceTransformer.return_value = mock_model

        with patch.dict('sys.modules', {'sentence_transformers': mock_st}):
            embedder = TextEmbedder(model_name="Qwen/Qwen3-Embedding-0.6B")
            embedder._model = mock_model
            result = embedder.embed(["text1", "text2"])
            assert result.shape == (2, 1024)
