from typing import Dict, List, Optional, Any
import numpy as np

from customer_retention.core.compat import DataFrame


EMBEDDING_MODELS: Dict[str, Dict[str, Any]] = {
    "minilm": {
        "model_name": "all-MiniLM-L6-v2",
        "embedding_dim": 384,
        "size_mb": 90,
        "description": "Fast, lightweight model. Good for CPU and quick experimentation.",
        "gpu_recommended": False,
    },
    "qwen3-0.6b": {
        "model_name": "Qwen/Qwen3-Embedding-0.6B",
        "embedding_dim": 1024,
        "size_mb": 1200,
        "description": "Higher quality embeddings, multilingual. Requires GPU for reasonable speed.",
        "gpu_recommended": True,
    },
    "qwen3-4b": {
        "model_name": "Qwen/Qwen3-Embedding-4B",
        "embedding_dim": 2560,
        "size_mb": 8000,
        "description": "High quality, large model. Requires significant GPU memory (16GB+).",
        "gpu_recommended": True,
    },
    "qwen3-8b": {
        "model_name": "Qwen/Qwen3-Embedding-8B",
        "embedding_dim": 4096,
        "size_mb": 16000,
        "description": "Highest quality, very large model. Requires 32GB+ GPU memory.",
        "gpu_recommended": True,
    },
}


def get_model_info(preset: str) -> Dict[str, Any]:
    if preset not in EMBEDDING_MODELS:
        raise ValueError(f"Unknown preset: {preset}. Available: {list(EMBEDDING_MODELS.keys())}")
    return EMBEDDING_MODELS[preset].copy()


def list_available_models() -> List[str]:
    return list(EMBEDDING_MODELS.keys())


class TextEmbedder:
    DEFAULT_MODEL = "all-MiniLM-L6-v2"

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self._model = None

    @classmethod
    def from_preset(cls, preset: str) -> "TextEmbedder":
        if preset not in EMBEDDING_MODELS:
            raise ValueError(f"Unknown preset: {preset}. Available: {list(EMBEDDING_MODELS.keys())}")
        model_name = EMBEDDING_MODELS[preset]["model_name"]
        return cls(model_name=model_name)

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    @property
    def embedding_dim(self) -> int:
        return self.model.get_sentence_embedding_dimension()

    def embed(self, texts: List[Optional[str]], batch_size: int = 32,
              show_progress: bool = False) -> np.ndarray:
        clean_texts = [self._clean_text(t) for t in texts]
        return self.model.encode(clean_texts, batch_size=batch_size,
                                  show_progress_bar=show_progress)

    def embed_column(self, df: DataFrame, column: str, batch_size: int = 32) -> np.ndarray:
        texts = df[column].fillna("").astype(str).tolist()
        return self.embed(texts, batch_size=batch_size)

    def _clean_text(self, text: Optional[str]) -> str:
        if not isinstance(text, str) or not text.strip():
            return ""
        return text
