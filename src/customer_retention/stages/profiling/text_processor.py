from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from customer_retention.core.compat import DataFrame

from .text_embedder import TextEmbedder
from .text_reducer import TextDimensionalityReducer

if TYPE_CHECKING:
    from customer_retention.artifacts import FitArtifactRegistry


@dataclass
class TextProcessingConfig:
    embedding_model: str = "all-MiniLM-L6-v2"
    variance_threshold: float = 0.95
    max_components: Optional[int] = None
    min_components: int = 2
    batch_size: int = 32


@dataclass
class TextColumnResult:
    column_name: str
    n_components: int
    explained_variance: float
    component_columns: List[str]
    embeddings_shape: Tuple[int, int]
    sample_size: int


class TextColumnProcessor:
    def __init__(self, config: Optional[TextProcessingConfig] = None,
                 registry: Optional["FitArtifactRegistry"] = None):
        self.config = config or TextProcessingConfig()
        self.registry = registry
        self._embedder: Optional[TextEmbedder] = None
        self._reducers: Dict[str, TextDimensionalityReducer] = {}

    @property
    def embedder(self) -> TextEmbedder:
        if self._embedder is None:
            self._embedder = TextEmbedder(self.config.embedding_model)
        return self._embedder

    def process_column(self, df: DataFrame, column: str,
                       fit: bool = True) -> Tuple[DataFrame, TextColumnResult]:
        embeddings = self.embedder.embed_column(df, column, batch_size=self.config.batch_size)
        reducer = self._get_or_create_reducer(column, fit)
        if fit:
            result = reducer.fit_transform(embeddings, column)
            self._register_reducer(column, reducer)
        else:
            result = reducer.transform(embeddings, column)
        output_df = self._add_components_to_df(df, result.components, result.component_names)
        return output_df, TextColumnResult(
            column_name=column,
            n_components=result.n_components,
            explained_variance=result.cumulative_variance,
            component_columns=result.component_names,
            embeddings_shape=embeddings.shape,
            sample_size=len(df)
        )

    def _register_reducer(self, column: str, reducer: TextDimensionalityReducer) -> None:
        if self.registry is None or reducer._pca is None:
            return
        self.registry.register(
            artifact_type="reducer",
            target_column=column,
            transformer=reducer._pca
        )

    def process_all_text_columns(self, df: DataFrame,
                                  text_columns: List[str]) -> Tuple[DataFrame, List[TextColumnResult]]:
        results = []
        output_df = df.copy()
        for column in text_columns:
            output_df, result = self.process_column(output_df, column)
            results.append(result)
        return output_df, results

    def _get_or_create_reducer(self, column: str, fit: bool) -> TextDimensionalityReducer:
        if fit:
            self._reducers[column] = TextDimensionalityReducer(
                variance_threshold=self.config.variance_threshold,
                max_components=self.config.max_components,
                min_components=self.config.min_components
            )
            return self._reducers[column]
        if column in self._reducers:
            return self._reducers[column]
        if self.registry is not None and self.registry.has_artifact(f"{column}_reducer"):
            pca = self.registry.load(f"{column}_reducer")
            reducer = TextDimensionalityReducer(
                variance_threshold=self.config.variance_threshold,
                max_components=self.config.max_components,
                min_components=self.config.min_components
            )
            reducer._pca = pca
            reducer._fitted = True
            self._reducers[column] = reducer
            return reducer
        self._reducers[column] = TextDimensionalityReducer(
            variance_threshold=self.config.variance_threshold,
            max_components=self.config.max_components,
            min_components=self.config.min_components
        )
        return self._reducers[column]

    def _add_components_to_df(self, df: DataFrame, components, names: List[str]) -> DataFrame:
        output_df = df.copy()
        for i, name in enumerate(names):
            output_df[name] = components[:, i]
        return output_df
