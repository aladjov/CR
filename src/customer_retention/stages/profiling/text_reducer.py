from dataclasses import dataclass
from typing import List, Optional
import numpy as np


@dataclass
class ReductionResult:
    components: np.ndarray
    n_components: int
    explained_variance_ratio: np.ndarray
    cumulative_variance: float
    component_names: List[str]


class TextDimensionalityReducer:
    def __init__(self, variance_threshold: float = 0.95,
                 max_components: Optional[int] = None, min_components: int = 2):
        self.variance_threshold = variance_threshold
        self.max_components = max_components
        self.min_components = min_components
        self._pca = None
        self._fitted = False

    def fit(self, embeddings: np.ndarray) -> "TextDimensionalityReducer":
        from sklearn.decomposition import PCA
        n_components = self._compute_n_components(embeddings)
        self._pca = PCA(n_components=n_components)
        self._pca.fit(embeddings)
        self._fitted = True
        return self

    def transform(self, embeddings: np.ndarray, column_prefix: str) -> ReductionResult:
        if not self._fitted:
            raise ValueError("Must call fit() before transform()")
        components = self._pca.transform(embeddings)
        return ReductionResult(
            components=components,
            n_components=self._pca.n_components_,
            explained_variance_ratio=self._pca.explained_variance_ratio_,
            cumulative_variance=float(np.sum(self._pca.explained_variance_ratio_)),
            component_names=[f"{column_prefix}_pc{i+1}" for i in range(self._pca.n_components_)]
        )

    def fit_transform(self, embeddings: np.ndarray, column_prefix: str) -> ReductionResult:
        self.fit(embeddings)
        return self.transform(embeddings, column_prefix)

    def _compute_n_components(self, embeddings: np.ndarray) -> int:
        from sklearn.decomposition import PCA
        n_samples, n_features = embeddings.shape
        max_possible = min(n_samples, n_features)
        full_pca = PCA(n_components=max_possible)
        full_pca.fit(embeddings)
        cumsum = np.cumsum(full_pca.explained_variance_ratio_)
        n_components = int(np.searchsorted(cumsum, self.variance_threshold) + 1)
        n_components = max(n_components, self.min_components)
        if self.max_components:
            n_components = min(n_components, self.max_components)
        return min(n_components, max_possible)
