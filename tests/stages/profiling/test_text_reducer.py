"""Tests for TextDimensionalityReducer - TDD first."""
import numpy as np
import pytest


class TestTextDimensionalityReducerInit:
    def test_default_variance_threshold(self):
        from customer_retention.stages.profiling.text_reducer import TextDimensionalityReducer
        reducer = TextDimensionalityReducer()
        assert reducer.variance_threshold == 0.95

    def test_custom_variance_threshold(self):
        from customer_retention.stages.profiling.text_reducer import TextDimensionalityReducer
        reducer = TextDimensionalityReducer(variance_threshold=0.80)
        assert reducer.variance_threshold == 0.80

    def test_default_min_components(self):
        from customer_retention.stages.profiling.text_reducer import TextDimensionalityReducer
        reducer = TextDimensionalityReducer()
        assert reducer.min_components == 2

    def test_default_max_components_is_none(self):
        from customer_retention.stages.profiling.text_reducer import TextDimensionalityReducer
        reducer = TextDimensionalityReducer()
        assert reducer.max_components is None


class TestTextDimensionalityReducerFit:
    def test_fit_returns_self(self):
        from customer_retention.stages.profiling.text_reducer import TextDimensionalityReducer
        reducer = TextDimensionalityReducer()
        embeddings = np.random.randn(100, 50)
        result = reducer.fit(embeddings)
        assert result is reducer

    def test_fit_sets_fitted_flag(self):
        from customer_retention.stages.profiling.text_reducer import TextDimensionalityReducer
        reducer = TextDimensionalityReducer()
        embeddings = np.random.randn(100, 50)
        reducer.fit(embeddings)
        assert reducer._fitted is True

    def test_respects_variance_threshold(self):
        from customer_retention.stages.profiling.text_reducer import TextDimensionalityReducer
        np.random.seed(42)
        embeddings = np.random.randn(100, 50)
        reducer = TextDimensionalityReducer(variance_threshold=0.50)
        reducer.fit(embeddings)
        assert reducer._pca.n_components_ <= 50
        cumvar = np.sum(reducer._pca.explained_variance_ratio_)
        assert cumvar >= 0.50 or reducer._pca.n_components_ == reducer.min_components


class TestTextDimensionalityReducerConstraints:
    def test_respects_max_components(self):
        from customer_retention.stages.profiling.text_reducer import TextDimensionalityReducer
        np.random.seed(42)
        embeddings = np.random.randn(100, 50)
        reducer = TextDimensionalityReducer(variance_threshold=0.99, max_components=5)
        reducer.fit(embeddings)
        assert reducer._pca.n_components_ <= 5

    def test_respects_min_components(self):
        from customer_retention.stages.profiling.text_reducer import TextDimensionalityReducer
        np.random.seed(42)
        embeddings = np.random.randn(100, 50)
        reducer = TextDimensionalityReducer(variance_threshold=0.01, min_components=3)
        reducer.fit(embeddings)
        assert reducer._pca.n_components_ >= 3

    def test_handles_small_sample_size(self):
        from customer_retention.stages.profiling.text_reducer import TextDimensionalityReducer
        embeddings = np.random.randn(5, 50)
        reducer = TextDimensionalityReducer(min_components=2)
        reducer.fit(embeddings)
        assert reducer._pca.n_components_ <= 5


class TestTextDimensionalityReducerTransform:
    def test_transform_requires_fit(self):
        from customer_retention.stages.profiling.text_reducer import TextDimensionalityReducer
        reducer = TextDimensionalityReducer()
        embeddings = np.random.randn(10, 50)
        with pytest.raises(ValueError, match="fit"):
            reducer.transform(embeddings, "text")

    def test_transform_returns_reduction_result(self):
        from customer_retention.stages.profiling.text_reducer import ReductionResult, TextDimensionalityReducer
        embeddings = np.random.randn(100, 50)
        reducer = TextDimensionalityReducer(variance_threshold=0.5)
        reducer.fit(embeddings)
        result = reducer.transform(embeddings, "text")
        assert isinstance(result, ReductionResult)

    def test_result_has_correct_shape(self):
        from customer_retention.stages.profiling.text_reducer import TextDimensionalityReducer
        embeddings = np.random.randn(100, 50)
        reducer = TextDimensionalityReducer(max_components=5)
        reducer.fit(embeddings)
        result = reducer.transform(embeddings, "text")
        assert result.components.shape[0] == 100
        assert result.components.shape[1] == result.n_components

    def test_component_names_use_prefix(self):
        from customer_retention.stages.profiling.text_reducer import TextDimensionalityReducer
        embeddings = np.random.randn(100, 50)
        reducer = TextDimensionalityReducer(max_components=3)
        reducer.fit(embeddings)
        result = reducer.transform(embeddings, "ticket_text")
        assert result.component_names == ["ticket_text_pc1", "ticket_text_pc2", "ticket_text_pc3"]

    def test_result_has_explained_variance(self):
        from customer_retention.stages.profiling.text_reducer import TextDimensionalityReducer
        embeddings = np.random.randn(100, 50)
        reducer = TextDimensionalityReducer(max_components=5)
        reducer.fit(embeddings)
        result = reducer.transform(embeddings, "text")
        assert len(result.explained_variance_ratio) == result.n_components
        assert result.cumulative_variance > 0


class TestTextDimensionalityReducerFitTransform:
    def test_fit_transform_combines_both(self):
        from customer_retention.stages.profiling.text_reducer import TextDimensionalityReducer
        embeddings = np.random.randn(100, 50)
        reducer = TextDimensionalityReducer(max_components=5)
        result = reducer.fit_transform(embeddings, "text")
        assert reducer._fitted is True
        assert result.n_components == 5


class TestReductionResult:
    def test_dataclass_fields(self):
        from customer_retention.stages.profiling.text_reducer import ReductionResult
        result = ReductionResult(
            components=np.array([[1, 2], [3, 4]]),
            n_components=2,
            explained_variance_ratio=np.array([0.6, 0.3]),
            cumulative_variance=0.9,
            component_names=["pc1", "pc2"]
        )
        assert result.n_components == 2
        assert result.cumulative_variance == 0.9
        assert result.component_names == ["pc1", "pc2"]
