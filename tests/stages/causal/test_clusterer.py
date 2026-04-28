"""Unit tests for the SHAP-space KMeans clusterer.

The tests target two layers:

1. **Module-level smoke tests** that import the public API and check
   parameter validation paths. These don't need a live Spark session.
2. **Mocked Spark tests** that swap out ``pyspark.ml`` and ``pyspark.sql``
   so the cluster_kmeans / cluster_centroids_raw flows can be exercised
   on CI without PySpark installed.
"""

from __future__ import annotations

from typing import Any, List
from unittest.mock import MagicMock

import pytest

pytest.importorskip("pyspark", reason="PySpark required for causal clusterer tests")

from customer_retention.stages.causal import clusterer
from customer_retention.stages.causal.clusterer import (
    CLUSTER_COL,
    DEFAULT_K_CAP,
    DEFAULT_K_RANGE,
    ClusterCandidate,
    ClusteringResult,
    cluster_kmeans,
)

# ---------------------------------------------------------------------------
# Defaults + dataclasses
# ---------------------------------------------------------------------------


class TestDefaults:
    def test_default_k_range_is_4_to_12(self):
        assert DEFAULT_K_RANGE == (4, 12)

    def test_default_k_cap_is_8(self):
        assert DEFAULT_K_CAP == 8

    def test_cluster_col_constant(self):
        assert CLUSTER_COL == "cluster_id"

    def test_clustering_result_cluster_count_property(self):
        result = ClusteringResult(best_k=5, best_silhouette=0.7)
        assert result.cluster_count == 5

    def test_cluster_candidate_dataclass(self):
        c = ClusterCandidate(k=4, silhouette=0.8)
        assert c.k == 4
        assert c.silhouette == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestClusterKmeansValidation:
    def test_empty_feature_columns_raises(self):
        fake_df = MagicMock(name="DataFrame")
        with pytest.raises(ValueError, match="at least one feature column"):
            cluster_kmeans(fake_df, feature_columns=[])

    def test_empty_centroids_raw_raises(self):
        fake_df = MagicMock(name="DataFrame")
        with pytest.raises(ValueError, match="at least one feature column"):
            clusterer.cluster_centroids_raw(fake_df, feature_columns=[])


# ---------------------------------------------------------------------------
# cluster_kmeans (mocked Spark)
# ---------------------------------------------------------------------------


class _FakeKMeansModel:
    def __init__(self, k: int) -> None:
        self.k = k
        self._centers = [[float(i + j) for j in range(3)] for i in range(k)]

    def transform(self, df: Any) -> Any:
        labelled = MagicMock(name="LabelledDF")
        labelled.drop.return_value = labelled
        return labelled

    def clusterCenters(self) -> List[List[float]]:  # noqa: N802
        return self._centers


class _FakeKMeansBuilder:
    """Mocks ``pyspark.ml.clustering.KMeans``: returns a model whose silhouette grows with k."""

    instances: List[Any] = []

    def __init__(self, **kwargs: Any) -> None:
        self.k = kwargs.get("k", 0)
        self.kwargs = kwargs
        _FakeKMeansBuilder.instances.append(self)

    def fit(self, _df: Any) -> _FakeKMeansModel:
        return _FakeKMeansModel(self.k)


class _FakeEvaluator:
    def __init__(self, **_kwargs: Any) -> None:
        self.calls = 0

    def evaluate(self, _df: Any) -> float:
        # silhouette increasing with order so the loop picks the last k
        self.calls += 1
        return 0.4 + 0.05 * self.calls


class _FakeAssembler:
    def __init__(self, **_kwargs: Any) -> None:
        pass

    def transform(self, df: Any) -> Any:
        cached = MagicMock(name="AssembledDF")
        cached.cache.return_value = cached
        return cached


@pytest.fixture
def patched_spark_ml(monkeypatch):
    import sys

    fake_clustering = MagicMock()
    fake_clustering.KMeans = _FakeKMeansBuilder
    fake_evaluation = MagicMock()
    fake_evaluation.ClusteringEvaluator = _FakeEvaluator
    fake_feature = MagicMock()
    fake_feature.VectorAssembler = _FakeAssembler

    monkeypatch.setitem(sys.modules, "pyspark.ml.clustering", fake_clustering)
    monkeypatch.setitem(sys.modules, "pyspark.ml.evaluation", fake_evaluation)
    monkeypatch.setitem(sys.modules, "pyspark.ml.feature", fake_feature)
    _FakeKMeansBuilder.instances = []
    yield


class TestClusterKmeans:
    def test_runs_silhouette_sweep_within_capped_range(self, patched_spark_ml):
        fake_df = MagicMock(name="ShapDF")
        fake_df.columns = ["shap_a", "shap_b", "shap_c"]
        result = cluster_kmeans(
            fake_df,
            feature_columns=["shap_a", "shap_b", "shap_c"],
            k_range=(2, 10),
            k_cap=4,
        )
        ks = [c.k for c in result.candidates]
        assert ks == [2, 3, 4]

    def test_picks_best_silhouette(self, patched_spark_ml):
        fake_df = MagicMock(name="ShapDF")
        result = cluster_kmeans(
            fake_df,
            feature_columns=["shap_a", "shap_b", "shap_c"],
            k_range=(2, 4),
            k_cap=4,
        )
        # _FakeEvaluator returns increasing values, so the highest k wins
        assert result.best_k == 4

    def test_centroids_are_lists_of_floats(self, patched_spark_ml):
        fake_df = MagicMock(name="ShapDF")
        result = cluster_kmeans(
            fake_df,
            feature_columns=["shap_a", "shap_b", "shap_c"],
            k_range=(2, 3),
            k_cap=3,
        )
        assert all(isinstance(centroid, list) for centroid in result.centroids)
        assert all(isinstance(v, float) for centroid in result.centroids for v in centroid)

    def test_feature_order_preserved(self, patched_spark_ml):
        fake_df = MagicMock(name="ShapDF")
        result = cluster_kmeans(
            fake_df,
            feature_columns=["shap_a", "shap_b", "shap_c"],
            k_range=(2, 2),
            k_cap=2,
        )
        assert result.feature_order == ["shap_a", "shap_b", "shap_c"]

    def test_lower_bound_clipped_to_two(self, patched_spark_ml):
        fake_df = MagicMock(name="ShapDF")
        result = cluster_kmeans(
            fake_df,
            feature_columns=["shap_a"],
            k_range=(1, 3),
            k_cap=3,
        )
        # k_low gets clipped to 2; k_high min(3, cap=3) = 3 → range 2..3
        assert [c.k for c in result.candidates] == [2, 3]

    def test_inverted_range_collapses_to_low(self, patched_spark_ml):
        fake_df = MagicMock(name="ShapDF")
        result = cluster_kmeans(
            fake_df,
            feature_columns=["shap_a"],
            k_range=(5, 3),
            k_cap=4,
        )
        # k_low=5, k_high=min(3,4)=3 → 3<5 → k_high becomes 5 → single candidate
        assert [c.k for c in result.candidates] == [5]


# ---------------------------------------------------------------------------
# cluster_centroids_raw / cluster_size_stats / cluster_target_means / cluster_shap_centroids
# ---------------------------------------------------------------------------


class _FakeRow:
    def __init__(self, data: dict) -> None:
        self._data = data

    def __getitem__(self, key: str) -> Any:
        return self._data[key]


def _make_chain_returning(rows: List[dict], columns: List[str] | None = None) -> Any:
    chain = MagicMock(name="DFChain")
    fake_collect = [_FakeRow(r) for r in rows]
    # groupBy(...).agg(...).orderBy(...).collect()
    chain.groupBy.return_value = chain
    chain.agg.return_value = chain
    chain.orderBy.return_value = chain
    chain.collect.return_value = fake_collect
    chain.count = MagicMock(return_value=chain)  # for cluster_size_stats fallback path: groupBy().count().orderBy().collect()
    # ``columns`` is a real list (not a MagicMock attribute) so callers can
    # control whether ``entity_id in df.columns`` evaluates to True/False
    # deterministically.
    chain.columns = list(columns or [])
    return chain


@pytest.fixture
def patched_pyspark_sql_functions(monkeypatch):
    import sys

    fake_functions = MagicMock(name="pyspark.sql.functions")
    fake_functions.mean = MagicMock(side_effect=lambda c: c)
    fake_functions.col = MagicMock(side_effect=lambda c: c if hasattr(c, "cast") else _FakeCol(c))
    fake_functions.nanvl = MagicMock(side_effect=lambda a, _b: a)
    fake_functions.lit = MagicMock(side_effect=lambda v: _FakeCol(f"lit({v})"))
    fake_functions.abs = MagicMock(side_effect=lambda c: c)
    fake_functions.stddev_pop = MagicMock(side_effect=lambda c: c)
    fake_functions.countDistinct = MagicMock(side_effect=lambda c: c)
    fake_sql = MagicMock(name="pyspark.sql")
    fake_sql.functions = fake_functions
    monkeypatch.setitem(sys.modules, "pyspark.sql", fake_sql)
    yield fake_functions


class _FakeCol:
    def __init__(self, name: str) -> None:
        self.name = name

    def cast(self, _t: str) -> "_FakeCol":
        return self

    def alias(self, name: str) -> "_FakeCol":
        return _FakeCol(name)


def _assert_nanvl_fallback_is_null(fake_functions: Any, *, expected_calls: int) -> None:
    """Assert every ``F.nanvl`` call received a NULL literal as its second arg.

    Using ``F.lit(0.0)`` in place of NULL would make ``F.mean`` include bogus
    zeros in the aggregation instead of skipping NaN rows — the clusterer's
    per-cluster means would silently skew toward zero for any feature with
    NaN rows. Pinning the fallback here stops that regression cold."""
    assert fake_functions.nanvl.call_count == expected_calls
    for call in fake_functions.nanvl.call_args_list:
        fallback = call.args[1]
        assert getattr(fallback, "name", None) == "lit(None)", (
            f"nanvl fallback must be F.lit(None) (DoubleType NULL); got {fallback!r}"
        )


class TestClusterCentroidsRaw:
    def test_returns_per_cluster_means(self, patched_pyspark_sql_functions):
        rows = [
            {CLUSTER_COL: 0, "__mean_0": 1.0, "__mean_1": 2.0},
            {CLUSTER_COL: 1, "__mean_0": 3.0, "__mean_1": 4.0},
        ]
        chain = _make_chain_returning(rows)
        centroids, order = clusterer.cluster_centroids_raw(chain, feature_columns=["a", "b"])
        assert centroids == [[1.0, 2.0], [3.0, 4.0]]
        assert order == ["a", "b"]

    def test_handles_null_means(self, patched_pyspark_sql_functions):
        rows = [{CLUSTER_COL: 0, "__mean_0": None, "__mean_1": 5.0}]
        chain = _make_chain_returning(rows)
        centroids, _ = clusterer.cluster_centroids_raw(chain, feature_columns=["a", "b"])
        assert centroids[0] == [0.0, 5.0]

    def test_mean_wraps_nanvl_so_nan_rows_are_skipped(self, patched_pyspark_sql_functions):
        """Regression: Spark's ``F.mean`` skips NULL but not NaN — a single NaN
        in a gold column poisons the whole cluster mean for that feature. The
        aggregation must wrap each column in ``F.nanvl(col, NULL)`` so NaN
        values are coerced to NULL (which ``F.mean`` then skips), preserving
        the contribution of finite rows.

        The fallback arg of every ``nanvl`` call must be a NULL literal — a
        future refactor that substituted ``F.lit(0.0)`` would include bogus
        zeros in the mean and silently skew centroids toward the origin; this
        assertion pins the NULL fallback so that regression can't slip in."""
        rows = [{CLUSTER_COL: 0, "__mean_0": 1.0, "__mean_1": 2.0}]
        chain = _make_chain_returning(rows)
        clusterer.cluster_centroids_raw(chain, feature_columns=["a", "b"])
        _assert_nanvl_fallback_is_null(patched_pyspark_sql_functions, expected_calls=2)


class TestClusterSizeStats:
    def test_returns_per_cluster_counts(self, patched_pyspark_sql_functions):
        # No ``entity_id`` column on the labelled DataFrame -> fall through
        # to the ``groupBy().count()`` path. Backward-compatible behaviour
        # for fixtures that don't carry the entity key.
        rows = [{CLUSTER_COL: 0, "count": 50}, {CLUSTER_COL: 1, "count": 75}]
        chain = _make_chain_returning(rows)
        result = clusterer.cluster_size_stats(chain)
        assert result == [(0, 50), (1, 75)]
        # ``count()`` (no agg() -> path) was the one invoked.
        chain.count.assert_called_once()
        chain.agg.assert_not_called()

    def test_counts_distinct_entities_when_entity_column_present(self, patched_pyspark_sql_functions):
        # When ``entity_id`` IS on the DataFrame, the size must collapse
        # to distinct entities -- not rows. The training input is keyed
        # on ``(entity_id, as_of_date)``, so the same entity contributes
        # multiple snapshot rows; counting rows multi-counts the cluster
        # by the number of as-of-dates and inflates the dashboard's
        # ``N customers in this archetype`` pill.
        rows = [{CLUSTER_COL: 0, "count": 50}, {CLUSTER_COL: 1, "count": 75}]
        chain = _make_chain_returning(rows, columns=["entity_id", "as_of_date", "f0"])
        result = clusterer.cluster_size_stats(chain)
        assert result == [(0, 50), (1, 75)]
        # countDistinct path: agg() invoked, raw count() path not used.
        chain.agg.assert_called_once()
        chain.count.assert_not_called()
        # ``F.countDistinct`` was the aggregation function chosen.
        assert patched_pyspark_sql_functions.countDistinct.call_count == 1


class TestClusterTargetMeans:
    def test_returns_mean_target_per_cluster(self, patched_pyspark_sql_functions):
        rows = [
            {CLUSTER_COL: 0, "__mean_target": 0.3},
            {CLUSTER_COL: 1, "__mean_target": 0.6},
        ]
        chain = _make_chain_returning(rows)
        result = clusterer.cluster_target_means(chain, target_col="target")
        assert result == [(0, 0.3), (1, 0.6)]

    def test_mean_target_wraps_nanvl(self, patched_pyspark_sql_functions):
        rows = [{CLUSTER_COL: 0, "__mean_target": 0.3}]
        chain = _make_chain_returning(rows)
        clusterer.cluster_target_means(chain, target_col="target")
        _assert_nanvl_fallback_is_null(patched_pyspark_sql_functions, expected_calls=1)


class TestClusterShapCentroids:
    def test_returns_per_cluster_shap_means(self, patched_pyspark_sql_functions):
        rows = [
            {CLUSTER_COL: 0, "__shap_mean_0": 0.1, "__shap_mean_1": -0.2},
            {CLUSTER_COL: 1, "__shap_mean_0": 0.4, "__shap_mean_1": 0.5},
        ]
        chain = _make_chain_returning(rows)
        centroids, order = clusterer.cluster_shap_centroids(
            chain, shap_columns=["shap_a", "shap_b"]
        )
        assert centroids == [[0.1, -0.2], [0.4, 0.5]]
        assert order == ["shap_a", "shap_b"]

    def test_shap_mean_wraps_nanvl(self, patched_pyspark_sql_functions):
        rows = [{CLUSTER_COL: 0, "__shap_mean_0": 0.1, "__shap_mean_1": -0.2}]
        chain = _make_chain_returning(rows)
        clusterer.cluster_shap_centroids(chain, shap_columns=["shap_a", "shap_b"])
        _assert_nanvl_fallback_is_null(patched_pyspark_sql_functions, expected_calls=2)


# ---------------------------------------------------------------------------
# select_top_shap_features — batched per Coding_Practices.md (≤200 / .agg())
# ---------------------------------------------------------------------------


class TestSelectTopShapFeatures:
    def test_returns_input_unchanged_when_under_cap(self, patched_pyspark_sql_functions):
        df = MagicMock(name="ShapDF")
        out = clusterer.select_top_shap_features(df, ["a", "b"], feature_cap=50)
        assert out == ["a", "b"]

    def test_batches_aggregations_at_200_columns_per_call(
        self, patched_pyspark_sql_functions
    ):
        """Coding_Practices.md: per-column aggregates must be batched in
        groups of ≤200 so Catalyst plans stay O(200²) instead of O(N²)."""
        agg_calls: list[int] = []

        class _Row:
            def __getitem__(self, key):
                return 0.1

        def _agg(*exprs):
            agg_calls.append(len(exprs))
            r = MagicMock()
            r.head.return_value = _Row()
            return r

        df = MagicMock(name="ShapDF")
        df.agg = _agg

        features = [f"shap_f{i}" for i in range(450)]
        selected = clusterer.select_top_shap_features(df, features, feature_cap=50)
        # 450 / 200 → batches of 200, 200, 50
        assert agg_calls == [200, 200, 50]
        assert len(selected) == 50

    def test_mean_abs_wraps_nanvl(self, patched_pyspark_sql_functions):
        """NaN rows in shap_<feature> (shouldn't happen after the emission
        fix, but defensive) must not poison the top-feature ranking — wrap
        each column in ``nanvl`` before ``F.mean(F.abs(...))``."""

        class _Row:
            def __getitem__(self, key):
                return 0.1

        def _agg(*_exprs):
            r = MagicMock()
            r.head.return_value = _Row()
            return r

        df = MagicMock(name="ShapDF")
        df.agg = _agg
        features = [f"shap_f{i}" for i in range(250)]  # forces batching
        clusterer.select_top_shap_features(df, features, feature_cap=50)
        assert patched_pyspark_sql_functions.nanvl.call_count == 250


# ---------------------------------------------------------------------------
# _compute_feature_scales batching (lives in derivation.py but shares pattern)
# ---------------------------------------------------------------------------


class TestComputeFeatureScales:
    def test_batches_stddev_at_200_columns_per_call(self):
        """Same batching contract as select_top_shap_features — keep Catalyst
        plans small when feature_order has 100+ columns."""

        from customer_retention.stages.causal import derivation

        fake_functions = MagicMock(name="pyspark.sql.functions")
        fake_functions.col = MagicMock(side_effect=lambda c: _FakeCol(c))
        fake_functions.stddev_pop = MagicMock(side_effect=lambda c: c)
        fake_sql = MagicMock(name="pyspark.sql")
        fake_sql.functions = fake_functions

        with _patch_module("pyspark.sql", fake_sql), _patch_module("pyspark.sql.functions", fake_functions):
            agg_calls: list[int] = []

            class _Row:
                def __getitem__(self, key):
                    return 1.5

            def _agg(*exprs):
                agg_calls.append(len(exprs))
                r = MagicMock()
                r.head.return_value = _Row()
                return r

            df = MagicMock(name="RawWithClusters")
            df.columns = [f"f{i}" for i in range(450)]
            df.agg = _agg

            features = [f"f{i}" for i in range(450)]
            scales = derivation._compute_feature_scales(df, features)
            assert agg_calls == [200, 200, 50]
            assert len(scales) == 450
            assert all(s == 1.5 for s in scales)


def _patch_module(name, module):
    import sys

    class _Ctx:
        def __enter__(self):
            self._prev = sys.modules.get(name)
            sys.modules[name] = module
            return module

        def __exit__(self, *exc):
            if self._prev is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = self._prev

    return _Ctx()
