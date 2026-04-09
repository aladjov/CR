from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from customer_retention.analysis.auto_explorer.active_dataset_store import (
    assert_landing_schema_matches_registry,
    load_active_dataset,
    load_active_dataset_distributed,
    load_gold_features,
    load_gold_features_distributed,
    load_merge_dataset,
    load_merge_dataset_distributed,
    load_silver_merged,
    load_silver_merged_distributed,
    merge_datasets_incremental,
    require_silver_merged,
    require_silver_merged_distributed,
    save_active_dataset,
    save_aggregated_dataset,
    save_gold_features,
)
from customer_retention.analysis.auto_explorer.run_namespace import RunNamespace
from customer_retention.core.config.column_config import DatasetGranularity
from customer_retention.integrations.adapters.factory import get_delta


@pytest.fixture()
def namespace(tmp_path):
    ns = RunNamespace(root=tmp_path, run_id="proj-test1234")
    ns.setup()
    return ns


@pytest.fixture()
def sample_df():
    return pd.DataFrame({"customer_id": [1, 2, 3], "revenue": [100.0, 200.0, 300.0]})


class TestSaveAndLoadRoundTrip:
    def test_save_and_load_round_trip(self, namespace, sample_df):
        save_active_dataset(namespace, "customers", sample_df)
        loaded = load_active_dataset(namespace, "customers")
        pd.testing.assert_frame_equal(loaded, sample_df)

    def test_returns_dlt_path(self, namespace, sample_df):
        path = save_active_dataset(namespace, "customers", sample_df)
        assert path == namespace.landing_table_dir("customers")


class TestLoadMissing:
    def test_load_missing_raises_file_not_found(self, namespace):
        with pytest.raises(FileNotFoundError):
            load_active_dataset(namespace, "nonexistent")


class _SpyDistributedDF:
    # Stand-in for pyspark.pandas / native Spark DataFrame: explodes if any
    # caller tries to collect via `.to_pandas()`. Used by every
    # "no driver collect" regression test in this file.
    def to_pandas(self):
        raise AssertionError(
            "loader collected distributed data to driver — must stay distributed",
        )


class TestLoadActiveDatasetNoDriverCollect:
    # Regression: a native-pandas-only load path forced `.to_pandas()` on the
    # full landing Delta, OOM'ing the driver after the SCD history augmentation
    # expanded `case` into (case × grid-anchor) fan-out (observed 4.4 GiB
    # collect). All load functions must dispatch via `get_delta()` and return
    # whatever the active backend hands back — pyspark.pandas on Databricks,
    # native pandas on local — without a full driver collect on the
    # distributed path.
    def test_load_active_dataset_returns_without_collect(self, namespace):
        (namespace.landing_table_dir("case")).mkdir(parents=True, exist_ok=True)

        fake_delta = MagicMock()
        fake_delta.read.return_value = _SpyDistributedDF()

        with patch(
            "customer_retention.analysis.auto_explorer.active_dataset_store.get_delta",
            return_value=fake_delta,
        ):
            result = load_active_dataset(namespace, "case")

        fake_delta.read.assert_called_once_with(
            str(namespace.landing_table_dir("case")),
        )
        assert isinstance(result, _SpyDistributedDF)

    def test_load_merge_dataset_event_level_returns_without_collect(self, namespace):
        (namespace.bronze_table_dir("events")).mkdir(parents=True, exist_ok=True)

        fake_delta = MagicMock()
        fake_delta.read.return_value = _SpyDistributedDF()

        with patch(
            "customer_retention.analysis.auto_explorer.active_dataset_store.get_delta",
            return_value=fake_delta,
        ):
            result = load_merge_dataset(
                namespace, "events", DatasetGranularity.EVENT_LEVEL,
            )

        fake_delta.read.assert_called_once_with(
            str(namespace.bronze_table_dir("events")),
        )
        assert isinstance(result, _SpyDistributedDF)

    def test_load_silver_merged_returns_without_collect(self, namespace):
        namespace.silver_merged_path.mkdir(parents=True, exist_ok=True)

        fake_delta = MagicMock()
        fake_delta.read.return_value = _SpyDistributedDF()

        with patch(
            "customer_retention.analysis.auto_explorer.active_dataset_store.get_delta",
            return_value=fake_delta,
        ):
            result = load_silver_merged(
                namespace, "events", DatasetGranularity.EVENT_LEVEL,
            )

        fake_delta.read.assert_called_once_with(str(namespace.silver_merged_path))
        assert isinstance(result, _SpyDistributedDF)

    def test_require_silver_merged_returns_without_collect(self, namespace):
        namespace.silver_merged_path.mkdir(parents=True, exist_ok=True)

        fake_delta = MagicMock()
        fake_delta.read.return_value = _SpyDistributedDF()

        with patch(
            "customer_retention.analysis.auto_explorer.active_dataset_store.get_delta",
            return_value=fake_delta,
        ):
            result = require_silver_merged(namespace)

        fake_delta.read.assert_called_once_with(str(namespace.silver_merged_path))
        assert isinstance(result, _SpyDistributedDF)

    def test_load_gold_features_returns_without_collect(self, namespace):
        (namespace.gold_table_dir("cust_emai__abc1234")).mkdir(
            parents=True, exist_ok=True,
        )

        fake_delta = MagicMock()
        fake_delta.read.return_value = _SpyDistributedDF()

        with patch(
            "customer_retention.analysis.auto_explorer.active_dataset_store.get_delta",
            return_value=fake_delta,
        ):
            result = load_gold_features(namespace, "cust_emai__abc1234")

        fake_delta.read.assert_called_once_with(
            str(namespace.gold_table_dir("cust_emai__abc1234")),
        )
        assert isinstance(result, _SpyDistributedDF)


class TestSaveNormalizesTimestamps:
    def test_tz_aware_timestamps_saved_as_tz_naive(self, namespace):
        df = pd.DataFrame({
            "customer_id": [1, 2],
            "event_time": pd.to_datetime(["2023-01-01", "2023-06-15"]).tz_localize("UTC"),
        })
        save_active_dataset(namespace, "customers", df)
        loaded = load_active_dataset(namespace, "customers")
        assert loaded["event_time"].dt.tz is None
        assert loaded["event_time"].iloc[0] == pd.Timestamp("2023-01-01")

    def test_mixed_tz_and_naive_timestamps(self, namespace):
        df = pd.DataFrame({
            "customer_id": [1],
            "ts_utc": pd.to_datetime(["2023-01-01"]).tz_localize("UTC"),
            "ts_naive": pd.to_datetime(["2023-06-15"]),
        })
        save_active_dataset(namespace, "customers", df)
        loaded = load_active_dataset(namespace, "customers")
        assert loaded["ts_utc"].dt.tz is None
        assert loaded["ts_naive"].dt.tz is None


class TestOverwrite:
    def test_overwrite_replaces_data(self, namespace, sample_df):
        save_active_dataset(namespace, "customers", sample_df)
        new_df = pd.DataFrame({"customer_id": [10], "score": [0.5]})
        save_active_dataset(namespace, "customers", new_df)
        loaded = load_active_dataset(namespace, "customers")
        pd.testing.assert_frame_equal(loaded, new_df)


class TestLandingTableDirPathConvention:
    def test_landing_table_dir_path_convention(self, namespace, sample_df):
        save_active_dataset(namespace, "orders", sample_df)
        expected_dir = namespace.landing_table_dir("orders")
        assert expected_dir.is_dir()


class TestLoadMergeDataset:
    def test_entity_level_returns_active_dataset(self, namespace, sample_df):
        save_active_dataset(namespace, "customers", sample_df)
        result = load_merge_dataset(namespace, "customers", DatasetGranularity.ENTITY_LEVEL)
        pd.testing.assert_frame_equal(result, sample_df)

    def test_event_level_prefers_aggregated_delta(self, namespace, sample_df):
        save_active_dataset(namespace, "events", sample_df)
        agg_df = pd.DataFrame({"customer_id": [1, 2], "as_of_date": ["2024-01-01", "2024-02-01"], "count": [10, 20]})
        save_aggregated_dataset(namespace, "events", agg_df)
        result = load_merge_dataset(namespace, "events", DatasetGranularity.EVENT_LEVEL)
        pd.testing.assert_frame_equal(result, agg_df)

    def test_event_level_falls_back_when_no_aggregated(self, namespace, sample_df):
        save_active_dataset(namespace, "events", sample_df)
        result = load_merge_dataset(namespace, "events", DatasetGranularity.EVENT_LEVEL)
        pd.testing.assert_frame_equal(result, sample_df)

    def test_unknown_granularity_returns_active_dataset(self, namespace, sample_df):
        save_active_dataset(namespace, "misc", sample_df)
        result = load_merge_dataset(namespace, "misc", DatasetGranularity.UNKNOWN)
        pd.testing.assert_frame_equal(result, sample_df)


class TestLoadSilverMerged:
    def test_returns_silver_merged_when_exists(self, namespace):
        merged_df = pd.DataFrame({"customer_id": [1, 2], "churned": [0, 1], "feature_a": [10, 20]})
        delta = get_delta(force_local=True)
        delta.write(merged_df, str(namespace.silver_merged_path), mode="overwrite")
        result = load_silver_merged(namespace, "events", DatasetGranularity.EVENT_LEVEL)
        pd.testing.assert_frame_equal(result, merged_df)

    def test_falls_back_to_bronze_when_no_silver(self, namespace, sample_df):
        agg_df = pd.DataFrame({"customer_id": [1, 2], "count": [10, 20]})
        save_active_dataset(namespace, "events", sample_df)
        save_aggregated_dataset(namespace, "events", agg_df)
        result = load_silver_merged(namespace, "events", DatasetGranularity.EVENT_LEVEL)
        pd.testing.assert_frame_equal(result, agg_df)

    def test_falls_back_to_landing_when_nothing(self, namespace, sample_df):
        save_active_dataset(namespace, "events", sample_df)
        result = load_silver_merged(namespace, "events", DatasetGranularity.EVENT_LEVEL)
        pd.testing.assert_frame_equal(result, sample_df)


class TestGoldFeatures:
    def test_save_and_load_gold_round_trip(self, namespace, sample_df):
        save_gold_features(namespace, "cust_emai__abc1234", sample_df)
        loaded = load_gold_features(namespace, "cust_emai__abc1234")
        pd.testing.assert_frame_equal(loaded, sample_df)

    def test_save_gold_returns_path(self, namespace, sample_df):
        path = save_gold_features(namespace, "cust_emai__abc1234", sample_df)
        assert path == namespace.gold_table_dir("cust_emai__abc1234")

    def test_load_gold_missing_raises(self, namespace):
        with pytest.raises(FileNotFoundError):
            load_gold_features(namespace, "nonexistent__0000000")


class TestLoadMergeDatasetDistributed:
    def test_reads_without_conversion(self, namespace, sample_df):
        save_active_dataset(namespace, "customers", sample_df)
        result = load_merge_dataset_distributed(
            namespace, "customers", DatasetGranularity.ENTITY_LEVEL,
        )
        pd.testing.assert_frame_equal(result, sample_df)

    def test_event_level_prefers_aggregated(self, namespace, sample_df):
        save_active_dataset(namespace, "events", sample_df)
        agg_df = pd.DataFrame({
            "customer_id": [1, 2],
            "as_of_date": ["2024-01-01", "2024-02-01"],
            "count": [10, 20],
        })
        save_aggregated_dataset(namespace, "events", agg_df)
        result = load_merge_dataset_distributed(
            namespace, "events", DatasetGranularity.EVENT_LEVEL,
        )
        pd.testing.assert_frame_equal(result, agg_df)

    def test_event_level_falls_back_to_landing(self, namespace, sample_df):
        save_active_dataset(namespace, "events", sample_df)
        result = load_merge_dataset_distributed(
            namespace, "events", DatasetGranularity.EVENT_LEVEL,
        )
        pd.testing.assert_frame_equal(result, sample_df)

    def test_missing_raises_file_not_found(self, namespace):
        with pytest.raises(FileNotFoundError):
            load_merge_dataset_distributed(
                namespace, "nonexistent", DatasetGranularity.ENTITY_LEVEL,
            )


class TestLoadActiveDatasetDistributed:
    def test_reads_without_conversion(self, namespace, sample_df):
        save_active_dataset(namespace, "customers", sample_df)
        result = load_active_dataset_distributed(namespace, "customers")
        pd.testing.assert_frame_equal(result, sample_df)

    def test_missing_raises_file_not_found(self, namespace):
        with pytest.raises(FileNotFoundError):
            load_active_dataset_distributed(namespace, "nonexistent")


class TestLoadSilverMergedDistributed:
    def test_returns_silver_when_exists(self, namespace):
        merged_df = pd.DataFrame({
            "customer_id": [1, 2],
            "churned": [0, 1],
            "feature_a": [10, 20],
        })
        delta = get_delta(force_local=True)
        delta.write(merged_df, str(namespace.silver_merged_path), mode="overwrite")
        result = load_silver_merged_distributed(
            namespace, "events", DatasetGranularity.EVENT_LEVEL,
        )
        pd.testing.assert_frame_equal(result, merged_df)

    def test_falls_back_to_bronze_when_no_silver(self, namespace, sample_df):
        agg_df = pd.DataFrame({"customer_id": [1, 2], "count": [10, 20]})
        save_active_dataset(namespace, "events", sample_df)
        save_aggregated_dataset(namespace, "events", agg_df)
        result = load_silver_merged_distributed(
            namespace, "events", DatasetGranularity.EVENT_LEVEL,
        )
        pd.testing.assert_frame_equal(result, agg_df)

    def test_falls_back_to_landing_when_nothing(self, namespace, sample_df):
        save_active_dataset(namespace, "events", sample_df)
        result = load_silver_merged_distributed(
            namespace, "events", DatasetGranularity.EVENT_LEVEL,
        )
        pd.testing.assert_frame_equal(result, sample_df)


class TestRequireSilverMerged:
    def test_returns_data_when_silver_merged_exists(self, namespace):
        merged_df = pd.DataFrame({"customer_id": [1, 2], "churned": [0, 1], "feature_a": [10, 20]})
        delta = get_delta(force_local=True)
        delta.write(merged_df, str(namespace.silver_merged_path), mode="overwrite")
        result = require_silver_merged(namespace)
        pd.testing.assert_frame_equal(result, merged_df)

    def test_raises_when_silver_merged_missing(self, namespace):
        with pytest.raises(FileNotFoundError, match="Silver merged dataset not found"):
            require_silver_merged(namespace)

    def test_does_not_fall_back_to_landing(self, namespace, sample_df):
        save_active_dataset(namespace, "customers", sample_df)
        with pytest.raises(FileNotFoundError, match="Silver merged dataset not found"):
            require_silver_merged(namespace)


class TestRequireSilverMergedDistributed:
    def test_returns_data_when_silver_merged_exists(self, namespace):
        merged_df = pd.DataFrame({"customer_id": [1, 2], "churned": [0, 1], "feature_a": [10, 20]})
        delta = get_delta(force_local=True)
        delta.write(merged_df, str(namespace.silver_merged_path), mode="overwrite")
        result = require_silver_merged_distributed(namespace)
        pd.testing.assert_frame_equal(result, merged_df)

    def test_raises_when_silver_merged_missing(self, namespace):
        with pytest.raises(FileNotFoundError, match="Silver merged dataset not found"):
            require_silver_merged_distributed(namespace)

    def test_does_not_fall_back_to_landing(self, namespace, sample_df):
        save_active_dataset(namespace, "customers", sample_df)
        with pytest.raises(FileNotFoundError, match="Silver merged dataset not found"):
            require_silver_merged_distributed(namespace)


class TestLoadGoldFeaturesDistributed:
    def test_round_trip(self, namespace, sample_df):
        save_gold_features(namespace, "cust_emai__abc1234", sample_df)
        result = load_gold_features_distributed(namespace, "cust_emai__abc1234")
        pd.testing.assert_frame_equal(result, sample_df)

    def test_missing_raises(self, namespace):
        with pytest.raises(FileNotFoundError):
            load_gold_features_distributed(namespace, "nonexistent__0000000")


class TestDistributedSavePath:
    def _make_spark_pandas_df(self):
        mock_df = MagicMock()
        mock_df.to_spark = MagicMock()
        return mock_df

    @patch("customer_retention.analysis.auto_explorer.active_dataset_store.as_spark_df")
    @patch("customer_retention.analysis.auto_explorer.active_dataset_store.get_delta")
    def test_save_active_converts_to_native_spark(self, mock_get_delta, mock_as_spark, namespace):
        mock_delta = MagicMock()
        mock_get_delta.return_value = mock_delta
        ps_df = self._make_spark_pandas_df()
        mock_native = MagicMock()
        mock_as_spark.return_value = mock_native

        save_active_dataset(namespace, "customers", ps_df)

        mock_as_spark.assert_called_once_with(ps_df)
        mock_delta.write.assert_called_once_with(
            mock_native, str(namespace.landing_table_dir("customers")),
            mode="overwrite", z_order_columns=None,
        )

    @patch("customer_retention.analysis.auto_explorer.active_dataset_store.as_spark_df")
    @patch("customer_retention.analysis.auto_explorer.active_dataset_store.get_delta")
    def test_save_aggregated_converts_to_native_spark(self, mock_get_delta, mock_as_spark, namespace):
        mock_delta = MagicMock()
        mock_get_delta.return_value = mock_delta
        ps_df = self._make_spark_pandas_df()
        mock_native = MagicMock()
        mock_as_spark.return_value = mock_native

        save_aggregated_dataset(namespace, "events", ps_df)

        mock_as_spark.assert_called_once_with(ps_df)
        mock_delta.write.assert_called_once_with(
            mock_native, str(namespace.bronze_table_dir("events")),
            mode="overwrite", z_order_columns=None,
        )

    @patch("customer_retention.analysis.auto_explorer.active_dataset_store.as_spark_df")
    @patch("customer_retention.analysis.auto_explorer.active_dataset_store.get_delta")
    def test_save_gold_converts_to_native_spark(self, mock_get_delta, mock_as_spark, namespace):
        mock_delta = MagicMock()
        mock_get_delta.return_value = mock_delta
        ps_df = self._make_spark_pandas_df()
        mock_native = MagicMock()
        mock_as_spark.return_value = mock_native

        ps_df.columns = ["feature_a", "feature_b"]  # no entity_id/as_of_date
        save_gold_features(namespace, "cust_emai__abc1234", ps_df)

        mock_as_spark.assert_called_once_with(ps_df)
        mock_delta.write.assert_called_once_with(
            mock_native, str(namespace.gold_table_dir("cust_emai__abc1234")),
            mode="overwrite", z_order_columns=None,
        )

    @patch("customer_retention.analysis.auto_explorer.active_dataset_store.as_spark_df")
    @patch("customer_retention.analysis.auto_explorer.active_dataset_store.get_delta")
    def test_delegates_to_delta_write(self, mock_get_delta, mock_as_spark, namespace):
        mock_delta = MagicMock()
        mock_get_delta.return_value = mock_delta
        mock_as_spark.return_value = MagicMock()
        ps_df = self._make_spark_pandas_df()

        save_active_dataset(namespace, "customers", ps_df)

        mock_as_spark.assert_called_once_with(ps_df)
        mock_delta.write.assert_called_once()

    def test_native_pandas_still_uses_local_delta(self, namespace):
        df = pd.DataFrame({"customer_id": [1, 2, 3], "revenue": [100.0, 200.0, 300.0]})
        save_active_dataset(namespace, "customers", df)
        loaded = load_active_dataset(namespace, "customers")
        pd.testing.assert_frame_equal(loaded, df)

    @patch("customer_retention.analysis.auto_explorer.active_dataset_store._is_native_spark_df")
    @patch("customer_retention.analysis.auto_explorer.active_dataset_store.as_spark_df")
    @patch("customer_retention.analysis.auto_explorer.active_dataset_store.get_delta")
    def test_save_active_routes_native_spark_df_distributed(
        self, mock_get_delta, mock_as_spark, mock_is_native, namespace,
    ):
        # Regression: native PySpark DataFrames lack `.to_spark()` (only
        # pyspark.pandas does), so a bare `hasattr(df, "to_spark")` check used
        # to silently fall through to `toPandas()` and OOM the driver. The
        # routing must recognize native Spark DFs and dispatch them through
        # `get_delta().write` distributed-style.
        mock_delta = MagicMock()
        mock_get_delta.return_value = mock_delta
        # Stand-in for pyspark.sql.DataFrame: no .to_spark, no .to_pandas, no
        # .toPandas — if anything tries to collect, it will AttributeError.
        class _FakeNativeSparkDF:
            pass

        native = _FakeNativeSparkDF()
        mock_is_native.return_value = True
        mock_as_spark.return_value = native

        save_active_dataset(namespace, "customers", native)

        mock_is_native.assert_called_once_with(native)
        mock_as_spark.assert_called_once_with(native)
        mock_delta.write.assert_called_once_with(
            native, str(namespace.landing_table_dir("customers")),
            mode="overwrite", z_order_columns=None,
        )


class TestAssertLandingSchemaMatchesRegistry:
    """Fail-fast schema-drift guard.

    Catches the bouncing failure mode where NB00 patches the in-memory
    registry but the landing Delta on disk is stale (or vice versa). The
    helper is schema-only — never iterates rows — so it stays distributed-
    safe even on multi-million row datasets.
    """

    def test_passes_when_columns_match_exactly(self, namespace):
        df = pd.DataFrame(
            {"CASE_ID": ["A"], "OWNER_NAME": ["Alice"], "Status": ["Open"]},
        )
        save_active_dataset(namespace, "case", df)

        assert_landing_schema_matches_registry(
            namespace, "case", ["CASE_ID", "OWNER_NAME", "Status"],
        )

    def test_passes_with_case_insensitive_match(self, namespace):
        df = pd.DataFrame(
            {"CASE_ID": ["A"], "OWNER_NAME": ["Alice"], "Status": ["Open"]},
        )
        save_active_dataset(namespace, "case", df)

        assert_landing_schema_matches_registry(
            namespace, "case", ["case_id", "owner_name", "status"],
        )

    def test_raises_when_landing_dir_missing(self, namespace):
        with pytest.raises(FileNotFoundError, match="Landing Delta missing"):
            assert_landing_schema_matches_registry(
                namespace, "nonexistent", ["a", "b"],
            )

    def test_raises_when_landing_has_extra_column(self, namespace):
        df = pd.DataFrame(
            {"CASE_ID": ["A"], "OWNER_NAME": ["Alice"], "stale_col": [1]},
        )
        save_active_dataset(namespace, "case", df)

        with pytest.raises(ValueError, match="stale_col"):
            assert_landing_schema_matches_registry(
                namespace, "case", ["CASE_ID", "OWNER_NAME"],
            )

    def test_raises_when_registry_has_extra_column(self, namespace):
        df = pd.DataFrame(
            {"CASE_ID": ["A"], "OWNER_NAME": ["Alice"]},
        )
        save_active_dataset(namespace, "case", df)

        with pytest.raises(ValueError, match="missing_col"):
            assert_landing_schema_matches_registry(
                namespace, "case", ["CASE_ID", "OWNER_NAME", "missing_col"],
            )

    def test_does_not_iterate_rows(self, namespace, monkeypatch):
        # Distributed-safety contract: the helper inspects the schema only,
        # never .collect() / .count() / per-row scans. We mock get_delta to
        # return a stub whose .read(path) returns a DF whose .columns is the
        # only attribute exercised.
        from customer_retention.analysis.auto_explorer import (
            active_dataset_store as store_mod,
        )

        landing_path = namespace.landing_table_dir("case")
        landing_path.mkdir(parents=True, exist_ok=True)

        captured_calls: list[str] = []

        class _SchemaOnlyDF:
            columns = ["CASE_ID", "Status"]

            def __getattr__(self, name):
                captured_calls.append(name)
                raise AssertionError(
                    f"schema-drift guard called {name!r} — must inspect columns only",
                )

        class _SchemaOnlyDelta:
            def read(self, path):
                return _SchemaOnlyDF()

        monkeypatch.setattr(store_mod, "get_delta", lambda: _SchemaOnlyDelta())

        assert_landing_schema_matches_registry(
            namespace, "case", ["CASE_ID", "Status"],
        )
        assert captured_calls == [], f"helper accessed forbidden attrs: {captured_calls}"


# ---------------------------------------------------------------------------
# FakeSparkDF for merge_datasets_incremental tests
# ---------------------------------------------------------------------------


class _FakeSparkDF:
    """Minimal native Spark DF simulation for merge_datasets_incremental."""

    _is_fake_spark_df = True

    def __init__(self, pdf):
        self._pdf = pdf.copy()

    @property
    def columns(self):
        return list(self._pdf.columns)

    def count(self):
        return len(self._pdf)

    def select(self, *cols):
        return _FakeSparkDF(self._pdf[list(cols)])

    def distinct(self):
        return _FakeSparkDF(self._pdf.drop_duplicates())


class _FakeDeltaForIncremental:
    """In-memory Delta stand-in that stores one table at a time."""

    def __init__(self):
        self.store: dict[str, pd.DataFrame] = {}
        self.optimize_calls: list[dict] = []

    def write(self, data, path, mode="overwrite"):
        if isinstance(data, _FakeSparkDF):
            self.store[path] = data._pdf.copy()
        elif isinstance(data, pd.DataFrame):
            self.store[path] = data.copy()
        else:
            # native Spark → just store columns for schema tests
            self.store[path] = data._pdf.copy()

    def read(self, path):
        return self.store[path].copy()

    def optimize(self, path, z_order_columns=None):
        self.optimize_calls.append({"path": path, "z_order_columns": z_order_columns})


class _FakeSparkSession:
    """Simulates spark.read.format('delta').load(path)."""

    def __init__(self, delta_store):
        self._delta = delta_store

    @property
    def read(self):
        return self

    def format(self, fmt):
        return self

    def load(self, path):
        pdf = self._delta.store[path].copy()
        return _FakeSparkDF(pdf)

    @property
    def catalog(self):
        return self

    def clearCache(self):  # noqa: N802
        pass


class TestMergeDatasetsIncremental:
    @pytest.fixture()
    def _setup(self, namespace, monkeypatch):
        """Wire up fakes for Delta, Spark, and merge_one."""
        fake_delta = _FakeDeltaForIncremental()
        fake_spark = _FakeSparkSession(fake_delta)

        monkeypatch.setattr(
            "customer_retention.analysis.auto_explorer.active_dataset_store.get_delta",
            lambda **kw: fake_delta,
        )
        # get_spark_session is imported inside the function from detection module
        monkeypatch.setattr(
            "customer_retention.core.compat.detection.get_spark_session",
            lambda: fake_spark,
        )
        monkeypatch.setattr(
            "customer_retention.analysis.auto_explorer.active_dataset_store.release_stage_memory",
            lambda: None,
        )
        return namespace, fake_delta, fake_spark

    def _make_merger(self, merge_one_side_effect=None):
        """Build a mock SparkTemporalMerger with controllable merge_one."""
        from customer_retention.stages.temporal.temporal_merger import MergeConfig

        merger = MagicMock()
        merger.config = MergeConfig(entity_key="entity_id")

        if merge_one_side_effect:
            merger.merge_one.side_effect = merge_one_side_effect
        return merger

    def _spine_sdf(self):
        pdf = pd.MultiIndex.from_product(
            [["A", "B"], pd.to_datetime(["2024-01-01", "2024-02-01"])],
            names=["entity_id", "as_of_date"],
        ).to_frame(index=False)
        return _FakeSparkDF(pdf)

    def test_produces_correct_merged_result(self, _setup):
        namespace, fake_delta, _ = _setup
        spine_sdf = self._spine_sdf()

        ds1 = MagicMock(name="ds1_input")
        ds1.name = "txns"
        ds2 = MagicMock(name="ds2_input")
        ds2.name = "customers"

        def _merge_one(current_sdf, ds):
            pdf = current_sdf._pdf.copy()
            if ds.name == "txns":
                pdf["amount"] = range(len(pdf))
                return _FakeSparkDF(pdf), {"amount"}
            else:
                pdf["tier"] = "gold"
                return _FakeSparkDF(pdf), {"tier"}

        merger = self._make_merger(merge_one_side_effect=_merge_one)
        report = merge_datasets_incremental(namespace, spine_sdf, [ds1, ds2], merger)

        final = fake_delta.store[str(namespace.silver_merged_path)]
        assert "amount" in final.columns
        assert "tier" in final.columns
        assert len(final) == 4

    def test_report_has_correct_column_counts(self, _setup):
        namespace, _, _ = _setup
        spine_sdf = self._spine_sdf()

        ds1 = MagicMock()
        ds1.name = "txns"

        def _merge_one(current_sdf, ds):
            pdf = current_sdf._pdf.copy()
            pdf["a"] = 1
            pdf["b"] = 2
            return _FakeSparkDF(pdf), {"a", "b"}

        merger = self._make_merger(merge_one_side_effect=_merge_one)
        report = merge_datasets_incremental(namespace, spine_sdf, [ds1], merger)

        assert report.columns_per_dataset == {"txns": 2}
        assert report.total_columns == 4  # entity_id, as_of_date, a, b

    def test_optimize_with_zorder_columns(self, _setup):
        namespace, fake_delta, _ = _setup
        spine_sdf = self._spine_sdf()

        merger = self._make_merger()
        merge_datasets_incremental(namespace, spine_sdf, [], merger)

        assert len(fake_delta.optimize_calls) == 1
        call = fake_delta.optimize_calls[0]
        assert call["z_order_columns"] == ["entity_id", "as_of_date"]

    def test_memory_released_between_steps(self, _setup, monkeypatch):
        namespace, _, _ = _setup
        spine_sdf = self._spine_sdf()
        release_calls = []

        monkeypatch.setattr(
            "customer_retention.analysis.auto_explorer.active_dataset_store.release_stage_memory",
            lambda: release_calls.append(1),
        )

        ds1 = MagicMock()
        ds1.name = "d1"
        ds2 = MagicMock()
        ds2.name = "d2"

        def _merge_one(current_sdf, ds):
            pdf = current_sdf._pdf.copy()
            pdf[f"col_{ds.name}"] = 1
            return _FakeSparkDF(pdf), {f"col_{ds.name}"}

        merger = self._make_merger(merge_one_side_effect=_merge_one)
        merge_datasets_incremental(namespace, spine_sdf, [ds1, ds2], merger)
        assert len(release_calls) == 2

    def test_empty_datasets_list(self, _setup):
        namespace, fake_delta, _ = _setup
        spine_sdf = self._spine_sdf()

        merger = self._make_merger()
        report = merge_datasets_incremental(namespace, spine_sdf, [], merger)

        final = fake_delta.store[str(namespace.silver_merged_path)]
        assert len(final) == 4
        assert report.datasets_merged == []
        assert report.total_columns == 2  # entity_id, as_of_date

    def test_column_conflicts_across_datasets(self, _setup):
        namespace, fake_delta, _ = _setup
        spine_sdf = self._spine_sdf()

        ds1 = MagicMock()
        ds1.name = "src1"
        ds2 = MagicMock()
        ds2.name = "src2"

        def _merge_one(current_sdf, ds):
            pdf = current_sdf._pdf.copy()
            if ds.name == "src1":
                pdf["val"] = 1
                return _FakeSparkDF(pdf), {"val"}
            else:
                pdf["src2__val"] = 2
                return _FakeSparkDF(pdf), {"src2__val"}

        merger = self._make_merger(merge_one_side_effect=_merge_one)
        report = merge_datasets_incremental(namespace, spine_sdf, [ds1, ds2], merger)

        assert "src2__val" in report.renamed_columns

    def test_spine_row_count_preserved(self, _setup):
        namespace, fake_delta, _ = _setup
        spine_sdf = self._spine_sdf()

        ds1 = MagicMock()
        ds1.name = "ds1"

        def _merge_one(current_sdf, ds):
            # Left join preserves rows
            pdf = current_sdf._pdf.copy()
            pdf["x"] = 1
            return _FakeSparkDF(pdf), {"x"}

        merger = self._make_merger(merge_one_side_effect=_merge_one)
        report = merge_datasets_incremental(namespace, spine_sdf, [ds1], merger)

        final = fake_delta.store[str(namespace.silver_merged_path)]
        assert len(final) == report.spine_rows
