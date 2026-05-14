from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from customer_retention.core import compat as compat_mod
from customer_retention.core.compat import (
    release_stage_memory,
    track_stage_object,
)


class TestTrackStageObject:

    def setup_method(self):
        compat_mod._stage_objects.clear()

    def teardown_method(self):
        compat_mod._stage_objects.clear()

    def test_adds_objects_to_registry(self):
        a, b = object(), object()
        track_stage_object(a, b)
        assert len(compat_mod._stage_objects) == 2

    def test_multiple_calls_accumulate(self):
        track_stage_object(object())
        track_stage_object(object(), object())
        assert len(compat_mod._stage_objects) == 3


class TestReleaseStageMemory:

    def setup_method(self):
        compat_mod._stage_objects.clear()

    def teardown_method(self):
        compat_mod._stage_objects.clear()

    @patch("customer_retention.core.compat.is_databricks", return_value=True)
    @patch("customer_retention.core.compat.get_spark_session", return_value=None)
    def test_no_spark_session_runs_gc_only(self, mock_get_session, _):
        release_stage_memory()
        mock_get_session.assert_called_once()

    @patch("customer_retention.core.compat.is_databricks", return_value=True)
    @patch("customer_retention.core.compat.get_spark_session")
    def test_with_spark_session_calls_clear_cache(self, mock_get_session, _):
        mock_session = MagicMock()
        mock_get_session.return_value = mock_session
        release_stage_memory()
        mock_session.catalog.clearCache.assert_called_once()

    @patch("customer_retention.core.compat.is_databricks", return_value=True)
    @patch("customer_retention.core.compat.get_spark_session")
    def test_no_jvm_access(self, mock_get_session, _):
        """_jvm.System.gc() is no longer called — verify no JVM access."""
        mock_session = MagicMock()
        type(mock_session)._jvm = PropertyMock(
            side_effect=RuntimeError("[JVM_ATTRIBUTE_NOT_SUPPORTED]"))
        mock_get_session.return_value = mock_session
        release_stage_memory()
        mock_session.catalog.clearCache.assert_called_once()

    @patch("customer_retention.core.compat.is_databricks", return_value=True)
    @patch("customer_retention.core.compat.get_spark_session")
    def test_clear_cache_runtime_failure_is_swallowed(self, mock_get_session, _):
        """``release_stage_memory`` is best-effort memory hygiene — it must
        NOT fail the cell when ``catalog.clearCache()`` is rejected. Spark
        Connect on DBR 14+ shared clusters reuses the
        ``[NOT_SUPPORTED_WITH_SERVERLESS] CLEAR CACHE is not supported``
        error code whenever the access-mode policy is in the serverless-
        compatibility subset; that exception used to propagate and kill
        the c05 release cell. The function now catches broadly so the
        Python GC + tracked-object unpersist above still produce the
        cleanup win even when the cluster rejects a full cache clear."""
        mock_session = MagicMock()
        mock_session.catalog.clearCache.side_effect = RuntimeError(
            "[NOT_SUPPORTED_WITH_SERVERLESS] CLEAR CACHE is not supported"
        )
        mock_get_session.return_value = mock_session
        release_stage_memory()
        mock_session.catalog.clearCache.assert_called_once()

    @patch("customer_retention.core.compat.is_databricks", return_value=True)
    @patch("customer_retention.core.compat.get_spark_session")
    def test_clear_cache_spark_connect_grpc_failure_is_swallowed(self, mock_get_session, _):
        """Same contract, but with a generic ``Exception`` subclass modelling
        the real ``SparkConnectGrpcException`` the cluster raises. Confirms
        the catch is broad enough to handle the actual Spark Connect
        exception class, not just ``RuntimeError`` / ``OSError``."""
        class _FakeGrpcError(Exception):
            pass

        mock_session = MagicMock()
        mock_session.catalog.clearCache.side_effect = _FakeGrpcError(
            "[NOT_SUPPORTED_WITH_SERVERLESS] CLEAR CACHE is not supported"
        )
        mock_get_session.return_value = mock_session
        release_stage_memory()
        mock_session.catalog.clearCache.assert_called_once()

    @patch("customer_retention.core.compat.get_spark_session", return_value=None)
    def test_unpersist_spark_dataframe(self, _):
        mock_df = MagicMock()
        mock_df.unpersist = MagicMock()
        mock_df.to_spark = MagicMock()
        track_stage_object(mock_df)
        release_stage_memory()
        mock_df.unpersist.assert_called_once()
        assert len(compat_mod._stage_objects) == 0

    @patch("customer_retention.core.compat.get_spark_session", return_value=None)
    def test_unpersist_pyspark_pandas_dataframe(self, _):
        mock_psdf = MagicMock(spec=[])
        mock_psdf.spark = MagicMock()
        mock_psdf.spark.unpersist = MagicMock()
        track_stage_object(mock_psdf)
        release_stage_memory()
        mock_psdf.spark.unpersist.assert_called_once()

    @patch("customer_retention.core.compat.get_spark_session", return_value=None)
    def test_plain_objects_cleared_from_registry(self, _):
        track_stage_object({"large": "dict"}, [1, 2, 3])
        release_stage_memory()
        assert len(compat_mod._stage_objects) == 0

    @patch("customer_retention.core.compat.get_spark_session", return_value=None)
    def test_reentrant_track_release_cycle(self, _):
        track_stage_object(MagicMock())
        release_stage_memory()
        assert len(compat_mod._stage_objects) == 0
        track_stage_object(MagicMock(), MagicMock())
        release_stage_memory()
        assert len(compat_mod._stage_objects) == 0

    @patch("customer_retention.core.compat.get_spark_session", return_value=None)
    def test_unpersist_failure_tolerated(self, _):
        mock_df = MagicMock()
        mock_df.unpersist.side_effect = RuntimeError("already disposed")
        mock_df.to_spark = MagicMock()
        track_stage_object(mock_df)
        release_stage_memory()
        assert len(compat_mod._stage_objects) == 0
