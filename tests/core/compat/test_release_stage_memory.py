from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from customer_retention.core.compat import release_stage_memory


class TestReleaseStageMemory:

    @patch("customer_retention.core.compat.get_spark_session", return_value=None)
    def test_no_spark_session_runs_gc_only(self, mock_get_session):
        release_stage_memory()
        mock_get_session.assert_called_once()

    @patch("customer_retention.core.compat.get_spark_session")
    def test_with_spark_session_calls_clear_cache_and_jvm_gc(self, mock_get_session):
        mock_session = MagicMock()
        mock_get_session.return_value = mock_session

        release_stage_memory()

        mock_session.catalog.clearCache.assert_called_once()
        mock_session._jvm.System.gc.assert_called_once()

    @patch("customer_retention.core.compat.get_spark_session")
    def test_jvm_access_blocked_on_shared_cluster(self, mock_get_session):
        mock_session = MagicMock()
        type(mock_session)._jvm = PropertyMock(
            side_effect=RuntimeError("[JVM_ATTRIBUTE_NOT_SUPPORTED]"))
        mock_get_session.return_value = mock_session

        release_stage_memory()

        mock_session.catalog.clearCache.assert_called_once()

    @patch("customer_retention.core.compat.get_spark_session")
    def test_jvm_gc_attribute_error_tolerated(self, mock_get_session):
        mock_session = MagicMock()
        type(mock_session)._jvm = PropertyMock(
            side_effect=AttributeError("no _jvm"))
        mock_get_session.return_value = mock_session

        release_stage_memory()

        mock_session.catalog.clearCache.assert_called_once()

    @patch("customer_retention.core.compat.get_spark_session")
    def test_jvm_gc_method_raises_tolerated(self, mock_get_session):
        mock_session = MagicMock()
        mock_session._jvm.System.gc.side_effect = RuntimeError("gc failed")
        mock_get_session.return_value = mock_session

        release_stage_memory()

        mock_session.catalog.clearCache.assert_called_once()

    @patch("customer_retention.core.compat.get_spark_session")
    def test_clear_cache_failure_propagates(self, mock_get_session):
        mock_session = MagicMock()
        mock_session.catalog.clearCache.side_effect = RuntimeError("cluster gone")
        mock_get_session.return_value = mock_session

        with pytest.raises(RuntimeError, match="cluster gone"):
            release_stage_memory()
