from unittest.mock import MagicMock, patch

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

    def test_does_not_call_clear_cache(self):
        with patch("customer_retention.core.compat.get_spark_session") as mock_get:
            mock_session = MagicMock()
            mock_get.return_value = mock_session
            release_stage_memory()
            mock_session.catalog.clearCache.assert_not_called()

    def test_runs_without_spark_session(self):
        release_stage_memory()
        assert len(compat_mod._stage_objects) == 0

    def test_unpersist_spark_dataframe(self):
        mock_df = MagicMock()
        mock_df.unpersist = MagicMock()
        mock_df.to_spark = MagicMock()
        track_stage_object(mock_df)
        release_stage_memory()
        mock_df.unpersist.assert_called_once()
        assert len(compat_mod._stage_objects) == 0

    def test_unpersist_pyspark_pandas_dataframe(self):
        mock_psdf = MagicMock(spec=[])
        mock_psdf.spark = MagicMock()
        mock_psdf.spark.unpersist = MagicMock()
        track_stage_object(mock_psdf)
        release_stage_memory()
        mock_psdf.spark.unpersist.assert_called_once()

    def test_plain_objects_cleared_from_registry(self):
        track_stage_object({"large": "dict"}, [1, 2, 3])
        release_stage_memory()
        assert len(compat_mod._stage_objects) == 0

    def test_reentrant_track_release_cycle(self):
        track_stage_object(MagicMock())
        release_stage_memory()
        assert len(compat_mod._stage_objects) == 0
        track_stage_object(MagicMock(), MagicMock())
        release_stage_memory()
        assert len(compat_mod._stage_objects) == 0

    def test_unpersist_failure_tolerated(self):
        mock_df = MagicMock()
        mock_df.unpersist.side_effect = RuntimeError("already disposed")
        mock_df.to_spark = MagicMock()
        track_stage_object(mock_df)
        release_stage_memory()
        assert len(compat_mod._stage_objects) == 0
