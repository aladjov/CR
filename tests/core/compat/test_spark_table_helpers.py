from unittest.mock import MagicMock, patch

import pytest

from customer_retention.core.compat import as_pandas_api, load_spark_table, register_temp_view


class TestLoadSparkTable:
    def test_raises_without_spark_session(self):
        with patch("customer_retention.core.compat.get_spark_session", return_value=None):
            with pytest.raises(RuntimeError, match="No active Spark session"):
                load_spark_table("catalog.schema.table")

    def test_delegates_to_spark_table(self):
        mock_spark = MagicMock()
        mock_spark.table.return_value = MagicMock(name="spark_df")
        with patch("customer_retention.core.compat.get_spark_session", return_value=mock_spark):
            result = load_spark_table("catalog.schema.table")
        mock_spark.table.assert_called_once_with("catalog.schema.table")
        assert result is mock_spark.table.return_value

    def test_propagates_spark_errors(self):
        mock_spark = MagicMock()
        mock_spark.table.side_effect = Exception("Table not found")
        with patch("customer_retention.core.compat.get_spark_session", return_value=mock_spark):
            with pytest.raises(Exception, match="Table not found"):
                load_spark_table("catalog.schema.missing")


class TestRegisterTempView:
    def test_returns_qualified_global_temp_name(self):
        mock_sdf = MagicMock()
        result = register_temp_view(mock_sdf, "enriched_account")
        assert result == "global_temp.enriched_account"
        mock_sdf.createOrReplaceGlobalTempView.assert_called_once_with("enriched_account")

    def test_qualified_name_recognized_by_is_table_name(self):
        from customer_retention.analysis.auto_explorer.dataset_fingerprinter import is_table_name
        mock_sdf = MagicMock()
        name = register_temp_view(mock_sdf, "enriched_account")
        assert is_table_name(name)

    def test_different_view_names(self):
        mock_sdf = MagicMock()
        assert register_temp_view(mock_sdf, "foo") == "global_temp.foo"
        assert register_temp_view(mock_sdf, "bar_baz") == "global_temp.bar_baz"


class TestAsPandasApi:
    @pytest.fixture(autouse=True)
    def _skip_without_pyspark(self):
        pytest.importorskip("pyspark")

    def test_calls_pandas_api_when_available(self):
        mock_sdf = MagicMock()
        mock_sdf.pandas_api.return_value = MagicMock(name="psdf")
        result = as_pandas_api(mock_sdf)
        mock_sdf.pandas_api.assert_called_once()
        assert result is mock_sdf.pandas_api.return_value

    def test_falls_back_to_to_pandas_on_spark(self):
        class FakeSparkDF:
            def to_pandas_on_spark(self):
                return "fallback_result"

        result = as_pandas_api(FakeSparkDF())
        assert result == "fallback_result"

    def test_load_and_convert_roundtrip(self):
        mock_spark = MagicMock()
        mock_sdf = MagicMock()
        mock_psdf = MagicMock(name="psdf")
        mock_spark.table.return_value = mock_sdf
        mock_sdf.pandas_api.return_value = mock_psdf
        with patch("customer_retention.core.compat.get_spark_session", return_value=mock_spark):
            result = as_pandas_api(load_spark_table("catalog.schema.table"))
        assert result is mock_psdf
