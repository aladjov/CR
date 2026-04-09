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
    def _fake_sdf(self, columns):
        sdf = MagicMock()
        sdf.columns = columns
        return sdf

    def test_returns_qualified_global_temp_name(self):
        mock_sdf = self._fake_sdf(["ACCOUNT_ID", "churned"])
        result = register_temp_view(mock_sdf, "enriched_account", purpose="test")
        assert result == "global_temp.enriched_account"
        mock_sdf.createOrReplaceGlobalTempView.assert_called_once_with("enriched_account")

    def test_qualified_name_recognized_by_is_table_name(self):
        from customer_retention.analysis.auto_explorer.dataset_fingerprinter import is_table_name
        mock_sdf = self._fake_sdf(["ACCOUNT_ID"])
        name = register_temp_view(mock_sdf, "enriched_account", purpose="test")
        assert is_table_name(name)

    def test_different_view_names(self):
        assert register_temp_view(
            self._fake_sdf(["x"]), "foo", purpose="test",
        ) == "global_temp.foo"
        assert register_temp_view(
            self._fake_sdf(["x"]), "bar_baz", purpose="test",
        ) == "global_temp.bar_baz"

    def test_raises_on_case_insensitive_duplicate_columns(self):
        # Idempotency / collision guard: catches the bug at the registration
        # site, not 6 cells later when as_pandas_api iterates the schema and
        # the case-insensitive Spark resolver raises [AMBIGUOUS_REFERENCE].
        mock_sdf = self._fake_sdf(["CASE_ID", "as_of_date", "as_of_date", "Status"])
        with pytest.raises(ValueError, match="case-insensitive duplicate columns"):
            register_temp_view(mock_sdf, "broken_view", purpose="test")
        mock_sdf.createOrReplaceGlobalTempView.assert_not_called()

    def test_raises_on_mixed_case_duplicate(self):
        mock_sdf = self._fake_sdf(["CASE_ID", "Origin", "ORIGIN"])
        with pytest.raises(ValueError, match="ORIGIN.*Origin|Origin.*ORIGIN"):
            register_temp_view(mock_sdf, "broken_view", purpose="test")

    def test_empty_columns_does_not_raise(self):
        # Defensive: an empty schema should not crash the registration call.
        mock_sdf = self._fake_sdf([])
        register_temp_view(mock_sdf, "empty_view", purpose="test")
        mock_sdf.createOrReplaceGlobalTempView.assert_called_once_with("empty_view")

    def test_purpose_is_required(self):
        # The audit-trail kwarg is required so every call site is reviewable —
        # temp views are session-scoped, code review must catch misuse.
        mock_sdf = self._fake_sdf(["x"])
        with pytest.raises(TypeError, match="purpose"):
            register_temp_view(mock_sdf, "foo")  # type: ignore[call-arg]
        mock_sdf.createOrReplaceGlobalTempView.assert_not_called()

    def test_purpose_must_be_non_empty(self):
        mock_sdf = self._fake_sdf(["x"])
        with pytest.raises(ValueError, match="purpose"):
            register_temp_view(mock_sdf, "foo", purpose="")
        mock_sdf.createOrReplaceGlobalTempView.assert_not_called()

    def test_purpose_must_not_be_whitespace(self):
        mock_sdf = self._fake_sdf(["x"])
        with pytest.raises(ValueError, match="purpose"):
            register_temp_view(mock_sdf, "foo", purpose="   ")
        mock_sdf.createOrReplaceGlobalTempView.assert_not_called()

    def test_purpose_must_be_string(self):
        mock_sdf = self._fake_sdf(["x"])
        with pytest.raises(ValueError, match="purpose"):
            register_temp_view(mock_sdf, "foo", purpose=42)  # type: ignore[arg-type]
        mock_sdf.createOrReplaceGlobalTempView.assert_not_called()


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
