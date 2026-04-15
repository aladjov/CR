from unittest.mock import MagicMock, patch

import pytest

from customer_retention.core.compat import as_pandas_api, load_spark_table, register_temp_view


class TestLoadSparkTable:
    def test_raises_without_spark_session(self):
        with patch("customer_retention.core.compat.get_spark_session", return_value=None):
            with pytest.raises(RuntimeError, match="No active Spark session"):
                load_spark_table("catalog.schema.table")

    def test_passes_3_part_name_unchanged(self):
        mock_spark = MagicMock()
        mock_spark.table.return_value = MagicMock(name="spark_df")
        with patch("customer_retention.core.compat.get_spark_session", return_value=mock_spark):
            result = load_spark_table("catalog.schema.table")
        mock_spark.table.assert_called_once_with("catalog.schema.table")
        mock_spark.catalog.currentCatalog.assert_not_called()
        mock_spark.catalog.currentDatabase.assert_not_called()
        assert result is mock_spark.table.return_value

    def test_qualifies_2_part_name_with_current_catalog(self):
        # Spark Connect / multi-task Databricks jobs can re-analyze plans with
        # a different current_catalog than at load time — pinning to the 3-part
        # identifier keeps the plan robust.
        mock_spark = MagicMock()
        mock_spark.catalog.currentCatalog.return_value = "main"
        mock_spark.catalog.currentDatabase.return_value = "bronze"
        with patch("customer_retention.core.compat.get_spark_session", return_value=mock_spark):
            load_spark_table("bronze.contract")
        mock_spark.table.assert_called_once_with("main.bronze.contract")

    def test_qualifies_1_part_name_with_current_catalog_and_schema(self):
        mock_spark = MagicMock()
        mock_spark.catalog.currentCatalog.return_value = "main"
        mock_spark.catalog.currentDatabase.return_value = "bronze"
        with patch("customer_retention.core.compat.get_spark_session", return_value=mock_spark):
            load_spark_table("contract")
        mock_spark.table.assert_called_once_with("main.bronze.contract")

    def test_qualification_falls_back_on_catalog_api_error(self):
        # Older Spark builds (pre 3.4) lack currentCatalog — don't break loads.
        mock_spark = MagicMock()
        mock_spark.catalog.currentCatalog.side_effect = Exception("not supported")
        with patch("customer_retention.core.compat.get_spark_session", return_value=mock_spark):
            load_spark_table("schema.contract")
        mock_spark.table.assert_called_once_with("schema.contract")

    def test_qualification_skipped_when_current_values_empty(self):
        mock_spark = MagicMock()
        mock_spark.catalog.currentCatalog.return_value = ""
        mock_spark.catalog.currentDatabase.return_value = None
        with patch("customer_retention.core.compat.get_spark_session", return_value=mock_spark):
            load_spark_table("contract")
        mock_spark.table.assert_called_once_with("contract")

    def test_propagates_spark_errors(self):
        mock_spark = MagicMock()
        mock_spark.table.side_effect = Exception("Table not found")
        with patch("customer_retention.core.compat.get_spark_session", return_value=mock_spark):
            with pytest.raises(Exception, match="Table not found"):
                load_spark_table("catalog.schema.missing")

    def test_global_temp_name_not_catalog_qualified(self):
        # `global_temp` is Spark's session-scoped view schema; it must be read
        # as "global_temp.<view>" directly. Prepending a catalog makes the
        # lookup fail with TABLE_OR_VIEW_NOT_FOUND — the exact failure that
        # register_temp_view + load_spark_table would hit on Databricks.
        mock_spark = MagicMock()
        mock_spark.conf.get.return_value = "global_temp"
        mock_spark.catalog.currentCatalog.return_value = "main"
        mock_spark.catalog.currentDatabase.return_value = "bronze"
        with patch("customer_retention.core.compat.get_spark_session", return_value=mock_spark):
            load_spark_table("global_temp.sps_enriched_contract")
        mock_spark.table.assert_called_once_with("global_temp.sps_enriched_contract")

    def test_global_temp_honors_custom_global_temp_database_conf(self):
        mock_spark = MagicMock()
        mock_spark.conf.get.return_value = "my_temp_db"
        mock_spark.catalog.currentCatalog.return_value = "main"
        mock_spark.catalog.currentDatabase.return_value = "bronze"
        with patch("customer_retention.core.compat.get_spark_session", return_value=mock_spark):
            load_spark_table("my_temp_db.enriched_view")
        mock_spark.table.assert_called_once_with("my_temp_db.enriched_view")

    def test_register_then_load_roundtrip(self):
        # The critical contract: a view name returned by register_temp_view
        # must be directly consumable by load_spark_table — otherwise the
        # NB00 enrichment → derive_target flow can't hand data forward.
        mock_spark = MagicMock()
        mock_spark.conf.get.return_value = "global_temp"
        mock_spark.catalog.currentCatalog.return_value = "main"
        mock_spark.catalog.currentDatabase.return_value = "bronze"
        mock_sdf = MagicMock()
        mock_sdf.columns = ["entity_id", "churned"]
        name = register_temp_view(mock_sdf, "enriched_contract", purpose="roundtrip-test")
        assert name == "global_temp.enriched_contract"
        with patch("customer_retention.core.compat.get_spark_session", return_value=mock_spark):
            load_spark_table(name)
        mock_spark.table.assert_called_once_with("global_temp.enriched_contract")


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
