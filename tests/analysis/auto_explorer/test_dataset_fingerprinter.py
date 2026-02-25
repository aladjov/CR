from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from customer_retention.analysis.auto_explorer.dataset_fingerprinter import (
    DatasetFingerprint,
    DatasetFingerprinter,
    is_table_name,
)
from customer_retention.core.config.column_config import DatasetGranularity

FIXTURES = Path(__file__).parent.parent.parent / "fixtures"
TINY_PROFILES = FIXTURES / "3set_tiny_customer_profiles.csv"
TINY_TRANSACTIONS = FIXTURES / "3set_tiny_edi_transactions.csv"
TINY_TICKETS = FIXTURES / "3set_tiny_support_tickets.csv"


def _entity_level_df():
    return pd.DataFrame({
        "customer_id": ["C001", "C002", "C003", "C004", "C005"],
        "age": [25, 35, 45, 55, 65],
        "churned": [0, 1, 0, 1, 0],
    })


def _event_level_df():
    return pd.DataFrame({
        "customer_id": ["C001", "C001", "C001", "C002", "C002",
                        "C003", "C003", "C003", "C003", "C004"],
        "event_timestamp": pd.to_datetime([
            "2024-01-01", "2024-02-01", "2024-03-01",
            "2024-01-15", "2024-04-15",
            "2024-01-01", "2024-03-01", "2024-06-01", "2024-09-01",
            "2024-01-01",
        ]),
        "amount": [100, 200, 150, 300, 250, 50, 75, 100, 125, 500],
    })


class TestDatasetFingerprintDataclass:
    def test_creation(self):
        fp = DatasetFingerprint(
            name="test",
            row_count=100,
            column_count=5,
            entity_columns=["customer_id"],
            time_columns=[],
            target_candidates=["churned"],
            granularity=DatasetGranularity.ENTITY_LEVEL,
        )
        assert fp.name == "test"
        assert fp.row_count == 100
        assert fp.entity_column is None
        assert fp.time_column is None

    def test_defaults(self):
        fp = DatasetFingerprint(
            name="test", row_count=10, column_count=3,
            entity_columns=[], time_columns=[], target_candidates=[],
            granularity=DatasetGranularity.UNKNOWN,
        )
        assert fp.unique_entities is None
        assert fp.avg_rows_per_entity is None
        assert fp.temporal_span_days is None


class TestFingerprinterEntityLevel:
    def test_detects_entity_level(self):
        fp = DatasetFingerprinter().fingerprint("profiles", _entity_level_df())
        assert fp.granularity == DatasetGranularity.ENTITY_LEVEL

    def test_detects_entity_column(self):
        fp = DatasetFingerprinter().fingerprint("profiles", _entity_level_df())
        assert fp.entity_column == "customer_id"

    def test_entity_columns_list(self):
        fp = DatasetFingerprinter().fingerprint("profiles", _entity_level_df())
        assert "customer_id" in fp.entity_columns

    def test_row_and_column_count(self):
        fp = DatasetFingerprinter().fingerprint("profiles", _entity_level_df())
        assert fp.row_count == 5
        assert fp.column_count == 3

    def test_target_candidate_detection(self):
        fp = DatasetFingerprinter().fingerprint("profiles", _entity_level_df())
        assert "churned" in fp.target_candidates

    def test_unique_entities(self):
        fp = DatasetFingerprinter().fingerprint("profiles", _entity_level_df())
        assert fp.unique_entities == 5

    def test_no_time_columns(self):
        fp = DatasetFingerprinter().fingerprint("profiles", _entity_level_df())
        assert fp.time_columns == []
        assert fp.time_column is None


class TestFingerprinterEventLevel:
    def test_detects_event_level(self):
        fp = DatasetFingerprinter().fingerprint("events", _event_level_df())
        assert fp.granularity == DatasetGranularity.EVENT_LEVEL

    def test_detects_time_column(self):
        fp = DatasetFingerprinter().fingerprint("events", _event_level_df())
        assert fp.time_column == "event_timestamp"

    def test_time_columns_list(self):
        fp = DatasetFingerprinter().fingerprint("events", _event_level_df())
        assert "event_timestamp" in fp.time_columns

    def test_unique_entities(self):
        fp = DatasetFingerprinter().fingerprint("events", _event_level_df())
        assert fp.unique_entities == 4

    def test_avg_rows_per_entity(self):
        fp = DatasetFingerprinter().fingerprint("events", _event_level_df())
        assert fp.avg_rows_per_entity == pytest.approx(2.5)

    def test_temporal_span(self):
        fp = DatasetFingerprinter().fingerprint("events", _event_level_df())
        assert fp.temporal_span_days is not None
        assert fp.temporal_span_days > 0

    def test_no_target_candidates(self):
        fp = DatasetFingerprinter().fingerprint("events", _event_level_df())
        assert fp.target_candidates == []


class TestFingerprinterFromPath:
    def test_load_from_csv_string(self):
        fp = DatasetFingerprinter().fingerprint("profiles", str(TINY_PROFILES))
        assert fp.row_count > 0
        assert fp.name == "profiles"

    def test_load_from_path_object(self):
        fp = DatasetFingerprinter().fingerprint("profiles", TINY_PROFILES)
        assert fp.row_count > 0

    def test_fingerprint_all_with_path_objects(self):
        datasets = {"profiles": TINY_PROFILES, "transactions": TINY_TRANSACTIONS}
        results = DatasetFingerprinter().fingerprint_all(datasets)
        assert len(results) == 2
        assert all(fp.row_count > 0 for fp in results.values())

    def test_count_rows_with_path_object(self):
        assert DatasetFingerprinter()._count_rows(TINY_PROFILES) > 0

    def test_load_with_path_object(self):
        assert len(DatasetFingerprinter()._load(TINY_PROFILES)) > 0


class TestFingerprinterDataFrameDetection:
    def test_fingerprint_accepts_dataframe_regardless_of_pd_alias(self):
        df = _entity_level_df()
        fp = DatasetFingerprinter().fingerprint("test", df)
        assert fp.row_count == 5

    def test_count_rows_uses_duck_typing(self):
        df = _entity_level_df()
        assert DatasetFingerprinter()._count_rows(df) == 5

    def test_load_uses_duck_typing(self):
        df = _entity_level_df()
        result = DatasetFingerprinter()._load(df)
        assert len(result) == 5


class TestFingerprinterMultiDataset:
    def test_fingerprint_all(self):
        datasets = {
            "profiles": _entity_level_df(),
            "events": _event_level_df(),
        }
        results = DatasetFingerprinter().fingerprint_all(datasets)
        assert len(results) == 2
        assert "profiles" in results
        assert "events" in results

    def test_fingerprint_all_types(self):
        results = DatasetFingerprinter().fingerprint_all({"df": _entity_level_df()})
        assert isinstance(results["df"], DatasetFingerprint)


class TestFingerprinterSummaryDataframe:
    def test_summary_shape(self):
        datasets = {
            "profiles": _entity_level_df(),
            "events": _event_level_df(),
        }
        fps = DatasetFingerprinter().fingerprint_all(datasets)
        summary = DatasetFingerprinter.to_summary_dataframe(fps)
        assert len(summary) == 2
        assert "name" in summary.columns
        assert "rows" in summary.columns
        assert "granularity" in summary.columns

    def test_summary_columns(self):
        fps = DatasetFingerprinter().fingerprint_all({"df": _entity_level_df()})
        summary = DatasetFingerprinter.to_summary_dataframe(fps)
        expected_cols = {"name", "rows", "columns", "granularity",
                         "entity_column", "time_column",
                         "target_candidates", "sampled"}
        assert expected_cols.issubset(set(summary.columns))

    def test_summary_excludes_entity_stats_columns(self):
        fps = DatasetFingerprinter().fingerprint_all({"df": _entity_level_df()})
        summary = DatasetFingerprinter.to_summary_dataframe(fps)
        assert "unique_entities" not in summary.columns
        assert "avg_rows_per_entity" not in summary.columns
        assert "temporal_span_days" not in summary.columns


class TestFingerprinterNrowsSampling:
    def test_respects_nrows_limit(self):
        big_df = pd.DataFrame({
            "customer_id": [f"C{i:04d}" for i in range(500)],
            "value": range(500),
        })
        fp = DatasetFingerprinter(nrows=100).fingerprint("big", big_df)
        assert fp.row_count == 500
        assert fp.sampled is True

    def test_small_df_not_affected(self):
        fp = DatasetFingerprinter(nrows=1000).fingerprint("small", _entity_level_df())
        assert fp.row_count == 5
        assert fp.sampled is False


class TestFingerprinterTinyFixtures:
    def test_profiles_has_target(self):
        fp = DatasetFingerprinter().fingerprint("profiles", TINY_PROFILES)
        assert "churned" in fp.target_candidates

    def test_profiles_entity_level(self):
        fp = DatasetFingerprinter().fingerprint("profiles", TINY_PROFILES)
        assert fp.granularity == DatasetGranularity.ENTITY_LEVEL

    def test_profiles_entity_column(self):
        fp = DatasetFingerprinter().fingerprint("profiles", TINY_PROFILES)
        assert fp.entity_column == "customer_id"


class TestFingerprintDataSpan:
    def test_event_level_has_data_start_and_end(self):
        fp = DatasetFingerprinter().fingerprint("events", _event_level_df())
        assert fp.data_start is not None
        assert fp.data_end is not None
        assert fp.data_start == "2024-01-01"
        assert fp.data_end == "2024-09-01"

    def test_entity_level_no_time_column_returns_none(self):
        fp = DatasetFingerprinter().fingerprint("profiles", _entity_level_df())
        assert fp.data_start is None
        assert fp.data_end is None

    def test_sampled_data_no_dates(self):
        big_df = pd.DataFrame({
            "customer_id": [f"C{i:04d}" for i in range(500)] * 2,
            "event_timestamp": pd.to_datetime(
                ["2024-01-01"] * 500 + ["2024-06-01"] * 500
            ),
            "amount": range(1000),
        })
        fp = DatasetFingerprinter(nrows=100).fingerprint("big", big_df)
        assert fp.data_start is None
        assert fp.data_end is None

    def test_dates_consistent_with_temporal_span(self):
        fp = DatasetFingerprinter().fingerprint("events", _event_level_df())
        from datetime import date
        start = date.fromisoformat(fp.data_start)
        end = date.fromisoformat(fp.data_end)
        assert (end - start).days == fp.temporal_span_days


class TestIsTableName:
    def test_three_part_unity_catalog_name(self):
        assert is_table_name("catalog.schema.table") is True

    def test_two_part_schema_table(self):
        assert is_table_name("schema.table_name") is True

    def test_real_snowflake_table(self):
        assert is_table_name("snowflake_corp.salesforce.contract") is True

    def test_long_dotted_name(self):
        assert is_table_name("prod_networkdata.reporting_gold.customer_visible_transaction_volume_daily") is True

    def test_two_part_catalog_name(self):
        assert is_table_name("prod_networkdata.orderexchange_gold") is True

    def test_csv_path_not_table(self):
        assert is_table_name("data/file.csv") is False

    def test_parquet_path_not_table(self):
        assert is_table_name("data/file.parquet") is False

    def test_relative_path_not_table(self):
        assert is_table_name("../tests/fixtures/data.csv") is False

    def test_absolute_path_not_table(self):
        assert is_table_name("/home/user/data.csv") is False

    def test_s3_path_not_table(self):
        assert is_table_name("s3://bucket/path/file.parquet") is False

    def test_abfss_path_not_table(self):
        assert is_table_name("abfss://container@account.dfs.core.windows.net/path") is False

    def test_dbfs_path_not_table(self):
        assert is_table_name("dbfs:/mnt/data/file.csv") is False

    def test_plain_name_without_dots_not_table(self):
        assert is_table_name("my_table") is False

    def test_local_csv_without_path_not_table(self):
        assert is_table_name("data.csv") is False

    def test_local_parquet_without_path_not_table(self):
        assert is_table_name("data.parquet") is False

    def test_delta_extension_not_table(self):
        assert is_table_name("data.delta") is False


class TestTableLoading:
    def _mock_spark(self, monkeypatch, mock_spark):
        monkeypatch.setattr(DatasetFingerprinter, "_ensure_spark", staticmethod(lambda: mock_spark))

    def test_load_dispatches_to_spark_table(self, monkeypatch):
        mock_spark = MagicMock()
        mock_df = pd.DataFrame({"a": [1, 2, 3]})
        mock_spark.table.return_value.limit.return_value.toPandas.return_value = mock_df
        self._mock_spark(monkeypatch, mock_spark)
        result = DatasetFingerprinter(nrows=100)._load("catalog.schema.my_table")
        mock_spark.table.assert_called_once_with("catalog.schema.my_table")
        assert len(result) == 3

    def test_count_rows_dispatches_to_spark_table(self, monkeypatch):
        mock_spark = MagicMock()
        mock_spark.table.return_value.count.return_value = 42
        self._mock_spark(monkeypatch, mock_spark)
        result = DatasetFingerprinter()._count_rows("catalog.schema.my_table")
        mock_spark.table.assert_called_once_with("catalog.schema.my_table")
        assert result == 42

    def test_load_table_applies_nrows_limit(self, monkeypatch):
        mock_spark = MagicMock()
        mock_spark.table.return_value.limit.return_value.toPandas.return_value = pd.DataFrame({"x": [1]})
        self._mock_spark(monkeypatch, mock_spark)
        DatasetFingerprinter(nrows=500)._load("schema.table")
        mock_spark.table.return_value.limit.assert_called_once_with(500)

    def test_fingerprint_all_with_table_names(self, monkeypatch):
        mock_spark = MagicMock()
        mock_df = pd.DataFrame({
            "customer_id": ["C001", "C002", "C003"],
            "amount": [100, 200, 300],
        })
        mock_spark.table.return_value.limit.return_value.toPandas.return_value = mock_df
        mock_spark.table.return_value.count.return_value = 3
        self._mock_spark(monkeypatch, mock_spark)
        results = DatasetFingerprinter().fingerprint_all({
            "contracts": "catalog.schema.contracts",
        })
        assert "contracts" in results
        assert results["contracts"].row_count == 3

    def test_load_table_no_spark_raises(self, monkeypatch):
        self._mock_spark(monkeypatch, None)
        with pytest.raises(RuntimeError, match="Spark session"):
            DatasetFingerprinter()._load("catalog.schema.my_table")

    def test_count_rows_table_no_spark_raises(self, monkeypatch):
        self._mock_spark(monkeypatch, None)
        with pytest.raises(RuntimeError, match="Spark session"):
            DatasetFingerprinter()._count_rows("catalog.schema.my_table")

    def test_file_path_still_loads_locally(self):
        fp = DatasetFingerprinter().fingerprint("profiles", str(TINY_PROFILES))
        assert fp.row_count > 0

    def test_dataframe_input_unaffected(self):
        fp = DatasetFingerprinter().fingerprint("test", _entity_level_df())
        assert fp.row_count == 5
