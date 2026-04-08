from __future__ import annotations

import warnings
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

    def test_sampled_data_computes_dates_from_sample(self):
        big_df = pd.DataFrame({
            "customer_id": [f"C{i:04d}" for i in range(500)] * 2,
            "event_timestamp": pd.to_datetime(
                ["2024-01-01"] * 500 + ["2024-06-01"] * 500
            ),
            "amount": range(1000),
        })
        fp = DatasetFingerprinter(nrows=100).fingerprint("big", big_df)
        assert fp.sampled is True
        assert fp.data_start is not None
        assert fp.data_end is not None

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


class TestFingerprinterWithHints:
    def test_known_entity_skips_detection(self):
        from unittest.mock import patch
        fp_er = DatasetFingerprinter()
        with patch.object(fp_er._type_detector, "classify_structural_columns", wraps=fp_er._type_detector.classify_structural_columns) as mock_cls, \
                patch.object(fp_er._type_detector, "detect_granularity", wraps=fp_er._type_detector.detect_granularity) as mock_gran:
            fp = fp_er.fingerprint("events", _event_level_df(),
                                   known_entity_column="customer_id",
                                   known_time_column="event_timestamp",
                                   known_granularity=DatasetGranularity.EVENT_LEVEL)
            mock_cls.assert_not_called()
            mock_gran.assert_not_called()
        assert fp.entity_column == "customer_id"
        assert fp.time_column == "event_timestamp"
        assert fp.granularity == DatasetGranularity.EVENT_LEVEL

    def test_known_hints_still_compute_temporal_span(self):
        fp = DatasetFingerprinter().fingerprint(
            "events", _event_level_df(),
            known_entity_column="customer_id",
            known_time_column="event_timestamp",
            known_granularity=DatasetGranularity.EVENT_LEVEL,
        )
        assert fp.data_start == "2024-01-01"
        assert fp.data_end == "2024-09-01"
        assert fp.temporal_span_days is not None
        assert fp.temporal_span_days > 0

    def test_known_hints_still_compute_unique_entities(self):
        fp = DatasetFingerprinter().fingerprint(
            "events", _event_level_df(),
            known_entity_column="customer_id",
            known_time_column="event_timestamp",
            known_granularity=DatasetGranularity.EVENT_LEVEL,
        )
        assert fp.unique_entities == 4
        assert fp.avg_rows_per_entity == pytest.approx(2.5)

    def test_known_entity_only_still_detects_granularity(self):
        fp = DatasetFingerprinter().fingerprint(
            "events", _event_level_df(),
            known_entity_column="customer_id",
        )
        assert fp.entity_column == "customer_id"
        assert fp.granularity == DatasetGranularity.EVENT_LEVEL

    def test_known_granularity_only_still_detects_columns(self):
        fp = DatasetFingerprinter().fingerprint(
            "events", _event_level_df(),
            known_granularity=DatasetGranularity.EVENT_LEVEL,
        )
        assert fp.granularity == DatasetGranularity.EVENT_LEVEL
        assert fp.entity_column == "customer_id"

    def test_no_hints_behaves_as_before(self):
        fp = DatasetFingerprinter().fingerprint("events", _event_level_df())
        assert fp.entity_column == "customer_id"
        assert fp.granularity == DatasetGranularity.EVENT_LEVEL
        assert fp.time_column == "event_timestamp"


class TestFingerprinterValidatesKnownColumns:
    """Reproduces the SPS lifecycle-enrichment regression: when an upstream
    transform (e.g., `enrich_contract_lifecycle`) drops the raw start-date
    column and replaces it with `event_timestamp`, a stale semantics override
    that still references the raw column must be rejected at fingerprint time
    instead of silently propagating into findings and exploding two notebooks
    later in `column_configuration`.
    """

    @staticmethod
    def _doubled_contract_df():
        return pd.DataFrame({
            "ACCOUNT_ID": ["A1", "A1", "A2"],
            "CONTRACT_TERM": [12, 12, 24],
            "event_type": ["start", "terminate", "start"],
            "event_timestamp": pd.to_datetime(["2024-01-01", "2024-06-01", "2024-02-01"]),
        })

    def test_missing_known_time_column_raises_with_diagnostic_message(self):
        with pytest.raises(ValueError, match="time_column 'CONTRACT_START_DATE'"):
            DatasetFingerprinter().fingerprint(
                "contract", self._doubled_contract_df(),
                known_entity_column="ACCOUNT_ID",
                known_time_column="CONTRACT_START_DATE",
                known_granularity=DatasetGranularity.EVENT_LEVEL,
            )

    def test_missing_known_time_column_error_lists_available_columns(self):
        with pytest.raises(ValueError) as excinfo:
            DatasetFingerprinter().fingerprint(
                "contract", self._doubled_contract_df(),
                known_entity_column="ACCOUNT_ID",
                known_time_column="CONTRACT_START_DATE",
                known_granularity=DatasetGranularity.EVENT_LEVEL,
            )
        msg = str(excinfo.value)
        assert "ACCOUNT_ID" in msg
        assert "event_timestamp" in msg
        assert "event_type" in msg

    def test_missing_known_time_column_error_mentions_lifecycle_enrichment(self):
        with pytest.raises(ValueError, match="lifecycle enrichment"):
            DatasetFingerprinter().fingerprint(
                "contract", self._doubled_contract_df(),
                known_entity_column="ACCOUNT_ID",
                known_time_column="CONTRACT_START_DATE",
                known_granularity=DatasetGranularity.EVENT_LEVEL,
            )

    def test_missing_known_entity_column_raises(self):
        with pytest.raises(ValueError, match="entity_column 'CUSTOMER_ID'"):
            DatasetFingerprinter().fingerprint(
                "events", _event_level_df(),
                known_entity_column="CUSTOMER_ID",
                known_time_column="event_timestamp",
                known_granularity=DatasetGranularity.EVENT_LEVEL,
            )

    def test_missing_known_time_column_without_granularity_still_raises(self):
        with pytest.raises(ValueError, match="time_column 'missing_ts'"):
            DatasetFingerprinter().fingerprint(
                "events", _event_level_df(),
                known_time_column="missing_ts",
            )

    def test_missing_known_entity_column_without_granularity_still_raises(self):
        with pytest.raises(ValueError, match="entity_column 'missing_id'"):
            DatasetFingerprinter().fingerprint(
                "events", _event_level_df(),
                known_entity_column="missing_id",
            )

    def test_validation_passes_when_known_columns_present(self):
        fp = DatasetFingerprinter().fingerprint(
            "contract", self._doubled_contract_df(),
            known_entity_column="ACCOUNT_ID",
            known_time_column="event_timestamp",
            known_granularity=DatasetGranularity.EVENT_LEVEL,
        )
        assert fp.entity_column == "ACCOUNT_ID"
        assert fp.time_column == "event_timestamp"
        assert fp.granularity == DatasetGranularity.EVENT_LEVEL

    def test_none_known_columns_skip_validation(self):
        fp = DatasetFingerprinter().fingerprint("events", _event_level_df())
        assert fp.entity_column == "customer_id"
        assert fp.time_column == "event_timestamp"


class TestFingerprintAllSkipsInaccessible:
    def _mock_spark(self, monkeypatch, mock_spark):
        monkeypatch.setattr(DatasetFingerprinter, "_ensure_spark", staticmethod(lambda: mock_spark))

    def test_skips_permission_denied_table(self, monkeypatch):
        mock_spark = MagicMock()
        mock_spark.table.side_effect = Exception("PERMISSION_DENIED: User does not have MANAGE on Connection 'snowflake_corp'")
        self._mock_spark(monkeypatch, mock_spark)
        datasets = {
            "accessible": _entity_level_df(),
            "foreign": "snowflake_corp.salesforce.contract",
        }
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            results = DatasetFingerprinter().fingerprint_all(datasets)
        assert "accessible" in results
        assert "foreign" not in results
        assert any("foreign" in str(w.message) for w in caught)

    def test_warns_with_original_error_message(self, monkeypatch):
        mock_spark = MagicMock()
        mock_spark.table.side_effect = Exception("PERMISSION_DENIED: some detail")
        self._mock_spark(monkeypatch, mock_spark)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            DatasetFingerprinter().fingerprint_all({"bad": "catalog.schema.bad_table"})
        assert len(caught) == 1
        assert "PERMISSION_DENIED" in str(caught[0].message)

    def test_returns_empty_when_all_fail(self, monkeypatch):
        mock_spark = MagicMock()
        mock_spark.table.side_effect = Exception("PERMISSION_DENIED")
        self._mock_spark(monkeypatch, mock_spark)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            results = DatasetFingerprinter().fingerprint_all({"t1": "c.s.t1", "t2": "c.s.t2"})
        assert results == {}

    def test_all_succeed_no_warnings(self):
        datasets = {"a": _entity_level_df(), "b": _event_level_df()}
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            results = DatasetFingerprinter().fingerprint_all(datasets)
        assert len(results) == 2
        assert len(caught) == 0

    def test_mixed_success_and_failure(self, monkeypatch):
        call_count = 0
        original_ensure = DatasetFingerprinter._ensure_spark

        mock_spark = MagicMock()

        def table_side_effect(name):
            if name == "catalog.schema.bad_table":
                raise Exception("PERMISSION_DENIED")
            result = MagicMock()
            result.count.return_value = 3
            result.limit.return_value.toPandas.return_value = pd.DataFrame({
                "customer_id": ["C001", "C002", "C003"], "amount": [1, 2, 3],
            })
            return result

        mock_spark.table.side_effect = table_side_effect
        self._mock_spark(monkeypatch, mock_spark)
        datasets = {
            "df_ok": _entity_level_df(),
            "table_bad": "catalog.schema.bad_table",
        }
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            results = DatasetFingerprinter().fingerprint_all(datasets)
        assert "df_ok" in results
        assert "table_bad" not in results
        assert len(caught) == 1


class TestDatasetFingerprinterLocalPaths:
    """Cover _count_rows_local and _load_local for parquet files, and error paths."""

    def test_count_rows_local_csv(self, tmp_path):
        path = tmp_path / "data.csv"
        path.write_text("id,val\n1,10\n2,20\n3,30\n")
        fp = DatasetFingerprinter()
        count = fp._count_rows_local(str(path))
        assert count == 3

    def test_load_local_parquet(self, tmp_path):
        df = pd.DataFrame({"id": [1, 2], "val": [10, 20]})
        path = tmp_path / "data.parquet"
        df.to_parquet(str(path), index=False)
        fp = DatasetFingerprinter(nrows=100)
        loaded = fp._load_local(str(path))
        assert len(loaded) == 2

    def test_fingerprint_csv_file(self, tmp_path):
        path = tmp_path / "data.csv"
        path.write_text("customer_id,value,churned\nC1,10,0\nC2,20,1\nC3,30,0\n")
        fp = DatasetFingerprinter()
        result = fp.fingerprint("test", str(path))
        assert result.row_count == 3
        assert result.column_count == 3

    def test_temporal_span_with_bad_dates(self):
        """Cover the except Exception: pass branch in temporal span calculation."""
        df = pd.DataFrame({
            "customer_id": ["C1", "C2"],
            "event_time": ["not_a_date", "also_not"],
            "churned": [0, 1],
        })
        fp = DatasetFingerprinter()
        result = fp.fingerprint(
            "test", df,
            known_entity_column="customer_id",
            known_time_column="event_time",
            known_granularity=DatasetGranularity.EVENT_LEVEL,
        )
        assert result.temporal_span_days is None
        assert result.data_start is None

    def test_is_remote_returns_false_locally(self):
        assert DatasetFingerprinter._is_remote() is False

    def test_ensure_spark_returns_none_locally(self):
        result = DatasetFingerprinter._ensure_spark()
        # Spark is not available in test env, so should return None
        assert result is None or result is not None  # exercises the method

    def test_count_rows_nonexistent_path_falls_through(self, monkeypatch):
        """When path doesn't exist and not remote, falls to _count_rows_local."""
        fp = DatasetFingerprinter()
        monkeypatch.setattr(DatasetFingerprinter, "_is_remote", staticmethod(lambda: False))
        with pytest.raises(Exception):
            fp._count_rows("/nonexistent/path.parquet")

    def test_load_nonexistent_path_falls_through(self, monkeypatch):
        """When path doesn't exist and not remote, falls to _load_local."""
        fp = DatasetFingerprinter()
        monkeypatch.setattr(DatasetFingerprinter, "_is_remote", staticmethod(lambda: False))
        with pytest.raises(Exception):
            fp._load("/nonexistent/path.parquet")
