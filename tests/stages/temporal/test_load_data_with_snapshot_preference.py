import shutil
import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from customer_retention.analysis.auto_explorer.findings import ExplorationFindings, TimeSeriesMetadata
from customer_retention.core.config.column_config import DatasetGranularity
from customer_retention.stages.temporal import (
    SnapshotManager,
    UnifiedDataPreparer,
    compute_composite_dataset_name,
    load_data_with_snapshot_preference,
)
from customer_retention.stages.temporal.timestamp_manager import TimestampConfig, TimestampStrategy


@pytest.fixture
def temp_dir():
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def event_df():
    return pd.DataFrame({
        "user_id": ["A", "A", "B", "B", "C"],
        "event_date": pd.to_datetime(["2024-01-01", "2024-01-05", "2024-01-02", "2024-01-06", "2024-01-03"]),
        "churned": [0, 1, 0, 0, 1],
        "feature_1": [10, 20, 30, 40, 50],
    })


def _make_findings(source_path, entity_col="user_id", target_col="churned"):
    return ExplorationFindings(
        source_path=source_path,
        source_format="csv",
        time_series_metadata=TimeSeriesMetadata(
            granularity=DatasetGranularity.EVENT_LEVEL,
            entity_column=entity_col,
            time_column="event_date",
        ),
        target_column=target_col,
    )


class TestLoadDataWithSnapshotPreferenceRestoresColumns:
    def test_snapshot_restores_entity_column(self, temp_dir, event_df):
        dataset_name = "dummy"
        output_dir = temp_dir / "output"
        output_dir.mkdir()
        renamed = event_df.rename(columns={"user_id": "entity_id", "churned": "target"})
        renamed["feature_timestamp"] = renamed["event_date"]
        renamed["label_available_flag"] = True
        SnapshotManager(output_dir, dataset_name=dataset_name).create_snapshot(
            renamed, datetime(2024, 12, 31), "target")

        findings = _make_findings(source_path=str(temp_dir / f"{dataset_name}.csv"),
                                  entity_col="user_id", target_col="churned")
        df, source = load_data_with_snapshot_preference(findings, output_dir=str(output_dir))

        assert "snapshot" in source
        assert "user_id" in df.columns, "entity column should be restored from entity_id"
        assert "entity_id" not in df.columns

    def test_snapshot_restores_target_column(self, temp_dir, event_df):
        dataset_name = "dummy"
        output_dir = temp_dir / "output"
        output_dir.mkdir()
        renamed = event_df.rename(columns={"user_id": "entity_id", "churned": "target"})
        renamed["feature_timestamp"] = renamed["event_date"]
        renamed["label_available_flag"] = True
        SnapshotManager(output_dir, dataset_name=dataset_name).create_snapshot(
            renamed, datetime(2024, 12, 31), "target")

        findings = _make_findings(source_path=str(temp_dir / f"{dataset_name}.csv"),
                                  entity_col="user_id", target_col="churned")
        df, source = load_data_with_snapshot_preference(findings, output_dir=str(output_dir))

        assert "snapshot" in source
        assert "churned" in df.columns, "target column should be restored from target"
        assert "target" not in df.columns

    def test_snapshot_manager_restores_entity_column(self, temp_dir, event_df):
        config = TimestampConfig(
            strategy=TimestampStrategy.SYNTHETIC_FIXED,
            synthetic_base_date="2024-01-01",
            observation_window_days=90,
        )
        output_dir = temp_dir / "findings"
        output_dir.mkdir()
        preparer = UnifiedDataPreparer(output_dir, config, dataset_name="dummy")
        unified = preparer.prepare_from_raw(event_df.copy(), "churned", "user_id")
        preparer.create_training_snapshot(unified, datetime(2024, 6, 1))

        findings = _make_findings(source_path=str(temp_dir / "dummy.csv"), entity_col="user_id")
        df, source = load_data_with_snapshot_preference(findings, output_dir=str(output_dir))

        assert "snapshot" in source
        assert "user_id" in df.columns
        assert "entity_id" not in df.columns

    def test_source_file_columns_unchanged(self, temp_dir, event_df):
        csv_path = temp_dir / "data.csv"
        event_df.to_csv(csv_path, index=False)

        findings = _make_findings(source_path=str(csv_path), entity_col="user_id")
        df, source = load_data_with_snapshot_preference(findings)

        assert source == "source"
        assert "user_id" in df.columns

    def test_no_rename_when_entity_id_not_present(self, temp_dir):
        dataset_name = "dummy"
        output_dir = temp_dir / "output"
        output_dir.mkdir()
        df_original = pd.DataFrame({"user_id": ["A", "B"], "value": [1, 2],
                                    "feature_timestamp": pd.to_datetime(["2024-01-01", "2024-01-02"]),
                                    "label_available_flag": [True, True]})
        SnapshotManager(output_dir, dataset_name=dataset_name).create_snapshot(
            df_original, datetime(2024, 12, 31), "value")

        findings = _make_findings(source_path=str(temp_dir / f"{dataset_name}.csv"),
                                  entity_col="user_id")
        df, _ = load_data_with_snapshot_preference(findings, output_dir=str(output_dir))

        assert "user_id" in df.columns

    def test_no_rename_when_entity_column_already_present(self, temp_dir):
        dataset_name = "dummy"
        output_dir = temp_dir / "output"
        output_dir.mkdir()
        df_both = pd.DataFrame({"user_id": ["A"], "entity_id": ["X"], "val": [1],
                                "feature_timestamp": pd.to_datetime(["2024-01-01"]),
                                "label_available_flag": [True]})
        SnapshotManager(output_dir, dataset_name=dataset_name).create_snapshot(
            df_both, datetime(2024, 12, 31), "val")

        findings = _make_findings(source_path=str(temp_dir / f"{dataset_name}.csv"),
                                  entity_col="user_id")
        df, _ = load_data_with_snapshot_preference(findings, output_dir=str(output_dir))

        assert "user_id" in df.columns
        assert "entity_id" in df.columns


class TestLoadDataNamespaceAwareSnapshotDiscovery:
    def test_finds_snapshot_in_dataset_namespace(self, temp_dir, event_df):
        dataset_name = "my_data"
        csv_path = temp_dir / f"{dataset_name}.csv"
        event_df.to_csv(csv_path, index=False)

        config = TimestampConfig(
            strategy=TimestampStrategy.SYNTHETIC_FIXED,
            synthetic_base_date="2024-01-01",
            observation_window_days=90,
        )
        output_dir = temp_dir / "findings"
        output_dir.mkdir()
        preparer = UnifiedDataPreparer(output_dir, config, dataset_name=dataset_name)
        unified = preparer.prepare_from_raw(event_df.copy(), "churned", "user_id")
        preparer.create_training_snapshot(unified, datetime(2024, 6, 1))

        findings = _make_findings(source_path=str(csv_path), entity_col="user_id")
        df, source = load_data_with_snapshot_preference(findings, output_dir=str(output_dir))

        assert "snapshot" in source
        assert "user_id" in df.columns

    def test_ignores_snapshot_in_wrong_namespace(self, temp_dir, event_df):
        csv_path = temp_dir / "data.csv"
        event_df.to_csv(csv_path, index=False)

        output_dir = temp_dir / "findings"
        output_dir.mkdir()
        mgr = SnapshotManager(output_dir)
        snapshot_df = event_df.copy()
        snapshot_df["feature_timestamp"] = snapshot_df["event_date"]
        snapshot_df["label_available_flag"] = True
        mgr.create_snapshot(snapshot_df, datetime(2024, 12, 31), "churned")

        findings = _make_findings(source_path=str(csv_path), entity_col="user_id")
        df, source = load_data_with_snapshot_preference(findings, output_dir=str(output_dir))

        assert source == "source"

    def test_source_fallback_when_no_snapshots(self, temp_dir, event_df):
        csv_path = temp_dir / "data.csv"
        event_df.to_csv(csv_path, index=False)

        findings = _make_findings(source_path=str(csv_path), entity_col="user_id")
        df, source = load_data_with_snapshot_preference(findings, output_dir=str(temp_dir))

        assert source == "source"
        assert "user_id" in df.columns


class TestLoadDataRequiresSourcePath:
    def test_raises_when_source_path_is_none(self):
        findings = ExplorationFindings(
            source_path=None, source_format="csv",
            time_series_metadata=TimeSeriesMetadata(
                granularity=DatasetGranularity.EVENT_LEVEL,
                entity_column="user_id", time_column="event_date"))
        with pytest.raises(ValueError, match="source_path is required"):
            load_data_with_snapshot_preference(findings)

    def test_raises_when_source_path_is_empty(self):
        findings = ExplorationFindings(
            source_path="", source_format="csv",
            time_series_metadata=TimeSeriesMetadata(
                granularity=DatasetGranularity.EVENT_LEVEL,
                entity_column="user_id", time_column="event_date"))
        with pytest.raises(ValueError, match="source_path is required"):
            load_data_with_snapshot_preference(findings)


class TestLoadDataSourceFallbackReturnsRawData:
    def test_source_fallback_preserves_original_columns(self, temp_dir):
        csv_path = temp_dir / "data.csv"
        pd.DataFrame({
            "user_id": ["A", "B"],
            "sent_date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "value": [1, 2],
        }).to_csv(csv_path, index=False)

        findings = ExplorationFindings(
            source_path=str(csv_path),
            source_format="csv",
            time_series_metadata=TimeSeriesMetadata(
                granularity=DatasetGranularity.EVENT_LEVEL,
                entity_column="user_id",
                time_column="feature_timestamp",
            ),
            datetime_columns=["sent_date"],
        )
        df, source = load_data_with_snapshot_preference(findings)

        assert source == "source"
        assert "sent_date" in df.columns
        assert "feature_timestamp" not in df.columns


class TestLoadDataWithCompositeDatasetName:
    def test_finds_snapshot_with_explicit_composite_name(self, temp_dir, event_df):
        output_dir = temp_dir / "output"
        output_dir.mkdir()
        composite_name = compute_composite_dataset_name(["customers", "transactions"])
        snapshot_df = event_df.rename(columns={"user_id": "entity_id", "churned": "target"})
        snapshot_df["feature_timestamp"] = snapshot_df["event_date"]
        snapshot_df["label_available_flag"] = True
        SnapshotManager(output_dir, dataset_name=composite_name).create_snapshot(
            snapshot_df, datetime(2024, 12, 31), "target")

        findings = _make_findings(source_path=str(temp_dir / "customers.csv"))
        df, source = load_data_with_snapshot_preference(
            findings, output_dir=str(output_dir), dataset_name=composite_name)

        assert "snapshot" in source

    def test_ignores_single_dataset_snapshot_when_composite_name_given(self, temp_dir, event_df):
        output_dir = temp_dir / "output"
        output_dir.mkdir()
        snapshot_df = event_df.copy()
        snapshot_df["feature_timestamp"] = snapshot_df["event_date"]
        snapshot_df["label_available_flag"] = True
        SnapshotManager(output_dir, dataset_name="customers").create_snapshot(
            snapshot_df, datetime(2024, 12, 31), "churned")

        csv_path = temp_dir / "customers.csv"
        event_df.to_csv(csv_path, index=False)
        composite_name = compute_composite_dataset_name(["customers", "transactions"])
        findings = _make_findings(source_path=str(csv_path))
        df, source = load_data_with_snapshot_preference(
            findings, output_dir=str(output_dir), dataset_name=composite_name)

        assert source == "source"

    def test_dataset_name_none_preserves_stem_behavior(self, temp_dir, event_df):
        dataset_name = "mydata"
        output_dir = temp_dir / "output"
        output_dir.mkdir()
        snapshot_df = event_df.rename(columns={"user_id": "entity_id", "churned": "target"})
        snapshot_df["feature_timestamp"] = snapshot_df["event_date"]
        snapshot_df["label_available_flag"] = True
        SnapshotManager(output_dir, dataset_name=dataset_name).create_snapshot(
            snapshot_df, datetime(2024, 12, 31), "target")

        findings = _make_findings(source_path=str(temp_dir / f"{dataset_name}.csv"))
        df, source = load_data_with_snapshot_preference(
            findings, output_dir=str(output_dir), dataset_name=None)

        assert "snapshot" in source
