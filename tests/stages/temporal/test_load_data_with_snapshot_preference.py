import shutil
import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from customer_retention.analysis.auto_explorer.findings import ExplorationFindings, TimeSeriesMetadata
from customer_retention.core.config.column_config import DatasetGranularity
from customer_retention.stages.temporal import UnifiedDataPreparer, load_data_with_snapshot_preference
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


def _make_findings(source_path, entity_col="user_id", target_col="churned",
                   snapshot_path=None):
    return ExplorationFindings(
        source_path=source_path,
        source_format="csv",
        time_series_metadata=TimeSeriesMetadata(
            granularity=DatasetGranularity.EVENT_LEVEL,
            entity_column=entity_col,
            time_column="event_date",
        ),
        target_column=target_col,
        snapshot_path=snapshot_path,
    )


class TestLoadDataWithSnapshotPreferenceRestoresColumns:
    def test_snapshot_path_restores_entity_column(self, temp_dir, event_df):
        snapshot_file = temp_dir / "snapshot.parquet"
        renamed = event_df.rename(columns={"user_id": "entity_id", "churned": "target"})
        renamed.to_parquet(snapshot_file, index=False)

        findings = _make_findings(
            source_path=str(temp_dir / "dummy.csv"),
            entity_col="user_id",
            target_col="churned",
            snapshot_path=str(snapshot_file),
        )
        df, source = load_data_with_snapshot_preference(findings)

        assert "user_id" in df.columns, "entity column should be restored from entity_id"
        assert "entity_id" not in df.columns

    def test_snapshot_path_restores_target_column(self, temp_dir, event_df):
        snapshot_file = temp_dir / "snapshot.parquet"
        renamed = event_df.rename(columns={"user_id": "entity_id", "churned": "target"})
        renamed.to_parquet(snapshot_file, index=False)

        findings = _make_findings(
            source_path=str(temp_dir / "dummy.csv"),
            entity_col="user_id",
            target_col="churned",
            snapshot_path=str(snapshot_file),
        )
        df, _ = load_data_with_snapshot_preference(findings)

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
        preparer = UnifiedDataPreparer(output_dir, config)
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
        snapshot_file = temp_dir / "snapshot.parquet"
        df_original = pd.DataFrame({"user_id": ["A", "B"], "value": [1, 2]})
        df_original.to_parquet(snapshot_file, index=False)

        findings = _make_findings(
            source_path=str(temp_dir / "dummy.csv"),
            entity_col="user_id",
            snapshot_path=str(snapshot_file),
        )
        df, _ = load_data_with_snapshot_preference(findings)

        assert "user_id" in df.columns

    def test_no_rename_when_entity_column_already_present(self, temp_dir):
        snapshot_file = temp_dir / "snapshot.parquet"
        df_both = pd.DataFrame({"user_id": ["A"], "entity_id": ["X"], "val": [1]})
        df_both.to_parquet(snapshot_file, index=False)

        findings = _make_findings(
            source_path=str(temp_dir / "dummy.csv"),
            entity_col="user_id",
            snapshot_path=str(snapshot_file),
        )
        df, _ = load_data_with_snapshot_preference(findings)

        assert "user_id" in df.columns
        assert "entity_id" in df.columns
