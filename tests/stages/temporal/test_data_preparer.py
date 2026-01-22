import pytest
import pandas as pd
from datetime import datetime
from pathlib import Path
import tempfile
import shutil

from customer_retention.stages.temporal.data_preparer import UnifiedDataPreparer, PreparedData
from customer_retention.stages.temporal.timestamp_manager import TimestampConfig, TimestampStrategy


class TestUnifiedDataPreparer:
    @pytest.fixture
    def temp_dir(self):
        temp_path = Path(tempfile.mkdtemp())
        yield temp_path
        shutil.rmtree(temp_path)

    @pytest.fixture
    def synthetic_config(self):
        return TimestampConfig(
            strategy=TimestampStrategy.SYNTHETIC_FIXED,
            synthetic_base_date="2024-01-01",
            observation_window_days=90
        )

    @pytest.fixture
    def production_config(self):
        return TimestampConfig(
            strategy=TimestampStrategy.PRODUCTION,
            feature_timestamp_column="snapshot_date",
            label_timestamp_column="outcome_date"
        )

    @pytest.fixture
    def raw_df(self):
        return pd.DataFrame({
            "customer_id": ["A", "B", "C", "D", "E"],
            "churned": [1, 0, 1, 0, 1],
            "feature_1": [100, 200, 300, 400, 500],
            "feature_2": [10, 20, 30, 40, 50]
        })

    @pytest.fixture
    def production_df(self):
        return pd.DataFrame({
            "customer_id": ["A", "B", "C"],
            "snapshot_date": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01"]),
            "outcome_date": pd.to_datetime(["2024-04-01", "2024-05-01", "2024-06-01"]),
            "churned": [1, 0, 1],
            "feature_1": [100, 200, 300]
        })


class TestPrepareFromRaw(TestUnifiedDataPreparer):
    def test_prepare_adds_timestamps(self, temp_dir, synthetic_config, raw_df):
        preparer = UnifiedDataPreparer(temp_dir, synthetic_config)
        result = preparer.prepare_from_raw(raw_df, "churned", "customer_id")

        assert "feature_timestamp" in result.columns
        assert "label_timestamp" in result.columns
        assert "label_available_flag" in result.columns

    def test_prepare_renames_columns(self, temp_dir, synthetic_config, raw_df):
        preparer = UnifiedDataPreparer(temp_dir, synthetic_config)
        result = preparer.prepare_from_raw(raw_df, "churned", "customer_id")

        assert "target" in result.columns
        assert "entity_id" in result.columns
        assert "churned" not in result.columns
        assert "customer_id" not in result.columns

    def test_prepare_saves_unified_dataset(self, temp_dir, synthetic_config, raw_df):
        preparer = UnifiedDataPreparer(temp_dir, synthetic_config)
        preparer.prepare_from_raw(raw_df, "churned", "customer_id")

        unified_path = temp_dir / "unified" / "unified_dataset.parquet"
        assert unified_path.exists()

    def test_prepare_production_validates_timestamps(self, temp_dir, production_config, production_df):
        preparer = UnifiedDataPreparer(temp_dir, production_config)
        result = preparer.prepare_from_raw(production_df, "churned", "customer_id")

        assert result["feature_timestamp"].iloc[0] == pd.to_datetime("2024-01-01")


class TestCreateTrainingSnapshot(TestUnifiedDataPreparer):
    def test_create_snapshot_returns_filtered_data(self, temp_dir, synthetic_config, raw_df):
        preparer = UnifiedDataPreparer(temp_dir, synthetic_config)
        unified = preparer.prepare_from_raw(raw_df, "churned", "customer_id")

        cutoff = datetime(2024, 6, 1)
        snapshot_df, metadata = preparer.create_training_snapshot(unified, cutoff)

        assert len(snapshot_df) > 0
        assert metadata["snapshot_id"] == "training_v1"

    def test_create_snapshot_increments_version(self, temp_dir, synthetic_config, raw_df):
        preparer = UnifiedDataPreparer(temp_dir, synthetic_config)
        unified = preparer.prepare_from_raw(raw_df, "churned", "customer_id")
        cutoff = datetime(2024, 6, 1)

        _, meta1 = preparer.create_training_snapshot(unified, cutoff)
        _, meta2 = preparer.create_training_snapshot(unified, cutoff)

        assert meta1["version"] == 1
        assert meta2["version"] == 2


class TestLoadForEda(TestUnifiedDataPreparer):
    def test_load_for_eda_returns_snapshot(self, temp_dir, synthetic_config, raw_df):
        preparer = UnifiedDataPreparer(temp_dir, synthetic_config)
        unified = preparer.prepare_from_raw(raw_df, "churned", "customer_id")
        cutoff = datetime(2024, 6, 1)
        _, metadata = preparer.create_training_snapshot(unified, cutoff)

        loaded = preparer.load_for_eda(metadata["snapshot_id"])

        assert len(loaded) > 0


class TestLoadForInference(TestUnifiedDataPreparer):
    def test_load_for_inference_sets_label_unavailable(self, temp_dir, synthetic_config, raw_df):
        preparer = UnifiedDataPreparer(temp_dir, synthetic_config)
        result = preparer.load_for_inference(raw_df)

        assert (result["label_available_flag"] == False).all()


class TestPrepareWithValidation(TestUnifiedDataPreparer):
    def test_prepare_with_validation_returns_prepared_data(self, temp_dir, synthetic_config, raw_df):
        preparer = UnifiedDataPreparer(temp_dir, synthetic_config)
        cutoff = datetime(2024, 6, 1)

        prepared = preparer.prepare_with_validation(raw_df, "churned", "customer_id", cutoff)

        assert isinstance(prepared, PreparedData)
        assert prepared.unified_df is not None
        assert prepared.snapshot_metadata is not None
        assert prepared.timestamp_strategy == "synthetic_fixed"
        assert prepared.validation_report is not None


class TestListAndGetSnapshots(TestUnifiedDataPreparer):
    def test_list_available_snapshots(self, temp_dir, synthetic_config, raw_df):
        preparer = UnifiedDataPreparer(temp_dir, synthetic_config)
        unified = preparer.prepare_from_raw(raw_df, "churned", "customer_id")
        cutoff = datetime(2024, 6, 1)

        preparer.create_training_snapshot(unified, cutoff)
        preparer.create_training_snapshot(unified, cutoff, snapshot_name="validation")

        snapshots = preparer.list_available_snapshots()

        assert len(snapshots) == 2
        assert "training_v1" in snapshots
        assert "validation_v1" in snapshots

    def test_get_snapshot_summary(self, temp_dir, synthetic_config, raw_df):
        preparer = UnifiedDataPreparer(temp_dir, synthetic_config)
        unified = preparer.prepare_from_raw(raw_df, "churned", "customer_id")
        cutoff = datetime(2024, 6, 1)
        _, metadata = preparer.create_training_snapshot(unified, cutoff)

        summary = preparer.get_snapshot_summary(metadata["snapshot_id"])

        assert "snapshot_id" in summary
        assert "row_count" in summary
        assert "feature_count" in summary
        assert "data_hash" in summary
