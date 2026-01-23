import shutil
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from customer_retention.stages.temporal import (
    CutoffAnalyzer,
    ScenarioDetector,
    UnifiedDataPreparer,
)
from customer_retention.stages.temporal.point_in_time_registry import PointInTimeRegistry
from customer_retention.stages.temporal.synthetic_coordinator import (
    SyntheticCoordinationParams,
    SyntheticTimestampCoordinator,
)
from customer_retention.stages.temporal.timestamp_manager import (
    TimestampConfig,
    TimestampManager,
    TimestampStrategy,
)


@pytest.fixture
def temp_dir():
    path = Path(tempfile.mkdtemp())
    yield path
    shutil.rmtree(path)


@pytest.fixture
def no_datetime_df():
    np.random.seed(42)
    return pd.DataFrame({
        "customer_id": [f"C{i}" for i in range(200)],
        "monthly_spend": np.random.uniform(10, 500, 200),
        "loyalty_score": np.random.randint(1, 100, 200),
        "support_tickets": np.random.randint(0, 10, 200),
        "churned": np.random.randint(0, 2, 200),
    })


@pytest.fixture
def mostly_null_datetime_df():
    np.random.seed(42)
    n = 200
    dates = pd.Series([None] * n, dtype="object")
    dates.iloc[:10] = pd.date_range("2022-01-01", periods=10, freq="M").astype(str)
    return pd.DataFrame({
        "customer_id": [f"C{i}" for i in range(n)],
        "signup_date": dates,
        "monthly_spend": np.random.uniform(10, 500, n),
        "churned": np.random.randint(0, 2, n),
    })


class TestSyntheticPipelineDetection:
    def test_no_datetime_columns_returns_synthetic_scenario(self, no_datetime_df):
        detector = ScenarioDetector()
        scenario, _, _ = detector.detect(no_datetime_df, "churned")
        assert scenario == "synthetic"

    def test_mostly_null_datetime_returns_synthetic_scenario(self, mostly_null_datetime_df):
        detector = ScenarioDetector()
        scenario, config, _ = detector.detect(mostly_null_datetime_df, "churned")
        assert scenario in ("synthetic", "partial")
        if scenario == "synthetic":
            assert config.strategy in (TimestampStrategy.SYNTHETIC_INDEX, TimestampStrategy.SYNTHETIC_FIXED)

    def test_synthetic_scenario_uses_index_strategy(self, no_datetime_df):
        detector = ScenarioDetector()
        _, config, _ = detector.detect(no_datetime_df, "churned")
        assert config.strategy == TimestampStrategy.SYNTHETIC_INDEX


class TestSyntheticPipelinePrepare:
    def test_prepare_adds_timestamp_columns(self, no_datetime_df, temp_dir):
        config = TimestampConfig(strategy=TimestampStrategy.SYNTHETIC_INDEX, observation_window_days=90)
        preparer = UnifiedDataPreparer(temp_dir, config)
        result = preparer.prepare_from_raw(no_datetime_df, "churned", "customer_id")
        assert "feature_timestamp" in result.columns
        assert "label_timestamp" in result.columns
        assert "label_available_flag" in result.columns

    def test_timestamps_are_sequentially_ordered(self, no_datetime_df, temp_dir):
        config = TimestampConfig(strategy=TimestampStrategy.SYNTHETIC_INDEX, observation_window_days=90)
        preparer = UnifiedDataPreparer(temp_dir, config)
        result = preparer.prepare_from_raw(no_datetime_df, "churned", "customer_id")
        ts = result["feature_timestamp"].reset_index(drop=True)
        assert (ts.iloc[1:].values >= ts.iloc[:-1].values).all()

    def test_point_in_time_validation_passes(self, no_datetime_df, temp_dir):
        config = TimestampConfig(strategy=TimestampStrategy.SYNTHETIC_INDEX, observation_window_days=90)
        preparer = UnifiedDataPreparer(temp_dir, config)
        result = preparer.prepare_from_raw(no_datetime_df, "churned", "customer_id")
        manager = TimestampManager(config)
        assert manager.validate_point_in_time(result) is True

    def test_original_features_preserved(self, no_datetime_df, temp_dir):
        config = TimestampConfig(strategy=TimestampStrategy.SYNTHETIC_INDEX, observation_window_days=90)
        preparer = UnifiedDataPreparer(temp_dir, config)
        result = preparer.prepare_from_raw(no_datetime_df, "churned", "customer_id")
        assert "monthly_spend" in result.columns
        assert "support_tickets" in result.columns


class TestSyntheticPipelineAnalyze:
    def test_analyze_produces_multiple_bins(self, no_datetime_df, temp_dir):
        config = TimestampConfig(strategy=TimestampStrategy.SYNTHETIC_INDEX, observation_window_days=90)
        preparer = UnifiedDataPreparer(temp_dir, config)
        prepared = preparer.prepare_from_raw(no_datetime_df, "churned", "customer_id")
        analyzer = CutoffAnalyzer()
        analysis = analyzer.analyze(prepared, timestamp_column="feature_timestamp")
        assert len(analysis.bins) > 1

    def test_date_range_spans_dataframe_length(self, no_datetime_df, temp_dir):
        config = TimestampConfig(
            strategy=TimestampStrategy.SYNTHETIC_INDEX, observation_window_days=90,
            synthetic_base_date="2024-01-01",
        )
        preparer = UnifiedDataPreparer(temp_dir, config)
        prepared = preparer.prepare_from_raw(no_datetime_df, "churned", "customer_id")
        analyzer = CutoffAnalyzer()
        analysis = analyzer.analyze(prepared, timestamp_column="feature_timestamp")
        base = pd.to_datetime("2024-01-01")
        expected_max = base + timedelta(days=len(no_datetime_df) - 1)
        assert analysis.date_range[0] >= base
        assert analysis.date_range[1] <= expected_max

    def test_coverage_is_full(self, no_datetime_df, temp_dir):
        config = TimestampConfig(strategy=TimestampStrategy.SYNTHETIC_INDEX, observation_window_days=90)
        preparer = UnifiedDataPreparer(temp_dir, config)
        prepared = preparer.prepare_from_raw(no_datetime_df, "churned", "customer_id")
        analyzer = CutoffAnalyzer()
        analysis = analyzer.analyze(prepared, timestamp_column="feature_timestamp")
        assert analysis.covered_rows == len(no_datetime_df)

    def test_suggest_cutoff_returns_date_within_range(self, no_datetime_df, temp_dir):
        config = TimestampConfig(strategy=TimestampStrategy.SYNTHETIC_INDEX, observation_window_days=90)
        preparer = UnifiedDataPreparer(temp_dir, config)
        prepared = preparer.prepare_from_raw(no_datetime_df, "churned", "customer_id")
        analyzer = CutoffAnalyzer()
        analysis = analyzer.analyze(prepared, timestamp_column="feature_timestamp")
        cutoff = analysis.suggest_cutoff(train_ratio=0.9)
        assert analysis.date_range[0] <= cutoff <= analysis.date_range[1]


class TestSyntheticPipelineSplit:
    @pytest.fixture
    def prepared_with_analysis(self, no_datetime_df, temp_dir):
        config = TimestampConfig(strategy=TimestampStrategy.SYNTHETIC_INDEX, observation_window_days=90)
        preparer = UnifiedDataPreparer(temp_dir, config)
        prepared = preparer.prepare_from_raw(no_datetime_df, "churned", "customer_id")
        analyzer = CutoffAnalyzer()
        analysis = analyzer.analyze(prepared, timestamp_column="feature_timestamp")
        return prepared, analysis

    def test_split_produces_no_unresolvable(self, prepared_with_analysis):
        prepared, analysis = prepared_with_analysis
        cutoff = analysis.suggest_cutoff(train_ratio=0.9)
        split = analysis.split_at_cutoff(cutoff)
        assert split.unresolvable_count == 0

    def test_train_count_approximately_matches_ratio(self, prepared_with_analysis):
        prepared, analysis = prepared_with_analysis
        cutoff = analysis.suggest_cutoff(train_ratio=0.9)
        split = analysis.split_at_cutoff(cutoff)
        ratio = split.train_count / (split.train_count + split.score_count)
        assert 0.80 <= ratio <= 0.95

    def test_train_rows_have_ts_le_cutoff(self, prepared_with_analysis):
        prepared, analysis = prepared_with_analysis
        cutoff = analysis.suggest_cutoff(train_ratio=0.9)
        split = analysis.split_at_cutoff(cutoff)
        assert (split.train_df["feature_timestamp"] <= cutoff).all()

    def test_score_rows_have_ts_gt_cutoff(self, prepared_with_analysis):
        prepared, analysis = prepared_with_analysis
        cutoff = analysis.suggest_cutoff(train_ratio=0.9)
        split = analysis.split_at_cutoff(cutoff)
        assert (split.score_df["feature_timestamp"] > cutoff).all()


class TestSyntheticPipelineSnapshot:
    def test_snapshot_row_count_matches_split_train_count(self, no_datetime_df, temp_dir):
        config = TimestampConfig(strategy=TimestampStrategy.SYNTHETIC_INDEX, observation_window_days=90)
        preparer = UnifiedDataPreparer(temp_dir, config)
        prepared = preparer.prepare_from_raw(no_datetime_df, "churned", "customer_id")
        analyzer = CutoffAnalyzer()
        analysis = analyzer.analyze(prepared, timestamp_column="feature_timestamp")
        cutoff = analysis.suggest_cutoff(train_ratio=0.9)
        split = analysis.split_at_cutoff(cutoff)
        _, meta = preparer.create_training_snapshot(prepared, cutoff)
        assert meta["row_count"] == split.train_count

    def test_snapshot_metadata_contains_cutoff(self, no_datetime_df, temp_dir):
        config = TimestampConfig(strategy=TimestampStrategy.SYNTHETIC_INDEX, observation_window_days=90)
        preparer = UnifiedDataPreparer(temp_dir, config)
        prepared = preparer.prepare_from_raw(no_datetime_df, "churned", "customer_id")
        analyzer = CutoffAnalyzer()
        analysis = analyzer.analyze(prepared, timestamp_column="feature_timestamp")
        cutoff = analysis.suggest_cutoff(train_ratio=0.9)
        _, meta = preparer.create_training_snapshot(prepared, cutoff)
        assert "cutoff_date" in meta

    def test_label_available_flag_all_true_for_synthetic(self, no_datetime_df, temp_dir):
        config = TimestampConfig(strategy=TimestampStrategy.SYNTHETIC_INDEX, observation_window_days=90)
        preparer = UnifiedDataPreparer(temp_dir, config)
        prepared = preparer.prepare_from_raw(no_datetime_df, "churned", "customer_id")
        assert prepared["label_available_flag"].all()


class TestSyntheticPipelineFullFlow:
    def test_full_pipeline_end_to_end(self, no_datetime_df, temp_dir):
        detector = ScenarioDetector()
        scenario, config, _ = detector.detect(no_datetime_df, "churned")
        assert scenario == "synthetic"

        preparer = UnifiedDataPreparer(temp_dir, config)
        prepared = preparer.prepare_from_raw(no_datetime_df, "churned", "customer_id")

        analyzer = CutoffAnalyzer()
        analysis = analyzer.analyze(prepared, timestamp_column="feature_timestamp")
        cutoff = analysis.suggest_cutoff(train_ratio=0.9)
        split = analysis.split_at_cutoff(cutoff)

        snapshot_df, meta = preparer.create_training_snapshot(prepared, cutoff)
        assert meta["row_count"] == split.train_count
        assert 0.80 <= meta["row_count"] / len(prepared) <= 0.95

    def test_shuffled_dataframe_produces_same_timestamp_range(self, no_datetime_df, temp_dir):
        config = TimestampConfig(
            strategy=TimestampStrategy.SYNTHETIC_INDEX, observation_window_days=90,
            synthetic_base_date="2024-01-01",
        )
        shuffled = no_datetime_df.sample(frac=1, random_state=99).reset_index(drop=True)

        preparer_orig = UnifiedDataPreparer(temp_dir / "orig", config)
        preparer_shuf = UnifiedDataPreparer(temp_dir / "shuf", config)

        prepared_orig = preparer_orig.prepare_from_raw(no_datetime_df, "churned", "customer_id")
        prepared_shuf = preparer_shuf.prepare_from_raw(shuffled, "churned", "customer_id")

        assert prepared_orig["feature_timestamp"].min() == prepared_shuf["feature_timestamp"].min()
        assert prepared_orig["feature_timestamp"].max() == prepared_shuf["feature_timestamp"].max()

    def test_non_sequential_index_works(self, temp_dir):
        df = pd.DataFrame(
            {"entity_id": ["A", "B", "C", "D", "E"], "val": [1, 2, 3, 4, 5], "target": [0, 1, 0, 1, 0]},
            index=[100, 200, 300, 400, 500],
        )
        config = TimestampConfig(
            strategy=TimestampStrategy.SYNTHETIC_INDEX, observation_window_days=90,
            synthetic_base_date="2024-01-01",
        )
        preparer = UnifiedDataPreparer(temp_dir, config)
        prepared = preparer.prepare_from_raw(df, "target", "entity_id")

        base = pd.to_datetime("2024-01-01")
        assert prepared["feature_timestamp"].min() == base
        assert prepared["feature_timestamp"].max() == base + timedelta(days=4)


class TestSyntheticPipelineMultiDataset:
    def test_two_datasets_with_coordinated_configs_share_base_date(self, temp_dir):
        coordinator = SyntheticTimestampCoordinator()
        config_a = coordinator.create_config("customers")
        config_b = coordinator.create_config("transactions")
        assert config_a.synthetic_base_date == config_b.synthetic_base_date

    def test_coordinated_datasets_produce_compatible_cutoff_ranges(self, temp_dir):
        coordinator = SyntheticTimestampCoordinator(
            SyntheticCoordinationParams(base_date="2023-01-01", observation_window_days=90)
        )
        config_a = coordinator.create_config("customers")
        config_b = coordinator.create_config("transactions")

        df_a = pd.DataFrame({"id": range(100), "val": range(100), "target": [0] * 100})
        df_b = pd.DataFrame({"id": range(50), "val": range(50), "target": [1] * 50})

        manager_a = TimestampManager(config_a)
        manager_b = TimestampManager(config_b)
        prepared_a = manager_a.ensure_timestamps(df_a)
        prepared_b = manager_b.ensure_timestamps(df_b)

        assert prepared_a["feature_timestamp"].min() == prepared_b["feature_timestamp"].min()

    def test_coordinator_plus_registry_consistency_check_passes(self, temp_dir):
        registry = PointInTimeRegistry(temp_dir)
        coordinator = SyntheticTimestampCoordinator.from_registry(registry)
        coordinator.create_config("a")
        coordinator.create_config("b")
        valid, _ = coordinator.validate_compatibility()
        assert valid is True

    def test_uncoordinated_configs_fail_validate_compatibility(self):
        coordinator = SyntheticTimestampCoordinator()
        config_a = coordinator.create_config("a")
        coordinator.create_config("b")
        config_a.synthetic_base_date = "2020-01-01"
        valid, _ = coordinator.validate_compatibility()
        assert valid is False


class TestSyntheticPipelineEdgeCases:
    def test_single_row_dataframe(self, temp_dir):
        df = pd.DataFrame({"id": ["X"], "val": [1], "target": [0]})
        config = TimestampConfig(strategy=TimestampStrategy.SYNTHETIC_INDEX, observation_window_days=90)
        preparer = UnifiedDataPreparer(temp_dir, config)
        prepared = preparer.prepare_from_raw(df, "target", "id")
        assert len(prepared) == 1
        assert prepared["feature_timestamp"].notna().all()

    def test_empty_dataframe_returns_empty_snapshot(self, temp_dir):
        df = pd.DataFrame({"id": pd.Series(dtype="str"), "val": pd.Series(dtype="float"), "target": pd.Series(dtype="int")})
        config = TimestampConfig(strategy=TimestampStrategy.SYNTHETIC_INDEX, observation_window_days=90)
        preparer = UnifiedDataPreparer(temp_dir, config)
        prepared = preparer.prepare_from_raw(df, "target", "id")
        _, meta = preparer.create_training_snapshot(prepared, datetime(2024, 7, 1))
        assert meta["row_count"] == 0

    def test_large_dataframe_has_correct_date_range(self, temp_dir):
        n = 1000
        df = pd.DataFrame({
            "id": [f"E{i}" for i in range(n)], "val": range(n),
            "target": np.random.RandomState(42).randint(0, 2, n),
        })
        config = TimestampConfig(
            strategy=TimestampStrategy.SYNTHETIC_INDEX, observation_window_days=90,
            synthetic_base_date="2024-01-01",
        )
        preparer = UnifiedDataPreparer(temp_dir, config)
        prepared = preparer.prepare_from_raw(df, "target", "id")
        base = pd.to_datetime("2024-01-01")
        assert prepared["feature_timestamp"].min() == base
        assert prepared["feature_timestamp"].max() == base + timedelta(days=n - 1)

    def test_all_null_datetime_columns_detected_as_synthetic(self, temp_dir):
        df = pd.DataFrame({
            "id": ["A", "B", "C"],
            "some_date": pd.Series([None, None, None], dtype="object"),
            "val": [1, 2, 3],
            "target": [0, 1, 0],
        })
        detector = ScenarioDetector()
        scenario, _, _ = detector.detect(df, "target")
        assert scenario == "synthetic"
