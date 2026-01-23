"""Integration tests verifying consistency between notebook sections.

Simulates the 01_data_discovery.ipynb flow end-to-end and asserts that:
- Cutoff analysis expected split matches actual snapshot row count
- Snapshot is not suspiciously small relative to the dataset
- The timestamp source used for analysis is the same used for splitting
"""

import shutil
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from customer_retention.stages.profiling import TypeDetector
from customer_retention.stages.temporal import (
    CutoffAnalyzer, DatetimeOrderAnalyzer, ScenarioDetector, UnifiedDataPreparer,
)


FIXTURES_DIR = Path(__file__).parent.parent.parent / "fixtures"


class TestNotebookFlowConsistency:
    """Simulates notebook flow and verifies section outputs are consistent."""

    @pytest.fixture
    def output_dir(self):
        path = Path(tempfile.mkdtemp())
        yield path
        shutil.rmtree(path)

    @pytest.fixture
    def customer_emails_flow(self, output_dir):
        raw_df = pd.read_csv(FIXTURES_DIR / "customer_emails.csv")
        target = "unsubscribed"

        type_detector = TypeDetector()
        entity_col = type_detector.detect_granularity(raw_df).entity_column

        detector = ScenarioDetector(label_window_days=180)
        scenario, ts_config, discovery_result = detector.detect(raw_df, target)

        datetime_order_analyzer = DatetimeOrderAnalyzer()
        last_action_series = datetime_order_analyzer.derive_last_action_date(raw_df)

        cutoff_analyzer = CutoffAnalyzer()
        analysis = cutoff_analyzer.analyze(raw_df, timestamp_series=last_action_series, n_bins=50)
        cutoff = analysis.suggest_cutoff(train_ratio=0.9)
        expected_split = analysis.get_split_at_date(cutoff)

        preparer = UnifiedDataPreparer(output_dir, ts_config)
        prepared_df = preparer.prepare_from_raw(raw_df, target, entity_col or "entity_id")
        # Use the SAME series from raw data for snapshot (not re-derived from prepared)
        snapshot_df, meta = preparer.create_training_snapshot(
            prepared_df, cutoff, timestamp_series=last_action_series
        )

        return {
            "raw_df": raw_df,
            "prepared_df": prepared_df,
            "snapshot_df": snapshot_df,
            "meta": meta,
            "analysis": analysis,
            "cutoff": cutoff,
            "expected_split": expected_split,
            "last_action_series": last_action_series,
        }

    def test_snapshot_rows_match_analysis_expected_split(self, customer_emails_flow):
        meta = customer_emails_flow["meta"]
        expected = customer_emails_flow["expected_split"]

        actual_ratio = meta["row_count"] / len(customer_emails_flow["prepared_df"]) * 100
        expected_ratio = expected["train_pct"]

        assert abs(actual_ratio - expected_ratio) < 5, (
            f"Snapshot ratio {actual_ratio:.1f}% differs from analysis expected "
            f"{expected_ratio:.1f}% by more than 5 percentage points"
        )

    def test_snapshot_not_suspiciously_small(self, customer_emails_flow):
        meta = customer_emails_flow["meta"]
        raw_rows = len(customer_emails_flow["raw_df"])

        ratio = meta["row_count"] / raw_rows
        assert ratio > 0.5, (
            f"Snapshot has {meta['row_count']} rows = {ratio:.1%} of {raw_rows} raw rows. "
            f"Expected at least 50% for a 90/10 split."
        )

    def test_snapshot_approximately_90_percent_of_covered(self, customer_emails_flow):
        meta = customer_emails_flow["meta"]
        analysis = customer_emails_flow["analysis"]

        ratio = meta["row_count"] / analysis.covered_rows
        assert 0.85 <= ratio <= 0.95, (
            f"Snapshot = {ratio:.1%} of covered rows, expected ~90%"
        )

    def test_label_available_flag_covers_most_rows(self, customer_emails_flow):
        prepared_df = customer_emails_flow["prepared_df"]

        available_ratio = prepared_df["label_available_flag"].mean()
        assert available_ratio > 0.9, (
            f"Only {available_ratio:.1%} of rows have label_available_flag=True. "
            f"For historical datasets this should be nearly all rows."
        )

    def test_cutoff_analysis_and_split_result_agree(self, customer_emails_flow):
        analysis = customer_emails_flow["analysis"]
        cutoff = customer_emails_flow["cutoff"]

        split = analysis.split_at_cutoff(cutoff)
        expected = customer_emails_flow["expected_split"]

        assert abs(split.train_count - expected["train_count"]) <= 1


class TestNotebookFlowWithSyntheticData:
    """Tests with controlled synthetic data to verify edge cases."""

    @pytest.fixture
    def output_dir(self):
        path = Path(tempfile.mkdtemp())
        yield path
        shutil.rmtree(path)

    def test_sparse_event_column_does_not_collapse_snapshot(self, output_dir):
        np.random.seed(42)
        n = 5000
        feature_dates = pd.date_range("2020-01-01", "2023-06-30", periods=n)
        # Only 5% have an event (like unsubscribe) — rest are NaT
        event_mask = np.random.random(n) < 0.05
        event_dates = pd.Series([None] * n, dtype="object")
        event_dates[event_mask] = (
            feature_dates[event_mask] + pd.Timedelta(days=60)
        ).astype(str)

        raw_df = pd.DataFrame({
            "customer_id": [f"C{i}" for i in range(n)],
            "activity_date": feature_dates.astype(str),
            "unsubscribe_date": event_dates,
            "target_col": np.random.randint(0, 2, n),
        })

        detector = ScenarioDetector(label_window_days=90)
        scenario, ts_config, _ = detector.detect(raw_df, "target_col")

        analyzer = DatetimeOrderAnalyzer()
        last_action = analyzer.derive_last_action_date(raw_df)

        cutoff_analyzer = CutoffAnalyzer()
        analysis = cutoff_analyzer.analyze(raw_df, timestamp_series=last_action, n_bins=50)
        cutoff = analysis.suggest_cutoff(train_ratio=0.9)

        preparer = UnifiedDataPreparer(output_dir, ts_config)
        prepared = preparer.prepare_from_raw(raw_df, "target_col", "customer_id")
        # Use same series from raw data (not re-derived from prepared)
        _, meta = preparer.create_training_snapshot(
            prepared, cutoff, timestamp_series=last_action
        )

        ratio = meta["row_count"] / n
        assert ratio > 0.7, (
            f"Sparse event column caused snapshot collapse: "
            f"{meta['row_count']}/{n} = {ratio:.1%}"
        )

    def test_full_coverage_timestamps_exact_match(self, output_dir):
        n = 1000
        ts = pd.date_range("2021-01-01", periods=n, freq="D")
        raw_df = pd.DataFrame({
            "customer_id": [f"C{i}" for i in range(n)],
            "observation_date": ts.astype(str),
            "target_col": np.random.RandomState(42).randint(0, 2, n),
        })

        detector = ScenarioDetector(label_window_days=90)
        _, ts_config, _ = detector.detect(raw_df, "target_col")

        analyzer = DatetimeOrderAnalyzer()
        last_action = analyzer.derive_last_action_date(raw_df)

        cutoff_analyzer = CutoffAnalyzer()
        analysis = cutoff_analyzer.analyze(raw_df, timestamp_series=last_action, n_bins=50)
        cutoff = analysis.suggest_cutoff(train_ratio=0.9)
        expected = analysis.get_split_at_date(cutoff)

        preparer = UnifiedDataPreparer(output_dir, ts_config)
        prepared = preparer.prepare_from_raw(raw_df, "target_col", "customer_id")
        # Use same series from raw data (not re-derived from prepared)
        _, meta = preparer.create_training_snapshot(
            prepared, cutoff, timestamp_series=last_action
        )

        actual_pct = meta["row_count"] / n * 100
        assert abs(actual_pct - expected["train_pct"]) < 5

    def test_consistency_between_split_result_and_snapshot(self, output_dir):
        n = 2000
        ts = pd.date_range("2020-06-01", periods=n, freq="D")
        raw_df = pd.DataFrame({
            "entity_id": [f"E{i}" for i in range(n)],
            "event_date": ts.astype(str),
            "churn": np.random.RandomState(99).randint(0, 2, n),
        })

        detector = ScenarioDetector(label_window_days=90)
        _, ts_config, _ = detector.detect(raw_df, "churn")

        analyzer = DatetimeOrderAnalyzer()
        last_action = analyzer.derive_last_action_date(raw_df)

        cutoff_analyzer = CutoffAnalyzer()
        analysis = cutoff_analyzer.analyze(raw_df, timestamp_series=last_action, n_bins=50)
        cutoff = analysis.suggest_cutoff(train_ratio=0.9)
        split = analysis.split_at_cutoff(cutoff)

        preparer = UnifiedDataPreparer(output_dir, ts_config)
        prepared = preparer.prepare_from_raw(raw_df, "churn", "entity_id")
        # Use same series from raw data (not re-derived from prepared)
        _, meta = preparer.create_training_snapshot(
            prepared, cutoff, timestamp_series=last_action
        )

        assert abs(meta["row_count"] - split.train_count) <= 1, (
            f"Snapshot ({meta['row_count']}) and split_at_cutoff ({split.train_count}) disagree"
        )

    def test_no_datetime_columns_falls_back_to_feature_timestamp(self, output_dir):
        n = 500
        raw_df = pd.DataFrame({
            "customer_id": [f"C{i}" for i in range(n)],
            "numeric_col": range(n),
            "target_col": np.random.RandomState(42).randint(0, 2, n),
        })

        analyzer = DatetimeOrderAnalyzer()
        last_action = analyzer.derive_last_action_date(raw_df)
        assert last_action is None  # No datetime columns

        # Fallback: use synthetic strategy since no timestamps
        from customer_retention.stages.temporal import TimestampConfig, TimestampStrategy
        ts_config = TimestampConfig(strategy=TimestampStrategy.SYNTHETIC_INDEX)

        preparer = UnifiedDataPreparer(output_dir, ts_config)
        prepared = preparer.prepare_from_raw(raw_df, "target_col", "customer_id")

        # Without last_action, snapshot uses feature_timestamp (synthetic)
        cutoff = datetime(2024, 7, 1)
        _, meta = preparer.create_training_snapshot(prepared, cutoff)

        assert meta["row_count"] > 0
        assert meta["row_count"] <= n

    def test_cutoff_beyond_all_data_captures_everything(self, output_dir):
        n = 300
        ts = pd.date_range("2020-01-01", periods=n, freq="D")
        raw_df = pd.DataFrame({
            "entity_id": [f"E{i}" for i in range(n)],
            "obs_date": ts.astype(str),
            "target_col": [0] * n,
        })

        detector = ScenarioDetector(label_window_days=90)
        _, ts_config, _ = detector.detect(raw_df, "target_col")

        analyzer = DatetimeOrderAnalyzer()
        last_action = analyzer.derive_last_action_date(raw_df)

        preparer = UnifiedDataPreparer(output_dir, ts_config)
        prepared = preparer.prepare_from_raw(raw_df, "target_col", "entity_id")

        # Cutoff far in the future — all rows should be in snapshot
        _, meta = preparer.create_training_snapshot(
            prepared, datetime(2030, 1, 1), timestamp_series=last_action
        )
        assert meta["row_count"] == n

    def test_label_available_flag_partially_filters_stays_consistent(self, output_dir):
        n = 1000
        now = datetime.now()
        # Half the data is recent (within observation window)
        old_ts = pd.date_range("2020-01-01", periods=n // 2, freq="D")
        recent_ts = pd.date_range(
            now - pd.Timedelta(days=30), periods=n // 2, freq="h"
        )
        all_ts = pd.concat([pd.Series(old_ts), pd.Series(recent_ts)]).reset_index(drop=True)

        raw_df = pd.DataFrame({
            "entity_id": [f"E{i}" for i in range(n)],
            "obs_date": all_ts.astype(str),
            "target_col": np.random.RandomState(42).randint(0, 2, n),
        })

        from customer_retention.stages.temporal import TimestampConfig, TimestampStrategy
        ts_config = TimestampConfig(
            strategy=TimestampStrategy.PRODUCTION,
            feature_timestamp_column="obs_date",
            derive_label_from_feature=True,
            observation_window_days=90,
        )

        analyzer = DatetimeOrderAnalyzer()
        last_action = analyzer.derive_last_action_date(raw_df)

        cutoff_analyzer = CutoffAnalyzer()
        analysis = cutoff_analyzer.analyze(raw_df, timestamp_series=last_action, n_bins=50)
        cutoff = analysis.suggest_cutoff(train_ratio=0.9)

        preparer = UnifiedDataPreparer(output_dir, ts_config)
        prepared = preparer.prepare_from_raw(raw_df, "target_col", "entity_id")

        # Recent rows should have label_available_flag=False (within window)
        available = prepared["label_available_flag"].sum()
        assert available < n  # Not all rows available

        _, meta = preparer.create_training_snapshot(
            prepared, cutoff, timestamp_series=last_action
        )

        # Snapshot should be <= available rows (filtered by both ts and flag)
        assert meta["row_count"] <= available
