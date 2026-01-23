import shutil
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from customer_retention.stages.temporal.cutoff_analyzer import CutoffAnalyzer
from customer_retention.stages.temporal.snapshot_manager import SnapshotManager
from customer_retention.stages.temporal.timestamp_discovery import DatetimeOrderAnalyzer


class TestSnapshotUsesAnalysisTimestamp:
    """Integration tests verifying snapshot creation uses the same timestamp
    source as cutoff analysis, preventing the bug where analysis uses
    last_action_date (coalesced) but snapshot filters on feature_timestamp."""

    @pytest.fixture
    def temp_dir(self):
        temp_path = Path(tempfile.mkdtemp())
        yield temp_path
        shutil.rmtree(temp_path)

    @pytest.fixture
    def event_df(self):
        np.random.seed(42)
        n = 1000
        sent_dates = pd.date_range("2022-01-01", "2023-06-30", periods=n)
        unsubscribe_mask = np.random.random(n) < 0.15
        unsubscribe_dates = pd.Series([None] * n, dtype="object")
        unsubscribe_dates[unsubscribe_mask] = (
            sent_dates[unsubscribe_mask] + pd.Timedelta(days=30)
        ).astype(str)
        return pd.DataFrame({
            "email_id": [f"E{i}" for i in range(n)],
            "customer_id": [f"C{i % 200}" for i in range(n)],
            "sent_date": sent_dates.astype(str),
            "campaign_type": np.random.choice(["promo", "newsletter", "welcome"], n),
            "opened": np.random.randint(0, 2, n),
            "unsubscribed": np.random.randint(0, 2, n),
            "unsubscribe_date": unsubscribe_dates,
        })

    @pytest.fixture
    def prepared_df(self, event_df):
        analyzer = DatetimeOrderAnalyzer()
        last_action = analyzer.derive_last_action_date(event_df)
        feature_ts = pd.to_datetime(event_df["sent_date"])
        label_ts = feature_ts + pd.Timedelta(days=180)
        df = event_df.copy()
        df["feature_timestamp"] = feature_ts
        df["label_timestamp"] = label_ts
        df["label_available_flag"] = True
        df["target"] = df["unsubscribed"]
        return df, last_action

    def test_snapshot_with_coalesced_series_matches_analysis_split(self, temp_dir, prepared_df):
        df, last_action = prepared_df
        manager = SnapshotManager(temp_dir)
        analyzer = CutoffAnalyzer()

        analysis = analyzer.analyze(df, timestamp_series=last_action)
        cutoff = analysis.suggest_cutoff(train_ratio=0.9)

        meta = manager.create_snapshot(
            df, cutoff, "target", timestamp_series=last_action
        )

        expected_train = (last_action.dropna() <= cutoff).sum()
        assert abs(meta.row_count - expected_train) <= 1

    def test_snapshot_without_series_uses_feature_timestamp(self, temp_dir, prepared_df):
        df, last_action = prepared_df
        manager = SnapshotManager(temp_dir)
        analyzer = CutoffAnalyzer()

        analysis = analyzer.analyze(df, timestamp_series=last_action)
        cutoff = analysis.suggest_cutoff(train_ratio=0.9)

        meta_default = manager.create_snapshot(df, cutoff, "target")
        meta_coalesced = manager.create_snapshot(
            df, cutoff, "target", snapshot_name="coalesced", timestamp_series=last_action
        )

        # Different timestamp sources produce different splits
        assert meta_coalesced.row_count != meta_default.row_count

    def test_snapshot_row_count_is_approximately_90_percent(self, temp_dir, prepared_df):
        df, last_action = prepared_df
        manager = SnapshotManager(temp_dir)
        analyzer = CutoffAnalyzer()

        analysis = analyzer.analyze(df, timestamp_series=last_action)
        cutoff = analysis.suggest_cutoff(train_ratio=0.9)

        meta = manager.create_snapshot(
            df, cutoff, "target", timestamp_series=last_action
        )

        ratio = meta.row_count / analysis.covered_rows
        assert 0.85 <= ratio <= 0.95, (
            f"Snapshot has {meta.row_count} rows = {ratio:.1%} of "
            f"{analysis.covered_rows} covered rows (expected ~90%)"
        )

    def test_split_result_consistent_with_snapshot(self, temp_dir, prepared_df):
        df, last_action = prepared_df
        manager = SnapshotManager(temp_dir)
        analyzer = CutoffAnalyzer()

        analysis = analyzer.analyze(df, timestamp_series=last_action)
        cutoff = analysis.suggest_cutoff(train_ratio=0.9)
        split = analysis.split_at_cutoff(cutoff)

        meta = manager.create_snapshot(
            df, cutoff, "target", timestamp_series=last_action
        )

        assert meta.row_count == split.train_count

    def test_discrepancy_detected_when_feature_timestamp_differs(self, temp_dir):
        np.random.seed(42)
        n = 500
        # feature_timestamp covers only first 50% of rows (rest is NaT)
        feature_ts = pd.Series([None] * n, dtype="datetime64[ns]")
        feature_ts.iloc[:250] = pd.date_range("2022-01-01", periods=250, freq="D")
        # coalesced series covers all rows
        coalesced = pd.Series(
            pd.date_range("2022-01-01", periods=n, freq="D"), name="last_action_date"
        )
        df = pd.DataFrame({
            "entity_id": range(n),
            "feature_timestamp": feature_ts,
            "label_timestamp": coalesced + pd.Timedelta(days=180),
            "label_available_flag": [True] * n,
            "target": np.random.randint(0, 2, n),
        })

        manager = SnapshotManager(temp_dir)
        analyzer = CutoffAnalyzer()
        analysis = analyzer.analyze(df, timestamp_series=coalesced)
        cutoff = analysis.suggest_cutoff(train_ratio=0.9)

        meta_feature_ts = manager.create_snapshot(df, cutoff, "target")
        meta_coalesced = manager.create_snapshot(
            df, cutoff, "target", snapshot_name="coalesced", timestamp_series=coalesced
        )

        # feature_timestamp only has 250 rows, so default path misses half the data
        assert meta_coalesced.row_count > meta_feature_ts.row_count
        # The coalesced version should be ~90% of 500
        assert meta_coalesced.row_count >= 400

    def test_unresolvable_rows_not_silently_dropped(self, temp_dir, prepared_df):
        df, last_action = prepared_df
        analyzer = CutoffAnalyzer()

        analysis = analyzer.analyze(df, timestamp_series=last_action)
        cutoff = analysis.suggest_cutoff(train_ratio=0.9)
        split = analysis.split_at_cutoff(cutoff)

        assert split.train_count + split.score_count + split.unresolvable_count == split.original_count
        assert split.original_count == len(df)


class TestSnapshotSanityChecks:
    """Sanity checks that snapshot row counts are reasonable relative to input."""

    @pytest.fixture
    def temp_dir(self):
        temp_path = Path(tempfile.mkdtemp())
        yield temp_path
        shutil.rmtree(temp_path)

    def test_snapshot_not_suspiciously_small(self, temp_dir):
        n = 2000
        ts = pd.Series(pd.date_range("2022-01-01", periods=n, freq="D"))
        df = pd.DataFrame({
            "feature_timestamp": ts,
            "label_timestamp": ts + pd.Timedelta(days=180),
            "label_available_flag": [True] * n,
            "target": [0] * n,
        })
        manager = SnapshotManager(temp_dir)
        analyzer = CutoffAnalyzer()
        analysis = analyzer.analyze(df, timestamp_column="feature_timestamp")
        cutoff = analysis.suggest_cutoff(train_ratio=0.9)

        meta = manager.create_snapshot(df, cutoff, "target")

        ratio = meta.row_count / len(df)
        assert ratio >= 0.5, (
            f"Snapshot suspiciously small: {meta.row_count}/{len(df)} = {ratio:.1%}"
        )

    def test_snapshot_90_percent_split_yields_expected_range(self, temp_dir):
        n = 1000
        ts = pd.Series(pd.date_range("2022-01-01", periods=n, freq="D"))
        df = pd.DataFrame({
            "feature_timestamp": ts,
            "label_timestamp": ts + pd.Timedelta(days=180),
            "label_available_flag": [True] * n,
            "target": [0] * n,
        })
        manager = SnapshotManager(temp_dir)
        analyzer = CutoffAnalyzer()
        analysis = analyzer.analyze(df, timestamp_column="feature_timestamp")
        cutoff = analysis.suggest_cutoff(train_ratio=0.9)

        meta = manager.create_snapshot(df, cutoff, "target")
        ratio = meta.row_count / len(df)
        assert 0.85 <= ratio <= 0.95, f"Expected ~90% but got {ratio:.1%}"

    def test_all_label_unavailable_yields_empty_snapshot(self, temp_dir):
        n = 100
        ts = pd.Series(pd.date_range("2022-01-01", periods=n, freq="D"))
        df = pd.DataFrame({
            "feature_timestamp": ts,
            "label_timestamp": ts + pd.Timedelta(days=180),
            "label_available_flag": [False] * n,
            "target": [0] * n,
        })
        manager = SnapshotManager(temp_dir)
        meta = manager.create_snapshot(df, datetime(2023, 6, 1), "target")
        assert meta.row_count == 0
