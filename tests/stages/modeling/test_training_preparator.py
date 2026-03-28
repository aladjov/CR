from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from customer_retention.core.compat.timing import TimingEntry

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def base_df():
    np.random.seed(42)
    n = 200
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    return pd.DataFrame(
        {
            "entity_id": [f"e{i % 30}" for i in range(n)],
            "as_of_date": dates,
            "feat_a": np.random.randn(n) * 10 + 50,
            "feat_b": np.random.randn(n) * 5 + 20,
            "feat_c": np.random.choice(["cat", "dog", "fish"], n),
            "feat_d": pd.date_range("2020-01-01", periods=n, freq="h"),
            "feat_e": np.random.randn(n),
            "target": np.random.choice([0, 1], n, p=[0.3, 0.7]),
        }
    )


@pytest.fixture
def feature_cols():
    return ["feat_a", "feat_b", "feat_c", "feat_d", "feat_e"]


@pytest.fixture
def preparator(feature_cols):
    from customer_retention.stages.modeling.training_preparator import TrainingPreparator

    return TrainingPreparator(
        target_column="target",
        feature_columns=feature_cols,
        purge_gap_days=30,
        test_size=0.2,
    )


# ---------------------------------------------------------------------------
# TestClassifyColumns
# ---------------------------------------------------------------------------


class TestClassifyColumns:
    def test_excludes_datetime_columns(self, preparator, base_df):
        non_dt, _ = preparator._classify_columns(base_df, ["feat_a", "feat_d", "feat_e"])
        assert "feat_d" not in non_dt
        assert "feat_a" in non_dt
        assert "feat_e" in non_dt

    def test_excludes_timedelta_columns(self, preparator):
        df = pd.DataFrame({"a": [1.0], "b": [pd.Timedelta("1 day")]})
        non_dt, _ = preparator._classify_columns(df, ["a", "b"])
        assert non_dt == ["a"]

    def test_keeps_numeric_and_object(self, preparator, base_df):
        non_dt, _ = preparator._classify_columns(base_df, ["feat_a", "feat_b", "feat_c"])
        assert non_dt == ["feat_a", "feat_b", "feat_c"]

    def test_returns_empty_when_all_datetime(self, preparator):
        df = pd.DataFrame({"a": pd.to_datetime(["2023-01-01"]), "b": pd.to_datetime(["2023-01-01"])})
        non_dt, _ = preparator._classify_columns(df, ["a", "b"])
        assert non_dt == []

    def test_detects_object_columns(self, preparator, base_df):
        _, obj_cols = preparator._classify_columns(base_df, ["feat_a", "feat_b", "feat_c"])
        assert "feat_c" in obj_cols
        assert "feat_a" not in obj_cols

    def test_object_columns_are_subset_of_non_dt(self, preparator, base_df):
        non_dt, obj_cols = preparator._classify_columns(
            base_df,
            ["feat_a", "feat_b", "feat_c", "feat_d"],
        )
        assert set(obj_cols).issubset(set(non_dt))
        assert "feat_d" not in non_dt  # datetime excluded


# ---------------------------------------------------------------------------
# TestDropMissingTarget
# ---------------------------------------------------------------------------


class TestDropMissingTarget:
    def test_drops_nan_target_rows(self, preparator):
        df = pd.DataFrame({"target": [1, np.nan, 0, np.nan], "x": [1, 2, 3, 4]})
        result, count = preparator._drop_missing_target(df)
        assert len(result) == 2
        assert count == 2

    def test_no_nans_returns_zero_count(self, preparator):
        df = pd.DataFrame({"target": [1, 0, 1], "x": [1, 2, 3]})
        result, count = preparator._drop_missing_target(df)
        assert len(result) == 3
        assert count == 0

    def test_all_nan_raises_value_error(self, preparator):
        df = pd.DataFrame({"target": [np.nan, np.nan], "x": [1, 2]})
        with pytest.raises(ValueError, match="all target values are NaN"):
            preparator._drop_missing_target(df)

    def test_preserves_other_columns(self, preparator):
        df = pd.DataFrame({"target": [1, np.nan, 0], "x": [10, 20, 30]})
        result, _ = preparator._drop_missing_target(df)
        assert list(result["x"]) == [10, 30]


# ---------------------------------------------------------------------------
# TestEncodeObjectColumns
# ---------------------------------------------------------------------------


class TestEncodeObjectColumns:
    def test_classify_detects_object_columns_for_encoding(self, preparator):
        from customer_retention.core.compat import bulk_label_encode

        df = pd.DataFrame(
            {
                "feat_c": ["cat", "dog", "cat", "fish"],
                "feat_a": [1.0, 2.0, 3.0, 4.0],
            }
        )
        _, obj_cols = preparator._classify_columns(df, ["feat_c", "feat_a"])
        assert obj_cols == ["feat_c"]
        result = bulk_label_encode(df, obj_cols)
        assert result["feat_c"].dtype in (np.int64, np.int32, int)

    def test_no_object_cols_returns_empty_list(self, preparator):
        df = pd.DataFrame({"feat_a": [1.0, 2.0], "feat_b": [3.0, 4.0]})
        _, obj_cols = preparator._classify_columns(df, ["feat_a", "feat_b"])
        assert obj_cols == []

    def test_only_feature_cols_checked_for_object(self, preparator):
        df = pd.DataFrame(
            {
                "feat_c": ["cat", "dog"],
                "other_obj": ["x", "y"],
                "feat_a": [1.0, 2.0],
            }
        )
        _, obj_cols = preparator._classify_columns(df, ["feat_c", "feat_a"])
        assert "other_obj" not in obj_cols
        assert "feat_c" in obj_cols


# ---------------------------------------------------------------------------
# TestSampleEntities
# ---------------------------------------------------------------------------


class TestSampleEntities:
    def test_samples_when_exceeds_max_rows(self, feature_cols):
        from customer_retention.stages.modeling.training_preparator import TrainingPreparator

        prep = TrainingPreparator(
            target_column="target",
            feature_columns=feature_cols,
            max_rows=500,
        )
        np.random.seed(42)
        n = 2000
        df = pd.DataFrame(
            {
                "entity_id": [f"e{i % 200}" for i in range(n)],
                "as_of_date": pd.date_range("2023-01-01", periods=n, freq="h"),
                "feat_a": np.random.randn(n),
                "target": np.random.choice([0, 1], n),
            }
        )
        result = prep._sample_entities(df)
        assert len(result) < len(df)
        sampled_entities = result["entity_id"].unique()
        for eid in sampled_entities:
            original_count = (df["entity_id"] == eid).sum()
            sampled_count = (result["entity_id"] == eid).sum()
            assert original_count == sampled_count

    def test_noop_when_below_max_rows(self, feature_cols):
        from customer_retention.stages.modeling.training_preparator import TrainingPreparator

        prep = TrainingPreparator(
            target_column="target",
            feature_columns=feature_cols,
            max_rows=1000,
        )
        df = pd.DataFrame(
            {
                "entity_id": ["e1"] * 10,
                "feat_a": range(10),
                "target": [0] * 10,
            }
        )
        result = prep._sample_entities(df)
        assert len(result) == 10

    def test_noop_when_max_rows_none(self, preparator):
        df = pd.DataFrame(
            {
                "entity_id": ["e1"] * 100,
                "feat_a": range(100),
                "target": [0] * 100,
            }
        )
        result = preparator._sample_entities(df)
        assert len(result) == 100

    def test_rows_per_entity_estimation(self, feature_cols):
        from customer_retention.stages.modeling.training_preparator import TrainingPreparator

        prep = TrainingPreparator(
            target_column="target",
            feature_columns=feature_cols,
            max_rows=500,
        )
        n = 2000
        df = pd.DataFrame(
            {
                "entity_id": [f"e{i % 200}" for i in range(n)],
                "feat_a": range(n),
                "target": [0] * n,
            }
        )
        result = prep._sample_entities(df)
        assert len(result) < n


# ---------------------------------------------------------------------------
# TestFillnaAndZeroVariance
# ---------------------------------------------------------------------------


class TestFillnaAndZeroVariance:
    def test_drops_constant_columns(self, preparator):
        X_train = pd.DataFrame({"a": [1.0, 1.0, 1.0], "b": [1.0, 2.0, 3.0]})
        X_test = pd.DataFrame({"a": [1.0, 1.0, 1.0], "b": [4.0, 5.0, 6.0]})
        Xtr, Xte, dropped, _nulls = preparator._fillna_and_drop_zero_variance(X_train, X_test)
        assert "a" not in Xtr.columns
        assert "a" not in Xte.columns
        assert "a" in dropped

    def test_reports_dropped_names(self, preparator):
        X_train = pd.DataFrame({"a": [5.0, 5.0], "b": [1.0, 2.0], "c": [3.0, 3.0]})
        X_test = pd.DataFrame({"a": [5.0, 5.0], "b": [3.0, 4.0], "c": [3.0, 3.0]})
        _, _, dropped, _nulls = preparator._fillna_and_drop_zero_variance(X_train, X_test)
        assert set(dropped) == {"a", "c"}

    def test_fillna_before_variance_check(self, preparator):
        X_train = pd.DataFrame({"a": [np.nan, np.nan, np.nan], "b": [1.0, 2.0, 3.0]})
        X_test = pd.DataFrame({"a": [np.nan, np.nan, np.nan], "b": [4.0, 5.0, 6.0]})
        Xtr, Xte, dropped, _nulls = preparator._fillna_and_drop_zero_variance(X_train, X_test)
        assert "a" in dropped

    def test_no_columns_dropped_when_all_vary(self, preparator):
        X_train = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        X_test = pd.DataFrame({"a": [5.0, 6.0], "b": [7.0, 8.0]})
        Xtr, Xte, dropped, _nulls = preparator._fillna_and_drop_zero_variance(X_train, X_test)
        assert dropped == []
        assert list(Xtr.columns) == ["a", "b"]

    def test_null_counts_computed_before_fillna(self, preparator):
        X_train = pd.DataFrame({"a": [np.nan, 1.0, 2.0], "b": [1.0, 2.0, 3.0]})
        X_test = pd.DataFrame({"a": [np.nan, np.nan, 5.0], "b": [4.0, np.nan, 6.0]})
        _, _, _, nulls = preparator._fillna_and_drop_zero_variance(X_train, X_test)
        assert nulls == {"a": 3, "b": 1}

    def test_null_counts_excludes_dropped_zero_variance(self, preparator):
        X_train = pd.DataFrame({"const": [np.nan, np.nan, np.nan], "vary": [1.0, np.nan, 3.0]})
        X_test = pd.DataFrame({"const": [np.nan, np.nan, np.nan], "vary": [4.0, 5.0, 6.0]})
        _, _, dropped, nulls = preparator._fillna_and_drop_zero_variance(X_train, X_test)
        assert "const" in dropped
        assert "const" not in nulls
        assert nulls == {"vary": 1}

    def test_null_counts_zero_when_no_nulls(self, preparator):
        X_train = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        X_test = pd.DataFrame({"a": [5.0, 6.0], "b": [7.0, 8.0]})
        _, _, _, nulls = preparator._fillna_and_drop_zero_variance(X_train, X_test)
        assert nulls == {"a": 0, "b": 0}


# ---------------------------------------------------------------------------
# TestClassDistribution
# ---------------------------------------------------------------------------


class TestClassDistribution:
    def test_binary_target(self, preparator):
        y = pd.Series([0, 0, 1, 1, 1])
        result = preparator._class_distribution(y)
        assert result == {0: 2, 1: 3}

    def test_handles_value_counts_with_to_pandas(self, preparator):
        y = pd.Series([0, 1, 1, 0, 0])
        result = preparator._class_distribution(y)
        assert sum(result.values()) == 5


# ---------------------------------------------------------------------------
# Integration: TestPrepareLocalEndToEnd
# ---------------------------------------------------------------------------


class TestPrepareLocalEndToEnd:
    def test_full_pipeline_produces_result(self, preparator, base_df):
        from customer_retention.stages.modeling.training_preparator import TrainingPreparationResult

        result = preparator.prepare(base_df)
        assert isinstance(result, TrainingPreparationResult)
        assert len(result.X_train) > 0
        assert len(result.X_test) > 0
        assert len(result.y_train) > 0
        assert len(result.y_test) > 0

    def test_feature_names_exclude_metadata(self, preparator, base_df):
        from customer_retention.stages.modeling.cross_validator import _CV_DATE_COL, _CV_ENTITY_COL

        result = preparator.prepare(base_df)
        assert "target" not in result.feature_names
        assert "as_of_date" not in result.feature_names
        assert "entity_id" not in result.feature_names
        assert _CV_ENTITY_COL not in result.feature_names
        assert _CV_DATE_COL not in result.feature_names

    def test_scaled_mean_near_zero(self, preparator, base_df):
        result = preparator.prepare(base_df)
        means = result.X_train_scaled.mean()
        assert all(abs(m) < 0.5 for m in means), f"Means not near zero: {means.to_dict()}"

    def test_y_test_np_is_numpy(self, preparator, base_df):
        result = preparator.prepare(base_df)
        assert isinstance(result.y_test_np, np.ndarray)

    def test_train_entities_are_native_pandas(self, preparator, base_df):
        result = preparator.prepare(base_df)
        assert isinstance(result.train_entities, pd.Series)
        assert isinstance(result.train_dates, pd.Series)

    def test_null_counts_populated(self, preparator, base_df):
        result = preparator.prepare(base_df)
        assert isinstance(result.null_counts, dict)
        for feat in result.feature_names:
            assert feat in result.null_counts
            assert isinstance(result.null_counts[feat], int)

    def test_float32_applied_pre_collect(self, base_df, feature_cols):
        from customer_retention.stages.modeling.training_preparator import TrainingPreparator

        prep = TrainingPreparator(
            target_column="target",
            feature_columns=feature_cols,
            use_float32=True,
        )
        result = prep.prepare(base_df)
        for col in result.feature_names:
            assert result.X_train[col].dtype == np.float32, f"{col} not float32"
            assert result.X_train_scaled[col].dtype == np.float32, f"{col} scaled not float32"

    def test_float32_off_preserves_original_dtypes(self, base_df, feature_cols):
        from customer_retention.stages.modeling.training_preparator import TrainingPreparator

        prep = TrainingPreparator(
            target_column="target",
            feature_columns=feature_cols,
            use_float32=False,
        )
        result = prep.prepare(base_df)
        float_cols = [c for c in result.feature_names if result.X_train[c].dtype.kind == "f"]
        assert len(float_cols) > 0, "Expected at least one float column"
        for col in float_cols:
            assert result.X_train[col].dtype == np.float64, f"{col} not float64"


# ---------------------------------------------------------------------------
# TestTimingProfiling
# ---------------------------------------------------------------------------


class TestTimingProfiling:
    def test_timing_entries_produced(self, preparator, base_df):
        result = preparator.prepare(base_df)
        assert len(result.timing_entries) > 0
        assert all(isinstance(e, TimingEntry) for e in result.timing_entries)

    def test_expected_labels_present(self, preparator, base_df):
        result = preparator.prepare(base_df)
        labels = {e.label for e in result.timing_entries}
        expected = {
            "classify_columns",
            "drop_missing_target",
            "encode_object_columns",
            "checkpoint",
            "median_impute",
            "temporal_split",
            "fillna_and_drop_zero_variance",
            "scale_features",
            "class_distribution",
        }
        assert expected.issubset(labels), f"Missing: {expected - labels}"


# ---------------------------------------------------------------------------
# TestProgressCallback
# ---------------------------------------------------------------------------


class TestProgressCallback:
    def test_callback_invoked_for_each_step(self, feature_cols, base_df):
        from customer_retention.stages.modeling.training_preparator import TrainingPreparator

        invocations = []
        prep = TrainingPreparator(
            target_column="target",
            feature_columns=feature_cols,
            purge_gap_days=30,
            test_size=0.2,
            on_progress=lambda s, t, label, elapsed: invocations.append((s, t, label, elapsed)),
        )
        prep.prepare(base_df)
        labels = [label for _, _, label, _ in invocations]
        assert "temporal_split" in labels
        assert "scale_features" in labels
        assert all(elapsed >= 0 for _, _, _, elapsed in invocations)

    def test_callback_reports_step_numbers(self, feature_cols, base_df):
        from customer_retention.stages.modeling.training_preparator import TrainingPreparator

        invocations = []
        prep = TrainingPreparator(
            target_column="target",
            feature_columns=feature_cols,
            purge_gap_days=30,
            test_size=0.2,
            on_progress=lambda s, t, label, elapsed: invocations.append((s, t)),
        )
        prep.prepare(base_df)
        steps = [s for s, _ in invocations]
        totals = [t for _, t in invocations]
        assert steps == list(range(1, len(invocations) + 1))
        assert all(t == totals[0] for t in totals)

    def test_callback_none_is_noop(self, feature_cols, base_df):
        from customer_retention.stages.modeling.training_preparator import TrainingPreparator

        prep = TrainingPreparator(
            target_column="target",
            feature_columns=feature_cols,
            purge_gap_days=30,
            test_size=0.2,
            on_progress=None,
        )
        result = prep.prepare(base_df)
        assert len(result.X_train) > 0

    def test_print_preparation_progress_format(self, capsys):
        from customer_retention.stages.modeling.training_preparator import print_preparation_progress

        print_preparation_progress(3, 10, "temporal_split", 1.234)
        captured = capsys.readouterr()
        assert "[3/10] temporal_split: 1.2s" in captured.out

    def test_progress_tracker_includes_wall_time(self, capsys):
        from customer_retention.stages.modeling.training_preparator import PreparationProgressTracker

        tracker = PreparationProgressTracker()
        tracker(1, 10, "classify_columns", 4.7)
        captured = capsys.readouterr()
        assert "[1/10] classify_columns: 4.7s" in captured.out
        assert "wall:" in captured.out
        assert "ETA" in captured.out

    def test_progress_tracker_no_eta_on_last_step(self, capsys):
        from customer_retention.stages.modeling.training_preparator import PreparationProgressTracker

        tracker = PreparationProgressTracker()
        tracker(10, 10, "class_distribution", 0.1)
        captured = capsys.readouterr()
        assert "ETA" not in captured.out

    def test_log_sub_prints_when_progress_enabled(self, capsys):
        from customer_retention.stages.modeling.training_preparator import TrainingPreparator

        prep = TrainingPreparator(
            target_column="target",
            feature_columns=["a"],
            on_progress=lambda s, t, label, e: None,
        )
        prep._log_sub("test substep: 1.2s")
        captured = capsys.readouterr()
        assert "→ test substep: 1.2s" in captured.out

    def test_log_sub_silent_when_progress_disabled(self, capsys):
        from customer_retention.stages.modeling.training_preparator import TrainingPreparator

        prep = TrainingPreparator(
            target_column="target",
            feature_columns=["a"],
            on_progress=None,
        )
        prep._log_sub("test substep: 1.2s")
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_callback_receives_labels_in_order(self, feature_cols, base_df):
        from customer_retention.stages.modeling.training_preparator import TrainingPreparator

        labels = []
        prep = TrainingPreparator(
            target_column="target",
            feature_columns=feature_cols,
            purge_gap_days=30,
            test_size=0.2,
            on_progress=lambda s, t, label, elapsed: labels.append(label),
        )
        prep.prepare(base_df)
        assert labels[0] == "classify_columns"
        assert labels[-1] == "class_distribution"


# ---------------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_single_feature_column(self):
        from customer_retention.stages.modeling.training_preparator import TrainingPreparator

        np.random.seed(42)
        n = 100
        df = pd.DataFrame(
            {
                "entity_id": [f"e{i % 10}" for i in range(n)],
                "as_of_date": pd.date_range("2023-01-01", periods=n, freq="D"),
                "feat_only": np.random.randn(n),
                "target": np.random.choice([0, 1], n, p=[0.3, 0.7]),
            }
        )
        prep = TrainingPreparator(
            target_column="target",
            feature_columns=["feat_only"],
            purge_gap_days=10,
            test_size=0.2,
        )
        result = prep.prepare(df)
        assert result.feature_names == ["feat_only"]

    def test_small_dataset(self):
        from customer_retention.stages.modeling.training_preparator import TrainingPreparator

        np.random.seed(42)
        n = 50
        df = pd.DataFrame(
            {
                "entity_id": [f"e{i % 5}" for i in range(n)],
                "as_of_date": pd.date_range("2023-01-01", periods=n, freq="D"),
                "feat_a": np.random.randn(n),
                "feat_b": np.random.randn(n),
                "target": np.random.choice([0, 1], n, p=[0.4, 0.6]),
            }
        )
        prep = TrainingPreparator(
            target_column="target",
            feature_columns=["feat_a", "feat_b"],
            purge_gap_days=5,
            test_size=0.2,
        )
        result = prep.prepare(df)
        assert len(result.X_train) + len(result.X_test) <= n

    def test_no_object_columns(self):
        from customer_retention.stages.modeling.training_preparator import TrainingPreparator

        np.random.seed(42)
        n = 100
        df = pd.DataFrame(
            {
                "entity_id": [f"e{i % 10}" for i in range(n)],
                "as_of_date": pd.date_range("2023-01-01", periods=n, freq="D"),
                "feat_a": np.random.randn(n),
                "feat_b": np.random.randn(n) * 2,
                "target": np.random.choice([0, 1], n, p=[0.3, 0.7]),
            }
        )
        prep = TrainingPreparator(
            target_column="target",
            feature_columns=["feat_a", "feat_b"],
            purge_gap_days=10,
            test_size=0.2,
        )
        result = prep.prepare(df)
        assert "feat_a" in result.feature_names
        assert "feat_b" in result.feature_names


# ---------------------------------------------------------------------------
# Distributed Path (mock-based)
# ---------------------------------------------------------------------------


class TestFinalizeDistributed:
    def _make_split_inputs(self):
        np.random.seed(42)
        X_train = pd.DataFrame({"f1": np.random.randn(80), "f2": np.random.randn(80)})
        X_test = pd.DataFrame({"f1": np.random.randn(20), "f2": np.random.randn(20)})
        y_train = pd.Series(np.random.choice([0, 1], 80), name="target")
        y_test = pd.Series(np.random.choice([0, 1], 20), name="target")
        train_entities = pd.Series([f"e{i}" for i in range(80)], name="entity_id")
        train_dates = pd.Series(pd.date_range("2023-01-01", periods=80, freq="D"), name="as_of_date")
        return X_train, X_test, y_train, y_test, train_entities, train_dates

    def test_no_banned_patterns_in_finalize_distributed(self):
        import inspect

        from customer_retention.stages.modeling.training_preparator import TrainingPreparator

        source = inspect.getsource(TrainingPreparator._finalize_distributed)
        assert "X_train[" not in source or "__setitem__" not in source, (
            "_finalize_distributed must not use __setitem__ on pyspark.pandas"
        )
        assert "fit_transform(" not in source, (
            "_finalize_distributed should use native Spark scaling, not fit_transform"
        )

    def test_uses_native_spark_scaling(self):
        import inspect

        from customer_retention.stages.modeling.training_preparator import TrainingPreparator

        source = inspect.getsource(TrainingPreparator._finalize_distributed)
        assert "_compute_params_spark" in source
        assert "_apply_spark" in source
        assert "as_spark_df" in source

    @patch("customer_retention.stages.modeling.training_preparator.spark_checkpoint", side_effect=lambda x: x)
    @patch(
        "customer_retention.stages.modeling.training_preparator.concat",
        side_effect=lambda objs, **kw: pd.concat(objs, **kw),
    )
    @patch("customer_retention.stages.modeling.training_preparator.collect_for_sklearn", side_effect=lambda x: x)
    def test_collect_only_for_metadata(self, mock_collect, mock_concat, mock_ckpt):
        from unittest.mock import MagicMock

        from customer_retention.stages.modeling.training_preparator import TrainingPreparator

        X_train, X_test, y_train, y_test, train_entities, train_dates = self._make_split_inputs()

        def fake_as_spark(df):
            mock = MagicMock()
            mock.columns = list(df.columns) if hasattr(df, "columns") else []
            mock.select.side_effect = lambda cols: fake_as_spark(df[cols] if isinstance(cols, list) else df)
            mock.drop.return_value = mock
            return mock

        prep = TrainingPreparator(target_column="target", feature_columns=["f1", "f2"])
        with (
            patch("customer_retention.stages.modeling.training_preparator.as_spark_df", side_effect=fake_as_spark),
            patch(
                "customer_retention.core.compat.spark_backend._as_pandas_api",
                side_effect=lambda x: pd.DataFrame({"f1": [1], "f2": [2]}),
            ),
            patch("customer_retention.stages.modeling.spark_feature_scaler.SparkFeatureScaler") as mock_cls,
        ):
            mock_scaler = mock_cls.return_value
            mock_scaler._compute_params_spark.return_value = {}
            mock_scaler._apply_spark.side_effect = lambda df: df
            prep._finalize_distributed(
                X_train,
                X_test,
                y_train,
                y_test,
                train_entities,
                train_dates,
                ["f1", "f2"],
            )

        assert mock_collect.call_count == 3  # y_test, train_entities, train_dates
