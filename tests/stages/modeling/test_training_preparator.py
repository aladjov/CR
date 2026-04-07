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


# ---------------------------------------------------------------------------
# TestPrepareForDiagnostics
# ---------------------------------------------------------------------------


class TestPrepareForDiagnostics:
    @pytest.fixture
    def gold_df(self):
        np.random.seed(7)
        n = 400
        dates = pd.date_range("2024-01-01", periods=n, freq="D")
        return pd.DataFrame({
            "entity_id": [f"e{i % 40}" for i in range(n)],
            "as_of_date": dates,
            "feat_value": np.random.randn(n) * 3 + 10,
            "feat_count": np.random.randint(0, 100, n).astype(float),
            "feat_count_is_zero": (np.arange(n) % 5 == 0).astype(int),
            "target": np.random.choice([0, 1], n, p=[0.4, 0.6]),
        })

    def test_returns_split_result_with_matching_feature_columns(self, gold_df):
        from customer_retention.stages.modeling.training_preparator import prepare_for_diagnostics

        split = prepare_for_diagnostics(
            gold_df,
            target_column="target",
            feature_names=["feat_value", "feat_count", "feat_count_is_zero"],
            purge_gap_days=14,
        )
        assert list(split.X_train.columns) == ["feat_value", "feat_count", "feat_count_is_zero"]
        assert list(split.X_test.columns) == ["feat_value", "feat_count", "feat_count_is_zero"]
        assert len(split.X_train) > 0 and len(split.X_test) > 0

    def test_raises_keyerror_listing_missing_features(self, gold_df):
        from customer_retention.stages.modeling.training_preparator import prepare_for_diagnostics

        with pytest.raises(KeyError, match="days_since_last_event_y_is_zero"):
            prepare_for_diagnostics(
                gold_df,
                target_column="target",
                feature_names=["feat_value", "days_since_last_event_y_is_zero"],
                purge_gap_days=14,
            )

    def test_preserves_column_with_constant_binary_flag(self, gold_df):
        from customer_retention.stages.modeling.training_preparator import prepare_for_diagnostics

        constant_flag = gold_df.assign(feat_all_zero=0)
        split = prepare_for_diagnostics(
            constant_flag,
            target_column="target",
            feature_names=["feat_value", "feat_all_zero"],
            purge_gap_days=14,
        )
        assert "feat_all_zero" in split.X_train.columns
        assert "feat_all_zero" in split.X_test.columns

    def test_drops_rows_with_missing_target(self, gold_df):
        from customer_retention.stages.modeling.training_preparator import prepare_for_diagnostics

        dirty = gold_df.copy()
        dirty.loc[dirty.index[:20], "target"] = np.nan
        split = prepare_for_diagnostics(
            dirty,
            target_column="target",
            feature_names=["feat_value", "feat_count"],
            purge_gap_days=14,
        )
        assert len(split.y_train) + len(split.y_test) <= len(gold_df) - 20
        assert not split.y_train.isna().any()
        assert not split.y_test.isna().any()

    def test_imputes_missing_feature_values(self, gold_df):
        from customer_retention.stages.modeling.training_preparator import prepare_for_diagnostics

        noisy = gold_df.copy()
        noisy.loc[noisy.index[:25], "feat_value"] = np.nan
        split = prepare_for_diagnostics(
            noisy,
            target_column="target",
            feature_names=["feat_value", "feat_count"],
            purge_gap_days=14,
        )
        assert not split.X_train["feat_value"].isna().any()
        assert not split.X_test["feat_value"].isna().any()

    def test_deterministic_split_across_invocations(self, gold_df):
        from customer_retention.stages.modeling.training_preparator import prepare_for_diagnostics

        kwargs = dict(
            target_column="target",
            feature_names=["feat_value", "feat_count", "feat_count_is_zero"],
            purge_gap_days=14,
        )
        first = prepare_for_diagnostics(gold_df, **kwargs)
        second = prepare_for_diagnostics(gold_df, **kwargs)
        assert len(first.X_train) == len(second.X_train)
        assert len(first.X_test) == len(second.X_test)
        pd.testing.assert_series_equal(
            first.y_train.reset_index(drop=True),
            second.y_train.reset_index(drop=True),
        )

    def test_purge_gap_removes_training_rows_close_to_cutoff(self, gold_df):
        from customer_retention.stages.modeling.training_preparator import prepare_for_diagnostics

        narrow = prepare_for_diagnostics(
            gold_df, target_column="target",
            feature_names=["feat_value", "feat_count"], purge_gap_days=0,
        )
        wide = prepare_for_diagnostics(
            gold_df, target_column="target",
            feature_names=["feat_value", "feat_count"], purge_gap_days=60,
        )
        assert len(wide.X_train) < len(narrow.X_train)

    def test_does_not_mutate_input_gold(self, gold_df):
        from customer_retention.stages.modeling.training_preparator import prepare_for_diagnostics

        original_cols = list(gold_df.columns)
        original_len = len(gold_df)
        prepare_for_diagnostics(
            gold_df, target_column="target",
            feature_names=["feat_value", "feat_count"], purge_gap_days=14,
        )
        assert list(gold_df.columns) == original_cols
        assert len(gold_df) == original_len


class TestFmtDuration:
    def test_seconds_format(self):
        from customer_retention.stages.modeling.training_preparator import _fmt_duration
        assert _fmt_duration(0) == "0s"
        assert _fmt_duration(45.7) == "46s"
        assert _fmt_duration(59.4) == "59s"

    def test_minutes_format(self):
        from customer_retention.stages.modeling.training_preparator import _fmt_duration
        assert _fmt_duration(60) == "1m00s"
        assert _fmt_duration(125) == "2m05s"
        assert _fmt_duration(3599) == "59m59s"

    def test_hours_format(self):
        from customer_retention.stages.modeling.training_preparator import _fmt_duration
        assert _fmt_duration(3600) == "1h00m"
        assert _fmt_duration(3600 + 120) == "1h02m"
        assert _fmt_duration(2 * 3600 + 30 * 60) == "2h30m"


class TestPreparationProgressTracker:
    def test_tracker_prints_progress(self, capsys):
        from customer_retention.stages.modeling.training_preparator import PreparationProgressTracker

        tracker = PreparationProgressTracker()
        tracker(1, 10, "step1", 0.5)
        tracker(5, 10, "step5", 1.2)
        tracker(10, 10, "step10", 0.3)
        captured = capsys.readouterr()
        assert "[1/10] step1" in captured.out
        assert "[5/10] step5" in captured.out
        assert "[10/10] step10" in captured.out
        # ETA is shown when step < total
        assert "ETA" in captured.out

    def test_print_preparation_progress(self, capsys):
        from customer_retention.stages.modeling.training_preparator import print_preparation_progress

        print_preparation_progress(3, 10, "classify", 2.5)
        captured = capsys.readouterr()
        assert "[3/10] classify: 2.5s" in captured.out


class _FakeSparkDFForPreparator:
    """Stub Spark DataFrame recording method calls for preparator tests."""

    def __init__(self, columns, *, schema_fields=None, agg_row=None, history=None):
        self._columns = list(columns)
        self._schema_fields = schema_fields or []
        self._agg_row = agg_row
        self.history = history if history is not None else []

    @property
    def columns(self):
        return list(self._columns)

    @property
    def schema(self):
        from unittest.mock import MagicMock
        s = MagicMock()
        s.fields = self._schema_fields
        return s

    def agg(self, *exprs):  # noqa: ARG002
        self.history.append(("agg",))
        return self

    def head(self):
        self.history.append(("head",))
        return self._agg_row

    def filter(self, expr):  # noqa: ARG002
        self.history.append(("filter",))
        return _FakeSparkDFForPreparator(
            self._columns, schema_fields=self._schema_fields,
            agg_row=self._agg_row, history=self.history,
        )

    def select(self, cols):
        self.history.append(("select", cols if isinstance(cols, list) else [cols]))
        kept = [c for c in self._columns if c in (cols if isinstance(cols, list) else [cols])]
        return _FakeSparkDFForPreparator(
            kept or self._columns,
            schema_fields=self._schema_fields,
            agg_row=self._agg_row,
            history=self.history,
        )

    def toDF(self, *aliases):  # noqa: N802
        self.history.append(("toDF", aliases))
        return _FakeSparkDFForPreparator(
            list(aliases),
            schema_fields=self._schema_fields,
            agg_row=self._agg_row,
            history=self.history,
        )

    @property
    def na(self):
        class _NaAccessor:
            def __init__(self, parent):
                self.parent = parent

            def fill(self, value):  # noqa: ARG002
                self.parent.history.append(("fillna",))
                return self.parent

        return _NaAccessor(self)

    def localCheckpoint(self, eager=True):  # noqa: ARG002, N802
        self.history.append(("localCheckpoint",))
        return self


class TestClassifyColumnsSpark:
    def test_classify_via_schema_excludes_datetime_types(self):
        pytest.importorskip("pyspark")
        from pyspark.sql.types import (
            DoubleType,
            IntegerType,
            StringType,
            StructField,
            TimestampNTZType,
        )

        from customer_retention.stages.modeling.training_preparator import TrainingPreparator

        fake = _FakeSparkDFForPreparator(
            ["feat_a", "feat_b", "feat_c", "feat_d"],
            schema_fields=[
                StructField("feat_a", DoubleType()),
                StructField("feat_b", IntegerType()),
                StructField("feat_c", StringType()),
                StructField("feat_d", TimestampNTZType()),
            ],
        )
        non_dt, str_cols = TrainingPreparator._classify_via_schema(
            fake, ["feat_a", "feat_b", "feat_c", "feat_d"],
        )
        assert "feat_d" not in non_dt  # datetime excluded
        assert "feat_a" in non_dt
        assert "feat_b" in non_dt
        assert "feat_c" in non_dt
        assert str_cols == ["feat_c"]  # only string cols

    def test_classify_via_schema_routes_from_dispatcher(self, preparator):
        from unittest.mock import MagicMock, patch

        fake_df = MagicMock()
        with patch("customer_retention.stages.modeling.training_preparator._is_spark_pandas", return_value=True), \
             patch("customer_retention.stages.modeling.training_preparator.as_spark_df", return_value=fake_df), \
             patch.object(TrainingPreparator := type(preparator), "_classify_via_schema", return_value=(["a"], [])) as mock_schema:
            result = preparator._classify_columns(fake_df, ["a", "b"])
        mock_schema.assert_called_once()
        assert result == (["a"], [])


class TestDropMissingTargetSpark:
    def test_drops_rows_with_null_target(self, preparator):
        from unittest.mock import MagicMock, patch

        fake_spark = MagicMock()
        fake_spark.agg.return_value.head.return_value = {"nulls": 10, "total": 100}
        fake_spark.filter.return_value = fake_spark
        fake_input = MagicMock()

        with patch("customer_retention.stages.modeling.training_preparator._is_spark_pandas", return_value=True), \
             patch("customer_retention.stages.modeling.training_preparator.as_spark_df", return_value=fake_spark), \
             patch("customer_retention.core.compat.spark_backend._as_pandas_api", return_value="filtered_ps"):
            result, nan_count = preparator._drop_missing_target(fake_input)

        assert nan_count == 10
        assert result == "filtered_ps"
        fake_spark.filter.assert_called_once()

    def test_raises_when_all_target_null(self, preparator):
        from unittest.mock import MagicMock, patch

        fake_spark = MagicMock()
        fake_spark.agg.return_value.head.return_value = {"nulls": 100, "total": 100}

        with patch("customer_retention.stages.modeling.training_preparator._is_spark_pandas", return_value=True), \
             patch("customer_retention.stages.modeling.training_preparator.as_spark_df", return_value=fake_spark):
            with pytest.raises(ValueError, match="all target values are NaN"):
                preparator._drop_missing_target(MagicMock())

    def test_no_filter_when_no_nulls(self, preparator):
        from unittest.mock import MagicMock, patch

        fake_spark = MagicMock()
        fake_spark.agg.return_value.head.return_value = {"nulls": 0, "total": 100}
        fake_input = MagicMock()

        with patch("customer_retention.stages.modeling.training_preparator._is_spark_pandas", return_value=True), \
             patch("customer_retention.stages.modeling.training_preparator.as_spark_df", return_value=fake_spark):
            result, nan_count = preparator._drop_missing_target(fake_input)

        assert nan_count == 0
        assert result is fake_input
        fake_spark.filter.assert_not_called()

    def test_nan_count_none_handled(self, preparator):
        from unittest.mock import MagicMock, patch

        fake_spark = MagicMock()
        # Simulate a SQL NULL: row["nulls"] is None
        fake_spark.agg.return_value.head.return_value = {"nulls": None, "total": 100}

        with patch("customer_retention.stages.modeling.training_preparator._is_spark_pandas", return_value=True), \
             patch("customer_retention.stages.modeling.training_preparator.as_spark_df", return_value=fake_spark):
            result, nan_count = preparator._drop_missing_target(MagicMock())

        assert nan_count == 0


class TestSampleEntitiesSpark:
    def test_returns_input_when_under_limit(self, feature_cols):
        from unittest.mock import MagicMock, patch

        from customer_retention.stages.modeling.training_preparator import TrainingPreparator

        prep = TrainingPreparator(
            target_column="target", feature_columns=feature_cols, max_rows=1000,
        )
        fake_spark = MagicMock()
        fake_spark.agg.return_value.head.return_value = {"total": 500, "n_entities": 50}
        fake_input = MagicMock()

        with patch("customer_retention.stages.modeling.training_preparator._is_spark_pandas", return_value=True), \
             patch("customer_retention.stages.modeling.training_preparator.as_spark_df", return_value=fake_spark):
            result = prep._sample_entities(fake_input)

        assert result is fake_input

    def test_samples_when_over_limit(self, feature_cols):
        from unittest.mock import MagicMock, patch

        from customer_retention.stages.modeling.training_preparator import TrainingPreparator

        prep = TrainingPreparator(
            target_column="target", feature_columns=feature_cols, max_rows=100,
        )
        # 500 rows / 100 entities → 5 rows per entity → target 20 entities
        fake_spark = MagicMock()
        fake_spark.agg.return_value.head.return_value = {"total": 500, "n_entities": 100}

        fake_input = MagicMock()
        fake_input.__getitem__ = MagicMock(return_value=MagicMock(drop_duplicates=MagicMock(return_value="entity_df")))
        fake_input.merge = MagicMock(return_value="merged_result")

        with patch("customer_retention.stages.modeling.training_preparator._is_spark_pandas", return_value=True), \
             patch("customer_retention.stages.modeling.training_preparator.as_spark_df", return_value=fake_spark), \
             patch("customer_retention.stages.modeling.training_preparator.safe_sample", return_value="sampled"):
            result = prep._sample_entities(fake_input)

        assert result == "merged_result"
        fake_input.merge.assert_called_once()

    def test_no_op_when_max_rows_none(self, preparator):
        from unittest.mock import MagicMock

        fake_input = MagicMock()
        result = preparator._sample_entities(fake_input)
        assert result is fake_input


class TestExtractTrainMetadataFromResult:
    def test_uses_train_metadata_when_available(self, preparator):
        from unittest.mock import MagicMock

        split_result = MagicMock()
        split_result.train_metadata = {
            "entity_id": "meta_entity_series",
            "as_of_date": "meta_date_series",
        }
        entities, dates = preparator._extract_train_metadata(split_result, MagicMock())
        assert entities == "meta_entity_series"
        assert dates == "meta_date_series"


class TestSparkFillnaAndDrop:
    def test_spark_fillna_drops_zero_variance_columns(self, preparator):
        from unittest.mock import MagicMock, patch

        fake_train_spark = MagicMock()
        fake_train_spark.columns = ["f1", "f2", "f3"]
        fake_train_spark.select.return_value.na.fill.return_value.localCheckpoint.return_value = "train_ckpt"
        fake_test_spark = MagicMock()
        fake_test_spark.columns = ["f1", "f2", "f3"]
        fake_test_spark.select.return_value.na.fill.return_value.localCheckpoint.return_value = "test_ckpt"

        call_sequence = [fake_train_spark, fake_test_spark]
        with patch("customer_retention.stages.modeling.training_preparator._is_spark_pandas", return_value=True), \
             patch("customer_retention.stages.modeling.training_preparator.as_spark_df", side_effect=lambda df: call_sequence.pop(0)), \
             patch.object(preparator, "_spark_nulls_and_zero_var", return_value=({"f1": 2, "f2": 0, "f3": 5}, ["f2"])), \
             patch("customer_retention.core.compat.spark_backend._as_pandas_api", side_effect=lambda x: f"ps_{x}"):
            X_train, X_test, zero_var, nulls = preparator._spark_fillna_and_drop(MagicMock(), MagicMock())

        assert zero_var == ["f2"]
        assert "f2" not in nulls  # Zero-variance col removed from nulls dict
        assert X_train == "ps_train_ckpt"
        assert X_test == "ps_test_ckpt"


class TestPandasNullsAndZeroVar:
    def test_returns_empty_when_no_numeric_cols(self):
        from customer_retention.stages.modeling.training_preparator import TrainingPreparator

        X_train = pd.DataFrame({"txt": ["a", "b", "c"]})
        X_test = pd.DataFrame({"txt": ["d", "e", "f"]})
        # Null counts dict built against training columns, zero_var list empty
        nulls, zero_var = TrainingPreparator._pandas_nulls_and_zero_var(X_train, X_test)
        assert "txt" in nulls
        assert zero_var == []

    def test_identifies_zero_variance_numeric_col(self):
        from customer_retention.stages.modeling.training_preparator import TrainingPreparator

        X_train = pd.DataFrame({"constant": [5.0, 5.0, 5.0], "varied": [1.0, 2.0, 3.0]})
        X_test = pd.DataFrame({"constant": [5.0], "varied": [2.5]})
        nulls, zero_var = TrainingPreparator._pandas_nulls_and_zero_var(X_train, X_test)
        assert "constant" in zero_var
        assert "varied" not in zero_var

    def test_identifies_nan_std_as_zero_variance(self):
        from customer_retention.stages.modeling.training_preparator import TrainingPreparator

        X_train = pd.DataFrame({"single": [5.0]})
        X_test = pd.DataFrame({"single": [5.0]})
        _, zero_var = TrainingPreparator._pandas_nulls_and_zero_var(X_train, X_test)
        assert "single" in zero_var


class TestSparkNullsAndZeroVar:
    def test_spark_batch_aggregation_collects_stats(self, preparator):
        pytest.importorskip("pyspark")
        from unittest.mock import MagicMock, patch

        from pyspark.sql.types import DoubleType, StringType, StructField

        fake_train = MagicMock()
        fake_train.columns = ["num1", "num2", "txt1"]
        fake_train.schema.fields = [
            StructField("num1", DoubleType()),
            StructField("num2", DoubleType()),
            StructField("txt1", StringType()),
        ]

        # toDF returns a work-df that supports .agg().head()
        work_df = MagicMock()
        # Train row: nulls + stddev per numeric col
        train_row = {
            "__a0____n": 3, "__a0____s": 1.5,  # num1: has variance
            "__a1____n": 2, "__a1____s": 0.0,  # num2: zero variance
            "__a2____n": 0,                    # txt1: null count only, no stddev
        }
        # Test row: just null counts
        test_row = {"__a0____n": 1, "__a1____n": 0, "__a2____n": 2}
        work_df.agg.return_value.head.side_effect = [train_row, test_row]
        fake_train.toDF.return_value = work_df

        fake_test = MagicMock()
        fake_test.columns = ["num1", "num2", "txt1"]
        fake_test.toDF.return_value = work_df

        call_sequence = [fake_train, fake_test]
        with patch("customer_retention.stages.modeling.training_preparator.as_spark_df", side_effect=lambda df: call_sequence.pop(0)):
            nulls, zero_var = preparator._spark_nulls_and_zero_var(MagicMock(), MagicMock())

        assert nulls["num1"] == 4  # 3 train + 1 test
        assert nulls["num2"] == 2
        assert nulls["txt1"] == 2
        assert "num2" in zero_var
        assert "num1" not in zero_var
        # txt1 is not numeric → never flagged as zero_var
        assert "txt1" not in zero_var


class TestDropMissingTargetRows:
    def test_pandas_filters_by_notna(self):
        from customer_retention.stages.modeling.training_preparator import _drop_missing_target_rows

        df = pd.DataFrame({"target": [1, np.nan, 0, np.nan, 1], "f": [1, 2, 3, 4, 5]})
        result = _drop_missing_target_rows(df, "target")
        assert len(result) == 3
        assert not result["target"].isna().any()

    def test_spark_path_uses_filter(self):
        from unittest.mock import MagicMock, patch

        from customer_retention.stages.modeling.training_preparator import _drop_missing_target_rows

        fake_spark_df = MagicMock()
        fake_filtered = MagicMock()
        fake_spark_df.filter.return_value = fake_filtered
        fake_input = MagicMock()

        with patch("customer_retention.stages.modeling.training_preparator._is_spark_pandas", return_value=True), \
             patch("customer_retention.stages.modeling.training_preparator.as_spark_df", return_value=fake_spark_df), \
             patch("customer_retention.core.compat.spark_backend._as_pandas_api", return_value="ps_filtered"):
            result = _drop_missing_target_rows(fake_input, "target")

        fake_spark_df.filter.assert_called_once()
        assert result == "ps_filtered"
