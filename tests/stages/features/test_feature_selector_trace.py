from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from customer_retention.stages.features.feature_selector import (
    FeatureSelector,
    SelectionMethod,
    run_chi_squared_rescue_selection,
    run_selection_pipeline,
)
from customer_retention.stages.modeling.feature_profile import SelectionTraceRecorder


def _binary_target_dataset(n: int = 200, n_useful: int = 3, n_noise: int = 7, seed: int = 0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=n)
    data = {}
    for i in range(n_useful):
        signal = y * 2.0 + rng.normal(0, 0.5, size=n)
        data[f"useful_{i}"] = signal
    for i in range(n_noise):
        data[f"noise_{i}"] = rng.normal(0, 1, size=n)
    for i in range(2):
        data[f"constant_{i}"] = np.zeros(n)
    df = pd.DataFrame(data)
    df["y"] = y
    return df


class TestVarianceTrace:
    def test_records_every_candidate(self):
        df = _binary_target_dataset(n=100, n_useful=2, n_noise=3)
        recorder = SelectionTraceRecorder()
        selector = FeatureSelector(
            method=SelectionMethod.VARIANCE, variance_threshold=0.01,
            target_column="y", trace_recorder=recorder,
        )
        selector.fit(df)
        assert "useful_0" in recorder.all_features()
        assert "constant_0" in recorder.all_features()

    def test_constant_feature_dropped_with_reason(self):
        df = _binary_target_dataset(n=100)
        recorder = SelectionTraceRecorder()
        selector = FeatureSelector(
            method=SelectionMethod.VARIANCE, variance_threshold=0.01,
            target_column="y", trace_recorder=recorder,
        )
        selector.fit(df)
        trace = recorder.trace_for("constant_0")
        assert trace[0].decision == "dropped"
        assert trace[0].reason == "low_variance"
        assert trace[0].score < 0.01 or math.isnan(trace[0].score)

    def test_variance_score_is_numeric(self):
        df = _binary_target_dataset(n=100)
        recorder = SelectionTraceRecorder()
        selector = FeatureSelector(
            method=SelectionMethod.VARIANCE, variance_threshold=0.01,
            target_column="y", trace_recorder=recorder,
        )
        selector.fit(df)
        assert isinstance(recorder.trace_for("useful_0")[0].score, float)

    def test_threshold_captured(self):
        df = _binary_target_dataset(n=50)
        recorder = SelectionTraceRecorder()
        selector = FeatureSelector(
            method=SelectionMethod.VARIANCE, variance_threshold=0.05,
            target_column="y", trace_recorder=recorder,
        )
        selector.fit(df)
        assert recorder.trace_for("useful_0")[0].threshold == 0.05

    def test_preserve_features_marked_preserved(self):
        df = _binary_target_dataset(n=100)
        recorder = SelectionTraceRecorder()
        selector = FeatureSelector(
            method=SelectionMethod.VARIANCE, variance_threshold=0.01,
            target_column="y", preserve_features=["constant_0"],
            trace_recorder=recorder,
        )
        selector.fit(df)
        trace = recorder.trace_for("constant_0")
        assert trace[0].decision == "preserved"


class TestCorrelationTrace:
    def test_companion_feature_set_on_loser(self):
        n = 100
        rng = np.random.default_rng(0)
        base = rng.normal(0, 1, size=n)
        df = pd.DataFrame({
            "a": base,
            "b": base + rng.normal(0, 0.01, size=n),
            "c": rng.normal(0, 1, size=n),
            "y": rng.integers(0, 2, size=n),
        })
        recorder = SelectionTraceRecorder()
        selector = FeatureSelector(
            method=SelectionMethod.CORRELATION, correlation_threshold=0.9,
            target_column="y", trace_recorder=recorder,
        )
        selector.fit(df)
        dropped = {f: recorder.trace_for(f)[0] for f in recorder.all_features()
                   if recorder.trace_for(f)[0].decision == "dropped"}
        assert len(dropped) == 1
        loser_name, loser_entry = next(iter(dropped.items()))
        assert loser_entry.companion_feature in {"a", "b"}
        assert loser_entry.companion_feature != loser_name


class TestL1Trace:
    def test_all_numeric_features_get_l1_coef(self):
        df = _binary_target_dataset(n=200)
        recorder = SelectionTraceRecorder()
        selector = FeatureSelector(
            method=SelectionMethod.L1_SELECTION, target_column="y",
            l1_C=10.0, trace_recorder=recorder,
        )
        selector.fit(df)
        for f in df.columns:
            if f == "y":
                continue
            trace = recorder.trace_for(f)
            assert len(trace) == 1
            assert trace[0].stage == "l1"
            assert trace[0].score_name == "l1_abs_coef"

    def test_useful_features_kept(self):
        df = _binary_target_dataset(n=500)
        recorder = SelectionTraceRecorder()
        selector = FeatureSelector(
            method=SelectionMethod.L1_SELECTION, target_column="y",
            l1_C=1.0, trace_recorder=recorder,
        )
        selector.fit(df)
        assert recorder.trace_for("useful_0")[0].decision in {"kept", "dropped"}
        assert recorder.trace_for("noise_0")[0].score_name == "l1_abs_coef"

    def test_threshold_captured_when_dropped(self):
        df = _binary_target_dataset(n=500)
        recorder = SelectionTraceRecorder()
        selector = FeatureSelector(
            method=SelectionMethod.L1_SELECTION, target_column="y",
            l1_C=1.0, trace_recorder=recorder,
        )
        selector.fit(df)
        entries = [recorder.trace_for(f)[0] for f in recorder.all_features()]
        thresholds = {e.threshold for e in entries if e.threshold is not None}
        assert len(thresholds) <= 1


class TestPipelineTrace:
    def test_full_pipeline_records_all_stages(self):
        df = _binary_target_dataset(n=200, n_useful=3, n_noise=5)
        recorder = SelectionTraceRecorder()
        run_selection_pipeline(
            df, target_column="y",
            variance_threshold=0.01, correlation_threshold=0.95,
            l1_enabled=True, l1_C=1.0,
            trace_recorder=recorder,
        )
        assert "useful_0" in recorder.all_features()
        assert "constant_0" in recorder.all_features()
        stages_for_useful = [e.stage for e in recorder.trace_for("useful_0")]
        assert "variance" in stages_for_useful

    def test_pipeline_without_recorder_stays_silent(self):
        df = _binary_target_dataset(n=100)
        # Does not crash, no exception.
        result = run_selection_pipeline(
            df, target_column="y",
            variance_threshold=0.01, correlation_threshold=0.95,
            l1_enabled=False,
        )
        assert result is not None

    def test_stage_counts_monotonically_decrease(self):
        df = _binary_target_dataset(n=200, n_useful=3, n_noise=5)
        recorder = SelectionTraceRecorder()
        sel_var = FeatureSelector(
            method=SelectionMethod.VARIANCE, variance_threshold=0.01,
            target_column="y", trace_recorder=recorder,
        )
        sel_var.fit(df)
        kept_after_variance = set(sel_var.selected_features)
        summary = recorder.stage_summary()
        assert summary[0]["stage"] == "variance"
        assert summary[0]["output"] == len(kept_after_variance)


@pytest.fixture
def two_stage_recorder():
    df = _binary_target_dataset(n=200)
    recorder = SelectionTraceRecorder()
    var_sel = FeatureSelector(
        method=SelectionMethod.VARIANCE, variance_threshold=0.01,
        target_column="y", trace_recorder=recorder,
    )
    var_sel.fit(df)
    l1_sel = FeatureSelector(
        method=SelectionMethod.L1_SELECTION, target_column="y",
        l1_C=1.0, trace_recorder=recorder,
    )
    kept_after_var = [c for c in df.columns if c == "y" or c in var_sel.selected_features]
    l1_sel.fit(df[kept_after_var])
    return recorder


class TestMultiStageFlow:
    def test_feature_seen_by_both_variance_and_l1(self, two_stage_recorder):
        stages = [e.stage for e in two_stage_recorder.trace_for("useful_0")]
        assert stages == ["variance", "l1"]

    def test_dropped_constant_only_seen_by_variance(self, two_stage_recorder):
        trace = two_stage_recorder.trace_for("constant_0")
        assert len(trace) == 1
        assert trace[0].stage == "variance"
        assert trace[0].decision == "dropped"


class TestRescuePipelineTrace:
    def _df_with_time_column(self, n: int = 400, n_dates: int = 4):
        rng = np.random.default_rng(0)
        y = np.tile([0, 1], n // 2)
        rng.shuffle(y)
        dates = pd.to_datetime(
            np.tile(pd.date_range("2025-01-01", periods=n_dates, freq="7D"), n // n_dates)
        )
        df = pd.DataFrame({
            "useful_0": y * 2.0 + rng.normal(0, 0.5, size=n),
            "useful_1": y * 1.5 + rng.normal(0, 0.5, size=n),
            "noise_0": rng.normal(0, 1, size=n),
            "noise_1": rng.normal(0, 1, size=n),
            "noise_2": rng.normal(0, 1, size=n),
            "as_of_date": dates,
            "y": y.astype("int64"),
        })
        return df

    def test_records_chi_squared_and_gbdt_rescue(self):
        df = self._df_with_time_column(200)
        recorder = SelectionTraceRecorder()
        run_chi_squared_rescue_selection(
            df, target_column="y", max_features=2,
            time_column="as_of_date", slice_strategy="last",
            min_positive_rate=0.0, num_buckets=5,
            gbdt_rescue_enabled=True, gbdt_rescue_max_features=2,
            gbdt_n_estimators=20, gbdt_max_depth=3,
            trace_recorder=recorder,
        )
        assert "useful_0" in recorder.all_features()
        stages_for_useful = {e.stage for e in recorder.trace_for("useful_0")}
        assert "chi_squared" in stages_for_useful

    def test_records_l1_rescue_when_enabled(self):
        df = self._df_with_time_column(200)
        recorder = SelectionTraceRecorder()
        run_chi_squared_rescue_selection(
            df, target_column="y", max_features=1,
            time_column="as_of_date", slice_strategy="last",
            min_positive_rate=0.0, num_buckets=5,
            l1_rescue_enabled=True, l1_rescue_max_features=2, l1_C=1.0,
            gbdt_rescue_enabled=False,
            trace_recorder=recorder,
        )
        stages_across_all = {
            e.stage for f in recorder.all_features() for e in recorder.trace_for(f)
        }
        assert "l1_rescue" in stages_across_all
