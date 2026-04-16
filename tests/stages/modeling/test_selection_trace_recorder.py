from __future__ import annotations

import math

import pytest

from customer_retention.stages.modeling.feature_profile import (
    ColumnProfile,
    FeatureProfile,
    SelectionTraceRecorder,
    StageDecision,
)


class TestRecordStage:
    def test_records_single_feature_kept(self):
        recorder = SelectionTraceRecorder()
        recorder.record_stage(
            stage="variance", score_name="variance",
            scores={"a": 0.5}, threshold=0.01,
            decisions={"a": "kept"}, reasons={"a": None},
            stage_input_count=1, stage_output_count=1,
        )
        trace = recorder.trace_for("a")
        assert len(trace) == 1
        assert trace[0].stage == "variance"
        assert trace[0].decision == "kept"
        assert trace[0].score == 0.5

    def test_assigns_ranks_by_score_desc(self):
        recorder = SelectionTraceRecorder()
        recorder.record_stage(
            stage="variance", score_name="variance",
            scores={"a": 0.1, "b": 0.5, "c": 0.3},
            threshold=0.01,
            decisions={"a": "dropped", "b": "kept", "c": "kept"},
            reasons={"a": "low_var", "b": None, "c": None},
            stage_input_count=3, stage_output_count=2,
        )
        assert recorder.trace_for("b")[0].rank == 1
        assert recorder.trace_for("c")[0].rank == 2
        assert recorder.trace_for("a")[0].rank == 3

    def test_nan_scores_get_no_rank(self):
        recorder = SelectionTraceRecorder()
        recorder.record_stage(
            stage="l1", score_name="l1_abs_coef",
            scores={"a": 0.5, "b": float("nan")},
            threshold=0.005,
            decisions={"a": "kept", "b": "not_evaluated"},
            reasons={"a": None, "b": "not_numeric"},
            stage_input_count=2, stage_output_count=1,
        )
        assert recorder.trace_for("a")[0].rank == 1
        assert recorder.trace_for("b")[0].rank is None
        assert math.isnan(recorder.trace_for("b")[0].score)

    def test_duplicate_stage_feature_raises(self):
        recorder = SelectionTraceRecorder()
        recorder.record_stage(
            stage="variance", score_name="variance",
            scores={"a": 0.5}, threshold=0.01,
            decisions={"a": "kept"}, reasons={"a": None},
            stage_input_count=1, stage_output_count=1,
        )
        with pytest.raises(ValueError, match="already recorded"):
            recorder.record_stage(
                stage="variance", score_name="variance",
                scores={"a": 0.4}, threshold=0.01,
                decisions={"a": "kept"}, reasons={"a": None},
                stage_input_count=1, stage_output_count=1,
            )

    def test_multiple_stages_accumulate_per_feature(self):
        recorder = SelectionTraceRecorder()
        recorder.record_stage(
            stage="variance", score_name="variance",
            scores={"a": 0.5}, threshold=0.01,
            decisions={"a": "kept"}, reasons={"a": None},
            stage_input_count=1, stage_output_count=1,
        )
        recorder.record_stage(
            stage="l1", score_name="l1_abs_coef",
            scores={"a": 0.8}, threshold=0.008,
            decisions={"a": "kept"}, reasons={"a": None},
            stage_input_count=1, stage_output_count=1,
        )
        trace = recorder.trace_for("a")
        assert [d.stage for d in trace] == ["variance", "l1"]

    def test_companion_map_captured(self):
        recorder = SelectionTraceRecorder()
        recorder.record_stage(
            stage="correlation", score_name="abs_pearson_r",
            scores={"loser": 0.97, "winner": 0.97},
            threshold=0.95,
            decisions={"loser": "dropped", "winner": "kept"},
            reasons={"loser": "high_correlation", "winner": None},
            stage_input_count=2, stage_output_count=1,
            companion_map={"loser": "winner"},
        )
        assert recorder.trace_for("loser")[0].companion_feature == "winner"
        assert recorder.trace_for("winner")[0].companion_feature is None

    def test_empty_scores_noop(self):
        recorder = SelectionTraceRecorder()
        recorder.record_stage(
            stage="variance", score_name="variance",
            scores={}, threshold=0.01,
            decisions={}, reasons={},
            stage_input_count=0, stage_output_count=0,
        )
        assert recorder.trace_for("a") == []


class TestApplyToProfile:
    def _recorder_with_a_kept_b_dropped(self):
        recorder = SelectionTraceRecorder()
        recorder.record_stage(
            stage="variance", score_name="variance",
            scores={"a": 0.5, "b": 0.005},
            threshold=0.01,
            decisions={"a": "kept", "b": "dropped"},
            reasons={"a": None, "b": "low_variance"},
            stage_input_count=2, stage_output_count=1,
        )
        return recorder

    def test_applies_to_kept_feature(self):
        recorder = self._recorder_with_a_kept_b_dropped()
        profile = FeatureProfile(
            stage="exploration", created_at="x", row_count=10, target_column="y",
            features={"a": ColumnProfile("double", 10, 0)},
            excluded={"b": "drop_zero_variance"},
        )
        recorder.apply_to_profile(profile)
        assert profile.features["a"].selection_trace[0].stage == "variance"

    def test_applies_to_dropped_feature_via_excluded_profiles(self):
        recorder = self._recorder_with_a_kept_b_dropped()
        profile = FeatureProfile(
            stage="exploration", created_at="x", row_count=10, target_column="y",
            features={"a": ColumnProfile("double", 10, 0)},
            excluded={"b": "drop_zero_variance"},
        )
        recorder.apply_to_profile(profile)
        assert "b" in profile.excluded_profiles
        assert profile.excluded_profiles["b"].selection_trace[0].decision == "dropped"

    def test_preserves_existing_excluded_profile_dtype(self):
        recorder = self._recorder_with_a_kept_b_dropped()
        profile = FeatureProfile(
            stage="exploration", created_at="x", row_count=10, target_column="y",
            features={"a": ColumnProfile("double", 10, 0)},
            excluded={"b": "drop_zero_variance"},
            excluded_profiles={"b": ColumnProfile("int64", 9, 1)},
        )
        recorder.apply_to_profile(profile)
        assert profile.excluded_profiles["b"].dtype == "int64"
        assert profile.excluded_profiles["b"].non_null_count == 9
        assert len(profile.excluded_profiles["b"].selection_trace) == 1

    def test_feature_absent_from_profile_is_ignored(self):
        recorder = SelectionTraceRecorder()
        recorder.record_stage(
            stage="variance", score_name="variance",
            scores={"orphan": 0.5}, threshold=0.01,
            decisions={"orphan": "kept"}, reasons={"orphan": None},
            stage_input_count=1, stage_output_count=1,
        )
        profile = FeatureProfile(
            stage="exploration", created_at="x", row_count=10, target_column="y",
            features={}, excluded={},
        )
        recorder.apply_to_profile(profile)
        assert "orphan" not in profile.features
        assert "orphan" not in profile.excluded_profiles

    def test_preserved_decision_recorded(self):
        recorder = SelectionTraceRecorder()
        recorder.record_stage(
            stage="l1", score_name="l1_abs_coef",
            scores={"p": 0.0}, threshold=0.005,
            decisions={"p": "preserved"},
            reasons={"p": "in_preserve_list"},
            stage_input_count=1, stage_output_count=1,
        )
        profile = FeatureProfile(
            stage="exploration", created_at="x", row_count=10, target_column="y",
            features={"p": ColumnProfile("double", 10, 0)},
        )
        recorder.apply_to_profile(profile)
        assert profile.features["p"].selection_trace[0].decision == "preserved"


class TestRecorderQueries:
    def test_all_features_seen(self):
        recorder = SelectionTraceRecorder()
        recorder.record_stage(
            stage="variance", score_name="variance",
            scores={"a": 0.5, "b": 0.1}, threshold=0.01,
            decisions={"a": "kept", "b": "kept"},
            reasons={"a": None, "b": None},
            stage_input_count=2, stage_output_count=2,
        )
        recorder.record_stage(
            stage="l1", score_name="l1_abs_coef",
            scores={"a": 0.8}, threshold=0.008,
            decisions={"a": "kept"}, reasons={"a": None},
            stage_input_count=1, stage_output_count=1,
        )
        assert recorder.all_features() == {"a", "b"}

    def test_stage_input_output_counts(self):
        recorder = SelectionTraceRecorder()
        recorder.record_stage(
            stage="variance", score_name="variance",
            scores={"a": 0.5}, threshold=0.01,
            decisions={"a": "kept"}, reasons={"a": None},
            stage_input_count=10, stage_output_count=7,
        )
        entry = recorder.trace_for("a")[0]
        assert entry.stage_input_count == 10
        assert entry.stage_output_count == 7

    def test_stage_summary(self):
        recorder = SelectionTraceRecorder()
        recorder.record_stage(
            stage="variance", score_name="variance",
            scores={"a": 0.5, "b": 0.005}, threshold=0.01,
            decisions={"a": "kept", "b": "dropped"},
            reasons={"a": None, "b": "low_variance"},
            stage_input_count=2, stage_output_count=1,
        )
        summary = recorder.stage_summary()
        assert summary[0]["stage"] == "variance"
        assert summary[0]["input"] == 2
        assert summary[0]["output"] == 1
        assert summary[0]["dropped"] == 1

    def test_record_single_accepts_stagedecision(self):
        recorder = SelectionTraceRecorder()
        decision = StageDecision(
            stage="l1", score=0.9, score_name="l1_abs_coef",
            threshold=0.009, decision="kept", reason=None, rank=1,
            stage_input_count=1, stage_output_count=1,
        )
        recorder.record_single("a", decision)
        assert recorder.trace_for("a") == [decision]
