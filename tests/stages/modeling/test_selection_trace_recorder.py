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


class TestRecordNB05Drops:
    """NB05 drops happen BEFORE the selection pipeline runs. Recording them as
    stage='nb05' entries makes the trace show the full funnel end-to-end instead
    of starting at variance/correlation/L1.
    """

    def test_records_drop_as_nb05_stage(self):
        recorder = SelectionTraceRecorder()
        recorder.record_nb05_drops(
            {"feat_a": "drop_weak", "feat_b": "drop_multicollinear"},
            total_pre_nb05_features=10,
            total_post_nb05_features=8,
        )
        trace_a = recorder.trace_for("feat_a")
        assert len(trace_a) == 1
        assert trace_a[0].stage == "nb05"
        assert trace_a[0].decision == "dropped"
        assert trace_a[0].reason == "drop_weak"

    def test_records_multiple_drop_reasons(self):
        recorder = SelectionTraceRecorder()
        recorder.record_nb05_drops(
            {"feat_a": "drop_weak", "feat_b": "drop_multicollinear"},
            total_pre_nb05_features=10,
            total_post_nb05_features=8,
        )
        reasons = {f: recorder.trace_for(f)[0].reason for f in ("feat_a", "feat_b")}
        assert reasons == {"feat_a": "drop_weak", "feat_b": "drop_multicollinear"}

    def test_input_output_counts_captured(self):
        recorder = SelectionTraceRecorder()
        recorder.record_nb05_drops(
            {"feat_a": "drop_weak"},
            total_pre_nb05_features=100,
            total_post_nb05_features=99,
        )
        entry = recorder.trace_for("feat_a")[0]
        assert entry.stage_input_count == 100
        assert entry.stage_output_count == 99

    def test_empty_drops_is_noop(self):
        recorder = SelectionTraceRecorder()
        recorder.record_nb05_drops(
            {},
            total_pre_nb05_features=100,
            total_post_nb05_features=100,
        )
        assert recorder.all_features() == set()

    def test_nb05_trace_precedes_variance_trace(self):
        """Multi-stage: NB05 drop recorded, then feature enters variance stage."""
        recorder = SelectionTraceRecorder()
        recorder.record_nb05_drops(
            {"dropped_early": "drop_weak"},
            total_pre_nb05_features=2,
            total_post_nb05_features=1,
        )
        recorder.record_stage(
            stage="variance", score_name="variance",
            scores={"survivor": 0.5}, threshold=0.01,
            decisions={"survivor": "kept"}, reasons={"survivor": None},
            stage_input_count=1, stage_output_count=1,
        )
        early = recorder.trace_for("dropped_early")
        survivor = recorder.trace_for("survivor")
        assert len(early) == 1 and early[0].stage == "nb05"
        assert len(survivor) == 1 and survivor[0].stage == "variance"

    def test_apply_to_profile_nb05_drops_go_to_excluded_profiles(self):
        from customer_retention.stages.modeling.feature_profile import FeatureProfile
        recorder = SelectionTraceRecorder()
        recorder.record_nb05_drops(
            {"early_drop": "drop_weak"},
            total_pre_nb05_features=2,
            total_post_nb05_features=1,
        )
        profile = FeatureProfile(
            stage="exploration", created_at="x", row_count=10, target_column="y",
            features={},
            excluded={"early_drop": "drop_weak"},
        )
        recorder.apply_to_profile(profile)
        assert "early_drop" in profile.excluded_profiles
        assert profile.excluded_profiles["early_drop"].selection_trace[0].stage == "nb05"

    def test_stage_summary_includes_nb05(self):
        recorder = SelectionTraceRecorder()
        recorder.record_nb05_drops(
            {"a": "drop_weak", "b": "drop_multicollinear"},
            total_pre_nb05_features=100,
            total_post_nb05_features=98,
        )
        summary = recorder.stage_summary()
        nb05_summaries = [s for s in summary if s["stage"] == "nb05"]
        assert len(nb05_summaries) == 1
        assert nb05_summaries[0]["input"] == 100
        assert nb05_summaries[0]["output"] == 98
        assert nb05_summaries[0]["dropped"] == 2
