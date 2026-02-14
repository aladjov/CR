from __future__ import annotations

from unittest.mock import patch

import pytest

from customer_retention.analysis.auto_explorer.objective_support_communicator import (
    ObjectiveSupportCommunicator,
    ObjectiveSynthesis,
    SectionRecord,
    SignalRule,
    apply_signal_rules,
    collect_indicators,
    signal_bar,
)


class TestSignalBar:
    def test_zero(self):
        assert signal_bar(0) == "░░░"

    def test_one(self):
        assert signal_bar(1) == "█░░"

    def test_two(self):
        assert signal_bar(2) == "██░"

    def test_three(self):
        assert signal_bar(3) == "███"

    def test_negative_clamped(self):
        assert signal_bar(-1) == "░░░"

    def test_above_three_clamped(self):
        assert signal_bar(5) == "███"


class TestFormatObjectiveLabel:
    def test_camel_case(self):
        comm = ObjectiveSupportCommunicator()
        assert comm._format_objective_label("ImmediateRisk") == "Immediate risk"

    def test_single_word(self):
        comm = ObjectiveSupportCommunicator()
        assert comm._format_objective_label("Renewal") == "Renewal"

    def test_multi_word(self):
        comm = ObjectiveSupportCommunicator()
        assert comm._format_objective_label("LongTermDisengagement") == "Long term disengagement"


class TestRankPhrases:
    def test_empty(self):
        comm = ObjectiveSupportCommunicator()
        assert comm._rank_phrases([]) == []

    def test_dedup_case_insensitive(self):
        comm = ObjectiveSupportCommunicator()
        result = comm._rank_phrases(["High density", "high density", "HIGH DENSITY"])
        assert len(result) == 1
        assert result[0] == ("high density", 3)

    def test_ranked_by_count(self):
        comm = ObjectiveSupportCommunicator()
        result = comm._rank_phrases(["alpha", "beta", "alpha", "gamma", "beta", "alpha"])
        assert result[0] == ("alpha", 3)
        assert result[1] == ("beta", 2)
        assert result[2] == ("gamma", 1)

    def test_preserves_lowercase(self):
        comm = ObjectiveSupportCommunicator()
        result = comm._rank_phrases(["Some Phrase"])
        assert result[0][0] == "some phrase"

    def test_multiple_same_count_stable(self):
        comm = ObjectiveSupportCommunicator()
        result = comm._rank_phrases(["aaa", "bbb", "ccc"])
        counts = [c for _, c in result]
        assert counts == [1, 1, 1]


class TestComputeConfidence:
    def test_no_signals_returns_zero(self):
        comm = ObjectiveSupportCommunicator()
        assert comm._compute_confidence([]) == 0

    def test_identical_signals_high_confidence(self):
        comm = ObjectiveSupportCommunicator()
        assert comm._compute_confidence([3, 3, 3]) == 3

    def test_low_variance(self):
        comm = ObjectiveSupportCommunicator()
        assert comm._compute_confidence([2, 2, 3]) == 2

    def test_medium_variance(self):
        comm = ObjectiveSupportCommunicator()
        assert comm._compute_confidence([0, 2, 2]) == 1

    def test_high_variance(self):
        comm = ObjectiveSupportCommunicator()
        assert comm._compute_confidence([0, 3, 0, 3]) == 0


class TestRecordSection:
    def test_basic_recording(self):
        comm = ObjectiveSupportCommunicator()
        comm.record_section(
            "section_1",
            signals={"ImmediateRisk": 3},
            why={"ImmediateRisk": ["reason"]},
            positives=["good"], negatives=["bad"], gaps=["gap"],
        )
        assert "section_1" in comm._sections

    def test_signal_out_of_range_raises(self):
        comm = ObjectiveSupportCommunicator()
        with pytest.raises(ValueError):
            comm.record_section("s1", signals={"ImmediateRisk": 4}, why={})

    def test_negative_signal_raises(self):
        comm = ObjectiveSupportCommunicator()
        with pytest.raises(ValueError):
            comm.record_section("s1", signals={"ImmediateRisk": -1}, why={})

    def test_unknown_objective_in_signals_raises(self):
        comm = ObjectiveSupportCommunicator()
        with pytest.raises(ValueError):
            comm.record_section("s1", signals={"Unknown": 2}, why={})

    def test_unknown_objective_in_why_raises(self):
        comm = ObjectiveSupportCommunicator()
        with pytest.raises(ValueError):
            comm.record_section("s1", signals={"ImmediateRisk": 2}, why={"Unknown": ["x"]})

    def test_why_capped_at_five_per_objective(self):
        comm = ObjectiveSupportCommunicator()
        comm.record_section(
            "s1", signals={"ImmediateRisk": 1},
            why={"ImmediateRisk": ["a", "b", "c", "d", "e", "f", "g"]},
        )
        assert len(comm._sections["s1"].why["ImmediateRisk"]) == 5

    def test_defaults_for_optional_lists(self):
        comm = ObjectiveSupportCommunicator()
        comm.record_section("s1", signals={"ImmediateRisk": 1}, why={"ImmediateRisk": ["reason"]})
        rec = comm._sections["s1"]
        assert rec.positives == []
        assert rec.negatives == []
        assert rec.gaps == []

    def test_multiple_objectives(self):
        comm = ObjectiveSupportCommunicator()
        comm.record_section(
            "s1",
            signals={"ImmediateRisk": 3, "Disengagement": 2, "Renewal": 1},
            why={"ImmediateRisk": ["reason a"], "Disengagement": ["reason b"]},
        )
        rec = comm._sections["s1"]
        assert rec.signals["ImmediateRisk"] == 3
        assert rec.signals["Disengagement"] == 2
        assert rec.signals["Renewal"] == 1
        assert rec.why["ImmediateRisk"] == ["reason a"]
        assert rec.why["Disengagement"] == ["reason b"]

    def test_duplicate_section_overwrites(self):
        comm = ObjectiveSupportCommunicator()
        comm.record_section("s1", signals={"ImmediateRisk": 1}, why={"ImmediateRisk": ["first"]})
        comm.record_section("s1", signals={"ImmediateRisk": 3}, why={"ImmediateRisk": ["second"]})
        assert comm._sections["s1"].signals["ImmediateRisk"] == 3
        assert comm._sections["s1"].why["ImmediateRisk"] == ["second"]

    def test_empty_why_dict_accepted(self):
        comm = ObjectiveSupportCommunicator()
        comm.record_section("s1", signals={"ImmediateRisk": 2}, why={})
        assert comm._sections["s1"].why == {}


class TestRenderSection:
    def test_renders_last_section_by_default(self):
        comm = ObjectiveSupportCommunicator()
        comm.record_section("s1", signals={"ImmediateRisk": 1}, why={"ImmediateRisk": ["a"]})
        comm.record_section("s2", signals={"ImmediateRisk": 3}, why={"ImmediateRisk": ["b"]})
        result = comm.render_section()
        assert "███" in result
        assert "b" in result

    def test_renders_specific_section(self):
        comm = ObjectiveSupportCommunicator()
        comm.record_section("s1", signals={"ImmediateRisk": 1}, why={"ImmediateRisk": ["a"]})
        comm.record_section("s2", signals={"ImmediateRisk": 3}, why={"ImmediateRisk": ["b"]})
        result = comm.render_section("s1")
        assert "█░░" in result

    def test_unknown_section_raises(self):
        comm = ObjectiveSupportCommunicator()
        with pytest.raises(KeyError):
            comm.render_section("missing")

    def test_header_present(self):
        comm = ObjectiveSupportCommunicator()
        comm.record_section("s1", signals={"ImmediateRisk": 2}, why={"ImmediateRisk": ["x"]})
        result = comm.render_section("s1")
        assert "OBJECTIVES IMPACT" in result

    def test_why_lines_under_objective(self):
        comm = ObjectiveSupportCommunicator()
        comm.record_section(
            "s1",
            signals={"ImmediateRisk": 2, "Renewal": 1},
            why={"ImmediateRisk": ["reason one"], "Renewal": ["reason two"]},
            positives=["good thing"], negatives=["bad thing"],
        )
        result = comm.render_section("s1")
        lines = result.split("\n")
        ir_idx = next(i for i, line in enumerate(lines) if "Immediate risk" in line)
        re_idx = next(i for i, line in enumerate(lines) if "Renewal" in line)
        assert lines[ir_idx + 1] == "  + reason one"
        assert lines[re_idx + 1] == "  - reason two"
        assert "  + good thing" in result
        assert "  - bad thing" in result

    def test_why_prefix_matches_signal_strength(self):
        comm = ObjectiveSupportCommunicator()
        comm.record_section(
            "s1",
            signals={"ImmediateRisk": 3, "Renewal": 0, "Disengagement": 2},
            why={
                "ImmediateRisk": ["strong support"],
                "Renewal": ["no data"],
                "Disengagement": ["moderate support"],
            },
        )
        result = comm.render_section("s1")
        assert "  + strong support" in result
        assert "  - no data" in result
        assert "  + moderate support" in result

    def test_gaps_formatted(self):
        comm = ObjectiveSupportCommunicator()
        comm.record_section("s1", signals={"ImmediateRisk": 1}, why={}, gaps=["missing data"])
        result = comm.render_section("s1")
        assert "Gaps:" in result
        assert "  ? missing data" in result

    def test_no_gaps_section_omitted(self):
        comm = ObjectiveSupportCommunicator()
        comm.record_section("s1", signals={"ImmediateRisk": 1}, why={})
        result = comm.render_section("s1")
        assert "Gaps:" not in result

    def test_label_before_bar_format(self):
        comm = ObjectiveSupportCommunicator()
        comm.record_section(
            "s1",
            signals={"ImmediateRisk": 3, "Disengagement": 2, "Renewal": 1},
            why={},
        )
        result = comm.render_section("s1")
        assert "Immediate risk: (███)" in result
        assert "Renewal: (█░░)" in result
        assert "Disengagement: (██░)" in result

    def test_empty_sections_raises(self):
        comm = ObjectiveSupportCommunicator()
        with pytest.raises(KeyError):
            comm.render_section()

    def test_auto_display_ipython(self):
        comm = ObjectiveSupportCommunicator()
        comm.record_section("s1", signals={"ImmediateRisk": 2}, why={})
        with patch("customer_retention.analysis.auto_explorer.objective_support_communicator._display_output") as mock_display:
            comm.render_section("s1")
            mock_display.assert_called_once()


class TestAggregate:
    def test_single_section(self):
        comm = ObjectiveSupportCommunicator()
        comm.record_section(
            "s1", signals={"ImmediateRisk": 3},
            why={"ImmediateRisk": ["reason"]},
            positives=["good"], negatives=["bad"], gaps=["gap"],
        )
        result = comm.aggregate()
        assert "ImmediateRisk" in result
        synth = result["ImmediateRisk"]
        assert synth.combined_strength == 3
        assert len(synth.drivers) >= 1
        assert len(synth.frictions) >= 1
        assert "gap" in synth.gaps

    def test_multiple_sections_averaging(self):
        comm = ObjectiveSupportCommunicator()
        comm.record_section("s1", signals={"ImmediateRisk": 2}, why={})
        comm.record_section("s2", signals={"ImmediateRisk": 3}, why={})
        result = comm.aggregate()
        assert result["ImmediateRisk"].combined_strength == 3  # mean 2.5 rounds up

    def test_mean_rounding(self):
        comm = ObjectiveSupportCommunicator()
        comm.record_section("s1", signals={"ImmediateRisk": 1}, why={})
        comm.record_section("s2", signals={"ImmediateRisk": 2}, why={})
        result = comm.aggregate()
        assert result["ImmediateRisk"].combined_strength == 2  # mean 1.5 rounds up

    def test_positives_attributed_to_active_objectives(self):
        comm = ObjectiveSupportCommunicator()
        comm.record_section(
            "s1",
            signals={"ImmediateRisk": 2, "Disengagement": 0},
            why={}, positives=["good thing"],
        )
        result = comm.aggregate()
        ir_drivers = [d[0] for d in result["ImmediateRisk"].drivers]
        dis_drivers = [d[0] for d in result["Disengagement"].drivers]
        assert "good thing" in ir_drivers
        assert "good thing" not in dis_drivers

    def test_gaps_deduplicated(self):
        comm = ObjectiveSupportCommunicator()
        comm.record_section("s1", signals={"ImmediateRisk": 1}, why={}, gaps=["gap1"])
        comm.record_section("s2", signals={"ImmediateRisk": 2}, why={}, gaps=["gap1", "gap2"])
        result = comm.aggregate()
        assert len(result["ImmediateRisk"].gaps) == 2

    def test_objectives_without_signals_get_zero(self):
        comm = ObjectiveSupportCommunicator()
        comm.record_section("s1", signals={"ImmediateRisk": 3}, why={})
        result = comm.aggregate()
        assert result["Disengagement"].combined_strength == 0
        assert result["Renewal"].combined_strength == 0

    def test_confidence_included(self):
        comm = ObjectiveSupportCommunicator()
        comm.record_section("s1", signals={"ImmediateRisk": 3}, why={})
        comm.record_section("s2", signals={"ImmediateRisk": 3}, why={})
        result = comm.aggregate()
        assert result["ImmediateRisk"].confidence == 3

    def test_combined_strength_clamped(self):
        comm = ObjectiveSupportCommunicator()
        comm.record_section("s1", signals={"ImmediateRisk": 3}, why={})
        result = comm.aggregate()
        assert 0 <= result["ImmediateRisk"].combined_strength <= 3


class TestRenderSummary:
    def test_header_present(self):
        comm = ObjectiveSupportCommunicator()
        comm.record_section("s1", signals={"ImmediateRisk": 2}, why={})
        result = comm.render_summary()
        assert "OBJECTIVES SYNTHESIS" in result

    def test_all_objectives_shown(self):
        comm = ObjectiveSupportCommunicator()
        comm.record_section(
            "s1",
            signals={"ImmediateRisk": 3, "Disengagement": 2, "Renewal": 1},
            why={},
        )
        result = comm.render_summary()
        assert "Immediate risk" in result
        assert "Disengagement" in result
        assert "Renewal" in result

    def test_drivers_shown(self):
        comm = ObjectiveSupportCommunicator()
        comm.record_section(
            "s1", signals={"ImmediateRisk": 3},
            why={}, positives=["recency effect"],
        )
        result = comm.render_summary()
        assert "recency effect" in result

    def test_frictions_shown(self):
        comm = ObjectiveSupportCommunicator()
        comm.record_section(
            "s1", signals={"ImmediateRisk": 3},
            why={}, negatives=["segment instability"],
        )
        result = comm.render_summary()
        assert "segment instability" in result

    def test_gaps_section_shown(self):
        comm = ObjectiveSupportCommunicator()
        comm.record_section(
            "s1", signals={"ImmediateRisk": 1},
            why={}, gaps=["no contract data"],
        )
        result = comm.render_summary()
        assert "Gaps:" in result
        assert "  ? no contract data" in result

    def test_confidence_section_shown(self):
        comm = ObjectiveSupportCommunicator()
        comm.record_section("s1", signals={"ImmediateRisk": 3}, why={})
        result = comm.render_summary()
        assert "Confidence:" in result

    def test_empty_sections_raises(self):
        comm = ObjectiveSupportCommunicator()
        with pytest.raises(ValueError):
            comm.render_summary()

    def test_auto_display_ipython(self):
        comm = ObjectiveSupportCommunicator()
        comm.record_section("s1", signals={"ImmediateRisk": 2}, why={})
        with patch("customer_retention.analysis.auto_explorer.objective_support_communicator._display_output") as mock_display:
            comm.render_summary()
            mock_display.assert_called_once()

    def test_label_before_bar_in_summary(self):
        comm = ObjectiveSupportCommunicator()
        comm.record_section("s1", signals={"ImmediateRisk": 3}, why={})
        result = comm.render_summary()
        assert "Immediate risk: (███)" in result


class TestCustomObjectives:
    def test_custom_objectives_accepted(self):
        comm = ObjectiveSupportCommunicator(objectives=("Churn", "Upsell"))
        comm.record_section("s1", signals={"Churn": 2}, why={"Churn": ["x"]})
        assert "s1" in comm._sections

    def test_custom_objectives_reject_unknown(self):
        comm = ObjectiveSupportCommunicator(objectives=("Churn", "Upsell"))
        with pytest.raises(ValueError):
            comm.record_section("s1", signals={"ImmediateRisk": 2}, why={})

    def test_custom_objectives_in_aggregate(self):
        comm = ObjectiveSupportCommunicator(objectives=("Churn", "Upsell"))
        comm.record_section("s1", signals={"Churn": 3, "Upsell": 1}, why={"Churn": ["x"]})
        result = comm.aggregate()
        assert "Churn" in result
        assert "Upsell" in result


class TestDataModelDefaults:
    def test_section_record_defaults(self):
        rec = SectionRecord(section_id="s1", signals={"ImmediateRisk": 1}, why={"ImmediateRisk": ["x"]})
        assert rec.positives == []
        assert rec.negatives == []
        assert rec.gaps == []

    def test_objective_synthesis_defaults(self):
        synth = ObjectiveSynthesis(combined_strength=2, confidence=1)
        assert synth.drivers == []
        assert synth.frictions == []
        assert synth.gaps == []


class TestSignalAlias:
    def test_signal_calls_record_section(self):
        comm = ObjectiveSupportCommunicator()
        comm.signal("s1", signals={"ImmediateRisk": 2}, why={"ImmediateRisk": ["via alias"]})
        assert "s1" in comm._sections
        assert comm._sections["s1"].signals["ImmediateRisk"] == 2


class TestDriverCountDisplay:
    def test_driver_count_in_summary(self):
        comm = ObjectiveSupportCommunicator()
        comm.record_section("s1", signals={"ImmediateRisk": 3}, why={}, positives=["recency effect"])
        comm.record_section("s2", signals={"ImmediateRisk": 2}, why={}, positives=["recency effect"])
        comm.record_section("s3", signals={"ImmediateRisk": 3}, why={}, positives=["recency effect"])
        result = comm.render_summary()
        assert "(3x)" in result

    def test_single_count_no_suffix(self):
        comm = ObjectiveSupportCommunicator()
        comm.record_section("s1", signals={"ImmediateRisk": 3}, why={}, positives=["unique thing"])
        result = comm.render_summary()
        assert "(1x)" not in result


class TestSectionOrder:
    def test_section_order_preserved(self):
        comm = ObjectiveSupportCommunicator()
        comm.record_section("alpha", signals={"ImmediateRisk": 1}, why={})
        comm.record_section("beta", signals={"ImmediateRisk": 2}, why={})
        comm.record_section("gamma", signals={"ImmediateRisk": 3}, why={})
        assert list(comm._sections.keys()) == ["alpha", "beta", "gamma"]

    def test_last_section_is_last_recorded(self):
        comm = ObjectiveSupportCommunicator()
        comm.record_section("first", signals={"ImmediateRisk": 1}, why={})
        comm.record_section("second", signals={"ImmediateRisk": 3}, why={})
        result = comm.render_section()
        assert "███" in result


class TestObjectiveOrder:
    def test_default_order_is_immediate_renewal_disengagement(self):
        comm = ObjectiveSupportCommunicator()
        comm.record_section(
            "s1",
            signals={"ImmediateRisk": 3, "Renewal": 2, "Disengagement": 1},
            why={},
        )
        result = comm.render_section("s1")
        lines = result.split("\n")
        obj_lines = [line for line in lines if "(" in line and ")" in line and not line.startswith(" ")]
        assert "Immediate risk" in obj_lines[0]
        assert "Renewal" in obj_lines[1]
        assert "Disengagement" in obj_lines[2]


class TestApplySignalRules:
    def test_no_rules_returns_base(self):
        signal, whys = apply_signal_rules(3, [])
        assert signal == 3
        assert whys == []

    def test_no_rules_with_default_why(self):
        signal, whys = apply_signal_rules(3, [], "no drift")
        assert signal == 3
        assert whys == ["no drift"]

    def test_active_decrement(self):
        signal, whys = apply_signal_rules(3, [SignalRule(True, decrement=1, why="volume drift")])
        assert signal == 2
        assert whys == ["volume drift"]

    def test_inactive_rule_ignored(self):
        signal, whys = apply_signal_rules(3, [SignalRule(False, decrement=1, why="volume drift")])
        assert signal == 3
        assert whys == []

    def test_cap_limits_signal(self):
        signal, whys = apply_signal_rules(3, [SignalRule(True, cap=1, why="short history")])
        assert signal == 1
        assert whys == ["short history"]

    def test_multiple_decrements_floor_at_zero(self):
        rules = [
            SignalRule(True, decrement=2, why="major drift"),
            SignalRule(True, decrement=2, why="another drift"),
        ]
        signal, whys = apply_signal_rules(3, rules)
        assert signal == 0
        assert whys == ["major drift", "another drift"]

    def test_decrement_and_cap_combined(self):
        rules = [
            SignalRule(True, decrement=1, why="drift"),
            SignalRule(True, cap=1, why="cap"),
        ]
        signal, whys = apply_signal_rules(3, rules)
        assert signal == 1
        assert whys == ["drift", "cap"]

    def test_default_why_only_when_no_rules_fire(self):
        signal, whys = apply_signal_rules(3, [SignalRule(True, why="fired")], "default")
        assert whys == ["fired"]

    def test_signal_clamped_to_three(self):
        signal, _ = apply_signal_rules(5, [])
        assert signal == 3

    def test_rule_without_why_adds_no_text(self):
        signal, whys = apply_signal_rules(3, [SignalRule(True, decrement=1)])
        assert signal == 2
        assert whys == []

    def test_mixed_active_inactive(self):
        rules = [
            SignalRule(True, decrement=1, why="active one"),
            SignalRule(False, decrement=1, why="inactive"),
            SignalRule(True, cap=2, why="active two"),
        ]
        signal, whys = apply_signal_rules(3, rules)
        assert signal == 2
        assert whys == ["active one", "active two"]

    def test_why_only_rule_no_signal_change(self):
        signal, whys = apply_signal_rules(2, [SignalRule(True, why="note")])
        assert signal == 2
        assert whys == ["note"]


class TestCollectIndicators:
    def test_empty_list(self):
        positives, negatives = collect_indicators([])
        assert positives == []
        assert negatives == []

    def test_active_gives_negative(self):
        positives, negatives = collect_indicators([(True, "good", "bad")])
        assert positives == []
        assert negatives == ["bad"]

    def test_inactive_gives_positive(self):
        positives, negatives = collect_indicators([(False, "good", "bad")])
        assert positives == ["good"]
        assert negatives == []

    def test_empty_strings_excluded(self):
        positives, negatives = collect_indicators([(True, "", "bad"), (False, "", "")])
        assert positives == []
        assert negatives == ["bad"]

    def test_mixed_indicators(self):
        indicators = [
            (True, "stable cadence", "cadence change"),
            (False, "consistent entities", "entity drift"),
            (False, "low missingness", "missingness drift"),
        ]
        positives, negatives = collect_indicators(indicators)
        assert positives == ["consistent entities", "low missingness"]
        assert negatives == ["cadence change"]

    def test_all_active(self):
        indicators = [(True, "a", "b"), (True, "c", "d")]
        positives, negatives = collect_indicators(indicators)
        assert positives == []
        assert negatives == ["b", "d"]

    def test_all_inactive(self):
        indicators = [(False, "a", "b"), (False, "c", "d")]
        positives, negatives = collect_indicators(indicators)
        assert positives == ["a", "c"]
        assert negatives == []
