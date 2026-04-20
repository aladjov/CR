"""Unit tests for the deterministic feature-overlap mapper."""

from __future__ import annotations

from typing import Any, List

from customer_retention.stages.causal.llm_namer import (
    ArchetypeContext,
    ArchetypeNaming,
    PlaybookFitDecision,
)
from customer_retention.stages.causal.playbook_mapper import (
    ArchetypeSummary,
    extract_features_from_text,
    map_archetypes_to_playbooks,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


GOLD_FEATURES: List[str] = [
    "tenure_days",
    "nps_score",
    "support_tickets_30d",
    "monthly_revenue",
    "logins_30d",
    "discount_used",
]


def _make_archetype(idx: int, top_positive: List[str], top_negative: List[str] | None = None) -> ArchetypeSummary:
    return ArchetypeSummary(
        cluster_index=idx,
        cluster_size=100 + idx,
        cluster_mean_churn_probability=0.4 + 0.05 * idx,
        top_positive_drivers=[
            {"feature": f, "mean_shap": 0.1, "mean_value": 1.0} for f in top_positive
        ],
        top_negative_drivers=[
            {"feature": f, "mean_shap": -0.05, "mean_value": 0.5}
            for f in (top_negative or [])
        ],
    )


def _make_playbook(playbook_id: str, description: str, **extra: Any) -> dict:
    base = {
        "playbook_id": playbook_id,
        "version": "1.0.0",
        "name": playbook_id.replace("_", " ").title(),
        "description": description,
        "expected_uplift_pct_default": 0.05,
    }
    base.update(extra)
    return base


# ---------------------------------------------------------------------------
# extract_features_from_text
# ---------------------------------------------------------------------------


class TestExtractFeaturesFromText:
    def test_extracts_matching_tokens(self):
        text = "Targets customers with low nps_score and high support_tickets_30d"
        out = extract_features_from_text(text, GOLD_FEATURES)
        assert "nps_score" in out
        assert "support_tickets_30d" in out

    def test_ignores_non_feature_words(self):
        text = "Drives engagement and retention via tenure_days outreach"
        out = extract_features_from_text(text, GOLD_FEATURES)
        assert out == ["tenure_days"]

    def test_dedupes_repeated_features(self):
        text = "tenure_days affects tenure_days strongly"
        assert extract_features_from_text(text, GOLD_FEATURES) == ["tenure_days"]

    def test_empty_text_returns_empty(self):
        assert extract_features_from_text("", GOLD_FEATURES) == []

    def test_empty_feature_list_returns_empty(self):
        assert extract_features_from_text("tenure_days", []) == []


# ---------------------------------------------------------------------------
# map_archetypes_to_playbooks (template namer path)
# ---------------------------------------------------------------------------


class TestMapArchetypesToPlaybooksTemplatePath:
    def test_archetype_with_matching_driver_finds_playbook(self):
        archetypes = [_make_archetype(1, ["nps_score"], ["tenure_days"])]
        playbooks = [
            _make_playbook("low_nps", "Recover accounts with declining nps_score scores"),
            _make_playbook("upsell", "Offer expansion to customers with monthly_revenue growth"),
        ]
        mappings = map_archetypes_to_playbooks(archetypes, playbooks, GOLD_FEATURES)
        assert len(mappings) == 1
        assert "low_nps" in mappings[0].candidate_playbook_ids

    def test_overlap_threshold_filters_unrelated(self):
        archetypes = [_make_archetype(1, ["tenure_days"])]
        playbooks = [
            _make_playbook("upsell", "Offer expansion to customers with monthly_revenue growth")
        ]
        mappings = map_archetypes_to_playbooks(archetypes, playbooks, GOLD_FEATURES)
        assert mappings[0].candidate_playbook_ids == []

    def test_primary_driver_match_overrides_low_overlap(self):
        archetypes = [_make_archetype(1, ["tenure_days"])]
        playbooks = [
            _make_playbook(
                "loyalty_program",
                "Reward customers with high tenure_days and bonus offers and gifts and tokens",
            )
        ]
        mappings = map_archetypes_to_playbooks(archetypes, playbooks, GOLD_FEATURES)
        assert "loyalty_program" in mappings[0].candidate_playbook_ids

    def test_template_namer_writes_rationale(self):
        archetypes = [_make_archetype(1, ["nps_score"])]
        playbooks = [_make_playbook("low_nps", "Recover accounts with low nps_score outreach")]
        mappings = map_archetypes_to_playbooks(archetypes, playbooks, GOLD_FEATURES)
        assert mappings[0].fit_decisions[0].rationale  # non-empty
        assert mappings[0].llm_model_id == "template"

    def test_archetype_without_drivers_returns_empty_candidate_set(self):
        archetypes = [_make_archetype(1, [], [])]
        playbooks = [_make_playbook("low_nps", "Recover accounts with low nps_score")]
        mappings = map_archetypes_to_playbooks(archetypes, playbooks, GOLD_FEATURES)
        assert mappings[0].candidate_playbook_ids == []
        assert mappings[0].confidence == 0.0

    def test_multiple_archetypes_each_get_their_own_mapping(self):
        archetypes = [
            _make_archetype(1, ["nps_score"]),
            _make_archetype(2, ["monthly_revenue"]),
        ]
        playbooks = [
            _make_playbook("low_nps", "Recover accounts with low nps_score outreach"),
            _make_playbook("upsell", "Offer expansion to customers with monthly_revenue growth"),
        ]
        mappings = map_archetypes_to_playbooks(archetypes, playbooks, GOLD_FEATURES)
        assert len(mappings) == 2
        assert "low_nps" in mappings[0].candidate_playbook_ids
        assert "upsell" in mappings[1].candidate_playbook_ids

    def test_sort_by_overlap_score_descending(self):
        archetypes = [_make_archetype(1, ["nps_score"])]
        playbooks = [
            # a: 1 of 1 referenced features matches → overlap 1.0
            _make_playbook("a", "matches nps_score only"),
            # b: 1 of 2 referenced features matches → overlap 0.5
            _make_playbook("b", "references nps_score and monthly_revenue"),
        ]
        mappings = map_archetypes_to_playbooks(archetypes, playbooks, GOLD_FEATURES)
        assert mappings[0].candidate_playbook_ids == ["a", "b"]


# ---------------------------------------------------------------------------
# map_archetypes_to_playbooks with custom LLM namer
# ---------------------------------------------------------------------------


class _StubNamer:
    """Captures call args and returns a canned ArchetypeNaming."""

    model_id: str = "stub-llm"

    def __init__(self) -> None:
        self.contexts: List[ArchetypeContext] = []

    def name_archetype(self, context: ArchetypeContext) -> ArchetypeNaming:
        self.contexts.append(context)
        return ArchetypeNaming(
            archetype_name=f"Stubbed Archetype {context.cluster_index}",
            archetype_description="Stubbed description.",
            playbooks=[
                PlaybookFitDecision(
                    playbook_id=cand["playbook_id"],
                    fit_score=0.99,
                    rationale="stubbed rationale",
                )
                for cand in context.candidate_playbooks
            ],
            confidence=0.91,
            llm_model_id="stub-llm",
        )


class TestMapArchetypesToPlaybooksCustomNamer:
    def test_uses_namer_results(self):
        namer = _StubNamer()
        archetypes = [_make_archetype(1, ["nps_score"])]
        playbooks = [_make_playbook("low_nps", "Recover accounts with low nps_score outreach")]
        mappings = map_archetypes_to_playbooks(
            archetypes, playbooks, GOLD_FEATURES, llm_namer=namer
        )
        assert mappings[0].archetype_name == "Stubbed Archetype 1"
        assert mappings[0].llm_model_id == "stub-llm"
        assert mappings[0].fit_decisions[0].rationale == "stubbed rationale"

    def test_derivation_method_changes_when_llm_used(self):
        namer = _StubNamer()
        archetypes = [_make_archetype(1, ["nps_score"])]
        playbooks = [_make_playbook("low_nps", "Recover accounts with low nps_score outreach")]
        mappings = map_archetypes_to_playbooks(
            archetypes, playbooks, GOLD_FEATURES, llm_namer=namer
        )
        assert mappings[0].derivation_method == "feature_overlap+llm"

    def test_namer_receives_full_archetype_context(self):
        namer = _StubNamer()
        archetypes = [_make_archetype(1, ["nps_score"], ["tenure_days"])]
        playbooks = [_make_playbook("low_nps", "Recover accounts with low nps_score outreach")]
        map_archetypes_to_playbooks(
            archetypes, playbooks, GOLD_FEATURES, llm_namer=namer
        )
        ctx = namer.contexts[0]
        assert ctx.cluster_index == 1
        assert ctx.top_positive_drivers
        assert ctx.top_negative_drivers


# ---------------------------------------------------------------------------
# validate_playbook_feature_references
# ---------------------------------------------------------------------------


class TestValidatePlaybookFeatureReferences:
    """Schema-drift guard: playbook target_features must name real gold columns."""

    def test_no_raise_when_all_target_features_exist(self):
        from customer_retention.stages.causal.playbook_mapper import (
            validate_playbook_feature_references,
        )
        playbooks = [
            {"playbook_id": "a", "target_features": ["tenure_days", "nps_score"]},
        ]
        validate_playbook_feature_references(playbooks, GOLD_FEATURES)

    def test_raises_when_target_feature_missing_from_gold(self):
        import pytest

        from customer_retention.stages.causal.playbook_mapper import (
            validate_playbook_feature_references,
        )
        playbooks = [
            {"playbook_id": "low_nps", "target_features": ["tenure_days", "renamed_col"]},
        ]
        with pytest.raises(RuntimeError, match="renamed_col"):
            validate_playbook_feature_references(playbooks, GOLD_FEATURES)

    def test_error_message_names_offending_playbook(self):
        import pytest

        from customer_retention.stages.causal.playbook_mapper import (
            validate_playbook_feature_references,
        )
        playbooks = [
            {"playbook_id": "low_nps", "target_features": ["bad_col"]},
        ]
        with pytest.raises(RuntimeError, match="low_nps"):
            validate_playbook_feature_references(playbooks, GOLD_FEATURES)

    def test_no_raise_when_target_features_empty(self):
        from customer_retention.stages.causal.playbook_mapper import (
            validate_playbook_feature_references,
        )
        playbooks = [
            {"playbook_id": "a", "target_features": []},
            {"playbook_id": "b"},
        ]
        validate_playbook_feature_references(playbooks, GOLD_FEATURES)

    def test_aggregates_multiple_offenders(self):
        import pytest

        from customer_retention.stages.causal.playbook_mapper import (
            validate_playbook_feature_references,
        )
        playbooks = [
            {"playbook_id": "a", "target_features": ["ghost_a"]},
            {"playbook_id": "b", "target_features": ["ghost_b", "tenure_days"]},
        ]
        with pytest.raises(RuntimeError) as excinfo:
            validate_playbook_feature_references(playbooks, GOLD_FEATURES)
        msg = str(excinfo.value)
        assert "ghost_a" in msg
        assert "ghost_b" in msg
