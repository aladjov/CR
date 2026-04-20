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
            _make_playbook("low_nps", "Recover accounts with declining NPS survey scores"),
            _make_playbook("upsell", "Offer expansion to customers with revenue growth"),
        ]
        mappings = map_archetypes_to_playbooks(archetypes, playbooks, GOLD_FEATURES)
        assert len(mappings) == 1
        # All playbooks are candidates now; matching appears in fit_decisions scores.
        assert "low_nps" in mappings[0].candidate_playbook_ids
        scores = {d.playbook_id: d.fit_score for d in mappings[0].fit_decisions}
        assert scores["low_nps"] > scores["upsell"]

    def test_unrelated_playbook_appears_with_low_score(self):
        """Every playbook stays visible for review; unrelated ones get a low score, not filtered out."""
        archetypes = [_make_archetype(1, ["tenure_days"])]
        playbooks = [
            _make_playbook("upsell", "Offer expansion based on revenue growth")
        ]
        mappings = map_archetypes_to_playbooks(archetypes, playbooks, GOLD_FEATURES)
        assert mappings[0].candidate_playbook_ids == ["upsell"]
        assert mappings[0].fit_decisions[0].fit_score == 0.0

    def test_matcher_writes_rationale(self):
        archetypes = [_make_archetype(1, ["nps_score"])]
        playbooks = [_make_playbook("low_nps", "Recover accounts with low NPS scores")]
        mappings = map_archetypes_to_playbooks(archetypes, playbooks, GOLD_FEATURES)
        assert mappings[0].fit_decisions[0].rationale  # non-empty
        assert mappings[0].llm_model_id == "prose_overlap"

    def test_archetype_without_drivers_emits_zero_scored_decisions(self):
        """No driver tokens → every fit decision scores 0, but all playbooks remain candidates."""
        archetypes = [_make_archetype(1, [], [])]
        playbooks = [_make_playbook("low_nps", "Recover accounts with low NPS scores")]
        mappings = map_archetypes_to_playbooks(archetypes, playbooks, GOLD_FEATURES)
        assert mappings[0].candidate_playbook_ids == ["low_nps"]
        assert mappings[0].fit_decisions[0].fit_score == 0.0
        assert mappings[0].confidence == 0.0

    def test_multiple_archetypes_each_get_their_own_mapping(self):
        archetypes = [
            _make_archetype(1, ["nps_score"]),
            _make_archetype(2, ["monthly_revenue"]),
        ]
        playbooks = [
            _make_playbook("low_nps", "Recover accounts with low NPS survey responses"),
            _make_playbook("upsell", "Offer expansion to customers with monthly revenue growth"),
        ]
        mappings = map_archetypes_to_playbooks(archetypes, playbooks, GOLD_FEATURES)
        assert len(mappings) == 2
        scores_0 = {d.playbook_id: d.fit_score for d in mappings[0].fit_decisions}
        scores_1 = {d.playbook_id: d.fit_score for d in mappings[1].fit_decisions}
        assert scores_0["low_nps"] > scores_0["upsell"]
        assert scores_1["upsell"] > scores_1["low_nps"]

    def test_fit_decisions_sorted_by_score_descending(self):
        archetypes = [_make_archetype(1, ["nps_score"])]
        playbooks = [
            _make_playbook("unrelated", "billing questions and invoice"),
            _make_playbook("low_nps", "NPS survey recovery outreach"),
        ]
        mappings = map_archetypes_to_playbooks(archetypes, playbooks, GOLD_FEATURES)
        # fit_decisions are sorted by score descending inside the namer
        decisions = mappings[0].fit_decisions
        assert decisions[0].playbook_id == "low_nps"
        assert decisions[0].fit_score >= decisions[1].fit_score


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
        assert mappings[0].derivation_method == "prose_overlap+llm"

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
# Prose-overlap scoring (no feature column coupling)
# ---------------------------------------------------------------------------


class TestProseOverlapScore:
    """Playbook prose ↔ archetype driver-token matching is the baseline."""

    def test_matches_driver_tokens_in_description(self):
        from customer_retention.stages.causal.playbook_mapper import prose_overlap_score

        playbook = {
            "playbook_id": "low_nps",
            "name": "Low NPS",
            "description": "Engage customers with low NPS survey response",
            "when_applicable": "",
        }
        archetype = _make_archetype(0, top_positive=["nps_score", "survey_count"])
        score, matched = prose_overlap_score(playbook, archetype)
        assert score > 0.0
        assert "nps" in matched

    def test_zero_score_when_no_driver_tokens_in_prose(self):
        from customer_retention.stages.causal.playbook_mapper import prose_overlap_score

        playbook = {
            "playbook_id": "x",
            "name": "X",
            "description": "something unrelated",
            "when_applicable": "",
        }
        archetype = _make_archetype(0, top_positive=["payment_overdue_days"])
        score, matched = prose_overlap_score(playbook, archetype)
        assert score == 0.0
        assert matched == []

    def test_matches_when_applicable_prose(self):
        from customer_retention.stages.causal.playbook_mapper import prose_overlap_score

        playbook = {
            "playbook_id": "credit",
            "name": "Credit",
            "description": "Internal workflow",
            "when_applicable": "Found useful when payment issues are the cause",
        }
        archetype = _make_archetype(0, top_positive=["payment_overdue_days"])
        score, matched = prose_overlap_score(playbook, archetype)
        assert score > 0.0
        assert "payment" in matched

    def test_no_feature_column_coupling(self):
        """The mapper does not consult gold_feature_names anywhere."""
        from customer_retention.stages.causal.playbook_mapper import (
            map_archetypes_to_playbooks,
        )
        archetypes = [_make_archetype(0, ["nps_score"])]
        playbooks = [_make_playbook("low_nps", "Low NPS engagement")]
        mappings_a = map_archetypes_to_playbooks(archetypes, playbooks, [])
        mappings_b = map_archetypes_to_playbooks(archetypes, playbooks, GOLD_FEATURES)
        assert len(mappings_a) == len(mappings_b) == 1
        assert mappings_a[0].fit_decisions[0].fit_score == mappings_b[0].fit_decisions[0].fit_score

    def test_every_playbook_becomes_a_candidate(self):
        """Transparency: all playbooks appear in fit_decisions with a score."""
        from customer_retention.stages.causal.playbook_mapper import (
            map_archetypes_to_playbooks,
        )
        archetypes = [_make_archetype(0, ["nps_score"])]
        playbooks = [
            _make_playbook("a", "about NPS dissatisfaction"),
            _make_playbook("b", "about billing"),
            _make_playbook("c", "about onboarding"),
        ]
        mappings = map_archetypes_to_playbooks(archetypes, playbooks, GOLD_FEATURES)
        ids = {d.playbook_id for d in mappings[0].fit_decisions}
        assert ids == {"a", "b", "c"}
