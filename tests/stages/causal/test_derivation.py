"""Unit tests for the derivation orchestrator.

The orchestrator wires together six modules. Heavy paths (SHAP UDF, KMeans
fit, surrogate trees) are exercised in their own dedicated tests; this
file focuses on the row-builder math: archetype/policy ID derivation,
top-driver splitting, and the eligibility-policy fan-out from the
mapping.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from customer_retention.stages.causal import derivation
from customer_retention.stages.causal.derivation import (
    DerivationConfig,
    DerivationResult,
    stability_score,
)
from customer_retention.stages.causal.llm_namer import PlaybookFitDecision
from customer_retention.stages.causal.playbook_mapper import (
    ArchetypeMapping,
    ArchetypeSummary,
)
from customer_retention.stages.causal.rule_extractor import ExtractedRule

# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


class TestPureHelpers:
    def test_archetype_id_is_deterministic(self):
        a = derivation._archetype_id("model", "v1", 3)
        b = derivation._archetype_id("model", "v1", 3)
        assert a == b
        assert a.startswith("arch_3_")

    def test_archetype_id_changes_with_inputs(self):
        a = derivation._archetype_id("model", "v1", 1)
        b = derivation._archetype_id("model", "v2", 1)
        c = derivation._archetype_id("model", "v1", 2)
        assert len({a, b, c}) == 3

    def test_archetype_version_uses_run_id(self):
        v = derivation._archetype_version("deriv_xyz", 1)
        assert v.startswith("v_")

    def test_policy_id_is_deterministic(self):
        a = derivation._policy_id("v1", "low_nps", "v_aaa")
        b = derivation._policy_id("v1", "low_nps", "v_aaa")
        assert a == b
        assert a.startswith("pol_")

    def test_safe_float_handles_none(self):
        assert derivation._safe_float(None) is None
        assert derivation._safe_float("3.14") == pytest.approx(3.14)
        assert derivation._safe_float("garbage") is None

    def test_make_run_id_format(self):
        cfg = MagicMock(spec=DerivationConfig)
        cfg.model_name = "model"
        cfg.model_version = "v1"
        run_id = derivation._make_run_id(cfg)
        assert run_id.startswith("deriv_")

    def test_stability_score_reexport(self):
        assert stability_score([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)

    def test_row_value_scrubs_nan(self):
        """Regression: ``_row_value`` feeds the surrogate-tree numpy matrix in
        ``_collect_surrogate_inputs``. ``float('nan')`` previously passed
        through — old sklearn (<1.3) raises on NaN, and newer sklearn learns
        'null pattern' as a tree-split predicate (wrong semantics for
        eligibility rules). NaN must be coerced to 0.0 the same way NULL is."""
        row = {"a": float("nan"), "b": float("inf"), "c": 3.14, "d": None, "e": "garbage"}
        # NaN and strings both land on the error branch → 0.0
        assert derivation._row_value(row, "a") == 0.0
        # +inf is a finite-but-extreme value; surrogate tree tolerates it. Leave as-is.
        assert derivation._row_value(row, "b") == float("inf")
        assert derivation._row_value(row, "c") == pytest.approx(3.14)
        assert derivation._row_value(row, "d") == 0.0
        assert derivation._row_value(row, "e") == 0.0

    def test_row_value_scrubs_negative_nan(self):
        """Bit-level defensiveness: a signaling NaN from pandas→Spark round-trip
        has the sign bit set. ``math.isnan`` catches both quiet and signaling,
        positive and negative NaN."""
        row = {"x": float("-nan")}
        assert derivation._row_value(row, "x") == 0.0


# ---------------------------------------------------------------------------
# Driver splitting
# ---------------------------------------------------------------------------


class TestSplitDrivers:
    def test_positive_negatives_split_correctly(self):
        positives, negatives = derivation._split_drivers(
            shap_vec=[0.3, -0.2, 0.1, -0.4],
            shap_feature_order=["shap_a", "shap_b", "shap_c", "shap_d"],
            raw_vec=[1.0, 2.0, 3.0, 4.0],
            raw_feature_order=["a", "b", "c", "d"],
        )
        pos_features = {p["feature"] for p in positives}
        neg_features = {n["feature"] for n in negatives}
        assert pos_features == {"a", "c"}
        assert neg_features == {"b", "d"}

    def test_strips_shap_prefix(self):
        positives, _ = derivation._split_drivers(
            shap_vec=[0.5],
            shap_feature_order=["shap_tenure_days"],
            raw_vec=[365.0],
            raw_feature_order=["tenure_days"],
        )
        assert positives[0]["feature"] == "tenure_days"
        assert positives[0]["mean_value"] == 365.0

    def test_sorted_by_absolute_shap(self):
        positives, _ = derivation._split_drivers(
            shap_vec=[0.1, 0.5, 0.3],
            shap_feature_order=["shap_a", "shap_b", "shap_c"],
            raw_vec=[1.0, 2.0, 3.0],
            raw_feature_order=["a", "b", "c"],
        )
        assert [p["feature"] for p in positives] == ["b", "c", "a"]


# ---------------------------------------------------------------------------
# Top-shap struct
# ---------------------------------------------------------------------------


class TestJoinClusterLabels:
    def test_dedupes_both_sides_on_join_key(self):
        """Regression: if gold is snapshot-grained (multiple rows per
        ``account_id``), the inner join fans out |raw| × |labelled-per-key|
        rows per account, and downstream per-cluster aggregations (centroids,
        sizes, mean churn) become biased toward entities with more snapshots.
        Both sides must be collapsed to one row per entity before joining —
        the archetype is a property of the entity, not the snapshot."""
        labelled = MagicMock(name="LabelledShap")
        labelled_selected = MagicMock(name="LabelledSelected")
        labelled_deduped = MagicMock(name="LabelledDeduped")
        labelled.select = MagicMock(return_value=labelled_selected)
        labelled_selected.dropDuplicates = MagicMock(return_value=labelled_deduped)

        raw = MagicMock(name="RawFeatures")
        raw_deduped = MagicMock(name="RawDeduped")
        raw.dropDuplicates = MagicMock(return_value=raw_deduped)
        raw_deduped.join = MagicMock(return_value=MagicMock(name="Joined"))

        derivation._join_cluster_labels(labelled, raw, join_key="entity_id")

        labelled_selected.dropDuplicates.assert_called_once_with(["entity_id"])
        raw.dropDuplicates.assert_called_once_with(["entity_id"])
        raw_deduped.join.assert_called_once()
        join_args, join_kwargs = raw_deduped.join.call_args
        assert join_args[0] is labelled_deduped or join_kwargs.get("other") is labelled_deduped
        assert join_kwargs.get("on") == "entity_id"
        assert join_kwargs.get("how") == "inner"

    def test_select_scopes_to_join_key_and_cluster_col(self):
        labelled = MagicMock(name="LabelledShap")
        labelled_selected = MagicMock(name="LabelledSelected")
        labelled_selected.dropDuplicates = MagicMock(return_value=MagicMock())
        labelled.select = MagicMock(return_value=labelled_selected)
        raw = MagicMock(name="RawFeatures")
        raw.dropDuplicates = MagicMock(return_value=MagicMock(join=MagicMock(return_value=MagicMock())))

        derivation._join_cluster_labels(labelled, raw, join_key="entity_id")

        labelled.select.assert_called_once()
        assert "entity_id" in labelled.select.call_args.args
        from customer_retention.stages.causal.clusterer import CLUSTER_COL
        assert CLUSTER_COL in labelled.select.call_args.args


class TestTopShapStruct:
    def test_combines_positive_and_negative_with_direction(self):
        summary = ArchetypeSummary(
            cluster_index=1,
            cluster_size=10,
            cluster_mean_churn_probability=0.5,
            top_positive_drivers=[{"feature": "a", "mean_shap": 0.2, "mean_value": 1.0}],
            top_negative_drivers=[{"feature": "b", "mean_shap": -0.1, "mean_value": 2.0}],
        )
        out = derivation._top_shap_struct(summary)
        assert len(out) == 2
        assert out[0]["direction"] == "positive"
        assert out[1]["direction"] == "negative"


# ---------------------------------------------------------------------------
# Row builders (no Spark needed)
# ---------------------------------------------------------------------------


def _make_extracted_rule(cluster_index: int) -> ExtractedRule:
    return ExtractedRule(
        cluster_index=cluster_index,
        predicate_json={"op": ">=", "feature": "tenure_days", "value": 365},
        predicate_sql="`tenure_days` >= 365",
        pure_leaf_count=1,
        coverage=0.85,
        used_features=["tenure_days"],
        feature_thresholds={"tenure_days": {"p25": 100.0, "p50": 200.0, "p75": 300.0}},
    )


def _make_mapping(cluster_index: int, playbook_ids: list[str]) -> ArchetypeMapping:
    return ArchetypeMapping(
        cluster_index=cluster_index,
        archetype_name=f"Test {cluster_index}",
        archetype_description="A test archetype",
        rationale="rationale",
        confidence=0.9,
        candidate_playbook_ids=list(playbook_ids),
        fit_decisions=[
            PlaybookFitDecision(playbook_id=pid, fit_score=0.8, rationale="r")
            for pid in playbook_ids
        ],
        llm_model_id="test-llm",
        derivation_method="feature_overlap",
    )


def _make_config(write: bool = False) -> DerivationConfig:
    return DerivationConfig(
        spark=MagicMock(),
        training_df=MagicMock(),
        raw_feature_df=MagicMock(),
        feature_columns=["a", "b"],
        model_uri="models:/test_model@production",
        target_column="target",
        archetype_catalog_fqn="cat.sch.archetype_catalog",
        eligibility_policy_fqn="cat.sch.eligibility_policy",
        playbooks=[
            {"playbook_id": "low_nps", "version": "1.0.0", "description": "low nps", "expected_uplift_pct_default": 0.05},
            {"playbook_id": "upsell", "version": "1.0.0", "description": "upsell", "expected_uplift_pct_default": 0.1},
        ],
        gold_feature_names=["a", "b"],
        model_name="test_model",
        model_version="v1",
        write=write,
    )


class TestBuildArchetypeRows:
    def test_one_row_per_summary(self):
        summaries = [
            ArchetypeSummary(
                cluster_index=0,
                cluster_size=50,
                cluster_mean_churn_probability=0.4,
                top_positive_drivers=[{"feature": "a", "mean_shap": 0.1, "mean_value": 1.0}],
                top_negative_drivers=[],
            )
        ]
        rules = [_make_extracted_rule(0)]
        mappings = [_make_mapping(0, ["low_nps"])]
        rows = derivation._build_archetype_rows(
            config=_make_config(),
            derivation_run_id="deriv_x",
            timestamp=None,
            summaries=summaries,
            clustering_silhouette=0.65,
            shap_centroids=[[0.1, 0.0]],
            raw_centroids=[[1.0, 2.0]],
            raw_feature_order=["a", "b"],
            feature_scales=[10.0, 5.0],
            extracted_rules=rules,
            mappings=mappings,
            sizes=[(0, 50)],
            mean_targets=[(0, 0.42)],
        )
        assert len(rows) == 1
        row = rows[0]
        assert row["model_name"] == "test_model"
        assert row["model_version"] == "v1"
        assert row["cluster_size"] == 50
        assert row["status"] == "pending_review"
        assert row["cluster_mean_churn_probability"] == pytest.approx(0.42)
        assert row["centroid_vector"] == [0.1, 0.0]
        assert row["centroid_vector_raw"] == [1.0, 2.0]
        assert row["centroid_feature_order"] == ["a", "b"]
        assert row["centroid_feature_scales"] == [10.0, 5.0]
        assert row["name"] == "Test 0"
        # derivation_params is JSON-encoded
        params = json.loads(row["derivation_params"])
        assert params["silhouette"] == pytest.approx(0.65)


class TestBuildPolicyRows:
    def test_one_policy_row_per_fit_decision(self):
        archetype_rows = [
            {
                "archetype_id": "arch_0_aaa",
                "archetype_version": "v_111",
                "cluster_raw_id": "0",
            }
        ]
        rules = [_make_extracted_rule(0)]
        mappings = [_make_mapping(0, ["low_nps", "upsell"])]
        rows = derivation._build_policy_rows(
            config=_make_config(),
            derivation_run_id="deriv_x",
            timestamp=None,
            archetype_rows=archetype_rows,
            extracted_rules=rules,
            mappings=mappings,
        )
        assert len(rows) == 2
        assert {r["playbook_id"] for r in rows} == {"low_nps", "upsell"}
        for row in rows:
            assert row["status"] == "pending_review"
            assert row["model_name"] == "test_model"
            assert json.loads(row["eligibility_rules"])["op"] == ">="
            assert row["requires_features"] == ["tenure_days"]
            assert row["archetype_ids"] == ["v_111"]

    def test_skips_decisions_for_unknown_playbooks(self):
        archetype_rows = [
            {"archetype_id": "arch_0_a", "archetype_version": "v_1", "cluster_raw_id": "0"}
        ]
        rules = [_make_extracted_rule(0)]
        mappings = [_make_mapping(0, ["unknown_playbook"])]
        rows = derivation._build_policy_rows(
            config=_make_config(),
            derivation_run_id="deriv_x",
            timestamp=None,
            archetype_rows=archetype_rows,
            extracted_rules=rules,
            mappings=mappings,
        )
        assert rows == []

    def test_skips_when_no_rule_for_cluster(self):
        archetype_rows = [
            {"archetype_id": "arch_0_a", "archetype_version": "v_1", "cluster_raw_id": "0"}
        ]
        rules: list[ExtractedRule] = []
        mappings = [_make_mapping(0, ["low_nps"])]
        rows = derivation._build_policy_rows(
            config=_make_config(),
            derivation_run_id="deriv_x",
            timestamp=None,
            archetype_rows=archetype_rows,
            extracted_rules=rules,
            mappings=mappings,
        )
        assert rows == []


class TestFitThresholdTiers:
    """Three-tier classification + catch-all routing in _build_policy_rows."""

    def _arch_row(self) -> dict:
        return {"archetype_id": "arch_0_a", "archetype_version": "v_1", "cluster_raw_id": "0"}

    def _mapping_with_scores(self, scored: dict[str, float]) -> ArchetypeMapping:
        return ArchetypeMapping(
            cluster_index=0,
            archetype_name="T",
            archetype_description="t",
            rationale="r",
            confidence=0.5,
            candidate_playbook_ids=list(scored.keys()),
            fit_decisions=[
                PlaybookFitDecision(playbook_id=pid, fit_score=score, rationale="r")
                for pid, score in scored.items()
            ],
            llm_model_id="prose_overlap",
        )

    def test_auto_tier_emits_row_with_fit_tier_auto(self):
        rows = derivation._build_policy_rows(
            config=_make_config(),
            derivation_run_id="dx",
            timestamp=None,
            archetype_rows=[self._arch_row()],
            extracted_rules=[_make_extracted_rule(0)],
            mappings=[self._mapping_with_scores({"low_nps": 0.9})],
        )
        assert len(rows) == 1
        assert rows[0]["fit_tier"] == "auto"
        assert rows[0]["fit_score"] == pytest.approx(0.9)
        assert "[auto]" in rows[0]["rationale"]

    def test_review_tier_emits_row_with_fit_tier_review(self):
        rows = derivation._build_policy_rows(
            config=_make_config(),
            derivation_run_id="dx",
            timestamp=None,
            archetype_rows=[self._arch_row()],
            extracted_rules=[_make_extracted_rule(0)],
            mappings=[self._mapping_with_scores({"low_nps": 0.3})],
        )
        assert len(rows) == 1
        assert rows[0]["fit_tier"] == "review"
        assert rows[0]["fit_score"] == pytest.approx(0.3)

    def test_manual_tier_is_dropped(self):
        rows = derivation._build_policy_rows(
            config=_make_config(),
            derivation_run_id="dx",
            timestamp=None,
            archetype_rows=[self._arch_row()],
            extracted_rules=[_make_extracted_rule(0)],
            mappings=[self._mapping_with_scores({"low_nps": 0.05})],
        )
        assert rows == []

    def test_catch_all_row_when_configured_and_no_matches(self):
        cfg = _make_config()
        cfg.default_playbook_id = "low_nps"
        rows = derivation._build_policy_rows(
            config=cfg,
            derivation_run_id="dx",
            timestamp=None,
            archetype_rows=[self._arch_row()],
            extracted_rules=[_make_extracted_rule(0)],
            mappings=[self._mapping_with_scores({"low_nps": 0.05, "upsell": 0.01})],
        )
        assert len(rows) == 1
        assert rows[0]["fit_tier"] == "catch_all"
        assert rows[0]["playbook_id"] == "low_nps"
        assert "needs manual review" in rows[0]["rationale"]

    def test_no_catch_all_row_when_auto_match_exists(self):
        cfg = _make_config()
        cfg.default_playbook_id = "upsell"
        rows = derivation._build_policy_rows(
            config=cfg,
            derivation_run_id="dx",
            timestamp=None,
            archetype_rows=[self._arch_row()],
            extracted_rules=[_make_extracted_rule(0)],
            mappings=[self._mapping_with_scores({"low_nps": 0.9, "upsell": 0.02})],
        )
        tiers = [r["fit_tier"] for r in rows]
        assert "catch_all" not in tiers
        assert "auto" in tiers

    def test_no_catch_all_when_default_is_disabled(self):
        cfg = _make_config()
        cfg.default_playbook_id = None
        rows = derivation._build_policy_rows(
            config=cfg,
            derivation_run_id="dx",
            timestamp=None,
            archetype_rows=[self._arch_row()],
            extracted_rules=[_make_extracted_rule(0)],
            mappings=[self._mapping_with_scores({"low_nps": 0.05})],
        )
        assert rows == []

    def test_custom_thresholds_classify_differently(self):
        cfg = _make_config()
        cfg.fit_thresholds = derivation.FitThresholds(auto=0.3, review=0.1)
        rows = derivation._build_policy_rows(
            config=cfg,
            derivation_run_id="dx",
            timestamp=None,
            archetype_rows=[self._arch_row()],
            extracted_rules=[_make_extracted_rule(0)],
            mappings=[self._mapping_with_scores({"low_nps": 0.35})],
        )
        assert rows[0]["fit_tier"] == "auto"


# ---------------------------------------------------------------------------
# DerivationResult.summary
# ---------------------------------------------------------------------------


class TestDerivationResult:
    def test_summary_includes_run_id_and_counts(self):
        from customer_retention.stages.modeling.shap_attribution import ShapAttribution
        result = DerivationResult(
            derivation_run_id="deriv_xyz",
            best_k=4,
            best_silhouette=0.6,
            cluster_sizes=[(0, 100), (1, 200)],
            cluster_target_means=[(0, 0.4), (1, 0.5)],
            archetype_rows=[{}, {}],
            eligibility_policy_rows=[{}, {}, {}],
            attribution=ShapAttribution(
                importances={"a": 1.0},
                background_means={"a": 0.0},
                feature_columns=["a"],
                sample_size=10,
            ),
            extracted_rules=[],
            mappings=[],
            llm_model_id="test-llm",
        )
        text = result.summary()
        assert "deriv_xyz" in text
        assert "best_k=4" in text
        assert "2 archetype rows" in text
        assert "3 policy rows" in text
        assert "test-llm" in text


# ---------------------------------------------------------------------------
# _write_rows validation
# ---------------------------------------------------------------------------


class TestWriteRowsValidation:
    def test_missing_archetype_catalog_fqn_raises(self):
        cfg = _make_config(write=True)
        cfg.archetype_catalog_fqn = ""
        with pytest.raises(ValueError, match="archetype_catalog_fqn"):
            derivation._write_rows(cfg, archetype_rows=[], policy_rows=[])

    def test_missing_eligibility_policy_fqn_raises(self):
        cfg = _make_config(write=True)
        cfg.eligibility_policy_fqn = ""
        with pytest.raises(ValueError, match="eligibility_policy_fqn"):
            derivation._write_rows(cfg, archetype_rows=[], policy_rows=[])


class TestValidatePolicyCoverage:
    """Guard against the silent failure where archetypes exist but no policies map.

    Original incident: playbook YAMLs had prose descriptions with no
    overlap against the archetype's top SHAP-driver tokens, the overlap
    baseline scored every (archetype, playbook) pair below the review
    threshold, ``_build_policy_rows`` dropped them all as ``manual``, and
    with no ``DEFAULT_PLAYBOOK_ID`` configured 0 policy rows got written.
    c05's snapshot then failed three steps downstream with "no active
    eligibility_policy rows". This validation converts the silent case
    into a RuntimeError at derivation time and points the operator at
    the actual unblocks.
    """

    def _archetype_row(self) -> dict:
        return {"archetype_id": "a1", "archetype_version": "v1"}

    def _mapping_empty_decisions(self) -> ArchetypeMapping:
        return ArchetypeMapping(
            cluster_index=0,
            archetype_name="Archetype 0",
            archetype_description="",
            rationale="",
            confidence=0.0,
            candidate_playbook_ids=[],
            fit_decisions=[],
        )

    def test_raises_when_archetypes_present_but_no_policies(self):
        cfg = _make_config(write=False)
        with pytest.raises(RuntimeError, match="0 eligibility_policy rows"):
            derivation._validate_policy_coverage(
                archetype_rows=[self._archetype_row()],
                policy_rows=[],
                mappings=[self._mapping_empty_decisions()],
                config=cfg,
            )

    def test_error_message_names_concrete_unblocks(self):
        cfg = _make_config(write=False)
        # The empty-decisions branch points operators at LLM endpoint /
        # playbook_catalog reachability — that's the concrete unblock when
        # nothing matched at all.
        with pytest.raises(RuntimeError, match="LLM_ENDPOINT_NAME"):
            derivation._validate_policy_coverage(
                archetype_rows=[self._archetype_row()],
                policy_rows=[],
                mappings=[self._mapping_empty_decisions()],
                config=cfg,
            )

    def test_error_message_names_threshold_unblocks_when_decisions_present(self):
        # When decisions WERE produced but every score fell below the review
        # threshold, the message must point at DEFAULT_PLAYBOOK_ID and
        # FIT_REVIEW_THRESHOLD as the two real unblocks.
        cfg = _make_config(write=False)
        decision = PlaybookFitDecision(
            playbook_id="pb_a", fit_score=0.05, rationale="weak overlap"
        )
        mapping = ArchetypeMapping(
            cluster_index=0,
            archetype_name="Archetype 0",
            archetype_description="",
            rationale="",
            confidence=0.0,
            candidate_playbook_ids=["pb_a"],
            fit_decisions=[decision],
        )
        with pytest.raises(RuntimeError, match="DEFAULT_PLAYBOOK_ID"):
            derivation._validate_policy_coverage(
                archetype_rows=[self._archetype_row()],
                policy_rows=[],
                mappings=[mapping],
                config=cfg,
            )

    def test_no_raise_when_policies_present(self):
        cfg = _make_config(write=False)
        derivation._validate_policy_coverage(
            archetype_rows=[self._archetype_row()],
            policy_rows=[{"eligibility_policy_id": "p1"}],
            mappings=[self._mapping_empty_decisions()],
            config=cfg,
        )

    def test_no_raise_when_no_archetypes(self):
        cfg = _make_config(write=False)
        derivation._validate_policy_coverage(
            archetype_rows=[],
            policy_rows=[],
            mappings=[],
            config=cfg,
        )
