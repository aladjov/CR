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
from customer_retention.stages.causal.shap_runner import BackgroundSample

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
        model=MagicMock(),
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


# ---------------------------------------------------------------------------
# DerivationResult.summary
# ---------------------------------------------------------------------------


class TestDerivationResult:
    def test_summary_includes_run_id_and_counts(self):
        result = DerivationResult(
            derivation_run_id="deriv_xyz",
            best_k=4,
            best_silhouette=0.6,
            cluster_sizes=[(0, 100), (1, 200)],
            cluster_target_means=[(0, 0.4), (1, 0.5)],
            archetype_rows=[{}, {}],
            eligibility_policy_rows=[{}, {}, {}],
            background=MagicMock(),
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


class TestShapMethodRouting:
    """Verify DerivationConfig.shap_method dispatches to the correct runner."""

    def test_default_method_is_linear(self):
        cfg = _make_config()
        assert cfg.shap_method == "linear"

    def test_linear_method_routes_to_compute_shap_distributed(self, monkeypatch):
        import customer_retention.stages.causal.derivation as mod

        linear = MagicMock(return_value="linear_result")
        sampling = MagicMock(return_value="sampling_result")
        monkeypatch.setattr(mod, "compute_shap_distributed", linear)
        monkeypatch.setattr(mod, "compute_shap_sampling_distributed", sampling)

        cfg = _make_config()
        cfg.shap_method = "linear"
        bg = BackgroundSample(rows=[], feature_columns=[], sample_size=0)
        out = mod._compute_shap(cfg, bg)
        assert out == "linear_result"
        linear.assert_called_once()
        sampling.assert_not_called()

    def test_sampling_method_routes_to_compute_shap_sampling_distributed(self, monkeypatch):
        import customer_retention.stages.causal.derivation as mod

        linear = MagicMock(return_value="linear_result")
        sampling = MagicMock(return_value="sampling_result")
        monkeypatch.setattr(mod, "compute_shap_distributed", linear)
        monkeypatch.setattr(mod, "compute_shap_sampling_distributed", sampling)

        cfg = _make_config()
        cfg.shap_method = "sampling"
        cfg.shap_model_uri = "models:/m@production"
        cfg.shap_num_samples = 30
        bg = BackgroundSample(rows=[], feature_columns=[], sample_size=0)
        out = mod._compute_shap(cfg, bg)
        assert out == "sampling_result"
        linear.assert_not_called()
        sampling.assert_called_once()
        call = sampling.call_args
        assert call.kwargs["model_uri"] == "models:/m@production"
        assert call.kwargs["num_samples"] == 30

    def test_sampling_method_without_model_uri_fails_fast(self):
        import customer_retention.stages.causal.derivation as mod

        cfg = _make_config()
        cfg.shap_method = "sampling"
        cfg.shap_model_uri = None
        bg = BackgroundSample(rows=[], feature_columns=[], sample_size=0)
        with pytest.raises(ValueError, match="shap_model_uri"):
            mod._compute_shap(cfg, bg)

    def test_unknown_method_fails_fast(self):
        import customer_retention.stages.causal.derivation as mod

        cfg = _make_config()
        cfg.shap_method = "bogus"
        bg = BackgroundSample(rows=[], feature_columns=[], sample_size=0)
        with pytest.raises(ValueError, match="shap_method"):
            mod._compute_shap(cfg, bg)
