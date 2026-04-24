"""Tests for ``enrich_archetype_from_namespace``."""
from __future__ import annotations

from types import SimpleNamespace

from customer_retention.analysis.auto_explorer.run_namespace import RunNamespace
from customer_retention.stages.causal.column_descriptions_writer import ColumnDescriptionRow
from customer_retention.stages.causal.feature_meta_writer import FeatureMetaRow
from customer_retention.stages.causal.interpretation.enrichment_pipeline import (
    enrich_archetype_from_namespace,
)
from customer_retention.stages.causal.interpretation.sidecars import (
    write_column_descriptions_sidecar,
    write_feature_meta_sidecar,
    write_population_stats_sidecar,
)
from customer_retention.stages.causal.population_stats import PopulationStatsRow


def _raw_ctx():
    return SimpleNamespace(
        cluster_index=0,
        cluster_size=250,
        cluster_mean_churn_probability=0.42,
        top_positive_drivers=[{"feature": "nps_mean_90d", "mean_shap": 0.18, "mean_value": 8.5}],
        top_negative_drivers=[],
        candidate_playbooks=[],
    )


class TestEnrichArchetypeFromNamespace:
    def test_empty_sidecars_degrades_gracefully(self, tmp_path):
        ns = RunNamespace(root=tmp_path, run_id="r1")
        enriched = enrich_archetype_from_namespace(_raw_ctx(), ns, composite_name="cn1")
        assert enriched.cluster_size == 250
        driver = enriched.top_positive_drivers[0]
        assert driver.business_phrase == "nps_mean_90d"  # fallback
        assert driver.value_phrase == "unknown"  # no stats

    def test_full_sidecar_path_produces_narrated_context(self, tmp_path):
        ns = RunNamespace(root=tmp_path, run_id="r1")
        write_feature_meta_sidecar(ns, "cn1", [
            FeatureMetaRow(
                composite_name="cn1", feature_name="nps_mean_90d",
                source_columns=["nps"], aggregation_kind="avg",
                window_days=90, polarity="high_is_good",
                business_phrase="average NPS score over last 90 days",
            ),
        ])
        write_population_stats_sidecar(ns, [
            PopulationStatsRow(
                run_id="r1", feature_name="nps_mean_90d", dtype="numeric",
                q05=1, q25=3, q50=5, q75=7, q95=9,
            ),
        ])
        write_column_descriptions_sidecar(ns, [
            ColumnDescriptionRow(
                table="account", column_name="nps",
                business_name="Net Promoter Score", polarity="high_is_good",
            ),
        ])
        enriched = enrich_archetype_from_namespace(
            _raw_ctx(), ns, composite_name="cn1",
            total_book_size=1000, population_mean_churn=0.30,
        )
        driver = enriched.top_positive_drivers[0]
        assert driver.business_phrase == "average NPS score over last 90 days"
        assert driver.value_phrase == "elevated"
        assert enriched.share_of_book == 0.25
        assert enriched.lift_vs_population == 1.4

    def test_eligibility_predicate_rendered_to_prose(self, tmp_path):
        ns = RunNamespace(root=tmp_path, run_id="r1")
        write_feature_meta_sidecar(ns, "cn1", [
            FeatureMetaRow(
                composite_name="cn1", feature_name="nps_mean_90d",
                business_phrase="average NPS score over last 90 days",
                polarity="high_is_good",
            ),
        ])
        predicate = {"op": "<", "feature": "nps_mean_90d", "value": 4}
        enriched = enrich_archetype_from_namespace(
            _raw_ctx(), ns, composite_name="cn1",
            eligibility_predicate=predicate,
        )
        assert enriched.eligibility_rule_prose is not None
        assert "average NPS score over last 90 days" in enriched.eligibility_rule_prose
        assert "4" in enriched.eligibility_rule_prose

    def test_no_predicate_yields_null_prose(self, tmp_path):
        ns = RunNamespace(root=tmp_path, run_id="r1")
        enriched = enrich_archetype_from_namespace(_raw_ctx(), ns, composite_name="cn1")
        assert enriched.eligibility_rule_prose is None

    def test_composite_name_filter_isolates_runs(self, tmp_path):
        ns = RunNamespace(root=tmp_path, run_id="r1")
        write_feature_meta_sidecar(ns, "cn_other", [
            FeatureMetaRow(
                composite_name="cn_other", feature_name="nps_mean_90d",
                business_phrase="OTHER phrase",
            ),
        ])
        enriched = enrich_archetype_from_namespace(_raw_ctx(), ns, composite_name="cn1")
        # feature_meta came from a different composite_name → falls back to raw name
        assert enriched.top_positive_drivers[0].business_phrase == "nps_mean_90d"
