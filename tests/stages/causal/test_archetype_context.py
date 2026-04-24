"""Tests for ``archetype_context.build_enriched_context``."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from customer_retention.stages.causal.column_descriptions_writer import ColumnDescriptionRow
from customer_retention.stages.causal.feature_meta_writer import FeatureMetaRow
from customer_retention.stages.causal.interpretation.archetype_context import (
    build_enriched_context,
)
from customer_retention.stages.causal.interpretation.quantile_phrasing import PopulationStats


def _stats(**kw):
    return PopulationStats(q05=1, q25=3, q50=5, q75=7, q95=9, **kw)


def _raw_context(**overrides):
    defaults = dict(
        cluster_index=0,
        cluster_size=250,
        cluster_mean_churn_probability=0.42,
        top_positive_drivers=[
            {"feature": "nps_mean_90d", "mean_shap": 0.18, "mean_value": 8.5},
        ],
        top_negative_drivers=[
            {"feature": "tenure_days", "mean_shap": -0.12, "mean_value": 30.0},
        ],
        candidate_playbooks=[
            {
                "playbook_id": "nps_followup",
                "name": "NPS Follow-up",
                "when_applicable": "low NPS accounts",
                "description": "CSM outreach for unhappy NPS.",
                "policy_summary": "14-day cooldown, 50/week.",
                "expected_effect": "~22% churn reduction historical.",
                "fit_score": 0.81,
            }
        ],
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


class TestBuildEnrichedContext:
    def test_driver_rendering_with_full_lineage(self):
        feature_meta = {
            "nps_mean_90d": FeatureMetaRow(
                composite_name="cn",
                feature_name="nps_mean_90d",
                source_columns=["nps"],
                aggregation_kind="avg",
                window_days=90,
                polarity="high_is_good",
                business_phrase="average NPS score over last 90 days",
            )
        }
        population_stats = {"nps_mean_90d": _stats()}
        ctx = build_enriched_context(
            _raw_context(),
            feature_meta=feature_meta,
            population_stats=population_stats,
            total_book_size=1000,
            population_mean_churn=0.30,
        )
        driver = ctx.top_positive_drivers[0]
        assert driver.feature_name == "nps_mean_90d"
        assert driver.business_phrase == "average NPS score over last 90 days"
        assert driver.value == pytest.approx(8.5)
        assert driver.value_phrase == "elevated"
        assert driver.polarity == "high_is_good"
        assert ctx.share_of_book == pytest.approx(0.25)
        assert ctx.lift_vs_population == pytest.approx(1.4)

    def test_missing_feature_meta_falls_back_to_raw_name(self):
        ctx = build_enriched_context(
            _raw_context(),
            feature_meta={},
            population_stats={},
        )
        driver = ctx.top_positive_drivers[0]
        assert driver.business_phrase == "nps_mean_90d"
        assert driver.value_phrase == "unknown"

    def test_descriptions_used_when_meta_absent(self):
        descriptions = {
            "nps_mean_90d": ColumnDescriptionRow(
                table="t", column_name="nps_mean_90d",
                business_name="NPS trailing average",
                polarity="high_is_good",
            )
        }
        ctx = build_enriched_context(
            _raw_context(),
            feature_meta={},
            population_stats={"nps_mean_90d": _stats()},
            column_descriptions=descriptions,
        )
        driver = ctx.top_positive_drivers[0]
        assert driver.business_phrase == "NPS trailing average"
        assert driver.polarity == "high_is_good"

    def test_polarity_inversion_flips_phrase(self):
        feature_meta = {
            "churn_rate_90d": FeatureMetaRow(
                composite_name="cn",
                feature_name="churn_rate_90d",
                polarity="high_is_bad",
                business_phrase="90-day churn rate",
            )
        }
        raw = _raw_context(top_positive_drivers=[
            {"feature": "churn_rate_90d", "mean_shap": 0.3, "mean_value": 8.5},
        ])
        ctx = build_enriched_context(
            raw,
            feature_meta=feature_meta,
            population_stats={"churn_rate_90d": _stats()},
        )
        assert ctx.top_positive_drivers[0].value_phrase == "low"

    def test_share_and_lift_handle_zero_population(self):
        ctx = build_enriched_context(
            _raw_context(),
            feature_meta={},
            population_stats={},
            total_book_size=0,
            population_mean_churn=0.0,
        )
        assert ctx.share_of_book is None
        assert ctx.lift_vs_population is None

    def test_driver_stability_attached(self):
        ctx = build_enriched_context(
            _raw_context(),
            feature_meta={},
            population_stats={},
            driver_stability={"nps_mean_90d": 0.9},
        )
        assert ctx.top_positive_drivers[0].stability == 0.9

    def test_playbook_enrichment(self):
        ctx = build_enriched_context(
            _raw_context(),
            feature_meta={},
            population_stats={},
        )
        pb = ctx.candidate_playbooks[0]
        assert pb.playbook_id == "nps_followup"
        assert pb.when_applicable == "low NPS accounts"
        assert pb.policy_summary == "14-day cooldown, 50/week."
        assert pb.expected_effect.startswith("~22%")
        assert pb.fit_score == pytest.approx(0.81)

    def test_sibling_contrast_renders_phrases(self):
        feature_meta = {
            "nps_mean_90d": FeatureMetaRow(
                composite_name="cn", feature_name="nps_mean_90d",
                polarity="high_is_good",
                business_phrase="average NPS score over last 90 days",
            )
        }
        contrast = [{"feature": "nps_mean_90d", "self_value": 2.0, "sibling_value": 8.0}]
        ctx = build_enriched_context(
            _raw_context(),
            feature_meta=feature_meta,
            population_stats={"nps_mean_90d": _stats()},
            sibling_contrast=contrast,
        )
        c = ctx.sibling_contrast[0]
        assert c.business_phrase == "average NPS score over last 90 days"
        assert c.self_phrase == "low"
        assert c.sibling_phrase == "elevated"

    def test_invalid_numeric_value_gracefully_handled(self):
        raw = _raw_context(top_positive_drivers=[
            {"feature": "f", "mean_shap": None, "mean_value": "not-a-number"},
        ])
        ctx = build_enriched_context(raw, feature_meta={}, population_stats={})
        assert ctx.top_positive_drivers[0].value is None
        assert ctx.top_positive_drivers[0].value_phrase == "unknown"

    def test_eligibility_rule_prose_passthrough(self):
        ctx = build_enriched_context(
            _raw_context(),
            feature_meta={},
            population_stats={},
            eligibility_rule_prose="NPS is low (≤ 4) AND tenure is elevated (≥ 365)",
        )
        assert "NPS is low" in ctx.eligibility_rule_prose
