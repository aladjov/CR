"""Tests for ``compile_predicate_prose``."""
from __future__ import annotations

from customer_retention.stages.causal.column_descriptions_writer import ColumnDescriptionRow
from customer_retention.stages.causal.feature_meta_writer import FeatureMetaRow
from customer_retention.stages.causal.interpretation.predicate_prose import (
    compile_predicate_prose,
)
from customer_retention.stages.causal.interpretation.quantile_phrasing import PopulationStats


def _meta(feature_name, **kw):
    return {feature_name: FeatureMetaRow(composite_name="cn", feature_name=feature_name, **kw)}


class TestBaseCases:
    def test_empty_predicate_is_always(self):
        assert compile_predicate_prose({}) == "always"

    def test_true_op(self):
        assert compile_predicate_prose({"op": "true"}) == "always"

    def test_false_op(self):
        assert compile_predicate_prose({"op": "false"}) == "never"


class TestComparisonsWithMetadata:
    def test_gte_with_quantile_lift_and_polarity(self):
        meta = _meta("nps_score", polarity="high_is_good", business_phrase="NPS score")
        stats = {"nps_score": PopulationStats(q05=2, q25=4, q50=7, q75=8, q95=9.5)}
        out = compile_predicate_prose(
            {"op": ">=", "feature": "nps_score", "value": 9},
            feature_meta=meta, population_stats=stats,
        )
        assert "NPS score" in out
        assert "elevated" in out
        assert "at or above" in out

    def test_lt_with_inverted_polarity(self):
        meta = _meta("churn_rate_90d", polarity="high_is_bad", business_phrase="90-day churn rate")
        stats = {"churn_rate_90d": PopulationStats(q05=0.02, q25=0.05, q50=0.1, q75=0.2, q95=0.4)}
        out = compile_predicate_prose(
            {"op": "<", "feature": "churn_rate_90d", "value": 0.01},
            feature_meta=meta, population_stats=stats,
        )
        assert "90-day churn rate" in out
        assert "very high" in out

    def test_comparison_without_stats_falls_back_to_raw_cutoff(self):
        meta = _meta("nps_score", business_phrase="NPS score")
        out = compile_predicate_prose(
            {"op": ">=", "feature": "nps_score", "value": 4},
            feature_meta=meta,
        )
        assert "NPS score" in out
        assert "4" in out
        assert "unknown" not in out

    def test_comparison_without_metadata_uses_raw_column(self):
        out = compile_predicate_prose({"op": "<", "feature": "raw_col_xyz", "value": 42})
        assert "raw_col_xyz" in out
        assert "42" in out


class TestLogicalNesting:
    def test_and_nesting(self):
        meta = {
            **_meta("nps_score", business_phrase="NPS score"),
            **_meta("tenure_days", business_phrase="tenure in days"),
        }
        predicate = {"op": "and", "clauses": [
            {"op": ">=", "feature": "nps_score", "value": 7},
            {"op": "<", "feature": "tenure_days", "value": 30},
        ]}
        out = compile_predicate_prose(predicate, feature_meta=meta)
        assert " AND " in out
        assert "NPS score" in out
        assert "tenure in days" in out

    def test_or_nesting(self):
        predicate = {"op": "or", "clauses": [
            {"op": "<", "feature": "a", "value": 1},
            {"op": ">", "feature": "b", "value": 2},
        ]}
        out = compile_predicate_prose(predicate)
        assert " OR " in out

    def test_and_with_single_clause_returns_clause(self):
        predicate = {"op": "and", "clauses": [
            {"op": ">=", "feature": "x", "value": 1},
        ]}
        out = compile_predicate_prose(predicate)
        assert "x" in out
        assert " AND " not in out

    def test_empty_and_is_always(self):
        assert compile_predicate_prose({"op": "and", "clauses": []}) == "always"

    def test_empty_or_is_never(self):
        assert compile_predicate_prose({"op": "or", "clauses": []}) == "never"

    def test_not_wraps_inner(self):
        out = compile_predicate_prose(
            {"op": "not", "clause": {"op": ">=", "feature": "x", "value": 1}}
        )
        assert out.startswith("NOT (")
        assert "x" in out

    def test_not_with_missing_clause_degrades(self):
        out = compile_predicate_prose({"op": "not"})
        assert "NOT" in out


class TestMembershipAndNull:
    def test_in_renders_set(self):
        out = compile_predicate_prose(
            {"op": "in", "feature": "segment", "values": ["SMB", "Mid"]}
        )
        assert "one of" in out
        assert "SMB" in out
        assert "Mid" in out

    def test_not_in_renders(self):
        out = compile_predicate_prose(
            {"op": "not_in", "feature": "segment", "values": ["SMB"]}
        )
        assert "not one of" in out

    def test_is_null(self):
        meta = _meta("nps_score", business_phrase="NPS score")
        out = compile_predicate_prose(
            {"op": "is_null", "feature": "nps_score"}, feature_meta=meta,
        )
        assert out == "NPS score is missing"

    def test_not_null(self):
        out = compile_predicate_prose({"op": "not_null", "feature": "nps_score"})
        assert out == "nps_score is present"


class TestFallbackSafety:
    def test_unknown_op_degrades_gracefully(self):
        out = compile_predicate_prose({"op": "weird_op", "feature": "x"})
        assert "x" in out
        assert "weird_op" in out

    def test_non_numeric_value_degrades_via_unknown_band(self):
        stats = {"x": PopulationStats(q05=1, q25=2, q50=5, q75=7, q95=9)}
        out = compile_predicate_prose(
            {"op": "<", "feature": "x", "value": "nan"},
            population_stats=stats,
        )
        assert "x" in out

    def test_descriptions_fallback_when_meta_missing(self):
        descriptions = {"nps_score": ColumnDescriptionRow(
            table="t", column_name="nps_score", business_name="NPS score",
        )}
        out = compile_predicate_prose(
            {"op": ">=", "feature": "nps_score", "value": 7},
            column_descriptions=descriptions,
        )
        assert "NPS score" in out
