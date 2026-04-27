"""Tests for ``prose_backfill.backfill_eligibility_prose``.

Cycle 013 P4 surfaced the failure mode this helper closes: c02 derivation
populates ``eligibility_rules_prose`` only at row-creation time. Rows
written before column_descriptions / feature_meta /
feature_population_stats sidecars existed remain ``NULL`` indefinitely.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from customer_retention.stages.causal.interpretation.prose_backfill import (
    ProseBackfillResult,
    _safe_render,
    backfill_eligibility_prose,
)


def _bundle(namespace=None, fm=None, ps=None, cd=None, warnings=None):
    return SimpleNamespace(
        namespace=namespace,
        feature_meta=fm or {},
        population_stats=ps or {},
        column_descriptions=cd or {},
        warnings=warnings or [],
        emit_warnings=lambda *, logger_=None: None,
    )


class TestProseBackfillResult:
    def test_default_warnings_is_empty_list(self):
        r = ProseBackfillResult()
        assert r.warnings == []
        assert r.candidates == 0
        assert r.rendered == 0
        assert r.updated == 0

    def test_summary_string(self):
        r = ProseBackfillResult(candidates=5, rendered=3, updated=3)
        assert "candidates=5" in r.summary()
        assert "rendered=3" in r.summary()
        assert "updated=3" in r.summary()


class TestBackfillEligibilityProse:
    def test_returns_zero_when_namespace_discovery_fails(self):
        spark = MagicMock()
        with patch(
            "customer_retention.stages.causal.interpretation.discovery"
            ".discover_interpretation_sidecars",
            return_value=_bundle(namespace=None, warnings=["discovery failed"]),
        ):
            result = backfill_eligibility_prose(spark, "c.s.eligibility_policy")
        assert result.candidates == 0
        assert result.rendered == 0
        assert result.updated == 0
        assert "discovery failed" in result.warnings

    def test_no_null_rows_short_circuits(self):
        """When every active row already has prose, no MERGE is issued."""
        spark = MagicMock()
        ns = SimpleNamespace(run_id="run-1")
        # collect() returns empty
        empty_df = MagicMock()
        empty_df.collect.return_value = []
        spark.sql.return_value = empty_df
        with patch(
            "customer_retention.stages.causal.interpretation.discovery"
            ".discover_interpretation_sidecars",
            return_value=_bundle(namespace=ns, fm={"f": "x"}, ps={"f": "x"}, cd={"c": "x"}),
        ):
            result = backfill_eligibility_prose(spark, "c.s.eligibility_policy")
        assert result.candidates == 0
        assert result.updated == 0
        # Verify no createDataFrame / DeltaTable calls — short-circuit holds
        assert spark.createDataFrame.call_count == 0

    def test_rendered_rows_merged(self):
        """Rows whose JSON predicate compiles to non-empty prose are MERGEd back."""
        spark = MagicMock()
        ns = SimpleNamespace(run_id="run-1")
        # Two candidate rows — dicts mirror Spark Row's __getitem__ semantics
        rows = [
            {
                "eligibility_policy_id": "pol_1",
                "version": "v1",
                "eligibility_rules": '{"op": "AND", "clauses": []}',
            },
            {
                "eligibility_policy_id": "pol_2",
                "version": "v1",
                "eligibility_rules": '{"op": "OR", "clauses": []}',
            },
        ]

        df = MagicMock()
        df.collect.return_value = rows
        spark.sql.return_value = df

        merged_source = MagicMock()
        spark.createDataFrame.return_value = merged_source

        delta_target = MagicMock()
        with patch(
            "customer_retention.stages.causal.interpretation.discovery"
            ".discover_interpretation_sidecars",
            return_value=_bundle(namespace=ns, fm={"f": "x"}, ps={"f": "x"}, cd={"c": "x"}),
        ), patch(
            "customer_retention.stages.causal.interpretation.predicate_prose"
            ".compile_predicate_prose",
            return_value="rendered prose",
        ), patch(
            "delta.tables.DeltaTable.forName", return_value=delta_target,
        ):
            # Wire the chained-call shape `t.alias().merge().whenMatchedUpdate().execute()`
            chain = MagicMock()
            delta_target.alias.return_value = chain
            chain.merge.return_value = chain
            chain.whenMatchedUpdate.return_value = chain
            chain.execute.return_value = None

            result = backfill_eligibility_prose(spark, "c.s.eligibility_policy")

        assert result.candidates == 2
        assert result.rendered == 2
        assert result.updated == 2
        # createDataFrame called once with the rendered tuples
        assert spark.createDataFrame.call_count == 1
        # MERGE was actually invoked
        chain.merge.assert_called_once()
        chain.whenMatchedUpdate.assert_called_once()
        chain.execute.assert_called_once()

    def test_malformed_json_predicate_skipped(self):
        """Rows whose predicate fails JSON parse don't crash the batch."""
        spark = MagicMock()
        ns = SimpleNamespace(run_id="run-1")
        bad = {
            "eligibility_policy_id": "pol_bad",
            "version": "v1",
            "eligibility_rules": "not-json",
        }
        df = MagicMock()
        df.collect.return_value = [bad]
        spark.sql.return_value = df
        with patch(
            "customer_retention.stages.causal.interpretation.discovery"
            ".discover_interpretation_sidecars",
            return_value=_bundle(namespace=ns, fm={"f": "x"}, ps={"f": "x"}, cd={"c": "x"}),
        ):
            result = backfill_eligibility_prose(spark, "c.s.eligibility_policy")
        assert result.candidates == 1
        assert result.rendered == 0
        assert result.updated == 0


class TestSafeRender:
    def test_invalid_json_returns_none(self):
        bundle = _bundle()
        assert _safe_render("not json", bundle) is None

    def test_empty_prose_returns_none(self):
        bundle = _bundle()
        with patch(
            "customer_retention.stages.causal.interpretation.predicate_prose"
            ".compile_predicate_prose",
            return_value="",
        ):
            assert _safe_render('{"op":"AND","clauses":[]}', bundle) is None

    def test_compile_exception_returns_none(self):
        bundle = _bundle()
        with patch(
            "customer_retention.stages.causal.interpretation.predicate_prose"
            ".compile_predicate_prose",
            side_effect=RuntimeError("oops"),
        ):
            assert _safe_render('{"op":"AND","clauses":[]}', bundle) is None

    def test_happy_path_returns_prose(self):
        bundle = _bundle()
        with patch(
            "customer_retention.stages.causal.interpretation.predicate_prose"
            ".compile_predicate_prose",
            return_value="rendered ok",
        ):
            assert _safe_render('{"op":"AND","clauses":[]}', bundle) == "rendered ok"
