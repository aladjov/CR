"""Unit tests for ``stages/causal/snapshot_writer.py``.

Pure-Python helpers (id derivation, predicate parsing, capacity lookup
construction) are tested directly. The Spark-orchestration paths
(``assign_archetype``, ``evaluate_eligibility``, ``apply_decision_policy``,
``build_eligibility_snapshot``) are exercised through ``pyspark.importorskip``
guards plus a thin local SparkSession when PySpark is available — the
test file degrades to a no-op on CI runners without PySpark.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

from customer_retention.stages.causal.snapshot_writer import (
    SnapshotConfig,
    SnapshotResult,
    _capacity_lookup_column,
    _compile_suppression,
    _holdout_fraction_lookup,
    _join_features_for_predicates,
    _parse_predicate,
    _required_features_union,
    _resolve_definition_version,
    _resolve_risk_tier_thresholds,
    _seed_columns,
    compute_eligibility_id,
    compute_scoring_run_id,
)

# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


class TestComputeScoringRunId:
    def test_deterministic_for_same_inputs(self):
        as_of = datetime(2026, 4, 9, 12, 0, 0, tzinfo=timezone.utc)
        a = compute_scoring_run_id("m", "1", "av", "pv", "dv", as_of)
        b = compute_scoring_run_id("m", "1", "av", "pv", "dv", as_of)
        assert a == b
        assert a.startswith("score_")

    def test_changes_when_anchor_changes(self):
        as_of = datetime(2026, 4, 9, tzinfo=timezone.utc)
        base = compute_scoring_run_id("m", "1", "av", "pv", "dv", as_of)
        for kwargs in [
            dict(model_name="m2"),
            dict(model_version="2"),
            dict(archetype_version="av2"),
            dict(eligibility_policy_version="pv2"),
            dict(decision_policy_version="dv2"),
        ]:
            mutated = compute_scoring_run_id(
                kwargs.get("model_name", "m"),
                kwargs.get("model_version", "1"),
                kwargs.get("archetype_version", "av"),
                kwargs.get("eligibility_policy_version", "pv"),
                kwargs.get("decision_policy_version", "dv"),
                as_of,
            )
            assert mutated != base

    def test_microseconds_truncated(self):
        # Microseconds are dropped so two near-simultaneous runs hash equal
        a = compute_scoring_run_id(
            "m", "1", "av", "pv", "dv",
            datetime(2026, 4, 9, 12, 0, 0, 123, tzinfo=timezone.utc),
        )
        b = compute_scoring_run_id(
            "m", "1", "av", "pv", "dv",
            datetime(2026, 4, 9, 12, 0, 0, 456, tzinfo=timezone.utc),
        )
        assert a == b


class TestComputeEligibilityId:
    def test_deterministic(self):
        a = compute_eligibility_id("score_1", "acct_1", "play_1")
        b = compute_eligibility_id("score_1", "acct_1", "play_1")
        assert a == b
        assert len(a) == 24

    def test_changes_per_account_or_playbook(self):
        base = compute_eligibility_id("score_1", "acct_1", "play_1")
        assert compute_eligibility_id("score_1", "acct_2", "play_1") != base
        assert compute_eligibility_id("score_1", "acct_1", "play_2") != base
        assert compute_eligibility_id("score_2", "acct_1", "play_1") != base


class TestParsePredicate:
    def test_dict_passthrough(self):
        rules = {"op": "and", "clauses": []}
        assert _parse_predicate(rules) is rules

    def test_string_parsed_to_dict(self):
        rules = '{"op": "and", "clauses": []}'
        result = _parse_predicate(rules)
        assert result == {"op": "and", "clauses": []}

    def test_invalid_json_returns_none(self):
        assert _parse_predicate("not valid json") is None

    def test_none_returns_none(self):
        assert _parse_predicate(None) is None

    def test_non_dict_json_returns_none(self):
        assert _parse_predicate("[1, 2, 3]") is None


class TestResolveDefinitionVersion:
    def test_picks_lexicographic_max(self):
        rows = [{"v": "v0.1"}, {"v": "v0.3"}, {"v": "v0.2"}]
        assert _resolve_definition_version(rows, "v") == "v0.3"

    def test_empty_returns_empty_string(self):
        assert _resolve_definition_version([], "v") == ""

    def test_skips_none_and_empty(self):
        rows = [{"v": None}, {"v": ""}, {"v": "v0.1"}]
        assert _resolve_definition_version(rows, "v") == "v0.1"


class TestSeedColumns:
    def test_empty_recipe_uses_account_id(self):
        assert _seed_columns("", ["entity_id", "x"], "entity_id") == ["entity_id"]

    def test_recipe_filters_to_available(self):
        cols = _seed_columns("a, b, c", ["a", "c"], "entity_id")
        assert cols == ["a", "c"]

    def test_no_overlap_falls_back(self):
        cols = _seed_columns("nope, missing", ["entity_id"], "entity_id")
        assert cols == ["entity_id"]


class TestResolveRiskTierThresholds:
    def _config(self, **kwargs) -> SnapshotConfig:
        defaults = dict(
            spark=MagicMock(),
            predictions_fqn="cat.sch.predictions",
            archetype_catalog_fqn="cat.sch.archetype_catalog",
            eligibility_policy_fqn="cat.sch.eligibility_policy",
            decision_policy_fqn="cat.sch.decision_policy",
            snapshot_table_fqn="cat.sch.eligibility_snapshot",
            model_name="m",
            model_version="1",
        )
        defaults.update(kwargs)
        return SnapshotConfig(**defaults)

    def test_config_overrides_win(self):
        cfg = self._config(risk_tier_high=0.8, risk_tier_medium=0.4)
        high, medium = _resolve_risk_tier_thresholds(cfg, {"risk_tier_high_threshold": 0.6})
        assert (high, medium) == (0.8, 0.4)

    def test_falls_back_to_decision_policy_values(self):
        cfg = self._config()
        high, medium = _resolve_risk_tier_thresholds(
            cfg, {"risk_tier_high_threshold": 0.7, "risk_tier_medium_threshold": 0.35}
        )
        assert (high, medium) == (0.7, 0.35)

    def test_falls_back_to_built_in_defaults(self):
        cfg = self._config()
        high, medium = _resolve_risk_tier_thresholds(cfg, None)
        assert (high, medium) == (0.6, 0.3)


# ---------------------------------------------------------------------------
# Spark-driven helpers — gated on pyspark availability
# ---------------------------------------------------------------------------


pyspark = pytest.importorskip("pyspark")


@pytest.fixture(scope="module")
def spark_session():
    from pyspark.sql import SparkSession

    try:
        spark = (
            SparkSession.builder.master("local[2]")
            .appName("snapshot-writer-tests")
            .config("spark.sql.shuffle.partitions", "1")
            .getOrCreate()
        )
    except Exception as exc:  # noqa: BLE001 — environment-specific failures
        pytest.skip(f"Local Spark session unavailable in this environment: {exc}")
        return  # pragma: no cover - pytest.skip raises
    yield spark
    try:
        spark.stop()
    except Exception:  # noqa: BLE001
        pass


# Re-import after the gate so the rest of the file does not crash on CI without
# PySpark — the imports above are pure-Python.
from customer_retention.stages.causal.snapshot_writer import (  # noqa: E402
    _LoadedDefinitions,
    apply_decision_policy,
    assign_archetype,
    build_eligibility_snapshot,
    evaluate_eligibility,
)


class TestAssignArchetype:
    def test_empty_archetypes_fills_nulls(self, spark_session):
        df = spark_session.createDataFrame(
            [(1, 0.1, 0.2), (2, 0.5, 0.6)], ["entity_id", "f1", "f2"]
        )
        out = assign_archetype(df, [])
        rows = out.collect()
        assert all(r["archetype_id"] is None for r in rows)
        assert all(r["archetype_version"] is None for r in rows)

    def test_nearest_centroid_assignment(self, spark_session):
        df = spark_session.createDataFrame(
            [
                ("a1", 0.0, 0.0),
                ("a2", 1.0, 1.0),
                ("a3", 5.0, 5.0),
            ],
            ["entity_id", "f1", "f2"],
        )
        archetypes = [
            {
                "archetype_id": "near_origin",
                "archetype_version": "v1",
                "centroid_vector_raw": [0.0, 0.0],
                "centroid_feature_order": ["f1", "f2"],
            },
            {
                "archetype_id": "far",
                "archetype_version": "v1",
                "centroid_vector_raw": [10.0, 10.0],
                "centroid_feature_order": ["f1", "f2"],
            },
        ]
        out = assign_archetype(df, archetypes)
        result = {row["entity_id"]: row["archetype_id"] for row in out.collect()}
        assert result["a1"] == "near_origin"
        assert result["a2"] == "near_origin"
        assert result["a3"] == "near_origin"  # 5,5 closer to 0,0 than to 10,10

    def test_skip_archetype_with_missing_centroid(self, spark_session):
        df = spark_session.createDataFrame([("a", 0.0, 0.0)], ["entity_id", "f1", "f2"])
        archetypes = [
            {
                "archetype_id": "incomplete",
                "archetype_version": "v1",
                "centroid_vector_raw": None,
                "centroid_feature_order": ["f1"],
            },
            {
                "archetype_id": "good",
                "archetype_version": "v1",
                "centroid_vector_raw": [0.0, 0.0],
                "centroid_feature_order": ["f1", "f2"],
            },
        ]
        out = assign_archetype(df, archetypes)
        rows = out.collect()
        assert rows[0]["archetype_id"] == "good"

    def test_missing_feature_columns_handled(self, spark_session):
        # Centroid references columns the predictions DF does not have —
        # the missing column is silently skipped per archetype.
        df = spark_session.createDataFrame([("a", 0.5)], ["entity_id", "f1"])
        archetypes = [
            {
                "archetype_id": "uses_missing_feature",
                "archetype_version": "v1",
                "centroid_vector_raw": [0.0, 0.0],
                "centroid_feature_order": ["f1", "f_missing"],
            }
        ]
        out = assign_archetype(df, archetypes)
        rows = out.collect()
        assert rows[0]["archetype_id"] == "uses_missing_feature"

    def test_scaled_distance_overrides_unscaled_winner(self, spark_session):
        """Scaling by centroid_feature_scales changes the winner.

        Customer (f1=10, f2=50) without scaling is dominated by f2:
          dist(a1_near_f1, centroid=(9, 1000))  ≈ sqrt(1 + 902500) ≈ 950  ← far
          dist(a2_near_f2, centroid=(0, 51))    ≈ sqrt(100 + 1)    ≈ 10   ← near

        After scaling by (1, 500), the 950-unit f2 gap shrinks to ±1.9 scaled
        units and a1's f1 closeness dominates:
          scaled_dist(a1) = sqrt((10-9)²/1² + (50-1000)²/500²) ≈ sqrt(1 + 3.61) ≈ 2.14
          scaled_dist(a2) = sqrt((10-0)²/1² + (50-51)²/500²)   ≈ sqrt(100 + 0.000004) ≈ 10
        """
        df = spark_session.createDataFrame(
            [("cust", 10.0, 50.0)], ["entity_id", "f1", "f2"]
        )
        archetypes = [
            {
                "archetype_id": "a1_near_f1",
                "archetype_version": "v1",
                "centroid_vector_raw": [9.0, 1000.0],
                "centroid_feature_order": ["f1", "f2"],
                "centroid_feature_scales": [1.0, 500.0],
            },
            {
                "archetype_id": "a2_near_f2",
                "archetype_version": "v1",
                "centroid_vector_raw": [0.0, 51.0],
                "centroid_feature_order": ["f1", "f2"],
                "centroid_feature_scales": [1.0, 500.0],
            },
        ]
        out = assign_archetype(df, archetypes)
        rows = out.collect()
        # Without scaling, a2 would win (raw f2-distance dominates).
        # With scaling (scale_f2=500), a1 wins because the f2 gap is divided out.
        assert rows[0]["archetype_id"] == "a1_near_f1"

    def test_missing_scales_defaults_to_unscaled(self, spark_session):
        """When centroid_feature_scales is absent, behaviour matches plain Euclidean."""
        df = spark_session.createDataFrame(
            [("a1", 0.0, 0.0), ("a3", 5.0, 5.0)], ["entity_id", "f1", "f2"]
        )
        archetypes = [
            {
                "archetype_id": "near_origin",
                "archetype_version": "v1",
                "centroid_vector_raw": [0.0, 0.0],
                "centroid_feature_order": ["f1", "f2"],
                # centroid_feature_scales absent — should default gracefully
            },
            {
                "archetype_id": "far",
                "archetype_version": "v1",
                "centroid_vector_raw": [10.0, 10.0],
                "centroid_feature_order": ["f1", "f2"],
            },
        ]
        out = assign_archetype(df, archetypes)
        result = {r["entity_id"]: r["archetype_id"] for r in out.collect()}
        assert result["a1"] == "near_origin"
        assert result["a3"] == "near_origin"  # (5,5) is closer to origin than (10,10)


class TestEvaluateEligibility:
    def _df(self, spark):
        return spark.createDataFrame(
            [
                ("a1", 100, 0.9),
                ("a2", 50, 0.4),
                ("a3", 200, 0.1),
            ],
            ["entity_id", "tenure_days", "churn_probability"],
        )

    def test_predicate_filter_fans_out_per_policy(self, spark_session):
        df = self._df(spark_session)
        policies: List[Dict[str, Any]] = [
            {
                "eligibility_policy_id": "ep1",
                "version": "v1",
                "playbook_id": "high_risk_outreach",
                "playbook_version": "v1",
                "eligibility_rules": {
                    "op": ">=",
                    "feature": "churn_probability",
                    "value": 0.5,
                },
                "eligibility_rules_sql": "`churn_probability` >= 0.5",
            },
            {
                "eligibility_policy_id": "ep2",
                "version": "v1",
                "playbook_id": "tenure_check",
                "playbook_version": "v1",
                "eligibility_rules": {
                    "op": ">=",
                    "feature": "tenure_days",
                    "value": 100,
                },
                "eligibility_rules_sql": "`tenure_days` >= 100",
            },
        ]
        out = evaluate_eligibility(df, policies)
        rows = out.collect()
        # a1 matches both policies → 2 rows; a2 none; a3 matches tenure → 1 row
        triples = sorted((r["entity_id"], r["playbook_id"]) for r in rows)
        assert triples == [
            ("a1", "high_risk_outreach"),
            ("a1", "tenure_check"),
            ("a3", "tenure_check"),
        ]
        # All annotation columns must be present
        assert {"playbook_id", "playbook_version", "eligibility_policy_id",
                "eligibility_policy_version", "eligibility_evidence"}.issubset(set(out.columns))

    def test_empty_policies_returns_empty_with_schema(self, spark_session):
        df = self._df(spark_session)
        out = evaluate_eligibility(df, [])
        assert out.count() == 0
        assert "playbook_id" in out.columns
        assert "eligibility_policy_id" in out.columns

    def test_skips_unparseable_policy(self, spark_session):
        df = self._df(spark_session)
        policies = [
            {
                "eligibility_policy_id": "broken",
                "version": "v1",
                "playbook_id": "x",
                "playbook_version": "v1",
                "eligibility_rules": "not valid json",
            },
            {
                "eligibility_policy_id": "ok",
                "version": "v1",
                "playbook_id": "y",
                "playbook_version": "v1",
                "eligibility_rules": {"op": "true"},
            },
        ]
        out = evaluate_eligibility(df, policies)
        assert {r["playbook_id"] for r in out.collect()} == {"y"}


class TestApplyDecisionPolicy:
    def _eligible_df(self, spark):
        from pyspark.sql import functions as F  # noqa: N812

        rows = [
            ("a1", "p1", 0.9),
            ("a1", "p2", 0.9),
            ("a2", "p1", 0.4),
            ("a3", "p1", 0.05),
        ]
        df = spark.createDataFrame(rows, ["entity_id", "playbook_id", "churn_probability"])
        # evaluate_eligibility would normally add these columns; mimic them
        return (
            df.withColumn("playbook_version", F.lit("v1"))
            .withColumn("eligibility_policy_id", F.lit("ep1"))
            .withColumn("eligibility_policy_version", F.lit("v1"))
            .withColumn("eligibility_evidence", F.lit(""))
            .withColumn("archetype_id", F.lit("arch1"))
            .withColumn("archetype_version", F.lit("v1"))
        )

    def test_risk_tier_assignment(self, spark_session):
        df = self._eligible_df(spark_session)
        out = apply_decision_policy(df, None, risk_tier_high=0.6, risk_tier_medium=0.3)
        tiers = {r["entity_id"]: r["risk_tier"] for r in out.collect() if r["playbook_id"] == "p1"}
        assert tiers["a1"] == "High"
        assert tiers["a2"] == "Medium"
        assert tiers["a3"] == "Low"

    def test_recommended_excludes_holdout(self, spark_session):
        df = self._eligible_df(spark_session)
        # Force everything into holdout via 100% holdout fraction
        policy = {"holdout_fractions_by_playbook": {"p1": 1.0, "p2": 1.0}}
        out = apply_decision_policy(df, policy)
        rows = out.collect()
        assert all(r["is_holdout"] for r in rows)
        assert all(not r["recommended"] for r in rows)

    def test_capacity_marks_overflow_as_suppressed(self, spark_session):
        df = self._eligible_df(spark_session)
        policy = {
            "holdout_fractions_by_playbook": {"p1": 0.0, "p2": 0.0},
            "capacity_by_playbook": {"p1": 1},
        }
        out = apply_decision_policy(df, policy)
        # p1 sorted by churn descending: a1 (0.9), a2 (0.4), a3 (0.05)
        # capacity 1 -> only top row recommended
        p1_rows = [r for r in out.collect() if r["playbook_id"] == "p1"]
        recommended_p1 = sorted(r["entity_id"] for r in p1_rows if r["recommended"])
        suppressed_p1 = sorted(r["entity_id"] for r in p1_rows if not r["recommended"])
        assert recommended_p1 == ["a1"]
        assert sorted(suppressed_p1) == ["a2", "a3"]
        for row in p1_rows:
            if row["entity_id"] != "a1":
                assert row["playbook_suppressed_reason"] == "capacity_exceeded"

    def test_eligible_playbook_count_per_account(self, spark_session):
        df = self._eligible_df(spark_session)
        out = apply_decision_policy(df, None)
        counts = {r["entity_id"]: r["eligible_playbook_count"] for r in out.collect()}
        assert counts["a1"] == 2  # a1 has p1 and p2
        assert counts["a2"] == 1

    def test_policy_rank_among_eligible_uses_fit_score(self, spark_session):
        """Regression for the alphabetical-first failure mode.

        churn_probability is replicated across (entity, playbook) rows so
        it doesn't differentiate playbooks within a partition. Without
        fit_score in the sort, every entity received the alphabetically
        first eligible playbook regardless of c02's matching. With the
        fix, the higher fit_score wins even when its playbook_id sorts
        later alphabetically.
        """
        from pyspark.sql import functions as F  # noqa: N812

        rows = [
            ("a1", "alpha_first", 0.9, 0.10),  # alpha first, weak fit
            ("a1", "zulu_last",   0.9, 0.80),  # alpha last, strong fit -> should win
        ]
        df = spark_session.createDataFrame(
            rows, ["entity_id", "playbook_id", "churn_probability", "fit_score"]
        )
        df = (
            df.withColumn("playbook_version", F.lit("v1"))
            .withColumn("eligibility_policy_id", F.lit("ep1"))
            .withColumn("eligibility_policy_version", F.lit("v1"))
            .withColumn("eligibility_evidence", F.lit(""))
            .withColumn("archetype_id", F.lit("arch1"))
            .withColumn("archetype_version", F.lit("v1"))
        )
        out = apply_decision_policy(df, None)
        ranks = {r["playbook_id"]: r["policy_rank_among_eligible"] for r in out.collect()}
        assert ranks["zulu_last"] == 1, (
            f"strong fit_score must beat alphabetical first; got ranks={ranks}"
        )
        assert ranks["alpha_first"] == 2

    def test_policy_rank_handles_null_fit_score(self, spark_session):
        """Pre-migration rows have NULL fit_score. NULLS LAST keeps a
        scored row above an unscored one even when the unscored row's
        playbook_id sorts earlier."""
        from pyspark.sql import functions as F  # noqa: N812

        rows = [
            ("a1", "alpha_unscored", 0.9, None),
            ("a1", "zulu_scored",    0.9, 0.30),
        ]
        df = spark_session.createDataFrame(
            rows, ["entity_id", "playbook_id", "churn_probability", "fit_score"]
        )
        df = (
            df.withColumn("playbook_version", F.lit("v1"))
            .withColumn("eligibility_policy_id", F.lit("ep1"))
            .withColumn("eligibility_policy_version", F.lit("v1"))
            .withColumn("eligibility_evidence", F.lit(""))
            .withColumn("archetype_id", F.lit("arch1"))
            .withColumn("archetype_version", F.lit("v1"))
        )
        out = apply_decision_policy(df, None)
        ranks = {r["playbook_id"]: r["policy_rank_among_eligible"] for r in out.collect()}
        assert ranks["zulu_scored"] == 1
        assert ranks["alpha_unscored"] == 2

    def test_policy_rank_falls_back_when_fit_score_column_absent(self, spark_session):
        """When the input frame predates the fit_score column, ranking
        falls back to (churn_probability DESC, playbook_id) — same
        behaviour as before the fix. Guards against breaking older
        callers that don't add fit_score."""
        df = self._eligible_df(spark_session)
        assert "fit_score" not in df.columns  # precondition
        out = apply_decision_policy(df, None)
        a1_ranks = {
            r["playbook_id"]: r["policy_rank_among_eligible"]
            for r in out.collect() if r["entity_id"] == "a1"
        }
        # Both rows have churn_probability=0.9, so alphabetical decides.
        assert a1_ranks["p1"] == 1
        assert a1_ranks["p2"] == 2


# ---------------------------------------------------------------------------
# Capacity / holdout column-construction helpers (driver-side)
# ---------------------------------------------------------------------------


class TestCapacityLookup:
    def test_returns_typed_null_for_empty(self, spark_session):
        col = _capacity_lookup_column({})
        df = spark_session.createDataFrame([("p1",)], ["playbook_id"])
        result = df.withColumn("cap", col).collect()[0]
        assert result["cap"] is None

    def test_maps_known_playbooks(self, spark_session):
        col = _capacity_lookup_column({"p1": 5, "p2": 10})
        df = spark_session.createDataFrame(
            [("p1",), ("p2",), ("p3",)], ["playbook_id"]
        )
        result = {row["playbook_id"]: row["cap"] for row in df.withColumn("cap", col).collect()}
        assert result == {"p1": 5, "p2": 10, "p3": None}


class TestHoldoutFractionLookup:
    def test_default_fraction_when_empty(self, spark_session):
        col = _holdout_fraction_lookup({})
        df = spark_session.createDataFrame([("p1",)], ["playbook_id"])
        result = df.withColumn("frac", col).collect()[0]
        assert result["frac"] == pytest.approx(0.1)

    def test_per_playbook_override(self, spark_session):
        col = _holdout_fraction_lookup({"p1": 0.2, "p2": 0.5})
        df = spark_session.createDataFrame([("p1",), ("p2",), ("p3",)], ["playbook_id"])
        result = {row["playbook_id"]: row["frac"] for row in df.withColumn("frac", col).collect()}
        assert result["p1"] == pytest.approx(0.2)
        assert result["p2"] == pytest.approx(0.5)
        assert result["p3"] == pytest.approx(0.1)


class TestCompileSuppression:
    def test_empty_returns_false_predicate(self, spark_session):
        column, reason = _compile_suppression(None)
        df = spark_session.createDataFrame([(1,)], ["x"])
        assert df.withColumn("s", column).collect()[0]["s"] is False
        assert reason == "suppressed"

    def test_predicate_with_reason(self, spark_session):
        rules = {
            "reason": "manual_block",
            "predicate": {"op": "==", "feature": "playbook_id", "value": "p1"},
        }
        column, reason = _compile_suppression(rules)
        df = spark_session.createDataFrame([("p1",), ("p2",)], ["playbook_id"])
        rows = df.withColumn("s", column).collect()
        result = {r["playbook_id"]: r["s"] for r in rows}
        assert result == {"p1": True, "p2": False}
        assert reason == "manual_block"


# ---------------------------------------------------------------------------
# SnapshotResult formatting
# ---------------------------------------------------------------------------


class TestSnapshotResultSummary:
    def test_summary_has_all_counts(self):
        result = SnapshotResult(
            scoring_run_id="score_abc",
            as_of_date=datetime(2026, 4, 9, tzinfo=timezone.utc),
            total_eligible_rows=10,
            recommended_rows=6,
            holdout_rows=2,
            suppressed_rows=2,
            archetypes_considered=4,
            policies_considered=8,
        )
        text = result.summary()
        assert "score_abc" in text
        assert "10 eligible" in text
        assert "6 recommended" in text
        assert "2 holdout" in text
        assert "4 archetypes" in text
        assert "8 policies" in text


# ---------------------------------------------------------------------------
# End-to-end orchestrator wiring (D5 from Phase 3 review)
# ---------------------------------------------------------------------------


class TestBuildEligibilitySnapshotE2E:
    """Exercises the full orchestrator path with real Spark transforms.

    Delta I/O is patched so no live catalog is required, but every Spark
    transform stage (assign_archetype → evaluate_eligibility →
    apply_decision_policy → _shape_snapshot_rows → _compute_snapshot_counts)
    runs for real. This catches column-name mismatches between stages that
    unit-testing individual helpers cannot detect.
    """

    def _definitions(self) -> "_LoadedDefinitions":
        return _LoadedDefinitions(
            archetypes=[
                {
                    "archetype_id": "arch_default",
                    "archetype_version": "v1",
                    "centroid_vector_raw": [5.0, 50.0],
                    "centroid_feature_order": ["f1", "f2"],
                }
            ],
            policies=[
                {
                    "eligibility_policy_id": "ep1",
                    "version": "v1",
                    "playbook_id": "high_risk_outreach",
                    "playbook_version": "v1",
                    "eligibility_rules": {
                        "op": ">=",
                        "feature": "churn_probability",
                        "value": 0.3,
                    },
                }
            ],
            decision_policy={
                "decision_policy_id": "dp1",
                "version": "v1",
                "holdout_fractions_by_playbook": {},
                "capacity_by_playbook": {},
                "suppression_rules": None,
                "holdout_seed_recipe": "",
                "risk_tier_high_threshold": 0.6,
                "risk_tier_medium_threshold": 0.3,
            },
            archetype_version="v1",
            eligibility_policy_version="v1",
            decision_policy_version="v1",
            decision_policy_id="dp1",
        )

    def test_happy_path_returns_correct_counts(self, spark_session):
        """Full pipeline: 3 accounts, policy matches 2 → SnapshotResult with
        correct totals and all expected snapshot columns present."""
        from unittest.mock import patch

        import customer_retention.stages.causal.snapshot_writer as sw

        config = SnapshotConfig(
            spark=spark_session,
            predictions_fqn="mock.predictions",
            archetype_catalog_fqn="mock.archetype_catalog",
            eligibility_policy_fqn="mock.eligibility_policy",
            decision_policy_fqn="mock.decision_policy",
            snapshot_table_fqn="mock.eligibility_snapshot",
            model_name="retention_model",
            model_version="v1",
            as_of_date=datetime(2026, 4, 9, 0, 0, 0, tzinfo=timezone.utc),
        )

        # Predictions: a1 and a2 match policy (churn ≥ 0.3); a3 does not.
        predictions_df = spark_session.createDataFrame(
            [
                ("a1", 0.9, 10.0, 100.0),
                ("a2", 0.4, 5.0, 50.0),
                ("a3", 0.05, 2.0, 20.0),
            ],
            ["entity_id", "churn_probability", "f1", "f2"],
        )

        captured: Dict[str, Any] = {}

        def fake_write_snapshot(spark, snapshot_df, table_fqn):
            captured["columns"] = set(snapshot_df.columns)
            # Stash the shaped snapshot so the count-from-table read-back
            # below can be served without a real Delta write.
            captured["snapshot_df"] = snapshot_df
            return {"target": table_fqn}

        def fake_compute_counts_from_table(spark, snapshot_table_fqn, scoring_run_id):  # noqa: ARG001
            # Same code path counts would take after a real Delta read-back.
            return sw._compute_snapshot_counts(captured["snapshot_df"])

        with (
            patch.object(sw, "_load_active_definitions", return_value=self._definitions()),
            patch.object(sw, "_load_latest_predictions", return_value=predictions_df),
            patch.object(
                sw, "_scratch_eligible_round_trip",
                side_effect=lambda spark, df, fqn: df,
            ),
            patch.object(sw, "write_snapshot", side_effect=fake_write_snapshot),
            patch.object(
                sw, "_compute_snapshot_counts_from_table",
                side_effect=fake_compute_counts_from_table,
            ),
        ):
            result = build_eligibility_snapshot(config)

        # Exactly 2 rows pass the eligibility predicate
        assert result.total_eligible_rows == 2
        assert result.scoring_run_id.startswith("score_")
        assert result.archetypes_considered == 1
        assert result.policies_considered == 1
        assert result.decision_policy_id == "dp1"

        # Counts partition correctly: recommended + holdout + suppressed == total
        assert (
            result.recommended_rows + result.holdout_rows + result.suppressed_rows
            == result.total_eligible_rows
        )

        # All schema columns must be present in the snapshot passed to write_snapshot
        required_cols = {
            "eligibility_id", "scoring_run_id", "as_of_date", "entity_id",
            "playbook_id", "archetype_id", "archetype_version", "churn_probability",
            "risk_tier", "recommended", "is_holdout", "holdout_stratum",
            "holdout_seed", "playbook_suppressed_reason", "decision_policy_id",
            "model_name", "model_version", "priority_rank_within_cohort",
            "policy_rank_among_eligible", "eligible_playbook_count",
        }
        assert required_cols.issubset(captured["columns"]), (
            f"missing snapshot columns: {required_cols - captured['columns']}"
        )

    def test_no_cache_call_and_counts_read_from_snapshot_table(self, spark_session):
        """Root-cause fix for Databricks shared/serverless clusters that reject
        ``df.cache()`` via Spark Connect with
        ``[NOT_SUPPORTED_WITH_SERVERLESS] PERSIST TABLE is not supported``.
        The orchestrator must NOT call cache anywhere — counts come from a
        read-back of the just-written snapshot Delta table filtered to this
        run's scoring_run_id."""
        from unittest.mock import patch

        import customer_retention.stages.causal.snapshot_writer as sw

        config = SnapshotConfig(
            spark=spark_session,
            predictions_fqn="mock.predictions",
            archetype_catalog_fqn="mock.archetype_catalog",
            eligibility_policy_fqn="mock.eligibility_policy",
            decision_policy_fqn="mock.decision_policy",
            snapshot_table_fqn="mock.eligibility_snapshot",
            model_name="m", model_version="v1",
            as_of_date=datetime(2026, 4, 9, 0, 0, 0, tzinfo=timezone.utc),
        )

        predictions_df = spark_session.createDataFrame(
            [("a1", 0.9, 10.0, 100.0), ("a2", 0.4, 5.0, 50.0)],
            ["entity_id", "churn_probability", "f1", "f2"],
        )

        # Track whether anybody calls .cache() — fail loudly if so.
        cache_calls: list[int] = []
        original_cache = type(predictions_df).cache

        def _tripwire_cache(self):
            cache_calls.append(1)
            return original_cache(self)

        captured: Dict[str, Any] = {}

        def fake_write_snapshot(spark, snapshot_df, table_fqn):
            captured["snapshot_df"] = snapshot_df
            captured["scoring_run_id"] = snapshot_df.select("scoring_run_id").first()[0]
            return {"target": table_fqn}

        count_calls: list[tuple] = []

        def fake_count_from_table(spark, table_fqn, scoring_run_id):
            count_calls.append((table_fqn, scoring_run_id))
            return sw._compute_snapshot_counts(captured["snapshot_df"])

        with (
            patch.object(type(predictions_df), "cache", _tripwire_cache),
            patch.object(sw, "_load_active_definitions", return_value=self._definitions()),
            patch.object(sw, "_load_latest_predictions", return_value=predictions_df),
            patch.object(
                sw, "_scratch_eligible_round_trip",
                side_effect=lambda spark, df, fqn: df,
            ),
            patch.object(sw, "write_snapshot", side_effect=fake_write_snapshot),
            patch.object(sw, "_compute_snapshot_counts_from_table", side_effect=fake_count_from_table),
        ):
            result = build_eligibility_snapshot(config)

        assert cache_calls == [], (
            f"build_eligibility_snapshot must not call .cache() — caught {len(cache_calls)} call(s). "
            "Spark Connect on shared/serverless clusters rejects cache via "
            "[NOT_SUPPORTED_WITH_SERVERLESS] PERSIST TABLE."
        )
        assert len(count_calls) == 1
        called_fqn, called_run_id = count_calls[0]
        assert called_fqn == "mock.eligibility_snapshot"
        assert called_run_id == captured["scoring_run_id"], (
            "counts read-back must filter on the THIS run's scoring_run_id so a "
            "second concurrent c05 doesn't double-count rows in the same table."
        )
        assert result.total_eligible_rows == 2

    def test_scratch_round_trip_called_between_evaluate_and_decision(self, spark_session):
        """Regression guard: build_eligibility_snapshot must collapse the
        ``evaluate_eligibility`` union plan via a scratch-Delta round-trip
        before ``apply_decision_policy`` runs. Without this break, the first
        ``.columns`` access in apply_decision_policy triggers a Spark Connect
        AnalyzePlan RPC that serializes the full N-policy unioned protobuf
        and OOMs the driver kernel on high policy counts (observed at 614)."""
        from unittest.mock import patch

        import customer_retention.stages.causal.snapshot_writer as sw

        config = SnapshotConfig(
            spark=spark_session,
            predictions_fqn="mock.predictions",
            archetype_catalog_fqn="mock.archetype_catalog",
            eligibility_policy_fqn="mock.eligibility_policy",
            decision_policy_fqn="mock.decision_policy",
            snapshot_table_fqn="mock.eligibility_snapshot",
            model_name="m", model_version="v1",
            as_of_date=datetime(2026, 4, 9, 0, 0, 0, tzinfo=timezone.utc),
        )
        predictions_df = spark_session.createDataFrame(
            [("a1", 0.9, 10.0, 100.0), ("a2", 0.4, 5.0, 50.0)],
            ["entity_id", "churn_probability", "f1", "f2"],
        )

        captured: Dict[str, Any] = {}
        round_trip_calls: list[tuple] = []

        def fake_round_trip(spark, df, snapshot_table_fqn):
            round_trip_calls.append((id(df), snapshot_table_fqn))
            return df

        def fake_write_snapshot(spark, snapshot_df, table_fqn):
            captured["snapshot_df"] = snapshot_df
            return {"target": table_fqn}

        def fake_count_from_table(spark, table_fqn, scoring_run_id):  # noqa: ARG001
            return sw._compute_snapshot_counts(captured["snapshot_df"])

        with (
            patch.object(sw, "_load_active_definitions", return_value=self._definitions()),
            patch.object(sw, "_load_latest_predictions", return_value=predictions_df),
            patch.object(sw, "_scratch_eligible_round_trip", side_effect=fake_round_trip),
            patch.object(sw, "write_snapshot", side_effect=fake_write_snapshot),
            patch.object(sw, "_compute_snapshot_counts_from_table", side_effect=fake_count_from_table),
        ):
            build_eligibility_snapshot(config)

        assert len(round_trip_calls) == 1, (
            f"_scratch_eligible_round_trip must be called exactly once between "
            f"evaluate_eligibility and apply_decision_policy; got {len(round_trip_calls)} call(s)."
        )
        _, called_fqn = round_trip_calls[0]
        assert called_fqn == "mock.eligibility_snapshot"


# ---------------------------------------------------------------------------
# _required_features_union + _join_features_for_predicates
# ---------------------------------------------------------------------------


class TestRequiredFeaturesUnion:
    def test_union_across_policies(self):
        policies = [
            {"requires_features": ["active_span_days", "event_count_180d"]},
            {"requires_features": ["event_count_180d", "regularity_score"]},
        ]
        assert _required_features_union(policies) == {
            "active_span_days", "event_count_180d", "regularity_score",
        }

    def test_empty_policies_returns_empty(self):
        assert _required_features_union([]) == set()

    def test_handles_missing_requires_features(self):
        assert _required_features_union([{}, {"requires_features": None}]) == set()


class TestJoinFeaturesForPredicates:
    """Drives the fix for UNRESOLVED_COLUMN on missing raw features in c05."""

    def _cfg(self, **overrides) -> SnapshotConfig:
        base = dict(
            spark=MagicMock(),
            predictions_fqn="cat.sch.predictions",
            archetype_catalog_fqn="cat.sch.archetype_catalog",
            eligibility_policy_fqn="cat.sch.eligibility_policy",
            decision_policy_fqn="cat.sch.decision_policy",
            snapshot_table_fqn="cat.sch.eligibility_snapshot",
            model_name="m",
            model_version="1",
        )
        base.update(overrides)
        return SnapshotConfig(**base)

    def _enriched_df(self, columns):
        df = MagicMock(name="EnrichedDF")
        df.columns = list(columns)
        df.join = MagicMock(return_value=df)
        return df

    def test_no_op_when_no_policy_requires_features(self):
        spark = MagicMock()
        enriched = self._enriched_df(["entity_id", "churn_probability"])
        cfg = self._cfg(spark=spark)
        result = _join_features_for_predicates(
            spark=spark, scored_df=enriched, config=cfg, policies=[]
        )
        assert result is enriched
        spark.table.assert_not_called()

    def test_raises_when_gold_missing_and_features_needed(self):
        spark = MagicMock()
        cfg = self._cfg(spark=spark, gold_features_fqn=None)
        enriched = self._enriched_df(["entity_id"])
        policies = [{"requires_features": ["active_span_days"]}]
        with pytest.raises(RuntimeError, match="gold_features_fqn is not set"):
            _join_features_for_predicates(
                spark=spark, scored_df=enriched, config=cfg, policies=policies
            )

    def test_raises_when_gold_table_does_not_exist(self):
        spark = MagicMock()
        spark.catalog.tableExists.return_value = False
        cfg = self._cfg(spark=spark, gold_features_fqn="cat.sch.gold")
        enriched = self._enriched_df(["entity_id"])
        policies = [{"requires_features": ["active_span_days"]}]
        with pytest.raises(RuntimeError, match="does not exist"):
            _join_features_for_predicates(
                spark=spark, scored_df=enriched, config=cfg, policies=policies
            )

    def test_raises_when_required_features_missing_from_gold(self):
        spark = MagicMock()
        spark.catalog.tableExists.return_value = True
        gold_df = MagicMock()
        gold_df.columns = ["entity_id", "active_span_days"]
        spark.table.return_value = gold_df
        enriched = self._enriched_df(["entity_id", "churn_probability"])
        cfg = self._cfg(spark=spark, gold_features_fqn="cat.sch.gold")
        policies = [{"requires_features": ["active_span_days", "missing_col"]}]
        with pytest.raises(RuntimeError, match="missing_col"):
            _join_features_for_predicates(
                spark=spark, scored_df=enriched, config=cfg, policies=policies
            )

    def test_raises_when_entity_key_missing_from_gold(self):
        spark = MagicMock()
        spark.catalog.tableExists.return_value = True
        gold_df = MagicMock()
        gold_df.columns = ["some_other_key", "active_span_days"]  # no entity_id
        spark.table.return_value = gold_df
        enriched = self._enriched_df(["entity_id"])
        cfg = self._cfg(spark=spark, gold_features_fqn="cat.sch.gold")
        policies = [{"requires_features": ["active_span_days"]}]
        with pytest.raises(RuntimeError, match="entity_id"):
            _join_features_for_predicates(
                spark=spark, scored_df=enriched, config=cfg, policies=policies
            )

    def test_happy_path_projects_and_joins(self):
        """Validates projection pushdown + left join semantics."""
        spark = MagicMock()
        spark.catalog.tableExists.return_value = True
        gold_df = MagicMock()
        gold_df.columns = [
            "entity_id", "active_span_days", "event_count_180d", "noise_col",
        ]
        # Track what columns we select before joining
        selected_df = MagicMock()
        gold_df.select = MagicMock(return_value=selected_df)
        spark.table.return_value = gold_df

        enriched = self._enriched_df(["entity_id", "churn_probability"])
        cfg = self._cfg(spark=spark, gold_features_fqn="cat.sch.gold")
        policies = [
            {"requires_features": ["active_span_days"]},
            {"requires_features": ["event_count_180d"]},
        ]
        result = _join_features_for_predicates(
            spark=spark, scored_df=enriched, config=cfg, policies=policies
        )
        # Should select only entity_key + required features (not noise_col)
        selected_args = gold_df.select.call_args[0]
        assert "entity_id" in selected_args
        assert "active_span_days" in selected_args
        assert "event_count_180d" in selected_args
        assert "noise_col" not in selected_args
        # Left join on entity_id
        enriched.join.assert_called_once_with(selected_df, on="entity_id", how="left")
        assert result is enriched


class TestArchetypeFeatureUnion:
    """The archetype centroid features are also needed on the join projection."""

    def test_union_across_archetypes(self):
        from customer_retention.stages.causal.snapshot_writer import (
            _archetype_feature_union,
        )
        archetypes = [
            {"centroid_feature_order": ["event_count_180d", "regularity_score"]},
            {"centroid_feature_order": ["active_span_days", "regularity_score"]},
        ]
        assert _archetype_feature_union(archetypes) == {
            "event_count_180d", "regularity_score", "active_span_days",
        }

    def test_handles_missing_centroid_feature_order(self):
        from customer_retention.stages.causal.snapshot_writer import (
            _archetype_feature_union,
        )
        assert _archetype_feature_union([{}, {"centroid_feature_order": None}]) == set()

    def test_empty_archetypes_returns_empty(self):
        from customer_retention.stages.causal.snapshot_writer import (
            _archetype_feature_union,
        )
        assert _archetype_feature_union([]) == set()


class TestJoinFeaturesProjectsBothUnions:
    """The join must project the union of policy + archetype features."""

    def _cfg(self, **overrides) -> SnapshotConfig:
        base = dict(
            spark=MagicMock(),
            predictions_fqn="cat.sch.predictions",
            archetype_catalog_fqn="cat.sch.archetype_catalog",
            eligibility_policy_fqn="cat.sch.eligibility_policy",
            decision_policy_fqn="cat.sch.decision_policy",
            snapshot_table_fqn="cat.sch.eligibility_snapshot",
            model_name="m",
            model_version="1",
            gold_features_fqn="cat.sch.gold",
        )
        base.update(overrides)
        return SnapshotConfig(**base)

    def test_projects_archetype_features_even_without_policy_requirements(self):
        """assign_archetype needs centroid features on input — join must provide."""
        spark = MagicMock()
        spark.catalog.tableExists.return_value = True
        gold_df = MagicMock()
        gold_df.columns = ["entity_id", "event_count_180d", "regularity_score"]
        gold_df.select = MagicMock(return_value=MagicMock())
        spark.table.return_value = gold_df

        scored = MagicMock()
        scored.columns = ["entity_id", "churn_probability"]
        scored.join = MagicMock(return_value=scored)

        cfg = self._cfg(spark=spark)
        policies = [{"requires_features": []}]  # no policy features
        archetypes = [{"centroid_feature_order": ["event_count_180d", "regularity_score"]}]

        _join_features_for_predicates(
            spark=spark, scored_df=scored, config=cfg,
            policies=policies, archetypes=archetypes,
        )
        selected_args = gold_df.select.call_args[0]
        assert "event_count_180d" in selected_args
        assert "regularity_score" in selected_args

    def test_soft_warns_when_archetype_feature_missing_from_gold(self, caplog):
        """Archetype features missing from gold log a warning but do not raise."""
        import logging
        spark = MagicMock()
        spark.catalog.tableExists.return_value = True
        gold_df = MagicMock()
        gold_df.columns = ["entity_id", "active_span_days"]
        gold_df.select = MagicMock(return_value=MagicMock())
        spark.table.return_value = gold_df

        scored = MagicMock()
        scored.columns = ["entity_id"]
        scored.join = MagicMock(return_value=scored)

        cfg = self._cfg(spark=spark)
        policies = [{"requires_features": ["active_span_days"]}]
        archetypes = [{"centroid_feature_order": ["never_materialized_feature"]}]

        with caplog.at_level(logging.WARNING, logger="customer_retention.stages.causal.snapshot_writer"):
            _join_features_for_predicates(
                spark=spark, scored_df=scored, config=cfg,
                policies=policies, archetypes=archetypes,
            )
        assert any("never_materialized_feature" in r.message for r in caplog.records)

    def test_policy_feature_missing_still_raises(self):
        """Policy-required features are a hard contract even with archetypes in play."""
        spark = MagicMock()
        spark.catalog.tableExists.return_value = True
        gold_df = MagicMock()
        gold_df.columns = ["entity_id", "event_count_180d"]  # no active_span_days
        gold_df.select = MagicMock(return_value=MagicMock())
        spark.table.return_value = gold_df

        scored = MagicMock()
        scored.columns = ["entity_id"]

        cfg = self._cfg(spark=spark)
        policies = [{"requires_features": ["active_span_days"]}]
        archetypes = [{"centroid_feature_order": ["event_count_180d"]}]

        with pytest.raises(RuntimeError, match="active_span_days"):
            _join_features_for_predicates(
                spark=spark, scored_df=scored, config=cfg,
                policies=policies, archetypes=archetypes,
            )
