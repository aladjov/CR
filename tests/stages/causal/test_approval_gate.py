"""Unit tests for the approval gate.

The cosine similarity helper is pure Python and tested directly. The
``auto_promote_stable`` and ``list_pending_review`` functions are
exercised through a fake SparkSession that records every SQL call so the
tests verify the right cascades happen without needing a live cluster.

After the bug-fix rewrite, ``auto_promote_stable`` loads candidates +
prior active rows in a single ``spark.sql(query, args=...)`` call (no
more N+1 ``spark.table()`` filtering), and writes through parameterized
UPDATE statements. The fake Spark in this file mirrors that contract:
``spark.sql(query, args=...)`` is the only entry point the production
code touches.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

from customer_retention.stages.causal.approval_gate import (
    AUTO_APPROVER,
    DEFAULT_STABILITY_THRESHOLD,
    ApprovalGateResult,
    StabilityDecision,
    auto_promote_stable,
    cosine_similarity,
    list_pending_review,
)

# ---------------------------------------------------------------------------
# cosine_similarity
# ---------------------------------------------------------------------------


class TestCosineSimilarity:
    def test_identical_vectors_score_one(self):
        assert cosine_similarity([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)

    def test_orthogonal_vectors_score_zero(self):
        assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_opposite_vectors_score_minus_one(self):
        assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)

    def test_zero_vector_returns_zero(self):
        assert cosine_similarity([0.0, 0.0], [1.0, 2.0]) == 0.0
        assert cosine_similarity([1.0, 2.0], [0.0, 0.0]) == 0.0

    def test_empty_vector_returns_zero(self):
        assert cosine_similarity([], []) == 0.0

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError, match="equal-length"):
            cosine_similarity([1.0, 2.0], [1.0, 2.0, 3.0])

    def test_handles_int_inputs_via_float_conversion(self):
        assert cosine_similarity([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Fake Spark wiring for the gate tests
# ---------------------------------------------------------------------------


class _FakeRow:
    def __init__(self, data: Dict[str, Any]) -> None:
        self._data = data

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def asDict(self, recursive: bool = False) -> Dict[str, Any]:  # noqa: N802
        # Mirrors pyspark.sql.Row.asDict — production code calls row.asDict(recursive=True)
        return dict(self._data)


class _FakeResultDF:
    """Static result DataFrame returned by the fake spark.sql."""

    def __init__(self, rows: List[Dict[str, Any]]) -> None:
        self.rows = rows

    def collect(self) -> List[_FakeRow]:
        return [_FakeRow(r) for r in self.rows]


@dataclass
class _SqlCall:
    query: str
    args: List[Any]


def _make_fake_spark(
    pending_rows: List[Dict[str, Any]],
    prior_rows_by_id: Dict[str, List[Dict[str, Any]]],
    table_exists: bool = True,
    superseded_count: int = 0,
):
    """Build a fake SparkSession that mimics the parameterized SQL contract.

    ``approval_gate`` only ever calls ``spark.sql(query, args=[...])`` —
    no ``spark.table().filter()`` chains anymore. The fake matches the
    real query patterns by substring and returns the right canned rows
    for each: ``WITH pending`` ⇒ join of pending × prior; ``COUNT(*)``
    ⇒ supersession count; everything else (UPDATE statements) ⇒ empty.
    """
    spark = MagicMock(name="SparkSession")
    spark.catalog = MagicMock()
    spark.catalog.tableExists.return_value = table_exists

    sql_calls: List[_SqlCall] = []
    spark.__sql_calls__ = sql_calls  # type: ignore[attr-defined]

    def _join_pending_with_prior() -> List[Dict[str, Any]]:
        joined: List[Dict[str, Any]] = []
        for pending in pending_rows:
            archetype_id = pending["archetype_id"]
            priors = prior_rows_by_id.get(archetype_id, [])
            prior = priors[0] if priors else None
            joined.append(
                {
                    "archetype_id": archetype_id,
                    "archetype_version": pending["archetype_version"],
                    "centroid_vector": pending.get("centroid_vector"),
                    "prior_archetype_version": prior["archetype_version"] if prior else None,
                    "prior_centroid_vector": prior.get("centroid_vector") if prior else None,
                }
            )
        return joined

    def _sql(query: str, args: List[Any] | None = None) -> _FakeResultDF:
        sql_calls.append(_SqlCall(query=query, args=list(args) if args else []))
        if "WITH pending AS" in query:
            return _FakeResultDF(_join_pending_with_prior())
        if "SELECT COUNT(*) AS c FROM" in query:
            return _FakeResultDF([{"c": superseded_count}])
        return _FakeResultDF([])

    spark.sql.side_effect = _sql
    return spark


# ---------------------------------------------------------------------------
# auto_promote_stable
# ---------------------------------------------------------------------------


class TestAutoPromoteStable:
    def test_first_version_no_prior_active_promotes(self):
        pending = [
            {
                "archetype_id": "arch_1_aaa",
                "archetype_version": "v_111",
                "centroid_vector": [1.0, 2.0, 3.0],
            }
        ]
        spark = _make_fake_spark(pending, prior_rows_by_id={}, superseded_count=0)
        result = auto_promote_stable(
            spark=spark,
            archetype_table_fqn="cat.sch.archetype_catalog",
            policy_table_fqn="cat.sch.eligibility_policy",
            derivation_run_id="deriv_1",
            now=datetime(2026, 4, 9),
        )
        assert len(result.promoted) == 1
        assert result.pending == []
        assert result.promoted[0].reason == "first_version_no_prior_active"

    def test_stable_centroid_auto_promotes(self):
        pending = [
            {
                "archetype_id": "arch_1_aaa",
                "archetype_version": "v_222",
                "centroid_vector": [1.0, 2.0, 3.0],
            }
        ]
        prior = {"arch_1_aaa": [{"archetype_version": "v_111", "centroid_vector": [1.0, 2.0, 3.0]}]}
        spark = _make_fake_spark(pending, prior_rows_by_id=prior)
        result = auto_promote_stable(
            spark=spark,
            archetype_table_fqn="cat.sch.archetype_catalog",
            policy_table_fqn="cat.sch.eligibility_policy",
            derivation_run_id="deriv_2",
            now=datetime(2026, 4, 9),
        )
        assert len(result.promoted) == 1
        assert result.promoted[0].cosine_similarity == pytest.approx(1.0)
        assert "stable" in result.promoted[0].reason

    def test_unstable_centroid_stays_pending(self):
        pending = [
            {
                "archetype_id": "arch_1_aaa",
                "archetype_version": "v_222",
                "centroid_vector": [1.0, 0.0, 0.0],
            }
        ]
        prior = {"arch_1_aaa": [{"archetype_version": "v_111", "centroid_vector": [0.0, 1.0, 0.0]}]}
        spark = _make_fake_spark(pending, prior_rows_by_id=prior)
        result = auto_promote_stable(
            spark=spark,
            archetype_table_fqn="cat.sch.archetype_catalog",
            policy_table_fqn="cat.sch.eligibility_policy",
            derivation_run_id="deriv_3",
            now=datetime(2026, 4, 9),
        )
        assert result.promoted == []
        assert len(result.pending) == 1
        assert "unstable" in result.pending[0].reason

    def test_force_approve_overrides_unstable(self):
        pending = [
            {
                "archetype_id": "arch_1_aaa",
                "archetype_version": "v_222",
                "centroid_vector": [1.0, 0.0, 0.0],
            }
        ]
        prior = {"arch_1_aaa": [{"archetype_version": "v_111", "centroid_vector": [0.0, 1.0, 0.0]}]}
        spark = _make_fake_spark(pending, prior_rows_by_id=prior)
        result = auto_promote_stable(
            spark=spark,
            archetype_table_fqn="cat.sch.archetype_catalog",
            policy_table_fqn="cat.sch.eligibility_policy",
            derivation_run_id="deriv_force",
            force=True,
            now=datetime(2026, 4, 9),
        )
        assert len(result.promoted) == 1
        assert result.promoted[0].reason == "force_approve"

    def test_missing_centroid_marks_pending(self):
        pending = [
            {
                "archetype_id": "arch_1_aaa",
                "archetype_version": "v_222",
                "centroid_vector": None,
            }
        ]
        prior = {"arch_1_aaa": [{"archetype_version": "v_111", "centroid_vector": [1.0, 2.0]}]}
        spark = _make_fake_spark(pending, prior_rows_by_id=prior)
        result = auto_promote_stable(
            spark=spark,
            archetype_table_fqn="cat.sch.archetype_catalog",
            policy_table_fqn="cat.sch.eligibility_policy",
            derivation_run_id="deriv_4",
            now=datetime(2026, 4, 9),
        )
        assert result.pending[0].reason == "missing_centroid_vector"

    def test_dimension_change_marks_pending(self):
        pending = [
            {
                "archetype_id": "arch_1_aaa",
                "archetype_version": "v_222",
                "centroid_vector": [1.0, 2.0, 3.0, 4.0],
            }
        ]
        prior = {"arch_1_aaa": [{"archetype_version": "v_111", "centroid_vector": [1.0, 2.0]}]}
        spark = _make_fake_spark(pending, prior_rows_by_id=prior)
        result = auto_promote_stable(
            spark=spark,
            archetype_table_fqn="cat.sch.archetype_catalog",
            policy_table_fqn="cat.sch.eligibility_policy",
            derivation_run_id="deriv_5",
            now=datetime(2026, 4, 9),
        )
        assert result.pending[0].reason == "centroid_dimension_changed"

    def test_promotion_emits_parameterized_update_sql(self):
        pending = [
            {
                "archetype_id": "arch_1_aaa",
                "archetype_version": "v_222",
                "centroid_vector": [1.0, 2.0, 3.0],
            }
        ]
        prior = {"arch_1_aaa": [{"archetype_version": "v_111", "centroid_vector": [1.0, 2.0, 3.0]}]}
        spark = _make_fake_spark(pending, prior_rows_by_id=prior)
        auto_promote_stable(
            spark=spark,
            archetype_table_fqn="cat.sch.archetype_catalog",
            policy_table_fqn="cat.sch.eligibility_policy",
            derivation_run_id="deriv_6",
            now=datetime(2026, 4, 9),
        )
        archetype_updates = [
            call for call in spark.__sql_calls__
            if "UPDATE cat.sch.archetype_catalog" in call.query
        ]
        # Two updates: one promotes v_222 to active, one supersedes v_111
        assert len(archetype_updates) == 2
        promote_call = next(c for c in archetype_updates if "status = 'active'" in c.query)
        supersede_call = next(c for c in archetype_updates if "superseded" in c.query)
        # Promotion binds the auto-approver and the new version via args
        assert AUTO_APPROVER in promote_call.args
        assert "v_222" in promote_call.args
        # Supersession binds the prior version via args
        assert "v_111" in supersede_call.args
        # No string interpolation of values into the SQL text itself
        assert "v_222" not in promote_call.query
        assert "v_111" not in supersede_call.query

    def test_cascade_uses_arrays_overlap(self):
        pending = [
            {
                "archetype_id": "arch_1_aaa",
                "archetype_version": "v_222",
                "centroid_vector": [1.0, 2.0, 3.0],
            }
        ]
        prior = {"arch_1_aaa": [{"archetype_version": "v_111", "centroid_vector": [1.0, 2.0, 3.0]}]}
        spark = _make_fake_spark(pending, prior_rows_by_id=prior)
        auto_promote_stable(
            spark=spark,
            archetype_table_fqn="cat.sch.archetype_catalog",
            policy_table_fqn="cat.sch.eligibility_policy",
            derivation_run_id="deriv_cascade",
            now=datetime(2026, 4, 9),
        )
        policy_updates = [
            call for call in spark.__sql_calls__
            if "UPDATE cat.sch.eligibility_policy" in call.query
        ]
        assert len(policy_updates) == 1
        cascade = policy_updates[0]
        assert "arrays_overlap(archetype_ids, array(" in cascade.query
        # Promoted archetype version is bound, not interpolated
        assert "v_222" in cascade.args
        assert "v_222" not in cascade.query

    def test_single_join_query_loads_all_candidates(self):
        """Verify the N+1 lookup is gone — one batched join query, not N filters."""
        pending = [
            {
                "archetype_id": "arch_1",
                "archetype_version": "v_1",
                "centroid_vector": [1.0, 0.0],
            },
            {
                "archetype_id": "arch_2",
                "archetype_version": "v_2",
                "centroid_vector": [0.0, 1.0],
            },
            {
                "archetype_id": "arch_3",
                "archetype_version": "v_3",
                "centroid_vector": [1.0, 1.0],
            },
        ]
        prior = {
            "arch_1": [{"archetype_version": "v_0", "centroid_vector": [1.0, 0.0]}],
            "arch_2": [{"archetype_version": "v_0", "centroid_vector": [0.0, 1.0]}],
        }
        spark = _make_fake_spark(pending, prior_rows_by_id=prior)
        auto_promote_stable(
            spark=spark,
            archetype_table_fqn="cat.sch.archetype_catalog",
            policy_table_fqn="cat.sch.eligibility_policy",
            derivation_run_id="deriv_batch",
            now=datetime(2026, 4, 9),
        )
        # Exactly one candidate-loading query, regardless of candidate count
        loader_calls = [
            call for call in spark.__sql_calls__ if "WITH pending AS" in call.query
        ]
        assert len(loader_calls) == 1


# ---------------------------------------------------------------------------
# list_pending_review
# ---------------------------------------------------------------------------


def _make_pending_review_spark(rows: List[Dict[str, Any]], table_exists: bool = True):
    """Tiny fake Spark whose only behaviour is to echo `rows` from the SELECT."""
    spark = MagicMock(name="SparkSession")
    spark.catalog = MagicMock()
    spark.catalog.tableExists.return_value = table_exists

    def _sql(query: str, args: List[Any] | None = None) -> _FakeResultDF:  # noqa: ARG001
        return _FakeResultDF(rows)

    spark.sql.side_effect = _sql
    return spark


class TestListPendingReview:
    def test_returns_empty_when_table_missing(self):
        spark = _make_pending_review_spark([], table_exists=False)
        assert list_pending_review(spark, "cat.sch.archetype_catalog") == []

    def test_returns_dict_per_pending_row(self):
        rows = [
            {
                "archetype_id": "arch_1",
                "archetype_version": "v_111",
                "name": "Tenured Stable",
                "stability_vs_prior_version": 0.93,
            }
        ]
        spark = _make_pending_review_spark(rows)
        result = list_pending_review(spark, "cat.sch.archetype_catalog", derivation_run_id="deriv")
        assert len(result) == 1
        assert result[0]["archetype_id"] == "arch_1"
        assert result[0]["name"] == "Tenured Stable"
        assert result[0]["stability_vs_prior_version"] == "0.9300"

    def test_handles_null_stability(self):
        rows = [
            {
                "archetype_id": "arch_1",
                "archetype_version": "v_111",
                "name": None,
                "stability_vs_prior_version": None,
            }
        ]
        spark = _make_pending_review_spark(rows)
        result = list_pending_review(spark, "cat.sch.archetype_catalog", derivation_run_id="d")
        assert result[0]["stability_vs_prior_version"] == "n/a"
        assert result[0]["name"] == ""


# ---------------------------------------------------------------------------
# ApprovalGateResult convenience
# ---------------------------------------------------------------------------


class TestApprovalGateResult:
    def test_total_counts_promoted_plus_pending(self):
        result = ApprovalGateResult(
            promoted=[
                StabilityDecision(
                    archetype_id="a", archetype_version="v1", prior_archetype_version=None,
                    cosine_similarity=None, promoted=True, reason="x",
                )
            ],
            pending=[
                StabilityDecision(
                    archetype_id="b", archetype_version="v1", prior_archetype_version=None,
                    cosine_similarity=None, promoted=False, reason="y",
                )
            ],
        )
        assert result.total == 2

    def test_summary_is_human_readable(self):
        result = ApprovalGateResult(
            promoted=[],
            pending=[],
            superseded_archetypes=2,
            superseded_policies=5,
            threshold=DEFAULT_STABILITY_THRESHOLD,
        )
        text = result.summary()
        assert "0 auto-promoted" in text
        assert "0 pending review" in text
        assert "superseded 2" in text
