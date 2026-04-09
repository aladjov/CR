"""Unit tests for the surrogate-tree rule extractor.

These tests use real numpy + scikit-learn (both already in the dev env)
because the rule extractor is pure CPU code with no Spark surface.
Synthetic clusters with clean separability ensure deterministic predicates.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("sklearn")

from customer_retention.stages.causal.rule_extractor import (
    DEFAULT_MAX_DEPTH,
    extract_eligibility_rules,
)

# ---------------------------------------------------------------------------
# Synthetic data builders
# ---------------------------------------------------------------------------


def _make_two_cluster_data(n_per_cluster: int = 200) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Two clean clusters separated on a single feature."""
    rng = np.random.default_rng(42)
    cluster_a = np.column_stack([rng.normal(0.0, 0.1, n_per_cluster), rng.normal(0.0, 1.0, n_per_cluster)])
    cluster_b = np.column_stack([rng.normal(5.0, 0.1, n_per_cluster), rng.normal(0.0, 1.0, n_per_cluster)])
    X = np.vstack([cluster_a, cluster_b])
    labels = np.concatenate([np.zeros(n_per_cluster, dtype=int), np.ones(n_per_cluster, dtype=int)])
    return X, labels, ["x", "y"]


def _make_three_cluster_data(n_per_cluster: int = 200) -> tuple[np.ndarray, np.ndarray, list[str]]:
    rng = np.random.default_rng(42)
    a = np.column_stack([rng.normal(0.0, 0.1, n_per_cluster), rng.normal(0.0, 0.1, n_per_cluster)])
    b = np.column_stack([rng.normal(5.0, 0.1, n_per_cluster), rng.normal(0.0, 0.1, n_per_cluster)])
    c = np.column_stack([rng.normal(0.0, 0.1, n_per_cluster), rng.normal(5.0, 0.1, n_per_cluster)])
    X = np.vstack([a, b, c])
    labels = np.concatenate(
        [
            np.zeros(n_per_cluster, dtype=int),
            np.ones(n_per_cluster, dtype=int),
            np.full(n_per_cluster, 2, dtype=int),
        ]
    )
    return X, labels, ["x", "y"]


# ---------------------------------------------------------------------------
# extract_eligibility_rules
# ---------------------------------------------------------------------------


class TestExtractEligibilityRules:
    def test_returns_one_rule_per_cluster(self):
        X, labels, names = _make_three_cluster_data()
        rules = extract_eligibility_rules(X, names, labels)
        assert len(rules) == 3
        assert {r.cluster_index for r in rules} == {0, 1, 2}

    def test_rule_has_predicate_json_and_sql(self):
        X, labels, names = _make_two_cluster_data()
        rules = extract_eligibility_rules(X, names, labels)
        for rule in rules:
            assert isinstance(rule.predicate_json, dict)
            assert "op" in rule.predicate_json
            assert isinstance(rule.predicate_sql, str)
            assert rule.predicate_sql

    def test_uses_only_separating_feature_when_separable(self):
        X, labels, names = _make_two_cluster_data()
        rules = extract_eligibility_rules(X, names, labels)
        for rule in rules:
            if rule.pure_leaf_count > 0:
                assert "x" in rule.used_features

    def test_pure_leaf_count_positive_for_separable_data(self):
        X, labels, names = _make_two_cluster_data()
        rules = extract_eligibility_rules(X, names, labels)
        assert any(r.pure_leaf_count > 0 for r in rules)

    def test_coverage_is_non_zero(self):
        X, labels, names = _make_two_cluster_data()
        rules = extract_eligibility_rules(X, names, labels)
        assert any(r.coverage > 0.0 for r in rules)

    def test_feature_thresholds_include_quartiles(self):
        X, labels, names = _make_two_cluster_data()
        rules = extract_eligibility_rules(X, names, labels)
        for rule in rules:
            for name, thresholds in rule.feature_thresholds.items():
                assert {"p25", "p50", "p75"} <= set(thresholds.keys())

    def test_max_depth_is_respected(self):
        X, labels, names = _make_three_cluster_data()
        rules = extract_eligibility_rules(X, names, labels, max_depth=2)
        for rule in rules:
            for cond_count in _conditions_per_path(rule.predicate_json):
                assert cond_count <= 2

    def test_min_samples_below_threshold_returns_empty_predicate(self):
        # 5 samples in cluster 0, plenty in cluster 1
        X = np.column_stack([np.linspace(0.0, 1.0, 105), np.linspace(0.0, 1.0, 105)])
        labels = np.concatenate([np.zeros(5, dtype=int), np.ones(100, dtype=int)])
        rules = extract_eligibility_rules(X, ["x", "y"], labels, min_samples_leaf=25)
        cluster_0 = next(r for r in rules if r.cluster_index == 0)
        assert cluster_0.predicate_json == {"op": "false"}
        assert cluster_0.pure_leaf_count == 0

    def test_subsamples_when_input_exceeds_max_sample_rows(self):
        n = 5000
        rng = np.random.default_rng(42)
        X = np.column_stack([rng.normal(0.0, 1.0, n), rng.normal(0.0, 1.0, n)])
        labels = (X[:, 0] > 0).astype(int)
        rules = extract_eligibility_rules(X, ["x", "y"], labels, max_sample_rows=500)
        # Subsampling shouldn't crash; both clusters should get rules
        assert len(rules) == 2

    def test_raises_on_dim_mismatch(self):
        X = np.zeros((10, 2))
        labels = np.zeros(10, dtype=int)
        with pytest.raises(ValueError, match="match feature_names"):
            extract_eligibility_rules(X, ["x"], labels)

    def test_raises_on_row_count_mismatch(self):
        X = np.zeros((10, 2))
        labels = np.zeros(5, dtype=int)
        with pytest.raises(ValueError, match="match cluster_labels"):
            extract_eligibility_rules(X, ["x", "y"], labels)

    def test_raises_on_non_2d_input(self):
        X = np.zeros(10)
        labels = np.zeros(10, dtype=int)
        with pytest.raises(ValueError, match="must be 2-D"):
            extract_eligibility_rules(X, ["x"], labels)

    def test_default_max_depth_is_four(self):
        assert DEFAULT_MAX_DEPTH == 4


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _conditions_per_path(predicate: dict) -> list[int]:
    """Return condition counts for every leaf path in an OR-of-ANDs tree."""
    if not predicate:
        return [0]
    op = predicate.get("op")
    if op == "and":
        return [len(predicate.get("clauses") or [])]
    if op == "or":
        out: list[int] = []
        for clause in predicate.get("clauses") or []:
            out.extend(_conditions_per_path(clause))
        return out
    if op in {">=", "<=", ">", "<", "==", "!="}:
        return [1]
    return [0]
