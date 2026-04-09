"""Per-cluster surrogate decision tree → JSON predicate.

Once SHAP-clustering has labelled each training row with an archetype, the
``eligibility_policy`` table needs a *runtime* rule for each (archetype,
playbook) pair so the snapshot writer can decide eligibility from raw
features alone — no SHAP at scoring time. This module fits a depth-4
sklearn ``DecisionTreeClassifier`` per cluster as a surrogate that
approximates "is the row in cluster *i*?" using only the raw features,
then extracts every leaf with predicted class 1 and purity ≥ 0.8 as a
conjunction of feature thresholds. ORing those conjunctions together
gives a JSON predicate the runtime can evaluate against any row.

This follows the surrogate-tree pattern from Molnar's *Interpretable ML*
§5.6: depth 4 keeps the rule readable, the purity floor avoids
mis-labelling rare leaves, and the binary "in this cluster" target is the
standard surrogate-tree formulation.

Driver-side execution. The mapper feeds in the small training cohort
sample (subsampled to ``MAX_SAMPLE_ROWS`` rows when the input is larger);
fitting a depth-4 tree on ≤100k rows is comfortably tractable on a single
node, so we don't need to push this back into Spark.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------


DEFAULT_MAX_DEPTH: int = 4
DEFAULT_MIN_PURITY: float = 0.8
DEFAULT_MIN_SAMPLES_LEAF: int = 25
MAX_SAMPLE_ROWS: int = 100_000
RANDOM_STATE: int = 42


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ExtractedRule:
    """Per-cluster surrogate-tree result.

    ``predicate_json`` is the JSON predicate the snapshot writer compiles
    via ``predicate_compiler.compile_predicate``. ``predicate_sql`` is a
    rendered SQL fragment for the dashboard's "why surfaced" view. Both
    encode the same logic.
    """

    cluster_index: int
    predicate_json: Dict[str, Any]
    predicate_sql: str
    pure_leaf_count: int
    coverage: float
    used_features: List[str] = field(default_factory=list)
    feature_thresholds: Dict[str, Dict[str, float]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_eligibility_rules(
    raw_features: np.ndarray,
    feature_names: Sequence[str],
    cluster_labels: np.ndarray,
    max_depth: int = DEFAULT_MAX_DEPTH,
    min_purity: float = DEFAULT_MIN_PURITY,
    min_samples_leaf: int = DEFAULT_MIN_SAMPLES_LEAF,
    max_sample_rows: int = MAX_SAMPLE_ROWS,
) -> List[ExtractedRule]:
    """Fit one surrogate tree per cluster and return per-cluster predicates.

    ``raw_features`` is a 2-D ``np.ndarray`` shaped ``(n_rows, n_features)``;
    ``feature_names`` aligns with the column axis. ``cluster_labels`` is a
    1-D integer array. Inputs are subsampled deterministically (seed 42) if
    they exceed ``max_sample_rows``.
    """
    from sklearn.tree import DecisionTreeClassifier  # local import — sklearn loads slowly

    if raw_features.ndim != 2:
        raise ValueError(f"raw_features must be 2-D; got shape {raw_features.shape}")
    if raw_features.shape[1] != len(feature_names):
        raise ValueError(
            f"raw_features columns ({raw_features.shape[1]}) must match feature_names "
            f"length ({len(feature_names)})"
        )
    if raw_features.shape[0] != cluster_labels.shape[0]:
        raise ValueError(
            f"raw_features rows ({raw_features.shape[0]}) must match cluster_labels "
            f"length ({cluster_labels.shape[0]})"
        )

    sampled_X, sampled_labels = _subsample(raw_features, cluster_labels, max_sample_rows)
    feature_quartiles = _per_feature_quartiles(sampled_X, feature_names)
    rules: List[ExtractedRule] = []
    unique_clusters = sorted(int(c) for c in np.unique(sampled_labels))
    for cluster_index in unique_clusters:
        target = (sampled_labels == cluster_index).astype(int)
        if target.sum() < min_samples_leaf:
            rules.append(_empty_rule(cluster_index))
            continue
        tree = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=RANDOM_STATE,
            class_weight="balanced",
        )
        tree.fit(sampled_X, target)
        leaves = _extract_pure_leaves(tree, feature_names, min_purity)
        if not leaves:
            rules.append(_empty_rule(cluster_index, used_features=[]))
            continue
        predicate_json = _leaves_to_predicate(leaves)
        coverage = _leaf_coverage(tree, sampled_X, target)
        used_feature_set = sorted({step["feature"] for path in leaves for step in path["conditions"]})
        rules.append(
            ExtractedRule(
                cluster_index=cluster_index,
                predicate_json=predicate_json,
                predicate_sql=_predicate_to_sql(predicate_json),
                pure_leaf_count=len(leaves),
                coverage=coverage,
                used_features=used_feature_set,
                feature_thresholds={
                    name: feature_quartiles[name]
                    for name in used_feature_set
                    if name in feature_quartiles
                },
            )
        )
    return rules


# ---------------------------------------------------------------------------
# Internals — sklearn tree extraction
# ---------------------------------------------------------------------------


def _subsample(
    X: np.ndarray, labels: np.ndarray, max_rows: int
) -> Tuple[np.ndarray, np.ndarray]:
    if X.shape[0] <= max_rows:
        return X, labels
    rng = np.random.default_rng(RANDOM_STATE)
    indices = rng.choice(X.shape[0], size=max_rows, replace=False)
    indices.sort()
    return X[indices], labels[indices]


def _extract_pure_leaves(
    tree: Any, feature_names: Sequence[str], min_purity: float
) -> List[Dict[str, Any]]:
    inner = tree.tree_
    feature_arr = inner.feature
    threshold_arr = inner.threshold
    children_left = inner.children_left
    children_right = inner.children_right
    value = inner.value  # shape (n_nodes, 1, n_classes)

    pure_leaves: List[Dict[str, Any]] = []

    def walk(node_id: int, conditions: List[Dict[str, Any]]) -> None:
        if children_left[node_id] == children_right[node_id]:
            counts = value[node_id][0]
            total = float(counts.sum())
            if total <= 0:
                return
            class_idx = int(np.argmax(counts))
            purity = float(counts[class_idx] / total)
            if class_idx == 1 and purity >= min_purity:
                pure_leaves.append(
                    {
                        "conditions": list(conditions),
                        "purity": purity,
                        "samples": int(total),
                    }
                )
            return
        feature_idx = int(feature_arr[node_id])
        threshold = float(threshold_arr[node_id])
        feature_name = str(feature_names[feature_idx])
        walk(
            int(children_left[node_id]),
            conditions + [{"feature": feature_name, "op": "<=", "value": threshold}],
        )
        walk(
            int(children_right[node_id]),
            conditions + [{"feature": feature_name, "op": ">", "value": threshold}],
        )

    walk(0, [])
    return pure_leaves


def _leaves_to_predicate(leaves: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not leaves:
        return {"op": "false"}
    or_clauses: List[Dict[str, Any]] = []
    for leaf in leaves:
        and_clauses = [
            {"op": cond["op"], "feature": cond["feature"], "value": cond["value"]}
            for cond in _coalesce_conditions(leaf["conditions"])
        ]
        if not and_clauses:
            return {"op": "true"}
        if len(and_clauses) == 1:
            or_clauses.append(and_clauses[0])
        else:
            or_clauses.append({"op": "and", "clauses": and_clauses})
    if len(or_clauses) == 1:
        return or_clauses[0]
    return {"op": "or", "clauses": or_clauses}


def _coalesce_conditions(conditions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Collapse repeated thresholds on the same (feature, op) pair into the tightest one.

    Surrogate trees often visit the same feature multiple times along one
    root-to-leaf path. Keeping only the tightest threshold removes
    redundant clauses without changing the predicate's truth value.
    """
    tightest: Dict[Tuple[str, str], float] = {}
    for cond in conditions:
        key = (str(cond["feature"]), str(cond["op"]))
        value = float(cond["value"])
        if key not in tightest:
            tightest[key] = value
        elif key[1] == "<=":
            tightest[key] = min(tightest[key], value)
        else:  # ">"
            tightest[key] = max(tightest[key], value)
    return [
        {"feature": feature, "op": op, "value": value}
        for (feature, op), value in tightest.items()
    ]


def _leaf_coverage(tree: Any, X: np.ndarray, target: np.ndarray) -> float:
    predictions = tree.predict(X)
    matches = np.sum((predictions == 1) & (target == 1))
    positives = float(target.sum())
    if positives <= 0:
        return 0.0
    return float(matches / positives)


def _per_feature_quartiles(
    X: np.ndarray, feature_names: Sequence[str]
) -> Dict[str, Dict[str, float]]:
    quantiles = np.nanpercentile(X, [25, 50, 75], axis=0)
    out: Dict[str, Dict[str, float]] = {}
    for idx, name in enumerate(feature_names):
        out[str(name)] = {
            "p25": float(quantiles[0, idx]),
            "p50": float(quantiles[1, idx]),
            "p75": float(quantiles[2, idx]),
        }
    return out


def _empty_rule(cluster_index: int, used_features: Optional[List[str]] = None) -> ExtractedRule:
    return ExtractedRule(
        cluster_index=cluster_index,
        predicate_json={"op": "false"},
        predicate_sql="FALSE",
        pure_leaf_count=0,
        coverage=0.0,
        used_features=used_features or [],
        feature_thresholds={},
    )


def _predicate_to_sql(predicate: Dict[str, Any]) -> str:
    """Render the predicate as a SQL fragment.

    Lazy import of ``predicate_compiler.predicate_to_sql`` so the rule
    extractor stays decoupled from the predicate compiler at module-load
    time.
    """
    from .predicate_compiler import predicate_to_sql

    return predicate_to_sql(predicate)
