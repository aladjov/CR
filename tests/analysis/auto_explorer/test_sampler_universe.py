"""ANCHOR invariant tests for the sampling universe.

Invariant under test (from `docs/sps_phased_fix_strategy.md` + Cycle 001):

    universe = (∩ over D in merge_scaffold) entity_universe(D)
               ∩ (∩ over F in filters) filter_passing_ids(F)

    sample ⊆ universe   (bulletproof — no orphans under any configuration)

All fixtures use generic names (`dataset_a`, `dataset_b`, …, entity IDs
`"E{n}"`). No client-identifying content.

Tests are spec-first. The production entry point `compute_sampling_universe`
is imported at module level; tests skip cleanly when the function is
absent, and activate automatically once the C1 fix lands.
"""
from __future__ import annotations

import pandas as pd
import pytest

try:
    from customer_retention.analysis.auto_explorer.sampling import (
        compute_sampling_universe,
    )
except ImportError:
    compute_sampling_universe = None

from customer_retention.analysis.auto_explorer.sampling import (
    SegmentEntitySelection,
    resolve_segment_entity_ids,
    stratified_entity_sample,
    stratified_holdout_split,
)

pytestmark = pytest.mark.skipif(
    compute_sampling_universe is None,
    reason="compute_sampling_universe not implemented yet (C1 fix pending)",
)


def _ids(prefix: str, start: int, stop: int) -> list[str]:
    return [f"{prefix}{i}" for i in range(start, stop)]


def _frame(entity_col: str, ids: list[str], **extra) -> pd.DataFrame:
    data = {entity_col: ids}
    for k, v in extra.items():
        data[k] = v if len(v) == len(ids) else [v] * len(ids)
    return pd.DataFrame(data)


def _expected_universe(
    frames: dict[str, pd.DataFrame],
    entity_columns: dict[str, str],
    filters: dict[str, str] | None = None,
    primary: str = "dataset_a",
) -> set[str]:
    """Oracle — the correct universe by definition.

    Anchors at the primary entity dataset; narrows by each filter's
    passing-ID set (a filter on a non-primary dataset counts an entity
    only when at least one of its rows there passes, matching
    _spark_passing_entities groupBy-then-all-match semantics). Pure
    pandas; no dependency on the production implementation.

    Entities appearing in auxiliary datasets but not in the primary are
    NOT contributed by this oracle — those would be orphans per cycle
    001's ANCHOR invariant.
    """
    universe = set(frames[primary][entity_columns[primary]].astype(str))
    for name, expr in (filters or {}).items():
        df = frames[name]
        pre = df.groupby(entity_columns[name]).size()
        passed = df.query(expr).groupby(entity_columns[name]).size()
        passing = {k for k in pre.index if passed.get(k, 0) == pre[k]}
        universe &= {str(x) for x in passing}
    return universe


class TestAnchorInvariant:
    """compute_sampling_universe must always return a subset of the oracle."""

    def test_single_dataset_no_filter(self):
        frames = {"dataset_a": _frame("id", _ids("E", 0, 50))}
        entity_cols = {"dataset_a": "id"}

        expected = _expected_universe(frames, entity_cols)
        universe = compute_sampling_universe(
            frames=frames,
            entity_columns=entity_cols,
            primary_entity_dataset="dataset_a",
        )
        actual = set(universe) if not universe.is_distributed else None
        assert actual == expected
        assert len(actual) == 50

    def test_two_datasets_no_filter_anchors_at_primary(self):
        # Auxiliary-dataset presence is NOT required. Entities that exist only
        # in dataset_b (X0..X74 here, after the slice) must not enter the
        # universe — they'd be orphans — but primary-only entities (E0..E24)
        # stay. Intersecting across all datasets was the pre-fix overreach.
        frames = {
            "dataset_a": _frame("id", _ids("E", 0, 50)),
            "dataset_b": _frame("id", _ids("E", 25, 100)),
        }
        entity_cols = {"dataset_a": "id", "dataset_b": "id"}

        expected = _expected_universe(frames, entity_cols)
        assert expected == set(_ids("E", 0, 50))

        universe = compute_sampling_universe(
            frames=frames,
            entity_columns=entity_cols,
            primary_entity_dataset="dataset_a",
        )
        actual = set(universe) if not universe.is_distributed else None
        assert actual == expected

    def test_filter_on_primary_dataset(self):
        frames = {
            "dataset_a": _frame("id", _ids("E", 0, 50), keep=[i % 2 for i in range(50)]),
            "dataset_b": _frame("id", _ids("E", 0, 50)),
        }
        entity_cols = {"dataset_a": "id", "dataset_b": "id"}
        filters = {"dataset_a": "keep == 1"}

        expected = _expected_universe(frames, entity_cols, filters)
        universe = compute_sampling_universe(
            frames=frames,
            entity_columns=entity_cols,
            primary_entity_dataset="dataset_a",
            filters=filters,
        )
        actual = set(universe) if not universe.is_distributed else None
        assert actual == expected
        assert len(actual) == 25

    def test_filter_on_non_primary_dataset(self):
        frames = {
            "dataset_a": _frame("id", _ids("E", 0, 50)),
            "dataset_b": _frame("id", _ids("E", 0, 50), active=[i < 10 for i in range(50)]),
        }
        entity_cols = {"dataset_a": "id", "dataset_b": "id"}
        filters = {"dataset_b": "active == True"}

        expected = _expected_universe(frames, entity_cols, filters)
        universe = compute_sampling_universe(
            frames=frames,
            entity_columns=entity_cols,
            primary_entity_dataset="dataset_a",
            filters=filters,
        )
        actual = set(universe) if not universe.is_distributed else None
        assert actual == expected
        assert len(actual) == 10

    def test_multi_filter_intersection(self):
        frames = {
            "dataset_a": _frame("id", _ids("E", 0, 100), seg=["small" if i < 30 else "large" for i in range(100)]),
            "dataset_b": _frame("id", _ids("E", 0, 100), lifecycle=["active" if i % 3 == 0 else "other" for i in range(100)]),
        }
        entity_cols = {"dataset_a": "id", "dataset_b": "id"}
        filters = {"dataset_a": "seg == 'small'", "dataset_b": "lifecycle == 'active'"}

        expected = _expected_universe(frames, entity_cols, filters)
        universe = compute_sampling_universe(
            frames=frames,
            entity_columns=entity_cols,
            primary_entity_dataset="dataset_a",
            filters=filters,
        )
        actual = set(universe) if not universe.is_distributed else None
        assert actual == expected

    def test_orphan_regression(self):
        """Reproduces the 4,107-orphan shape from engagement_e4ad6e1b.

        Pre-fix: sampler pulls from `dataset_b` (broader) and never anchors
        to `dataset_a` (primary, filtered). Post-fix: universe is always a
        subset of dataset_a, so the 100 "orphan" entities in dataset_b-only
        never enter the pool.
        """
        primary = _ids("E", 0, 50)
        orphans = _ids("X", 0, 100)
        frames = {
            "dataset_a": _frame("id", primary,
                                 seg=["small" if i < 25 else "large" for i in range(50)]),
            "dataset_b": _frame("id", primary + orphans),
        }
        entity_cols = {"dataset_a": "id", "dataset_b": "id"}
        filters = {"dataset_a": "seg == 'small'"}

        universe = compute_sampling_universe(
            frames=frames,
            entity_columns=entity_cols,
            primary_entity_dataset="dataset_a",
            filters=filters,
        )
        actual = set(universe) if not universe.is_distributed else None
        assert not (actual & set(orphans)), "universe must never contain dataset_b-only orphans"
        assert len(actual) == 25

    def test_bridge_dataset_resolved_to_primary(self):
        """opportunity_product-style bridge — entity reaches primary through an
        intermediate table. Only bridged-resolvable IDs count toward the universe."""
        primary = _ids("E", 0, 50)
        opps = [("O" + str(i), "E" + str(i)) for i in range(50)]
        opp_products = [("OP" + str(i), "O" + str(i)) for i in range(60)]

        frames = {
            "dataset_a": _frame("id", primary),
            "dataset_b": pd.DataFrame(opps, columns=["opp_id", "id"]),
            "dataset_c": pd.DataFrame(opp_products, columns=["op_id", "opp_id"]),
        }
        entity_cols = {"dataset_a": "id", "dataset_b": "id", "dataset_c": "opp_id"}
        bridges = {
            "dataset_c": {"through": "dataset_b", "on": "opp_id", "resolves_to": "id"},
        }

        universe = compute_sampling_universe(
            frames=frames,
            entity_columns=entity_cols,
            primary_entity_dataset="dataset_a",
            bridges=bridges,
        )
        actual = set(universe) if not universe.is_distributed else None
        assert actual <= set(primary)
        assert len(actual & set(primary)) == 50

    def test_empty_universe_raises(self):
        # Under anchor-at-primary, an empty universe requires a filter that
        # eliminates every primary entity — not merely a disjoint auxiliary.
        frames = {
            "dataset_a": _frame("id", _ids("E", 0, 50),
                                 seg=["keep" if i < 0 else "drop" for i in range(50)]),
        }
        entity_cols = {"dataset_a": "id"}
        filters = {"dataset_a": "seg == 'keep'"}
        with pytest.raises((ValueError, RuntimeError), match=r"(empty|no entities|universe)"):
            compute_sampling_universe(
                frames=frames,
                entity_columns=entity_cols,
                primary_entity_dataset="dataset_a",
                filters=filters,
            )


class TestSamplerRespectsAnchor:
    """The full sampler chain must produce IDs that are a subset of the universe."""

    def _run_sampler(self, frames, entity_cols, filters, n_entities, holdout_fraction, primary):
        universe = compute_sampling_universe(
            frames=frames,
            entity_columns=entity_cols,
            primary_entity_dataset=primary,
            filters=filters,
        )
        primary_frame = frames[primary]
        mask = primary_frame[entity_cols[primary]].astype(str).isin(
            set(universe) if not universe.is_distributed else set()
        )
        filtered = primary_frame[mask].copy()
        train_ids = stratified_entity_sample(
            entity_df=filtered, n_entities=n_entities, entity_col=entity_cols[primary],
        )
        train_ids, holdout_ids = stratified_holdout_split(
            entity_df=filtered, entity_ids=train_ids,
            holdout_fraction=holdout_fraction, entity_col=entity_cols[primary],
        )
        return set(universe) if not universe.is_distributed else set(), train_ids, holdout_ids

    def test_sample_subset_of_universe(self):
        frames = {
            "dataset_a": _frame("id", _ids("E", 0, 50),
                                 seg=["small" if i < 25 else "large" for i in range(50)]),
            "dataset_b": _frame("id", _ids("E", 0, 50) + _ids("X", 0, 100)),
        }
        entity_cols = {"dataset_a": "id", "dataset_b": "id"}
        universe, train, holdout = self._run_sampler(
            frames, entity_cols, {"dataset_a": "seg == 'small'"},
            n_entities=20, holdout_fraction=0.5, primary="dataset_a",
        )
        assert set(train) <= universe
        assert set(holdout) <= universe
        assert set(train).isdisjoint(set(holdout))

    def test_train_holdout_cardinality_matches_config(self):
        n_total = 40
        frames = {"dataset_a": _frame("id", _ids("E", 0, 100))}
        entity_cols = {"dataset_a": "id"}
        _, train, holdout = self._run_sampler(
            frames, entity_cols, None,
            n_entities=n_total, holdout_fraction=0.5, primary="dataset_a",
        )
        assert len(train) + len(holdout) == n_total
        assert abs(len(train) - n_total // 2) <= 1

    def test_n_larger_than_universe_caps_at_universe(self):
        frames = {"dataset_a": _frame("id", _ids("E", 0, 10))}
        entity_cols = {"dataset_a": "id"}
        universe, train, holdout = self._run_sampler(
            frames, entity_cols, None,
            n_entities=1000, holdout_fraction=0.5, primary="dataset_a",
        )
        assert len(train) + len(holdout) <= len(universe)
        assert set(train) | set(holdout) <= universe

    def test_holdout_fraction_zero_emits_empty_holdout(self):
        frames = {"dataset_a": _frame("id", _ids("E", 0, 50))}
        entity_cols = {"dataset_a": "id"}
        _, train, holdout = self._run_sampler(
            frames, entity_cols, None,
            n_entities=20, holdout_fraction=0.0, primary="dataset_a",
        )
        assert len(train) == 20
        assert holdout == []

    def test_holdout_fraction_one_emits_empty_train(self):
        frames = {"dataset_a": _frame("id", _ids("E", 0, 50))}
        entity_cols = {"dataset_a": "id"}
        _, train, holdout = self._run_sampler(
            frames, entity_cols, None,
            n_entities=20, holdout_fraction=1.0, primary="dataset_a",
        )
        assert len(holdout) == 20
        assert train == []

    def test_stratification_preserves_class_proportions(self):
        frames = {
            "dataset_a": _frame(
                "id", _ids("E", 0, 100),
                target=[1 if i < 20 else 0 for i in range(100)],
            ),
        }
        entity_cols = {"dataset_a": "id"}
        universe = compute_sampling_universe(
            frames=frames,
            entity_columns=entity_cols,
            primary_entity_dataset="dataset_a",
        )
        primary_frame = frames["dataset_a"]
        filtered = primary_frame[primary_frame["id"].astype(str).isin(
            set(universe) if not universe.is_distributed else set()
        )].copy()

        train = stratified_entity_sample(
            entity_df=filtered, n_entities=40, entity_col="id", target_col="target",
        )
        sample = filtered[filtered["id"].isin(train)]
        positive_rate = sample["target"].mean()
        assert abs(positive_rate - 0.2) < 0.1


class TestExistingSamplerRegressionGuard:
    """Snapshot the current behaviour of the existing API so a fix cannot
    silently regress intersection semantics already in place."""

    def test_resolve_segment_entity_ids_intersects_filters(self):
        frames = {
            "dataset_a": _frame("id", _ids("E", 0, 30), seg=["small"] * 15 + ["large"] * 15),
            "dataset_b": _frame("id", _ids("E", 0, 30), active=[True] * 10 + [False] * 20),
        }
        selection = resolve_segment_entity_ids(
            frames=frames,
            filters={"dataset_a": "seg == 'small'", "dataset_b": "active == True"},
            entity_columns={"dataset_a": "id", "dataset_b": "id"},
        )
        assert isinstance(selection, SegmentEntitySelection)
        assert set(selection) == set(_ids("E", 0, 10))
