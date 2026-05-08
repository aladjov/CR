"""Tests for stages.profiling.case_collision (PA-3)."""

from __future__ import annotations

import pytest

from customer_retention.analysis.auto_explorer.layered_recommendations import (
    RecommendationRegistry,
)
from customer_retention.stages.profiling.case_collision import (
    CaseCollision,
    detect_case_collisions,
    register_case_collision_drops,
)


class TestDetectCaseCollisions:
    def test_no_collisions_returns_empty(self):
        cols = ["account_id", "ACCOUNT_NAME", "created_date"]
        assert detect_case_collisions(cols) == []

    def test_lowercase_first_then_upper_keeps_lowercase(self):
        cols = ["opportunity_id", "OPPORTUNITY_ID"]
        result = detect_case_collisions(cols)
        assert result == [
            CaseCollision(kept="opportunity_id", dropped=["OPPORTUNITY_ID"]),
        ]

    def test_uppercase_first_then_lower_keeps_uppercase(self):
        cols = ["OPPORTUNITY_ID", "opportunity_id"]
        result = detect_case_collisions(cols)
        assert result == [
            CaseCollision(kept="OPPORTUNITY_ID", dropped=["opportunity_id"]),
        ]

    def test_three_variants_collision_drops_all_after_first(self):
        cols = ["foo", "FOO", "Foo"]
        result = detect_case_collisions(cols)
        assert result == [CaseCollision(kept="foo", dropped=["FOO", "Foo"])]

    def test_independent_collisions_are_separate_entries(self):
        cols = ["a", "A", "b", "B"]
        result = detect_case_collisions(cols)
        assert result == [
            CaseCollision(kept="a", dropped=["A"]),
            CaseCollision(kept="b", dropped=["B"]),
        ]

    def test_mixed_collisions_and_uniques(self):
        cols = ["entity_id", "Foo", "FOO", "as_of_date"]
        result = detect_case_collisions(cols)
        assert result == [CaseCollision(kept="Foo", dropped=["FOO"])]

    def test_none_columns_skipped(self):
        cols = ["foo", None, "FOO"]
        result = detect_case_collisions(cols)
        assert result == [CaseCollision(kept="foo", dropped=["FOO"])]

    def test_empty_input(self):
        assert detect_case_collisions([]) == []

    def test_iterates_iterable_not_just_lists(self):
        result = detect_case_collisions(c for c in ("a", "A"))
        assert result == [CaseCollision(kept="a", dropped=["A"])]


class TestRegisterCaseCollisionDrops:
    def test_collisions_registered_to_landing_drop_columns(self):
        reg = RecommendationRegistry()
        reg.init_landing()
        register_case_collision_drops(
            reg, dataset="opportunity_product",
            columns=["opportunity_id", "OPPORTUNITY_ID", "QUANTITY"],
        )
        assert len(reg.landing.drop_columns) == 1
        rec = reg.landing.drop_columns[0]
        assert rec.parameters["dataset"] == "opportunity_product"
        assert rec.parameters["columns"] == ["OPPORTUNITY_ID"]
        assert rec.action == "drop"
        assert rec.category == "drop_columns"

    def test_no_collisions_no_recs_added(self):
        reg = RecommendationRegistry()
        reg.init_landing()
        register_case_collision_drops(
            reg, dataset="account",
            columns=["entity_id", "as_of_date"],
        )
        assert reg.landing.drop_columns == []

    def test_returns_collisions_list(self):
        reg = RecommendationRegistry()
        reg.init_landing()
        result = register_case_collision_drops(
            reg, dataset="ds", columns=["foo", "FOO"],
        )
        assert result == [CaseCollision(kept="foo", dropped=["FOO"])]

    def test_idempotent_on_repeat_call(self):
        reg = RecommendationRegistry()
        reg.init_landing()
        register_case_collision_drops(
            reg, dataset="ds", columns=["foo", "FOO"],
        )
        register_case_collision_drops(
            reg, dataset="ds", columns=["foo", "FOO"],
        )
        # add_landing_drop_columns dedupes per (dataset, column), so the
        # second call is a no-op (no new rec, same column already
        # registered).
        all_drops = []
        for rec in reg.landing.drop_columns:
            all_drops.extend(rec.parameters["columns"])
        assert all_drops.count("FOO") == 1

    def test_default_source_notebook(self):
        reg = RecommendationRegistry()
        reg.init_landing()
        register_case_collision_drops(
            reg, dataset="ds", columns=["foo", "FOO"],
        )
        assert reg.landing.drop_columns[0].source_notebook == "02_source_integrity"

    def test_explicit_source_notebook_propagated(self):
        reg = RecommendationRegistry()
        reg.init_landing()
        register_case_collision_drops(
            reg, dataset="ds", columns=["foo", "FOO"],
            source_notebook="custom_nb",
        )
        assert reg.landing.drop_columns[0].source_notebook == "custom_nb"


class TestRegistryAddLandingDropColumnsAPI:
    """Direct tests for the new RecommendationRegistry method."""

    def test_init_landing_includes_drop_columns(self):
        reg = RecommendationRegistry()
        reg.init_landing()
        assert reg.landing.drop_columns == []

    def test_add_creates_landing_lazily(self):
        reg = RecommendationRegistry()
        # No init_landing() — add should auto-init like the other landing
        # add_* methods do.
        reg.add_landing_drop_columns(
            dataset="ds", columns=["FOO"], rationale="r", source_notebook="nb",
        )
        assert reg.landing is not None
        assert len(reg.landing.drop_columns) == 1

    def test_add_dedups_within_dataset(self):
        reg = RecommendationRegistry()
        reg.add_landing_drop_columns(
            dataset="ds", columns=["A"], rationale="r1", source_notebook="nb1",
        )
        reg.add_landing_drop_columns(
            dataset="ds", columns=["A", "B"], rationale="r2", source_notebook="nb2",
        )
        # First call: 1 rec with [A]; second call: only B is new, 1 rec with [B]
        all_columns = []
        for rec in reg.landing.drop_columns:
            all_columns.extend(rec.parameters["columns"])
        assert sorted(all_columns) == ["A", "B"]

    def test_add_does_not_dedup_across_datasets(self):
        reg = RecommendationRegistry()
        reg.add_landing_drop_columns(
            dataset="ds1", columns=["A"], rationale="r", source_notebook="nb",
        )
        reg.add_landing_drop_columns(
            dataset="ds2", columns=["A"], rationale="r", source_notebook="nb",
        )
        # Same column, different datasets — both kept.
        assert len(reg.landing.drop_columns) == 2

    def test_empty_columns_raises(self):
        reg = RecommendationRegistry()
        with pytest.raises(ValueError, match="at least one column"):
            reg.add_landing_drop_columns(
                dataset="ds", columns=[], rationale="r", source_notebook="nb",
            )

    def test_round_trip_through_to_dict_from_dict(self):
        reg = RecommendationRegistry()
        reg.add_landing_drop_columns(
            dataset="ds", columns=["A", "B"], rationale="r", source_notebook="nb",
        )
        round_tripped = RecommendationRegistry.from_dict(reg.to_dict())
        assert len(round_tripped.landing.drop_columns) == 1
        rec = round_tripped.landing.drop_columns[0]
        assert rec.parameters["dataset"] == "ds"
        assert rec.parameters["columns"] == ["A", "B"]
