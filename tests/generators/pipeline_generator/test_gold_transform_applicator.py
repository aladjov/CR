"""Tests for gold_transform_applicator: build_gold_steps, build_silver_derived_steps, _derived_source_columns."""

from __future__ import annotations

from customer_retention.analysis.auto_explorer.layered_recommendations import (
    RecommendationRegistry,
)
from customer_retention.generators.pipeline_generator.gold_transform_applicator import (
    _derived_source_columns,
    build_gold_steps,
    build_silver_derived_steps,
)
from customer_retention.generators.pipeline_generator.models import (
    PipelineTransformationType,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _registry_with_gold(*transforms):
    """Return a registry with gold initialised and the given (column, action, params) tuples added."""
    reg = RecommendationRegistry()
    reg.init_gold("churn")
    for col, action, params in transforms:
        reg.add_gold_transformation(col, action, params, "rationale", "04")
    return reg


def _registry_with_silver(*derived):
    """Return a registry with silver initialised and the given derived columns added.

    Each entry is (method, *args) where method is 'ratio', 'interaction', or 'composite'.
    """
    reg = RecommendationRegistry()
    reg.init_silver("customer_id")
    for entry in derived:
        method = entry[0]
        if method == "ratio":
            reg.add_silver_ratio(entry[1], entry[2], entry[3], "rationale", "06")
        elif method == "interaction":
            reg.add_silver_interaction(entry[1], entry[2], "rationale", "06")
        elif method == "composite":
            reg.add_silver_composite(entry[1], entry[2], "rationale", "06")
    return reg


# ---------------------------------------------------------------------------
# build_gold_steps
# ---------------------------------------------------------------------------

class TestBuildGoldSteps:
    def test_no_gold_attribute_returns_empty(self):
        reg = RecommendationRegistry()  # gold is None by default
        assert build_gold_steps(reg, {"revenue"}) == []

    def test_gold_is_none_returns_empty(self):
        reg = RecommendationRegistry()
        reg.gold = None
        assert build_gold_steps(reg, {"revenue"}) == []

    def test_no_gold_attr_at_all_returns_empty(self):
        """Registry-like object without 'gold' attribute at all."""

        class FakeRegistry:
            pass

        assert build_gold_steps(FakeRegistry(), {"revenue"}) == []

    def test_unknown_action_is_skipped(self):
        reg = _registry_with_gold(("revenue", "unknown_transform", {}))
        steps = build_gold_steps(reg, {"revenue"})
        assert steps == []

    def test_column_not_in_pipeline_is_skipped(self):
        reg = _registry_with_gold(("revenue", "log", {}))
        steps = build_gold_steps(reg, {"orders"})  # 'revenue' not in set
        assert steps == []

    def test_valid_log_transform(self):
        reg = _registry_with_gold(("revenue", "log", {}))
        steps = build_gold_steps(reg, {"revenue"})
        assert len(steps) == 1
        assert steps[0].type == PipelineTransformationType.LOG_TRANSFORM
        assert steps[0].column == "revenue"

    def test_valid_log_transform_alias(self):
        reg = _registry_with_gold(("revenue", "log_transform", {}))
        steps = build_gold_steps(reg, {"revenue"})
        assert len(steps) == 1
        assert steps[0].type == PipelineTransformationType.LOG_TRANSFORM

    def test_valid_sqrt_transform(self):
        reg = _registry_with_gold(("orders", "sqrt", {}))
        steps = build_gold_steps(reg, {"orders"})
        assert len(steps) == 1
        assert steps[0].type == PipelineTransformationType.SQRT_TRANSFORM

    def test_valid_sqrt_transform_alias(self):
        reg = _registry_with_gold(("orders", "sqrt_transform", {}))
        steps = build_gold_steps(reg, {"orders"})
        assert len(steps) == 1
        assert steps[0].type == PipelineTransformationType.SQRT_TRANSFORM

    def test_valid_yeo_johnson_transform(self):
        reg = _registry_with_gold(("revenue", "yeo_johnson", {"lambda": 0.5}))
        steps = build_gold_steps(reg, {"revenue"})
        assert len(steps) == 1
        assert steps[0].type == PipelineTransformationType.YEO_JOHNSON
        assert steps[0].parameters == {"lambda": 0.5}

    def test_valid_zero_inflation_handling(self):
        reg = _registry_with_gold(("orders", "zero_inflation_handling", {}))
        steps = build_gold_steps(reg, {"orders"})
        assert len(steps) == 1
        assert steps[0].type == PipelineTransformationType.ZERO_INFLATION_HANDLING

    def test_valid_cap_then_log(self):
        reg = _registry_with_gold(("amount", "cap_then_log", {"cap": 99}))
        steps = build_gold_steps(reg, {"amount"})
        assert len(steps) == 1
        assert steps[0].type == PipelineTransformationType.CAP_THEN_LOG
        assert steps[0].parameters == {"cap": 99}

    def test_multiple_transforms_mixed_validity(self):
        """Some valid, one unknown, one missing column -> only valid ones returned."""
        reg = _registry_with_gold(
            ("revenue", "log", {}),
            ("orders", "unknown_op", {}),
            ("missing_col", "sqrt", {}),
            ("amount", "yeo_johnson", {}),
        )
        steps = build_gold_steps(reg, {"revenue", "amount", "orders"})
        assert len(steps) == 2
        columns = {s.column for s in steps}
        assert columns == {"revenue", "amount"}

    def test_rationale_and_source_notebook_propagated(self):
        reg = RecommendationRegistry()
        reg.init_gold("churn")
        reg.add_gold_transformation("revenue", "log", {}, "fix skew", "nb04")
        steps = build_gold_steps(reg, {"revenue"})
        assert steps[0].rationale == "fix skew"
        assert steps[0].source_notebook == "nb04"

    def test_empty_parameters_become_empty_dict(self):
        reg = _registry_with_gold(("revenue", "log", {}))
        steps = build_gold_steps(reg, {"revenue"})
        assert steps[0].parameters == {}

    def test_none_parameters_become_empty_dict(self):
        """When rec.parameters is None, the step should get {}."""
        reg = RecommendationRegistry()
        reg.init_gold("churn")
        reg.add_gold_transformation("revenue", "log", None, "rationale", "04")
        steps = build_gold_steps(reg, {"revenue"})
        assert steps[0].parameters == {}


# ---------------------------------------------------------------------------
# build_silver_derived_steps
# ---------------------------------------------------------------------------

class TestBuildSilverDerivedSteps:
    def test_no_silver_attribute_returns_empty(self):
        reg = RecommendationRegistry()  # silver is None by default
        assert build_silver_derived_steps(reg, {"col_a"}) == []

    def test_silver_is_none_returns_empty(self):
        reg = RecommendationRegistry()
        reg.silver = None
        assert build_silver_derived_steps(reg, {"col_a"}) == []

    def test_no_silver_attr_at_all_returns_empty(self):

        class FakeRegistry:
            pass

        assert build_silver_derived_steps(FakeRegistry(), {"col_a"}) == []

    def test_action_not_ratio_interaction_composite_is_skipped(self):
        """Derived columns with actions like 'polynomial' should be skipped."""
        reg = RecommendationRegistry()
        reg.init_silver("customer_id")
        # Use add_silver_derived with a non-standard feature_type
        reg.add_silver_derived("new_col", "a + b", "polynomial", "rationale", "06")
        steps = build_silver_derived_steps(reg, {"a", "b"})
        assert steps == []

    def test_source_columns_not_in_pipeline_is_skipped(self):
        reg = _registry_with_silver(("ratio", "avg_order", "total_amount", "order_count"))
        # pipeline_columns missing "order_count"
        steps = build_silver_derived_steps(reg, {"total_amount"})
        assert steps == []

    def test_valid_ratio(self):
        reg = _registry_with_silver(("ratio", "avg_order", "total_amount", "order_count"))
        steps = build_silver_derived_steps(reg, {"total_amount", "order_count"})
        assert len(steps) == 1
        assert steps[0].type == PipelineTransformationType.DERIVED_COLUMN
        assert steps[0].column == "avg_order"
        assert steps[0].parameters["action"] == "ratio"
        assert steps[0].parameters["numerator"] == "total_amount"
        assert steps[0].parameters["denominator"] == "order_count"

    def test_valid_interaction(self):
        """interaction uses col_a/col_b in _derived_source_columns but features in add_silver_interaction.

        We need to construct a rec that has col_a and col_b in params for the source check.
        """
        reg = RecommendationRegistry()
        reg.init_silver("customer_id")
        # Manually create a derived column with col_a / col_b params
        from customer_retention.analysis.auto_explorer.layered_recommendations import (
            LayeredRecommendation,
        )

        rec = LayeredRecommendation(
            id="test_1",
            layer="silver",
            category="derived",
            action="interaction",
            target_column="feat_product",
            parameters={"col_a": "feat_x", "col_b": "feat_y"},
            rationale="interaction feature",
            source_notebook="06",
        )
        reg.silver.derived_columns.append(rec)
        steps = build_silver_derived_steps(reg, {"feat_x", "feat_y"})
        assert len(steps) == 1
        assert steps[0].parameters["action"] == "interaction"
        assert steps[0].parameters["col_a"] == "feat_x"
        assert steps[0].parameters["col_b"] == "feat_y"

    def test_valid_composite(self):
        reg = _registry_with_silver(("composite", "score", ["f1", "f2", "f3"]))
        steps = build_silver_derived_steps(reg, {"f1", "f2", "f3"})
        assert len(steps) == 1
        assert steps[0].parameters["action"] == "composite"
        assert steps[0].parameters["columns"] == ["f1", "f2", "f3"]

    def test_composite_missing_one_column_is_skipped(self):
        reg = _registry_with_silver(("composite", "score", ["f1", "f2", "f3"]))
        steps = build_silver_derived_steps(reg, {"f1", "f2"})  # missing f3
        assert steps == []


# ---------------------------------------------------------------------------
# _derived_source_columns
# ---------------------------------------------------------------------------

class TestDerivedSourceColumns:
    def test_ratio_returns_numerator_denominator(self):
        result = _derived_source_columns("ratio", {"numerator": "a", "denominator": "b"})
        assert result == {"a", "b"}

    def test_ratio_missing_keys_returns_non_empty_subset(self):
        result = _derived_source_columns("ratio", {"numerator": "a"})
        assert result == {"a"}

    def test_ratio_empty_params(self):
        result = _derived_source_columns("ratio", {})
        assert result == set()

    def test_interaction_returns_col_a_col_b(self):
        result = _derived_source_columns("interaction", {"col_a": "x", "col_b": "y"})
        assert result == {"x", "y"}

    def test_interaction_missing_one_key(self):
        result = _derived_source_columns("interaction", {"col_a": "x"})
        assert result == {"x"}

    def test_interaction_empty_params(self):
        result = _derived_source_columns("interaction", {})
        assert result == set()

    def test_composite_returns_columns_set(self):
        result = _derived_source_columns("composite", {"columns": ["a", "b", "c"]})
        assert result == {"a", "b", "c"}

    def test_composite_empty_columns_list(self):
        result = _derived_source_columns("composite", {"columns": []})
        assert result == set()

    def test_composite_missing_columns_key(self):
        result = _derived_source_columns("composite", {})
        assert result == set()

    def test_unknown_action_returns_empty_set(self):
        result = _derived_source_columns("polynomial", {"columns": ["a"]})
        assert result == set()

    def test_another_unknown_action(self):
        result = _derived_source_columns("custom_formula", {})
        assert result == set()
