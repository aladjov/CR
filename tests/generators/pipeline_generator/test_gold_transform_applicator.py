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


class TestBuildGoldStepsIncludesEncoding:
    """Parity: exploration must apply encodings so _feature_names matches Databricks gold."""

    def test_one_hot_encoding_emitted(self):
        reg = RecommendationRegistry()
        reg.init_gold("churn")
        reg.add_gold_encoding("recency_bucket", "one_hot", "low cardinality", "04")
        steps = build_gold_steps(reg, {"recency_bucket"})
        assert len(steps) == 1
        assert steps[0].type == PipelineTransformationType.ENCODE
        assert steps[0].column == "recency_bucket"
        assert steps[0].parameters == {"method": "one_hot"}

    def test_target_encoding_emitted(self):
        reg = RecommendationRegistry()
        reg.init_gold("churn")
        reg.add_gold_encoding("zip_code", "target", "high cardinality", "04")
        steps = build_gold_steps(reg, {"zip_code"})
        assert len(steps) == 1
        assert steps[0].type == PipelineTransformationType.ENCODE
        assert steps[0].parameters == {"method": "target"}

    def test_onehot_alias_normalised_to_one_hot(self):
        reg = RecommendationRegistry()
        reg.init_gold("churn")
        reg.add_gold_encoding("recency_bucket", "onehot", "low cardinality", "04")
        steps = build_gold_steps(reg, {"recency_bucket"})
        assert steps[0].parameters == {"method": "one_hot"}

    def test_encoding_column_not_in_pipeline_is_skipped(self):
        reg = RecommendationRegistry()
        reg.init_gold("churn")
        reg.add_gold_encoding("stale_col", "one_hot", "low cardinality", "04")
        steps = build_gold_steps(reg, {"other_col"})
        assert steps == []

    def test_encoding_rationale_and_source_notebook_propagated(self):
        reg = RecommendationRegistry()
        reg.init_gold("churn")
        reg.add_gold_encoding("city", "one_hot", "low cardinality", "nb04")
        steps = build_gold_steps(reg, {"city"})
        assert steps[0].rationale == "low cardinality"
        assert steps[0].source_notebook == "nb04"


class TestBuildGoldStepsEncodingDedup:
    """PA-2: parity-of-encoding between NB08 and NB10 codegen.

    NB10's `FindingsParser._apply_gold_recommendations` seeds
    `seen_encoding_columns` with the baseline `one_hot` step from
    `_build_gold_config` (always one_hot for every categorical) and
    skips registry recs whose target column is already in the seen set.
    `build_gold_steps` (called by NB08's `apply_gold_transforms`) must
    follow the same de-dup-with-one_hot-preference rule, otherwise NB08
    runs both `binary` and `one_hot` recs back-to-back: the first
    label-encodes the column to integer codes, the second one-hots
    those codes into positional `_0`/`_1` suffixes. Production codegen
    only runs one_hot, so it emits value-based `_Emerging`/`_Enterprise`
    suffixes, leaving NB08-selected `*_0`/`*_1` features absent from
    gold (engagement spschurn-fa23ccd0 `REVENUE_MARKET_SEGMENT_1`
    parity gap).
    """

    def test_two_encoding_recs_same_column_emit_one_step(self):
        reg = RecommendationRegistry()
        reg.init_gold("churn")
        reg.add_gold_encoding(
            "REVENUE_MARKET_SEGMENT", "binary",
            "Binary categorical - simple 0/1 encoding", "04",
        )
        reg.add_gold_encoding(
            "REVENUE_MARKET_SEGMENT", "one_hot",
            "Low cardinality (4 unique values)", "06",
        )
        steps = build_gold_steps(reg, {"REVENUE_MARKET_SEGMENT"})
        assert len(steps) == 1
        assert steps[0].column == "REVENUE_MARKET_SEGMENT"

    def test_one_hot_wins_when_competing_with_binary(self):
        reg = RecommendationRegistry()
        reg.init_gold("churn")
        reg.add_gold_encoding("MARKET", "binary", "binary", "04")
        reg.add_gold_encoding("MARKET", "one_hot", "low cardinality", "06")
        steps = build_gold_steps(reg, {"MARKET"})
        assert steps[0].parameters == {"method": "one_hot"}
        assert steps[0].source_notebook == "06"

    def test_one_hot_wins_regardless_of_order(self):
        reg = RecommendationRegistry()
        reg.init_gold("churn")
        reg.add_gold_encoding("REGION", "one_hot", "low cardinality", "06")
        reg.add_gold_encoding("REGION", "binary", "binary", "04")
        steps = build_gold_steps(reg, {"REGION"})
        assert steps[0].parameters == {"method": "one_hot"}
        assert steps[0].source_notebook == "06"

    def test_first_rec_wins_when_no_one_hot(self):
        reg = RecommendationRegistry()
        reg.init_gold("churn")
        reg.add_gold_encoding("ZIP", "target", "high cardinality", "04")
        reg.add_gold_encoding("ZIP", "frequency", "high cardinality", "06")
        steps = build_gold_steps(reg, {"ZIP"})
        assert len(steps) == 1
        assert steps[0].parameters == {"method": "target"}
        assert steps[0].source_notebook == "04"

    def test_dedup_preserves_first_seen_order_across_columns(self):
        reg = RecommendationRegistry()
        reg.init_gold("churn")
        reg.add_gold_encoding("CITY", "one_hot", "first", "04")
        reg.add_gold_encoding("STATE", "binary", "second", "04")
        reg.add_gold_encoding("STATE", "one_hot", "second-onehot", "06")
        reg.add_gold_encoding("ZIP", "target", "third", "04")
        steps = build_gold_steps(reg, {"CITY", "STATE", "ZIP"})
        assert [s.column for s in steps] == ["CITY", "STATE", "ZIP"]

    def test_onehot_alias_treated_as_one_hot_for_dedup(self):
        reg = RecommendationRegistry()
        reg.init_gold("churn")
        reg.add_gold_encoding("BRAND", "binary", "binary", "04")
        reg.add_gold_encoding("BRAND", "onehot", "low cardinality", "06")
        steps = build_gold_steps(reg, {"BRAND"})
        assert steps[0].parameters == {"method": "one_hot"}


class TestBuildGoldStepsCategoricalBaseline:
    """PA-2 closure: ``build_gold_steps(..., categorical_columns=...)`` must
    mirror ``FindingsParser._build_gold_config`` baseline.

    The parser unconditionally seeds ``config.gold.encodings`` with one
    ``one_hot`` step per nominal/ordinal/cyclical column from findings,
    THEN iterates registry recs and skips any whose ``target_column``
    is already in the baseline-seeded ``seen_encoding_columns`` set.
    Net effect:
      * Nominal/ordinal/cyclical column with a registered ``binary``
        rec → production emits one_hot (rec is skipped, baseline wins).
      * Nominal/ordinal/cyclical column with NO registered rec →
        production still emits one_hot (baseline-only).
      * Nominal/ordinal/cyclical column with a ``one_hot`` rec →
        production emits one_hot (baseline wins on dedup; rec details
        like rationale are lost — same as the parser).
      * Non-categorical column with any registered rec → behaves as
        before (rec wins).

    ``build_gold_steps`` must mirror this exactly when the caller
    supplies ``categorical_columns`` so NB08's exploration path emits
    the same encoding as production.
    """

    def test_baseline_one_hot_for_categorical_with_no_rec(self):
        reg = RecommendationRegistry()
        reg.init_gold("churn")
        steps = build_gold_steps(
            reg, {"REVENUE_MARKET_SEGMENT", "ACCOUNT_ID"},
            categorical_columns={"REVENUE_MARKET_SEGMENT"},
        )
        assert len(steps) == 1
        assert steps[0].type == PipelineTransformationType.ENCODE
        assert steps[0].column == "REVENUE_MARKET_SEGMENT"
        assert steps[0].parameters == {"method": "one_hot"}

    def test_categorical_with_only_binary_rec_emits_one_hot(self):
        """Mirror parser baseline-wins semantics: binary rec on a
        categorical column is overridden by the always-one_hot baseline.
        """
        reg = RecommendationRegistry()
        reg.init_gold("churn")
        reg.add_gold_encoding(
            "REVENUE_MARKET_SEGMENT", "binary",
            "Binary 2-distinct categorical", "04",
        )
        steps = build_gold_steps(
            reg, {"REVENUE_MARKET_SEGMENT"},
            categorical_columns={"REVENUE_MARKET_SEGMENT"},
        )
        assert len(steps) == 1
        assert steps[0].parameters == {"method": "one_hot"}, (
            "Categorical column with only a binary rec must emit one_hot "
            "to match parser's baseline-wins semantics. Without this, "
            "NB08 would label-encode (binary) while production codegen "
            "one-hots — different feature column names."
        )

    def test_non_categorical_keeps_registered_method(self):
        """Columns NOT marked categorical keep the registered method —
        NB04's `target` encoding for high-cardinality numerics or IDs
        must round-trip through NB08 unchanged.
        """
        reg = RecommendationRegistry()
        reg.init_gold("churn")
        reg.add_gold_encoding("ZIP", "target", "high cardinality", "04")
        steps = build_gold_steps(
            reg, {"ZIP"},
            categorical_columns=set(),  # ZIP not declared categorical
        )
        assert len(steps) == 1
        assert steps[0].parameters == {"method": "target"}

    def test_categorical_baseline_skipped_when_column_not_in_pipeline(self):
        """Categorical column not present in pipeline_columns must NOT
        emit a baseline one_hot — same gate the registered-rec path
        already enforces.
        """
        reg = RecommendationRegistry()
        reg.init_gold("churn")
        steps = build_gold_steps(
            reg, {"OTHER_COL"},
            categorical_columns={"REVENUE_MARKET_SEGMENT"},
        )
        assert steps == []

    def test_target_column_excluded_from_categorical_baseline(self):
        """The target column must never be emitted as a feature
        encoding step — it's the label, not a feature. Parser excludes
        ``target_column`` in ``_build_gold_config``.
        """
        reg = RecommendationRegistry()
        reg.init_gold("churn")
        steps = build_gold_steps(
            reg, {"churn", "REVENUE_MARKET_SEGMENT"},
            categorical_columns={"churn", "REVENUE_MARKET_SEGMENT"},
            target_column="churn",
        )
        assert {s.column for s in steps} == {"REVENUE_MARKET_SEGMENT"}

    def test_categorical_baseline_dedups_against_registered_one_hot(self):
        """A categorical column with a registered one_hot rec must emit
        exactly ONE step, not two (baseline + rec).
        """
        reg = RecommendationRegistry()
        reg.init_gold("churn")
        reg.add_gold_encoding(
            "REVENUE_MARKET_SEGMENT", "one_hot",
            "Low cardinality (4 unique values)", "06",
        )
        steps = build_gold_steps(
            reg, {"REVENUE_MARKET_SEGMENT"},
            categorical_columns={"REVENUE_MARKET_SEGMENT"},
        )
        assert len(steps) == 1
        assert steps[0].parameters == {"method": "one_hot"}

    def test_default_no_categorical_columns_keeps_legacy_behavior(self):
        """When ``categorical_columns`` is None or omitted, build_gold_steps
        keeps existing dedup-with-one_hot-preference semantics. No
        synthetic baseline rows. Backward compat.
        """
        reg = RecommendationRegistry()
        reg.init_gold("churn")
        reg.add_gold_encoding("ZIP", "target", "rationale", "04")
        steps_legacy = build_gold_steps(reg, {"ZIP"})
        assert len(steps_legacy) == 1
        assert steps_legacy[0].parameters == {"method": "target"}
        steps_explicit_none = build_gold_steps(reg, {"ZIP"}, categorical_columns=None)
        assert steps_explicit_none == steps_legacy

    def test_categorical_baseline_preserves_first_seen_order(self):
        """When mixing registered and baseline-emitted columns, the
        deterministic order is the same as the parser: registered recs
        first (in registry order), baseline-only columns after (in the
        order they appear in ``categorical_columns`` iteration). Tests
        the 3-column SPS-shape: one column with a one_hot rec, one with
        only a binary rec, one with no rec at all.
        """
        reg = RecommendationRegistry()
        reg.init_gold("churn")
        reg.add_gold_encoding("REVENUE_MARKET_SEGMENT", "one_hot", "rec", "06")
        reg.add_gold_encoding("INDUSTRY", "binary", "rec", "04")
        steps = build_gold_steps(
            reg, {"REVENUE_MARKET_SEGMENT", "INDUSTRY", "BUSINESS_TYPE"},
            categorical_columns={"REVENUE_MARKET_SEGMENT", "INDUSTRY", "BUSINESS_TYPE"},
        )
        assert {s.column for s in steps} == {
            "REVENUE_MARKET_SEGMENT", "INDUSTRY", "BUSINESS_TYPE",
        }
        for step in steps:
            assert step.parameters == {"method": "one_hot"}


class TestBuildGoldStepsIncludesScaling:
    """Parity: exploration must apply scalings so numeric features match Databricks gold."""

    def test_standard_scaling_emitted(self):
        reg = RecommendationRegistry()
        reg.init_gold("churn")
        reg.add_gold_scaling("revenue", "standard", "normalize", "04")
        steps = build_gold_steps(reg, {"revenue"})
        assert len(steps) == 1
        assert steps[0].type == PipelineTransformationType.SCALE
        assert steps[0].parameters == {"method": "standard"}

    def test_minmax_scaling_emitted(self):
        reg = RecommendationRegistry()
        reg.init_gold("churn")
        reg.add_gold_scaling("orders", "minmax", "normalize", "04")
        steps = build_gold_steps(reg, {"orders"})
        assert len(steps) == 1
        assert steps[0].type == PipelineTransformationType.SCALE
        assert steps[0].parameters == {"method": "minmax"}

    def test_scaling_column_not_in_pipeline_is_skipped(self):
        reg = RecommendationRegistry()
        reg.init_gold("churn")
        reg.add_gold_scaling("stale_col", "standard", "normalize", "04")
        steps = build_gold_steps(reg, {"other_col"})
        assert steps == []


class TestBuildGoldStepsOrdering:
    """Order must match Databricks gold run_gold(): transforms → encodings → scalings."""

    def test_transformations_then_encodings_then_scalings(self):
        reg = RecommendationRegistry()
        reg.init_gold("churn")
        reg.add_gold_transformation("revenue", "log", {}, "skew", "04")
        reg.add_gold_encoding("city", "one_hot", "low card", "04")
        reg.add_gold_scaling("orders", "standard", "normalize", "04")
        steps = build_gold_steps(reg, {"revenue", "city", "orders"})
        assert [s.type for s in steps] == [
            PipelineTransformationType.LOG_TRANSFORM,
            PipelineTransformationType.ENCODE,
            PipelineTransformationType.SCALE,
        ]

    def test_lifecycle_columns_one_hot_encoded_end_to_end(self):
        """Regression for FeatureSpec parity violation: gold missing ['lifecycle_quadrant', 'recency_bucket']."""
        reg = RecommendationRegistry()
        reg.init_gold("churn")
        reg.add_gold_encoding("recency_bucket", "one_hot", "low cardinality", "04")
        reg.add_gold_encoding("lifecycle_quadrant", "one_hot", "low cardinality", "04")
        steps = build_gold_steps(reg, {"recency_bucket", "lifecycle_quadrant", "days_since_last"})
        encoded = [s for s in steps if s.type == PipelineTransformationType.ENCODE]
        assert {s.column for s in encoded} == {"recency_bucket", "lifecycle_quadrant"}
        for step in encoded:
            assert step.parameters == {"method": "one_hot"}


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


# ---------------------------------------------------------------------------
# Sequential silver + gold: no duplicate feature_cols
# ---------------------------------------------------------------------------

class TestSilverThenGoldNoDuplicateFeatures:
    """Reproduce NB08 bug where _cols captured pre-silver caused
    silver-derived columns to appear in _new_cols after gold,
    duplicating entries in feature_cols."""

    def test_feature_cols_no_duplicates_after_silver_and_gold(self):
        import pandas as pd

        from customer_retention.transforms import TransformExecutor

        reg = RecommendationRegistry()
        reg.init_silver("customer_id")
        reg.add_silver_ratio("avg_order", "total_amount", "order_count", "ratio feature", "06")
        reg.init_gold("churn")
        reg.add_gold_transformation("total_amount", "log", {}, "log transform", "04")

        df = pd.DataFrame({
            "total_amount": [100.0, 200.0, 300.0],
            "order_count": [1.0, 2.0, 3.0],
        })
        feature_cols = ["total_amount", "order_count"]
        executor = TransformExecutor()

        silver_steps = build_silver_derived_steps(reg, set(df.columns))
        if silver_steps:
            df = executor.apply_all(df, silver_steps)
            feature_cols = [c for c in df.columns if c in set(feature_cols) | {s.column for s in silver_steps}]

        pre_gold_cols = set(df.columns)
        gold_steps = build_gold_steps(reg, pre_gold_cols)
        if gold_steps:
            df = executor.apply_all(df, gold_steps)
            new_cols = [c for c in df.columns if c not in pre_gold_cols]
            feature_cols = feature_cols + new_cols

        assert len(feature_cols) == len(set(feature_cols)), (
            f"Duplicate feature columns: {[c for c in feature_cols if feature_cols.count(c) > 1]}"
        )

    def test_pre_silver_cols_causes_duplicates(self):
        """Verify that using pre-silver _cols for gold detection DOES produce duplicates."""
        import pandas as pd

        from customer_retention.transforms import TransformExecutor

        reg = RecommendationRegistry()
        reg.init_silver("customer_id")
        reg.add_silver_ratio("avg_order", "total_amount", "order_count", "ratio feature", "06")
        reg.init_gold("churn")
        reg.add_gold_transformation("total_amount", "log", {}, "log transform", "04")

        df = pd.DataFrame({
            "total_amount": [100.0, 200.0, 300.0],
            "order_count": [1.0, 2.0, 3.0],
        })
        feature_cols = ["total_amount", "order_count"]
        executor = TransformExecutor()
        pre_silver_cols = set(df.columns)

        silver_steps = build_silver_derived_steps(reg, pre_silver_cols)
        if silver_steps:
            df = executor.apply_all(df, silver_steps)
            feature_cols = [c for c in df.columns if c in set(feature_cols) | {s.column for s in silver_steps}]

        gold_steps = build_gold_steps(reg, set(df.columns))
        if gold_steps:
            df = executor.apply_all(df, gold_steps)
            new_cols = [c for c in df.columns if c not in pre_silver_cols]
            feature_cols = feature_cols + new_cols

        duplicates = [c for c in feature_cols if feature_cols.count(c) > 1]
        assert len(duplicates) > 0, "Expected duplicates from pre-silver _cols bug"
