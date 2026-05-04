"""Tests for FW-12 — target dtype classification + encoding helpers."""
from __future__ import annotations

import pandas as pd
import pytest

from customer_retention.stages.profiling.target_validator import (
    KIND_BINARY_STRING,
    KIND_BOOLEAN,
    KIND_MISSING,
    KIND_MULTI_CLASS_STRING,
    KIND_NUMERIC,
    KIND_SINGLE_VALUE,
    apply_target_encoding,
    auto_binary_encoding,
    classify_target_dtype,
    render_label_map_template,
    validate_target_or_raise,
)


class TestClassifyTargetDtype:
    def test_numeric_int_target_is_numeric(self):
        df = pd.DataFrame({"y": [0, 1, 0, 1, 1]})
        c = classify_target_dtype(df, "y")
        assert c.kind == KIND_NUMERIC
        assert c.proposed_mapping is None
        assert c.is_actionable

    def test_numeric_float_target_is_numeric(self):
        df = pd.DataFrame({"y": [0.0, 1.0, 0.5, 1.0]})
        c = classify_target_dtype(df, "y")
        assert c.kind == KIND_NUMERIC

    def test_boolean_target_is_boolean(self):
        df = pd.DataFrame({"y": [True, False, True, False]})
        c = classify_target_dtype(df, "y")
        assert c.kind == KIND_BOOLEAN
        assert c.is_actionable

    def test_binary_string_target_yields_proposed_mapping(self):
        df = pd.DataFrame({"y": ["Active", "Churned", "Active", "Churned"]})
        c = classify_target_dtype(df, "y")
        assert c.kind == KIND_BINARY_STRING
        assert c.distinct_count == 2
        # Sort order: 'Active' < 'Churned' → 0, 1
        assert c.proposed_mapping == {"Active": 0, "Churned": 1}
        assert c.is_actionable

    def test_multi_class_string_returns_distinct_values(self):
        df = pd.DataFrame({
            "y": ["Reseller", "Distributor", "Retailer",
                  "Reseller", "Distributor", "Retailer"],
        })
        c = classify_target_dtype(df, "y")
        assert c.kind == KIND_MULTI_CLASS_STRING
        assert c.distinct_count == 3
        assert c.proposed_mapping is None
        assert not c.is_actionable
        assert set(c.distinct_values) == {"Reseller", "Distributor", "Retailer"}

    def test_single_value_target_yields_single_value_kind(self):
        df = pd.DataFrame({"y": ["Active"] * 10})
        c = classify_target_dtype(df, "y")
        assert c.kind == KIND_SINGLE_VALUE
        assert c.distinct_count == 1

    def test_missing_target_column(self):
        df = pd.DataFrame({"x": [1, 2, 3]})
        c = classify_target_dtype(df, "y")
        assert c.kind == KIND_MISSING

    def test_distinct_cap_bounds_collected_values(self):
        """High-cardinality target: distinct_count reflects truth, but
        distinct_values is capped at distinct_cap so the diagnostic stays
        readable and the call doesn't OOM."""
        df = pd.DataFrame({"y": [f"v{i % 50}" for i in range(500)]})
        c = classify_target_dtype(df, "y", distinct_cap=10)
        assert c.kind == KIND_MULTI_CLASS_STRING
        assert c.distinct_count == 50
        assert len(c.distinct_values) == 10


class TestAutoBinaryEncoding:
    def test_two_distinct_values_yields_sorted_mapping(self):
        assert auto_binary_encoding(["Active", "Churned"]) == {"Active": 0, "Churned": 1}

    def test_two_distinct_values_sort_is_deterministic(self):
        # Order of input doesn't matter — output is deterministic.
        a = auto_binary_encoding(["Z", "A"])
        b = auto_binary_encoding(["A", "Z"])
        assert a == b == {"A": 0, "Z": 1}

    def test_three_distinct_values_returns_none(self):
        assert auto_binary_encoding(["A", "B", "C"]) is None

    def test_one_distinct_value_returns_none(self):
        assert auto_binary_encoding(["A"]) is None

    def test_filters_none_values(self):
        # `None` should be ignored when counting distinct.
        assert auto_binary_encoding([None, "A", "B"]) == {"A": 0, "B": 1}


class TestApplyTargetEncoding:
    def test_pandas_path_maps_values_and_unmapped_become_na(self):
        df = pd.DataFrame({"y": ["Reseller", "Distributor", "Retailer", "Other"]})
        out = apply_target_encoding(df, "y", {"Reseller": 1, "Distributor": 0, "Retailer": 0})
        # Unmapped 'Other' becomes <NA>; the others map to 1/0/0.
        assert list(out["y"][:3]) == [1, 0, 0]
        assert pd.isna(out["y"].iloc[3])
        assert str(out["y"].dtype) == "Int64"

    def test_empty_mapping_is_no_op(self):
        df = pd.DataFrame({"y": ["A", "B"]})
        out = apply_target_encoding(df, "y", {})
        assert list(out["y"]) == ["A", "B"]

    def test_does_not_mutate_input(self):
        df = pd.DataFrame({"y": ["Reseller", "Distributor"]})
        snapshot = df.copy()
        apply_target_encoding(df, "y", {"Reseller": 1, "Distributor": 0})
        assert list(df["y"]) == list(snapshot["y"])


class TestValidateTargetOrRaise:
    def test_numeric_target_returns_classification(self):
        df = pd.DataFrame({"y": [0, 1, 0, 1]})
        c = validate_target_or_raise(df, "y")
        assert c.kind == KIND_NUMERIC

    def test_binary_string_target_returns_classification_no_raise(self):
        df = pd.DataFrame({"y": ["Active", "Churned"] * 5})
        c = validate_target_or_raise(df, "y")
        assert c.kind == KIND_BINARY_STRING
        assert c.proposed_mapping == {"Active": 0, "Churned": 1}

    def test_multi_class_string_without_map_raises_with_template(self):
        df = pd.DataFrame({"y": ["Reseller", "Distributor", "Retailer"] * 3})
        with pytest.raises(ValueError) as exc:
            validate_target_or_raise(df, "y")
        msg = str(exc.value)
        # Message names the column, lists distinct values, and embeds the
        # paste-ready registry call so the operator copy-pastes once.
        assert "'y'" in msg
        assert "Reseller" in msg and "Distributor" in msg and "Retailer" in msg
        assert "registry.set_target_label_map" in msg
        assert "set to 1 if this value is the positive (churn) class" in msg

    def test_multi_class_string_with_complete_map_passes(self):
        df = pd.DataFrame({"y": ["Reseller", "Distributor", "Retailer"] * 3})
        c = validate_target_or_raise(df, "y", label_map={
            "Reseller": 1, "Distributor": 0, "Retailer": 0,
        })
        assert c.kind == KIND_MULTI_CLASS_STRING

    def test_multi_class_string_with_incomplete_map_raises(self):
        df = pd.DataFrame({"y": ["Reseller", "Distributor", "Retailer"] * 3})
        with pytest.raises(ValueError, match="Retailer"):
            validate_target_or_raise(df, "y", label_map={
                "Reseller": 1, "Distributor": 0,
            })

    def test_single_value_target_raises(self):
        df = pd.DataFrame({"y": ["Only"] * 10})
        with pytest.raises(ValueError, match="only one distinct value"):
            validate_target_or_raise(df, "y")

    def test_missing_target_raises(self):
        df = pd.DataFrame({"x": [1, 2, 3]})
        with pytest.raises(ValueError, match="missing"):
            validate_target_or_raise(df, "y")

    def test_strict_false_does_not_raise_on_multi_class(self):
        df = pd.DataFrame({"y": ["A", "B", "C"] * 3})
        c = validate_target_or_raise(df, "y", strict=False)
        assert c.kind == KIND_MULTI_CLASS_STRING


class TestRenderLabelMapTemplate:
    def test_renders_one_line_per_value_with_zero_default(self):
        out = render_label_map_template(["Reseller", "Distributor"])
        assert "'Reseller': 0," in out
        assert "'Distributor': 0," in out
        assert "set to 1 if" in out

    def test_empty_distinct_yields_safe_placeholder(self):
        out = render_label_map_template([])
        assert "no distinct values observed" in out


class TestRegistryIntegration:
    """`set_target_label_map` is the registry-level API surfaced from NB05.
    Tests the validation + round-trip behavior."""

    def test_set_target_label_map_validates_binary_labels(self):
        from customer_retention.analysis.auto_explorer.layered_recommendations import (
            RecommendationRegistry,
        )
        r = RecommendationRegistry()
        r.init_gold("PARTNER_CLASSIFICATION")
        with pytest.raises(ValueError, match="must be 0 or 1"):
            r.set_target_label_map(
                "PARTNER_CLASSIFICATION",
                {"Reseller": 1, "Distributor": 2},  # bad: 2 is not binary
            )

    def test_set_target_label_map_rejects_empty(self):
        from customer_retention.analysis.auto_explorer.layered_recommendations import (
            RecommendationRegistry,
        )
        r = RecommendationRegistry()
        r.init_gold("y")
        with pytest.raises(ValueError, match="non-empty"):
            r.set_target_label_map("y", {})

    def test_set_target_label_map_rejects_target_mismatch(self):
        from customer_retention.analysis.auto_explorer.layered_recommendations import (
            RecommendationRegistry,
        )
        r = RecommendationRegistry()
        r.init_gold("churn")
        with pytest.raises(ValueError, match="target_column mismatch"):
            r.set_target_label_map("PARTNER_CLASSIFICATION", {"A": 0, "B": 1})

    def test_set_target_label_map_round_trips_through_save_load(self):
        from customer_retention.analysis.auto_explorer.layered_recommendations import (
            RecommendationRegistry,
        )
        r = RecommendationRegistry()
        r.init_gold("PARTNER_CLASSIFICATION")
        r.set_target_label_map(
            "PARTNER_CLASSIFICATION",
            {"Reseller": 1, "Distributor": 0, "Retailer": 0},
            rationale="reseller-as-churn",
            source_notebook="01_data_discovery.ipynb",
        )
        d = r.to_dict()
        r2 = RecommendationRegistry.from_dict(d)
        assert r2.get_target_label_map() == {
            "Reseller": 1, "Distributor": 0, "Retailer": 0,
        }
        assert r2.gold.target_label_map_rationale == "reseller-as-churn"

    def test_recommendations_hash_includes_target_label_map(self):
        """Hash must change when the mapping changes — otherwise codegen
        would silently re-use cached gold against a different mapping."""
        from customer_retention.analysis.auto_explorer.layered_recommendations import (
            RecommendationRegistry,
        )
        r1 = RecommendationRegistry()
        r1.init_gold("y")
        r1.set_target_label_map("y", {"A": 0, "B": 1})
        r2 = RecommendationRegistry()
        r2.init_gold("y")
        r2.set_target_label_map("y", {"A": 1, "B": 0})  # polarity flipped
        assert r1.compute_recommendations_hash() != r2.compute_recommendations_hash()
