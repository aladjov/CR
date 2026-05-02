"""Tests for `project_codegen_emit_set` and `prune_unreachable_features`.

These two helpers turn the FeatureSpec parity gate from a hard codegen-time
ValueError into a build-time pre-filter at NB08:

- `project_codegen_emit_set` walks the same paths
  `FindingsParser._collect_known_pipeline_columns` walks during the parity
  check, but with `_feature_spec` cleared so no gate fires.
- `prune_unreachable_features` filters NB08's `_feature_names` to those
  reachable in the projected emit set, returning `(kept, dropped)`.

The end-to-end emit-set projection is covered by the existing
`test_findings_parser_*.py` suite (which exercises `_collect_known_pipeline_columns`
through the production code path); these tests focus on the new entry
points + the spec-augmented value_counts harvesting (Fix #2).
"""
from __future__ import annotations

from customer_retention.generators.pipeline_generator.findings_parser import (
    FindingsParser,
    prune_unreachable_features,
)
from customer_retention.stages.modeling.feature_spec import (
    FeatureSpec,
    FittedTransform,
)


class TestPruneUnreachableFeatures:
    def test_kept_features_stay_when_in_emit_set(self):
        kept, dropped = prune_unreachable_features(
            ["a", "b", "c"], emit_set={"a", "b", "c", "d"},
        )
        assert kept == ["a", "b", "c"]
        assert dropped == []

    def test_drops_features_not_in_emit_set(self):
        kept, dropped = prune_unreachable_features(
            ["reachable", "drift_x_y", "another"], emit_set={"reachable", "another"},
        )
        assert kept == ["reachable", "another"]
        assert dropped == ["drift_x_y"]

    def test_protected_columns_never_dropped(self):
        kept, dropped = prune_unreachable_features(
            ["entity_id", "as_of_date", "churn", "feat_a"],
            emit_set={"feat_a"},
            protected=["churn"],
        )
        # entity_id, as_of_date, churn are protected; feat_a is in emit_set
        assert "churn" in kept
        assert "entity_id" in kept
        assert "as_of_date" in kept
        assert "feat_a" in kept
        assert dropped == []

    def test_preserves_input_order(self):
        kept, dropped = prune_unreachable_features(
            ["z", "a", "m", "drop_me"], emit_set={"a", "m", "z"},
        )
        assert kept == ["z", "a", "m"]  # original order

    def test_empty_input(self):
        kept, dropped = prune_unreachable_features([], emit_set={"a", "b"})
        assert kept == []
        assert dropped == []

    def test_protected_kept_when_also_in_emit_set(self):
        # Protected status doesn't cause duplicates
        kept, _ = prune_unreachable_features(
            ["churn"], emit_set={"churn"}, protected=["churn"],
        )
        assert kept == ["churn"]
        assert kept.count("churn") == 1


class TestAugmentValueCountsFromSpec:
    """`Fix #2`: when a categorical value (e.g. `terminate`) was missed by
    the sampled findings' `type_metrics["value_counts"]` but DOES appear in
    a `selected_features` entry of the form `{col}_{val}_count_{window}`,
    the parser augments `_value_counts_by_source` so codegen still emits
    the column.

    Direct test on `FindingsParser._augment_value_counts_from_spec` to keep
    the unit narrow — the end-to-end emit-set behaviour relies on existing
    `_event_aggregated_columns` tests in the rest of this directory.
    """

    def _make_parser_with_state(
        self, *, selected_features, vc_targets, observed_value_counts=None,
    ):
        parser = FindingsParser.__new__(FindingsParser)
        # Minimal state needed by `_augment_value_counts_from_spec`
        parser._bronze_aggregation_overrides = {
            src: {"value_counts_columns": list(cols)}
            for src, cols in vc_targets.items()
        }
        parser._value_counts_by_source = dict(observed_value_counts or {})
        parser._feature_spec = FeatureSpec(
            target_column="churn",
            selected_features=list(selected_features),
            fitted_transforms=[
                FittedTransform(column=c, action="impute", method="median")
                for c in selected_features
            ],
        )
        return parser

    def test_adds_new_value_when_spec_references_unobserved_value(self):
        parser = self._make_parser_with_state(
            selected_features=["event_type_terminate_count_365d"],
            vc_targets={"contract": ["event_type"]},
            observed_value_counts={"contract": {"event_type": ["start"]}},
        )
        parser._augment_value_counts_from_spec()

        values = parser._value_counts_by_source["contract"]["event_type"]
        assert "start" in values
        assert "terminate" in values

    def test_creates_bucket_when_source_was_unseen_in_findings(self):
        parser = self._make_parser_with_state(
            selected_features=["event_type_renewal_count_30d"],
            vc_targets={"contract": ["event_type"]},
            observed_value_counts={},  # findings had no value_counts at all
        )
        parser._augment_value_counts_from_spec()

        assert parser._value_counts_by_source["contract"]["event_type"] == ["renewal"]

    def test_does_not_duplicate_existing_value(self):
        parser = self._make_parser_with_state(
            selected_features=["event_type_start_count_7d"],
            vc_targets={"contract": ["event_type"]},
            observed_value_counts={"contract": {"event_type": ["start"]}},
        )
        parser._augment_value_counts_from_spec()

        assert parser._value_counts_by_source["contract"]["event_type"].count("start") == 1

    def test_ignores_features_not_matching_value_count_pattern(self):
        parser = self._make_parser_with_state(
            selected_features=[
                "event_type_terminate_count_365d",
                "some_unrelated_feature",
                "event_count_30d",        # not value_count shape
                "event_type_avg_30d",     # has _count_ missing
            ],
            vc_targets={"contract": ["event_type"]},
            observed_value_counts={},
        )
        parser._augment_value_counts_from_spec()

        # Only `terminate` was harvested; the other 3 didn't match the regex
        assert parser._value_counts_by_source["contract"]["event_type"] == ["terminate"]

    def test_skips_columns_not_declared_as_value_counts_targets(self):
        # Spec references `cancel_reason_voluntary_count_30d`, but
        # `cancel_reason` is NOT in the bronze override's value_counts_columns,
        # so we must NOT add its value to the bucket.
        parser = self._make_parser_with_state(
            selected_features=[
                "event_type_terminate_count_365d",
                "cancel_reason_voluntary_count_30d",
            ],
            vc_targets={"contract": ["event_type"]},  # not "cancel_reason"
            observed_value_counts={},
        )
        parser._augment_value_counts_from_spec()

        assert parser._value_counts_by_source["contract"]["event_type"] == ["terminate"]
        assert "cancel_reason" not in parser._value_counts_by_source.get("contract", {})

    def test_no_op_when_spec_is_none(self):
        parser = FindingsParser.__new__(FindingsParser)
        parser._feature_spec = None
        parser._bronze_aggregation_overrides = {"contract": {"value_counts_columns": ["event_type"]}}
        parser._value_counts_by_source = {}
        # Should not raise
        parser._augment_value_counts_from_spec()
        assert parser._value_counts_by_source == {}

    def test_no_op_when_no_value_counts_targets(self):
        parser = self._make_parser_with_state(
            selected_features=["event_type_terminate_count_365d"],
            vc_targets={},  # no overrides at all
            observed_value_counts={},
        )
        parser._augment_value_counts_from_spec()
        assert parser._value_counts_by_source == {}
