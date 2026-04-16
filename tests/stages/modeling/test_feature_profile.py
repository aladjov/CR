
import math

import pytest
import yaml

from customer_retention.stages.modeling.feature_profile import (
    ColumnProfile,
    FeatureOrigin,
    FeatureProfile,
    StageDecision,
    build_feature_profile,
    compare_feature_profiles,
)


class TestColumnProfile:
    def test_basic_creation(self):
        col = ColumnProfile(dtype="double", non_null_count=100, null_count=5)
        assert col.dtype == "double"
        assert col.non_null_count == 100
        assert col.null_count == 5


class TestFeatureProfile:
    @pytest.fixture
    def sample_profile(self):
        return FeatureProfile(
            stage="exploration", created_at="2024-01-15T10:00:00",
            row_count=1000, target_column="churn",
            features={
                "amount_sum": ColumnProfile("double", 950, 50),
                "email_count": ColumnProfile("integer", 1000, 0),
            },
            excluded={"entity_id": "metadata", "as_of_date": "temporal"},
        )

    def test_feature_count(self, sample_profile):
        assert sample_profile.feature_count == 2

    def test_to_dict_roundtrip(self, sample_profile):
        d = sample_profile.to_dict()
        restored = FeatureProfile.from_dict(d)
        assert restored.stage == "exploration"
        assert restored.row_count == 1000
        assert restored.feature_count == 2
        assert restored.features["amount_sum"].dtype == "double"
        assert restored.features["amount_sum"].null_count == 50
        assert restored.excluded == {"entity_id": "metadata", "as_of_date": "temporal"}

    def test_save_load_roundtrip(self, sample_profile, tmp_path):
        path = tmp_path / "profile.yaml"
        sample_profile.save(path)
        loaded = FeatureProfile.load(path)
        assert loaded is not None
        assert loaded.feature_count == 2
        assert loaded.features["email_count"].non_null_count == 1000

    def test_load_missing_file_returns_none(self, tmp_path):
        assert FeatureProfile.load(tmp_path / "nonexistent.yaml") is None

    def test_to_dict_includes_feature_count(self, sample_profile):
        d = sample_profile.to_dict()
        assert d["feature_count"] == 2


class TestBuildFeatureProfile:
    def test_builds_from_stats(self):
        stats = {
            "col_a": ColumnProfile("double", 90, 10),
            "col_b": ColumnProfile("integer", 100, 0),
        }
        profile = build_feature_profile("production", "churn", 100, stats, {"id": "metadata"})
        assert profile.stage == "production"
        assert profile.feature_count == 2
        assert profile.excluded == {"id": "metadata"}
        assert profile.created_at  # not empty


class TestCompareFeatureProfiles:
    def _make(self, stage, features, row_count=1000):
        return FeatureProfile(
            stage=stage, created_at="2024-01-01", row_count=row_count,
            target_column="churn", features=features,
        )

    def test_identical_profiles_no_discrepancies(self):
        features = {"a": ColumnProfile("double", 1000, 0), "b": ColumnProfile("integer", 900, 100)}
        exp = self._make("exploration", features)
        prod = self._make("production", dict(features))
        assert compare_feature_profiles(exp, prod) == []

    def test_missing_in_production(self):
        exp = self._make("exploration", {"a": ColumnProfile("double", 100, 0), "b": ColumnProfile("double", 100, 0)})
        prod = self._make("production", {"a": ColumnProfile("double", 100, 0)})
        discrepancies = compare_feature_profiles(exp, prod)
        assert any("MISSING" in d and "b" in d for d in discrepancies)

    def test_each_missing_column_is_separate_discrepancy(self):
        exp = self._make("exploration", {"a": ColumnProfile("double", 100, 0), "b": ColumnProfile("double", 100, 0), "c": ColumnProfile("double", 100, 0)})
        prod = self._make("production", {"a": ColumnProfile("double", 100, 0)})
        discrepancies = compare_feature_profiles(exp, prod)
        missing_lines = [d for d in discrepancies if "MISSING" in d]
        assert len(missing_lines) == 2

    def test_each_extra_column_is_separate_discrepancy(self):
        exp = self._make("exploration", {"a": ColumnProfile("double", 100, 0)})
        prod = self._make("production", {"a": ColumnProfile("double", 100, 0), "x": ColumnProfile("double", 100, 0), "y": ColumnProfile("double", 100, 0)})
        discrepancies = compare_feature_profiles(exp, prod)
        extra_lines = [d for d in discrepancies if "EXTRA" in d]
        assert len(extra_lines) == 2

    def test_extra_in_production(self):
        exp = self._make("exploration", {"a": ColumnProfile("double", 100, 0)})
        prod = self._make("production", {"a": ColumnProfile("double", 100, 0), "new_col": ColumnProfile("double", 100, 0)})
        discrepancies = compare_feature_profiles(exp, prod)
        assert any("EXTRA" in d and "new_col" in d for d in discrepancies)

    def test_type_mismatch(self):
        exp = self._make("exploration", {"a": ColumnProfile("double", 100, 0)})
        prod = self._make("production", {"a": ColumnProfile("string", 100, 0)})
        discrepancies = compare_feature_profiles(exp, prod)
        assert any("TYPE MISMATCH" in d and "a" in d for d in discrepancies)

    def test_new_nulls(self):
        exp = self._make("exploration", {"a": ColumnProfile("double", 100, 0)})
        prod = self._make("production", {"a": ColumnProfile("double", 80, 20)})
        discrepancies = compare_feature_profiles(exp, prod)
        assert any("NEW NULLS" in d for d in discrepancies)

    def test_null_drift_detected(self):
        exp = self._make("exploration", {"a": ColumnProfile("double", 900, 100)})
        prod = self._make("production", {"a": ColumnProfile("double", 500, 500)})
        discrepancies = compare_feature_profiles(exp, prod)
        assert any("NULL DRIFT" in d for d in discrepancies)

    def test_null_drift_below_threshold_not_reported(self):
        exp = self._make("exploration", {"a": ColumnProfile("double", 950, 50)})
        prod = self._make("production", {"a": ColumnProfile("double", 920, 80)})
        discrepancies = compare_feature_profiles(exp, prod)
        assert not any("NULL DRIFT" in d for d in discrepancies)

    def test_feature_dropped_in_exploration_not_flagged_as_extra(self):
        exp = FeatureProfile(
            stage="exploration", created_at="2024-01-01", row_count=1000,
            target_column="churn",
            features={"a": ColumnProfile("double", 1000, 0)},
            excluded={"b": "drop_multicollinear"},
        )
        prod = FeatureProfile(
            stage="production", created_at="2024-01-01", row_count=1000,
            target_column="churn",
            features={"a": ColumnProfile("double", 1000, 0)},
            excluded={"b": "drop_multicollinear"},
        )
        assert compare_feature_profiles(exp, prod) == []

    def test_feature_dropped_in_exploration_but_present_in_production(self):
        exp = FeatureProfile(
            stage="exploration", created_at="2024-01-01", row_count=1000,
            target_column="churn",
            features={"a": ColumnProfile("double", 1000, 0)},
            excluded={"b": "drop_l1_zero"},
        )
        prod = FeatureProfile(
            stage="production", created_at="2024-01-01", row_count=1000,
            target_column="churn",
            features={"a": ColumnProfile("double", 1000, 0), "b": ColumnProfile("double", 1000, 0)},
        )
        discrepancies = compare_feature_profiles(exp, prod)
        assert any("SELECTION DRIFT" in d and "b" in d and "drop_l1_zero" in d for d in discrepancies)

    def test_feature_dropped_in_production_but_present_in_exploration(self):
        exp = FeatureProfile(
            stage="exploration", created_at="2024-01-01", row_count=1000,
            target_column="churn",
            features={"a": ColumnProfile("double", 1000, 0), "b": ColumnProfile("double", 1000, 0)},
        )
        prod = FeatureProfile(
            stage="production", created_at="2024-01-01", row_count=1000,
            target_column="churn",
            features={"a": ColumnProfile("double", 1000, 0)},
            excluded={"b": "drop_weak"},
        )
        discrepancies = compare_feature_profiles(exp, prod)
        assert any("SELECTION DRIFT" in d and "b" in d and "drop_weak" in d for d in discrepancies)

    def test_matching_excluded_not_reported_as_missing_or_extra(self):
        exp = FeatureProfile(
            stage="exploration", created_at="2024-01-01", row_count=1000,
            target_column="churn",
            features={"a": ColumnProfile("double", 1000, 0)},
            excluded={"b": "drop_multicollinear", "c": "drop_l1_zero"},
        )
        prod = FeatureProfile(
            stage="production", created_at="2024-01-01", row_count=1000,
            target_column="churn",
            features={"a": ColumnProfile("double", 1000, 0)},
            excluded={"b": "drop_multicollinear", "c": "drop_l1_zero"},
        )
        discrepancies = compare_feature_profiles(exp, prod)
        assert not any("MISSING" in d for d in discrepancies)
        assert not any("EXTRA" in d for d in discrepancies)

    def test_matching_excluded_availability_and_zero_var_not_flagged(self):
        exp = FeatureProfile(
            stage="exploration", created_at="2024-01-01", row_count=1000,
            target_column="churn",
            features={"a": ColumnProfile("double", 1000, 0)},
            excluded={"b": "drop_availability", "c": "drop_zero_variance"},
        )
        prod = FeatureProfile(
            stage="production", created_at="2024-01-01", row_count=1000,
            target_column="churn",
            features={"a": ColumnProfile("double", 1000, 0)},
            excluded={"b": "drop_availability", "c": "drop_zero_variance"},
        )
        discrepancies = compare_feature_profiles(exp, prod)
        assert not any("MISSING" in d for d in discrepancies)
        assert not any("EXTRA" in d for d in discrepancies)

    def test_excluded_reason_mismatch_reported(self):
        exp = FeatureProfile(
            stage="exploration", created_at="2024-01-01", row_count=1000,
            target_column="churn",
            features={"a": ColumnProfile("double", 1000, 0)},
            excluded={"b": "drop_weak"},
        )
        prod = FeatureProfile(
            stage="production", created_at="2024-01-01", row_count=1000,
            target_column="churn",
            features={"a": ColumnProfile("double", 1000, 0)},
            excluded={"b": "drop_l1_zero"},
        )
        discrepancies = compare_feature_profiles(exp, prod)
        assert any("EXCLUSION REASON" in d and "b" in d for d in discrepancies)

    def test_float64_vs_double_not_type_mismatch(self):
        exp = self._make("exploration", {"a": ColumnProfile("float64", 100, 0)})
        prod = self._make("production", {"a": ColumnProfile("double", 100, 0)})
        assert not any("TYPE MISMATCH" in d for d in compare_feature_profiles(exp, prod))

    def test_int32_vs_integer_not_type_mismatch(self):
        exp = self._make("exploration", {"a": ColumnProfile("int32", 100, 0)})
        prod = self._make("production", {"a": ColumnProfile("integer", 100, 0)})
        assert not any("TYPE MISMATCH" in d for d in compare_feature_profiles(exp, prod))

    def test_int64_vs_long_not_type_mismatch(self):
        exp = self._make("exploration", {"a": ColumnProfile("int64", 100, 0)})
        prod = self._make("production", {"a": ColumnProfile("long", 100, 0)})
        assert not any("TYPE MISMATCH" in d for d in compare_feature_profiles(exp, prod))

    def test_float32_vs_float_not_type_mismatch(self):
        exp = self._make("exploration", {"a": ColumnProfile("float32", 100, 0)})
        prod = self._make("production", {"a": ColumnProfile("float", 100, 0)})
        assert not any("TYPE MISMATCH" in d for d in compare_feature_profiles(exp, prod))

    def test_bool_vs_boolean_not_type_mismatch(self):
        exp = self._make("exploration", {"a": ColumnProfile("bool", 100, 0)})
        prod = self._make("production", {"a": ColumnProfile("boolean", 100, 0)})
        assert not any("TYPE MISMATCH" in d for d in compare_feature_profiles(exp, prod))

    def test_genuine_type_mismatch_still_reported(self):
        exp = self._make("exploration", {"a": ColumnProfile("float64", 100, 0)})
        prod = self._make("production", {"a": ColumnProfile("string", 100, 0)})
        assert any("TYPE MISMATCH" in d for d in compare_feature_profiles(exp, prod))

    def test_normalization_is_symmetric(self):
        exp = self._make("exploration", {"a": ColumnProfile("double", 100, 0)})
        prod = self._make("production", {"a": ColumnProfile("float64", 100, 0)})
        assert not any("TYPE MISMATCH" in d for d in compare_feature_profiles(exp, prod))


class TestStageDecision:
    def test_required_fields(self):
        decision = StageDecision(
            stage="variance", score=0.25, score_name="variance",
            threshold=0.01, decision="kept", reason=None, rank=3,
            stage_input_count=100, stage_output_count=80,
        )
        assert decision.stage == "variance"
        assert decision.score == 0.25
        assert decision.decision == "kept"
        assert decision.companion_feature is None

    def test_frozen_dataclass(self):
        decision = StageDecision(
            stage="l1", score=0.5, score_name="l1_abs_coef",
            threshold=0.01, decision="dropped", reason="below_threshold", rank=42,
            stage_input_count=358, stage_output_count=2,
        )
        with pytest.raises((AttributeError, TypeError)):
            decision.stage = "variance"  # type: ignore

    def test_companion_feature_for_correlation(self):
        decision = StageDecision(
            stage="correlation", score=0.97, score_name="abs_pearson_r",
            threshold=0.95, decision="dropped", reason="high_correlation", rank=None,
            stage_input_count=50, stage_output_count=49, companion_feature="winner_col",
        )
        assert decision.companion_feature == "winner_col"

    def test_nan_score_preserved(self):
        decision = StageDecision(
            stage="l1", score=float("nan"), score_name="l1_abs_coef",
            threshold=None, decision="not_evaluated", reason="not_numeric", rank=None,
            stage_input_count=10, stage_output_count=10,
        )
        assert math.isnan(decision.score)


class TestFeatureOrigin:
    def test_all_fields(self):
        origin = FeatureOrigin(
            source="contract", base_column="ACTIVATED_DATE",
            family="hour_max_all_time", lag_prefix=None,
            derivation="bronze_aggregate", parents=(),
        )
        assert origin.source == "contract"
        assert origin.derivation == "bronze_aggregate"
        assert origin.parents == ()

    def test_frozen(self):
        origin = FeatureOrigin(source="contract", base_column="X", family="sum")
        with pytest.raises((AttributeError, TypeError)):
            origin.source = "other"  # type: ignore

    def test_defaults(self):
        origin = FeatureOrigin(source="subscription", base_column="NET_PRICE", family="mean_365d")
        assert origin.lag_prefix is None
        assert origin.derivation is None
        assert origin.parents == ()

    def test_parents_for_derived_ratio(self):
        origin = FeatureOrigin(
            source=None, base_column=None, family="ratio",
            derivation="derived_ratio", parents=("CREATED_DATE_delta_hours", "ANNIVERSARY_DATE__C_dow"),
        )
        assert origin.parents == ("CREATED_DATE_delta_hours", "ANNIVERSARY_DATE__C_dow")


class TestColumnProfileExtensions:
    def test_defaults_are_empty(self):
        col = ColumnProfile(dtype="double", non_null_count=100, null_count=0)
        assert col.origin is None
        assert col.selection_trace == []
        assert col.final_score is None

    def test_accepts_origin(self):
        origin = FeatureOrigin(source="contract", base_column="X", family="")
        col = ColumnProfile(dtype="double", non_null_count=1, null_count=0, origin=origin)
        assert col.origin.source == "contract"

    def test_accepts_trace(self):
        trace = [
            StageDecision(
                stage="variance", score=0.5, score_name="variance",
                threshold=0.01, decision="kept", reason=None, rank=1,
                stage_input_count=10, stage_output_count=10,
            )
        ]
        col = ColumnProfile(dtype="double", non_null_count=1, null_count=0, selection_trace=trace)
        assert col.selection_trace[0].stage == "variance"

    def test_final_score_populated(self):
        col = ColumnProfile(dtype="double", non_null_count=1, null_count=0, final_score=0.865)
        assert col.final_score == 0.865


class TestFeatureProfileSchemaV2:
    def _origin_contract(self):
        return FeatureOrigin(
            source="contract", base_column="ACTIVATED_DATE",
            family="hour_max_all_time", derivation="bronze_aggregate",
        )

    def _decision_variance_kept(self):
        return StageDecision(
            stage="variance", score=0.25, score_name="variance",
            threshold=0.01, decision="kept", reason=None, rank=1,
            stage_input_count=2, stage_output_count=2,
        )

    def _decision_l1_dropped(self):
        return StageDecision(
            stage="l1", score=0.001, score_name="l1_abs_coef",
            threshold=0.01, decision="dropped", reason="below_threshold", rank=42,
            stage_input_count=358, stage_output_count=2,
        )

    def test_schema_version_defaults_2(self):
        profile = FeatureProfile(
            stage="exploration", created_at="2026-04-16",
            row_count=10, target_column="y", features={},
        )
        assert profile.schema_version == 2

    def test_excluded_profiles_defaults_empty(self):
        profile = FeatureProfile(
            stage="exploration", created_at="2026-04-16",
            row_count=10, target_column="y", features={},
        )
        assert profile.excluded_profiles == {}

    def test_roundtrip_empty_extensions_omits_fields(self):
        profile = FeatureProfile(
            stage="exploration", created_at="2026-04-16",
            row_count=10, target_column="y",
            features={"a": ColumnProfile("double", 10, 0)},
        )
        d = profile.to_dict()
        assert d["schema_version"] == 2
        assert "excluded_profiles" not in d or d["excluded_profiles"] == {}
        assert "origin" not in d["features"]["a"]
        assert "selection_trace" not in d["features"]["a"]

    def test_roundtrip_with_origin_on_feature(self):
        profile = FeatureProfile(
            stage="exploration", created_at="2026-04-16",
            row_count=10, target_column="y",
            features={
                "a": ColumnProfile("double", 10, 0, origin=self._origin_contract()),
            },
        )
        restored = FeatureProfile.from_dict(profile.to_dict())
        assert restored.features["a"].origin.source == "contract"
        assert restored.features["a"].origin.base_column == "ACTIVATED_DATE"
        assert restored.features["a"].origin.derivation == "bronze_aggregate"

    def test_roundtrip_with_selection_trace(self):
        profile = FeatureProfile(
            stage="exploration", created_at="2026-04-16",
            row_count=10, target_column="y",
            features={
                "a": ColumnProfile(
                    "double", 10, 0,
                    selection_trace=[self._decision_variance_kept()],
                ),
            },
        )
        restored = FeatureProfile.from_dict(profile.to_dict())
        assert len(restored.features["a"].selection_trace) == 1
        assert restored.features["a"].selection_trace[0].stage == "variance"
        assert restored.features["a"].selection_trace[0].decision == "kept"

    def test_roundtrip_with_excluded_profiles(self):
        profile = FeatureProfile(
            stage="exploration", created_at="2026-04-16",
            row_count=10, target_column="y", features={},
            excluded={"b": "drop_l1_zero"},
            excluded_profiles={
                "b": ColumnProfile(
                    "double", 9, 1, origin=self._origin_contract(),
                    selection_trace=[self._decision_l1_dropped()],
                ),
            },
        )
        restored = FeatureProfile.from_dict(profile.to_dict())
        assert restored.excluded["b"] == "drop_l1_zero"
        assert restored.excluded_profiles["b"].origin.source == "contract"
        assert restored.excluded_profiles["b"].selection_trace[0].reason == "below_threshold"

    def test_save_load_full_roundtrip(self, tmp_path):
        profile = FeatureProfile(
            stage="exploration", created_at="2026-04-16",
            row_count=10, target_column="y",
            features={
                "a": ColumnProfile(
                    "double", 10, 0, origin=self._origin_contract(),
                    selection_trace=[self._decision_variance_kept()],
                    final_score=0.865,
                ),
            },
            excluded={"b": "drop_l1_zero"},
            excluded_profiles={
                "b": ColumnProfile(
                    "double", 9, 1,
                    selection_trace=[self._decision_l1_dropped()],
                ),
            },
        )
        path = tmp_path / "profile.yaml"
        profile.save(path)
        loaded = FeatureProfile.load(path)
        assert loaded.schema_version == 2
        assert loaded.features["a"].final_score == 0.865
        assert loaded.excluded_profiles["b"].selection_trace[0].rank == 42

    def test_load_v1_yaml_backfills_defaults(self, tmp_path):
        v1_yaml = {
            "stage": "exploration", "created_at": "2024-01-01",
            "row_count": 100, "feature_count": 1, "target_column": "y",
            "features": {"a": {"dtype": "double", "non_null": 100, "null_count": 0}},
            "excluded": {"b": "drop_l1_zero"},
        }
        path = tmp_path / "v1.yaml"
        with open(path, "w") as f:
            yaml.dump(v1_yaml, f)
        loaded = FeatureProfile.load(path)
        assert loaded.schema_version == 1
        assert loaded.features["a"].origin is None
        assert loaded.features["a"].selection_trace == []
        assert loaded.excluded_profiles == {}

    def test_companion_feature_roundtrip(self):
        decision = StageDecision(
            stage="correlation", score=0.97, score_name="abs_pearson_r",
            threshold=0.95, decision="dropped", reason="high_correlation", rank=None,
            stage_input_count=50, stage_output_count=49, companion_feature="winner_col",
        )
        profile = FeatureProfile(
            stage="exploration", created_at="2026-04-16",
            row_count=10, target_column="y", features={},
            excluded={"loser": "drop_multicollinear"},
            excluded_profiles={
                "loser": ColumnProfile("double", 10, 0, selection_trace=[decision]),
            },
        )
        restored = FeatureProfile.from_dict(profile.to_dict())
        assert restored.excluded_profiles["loser"].selection_trace[0].companion_feature == "winner_col"

    def test_compare_ignores_new_fields(self):
        exp = FeatureProfile(
            stage="exploration", created_at="2024-01-01",
            row_count=100, target_column="y",
            features={"a": ColumnProfile("double", 100, 0, origin=self._origin_contract())},
        )
        prod = FeatureProfile(
            stage="production", created_at="2024-01-01",
            row_count=100, target_column="y",
            features={"a": ColumnProfile("double", 100, 0)},
        )
        assert compare_feature_profiles(exp, prod) == []


class TestBuildFeatureProfileExtensions:
    def test_accepts_excluded_profiles_kw(self):
        excluded_profiles = {
            "b": ColumnProfile(
                "double", 10, 0,
                selection_trace=[StageDecision(
                    stage="variance", score=0.0, score_name="variance",
                    threshold=0.01, decision="dropped", reason="zero_variance", rank=None,
                    stage_input_count=5, stage_output_count=4,
                )],
            ),
        }
        profile = build_feature_profile(
            "exploration", "y", 100,
            {"a": ColumnProfile("double", 100, 0)},
            {"b": "drop_zero_variance"},
            excluded_profiles=excluded_profiles,
        )
        assert profile.excluded_profiles["b"].selection_trace[0].reason == "zero_variance"

    def test_builds_without_excluded_profiles(self):
        profile = build_feature_profile(
            "exploration", "y", 100,
            {"a": ColumnProfile("double", 100, 0)},
        )
        assert profile.excluded_profiles == {}
        assert profile.schema_version == 2
