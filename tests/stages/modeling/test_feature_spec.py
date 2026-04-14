import pytest
import yaml

from customer_retention.stages.modeling.feature_spec import (
    FeatureSpec,
    FittedTransform,
    LeakageExclusion,
    NB05Drops,
    SplitSpec,
    TransformStep,
    Verdict,
    build_fitted_impute_transforms,
)


def _minimal_valid_spec() -> FeatureSpec:
    return FeatureSpec(
        exploration_run_id="email-4f155d9c",
        target_column="unsubscribed",
        entity_column="customer_id",
        timestamp_column="as_of_date",
        horizon_days=180,
        selected_features=["event_count_180d", "send_hour_sum_180d"],
        fitted_transforms=[
            FittedTransform(column="event_count_180d", action="impute", method="median"),
            FittedTransform(column="send_hour_sum_180d", action="impute", method="median"),
        ],
    )


class TestLeakageExclusion:
    def test_fields_default_to_hard_defaults(self):
        excl = LeakageExclusion(column="event_count_365d")
        assert excl.column == "event_count_365d"
        assert excl.code == ""
        assert excl.severity == "HIGH"
        assert excl.rationale == ""

    def test_roundtrip_dict(self):
        excl = LeakageExclusion(
            column="event_count_365d", code="LD063",
            severity="HIGH", rationale="180d horizon inside 365d window",
        )
        restored = LeakageExclusion.from_dict(excl.to_dict())
        assert restored == excl

    def test_from_dict_rejects_flat_string(self):
        with pytest.raises(TypeError, match="flat-list"):
            LeakageExclusion.from_dict("just_a_column_name")


class TestTransformStep:
    def test_roundtrip_preserves_parameters(self):
        step = TransformStep(
            column="event_count_180d",
            action="zero_inflation_handling",
            parameters={"strategy": "separate_indicator", "transform_non_zero": "log"},
            source_notebook="04_column_deep_dive",
        )
        restored = TransformStep.from_dict(step.to_dict())
        assert restored == step


class TestFittedTransform:
    def test_default_strategy_source_empty(self):
        ft = FittedTransform(column="c", action="impute", method="median")
        assert ft.strategy_source == ""
        assert ft.exploration_null_count == 0

    def test_roundtrip(self):
        ft = FittedTransform(
            column="time_to_open_hours_mean_180d", action="impute",
            method="zero_with_indicator", exploration_null_count=412,
            strategy_source="zero_inflation_opt_in",
        )
        restored = FittedTransform.from_dict(ft.to_dict())
        assert restored == ft


class TestVerdict:
    def test_valid_statuses_accepted(self):
        for status in ["solid", "caution", "unstable", "overfit", "leaky"]:
            Verdict(status=status)

    def test_invalid_status_raises(self):
        with pytest.raises(ValueError, match="status"):
            Verdict(status="excellent")

    def test_roundtrip(self):
        v = Verdict(
            status="unstable", cv_mean=0.665, cv_std=0.279,
            fold_aucs=[0.9432, 0.3861], best_model="random_forest",
            critical_issues=["random_forest: CV stability critical"],
        )
        restored = Verdict.from_dict(v.to_dict())
        assert restored == v


class TestSplitSpec:
    def test_defaults_match_plan(self):
        s = SplitSpec()
        assert s.strategy == "temporal_entity_aware"
        assert s.temporal_column == "as_of_date"
        assert s.test_size == 0.2
        assert s.random_state == 42
        assert s.cutoff_date is None


class TestFeatureSpecValidation:
    def test_valid_spec_passes(self):
        spec = _minimal_valid_spec()
        spec.validate()

    def test_empty_selected_features_raises(self):
        spec = _minimal_valid_spec()
        spec.selected_features = []
        with pytest.raises(ValueError, match="selected_features"):
            spec.validate()

    def test_duplicate_selected_features_raises(self):
        spec = _minimal_valid_spec()
        spec.selected_features = ["a", "a", "b"]
        with pytest.raises(ValueError, match="unique"):
            spec.validate()

    def test_missing_impute_for_selected_feature_raises(self):
        spec = _minimal_valid_spec()
        spec.selected_features.append("new_feature_never_imputed")
        with pytest.raises(ValueError, match="impute"):
            spec.validate()

    def test_leakage_excluded_column_in_selected_raises(self):
        spec = _minimal_valid_spec()
        spec.leakage_exclusions = [
            LeakageExclusion(column="event_count_180d", code="LD063"),
        ]
        with pytest.raises(ValueError, match="leakage"):
            spec.validate()


class TestFeatureSpecSerialization:
    def test_yaml_roundtrip_preserves_order(self, tmp_path):
        spec = FeatureSpec(
            exploration_run_id="email-4f155d9c",
            target_column="unsubscribed",
            entity_column="customer_id",
            timestamp_column="as_of_date",
            horizon_days=180,
            selected_features=[
                "event_count_180d",
                "send_hour_sum_180d",
                "time_to_open_hours_mean_180d",
            ],
            transform_topology=[
                TransformStep(
                    column="event_count_180d", action="zero_inflation_handling",
                    parameters={"strategy": "separate_indicator", "transform_non_zero": "log"},
                    source_notebook="04_column_deep_dive",
                ),
            ],
            fitted_transforms=[
                FittedTransform(column="event_count_180d", action="impute", method="median",
                                strategy_source="default_numeric"),
                FittedTransform(column="send_hour_sum_180d", action="impute", method="median",
                                strategy_source="default_numeric"),
                FittedTransform(column="time_to_open_hours_mean_180d", action="impute",
                                method="zero_with_indicator", exploration_null_count=412,
                                strategy_source="zero_inflation_opt_in"),
                FittedTransform(column="event_count_180d", action="scale", method="standard"),
            ],
            leakage_exclusions=[
                LeakageExclusion(column="event_count_365d", code="LD063", severity="HIGH",
                                 rationale="180d label horizon inside 365d window"),
            ],
            nb05_drops=NB05Drops(permanent=["send_hour_mean_180d"], advisory=["some_correlated_col"]),
            split=SplitSpec(purge_gap_days=104, cutoff_date="2021-11-29"),
            verdict=Verdict(status="unstable", cv_mean=0.665, cv_std=0.279,
                            fold_aucs=[0.9432, 0.3861], best_model="random_forest"),
            holdout_entity_ids_path="runs/email-4f155d9c/holdout_entity_ids.json",
            holdout_entity_count=96000,
        )
        path = tmp_path / "feature_spec.yaml"
        spec.save(path)
        loaded = FeatureSpec.load(path)
        assert loaded.selected_features == spec.selected_features
        assert loaded.transform_topology[0].parameters["strategy"] == "separate_indicator"
        assert loaded.fitted_transforms[2].method == "zero_with_indicator"
        assert loaded.fitted_transforms[2].exploration_null_count == 412
        assert loaded.leakage_exclusions[0].code == "LD063"
        assert loaded.nb05_drops.permanent == ["send_hour_mean_180d"]
        assert loaded.split.purge_gap_days == 104
        assert loaded.verdict.status == "unstable"
        assert loaded.verdict.fold_aucs == [0.9432, 0.3861]
        assert loaded.holdout_entity_count == 96000

    def test_selected_features_preserved_in_yaml_order(self, tmp_path):
        spec = _minimal_valid_spec()
        spec.selected_features = ["z_first", "a_second", "m_third"]
        spec.fitted_transforms = [
            FittedTransform(column=c, action="impute", method="median")
            for c in spec.selected_features
        ]
        path = tmp_path / "feature_spec.yaml"
        spec.save(path)
        with open(path) as f:
            raw = yaml.safe_load(f)
        assert raw["selected_features"] == ["z_first", "a_second", "m_third"]
        loaded = FeatureSpec.load(path)
        assert loaded.selected_features == ["z_first", "a_second", "m_third"]

    def test_save_and_load_idempotent(self, tmp_path):
        spec = _minimal_valid_spec()
        path = tmp_path / "spec.yaml"
        spec.save(path)
        first = FeatureSpec.load(path)
        first.save(path)
        second = FeatureSpec.load(path)
        assert first.to_dict() == second.to_dict()

    def test_load_validates_by_default(self, tmp_path):
        path = tmp_path / "bad.yaml"
        bad = {
            "spec_version": 1,
            "selected_features": ["a", "b"],
            "fitted_transforms": [],
        }
        with open(path, "w") as f:
            yaml.dump(bad, f)
        with pytest.raises(ValueError, match="impute"):
            FeatureSpec.load(path)

    def test_load_skip_validation(self, tmp_path):
        path = tmp_path / "bad.yaml"
        bad = {"spec_version": 1, "selected_features": [], "fitted_transforms": []}
        with open(path, "w") as f:
            yaml.dump(bad, f)
        spec = FeatureSpec.load(path, validate=False)
        assert spec.selected_features == []


class TestFeatureSpecHardBlock:
    @pytest.mark.parametrize("status,hard_block", [
        ("solid", False), ("caution", False), ("unstable", False),
        ("overfit", True), ("leaky", True),
    ])
    def test_hard_block_membership(self, status, hard_block):
        v = Verdict(status=status)
        assert v.is_hard_block() is hard_block


class TestToTransformationStep:
    def test_stateless_log_transform_maps_to_log(self):
        step = TransformStep(column="c", action="log_transform", parameters={"clip": 0.0})
        ts = step.to_transformation_step()
        assert ts.type.value == "log_transform"
        assert ts.column == "c"
        assert ts.parameters == {"clip": 0.0}

    def test_stateless_yj_transform_maps_to_yeo_johnson(self):
        step = TransformStep(column="c", action="yj_transform")
        assert step.to_transformation_step().type.value == "yeo_johnson"

    def test_stateless_zero_inflation_handling_maps_directly(self):
        step = TransformStep(column="x", action="zero_inflation_handling",
                             parameters={"strategy": "separate_indicator"})
        ts = step.to_transformation_step()
        assert ts.type.value == "zero_inflation_handling"
        assert ts.parameters["strategy"] == "separate_indicator"

    def test_stateless_unknown_action_raises(self):
        with pytest.raises(ValueError, match="unknown stateless action"):
            TransformStep(column="c", action="bogus_transform").to_transformation_step()

    def test_fitted_impute_merges_method_into_parameters(self):
        ft = FittedTransform(column="x", action="impute", method="median")
        ts = ft.to_transformation_step()
        assert ts.type.value == "impute_null"
        assert ts.parameters["method"] == "median"

    def test_fitted_scale_standard(self):
        ft = FittedTransform(column="y", action="scale", method="standard")
        ts = ft.to_transformation_step()
        assert ts.type.value == "scale"
        assert ts.parameters["method"] == "standard"

    def test_fitted_unknown_action_raises(self):
        with pytest.raises(ValueError, match="unknown stateful action"):
            FittedTransform(column="x", action="bogus", method="m").to_transformation_step()


class TestBuildFittedImputeTransforms:
    def test_numeric_default_to_median(self):
        ft = build_fitted_impute_transforms(
            selected_features=["revenue_sum_180d", "count_30d"],
            feature_kinds={"revenue_sum_180d": "numeric", "count_30d": "numeric"},
            zero_inflation_bases=[],
        )
        assert [t.method for t in ft] == ["median", "median"]
        assert [t.strategy_source for t in ft] == ["default_numeric", "default_numeric"]

    def test_categorical_default_to_mode(self):
        ft = build_fitted_impute_transforms(
            selected_features=["CASE_TYPE"],
            feature_kinds={"CASE_TYPE": "categorical"},
            zero_inflation_bases=[],
        )
        assert ft[0].method == "mode"
        assert ft[0].strategy_source == "default_categorical"

    def test_zero_inflation_opt_in_uses_zero_with_indicator(self):
        ft = build_fitted_impute_transforms(
            selected_features=["FIRST_RESPONSE_TIME_count_180d", "OTHER_COL_sum_30d"],
            feature_kinds={
                "FIRST_RESPONSE_TIME_count_180d": "numeric",
                "OTHER_COL_sum_30d": "numeric",
            },
            zero_inflation_bases=["FIRST_RESPONSE_TIME"],
        )
        assert ft[0].method == "zero_with_indicator"
        assert ft[0].strategy_source == "zero_inflation_opt_in"
        assert ft[1].method == "median"
        assert ft[1].strategy_source == "default_numeric"

    def test_null_counts_surface_into_transform(self):
        ft = build_fitted_impute_transforms(
            selected_features=["x"], feature_kinds={"x": "numeric"},
            zero_inflation_bases=[], null_counts={"x": 412},
        )
        assert ft[0].exploration_null_count == 412

    def test_covers_every_selected_feature(self):
        selected = ["a_count_30d", "b_mean_90d", "c", "D_sum_180d_is_zero"]
        ft = build_fitted_impute_transforms(
            selected_features=selected, feature_kinds={}, zero_inflation_bases=[],
        )
        assert [t.column for t in ft] == selected
        assert all(t.action == "impute" for t in ft)
