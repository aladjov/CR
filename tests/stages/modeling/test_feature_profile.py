
import pytest

from customer_retention.stages.modeling.feature_profile import (
    ColumnProfile,
    FeatureProfile,
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
