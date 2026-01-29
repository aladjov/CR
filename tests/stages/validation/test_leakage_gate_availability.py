from typing import List

import pandas as pd

from customer_retention.analysis.auto_explorer.findings import FeatureAvailabilityInfo, FeatureAvailabilityMetadata
from customer_retention.stages.validation.leakage_gate import LeakageGate, LeakageIssue


class TestAvailabilityLeakageChecks:
    def _get_issues_by_check_id(self, result, check_id: str) -> List[LeakageIssue]:
        return [i for i in result.high_issues if i.check_id == check_id]

    def create_sample_df(self) -> pd.DataFrame:
        return pd.DataFrame({
            "feature_a": range(100),
            "feature_b": range(100),
            "target": [0] * 90 + [1] * 10,
        })

    def create_availability_with_unavailable_feature(self) -> FeatureAvailabilityMetadata:
        return FeatureAvailabilityMetadata(
            data_start="2020-01-01",
            data_end="2022-12-31",
            time_span_days=1095,
            new_tracking=["feature_a"],
            retired_tracking=[],
            partial_window=[],
            features={
                "feature_a": FeatureAvailabilityInfo(
                    first_valid_date="2021-07-01",
                    last_valid_date="2022-12-31",
                    coverage_pct=50.0,
                    availability_type="new_tracking",
                    days_from_start=547,
                    days_before_end=0,
                ),
                "feature_b": FeatureAvailabilityInfo(
                    first_valid_date="2020-01-01",
                    last_valid_date="2022-12-31",
                    coverage_pct=100.0,
                    availability_type="full",
                    days_from_start=0,
                    days_before_end=0,
                ),
            },
        )

    def create_availability_with_sparse_feature(self) -> FeatureAvailabilityMetadata:
        return FeatureAvailabilityMetadata(
            data_start="2020-01-01",
            data_end="2022-12-31",
            time_span_days=1095,
            new_tracking=[],
            retired_tracking=[],
            partial_window=[],
            features={
                "feature_a": FeatureAvailabilityInfo(
                    first_valid_date="2020-01-01",
                    last_valid_date="2022-12-31",
                    coverage_pct=40.0,
                    availability_type="full",
                    days_from_start=0,
                    days_before_end=0,
                ),
            },
        )

    def create_full_coverage_availability(self) -> FeatureAvailabilityMetadata:
        return FeatureAvailabilityMetadata(
            data_start="2020-01-01",
            data_end="2022-12-31",
            time_span_days=1095,
            new_tracking=[],
            retired_tracking=[],
            partial_window=[],
            features={
                "feature_a": FeatureAvailabilityInfo(
                    first_valid_date="2020-01-01",
                    last_valid_date="2022-12-31",
                    coverage_pct=100.0,
                    availability_type="full",
                    days_from_start=0,
                    days_before_end=0,
                ),
            },
        )

    def test_lk011_triggers_for_unavailable_feature(self):
        df = self.create_sample_df()
        gate = LeakageGate(target_column="target")
        availability = self.create_availability_with_unavailable_feature()
        result = gate.run(df, feature_availability=availability)
        lk011_issues = self._get_issues_by_check_id(result, "LK011")
        assert len(lk011_issues) >= 1
        assert any(i.feature == "feature_a" for i in lk011_issues)

    def test_lk011_passes_for_available_feature(self):
        df = self.create_sample_df()
        gate = LeakageGate(target_column="target")
        availability = self.create_full_coverage_availability()
        result = gate.run(df, feature_availability=availability)
        assert len(self._get_issues_by_check_id(result, "LK011")) == 0

    def test_lk012_triggers_for_sparse_feature(self):
        df = self.create_sample_df()
        gate = LeakageGate(target_column="target", availability_coverage_threshold=50.0)
        availability = self.create_availability_with_sparse_feature()
        result = gate.run(df, feature_availability=availability)
        lk012_issues = self._get_issues_by_check_id(result, "LK012")
        assert len(lk012_issues) >= 1
        assert any(i.feature == "feature_a" for i in lk012_issues)

    def test_lk012_passes_for_adequate_coverage(self):
        df = self.create_sample_df()
        gate = LeakageGate(target_column="target", availability_coverage_threshold=50.0)
        availability = self.create_full_coverage_availability()
        result = gate.run(df, feature_availability=availability)
        assert len(self._get_issues_by_check_id(result, "LK012")) == 0

    def test_lk011_lk012_independent(self):
        availability = FeatureAvailabilityMetadata(
            data_start="2020-01-01",
            data_end="2022-12-31",
            time_span_days=1095,
            new_tracking=["feature_a"],
            retired_tracking=[],
            partial_window=[],
            features={
                "feature_a": FeatureAvailabilityInfo(
                    first_valid_date="2021-07-01",
                    last_valid_date="2022-12-31",
                    coverage_pct=50.0,
                    availability_type="new_tracking",
                    days_from_start=547,
                    days_before_end=0,
                ),
                "feature_b": FeatureAvailabilityInfo(
                    first_valid_date="2020-01-01",
                    last_valid_date="2022-12-31",
                    coverage_pct=30.0,
                    availability_type="full",
                    days_from_start=0,
                    days_before_end=0,
                ),
            },
        )
        df = self.create_sample_df()
        gate = LeakageGate(target_column="target", availability_coverage_threshold=50.0)
        result = gate.run(df, feature_availability=availability)
        lk011_features = {i.feature for i in self._get_issues_by_check_id(result, "LK011")}
        lk012_features = {i.feature for i in self._get_issues_by_check_id(result, "LK012")}
        assert "feature_a" in lk011_features
        assert "feature_b" in lk012_features

    def test_checks_skip_when_no_availability_metadata(self):
        df = self.create_sample_df()
        gate = LeakageGate(target_column="target")
        result = gate.run(df, feature_availability=None)
        assert len(self._get_issues_by_check_id(result, "LK011")) == 0
        assert len(self._get_issues_by_check_id(result, "LK012")) == 0

    def test_availability_issues_are_high_severity_not_critical(self):
        df = self.create_sample_df()
        gate = LeakageGate(target_column="target")
        availability = self.create_availability_with_unavailable_feature()
        result = gate.run(df, feature_availability=availability)
        assert not any(i.check_id == "LK011" for i in result.critical_issues)
        assert len(self._get_issues_by_check_id(result, "LK011")) >= 1

    def test_lk011_includes_descriptive_message(self):
        df = self.create_sample_df()
        gate = LeakageGate(target_column="target")
        availability = self.create_availability_with_unavailable_feature()
        result = gate.run(df, feature_availability=availability)
        lk011_issues = self._get_issues_by_check_id(result, "LK011")
        assert len(lk011_issues) >= 1
        assert "2021-07-01" in lk011_issues[0].description or "tracking" in lk011_issues[0].description.lower()

    def test_backward_compatible_run_without_availability_param(self):
        df = self.create_sample_df()
        gate = LeakageGate(target_column="target")
        result = gate.run(df)
        assert result is not None
        assert isinstance(result.passed, bool)
