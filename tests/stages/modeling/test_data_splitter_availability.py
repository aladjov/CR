from typing import List, Optional

import pandas as pd

from customer_retention.analysis.auto_explorer.findings import FeatureAvailabilityInfo, FeatureAvailabilityMetadata
from customer_retention.stages.modeling.data_splitter import DataSplitter, SplitStrategy, SplitWarning


class TestSplitAvailabilityValidation:
    def _get_warning_by_column(self, warnings: List[SplitWarning], column: str) -> Optional[SplitWarning]:
        return next((w for w in warnings if w.column == column), None)

    def create_sample_df(self) -> pd.DataFrame:
        return pd.DataFrame({
            "feature_timestamp": pd.date_range("2020-01-01", periods=1000, freq="D"),
            "feature_a": range(1000),
            "feature_b": range(1000),
            "target": [0] * 900 + [1] * 100,
        })

    def create_availability_with_new_tracking(self) -> FeatureAvailabilityMetadata:
        return FeatureAvailabilityMetadata(
            data_start="2020-01-01",
            data_end="2022-09-26",
            time_span_days=1000,
            new_tracking=["feature_a"],
            retired_tracking=[],
            partial_window=[],
            features={
                "feature_a": FeatureAvailabilityInfo(
                    first_valid_date="2021-07-01",
                    last_valid_date="2022-09-26",
                    coverage_pct=45.0,
                    availability_type="new_tracking",
                    days_from_start=547,
                    days_before_end=0,
                ),
                "feature_b": FeatureAvailabilityInfo(
                    first_valid_date="2020-01-01",
                    last_valid_date="2022-09-26",
                    coverage_pct=100.0,
                    availability_type="full",
                    days_from_start=0,
                    days_before_end=0,
                ),
            },
        )

    def create_availability_with_retired(self) -> FeatureAvailabilityMetadata:
        return FeatureAvailabilityMetadata(
            data_start="2020-01-01",
            data_end="2022-09-26",
            time_span_days=1000,
            new_tracking=[],
            retired_tracking=["feature_a"],
            partial_window=[],
            features={
                "feature_a": FeatureAvailabilityInfo(
                    first_valid_date="2020-01-01",
                    last_valid_date="2021-06-30",
                    coverage_pct=55.0,
                    availability_type="retired",
                    days_from_start=0,
                    days_before_end=453,
                ),
            },
        )

    def create_full_coverage_availability(self) -> FeatureAvailabilityMetadata:
        return FeatureAvailabilityMetadata(
            data_start="2020-01-01",
            data_end="2022-09-26",
            time_span_days=1000,
            new_tracking=[],
            retired_tracking=[],
            partial_window=[],
            features={
                "feature_a": FeatureAvailabilityInfo(
                    first_valid_date="2020-01-01",
                    last_valid_date="2022-09-26",
                    coverage_pct=100.0,
                    availability_type="full",
                    days_from_start=0,
                    days_before_end=0,
                ),
            },
        )

    def test_no_warning_when_all_features_available(self):
        df = self.create_sample_df()
        splitter = DataSplitter(
            target_column="target",
            strategy=SplitStrategy.TEMPORAL,
            temporal_column="feature_timestamp",
        )
        availability = self.create_full_coverage_availability()
        warnings = splitter.validate_feature_availability(df, availability)
        assert warnings == []

    def test_warning_for_new_tracking_feature(self):
        df = self.create_sample_df()
        splitter = DataSplitter(
            target_column="target",
            strategy=SplitStrategy.TEMPORAL,
            temporal_column="feature_timestamp",
        )
        availability = self.create_availability_with_new_tracking()
        warnings = splitter.validate_feature_availability(df, availability)
        assert len(warnings) >= 1
        feature_a_warning = self._get_warning_by_column(warnings, "feature_a")
        assert feature_a_warning is not None
        assert feature_a_warning.issue == "new_tracking"
        assert "train" in feature_a_warning.recommendation.lower()

    def test_warning_for_retired_feature(self):
        df = self.create_sample_df()
        splitter = DataSplitter(
            target_column="target",
            strategy=SplitStrategy.TEMPORAL,
            temporal_column="feature_timestamp",
        )
        availability = self.create_availability_with_retired()
        warnings = splitter.validate_feature_availability(df, availability)
        assert len(warnings) >= 1
        feature_a_warning = self._get_warning_by_column(warnings, "feature_a")
        assert feature_a_warning is not None
        assert feature_a_warning.issue == "retired"
        assert "test" in feature_a_warning.recommendation.lower()

    def test_multiple_warnings_aggregated(self):
        availability = FeatureAvailabilityMetadata(
            data_start="2020-01-01",
            data_end="2022-09-26",
            time_span_days=1000,
            new_tracking=["feature_a"],
            retired_tracking=["feature_b"],
            partial_window=[],
            features={
                "feature_a": FeatureAvailabilityInfo(
                    first_valid_date="2021-07-01",
                    last_valid_date="2022-09-26",
                    coverage_pct=45.0,
                    availability_type="new_tracking",
                    days_from_start=547,
                    days_before_end=0,
                ),
                "feature_b": FeatureAvailabilityInfo(
                    first_valid_date="2020-01-01",
                    last_valid_date="2021-06-30",
                    coverage_pct=55.0,
                    availability_type="retired",
                    days_from_start=0,
                    days_before_end=453,
                ),
            },
        )
        df = self.create_sample_df()
        splitter = DataSplitter(
            target_column="target",
            strategy=SplitStrategy.TEMPORAL,
            temporal_column="feature_timestamp",
        )
        warnings = splitter.validate_feature_availability(df, availability)
        columns_warned = {w.column for w in warnings}
        assert "feature_a" in columns_warned
        assert "feature_b" in columns_warned

    def test_non_temporal_split_skips_validation(self):
        df = self.create_sample_df()
        splitter = DataSplitter(
            target_column="target",
            strategy=SplitStrategy.RANDOM_STRATIFIED,
        )
        availability = self.create_availability_with_new_tracking()
        warnings = splitter.validate_feature_availability(df, availability)
        assert warnings == []

    def test_warning_includes_recommendation(self):
        df = self.create_sample_df()
        splitter = DataSplitter(
            target_column="target",
            strategy=SplitStrategy.TEMPORAL,
            temporal_column="feature_timestamp",
        )
        availability = self.create_availability_with_new_tracking()
        warnings = splitter.validate_feature_availability(df, availability)
        for warning in warnings:
            assert warning.recommendation
            assert len(warning.recommendation) > 10

    def test_none_availability_returns_empty_warnings(self):
        df = self.create_sample_df()
        splitter = DataSplitter(
            target_column="target",
            strategy=SplitStrategy.TEMPORAL,
            temporal_column="feature_timestamp",
        )
        warnings = splitter.validate_feature_availability(df, None)
        assert warnings == []

    def test_split_result_contains_availability_warnings(self):
        df = self.create_sample_df()
        splitter = DataSplitter(
            target_column="target",
            strategy=SplitStrategy.TEMPORAL,
            temporal_column="feature_timestamp",
        )
        availability = self.create_availability_with_new_tracking()
        result = splitter.split(df, feature_availability=availability)
        assert "availability_warnings" in result.split_info
        assert len(result.split_info["availability_warnings"]) >= 1

    def test_split_without_availability_has_no_warnings_key(self):
        df = self.create_sample_df()
        splitter = DataSplitter(
            target_column="target",
            strategy=SplitStrategy.TEMPORAL,
            temporal_column="feature_timestamp",
        )
        result = splitter.split(df)
        assert "availability_warnings" not in result.split_info or result.split_info["availability_warnings"] == []


class TestSplitWarningDataclass:
    def test_split_warning_creation(self):
        warning = SplitWarning(
            column="feature_a",
            issue="new_tracking",
            severity="warning",
            recommendation="Filter training data to start from 2021-07-01",
        )
        assert warning.column == "feature_a"
        assert warning.issue == "new_tracking"
        assert warning.severity == "warning"

    def test_split_warning_to_dict(self):
        warning = SplitWarning(
            column="feature_a",
            issue="retired",
            severity="warning",
            recommendation="Test data may have missing values for this feature",
        )
        d = warning.to_dict()
        assert d["column"] == "feature_a"
        assert d["issue"] == "retired"
        assert d["severity"] == "warning"
        assert "recommendation" in d
