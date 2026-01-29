from typing import List, Optional

import pandas as pd

from customer_retention.analysis.auto_explorer.findings import FeatureAvailabilityInfo, FeatureAvailabilityMetadata
from customer_retention.stages.features.feature_selector import (
    AvailabilityRecommendation,
    FeatureSelector,
)


class TestAvailabilityRecommendations:
    def _get_recommendation_by_column(self, recs: List[AvailabilityRecommendation], column: str) -> Optional[AvailabilityRecommendation]:
        return next((r for r in recs if r.column == column), None)

    def create_sample_df(self) -> pd.DataFrame:
        return pd.DataFrame({
            "feature_a": range(100),
            "feature_b": range(100),
            "target": [0] * 90 + [1] * 10,
        })

    def create_availability_with_issues(self) -> FeatureAvailabilityMetadata:
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

    def test_get_recommendations_returns_list(self):
        selector = FeatureSelector(target_column="target")
        availability = self.create_availability_with_issues()
        recs = selector.get_availability_recommendations(availability)
        assert isinstance(recs, list)

    def test_no_recommendations_for_full_coverage(self):
        selector = FeatureSelector(target_column="target")
        availability = self.create_full_coverage_availability()
        recs = selector.get_availability_recommendations(availability)
        assert len(recs) == 0

    def test_recommendation_for_new_tracking_feature(self):
        selector = FeatureSelector(target_column="target")
        availability = self.create_availability_with_issues()
        recs = selector.get_availability_recommendations(availability)
        feature_a_recs = [r for r in recs if r.column == "feature_a"]
        assert len(feature_a_recs) == 1
        assert feature_a_recs[0].issue_type == "new_tracking"

    def test_recommendation_includes_multiple_options(self):
        selector = FeatureSelector(target_column="target")
        availability = self.create_availability_with_issues()
        recs = selector.get_availability_recommendations(availability)
        feature_a_rec = self._get_recommendation_by_column(recs, "feature_a")
        assert feature_a_rec is not None
        assert len(feature_a_rec.options) >= 3

    def test_recommendation_options_include_data_preserving_methods(self):
        selector = FeatureSelector(target_column="target")
        availability = self.create_availability_with_issues()
        recs = selector.get_availability_recommendations(availability)
        feature_a_rec = self._get_recommendation_by_column(recs, "feature_a")
        assert feature_a_rec is not None
        option_types = [o["type"] for o in feature_a_rec.options]
        assert "add_indicator" in option_types
        assert "impute" in option_types or "filter_window" in option_types
        assert "remove" in option_types

    def test_recommendation_includes_coverage_info(self):
        selector = FeatureSelector(target_column="target")
        availability = self.create_availability_with_issues()
        recs = selector.get_availability_recommendations(availability)
        feature_a_rec = self._get_recommendation_by_column(recs, "feature_a")
        assert feature_a_rec is not None
        assert feature_a_rec.coverage_pct == 50.0

    def test_none_availability_returns_empty_list(self):
        selector = FeatureSelector(target_column="target")
        recs = selector.get_availability_recommendations(None)
        assert recs == []

    def test_retired_feature_recommendation(self):
        availability = FeatureAvailabilityMetadata(
            data_start="2020-01-01",
            data_end="2022-12-31",
            time_span_days=1095,
            new_tracking=[],
            retired_tracking=["legacy_col"],
            partial_window=[],
            features={
                "legacy_col": FeatureAvailabilityInfo(
                    first_valid_date="2020-01-01",
                    last_valid_date="2021-06-30",
                    coverage_pct=55.0,
                    availability_type="retired",
                    days_from_start=0,
                    days_before_end=549,
                ),
            },
        )
        selector = FeatureSelector(target_column="target")
        recs = selector.get_availability_recommendations(availability)
        assert len(recs) == 1
        assert recs[0].column == "legacy_col"
        assert recs[0].issue_type == "retired"

    def test_partial_window_feature_recommendation(self):
        availability = FeatureAvailabilityMetadata(
            data_start="2020-01-01",
            data_end="2022-12-31",
            time_span_days=1095,
            new_tracking=[],
            retired_tracking=[],
            partial_window=["temp_col"],
            features={
                "temp_col": FeatureAvailabilityInfo(
                    first_valid_date="2021-01-01",
                    last_valid_date="2021-12-31",
                    coverage_pct=33.3,
                    availability_type="partial_window",
                    days_from_start=366,
                    days_before_end=365,
                ),
            },
        )
        selector = FeatureSelector(target_column="target")
        recs = selector.get_availability_recommendations(availability)
        assert len(recs) == 1
        assert recs[0].column == "temp_col"
        assert recs[0].issue_type == "partial_window"


class TestAvailabilityRecommendationDataclass:
    def test_recommendation_creation(self):
        rec = AvailabilityRecommendation(
            column="feature_a",
            issue_type="new_tracking",
            coverage_pct=50.0,
            first_valid_date="2021-07-01",
            last_valid_date="2022-12-31",
            options=[
                {"type": "add_indicator", "description": "Create feature_a_available indicator"},
                {"type": "remove", "description": "Remove feature from selection"},
            ],
        )
        assert rec.column == "feature_a"
        assert rec.issue_type == "new_tracking"
        assert len(rec.options) == 2

    def test_recommendation_to_dict(self):
        rec = AvailabilityRecommendation(
            column="col",
            issue_type="retired",
            coverage_pct=40.0,
            first_valid_date="2020-01-01",
            last_valid_date="2021-06-30",
            options=[{"type": "remove", "description": "Remove"}],
        )
        d = rec.to_dict()
        assert d["column"] == "col"
        assert d["issue_type"] == "retired"
        assert "options" in d
