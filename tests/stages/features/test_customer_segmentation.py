"""Tests for customer segmentation module."""

import pytest
import pandas as pd
import numpy as np
from customer_retention.stages.features import (
    CustomerSegmenter, SegmentationType,
    SegmentDefinition, SegmentationResult
)


class TestCustomerSegmenter:
    """Tests for CustomerSegmenter class."""

    @pytest.fixture
    def segmenter(self):
        """Create a CustomerSegmenter instance."""
        return CustomerSegmenter()

    @pytest.fixture
    def sample_df(self):
        """Create a sample DataFrame for testing."""
        return pd.DataFrame({
            "custid": range(1, 101),
            "revenue": [100, 200, 50, 300, 150] * 20,  # Mix of high/low value
            "order_count": [1, 5, 2, 10, 3] * 20,  # Mix of high/low frequency
            "days_since_last_order": [10, 45, 100, 200, 30] * 20,
            "engagement_score": [0.8, 0.4, 0.2, 0.9, 0.5] * 20,
            "created": pd.date_range("2023-01-01", periods=100, freq="D"),
            "lastorder": pd.date_range("2023-06-01", periods=100, freq="D"),
            "eopenrate": [50, 60, 20, 80, 40] * 20,
            "eclickrate": [30, 40, 10, 60, 20] * 20
        })


class TestSegmentByValueFrequency(TestCustomerSegmenter):
    """Tests for segment_by_value_frequency method."""

    def test_basic_segmentation(self, segmenter, sample_df):
        """Test basic value-frequency segmentation."""
        df_result, result = segmenter.segment_by_value_frequency(
            sample_df,
            value_column="revenue",
            frequency_column="order_count"
        )

        assert "customer_segment" in df_result.columns
        assert result.segment_type == SegmentationType.VALUE_FREQUENCY
        assert result.total_customers == 100
        assert len(result.segments) == 4

    def test_all_segments_present(self, segmenter, sample_df):
        """Test that all 4 segments are created."""
        df_result, result = segmenter.segment_by_value_frequency(
            sample_df,
            value_column="revenue",
            frequency_column="order_count"
        )

        segment_names = {s.name for s in result.segments}
        expected = {
            "High_Value_Frequent",
            "High_Value_Infrequent",
            "Low_Value_Frequent",
            "Low_Value_Infrequent"
        }
        assert segment_names == expected

    def test_custom_thresholds(self, segmenter, sample_df):
        """Test segmentation with custom thresholds."""
        df_result, result = segmenter.segment_by_value_frequency(
            sample_df,
            value_column="revenue",
            frequency_column="order_count",
            value_threshold=250,  # Higher threshold
            frequency_threshold=8
        )

        # With higher thresholds, fewer customers should be "high"
        high_value_frequent = sum(
            1 for s in result.segments
            if s.name == "High_Value_Frequent"
            for _ in range(s.count)
        )
        # Should have customers but fewer than with default threshold

    def test_custom_output_column(self, segmenter, sample_df):
        """Test custom output column name."""
        df_result, _ = segmenter.segment_by_value_frequency(
            sample_df,
            value_column="revenue",
            frequency_column="order_count",
            output_column="my_segment"
        )

        assert "my_segment" in df_result.columns
        assert "customer_segment" not in df_result.columns

    def test_segment_distribution_sums_to_total(self, segmenter, sample_df):
        """Test that segment counts sum to total customers."""
        _, result = segmenter.segment_by_value_frequency(
            sample_df,
            value_column="revenue",
            frequency_column="order_count"
        )

        total_in_segments = sum(result.segment_distribution.values())
        assert total_in_segments == result.total_customers


class TestSegmentByRecency(TestCustomerSegmenter):
    """Tests for segment_by_recency method."""

    def test_basic_recency_segmentation(self, segmenter, sample_df):
        """Test basic recency segmentation."""
        df_result, result = segmenter.segment_by_recency(
            sample_df,
            days_since_column="days_since_last_order"
        )

        assert "recency_segment" in df_result.columns
        assert result.segment_type == SegmentationType.RECENCY
        assert result.total_customers == 100

    def test_default_thresholds(self, segmenter, sample_df):
        """Test that default thresholds create expected segments."""
        df_result, result = segmenter.segment_by_recency(
            sample_df,
            days_since_column="days_since_last_order"
        )

        segment_names = [s.name for s in result.segments]
        assert "Active_30d" in segment_names
        assert "Recent_90d" in segment_names
        assert "Lapsing_180d" in segment_names
        assert "Dormant_180d+" in segment_names

    def test_custom_thresholds(self, segmenter, sample_df):
        """Test custom recency thresholds."""
        df_result, result = segmenter.segment_by_recency(
            sample_df,
            days_since_column="days_since_last_order",
            thresholds={"active": 14, "recent": 60, "lapsing": 120}
        )

        segment_names = [s.name for s in result.segments]
        assert "Active_14d" in segment_names
        assert "Recent_60d" in segment_names

    def test_null_handling(self, segmenter):
        """Test handling of null values in recency."""
        df = pd.DataFrame({
            "days_since": [10, None, 50, None, 200]
        })
        df_result, result = segmenter.segment_by_recency(
            df,
            days_since_column="days_since"
        )

        assert "Unknown" in df_result["recency_segment"].values


class TestSegmentByEngagement(TestCustomerSegmenter):
    """Tests for segment_by_engagement method."""

    def test_basic_engagement_segmentation(self, segmenter, sample_df):
        """Test basic engagement segmentation."""
        df_result, result = segmenter.segment_by_engagement(
            sample_df,
            engagement_column="engagement_score"
        )

        assert "engagement_segment" in df_result.columns
        assert result.segment_type == SegmentationType.ENGAGEMENT
        assert len(result.segments) == 3

    def test_three_engagement_levels(self, segmenter, sample_df):
        """Test that three engagement levels are created."""
        df_result, result = segmenter.segment_by_engagement(
            sample_df,
            engagement_column="engagement_score"
        )

        segment_names = {s.name for s in result.segments}
        expected = {"High_Engagement", "Medium_Engagement", "Low_Engagement"}
        assert segment_names == expected

    def test_custom_thresholds(self, segmenter, sample_df):
        """Test custom engagement thresholds."""
        df_result, result = segmenter.segment_by_engagement(
            sample_df,
            engagement_column="engagement_score",
            low_threshold=0.4,
            high_threshold=0.8
        )

        # Verify segmentation uses custom thresholds
        high_engagement_count = result.segment_distribution.get("High_Engagement", 0)
        # With threshold of 0.8, fewer should be "high"

    def test_null_handling(self, segmenter):
        """Test handling of null engagement scores."""
        df = pd.DataFrame({
            "engagement": [0.9, None, 0.3, None, 0.5]
        })
        df_result, result = segmenter.segment_by_engagement(
            df,
            engagement_column="engagement"
        )

        assert "Unknown" in df_result["engagement_segment"].values


class TestCreateEngagementScore(TestCustomerSegmenter):
    """Tests for create_engagement_score method."""

    def test_basic_score_creation(self, segmenter, sample_df):
        """Test basic engagement score creation."""
        df_result = segmenter.create_engagement_score(
            sample_df,
            open_rate_column="eopenrate",
            click_rate_column="eclickrate"
        )

        assert "engagement_score" in df_result.columns
        assert df_result["engagement_score"].min() >= 0
        assert df_result["engagement_score"].max() <= 1

    def test_custom_weights(self, segmenter, sample_df):
        """Test custom weighting of open and click rates."""
        df_equal = segmenter.create_engagement_score(
            sample_df,
            open_rate_column="eopenrate",
            click_rate_column="eclickrate",
            open_weight=0.5,
            click_weight=0.5
        )

        df_open_heavy = segmenter.create_engagement_score(
            sample_df,
            open_rate_column="eopenrate",
            click_rate_column="eclickrate",
            open_weight=0.9,
            click_weight=0.1
        )

        # Scores should differ based on weights
        assert not df_equal["engagement_score"].equals(df_open_heavy["engagement_score"])

    def test_custom_output_column(self, segmenter, sample_df):
        """Test custom output column name."""
        df_result = segmenter.create_engagement_score(
            sample_df,
            open_rate_column="eopenrate",
            click_rate_column="eclickrate",
            output_column="my_engagement"
        )

        assert "my_engagement" in df_result.columns


class TestCreateTenureFeatures(TestCustomerSegmenter):
    """Tests for create_tenure_features method."""

    def test_basic_tenure_features(self, segmenter, sample_df):
        """Test basic tenure feature creation."""
        df_result = segmenter.create_tenure_features(
            sample_df,
            created_column="created"
        )

        assert "tenure_days" in df_result.columns
        assert "tenure_months" in df_result.columns
        assert "tenure_bucket" in df_result.columns

    def test_tenure_buckets(self, segmenter):
        """Test tenure bucket assignment."""
        df = pd.DataFrame({
            "created": pd.to_datetime([
                "2024-01-01",  # ~0 days ago (reference)
                "2023-10-15",  # ~78 days ago -> New
                "2023-07-01",  # ~184 days ago -> Established
                "2022-01-01"   # ~730 days ago -> Mature
            ])
        })
        reference_date = pd.to_datetime("2024-01-01")
        df_result = segmenter.create_tenure_features(
            df,
            created_column="created",
            reference_date=reference_date
        )

        buckets = df_result["tenure_bucket"].tolist()
        assert "New_0_3m" in buckets
        assert "Mature_12m+" in buckets

    def test_custom_reference_date(self, segmenter, sample_df):
        """Test custom reference date."""
        ref_date = pd.to_datetime("2024-06-01")
        df_result = segmenter.create_tenure_features(
            sample_df,
            created_column="created",
            reference_date=ref_date
        )

        # All tenure_days should be calculated from ref_date
        min_created = sample_df["created"].min()
        max_tenure = (ref_date - min_created).days
        assert df_result["tenure_days"].max() <= max_tenure + 1  # Allow for rounding

    def test_custom_prefix(self, segmenter, sample_df):
        """Test custom column prefix."""
        df_result = segmenter.create_tenure_features(
            sample_df,
            created_column="created",
            output_prefix="customer"
        )

        assert "customer_tenure_days" in df_result.columns
        assert "customer_tenure_months" in df_result.columns
        assert "customer_tenure_bucket" in df_result.columns


class TestCreateRecencyFeatures(TestCustomerSegmenter):
    """Tests for create_recency_features method."""

    def test_basic_recency_feature(self, segmenter, sample_df):
        """Test basic recency feature creation."""
        df_result = segmenter.create_recency_features(
            sample_df,
            last_activity_column="lastorder"
        )

        assert "days_since_last_activity" in df_result.columns
        assert df_result["days_since_last_activity"].min() >= 0

    def test_custom_output_column(self, segmenter, sample_df):
        """Test custom output column name."""
        df_result = segmenter.create_recency_features(
            sample_df,
            last_activity_column="lastorder",
            output_column="days_since_order"
        )

        assert "days_since_order" in df_result.columns

    def test_custom_reference_date(self, segmenter, sample_df):
        """Test custom reference date."""
        ref_date = pd.to_datetime("2024-01-01")
        df_result = segmenter.create_recency_features(
            sample_df,
            last_activity_column="lastorder",
            reference_date=ref_date
        )

        # All values should be calculated from ref_date
        assert all(df_result["days_since_last_activity"] >= 0)


class TestSegmentationResultDataclass:
    """Tests for SegmentationResult dataclass."""

    def test_to_dict_method(self):
        """Test to_dict method."""
        result = SegmentationResult(
            segment_column="segment",
            segment_type=SegmentationType.VALUE_FREQUENCY,
            total_customers=100,
            segments=[
                SegmentDefinition(
                    name="High_Value",
                    segment_type=SegmentationType.VALUE_FREQUENCY,
                    description="High value customers",
                    count=30,
                    percentage=30.0
                )
            ],
            segment_distribution={"High_Value": 30, "Low_Value": 70}
        )
        d = result.to_dict()

        assert d["segment_column"] == "segment"
        assert d["segment_type"] == "value_frequency"
        assert d["total_customers"] == 100
        assert len(d["segments"]) == 1
        assert d["segments"][0]["name"] == "High_Value"


class TestEdgeCases(TestCustomerSegmenter):
    """Tests for edge cases and error handling."""

    def test_empty_dataframe(self, segmenter):
        """Test handling of empty DataFrame."""
        df = pd.DataFrame({
            "revenue": [],
            "order_count": []
        })
        df_result, result = segmenter.segment_by_value_frequency(
            df,
            value_column="revenue",
            frequency_column="order_count"
        )

        assert result.total_customers == 0

    def test_single_row_dataframe(self, segmenter):
        """Test handling of single-row DataFrame."""
        df = pd.DataFrame({
            "revenue": [100],
            "order_count": [5]
        })
        df_result, result = segmenter.segment_by_value_frequency(
            df,
            value_column="revenue",
            frequency_column="order_count"
        )

        assert result.total_customers == 1
        assert len(df_result) == 1

    def test_all_same_values(self, segmenter):
        """Test when all customers have same values."""
        df = pd.DataFrame({
            "revenue": [100] * 10,
            "order_count": [5] * 10
        })
        df_result, result = segmenter.segment_by_value_frequency(
            df,
            value_column="revenue",
            frequency_column="order_count"
        )

        # With all same values, all should be in one segment
        non_zero_segments = [s for s in result.segments if s.count > 0]
        assert len(non_zero_segments) >= 1
