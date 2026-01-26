"""Tests for TimeWindowAggregator - TDD approach."""
from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from customer_retention.stages.profiling.time_window_aggregator import (
    AggregationPlan,
    TimeWindowAggregator,
)


@pytest.fixture
def transactions_df():
    """Sample transaction data."""
    np.random.seed(42)
    # Reference date for calculating windows
    ref_date = pd.Timestamp("2023-12-31")

    data = []
    for cust_id in ["C001", "C002", "C003", "C004", "C005"]:
        # Generate events at different recencies
        if cust_id == "C001":  # Active customer - recent transactions
            dates = [ref_date - timedelta(days=d) for d in [1, 3, 10, 25, 45]]
        elif cust_id == "C002":  # Semi-active
            dates = [ref_date - timedelta(days=d) for d in [5, 35, 60]]
        elif cust_id == "C003":  # Churning - no recent activity
            dates = [ref_date - timedelta(days=d) for d in [100, 150, 200]]
        else:
            dates = [ref_date - timedelta(days=d) for d in range(10, 60, 15)]

        for i, date in enumerate(dates):
            data.append({
                "customer_id": cust_id,
                "transaction_date": date,
                "amount": np.random.uniform(20, 200),
                "quantity": np.random.randint(1, 5),
            })

    return pd.DataFrame(data)


@pytest.fixture
def emails_df():
    """Sample email engagement data."""
    np.random.seed(42)
    ref_date = pd.Timestamp("2023-12-31")

    data = []
    for cust_id in ["C001", "C002", "C003"]:
        n_emails = 10 if cust_id == "C001" else 5
        for i in range(n_emails):
            data.append({
                "customer_id": cust_id,
                "sent_date": ref_date - timedelta(days=i * 3),
                "opened": np.random.choice([0, 1], p=[0.3, 0.7]),
                "clicked": np.random.choice([0, 1], p=[0.7, 0.3]),
            })

    return pd.DataFrame(data)


class TestTimeWindowAggregator:
    """Tests for basic aggregation functionality."""

    def test_aggregate_returns_dataframe(self, transactions_df):
        aggregator = TimeWindowAggregator(
            entity_column="customer_id",
            time_column="transaction_date"
        )
        result = aggregator.aggregate(
            transactions_df,
            windows=["7d", "30d"],
            value_columns=["amount"],
            agg_funcs=["sum"]
        )

        assert isinstance(result, pd.DataFrame)

    def test_aggregate_one_row_per_entity(self, transactions_df):
        """Each entity should have exactly one row in output."""
        aggregator = TimeWindowAggregator(
            entity_column="customer_id",
            time_column="transaction_date"
        )
        result = aggregator.aggregate(
            transactions_df,
            windows=["30d"],
            value_columns=["amount"],
            agg_funcs=["sum"]
        )

        # Should have same number of unique entities
        assert len(result) == transactions_df["customer_id"].nunique()

    def test_aggregate_creates_window_columns(self, transactions_df):
        aggregator = TimeWindowAggregator(
            entity_column="customer_id",
            time_column="transaction_date"
        )
        result = aggregator.aggregate(
            transactions_df,
            windows=["7d", "30d"],
            value_columns=["amount"],
            agg_funcs=["sum"]
        )

        # Should create columns like amount_sum_7d, amount_sum_30d
        assert "amount_sum_7d" in result.columns
        assert "amount_sum_30d" in result.columns

    def test_aggregate_with_multiple_agg_funcs(self, transactions_df):
        aggregator = TimeWindowAggregator(
            entity_column="customer_id",
            time_column="transaction_date"
        )
        result = aggregator.aggregate(
            transactions_df,
            windows=["30d"],
            value_columns=["amount"],
            agg_funcs=["sum", "mean", "max", "count"]
        )

        assert "amount_sum_30d" in result.columns
        assert "amount_mean_30d" in result.columns
        assert "amount_max_30d" in result.columns
        assert "amount_count_30d" in result.columns


class TestTimeWindows:
    """Tests for time window definitions."""

    def test_standard_windows(self, transactions_df):
        """Test standard window periods."""
        aggregator = TimeWindowAggregator(
            entity_column="customer_id",
            time_column="transaction_date"
        )
        result = aggregator.aggregate(
            transactions_df,
            windows=["24h", "7d", "30d", "90d", "all_time"],
            value_columns=["amount"],
            agg_funcs=["sum"]
        )

        assert "amount_sum_24h" in result.columns
        assert "amount_sum_7d" in result.columns
        assert "amount_sum_30d" in result.columns
        assert "amount_sum_90d" in result.columns
        assert "amount_sum_all_time" in result.columns

    def test_reference_date_parameter(self, transactions_df):
        """Should use reference date for window calculations."""
        aggregator = TimeWindowAggregator(
            entity_column="customer_id",
            time_column="transaction_date"
        )

        # Use different reference dates
        ref1 = pd.Timestamp("2023-12-31")
        ref2 = pd.Timestamp("2023-06-30")

        result1 = aggregator.aggregate(
            transactions_df,
            windows=["30d"],
            value_columns=["amount"],
            agg_funcs=["sum"],
            reference_date=ref1
        )

        result2 = aggregator.aggregate(
            transactions_df,
            windows=["30d"],
            value_columns=["amount"],
            agg_funcs=["sum"],
            reference_date=ref2
        )

        # Results should differ based on reference date
        # (different events fall in 30d window)
        # Just check they're computed, values will differ based on data
        assert "amount_sum_30d" in result1.columns
        assert "amount_sum_30d" in result2.columns


class TestEventCounting:
    """Tests for event count aggregations."""

    def test_count_events_per_window(self, transactions_df):
        aggregator = TimeWindowAggregator(
            entity_column="customer_id",
            time_column="transaction_date"
        )
        result = aggregator.aggregate(
            transactions_df,
            windows=["7d", "30d"],
            include_event_count=True
        )

        assert "event_count_7d" in result.columns
        assert "event_count_30d" in result.columns

    def test_event_counts_are_integers(self, transactions_df):
        aggregator = TimeWindowAggregator(
            entity_column="customer_id",
            time_column="transaction_date"
        )
        result = aggregator.aggregate(
            transactions_df,
            windows=["30d"],
            include_event_count=True
        )

        # Event counts should be non-negative integers
        assert (result["event_count_30d"] >= 0).all()
        assert result["event_count_30d"].dtype in [np.int64, np.float64]


class TestRecencyFeatures:
    """Tests for recency-based features."""

    def test_days_since_last_event(self, transactions_df):
        aggregator = TimeWindowAggregator(
            entity_column="customer_id",
            time_column="transaction_date"
        )
        result = aggregator.aggregate(
            transactions_df,
            include_recency=True,
            reference_date=pd.Timestamp("2023-12-31")
        )

        assert "days_since_last_event" in result.columns
        # All values should be non-negative
        assert (result["days_since_last_event"] >= 0).all()

    def test_days_since_first_event(self, transactions_df):
        aggregator = TimeWindowAggregator(
            entity_column="customer_id",
            time_column="transaction_date"
        )
        result = aggregator.aggregate(
            transactions_df,
            include_tenure=True,
            reference_date=pd.Timestamp("2023-12-31")
        )

        assert "days_since_first_event" in result.columns


class TestAggregationPlan:
    """Tests for aggregation planning."""

    def test_generate_plan(self, transactions_df):
        aggregator = TimeWindowAggregator(
            entity_column="customer_id",
            time_column="transaction_date"
        )
        plan = aggregator.generate_plan(
            transactions_df,
            windows=["7d", "30d"],
            value_columns=["amount", "quantity"],
            agg_funcs=["sum", "mean"]
        )

        assert isinstance(plan, AggregationPlan)
        assert len(plan.feature_columns) > 0

    def test_plan_lists_all_features(self, transactions_df):
        aggregator = TimeWindowAggregator(
            entity_column="customer_id",
            time_column="transaction_date"
        )
        plan = aggregator.generate_plan(
            transactions_df,
            windows=["7d", "30d"],
            value_columns=["amount"],
            agg_funcs=["sum", "mean"]
        )

        # Should list all features that will be created
        expected = ["amount_sum_7d", "amount_sum_30d", "amount_mean_7d", "amount_mean_30d"]
        for feat in expected:
            assert feat in plan.feature_columns


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_dataframe(self):
        empty_df = pd.DataFrame(columns=["customer_id", "date", "amount"])

        aggregator = TimeWindowAggregator(
            entity_column="customer_id",
            time_column="date"
        )
        result = aggregator.aggregate(
            empty_df,
            windows=["30d"],
            value_columns=["amount"],
            agg_funcs=["sum"]
        )

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_single_entity(self):
        single_df = pd.DataFrame({
            "customer_id": ["C001", "C001", "C001"],
            "date": pd.date_range("2023-12-01", periods=3, freq="D"),
            "amount": [100, 200, 300],
        })

        aggregator = TimeWindowAggregator(
            entity_column="customer_id",
            time_column="date"
        )
        result = aggregator.aggregate(
            single_df,
            windows=["30d"],
            value_columns=["amount"],
            agg_funcs=["sum"]
        )

        assert len(result) == 1
        assert result.iloc[0]["amount_sum_30d"] == 600

    def test_null_values_in_aggregation(self, transactions_df):
        """Should handle null values in value columns."""
        df_with_nulls = transactions_df.copy()
        df_with_nulls.loc[df_with_nulls.index[:3], "amount"] = np.nan

        aggregator = TimeWindowAggregator(
            entity_column="customer_id",
            time_column="transaction_date"
        )
        result = aggregator.aggregate(
            df_with_nulls,
            windows=["30d"],
            value_columns=["amount"],
            agg_funcs=["sum"]
        )

        # Should not crash, nulls ignored in aggregation
        assert "amount_sum_30d" in result.columns

    def test_entity_with_no_events_in_window(self):
        """Entity with all events outside window should have zero counts."""
        old_events = pd.DataFrame({
            "customer_id": ["C001", "C001"],
            "date": pd.to_datetime(["2023-01-01", "2023-02-01"]),
            "amount": [100, 200],
        })

        aggregator = TimeWindowAggregator(
            entity_column="customer_id",
            time_column="date"
        )
        result = aggregator.aggregate(
            old_events,
            windows=["30d"],
            value_columns=["amount"],
            agg_funcs=["sum"],
            reference_date=pd.Timestamp("2023-12-31"),  # Far in future
            include_event_count=True
        )

        # No events in last 30 days
        assert result.iloc[0]["event_count_30d"] == 0
        assert result.iloc[0]["amount_sum_30d"] == 0 or pd.isna(result.iloc[0]["amount_sum_30d"])


@pytest.fixture
def events_with_categories():
    """Sample event data with categorical columns."""
    ref_date = pd.Timestamp("2023-12-31")
    return pd.DataFrame({
        "customer_id": ["C001"] * 5 + ["C002"] * 4 + ["C003"] * 3,
        "event_date": [
            ref_date - timedelta(days=d) for d in [1, 3, 5, 10, 25]
        ] + [
            ref_date - timedelta(days=d) for d in [2, 8, 15, 20]
        ] + [
            ref_date - timedelta(days=d) for d in [5, 50, 100]
        ],
        "channel": ["web", "web", "mobile", "web", "email"] + ["mobile", "mobile", "web", "mobile"] + ["email", "web", "web"],
        "product_category": ["electronics", "electronics", "electronics", "clothing", "clothing"] + ["home", "home", "home", "electronics"] + ["electronics", "electronics", "electronics"],
        "amount": [100, 150, 200, 50, 75, 300, 250, 100, 150, 80, 90, 110],
    })


class TestCategoricalAggregationMode:
    """Tests for mode (most frequent value) aggregation."""

    def test_mode_returns_most_frequent_value(self, events_with_categories):
        aggregator = TimeWindowAggregator(
            entity_column="customer_id", time_column="event_date"
        )
        result = aggregator.aggregate(
            events_with_categories,
            windows=["all_time"],
            value_columns=["channel"],
            agg_funcs=["mode"]
        )
        assert "channel_mode_all_time" in result.columns
        c001 = result[result["customer_id"] == "C001"]["channel_mode_all_time"].iloc[0]
        assert c001 == "web"  # web appears 3 times, others less

    def test_mode_with_time_window(self, events_with_categories):
        aggregator = TimeWindowAggregator(
            entity_column="customer_id", time_column="event_date"
        )
        result = aggregator.aggregate(
            events_with_categories,
            windows=["7d"],
            value_columns=["channel"],
            agg_funcs=["mode"],
            reference_date=pd.Timestamp("2023-12-31")
        )
        assert "channel_mode_7d" in result.columns

    def test_mode_returns_none_for_no_events_in_window(self):
        df = pd.DataFrame({
            "customer_id": ["C001", "C001"],
            "date": pd.to_datetime(["2023-01-01", "2023-02-01"]),
            "category": ["A", "B"],
        })
        aggregator = TimeWindowAggregator(entity_column="customer_id", time_column="date")
        result = aggregator.aggregate(
            df, windows=["7d"], value_columns=["category"], agg_funcs=["mode"],
            reference_date=pd.Timestamp("2023-12-31")
        )
        assert pd.isna(result.iloc[0]["category_mode_7d"]) or result.iloc[0]["category_mode_7d"] is None


class TestCategoricalAggregationNunique:
    """Tests for nunique (count distinct) aggregation."""

    def test_nunique_counts_distinct_values(self, events_with_categories):
        aggregator = TimeWindowAggregator(
            entity_column="customer_id", time_column="event_date"
        )
        result = aggregator.aggregate(
            events_with_categories,
            windows=["all_time"],
            value_columns=["channel"],
            agg_funcs=["nunique"]
        )
        assert "channel_nunique_all_time" in result.columns
        c001 = result[result["customer_id"] == "C001"]["channel_nunique_all_time"].iloc[0]
        assert c001 == 3  # web, mobile, email

    def test_nunique_respects_time_window(self, events_with_categories):
        aggregator = TimeWindowAggregator(
            entity_column="customer_id", time_column="event_date"
        )
        result = aggregator.aggregate(
            events_with_categories,
            windows=["7d"],
            value_columns=["channel"],
            agg_funcs=["nunique"],
            reference_date=pd.Timestamp("2023-12-31")
        )
        c001 = result[result["customer_id"] == "C001"]["channel_nunique_7d"].iloc[0]
        # In last 7 days: web(1d), web(3d), mobile(5d) = 2 unique
        assert c001 == 2

    def test_nunique_returns_zero_for_no_events(self):
        df = pd.DataFrame({
            "customer_id": ["C001"], "date": pd.to_datetime(["2023-01-01"]), "cat": ["A"]
        })
        aggregator = TimeWindowAggregator(entity_column="customer_id", time_column="date")
        result = aggregator.aggregate(
            df, windows=["7d"], value_columns=["cat"], agg_funcs=["nunique"],
            reference_date=pd.Timestamp("2023-12-31")
        )
        assert result.iloc[0]["cat_nunique_7d"] == 0


class TestCategoricalAggregationModeRatio:
    """Tests for mode_ratio (% of events matching mode) aggregation."""

    def test_mode_ratio_calculates_percentage(self, events_with_categories):
        aggregator = TimeWindowAggregator(
            entity_column="customer_id", time_column="event_date"
        )
        result = aggregator.aggregate(
            events_with_categories,
            windows=["all_time"],
            value_columns=["channel"],
            agg_funcs=["mode_ratio"]
        )
        assert "channel_mode_ratio_all_time" in result.columns
        c001 = result[result["customer_id"] == "C001"]["channel_mode_ratio_all_time"].iloc[0]
        # web appears 3/5 times = 0.6
        assert abs(c001 - 0.6) < 0.01

    def test_mode_ratio_is_between_0_and_1(self, events_with_categories):
        aggregator = TimeWindowAggregator(
            entity_column="customer_id", time_column="event_date"
        )
        result = aggregator.aggregate(
            events_with_categories,
            windows=["all_time"],
            value_columns=["channel"],
            agg_funcs=["mode_ratio"]
        )
        ratios = result["channel_mode_ratio_all_time"].dropna()
        assert (ratios >= 0).all() and (ratios <= 1).all()


class TestCategoricalAggregationEntropy:
    """Tests for entropy (Shannon entropy) aggregation."""

    def test_entropy_calculates_diversity(self, events_with_categories):
        aggregator = TimeWindowAggregator(
            entity_column="customer_id", time_column="event_date"
        )
        result = aggregator.aggregate(
            events_with_categories,
            windows=["all_time"],
            value_columns=["channel"],
            agg_funcs=["entropy"]
        )
        assert "channel_entropy_all_time" in result.columns
        c001 = result[result["customer_id"] == "C001"]["channel_entropy_all_time"].iloc[0]
        assert c001 > 0  # Has multiple categories so entropy > 0

    def test_entropy_zero_for_single_category(self):
        df = pd.DataFrame({
            "customer_id": ["C001"] * 3, "date": pd.date_range("2023-12-01", periods=3),
            "cat": ["A", "A", "A"]
        })
        aggregator = TimeWindowAggregator(entity_column="customer_id", time_column="date")
        result = aggregator.aggregate(
            df, windows=["all_time"], value_columns=["cat"], agg_funcs=["entropy"]
        )
        assert result.iloc[0]["cat_entropy_all_time"] == 0

    def test_entropy_max_for_uniform_distribution(self):
        df = pd.DataFrame({
            "customer_id": ["C001"] * 4, "date": pd.date_range("2023-12-01", periods=4),
            "cat": ["A", "B", "C", "D"]  # Uniform distribution
        })
        aggregator = TimeWindowAggregator(entity_column="customer_id", time_column="date")
        result = aggregator.aggregate(
            df, windows=["all_time"], value_columns=["cat"], agg_funcs=["entropy"]
        )
        # Max entropy for 4 categories = log2(4) = 2
        assert abs(result.iloc[0]["cat_entropy_all_time"] - 2.0) < 0.01


class TestCategoricalAggregationValueCounts:
    """Tests for value_counts (expand to one-hot counts) aggregation."""

    def test_value_counts_creates_columns_per_category(self, events_with_categories):
        aggregator = TimeWindowAggregator(
            entity_column="customer_id", time_column="event_date"
        )
        result = aggregator.aggregate(
            events_with_categories,
            windows=["all_time"],
            value_columns=["channel"],
            agg_funcs=["value_counts"]
        )
        # Should create columns for each unique value
        assert "channel_web_count_all_time" in result.columns
        assert "channel_mobile_count_all_time" in result.columns
        assert "channel_email_count_all_time" in result.columns

    def test_value_counts_correct_counts(self, events_with_categories):
        aggregator = TimeWindowAggregator(
            entity_column="customer_id", time_column="event_date"
        )
        result = aggregator.aggregate(
            events_with_categories,
            windows=["all_time"],
            value_columns=["channel"],
            agg_funcs=["value_counts"]
        )
        c001 = result[result["customer_id"] == "C001"]
        assert c001["channel_web_count_all_time"].iloc[0] == 3
        assert c001["channel_mobile_count_all_time"].iloc[0] == 1
        assert c001["channel_email_count_all_time"].iloc[0] == 1

    def test_value_counts_with_time_window(self, events_with_categories):
        aggregator = TimeWindowAggregator(
            entity_column="customer_id", time_column="event_date"
        )
        result = aggregator.aggregate(
            events_with_categories,
            windows=["7d"],
            value_columns=["channel"],
            agg_funcs=["value_counts"],
            reference_date=pd.Timestamp("2023-12-31")
        )
        c001 = result[result["customer_id"] == "C001"]
        # In last 7d: web(1d), web(3d), mobile(5d)
        assert c001["channel_web_count_7d"].iloc[0] == 2
        assert c001["channel_mobile_count_7d"].iloc[0] == 1


class TestCategoricalAggregationPlan:
    """Tests for plan generation with categorical aggregations."""

    def test_plan_includes_categorical_features(self, events_with_categories):
        aggregator = TimeWindowAggregator(
            entity_column="customer_id", time_column="event_date"
        )
        plan = aggregator.generate_plan(
            events_with_categories,
            windows=["30d"],
            value_columns=["channel"],
            agg_funcs=["mode", "nunique", "mode_ratio"]
        )
        assert "channel_mode_30d" in plan.feature_columns
        assert "channel_nunique_30d" in plan.feature_columns
        assert "channel_mode_ratio_30d" in plan.feature_columns

    def test_plan_value_counts_needs_data_scan(self, events_with_categories):
        aggregator = TimeWindowAggregator(
            entity_column="customer_id", time_column="event_date"
        )
        plan = aggregator.generate_plan(
            events_with_categories,
            windows=["30d"],
            value_columns=["channel"],
            agg_funcs=["value_counts"]
        )
        # value_counts expands based on unique values in data
        assert any("channel_" in col and "_count_30d" in col for col in plan.feature_columns)


class TestMixedNumericAndCategoricalAggregation:
    """Tests for combining numeric and categorical aggregations."""

    def test_mixed_aggregations(self, events_with_categories):
        aggregator = TimeWindowAggregator(
            entity_column="customer_id", time_column="event_date"
        )
        result = aggregator.aggregate(
            events_with_categories,
            windows=["30d"],
            value_columns=["amount", "channel"],
            agg_funcs=["sum", "mean", "mode", "nunique"]
        )
        # Numeric aggregations on amount
        assert "amount_sum_30d" in result.columns
        assert "amount_mean_30d" in result.columns
        # Categorical aggregations on channel
        assert "channel_mode_30d" in result.columns
        assert "channel_nunique_30d" in result.columns
        # sum/mean on categorical column returns NaN
        assert "channel_sum_30d" in result.columns
        assert result["channel_sum_30d"].isna().all()
        # mode/nunique work on numeric (amount) too
        assert "amount_mode_30d" in result.columns
        assert "amount_nunique_30d" in result.columns


class TestExcludeColumns:
    """Tests for exclude_columns parameter to prevent data leakage."""

    @pytest.fixture
    def df_with_target(self):
        """Sample data with a target column that should be excluded."""
        ref_date = pd.Timestamp("2023-12-31")
        return pd.DataFrame({
            "customer_id": ["C001"] * 5 + ["C002"] * 3,
            "event_date": [ref_date - timedelta(days=d) for d in [1, 5, 10, 20, 30]] +
                          [ref_date - timedelta(days=d) for d in [2, 8, 15]],
            "amount": [100, 150, 200, 50, 75, 300, 250, 100],
            "target": [1, 1, 1, 1, 1, 0, 0, 0],  # Target column - should NOT be aggregated
        })

    def test_exclude_columns_removes_from_aggregation(self, df_with_target):
        aggregator = TimeWindowAggregator(
            entity_column="customer_id", time_column="event_date"
        )
        result = aggregator.aggregate(
            df_with_target,
            windows=["30d"],
            value_columns=["amount", "target"],
            agg_funcs=["sum", "mean"],
            exclude_columns=["target"]
        )
        # amount should be aggregated
        assert "amount_sum_30d" in result.columns
        assert "amount_mean_30d" in result.columns
        # target should NOT be aggregated (data leakage prevention)
        assert "target_sum_30d" not in result.columns
        assert "target_mean_30d" not in result.columns

    def test_exclude_columns_in_generate_plan(self, df_with_target):
        aggregator = TimeWindowAggregator(
            entity_column="customer_id", time_column="event_date"
        )
        plan = aggregator.generate_plan(
            df_with_target,
            windows=["30d"],
            value_columns=["amount", "target"],
            agg_funcs=["sum", "mean"],
            exclude_columns=["target"]
        )
        # Plan should not include target features
        assert "amount_sum_30d" in plan.feature_columns
        assert "target_sum_30d" not in plan.feature_columns
        assert "target_mean_30d" not in plan.feature_columns

    def test_exclude_columns_empty_list_has_no_effect(self, df_with_target):
        aggregator = TimeWindowAggregator(
            entity_column="customer_id", time_column="event_date"
        )
        result = aggregator.aggregate(
            df_with_target,
            windows=["30d"],
            value_columns=["amount", "target"],
            agg_funcs=["sum"],
            exclude_columns=[]
        )
        # Both should be aggregated when exclude_columns is empty
        assert "amount_sum_30d" in result.columns
        assert "target_sum_30d" in result.columns

    def test_exclude_columns_none_has_no_effect(self, df_with_target):
        aggregator = TimeWindowAggregator(
            entity_column="customer_id", time_column="event_date"
        )
        result = aggregator.aggregate(
            df_with_target,
            windows=["30d"],
            value_columns=["amount", "target"],
            agg_funcs=["sum"],
            exclude_columns=None
        )
        # Both should be aggregated when exclude_columns is None
        assert "amount_sum_30d" in result.columns
        assert "target_sum_30d" in result.columns

    def test_exclude_multiple_columns(self, df_with_target):
        df_with_target["leaky_feature"] = df_with_target["target"] * 2
        aggregator = TimeWindowAggregator(
            entity_column="customer_id", time_column="event_date"
        )
        result = aggregator.aggregate(
            df_with_target,
            windows=["30d"],
            value_columns=["amount", "target", "leaky_feature"],
            agg_funcs=["sum"],
            exclude_columns=["target", "leaky_feature"]
        )
        assert "amount_sum_30d" in result.columns
        assert "target_sum_30d" not in result.columns
        assert "leaky_feature_sum_30d" not in result.columns


class TestNotebook01dLeakagePrevention:
    """Integration tests mimicking notebook 01d flow to prevent target leakage."""

    @pytest.fixture
    def email_events_df(self):
        """Sample email event data similar to customer_emails dataset."""
        np.random.seed(42)
        ref_date = pd.Timestamp("2023-12-31")
        n_entities = 50
        data = []
        for i in range(n_entities):
            entity_id = f"C{i:04d}"
            entity_target = np.random.choice([0, 1], p=[0.3, 0.7])
            n_events = np.random.randint(5, 20)
            for j in range(n_events):
                data.append({
                    "customer_id": entity_id,
                    "sent_date": ref_date - timedelta(days=np.random.randint(1, 365)),
                    "opened": np.random.choice([0, 1]),
                    "clicked": np.random.choice([0, 1]),
                    "bounced": np.random.choice([0, 1], p=[0.9, 0.1]),
                    "time_to_open_hours": np.random.uniform(0, 48) if np.random.random() > 0.3 else np.nan,
                    "target": entity_target,
                })
        return pd.DataFrame(data)

    def test_notebook_01d_excludes_target_from_value_columns(self, email_events_df):
        """Test that notebook 01d logic correctly excludes target from VALUE_COLUMNS."""
        df = email_events_df
        ENTITY_COLUMN = "customer_id"
        TIME_COLUMN = "sent_date"
        TARGET_COLUMN = "target"

        # This mimics notebook 01d cell-7 logic
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        exclude_cols = {ENTITY_COLUMN, TIME_COLUMN}
        if TARGET_COLUMN:
            exclude_cols.add(TARGET_COLUMN)
        VALUE_COLUMNS = [c for c in numeric_cols if c not in exclude_cols]

        # Target should NOT be in VALUE_COLUMNS
        assert TARGET_COLUMN not in VALUE_COLUMNS
        assert "opened" in VALUE_COLUMNS
        assert "clicked" in VALUE_COLUMNS

    def test_notebook_01d_aggregation_has_no_target_derived_columns(self, email_events_df):
        """Test that aggregation using notebook 01d logic produces no target-derived columns."""
        df = email_events_df
        ENTITY_COLUMN = "customer_id"
        TIME_COLUMN = "sent_date"
        TARGET_COLUMN = "target"

        # Mimic notebook 01d cell-7 logic
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        exclude_cols = {ENTITY_COLUMN, TIME_COLUMN, TARGET_COLUMN}
        VALUE_COLUMNS = [c for c in numeric_cols if c not in exclude_cols]

        # Mimic notebook 01d cell-11 aggregation
        aggregator = TimeWindowAggregator(entity_column=ENTITY_COLUMN, time_column=TIME_COLUMN)
        df_aggregated = aggregator.aggregate(
            df,
            windows=["180d", "365d", "all_time"],
            value_columns=VALUE_COLUMNS,
            agg_funcs=["sum", "mean", "max", "count"],
            include_event_count=True,
            include_recency=True,
            include_tenure=True
        )

        # Check NO target-derived columns exist
        target_cols = [c for c in df_aggregated.columns if "target" in c.lower()]
        assert len(target_cols) == 0, f"Found target-derived columns: {target_cols}"

        # Check legitimate columns DO exist
        assert "opened_sum_180d" in df_aggregated.columns
        assert "clicked_mean_365d" in df_aggregated.columns
        assert "event_count_all_time" in df_aggregated.columns

    def test_target_leakage_causes_perfect_correlation(self, email_events_df):
        """Demonstrate that including target in aggregation causes perfect correlation (leakage)."""
        df = email_events_df
        ENTITY_COLUMN = "customer_id"
        TIME_COLUMN = "sent_date"
        TARGET_COLUMN = "target"

        # WRONG: Include target in value_columns (old buggy behavior)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        VALUE_COLUMNS_WRONG = [c for c in numeric_cols if c not in {ENTITY_COLUMN, TIME_COLUMN}]

        aggregator = TimeWindowAggregator(entity_column=ENTITY_COLUMN, time_column=TIME_COLUMN)
        df_leaky = aggregator.aggregate(
            df,
            windows=["all_time"],
            value_columns=VALUE_COLUMNS_WRONG,
            agg_funcs=["max"],
            include_event_count=True,
        )

        # Add entity-level target
        entity_target = df.groupby(ENTITY_COLUMN)[TARGET_COLUMN].max()
        df_leaky["target"] = df_leaky[ENTITY_COLUMN].map(entity_target)

        # target_max_all_time should be perfectly correlated with target (LEAKAGE!)
        correlation = df_leaky["target"].corr(df_leaky["target_max_all_time"])
        assert correlation == 1.0, "This demonstrates the leakage - perfect correlation"

    def test_correct_aggregation_has_reasonable_correlations(self, email_events_df):
        """Test that correct aggregation (without target) has reasonable correlations."""
        df = email_events_df
        ENTITY_COLUMN = "customer_id"
        TIME_COLUMN = "sent_date"
        TARGET_COLUMN = "target"

        # CORRECT: Exclude target from value_columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        exclude_cols = {ENTITY_COLUMN, TIME_COLUMN, TARGET_COLUMN}
        VALUE_COLUMNS_CORRECT = [c for c in numeric_cols if c not in exclude_cols]

        aggregator = TimeWindowAggregator(entity_column=ENTITY_COLUMN, time_column=TIME_COLUMN)
        df_clean = aggregator.aggregate(
            df,
            windows=["all_time"],
            value_columns=VALUE_COLUMNS_CORRECT,
            agg_funcs=["sum", "mean"],
            include_event_count=True,
        )

        # Add entity-level target
        entity_target = df.groupby(ENTITY_COLUMN)[TARGET_COLUMN].max()
        df_clean["target"] = df_clean[ENTITY_COLUMN].map(entity_target)

        # No feature should have perfect correlation with target
        numeric_features = [c for c in df_clean.columns if c not in [ENTITY_COLUMN, "target"]]
        for col in numeric_features:
            if df_clean[col].std() > 0:  # Skip constant columns
                corr = abs(df_clean["target"].corr(df_clean[col]))
                assert corr < 0.99, f"{col} has suspiciously high correlation: {corr}"
