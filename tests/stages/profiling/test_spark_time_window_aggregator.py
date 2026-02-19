from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from customer_retention.stages.profiling.spark_time_window_aggregator import (
    SparkTimeWindowAggregator,
)
from customer_retention.stages.profiling.time_window_aggregator import (
    TimeWindowAggregator,
)


@pytest.fixture
def transactions_df():
    np.random.seed(42)
    ref_date = pd.Timestamp("2023-12-31")
    data = []
    for cust_id in ["C001", "C002", "C003", "C004", "C005"]:
        if cust_id == "C001":
            dates = [ref_date - timedelta(days=d) for d in [1, 3, 10, 25, 45]]
        elif cust_id == "C002":
            dates = [ref_date - timedelta(days=d) for d in [5, 35, 60]]
        elif cust_id == "C003":
            dates = [ref_date - timedelta(days=d) for d in [100, 150, 200]]
        else:
            dates = [ref_date - timedelta(days=d) for d in range(10, 60, 15)]
        for date in dates:
            data.append({
                "customer_id": cust_id,
                "transaction_date": date,
                "amount": np.random.uniform(20, 200),
                "quantity": np.random.randint(1, 5),
            })
    return pd.DataFrame(data)


@pytest.fixture
def events_with_categories():
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
        "channel": (
            ["web", "web", "mobile", "web", "email"]
            + ["mobile", "mobile", "web", "mobile"]
            + ["email", "web", "web"]
        ),
        "product_category": (
            ["electronics", "electronics", "electronics", "clothing", "clothing"]
            + ["home", "home", "home", "electronics"]
            + ["electronics", "electronics", "electronics"]
        ),
        "amount": [100, 150, 200, 50, 75, 300, 250, 100, 150, 80, 90, 110],
    })


BOTH_AGGREGATORS = [TimeWindowAggregator, SparkTimeWindowAggregator]


def _sort_result(df, entity_col="customer_id"):
    return df.sort_values(entity_col).reset_index(drop=True)


class TestSparkAggregatorMatchesLocal:

    @pytest.mark.parametrize("cls", BOTH_AGGREGATORS)
    def test_aggregate_returns_one_row_per_entity(self, transactions_df, cls):
        agg = cls(entity_column="customer_id", time_column="transaction_date")
        result = agg.aggregate(
            transactions_df, windows=["30d"],
            value_columns=["amount"], agg_funcs=["sum"],
        )
        assert len(result) == transactions_df["customer_id"].nunique()

    @pytest.mark.parametrize("cls", BOTH_AGGREGATORS)
    def test_aggregate_creates_window_columns(self, transactions_df, cls):
        agg = cls(entity_column="customer_id", time_column="transaction_date")
        result = agg.aggregate(
            transactions_df, windows=["7d", "30d"],
            value_columns=["amount"], agg_funcs=["sum"],
        )
        assert "amount_sum_7d" in result.columns
        assert "amount_sum_30d" in result.columns

    @pytest.mark.parametrize("cls", BOTH_AGGREGATORS)
    def test_multiple_agg_funcs(self, transactions_df, cls):
        agg = cls(entity_column="customer_id", time_column="transaction_date")
        result = agg.aggregate(
            transactions_df, windows=["30d"],
            value_columns=["amount"],
            agg_funcs=["sum", "mean", "max", "min", "count"],
        )
        for func in ["sum", "mean", "max", "min", "count"]:
            assert f"amount_{func}_30d" in result.columns

    @pytest.mark.parametrize("cls", BOTH_AGGREGATORS)
    def test_event_count(self, transactions_df, cls):
        agg = cls(entity_column="customer_id", time_column="transaction_date")
        result = agg.aggregate(
            transactions_df, windows=["7d", "30d"], include_event_count=True,
        )
        assert "event_count_7d" in result.columns
        assert "event_count_30d" in result.columns
        assert (result["event_count_30d"] >= 0).all()

    @pytest.mark.parametrize("cls", BOTH_AGGREGATORS)
    def test_recency_and_tenure(self, transactions_df, cls):
        ref = pd.Timestamp("2023-12-31")
        agg = cls(entity_column="customer_id", time_column="transaction_date")
        result = agg.aggregate(
            transactions_df, include_recency=True, include_tenure=True,
            reference_date=ref,
        )
        assert "days_since_last_event" in result.columns
        assert "days_since_first_event" in result.columns
        assert (result["days_since_last_event"] >= 0).all()
        assert (result["days_since_first_event"] >= 0).all()

    @pytest.mark.parametrize("cls", BOTH_AGGREGATORS)
    def test_exclude_columns(self, cls):
        ref = pd.Timestamp("2023-12-31")
        df = pd.DataFrame({
            "customer_id": ["C001"] * 5 + ["C002"] * 3,
            "event_date": [ref - timedelta(days=d) for d in [1, 5, 10, 20, 30]]
            + [ref - timedelta(days=d) for d in [2, 8, 15]],
            "amount": [100, 150, 200, 50, 75, 300, 250, 100],
            "target": [1, 1, 1, 1, 1, 0, 0, 0],
        })
        agg = cls(entity_column="customer_id", time_column="event_date")
        result = agg.aggregate(
            df, windows=["30d"], value_columns=["amount", "target"],
            agg_funcs=["sum", "mean"], exclude_columns=["target"],
        )
        assert "amount_sum_30d" in result.columns
        assert "target_sum_30d" not in result.columns

    @pytest.mark.parametrize("cls", BOTH_AGGREGATORS)
    def test_all_time_window(self, transactions_df, cls):
        agg = cls(entity_column="customer_id", time_column="transaction_date")
        result = agg.aggregate(
            transactions_df, windows=["all_time"],
            value_columns=["amount"], agg_funcs=["sum"],
        )
        assert "amount_sum_all_time" in result.columns

    def test_numeric_values_match(self, transactions_df):
        ref = pd.Timestamp("2023-12-31")
        kwargs = dict(
            windows=["7d", "30d", "90d", "all_time"],
            value_columns=["amount", "quantity"],
            agg_funcs=["sum", "mean", "max", "min", "count"],
            reference_date=ref, include_event_count=True,
            include_recency=True, include_tenure=True,
        )
        local = TimeWindowAggregator("customer_id", "transaction_date")
        spark = SparkTimeWindowAggregator("customer_id", "transaction_date")
        r_local = _sort_result(local.aggregate(transactions_df, **kwargs))
        r_spark = _sort_result(spark.aggregate(transactions_df, **kwargs))

        assert set(r_local.columns) == set(r_spark.columns)
        for col in r_local.columns:
            if col == "customer_id":
                continue
            np.testing.assert_allclose(
                r_local[col].to_numpy().astype(float),
                r_spark[col].to_numpy().astype(float),
                rtol=1e-6, atol=1e-10,
                err_msg=f"Column {col} mismatch",
            )


class TestSparkCategoricalAggregation:

    @pytest.mark.parametrize("cls", BOTH_AGGREGATORS)
    def test_mode(self, events_with_categories, cls):
        agg = cls(entity_column="customer_id", time_column="event_date")
        result = agg.aggregate(
            events_with_categories, windows=["all_time"],
            value_columns=["channel"], agg_funcs=["mode"],
        )
        assert "channel_mode_all_time" in result.columns
        c001 = result[result["customer_id"] == "C001"]["channel_mode_all_time"].iloc[0]
        assert c001 == "web"

    @pytest.mark.parametrize("cls", BOTH_AGGREGATORS)
    def test_nunique(self, events_with_categories, cls):
        agg = cls(entity_column="customer_id", time_column="event_date")
        result = agg.aggregate(
            events_with_categories, windows=["all_time"],
            value_columns=["channel"], agg_funcs=["nunique"],
        )
        c001 = result[result["customer_id"] == "C001"]["channel_nunique_all_time"].iloc[0]
        assert c001 == 3

    @pytest.mark.parametrize("cls", BOTH_AGGREGATORS)
    def test_mode_ratio(self, events_with_categories, cls):
        agg = cls(entity_column="customer_id", time_column="event_date")
        result = agg.aggregate(
            events_with_categories, windows=["all_time"],
            value_columns=["channel"], agg_funcs=["mode_ratio"],
        )
        c001 = result[result["customer_id"] == "C001"]["channel_mode_ratio_all_time"].iloc[0]
        assert abs(c001 - 0.6) < 0.01

    @pytest.mark.parametrize("cls", BOTH_AGGREGATORS)
    def test_entropy(self, events_with_categories, cls):
        agg = cls(entity_column="customer_id", time_column="event_date")
        result = agg.aggregate(
            events_with_categories, windows=["all_time"],
            value_columns=["channel"], agg_funcs=["entropy"],
        )
        c001 = result[result["customer_id"] == "C001"]["channel_entropy_all_time"].iloc[0]
        assert c001 > 0

    @pytest.mark.parametrize("cls", BOTH_AGGREGATORS)
    def test_value_counts(self, events_with_categories, cls):
        agg = cls(entity_column="customer_id", time_column="event_date")
        result = agg.aggregate(
            events_with_categories, windows=["all_time"],
            value_columns=["channel"], agg_funcs=["value_counts"],
        )
        c001 = result[result["customer_id"] == "C001"]
        assert c001["channel_web_count_all_time"].iloc[0] == 3
        assert c001["channel_mobile_count_all_time"].iloc[0] == 1
        assert c001["channel_email_count_all_time"].iloc[0] == 1

    def test_categorical_values_match(self, events_with_categories):
        ref = pd.Timestamp("2023-12-31")
        kwargs = dict(
            windows=["7d", "30d", "all_time"],
            value_columns=["channel"],
            agg_funcs=["mode", "nunique", "mode_ratio", "entropy"],
            reference_date=ref,
        )
        local = TimeWindowAggregator("customer_id", "event_date")
        spark = SparkTimeWindowAggregator("customer_id", "event_date")
        r_local = _sort_result(local.aggregate(events_with_categories, **kwargs))
        r_spark = _sort_result(spark.aggregate(events_with_categories, **kwargs))

        assert set(r_local.columns) == set(r_spark.columns)
        for col in r_local.columns:
            if col == "customer_id":
                continue
            if "_mode_" in col and "ratio" not in col:
                assert list(r_local[col]) == list(r_spark[col])
            else:
                np.testing.assert_allclose(
                    r_local[col].to_numpy().astype(float),
                    r_spark[col].to_numpy().astype(float),
                    rtol=1e-6, atol=1e-10,
                    err_msg=f"Column {col} mismatch",
                )


class TestSparkEdgeCases:

    @pytest.mark.parametrize("cls", BOTH_AGGREGATORS)
    def test_empty_dataframe(self, cls):
        empty_df = pd.DataFrame(columns=["customer_id", "date", "amount"])
        agg = cls(entity_column="customer_id", time_column="date")
        result = agg.aggregate(
            empty_df, windows=["30d"], value_columns=["amount"], agg_funcs=["sum"],
        )
        assert len(result) == 0

    @pytest.mark.parametrize("cls", BOTH_AGGREGATORS)
    def test_single_entity(self, cls):
        df = pd.DataFrame({
            "customer_id": ["C001", "C001", "C001"],
            "date": pd.date_range("2023-12-01", periods=3, freq="D"),
            "amount": [100, 200, 300],
        })
        agg = cls(entity_column="customer_id", time_column="date")
        result = agg.aggregate(
            df, windows=["30d"], value_columns=["amount"], agg_funcs=["sum"],
        )
        assert len(result) == 1
        assert result.iloc[0]["amount_sum_30d"] == 600

    @pytest.mark.parametrize("cls", BOTH_AGGREGATORS)
    def test_null_values(self, transactions_df, cls):
        df = transactions_df.copy()
        df.loc[df.index[:3], "amount"] = np.nan
        agg = cls(entity_column="customer_id", time_column="transaction_date")
        result = agg.aggregate(
            df, windows=["30d"], value_columns=["amount"], agg_funcs=["sum"],
        )
        assert "amount_sum_30d" in result.columns

    @pytest.mark.parametrize("cls", BOTH_AGGREGATORS)
    def test_entity_with_no_events_in_window(self, cls):
        df = pd.DataFrame({
            "customer_id": ["C001", "C001"],
            "date": pd.to_datetime(["2023-01-01", "2023-02-01"]),
            "amount": [100, 200],
        })
        agg = cls(entity_column="customer_id", time_column="date")
        result = agg.aggregate(
            df, windows=["30d"], value_columns=["amount"], agg_funcs=["sum"],
            reference_date=pd.Timestamp("2023-12-31"), include_event_count=True,
        )
        assert result.iloc[0]["event_count_30d"] == 0

    @pytest.mark.parametrize("cls", BOTH_AGGREGATORS)
    def test_entropy_zero_for_single_category(self, cls):
        df = pd.DataFrame({
            "customer_id": ["C001"] * 3,
            "date": pd.date_range("2023-12-01", periods=3),
            "cat": ["A", "A", "A"],
        })
        agg = cls(entity_column="customer_id", time_column="date")
        result = agg.aggregate(
            df, windows=["all_time"], value_columns=["cat"], agg_funcs=["entropy"],
        )
        assert result.iloc[0]["cat_entropy_all_time"] == 0

    @pytest.mark.parametrize("cls", BOTH_AGGREGATORS)
    def test_entropy_max_for_uniform_distribution(self, cls):
        df = pd.DataFrame({
            "customer_id": ["C001"] * 4,
            "date": pd.date_range("2023-12-01", periods=4),
            "cat": ["A", "B", "C", "D"],
        })
        agg = cls(entity_column="customer_id", time_column="date")
        result = agg.aggregate(
            df, windows=["all_time"], value_columns=["cat"], agg_funcs=["entropy"],
        )
        assert abs(result.iloc[0]["cat_entropy_all_time"] - 2.0) < 0.01

    @pytest.mark.parametrize("cls", BOTH_AGGREGATORS)
    def test_mode_returns_none_for_no_events_in_window(self, cls):
        df = pd.DataFrame({
            "customer_id": ["C001", "C001"],
            "date": pd.to_datetime(["2023-01-01", "2023-02-01"]),
            "category": ["A", "B"],
        })
        agg = cls(entity_column="customer_id", time_column="date")
        result = agg.aggregate(
            df, windows=["7d"], value_columns=["category"], agg_funcs=["mode"],
            reference_date=pd.Timestamp("2023-12-31"),
        )
        val = result.iloc[0]["category_mode_7d"]
        assert pd.isna(val) or val is None

    @pytest.mark.parametrize("cls", BOTH_AGGREGATORS)
    def test_tz_aware_timestamps(self, cls):
        ref = pd.Timestamp("2023-12-31", tz="UTC")
        df = pd.DataFrame({
            "customer_id": ["C001"] * 3 + ["C002"] * 2,
            "event_date": pd.to_datetime([
                "2023-12-20", "2023-12-25", "2023-12-30",
                "2023-12-10", "2023-12-28",
            ]).tz_localize("UTC"),
            "amount": [100, 150, 200, 300, 250],
        })
        agg = cls(entity_column="customer_id", time_column="event_date")
        result = agg.aggregate(
            df, windows=["30d"], value_columns=["amount"], agg_funcs=["sum"],
            reference_date=ref, include_event_count=True,
        )
        assert len(result) == 2
        assert "amount_sum_30d" in result.columns


class TestSparkMixedAggregation:

    @pytest.mark.parametrize("cls", BOTH_AGGREGATORS)
    def test_mixed_numeric_and_categorical(self, events_with_categories, cls):
        agg = cls(entity_column="customer_id", time_column="event_date")
        result = agg.aggregate(
            events_with_categories, windows=["30d"],
            value_columns=["amount", "channel"],
            agg_funcs=["sum", "mean", "mode", "nunique"],
        )
        assert "amount_sum_30d" in result.columns
        assert "amount_mean_30d" in result.columns
        assert "channel_mode_30d" in result.columns
        assert "channel_nunique_30d" in result.columns
        assert "channel_sum_30d" in result.columns
        assert result["channel_sum_30d"].isna().all()

    def test_mixed_values_match(self, events_with_categories):
        ref = pd.Timestamp("2023-12-31")
        kwargs = dict(
            windows=["30d", "all_time"],
            value_columns=["amount", "channel"],
            agg_funcs=["sum", "mean", "mode", "nunique"],
            reference_date=ref, include_event_count=True,
        )
        local = TimeWindowAggregator("customer_id", "event_date")
        spark = SparkTimeWindowAggregator("customer_id", "event_date")
        r_local = _sort_result(local.aggregate(events_with_categories, **kwargs))
        r_spark = _sort_result(spark.aggregate(events_with_categories, **kwargs))

        assert set(r_local.columns) == set(r_spark.columns)


class TestSparkGeneratePlan:

    @pytest.mark.parametrize("cls", BOTH_AGGREGATORS)
    def test_generate_plan(self, transactions_df, cls):
        agg = cls(entity_column="customer_id", time_column="transaction_date")
        plan = agg.generate_plan(
            transactions_df, windows=["7d", "30d"],
            value_columns=["amount", "quantity"], agg_funcs=["sum", "mean"],
        )
        assert len(plan.feature_columns) > 0
        assert "amount_sum_7d" in plan.feature_columns

    @pytest.mark.parametrize("cls", BOTH_AGGREGATORS)
    def test_plan_with_exclude_columns(self, cls):
        df = pd.DataFrame({
            "customer_id": ["C001"] * 3,
            "date": pd.date_range("2023-12-01", periods=3),
            "amount": [100, 200, 300],
            "target": [1, 1, 0],
        })
        agg = cls(entity_column="customer_id", time_column="date")
        plan = agg.generate_plan(
            df, windows=["30d"], value_columns=["amount", "target"],
            agg_funcs=["sum"], exclude_columns=["target"],
        )
        assert "amount_sum_30d" in plan.feature_columns
        assert "target_sum_30d" not in plan.feature_columns


class TestSparkAttrsMetadata:

    @pytest.mark.parametrize("cls", BOTH_AGGREGATORS)
    def test_attrs_stored_as_strings(self, transactions_df, cls):
        agg = cls(entity_column="customer_id", time_column="transaction_date")
        result = agg.aggregate(
            transactions_df, windows=["30d"],
            value_columns=["amount"], agg_funcs=["sum"],
            reference_date=pd.Timestamp("2023-12-31"),
        )
        assert isinstance(result.attrs.get("aggregation_reference_date"), str)
        assert isinstance(result.attrs.get("aggregation_timestamp"), str)
