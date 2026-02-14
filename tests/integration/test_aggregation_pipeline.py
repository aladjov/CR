import numpy as np
import pandas as pd
import pytest

from customer_retention.stages.profiling import TimeWindowAggregator

WINDOWS = ["7d", "14d", "30d", "90d", "all_time"]


@pytest.fixture
def controlled_temporal_distribution():
    ref = pd.Timestamp("2024-06-01")
    rng = np.random.default_rng(42)
    rows = []
    entity_id = 0

    for days_lo, days_hi, n in [(0, 6, 30), (8, 13, 20), (15, 29, 20), (31, 89, 30)]:
        for _ in range(n):
            entity = f"E{entity_id:04d}"
            offset = rng.integers(days_lo, days_hi + 1)
            event_date = ref - pd.Timedelta(days=int(offset))
            amount = float(rng.integers(10, 500))
            rows.append({"entity_id": entity, "event_time": event_date, "amount": amount})
            entity_id += 1

    df = pd.DataFrame(rows)
    return df, ref


@pytest.fixture
def two_entity_sparse_dataset():
    ref = pd.Timestamp("2024-06-01")
    rows = [
        {"entity_id": "A", "event_time": ref - pd.Timedelta(days=3), "amount": 10.0},
        {"entity_id": "A", "event_time": ref - pd.Timedelta(days=10), "amount": 20.0},
        {"entity_id": "A", "event_time": ref - pd.Timedelta(days=20), "amount": 30.0},
        {"entity_id": "A", "event_time": ref - pd.Timedelta(days=60), "amount": 40.0},
        {"entity_id": "B", "event_time": ref - pd.Timedelta(days=45), "amount": 100.0},
    ]
    return pd.DataFrame(rows), ref


@pytest.fixture
def boundary_events_dataset():
    ref = pd.Timestamp("2024-06-01")
    cutoff_7d = ref - pd.Timedelta(days=7)
    rows = [
        {"entity_id": "X", "event_time": cutoff_7d, "amount": 50.0},
        {"entity_id": "Y", "event_time": cutoff_7d - pd.Timedelta(seconds=1), "amount": 75.0},
    ]
    return pd.DataFrame(rows), ref


@pytest.fixture
def recency_known_dates_dataset():
    ref = pd.Timestamp("2024-06-01")
    rows = [
        {"entity_id": "R1", "event_time": ref - pd.Timedelta(days=1), "amount": 10.0},
        {"entity_id": "R1", "event_time": ref - pd.Timedelta(days=61), "amount": 20.0},
        {"entity_id": "R2", "event_time": ref - pd.Timedelta(days=12), "amount": 30.0},
        {"entity_id": "R2", "event_time": ref - pd.Timedelta(days=92), "amount": 40.0},
    ]
    return pd.DataFrame(rows), ref


@pytest.fixture
def leakage_prevention_dataset():
    ref = pd.Timestamp("2024-06-01")
    rows = [
        {"entity_id": "L1", "event_time": ref - pd.Timedelta(days=5), "amount": 10.0},
        {"entity_id": "L1", "event_time": ref - pd.Timedelta(days=2), "amount": 20.0},
        {"entity_id": "L1", "event_time": ref + pd.Timedelta(days=3), "amount": 999.0},
        {"entity_id": "L1", "event_time": ref + pd.Timedelta(days=10), "amount": 888.0},
        {"entity_id": "L2", "event_time": ref - pd.Timedelta(days=1), "amount": 50.0},
        {"entity_id": "L2", "event_time": ref + pd.Timedelta(days=1), "amount": 777.0},
    ]
    return pd.DataFrame(rows), ref


class TestUpperBoundPreventsLeakage:

    def test_events_after_reference_date_excluded_from_window(self, leakage_prevention_dataset):
        df, ref = leakage_prevention_dataset
        agg = TimeWindowAggregator(entity_column="entity_id", time_column="event_time")
        result = agg.aggregate(
            df, windows=["7d"], reference_date=ref, include_event_count=True,
            value_columns=["amount"], agg_funcs=["sum"],
        )
        l1 = result[result["entity_id"] == "L1"]
        assert l1["event_count_7d"].values[0] == 2
        assert l1["amount_sum_7d"].values[0] == pytest.approx(30.0)

    def test_events_after_reference_date_excluded_from_all_time(self, leakage_prevention_dataset):
        df, ref = leakage_prevention_dataset
        agg = TimeWindowAggregator(entity_column="entity_id", time_column="event_time")
        result = agg.aggregate(
            df, windows=["all_time"], reference_date=ref, include_event_count=True,
            value_columns=["amount"], agg_funcs=["sum"],
        )
        l1 = result[result["entity_id"] == "L1"]
        assert l1["event_count_all_time"].values[0] == 2
        assert l1["amount_sum_all_time"].values[0] == pytest.approx(30.0)

        l2 = result[result["entity_id"] == "L2"]
        assert l2["event_count_all_time"].values[0] == 1
        assert l2["amount_sum_all_time"].values[0] == pytest.approx(50.0)

    def test_recency_ignores_future_events(self, leakage_prevention_dataset):
        df, ref = leakage_prevention_dataset
        agg = TimeWindowAggregator(entity_column="entity_id", time_column="event_time")
        result = agg.aggregate(
            df, windows=["7d"], reference_date=ref,
            include_recency=True,
        )
        l1 = result[result["entity_id"] == "L1"]
        assert l1["days_since_last_event"].values[0] == 2

        l2 = result[result["entity_id"] == "L2"]
        assert l2["days_since_last_event"].values[0] == 1

    def test_tenure_ignores_future_events(self, leakage_prevention_dataset):
        df, ref = leakage_prevention_dataset
        agg = TimeWindowAggregator(entity_column="entity_id", time_column="event_time")
        result = agg.aggregate(
            df, windows=["7d"], reference_date=ref,
            include_tenure=True,
        )
        l1 = result[result["entity_id"] == "L1"]
        assert l1["days_since_first_event"].values[0] == 5

        l2 = result[result["entity_id"] == "L2"]
        assert l2["days_since_first_event"].values[0] == 1


class TestCountSumFeaturesNeverNaN:

    def test_event_count_zero_for_inactive_entity(self, two_entity_sparse_dataset):
        df, ref = two_entity_sparse_dataset
        agg = TimeWindowAggregator(entity_column="entity_id", time_column="event_time")
        result = agg.aggregate(df, windows=["7d"], reference_date=ref, include_event_count=True)
        b_count = result[result["entity_id"] == "B"]["event_count_7d"].values[0]
        assert b_count == 0
        assert not np.isnan(b_count)

    def test_sum_zero_for_inactive_entity(self, two_entity_sparse_dataset):
        df, ref = two_entity_sparse_dataset
        agg = TimeWindowAggregator(entity_column="entity_id", time_column="event_time")
        result = agg.aggregate(
            df, windows=["7d"], reference_date=ref,
            value_columns=["amount"], agg_funcs=["sum"],
        )
        b_sum = result[result["entity_id"] == "B"]["amount_sum_7d"].values[0]
        assert b_sum == pytest.approx(0.0)

    def test_count_zero_for_inactive_entity(self, two_entity_sparse_dataset):
        df, ref = two_entity_sparse_dataset
        agg = TimeWindowAggregator(entity_column="entity_id", time_column="event_time")
        result = agg.aggregate(
            df, windows=["7d"], reference_date=ref,
            value_columns=["amount"], agg_funcs=["count"],
        )
        b_count = result[result["entity_id"] == "B"]["amount_count_7d"].values[0]
        assert b_count == 0

    def test_event_count_all_time_always_positive(self, two_entity_sparse_dataset):
        df, ref = two_entity_sparse_dataset
        agg = TimeWindowAggregator(entity_column="entity_id", time_column="event_time")
        result = agg.aggregate(df, windows=["all_time"], reference_date=ref, include_event_count=True)
        assert (result["event_count_all_time"] > 0).all()

    def test_sum_all_time_never_nan(self, two_entity_sparse_dataset):
        df, ref = two_entity_sparse_dataset
        agg = TimeWindowAggregator(entity_column="entity_id", time_column="event_time")
        result = agg.aggregate(
            df, windows=["all_time"], reference_date=ref,
            value_columns=["amount"], agg_funcs=["sum"],
        )
        assert not result["amount_sum_all_time"].isna().any()

    def test_count_columns_no_nan_across_windows(self, two_entity_sparse_dataset):
        df, ref = two_entity_sparse_dataset
        agg = TimeWindowAggregator(entity_column="entity_id", time_column="event_time")
        result = agg.aggregate(
            df, windows=WINDOWS, reference_date=ref,
            value_columns=["amount"], agg_funcs=["count"],
        )
        for w in WINDOWS:
            col = f"amount_count_{w}"
            assert not result[col].isna().any(), f"{col} has NaN"


class TestMeanMaxNaNWhenExpected:

    def test_mean_nan_for_inactive_entity_7d(self, two_entity_sparse_dataset):
        df, ref = two_entity_sparse_dataset
        agg = TimeWindowAggregator(entity_column="entity_id", time_column="event_time")
        result = agg.aggregate(
            df, windows=["7d"], reference_date=ref,
            value_columns=["amount"], agg_funcs=["mean"],
        )
        b_mean = result[result["entity_id"] == "B"]["amount_mean_7d"].values[0]
        assert np.isnan(b_mean)

        a_mean = result[result["entity_id"] == "A"]["amount_mean_7d"].values[0]
        assert not np.isnan(a_mean)

    def test_max_nan_for_inactive_entity_7d(self, two_entity_sparse_dataset):
        df, ref = two_entity_sparse_dataset
        agg = TimeWindowAggregator(entity_column="entity_id", time_column="event_time")
        result = agg.aggregate(
            df, windows=["7d"], reference_date=ref,
            value_columns=["amount"], agg_funcs=["max"],
        )
        b_max = result[result["entity_id"] == "B"]["amount_max_7d"].values[0]
        assert np.isnan(b_max)

    def test_mean_non_nan_for_both_in_90d(self, two_entity_sparse_dataset):
        df, ref = two_entity_sparse_dataset
        agg = TimeWindowAggregator(entity_column="entity_id", time_column="event_time")
        result = agg.aggregate(
            df, windows=["90d"], reference_date=ref,
            value_columns=["amount"], agg_funcs=["mean"],
        )
        assert not result["amount_mean_90d"].isna().any()

    def test_max_matches_known_value(self, two_entity_sparse_dataset):
        df, ref = two_entity_sparse_dataset
        agg = TimeWindowAggregator(entity_column="entity_id", time_column="event_time")
        result = agg.aggregate(
            df, windows=["all_time"], reference_date=ref,
            value_columns=["amount"], agg_funcs=["max"],
        )
        a_max = result[result["entity_id"] == "A"]["amount_max_all_time"].values[0]
        assert a_max == pytest.approx(40.0)

        b_max = result[result["entity_id"] == "B"]["amount_max_all_time"].values[0]
        assert b_max == pytest.approx(100.0)


class TestNullRateMatchesDataDistribution:

    def test_null_rate_7d_is_70_percent(self, controlled_temporal_distribution):
        df, ref = controlled_temporal_distribution
        agg = TimeWindowAggregator(entity_column="entity_id", time_column="event_time")
        result = agg.aggregate(
            df, windows=["7d"], reference_date=ref,
            value_columns=["amount"], agg_funcs=["mean"],
        )
        null_rate = result["amount_mean_7d"].isna().mean()
        assert null_rate == pytest.approx(0.70, abs=0.01)

    def test_null_rate_14d_is_50_percent(self, controlled_temporal_distribution):
        df, ref = controlled_temporal_distribution
        agg = TimeWindowAggregator(entity_column="entity_id", time_column="event_time")
        result = agg.aggregate(
            df, windows=["14d"], reference_date=ref,
            value_columns=["amount"], agg_funcs=["mean"],
        )
        null_rate = result["amount_mean_14d"].isna().mean()
        assert null_rate == pytest.approx(0.50, abs=0.01)

    def test_null_rate_30d_is_30_percent(self, controlled_temporal_distribution):
        df, ref = controlled_temporal_distribution
        agg = TimeWindowAggregator(entity_column="entity_id", time_column="event_time")
        result = agg.aggregate(
            df, windows=["30d"], reference_date=ref,
            value_columns=["amount"], agg_funcs=["mean"],
        )
        null_rate = result["amount_mean_30d"].isna().mean()
        assert null_rate == pytest.approx(0.30, abs=0.01)

    def test_null_rate_90d_is_zero(self, controlled_temporal_distribution):
        df, ref = controlled_temporal_distribution
        agg = TimeWindowAggregator(entity_column="entity_id", time_column="event_time")
        result = agg.aggregate(
            df, windows=["90d"], reference_date=ref,
            value_columns=["amount"], agg_funcs=["mean"],
        )
        null_rate = result["amount_mean_90d"].isna().mean()
        assert null_rate == pytest.approx(0.0, abs=0.01)

    def test_null_rate_all_time_is_zero(self, controlled_temporal_distribution):
        df, ref = controlled_temporal_distribution
        agg = TimeWindowAggregator(entity_column="entity_id", time_column="event_time")
        result = agg.aggregate(
            df, windows=["all_time"], reference_date=ref,
            value_columns=["amount"], agg_funcs=["mean"],
        )
        assert not result["amount_mean_all_time"].isna().any()

    def test_null_rate_monotonically_decreases(self, controlled_temporal_distribution):
        df, ref = controlled_temporal_distribution
        agg = TimeWindowAggregator(entity_column="entity_id", time_column="event_time")
        result = agg.aggregate(
            df, windows=WINDOWS, reference_date=ref,
            value_columns=["amount"], agg_funcs=["mean"],
        )
        null_rates = [result[f"amount_mean_{w}"].isna().mean() for w in WINDOWS]
        for i in range(len(null_rates) - 1):
            assert null_rates[i] >= null_rates[i + 1], (
                f"Null rate for {WINDOWS[i]} ({null_rates[i]:.2f}) < {WINDOWS[i+1]} ({null_rates[i+1]:.2f})"
            )


class TestAllTimeWindowNeverNull:

    @pytest.fixture
    def result_all_time(self, two_entity_sparse_dataset):
        df, ref = two_entity_sparse_dataset
        agg = TimeWindowAggregator(entity_column="entity_id", time_column="event_time")
        return agg.aggregate(
            df, windows=["all_time"], reference_date=ref,
            include_event_count=True,
            value_columns=["amount"], agg_funcs=["sum", "mean", "max", "count"],
        )

    def test_event_count_all_time_no_nulls(self, result_all_time):
        assert not result_all_time["event_count_all_time"].isna().any()

    def test_sum_all_time_no_nulls(self, result_all_time):
        assert not result_all_time["amount_sum_all_time"].isna().any()

    def test_mean_all_time_no_nulls(self, result_all_time):
        assert not result_all_time["amount_mean_all_time"].isna().any()

    def test_max_all_time_no_nulls(self, result_all_time):
        assert not result_all_time["amount_max_all_time"].isna().any()

    def test_count_all_time_no_nulls(self, result_all_time):
        assert not result_all_time["amount_count_all_time"].isna().any()


class TestReferenceDateShiftsNullRates:

    def test_older_reference_date_increases_null_rates(self, controlled_temporal_distribution):
        df, ref = controlled_temporal_distribution
        agg = TimeWindowAggregator(entity_column="entity_id", time_column="event_time")

        result_current = agg.aggregate(
            df, windows=["30d"], reference_date=ref,
            value_columns=["amount"], agg_funcs=["mean"],
        )
        older_ref = ref - pd.Timedelta(days=60)
        result_older = agg.aggregate(
            df, windows=["30d"], reference_date=older_ref,
            value_columns=["amount"], agg_funcs=["mean"],
        )
        null_current = result_current["amount_mean_30d"].isna().mean()
        null_older = result_older["amount_mean_30d"].isna().mean()
        assert null_older >= null_current

    def test_ref_date_at_data_min_makes_7d_all_null(self, controlled_temporal_distribution):
        df, ref = controlled_temporal_distribution
        data_min = df["event_time"].min()
        very_early_ref = data_min - pd.Timedelta(days=1)
        agg = TimeWindowAggregator(entity_column="entity_id", time_column="event_time")
        result = agg.aggregate(
            df, windows=["7d"], reference_date=very_early_ref,
            value_columns=["amount"], agg_funcs=["mean"],
        )
        assert result["amount_mean_7d"].isna().all()


class TestEndToEndAggregationPipeline:

    def test_one_row_per_entity(self, two_entity_sparse_dataset):
        df, ref = two_entity_sparse_dataset
        agg = TimeWindowAggregator(entity_column="entity_id", time_column="event_time")
        result = agg.aggregate(
            df, windows=WINDOWS, reference_date=ref,
            include_event_count=True,
            value_columns=["amount"], agg_funcs=["sum", "count"],
        )
        assert len(result) == df["entity_id"].nunique()

    def test_no_nan_in_count_sum_columns(self, two_entity_sparse_dataset):
        df, ref = two_entity_sparse_dataset
        agg = TimeWindowAggregator(entity_column="entity_id", time_column="event_time")
        result = agg.aggregate(
            df, windows=WINDOWS, reference_date=ref,
            include_event_count=True,
            value_columns=["amount"], agg_funcs=["sum", "count"],
        )
        for w in WINDOWS:
            assert not result[f"event_count_{w}"].isna().any()
            assert not result[f"amount_sum_{w}"].isna().any()
            assert not result[f"amount_count_{w}"].isna().any()

    def test_column_naming_pattern(self, two_entity_sparse_dataset):
        df, ref = two_entity_sparse_dataset
        agg = TimeWindowAggregator(entity_column="entity_id", time_column="event_time")
        result = agg.aggregate(
            df, windows=["30d"], reference_date=ref,
            value_columns=["amount"], agg_funcs=["sum", "mean"],
        )
        assert "amount_sum_30d" in result.columns
        assert "amount_mean_30d" in result.columns


class TestRecencyFeatureCorrectness:

    def test_exact_recency_values(self, recency_known_dates_dataset):
        df, ref = recency_known_dates_dataset
        agg = TimeWindowAggregator(entity_column="entity_id", time_column="event_time")
        result = agg.aggregate(
            df, windows=["all_time"], reference_date=ref, include_recency=True,
        )
        r1_recency = result[result["entity_id"] == "R1"]["days_since_last_event"].values[0]
        assert r1_recency == 1

        r2_recency = result[result["entity_id"] == "R2"]["days_since_last_event"].values[0]
        assert r2_recency == 12

    def test_exact_tenure_values(self, recency_known_dates_dataset):
        df, ref = recency_known_dates_dataset
        agg = TimeWindowAggregator(entity_column="entity_id", time_column="event_time")
        result = agg.aggregate(
            df, windows=["all_time"], reference_date=ref, include_tenure=True,
        )
        r1_tenure = result[result["entity_id"] == "R1"]["days_since_first_event"].values[0]
        assert r1_tenure == 61

        r2_tenure = result[result["entity_id"] == "R2"]["days_since_first_event"].values[0]
        assert r2_tenure == 92

    def test_recency_non_negative(self, recency_known_dates_dataset):
        df, ref = recency_known_dates_dataset
        agg = TimeWindowAggregator(entity_column="entity_id", time_column="event_time")
        result = agg.aggregate(
            df, windows=["all_time"], reference_date=ref, include_recency=True,
        )
        assert (result["days_since_last_event"] >= 0).all()

    def test_tenure_gte_recency(self, recency_known_dates_dataset):
        df, ref = recency_known_dates_dataset
        agg = TimeWindowAggregator(entity_column="entity_id", time_column="event_time")
        result = agg.aggregate(
            df, windows=["all_time"], reference_date=ref,
            include_recency=True, include_tenure=True,
        )
        assert (result["days_since_first_event"] >= result["days_since_last_event"]).all()


class TestWindowBoundaryCorrectness:

    def test_event_at_exact_cutoff_included(self, boundary_events_dataset):
        df, ref = boundary_events_dataset
        agg = TimeWindowAggregator(entity_column="entity_id", time_column="event_time")
        result = agg.aggregate(df, windows=["7d"], reference_date=ref, include_event_count=True)
        x_count = result[result["entity_id"] == "X"]["event_count_7d"].values[0]
        assert x_count == 1

    def test_event_1s_before_cutoff_excluded(self, boundary_events_dataset):
        df, ref = boundary_events_dataset
        agg = TimeWindowAggregator(entity_column="entity_id", time_column="event_time")
        result = agg.aggregate(df, windows=["7d"], reference_date=ref, include_event_count=True)
        y_count = result[result["entity_id"] == "Y"]["event_count_7d"].values[0]
        assert y_count == 0

    def test_all_time_includes_all_events_before_ref(self, boundary_events_dataset):
        df, ref = boundary_events_dataset
        agg = TimeWindowAggregator(entity_column="entity_id", time_column="event_time")
        result = agg.aggregate(df, windows=["all_time"], reference_date=ref, include_event_count=True)
        assert (result["event_count_all_time"] == 1).all()
