import numpy as np
import pandas as pd
import pytest

from customer_retention.stages.profiling.spark_temporal_feature_engineer import (
    SparkTemporalFeatureEngineer,
)
from customer_retention.stages.profiling.temporal_feature_engineer import (
    FeatureGroup,
    ReferenceMode,
    TemporalAggregationConfig,
    TemporalFeatureEngineer,
    TemporalFeatureResult,
)

BOTH_ENGINEERS = [TemporalFeatureEngineer, SparkTemporalFeatureEngineer]


@pytest.fixture
def sample_events():
    np.random.seed(42)
    dates_a = pd.date_range("2023-06-01", periods=20, freq="7D")
    customer_a = pd.DataFrame({
        "customer_id": "A",
        "event_date": dates_a,
        "amount": np.random.uniform(80, 120, 20),
    })
    dates_b = pd.date_range("2023-06-01", periods=15, freq="10D")
    amounts_b = np.linspace(150, 30, 15) + np.random.uniform(-10, 10, 15)
    customer_b = pd.DataFrame({
        "customer_id": "B",
        "event_date": dates_b,
        "amount": amounts_b,
    })
    dates_c = pd.date_range("2023-10-01", periods=5, freq="7D")
    customer_c = pd.DataFrame({
        "customer_id": "C",
        "event_date": dates_c,
        "amount": np.random.uniform(50, 70, 5),
    })
    dates_d = list(pd.date_range("2023-06-01", periods=3, freq="3D")) + \
              list(pd.date_range("2023-08-15", periods=5, freq="2D"))
    customer_d = pd.DataFrame({
        "customer_id": "D",
        "event_date": dates_d,
        "amount": np.random.uniform(40, 60, 8),
    })
    return pd.concat([customer_a, customer_b, customer_c, customer_d], ignore_index=True)


@pytest.fixture
def reference_dates():
    return pd.DataFrame({
        "customer_id": ["A", "B", "C", "D"],
        "reference_date": pd.to_datetime([
            "2023-11-01", "2023-10-15", "2023-11-01", "2023-09-01",
        ])
    })


def _sort_result(df, entity_col="customer_id"):
    return df.sort_values(entity_col).reset_index(drop=True)


class TestSparkEngineerMatchesLocal:

    @pytest.mark.parametrize("cls", BOTH_ENGINEERS)
    def test_compute_returns_result(self, sample_events, reference_dates, cls):
        engineer = cls()
        result = engineer.compute(
            events_df=sample_events, entity_col="customer_id",
            time_col="event_date", value_cols=["amount"],
            reference_dates=reference_dates, reference_col="reference_date",
        )
        assert isinstance(result, TemporalFeatureResult)
        assert "customer_id" in result.features_df.columns

    @pytest.mark.parametrize("cls", BOTH_ENGINEERS)
    def test_lagged_features_created(self, sample_events, reference_dates, cls):
        engineer = cls()
        result = engineer.compute(
            events_df=sample_events, entity_col="customer_id",
            time_col="event_date", value_cols=["amount"],
            reference_dates=reference_dates, reference_col="reference_date",
        )
        df = result.features_df
        for lag in range(4):
            assert f"lag{lag}_amount_sum" in df.columns

    @pytest.mark.parametrize("cls", BOTH_ENGINEERS)
    def test_velocity_features_created(self, sample_events, reference_dates, cls):
        engineer = cls()
        result = engineer.compute(
            events_df=sample_events, entity_col="customer_id",
            time_col="event_date", value_cols=["amount"],
            reference_dates=reference_dates, reference_col="reference_date",
        )
        assert "amount_velocity" in result.features_df.columns
        assert "amount_velocity_pct" in result.features_df.columns

    @pytest.mark.parametrize("cls", BOTH_ENGINEERS)
    def test_acceleration_features_created(self, sample_events, reference_dates, cls):
        engineer = cls()
        result = engineer.compute(
            events_df=sample_events, entity_col="customer_id",
            time_col="event_date", value_cols=["amount"],
            reference_dates=reference_dates, reference_col="reference_date",
        )
        assert "amount_acceleration" in result.features_df.columns
        assert "amount_momentum" in result.features_df.columns

    @pytest.mark.parametrize("cls", BOTH_ENGINEERS)
    def test_lifecycle_features_created(self, sample_events, reference_dates, cls):
        engineer = cls()
        result = engineer.compute(
            events_df=sample_events, entity_col="customer_id",
            time_col="event_date", value_cols=["amount"],
            reference_dates=reference_dates, reference_col="reference_date",
        )
        df = result.features_df
        assert "amount_beginning" in df.columns
        assert "amount_middle" in df.columns
        assert "amount_end" in df.columns
        assert "amount_trend_ratio" in df.columns

    @pytest.mark.parametrize("cls", BOTH_ENGINEERS)
    def test_recency_features_created(self, sample_events, reference_dates, cls):
        engineer = cls()
        result = engineer.compute(
            events_df=sample_events, entity_col="customer_id",
            time_col="event_date", value_cols=["amount"],
            reference_dates=reference_dates, reference_col="reference_date",
        )
        df = result.features_df
        assert "days_since_last_event" in df.columns
        assert "days_since_first_event" in df.columns
        assert "active_span_days" in df.columns
        assert "recency_ratio" in df.columns

    @pytest.mark.parametrize("cls", BOTH_ENGINEERS)
    def test_regularity_features_created(self, sample_events, reference_dates, cls):
        engineer = cls()
        result = engineer.compute(
            events_df=sample_events, entity_col="customer_id",
            time_col="event_date", value_cols=["amount"],
            reference_dates=reference_dates, reference_col="reference_date",
        )
        df = result.features_df
        assert "event_frequency" in df.columns
        assert "regularity_score" in df.columns

    @pytest.mark.parametrize("cls", BOTH_ENGINEERS)
    def test_cohort_features_created(self, sample_events, reference_dates, cls):
        engineer = cls()
        result = engineer.compute(
            events_df=sample_events, entity_col="customer_id",
            time_col="event_date", value_cols=["amount"],
            reference_dates=reference_dates, reference_col="reference_date",
        )
        df = result.features_df
        assert "amount_vs_cohort_mean" in df.columns
        assert "amount_vs_cohort_pct" in df.columns

    @pytest.mark.parametrize("cls", BOTH_ENGINEERS)
    def test_seven_feature_groups(self, sample_events, reference_dates, cls):
        engineer = cls()
        result = engineer.compute(
            events_df=sample_events, entity_col="customer_id",
            time_col="event_date", value_cols=["amount"],
            reference_dates=reference_dates, reference_col="reference_date",
        )
        assert len(result.feature_groups) == 7
        groups = {g.group for g in result.feature_groups}
        assert groups == {fg for fg in FeatureGroup}


class TestSparkEngineerNumericMatch:

    def test_full_output_matches(self, sample_events, reference_dates):
        kwargs = dict(
            events_df=sample_events, entity_col="customer_id",
            time_col="event_date", value_cols=["amount"],
            reference_dates=reference_dates, reference_col="reference_date",
        )
        r_local = _sort_result(TemporalFeatureEngineer().compute(**kwargs).features_df)
        r_spark = _sort_result(SparkTemporalFeatureEngineer().compute(**kwargs).features_df)

        assert set(r_local.columns) == set(r_spark.columns)
        for col in r_local.columns:
            if col == "customer_id":
                continue
            local_vals = r_local[col].to_numpy().astype(float)
            spark_vals = r_spark[col].to_numpy().astype(float)
            both_nan = np.isnan(local_vals) & np.isnan(spark_vals)
            if both_nan.all():
                continue
            valid = ~np.isnan(local_vals) & ~np.isnan(spark_vals)
            if valid.any():
                np.testing.assert_allclose(
                    local_vals[valid], spark_vals[valid],
                    rtol=1e-6, atol=1e-10,
                    err_msg=f"Column {col} mismatch",
                )

    def test_lifecycle_values_match(self, sample_events, reference_dates):
        config = TemporalAggregationConfig(min_history_days=30)
        kwargs = dict(
            events_df=sample_events, entity_col="customer_id",
            time_col="event_date", value_cols=["amount"],
            reference_dates=reference_dates, reference_col="reference_date",
        )
        r_local = _sort_result(TemporalFeatureEngineer(config).compute(**kwargs).features_df)
        r_spark = _sort_result(SparkTemporalFeatureEngineer(config).compute(**kwargs).features_df)

        for col in ["amount_beginning", "amount_middle", "amount_end", "amount_trend_ratio"]:
            local_vals = r_local[col].to_numpy().astype(float)
            spark_vals = r_spark[col].to_numpy().astype(float)
            both_nan = np.isnan(local_vals) & np.isnan(spark_vals)
            valid = ~np.isnan(local_vals) & ~np.isnan(spark_vals)
            if valid.any():
                np.testing.assert_allclose(
                    local_vals[valid], spark_vals[valid],
                    rtol=1e-6, atol=1e-10,
                    err_msg=f"Lifecycle column {col} mismatch",
                )

    def test_regularity_values_match(self, sample_events, reference_dates):
        kwargs = dict(
            events_df=sample_events, entity_col="customer_id",
            time_col="event_date", value_cols=["amount"],
            reference_dates=reference_dates, reference_col="reference_date",
        )
        r_local = _sort_result(TemporalFeatureEngineer().compute(**kwargs).features_df)
        r_spark = _sort_result(SparkTemporalFeatureEngineer().compute(**kwargs).features_df)

        for col in ["event_frequency", "inter_event_gap_mean", "inter_event_gap_std",
                     "inter_event_gap_max", "regularity_score"]:
            local_vals = r_local[col].to_numpy().astype(float)
            spark_vals = r_spark[col].to_numpy().astype(float)
            valid = ~np.isnan(local_vals) & ~np.isnan(spark_vals)
            if valid.any():
                np.testing.assert_allclose(
                    local_vals[valid], spark_vals[valid],
                    rtol=1e-6, atol=1e-10,
                    err_msg=f"Regularity column {col} mismatch",
                )


class TestSparkEngineerEdgeCases:

    @pytest.mark.parametrize("cls", BOTH_ENGINEERS)
    def test_single_entity_single_event(self, cls):
        df = pd.DataFrame({
            "customer_id": ["A"],
            "event_date": [pd.Timestamp("2023-01-01")],
            "amount": [100.0],
        })
        engineer = cls()
        result = engineer.compute(
            events_df=df, entity_col="customer_id",
            time_col="event_date", value_cols=["amount"],
        )
        assert len(result.features_df) == 1
        assert pd.isna(result.features_df.iloc[0]["inter_event_gap_mean"])

    @pytest.mark.parametrize("cls", BOTH_ENGINEERS)
    def test_all_events_same_timestamp(self, cls):
        df = pd.DataFrame({
            "customer_id": ["A"] * 5,
            "event_date": [pd.Timestamp("2023-01-01")] * 5,
            "amount": [10.0, 20.0, 30.0, 40.0, 50.0],
        })
        engineer = cls()
        result = engineer.compute(
            events_df=df, entity_col="customer_id",
            time_col="event_date", value_cols=["amount"],
        )
        df_out = result.features_df
        assert df_out.iloc[0]["event_frequency"] == 5

    @pytest.mark.parametrize("cls", BOTH_ENGINEERS)
    def test_lifecycle_nan_for_short_history(self, cls):
        df = pd.DataFrame({
            "customer_id": ["A"] * 5,
            "event_date": pd.date_range("2023-01-01", periods=5, freq="3D"),
            "amount": [10.0, 20.0, 30.0, 40.0, 50.0],
        })
        config = TemporalAggregationConfig(min_history_days=60)
        engineer = cls(config)
        result = engineer.compute(
            events_df=df, entity_col="customer_id",
            time_col="event_date", value_cols=["amount"],
        )
        assert pd.isna(result.features_df.iloc[0]["amount_beginning"])

    @pytest.mark.parametrize("cls", BOTH_ENGINEERS)
    def test_zero_beginning_no_trend_ratio(self, cls):
        df = pd.DataFrame({
            "customer_id": ["A"] * 30,
            "event_date": pd.date_range("2023-01-01", periods=30, freq="5D"),
            "amount": [0.0] * 10 + [50.0] * 10 + [100.0] * 10,
        })
        config = TemporalAggregationConfig(min_history_days=30)
        engineer = cls(config)
        result = engineer.compute(
            events_df=df, entity_col="customer_id",
            time_col="event_date", value_cols=["amount"],
        )
        assert pd.isna(result.features_df.iloc[0]["amount_trend_ratio"])

    @pytest.mark.parametrize("cls", BOTH_ENGINEERS)
    def test_disabled_groups(self, sample_events, reference_dates, cls):
        config = TemporalAggregationConfig(
            compute_velocity=False, compute_lifecycle=False,
            compute_regularity=False, compute_cohort=False,
        )
        engineer = cls(config)
        result = engineer.compute(
            events_df=sample_events, entity_col="customer_id",
            time_col="event_date", value_cols=["amount"],
            reference_dates=reference_dates, reference_col="reference_date",
        )
        assert "amount_velocity" not in result.features_df.columns
        assert "amount_beginning" not in result.features_df.columns
        assert "regularity_score" not in result.features_df.columns

    @pytest.mark.parametrize("cls", BOTH_ENGINEERS)
    def test_global_reference_mode(self, sample_events, cls):
        from datetime import datetime
        config = TemporalAggregationConfig(
            reference_mode=ReferenceMode.GLOBAL_DATE,
            global_reference_date=datetime(2023, 11, 1),
        )
        engineer = cls(config)
        result = engineer.compute(
            events_df=sample_events, entity_col="customer_id",
            time_col="event_date", value_cols=["amount"],
        )
        assert len(result.features_df) == sample_events["customer_id"].nunique()

    @pytest.mark.parametrize("cls", BOTH_ENGINEERS)
    def test_multiple_value_columns(self, cls):
        df = pd.DataFrame({
            "customer_id": ["A"] * 10 + ["B"] * 10,
            "event_date": list(pd.date_range("2023-08-01", periods=10, freq="7D")) * 2,
            "amount": np.random.uniform(50, 150, 20),
            "quantity": np.random.randint(1, 10, 20).astype(float),
        })
        engineer = cls()
        result = engineer.compute(
            events_df=df, entity_col="customer_id",
            time_col="event_date", value_cols=["amount", "quantity"],
        )
        assert "lag0_amount_sum" in result.features_df.columns
        assert "lag0_quantity_sum" in result.features_df.columns

    @pytest.mark.parametrize("cls", BOTH_ENGINEERS)
    def test_tz_aware_timestamps(self, cls):
        df = pd.DataFrame({
            "customer_id": ["A"] * 10 + ["B"] * 10,
            "event_date": list(pd.date_range("2023-08-01", periods=10, freq="7D", tz="UTC")) * 2,
            "amount": np.random.uniform(50, 150, 20),
        })
        engineer = cls()
        result = engineer.compute(
            events_df=df, entity_col="customer_id",
            time_col="event_date", value_cols=["amount"],
        )
        assert len(result.features_df) == 2

    @pytest.mark.parametrize("cls", BOTH_ENGINEERS)
    def test_nullable_integer_columns(self, cls):
        df = pd.DataFrame({
            "customer_id": ["A"] * 10 + ["B"] * 10,
            "event_date": list(pd.date_range("2023-01-01", periods=10, freq="5D")) * 2,
            "amount": pd.array(
                [10, 20, 30, pd.NA, 50, 60, pd.NA, 80, 90, 100] * 2,
                dtype="Int64",
            ),
        })
        engineer = cls()
        result = engineer.compute(
            events_df=df, entity_col="customer_id",
            time_col="event_date", value_cols=["amount"],
        )
        assert len(result.features_df) == 2


class TestSparkEngineerCatalogAndMetadata:

    @pytest.mark.parametrize("cls", BOTH_ENGINEERS)
    def test_get_catalog(self, sample_events, reference_dates, cls):
        result = cls().compute(
            events_df=sample_events, entity_col="customer_id",
            time_col="event_date", value_cols=["amount"],
            reference_dates=reference_dates, reference_col="reference_date",
        )
        catalog = result.get_catalog()
        assert "TEMPORAL FEATURE CATALOG" in catalog

    @pytest.mark.parametrize("cls", BOTH_ENGINEERS)
    def test_to_dict(self, sample_events, reference_dates, cls):
        result = cls().compute(
            events_df=sample_events, entity_col="customer_id",
            time_col="event_date", value_cols=["amount"],
            reference_dates=reference_dates, reference_col="reference_date",
        )
        d = result.to_dict()
        assert d["n_entities"] == 4
        assert d["n_features"] > 0
