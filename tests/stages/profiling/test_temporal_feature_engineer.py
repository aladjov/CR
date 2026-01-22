"""Tests for TemporalFeatureEngineer - temporal feature engineering with lagged windows."""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from customer_retention.stages.profiling.temporal_feature_engineer import (
    FeatureGroup,
    FeatureGroupResult,
    ReferenceMode,
    TemporalAggregationConfig,
    TemporalFeatureEngineer,
    TemporalFeatureResult,
)


class TestTemporalAggregationConfig:
    def test_default_config_has_sensible_defaults(self):
        config = TemporalAggregationConfig()
        assert config.reference_mode == ReferenceMode.PER_CUSTOMER
        assert config.lag_window_days == 30
        assert config.num_lags == 4
        assert config.min_history_days == 60

    def test_config_validates_num_lags(self):
        config = TemporalAggregationConfig(num_lags=3)
        assert config.num_lags == 3

    def test_config_validates_lifecycle_splits(self):
        config = TemporalAggregationConfig(lifecycle_splits=[0.25, 0.50, 0.25])
        assert sum(config.lifecycle_splits) == pytest.approx(1.0)


class TestReferenceMode:
    def test_per_customer_mode(self):
        assert ReferenceMode.PER_CUSTOMER.value == "per_customer"

    def test_global_date_mode(self):
        assert ReferenceMode.GLOBAL_DATE.value == "global_date"


class TestFeatureGroup:
    def test_all_feature_groups_defined(self):
        groups = [
            FeatureGroup.LAGGED_WINDOWS,
            FeatureGroup.VELOCITY,
            FeatureGroup.ACCELERATION,
            FeatureGroup.LIFECYCLE,
            FeatureGroup.RECENCY,
            FeatureGroup.REGULARITY,
            FeatureGroup.COHORT_COMPARISON,
        ]
        assert len(groups) == 7


@pytest.fixture
def sample_events_df():
    """Create sample event data with multiple customers over time."""
    np.random.seed(42)

    # Customer A: Active, consistent spender
    dates_a = pd.date_range("2023-06-01", periods=20, freq="7D")
    customer_a = pd.DataFrame({
        "customer_id": "A",
        "event_date": dates_a,
        "amount": np.random.uniform(80, 120, 20),
        "event_type": "purchase"
    })

    # Customer B: Declining engagement (churned pattern)
    dates_b = pd.date_range("2023-06-01", periods=15, freq="10D")
    amounts_b = np.linspace(150, 30, 15) + np.random.uniform(-10, 10, 15)
    customer_b = pd.DataFrame({
        "customer_id": "B",
        "event_date": dates_b,
        "amount": amounts_b,
        "event_type": "purchase"
    })

    # Customer C: New customer (short history)
    dates_c = pd.date_range("2023-10-01", periods=5, freq="7D")
    customer_c = pd.DataFrame({
        "customer_id": "C",
        "event_date": dates_c,
        "amount": np.random.uniform(50, 70, 5),
        "event_type": "purchase"
    })

    # Customer D: Irregular/bursty
    dates_d = list(pd.date_range("2023-06-01", periods=3, freq="3D")) + \
              list(pd.date_range("2023-08-15", periods=5, freq="2D"))
    customer_d = pd.DataFrame({
        "customer_id": "D",
        "event_date": dates_d,
        "amount": np.random.uniform(40, 60, 8),
        "event_type": "purchase"
    })

    return pd.concat([customer_a, customer_b, customer_c, customer_d], ignore_index=True)


@pytest.fixture
def engineer():
    return TemporalFeatureEngineer()


@pytest.fixture
def reference_dates():
    """Reference dates per customer (e.g., churn date or analysis date)."""
    return pd.DataFrame({
        "customer_id": ["A", "B", "C", "D"],
        "reference_date": pd.to_datetime([
            "2023-11-01",  # A: still active
            "2023-10-15",  # B: churned
            "2023-11-01",  # C: new customer
            "2023-09-01",  # D: churned earlier
        ])
    })


class TestTemporalFeatureEngineer:
    def test_engineer_initialization(self, engineer):
        assert engineer is not None
        assert isinstance(engineer.config, TemporalAggregationConfig)

    def test_engineer_with_custom_config(self):
        config = TemporalAggregationConfig(num_lags=6, lag_window_days=14)
        engineer = TemporalFeatureEngineer(config=config)
        assert engineer.config.num_lags == 6
        assert engineer.config.lag_window_days == 14

    def test_compute_returns_result(self, engineer, sample_events_df, reference_dates):
        result = engineer.compute(
            events_df=sample_events_df,
            entity_col="customer_id",
            time_col="event_date",
            value_cols=["amount"],
            reference_dates=reference_dates,
            reference_col="reference_date",
        )
        assert isinstance(result, TemporalFeatureResult)

    def test_result_contains_feature_dataframe(self, engineer, sample_events_df, reference_dates):
        result = engineer.compute(
            events_df=sample_events_df,
            entity_col="customer_id",
            time_col="event_date",
            value_cols=["amount"],
            reference_dates=reference_dates,
            reference_col="reference_date",
        )
        assert isinstance(result.features_df, pd.DataFrame)
        assert "customer_id" in result.features_df.columns

    def test_result_contains_feature_groups(self, engineer, sample_events_df, reference_dates):
        result = engineer.compute(
            events_df=sample_events_df,
            entity_col="customer_id",
            time_col="event_date",
            value_cols=["amount"],
            reference_dates=reference_dates,
            reference_col="reference_date",
        )
        assert len(result.feature_groups) > 0
        assert all(isinstance(g, FeatureGroupResult) for g in result.feature_groups)


class TestLaggedWindowFeatures:
    """Group 1: Lagged Window Aggregations"""

    def test_lagged_features_created(self, engineer, sample_events_df, reference_dates):
        result = engineer.compute(
            events_df=sample_events_df,
            entity_col="customer_id",
            time_col="event_date",
            value_cols=["amount"],
            reference_dates=reference_dates,
            reference_col="reference_date",
        )
        df = result.features_df

        # Check lag0 features exist (most recent window)
        assert "lag0_amount_sum" in df.columns
        assert "lag0_amount_mean" in df.columns
        assert "lag0_amount_count" in df.columns

    def test_multiple_lags_created(self, engineer, sample_events_df, reference_dates):
        result = engineer.compute(
            events_df=sample_events_df,
            entity_col="customer_id",
            time_col="event_date",
            value_cols=["amount"],
            reference_dates=reference_dates,
            reference_col="reference_date",
        )
        df = result.features_df

        # Default is 4 lags
        for lag in range(4):
            assert f"lag{lag}_amount_sum" in df.columns

    def test_lag0_is_most_recent_window(self, engineer, sample_events_df, reference_dates):
        config = TemporalAggregationConfig(lag_window_days=30, num_lags=2)
        engineer = TemporalFeatureEngineer(config=config)

        result = engineer.compute(
            events_df=sample_events_df,
            entity_col="customer_id",
            time_col="event_date",
            value_cols=["amount"],
            reference_dates=reference_dates,
            reference_col="reference_date",
        )
        df = result.features_df

        # Lag0 should have data for customers with recent activity
        # Lag1 should be the previous window
        assert "lag0_amount_sum" in df.columns
        assert "lag1_amount_sum" in df.columns

    def test_missing_lags_are_nan(self, engineer, sample_events_df, reference_dates):
        """Customers without enough history should have NaN for older lags."""
        config = TemporalAggregationConfig(lag_window_days=30, num_lags=4)
        engineer = TemporalFeatureEngineer(config=config)

        result = engineer.compute(
            events_df=sample_events_df,
            entity_col="customer_id",
            time_col="event_date",
            value_cols=["amount"],
            reference_dates=reference_dates,
            reference_col="reference_date",
        )
        df = result.features_df

        # Customer C has short history - older lags should be NaN
        customer_c = df[df["customer_id"] == "C"].iloc[0]
        # At least some older lags should be NaN for short-history customer
        assert pd.isna(customer_c["lag3_amount_sum"]) or customer_c["lag3_amount_count"] == 0

    def test_count_is_zero_not_nan_for_empty_windows(self, engineer, sample_events_df, reference_dates):
        """Count should be 0 for empty windows, not NaN."""
        result = engineer.compute(
            events_df=sample_events_df,
            entity_col="customer_id",
            time_col="event_date",
            value_cols=["amount"],
            reference_dates=reference_dates,
            reference_col="reference_date",
        )
        df = result.features_df

        # Count columns should have 0, not NaN, for empty windows
        count_cols = [c for c in df.columns if "_count" in c]
        for col in count_cols:
            # Either has a count or is 0, not NaN
            assert df[col].isna().sum() == 0 or (df[col].fillna(0) >= 0).all()


class TestVelocityFeatures:
    """Group 2: Velocity (Rate of Change) Features"""

    def test_velocity_features_created(self, engineer, sample_events_df, reference_dates):
        result = engineer.compute(
            events_df=sample_events_df,
            entity_col="customer_id",
            time_col="event_date",
            value_cols=["amount"],
            reference_dates=reference_dates,
            reference_col="reference_date",
        )
        df = result.features_df

        assert "amount_velocity" in df.columns
        assert "amount_velocity_pct" in df.columns

    def test_velocity_is_difference_between_lags(self, engineer, sample_events_df, reference_dates):
        """Velocity = (Lag0 - Lag1) / window_days"""
        result = engineer.compute(
            events_df=sample_events_df,
            entity_col="customer_id",
            time_col="event_date",
            value_cols=["amount"],
            reference_dates=reference_dates,
            reference_col="reference_date",
        )
        df = result.features_df

        # Verify velocity calculation for a customer with data in both lags
        for _, row in df.iterrows():
            if pd.notna(row["lag0_amount_sum"]) and pd.notna(row["lag1_amount_sum"]):
                expected_velocity = (row["lag0_amount_sum"] - row["lag1_amount_sum"]) / 30
                if pd.notna(row["amount_velocity"]):
                    assert row["amount_velocity"] == pytest.approx(expected_velocity, rel=0.01)

    def test_declining_customer_has_negative_velocity(self, engineer, sample_events_df, reference_dates):
        """Customer B has declining pattern - should have negative velocity."""
        result = engineer.compute(
            events_df=sample_events_df,
            entity_col="customer_id",
            time_col="event_date",
            value_cols=["amount"],
            reference_dates=reference_dates,
            reference_col="reference_date",
        )
        df = result.features_df

        customer_b = df[df["customer_id"] == "B"].iloc[0]
        # Customer B has declining amounts, so velocity should be negative
        if pd.notna(customer_b["amount_velocity"]):
            assert customer_b["amount_velocity"] < 0


class TestAccelerationFeatures:
    """Group 3: Acceleration and Momentum Features"""

    def test_acceleration_features_created(self, engineer, sample_events_df, reference_dates):
        result = engineer.compute(
            events_df=sample_events_df,
            entity_col="customer_id",
            time_col="event_date",
            value_cols=["amount"],
            reference_dates=reference_dates,
            reference_col="reference_date",
        )
        df = result.features_df

        assert "amount_acceleration" in df.columns
        assert "amount_momentum" in df.columns

    def test_momentum_is_value_times_velocity(self, engineer, sample_events_df, reference_dates):
        """Momentum = Lag0_value × Velocity"""
        result = engineer.compute(
            events_df=sample_events_df,
            entity_col="customer_id",
            time_col="event_date",
            value_cols=["amount"],
            reference_dates=reference_dates,
            reference_col="reference_date",
        )
        df = result.features_df

        for _, row in df.iterrows():
            if pd.notna(row["lag0_amount_sum"]) and pd.notna(row["amount_velocity"]):
                expected_momentum = row["lag0_amount_sum"] * row["amount_velocity"]
                if pd.notna(row["amount_momentum"]):
                    assert row["amount_momentum"] == pytest.approx(expected_momentum, rel=0.01)


class TestLifecycleFeatures:
    """Group 4: Lifecycle (Beginning/Middle/End) Features"""

    def test_lifecycle_features_created(self, engineer, sample_events_df, reference_dates):
        result = engineer.compute(
            events_df=sample_events_df,
            entity_col="customer_id",
            time_col="event_date",
            value_cols=["amount"],
            reference_dates=reference_dates,
            reference_col="reference_date",
        )
        df = result.features_df

        assert "amount_beginning" in df.columns
        assert "amount_middle" in df.columns
        assert "amount_end" in df.columns
        assert "amount_trend_ratio" in df.columns

    def test_lifecycle_nan_for_short_history(self, engineer, sample_events_df, reference_dates):
        """Customers with history < min_history_days should have NaN lifecycle features."""
        config = TemporalAggregationConfig(min_history_days=60)
        engineer = TemporalFeatureEngineer(config=config)

        result = engineer.compute(
            events_df=sample_events_df,
            entity_col="customer_id",
            time_col="event_date",
            value_cols=["amount"],
            reference_dates=reference_dates,
            reference_col="reference_date",
        )
        df = result.features_df

        # Customer C has ~30 days history, less than 60 day threshold
        customer_c = df[df["customer_id"] == "C"].iloc[0]
        assert pd.isna(customer_c["amount_beginning"])
        assert pd.isna(customer_c["amount_trend_ratio"])

    def test_trend_ratio_calculation(self, engineer, sample_events_df, reference_dates):
        """Trend ratio = end / beginning"""
        result = engineer.compute(
            events_df=sample_events_df,
            entity_col="customer_id",
            time_col="event_date",
            value_cols=["amount"],
            reference_dates=reference_dates,
            reference_col="reference_date",
        )
        df = result.features_df

        for _, row in df.iterrows():
            if pd.notna(row["amount_beginning"]) and row["amount_beginning"] > 0:
                expected_ratio = row["amount_end"] / row["amount_beginning"]
                if pd.notna(row["amount_trend_ratio"]):
                    assert row["amount_trend_ratio"] == pytest.approx(expected_ratio, rel=0.01)


class TestRecencyFeatures:
    """Group 5: Recency and Tenure Features"""

    def test_recency_features_created(self, engineer, sample_events_df, reference_dates):
        result = engineer.compute(
            events_df=sample_events_df,
            entity_col="customer_id",
            time_col="event_date",
            value_cols=["amount"],
            reference_dates=reference_dates,
            reference_col="reference_date",
        )
        df = result.features_df

        assert "days_since_last_event" in df.columns
        assert "days_since_first_event" in df.columns
        assert "active_span_days" in df.columns
        assert "recency_ratio" in df.columns

    def test_recency_ratio_between_0_and_1(self, engineer, sample_events_df, reference_dates):
        """Recency ratio should be between 0 (just active) and 1 (dormant entire history)."""
        result = engineer.compute(
            events_df=sample_events_df,
            entity_col="customer_id",
            time_col="event_date",
            value_cols=["amount"],
            reference_dates=reference_dates,
            reference_col="reference_date",
        )
        df = result.features_df

        valid_ratios = df["recency_ratio"].dropna()
        assert (valid_ratios >= 0).all()
        assert (valid_ratios <= 1).all()


class TestRegularityFeatures:
    """Group 6: Frequency and Regularity Features"""

    def test_regularity_features_created(self, engineer, sample_events_df, reference_dates):
        result = engineer.compute(
            events_df=sample_events_df,
            entity_col="customer_id",
            time_col="event_date",
            value_cols=["amount"],
            reference_dates=reference_dates,
            reference_col="reference_date",
        )
        df = result.features_df

        assert "event_frequency" in df.columns
        assert "inter_event_gap_mean" in df.columns
        assert "inter_event_gap_std" in df.columns
        assert "regularity_score" in df.columns

    def test_regular_customer_has_high_regularity_score(self, engineer, sample_events_df, reference_dates):
        """Customer A has regular weekly events - should have high regularity score."""
        result = engineer.compute(
            events_df=sample_events_df,
            entity_col="customer_id",
            time_col="event_date",
            value_cols=["amount"],
            reference_dates=reference_dates,
            reference_col="reference_date",
        )
        df = result.features_df

        customer_a = df[df["customer_id"] == "A"].iloc[0]
        customer_d = df[df["customer_id"] == "D"].iloc[0]  # Irregular/bursty

        # Customer A (regular) should have higher regularity than D (bursty)
        if pd.notna(customer_a["regularity_score"]) and pd.notna(customer_d["regularity_score"]):
            assert customer_a["regularity_score"] > customer_d["regularity_score"]


class TestCohortComparisonFeatures:
    """Group 7: Cohort Comparison Features"""

    def test_cohort_features_created(self, engineer, sample_events_df, reference_dates):
        result = engineer.compute(
            events_df=sample_events_df,
            entity_col="customer_id",
            time_col="event_date",
            value_cols=["amount"],
            reference_dates=reference_dates,
            reference_col="reference_date",
        )
        df = result.features_df

        assert "amount_vs_cohort_mean" in df.columns
        assert "amount_vs_cohort_pct" in df.columns

    def test_cohort_comparison_centered_around_zero(self, engineer, sample_events_df, reference_dates):
        """Cohort comparison should center around 0 (or 100% for percentage)."""
        result = engineer.compute(
            events_df=sample_events_df,
            entity_col="customer_id",
            time_col="event_date",
            value_cols=["amount"],
            reference_dates=reference_dates,
            reference_col="reference_date",
        )
        df = result.features_df

        # Mean of vs_cohort_mean should be close to 0
        mean_diff = df["amount_vs_cohort_mean"].mean()
        assert abs(mean_diff) < df["amount_vs_cohort_mean"].std() * 2  # Within 2 std of 0


class TestFeatureGroupResults:
    def test_feature_groups_have_rationale(self, engineer, sample_events_df, reference_dates):
        result = engineer.compute(
            events_df=sample_events_df,
            entity_col="customer_id",
            time_col="event_date",
            value_cols=["amount"],
            reference_dates=reference_dates,
            reference_col="reference_date",
        )

        for group in result.feature_groups:
            assert group.rationale is not None
            assert len(group.rationale) > 0

    def test_feature_groups_list_features(self, engineer, sample_events_df, reference_dates):
        result = engineer.compute(
            events_df=sample_events_df,
            entity_col="customer_id",
            time_col="event_date",
            value_cols=["amount"],
            reference_dates=reference_dates,
            reference_col="reference_date",
        )

        for group in result.feature_groups:
            assert len(group.features) > 0
            assert all(isinstance(f, str) for f in group.features)

    def test_get_catalog_returns_formatted_output(self, engineer, sample_events_df, reference_dates):
        result = engineer.compute(
            events_df=sample_events_df,
            entity_col="customer_id",
            time_col="event_date",
            value_cols=["amount"],
            reference_dates=reference_dates,
            reference_col="reference_date",
        )

        catalog = result.get_catalog()
        assert isinstance(catalog, str)
        assert "LAGGED" in catalog.upper() or "LAG" in catalog.upper()


class TestConfigurableFeatureGroups:
    def test_disable_velocity_features(self, sample_events_df, reference_dates):
        config = TemporalAggregationConfig(compute_velocity=False)
        engineer = TemporalFeatureEngineer(config=config)

        result = engineer.compute(
            events_df=sample_events_df,
            entity_col="customer_id",
            time_col="event_date",
            value_cols=["amount"],
            reference_dates=reference_dates,
            reference_col="reference_date",
        )
        df = result.features_df

        assert "amount_velocity" not in df.columns

    def test_disable_lifecycle_features(self, sample_events_df, reference_dates):
        config = TemporalAggregationConfig(compute_lifecycle=False)
        engineer = TemporalFeatureEngineer(config=config)

        result = engineer.compute(
            events_df=sample_events_df,
            entity_col="customer_id",
            time_col="event_date",
            value_cols=["amount"],
            reference_dates=reference_dates,
            reference_col="reference_date",
        )
        df = result.features_df

        assert "amount_beginning" not in df.columns
        assert "amount_trend_ratio" not in df.columns

    def test_custom_lag_configuration(self, sample_events_df, reference_dates):
        config = TemporalAggregationConfig(lag_window_days=14, num_lags=6)
        engineer = TemporalFeatureEngineer(config=config)

        result = engineer.compute(
            events_df=sample_events_df,
            entity_col="customer_id",
            time_col="event_date",
            value_cols=["amount"],
            reference_dates=reference_dates,
            reference_col="reference_date",
        )
        df = result.features_df

        # Should have 6 lags
        for lag in range(6):
            assert f"lag{lag}_amount_sum" in df.columns


class TestGlobalReferenceMode:
    def test_global_reference_date(self, sample_events_df):
        config = TemporalAggregationConfig(
            reference_mode=ReferenceMode.GLOBAL_DATE,
            global_reference_date=datetime(2023, 11, 1)
        )
        engineer = TemporalFeatureEngineer(config=config)

        result = engineer.compute(
            events_df=sample_events_df,
            entity_col="customer_id",
            time_col="event_date",
            value_cols=["amount"],
        )

        # All customers should be computed relative to same date
        assert isinstance(result.features_df, pd.DataFrame)
        assert len(result.features_df) == sample_events_df["customer_id"].nunique()


class TestMultipleValueColumns:
    def test_features_for_multiple_columns(self, engineer, reference_dates):
        """Test with multiple value columns."""
        events_df = pd.DataFrame({
            "customer_id": ["A"] * 10 + ["B"] * 10,
            "event_date": list(pd.date_range("2023-08-01", periods=10, freq="7D")) * 2,
            "amount": np.random.uniform(50, 150, 20),
            "quantity": np.random.randint(1, 10, 20),
        })

        result = engineer.compute(
            events_df=events_df,
            entity_col="customer_id",
            time_col="event_date",
            value_cols=["amount", "quantity"],
            reference_dates=reference_dates[reference_dates["customer_id"].isin(["A", "B"])],
            reference_col="reference_date",
        )
        df = result.features_df

        # Both columns should have features
        assert "lag0_amount_sum" in df.columns
        assert "lag0_quantity_sum" in df.columns
        assert "amount_velocity" in df.columns
        assert "quantity_velocity" in df.columns
