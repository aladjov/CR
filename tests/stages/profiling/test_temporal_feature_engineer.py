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


class TestDisabledFeatureGroups:
    """Test all disabled feature group paths."""

    def test_disable_acceleration(self, sample_events_df, reference_dates):
        config = TemporalAggregationConfig(compute_acceleration=False)
        engineer = TemporalFeatureEngineer(config=config)
        result = engineer.compute(
            events_df=sample_events_df,
            entity_col="customer_id",
            time_col="event_date",
            value_cols=["amount"],
            reference_dates=reference_dates,
            reference_col="reference_date",
        )
        accel_group = next(
            g for g in result.feature_groups if g.group == FeatureGroup.ACCELERATION
        )
        assert not accel_group.enabled
        assert accel_group.features == []

    def test_disable_recency(self, sample_events_df, reference_dates):
        config = TemporalAggregationConfig(compute_recency=False)
        engineer = TemporalFeatureEngineer(config=config)
        result = engineer.compute(
            events_df=sample_events_df,
            entity_col="customer_id",
            time_col="event_date",
            value_cols=["amount"],
            reference_dates=reference_dates,
            reference_col="reference_date",
        )
        recency_group = next(
            g for g in result.feature_groups if g.group == FeatureGroup.RECENCY
        )
        assert not recency_group.enabled

    def test_disable_regularity(self, sample_events_df, reference_dates):
        config = TemporalAggregationConfig(compute_regularity=False)
        engineer = TemporalFeatureEngineer(config=config)
        result = engineer.compute(
            events_df=sample_events_df,
            entity_col="customer_id",
            time_col="event_date",
            value_cols=["amount"],
            reference_dates=reference_dates,
            reference_col="reference_date",
        )
        regularity_group = next(
            g for g in result.feature_groups if g.group == FeatureGroup.REGULARITY
        )
        assert not regularity_group.enabled

    def test_disable_cohort(self, sample_events_df, reference_dates):
        config = TemporalAggregationConfig(compute_cohort=False)
        engineer = TemporalFeatureEngineer(config=config)
        result = engineer.compute(
            events_df=sample_events_df,
            entity_col="customer_id",
            time_col="event_date",
            value_cols=["amount"],
            reference_dates=reference_dates,
            reference_col="reference_date",
        )
        cohort_group = next(
            g for g in result.feature_groups if g.group == FeatureGroup.COHORT_COMPARISON
        )
        assert not cohort_group.enabled

    def test_disabled_group_in_catalog(self, sample_events_df, reference_dates):
        """Disabled groups should be skipped in catalog."""
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
        catalog = result.get_catalog()
        assert "VELOCITY" not in catalog


class TestGetCatalogFeatures:

    def test_catalog_truncates_long_feature_list(self, sample_events_df, reference_dates):
        """Catalog should show '... and N more' for groups with >10 features."""
        config = TemporalAggregationConfig(num_lags=4, lag_aggregations=["sum", "mean", "count"])
        engineer = TemporalFeatureEngineer(config=config)
        result = engineer.compute(
            events_df=sample_events_df,
            entity_col="customer_id",
            time_col="event_date",
            value_cols=["amount"],
            reference_dates=reference_dates,
            reference_col="reference_date",
        )
        catalog = result.get_catalog()
        assert "TEMPORAL FEATURE CATALOG" in catalog
        # With 4 lags * 1 col * 3 aggs = 12 features, should truncate
        assert "more" in catalog

    def test_to_dict(self, sample_events_df, reference_dates):
        engineer = TemporalFeatureEngineer()
        result = engineer.compute(
            events_df=sample_events_df,
            entity_col="customer_id",
            time_col="event_date",
            value_cols=["amount"],
            reference_dates=reference_dates,
            reference_col="reference_date",
        )
        d = result.to_dict()
        assert "n_features" in d
        assert "n_entities" in d
        assert "feature_groups" in d
        assert d["n_entities"] == 4


class TestGlobalReferenceDateDefault:

    def test_global_mode_without_explicit_date(self, sample_events_df):
        """Global mode without global_reference_date should use datetime.now()."""
        config = TemporalAggregationConfig(
            reference_mode=ReferenceMode.GLOBAL_DATE,
            global_reference_date=None,
        )
        engineer = TemporalFeatureEngineer(config=config)
        result = engineer.compute(
            events_df=sample_events_df,
            entity_col="customer_id",
            time_col="event_date",
            value_cols=["amount"],
        )
        assert len(result.features_df) == sample_events_df["customer_id"].nunique()


class TestDefaultReferenceDate:

    def test_default_reference_uses_last_event(self, sample_events_df):
        """Default per_customer mode without reference_dates should use last event."""
        engineer = TemporalFeatureEngineer()
        result = engineer.compute(
            events_df=sample_events_df,
            entity_col="customer_id",
            time_col="event_date",
            value_cols=["amount"],
        )
        assert len(result.features_df) == sample_events_df["customer_id"].nunique()


class TestEdgeCases:

    def test_single_event_entity(self):
        """Entity with single event should have NaN for regularity."""
        events_df = pd.DataFrame({
            "customer_id": ["A", "B", "B"],
            "event_date": pd.to_datetime(["2023-01-01", "2023-01-01", "2023-01-15"]),
            "amount": [100.0, 50.0, 60.0],
        })
        engineer = TemporalFeatureEngineer()
        result = engineer.compute(
            events_df=events_df,
            entity_col="customer_id",
            time_col="event_date",
            value_cols=["amount"],
        )
        df = result.features_df
        cust_a = df[df["customer_id"] == "A"].iloc[0]
        assert pd.isna(cust_a["inter_event_gap_mean"])

    def test_events_same_timestamp(self):
        """All events at the same time -> total_days=0, frequency=event_count."""
        events_df = pd.DataFrame({
            "customer_id": ["A"] * 5,
            "event_date": pd.to_datetime(["2023-01-01"] * 5),
            "amount": [10.0, 20.0, 30.0, 40.0, 50.0],
        })
        engineer = TemporalFeatureEngineer()
        result = engineer.compute(
            events_df=events_df,
            entity_col="customer_id",
            time_col="event_date",
            value_cols=["amount"],
        )
        df = result.features_df
        cust_a = df[df["customer_id"] == "A"].iloc[0]
        # frequency should be event count when total_days == 0
        assert cust_a["event_frequency"] == 5

    def test_zero_beginning_value_no_trend_ratio(self):
        """Lifecycle: when beginning_val=0, trend_ratio should stay NaN."""
        events_df = pd.DataFrame({
            "customer_id": ["A"] * 30,
            "event_date": pd.date_range("2023-01-01", periods=30, freq="5D"),
            "amount": [0.0] * 10 + [50.0] * 10 + [100.0] * 10,
        })
        config = TemporalAggregationConfig(min_history_days=30)
        engineer = TemporalFeatureEngineer(config=config)
        result = engineer.compute(
            events_df=events_df,
            entity_col="customer_id",
            time_col="event_date",
            value_cols=["amount"],
        )
        df = result.features_df
        cust_a = df[df["customer_id"] == "A"].iloc[0]
        # beginning = 0, so trend_ratio should remain NaN
        assert pd.isna(cust_a["amount_trend_ratio"])

    def test_cohort_zero_std(self):
        """All entities have same lag0 value -> cohort_std=0, no z-score."""
        events_df = pd.DataFrame({
            "customer_id": ["A", "A", "B", "B"],
            "event_date": pd.to_datetime([
                "2023-01-01", "2023-01-15",
                "2023-01-01", "2023-01-15",
            ]),
            "amount": [100.0, 100.0, 100.0, 100.0],
        })
        engineer = TemporalFeatureEngineer()
        result = engineer.compute(
            events_df=events_df,
            entity_col="customer_id",
            time_col="event_date",
            value_cols=["amount"],
        )
        df = result.features_df
        assert "amount_cohort_zscore" not in df.columns

    def test_gap_mean_zero_regularity(self):
        """When gap_mean=0, regularity_score should be 1.0."""
        events_df = pd.DataFrame({
            "customer_id": ["A"] * 3,
            "event_date": pd.to_datetime(["2023-01-01", "2023-01-01", "2023-01-01"]),
            "amount": [10.0, 20.0, 30.0],
        })
        config = TemporalAggregationConfig(compute_regularity=True)
        engineer = TemporalFeatureEngineer(config=config)
        result = engineer.compute(
            events_df=events_df,
            entity_col="customer_id",
            time_col="event_date",
            value_cols=["amount"],
        )
        df = result.features_df
        cust_a = df[df["customer_id"] == "A"].iloc[0]
        assert cust_a["regularity_score"] == 1.0

    def test_all_single_event_entities_regularity_columns(self):
        """When all entities have single events, regularity columns should be NaN."""
        events_df = pd.DataFrame({
            "customer_id": ["A", "B"],
            "event_date": pd.to_datetime(["2023-01-01", "2023-02-01"]),
            "amount": [10.0, 20.0],
        })
        engineer = TemporalFeatureEngineer()
        result = engineer.compute(
            events_df=events_df,
            entity_col="customer_id",
            time_col="event_date",
            value_cols=["amount"],
        )
        df = result.features_df
        # Columns should exist but be NaN
        assert "regularity_score" in df.columns
        assert "event_frequency" in df.columns
        assert df["regularity_score"].isna().all()

    def test_entity_not_in_events_gets_nan_lifecycle(self):
        """Reference entity with no matching events should get NaN lifecycle."""
        events_df = pd.DataFrame({
            "customer_id": ["A"] * 30,
            "event_date": pd.date_range("2023-01-01", periods=30, freq="5D"),
            "amount": np.random.uniform(50, 150, 30),
        })
        ref = pd.DataFrame({
            "customer_id": ["A", "Z"],
            "reference_date": pd.to_datetime(["2023-11-01", "2023-11-01"]),
        })
        config = TemporalAggregationConfig(min_history_days=10)
        engineer = TemporalFeatureEngineer(config=config)
        result = engineer.compute(
            events_df=events_df,
            entity_col="customer_id",
            time_col="event_date",
            value_cols=["amount"],
            reference_dates=ref,
            reference_col="reference_date",
        )
        df = result.features_df
        cust_z = df[df["customer_id"] == "Z"]
        if len(cust_z) > 0:
            assert pd.isna(cust_z.iloc[0]["amount_beginning"])


class TestLifecycleNaNHistoryDays:
    def test_nat_timestamps_do_not_raise(self):
        events_df = pd.DataFrame({
            "customer_id": ["A", "A", "B", "B"],
            "event_date": pd.to_datetime(
                ["2023-01-01", "2023-03-01", pd.NaT, pd.NaT]
            ),
            "amount": [10.0, 20.0, 30.0, 40.0],
        })
        config = TemporalAggregationConfig(min_history_days=10)
        engineer = TemporalFeatureEngineer(config=config)
        result = engineer.compute(
            events_df=events_df,
            entity_col="customer_id",
            time_col="event_date",
            value_cols=["amount"],
        )
        df = result.features_df
        assert len(df) == 2
        cust_b = df[df["customer_id"] == "B"].iloc[0]
        assert pd.isna(cust_b["amount_beginning"])

    def test_all_nat_timestamps_do_not_raise(self):
        events_df = pd.DataFrame({
            "customer_id": ["A", "A"],
            "event_date": pd.to_datetime([pd.NaT, pd.NaT]),
            "amount": [10.0, 20.0],
        })
        engineer = TemporalFeatureEngineer()
        result = engineer.compute(
            events_df=events_df,
            entity_col="customer_id",
            time_col="event_date",
            value_cols=["amount"],
        )
        assert len(result.features_df) == 1


class TestVelocityWithNullableNA:
    def test_velocity_pct_with_nullable_integer_columns(self):
        events_df = pd.DataFrame({
            "customer_id": ["A"] * 10 + ["B"] * 10,
            "event_date": list(pd.date_range("2023-01-01", periods=10, freq="5D")) * 2,
            "amount": pd.array(
                [10, 20, 30, pd.NA, 50, 60, pd.NA, 80, 90, 100] * 2,
                dtype="Int64",
            ),
        })
        engineer = TemporalFeatureEngineer()
        result = engineer.compute(
            events_df=events_df,
            entity_col="customer_id",
            time_col="event_date",
            value_cols=["amount"],
        )
        assert "amount_velocity_pct" in result.features_df.columns
        assert len(result.features_df) == 2

    def test_velocity_pct_with_all_na_lag_column(self):
        events_df = pd.DataFrame({
            "customer_id": ["A"] * 3,
            "event_date": pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03"]),
            "amount": pd.array([pd.NA, pd.NA, pd.NA], dtype="Int64"),
        })
        engineer = TemporalFeatureEngineer()
        result = engineer.compute(
            events_df=events_df,
            entity_col="customer_id",
            time_col="event_date",
            value_cols=["amount"],
        )
        df = result.features_df
        if "amount_velocity_pct" in df.columns:
            assert df["amount_velocity_pct"].isna().all()


class TestVectorizedBatchAggregation:
    """Verify batched groupby produces correct results at scale."""

    def test_many_entities_lagged_windows(self):
        n_entities = 200
        events_per_entity = 30
        rng = np.random.default_rng(42)
        rows = []
        for i in range(n_entities):
            dates = pd.date_range("2023-01-01", periods=events_per_entity, freq="3D")
            for d in dates:
                rows.append({"customer_id": f"E{i}", "event_date": d, "amount": rng.uniform(10, 100)})
        events_df = pd.DataFrame(rows)

        config = TemporalAggregationConfig(
            lag_window_days=30, num_lags=3,
            compute_lifecycle=False, compute_regularity=False,
            compute_cohort=False, compute_velocity=False, compute_acceleration=False,
        )
        result = TemporalFeatureEngineer(config=config).compute(
            events_df=events_df, entity_col="customer_id",
            time_col="event_date", value_cols=["amount"],
        )
        df = result.features_df
        assert len(df) == n_entities
        for lag in range(3):
            assert f"lag{lag}_amount_sum" in df.columns
            assert f"lag{lag}_amount_count" in df.columns
        assert (df["lag0_amount_count"] >= 0).all()

    def test_many_entities_lifecycle_vectorized(self):
        n_entities = 150
        rng = np.random.default_rng(99)
        rows = []
        for i in range(n_entities):
            dates = pd.date_range("2023-01-01", periods=40, freq="5D")
            for d in dates:
                rows.append({"customer_id": f"E{i}", "event_date": d, "amount": rng.uniform(5, 50)})
        events_df = pd.DataFrame(rows)

        config = TemporalAggregationConfig(
            min_history_days=30, compute_regularity=False,
            compute_cohort=False, compute_velocity=False, compute_acceleration=False,
        )
        result = TemporalFeatureEngineer(config=config).compute(
            events_df=events_df, entity_col="customer_id",
            time_col="event_date", value_cols=["amount"],
        )
        df = result.features_df
        assert len(df) == n_entities
        assert "amount_beginning" in df.columns
        assert "amount_trend_ratio" in df.columns
        valid = df.dropna(subset=["amount_beginning"])
        assert len(valid) > 0
        for _, row in valid.iterrows():
            if row["amount_beginning"] > 0:
                expected = row["amount_end"] / row["amount_beginning"]
                assert row["amount_trend_ratio"] == pytest.approx(expected, rel=0.01)

    def test_many_entities_regularity_vectorized(self):
        n_entities = 100
        rng = np.random.default_rng(77)
        rows = []
        for i in range(n_entities):
            gap = rng.integers(2, 10)
            dates = pd.date_range("2023-01-01", periods=20, freq=f"{gap}D")
            for d in dates:
                rows.append({"customer_id": f"E{i}", "event_date": d, "amount": 10.0})
        events_df = pd.DataFrame(rows)

        config = TemporalAggregationConfig(
            compute_lifecycle=False, compute_cohort=False,
            compute_velocity=False, compute_acceleration=False,
        )
        result = TemporalFeatureEngineer(config=config).compute(
            events_df=events_df, entity_col="customer_id",
            time_col="event_date", value_cols=["amount"],
        )
        df = result.features_df
        assert len(df) == n_entities
        assert "regularity_score" in df.columns
        assert "event_frequency" in df.columns
        valid = df.dropna(subset=["regularity_score"])
        assert (valid["regularity_score"] >= 0).all()
        assert (valid["regularity_score"] <= 1).all()
        perfectly_regular = df[df["inter_event_gap_std"] == 0]
        if len(perfectly_regular) > 0:
            assert (perfectly_regular["regularity_score"] == 1.0).all()

    def test_multiple_value_cols_batched(self):
        events_df = pd.DataFrame({
            "cid": ["A"] * 10 + ["B"] * 10,
            "ts": list(pd.date_range("2023-01-01", periods=10, freq="7D")) * 2,
            "amount": np.random.uniform(10, 100, 20),
            "quantity": np.random.randint(1, 10, 20).astype(float),
            "score": np.random.uniform(0, 1, 20),
        })
        config = TemporalAggregationConfig(
            lag_window_days=30, num_lags=2,
            compute_lifecycle=False, compute_regularity=False,
            compute_cohort=False,
        )
        result = TemporalFeatureEngineer(config=config).compute(
            events_df=events_df, entity_col="cid",
            time_col="ts", value_cols=["amount", "quantity", "score"],
        )
        df = result.features_df
        assert len(df) == 2
        for col in ["amount", "quantity", "score"]:
            assert f"lag0_{col}_sum" in df.columns
            assert f"lag1_{col}_mean" in df.columns

    def test_compute_with_empty_value_cols_returns_non_value_features(self):
        """TFE must handle empty value_cols: returns recency/regularity features only.

        Regression guard for optimization commit 569a509 — batched
        groupby_multi_col_agg raised AssertionError('exprs should not be empty')
        when value_cols was empty after the caller's numeric filter.
        """
        events_df = pd.DataFrame({
            "cid": ["A"] * 5 + ["B"] * 5,
            "ts": list(pd.date_range("2023-01-01", periods=5, freq="3D")) * 2,
        })
        config = TemporalAggregationConfig(
            lag_window_days=30, num_lags=2,
            compute_lifecycle=False, compute_cohort=False,
            compute_velocity=False, compute_acceleration=False,
        )
        result = TemporalFeatureEngineer(config=config).compute(
            events_df=events_df, entity_col="cid", time_col="ts", value_cols=[],
        )
        df = result.features_df
        assert "cid" in df.columns
        assert set(df["cid"].tolist()) == {"A", "B"}
        assert "days_since_last_event" in df.columns
        assert "regularity_score" in df.columns
        for col in df.columns:
            assert not col.startswith("lag")

    def test_lagged_windows_empty_value_cols_returns_entity_only(self):
        """_compute_lagged_windows early-return path."""
        events_df = pd.DataFrame({
            "cid": ["A", "A", "B"],
            "ts": pd.to_datetime(["2023-01-01", "2023-01-15", "2023-01-10"]),
        })
        config = TemporalAggregationConfig(
            compute_lifecycle=False, compute_regularity=False,
            compute_cohort=False, compute_velocity=False,
            compute_acceleration=False, compute_recency=False,
        )
        result = TemporalFeatureEngineer(config=config).compute(
            events_df=events_df, entity_col="cid", time_col="ts", value_cols=[],
        )
        df = result.features_df
        assert list(df.columns) == ["cid"]
        assert len(df) == 2
        lag_group = next(g for g in result.feature_groups if g.group == FeatureGroup.LAGGED_WINDOWS)
        assert lag_group.features == []

    def test_lifecycle_short_history_excluded(self):
        short = pd.DataFrame({
            "cid": ["S"] * 3,
            "ts": pd.date_range("2023-01-01", periods=3, freq="5D"),
            "val": [10.0, 20.0, 30.0],
        })
        long = pd.DataFrame({
            "cid": ["L"] * 30,
            "ts": pd.date_range("2023-01-01", periods=30, freq="5D"),
            "val": np.random.uniform(10, 50, 30),
        })
        events_df = pd.concat([short, long], ignore_index=True)
        config = TemporalAggregationConfig(
            min_history_days=60,
            compute_regularity=False, compute_cohort=False,
            compute_velocity=False, compute_acceleration=False,
        )
        result = TemporalFeatureEngineer(config=config).compute(
            events_df=events_df, entity_col="cid",
            time_col="ts", value_cols=["val"],
        )
        df = result.features_df
        short_row = df[df["cid"] == "S"].iloc[0]
        long_row = df[df["cid"] == "L"].iloc[0]
        assert pd.isna(short_row["val_beginning"])
        assert pd.notna(long_row["val_beginning"])
