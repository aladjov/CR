import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from customer_retention.stages.features import TemporalFeatureGenerator, ReferenceDateSource


class TestReferenceDateHandling:
    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({
            "custid": ["C001", "C002", "C003"],
            "created": pd.to_datetime(["2023-01-01", "2023-06-15", "2024-01-01"]),
            "firstorder": pd.to_datetime(["2023-01-15", "2023-06-20", "2024-01-10"]),
            "lastorder": pd.to_datetime(["2024-06-01", "2024-05-15", "2024-06-15"])
        })

    def test_reference_date_from_config(self, sample_df):
        reference_date = pd.Timestamp("2024-07-01")
        generator = TemporalFeatureGenerator(
            reference_date=reference_date,
            reference_date_source=ReferenceDateSource.CONFIG
        )
        result = generator.fit_transform(sample_df)

        assert generator.reference_date == reference_date

    def test_reference_date_from_max_date(self, sample_df):
        generator = TemporalFeatureGenerator(
            reference_date_source=ReferenceDateSource.MAX_DATE,
            date_column="lastorder"
        )
        result = generator.fit_transform(sample_df)

        expected = sample_df["lastorder"].max()
        assert generator.reference_date == expected

    def test_reference_date_from_column(self, sample_df):
        sample_df["observation_date"] = pd.to_datetime([
            "2024-07-01", "2024-07-15", "2024-08-01"
        ])
        generator = TemporalFeatureGenerator(
            reference_date_source=ReferenceDateSource.COLUMN,
            reference_date_column="observation_date"
        )
        result = generator.fit_transform(sample_df)

        assert "tenure_days" in result.columns
        assert result["tenure_days"].iloc[0] != result["tenure_days"].iloc[1]


class TestTenureFeatures:
    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({
            "custid": ["C001", "C002"],
            "created": pd.to_datetime(["2024-01-01", "2024-03-15"]),
            "firstorder": pd.to_datetime(["2024-01-10", "2024-03-20"]),
            "lastorder": pd.to_datetime(["2024-06-01", "2024-06-15"])
        })

    def test_tenure_days_calculation(self, sample_df):
        reference_date = pd.Timestamp("2024-07-01")
        generator = TemporalFeatureGenerator(
            reference_date=reference_date,
            created_column="created"
        )
        result = generator.fit_transform(sample_df)

        expected_tenure = (reference_date - sample_df["created"]).dt.days
        pd.testing.assert_series_equal(
            result["tenure_days"], expected_tenure, check_names=False
        )

    def test_account_age_months_calculation(self, sample_df):
        reference_date = pd.Timestamp("2024-07-01")
        generator = TemporalFeatureGenerator(
            reference_date=reference_date,
            created_column="created"
        )
        result = generator.fit_transform(sample_df)

        expected_months = result["tenure_days"] / 30.44
        pd.testing.assert_series_equal(
            result["account_age_months"], expected_months, check_names=False
        )


class TestRecencyFeatures:
    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({
            "custid": ["C001", "C002", "C003"],
            "created": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-01"]),
            "lastorder": pd.to_datetime(["2024-06-01", "2024-03-01", "2024-06-25"])
        })

    def test_days_since_last_order(self, sample_df):
        reference_date = pd.Timestamp("2024-07-01")
        generator = TemporalFeatureGenerator(
            reference_date=reference_date,
            last_order_column="lastorder"
        )
        result = generator.fit_transform(sample_df)

        expected = (reference_date - sample_df["lastorder"]).dt.days
        pd.testing.assert_series_equal(
            result["days_since_last_order"], expected, check_names=False
        )

    def test_days_since_last_order_is_positive(self, sample_df):
        reference_date = pd.Timestamp("2024-07-01")
        generator = TemporalFeatureGenerator(
            reference_date=reference_date,
            last_order_column="lastorder"
        )
        result = generator.fit_transform(sample_df)

        assert (result["days_since_last_order"] >= 0).all()


class TestActivationFeatures:
    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({
            "custid": ["C001", "C002"],
            "created": pd.to_datetime(["2024-01-01", "2024-03-01"]),
            "firstorder": pd.to_datetime(["2024-01-15", "2024-03-01"]),
            "lastorder": pd.to_datetime(["2024-06-01", "2024-06-15"])
        })

    def test_days_to_first_order(self, sample_df):
        generator = TemporalFeatureGenerator(
            reference_date=pd.Timestamp("2024-07-01"),
            created_column="created",
            first_order_column="firstorder"
        )
        result = generator.fit_transform(sample_df)

        expected = (sample_df["firstorder"] - sample_df["created"]).dt.days
        pd.testing.assert_series_equal(
            result["days_to_first_order"], expected, check_names=False
        )

    def test_days_to_first_order_same_day(self, sample_df):
        generator = TemporalFeatureGenerator(
            reference_date=pd.Timestamp("2024-07-01"),
            created_column="created",
            first_order_column="firstorder"
        )
        result = generator.fit_transform(sample_df)

        assert result["days_to_first_order"].iloc[1] == 0

    def test_active_period_days(self, sample_df):
        generator = TemporalFeatureGenerator(
            reference_date=pd.Timestamp("2024-07-01"),
            first_order_column="firstorder",
            last_order_column="lastorder"
        )
        result = generator.fit_transform(sample_df)

        expected = (sample_df["lastorder"] - sample_df["firstorder"]).dt.days
        pd.testing.assert_series_equal(
            result["active_period_days"], expected, check_names=False
        )


class TestTemporalValidation:
    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({
            "custid": ["C001"],
            "created": pd.to_datetime(["2024-01-01"]),
            "firstorder": pd.to_datetime(["2024-01-15"]),
            "lastorder": pd.to_datetime(["2024-06-01"])
        })

    def test_tenure_greater_than_active_period(self, sample_df):
        generator = TemporalFeatureGenerator(
            reference_date=pd.Timestamp("2024-07-01"),
            created_column="created",
            first_order_column="firstorder",
            last_order_column="lastorder"
        )
        result = generator.fit_transform(sample_df)

        assert (result["tenure_days"] >= result["active_period_days"]).all()

    def test_warns_on_negative_temporal_features(self, sample_df):
        future_reference = pd.Timestamp("2023-06-01")
        generator = TemporalFeatureGenerator(
            reference_date=future_reference,
            created_column="created"
        )

        with pytest.warns(UserWarning, match="negative"):
            generator.fit_transform(sample_df)


class TestNullHandling:
    def test_handles_null_dates(self):
        df = pd.DataFrame({
            "custid": ["C001", "C002", "C003"],
            "created": pd.to_datetime(["2024-01-01", "2024-01-01", None]),
            "lastorder": pd.to_datetime(["2024-06-01", None, "2024-06-01"])
        })
        generator = TemporalFeatureGenerator(
            reference_date=pd.Timestamp("2024-07-01"),
            created_column="created",
            last_order_column="lastorder"
        )
        result = generator.fit_transform(df)

        assert pd.isna(result["tenure_days"].iloc[2])
        assert pd.isna(result["days_since_last_order"].iloc[1])


class TestFeatureOutput:
    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({
            "custid": ["C001"],
            "created": pd.to_datetime(["2024-01-01"]),
            "firstorder": pd.to_datetime(["2024-01-15"]),
            "lastorder": pd.to_datetime(["2024-06-01"])
        })

    def test_all_temporal_features_created(self, sample_df):
        generator = TemporalFeatureGenerator(
            reference_date=pd.Timestamp("2024-07-01"),
            created_column="created",
            first_order_column="firstorder",
            last_order_column="lastorder"
        )
        result = generator.fit_transform(sample_df)

        expected_features = [
            "tenure_days", "account_age_months", "days_since_last_order",
            "days_to_first_order", "active_period_days"
        ]
        for feature in expected_features:
            assert feature in result.columns

    def test_generated_features_info(self, sample_df):
        generator = TemporalFeatureGenerator(
            reference_date=pd.Timestamp("2024-07-01"),
            created_column="created"
        )
        result = generator.fit_transform(sample_df)

        assert hasattr(generator, 'generated_features')
        assert len(generator.generated_features) > 0
