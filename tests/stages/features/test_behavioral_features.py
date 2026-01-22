import numpy as np
import pandas as pd
import pytest

from customer_retention.stages.features import BehavioralFeatureGenerator


class TestFrequencyFeatures:
    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({
            "custid": ["C001", "C002", "C003"],
            "tenure_months": [12.0, 6.0, 24.0],
            "total_orders": [24, 12, 36],
            "esent": [100, 50, 200],
            "total_visits": [120, 60, 300],
            "days_since_last_order": [10, 30, 5],
            "tenure_days": [365, 183, 730]
        })

    def test_order_frequency_calculation(self, sample_df):
        generator = BehavioralFeatureGenerator(
            tenure_months_column="tenure_months",
            total_orders_column="total_orders"
        )
        result = generator.fit_transform(sample_df)

        expected = sample_df["total_orders"] / sample_df["tenure_months"]
        pd.testing.assert_series_equal(
            result["order_frequency"], expected, check_names=False
        )

    def test_email_frequency_calculation(self, sample_df):
        generator = BehavioralFeatureGenerator(
            tenure_months_column="tenure_months",
            emails_sent_column="esent"
        )
        result = generator.fit_transform(sample_df)

        expected = sample_df["esent"] / sample_df["tenure_months"]
        pd.testing.assert_series_equal(
            result["email_frequency"], expected, check_names=False
        )

    def test_order_recency_ratio(self, sample_df):
        generator = BehavioralFeatureGenerator(
            tenure_days_column="tenure_days",
            days_since_last_order_column="days_since_last_order"
        )
        result = generator.fit_transform(sample_df)

        expected = sample_df["days_since_last_order"] / sample_df["tenure_days"]
        pd.testing.assert_series_equal(
            result["order_recency_ratio"], expected, check_names=False
        )


class TestEngagementFeatures:
    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({
            "custid": ["C001", "C002", "C003"],
            "eopenrate": [0.4, 0.6, 0.0],
            "eclickrate": [0.2, 0.3, 0.0]
        })

    def test_email_engagement_score(self, sample_df):
        generator = BehavioralFeatureGenerator(
            open_rate_column="eopenrate",
            click_rate_column="eclickrate"
        )
        result = generator.fit_transform(sample_df)

        expected = (sample_df["eopenrate"] + sample_df["eclickrate"]) / 2
        pd.testing.assert_series_equal(
            result["email_engagement_score"], expected, check_names=False
        )

    def test_click_to_open_rate_normal(self, sample_df):
        generator = BehavioralFeatureGenerator(
            open_rate_column="eopenrate",
            click_rate_column="eclickrate"
        )
        result = generator.fit_transform(sample_df)

        assert result["click_to_open_rate"].iloc[0] == pytest.approx(0.5)  # 0.2/0.4
        assert result["click_to_open_rate"].iloc[1] == pytest.approx(0.5)  # 0.3/0.6

    def test_click_to_open_rate_handles_zero_open_rate(self, sample_df):
        generator = BehavioralFeatureGenerator(
            open_rate_column="eopenrate",
            click_rate_column="eclickrate"
        )
        result = generator.fit_transform(sample_df)

        # When open rate is 0, click_to_open should be 0 (or NaN)
        assert result["click_to_open_rate"].iloc[2] == 0.0 or pd.isna(result["click_to_open_rate"].iloc[2])


class TestServiceAdoptionFeatures:
    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({
            "custid": ["C001", "C002", "C003"],
            "paperless": [1, 0, 1],
            "refill": [1, 1, 0],
            "doorstep": [0, 1, 1]
        })

    def test_service_adoption_score(self, sample_df):
        generator = BehavioralFeatureGenerator(
            service_columns=["paperless", "refill", "doorstep"]
        )
        result = generator.fit_transform(sample_df)

        expected = sample_df["paperless"] + sample_df["refill"] + sample_df["doorstep"]
        pd.testing.assert_series_equal(
            result["service_adoption_score"], expected.astype(float), check_names=False
        )

    def test_service_adoption_percentage(self, sample_df):
        generator = BehavioralFeatureGenerator(
            service_columns=["paperless", "refill", "doorstep"]
        )
        result = generator.fit_transform(sample_df)

        expected = (sample_df["paperless"] + sample_df["refill"] + sample_df["doorstep"]) / 3
        pd.testing.assert_series_equal(
            result["service_adoption_pct"], expected, check_names=False
        )


class TestRecencyBuckets:
    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({
            "custid": ["C001", "C002", "C003", "C004", "C005"],
            "days_since_last_order": [5, 15, 35, 65, 100]
        })

    def test_recency_bucket_creation(self, sample_df):
        generator = BehavioralFeatureGenerator(
            days_since_last_order_column="days_since_last_order",
            recency_bins=[0, 7, 30, 60, 90, float('inf')],
            recency_labels=["active", "recent", "warm", "cooling", "dormant"]
        )
        result = generator.fit_transform(sample_df)

        assert "recency_bucket" in result.columns
        assert result["recency_bucket"].iloc[0] == "active"
        assert result["recency_bucket"].iloc[1] == "recent"
        assert result["recency_bucket"].iloc[2] == "warm"
        assert result["recency_bucket"].iloc[3] == "cooling"
        assert result["recency_bucket"].iloc[4] == "dormant"

    def test_recency_bucket_default_bins(self, sample_df):
        generator = BehavioralFeatureGenerator(
            days_since_last_order_column="days_since_last_order"
        )
        result = generator.fit_transform(sample_df)

        assert "recency_bucket" in result.columns


class TestNullHandling:
    def test_handles_null_tenure(self):
        df = pd.DataFrame({
            "custid": ["C001", "C002"],
            "tenure_months": [12.0, None],
            "total_orders": [24, 12]
        })
        generator = BehavioralFeatureGenerator(
            tenure_months_column="tenure_months",
            total_orders_column="total_orders"
        )
        result = generator.fit_transform(df)

        assert not pd.isna(result["order_frequency"].iloc[0])
        assert pd.isna(result["order_frequency"].iloc[1])

    def test_handles_zero_tenure(self):
        df = pd.DataFrame({
            "custid": ["C001", "C002"],
            "tenure_months": [12.0, 0.0],
            "total_orders": [24, 0]
        })
        generator = BehavioralFeatureGenerator(
            tenure_months_column="tenure_months",
            total_orders_column="total_orders"
        )
        result = generator.fit_transform(df)

        # Zero tenure should result in inf or nan for frequency
        assert result["order_frequency"].iloc[0] == pytest.approx(2.0)
        assert pd.isna(result["order_frequency"].iloc[1]) or np.isinf(result["order_frequency"].iloc[1])


class TestFitTransformSeparation:
    def test_fit_then_transform(self):
        train = pd.DataFrame({
            "custid": ["C001", "C002"],
            "tenure_months": [12.0, 6.0],
            "total_orders": [24, 12]
        })
        test = pd.DataFrame({
            "custid": ["C003"],
            "tenure_months": [3.0],
            "total_orders": [9]
        })

        generator = BehavioralFeatureGenerator(
            tenure_months_column="tenure_months",
            total_orders_column="total_orders"
        )
        generator.fit(train)
        result = generator.transform(test)

        assert result["order_frequency"].iloc[0] == pytest.approx(3.0)


class TestGeneratedFeaturesInfo:
    def test_generated_features_tracked(self):
        df = pd.DataFrame({
            "custid": ["C001"],
            "tenure_months": [12.0],
            "total_orders": [24],
            "eopenrate": [0.4],
            "eclickrate": [0.2]
        })
        generator = BehavioralFeatureGenerator(
            tenure_months_column="tenure_months",
            total_orders_column="total_orders",
            open_rate_column="eopenrate",
            click_rate_column="eclickrate"
        )
        result = generator.fit_transform(df)

        assert hasattr(generator, 'generated_features')
        assert "order_frequency" in generator.generated_features
        assert "email_engagement_score" in generator.generated_features
