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


class TestTransformBeforeFit:
    def test_transform_before_fit_raises_value_error(self):
        df = pd.DataFrame({
            "custid": ["C001", "C002"],
            "tenure_months": [12.0, 6.0],
            "total_orders": [24, 12]
        })
        generator = BehavioralFeatureGenerator(
            tenure_months_column="tenure_months",
            total_orders_column="total_orders"
        )
        with pytest.raises(ValueError, match="not fitted"):
            generator.transform(df)


class TestVisitFrequency:
    def test_visit_frequency_calculation(self):
        df = pd.DataFrame({
            "custid": ["C001", "C002", "C003"],
            "tenure_months": [12.0, 6.0, 24.0],
            "total_visits": [120, 60, 300],
        })
        generator = BehavioralFeatureGenerator(
            tenure_months_column="tenure_months",
            total_visits_column="total_visits"
        )
        result = generator.fit_transform(df)

        assert "visit_frequency" in result.columns
        expected = df["total_visits"] / df["tenure_months"]
        pd.testing.assert_series_equal(
            result["visit_frequency"], expected, check_names=False
        )
        assert "visit_frequency" in generator.generated_features

    def test_visit_frequency_with_zero_tenure(self):
        df = pd.DataFrame({
            "custid": ["C001"],
            "tenure_months": [0.0],
            "total_visits": [10],
        })
        generator = BehavioralFeatureGenerator(
            tenure_months_column="tenure_months",
            total_visits_column="total_visits"
        )
        result = generator.fit_transform(df)

        assert "visit_frequency" in result.columns
        assert pd.isna(result["visit_frequency"].iloc[0]) or np.isinf(result["visit_frequency"].iloc[0])


class TestEmailFrequencySeparate:
    def test_email_frequency_in_generated_features(self):
        df = pd.DataFrame({
            "custid": ["C001", "C002"],
            "tenure_months": [10.0, 5.0],
            "esent": [50, 25],
        })
        generator = BehavioralFeatureGenerator(
            tenure_months_column="tenure_months",
            emails_sent_column="esent"
        )
        result = generator.fit_transform(df)

        assert "email_frequency" in result.columns
        assert "email_frequency" in generator.generated_features
        assert result["email_frequency"].iloc[0] == pytest.approx(5.0)
        assert result["email_frequency"].iloc[1] == pytest.approx(5.0)


class TestOrderRecencyRatioSeparate:
    def test_order_recency_ratio_in_generated_features(self):
        df = pd.DataFrame({
            "custid": ["C001", "C002"],
            "tenure_days": [365, 180],
            "days_since_last_order": [10, 30],
        })
        generator = BehavioralFeatureGenerator(
            tenure_days_column="tenure_days",
            days_since_last_order_column="days_since_last_order"
        )
        result = generator.fit_transform(df)

        assert "order_recency_ratio" in result.columns
        assert "order_recency_ratio" in generator.generated_features
        assert result["order_recency_ratio"].iloc[0] == pytest.approx(10.0 / 365.0)
        assert result["order_recency_ratio"].iloc[1] == pytest.approx(30.0 / 180.0)


class TestEngagementFeaturesSeparate:
    def test_engagement_features_generated(self):
        df = pd.DataFrame({
            "custid": ["C001", "C002"],
            "eopenrate": [0.5, 0.8],
            "eclickrate": [0.1, 0.4],
        })
        generator = BehavioralFeatureGenerator(
            open_rate_column="eopenrate",
            click_rate_column="eclickrate"
        )
        result = generator.fit_transform(df)

        assert "email_engagement_score" in result.columns
        assert "click_to_open_rate" in result.columns
        assert "email_engagement_score" in generator.generated_features
        assert "click_to_open_rate" in generator.generated_features

        # email_engagement_score = (open_rate + click_rate) / 2
        assert result["email_engagement_score"].iloc[0] == pytest.approx(0.3)
        assert result["email_engagement_score"].iloc[1] == pytest.approx(0.6)

        # click_to_open_rate = click_rate / open_rate
        assert result["click_to_open_rate"].iloc[0] == pytest.approx(0.2)
        assert result["click_to_open_rate"].iloc[1] == pytest.approx(0.5)


class TestServiceAdoptionSeparate:
    def test_service_adoption_score_and_pct(self):
        df = pd.DataFrame({
            "custid": ["C001", "C002", "C003"],
            "svc_a": [1, 0, 1],
            "svc_b": [1, 1, 0],
            "svc_c": [0, 0, 1],
            "svc_d": [1, 1, 1],
        })
        generator = BehavioralFeatureGenerator(
            service_columns=["svc_a", "svc_b", "svc_c", "svc_d"]
        )
        result = generator.fit_transform(df)

        assert "service_adoption_score" in result.columns
        assert "service_adoption_pct" in result.columns
        assert "service_adoption_score" in generator.generated_features
        assert "service_adoption_pct" in generator.generated_features

        # C001: 3 services adopted out of 4
        assert result["service_adoption_score"].iloc[0] == pytest.approx(3.0)
        assert result["service_adoption_pct"].iloc[0] == pytest.approx(0.75)

        # C002: 2 services adopted out of 4
        assert result["service_adoption_score"].iloc[1] == pytest.approx(2.0)
        assert result["service_adoption_pct"].iloc[1] == pytest.approx(0.5)

    def test_service_columns_partially_missing(self):
        df = pd.DataFrame({
            "custid": ["C001"],
            "svc_a": [1],
        })
        generator = BehavioralFeatureGenerator(
            service_columns=["svc_a", "svc_nonexistent"]
        )
        result = generator.fit_transform(df)

        # Should only use existing columns
        assert "service_adoption_score" in result.columns
        assert result["service_adoption_score"].iloc[0] == pytest.approx(1.0)
        # pct denominator is number of existing columns (1)
        assert result["service_adoption_pct"].iloc[0] == pytest.approx(1.0)


class TestRecencyBucketSeparate:
    def test_recency_bucket_in_generated_features(self):
        df = pd.DataFrame({
            "custid": ["C001", "C002", "C003"],
            "days_since_last_order": [3, 20, 50],
        })
        generator = BehavioralFeatureGenerator(
            days_since_last_order_column="days_since_last_order"
        )
        result = generator.fit_transform(df)

        assert "recency_bucket" in result.columns
        assert "recency_bucket" in generator.generated_features
        assert result["recency_bucket"].iloc[0] == "active"
        assert result["recency_bucket"].iloc[1] == "recent"
        assert result["recency_bucket"].iloc[2] == "warm"

    def test_recency_bucket_custom_bins(self):
        df = pd.DataFrame({
            "custid": ["C001", "C002"],
            "days_since_last_order": [5, 15],
        })
        generator = BehavioralFeatureGenerator(
            days_since_last_order_column="days_since_last_order",
            recency_bins=[0, 10, float('inf')],
            recency_labels=["new", "old"]
        )
        result = generator.fit_transform(df)

        assert result["recency_bucket"].iloc[0] == "new"
        assert result["recency_bucket"].iloc[1] == "old"


class TestPITValidation:
    def test_pit_validation_detects_future_timestamps(self):
        df = pd.DataFrame({
            "custid": ["C001", "C002"],
            "feature_timestamp": pd.to_datetime(["2024-01-15", "2024-01-15"]),
            "event_date": pd.to_datetime(["2024-01-10", "2024-02-01"]),
        })
        generator = BehavioralFeatureGenerator(
            enforce_point_in_time=True,
            feature_timestamp_column="feature_timestamp"
        )
        generator.fit(df)
        generator.transform(df)

        assert len(generator.pit_warnings) > 0
        assert any("event_date" in w for w in generator.pit_warnings)
        assert any("1 rows" in w for w in generator.pit_warnings)

    def test_pit_validation_no_warnings_when_compliant(self):
        df = pd.DataFrame({
            "custid": ["C001", "C002"],
            "feature_timestamp": pd.to_datetime(["2024-06-01", "2024-06-01"]),
            "event_date": pd.to_datetime(["2024-01-10", "2024-03-01"]),
        })
        generator = BehavioralFeatureGenerator(
            enforce_point_in_time=True,
            feature_timestamp_column="feature_timestamp"
        )
        generator.fit(df)
        generator.transform(df)

        assert len(generator.pit_warnings) == 0

    def test_pit_validation_skipped_when_no_timestamp_column(self):
        df = pd.DataFrame({
            "custid": ["C001"],
            "some_col": [1.0],
        })
        generator = BehavioralFeatureGenerator(
            enforce_point_in_time=True,
            feature_timestamp_column="feature_timestamp"
        )
        generator.fit(df)
        # Should not raise even though feature_timestamp is missing
        result = generator.transform(df)
        assert len(generator.pit_warnings) == 0

    def test_pit_validation_not_run_when_not_enforced(self):
        df = pd.DataFrame({
            "custid": ["C001"],
            "feature_timestamp": pd.to_datetime(["2024-01-01"]),
            "event_date": pd.to_datetime(["2025-12-31"]),
        })
        generator = BehavioralFeatureGenerator(
            enforce_point_in_time=False,
            feature_timestamp_column="feature_timestamp"
        )
        generator.fit(df)
        generator.transform(df)

        # PIT validation not enforced, so no warnings collected
        assert len(generator.pit_warnings) == 0


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
