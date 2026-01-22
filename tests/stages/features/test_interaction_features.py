import numpy as np
import pandas as pd
import pytest

from customer_retention.stages.features import InteractionFeatureGenerator


class TestFeatureCombinations:
    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({
            "custid": ["C001", "C002", "C003"],
            "avgorder": [50.0, 100.0, 75.0],
            "order_frequency": [2.0, 1.0, 3.0],
            "email_engagement_score": [0.5, 0.3, 0.7],
            "tenure_months": [12.0, 6.0, 24.0],
            "days_since_last_order": [10, 30, 5],
            "service_adoption_score": [2, 1, 3]
        })

    def test_value_x_frequency(self, sample_df):
        generator = InteractionFeatureGenerator(
            combinations=[
                ("avgorder", "order_frequency", "value_x_frequency", "multiply")
            ]
        )
        result = generator.fit_transform(sample_df)

        expected = sample_df["avgorder"] * sample_df["order_frequency"]
        pd.testing.assert_series_equal(
            result["value_x_frequency"], expected, check_names=False
        )

    def test_engagement_x_tenure(self, sample_df):
        generator = InteractionFeatureGenerator(
            combinations=[
                ("email_engagement_score", "tenure_months", "engagement_x_tenure", "multiply")
            ]
        )
        result = generator.fit_transform(sample_df)

        expected = sample_df["email_engagement_score"] * sample_df["tenure_months"]
        pd.testing.assert_series_equal(
            result["engagement_x_tenure"], expected, check_names=False
        )

    def test_recency_x_value(self, sample_df):
        generator = InteractionFeatureGenerator(
            combinations=[
                ("avgorder", "days_since_last_order", "recency_x_value", "divide")
            ]
        )
        result = generator.fit_transform(sample_df)

        expected = sample_df["avgorder"] / sample_df["days_since_last_order"]
        pd.testing.assert_series_equal(
            result["recency_x_value"], expected, check_names=False
        )

    def test_multiple_combinations(self, sample_df):
        generator = InteractionFeatureGenerator(
            combinations=[
                ("avgorder", "order_frequency", "value_x_frequency", "multiply"),
                ("email_engagement_score", "tenure_months", "engagement_x_tenure", "multiply"),
                ("service_adoption_score", "tenure_months", "adoption_x_tenure", "multiply")
            ]
        )
        result = generator.fit_transform(sample_df)

        assert "value_x_frequency" in result.columns
        assert "engagement_x_tenure" in result.columns
        assert "adoption_x_tenure" in result.columns


class TestRatioFeatures:
    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({
            "custid": ["C001", "C002", "C003"],
            "total_orders": [24, 12, 36],
            "esent": [100, 50, 200],
            "total_revenue": [1200.0, 800.0, 2700.0],
            "total_visits": [120, 60, 300]
        })

    def test_orders_per_email(self, sample_df):
        generator = InteractionFeatureGenerator(
            ratios=[
                ("total_orders", "esent", "orders_per_email")
            ]
        )
        result = generator.fit_transform(sample_df)

        expected = sample_df["total_orders"] / sample_df["esent"]
        pd.testing.assert_series_equal(
            result["orders_per_email"], expected, check_names=False
        )

    def test_value_per_email(self, sample_df):
        generator = InteractionFeatureGenerator(
            ratios=[
                ("total_revenue", "esent", "value_per_email")
            ]
        )
        result = generator.fit_transform(sample_df)

        expected = sample_df["total_revenue"] / sample_df["esent"]
        pd.testing.assert_series_equal(
            result["value_per_email"], expected, check_names=False
        )

    def test_orders_per_visit(self, sample_df):
        generator = InteractionFeatureGenerator(
            ratios=[
                ("total_orders", "total_visits", "orders_per_visit")
            ]
        )
        result = generator.fit_transform(sample_df)

        expected = sample_df["total_orders"] / sample_df["total_visits"]
        pd.testing.assert_series_equal(
            result["orders_per_visit"], expected, check_names=False
        )


class TestDivisionByZeroHandling:
    def test_handles_zero_denominator_in_ratio(self):
        df = pd.DataFrame({
            "custid": ["C001", "C002"],
            "total_orders": [24, 12],
            "esent": [100, 0]
        })
        generator = InteractionFeatureGenerator(
            ratios=[
                ("total_orders", "esent", "orders_per_email")
            ]
        )
        result = generator.fit_transform(df)

        assert not np.isinf(result["orders_per_email"].iloc[1])
        assert pd.isna(result["orders_per_email"].iloc[1]) or result["orders_per_email"].iloc[1] == 0

    def test_handles_zero_denominator_in_combination(self):
        df = pd.DataFrame({
            "custid": ["C001", "C002"],
            "avgorder": [50.0, 100.0],
            "days_since_last_order": [10, 0]
        })
        generator = InteractionFeatureGenerator(
            combinations=[
                ("avgorder", "days_since_last_order", "value_by_recency", "divide")
            ]
        )
        result = generator.fit_transform(df)

        assert not np.isinf(result["value_by_recency"].iloc[1])


class TestNullHandling:
    def test_handles_null_in_first_column(self):
        df = pd.DataFrame({
            "custid": ["C001", "C002"],
            "avgorder": [50.0, None],
            "order_frequency": [2.0, 1.0]
        })
        generator = InteractionFeatureGenerator(
            combinations=[
                ("avgorder", "order_frequency", "value_x_frequency", "multiply")
            ]
        )
        result = generator.fit_transform(df)

        assert not pd.isna(result["value_x_frequency"].iloc[0])
        assert pd.isna(result["value_x_frequency"].iloc[1])

    def test_handles_null_in_second_column(self):
        df = pd.DataFrame({
            "custid": ["C001", "C002"],
            "avgorder": [50.0, 100.0],
            "order_frequency": [2.0, None]
        })
        generator = InteractionFeatureGenerator(
            combinations=[
                ("avgorder", "order_frequency", "value_x_frequency", "multiply")
            ]
        )
        result = generator.fit_transform(df)

        assert pd.isna(result["value_x_frequency"].iloc[1])


class TestOperationTypes:
    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({
            "custid": ["C001", "C002"],
            "col_a": [10.0, 20.0],
            "col_b": [2.0, 4.0]
        })

    def test_multiply_operation(self, sample_df):
        generator = InteractionFeatureGenerator(
            combinations=[("col_a", "col_b", "result", "multiply")]
        )
        result = generator.fit_transform(sample_df)
        pd.testing.assert_series_equal(
            result["result"],
            pd.Series([20.0, 80.0]),
            check_names=False
        )

    def test_divide_operation(self, sample_df):
        generator = InteractionFeatureGenerator(
            combinations=[("col_a", "col_b", "result", "divide")]
        )
        result = generator.fit_transform(sample_df)
        pd.testing.assert_series_equal(
            result["result"],
            pd.Series([5.0, 5.0]),
            check_names=False
        )

    def test_add_operation(self, sample_df):
        generator = InteractionFeatureGenerator(
            combinations=[("col_a", "col_b", "result", "add")]
        )
        result = generator.fit_transform(sample_df)
        pd.testing.assert_series_equal(
            result["result"],
            pd.Series([12.0, 24.0]),
            check_names=False
        )

    def test_subtract_operation(self, sample_df):
        generator = InteractionFeatureGenerator(
            combinations=[("col_a", "col_b", "result", "subtract")]
        )
        result = generator.fit_transform(sample_df)
        pd.testing.assert_series_equal(
            result["result"],
            pd.Series([8.0, 16.0]),
            check_names=False
        )


class TestFitTransformSeparation:
    def test_fit_then_transform(self):
        train = pd.DataFrame({
            "custid": ["C001", "C002"],
            "col_a": [10.0, 20.0],
            "col_b": [2.0, 4.0]
        })
        test = pd.DataFrame({
            "custid": ["C003"],
            "col_a": [30.0],
            "col_b": [3.0]
        })

        generator = InteractionFeatureGenerator(
            combinations=[("col_a", "col_b", "result", "multiply")]
        )
        generator.fit(train)
        result = generator.transform(test)

        assert result["result"].iloc[0] == pytest.approx(90.0)


class TestGeneratedFeaturesTracking:
    def test_generated_features_tracked(self):
        df = pd.DataFrame({
            "custid": ["C001"],
            "col_a": [10.0],
            "col_b": [2.0],
            "col_c": [5.0]
        })
        generator = InteractionFeatureGenerator(
            combinations=[
                ("col_a", "col_b", "combo1", "multiply"),
                ("col_a", "col_c", "combo2", "add")
            ],
            ratios=[
                ("col_a", "col_b", "ratio1")
            ]
        )
        result = generator.fit_transform(df)

        assert hasattr(generator, 'generated_features')
        assert "combo1" in generator.generated_features
        assert "combo2" in generator.generated_features
        assert "ratio1" in generator.generated_features


class TestMissingColumnHandling:
    def test_skips_missing_columns(self):
        df = pd.DataFrame({
            "custid": ["C001"],
            "col_a": [10.0]
        })
        generator = InteractionFeatureGenerator(
            combinations=[("col_a", "col_missing", "result", "multiply")]
        )
        result = generator.fit_transform(df)

        # Should not raise error, just skip the combination
        assert "result" not in result.columns or pd.isna(result["result"].iloc[0])
