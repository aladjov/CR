
import pandas as pd
import pytest

from customer_retention.stages.temporal.timestamp_discovery import (
    DatetimeOrderAnalyzer,
    TimestampDiscoveryEngine,
)


class TestDatetimeOrderAnalyzer:
    @pytest.fixture
    def analyzer(self):
        return DatetimeOrderAnalyzer()

    @pytest.fixture
    def df_with_ordered_dates(self):
        return pd.DataFrame({
            "customer_id": ["A", "B", "C"],
            "created": pd.to_datetime(["2020-01-01", "2020-02-01", "2020-03-01"]),
            "first_order": pd.to_datetime(["2020-06-01", "2020-07-01", "2020-08-01"]),
            "last_order": pd.to_datetime(["2023-01-01", "2023-06-01", "2024-01-01"]),
            "amount": [100.0, 200.0, 300.0],
        })

    def test_detect_chronological_order(self, analyzer, df_with_ordered_dates):
        ordering = analyzer.analyze_datetime_ordering(df_with_ordered_dates)

        assert ordering[0] == "created"
        assert ordering[1] == "first_order"
        assert ordering[2] == "last_order"

    def test_identify_latest_activity_column(self, analyzer, df_with_ordered_dates):
        latest = analyzer.find_latest_activity_column(df_with_ordered_dates)

        assert latest == "last_order"

    def test_identify_earliest_column(self, analyzer, df_with_ordered_dates):
        earliest = analyzer.find_earliest_column(df_with_ordered_dates)

        assert earliest == "created"

    def test_single_datetime_column(self, analyzer):
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "order_date": pd.to_datetime(["2023-01-01", "2023-02-01", "2023-03-01"]),
        })

        ordering = analyzer.analyze_datetime_ordering(df)
        assert ordering == ["order_date"]

        latest = analyzer.find_latest_activity_column(df)
        assert latest == "order_date"

    def test_no_datetime_columns(self, analyzer):
        df = pd.DataFrame({"id": [1, 2, 3], "amount": [100, 200, 300]})

        ordering = analyzer.analyze_datetime_ordering(df)
        assert ordering == []

        latest = analyzer.find_latest_activity_column(df)
        assert latest is None

    def test_handles_null_dates(self, analyzer):
        df = pd.DataFrame({
            "created": pd.to_datetime(["2020-01-01", None, "2020-03-01"]),
            "last_order": pd.to_datetime(["2023-01-01", "2023-06-01", None]),
        })

        ordering = analyzer.analyze_datetime_ordering(df)
        assert "created" in ordering
        assert "last_order" in ordering

    def test_activity_patterns_preferred(self, analyzer):
        df = pd.DataFrame({
            "signup_date": pd.to_datetime(["2020-01-01", "2020-02-01"]),
            "random_date": pd.to_datetime(["2023-06-01", "2023-07-01"]),
            "last_login": pd.to_datetime(["2023-01-01", "2023-02-01"]),
        })

        latest = analyzer.find_latest_activity_column(df)
        assert latest == "last_login"


class TestDeriveLastActionDate:
    @pytest.fixture
    def analyzer(self):
        return DatetimeOrderAnalyzer()

    def test_coalesces_latest_first(self, analyzer):
        df = pd.DataFrame({
            "created": pd.to_datetime(["2020-01-01", "2020-02-01", "2020-03-01"]),
            "last_order": pd.to_datetime(["2023-01-01", None, None]),
        })

        result = analyzer.derive_last_action_date(df)

        assert result is not None
        assert result.notna().all()
        assert result.iloc[0] == pd.Timestamp("2023-01-01")
        assert result.iloc[1] == pd.Timestamp("2020-02-01")
        assert result.iloc[2] == pd.Timestamp("2020-03-01")

    def test_full_coverage_with_sparse_columns(self, analyzer):
        n = 100
        created = pd.date_range("2020-01-01", periods=n, freq="D")
        last_order = pd.Series([pd.NaT] * n)
        last_order.iloc[0] = pd.Timestamp("2023-06-01")
        last_order.iloc[50] = pd.Timestamp("2023-07-01")
        last_order.iloc[99] = pd.Timestamp("2023-08-01")

        df = pd.DataFrame({"created": created, "last_order": last_order})

        result = analyzer.derive_last_action_date(df)

        assert result is not None
        assert result.notna().sum() == 100

    def test_returns_none_for_no_datetime_columns(self, analyzer):
        df = pd.DataFrame({"id": [1, 2, 3], "amount": [100, 200, 300]})

        result = analyzer.derive_last_action_date(df)

        assert result is None

    def test_single_datetime_column(self, analyzer):
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "order_date": pd.to_datetime(["2023-01-01", "2023-02-01", "2023-03-01"]),
        })

        result = analyzer.derive_last_action_date(df)

        assert result is not None
        assert len(result) == 3
        assert result.iloc[0] == pd.Timestamp("2023-01-01")

    def test_preserves_index(self, analyzer):
        df = pd.DataFrame(
            {"ts": pd.to_datetime(["2023-01-01", "2023-02-01", "2023-03-01"])},
            index=[10, 20, 30],
        )

        result = analyzer.derive_last_action_date(df)

        assert list(result.index) == [10, 20, 30]

    def test_respects_ordering_by_median(self, analyzer):
        df = pd.DataFrame({
            "early": pd.to_datetime(["2020-01-01", "2020-02-01", "2020-03-01"]),
            "mid": pd.to_datetime(["2021-06-01", "2021-07-01", None]),
            "late": pd.to_datetime(["2023-01-01", None, None]),
        })

        result = analyzer.derive_last_action_date(df)

        # Row 0: has late value -> picks 2023-01-01
        assert result.iloc[0] == pd.Timestamp("2023-01-01")
        # Row 1: no late, has mid -> picks 2021-07-01
        assert result.iloc[1] == pd.Timestamp("2021-07-01")
        # Row 2: no late, no mid, has early -> picks 2020-03-01
        assert result.iloc[2] == pd.Timestamp("2020-03-01")

    def test_returns_series_named_last_action_date(self, analyzer):
        df = pd.DataFrame({
            "ts": pd.to_datetime(["2023-01-01", "2023-02-01"]),
        })

        result = analyzer.derive_last_action_date(df)

        assert result.name == "last_action_date"


class TestTimestampDiscoveryWithOrdering:
    @pytest.fixture
    def engine(self):
        return TimestampDiscoveryEngine()

    @pytest.fixture
    def retail_like_df(self):
        return pd.DataFrame({
            "customer_id": ["A", "B", "C"],
            "created": pd.to_datetime(["2020-01-01", "2020-02-01", "2020-03-01"]),
            "firstorder": pd.to_datetime(["2020-06-01", "2020-07-01", "2020-08-01"]),
            "lastorder": pd.to_datetime(["2023-01-01", "2023-06-01", "2024-01-01"]),
            "retained": [1, 0, 1],
        })

    def test_uses_latest_activity_as_feature_timestamp(self, engine, retail_like_df):
        result = engine.discover(retail_like_df, target_column="retained")

        assert result.feature_timestamp is not None
        assert result.feature_timestamp.column_name == "lastorder"

    def test_derives_label_from_latest_activity(self, engine, retail_like_df):
        result = engine.discover(retail_like_df, target_column="retained")

        assert result.label_timestamp is not None
        assert result.label_timestamp.is_derived
        assert "lastorder" in result.label_timestamp.source_columns

    def test_label_derivation_uses_180_day_window(self, engine, retail_like_df):
        result = engine.discover(retail_like_df, target_column="retained")

        assert result.label_timestamp is not None
        assert "180" in result.label_timestamp.derivation_formula

    def test_explicit_label_timestamp_not_overridden(self, engine):
        df = pd.DataFrame({
            "customer_id": ["A", "B"],
            "last_order": pd.to_datetime(["2023-01-01", "2023-06-01"]),
            "churn_date": pd.to_datetime(["2023-04-01", "2023-09-01"]),
            "retained": [1, 0],
        })

        result = engine.discover(df, target_column="retained")

        assert result.label_timestamp.column_name == "churn_date"
        assert not result.label_timestamp.is_derived

    def test_ordering_in_discovery_report(self, engine, retail_like_df):
        result = engine.discover(retail_like_df, target_column="retained")

        assert "datetime_ordering" in result.discovery_report
        assert result.discovery_report["datetime_ordering"][0] == "created"


class TestLabelDerivationWindow:
    @pytest.fixture
    def engine(self):
        return TimestampDiscoveryEngine(label_window_days=180)

    def test_configurable_window(self):
        engine_90 = TimestampDiscoveryEngine(label_window_days=90)
        engine_180 = TimestampDiscoveryEngine(label_window_days=180)

        assert engine_90.label_window_days == 90
        assert engine_180.label_window_days == 180

    def test_derived_date_range_reflects_window(self, engine):
        df = pd.DataFrame({
            "customer_id": ["A"],
            "last_order": pd.to_datetime(["2023-01-01"]),
            "retained": [1],
        })

        result = engine.discover(df, target_column="retained")

        if result.label_timestamp and result.label_timestamp.date_range[0]:
            feature_max = result.feature_timestamp.date_range[1]
            label_min = result.label_timestamp.date_range[0]
            diff = (label_min - feature_max).days
            assert diff == 180
