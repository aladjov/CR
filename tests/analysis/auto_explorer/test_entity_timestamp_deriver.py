from datetime import datetime, timedelta

import pandas as pd
import pytest

from customer_retention.analysis.auto_explorer.entity_timestamp_deriver import (
    EntityFeatureTimestampDeriver,
    EntityTimestampResult,
)


@pytest.fixture()
def deriver():
    return EntityFeatureTimestampDeriver()


class TestDirectFeatureTimestamp:
    def test_direct_feature_timestamp_found(self, deriver):
        df = pd.DataFrame({
            "customer_id": [1, 2, 3],
            "last_activity_date": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01"]),
            "revenue": [100.0, 200.0, 300.0],
        })
        result = deriver.derive(df, target_column="revenue")
        assert result.method == "direct"
        assert result.column_name is not None
        assert "last_activity" in result.column_name.lower()


class TestCoalescedTimestamp:
    def test_coalesced_from_multiple_datetime_columns(self, deriver):
        df = pd.DataFrame({
            "customer_id": [1, 2, 3],
            "created_at": pd.to_datetime(["2023-01-01", "2023-06-01", "2023-09-01"]),
            "signup_date": pd.to_datetime(["2023-01-15", "2023-06-15", "2023-09-15"]),
            "revenue": [100.0, 200.0, 300.0],
        })
        result = deriver.derive(df, target_column="revenue")
        assert result.method in ("direct", "coalesced")
        assert result.column_name is not None


class TestNoDatetimeColumns:
    def test_no_datetime_columns_returns_none(self, deriver):
        df = pd.DataFrame({
            "customer_id": [1, 2, 3],
            "revenue": [100.0, 200.0, 300.0],
            "category": ["A", "B", "C"],
        })
        result = deriver.derive(df, target_column="revenue")
        assert result.method == "none"
        assert result.column_name is None


class TestApplyDirect:
    def test_apply_direct_adds_column(self, deriver):
        dates = pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01"])
        df = pd.DataFrame({
            "customer_id": [1, 2, 3],
            "last_activity_date": dates,
            "revenue": [100.0, 200.0, 300.0],
        })
        result = deriver.derive(df, target_column="revenue")
        applied = deriver.apply(df, result)
        assert "feature_timestamp" in applied.columns
        pd.testing.assert_series_equal(
            applied["feature_timestamp"], pd.Series(dates, name="feature_timestamp"),
            check_names=False,
        )


class TestApplyCoalesced:
    def test_apply_coalesced_adds_column(self, deriver):
        df = pd.DataFrame({
            "customer_id": [1, 2, 3],
            "created_at": pd.to_datetime(["2023-01-01", None, "2023-09-01"]),
            "signup_date": pd.to_datetime([None, "2023-06-15", "2023-09-15"]),
            "revenue": [100.0, 200.0, 300.0],
        })
        result = deriver.derive(df, target_column="revenue")
        if result.method != "none":
            applied = deriver.apply(df, result)
            assert "feature_timestamp" in applied.columns
            assert applied["feature_timestamp"].notna().sum() >= 2


class TestFutureDatesExcluded:
    def test_future_dates_excluded(self, deriver):
        future = datetime.now() + timedelta(days=365 * 10)
        df = pd.DataFrame({
            "customer_id": [1, 2, 3],
            "future_col": [future, future, future],
            "created_at": pd.to_datetime(["2023-01-01", "2023-06-01", "2023-09-01"]),
            "revenue": [100.0, 200.0, 300.0],
        })
        result = deriver.derive(df, target_column="revenue")
        if result.column_name is not None:
            assert "future_col" != result.column_name


class TestEntityTimestampResult:
    def test_dataclass_fields(self):
        result = EntityTimestampResult(column_name="ts", source_columns=["ts"], method="direct")
        assert result.column_name == "ts"
        assert result.source_columns == ["ts"]
        assert result.method == "direct"

    def test_none_result(self):
        result = EntityTimestampResult(column_name=None, source_columns=[], method="none")
        assert result.column_name is None
        assert result.method == "none"
