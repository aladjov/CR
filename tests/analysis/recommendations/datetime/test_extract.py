from datetime import datetime

import pandas as pd
import pytest

from customer_retention.analysis.recommendations.base import Platform, RecommendationResult
from customer_retention.analysis.recommendations.datetime.extract import (
    DaysSinceRecommendation,
    ExtractDayOfWeekRecommendation,
    ExtractMonthRecommendation,
)


class TestExtractMonthRecommendationInit:
    def test_default_rationale(self):
        rec = ExtractMonthRecommendation(columns=["date"])
        assert "month" in rec.rationale.lower()

    def test_recommendation_type(self):
        rec = ExtractMonthRecommendation(columns=["date"])
        assert rec.recommendation_type == "extract_month"

    def test_category_is_datetime(self):
        rec = ExtractMonthRecommendation(columns=["date"])
        assert rec.category == "datetime"


class TestExtractMonthRecommendationTransform:
    @pytest.fixture
    def date_df(self):
        return pd.DataFrame({
            "date": pd.to_datetime(["2023-01-15", "2023-06-20", "2023-12-01"])
        })

    def test_transform_extracts_month(self, date_df):
        rec = ExtractMonthRecommendation(columns=["date"])
        result = rec.fit_transform(date_df)
        assert "date_month" in result.data.columns
        assert list(result.data["date_month"]) == [1, 6, 12]

    def test_transform_returns_recommendation_result(self, date_df):
        rec = ExtractMonthRecommendation(columns=["date"])
        result = rec.fit_transform(date_df)
        assert isinstance(result, RecommendationResult)


class TestExtractDayOfWeekRecommendationTransform:
    @pytest.fixture
    def date_df(self):
        return pd.DataFrame({
            "date": pd.to_datetime(["2023-01-02", "2023-01-07"])
        })

    def test_transform_extracts_dayofweek(self, date_df):
        rec = ExtractDayOfWeekRecommendation(columns=["date"])
        result = rec.fit_transform(date_df)
        assert "date_dayofweek" in result.data.columns
        assert result.data["date_dayofweek"].iloc[0] == 0
        assert result.data["date_dayofweek"].iloc[1] == 5

    def test_recommendation_type(self):
        rec = ExtractDayOfWeekRecommendation(columns=["date"])
        assert rec.recommendation_type == "extract_dayofweek"


class TestDaysSinceRecommendationInit:
    def test_default_reference_date_is_today(self):
        rec = DaysSinceRecommendation(columns=["date"])
        assert rec.reference_date is not None

    def test_custom_reference_date(self):
        ref = datetime(2023, 1, 1)
        rec = DaysSinceRecommendation(columns=["date"], reference_date=ref)
        assert rec.reference_date == ref

    def test_recommendation_type(self):
        rec = DaysSinceRecommendation(columns=["date"])
        assert rec.recommendation_type == "days_since"


class TestDaysSinceRecommendationTransform:
    @pytest.fixture
    def date_df(self):
        return pd.DataFrame({
            "date": pd.to_datetime(["2023-01-01", "2023-01-11"])
        })

    def test_transform_calculates_days_since(self, date_df):
        ref = datetime(2023, 1, 21)
        rec = DaysSinceRecommendation(columns=["date"], reference_date=ref)
        result = rec.fit_transform(date_df)
        assert "date_days_since" in result.data.columns
        assert result.data["date_days_since"].iloc[0] == 20
        assert result.data["date_days_since"].iloc[1] == 10

    def test_transform_returns_recommendation_result(self, date_df):
        rec = DaysSinceRecommendation(columns=["date"])
        result = rec.fit_transform(date_df)
        assert isinstance(result, RecommendationResult)


class TestDatetimeRecommendationCodeGeneration:
    def test_extract_month_local_code(self):
        rec = ExtractMonthRecommendation(columns=["date"])
        code = rec.generate_code(Platform.LOCAL)
        assert "dt.month" in code

    def test_extract_month_databricks_code(self):
        rec = ExtractMonthRecommendation(columns=["date"])
        code = rec.generate_code(Platform.DATABRICKS)
        assert "month" in code

    def test_days_since_local_code(self):
        rec = DaysSinceRecommendation(columns=["date"])
        code = rec.generate_code(Platform.LOCAL)
        assert "days" in code.lower()


class TestDatetimeRecommendationEdgeCases:
    def test_column_not_in_dataframe(self):
        df = pd.DataFrame({"other": pd.to_datetime(["2023-01-01"])})
        rec = ExtractMonthRecommendation(columns=["date"])
        rec.fit(df)
        result = rec.transform(df)
        assert "other" in result.data.columns

    def test_multiple_columns(self):
        df = pd.DataFrame({
            "date1": pd.to_datetime(["2023-01-15"]),
            "date2": pd.to_datetime(["2023-06-20"])
        })
        rec = ExtractMonthRecommendation(columns=["date1", "date2"])
        result = rec.fit_transform(df)
        assert "date1_month" in result.data.columns
        assert "date2_month" in result.data.columns
