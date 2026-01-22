import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from customer_retention.stages.transformation import DatetimeTransformer, DatetimeTransformResult


class TestDatetimeExtraction:
    @pytest.fixture
    def datetime_series(self):
        return pd.Series(pd.to_datetime([
            "2024-01-15", "2024-03-20", "2024-06-10",
            "2024-09-05", "2024-12-25"
        ]))

    def test_extract_year(self, datetime_series):
        transformer = DatetimeTransformer(extract_features=["year"])
        result = transformer.fit_transform(datetime_series)

        assert "year" in result.df.columns
        assert result.df["year"].iloc[0] == 2024

    def test_extract_month(self, datetime_series):
        transformer = DatetimeTransformer(extract_features=["month"])
        result = transformer.fit_transform(datetime_series)

        assert "month" in result.df.columns
        assert result.df["month"].iloc[0] == 1
        assert result.df["month"].iloc[1] == 3

    def test_extract_day(self, datetime_series):
        transformer = DatetimeTransformer(extract_features=["day"])
        result = transformer.fit_transform(datetime_series)

        assert "day" in result.df.columns
        assert result.df["day"].iloc[0] == 15
        assert result.df["day"].iloc[1] == 20

    def test_extract_day_of_week(self, datetime_series):
        transformer = DatetimeTransformer(extract_features=["day_of_week"])
        result = transformer.fit_transform(datetime_series)

        assert "day_of_week" in result.df.columns
        assert 0 <= result.df["day_of_week"].iloc[0] <= 6

    def test_extract_is_weekend(self, datetime_series):
        transformer = DatetimeTransformer(extract_features=["is_weekend"])
        result = transformer.fit_transform(datetime_series)

        assert "is_weekend" in result.df.columns
        assert result.df["is_weekend"].dtype in [np.int64, np.int32, int]

    def test_extract_quarter(self, datetime_series):
        transformer = DatetimeTransformer(extract_features=["quarter"])
        result = transformer.fit_transform(datetime_series)

        assert "quarter" in result.df.columns
        assert result.df["quarter"].iloc[0] == 1
        assert result.df["quarter"].iloc[2] == 2

    def test_extract_is_month_start(self):
        series = pd.Series(pd.to_datetime(["2024-01-01", "2024-01-15", "2024-02-01"]))
        transformer = DatetimeTransformer(extract_features=["is_month_start"])
        result = transformer.fit_transform(series)

        assert "is_month_start" in result.df.columns
        assert result.df["is_month_start"].iloc[0] == 1
        assert result.df["is_month_start"].iloc[1] == 0
        assert result.df["is_month_start"].iloc[2] == 1

    def test_extract_is_month_end(self):
        series = pd.Series(pd.to_datetime(["2024-01-31", "2024-01-15", "2024-02-29"]))
        transformer = DatetimeTransformer(extract_features=["is_month_end"])
        result = transformer.fit_transform(series)

        assert "is_month_end" in result.df.columns
        assert result.df["is_month_end"].iloc[0] == 1
        assert result.df["is_month_end"].iloc[1] == 0
        assert result.df["is_month_end"].iloc[2] == 1

    def test_extract_day_of_year(self):
        series = pd.Series(pd.to_datetime(["2024-01-01", "2024-12-31"]))
        transformer = DatetimeTransformer(extract_features=["day_of_year"])
        result = transformer.fit_transform(series)

        assert "day_of_year" in result.df.columns
        assert result.df["day_of_year"].iloc[0] == 1
        assert result.df["day_of_year"].iloc[1] == 366

    def test_extract_week_of_year(self):
        series = pd.Series(pd.to_datetime(["2024-01-08", "2024-06-15"]))
        transformer = DatetimeTransformer(extract_features=["week_of_year"])
        result = transformer.fit_transform(series)

        assert "week_of_year" in result.df.columns
        assert result.df["week_of_year"].iloc[0] >= 1

    def test_extract_hour_minute(self):
        series = pd.Series(pd.to_datetime(["2024-01-01 14:30:00", "2024-01-01 23:59:00"]))
        transformer = DatetimeTransformer(extract_features=["hour", "minute"])
        result = transformer.fit_transform(series)

        assert "hour" in result.df.columns
        assert "minute" in result.df.columns
        assert result.df["hour"].iloc[0] == 14
        assert result.df["minute"].iloc[0] == 30


class TestCyclicalDatetimeFeatures:
    @pytest.fixture
    def datetime_series(self):
        return pd.Series(pd.to_datetime([
            "2024-01-01", "2024-04-01", "2024-07-01", "2024-10-01"
        ]))

    def test_cyclical_month_creates_sin_cos(self, datetime_series):
        transformer = DatetimeTransformer(
            extract_features=["month"], cyclical_features=["month"]
        )
        result = transformer.fit_transform(datetime_series)

        assert "month_sin" in result.df.columns
        assert "month_cos" in result.df.columns

    def test_cyclical_day_of_week_values_in_range(self):
        series = pd.Series(pd.to_datetime([
            "2024-01-01", "2024-01-02", "2024-01-03",
            "2024-01-04", "2024-01-05", "2024-01-06", "2024-01-07"
        ]))
        transformer = DatetimeTransformer(
            extract_features=["day_of_week"], cyclical_features=["day_of_week"]
        )
        result = transformer.fit_transform(series)

        assert result.df["day_of_week_sin"].min() >= -1
        assert result.df["day_of_week_sin"].max() <= 1
        assert result.df["day_of_week_cos"].min() >= -1
        assert result.df["day_of_week_cos"].max() <= 1


class TestDaysSinceFeatures:
    @pytest.fixture
    def datetime_series(self):
        return pd.Series(pd.to_datetime([
            "2024-01-01", "2024-01-15", "2024-02-01"
        ]))

    def test_days_since_reference(self, datetime_series):
        reference = pd.Timestamp("2024-03-01")
        transformer = DatetimeTransformer(reference_date=reference)
        result = transformer.fit_transform(datetime_series)

        assert "days_since" in result.df.columns
        assert result.df["days_since"].iloc[0] == (reference - datetime_series.iloc[0]).days

    def test_days_since_with_string_reference(self, datetime_series):
        transformer = DatetimeTransformer(reference_date="2024-03-01")
        result = transformer.fit_transform(datetime_series)

        assert "days_since" in result.df.columns


class TestMultipleFeatureExtraction:
    def test_extract_multiple_features(self):
        series = pd.Series(pd.to_datetime(["2024-06-15 14:30:00"]))
        transformer = DatetimeTransformer(
            extract_features=["year", "month", "day", "day_of_week", "quarter"]
        )
        result = transformer.fit_transform(series)

        assert "year" in result.df.columns
        assert "month" in result.df.columns
        assert "day" in result.df.columns
        assert "day_of_week" in result.df.columns
        assert "quarter" in result.df.columns


class TestDropOriginal:
    def test_drop_original_column(self):
        series = pd.Series(pd.to_datetime(["2024-01-01"]), name="created")
        transformer = DatetimeTransformer(
            extract_features=["year"], drop_original=True
        )
        result = transformer.fit_transform(series)

        assert "created" not in result.df.columns or result.drop_original

    def test_keep_original_column(self):
        series = pd.Series(pd.to_datetime(["2024-01-01"]), name="created")
        transformer = DatetimeTransformer(
            extract_features=["year"], drop_original=False
        )
        result = transformer.fit_transform(series)

        assert not result.drop_original


class TestEdgeCases:
    def test_handles_nulls(self):
        series = pd.Series(pd.to_datetime([
            "2024-01-01", None, "2024-03-01"
        ]))
        transformer = DatetimeTransformer(extract_features=["year", "month"])
        result = transformer.fit_transform(series)

        assert pd.isna(result.df["year"].iloc[1])
        assert pd.isna(result.df["month"].iloc[1])

    def test_handles_string_dates(self):
        series = pd.Series(["2024-01-01", "2024-06-15", "2024-12-31"])
        transformer = DatetimeTransformer(extract_features=["year", "month"])
        result = transformer.fit_transform(series)

        assert result.df["year"].iloc[0] == 2024
        assert result.df["month"].iloc[1] == 6


class TestResultOutput:
    def test_result_contains_extracted_features(self):
        series = pd.Series(pd.to_datetime(["2024-01-01"]))
        transformer = DatetimeTransformer(
            extract_features=["year", "month", "day"]
        )
        result = transformer.fit_transform(series)

        assert result.extracted_features is not None
        assert "year" in result.extracted_features
        assert "month" in result.extracted_features
        assert "day" in result.extracted_features
