from typing import Optional, Union
from dataclasses import dataclass, field
import numpy as np

from customer_retention.core.compat import pd, DataFrame, Series, Timestamp, is_datetime64_any_dtype


@dataclass
class DatetimeTransformResult:
    df: DataFrame
    extracted_features: list = field(default_factory=list)
    cyclical_features: list = field(default_factory=list)
    drop_original: bool = True


class DatetimeTransformer:
    FEATURE_EXTRACTORS = {
        "year": lambda s: s.dt.year,
        "month": lambda s: s.dt.month,
        "day": lambda s: s.dt.day,
        "day_of_week": lambda s: s.dt.dayofweek,
        "day_of_year": lambda s: s.dt.dayofyear,
        "week_of_year": lambda s: s.dt.isocalendar().week.astype(int),
        "quarter": lambda s: s.dt.quarter,
        "hour": lambda s: s.dt.hour,
        "minute": lambda s: s.dt.minute,
        "is_weekend": lambda s: s.dt.dayofweek.isin([5, 6]).astype(int),
        "is_month_start": lambda s: s.dt.is_month_start.astype(int),
        "is_month_end": lambda s: s.dt.is_month_end.astype(int),
        "is_quarter_start": lambda s: s.dt.is_quarter_start.astype(int),
        "is_quarter_end": lambda s: s.dt.is_quarter_end.astype(int),
    }

    CYCLICAL_PERIODS = {
        "month": 12,
        "day_of_week": 7,
        "day_of_year": 365,
        "quarter": 4,
        "hour": 24,
        "minute": 60,
    }

    def __init__(
        self,
        extract_features: Optional[list[str]] = None,
        cyclical_features: Optional[list[str]] = None,
        reference_date: Optional[Union[str, Timestamp]] = None,
        drop_original: bool = True
    ):
        self.extract_features = extract_features or ["year", "month", "day_of_week"]
        self.cyclical_features = cyclical_features or []
        self.reference_date = Timestamp(reference_date) if reference_date else None
        self.drop_original = drop_original

    def fit(self, series: Series) -> "DatetimeTransformer":
        return self

    def transform(self, series: Series) -> DatetimeTransformResult:
        return self._apply_transform(series)

    def fit_transform(self, series: Series) -> DatetimeTransformResult:
        return self._apply_transform(series)

    def _apply_transform(self, series: Series) -> DatetimeTransformResult:
        dt_series = self._ensure_datetime(series)
        result_dict = {}
        extracted = []

        for feature in self.extract_features:
            if feature in self.FEATURE_EXTRACTORS:
                values = self.FEATURE_EXTRACTORS[feature](dt_series)
                result_dict[feature] = values
                extracted.append(feature)

                if feature in self.cyclical_features:
                    period = self.CYCLICAL_PERIODS.get(feature)
                    if period:
                        sin_col = f"{feature}_sin"
                        cos_col = f"{feature}_cos"
                        result_dict[sin_col] = np.sin(2 * np.pi * values / period)
                        result_dict[cos_col] = np.cos(2 * np.pi * values / period)

        if self.reference_date is not None:
            result_dict["days_since"] = (self.reference_date - dt_series).dt.days

        df = DataFrame(result_dict)

        return DatetimeTransformResult(
            df=df, extracted_features=extracted,
            cyclical_features=self.cyclical_features, drop_original=self.drop_original
        )

    def _ensure_datetime(self, series: Series) -> Series:
        if is_datetime64_any_dtype(series):
            return series
        return pd.to_datetime(series, errors='coerce', format='mixed')
