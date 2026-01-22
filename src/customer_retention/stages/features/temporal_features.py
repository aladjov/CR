"""
Temporal feature generation for customer retention analysis.

This module provides temporal feature calculations such as tenure,
recency, activation time, and active period.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Union
import warnings

import numpy as np

from customer_retention.core.compat import pd, DataFrame, Series, Timestamp, Timedelta


class ReferenceDateSource(Enum):
    """Source for the reference date used in temporal calculations."""
    CONFIG = "config"
    MAX_DATE = "max_date"
    COLUMN = "column"
    FEATURE_TIMESTAMP = "feature_timestamp"


@dataclass
class TemporalFeatureResult:
    """Result of temporal feature generation."""
    df: DataFrame
    reference_date: Union[Timestamp, Series]
    generated_features: List[str]
    warnings: List[str] = field(default_factory=list)


class TemporalFeatureGenerator:
    """
    Generates temporal features from datetime columns.

    Temporal features are calculated relative to a reference date, which can
    be specified explicitly, derived from the data, or per-row from a column.

    Parameters
    ----------
    reference_date : Timestamp, optional
        Explicit reference date for calculations. Used when reference_date_source
        is CONFIG.
    reference_date_source : ReferenceDateSource, default CONFIG
        How to determine the reference date:
        - CONFIG: Use the explicit reference_date parameter
        - MAX_DATE: Use the maximum date in date_column
        - COLUMN: Use per-row dates from reference_date_column
    reference_date_column : str, optional
        Column name for per-row reference dates. Required when source is COLUMN.
    date_column : str, optional
        Column used to determine max date when source is MAX_DATE.
    created_column : str, default "created"
        Column containing customer account creation date.
    first_order_column : str, optional
        Column containing date of first order.
    last_order_column : str, optional
        Column containing date of last order.

    Attributes
    ----------
    reference_date : Timestamp or Series
        The reference date(s) used for calculations after fitting.
    generated_features : List[str]
        Names of features generated during last transform.
    """

    def __init__(
        self,
        reference_date: Optional[Timestamp] = None,
        reference_date_source: ReferenceDateSource = ReferenceDateSource.CONFIG,
        reference_date_column: Optional[str] = None,
        date_column: Optional[str] = None,
        created_column: str = "created",
        first_order_column: Optional[str] = None,
        last_order_column: Optional[str] = None,
    ):
        self._reference_date_param = reference_date
        self.reference_date_source = reference_date_source
        self.reference_date_column = reference_date_column
        self.date_column = date_column
        self.created_column = created_column
        self.first_order_column = first_order_column
        self.last_order_column = last_order_column

        self.reference_date: Optional[Union[Timestamp, Series]] = None
        self.generated_features: List[str] = []
        self._is_fitted = False

    def fit(self, df: DataFrame) -> "TemporalFeatureGenerator":
        """
        Fit the generator by determining the reference date.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame containing datetime columns.

        Returns
        -------
        self
        """
        self._determine_reference_date(df)
        self._is_fitted = True
        return self

    def transform(self, df: DataFrame) -> DataFrame:
        """
        Generate temporal features for the input DataFrame.

        Parameters
        ----------
        df : DataFrame
            Input DataFrame containing datetime columns.

        Returns
        -------
        DataFrame
            DataFrame with temporal features added.
        """
        if not self._is_fitted:
            raise ValueError("Generator not fitted. Call fit() first.")

        result = df.copy()
        self.generated_features = []
        warnings_list = []

        # Get reference date(s) for this transform
        if self.reference_date_source in [ReferenceDateSource.COLUMN, ReferenceDateSource.FEATURE_TIMESTAMP]:
            ref_dates = pd.to_datetime(df[self.reference_date_column], format='mixed')
        else:
            ref_dates = self.reference_date

        # Tenure features
        if self.created_column and self.created_column in df.columns:
            created = pd.to_datetime(df[self.created_column], format='mixed')
            tenure_days = self._compute_days_diff(ref_dates, created)
            result["tenure_days"] = tenure_days
            self.generated_features.append("tenure_days")

            # Check for negative values
            if (tenure_days < 0).any():
                warnings.warn(
                    "negative tenure_days detected. Reference date may be before "
                    "some created dates.",
                    UserWarning
                )
                warnings_list.append("negative_tenure_days")

            # Account age in months
            result["account_age_months"] = tenure_days / 30.44
            self.generated_features.append("account_age_months")

        # Recency features
        if self.last_order_column and self.last_order_column in df.columns:
            last_order = pd.to_datetime(df[self.last_order_column], format='mixed')
            days_since_last = self._compute_days_diff(ref_dates, last_order)
            result["days_since_last_order"] = days_since_last
            self.generated_features.append("days_since_last_order")

        # Activation features
        if (self.first_order_column and self.first_order_column in df.columns and
                self.created_column and self.created_column in df.columns):
            created = pd.to_datetime(df[self.created_column], format='mixed')
            first_order = pd.to_datetime(df[self.first_order_column], format='mixed')
            days_to_first = self._compute_days_diff(first_order, created)
            result["days_to_first_order"] = days_to_first
            self.generated_features.append("days_to_first_order")

        # Active period
        if (self.first_order_column and self.first_order_column in df.columns and
                self.last_order_column and self.last_order_column in df.columns):
            first_order = pd.to_datetime(df[self.first_order_column], format='mixed')
            last_order = pd.to_datetime(df[self.last_order_column], format='mixed')
            active_period = self._compute_days_diff(last_order, first_order)
            result["active_period_days"] = active_period
            self.generated_features.append("active_period_days")

        return result

    def fit_transform(self, df: DataFrame) -> DataFrame:
        """
        Fit and transform in one step.

        Parameters
        ----------
        df : DataFrame
            Input DataFrame containing datetime columns.

        Returns
        -------
        DataFrame
            DataFrame with temporal features added.
        """
        self.fit(df)
        return self.transform(df)

    def _determine_reference_date(self, df: DataFrame) -> None:
        """Determine the reference date based on configuration."""
        if self.reference_date_source == ReferenceDateSource.CONFIG:
            if self._reference_date_param is None:
                raise ValueError(
                    "reference_date must be provided when source is CONFIG"
                )
            self.reference_date = self._reference_date_param

        elif self.reference_date_source == ReferenceDateSource.MAX_DATE:
            if self.date_column is None:
                raise ValueError(
                    "date_column must be provided when source is MAX_DATE"
                )
            self.reference_date = pd.to_datetime(df[self.date_column], format='mixed').max()

        elif self.reference_date_source == ReferenceDateSource.COLUMN:
            if self.reference_date_column is None:
                raise ValueError(
                    "reference_date_column must be provided when source is COLUMN"
                )
            self.reference_date = pd.to_datetime(df[self.reference_date_column], format='mixed')

        elif self.reference_date_source == ReferenceDateSource.FEATURE_TIMESTAMP:
            if "feature_timestamp" not in df.columns:
                raise ValueError(
                    "feature_timestamp column required when source is FEATURE_TIMESTAMP"
                )
            self.reference_date = pd.to_datetime(df["feature_timestamp"], format='mixed')
            self.reference_date_column = "feature_timestamp"

    def _compute_days_diff(
        self,
        later: Union[Timestamp, Series],
        earlier: Union[Timestamp, Series]
    ) -> Series:
        """
        Compute the difference in days between two dates.

        Handles both scalar and Series inputs, preserving NaN values.
        """
        diff = later - earlier
        if isinstance(diff, Timedelta):
            return pd.Series([diff.days])
        return diff.dt.days
