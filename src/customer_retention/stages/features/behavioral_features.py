"""
Behavioral feature generation for customer retention analysis.

This module provides behavioral feature calculations such as frequency,
engagement, service adoption, and recency buckets.
"""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from customer_retention.core.compat import pd, DataFrame, Series


@dataclass
class BehavioralFeatureResult:
    """Result of behavioral feature generation."""
    df: DataFrame
    generated_features: List[str]
    warnings: List[str] = field(default_factory=list)
    pit_warnings: List[str] = field(default_factory=list)


class BehavioralFeatureGenerator:
    """
    Generates behavioral features from customer data.

    Behavioral features capture customer activity patterns, engagement levels,
    and service adoption metrics.

    Parameters
    ----------
    tenure_months_column : str, optional
        Column containing customer tenure in months.
    tenure_days_column : str, optional
        Column containing customer tenure in days.
    total_orders_column : str, optional
        Column containing total number of orders.
    emails_sent_column : str, optional
        Column containing number of emails sent.
    total_visits_column : str, optional
        Column containing total visits.
    days_since_last_order_column : str, optional
        Column containing days since last order.
    open_rate_column : str, optional
        Column containing email open rate.
    click_rate_column : str, optional
        Column containing email click rate.
    service_columns : List[str], optional
        List of columns indicating service adoption (binary).
    recency_bins : List[float], optional
        Bin edges for recency buckets.
    recency_labels : List[str], optional
        Labels for recency buckets.

    Attributes
    ----------
    generated_features : List[str]
        Names of features generated during last transform.
    """

    DEFAULT_RECENCY_BINS = [0, 7, 30, 60, 90, float('inf')]
    DEFAULT_RECENCY_LABELS = ["active", "recent", "warm", "cooling", "dormant"]

    def __init__(
        self,
        tenure_months_column: Optional[str] = None,
        tenure_days_column: Optional[str] = None,
        total_orders_column: Optional[str] = None,
        emails_sent_column: Optional[str] = None,
        total_visits_column: Optional[str] = None,
        days_since_last_order_column: Optional[str] = None,
        open_rate_column: Optional[str] = None,
        click_rate_column: Optional[str] = None,
        service_columns: Optional[List[str]] = None,
        recency_bins: Optional[List[float]] = None,
        recency_labels: Optional[List[str]] = None,
        enforce_point_in_time: bool = False,
        feature_timestamp_column: Optional[str] = None,
    ):
        self.tenure_months_column = tenure_months_column
        self.tenure_days_column = tenure_days_column
        self.total_orders_column = total_orders_column
        self.emails_sent_column = emails_sent_column
        self.total_visits_column = total_visits_column
        self.days_since_last_order_column = days_since_last_order_column
        self.open_rate_column = open_rate_column
        self.click_rate_column = click_rate_column
        self.service_columns = service_columns or []
        self.recency_bins = recency_bins or self.DEFAULT_RECENCY_BINS
        self.recency_labels = recency_labels or self.DEFAULT_RECENCY_LABELS
        self.enforce_point_in_time = enforce_point_in_time
        self.feature_timestamp_column = feature_timestamp_column or "feature_timestamp"

        self.generated_features: List[str] = []
        self.pit_warnings: List[str] = []
        self._is_fitted = False

    def fit(self, df: DataFrame) -> "BehavioralFeatureGenerator":
        """
        Fit the generator (stores configuration but no learning required).

        Parameters
        ----------
        df : DataFrame
            Input DataFrame.

        Returns
        -------
        self
        """
        self._is_fitted = True
        return self

    def transform(self, df: DataFrame) -> DataFrame:
        """
        Generate behavioral features for the input DataFrame.

        Parameters
        ----------
        df : DataFrame
            Input DataFrame.

        Returns
        -------
        DataFrame
            DataFrame with behavioral features added.
        """
        if not self._is_fitted:
            raise ValueError("Generator not fitted. Call fit() first.")

        result = df.copy()
        self.generated_features = []
        self.pit_warnings = []

        if self.enforce_point_in_time:
            self._validate_point_in_time(result)

        # Frequency features
        result = self._generate_frequency_features(result)

        # Engagement features
        result = self._generate_engagement_features(result)

        # Service adoption features
        result = self._generate_service_adoption_features(result)

        # Recency bucket
        result = self._generate_recency_bucket(result)

        return result

    def fit_transform(self, df: DataFrame) -> DataFrame:
        """
        Fit and transform in one step.

        Parameters
        ----------
        df : DataFrame
            Input DataFrame.

        Returns
        -------
        DataFrame
            DataFrame with behavioral features added.
        """
        self.fit(df)
        return self.transform(df)

    def _generate_frequency_features(self, df: DataFrame) -> DataFrame:
        """Generate frequency-based features."""
        # Order frequency
        if self.tenure_months_column and self.total_orders_column:
            if self.tenure_months_column in df.columns and self.total_orders_column in df.columns:
                tenure = df[self.tenure_months_column].replace(0, np.nan)
                df["order_frequency"] = df[self.total_orders_column] / tenure
                self.generated_features.append("order_frequency")

        # Email frequency
        if self.tenure_months_column and self.emails_sent_column:
            if self.tenure_months_column in df.columns and self.emails_sent_column in df.columns:
                tenure = df[self.tenure_months_column].replace(0, np.nan)
                df["email_frequency"] = df[self.emails_sent_column] / tenure
                self.generated_features.append("email_frequency")

        # Visit frequency
        if self.tenure_months_column and self.total_visits_column:
            if self.tenure_months_column in df.columns and self.total_visits_column in df.columns:
                tenure = df[self.tenure_months_column].replace(0, np.nan)
                df["visit_frequency"] = df[self.total_visits_column] / tenure
                self.generated_features.append("visit_frequency")

        # Order recency ratio
        if self.tenure_days_column and self.days_since_last_order_column:
            if self.tenure_days_column in df.columns and self.days_since_last_order_column in df.columns:
                tenure = df[self.tenure_days_column].replace(0, np.nan)
                df["order_recency_ratio"] = df[self.days_since_last_order_column] / tenure
                self.generated_features.append("order_recency_ratio")

        return df

    def _generate_engagement_features(self, df: DataFrame) -> DataFrame:
        """Generate engagement-based features."""
        if self.open_rate_column and self.click_rate_column:
            if self.open_rate_column in df.columns and self.click_rate_column in df.columns:
                # Email engagement score
                df["email_engagement_score"] = (
                    df[self.open_rate_column] + df[self.click_rate_column]
                ) / 2
                self.generated_features.append("email_engagement_score")

                # Click to open rate (handle division by zero)
                open_rate = df[self.open_rate_column].replace(0, np.nan)
                df["click_to_open_rate"] = df[self.click_rate_column] / open_rate
                df["click_to_open_rate"] = df["click_to_open_rate"].fillna(0)
                self.generated_features.append("click_to_open_rate")

        return df

    def _generate_service_adoption_features(self, df: DataFrame) -> DataFrame:
        """Generate service adoption features."""
        if self.service_columns:
            # Check which columns exist
            existing_cols = [c for c in self.service_columns if c in df.columns]
            if existing_cols:
                # Service adoption score (count of services)
                df["service_adoption_score"] = df[existing_cols].sum(axis=1).astype(float)
                self.generated_features.append("service_adoption_score")

                # Service adoption percentage
                df["service_adoption_pct"] = df[existing_cols].sum(axis=1) / len(existing_cols)
                self.generated_features.append("service_adoption_pct")

        return df

    def _generate_recency_bucket(self, df: DataFrame) -> DataFrame:
        """Generate recency bucket feature."""
        if self.days_since_last_order_column:
            if self.days_since_last_order_column in df.columns:
                df["recency_bucket"] = pd.cut(
                    df[self.days_since_last_order_column],
                    bins=self.recency_bins,
                    labels=self.recency_labels,
                    include_lowest=True
                )
                self.generated_features.append("recency_bucket")

        return df

    def _validate_point_in_time(self, df: DataFrame) -> None:
        """Validate that behavioral inputs respect point-in-time constraints."""
        if self.feature_timestamp_column not in df.columns:
            return

        feature_ts = pd.to_datetime(df[self.feature_timestamp_column], format='mixed')
        datetime_cols = df.select_dtypes(include=["datetime64"]).columns

        for col in datetime_cols:
            if col == self.feature_timestamp_column:
                continue
            violations = df[df[col] > feature_ts]
            if len(violations) > 0:
                self.pit_warnings.append(
                    f"PIT Warning: {len(violations)} rows have {col} > {self.feature_timestamp_column}"
                )
