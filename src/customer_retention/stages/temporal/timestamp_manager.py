"""Timestamp management for leakage-safe ML pipelines.

This module provides the core timestamp handling infrastructure for ensuring
point-in-time (PIT) correctness in ML training pipelines. It supports multiple
strategies for managing timestamps depending on data availability.

Key concepts:
    - feature_timestamp: When features were observed
    - label_timestamp: When the label became known
    - label_available_flag: Whether the label can be used for training

Example:
    >>> from customer_retention.stages.temporal import TimestampManager, TimestampConfig, TimestampStrategy
    >>> config = TimestampConfig(
    ...     strategy=TimestampStrategy.PRODUCTION,
    ...     feature_timestamp_column="last_activity_date",
    ...     label_timestamp_column="churn_date"
    ... )
    >>> manager = TimestampManager(config)
    >>> df_with_timestamps = manager.ensure_timestamps(df)
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional, Any
import pandas as pd
import numpy as np


class TimestampStrategy(Enum):
    """Strategy for handling timestamps in the ML pipeline.

    Attributes:
        PRODUCTION: Use explicit timestamp columns from the data
        SYNTHETIC_RANDOM: Generate random timestamps within a date range
        SYNTHETIC_INDEX: Generate timestamps based on row index
        SYNTHETIC_FIXED: Use a fixed timestamp for all rows
        DERIVED: Derive timestamps from other columns (e.g., tenure)
    """
    PRODUCTION = "production"
    SYNTHETIC_RANDOM = "synthetic_random"
    SYNTHETIC_INDEX = "synthetic_index"
    SYNTHETIC_FIXED = "synthetic_fixed"
    DERIVED = "derived"


@dataclass
class TimestampConfig:
    """Configuration for timestamp handling strategy.

    Attributes:
        strategy: The timestamp handling strategy to use
        feature_timestamp_column: Column name for feature timestamps (production strategy)
        label_timestamp_column: Column name for label timestamps (production strategy)
        observation_window_days: Days between feature observation and label availability
        synthetic_base_date: Base date for synthetic timestamp generation
        synthetic_range_days: Range of days for synthetic random timestamps
        derive_label_from_feature: If True, derive label_timestamp from feature_timestamp
        derivation_config: Configuration for derived timestamps (formula, source columns)
    """

    strategy: TimestampStrategy
    feature_timestamp_column: Optional[str] = None
    label_timestamp_column: Optional[str] = None
    observation_window_days: int = 90
    synthetic_base_date: str = "2024-01-01"
    synthetic_range_days: int = 365
    derive_label_from_feature: bool = False
    derivation_config: Optional[dict[str, Any]] = None


class TimestampManager:
    """Manages timestamp columns to ensure point-in-time correctness.

    The TimestampManager ensures that all data has proper feature_timestamp,
    label_timestamp, and label_available_flag columns, regardless of whether
    the source data has explicit timestamps or needs synthetic ones.

    Example:
        >>> config = TimestampConfig(strategy=TimestampStrategy.SYNTHETIC_FIXED)
        >>> manager = TimestampManager(config)
        >>> df = manager.ensure_timestamps(df)
        >>> assert "feature_timestamp" in df.columns
        >>> assert "label_timestamp" in df.columns
        >>> assert "label_available_flag" in df.columns
    """

    def __init__(self, config: TimestampConfig):
        """Initialize the TimestampManager.

        Args:
            config: Configuration specifying the timestamp strategy
        """
        self.config = config

    def ensure_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add or validate timestamp columns based on the configured strategy.

        This is the main entry point for timestamp handling. It adds feature_timestamp,
        label_timestamp, and label_available_flag columns to the DataFrame.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with timestamp columns added

        Raises:
            ValueError: If production strategy is used but required columns are missing
        """
        if self.config.strategy == TimestampStrategy.PRODUCTION:
            return self._validate_production_timestamps(df)
        elif self.config.strategy == TimestampStrategy.DERIVED:
            return self._derive_timestamps(df)
        return self._add_synthetic_timestamps(df)

    def _validate_production_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        required = [self.config.feature_timestamp_column, self.config.label_timestamp_column]
        missing = [col for col in required if col and col not in df.columns]
        if missing:
            raise ValueError(f"Missing required timestamp columns: {missing}")

        df = df.copy()
        if self.config.feature_timestamp_column:
            df["feature_timestamp"] = self._parse_datetime_column(
                df[self.config.feature_timestamp_column], self.config.feature_timestamp_column
            )
        if self.config.label_timestamp_column:
            df["label_timestamp"] = self._parse_datetime_column(
                df[self.config.label_timestamp_column], self.config.label_timestamp_column
            )
        elif self.config.derive_label_from_feature:
            window = timedelta(days=self.config.observation_window_days)
            df["label_timestamp"] = df["feature_timestamp"] + window
        df["label_available_flag"] = df["label_timestamp"] <= datetime.now()
        return df

    def _parse_datetime_column(self, series: pd.Series, col_name: str) -> pd.Series:
        if pd.api.types.is_datetime64_any_dtype(series):
            return series
        parsed = pd.to_datetime(series, format="mixed", errors="coerce")
        invalid_count = parsed.isna().sum() - series.isna().sum()
        if invalid_count > 0:
            import warnings
            warnings.warn(f"Column '{col_name}': {invalid_count} invalid dates coerced to NaT")
        return parsed

    def _derive_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.config.derivation_config:
            raise ValueError("derivation_config required for DERIVED strategy")

        df = df.copy()
        config = self.config.derivation_config

        if "feature_derivation" in config:
            df = self._apply_derivation(df, config["feature_derivation"], "feature_timestamp")
        if "label_derivation" in config:
            df = self._apply_derivation(df, config["label_derivation"], "label_timestamp")
        elif "feature_timestamp" in df.columns:
            window = timedelta(days=self.config.observation_window_days)
            df["label_timestamp"] = df["feature_timestamp"] + window

        df["label_available_flag"] = True
        return df

    def _apply_derivation(self, df: pd.DataFrame, derivation: dict, target_col: str) -> pd.DataFrame:
        sources = derivation.get("sources", [])
        formula = derivation.get("formula", "")

        if not sources or not formula:
            return df

        if "tenure" in formula.lower() and len(sources) >= 1:
            tenure_col = sources[0]
            if tenure_col in df.columns:
                reference_date = datetime.now()
                df[target_col] = reference_date - pd.to_timedelta(df[tenure_col] * 30, unit="D")
        return df

    def _add_synthetic_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        base = pd.to_datetime(self.config.synthetic_base_date)
        window = timedelta(days=self.config.observation_window_days)

        if self.config.strategy == TimestampStrategy.SYNTHETIC_FIXED:
            df["feature_timestamp"] = base
            df["label_timestamp"] = base + window
        elif self.config.strategy == TimestampStrategy.SYNTHETIC_INDEX:
            df["feature_timestamp"] = base + pd.to_timedelta(df.index, unit="D")
            df["label_timestamp"] = df["feature_timestamp"] + window
        elif self.config.strategy == TimestampStrategy.SYNTHETIC_RANDOM:
            np.random.seed(42)
            days = np.random.randint(0, self.config.synthetic_range_days, len(df))
            df["feature_timestamp"] = base + pd.to_timedelta(days, unit="D")
            df["label_timestamp"] = df["feature_timestamp"] + window

        df["label_available_flag"] = True
        return df

    def validate_point_in_time(self, df: pd.DataFrame) -> bool:
        """Validate that timestamps maintain point-in-time correctness.

        Ensures that feature_timestamp is always <= label_timestamp for all rows,
        which is required to prevent data leakage during training.

        Args:
            df: DataFrame with timestamp columns

        Returns:
            True if validation passes

        Raises:
            ValueError: If timestamp columns are missing or violations are found
        """
        if "feature_timestamp" not in df.columns or "label_timestamp" not in df.columns:
            raise ValueError("Missing timestamp columns for point-in-time validation")

        violations = df[df["feature_timestamp"] > df["label_timestamp"]]
        if len(violations) > 0:
            raise ValueError(
                f"Point-in-time violation: {len(violations)} rows have "
                f"feature_timestamp > label_timestamp"
            )
        return True

    def get_timestamp_summary(self, df: pd.DataFrame) -> dict[str, Any]:
        """Generate a summary of timestamp column statistics.

        Args:
            df: DataFrame with timestamp columns

        Returns:
            Dictionary containing timestamp statistics including min/max dates,
            null percentages, and label availability rates
        """
        summary = {"strategy": self.config.strategy.value}

        for col in ["feature_timestamp", "label_timestamp"]:
            if col in df.columns:
                summary[f"{col}_min"] = df[col].min()
                summary[f"{col}_max"] = df[col].max()
                summary[f"{col}_null_pct"] = df[col].isna().mean()

        if "label_available_flag" in df.columns:
            summary["label_available_pct"] = df["label_available_flag"].mean()

        return summary
