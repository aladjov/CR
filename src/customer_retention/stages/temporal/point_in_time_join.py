"""Point-in-time correct joins for feature engineering.

This module provides utilities for joining feature tables while maintaining
temporal correctness. It ensures that features from the future are never
used to predict past events, preventing data leakage.

Key functions:
    - join_features: Join feature tables with PIT correctness
    - asof_join: Pandas merge_asof wrapper for temporal joins
    - validate_no_future_data: Check for temporal violations
    - validate_temporal_integrity: Comprehensive integrity check

Example:
    >>> from customer_retention.stages.temporal import PointInTimeJoiner
    >>> joiner = PointInTimeJoiner()
    >>> merged = joiner.join_features(
    ...     base_df=customers,
    ...     feature_df=transactions_agg,
    ...     entity_key="customer_id"
    ... )
"""

from typing import Any

import pandas as pd


class PointInTimeJoiner:
    """Utility class for point-in-time correct feature joins.

    The PointInTimeJoiner ensures that when joining feature tables,
    only features that were available at the time of the base record
    are included. This prevents temporal leakage.

    Example:
        >>> joiner = PointInTimeJoiner()
        >>> # Only features from before base_df's feature_timestamp are included
        >>> merged = joiner.join_features(base_df, feature_df, "customer_id")
    """
    @staticmethod
    def join_features(
        base_df: pd.DataFrame, feature_df: pd.DataFrame, entity_key: str,
        base_timestamp_col: str = "feature_timestamp", feature_timestamp_col: str = "feature_timestamp"
    ) -> pd.DataFrame:
        if base_timestamp_col not in base_df.columns:
            raise ValueError(f"Base df missing timestamp column: {base_timestamp_col}")
        if feature_timestamp_col not in feature_df.columns:
            raise ValueError(f"Feature df missing timestamp column: {feature_timestamp_col}")

        feature_df = feature_df.rename(columns={feature_timestamp_col: "_feature_ts"})
        merged = base_df.merge(feature_df, on=entity_key, how="left")
        valid_mask = merged["_feature_ts"] <= merged[base_timestamp_col]

        merged = (
            merged[valid_mask]
            .sort_values([entity_key, "_feature_ts"])
            .groupby(entity_key)
            .last()
            .reset_index()
            .drop(columns=["_feature_ts"])
        )
        return merged

    @staticmethod
    def validate_no_future_data(
        df: pd.DataFrame, reference_timestamp_col: str, check_columns: list[str]
    ) -> dict[str, Any]:
        issues: dict[str, Any] = {}
        for col in check_columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                future_rows = df[df[col] > df[reference_timestamp_col]]
                if len(future_rows) > 0:
                    issues[col] = {
                        "violation_count": len(future_rows),
                        "example_ids": future_rows.index[:5].tolist()
                    }
        return issues

    @staticmethod
    def asof_join(
        left_df: pd.DataFrame, right_df: pd.DataFrame, entity_key: str,
        left_time_col: str, right_time_col: str, direction: str = "backward"
    ) -> pd.DataFrame:
        left_sorted = left_df.sort_values(left_time_col).reset_index(drop=True)
        right_sorted = right_df.sort_values(right_time_col).reset_index(drop=True)

        return pd.merge_asof(
            left_sorted, right_sorted, left_on=left_time_col, right_on=right_time_col,
            by=entity_key, direction=direction
        )

    @staticmethod
    def create_training_labels(
        df: pd.DataFrame, label_column: str, entity_key: str = "entity_id"
    ) -> pd.DataFrame:
        if "label_available_flag" not in df.columns:
            raise ValueError("DataFrame must have label_available_flag column")

        training_df = df[df["label_available_flag"] == True].copy()
        if label_column not in training_df.columns:
            raise ValueError(f"Label column '{label_column}' not found")

        return training_df[[entity_key, "feature_timestamp", "label_timestamp", label_column]]

    @staticmethod
    def validate_temporal_integrity(df: pd.DataFrame) -> dict[str, Any]:
        report = {"valid": True, "issues": []}

        if "feature_timestamp" in df.columns and "label_timestamp" in df.columns:
            violations = df[df["feature_timestamp"] > df["label_timestamp"]]
            if len(violations) > 0:
                report["valid"] = False
                report["issues"].append({
                    "type": "feature_after_label",
                    "count": len(violations),
                    "message": f"{len(violations)} rows have feature_timestamp > label_timestamp"
                })

        datetime_cols = df.select_dtypes(include=["datetime64"]).columns
        for col in datetime_cols:
            if col in ["feature_timestamp", "label_timestamp"]:
                continue
            if "feature_timestamp" in df.columns:
                future = df[df[col] > df["feature_timestamp"]]
                if len(future) > 0:
                    report["valid"] = False
                    report["issues"].append({
                        "type": "future_data",
                        "column": col,
                        "count": len(future),
                        "message": f"Column {col} has {len(future)} values after feature_timestamp"
                    })

        return report
