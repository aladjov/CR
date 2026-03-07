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

from customer_retention.core.compat import (
    as_tz_naive,
    is_datetime64_any_dtype,
    native_pd,
    resolve_column_name,
    to_datetime,
)


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
        base_df: Any, feature_df: Any, entity_key: str,
        base_timestamp_col: str = "feature_timestamp", feature_timestamp_col: str = "feature_timestamp"
    ) -> Any:
        try:
            base_timestamp_col = resolve_column_name(base_df.columns, base_timestamp_col)
        except KeyError:
            raise ValueError(f"Base df missing timestamp column: {base_timestamp_col}")
        try:
            feature_timestamp_col = resolve_column_name(feature_df.columns, feature_timestamp_col)
        except KeyError:
            raise ValueError(f"Feature df missing timestamp column: {feature_timestamp_col}")

        feature_df = feature_df.rename(columns={feature_timestamp_col: "_feature_ts"})
        merged = base_df.merge(feature_df, on=entity_key, how="left")
        valid_mask = as_tz_naive(merged["_feature_ts"]) <= as_tz_naive(merged[base_timestamp_col])

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
        df: Any, reference_timestamp_col: str, check_columns: list[str]
    ) -> dict[str, Any]:
        issues: dict[str, Any] = {}
        for col in check_columns:
            if is_datetime64_any_dtype(df[col]):
                future_rows = df[as_tz_naive(df[col]) > as_tz_naive(df[reference_timestamp_col])]
                if len(future_rows) > 0:
                    issues[col] = {
                        "violation_count": len(future_rows),
                        "example_ids": future_rows.index[:5].tolist()
                    }
        return issues

    @staticmethod
    def _normalize_time_col(df: Any, col: str) -> Any:
        df = df.copy()
        df[col] = as_tz_naive(to_datetime(df[col])).astype("datetime64[ns]")
        return df

    @staticmethod
    def asof_join(
        left_df: Any, right_df: Any, entity_key: str,
        left_time_col: str, right_time_col: str, direction: str = "backward"
    ) -> Any:
        left_sorted = PointInTimeJoiner._normalize_time_col(
            left_df.sort_values(left_time_col).reset_index(drop=True), left_time_col,
        )
        right_sorted = PointInTimeJoiner._normalize_time_col(
            right_df.sort_values(right_time_col).reset_index(drop=True), right_time_col,
        )

        return native_pd.merge_asof(
            left_sorted, right_sorted, left_on=left_time_col, right_on=right_time_col,
            by=entity_key, direction=direction
        )

    @staticmethod
    def create_training_labels(
        df: Any, label_column: str, entity_key: str = "entity_id"
    ) -> Any:
        if "label_available_flag" not in df.columns:
            raise ValueError("DataFrame must have label_available_flag column")

        training_df = df[df["label_available_flag"] == True].copy()
        if label_column not in training_df.columns:
            raise ValueError(f"Label column '{label_column}' not found")

        return training_df[[entity_key, "feature_timestamp", "label_timestamp", label_column]]

    @staticmethod
    def validate_temporal_integrity(df: Any) -> dict[str, Any]:
        report = {"valid": True, "issues": []}

        ft_ok = "feature_timestamp" in df.columns and is_datetime64_any_dtype(df["feature_timestamp"])
        lt_ok = "label_timestamp" in df.columns and is_datetime64_any_dtype(df["label_timestamp"])

        if ft_ok and lt_ok:
            violations = df[as_tz_naive(df["feature_timestamp"]) > as_tz_naive(df["label_timestamp"])]
            if len(violations) > 0:
                report["valid"] = False
                report["issues"].append({
                    "type": "feature_after_label",
                    "count": len(violations),
                    "message": f"{len(violations)} rows have feature_timestamp > label_timestamp"
                })

        datetime_cols = [c for c in df.columns if is_datetime64_any_dtype(df[c])]
        for col in datetime_cols:
            if col in ["feature_timestamp", "label_timestamp"]:
                continue
            if ft_ok:
                future = df[as_tz_naive(df[col]) > as_tz_naive(df["feature_timestamp"])]
                if len(future) > 0:
                    report["valid"] = False
                    report["issues"].append({
                        "type": "future_data",
                        "column": col,
                        "count": len(future),
                        "message": f"Column {col} has {len(future)} values after feature_timestamp"
                    })

        return report
