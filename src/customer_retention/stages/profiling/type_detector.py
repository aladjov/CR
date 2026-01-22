import re
from typing import Optional

from customer_retention.core.compat import pd, Series, DataFrame, is_numeric_dtype, is_string_dtype, is_datetime64_any_dtype
from customer_retention.core.config.column_config import ColumnType, DatasetGranularity
from .profile_result import TypeInference, TypeConfidence, GranularityResult


class TypeDetector:
    IDENTIFIER_PATTERNS = ["id", "key", "code", "uuid", "guid"]
    TARGET_PATTERNS_PRIMARY = ["churned", "retained", "churn", "retention", "attrition"]
    TARGET_PATTERNS_SECONDARY = [
        "unsubscribe", "unsubscribed", "terminate", "terminated", "cancel", "cancelled",
        "close", "closed", "discontinue", "discontinued", "exit", "exited", "leave", "left",
    ]
    TARGET_PATTERNS_GENERIC = ["target", "label", "outcome", "class", "flag"]
    CYCLICAL_DAY_PATTERNS = ["mon", "tue", "wed", "thu", "fri", "sat", "sun", "monday", "tuesday", "wednesday"]
    CYCLICAL_MONTH_PATTERNS = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]

    def __init__(self):
        self.evidence = []

    def detect_type(self, series: pd.Series, column_name: str) -> TypeInference:
        self.evidence = []

        if self.is_identifier(series, column_name):
            return TypeInference(
                inferred_type=ColumnType.IDENTIFIER,
                confidence=TypeConfidence.HIGH,
                evidence=self.evidence.copy()
            )

        if self.is_target(series, column_name):
            return TypeInference(
                inferred_type=ColumnType.TARGET,
                confidence=TypeConfidence.HIGH,
                evidence=self.evidence.copy()
            )

        if self.is_binary(series):
            return TypeInference(
                inferred_type=ColumnType.BINARY,
                confidence=TypeConfidence.HIGH,
                evidence=self.evidence.copy()
            )

        if self.is_datetime(series):
            return TypeInference(
                inferred_type=ColumnType.DATETIME,
                confidence=TypeConfidence.HIGH,
                evidence=self.evidence.copy()
            )

        if is_numeric_dtype(series):
            return self.detect_numeric_type(series)

        if is_string_dtype(series) or series.dtype == object:
            return self.detect_categorical_type(series)

        return TypeInference(
            inferred_type=ColumnType.UNKNOWN,
            confidence=TypeConfidence.LOW,
            evidence=["Could not determine type"]
        )

    def is_identifier(self, series: pd.Series, column_name: str) -> bool:
        column_lower = column_name.lower()
        if any(pattern in column_lower for pattern in self.IDENTIFIER_PATTERNS):
            self.evidence.append(f"Column name contains identifier pattern")
            return True

        if len(series) == 0:
            return False

        if is_datetime64_any_dtype(series):
            return False

        if is_numeric_dtype(series):
            return False

        distinct_count = series.nunique()
        distinct_ratio = distinct_count / len(series)

        if distinct_ratio == 1.0 and distinct_count <= 100:
            if series.dtype == object:
                sample = series.dropna().head(100)
                if len(sample) > 0:
                    parseable_count = 0
                    for value in sample:
                        try:
                            pd.to_datetime(value, format='mixed')
                            parseable_count += 1
                        except (ValueError, TypeError):
                            pass

                    if parseable_count / len(sample) > 0.8:
                        return False

            self.evidence.append("All values are unique (100%)")
            return True

        return False

    def is_target(self, series: pd.Series, column_name: str) -> bool:
        column_lower = column_name.lower()
        distinct_count = series.nunique()
        if distinct_count > 10:
            return False

        for pattern in self.TARGET_PATTERNS_PRIMARY:
            if pattern in column_lower:
                self.evidence.append(f"Column name contains primary target pattern '{pattern}' with {distinct_count} classes")
                return True

        for pattern in self.TARGET_PATTERNS_SECONDARY:
            if pattern in column_lower:
                self.evidence.append(f"Column name contains secondary target pattern '{pattern}' with {distinct_count} classes")
                return True

        for pattern in self.TARGET_PATTERNS_GENERIC:
            if pattern in column_lower:
                self.evidence.append(f"Column name contains generic target pattern '{pattern}' with {distinct_count} classes")
                return True

        return False

    def is_binary(self, series: pd.Series) -> bool:
        distinct_count = series.nunique()
        if distinct_count != 2:
            return False

        unique_values = set(series.dropna().unique())

        binary_sets = [
            {0, 1}, {0.0, 1.0},
            {True, False},
            {"0", "1"},
            {"yes", "no"}, {"Yes", "No"}, {"YES", "NO"},
            {"true", "false"}, {"True", "False"}, {"TRUE", "FALSE"},
            {"y", "n"}, {"Y", "N"}
        ]

        for binary_set in binary_sets:
            if unique_values == binary_set or unique_values.issubset(binary_set):
                self.evidence.append(f"Exactly 2 unique values: {unique_values}")
                return True

        if distinct_count == 2:
            self.evidence.append(f"Exactly 2 unique values (non-standard): {unique_values}")
            return True

        return False

    def is_datetime(self, series: pd.Series) -> bool:
        if is_datetime64_any_dtype(series):
            self.evidence.append("Column is datetime dtype")
            return True

        if series.dtype == object:
            sample = series.dropna().head(100)
            if len(sample) == 0:
                return False

            parseable_count = 0
            for value in sample:
                try:
                    pd.to_datetime(value, format='mixed')
                    parseable_count += 1
                except (ValueError, TypeError):
                    pass

            if parseable_count / len(sample) > 0.8:
                self.evidence.append(f"{parseable_count}/{len(sample)} values parseable as datetime")
                return True

        return False

    def detect_numeric_type(self, series: pd.Series) -> TypeInference:
        distinct_count = series.nunique()

        if distinct_count <= 20:
            self.evidence.append(f"Numeric with {distinct_count} unique values (≤20)")
            return TypeInference(
                inferred_type=ColumnType.NUMERIC_DISCRETE,
                confidence=TypeConfidence.MEDIUM,
                evidence=self.evidence.copy(),
                alternatives=[ColumnType.NUMERIC_CONTINUOUS]
            )

        self.evidence.append(f"Numeric with {distinct_count} unique values (>20)")
        return TypeInference(
            inferred_type=ColumnType.NUMERIC_CONTINUOUS,
            confidence=TypeConfidence.HIGH,
            evidence=self.evidence.copy()
        )

    def detect_categorical_type(self, series: pd.Series) -> TypeInference:
        if len(series) == 0 or series.dropna().empty:
            return TypeInference(
                inferred_type=ColumnType.UNKNOWN,
                confidence=TypeConfidence.LOW,
                evidence=["Empty or all-null series"]
            )

        distinct_count = series.nunique()

        if self.is_cyclical_pattern(series):
            return TypeInference(
                inferred_type=ColumnType.CATEGORICAL_CYCLICAL,
                confidence=TypeConfidence.MEDIUM,
                evidence=self.evidence.copy()
            )

        if distinct_count <= 10:
            self.evidence.append(f"String with {distinct_count} unique values (≤10)")
            return TypeInference(
                inferred_type=ColumnType.CATEGORICAL_NOMINAL,
                confidence=TypeConfidence.HIGH,
                evidence=self.evidence.copy()
            )

        if distinct_count <= 100:
            self.evidence.append(f"String with {distinct_count} unique values (≤100)")
            return TypeInference(
                inferred_type=ColumnType.CATEGORICAL_NOMINAL,
                confidence=TypeConfidence.MEDIUM,
                evidence=self.evidence.copy()
            )

        self.evidence.append(f"String with {distinct_count} unique values (>100)")
        return TypeInference(
            inferred_type=ColumnType.TEXT,
            confidence=TypeConfidence.MEDIUM,
            evidence=self.evidence.copy(),
            alternatives=[ColumnType.CATEGORICAL_NOMINAL]
        )

    def is_cyclical_pattern(self, series: pd.Series) -> bool:
        sample_values = [str(v).lower() for v in series.dropna().unique()[:20]]

        if len(sample_values) == 0:
            return False

        day_matches = sum(1 for v in sample_values if any(day in v for day in self.CYCLICAL_DAY_PATTERNS))
        if day_matches >= min(3, len(sample_values)):
            self.evidence.append("Contains day name patterns (cyclical)")
            return True

        month_matches = sum(1 for v in sample_values if any(month in v for month in self.CYCLICAL_MONTH_PATTERNS))
        if month_matches >= min(3, len(sample_values)):
            self.evidence.append("Contains month name patterns (cyclical)")
            return True

        return False

    def detect_granularity(self, df: DataFrame) -> GranularityResult:
        """Detect whether dataset is entity-level or event-level (time series)."""
        evidence = []

        if df is None or len(df) == 0 or len(df.columns) == 0:
            return GranularityResult(
                granularity=DatasetGranularity.UNKNOWN,
                evidence=["Empty or invalid DataFrame"]
            )

        entity_column = self._detect_entity_column(df)
        time_column = self._detect_time_column(df)

        if entity_column is None:
            evidence.append("No clear entity/ID column detected")
            return GranularityResult(
                granularity=DatasetGranularity.UNKNOWN,
                evidence=evidence
            )

        unique_entities = df[entity_column].nunique()
        total_rows = len(df)
        avg_events = total_rows / unique_entities if unique_entities > 0 else 0

        if unique_entities == total_rows:
            evidence.append(f"Each {entity_column} appears exactly once")
            return GranularityResult(
                granularity=DatasetGranularity.ENTITY_LEVEL,
                entity_column=entity_column,
                time_column=time_column,
                unique_entities=unique_entities,
                total_rows=total_rows,
                avg_events_per_entity=1.0,
                evidence=evidence
            )

        if avg_events > 1.5 and time_column is not None:
            evidence.append(f"Multiple rows per {entity_column} (avg {avg_events:.1f})")
            evidence.append(f"Temporal column detected: {time_column}")
            return GranularityResult(
                granularity=DatasetGranularity.EVENT_LEVEL,
                entity_column=entity_column,
                time_column=time_column,
                unique_entities=unique_entities,
                total_rows=total_rows,
                avg_events_per_entity=round(avg_events, 2),
                evidence=evidence
            )

        if avg_events > 1.5:
            evidence.append(f"Multiple rows per {entity_column} but no datetime column")
            return GranularityResult(
                granularity=DatasetGranularity.EVENT_LEVEL,
                entity_column=entity_column,
                time_column=None,
                unique_entities=unique_entities,
                total_rows=total_rows,
                avg_events_per_entity=round(avg_events, 2),
                evidence=evidence
            )

        evidence.append("Could not determine granularity with confidence")
        return GranularityResult(
            granularity=DatasetGranularity.UNKNOWN,
            entity_column=entity_column,
            time_column=time_column,
            evidence=evidence
        )

    def _detect_entity_column(self, df: DataFrame) -> Optional[str]:
        """Find the most likely entity/ID column."""
        candidates = []

        for col in df.columns:
            col_lower = col.lower()

            if any(pattern in col_lower for pattern in self.IDENTIFIER_PATTERNS):
                unique_ratio = df[col].nunique() / len(df)
                if 0.01 < unique_ratio < 1.0:
                    candidates.append((col, unique_ratio, "name_match"))
                elif unique_ratio == 1.0:
                    candidates.append((col, unique_ratio, "unique_id"))

        if not candidates:
            for col in df.columns:
                if df[col].dtype == object or str(df[col].dtype).startswith("str"):
                    unique_ratio = df[col].nunique() / len(df)
                    if 0.01 < unique_ratio < 0.5:
                        candidates.append((col, unique_ratio, "string_repeating"))

        if not candidates:
            return None

        for col, ratio, match_type in candidates:
            if match_type == "name_match" and ratio < 1.0:
                return col

        for col, ratio, match_type in candidates:
            if match_type == "unique_id":
                return col

        return candidates[0][0] if candidates else None

    def _detect_time_column(self, df: DataFrame) -> Optional[str]:
        """Find the most likely datetime/timestamp column."""
        for col in df.columns:
            if is_datetime64_any_dtype(df[col]):
                return col

        datetime_patterns = ["date", "time", "timestamp", "created", "updated", "sent", "event"]
        for col in df.columns:
            col_lower = col.lower()
            if any(pattern in col_lower for pattern in datetime_patterns):
                if df[col].dtype == object:
                    sample = df[col].dropna().head(20)
                    if len(sample) > 0:
                        parseable = 0
                        for val in sample:
                            try:
                                pd.to_datetime(val, format='mixed')
                                parseable += 1
                            except (ValueError, TypeError):
                                pass
                        if parseable / len(sample) > 0.8:
                            return col

        return None
