"""Automatic timestamp discovery for ML datasets.

This module provides intelligent detection of timestamp columns in datasets,
identifying which columns represent feature observation times vs. label
availability times. It supports:

- Direct datetime columns
- Unix timestamps (seconds or milliseconds)
- Derivable timestamps (e.g., calculating signup date from tenure)
- Pattern-based column name matching

Example:
    >>> from customer_retention.stages.temporal import TimestampDiscoveryEngine
    >>> engine = TimestampDiscoveryEngine()
    >>> result = engine.discover(df, target_column="churn")
    >>> print(f"Feature timestamp: {result.feature_timestamp.column_name}")
    >>> print(f"Label timestamp: {result.label_timestamp.column_name}")
    >>> print(f"Recommendation: {result.recommendation}")
"""

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Optional

import pandas as pd


class TimestampRole(Enum):
    """Role classification for timestamp columns.

    Attributes:
        FEATURE_TIMESTAMP: When features were observed (e.g., last_activity_date)
        LABEL_TIMESTAMP: When the label became known (e.g., churn_date)
        ENTITY_CREATED: When the entity was created (e.g., signup_date)
        ENTITY_UPDATED: When the entity was last updated
        EVENT_TIME: Generic event timestamp
        DERIVABLE: Can be derived from other columns
        UNKNOWN: Role could not be determined
    """
    FEATURE_TIMESTAMP = "feature_timestamp"
    LABEL_TIMESTAMP = "label_timestamp"
    ENTITY_CREATED = "entity_created"
    ENTITY_UPDATED = "entity_updated"
    EVENT_TIME = "event_time"
    DERIVABLE = "derivable"
    UNKNOWN = "unknown"


@dataclass
class TimestampCandidate:
    """A candidate column that may serve as a timestamp.

    Attributes:
        column_name: Name of the column (or derived name if is_derived=True)
        role: The inferred role for this timestamp
        confidence: Confidence score (0-1) in the role assignment
        coverage: Fraction of non-null values (0-1)
        date_range: Tuple of (min_date, max_date) for the values
        is_derived: Whether this timestamp is derived from other columns
        derivation_formula: Formula used to derive this timestamp
        source_columns: Columns used in derivation
        notes: Additional notes about the candidate
    """

    column_name: str
    role: TimestampRole
    confidence: float
    coverage: float
    date_range: tuple[Optional[datetime], Optional[datetime]]
    is_derived: bool = False
    derivation_formula: Optional[str] = None
    source_columns: list[str] = field(default_factory=list)
    notes: str = ""


@dataclass
class TimestampDiscoveryResult:
    """Result of timestamp discovery analysis.

    Attributes:
        feature_timestamp: Best candidate for feature timestamp, if found
        label_timestamp: Best candidate for label timestamp, if found
        all_candidates: All discovered timestamp candidates
        derivable_options: Candidates that can be derived from other columns
        recommendation: Human-readable recommendation string
        requires_synthetic: True if synthetic timestamps are needed
        discovery_report: Detailed report of the discovery process
    """

    feature_timestamp: Optional[TimestampCandidate]
    label_timestamp: Optional[TimestampCandidate]
    all_candidates: list[TimestampCandidate]
    derivable_options: list[TimestampCandidate]
    recommendation: str
    requires_synthetic: bool
    discovery_report: dict[str, Any]

    @property
    def datetime_columns(self) -> list[str]:
        """Get list of datetime column names (excluding feature/label timestamps).

        Returns column names of all datetime candidates that are not already
        selected as feature_timestamp or label_timestamp.
        """
        excluded = set()
        if self.feature_timestamp:
            excluded.add(self.feature_timestamp.column_name)
        if self.label_timestamp:
            excluded.add(self.label_timestamp.column_name)
        return [
            c.column_name for c in self.all_candidates
            if not c.is_derived and c.column_name not in excluded
        ]


def _looks_like_datetime_strings(sample: pd.Series) -> bool:
    if len(sample) == 0:
        return False
    str_sample = sample.astype(str)
    datetime_pattern = re.compile(
        r"\d{4}[-/]|\d{1,2}[-/]\d{1,2}[-/]|\d{1,2}:\d{2}|"
        r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)", re.IGNORECASE
    )
    matches = str_sample.apply(lambda x: bool(datetime_pattern.search(str(x))))
    return matches.mean() > 0.8


class DatetimeOrderAnalyzer:
    ACTIVITY_PATTERNS = [
        r"last_", r"latest_", r"recent_", r"final_", r"most_recent",
        r"lastorder", r"lastlogin", r"lastpurchase", r"lastvisit",
    ]

    def analyze_datetime_ordering(self, df: pd.DataFrame) -> list[str]:
        datetime_cols = self._get_datetime_columns(df)
        if not datetime_cols:
            return []
        median_dates = {}
        for col in datetime_cols:
            series = df[col].dropna()
            if not pd.api.types.is_datetime64_any_dtype(series):
                series = pd.to_datetime(series, format="mixed", errors="coerce")
            median_dates[col] = series.dropna().median()
        return sorted(datetime_cols, key=lambda c: median_dates[c])

    def find_latest_activity_column(self, df: pd.DataFrame) -> Optional[str]:
        datetime_cols = self._get_datetime_columns(df)
        if not datetime_cols:
            return None
        activity_cols = [c for c in datetime_cols if self._is_activity_column(c)]
        if activity_cols:
            return self._select_chronologically_latest(df, activity_cols)
        return self._select_chronologically_latest(df, datetime_cols)

    def find_earliest_column(self, df: pd.DataFrame) -> Optional[str]:
        ordering = self.analyze_datetime_ordering(df)
        return ordering[0] if ordering else None

    def derive_last_action_date(self, df: pd.DataFrame) -> Optional[pd.Series]:
        ordering = self.analyze_datetime_ordering(df)
        if not ordering:
            return None
        coalesced = self._coalesce_datetime_columns(df, list(reversed(ordering)))
        coalesced.name = "last_action_date"
        return coalesced

    def _coalesce_datetime_columns(self, df: pd.DataFrame, columns: list[str]) -> pd.Series:
        result = self._ensure_datetime_column(df, columns[0])
        for col in columns[1:]:
            result = result.fillna(self._ensure_datetime_column(df, col))
        return result

    def _ensure_datetime_column(self, df: pd.DataFrame, col: str) -> pd.Series:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            return df[col]
        return pd.to_datetime(df[col], format="mixed", errors="coerce")

    def _get_datetime_columns(self, df: pd.DataFrame) -> list[str]:
        result = []
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                result.append(col)
            elif df[col].dtype == object:
                sample = df[col].dropna().head(100)
                if _looks_like_datetime_strings(sample):
                    parsed = pd.to_datetime(sample, format="mixed", errors="coerce")
                    if parsed.notna().mean() > 0.8:
                        result.append(col)
        return result

    def _is_activity_column(self, col_name: str) -> bool:
        col_lower = col_name.lower()
        return any(re.search(p, col_lower) for p in self.ACTIVITY_PATTERNS)

    def _select_chronologically_latest(self, df: pd.DataFrame, cols: list[str]) -> str:
        max_dates = {}
        for col in cols:
            series = df[col].dropna()
            if not pd.api.types.is_datetime64_any_dtype(series):
                series = pd.to_datetime(series, format="mixed", errors="coerce")
            max_dates[col] = series.dropna().max()
        return max(cols, key=lambda c: max_dates[c])


class TimestampDiscoveryEngine:
    """Engine for automatically discovering timestamp columns in datasets.

    The discovery engine analyzes column names and values to identify which
    columns represent feature observation times vs. label availability times.
    It uses pattern matching on column names and validates data types.

    Example:
        >>> engine = TimestampDiscoveryEngine()
        >>> result = engine.discover(df, target_column="churn")
        >>> if result.requires_synthetic:
        ...     print("No timestamps found, will use synthetic")
        >>> else:
        ...     print(f"Using {result.feature_timestamp.column_name}")
    """
    FEATURE_TIMESTAMP_PATTERNS = [
        r"last_activity", r"last_login", r"last_purchase", r"last_order",
        r"last_seen", r"last_visit", r"last_interaction", r"last_transaction",
        r"snapshot_date", r"observation_date", r"record_date", r"as_of_date",
        r"updated_at", r"modified_date", r"last_updated", r"last_modified",
        r"effective_date", r"data_date", r"reporting_date",
    ]

    LABEL_TIMESTAMP_PATTERNS = [
        r"churn_date", r"churned_date", r"customer_churn_date", r"churn_timestamp",
        r"unsubscribe_date", r"unsubscribed_date", r"unsub_date",
        r"cancellation_date", r"cancel_date", r"cancelled_date",
        r"termination_date", r"terminate_date", r"terminated_date",
        r"discontinue_date", r"discontinued_date", r"discontinuation_date",
        r"close_date", r"closed_date", r"account_close_date", r"closure_date",
        r"end_date", r"exit_date", r"leave_date", r"left_date",
        r"expiry_date", r"expiration_date", r"expired_date",
        r"outcome_date", r"event_date", r"target_date", r"label_date", r"prediction_date",
    ]

    ENTITY_CREATED_PATTERNS = [
        r"signup_date", r"registration_date", r"created_at", r"create_date",
        r"join_date", r"account_created", r"first_order", r"first_purchase",
        r"onboarding_date", r"start_date", r"activation_date",
    ]

    TENURE_PATTERNS = [r"tenure", r"account_age", r"customer_age", r"months_active"]
    CONTRACT_PATTERNS = [r"contract_length", r"contract_duration", r"subscription_length"]

    def __init__(self, reference_date: Optional[datetime] = None, label_window_days: int = 180):
        self.reference_date = reference_date or datetime.now()
        self.label_window_days = label_window_days
        self.order_analyzer = DatetimeOrderAnalyzer()

    def discover(self, df: pd.DataFrame, target_column: Optional[str] = None) -> TimestampDiscoveryResult:
        datetime_candidates = self._discover_datetime_columns(df)
        derivable_candidates = self._discover_derivable_timestamps(df)
        all_candidates = datetime_candidates + derivable_candidates
        classified = self._classify_candidates(all_candidates)
        datetime_ordering = self.order_analyzer.analyze_datetime_ordering(df)

        feature_ts = self._select_best_candidate(classified, TimestampRole.FEATURE_TIMESTAMP)
        label_ts = self._select_best_candidate(classified, TimestampRole.LABEL_TIMESTAMP)

        if not feature_ts and datetime_ordering:
            feature_ts = self._promote_latest_to_feature(df, classified)

        if feature_ts and not label_ts:
            label_ts = self._derive_label_timestamp(feature_ts)

        recommendation, requires_synthetic = self._generate_recommendation(feature_ts, label_ts, all_candidates)
        discovery_report = self._build_report(df, datetime_candidates, derivable_candidates, classified)
        discovery_report["datetime_ordering"] = datetime_ordering

        return TimestampDiscoveryResult(
            feature_timestamp=feature_ts,
            label_timestamp=label_ts,
            all_candidates=all_candidates,
            derivable_options=derivable_candidates,
            recommendation=recommendation,
            requires_synthetic=requires_synthetic,
            discovery_report=discovery_report,
        )

    def _discover_datetime_columns(self, df: pd.DataFrame) -> list[TimestampCandidate]:
        return [c for col in df.columns if (c := self._analyze_column_for_datetime(df, col))]

    def _analyze_column_for_datetime(self, df: pd.DataFrame, col: str) -> Optional[TimestampCandidate]:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            return self._create_datetime_candidate(df, col)

        if df[col].dtype == object:
            sample = df[col].dropna().head(100)
            if _looks_like_datetime_strings(sample):
                parsed = pd.to_datetime(sample, format="mixed", errors="coerce")
                if parsed.notna().mean() > 0.8:
                    return self._create_datetime_candidate(df, col, needs_parsing=True)

        if pd.api.types.is_numeric_dtype(df[col]) and self._looks_like_unix_timestamp(df[col]):
            return self._create_datetime_candidate(df, col, is_unix=True)

        return None

    def _looks_like_unix_timestamp(self, series: pd.Series) -> bool:
        sample = series.dropna().head(100)
        if len(sample) == 0:
            return False
        mean_val = sample.mean()
        min_unix_seconds = 946684800  # 2000-01-01
        max_unix_seconds = 4102444800  # 2100-01-01
        min_unix_ms = min_unix_seconds * 1000
        max_unix_ms = max_unix_seconds * 1000
        is_seconds = min_unix_seconds < mean_val < max_unix_seconds
        is_milliseconds = min_unix_ms < mean_val < max_unix_ms
        return is_seconds or is_milliseconds

    def _create_datetime_candidate(
        self, df: pd.DataFrame, col: str, needs_parsing: bool = False, is_unix: bool = False
    ) -> TimestampCandidate:
        if is_unix:
            try:
                dt_series = pd.to_datetime(df[col], unit="s", errors="coerce")
            except Exception:
                dt_series = pd.to_datetime(df[col], unit="ms", errors="coerce")
        elif needs_parsing:
            dt_series = pd.to_datetime(df[col], format="mixed", errors="coerce")
        else:
            dt_series = df[col]

        coverage = float(dt_series.notna().mean())
        min_date = dt_series.min() if coverage > 0 else None
        max_date = dt_series.max() if coverage > 0 else None
        role = self._infer_role_from_name(col)
        confidence = self._calculate_confidence(col, role, coverage)

        return TimestampCandidate(
            column_name=col, role=role, confidence=confidence, coverage=coverage,
            date_range=(min_date, max_date), is_derived=False,
            notes=f"{'Unix timestamp' if is_unix else 'Datetime column'}",
        )

    def _discover_derivable_timestamps(self, df: pd.DataFrame) -> list[TimestampCandidate]:
        derivable = []
        for col in df.columns:
            col_lower = col.lower()
            if any(re.search(p, col_lower) for p in self.TENURE_PATTERNS):
                if pd.api.types.is_numeric_dtype(df[col]):
                    derivable.append(self._create_tenure_derived_candidate(df, col))
            if any(re.search(p, col_lower) for p in self.CONTRACT_PATTERNS):
                if pd.api.types.is_numeric_dtype(df[col]):
                    start_col = self._find_related_start_date(df, col)
                    if start_col:
                        derivable.append(self._create_contract_derived_candidate(df, col, start_col))
        return derivable

    def _create_tenure_derived_candidate(self, df: pd.DataFrame, tenure_col: str) -> TimestampCandidate:
        sample_tenure = df[tenure_col].dropna().head(100)
        avg_tenure = sample_tenure.mean() if len(sample_tenure) > 0 else 0

        max_val = sample_tenure.max() if len(sample_tenure) > 0 else 0
        min_val = sample_tenure.min() if len(sample_tenure) > 0 else 0
        min_signup = self.reference_date - timedelta(days=int(max_val * 30))
        max_signup = self.reference_date - timedelta(days=int(min_val * 30))

        return TimestampCandidate(
            column_name=f"derived_signup_date_from_{tenure_col}",
            role=TimestampRole.ENTITY_CREATED, confidence=0.7,
            coverage=float(df[tenure_col].notna().mean()), date_range=(min_signup, max_signup),
            is_derived=True, derivation_formula=f"reference_date - ({tenure_col} * 30 days)",
            source_columns=[tenure_col], notes=f"Derived from {tenure_col} (avg={avg_tenure:.1f} months)",
        )

    def _create_contract_derived_candidate(
        self, df: pd.DataFrame, length_col: str, start_col: str
    ) -> TimestampCandidate:
        return TimestampCandidate(
            column_name=f"derived_contract_end_from_{length_col}",
            role=TimestampRole.LABEL_TIMESTAMP, confidence=0.6,
            coverage=min(float(df[length_col].notna().mean()), float(df[start_col].notna().mean())),
            date_range=(None, None), is_derived=True,
            derivation_formula=f"{start_col} + ({length_col} * 30 days)",
            source_columns=[length_col, start_col],
            notes=f"Derived contract end from {start_col} + {length_col}",
        )

    def _find_related_start_date(self, df: pd.DataFrame, length_col: str) -> Optional[str]:
        for col in df.columns:
            if any(p in col.lower() for p in ["start", "begin", "signup", "created"]):
                if pd.api.types.is_datetime64_any_dtype(df[col]):
                    return col
                try:
                    pd.to_datetime(df[col].dropna().head(10), format="mixed")
                    return col
                except Exception:
                    pass
        return None

    def _infer_role_from_name(self, col_name: str) -> TimestampRole:
        col_lower = col_name.lower()
        for pattern in self.FEATURE_TIMESTAMP_PATTERNS:
            if re.search(pattern, col_lower):
                return TimestampRole.FEATURE_TIMESTAMP
        for pattern in self.LABEL_TIMESTAMP_PATTERNS:
            if re.search(pattern, col_lower):
                return TimestampRole.LABEL_TIMESTAMP
        for pattern in self.ENTITY_CREATED_PATTERNS:
            if re.search(pattern, col_lower):
                return TimestampRole.ENTITY_CREATED
        if re.search(r"update|modif", col_lower):
            return TimestampRole.ENTITY_UPDATED
        return TimestampRole.UNKNOWN

    def _calculate_confidence(self, col_name: str, role: TimestampRole, coverage: float) -> float:
        base = 0.5
        if role in [TimestampRole.FEATURE_TIMESTAMP, TimestampRole.LABEL_TIMESTAMP]:
            base += 0.3
        elif role == TimestampRole.ENTITY_CREATED:
            base += 0.2
        return min(base + coverage * 0.2, 1.0)

    def _classify_candidates(self, candidates: list[TimestampCandidate]) -> list[TimestampCandidate]:
        has_feature_ts = any(c.role == TimestampRole.FEATURE_TIMESTAMP for c in candidates)
        if not has_feature_ts:
            for c in candidates:
                if c.role == TimestampRole.ENTITY_UPDATED:
                    c.role = TimestampRole.FEATURE_TIMESTAMP
                    c.notes += " (promoted to feature_timestamp)"
                    break
        return candidates

    def _select_best_candidate(
        self, candidates: list[TimestampCandidate], role: TimestampRole
    ) -> Optional[TimestampCandidate]:
        matching = [c for c in candidates if c.role == role]
        if not matching:
            return None
        matching.sort(key=lambda c: (c.confidence, c.coverage), reverse=True)
        return matching[0]

    def _promote_latest_to_feature(
        self, df: pd.DataFrame, candidates: list[TimestampCandidate]
    ) -> Optional[TimestampCandidate]:
        latest_col = self.order_analyzer.find_latest_activity_column(df)
        if not latest_col:
            return None
        for c in candidates:
            if c.column_name == latest_col and c.role != TimestampRole.LABEL_TIMESTAMP:
                c.role = TimestampRole.FEATURE_TIMESTAMP
                c.notes += " (promoted: latest activity column)"
                c.confidence = max(c.confidence, 0.7)
                return c
        non_label_candidates = [c for c in candidates if c.role != TimestampRole.LABEL_TIMESTAMP]
        if non_label_candidates:
            best = max(non_label_candidates, key=lambda c: c.coverage)
            best.role = TimestampRole.FEATURE_TIMESTAMP
            best.notes += " (promoted: fallback latest)"
            best.confidence = max(best.confidence, 0.6)
            return best
        return None

    def _derive_label_timestamp(self, feature_ts: TimestampCandidate) -> TimestampCandidate:
        window = self.label_window_days
        min_date = feature_ts.date_range[0] + timedelta(days=window) if feature_ts.date_range[0] else None
        max_date = feature_ts.date_range[1] + timedelta(days=window) if feature_ts.date_range[1] else None

        return TimestampCandidate(
            column_name="derived_label_timestamp", role=TimestampRole.LABEL_TIMESTAMP,
            confidence=0.6, coverage=feature_ts.coverage, date_range=(min_date, max_date),
            is_derived=True, derivation_formula=f"{feature_ts.column_name} + {window} days",
            source_columns=[feature_ts.column_name],
            notes=f"Derived from feature_timestamp + {window}-day observation window",
        )

    def _generate_recommendation(
        self, feature_ts: Optional[TimestampCandidate], label_ts: Optional[TimestampCandidate],
        all_candidates: list[TimestampCandidate]
    ) -> tuple[str, bool]:
        if feature_ts and label_ts:
            derived_note = ""
            if feature_ts.is_derived:
                derived_note += f"\n  - feature_timestamp derived via: {feature_ts.derivation_formula}"
            if label_ts.is_derived:
                derived_note += f"\n  - label_timestamp derived via: {label_ts.derivation_formula}"
            return (
                f"RECOMMENDED: Use discovered timestamps\n"
                f"  - feature_timestamp: {feature_ts.column_name} (confidence: {feature_ts.confidence:.0%})\n"
                f"  - label_timestamp: {label_ts.column_name} (confidence: {label_ts.confidence:.0%})"
                f"{derived_note}",
                False
            )
        elif feature_ts:
            return (
                f"PARTIAL: Found feature_timestamp ({feature_ts.column_name}), "
                f"but no label_timestamp. Will derive from feature_timestamp + observation window.",
                False
            )
        elif all_candidates:
            return (
                f"WARNING: Found {len(all_candidates)} datetime column(s) but could not determine "
                f"feature/label timestamps. Manual review recommended.\n"
                f"Candidates: {[c.column_name for c in all_candidates]}",
                True
            )
        return (
            "FALLBACK: No datetime columns found. Using synthetic timestamps. "
            "This should be rare - verify the data truly has no temporal information.",
            True
        )

    def _build_report(
        self, df: pd.DataFrame, datetime_candidates: list[TimestampCandidate],
        derivable_candidates: list[TimestampCandidate], classified: list[TimestampCandidate]
    ) -> dict[str, Any]:
        return {
            "total_columns": len(df.columns),
            "datetime_columns_found": len(datetime_candidates),
            "derivable_timestamps_found": len(derivable_candidates),
            "candidates_by_role": {
                role.value: [c.column_name for c in classified if c.role == role]
                for role in TimestampRole
            },
            "all_candidates": [
                {
                    "column": c.column_name, "role": c.role.value, "confidence": c.confidence,
                    "coverage": c.coverage, "is_derived": c.is_derived,
                    "derivation": c.derivation_formula, "notes": c.notes,
                }
                for c in classified
            ],
        }
