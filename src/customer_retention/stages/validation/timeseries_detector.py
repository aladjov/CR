"""
Time series detection and validation for exploratory data analysis.

This module provides detection of time series data patterns and
quality validation specific to temporal datasets.
"""

import warnings
from dataclasses import dataclass, field
from datetime import timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from customer_retention.core.compat import (
    DataFrame,
    _is_spark_pandas,
    as_spark_df,
    head_as_list,
    is_datetime64_any_dtype,
    pd,
    timestamp_diffs_seconds,
    to_datetime,
)


class DatasetType(Enum):
    """Classification of dataset structure."""
    SNAPSHOT = "snapshot"
    TIME_SERIES = "time_series"
    EVENT_LOG = "event_log"
    UNKNOWN = "unknown"


class TimeSeriesFrequency(Enum):
    """Detected frequency of time series."""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"
    HOURLY = "hourly"
    IRREGULAR = "irregular"
    UNKNOWN = "unknown"


@dataclass
class TimeSeriesCharacteristics:
    """Characteristics of detected time series data."""
    is_time_series: bool
    dataset_type: DatasetType
    entity_column: Optional[str] = None
    timestamp_column: Optional[str] = None

    total_entities: int = 0
    min_observations_per_entity: int = 0
    max_observations_per_entity: int = 0
    avg_observations_per_entity: float = 0.0
    median_observations_per_entity: float = 0.0

    time_span_days: float = 0.0
    detected_frequency: TimeSeriesFrequency = TimeSeriesFrequency.UNKNOWN
    median_interval_hours: float = 0.0

    entities_with_single_observation: int = 0
    entities_with_gaps: int = 0
    duplicate_timestamps_count: int = 0

    confidence: float = 0.0
    evidence: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "is_time_series": self.is_time_series,
            "dataset_type": self.dataset_type.value,
            "entity_column": self.entity_column,
            "timestamp_column": self.timestamp_column,
            "total_entities": self.total_entities,
            "avg_observations_per_entity": round(self.avg_observations_per_entity, 2),
            "time_span_days": round(self.time_span_days, 1),
            "detected_frequency": self.detected_frequency.value,
            "confidence": round(self.confidence, 2),
            "evidence": self.evidence
        }


@dataclass
class TimeSeriesValidationResult:
    """Result of time series quality validation."""
    total_expected_periods: int = 0
    total_actual_periods: int = 0
    coverage_percentage: float = 100.0

    entities_with_gaps: int = 0
    total_gaps: int = 0
    max_gap_periods: int = 0
    gap_examples: List[Dict[str, Any]] = field(default_factory=list)

    entities_with_duplicate_timestamps: int = 0
    total_duplicate_timestamps: int = 0
    duplicate_examples: List[Dict[str, Any]] = field(default_factory=list)

    entities_with_ordering_issues: int = 0
    ordering_issue_examples: List[Dict[str, Any]] = field(default_factory=list)

    frequency_consistent: bool = True
    frequency_deviation_percentage: float = 0.0

    temporal_quality_score: float = 100.0
    issues: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "coverage_percentage": round(self.coverage_percentage, 2),
            "entities_with_gaps": self.entities_with_gaps,
            "total_gaps": self.total_gaps,
            "entities_with_duplicate_timestamps": self.entities_with_duplicate_timestamps,
            "total_duplicate_timestamps": self.total_duplicate_timestamps,
            "frequency_consistent": self.frequency_consistent,
            "temporal_quality_score": round(self.temporal_quality_score, 1),
            "issues": self.issues
        }


class TimeSeriesDetector:
    """
    Detect time series patterns in datasets.

    Analyzes a dataset to determine if it represents:
    - Snapshot data (single observation per entity)
    - Time series data (multiple observations per entity over time)
    - Event log data (irregular events per entity)

    Example
    -------
    >>> detector = TimeSeriesDetector()
    >>> result = detector.detect(df, entity_column='customer_id')
    >>> if result.is_time_series:
    ...     print(f"Time series detected with {result.avg_observations_per_entity:.1f} obs/entity")
    """

    TIMESTAMP_PATTERNS = [
        'date', 'time', 'timestamp', 'datetime', 'created', 'updated',
        'event_date', 'transaction_date', 'order_date', 'period',
        'month', 'year', 'week', 'day', 'ts', 'dt'
    ]

    ENTITY_PATTERNS = [
        'id', 'customer_id', 'user_id', 'account_id', 'entity_id',
        'custid', 'userid', 'client_id', 'member_id', 'subscriber_id'
    ]

    def detect(
        self,
        df: DataFrame,
        entity_column: Optional[str] = None,
        timestamp_column: Optional[str] = None,
        min_observations_threshold: int = 2
    ) -> TimeSeriesCharacteristics:
        """
        Detect if dataset contains time series data.

        Parameters
        ----------
        df : DataFrame
            Data to analyze
        entity_column : str, optional
            Column identifying entities (e.g., customer_id).
            If not provided, will attempt to auto-detect.
        timestamp_column : str, optional
            Column containing timestamps.
            If not provided, will attempt to auto-detect.
        min_observations_threshold : int
            Minimum average observations per entity to classify as time series

        Returns
        -------
        TimeSeriesCharacteristics
            Detected characteristics of the dataset
        """
        evidence = []

        if entity_column is None:
            entity_column = self._detect_entity_column(df)
            if entity_column:
                evidence.append(f"Auto-detected entity column: {entity_column}")

        if timestamp_column is None:
            timestamp_column = self._detect_timestamp_column(df)
            if timestamp_column:
                evidence.append(f"Auto-detected timestamp column: {timestamp_column}")

        if entity_column is None or entity_column not in df.columns:
            return TimeSeriesCharacteristics(
                is_time_series=False,
                dataset_type=DatasetType.UNKNOWN,
                confidence=0.0,
                evidence=["Could not detect entity column"]
            )

        entity_counts = df[entity_column].value_counts()
        total_entities = len(entity_counts)

        if total_entities == 0:
            return TimeSeriesCharacteristics(
                is_time_series=False,
                dataset_type=DatasetType.SNAPSHOT,
                entity_column=entity_column,
                timestamp_column=timestamp_column,
                total_entities=0,
                confidence=0.0,
                evidence=["Empty dataset - no entities found"]
            )

        min_obs = int(entity_counts.min())
        max_obs = int(entity_counts.max())
        avg_obs = float(entity_counts.mean())
        median_obs = float(entity_counts.median())
        single_obs_entities = int((entity_counts == 1).sum())

        evidence.append(f"Found {total_entities:,} unique entities")
        evidence.append(f"Observations per entity: min={min_obs}, max={max_obs}, avg={avg_obs:.1f}")

        if avg_obs < min_observations_threshold:
            return TimeSeriesCharacteristics(
                is_time_series=False,
                dataset_type=DatasetType.SNAPSHOT,
                entity_column=entity_column,
                timestamp_column=timestamp_column,
                total_entities=total_entities,
                min_observations_per_entity=min_obs,
                max_observations_per_entity=max_obs,
                avg_observations_per_entity=avg_obs,
                median_observations_per_entity=median_obs,
                entities_with_single_observation=single_obs_entities,
                confidence=0.8 if avg_obs < 1.5 else 0.6,
                evidence=evidence + ["Dataset appears to be snapshot (single observation per entity)"]
            )

        time_span_days = 0.0
        detected_frequency = TimeSeriesFrequency.UNKNOWN
        median_interval_hours = 0.0
        duplicate_timestamps = 0

        if timestamp_column and timestamp_column in df.columns:
            time_span_days, detected_frequency, median_interval_hours, duplicate_timestamps = (
                self._analyze_temporal_aspects(
                    df, entity_column, timestamp_column, evidence
                )
            )

        if detected_frequency == TimeSeriesFrequency.IRREGULAR:
            dataset_type = DatasetType.EVENT_LOG
            evidence.append("Irregular intervals suggest event log data")
        else:
            dataset_type = DatasetType.TIME_SERIES
            evidence.append("Regular intervals suggest time series data")

        confidence = self._calculate_confidence(
            avg_obs, timestamp_column is not None,
            detected_frequency != TimeSeriesFrequency.UNKNOWN
        )

        return TimeSeriesCharacteristics(
            is_time_series=True,
            dataset_type=dataset_type,
            entity_column=entity_column,
            timestamp_column=timestamp_column,
            total_entities=total_entities,
            min_observations_per_entity=min_obs,
            max_observations_per_entity=max_obs,
            avg_observations_per_entity=avg_obs,
            median_observations_per_entity=median_obs,
            time_span_days=time_span_days,
            detected_frequency=detected_frequency,
            median_interval_hours=median_interval_hours,
            entities_with_single_observation=single_obs_entities,
            duplicate_timestamps_count=duplicate_timestamps,
            confidence=confidence,
            evidence=evidence
        )

    def _analyze_temporal_aspects(
        self,
        df: DataFrame,
        entity_column: str,
        timestamp_column: str,
        evidence: List[str],
    ) -> Tuple[float, TimeSeriesFrequency, float, int]:
        ts_series = to_datetime(
            df[timestamp_column], errors='coerce', format='mixed'
        )
        valid_ts = ts_series.notna()

        if valid_ts.sum() == 0:
            return 0.0, TimeSeriesFrequency.UNKNOWN, 0.0, 0

        time_span = ts_series.max() - ts_series.min()
        time_span_days = time_span.total_seconds() / 86400
        evidence.append(f"Time span: {time_span_days:.1f} days")

        detected_frequency, median_interval_hours = self._detect_frequency(
            df, entity_column, timestamp_column
        )
        evidence.append(f"Detected frequency: {detected_frequency.value}")

        dup_check = df.groupby([entity_column, timestamp_column]).size()
        duplicate_timestamps = int((dup_check > 1).sum())
        if duplicate_timestamps > 0:
            evidence.append(f"Found {duplicate_timestamps} duplicate timestamps")

        return time_span_days, detected_frequency, median_interval_hours, duplicate_timestamps

    def _detect_entity_column(self, df: DataFrame) -> Optional[str]:
        matched = self._find_column_by_name_pattern(df)
        if matched:
            return matched
        return self._find_column_by_cardinality(df)

    def _find_column_by_name_pattern(self, df: DataFrame) -> Optional[str]:
        for col in df.columns:
            col_lower = col.lower()
            for pattern in self.ENTITY_PATTERNS:
                if pattern in col_lower:
                    return col
        return None

    def _find_column_by_cardinality(self, df: DataFrame) -> Optional[str]:
        for col in df.columns:
            if df[col].dtype == 'object' or df[col].dtype.name.startswith('int'):
                distinct_ratio = df[col].nunique() / len(df)
                if 0.01 < distinct_ratio < 0.9:
                    if df[col].value_counts().max() > 1:
                        return col
        return None

    def _detect_timestamp_column(self, df: DataFrame) -> Optional[str]:
        candidates = []

        for col in df.columns:
            priority = self._timestamp_column_priority(df, col)
            if priority > 0:
                candidates.append((col, priority))

        if candidates:
            candidates.sort(key=lambda x: x[1], reverse=True)
            return candidates[0][0]

        return None

    def _timestamp_column_priority(self, df: DataFrame, col: str) -> int:
        col_lower = col.lower()
        name_match = any(pattern in col_lower for pattern in self.TIMESTAMP_PATTERNS)
        is_datetime = is_datetime64_any_dtype(df[col])

        if is_datetime:
            return 3

        can_parse = self._can_parse_as_datetime(df[col])

        if name_match and can_parse:
            return 2
        if name_match or can_parse:
            return 1
        return 0

    def _can_parse_as_datetime(self, series) -> bool:
        if series.dtype != 'object':
            return False
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=FutureWarning)
                parsed = to_datetime(
                    series.head(100), errors='coerce', format='mixed'
                )
            return parsed.notna().mean() > 0.8
        except (ValueError, TypeError, OverflowError):
            return False

    def _detect_frequency(
        self,
        df: DataFrame,
        entity_column: str,
        timestamp_column: str
    ) -> Tuple[TimeSeriesFrequency, float]:
        """Detect the frequency of the time series."""
        if _is_spark_pandas(df):
            return self._spark_detect_frequency(df, entity_column, timestamp_column)

        sample_entities = head_as_list(df[entity_column].unique(), 100)

        intervals = []
        for entity in sample_entities:
            entity_data = df[df[entity_column] == entity]
            if len(entity_data) < 2:
                continue

            ts = to_datetime(
                entity_data[timestamp_column], errors='coerce', format='mixed'
            )
            ts = ts.dropna().sort_values()

            if len(ts) < 2:
                continue

            diff_seconds = timestamp_diffs_seconds(ts).dropna()
            intervals.extend(head_as_list(diff_seconds / 3600, 10000))

        if not intervals:
            return TimeSeriesFrequency.UNKNOWN, 0.0

        median_hours = float(pd.Series(intervals).median())
        std_hours = float(pd.Series(intervals).std())
        return self._classify_frequency(median_hours, std_hours), median_hours

    def _classify_frequency(
        self, median_hours: float, std_hours: float
    ) -> TimeSeriesFrequency:
        if median_hours < 2:
            return TimeSeriesFrequency.HOURLY
        if 20 <= median_hours <= 28:
            return TimeSeriesFrequency.DAILY
        if 144 <= median_hours <= 192:
            return TimeSeriesFrequency.WEEKLY
        if 672 <= median_hours <= 768:
            return TimeSeriesFrequency.MONTHLY
        if 2016 <= median_hours <= 2208:
            return TimeSeriesFrequency.QUARTERLY
        if 8400 <= median_hours <= 8880:
            return TimeSeriesFrequency.YEARLY
        return TimeSeriesFrequency.IRREGULAR

    def _spark_detect_frequency(
        self,
        df: DataFrame,
        entity_column: str,
        timestamp_column: str,
    ) -> Tuple[TimeSeriesFrequency, float]:
        import pyspark.sql.functions as F  # noqa: N812
        from pyspark.sql.window import Window

        spark_df = as_spark_df(df)
        spark_df = spark_df.withColumn(
            "__ts__", F.to_timestamp(F.col(timestamp_column))
        ).filter(F.col("__ts__").isNotNull())

        w = Window.partitionBy(entity_column).orderBy("__ts__")
        diffs_df = (
            spark_df
            .withColumn("__prev__", F.lag("__ts__").over(w))
            .filter(F.col("__prev__").isNotNull())
            .withColumn(
                "__diff_h__",
                (F.unix_timestamp("__ts__") - F.unix_timestamp("__prev__"))
                .cast("double") / 3600.0,
            )
        )

        row = diffs_df.agg(
            F.percentile_approx("__diff_h__", 0.5).alias("med"),
            F.stddev("__diff_h__").alias("std"),
            F.count("__diff_h__").alias("cnt"),
        ).head()

        if row["cnt"] == 0:
            return TimeSeriesFrequency.UNKNOWN, 0.0

        median_hours = float(row["med"])
        std_hours = float(row["std"]) if row["std"] is not None else 0.0
        return self._classify_frequency(median_hours, std_hours), median_hours

    def _calculate_confidence(
        self,
        avg_observations: float,
        has_timestamp: bool,
        has_frequency: bool
    ) -> float:
        confidence = 0.5
        confidence += self._observation_count_bonus(avg_observations)
        if has_timestamp:
            confidence += 0.1
        if has_frequency:
            confidence += 0.1
        return min(1.0, confidence)

    def _observation_count_bonus(self, avg_observations: float) -> float:
        if avg_observations >= 10:
            return 0.3
        if avg_observations >= 5:
            return 0.2
        if avg_observations >= 2:
            return 0.1
        return 0.0


class TimeSeriesValidator:
    """
    Validate time series data quality.

    Performs quality checks specific to time series data:
    - Temporal coverage and gaps
    - Duplicate timestamps
    - Temporal ordering
    - Frequency consistency

    Example
    -------
    >>> validator = TimeSeriesValidator()
    >>> result = validator.validate(
    ...     df,
    ...     entity_column='customer_id',
    ...     timestamp_column='date',
    ...     expected_frequency='daily'
    ... )
    >>> print(f"Temporal quality: {result.temporal_quality_score:.1f}/100")
    """

    def validate(
        self,
        df: DataFrame,
        entity_column: str,
        timestamp_column: str,
        expected_frequency: Optional[str] = None,
        max_allowed_gap_periods: int = 3
    ) -> TimeSeriesValidationResult:
        """
        Validate time series data quality.

        Parameters
        ----------
        df : DataFrame
            Time series data to validate
        entity_column : str
            Column identifying entities
        timestamp_column : str
            Column containing timestamps
        expected_frequency : str, optional
            Expected frequency ('daily', 'weekly', 'monthly', etc.)
        max_allowed_gap_periods : int
            Maximum gap periods before flagging as issue

        Returns
        -------
        TimeSeriesValidationResult
            Validation results with quality metrics
        """
        if entity_column not in df.columns:
            return TimeSeriesValidationResult(
                temporal_quality_score=0,
                issues=[f"Entity column '{entity_column}' not found"]
            )

        if timestamp_column not in df.columns:
            return TimeSeriesValidationResult(
                temporal_quality_score=0,
                issues=[f"Timestamp column '{timestamp_column}' not found"]
            )

        df_copy = df.copy()
        df_copy['_ts'] = to_datetime(
            df_copy[timestamp_column], errors='coerce', format='mixed'
        )

        issues = []
        dup_result = self._check_duplicate_timestamps(df_copy, entity_column)
        if dup_result['total'] > 0:
            issues.append(
                f"{dup_result['total']} duplicate timestamps across "
                f"{dup_result['entities']} entities"
            )

        order_result = self._check_ordering(df_copy, entity_column)
        if order_result['entities'] > 0:
            issues.append(
                f"{order_result['entities']} entities have ordering issues"
            )

        gap_result = self._analyze_gaps(
            df_copy, entity_column, expected_frequency, max_allowed_gap_periods
        )
        if gap_result['entities_with_gaps'] > 0:
            issues.append(
                f"{gap_result['entities_with_gaps']} entities have significant gaps"
            )

        total_entities = df[entity_column].nunique()
        temporal_quality_score = self._compute_temporal_quality_score(
            total_entities, dup_result, order_result, gap_result
        )

        return TimeSeriesValidationResult(
            total_expected_periods=gap_result.get('expected_periods', 0),
            total_actual_periods=gap_result.get('actual_periods', 0),
            coverage_percentage=gap_result.get('coverage', 100.0),
            entities_with_gaps=gap_result['entities_with_gaps'],
            total_gaps=gap_result['total_gaps'],
            max_gap_periods=gap_result['max_gap'],
            gap_examples=gap_result['examples'],
            entities_with_duplicate_timestamps=dup_result['entities'],
            total_duplicate_timestamps=dup_result['total'],
            duplicate_examples=dup_result['examples'],
            entities_with_ordering_issues=order_result['entities'],
            ordering_issue_examples=order_result['examples'],
            frequency_consistent=gap_result.get('frequency_consistent', True),
            frequency_deviation_percentage=gap_result.get('frequency_deviation', 0.0),
            temporal_quality_score=temporal_quality_score,
            issues=issues
        )

    def _compute_temporal_quality_score(
        self,
        total_entities: int,
        dup_result: Dict[str, Any],
        order_result: Dict[str, Any],
        gap_result: Dict[str, Any],
    ) -> float:
        penalties = 0
        penalties += self._rate_penalty(
            dup_result['entities'], total_entities, high_threshold=0.1, low_threshold=0.01
        )
        penalties += self._rate_penalty(
            order_result['entities'], total_entities, high_threshold=0.1, low_threshold=0.01
        )
        penalties += self._gap_penalty(gap_result['entities_with_gaps'], total_entities)
        return max(0, 100 - penalties)

    def _rate_penalty(
        self, affected: int, total: int, high_threshold: float, low_threshold: float
    ) -> int:
        rate = affected / total if total > 0 else 0
        if rate > high_threshold:
            return 20
        if rate > low_threshold:
            return 10
        return 0

    def _gap_penalty(self, entities_with_gaps: int, total_entities: int) -> int:
        gap_rate = entities_with_gaps / total_entities if total_entities > 0 else 0
        if gap_rate > 0.2:
            return 20
        if gap_rate > 0.1:
            return 10
        if gap_rate > 0.05:
            return 5
        return 0

    def _check_duplicate_timestamps(
        self,
        df: DataFrame,
        entity_column: str
    ) -> Dict[str, Any]:
        if _is_spark_pandas(df):
            return self._spark_check_duplicate_timestamps(df, entity_column)

        dup_counts = df.groupby([entity_column, '_ts']).size()
        duplicates = dup_counts[dup_counts > 1]

        examples = []
        if len(duplicates) > 0:
            for (entity, ts), count in duplicates.head(3).items():
                examples.append({
                    'entity': entity,
                    'timestamp': str(ts),
                    'count': int(count)
                })

        return {
            'total': len(duplicates),
            'entities': duplicates.index.get_level_values(0).nunique() if len(duplicates) > 0 else 0,
            'examples': examples
        }

    def _spark_check_duplicate_timestamps(
        self, df: DataFrame, entity_column: str
    ) -> Dict[str, Any]:
        import pyspark.sql.functions as F  # noqa: N812

        spark_df = as_spark_df(df)
        dup_df = (
            spark_df
            .groupBy(entity_column, "_ts")
            .agg(F.count("*").alias("__cnt__"))
            .filter(F.col("__cnt__") > 1)
        )

        stats = dup_df.agg(
            F.count("*").alias("total"),
            F.countDistinct(entity_column).alias("entities"),
        ).head()

        total = int(stats["total"] or 0)
        entities = int(stats["entities"] or 0)

        examples = []
        if total > 0:
            for row in dup_df.limit(3).collect():
                examples.append({
                    'entity': row[entity_column],
                    'timestamp': str(row["_ts"]),
                    'count': int(row["__cnt__"]),
                })

        return {'total': total, 'entities': entities, 'examples': examples}

    def _check_ordering(
        self,
        df: DataFrame,
        entity_column: str
    ) -> Dict[str, Any]:
        if _is_spark_pandas(df):
            return self._spark_check_ordering(df, entity_column)

        entities_with_issues = []
        examples = []

        sample_entities = head_as_list(df[entity_column].unique(), 1000)

        for entity in sample_entities:
            entity_data = df[df[entity_column] == entity]['_ts'].dropna()
            if len(entity_data) < 2:
                continue

            if not entity_data.is_monotonic_increasing:
                entities_with_issues.append(entity)
                if len(examples) < 3:
                    examples.append({
                        'entity': entity,
                        'issue': 'timestamps not in ascending order'
                    })

        return {
            'entities': len(entities_with_issues),
            'examples': examples
        }

    def _spark_check_ordering(
        self, df: DataFrame, entity_column: str
    ) -> Dict[str, Any]:
        import pyspark.sql.functions as F  # noqa: N812
        from pyspark.sql.window import Window

        spark_df = as_spark_df(df).filter(F.col("_ts").isNotNull())
        spark_df = spark_df.withColumn("__rid__", F.monotonically_increasing_id())

        w = Window.partitionBy(entity_column).orderBy("__rid__")
        unordered = (
            spark_df
            .withColumn("__prev__", F.lag("_ts").over(w))
            .filter(F.col("__prev__").isNotNull())
            .filter(F.col("_ts") < F.col("__prev__"))
            .select(entity_column)
            .distinct()
        )

        entity_count = unordered.count()
        examples = []
        if entity_count > 0:
            for row in unordered.limit(3).collect():
                examples.append({
                    'entity': row[entity_column],
                    'issue': 'timestamps not in ascending order',
                })

        return {'entities': entity_count, 'examples': examples}

    def _analyze_gaps(
        self,
        df: DataFrame,
        entity_column: str,
        expected_frequency: Optional[str],
        max_allowed_gap_periods: int
    ) -> Dict[str, Any]:
        if _is_spark_pandas(df):
            return self._spark_analyze_gaps(
                df, entity_column, expected_frequency, max_allowed_gap_periods,
            )

        if expected_frequency:
            expected_interval = self._frequency_to_timedelta(expected_frequency)
        else:
            expected_interval = self._estimate_interval(df, entity_column)

        if expected_interval is None:
            return {
                'entities_with_gaps': 0,
                'total_gaps': 0,
                'max_gap': 0,
                'examples': [],
                'coverage': 100.0,
                'frequency_consistent': True,
                'frequency_deviation': 0.0
            }

        entities_with_gaps = []
        total_gaps = 0
        max_gap = 0
        gap_examples = []

        sample_entities = head_as_list(df[entity_column].unique(), 500)

        for entity in sample_entities:
            entity_data = df[df[entity_column] == entity]['_ts'].dropna().sort_values()
            if len(entity_data) < 2:
                continue

            diffs_sec = timestamp_diffs_seconds(entity_data).dropna()
            interval_sec = expected_interval.total_seconds()
            threshold_sec = interval_sec * max_allowed_gap_periods
            large_gaps = diffs_sec[diffs_sec > threshold_sec]

            if len(large_gaps) > 0:
                entities_with_gaps.append(entity)
                total_gaps += len(large_gaps)

                gap_periods = int(large_gaps.max() / interval_sec)
                max_gap = max(max_gap, gap_periods)

                if len(gap_examples) < 3:
                    gap_examples.append({
                        'entity': entity,
                        'gap_size': f"{large_gaps.max() / 86400:.1f} days",
                        'gap_periods': gap_periods,
                    })

        coverage = 100.0
        if len(sample_entities) > 0:
            coverage = 100.0 * (1 - len(entities_with_gaps) / len(sample_entities))

        return {
            'entities_with_gaps': len(entities_with_gaps),
            'total_gaps': total_gaps,
            'max_gap': max_gap,
            'examples': gap_examples,
            'coverage': coverage,
            'frequency_consistent': len(entities_with_gaps) < len(sample_entities) * 0.1,
            'frequency_deviation': 0.0,
            'expected_periods': 0,
            'actual_periods': 0
        }

    def _spark_analyze_gaps(
        self,
        df: DataFrame,
        entity_column: str,
        expected_frequency: Optional[str],
        max_allowed_gap_periods: int,
    ) -> Dict[str, Any]:
        import pyspark.sql.functions as F  # noqa: N812
        from pyspark.sql.window import Window

        _EMPTY = {
            'entities_with_gaps': 0, 'total_gaps': 0, 'max_gap': 0,
            'examples': [], 'coverage': 100.0,
            'frequency_consistent': True, 'frequency_deviation': 0.0,
            'expected_periods': 0, 'actual_periods': 0,
        }

        if expected_frequency:
            expected_interval = self._frequency_to_timedelta(expected_frequency)
        else:
            expected_interval = self._estimate_interval(df, entity_column)

        if expected_interval is None:
            return _EMPTY

        interval_sec = expected_interval.total_seconds()
        threshold_sec = interval_sec * max_allowed_gap_periods

        spark_df = as_spark_df(df).filter(F.col("_ts").isNotNull())
        w = Window.partitionBy(entity_column).orderBy("_ts")
        diffs_df = (
            spark_df
            .withColumn("__prev__", F.lag("_ts").over(w))
            .filter(F.col("__prev__").isNotNull())
            .withColumn(
                "__diff_sec__",
                (F.unix_timestamp("_ts") - F.unix_timestamp("__prev__"))
                .cast("double"),
            )
            .filter(F.col("__diff_sec__") > threshold_sec)
        )

        gap_stats = diffs_df.groupBy(entity_column).agg(
            F.count("__diff_sec__").alias("gap_count"),
            F.max("__diff_sec__").alias("max_gap_sec"),
        )

        agg_row = gap_stats.agg(
            F.count(entity_column).alias("ent"),
            F.coalesce(F.sum("gap_count"), F.lit(0)).alias("gaps"),
            F.max("max_gap_sec").alias("max_sec"),
        ).head()

        ent_with_gaps = int(agg_row["ent"] or 0)
        total_gaps_count = int(agg_row["gaps"] or 0)
        max_sec = float(agg_row["max_sec"]) if agg_row["max_sec"] is not None else 0
        max_gap_periods = int(max_sec / interval_sec) if interval_sec > 0 else 0

        examples = []
        if ent_with_gaps > 0:
            for row in gap_stats.limit(3).collect():
                examples.append({
                    'entity': row[entity_column],
                    'gap_size': f"{float(row['max_gap_sec']) / 86400:.1f} days",
                    'gap_periods': int(float(row['max_gap_sec']) / interval_sec),
                })

        total_entity_count = spark_df.select(entity_column).distinct().count()
        coverage = (
            100.0 * (1 - ent_with_gaps / total_entity_count)
            if total_entity_count > 0 else 100.0
        )

        return {
            'entities_with_gaps': ent_with_gaps,
            'total_gaps': total_gaps_count,
            'max_gap': max_gap_periods,
            'examples': examples,
            'coverage': coverage,
            'frequency_consistent': ent_with_gaps < total_entity_count * 0.1,
            'frequency_deviation': 0.0,
            'expected_periods': 0,
            'actual_periods': 0,
        }

    def _frequency_to_timedelta(self, frequency: str) -> Optional[timedelta]:
        """Convert frequency string to timedelta."""
        freq_map = {
            'hourly': timedelta(hours=1),
            'daily': timedelta(days=1),
            'weekly': timedelta(weeks=1),
            'monthly': timedelta(days=30),
            'quarterly': timedelta(days=91),
            'yearly': timedelta(days=365),
        }
        return freq_map.get(frequency.lower())

    def _estimate_interval(
        self,
        df: DataFrame,
        entity_column: str
    ) -> Optional[timedelta]:
        if _is_spark_pandas(df):
            return self._spark_estimate_interval(df, entity_column)

        intervals = []

        sample_entities = head_as_list(df[entity_column].unique(), 100)

        for entity in sample_entities:
            entity_data = df[df[entity_column] == entity]['_ts'].dropna().sort_values()
            if len(entity_data) < 2:
                continue

            diffs_sec = timestamp_diffs_seconds(entity_data).dropna()
            intervals.extend(head_as_list(diffs_sec, 10000))

        if not intervals:
            return None

        return timedelta(seconds=float(pd.Series(intervals).median()))

    def _spark_estimate_interval(
        self, df: DataFrame, entity_column: str
    ) -> Optional[timedelta]:
        import pyspark.sql.functions as F  # noqa: N812
        from pyspark.sql.window import Window

        spark_df = as_spark_df(df).filter(F.col("_ts").isNotNull())
        w = Window.partitionBy(entity_column).orderBy("_ts")
        diffs_df = (
            spark_df
            .withColumn("__prev__", F.lag("_ts").over(w))
            .filter(F.col("__prev__").isNotNull())
            .withColumn(
                "__diff_sec__",
                (F.unix_timestamp("_ts") - F.unix_timestamp("__prev__"))
                .cast("double"),
            )
        )

        row = diffs_df.agg(
            F.percentile_approx("__diff_sec__", 0.5).alias("med"),
            F.count("__diff_sec__").alias("cnt"),
        ).head()

        if row["cnt"] == 0:
            return None

        return timedelta(seconds=float(row["med"]))
