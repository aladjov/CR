"""
Time series detection and validation for exploratory data analysis.

This module provides detection of time series data patterns and
quality validation specific to temporal datasets.
"""

import warnings
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
from enum import Enum
from datetime import timedelta

from customer_retention.core.compat import pd, DataFrame


class DatasetType(Enum):
    """Classification of dataset structure."""
    SNAPSHOT = "snapshot"           # Single row per entity (point-in-time)
    TIME_SERIES = "time_series"     # Multiple rows per entity over time
    EVENT_LOG = "event_log"         # Irregular events per entity
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

    # Entity statistics
    total_entities: int = 0
    min_observations_per_entity: int = 0
    max_observations_per_entity: int = 0
    avg_observations_per_entity: float = 0.0
    median_observations_per_entity: float = 0.0

    # Temporal statistics
    time_span_days: float = 0.0
    detected_frequency: TimeSeriesFrequency = TimeSeriesFrequency.UNKNOWN
    median_interval_hours: float = 0.0

    # Quality indicators
    entities_with_single_observation: int = 0
    entities_with_gaps: int = 0
    duplicate_timestamps_count: int = 0

    confidence: float = 0.0  # 0-1 confidence in detection
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
    # Temporal coverage
    total_expected_periods: int = 0
    total_actual_periods: int = 0
    coverage_percentage: float = 100.0

    # Gap analysis
    entities_with_gaps: int = 0
    total_gaps: int = 0
    max_gap_periods: int = 0
    gap_examples: List[Dict[str, Any]] = field(default_factory=list)

    # Duplicate timestamps
    entities_with_duplicate_timestamps: int = 0
    total_duplicate_timestamps: int = 0
    duplicate_examples: List[Dict[str, Any]] = field(default_factory=list)

    # Temporal ordering
    entities_with_ordering_issues: int = 0
    ordering_issue_examples: List[Dict[str, Any]] = field(default_factory=list)

    # Frequency consistency
    frequency_consistent: bool = True
    frequency_deviation_percentage: float = 0.0

    # Overall quality score for time series aspects
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

    # Common timestamp column name patterns
    TIMESTAMP_PATTERNS = [
        'date', 'time', 'timestamp', 'datetime', 'created', 'updated',
        'event_date', 'transaction_date', 'order_date', 'period',
        'month', 'year', 'week', 'day', 'ts', 'dt'
    ]

    # Common entity/ID column name patterns
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

        # Auto-detect entity column if not provided
        if entity_column is None:
            entity_column = self._detect_entity_column(df)
            if entity_column:
                evidence.append(f"Auto-detected entity column: {entity_column}")

        # Auto-detect timestamp column if not provided
        if timestamp_column is None:
            timestamp_column = self._detect_timestamp_column(df)
            if timestamp_column:
                evidence.append(f"Auto-detected timestamp column: {timestamp_column}")

        # If we can't detect both, return as unknown
        if entity_column is None or entity_column not in df.columns:
            return TimeSeriesCharacteristics(
                is_time_series=False,
                dataset_type=DatasetType.UNKNOWN,
                confidence=0.0,
                evidence=["Could not detect entity column"]
            )

        # Calculate entity statistics
        entity_counts = df[entity_column].value_counts()
        total_entities = len(entity_counts)

        # Handle empty dataframe
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

        # Determine dataset type based on observations per entity
        if avg_obs < min_observations_threshold:
            # Mostly single observations - likely snapshot data
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

        # Multiple observations per entity - analyze temporal aspects
        time_span_days = 0.0
        detected_frequency = TimeSeriesFrequency.UNKNOWN
        median_interval_hours = 0.0
        duplicate_timestamps = 0
        entities_with_gaps = 0

        if timestamp_column and timestamp_column in df.columns:
            # Convert to datetime if needed
            ts_series = pd.to_datetime(
                df[timestamp_column], errors='coerce', format='mixed'
            )
            valid_ts = ts_series.notna()

            if valid_ts.sum() > 0:
                time_span = ts_series.max() - ts_series.min()
                time_span_days = time_span.total_seconds() / 86400
                evidence.append(f"Time span: {time_span_days:.1f} days")

                # Detect frequency
                detected_frequency, median_interval_hours = self._detect_frequency(
                    df, entity_column, timestamp_column
                )
                evidence.append(f"Detected frequency: {detected_frequency.value}")

                # Check for duplicate timestamps per entity
                dup_check = df.groupby([entity_column, timestamp_column]).size()
                duplicate_timestamps = int((dup_check > 1).sum())
                if duplicate_timestamps > 0:
                    evidence.append(f"Found {duplicate_timestamps} duplicate timestamps")

        # Determine if this is time series or event log
        if detected_frequency == TimeSeriesFrequency.IRREGULAR:
            dataset_type = DatasetType.EVENT_LOG
            evidence.append("Irregular intervals suggest event log data")
        else:
            dataset_type = DatasetType.TIME_SERIES
            evidence.append("Regular intervals suggest time series data")

        # Calculate confidence
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

    def _detect_entity_column(self, df: DataFrame) -> Optional[str]:
        """Auto-detect the entity/ID column."""
        # First, look for columns matching common patterns
        for col in df.columns:
            col_lower = col.lower()
            for pattern in self.ENTITY_PATTERNS:
                if pattern in col_lower:
                    return col

        # Look for columns that might be identifiers based on characteristics
        for col in df.columns:
            if df[col].dtype == 'object' or df[col].dtype.name.startswith('int'):
                # High cardinality but not unique (multiple rows per entity)
                distinct_ratio = df[col].nunique() / len(df)
                if 0.01 < distinct_ratio < 0.9:  # Not constant, not unique
                    # Check if values repeat
                    if df[col].value_counts().max() > 1:
                        return col

        return None

    def _detect_timestamp_column(self, df: DataFrame) -> Optional[str]:
        """Auto-detect the timestamp column."""
        candidates = []

        for col in df.columns:
            col_lower = col.lower()

            # Check if column name matches timestamp patterns
            name_match = any(pattern in col_lower for pattern in self.TIMESTAMP_PATTERNS)

            # Check if column is datetime type
            is_datetime = pd.api.types.is_datetime64_any_dtype(df[col])

            # Try to parse as datetime
            can_parse = False
            if not is_datetime and df[col].dtype == 'object':
                try:
                    with warnings.catch_warnings():
                        warnings.filterwarnings('ignore', category=FutureWarning)
                        parsed = pd.to_datetime(
                            df[col].head(100), errors='coerce', format='mixed'
                        )
                    can_parse = parsed.notna().mean() > 0.8
                except Exception:
                    pass

            if is_datetime:
                candidates.append((col, 3))  # Highest priority
            elif name_match and can_parse:
                candidates.append((col, 2))
            elif name_match:
                candidates.append((col, 1))
            elif can_parse:
                candidates.append((col, 1))

        if candidates:
            # Return highest priority candidate
            candidates.sort(key=lambda x: x[1], reverse=True)
            return candidates[0][0]

        return None

    def _detect_frequency(
        self,
        df: DataFrame,
        entity_column: str,
        timestamp_column: str
    ) -> Tuple[TimeSeriesFrequency, float]:
        """Detect the frequency of the time series."""
        # Sample entities for efficiency
        sample_entities = df[entity_column].unique()[:100]

        intervals = []
        for entity in sample_entities:
            entity_data = df[df[entity_column] == entity].copy()
            if len(entity_data) < 2:
                continue

            ts = pd.to_datetime(
                entity_data[timestamp_column], errors='coerce', format='mixed'
            )
            ts = ts.dropna().sort_values()

            if len(ts) < 2:
                continue

            diffs = ts.diff().dropna()
            intervals.extend([d.total_seconds() / 3600 for d in diffs])  # Hours

        if not intervals:
            return TimeSeriesFrequency.UNKNOWN, 0.0

        median_hours = float(pd.Series(intervals).median())

        # Classify frequency based on median interval
        if median_hours < 2:
            freq = TimeSeriesFrequency.HOURLY
        elif 20 <= median_hours <= 28:
            freq = TimeSeriesFrequency.DAILY
        elif 144 <= median_hours <= 192:  # 6-8 days
            freq = TimeSeriesFrequency.WEEKLY
        elif 672 <= median_hours <= 768:  # 28-32 days
            freq = TimeSeriesFrequency.MONTHLY
        elif 2016 <= median_hours <= 2208:  # ~84-92 days
            freq = TimeSeriesFrequency.QUARTERLY
        elif 8400 <= median_hours <= 8880:  # ~350-370 days
            freq = TimeSeriesFrequency.YEARLY
        else:
            # Check variance to determine if irregular
            std_hours = float(pd.Series(intervals).std())
            cv = std_hours / median_hours if median_hours > 0 else 1
            if cv > 0.5:  # High coefficient of variation
                freq = TimeSeriesFrequency.IRREGULAR
            else:
                freq = TimeSeriesFrequency.IRREGULAR

        return freq, median_hours

    def _calculate_confidence(
        self,
        avg_observations: float,
        has_timestamp: bool,
        has_frequency: bool
    ) -> float:
        """Calculate confidence score for time series detection."""
        confidence = 0.5  # Base confidence

        # More observations per entity = higher confidence
        if avg_observations >= 10:
            confidence += 0.3
        elif avg_observations >= 5:
            confidence += 0.2
        elif avg_observations >= 2:
            confidence += 0.1

        # Having a timestamp column increases confidence
        if has_timestamp:
            confidence += 0.1

        # Having detected frequency increases confidence
        if has_frequency:
            confidence += 0.1

        return min(1.0, confidence)


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
        issues = []

        # Validate inputs
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

        # Convert timestamp
        df_copy = df.copy()
        df_copy['_ts'] = pd.to_datetime(
            df_copy[timestamp_column], errors='coerce', format='mixed'
        )

        # Check for duplicate timestamps per entity
        dup_result = self._check_duplicate_timestamps(df_copy, entity_column)
        if dup_result['total'] > 0:
            issues.append(
                f"{dup_result['total']} duplicate timestamps across "
                f"{dup_result['entities']} entities"
            )

        # Check temporal ordering
        order_result = self._check_ordering(df_copy, entity_column)
        if order_result['entities'] > 0:
            issues.append(
                f"{order_result['entities']} entities have ordering issues"
            )

        # Analyze gaps
        gap_result = self._analyze_gaps(
            df_copy, entity_column, expected_frequency, max_allowed_gap_periods
        )
        if gap_result['entities_with_gaps'] > 0:
            issues.append(
                f"{gap_result['entities_with_gaps']} entities have significant gaps"
            )

        # Calculate temporal quality score
        total_entities = df[entity_column].nunique()

        penalties = 0

        # Duplicate timestamp penalty
        dup_rate = dup_result['entities'] / total_entities if total_entities > 0 else 0
        if dup_rate > 0.1:
            penalties += 20
        elif dup_rate > 0.01:
            penalties += 10

        # Ordering issues penalty
        order_rate = order_result['entities'] / total_entities if total_entities > 0 else 0
        if order_rate > 0.1:
            penalties += 20
        elif order_rate > 0.01:
            penalties += 10

        # Gap penalty
        gap_rate = gap_result['entities_with_gaps'] / total_entities if total_entities > 0 else 0
        if gap_rate > 0.2:
            penalties += 20
        elif gap_rate > 0.1:
            penalties += 10
        elif gap_rate > 0.05:
            penalties += 5

        temporal_quality_score = max(0, 100 - penalties)

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

    def _check_duplicate_timestamps(
        self,
        df: DataFrame,
        entity_column: str
    ) -> Dict[str, Any]:
        """Check for duplicate timestamps within each entity."""
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

    def _check_ordering(
        self,
        df: DataFrame,
        entity_column: str
    ) -> Dict[str, Any]:
        """Check if timestamps are properly ordered within each entity."""
        entities_with_issues = []
        examples = []

        # Sample for efficiency
        sample_entities = df[entity_column].unique()[:1000]

        for entity in sample_entities:
            entity_data = df[df[entity_column] == entity]['_ts'].dropna()
            if len(entity_data) < 2:
                continue

            # Check if sorted
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

    def _analyze_gaps(
        self,
        df: DataFrame,
        entity_column: str,
        expected_frequency: Optional[str],
        max_allowed_gap_periods: int
    ) -> Dict[str, Any]:
        """Analyze gaps in time series."""
        # Determine expected interval
        if expected_frequency:
            expected_interval = self._frequency_to_timedelta(expected_frequency)
        else:
            # Estimate from data
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

        # Sample for efficiency
        sample_entities = df[entity_column].unique()[:500]

        for entity in sample_entities:
            entity_data = df[df[entity_column] == entity]['_ts'].dropna().sort_values()
            if len(entity_data) < 2:
                continue

            diffs = entity_data.diff().dropna()

            # Find gaps larger than allowed
            threshold = expected_interval * max_allowed_gap_periods
            large_gaps = diffs[diffs > threshold]

            if len(large_gaps) > 0:
                entities_with_gaps.append(entity)
                total_gaps += len(large_gaps)

                gap_periods = int((large_gaps.max() / expected_interval))
                max_gap = max(max_gap, gap_periods)

                if len(gap_examples) < 3:
                    gap_examples.append({
                        'entity': entity,
                        'gap_size': str(large_gaps.max()),
                        'gap_periods': gap_periods
                    })

        # Calculate coverage
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
        """Estimate the typical interval from the data."""
        intervals = []

        sample_entities = df[entity_column].unique()[:100]

        for entity in sample_entities:
            entity_data = df[df[entity_column] == entity]['_ts'].dropna().sort_values()
            if len(entity_data) < 2:
                continue

            diffs = entity_data.diff().dropna()
            intervals.extend(diffs.tolist())

        if not intervals:
            return None

        return pd.Series(intervals).median()
