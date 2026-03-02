"""Tests for time series detection and validation module."""

from datetime import datetime, timedelta
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from customer_retention.stages.validation.timeseries_detector import (
    DatasetType,
    TimeSeriesCharacteristics,
    TimeSeriesDetector,
    TimeSeriesFrequency,
    TimeSeriesValidationResult,
    TimeSeriesValidator,
)


class TestTimeSeriesDetector:
    """Tests for TimeSeriesDetector class."""

    @pytest.fixture
    def detector(self):
        """Create a TimeSeriesDetector instance."""
        return TimeSeriesDetector()

    @pytest.fixture
    def snapshot_df(self):
        """Create a snapshot dataset (single row per customer)."""
        return pd.DataFrame({
            'customer_id': [1, 2, 3, 4, 5],
            'name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
            'age': [25, 30, 35, 40, 45],
            'signup_date': pd.to_datetime(['2023-01-01', '2023-02-01', '2023-03-01', '2023-04-01', '2023-05-01'])
        })

    @pytest.fixture
    def timeseries_df(self):
        """Create a time series dataset (multiple rows per customer)."""
        dates = pd.date_range('2023-01-01', periods=12, freq='ME')
        data = []
        for cust_id in range(1, 6):  # 5 customers
            for date in dates:
                data.append({
                    'customer_id': cust_id,
                    'date': date,
                    'amount': np.random.uniform(10, 100),
                    'quantity': np.random.randint(1, 10)
                })
        return pd.DataFrame(data)

    @pytest.fixture
    def event_log_df(self):
        """Create an event log dataset (irregular events per customer)."""
        # Create truly irregular intervals - some same day, some weeks apart
        data = [
            {'customer_id': 1, 'event_timestamp': datetime(2023, 1, 1), 'event_type': 'login'},
            {'customer_id': 1, 'event_timestamp': datetime(2023, 1, 1, 14, 30), 'event_type': 'view'},
            {'customer_id': 1, 'event_timestamp': datetime(2023, 1, 3), 'event_type': 'purchase'},
            {'customer_id': 1, 'event_timestamp': datetime(2023, 2, 15), 'event_type': 'login'},
            {'customer_id': 1, 'event_timestamp': datetime(2023, 2, 15, 10, 0), 'event_type': 'view'},
            {'customer_id': 2, 'event_timestamp': datetime(2023, 1, 5), 'event_type': 'login'},
            {'customer_id': 2, 'event_timestamp': datetime(2023, 1, 5, 9, 15), 'event_type': 'view'},
            {'customer_id': 2, 'event_timestamp': datetime(2023, 3, 20), 'event_type': 'purchase'},
            {'customer_id': 2, 'event_timestamp': datetime(2023, 3, 21), 'event_type': 'login'},
            {'customer_id': 3, 'event_timestamp': datetime(2023, 1, 10), 'event_type': 'login'},
            {'customer_id': 3, 'event_timestamp': datetime(2023, 1, 10, 12, 0), 'event_type': 'purchase'},
            {'customer_id': 3, 'event_timestamp': datetime(2023, 5, 1), 'event_type': 'view'},
        ]
        return pd.DataFrame(data)


class TestDetection(TestTimeSeriesDetector):
    """Tests for time series detection."""

    def test_detect_snapshot_data(self, detector, snapshot_df):
        """Test detection of snapshot data."""
        result = detector.detect(snapshot_df, entity_column='customer_id')

        assert result.is_time_series is False
        assert result.dataset_type == DatasetType.SNAPSHOT
        assert result.entity_column == 'customer_id'
        assert result.avg_observations_per_entity == 1.0

    def test_detect_timeseries_data(self, detector, timeseries_df):
        """Test detection of time series data."""
        result = detector.detect(
            timeseries_df,
            entity_column='customer_id',
            timestamp_column='date'
        )

        assert result.is_time_series is True
        assert result.dataset_type == DatasetType.TIME_SERIES
        assert result.entity_column == 'customer_id'
        assert result.timestamp_column == 'date'
        assert result.avg_observations_per_entity == 12.0
        assert result.total_entities == 5

    def test_detect_event_log(self, detector, event_log_df):
        """Test detection of event log data."""
        result = detector.detect(
            event_log_df,
            entity_column='customer_id',
            timestamp_column='event_timestamp'
        )

        assert result.is_time_series is True
        # Event logs should have multiple observations per entity
        assert result.avg_observations_per_entity > 1
        # The dataset type could be TIME_SERIES or EVENT_LOG depending on interval variance
        assert result.dataset_type in [DatasetType.TIME_SERIES, DatasetType.EVENT_LOG]

    def test_auto_detect_entity_column(self, detector, timeseries_df):
        """Test auto-detection of entity column."""
        result = detector.detect(timeseries_df)

        # Should auto-detect customer_id
        assert result.entity_column == 'customer_id'

    def test_auto_detect_timestamp_column(self, detector, timeseries_df):
        """Test auto-detection of timestamp column."""
        result = detector.detect(timeseries_df, entity_column='customer_id')

        # Should auto-detect date column
        assert result.timestamp_column == 'date'

    def test_frequency_detection_daily(self, detector):
        """Test daily frequency detection."""
        dates = pd.date_range('2023-01-01', periods=30, freq='D')
        df = pd.DataFrame({
            'customer_id': [1] * 30,
            'date': dates,
            'value': range(30)
        })
        result = detector.detect(df, entity_column='customer_id', timestamp_column='date')

        assert result.detected_frequency == TimeSeriesFrequency.DAILY

    def test_frequency_detection_weekly(self, detector):
        """Test weekly frequency detection."""
        dates = pd.date_range('2023-01-01', periods=20, freq='W')
        df = pd.DataFrame({
            'customer_id': [1] * 20,
            'date': dates,
            'value': range(20)
        })
        result = detector.detect(df, entity_column='customer_id', timestamp_column='date')

        assert result.detected_frequency == TimeSeriesFrequency.WEEKLY

    def test_frequency_detection_monthly(self, detector):
        """Test monthly frequency detection."""
        dates = pd.date_range('2023-01-01', periods=12, freq='ME')
        df = pd.DataFrame({
            'customer_id': [1] * 12,
            'date': dates,
            'value': range(12)
        })
        result = detector.detect(df, entity_column='customer_id', timestamp_column='date')

        assert result.detected_frequency == TimeSeriesFrequency.MONTHLY

    def test_confidence_increases_with_observations(self, detector):
        """Test that confidence increases with more observations."""
        # Create dataset with few observations
        df_few = pd.DataFrame({
            'customer_id': [1, 1, 2, 2],
            'date': pd.to_datetime(['2023-01-01', '2023-02-01', '2023-01-01', '2023-02-01']),
            'value': [1, 2, 3, 4]
        })
        result_few = detector.detect(df_few, entity_column='customer_id')

        # Create dataset with many observations
        dates = pd.date_range('2023-01-01', periods=20, freq='ME')
        df_many = pd.DataFrame({
            'customer_id': [1] * 20 + [2] * 20,
            'date': list(dates) * 2,
            'value': range(40)
        })
        result_many = detector.detect(df_many, entity_column='customer_id')

        assert result_many.confidence > result_few.confidence

    def test_missing_entity_column(self, detector, snapshot_df):
        """Test behavior when entity column doesn't exist."""
        result = detector.detect(snapshot_df, entity_column='nonexistent')

        assert result.is_time_series is False
        assert result.dataset_type == DatasetType.UNKNOWN
        assert result.confidence == 0.0


class TestTimeSeriesCharacteristics:
    """Tests for TimeSeriesCharacteristics dataclass."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        characteristics = TimeSeriesCharacteristics(
            is_time_series=True,
            dataset_type=DatasetType.TIME_SERIES,
            entity_column='customer_id',
            timestamp_column='date',
            total_entities=100,
            avg_observations_per_entity=12.5,
            detected_frequency=TimeSeriesFrequency.MONTHLY,
            confidence=0.85,
            evidence=['Test evidence']
        )
        d = characteristics.to_dict()

        assert d['is_time_series'] is True
        assert d['dataset_type'] == 'time_series'
        assert d['entity_column'] == 'customer_id'
        assert d['detected_frequency'] == 'monthly'
        assert d['confidence'] == 0.85


class TestTimeSeriesValidator:
    """Tests for TimeSeriesValidator class."""

    @pytest.fixture
    def validator(self):
        """Create a TimeSeriesValidator instance."""
        return TimeSeriesValidator()

    @pytest.fixture
    def clean_timeseries_df(self):
        """Create a clean time series dataset."""
        dates = pd.date_range('2023-01-01', periods=12, freq='ME')
        data = []
        for cust_id in range(1, 6):
            for date in dates:
                data.append({
                    'customer_id': cust_id,
                    'date': date,
                    'amount': 100
                })
        return pd.DataFrame(data)

    @pytest.fixture
    def gappy_timeseries_df(self):
        """Create a time series dataset with gaps."""
        dates = pd.date_range('2023-01-01', periods=12, freq='ME')
        data = []
        for cust_id in range(1, 6):
            for i, date in enumerate(dates):
                # Skip some months to create gaps
                if cust_id == 1 and i in [3, 4, 5]:  # 3 month gap
                    continue
                data.append({
                    'customer_id': cust_id,
                    'date': date,
                    'amount': 100
                })
        return pd.DataFrame(data)

    @pytest.fixture
    def duplicate_timestamps_df(self):
        """Create a time series dataset with duplicate timestamps."""
        data = [
            {'customer_id': 1, 'date': '2023-01-01', 'amount': 100},
            {'customer_id': 1, 'date': '2023-01-01', 'amount': 150},  # Duplicate
            {'customer_id': 1, 'date': '2023-02-01', 'amount': 200},
            {'customer_id': 2, 'date': '2023-01-01', 'amount': 100},
            {'customer_id': 2, 'date': '2023-02-01', 'amount': 200},
        ]
        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date'])
        return df


class TestValidation(TestTimeSeriesValidator):
    """Tests for time series validation."""

    def test_validate_clean_data(self, validator, clean_timeseries_df):
        """Test validation of clean time series data."""
        result = validator.validate(
            clean_timeseries_df,
            entity_column='customer_id',
            timestamp_column='date'
        )

        assert result.temporal_quality_score >= 90
        assert result.entities_with_gaps == 0
        assert result.entities_with_duplicate_timestamps == 0
        assert len(result.issues) == 0

    def test_detect_gaps(self, validator, gappy_timeseries_df):
        """Test detection of gaps in time series."""
        result = validator.validate(
            gappy_timeseries_df,
            entity_column='customer_id',
            timestamp_column='date',
            expected_frequency='monthly'
        )

        assert result.entities_with_gaps >= 1
        assert result.total_gaps >= 1
        assert any('gap' in issue.lower() for issue in result.issues)

    def test_detect_duplicate_timestamps(self, validator, duplicate_timestamps_df):
        """Test detection of duplicate timestamps."""
        result = validator.validate(
            duplicate_timestamps_df,
            entity_column='customer_id',
            timestamp_column='date'
        )

        assert result.entities_with_duplicate_timestamps >= 1
        assert result.total_duplicate_timestamps >= 1
        assert any('duplicate' in issue.lower() for issue in result.issues)

    def test_detect_ordering_issues(self, validator):
        """Test detection of timestamp ordering issues."""
        df = pd.DataFrame({
            'customer_id': [1, 1, 1],
            'date': pd.to_datetime(['2023-03-01', '2023-01-01', '2023-02-01']),  # Out of order
            'amount': [100, 200, 300]
        })
        result = validator.validate(
            df,
            entity_column='customer_id',
            timestamp_column='date'
        )

        assert result.entities_with_ordering_issues >= 1

    def test_high_dup_rate_penalty(self, validator):
        """Test dup_rate > 0.1 triggers 20 point penalty."""
        # 2 entities, both with duplicate timestamps => dup_rate = 1.0 > 0.1
        data = [
            {'customer_id': 1, 'date': '2023-01-01', 'amount': 100},
            {'customer_id': 1, 'date': '2023-01-01', 'amount': 150},
            {'customer_id': 1, 'date': '2023-02-01', 'amount': 200},
            {'customer_id': 2, 'date': '2023-01-01', 'amount': 100},
            {'customer_id': 2, 'date': '2023-01-01', 'amount': 200},
            {'customer_id': 2, 'date': '2023-02-01', 'amount': 300},
        ]
        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date'])
        result = validator.validate(df, entity_column='customer_id', timestamp_column='date')

        # dup_rate = 2/2 = 1.0 > 0.1 => -20 penalty => score <= 80
        assert result.temporal_quality_score <= 80

    def test_moderate_dup_rate_penalty(self, validator):
        """Test 0.01 < dup_rate <= 0.1 triggers 10 point penalty."""
        # Many entities, only one with dups => dup_rate ~0.05
        data = []
        for cust_id in range(1, 21):  # 20 entities
            for month in range(1, 4):
                data.append({
                    'customer_id': cust_id,
                    'date': pd.Timestamp(f'2023-{month:02d}-01'),
                    'amount': 100,
                })
        # Add a duplicate for customer 1
        data.append({'customer_id': 1, 'date': pd.Timestamp('2023-01-01'), 'amount': 200})
        df = pd.DataFrame(data)
        result = validator.validate(df, entity_column='customer_id', timestamp_column='date')

        # dup_rate = 1/20 = 0.05 > 0.01 => -10
        assert result.temporal_quality_score <= 90

    def test_high_order_rate_penalty(self, validator):
        """Test order_rate > 0.1 triggers 20 point penalty."""
        # All entities have ordering issues
        data = []
        for cust_id in range(1, 4):
            data.extend([
                {'customer_id': cust_id, 'date': pd.Timestamp('2023-03-01'), 'amount': 300},
                {'customer_id': cust_id, 'date': pd.Timestamp('2023-01-01'), 'amount': 100},
                {'customer_id': cust_id, 'date': pd.Timestamp('2023-02-01'), 'amount': 200},
            ])
        df = pd.DataFrame(data)
        result = validator.validate(df, entity_column='customer_id', timestamp_column='date')

        # order_rate = 3/3 = 1.0 > 0.1 => -20
        assert result.temporal_quality_score <= 80

    def test_moderate_order_rate_penalty(self, validator):
        """Test 0.01 < order_rate <= 0.1 triggers 10 point penalty."""
        data = []
        for cust_id in range(1, 21):
            data.extend([
                {'customer_id': cust_id, 'date': pd.Timestamp('2023-01-01'), 'amount': 100},
                {'customer_id': cust_id, 'date': pd.Timestamp('2023-02-01'), 'amount': 200},
                {'customer_id': cust_id, 'date': pd.Timestamp('2023-03-01'), 'amount': 300},
            ])
        # Make customer 1 out of order
        data[0] = {'customer_id': 1, 'date': pd.Timestamp('2023-03-01'), 'amount': 300}
        data[2] = {'customer_id': 1, 'date': pd.Timestamp('2023-01-01'), 'amount': 100}
        df = pd.DataFrame(data)
        result = validator.validate(df, entity_column='customer_id', timestamp_column='date')

        # order_rate = 1/20 = 0.05 > 0.01 => -10
        assert result.temporal_quality_score <= 90

    def test_high_gap_rate_penalty(self, validator):
        """Test gap_rate > 0.2 triggers 20 point penalty."""
        # All entities have gaps
        data = []
        for cust_id in range(1, 4):
            data.extend([
                {'customer_id': cust_id, 'date': pd.Timestamp('2023-01-01'), 'amount': 100},
                {'customer_id': cust_id, 'date': pd.Timestamp('2023-06-01'), 'amount': 200},
            ])
        df = pd.DataFrame(data)
        result = validator.validate(
            df, entity_column='customer_id', timestamp_column='date',
            expected_frequency='monthly', max_allowed_gap_periods=1
        )

        # gap_rate = 3/3 = 1.0 > 0.2 => -20
        assert result.temporal_quality_score <= 80

    def test_moderate_gap_rate_penalty(self, validator):
        """Test 0.05 < gap_rate <= 0.1 triggers 5 point penalty."""
        data = []
        for cust_id in range(1, 21):
            data.extend([
                {'customer_id': cust_id, 'date': pd.Timestamp('2023-01-01'), 'amount': 100},
                {'customer_id': cust_id, 'date': pd.Timestamp('2023-02-01'), 'amount': 200},
                {'customer_id': cust_id, 'date': pd.Timestamp('2023-03-01'), 'amount': 300},
            ])
        # Customer 1 gets a big gap
        data[0] = {'customer_id': 1, 'date': pd.Timestamp('2023-01-01'), 'amount': 100}
        data[1] = {'customer_id': 1, 'date': pd.Timestamp('2023-08-01'), 'amount': 200}
        data[2] = {'customer_id': 1, 'date': pd.Timestamp('2023-09-01'), 'amount': 300}
        # Customer 2 also gets a big gap
        data[3] = {'customer_id': 2, 'date': pd.Timestamp('2023-01-01'), 'amount': 100}
        data[4] = {'customer_id': 2, 'date': pd.Timestamp('2023-08-01'), 'amount': 200}
        data[5] = {'customer_id': 2, 'date': pd.Timestamp('2023-09-01'), 'amount': 300}
        df = pd.DataFrame(data)
        result = validator.validate(
            df, entity_column='customer_id', timestamp_column='date',
            expected_frequency='monthly', max_allowed_gap_periods=1
        )

        # gap_rate = 2/20 = 0.1 => triggers 10 (0.1 > 0.05)
        assert result.temporal_quality_score <= 95

    def test_gap_analysis_with_estimated_interval(self, validator):
        """Test gap analysis when no expected_frequency is given (estimated from data)."""
        data = []
        for cust_id in range(1, 4):
            for month in range(1, 7):
                data.append({
                    'customer_id': cust_id,
                    'date': pd.Timestamp(f'2023-{month:02d}-01'),
                    'amount': 100,
                })
        df = pd.DataFrame(data)
        result = validator.validate(
            df, entity_column='customer_id', timestamp_column='date',
            # No expected_frequency - will estimate from data
        )

        assert result.temporal_quality_score >= 80

    def test_gap_analysis_none_interval(self, validator):
        """Test gap analysis when interval cannot be estimated."""
        # Single obs per entity => no interval can be estimated
        df = pd.DataFrame({
            'customer_id': [1, 2, 3],
            'date': pd.to_datetime(['2023-01-01', '2023-02-01', '2023-03-01']),
            'amount': [100, 200, 300],
        })
        result = validator.validate(
            df, entity_column='customer_id', timestamp_column='date',
        )

        # Should handle gracefully
        assert result.entities_with_gaps == 0

    def test_ordering_check_entity_with_single_observation(self, validator):
        """Test ordering check skips entities with < 2 observations."""
        data = [
            {'customer_id': 1, 'date': pd.Timestamp('2023-01-01'), 'amount': 100},
            {'customer_id': 2, 'date': pd.Timestamp('2023-01-01'), 'amount': 200},
            {'customer_id': 2, 'date': pd.Timestamp('2023-02-01'), 'amount': 300},
        ]
        df = pd.DataFrame(data)
        result = validator.validate(
            df, entity_column='customer_id', timestamp_column='date',
        )

        assert result.entities_with_ordering_issues == 0

    def test_missing_entity_column(self, validator, clean_timeseries_df):
        """Test validation with missing entity column."""
        result = validator.validate(
            clean_timeseries_df,
            entity_column='nonexistent',
            timestamp_column='date'
        )

        assert result.temporal_quality_score == 0
        assert len(result.issues) > 0

    def test_missing_timestamp_column(self, validator, clean_timeseries_df):
        """Test validation with missing timestamp column."""
        result = validator.validate(
            clean_timeseries_df,
            entity_column='customer_id',
            timestamp_column='nonexistent'
        )

        assert result.temporal_quality_score == 0
        assert len(result.issues) > 0


class TestTimeSeriesValidationResult:
    """Tests for TimeSeriesValidationResult dataclass."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = TimeSeriesValidationResult(
            coverage_percentage=95.5,
            entities_with_gaps=5,
            total_gaps=10,
            temporal_quality_score=85.0,
            issues=['Some issue']
        )
        d = result.to_dict()

        assert d['coverage_percentage'] == 95.5
        assert d['entities_with_gaps'] == 5
        assert d['temporal_quality_score'] == 85.0


class TestEdgeCases:
    """Tests for edge cases."""

    @pytest.fixture
    def detector(self):
        return TimeSeriesDetector()

    @pytest.fixture
    def validator(self):
        return TimeSeriesValidator()

    def test_empty_dataframe(self, detector):
        """Test detection on empty dataframe."""
        df = pd.DataFrame({'customer_id': [], 'date': []})
        result = detector.detect(df, entity_column='customer_id')

        # Should handle gracefully
        assert result.total_entities == 0

    def test_single_row_dataframe(self, detector):
        """Test detection on single row dataframe."""
        df = pd.DataFrame({
            'customer_id': [1],
            'date': [pd.Timestamp('2023-01-01')],
            'value': [100]
        })
        result = detector.detect(df, entity_column='customer_id')

        assert result.is_time_series is False
        assert result.dataset_type == DatasetType.SNAPSHOT

    def test_all_null_timestamps(self, detector):
        """Test detection with all null timestamps."""
        df = pd.DataFrame({
            'customer_id': [1, 1, 2, 2],
            'date': [None, None, None, None],
            'value': [1, 2, 3, 4]
        })
        result = detector.detect(df, entity_column='customer_id', timestamp_column='date')

        # Should still detect as time series based on entity repetition
        assert result.avg_observations_per_entity == 2.0

    def test_mixed_valid_invalid_timestamps(self, detector):
        """Test detection with mixed valid/invalid timestamps."""
        df = pd.DataFrame({
            'customer_id': [1, 1, 1, 1],
            'date': ['2023-01-01', '2023-02-01', 'invalid', '2023-04-01'],
            'value': [1, 2, 3, 4]
        })
        result = detector.detect(df, entity_column='customer_id', timestamp_column='date')

        # Should handle gracefully
        assert result.is_time_series is True

    def test_detect_timeseries_with_duplicate_timestamps(self, detector):
        """Test detection of time series data that contains duplicate timestamps."""
        data = []
        for cust_id in range(1, 4):
            for month in range(1, 7):
                data.append({
                    'customer_id': cust_id,
                    'date': pd.Timestamp(f'2023-{month:02d}-01'),
                    'value': 100,
                })
        # Add duplicate timestamps for customer 1
        data.append({'customer_id': 1, 'date': pd.Timestamp('2023-01-01'), 'value': 200})
        data.append({'customer_id': 1, 'date': pd.Timestamp('2023-02-01'), 'value': 300})
        df = pd.DataFrame(data)
        result = detector.detect(df, entity_column='customer_id', timestamp_column='date')

        assert result.is_time_series is True
        assert result.duplicate_timestamps_count > 0
        assert any('duplicate' in e.lower() for e in result.evidence)

    def test_auto_detect_entity_column_by_characteristics(self, detector):
        """Test entity column auto-detection by column characteristics (fallback path)."""
        # No column matches ENTITY_PATTERNS; detection falls through to heuristic
        df = pd.DataFrame({
            'category': ['A', 'A', 'B', 'B', 'C', 'C', 'A', 'B', 'C', 'A'],
            'date': pd.date_range('2023-01-01', periods=10, freq='D'),
            'value': range(10),
        })
        result = detector.detect(df)

        # Should auto-detect 'category' based on cardinality heuristic
        assert result.entity_column == 'category'

    def test_auto_detect_entity_column_returns_none(self, detector):
        """Test entity column auto-detection returns None when no column fits."""
        # All columns are unique or don't match
        df = pd.DataFrame({
            'col_a': [1.0, 2.0, 3.0],
            'col_b': [4.0, 5.0, 6.0],
        })
        result = detector.detect(df)

        assert result.is_time_series is False
        assert result.dataset_type == DatasetType.UNKNOWN

    def test_detect_timestamp_column_name_match_and_parsable(self, detector):
        """Test timestamp detection with a name-matching parsable string column."""
        df = pd.DataFrame({
            'customer_id': [1, 1, 2, 2, 3, 3],
            'event_date': ['2023-01-01', '2023-02-01', '2023-01-01', '2023-02-01',
                           '2023-01-01', '2023-02-01'],
            'value': range(6),
        })
        result = detector.detect(df, entity_column='customer_id')

        assert result.timestamp_column == 'event_date'

    def test_detect_timestamp_column_parsable_no_name_match(self, detector):
        """Test timestamp detection for a parsable column with no name match."""
        df = pd.DataFrame({
            'customer_id': [1, 1, 2, 2, 3, 3],
            'xyz': ['2023-01-01', '2023-02-01', '2023-01-01', '2023-02-01',
                    '2023-01-01', '2023-02-01'],
            'value': range(6),
        })
        result = detector.detect(df, entity_column='customer_id')

        # Should detect 'xyz' as parsable datetime even though name doesn't match
        assert result.timestamp_column == 'xyz'

    def test_detect_timestamp_column_returns_none(self, detector):
        """Test timestamp detection returns None when no columns match."""
        df = pd.DataFrame({
            'customer_id': [1, 1, 2, 2, 3, 3],
            'value': [10, 20, 30, 40, 50, 60],
        })
        result = detector.detect(df, entity_column='customer_id')

        assert result.timestamp_column is None

    def test_frequency_detection_hourly(self, detector):
        """Test hourly frequency detection."""
        dates = pd.date_range('2023-01-01', periods=30, freq='h')
        df = pd.DataFrame({
            'customer_id': [1] * 30,
            'date': dates,
            'value': range(30),
        })
        result = detector.detect(df, entity_column='customer_id', timestamp_column='date')

        assert result.detected_frequency == TimeSeriesFrequency.HOURLY

    def test_frequency_detection_quarterly(self, detector):
        """Test quarterly frequency detection."""
        dates = pd.date_range('2020-01-01', periods=12, freq='QS')
        df = pd.DataFrame({
            'customer_id': [1] * 12,
            'date': dates,
            'value': range(12),
        })
        result = detector.detect(df, entity_column='customer_id', timestamp_column='date')

        assert result.detected_frequency == TimeSeriesFrequency.QUARTERLY

    def test_frequency_detection_yearly(self, detector):
        """Test yearly frequency detection."""
        dates = pd.date_range('2010-01-01', periods=10, freq='YS')
        df = pd.DataFrame({
            'customer_id': [1] * 10,
            'date': dates,
            'value': range(10),
        })
        result = detector.detect(df, entity_column='customer_id', timestamp_column='date')

        assert result.detected_frequency == TimeSeriesFrequency.YEARLY

    def test_frequency_detection_empty_intervals(self, detector):
        """Test frequency detection when all entities have single observations."""
        df = pd.DataFrame({
            'customer_id': [1, 2, 3],
            'date': pd.to_datetime(['2023-01-01', '2023-02-01', '2023-03-01']),
            'value': [10, 20, 30],
        })
        # avg_obs = 1 so this is a snapshot, test detection directly
        freq, median = detector._detect_frequency(df, 'customer_id', 'date')

        assert freq == TimeSeriesFrequency.UNKNOWN
        assert median == 0.0

    def test_frequency_detection_entity_with_single_observation(self, detector):
        """Test frequency detection skips entities with < 2 observations."""
        df = pd.DataFrame({
            'customer_id': [1, 2, 2, 2],
            'date': pd.to_datetime(['2023-01-01', '2023-01-01', '2023-02-01', '2023-03-01']),
            'value': [10, 20, 30, 40],
        })
        freq, median = detector._detect_frequency(df, 'customer_id', 'date')

        # Should still compute from entity 2 only
        assert freq != TimeSeriesFrequency.UNKNOWN

    def test_frequency_detection_entity_with_null_timestamps(self, detector):
        """Test frequency detection when entity has all null timestamps after coerce."""
        df = pd.DataFrame({
            'customer_id': [1, 1, 1],
            'date': ['not-a-date', 'also-bad', 'still-bad'],
            'value': [1, 2, 3],
        })
        freq, median = detector._detect_frequency(df, 'customer_id', 'date')

        assert freq == TimeSeriesFrequency.UNKNOWN
        assert median == 0.0

    def test_confidence_mid_range_observations(self, detector):
        """Test confidence calculation with 5-9 avg observations."""
        dates = pd.date_range('2023-01-01', periods=7, freq='D')
        df = pd.DataFrame({
            'customer_id': [1] * 7 + [2] * 7,
            'date': list(dates) * 2,
            'value': range(14),
        })
        result = detector.detect(df, entity_column='customer_id', timestamp_column='date')

        # avg_obs=7, has_timestamp=True, has_frequency=True
        # confidence = 0.5 + 0.2 + 0.1 + 0.1 = 0.9
        assert result.confidence == pytest.approx(0.9, abs=0.05)

    def test_detect_timeseries_no_timestamp_column(self, detector):
        """Test detection with multiple obs per entity but no timestamp column found."""
        df = pd.DataFrame({
            'customer_id': [1, 1, 1, 2, 2, 2],
            'value': [10, 20, 30, 40, 50, 60],
        })
        result = detector.detect(df, entity_column='customer_id')

        # Should be time series (avg_obs=3) but unknown frequency
        assert result.is_time_series is True
        assert result.detected_frequency == TimeSeriesFrequency.UNKNOWN

    def test_snapshot_with_borderline_avg(self, detector):
        """Test snapshot detection when avg_obs is between 1.5 and 2."""
        df = pd.DataFrame({
            'customer_id': [1, 1, 2, 3],
            'date': pd.to_datetime(['2023-01-01', '2023-02-01', '2023-01-01', '2023-01-01']),
            'value': [10, 20, 30, 40],
        })
        # avg_obs = 4/3 ~= 1.33 which is < 1.5
        result = detector.detect(df, entity_column='customer_id', timestamp_column='date')

        assert result.is_time_series is False
        assert result.dataset_type == DatasetType.SNAPSHOT
        assert result.confidence == 0.8


class TestClassifyFrequency:

    @pytest.fixture
    def detector(self):
        return TimeSeriesDetector()

    @pytest.mark.parametrize("median_hours,std_hours,expected", [
        (1.0, 0.2, TimeSeriesFrequency.HOURLY),
        (24.0, 1.0, TimeSeriesFrequency.DAILY),
        (168.0, 5.0, TimeSeriesFrequency.WEEKLY),
        (730.0, 10.0, TimeSeriesFrequency.MONTHLY),
        (2100.0, 50.0, TimeSeriesFrequency.QUARTERLY),
        (8760.0, 100.0, TimeSeriesFrequency.YEARLY),
        (500.0, 300.0, TimeSeriesFrequency.IRREGULAR),
        (50.0, 5.0, TimeSeriesFrequency.IRREGULAR),
    ])
    def test_classify_frequency(self, detector, median_hours, std_hours, expected):
        assert detector._classify_frequency(median_hours, std_hours) == expected

    def test_classify_frequency_zero_median(self, detector):
        assert detector._classify_frequency(0.0, 0.0) == TimeSeriesFrequency.HOURLY


class TestSparkDispatch:

    @pytest.fixture
    def detector(self):
        return TimeSeriesDetector()

    @pytest.fixture
    def validator(self):
        return TimeSeriesValidator()

    def test_detect_frequency_dispatches_to_spark(self, detector, monkeypatch):
        called = {}

        def mock_spark_freq(df, ec, tc):
            called['spark'] = True
            return TimeSeriesFrequency.DAILY, 24.0

        monkeypatch.setattr(detector, '_spark_detect_frequency', mock_spark_freq)
        monkeypatch.setattr(
            'customer_retention.stages.validation.timeseries_detector._is_spark_pandas',
            lambda x: True,
        )
        freq, hours = detector._detect_frequency(MagicMock(), 'entity', 'ts')
        assert called.get('spark') is True
        assert freq == TimeSeriesFrequency.DAILY

    def test_detect_frequency_stays_pandas(self, detector, monkeypatch):
        dates = pd.date_range('2023-01-01', periods=30, freq='D')
        df = pd.DataFrame({'cid': [1] * 30, 'date': dates})
        monkeypatch.setattr(
            'customer_retention.stages.validation.timeseries_detector._is_spark_pandas',
            lambda x: False,
        )
        freq, hours = detector._detect_frequency(df, 'cid', 'date')
        assert freq == TimeSeriesFrequency.DAILY

    def test_check_ordering_dispatches_to_spark(self, validator, monkeypatch):
        called = {}

        def mock_spark_ordering(df, ec):
            called['spark'] = True
            return {'entities': 0, 'examples': []}

        monkeypatch.setattr(validator, '_spark_check_ordering', mock_spark_ordering)
        monkeypatch.setattr(
            'customer_retention.stages.validation.timeseries_detector._is_spark_pandas',
            lambda x: True,
        )
        result = validator._check_ordering(MagicMock(), 'entity')
        assert called.get('spark') is True
        assert result['entities'] == 0

    def test_analyze_gaps_dispatches_to_spark(self, validator, monkeypatch):
        called = {}

        def mock_spark_gaps(df, ec, ef, m):
            called['spark'] = True
            return {
                'entities_with_gaps': 0, 'total_gaps': 0, 'max_gap': 0,
                'examples': [], 'coverage': 100.0,
                'frequency_consistent': True, 'frequency_deviation': 0.0,
                'expected_periods': 0, 'actual_periods': 0,
            }

        monkeypatch.setattr(validator, '_spark_analyze_gaps', mock_spark_gaps)
        monkeypatch.setattr(
            'customer_retention.stages.validation.timeseries_detector._is_spark_pandas',
            lambda x: True,
        )
        result = validator._analyze_gaps(MagicMock(), 'entity', 'monthly', 3)
        assert called.get('spark') is True
        assert result['entities_with_gaps'] == 0

    def test_estimate_interval_dispatches_to_spark(self, validator, monkeypatch):
        called = {}

        def mock_spark_interval(df, ec):
            called['spark'] = True
            return timedelta(days=30)

        monkeypatch.setattr(validator, '_spark_estimate_interval', mock_spark_interval)
        monkeypatch.setattr(
            'customer_retention.stages.validation.timeseries_detector._is_spark_pandas',
            lambda x: True,
        )
        result = validator._estimate_interval(MagicMock(), 'entity')
        assert called.get('spark') is True
        assert result == timedelta(days=30)

    def test_check_duplicate_timestamps_dispatches_to_spark(self, validator, monkeypatch):
        called = {}

        def mock_spark_dups(df, ec):
            called['spark'] = True
            return {'total': 0, 'entities': 0, 'examples': []}

        monkeypatch.setattr(
            validator, '_spark_check_duplicate_timestamps', mock_spark_dups,
        )
        monkeypatch.setattr(
            'customer_retention.stages.validation.timeseries_detector._is_spark_pandas',
            lambda x: True,
        )
        result = validator._check_duplicate_timestamps(MagicMock(), 'entity')
        assert called.get('spark') is True
        assert result['total'] == 0
