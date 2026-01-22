"""Tests for time series detection and validation module."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from customer_retention.stages.validation.timeseries_detector import (
    TimeSeriesDetector, TimeSeriesValidator,
    TimeSeriesCharacteristics, TimeSeriesValidationResult,
    DatasetType, TimeSeriesFrequency
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
