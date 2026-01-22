"""
Time series profiler for analyzing event-level datasets.

Provides profiling capabilities specific to time series data including:
- Entity lifecycle analysis
- Event frequency distributions
- Inter-event timing statistics
"""
from dataclasses import dataclass, field
from typing import Optional

from customer_retention.core.compat import pd, DataFrame


@dataclass
class DistributionStats:
    """Statistics describing a distribution."""
    min: float
    max: float
    mean: float
    median: float
    std: float
    q25: Optional[float] = None
    q75: Optional[float] = None


@dataclass
class EntityLifecycle:
    """Lifecycle information for a single entity."""
    entity: str
    first_event: pd.Timestamp
    last_event: pd.Timestamp
    duration_days: int
    event_count: int


@dataclass
class TimeSeriesProfile:
    """Complete profile of a time series dataset."""
    entity_column: str
    time_column: str
    total_events: int
    unique_entities: int
    time_span_days: int
    events_per_entity: DistributionStats
    entity_lifecycles: DataFrame
    avg_inter_event_days: Optional[float] = None
    first_event_date: Optional[pd.Timestamp] = None
    last_event_date: Optional[pd.Timestamp] = None


class TimeSeriesProfiler:
    """Profiles time series (event-level) datasets."""

    def __init__(self, entity_column: str, time_column: str):
        self.entity_column = entity_column
        self.time_column = time_column

    def profile(self, df: DataFrame) -> TimeSeriesProfile:
        """Generate a complete profile of the time series data."""
        if len(df) == 0:
            return self._empty_profile()

        self._validate_columns(df)
        df = self._prepare_dataframe(df)

        total_events = len(df)
        unique_entities = df[self.entity_column].nunique()

        lifecycles = self._compute_entity_lifecycles(df)
        events_per_entity = self._compute_events_distribution(lifecycles)
        time_span = self._compute_time_span(df)
        avg_inter_event = self._compute_avg_inter_event_time(df)

        return TimeSeriesProfile(
            entity_column=self.entity_column,
            time_column=self.time_column,
            total_events=total_events,
            unique_entities=unique_entities,
            time_span_days=time_span,
            events_per_entity=events_per_entity,
            entity_lifecycles=lifecycles,
            avg_inter_event_days=avg_inter_event,
            first_event_date=df[self.time_column].min(),
            last_event_date=df[self.time_column].max(),
        )

    def _validate_columns(self, df: DataFrame) -> None:
        """Ensure required columns exist."""
        if self.entity_column not in df.columns:
            raise KeyError(f"Entity column '{self.entity_column}' not found")
        if self.time_column not in df.columns:
            raise KeyError(f"Time column '{self.time_column}' not found")

    def _prepare_dataframe(self, df: DataFrame) -> DataFrame:
        """Prepare dataframe by parsing dates if needed."""
        df = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(df[self.time_column]):
            df[self.time_column] = pd.to_datetime(df[self.time_column])
        return df

    def _compute_entity_lifecycles(self, df: DataFrame) -> DataFrame:
        """Compute lifecycle metrics for each entity."""
        grouped = df.groupby(self.entity_column)[self.time_column]

        lifecycles = pd.DataFrame({
            "entity": grouped.first().index.tolist(),
            "first_event": grouped.min().values,
            "last_event": grouped.max().values,
            "event_count": grouped.count().values,
        })

        lifecycles["duration_days"] = (
            (lifecycles["last_event"] - lifecycles["first_event"]).dt.days
        )

        return lifecycles

    def _compute_events_distribution(self, lifecycles: DataFrame) -> DistributionStats:
        """Compute distribution statistics for events per entity."""
        counts = lifecycles["event_count"]

        if len(counts) == 0:
            return DistributionStats(
                min=0, max=0, mean=0, median=0, std=0, q25=0, q75=0
            )

        return DistributionStats(
            min=float(counts.min()),
            max=float(counts.max()),
            mean=float(counts.mean()),
            median=float(counts.median()),
            std=float(counts.std()) if len(counts) > 1 else 0.0,
            q25=float(counts.quantile(0.25)),
            q75=float(counts.quantile(0.75)),
        )

    def _compute_time_span(self, df: DataFrame) -> int:
        """Compute total time span in days."""
        if len(df) == 0:
            return 0
        min_date = df[self.time_column].min()
        max_date = df[self.time_column].max()
        return (max_date - min_date).days

    def _compute_avg_inter_event_time(self, df: DataFrame) -> Optional[float]:
        """Compute average time between consecutive events per entity."""
        if len(df) < 2:
            return None

        inter_event_days = []
        for _, group in df.groupby(self.entity_column):
            if len(group) < 2:
                continue
            sorted_dates = group[self.time_column].sort_values()
            diffs = sorted_dates.diff().dropna()
            inter_event_days.extend(diffs.dt.total_seconds() / 86400)

        if not inter_event_days:
            return None

        return float(sum(inter_event_days) / len(inter_event_days))

    def _empty_profile(self) -> TimeSeriesProfile:
        """Return profile for empty dataset."""
        return TimeSeriesProfile(
            entity_column=self.entity_column,
            time_column=self.time_column,
            total_events=0,
            unique_entities=0,
            time_span_days=0,
            events_per_entity=DistributionStats(
                min=0, max=0, mean=0, median=0, std=0, q25=0, q75=0
            ),
            entity_lifecycles=pd.DataFrame(columns=[
                "entity", "first_event", "last_event", "duration_days", "event_count"
            ]),
            avg_inter_event_days=None,
        )
