"""
Time series profiler for analyzing event-level datasets.

Provides profiling capabilities specific to time series data including:
- Entity lifecycle analysis
- Event frequency distributions
- Inter-event timing statistics
"""
from dataclasses import dataclass
from typing import Optional

from customer_retention.core.compat import DataFrame, pd


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
class LifecycleQuadrantResult:
    """Result of lifecycle quadrant classification."""
    lifecycles: DataFrame
    tenure_threshold: float
    intensity_threshold: float
    recommendations: DataFrame


_QUADRANT_RECOMMENDATIONS = {
    "Steady & Loyal": {
        "Windows": "All (7d-365d)",
        "Feature Strategy": "Trend/seasonality features, engagement decay",
        "Risk": "Low churn risk; monitor for engagement decline",
    },
    "Occasional & Loyal": {
        "Windows": "Wide (90d, 180d, 365d)",
        "Feature Strategy": "Long-window aggregations, recency gap",
        "Risk": "May churn silently; long gaps are normal",
    },
    "Intense & Brief": {
        "Windows": "Short (7d, 24h, 30d)",
        "Feature Strategy": "Recency features, burst detection",
        "Risk": "High churn risk; may be early churners",
    },
    "One-shot": {
        "Windows": "N/A (insufficient history)",
        "Feature Strategy": "Cold-start fallback, population-level stats",
        "Risk": "Cannot build temporal features; consider separate handling",
    },
}


def classify_lifecycle_quadrants(entity_lifecycles: DataFrame) -> LifecycleQuadrantResult:
    """Classify entities into lifecycle quadrants and produce recommendations.

    Splits on median tenure and median intensity (events/day) to create four
    quadrants: Steady & Loyal, Occasional & Loyal, Intense & Brief, One-shot.

    Returns a LifecycleQuadrantResult with the classified lifecycles and a
    recommendations DataFrame ordered by quadrant significance (share desc).
    """
    lc = entity_lifecycles.copy()
    tenure_threshold = float(lc["duration_days"].median())
    lc["intensity"] = lc["event_count"] / lc["duration_days"].clip(lower=1)
    intensity_threshold = float(lc["intensity"].median())

    def _assign(row):
        long = row["duration_days"] >= tenure_threshold
        high = row["intensity"] >= intensity_threshold
        if long and high:
            return "Steady & Loyal"
        if long:
            return "Occasional & Loyal"
        if high:
            return "Intense & Brief"
        return "One-shot"

    lc["lifecycle_quadrant"] = lc.apply(_assign, axis=1)

    counts = lc["lifecycle_quadrant"].value_counts()
    total = len(lc)
    rows = []
    for quadrant in counts.index:
        n = counts[quadrant]
        rec = _QUADRANT_RECOMMENDATIONS[quadrant]
        rows.append({
            "Quadrant": quadrant,
            "Entities": n,
            "Share": f"{n / total * 100:.1f}%",
            "Windows": rec["Windows"],
            "Feature Strategy": rec["Feature Strategy"],
            "Risk": rec["Risk"],
        })

    return LifecycleQuadrantResult(
        lifecycles=lc,
        tenure_threshold=tenure_threshold,
        intensity_threshold=intensity_threshold,
        recommendations=pd.DataFrame(rows),
    )


@dataclass
class ActivitySegmentResult:
    """Result of activity segment classification."""
    lifecycles: DataFrame
    q25_threshold: float
    q75_threshold: float
    recommendations: DataFrame


_SEGMENT_RECOMMENDATIONS = {
    "One-time": {
        "Feature Approach": "No temporal features possible; use event-level attributes only",
        "Modeling Implication": "Cold-start problem; consider population-level fallback or separate model",
    },
    "Low Activity": {
        "Feature Approach": "Wide windows (90d+) with count/recency; sparse aggregations",
        "Modeling Implication": "Features will be noisy; log-transform counts, handle many zeros",
    },
    "Medium Activity": {
        "Feature Approach": "Standard windows (7d-90d); mean/std aggregations reliable",
        "Modeling Implication": "Core modeling population; most features well-populated",
    },
    "High Activity": {
        "Feature Approach": "All windows including short (24h, 7d); trends and velocity meaningful",
        "Modeling Implication": "Rich feature space; watch for dominance in training set",
    },
}


def classify_activity_segments(entity_lifecycles: DataFrame) -> ActivitySegmentResult:
    """Classify entities by event count into activity segments with recommendations.

    Segments are defined by quartiles of event_count:
    - One-time: exactly 1 event
    - Low Activity: 2 to q25
    - Medium Activity: q25 to q75
    - High Activity: above q75

    Returns an ActivitySegmentResult with classified lifecycles and a
    recommendations DataFrame ordered by segment share (descending).
    """
    lc = entity_lifecycles.copy()
    q25 = float(lc["event_count"].quantile(0.25))
    q75 = float(lc["event_count"].quantile(0.75))

    def _assign(count):
        if count <= 1:
            return "One-time"
        if count <= q25:
            return "Low Activity"
        if count <= q75:
            return "Medium Activity"
        return "High Activity"

    lc["activity_segment"] = lc["event_count"].apply(_assign)

    counts = lc["activity_segment"].value_counts()
    total = len(lc)
    rows = []
    for segment in counts.index:
        n = counts[segment]
        subset = lc[lc["activity_segment"] == segment]
        rec = _SEGMENT_RECOMMENDATIONS[segment]
        rows.append({
            "Segment": segment,
            "Entities": n,
            "Share": f"{n / total * 100:.1f}%",
            "Avg Events": f"{subset['event_count'].mean():.1f}",
            "Feature Approach": rec["Feature Approach"],
            "Modeling Implication": rec["Modeling Implication"],
        })

    return ActivitySegmentResult(
        lifecycles=lc,
        q25_threshold=q25,
        q75_threshold=q75,
        recommendations=pd.DataFrame(rows),
    )


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
