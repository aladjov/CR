from dataclasses import dataclass
from typing import Optional

import numpy as np

from customer_retention.core.compat import (
    DataFrame,
    Timestamp,
    _infer_epoch_unit,
    _is_spark_pandas,
    as_spark_df,
    groupby_multi_agg,
    head_as_list,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    native_pd,
    pd,
    resolve_column_name,
    timedelta_to_days,
    timestamp_diffs_seconds,
)


@dataclass
class DistributionStats:
    min: float
    max: float
    mean: float
    median: float
    std: float
    q25: Optional[float] = None
    q75: Optional[float] = None


@dataclass
class LifecycleQuadrantResult:
    lifecycles: DataFrame
    tenure_threshold: float
    intensity_threshold: float
    recommendations: DataFrame


LIFECYCLE_LABELS = {
    "high_high": "steady_loyal_lifecycle",
    "high_low": "occasional_loyal_lifecycle",
    "low_high": "intense_brief_lifecycle",
    "low_low": "one_shot_lifecycle",
}

_QUADRANT_RECOMMENDATIONS = {
    LIFECYCLE_LABELS["high_high"]: {
        "Windows": "All available windows",
        "Feature Strategy": "Trend/seasonality features, engagement decay",
        "Risk": "Low churn risk; monitor for engagement decline",
    },
    LIFECYCLE_LABELS["high_low"]: {
        "Windows": "Wider windows (capture sparse events)",
        "Feature Strategy": "Long-window aggregations, recency gap",
        "Risk": "May churn silently; long gaps are normal",
    },
    LIFECYCLE_LABELS["low_high"]: {
        "Windows": "Narrower windows (capture recency)",
        "Feature Strategy": "Recency features, burst detection",
        "Risk": "High churn risk; may be early churners",
    },
    LIFECYCLE_LABELS["low_low"]: {
        "Windows": "N/A (insufficient history)",
        "Feature Strategy": "Cold-start fallback, population-level stats",
        "Risk": "Cannot build temporal features; consider separate handling",
    },
}


def _assign_lifecycle_quadrant(duration_days: np.ndarray, intensity: np.ndarray,
                               tenure_threshold: float, intensity_threshold: float) -> np.ndarray:
    long = duration_days >= tenure_threshold
    high = intensity >= intensity_threshold
    result = np.full(len(duration_days), LIFECYCLE_LABELS["low_low"], dtype=object)
    result[high & ~long] = LIFECYCLE_LABELS["low_high"]
    result[long & ~high] = LIFECYCLE_LABELS["high_low"]
    result[long & high] = LIFECYCLE_LABELS["high_high"]
    return result


def classify_lifecycle_quadrants(entity_lifecycles: DataFrame) -> LifecycleQuadrantResult:
    lc = entity_lifecycles.copy()
    tenure_threshold = float(lc["duration_days"].median())
    lc["intensity"] = lc["event_count"] / lc["duration_days"].clip(lower=1)
    intensity_threshold = float(lc["intensity"].median())

    lc["lifecycle_quadrant"] = _assign_lifecycle_quadrant(
        lc["duration_days"].to_numpy(), lc["intensity"].to_numpy(),
        tenure_threshold, intensity_threshold
    )

    counts = lc["lifecycle_quadrant"].value_counts()
    total = len(lc)
    rows = []
    for quadrant in head_as_list(counts.index, 10):
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
        recommendations=native_pd.DataFrame(rows),
    )


@dataclass
class ActivitySegmentResult:
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
        "Feature Approach": "Wider windows with count/recency; sparse aggregations",
        "Modeling Implication": "Features will be noisy; log-transform counts, handle many zeros",
    },
    "Medium Activity": {
        "Feature Approach": "Standard windows; mean/std aggregations reliable",
        "Modeling Implication": "Core modeling population; most features well-populated",
    },
    "High Activity": {
        "Feature Approach": "All windows including narrower; trends and velocity meaningful",
        "Modeling Implication": "Rich feature space; watch for dominance in training set",
    },
}


def _assign_activity_segment_numpy(event_count: np.ndarray, q25: float, q75: float) -> np.ndarray:
    result = np.full(len(event_count), "High Activity", dtype=object)
    result[event_count <= q75] = "Medium Activity"
    result[event_count <= q25] = "Low Activity"
    result[event_count <= 1] = "One-time"
    return result


def _assign_activity_segment_spark(event_count_col, q25: float, q75: float):
    import pyspark.sql.functions as F  # noqa: N812
    return (F.when(event_count_col <= 1, "One-time")
             .when(event_count_col <= F.lit(q25), "Low Activity")
             .when(event_count_col <= F.lit(q75), "Medium Activity")
             .otherwise("High Activity"))


def classify_activity_segments(entity_lifecycles: DataFrame) -> ActivitySegmentResult:
    lc = entity_lifecycles.copy()
    q25 = float(lc["event_count"].quantile(0.25))
    q75 = float(lc["event_count"].quantile(0.75))

    if _is_spark_pandas(lc):
        lc["activity_segment"] = lc["event_count"].spark.transform(
            lambda c: _assign_activity_segment_spark(c, q25, q75))
    else:
        lc["activity_segment"] = _assign_activity_segment_numpy(lc["event_count"].to_numpy(), q25, q75)

    counts = lc["activity_segment"].value_counts()
    total = len(lc)
    rows = []
    for segment in head_as_list(counts.index, 50):
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
        recommendations=native_pd.DataFrame(rows),
    )


@dataclass
class EntityLifecycle:
    entity: str
    first_event: Timestamp
    last_event: Timestamp
    duration_days: int
    event_count: int


@dataclass
class TimeSeriesProfile:
    entity_column: str
    time_column: str
    total_events: int
    unique_entities: int
    time_span_days: int
    events_per_entity: DistributionStats
    entity_lifecycles: DataFrame
    avg_inter_event_days: Optional[float] = None
    first_event_date: Optional[Timestamp] = None
    last_event_date: Optional[Timestamp] = None


class TimeSeriesProfiler:
    SECONDS_PER_DAY = 86400

    def __init__(self, entity_column: str, time_column: str,
                 fallback_entity_columns: Optional[list[str]] = None):
        self.entity_column = entity_column
        self.time_column = time_column
        self._fallback_entity_columns = fallback_entity_columns or []

    def profile(self, df: DataFrame) -> TimeSeriesProfile:
        self._resolve_columns(df)
        df = self._prepare_dataframe(df)

        if len(df) == 0:
            return self._empty_profile()

        self._validate_columns(df)

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

    def _resolve_columns(self, df: DataFrame) -> None:
        candidates = [self.entity_column, *self._fallback_entity_columns]
        for col in candidates:
            try:
                self.entity_column = resolve_column_name(df.columns, col)
                break
            except KeyError:
                continue
        else:
            resolve_column_name(df.columns, self.entity_column)
        self.time_column = resolve_column_name(df.columns, self.time_column)

    def _validate_columns(self, df: DataFrame) -> None:
        if self.entity_column not in df.columns:
            raise KeyError(f"Entity column '{self.entity_column}' not found")
        if self.time_column not in df.columns:
            raise KeyError(f"Time column '{self.time_column}' not found")

    def _prepare_dataframe(self, df: DataFrame) -> DataFrame:
        df = df.copy()
        if not is_datetime64_any_dtype(df[self.time_column]):
            col = df[self.time_column]
            if is_numeric_dtype(col):
                sample_val = float(col.dropna().iloc[0])
                unit = _infer_epoch_unit(sample_val)
                df[self.time_column] = pd.to_datetime(col, unit=unit)
            else:
                df[self.time_column] = pd.to_datetime(col)
        return df

    def _compute_entity_lifecycles(self, df: DataFrame) -> DataFrame:
        lifecycles = groupby_multi_agg(df, self.entity_column, self.time_column, ["min", "max", "count"])
        lifecycles.columns = ["entity", "first_event", "last_event", "event_count"]
        lifecycles["duration_days"] = timedelta_to_days(lifecycles["last_event"] - lifecycles["first_event"])
        return lifecycles

    def _compute_events_distribution(self, lifecycles: DataFrame) -> DistributionStats:
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
        if len(df) == 0:
            return 0
        min_date = df[self.time_column].min()
        max_date = df[self.time_column].max()
        return (max_date - min_date).days

    def _compute_avg_inter_event_time(self, df: DataFrame) -> Optional[float]:
        if len(df) < 2:
            return None
        if _is_spark_pandas(df):
            return self._spark_avg_inter_event_time(df)
        sorted_df = df[[self.entity_column, self.time_column]].sort_values(
            [self.entity_column, self.time_column]
        )
        same_entity = sorted_df[self.entity_column] == sorted_df[self.entity_column].shift(1)
        tc = sorted_df[self.time_column]
        time_diffs = timestamp_diffs_seconds(tc) / self.SECONDS_PER_DAY
        valid = time_diffs[same_entity].dropna()
        if len(valid) == 0:
            return None
        return float(valid.mean())

    def _spark_avg_inter_event_time(self, df: DataFrame) -> Optional[float]:
        import pyspark.sql.functions as F  # noqa: N812
        from pyspark.sql.window import Window
        w = Window.partitionBy(self.entity_column).orderBy(self.time_column)
        tc, prev = F.col(self.time_column).cast("timestamp"), F.lag(self.time_column).over(w).cast("timestamp")
        result = (
            as_spark_df(df[[self.entity_column, self.time_column]])
            .withColumn("_diff", (F.unix_timestamp(tc) - F.unix_timestamp(prev)) / self.SECONDS_PER_DAY)
            .filter(F.col("_diff").isNotNull())
            .agg(F.mean("_diff"))
            .collect()[0][0]
        )
        return float(result) if result is not None else None

    def _empty_profile(self) -> TimeSeriesProfile:
        return TimeSeriesProfile(
            entity_column=self.entity_column,
            time_column=self.time_column,
            total_events=0,
            unique_entities=0,
            time_span_days=0,
            events_per_entity=DistributionStats(
                min=0, max=0, mean=0, median=0, std=0, q25=0, q75=0
            ),
            entity_lifecycles=native_pd.DataFrame(columns=[
                "entity", "first_event", "last_event", "duration_days", "event_count"
            ]),
            avg_inter_event_days=None,
        )
