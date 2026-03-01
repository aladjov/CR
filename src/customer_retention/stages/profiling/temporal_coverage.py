from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from customer_retention.core.compat import (
    Timedelta,
    Timestamp,
    is_datetime64_any_dtype,
    native_pd,
    to_datetime,
)

from .window_recommendation import WINDOW_DAYS_MAP

DEFAULT_CANDIDATE_WINDOWS = ["7d", "30d", "90d", "180d", "365d", "all_time"]
GAP_THRESHOLD_MULTIPLIER = 3.0
VOLUME_CHANGE_GROWING = 0.25
VOLUME_CHANGE_DECLINING = -0.25


@dataclass
class TemporalGap:
    start: Timestamp
    end: Timestamp
    duration_days: float
    severity: str


@dataclass
class EntityWindowCoverage:
    window: str
    window_days: Optional[float]
    active_entities: int
    coverage_pct: float


@dataclass
class DriftImplication:
    risk_level: str
    volume_drift_risk: str
    population_stability: float
    regime_count: int
    regime_boundaries: List[Timestamp]
    recommended_training_start: Optional[Timestamp]
    rationale: List[str]


@dataclass
class TemporalCoverageResult:
    time_span_days: int
    first_event: Timestamp
    last_event: Timestamp
    gaps: List[TemporalGap]
    entity_window_coverage: List[EntityWindowCoverage]
    volume_trend: str
    volume_change_pct: float
    recommendations: List[str]
    events_over_time: native_pd.Series
    new_entities_over_time: native_pd.Series


@dataclass
class TemporalComparison:
    entity_ratio: float
    event_ratio: float
    events_per_entity_change: float
    numeric_columns_drifted_pct: float
    categorical_shift_detected: bool
    regime_shift_detected: bool
    regime_shift_severity: str
    target_rate_delta: Optional[float]
    representativeness_score: float


def _to_native_series(series, name: str = "") -> native_pd.Series:
    if isinstance(series, native_pd.Series):
        result = series
    elif hasattr(series, "to_pandas"):
        result = series.to_pandas()
    else:
        result = native_pd.Series(series)
    if name:
        result.name = name
    return result


def analyze_temporal_coverage(
    df, entity_column: str, time_column: str,
    candidate_windows: Optional[List[str]] = None,
    reference_date: Optional[Timestamp] = None,
) -> TemporalCoverageResult:
    if not is_datetime64_any_dtype(df[time_column]):
        df = df.copy()
        df[time_column] = to_datetime(df[time_column])

    first_event = df[time_column].min()
    last_event = df[time_column].max()
    time_span_days = max(0, (last_event - first_event).days)
    ref_date = reference_date if reference_date is not None else last_event
    windows = candidate_windows if candidate_windows is not None else DEFAULT_CANDIDATE_WINDOWS

    grouper_freq, range_freq = _choose_freq(time_span_days)
    events_over_time = _resample_event_counts(df, time_column, grouper_freq)
    new_entities = _resample_new_entities(df, entity_column, time_column, grouper_freq)

    gaps = _detect_gaps(events_over_time, range_freq)
    coverage = _compute_entity_window_coverage(df, entity_column, time_column, ref_date, windows)
    volume_trend, volume_change = _assess_volume_trend(events_over_time)
    recommendations = _build_recommendations(gaps, volume_trend, volume_change, time_span_days, coverage)

    return TemporalCoverageResult(
        time_span_days=time_span_days, first_event=first_event, last_event=last_event,
        gaps=gaps, entity_window_coverage=coverage,
        volume_trend=volume_trend, volume_change_pct=volume_change,
        recommendations=recommendations,
        events_over_time=events_over_time, new_entities_over_time=new_entities,
    )


def _resample_event_counts(df, time_column: str, freq: str) -> native_pd.Series:
    daily_counts = df.groupby(df[time_column].dt.date).size()
    daily = _to_native_series(daily_counts).sort_index()
    daily.index = native_pd.DatetimeIndex(daily.index)
    raw = daily.resample(freq).sum().fillna(0).astype(int)
    return _to_native_series(raw, "event_count")


def _resample_new_entities(df, entity_column: str, time_column: str, freq: str) -> native_pd.Series:
    first_date = df.groupby(entity_column)[time_column].min().dt.date
    daily_counts = first_date.value_counts().sort_index()
    daily = _to_native_series(daily_counts).sort_index()
    daily.index = native_pd.DatetimeIndex(daily.index)
    raw = daily.resample(freq).sum().fillna(0).astype(int)
    return _to_native_series(raw, "new_entities")


def derive_drift_implications(result: TemporalCoverageResult) -> DriftImplication:
    major_gaps = [g for g in result.gaps if g.severity == "major"]
    regime_boundaries = [g.end for g in major_gaps]
    regime_count = len(regime_boundaries) + 1
    recommended_start = regime_boundaries[-1] if regime_boundaries else None
    volume_drift_risk = _volume_to_drift_risk(result.volume_trend)
    population_stability = _compute_population_stability(result.new_entities_over_time)
    risk_level = _assess_overall_drift_risk(
        volume_drift_risk, population_stability, regime_count, result.time_span_days,
    )
    rationale = _build_drift_rationale(
        volume_drift_risk, result.volume_change_pct, population_stability,
        regime_count, result.time_span_days, major_gaps,
    )
    return DriftImplication(
        risk_level=risk_level, volume_drift_risk=volume_drift_risk,
        population_stability=population_stability, regime_count=regime_count,
        regime_boundaries=regime_boundaries, recommended_training_start=recommended_start,
        rationale=rationale,
    )


def _volume_to_drift_risk(volume_trend: str) -> str:
    if volume_trend == "growing":
        return "growing"
    if volume_trend == "declining":
        return "declining"
    return "none"


def _compute_population_stability(new_entities: native_pd.Series) -> float:
    if len(new_entities) < 4:
        return 0.5
    total_new = new_entities.sum()
    if total_new == 0:
        return 1.0
    mid = len(new_entities) // 2
    second_half_new = new_entities.iloc[mid:].sum()
    fresh_fraction = second_half_new / total_new
    positive = new_entities[new_entities > 0]
    burstiness = min(1.0, (positive.std() / positive.mean()) / 2.0) if len(positive) >= 2 and positive.mean() > 0 else 0.5
    return round(max(0.0, min(1.0, 1.0 - fresh_fraction * 0.6 - burstiness * 0.4)), 4)


def _assess_overall_drift_risk(
    volume_drift_risk: str, population_stability: float,
    regime_count: int, time_span_days: int,
) -> str:
    risk_score = 0.0
    if volume_drift_risk != "none":
        risk_score += 0.3 if volume_drift_risk == "growing" else 0.4
    if population_stability < 0.5:
        risk_score += 0.3
    elif population_stability < 0.7:
        risk_score += 0.15
    if regime_count > 1:
        risk_score += 0.2 * min(regime_count - 1, 3)
    if time_span_days < 90:
        risk_score += 0.3
    if risk_score < 0.25:
        return "low"
    if risk_score < 0.5:
        return "moderate"
    return "high"


def _build_drift_rationale(
    volume_drift_risk: str, volume_change_pct: float,
    population_stability: float, regime_count: int,
    time_span_days: int, major_gaps: List[TemporalGap],
) -> List[str]:
    rationale = []
    if volume_drift_risk == "declining":
        rationale.append(
            f"Volume declining ({volume_change_pct:+.0%}) — feature distributions "
            f"computed over recent windows will differ from historical baselines"
        )
    elif volume_drift_risk == "growing":
        rationale.append(
            f"Volume growing ({volume_change_pct:+.0%}) — earlier periods have sparser "
            f"data; model trained on full history may underweight recent patterns"
        )
    if regime_count > 1:
        total_gap_days = sum(g.duration_days for g in major_gaps)
        rationale.append(
            f"{regime_count} distinct data regimes separated by {len(major_gaps)} major "
            f"gap(s) ({total_gap_days:.0f}d total) — training across regime boundaries "
            f"mixes incompatible distributions"
        )
    if population_stability < 0.5:
        rationale.append(
            f"Low population stability ({population_stability:.2f}) — entity influx is "
            f"highly uneven, indicating population composition drift"
        )
    elif population_stability < 0.7:
        rationale.append(
            f"Moderate population stability ({population_stability:.2f}) — some variation "
            f"in entity influx rate suggests gradual population shift"
        )
    if time_span_days < 90:
        rationale.append(
            f"Short observation span ({time_span_days}d) — insufficient history to "
            f"establish stable baselines for drift detection"
        )
    if not rationale:
        rationale.append("Stable volume, consistent population influx, no regime breaks detected")
    return rationale


def _choose_freq(time_span_days: int) -> tuple:
    if time_span_days <= 90:
        return "D", "D"
    if time_span_days <= 730:
        return "W-MON", "W-MON"
    return "ME", "ME"


def _detect_gaps(events_over_time: native_pd.Series, freq: str) -> List[TemporalGap]:
    if len(events_over_time) < 3:
        return []
    series = events_over_time
    median_volume = series[series > 0].median() if (series > 0).any() else 0
    if median_volume == 0:
        return []
    threshold = max(1, median_volume / GAP_THRESHOLD_MULTIPLIER)

    gaps: List[TemporalGap] = []
    gap_start = None
    for ts, vol in series.items():
        if vol < threshold:
            if gap_start is None:
                gap_start = ts
        else:
            if gap_start is not None:
                duration = (ts - gap_start).days
                if duration >= 3:
                    gaps.append(TemporalGap(
                        start=gap_start, end=ts,
                        duration_days=float(duration),
                        severity=_classify_gap_severity(duration),
                    ))
                gap_start = None
    if gap_start is not None:
        idx_list = list(series.index)
        end = idx_list[-1]
        duration = (end - gap_start).days
        if duration >= 3:
            gaps.append(TemporalGap(
                start=gap_start, end=end,
                duration_days=float(duration),
                severity=_classify_gap_severity(duration),
            ))
    return gaps


def _classify_gap_severity(duration_days: float) -> str:
    if duration_days < 7:
        return "minor"
    if duration_days < 30:
        return "moderate"
    return "major"


def _compute_entity_window_coverage(
    df, entity_column: str, time_column: str, reference_date, windows: List[str],
) -> List[EntityWindowCoverage]:
    total_entities = df[entity_column].nunique()
    results = []
    for window in windows:
        window_days = WINDOW_DAYS_MAP.get(window)
        if window_days is None:
            results.append(EntityWindowCoverage(
                window=window, window_days=None,
                active_entities=total_entities, coverage_pct=1.0,
            ))
            continue
        cutoff = reference_date - Timedelta(days=window_days)
        mask = (df[time_column] >= cutoff) & (df[time_column] <= reference_date)
        active = df.loc[mask, entity_column].nunique()
        results.append(EntityWindowCoverage(
            window=window, window_days=window_days,
            active_entities=active, coverage_pct=active / total_entities if total_entities > 0 else 0.0,
        ))
    return results


def _assess_volume_trend(events_over_time: native_pd.Series) -> tuple:
    if len(events_over_time) < 4:
        return "stable", 0.0
    mid = len(events_over_time) // 2
    first_half = events_over_time.iloc[:mid].mean()
    second_half = events_over_time.iloc[mid:].mean()
    if first_half == 0:
        change_pct = 1.0 if second_half > 0 else 0.0
    else:
        change_pct = (second_half - first_half) / first_half
    if change_pct > VOLUME_CHANGE_GROWING:
        return "growing", round(float(change_pct), 4)
    if change_pct < VOLUME_CHANGE_DECLINING:
        return "declining", round(float(change_pct), 4)
    return "stable", round(float(change_pct), 4)


def _build_recommendations(
    gaps: List[TemporalGap], volume_trend: str, volume_change: float,
    time_span_days: int, coverage: List[EntityWindowCoverage],
) -> List[str]:
    recs = []
    major_gaps = [g for g in gaps if g.severity == "major"]
    if major_gaps:
        total_gap_days = sum(g.duration_days for g in major_gaps)
        recs.append(
            f"Data has {len(major_gaps)} major gap(s) totaling {total_gap_days:.0f} days "
            f"— consider excluding gap periods or treating them as separate epochs"
        )
    if volume_trend == "declining":
        recs.append(
            f"Volume declining ({volume_change:+.0%}) — recent data may underrepresent entity activity; "
            f"verify data pipeline completeness"
        )
    if volume_trend == "growing":
        recs.append(
            f"Volume growing ({volume_change:+.0%}) — earlier periods have sparser data; "
            f"longer windows may mix density regimes"
        )
    if time_span_days < 90:
        recs.append(
            f"Limited time span ({time_span_days}d) — only short aggregation windows (7d, 30d) are reliable"
        )
    low_coverage = [c for c in coverage if c.window_days is not None and c.coverage_pct < 0.10]
    if low_coverage:
        windows_str = ", ".join(c.window for c in low_coverage)
        recs.append(f"Very few entities active in windows [{windows_str}] — these may produce mostly zeros")
    return recs


def compute_temporal_comparison(
    df,
    entity_column: str,
    time_column: str,
    target_column: Optional[str] = None,
    recent_days: int = 90,
    reference_date: Optional[Timestamp] = None,
) -> TemporalComparison:
    if not is_datetime64_any_dtype(df[time_column]):
        df = df.copy()
        df[time_column] = to_datetime(df[time_column])
    ref_date = reference_date if reference_date is not None else df[time_column].max()
    cutoff = ref_date - Timedelta(days=recent_days)

    mask_recent = df[time_column] >= cutoff
    df_full = df
    df_recent = df.loc[mask_recent]

    entities_full = df_full[entity_column].nunique()
    entities_recent = df_recent[entity_column].nunique() if len(df_recent) > 0 else 0
    entity_ratio = entities_recent / entities_full if entities_full > 0 else 0.0

    events_full = len(df_full)
    events_recent = len(df_recent)
    event_ratio = events_recent / events_full if events_full > 0 else 0.0

    density_full = events_full / entities_full if entities_full > 0 else 0.0
    density_recent = events_recent / entities_recent if entities_recent > 0 else 0.0
    events_per_entity_change = (
        (density_recent - density_full) / density_full if density_full > 0 else 0.0
    )

    numeric_drifted_pct, categorical_shift = _compute_drift_signals(
        df_full, df_recent, entity_column, time_column,
    )

    coverage_result = analyze_temporal_coverage(
        df_full, entity_column, time_column, reference_date=ref_date,
    )
    drift_impl = derive_drift_implications(coverage_result)
    regime_shift_detected = drift_impl.regime_count > 1
    regime_shift_severity = _drift_risk_to_severity(drift_impl.risk_level)

    target_rate_delta = None
    if target_column and target_column in df_full.columns and target_column in df_recent.columns:
        full_series = df_full[target_column].dropna()
        recent_series = df_recent[target_column].dropna()
        if len(full_series) > 0 and len(recent_series) > 0:
            try:
                target_rate_delta = float(recent_series.mean() - full_series.mean())
            except (TypeError, ValueError):
                pass

    population_stability = drift_impl.population_stability
    representativeness_score = _compute_representativeness_score(
        entity_ratio, event_ratio, numeric_drifted_pct, population_stability,
    )

    return TemporalComparison(
        entity_ratio=round(entity_ratio, 4),
        event_ratio=round(event_ratio, 4),
        events_per_entity_change=round(events_per_entity_change, 4),
        numeric_columns_drifted_pct=round(numeric_drifted_pct, 4),
        categorical_shift_detected=categorical_shift,
        regime_shift_detected=regime_shift_detected,
        regime_shift_severity=regime_shift_severity,
        target_rate_delta=round(target_rate_delta, 4) if target_rate_delta is not None else None,
        representativeness_score=round(representativeness_score, 4),
    )


def _compute_drift_signals(
    df_full, df_recent, entity_column: str, time_column: str,
) -> tuple:
    from customer_retention.core.config import ColumnType

    from .drift_detector import BaselineDriftChecker

    exclude = {entity_column, time_column}
    numeric_cols = [
        c for c in df_full.select_dtypes(include=[np.number]).columns
        if c not in exclude
    ]
    categorical_cols = [
        c for c in df_full.select_dtypes(include=["object", "category"]).columns
        if c not in exclude
    ]

    if len(df_recent) == 0:
        return 0.0, False

    numeric_drifted = 0
    if numeric_cols:
        checker = BaselineDriftChecker()
        for col in numeric_cols:
            checker.set_baseline(col, df_full[col].dropna(), ColumnType.NUMERIC_CONTINUOUS)
        drift_results = checker.detect_drift_all(df_recent)
        numeric_drifted = sum(1 for r in drift_results if r.has_drift)

    numeric_drifted_pct = numeric_drifted / len(numeric_cols) if numeric_cols else 0.0

    categorical_shift = False
    if categorical_cols:
        checker = BaselineDriftChecker()
        for col in categorical_cols:
            checker.set_baseline(col, df_full[col].dropna(), ColumnType.CATEGORICAL_NOMINAL)
        cat_results = checker.detect_drift_all(df_recent)
        categorical_shift = any(r.has_drift for r in cat_results)

    return numeric_drifted_pct, categorical_shift


def _drift_risk_to_severity(risk_level: str) -> str:
    if risk_level == "high":
        return "high"
    if risk_level == "moderate":
        return "medium"
    return "low"


def _compute_representativeness_score(
    entity_ratio: float,
    event_ratio: float,
    drift_pct: float,
    population_stability: float,
) -> float:
    score = (
        entity_ratio * 0.30
        + event_ratio * 0.25
        + (1.0 - drift_pct) * 0.25
        + population_stability * 0.20
    )
    return max(0.0, min(1.0, score))
