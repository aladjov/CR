from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd

from .window_recommendation import WINDOW_DAYS_MAP

DEFAULT_CANDIDATE_WINDOWS = ["7d", "30d", "90d", "180d", "365d", "all_time"]
GAP_THRESHOLD_MULTIPLIER = 3.0
VOLUME_CHANGE_GROWING = 0.25
VOLUME_CHANGE_DECLINING = -0.25


@dataclass
class FeatureAvailability:
    column: str
    first_valid_date: Optional[pd.Timestamp]
    last_valid_date: Optional[pd.Timestamp]
    valid_count: int
    total_count: int
    coverage_pct: float
    availability_type: str
    days_from_start: Optional[int]
    days_before_end: Optional[int]


@dataclass
class FeatureAvailabilityResult:
    data_start: pd.Timestamp
    data_end: pd.Timestamp
    time_span_days: int
    features: List[FeatureAvailability]
    new_tracking: List[str]
    retired_tracking: List[str]
    partial_window: List[str]
    recommendations: List[Dict]


@dataclass
class TemporalGap:
    start: pd.Timestamp
    end: pd.Timestamp
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
    regime_boundaries: List[pd.Timestamp]
    recommended_training_start: Optional[pd.Timestamp]
    rationale: List[str]


@dataclass
class TemporalCoverageResult:
    time_span_days: int
    first_event: pd.Timestamp
    last_event: pd.Timestamp
    gaps: List[TemporalGap]
    entity_window_coverage: List[EntityWindowCoverage]
    volume_trend: str
    volume_change_pct: float
    recommendations: List[str]
    events_over_time: pd.Series
    new_entities_over_time: pd.Series


def analyze_temporal_coverage(
    df: pd.DataFrame, entity_column: str, time_column: str,
    candidate_windows: Optional[List[str]] = None,
    reference_date: Optional[pd.Timestamp] = None,
) -> TemporalCoverageResult:
    times = pd.to_datetime(df[time_column])
    first_event = times.min()
    last_event = times.max()
    time_span_days = max(0, (last_event - first_event).days)
    ref_date = reference_date if reference_date is not None else last_event
    windows = candidate_windows if candidate_windows is not None else DEFAULT_CANDIDATE_WINDOWS

    grouper_freq, range_freq = _choose_freq(time_span_days)
    df_indexed = pd.DataFrame({"_t": times, "_e": df[entity_column].values})
    df_indexed = df_indexed.set_index("_t").sort_index()

    events_over_time = df_indexed.resample(grouper_freq).size()
    events_over_time.name = "event_count"

    first_per_entity = df.assign(_t=times).groupby(entity_column)["_t"].min()
    fpe_indexed = pd.DataFrame({"_count": 1}, index=first_per_entity.values)
    fpe_indexed.index.name = "_t"
    new_entities = fpe_indexed.resample(grouper_freq)["_count"].sum().fillna(0).astype(int)
    new_entities.name = "new_entities"

    gaps = _detect_gaps(events_over_time, range_freq)
    coverage = _compute_entity_window_coverage(df, entity_column, times, ref_date, windows)
    volume_trend, volume_change = _assess_volume_trend(events_over_time)
    recommendations = _build_recommendations(gaps, volume_trend, volume_change, time_span_days, coverage)

    return TemporalCoverageResult(
        time_span_days=time_span_days, first_event=first_event, last_event=last_event,
        gaps=gaps, entity_window_coverage=coverage,
        volume_trend=volume_trend, volume_change_pct=volume_change,
        recommendations=recommendations,
        events_over_time=events_over_time, new_entities_over_time=new_entities,
    )


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


def _compute_population_stability(new_entities: pd.Series) -> float:
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


def _detect_gaps(events_over_time: pd.Series, freq: str) -> List[TemporalGap]:
    if len(events_over_time) < 3:
        return []
    series = events_over_time.copy()
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
        end = series.index[-1]
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
    df: pd.DataFrame, entity_column: str, times: pd.Series,
    reference_date: pd.Timestamp, windows: List[str],
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
        cutoff = reference_date - pd.Timedelta(days=window_days)
        mask = (times >= cutoff) & (times <= reference_date)
        active = df.loc[mask, entity_column].nunique()
        results.append(EntityWindowCoverage(
            window=window, window_days=window_days,
            active_entities=active, coverage_pct=active / total_entities if total_entities > 0 else 0.0,
        ))
    return results


def _assess_volume_trend(events_over_time: pd.Series) -> tuple:
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


def analyze_feature_availability(df: pd.DataFrame, time_column: str, exclude_columns: Optional[List[str]] = None, late_start_threshold_pct: float = 10.0, early_end_threshold_pct: float = 10.0) -> FeatureAvailabilityResult:
    times = pd.to_datetime(df[time_column])
    data_start, data_end = times.min(), times.max()
    time_span_days = max(1, (data_end - data_start).days)
    late_threshold_days = time_span_days * late_start_threshold_pct / 100
    early_threshold_days = time_span_days * early_end_threshold_pct / 100

    exclude = set(exclude_columns or []) | {time_column}
    columns_to_check = [c for c in df.columns if c not in exclude]

    features = []
    new_tracking, retired_tracking, partial_window = [], [], []

    for col in columns_to_check:
        valid_mask = df[col].notna()
        valid_count = valid_mask.sum()
        total_count = len(df)
        coverage_pct = valid_count / total_count * 100 if total_count > 0 else 0

        if valid_count == 0:
            features.append(FeatureAvailability(
                column=col, first_valid_date=None, last_valid_date=None,
                valid_count=0, total_count=total_count, coverage_pct=0,
                availability_type="empty", days_from_start=None, days_before_end=None
            ))
            continue

        valid_times = times[valid_mask]
        first_valid, last_valid = valid_times.min(), valid_times.max()
        days_from_start = (first_valid - data_start).days
        days_before_end = (data_end - last_valid).days

        is_late_start = days_from_start > late_threshold_days
        is_early_end = days_before_end > early_threshold_days

        if is_late_start and is_early_end:
            availability_type = "partial_window"
            partial_window.append(col)
        elif is_late_start:
            availability_type = "new_tracking"
            new_tracking.append(col)
        elif is_early_end:
            availability_type = "retired"
            retired_tracking.append(col)
        else:
            availability_type = "full"

        features.append(FeatureAvailability(
            column=col, first_valid_date=first_valid, last_valid_date=last_valid,
            valid_count=valid_count, total_count=total_count, coverage_pct=coverage_pct,
            availability_type=availability_type, days_from_start=days_from_start,
            days_before_end=days_before_end
        ))

    recommendations = _build_availability_recommendations(
        features, new_tracking, retired_tracking, partial_window, time_span_days
    )

    return FeatureAvailabilityResult(
        data_start=data_start, data_end=data_end, time_span_days=time_span_days,
        features=features, new_tracking=new_tracking, retired_tracking=retired_tracking,
        partial_window=partial_window, recommendations=recommendations
    )


def _find_feature(features: List[FeatureAvailability], col: str) -> Optional[FeatureAvailability]:
    return next((f for f in features if f.column == col), None)


def _build_new_tracking_rec(feat: FeatureAvailability, col: str) -> Dict:
    return {
        "column": col, "issue": "new_tracking", "priority": "high",
        "reason": f"Tracking started {feat.days_from_start}d after data start ({feat.coverage_pct:.0f}% coverage)",
        "options": [
            f"Filter training data to start from {feat.first_valid_date.date()}",
            f"Create '{col}_available' indicator for models",
            "Exclude from features if coverage too low"
        ]
    }


def _build_retired_rec(feat: FeatureAvailability, col: str) -> Dict:
    return {
        "column": col, "issue": "retired", "priority": "high",
        "reason": f"Tracking stopped {feat.days_before_end}d before data end ({feat.coverage_pct:.0f}% coverage)",
        "options": [
            f"Filter data to end at {feat.last_valid_date.date()} for this feature",
            f"Create '{col}_available' indicator",
            "Exclude if feature won't be available for scoring"
        ]
    }


def _build_partial_window_rec(feat: FeatureAvailability, col: str) -> Dict:
    return {
        "column": col, "issue": "partial_window", "priority": "high",
        "reason": f"Only available {feat.first_valid_date.date()} to {feat.last_valid_date.date()} ({feat.coverage_pct:.0f}% coverage)",
        "options": [
            "Use only within available window",
            "Consider excluding - limited applicability",
            f"Create '{col}_available' indicator if keeping"
        ]
    }


def _build_availability_recommendations(
    features: List[FeatureAvailability], new_tracking: List[str],
    retired_tracking: List[str], partial_window: List[str], time_span_days: int,
) -> List[Dict]:
    recs = []
    builders = [
        (new_tracking, _build_new_tracking_rec),
        (retired_tracking, _build_retired_rec),
        (partial_window, _build_partial_window_rec),
    ]
    for tracking_list, build_fn in builders:
        for col in tracking_list:
            feat = _find_feature(features, col)
            if feat is not None:
                recs.append(build_fn(feat, col))

    problem_cols = new_tracking + retired_tracking + partial_window
    if problem_cols:
        recs.append({
            "column": "_general_", "issue": "train_test_split", "priority": "high",
            "reason": f"{len(problem_cols)} columns have availability boundaries",
            "options": [
                "Ensure train/test split doesn't cross availability boundaries",
                "Use time-based split after latest tracking start date",
                "Document which features are unavailable for which periods"
            ]
        })

    return recs
