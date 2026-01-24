from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .temporal_pattern_analyzer import SeasonalityPeriod
from .time_series_profiler import ActivitySegmentResult, LifecycleQuadrantResult

WINDOW_DAYS_MAP: Dict[str, Optional[float]] = {
    "24h": 1.0, "7d": 7.0, "14d": 14.0, "30d": 30.0,
    "90d": 90.0, "180d": 180.0, "365d": 365.0, "all_time": None,
}

SEASONALITY_WINDOW_MAP: Dict[int, str] = {
    1: "24h", 7: "7d", 14: "14d", 30: "30d", 90: "90d", 365: "365d",
}

TIMING_TOLERANCE = 0.5


@dataclass
class TemporalHeterogeneityResult:
    eta_squared_intensity: float
    eta_squared_event_count: float
    heterogeneity_level: str
    segmentation_advisory: str
    advisory_rationale: List[str]
    coverage_table: pd.DataFrame


@dataclass
class WindowUnionResult:
    windows: List[str]
    explanation: pd.DataFrame
    heterogeneity: TemporalHeterogeneityResult
    coverage_threshold: float
    feature_count_estimate: int


class WindowRecommendationCollector:
    ALL_CANDIDATE_WINDOWS = ["24h", "7d", "14d", "30d", "90d", "180d", "365d", "all_time"]

    def __init__(self, coverage_threshold: float = 0.10, always_include: Optional[List[str]] = None):
        self._coverage_threshold = coverage_threshold
        self._always_include = always_include if always_include is not None else ["all_time"]
        self._segment_lifecycles: Optional[pd.DataFrame] = None
        self._quadrant_lifecycles: Optional[pd.DataFrame] = None
        self._seasonality_periods: List[SeasonalityPeriod] = []
        self._inter_event_median: Optional[float] = None
        self._inter_event_mean: Optional[float] = None

    def add_segment_context(self, result: ActivitySegmentResult) -> None:
        self._segment_lifecycles = result.lifecycles

    def add_quadrant_context(self, result: LifecycleQuadrantResult) -> None:
        self._quadrant_lifecycles = result.lifecycles

    def add_seasonality_context(self, periods: List[SeasonalityPeriod]) -> None:
        self._seasonality_periods = periods

    def add_inter_event_context(self, median_days: float, mean_days: float) -> None:
        self._inter_event_median = median_days
        self._inter_event_mean = mean_days

    def compute_union(
        self, lifecycles: pd.DataFrame, time_span_days: int,
        min_coverage_ratio: float = 2.0,
        value_columns: int = 0, agg_funcs: int = 4,
    ) -> WindowUnionResult:
        rows = self._compute_coverage_rows(lifecycles, time_span_days, min_coverage_ratio)
        self._annotate_context(rows, lifecycles)
        selected = [r["window"] for r in rows if r["included"]]
        explanation = pd.DataFrame(rows)
        heterogeneity = self._compute_heterogeneity(lifecycles, selected)
        feature_count = value_columns * agg_funcs * len(selected) + len(selected) if value_columns > 0 else len(selected)
        return WindowUnionResult(
            windows=selected, explanation=explanation,
            heterogeneity=heterogeneity, coverage_threshold=self._coverage_threshold,
            feature_count_estimate=feature_count,
        )

    def _compute_coverage_rows(
        self, lifecycles: pd.DataFrame, time_span_days: int, min_coverage_ratio: float,
    ) -> List[Dict]:
        duration = lifecycles["duration_days"].astype(float)
        event_count = lifecycles["event_count"].astype(float)
        n = len(lifecycles)
        rows = []
        for window in self.ALL_CANDIDATE_WINDOWS:
            window_days = WINDOW_DAYS_MAP[window]
            if window_days is None:
                rows.append(self._all_time_row(n))
                continue
            has_span = duration >= window_days
            expected_events = event_count * (window_days / duration.clip(lower=1))
            has_density = expected_events >= 2
            beneficial = has_span & has_density
            coverage_pct = beneficial.mean()
            meaningful_pct = has_density[has_span].mean() if has_span.any() else 0.0
            beneficial_count = int(beneficial.sum())
            hard_excluded = time_span_days < window_days * min_coverage_ratio
            included, exclusion_reason = self._determine_inclusion(
                window, coverage_pct, hard_excluded,
            )
            rows.append({
                "window": window, "window_days": window_days,
                "coverage_pct": round(coverage_pct, 4),
                "meaningful_pct": round(meaningful_pct, 4),
                "beneficial_entities": beneficial_count,
                "primary_segments": [], "included": included,
                "exclusion_reason": exclusion_reason, "note": "",
            })
        return rows

    def _all_time_row(self, n: int) -> Dict:
        return {
            "window": "all_time", "window_days": None,
            "coverage_pct": 1.0, "meaningful_pct": 1.0,
            "beneficial_entities": n, "primary_segments": [],
            "included": True, "exclusion_reason": "", "note": "",
        }

    def _determine_inclusion(self, window: str, coverage_pct: float, hard_excluded: bool) -> Tuple[bool, str]:
        if hard_excluded:
            if window in self._always_include:
                return True, ""
            return False, f"Excluded: span < {WINDOW_DAYS_MAP[window] * 2:.0f}d required"
        if window in self._always_include:
            return True, ""
        if coverage_pct >= self._coverage_threshold:
            return True, ""
        return False, f"Coverage {coverage_pct:.1%} < threshold {self._coverage_threshold:.1%}"

    def _annotate_context(self, rows: List[Dict], lifecycles: pd.DataFrame) -> None:
        self._annotate_segments(rows, lifecycles)
        self._annotate_seasonality(rows)
        self._annotate_timing(rows)

    def _annotate_segments(self, rows: List[Dict], lifecycles: pd.DataFrame) -> None:
        context_lc = self._segment_lifecycles if self._segment_lifecycles is not None else self._quadrant_lifecycles
        if context_lc is None:
            return
        group_col = "activity_segment" if "activity_segment" in (context_lc.columns if self._segment_lifecycles is not None else []) else None
        if group_col is None and self._quadrant_lifecycles is not None and "lifecycle_quadrant" in self._quadrant_lifecycles.columns:
            group_col = "lifecycle_quadrant"
            context_lc = self._quadrant_lifecycles
        if group_col is None:
            return
        duration = context_lc["duration_days"].astype(float)
        event_count = context_lc["event_count"].astype(float)
        groups = context_lc[group_col]
        for row in rows:
            window_days = row["window_days"]
            if window_days is None:
                row["primary_segments"] = sorted(groups.unique().tolist())
                continue
            has_span = duration >= window_days
            expected_events = event_count * (window_days / duration.clip(lower=1))
            beneficial = has_span & (expected_events >= 2)
            if not beneficial.any():
                continue
            group_coverage = groups[beneficial].value_counts(normalize=True)
            top = group_coverage[group_coverage >= 0.15].index.tolist()
            row["primary_segments"] = sorted(top[:3])

    def _annotate_seasonality(self, rows: List[Dict]) -> None:
        if not self._seasonality_periods:
            return
        detected_windows = set()
        for sp in self._seasonality_periods:
            if sp.period in SEASONALITY_WINDOW_MAP:
                detected_windows.add(SEASONALITY_WINDOW_MAP[sp.period])
        for row in rows:
            if row["window"] in detected_windows:
                period_name = next(
                    (sp.period_name or f"{sp.period}d" for sp in self._seasonality_periods
                     if SEASONALITY_WINDOW_MAP.get(sp.period) == row["window"]), ""
                )
                row["note"] = f"Seasonality detected ({period_name})"

    def _annotate_timing(self, rows: List[Dict]) -> None:
        if self._inter_event_median is None:
            return
        for row in rows:
            window_days = row["window_days"]
            if window_days is None:
                continue
            ratio = self._inter_event_median / window_days if window_days > 0 else 0
            if TIMING_TOLERANCE <= ratio <= (1.0 / TIMING_TOLERANCE):
                existing = row["note"]
                timing_note = "Timing-aligned (median inter-event)"
                row["note"] = f"{existing}; {timing_note}" if existing else timing_note

    def _compute_heterogeneity(self, lifecycles: pd.DataFrame, selected_windows: List[str]) -> TemporalHeterogeneityResult:
        eta_intensity, eta_event = self._compute_eta_squared(lifecycles)
        level = self._classify_heterogeneity(max(eta_intensity, eta_event))
        cold_start_frac = self._cold_start_fraction(lifecycles)
        advisory, rationale = self._build_advisory(level, cold_start_frac, selected_windows, lifecycles)
        coverage_table = self._build_coverage_table(lifecycles, selected_windows)
        return TemporalHeterogeneityResult(
            eta_squared_intensity=eta_intensity,
            eta_squared_event_count=eta_event,
            heterogeneity_level=level,
            segmentation_advisory=advisory,
            advisory_rationale=rationale,
            coverage_table=coverage_table,
        )

    def _compute_eta_squared(self, lifecycles: pd.DataFrame) -> Tuple[float, float]:
        group_col = "lifecycle_quadrant" if "lifecycle_quadrant" in lifecycles.columns else None
        if group_col is None:
            return 0.0, 0.0
        groups = lifecycles[group_col]
        if groups.nunique() < 2:
            return 0.0, 0.0
        eta_intensity = self._eta_squared_for_variable(lifecycles, "intensity", groups)
        eta_event = self._eta_squared_for_variable(lifecycles, "event_count", groups)
        return eta_intensity, eta_event

    def _eta_squared_for_variable(self, df: pd.DataFrame, var: str, groups: pd.Series) -> float:
        if var not in df.columns:
            return 0.0
        values = df[var].astype(float)
        grand_mean = values.mean()
        ss_total = ((values - grand_mean) ** 2).sum()
        if ss_total == 0:
            return 0.0
        ss_between = 0.0
        for _, group_vals in values.groupby(groups):
            n_k = len(group_vals)
            mean_k = group_vals.mean()
            ss_between += n_k * (mean_k - grand_mean) ** 2
        return float(ss_between / ss_total)

    def _classify_heterogeneity(self, eta_max: float) -> str:
        if eta_max < 0.06:
            return "low"
        if eta_max < 0.14:
            return "moderate"
        return "high"

    def _cold_start_fraction(self, lifecycles: pd.DataFrame) -> float:
        cold_labels = {"One-shot", "One-time"}
        cold_count = 0
        for col in ("lifecycle_quadrant", "activity_segment"):
            if col in lifecycles.columns:
                cold_count = max(cold_count, lifecycles[col].isin(cold_labels).sum())
        return cold_count / len(lifecycles) if len(lifecycles) > 0 else 0.0

    def _build_advisory(
        self, level: str, cold_start_frac: float, selected_windows: List[str], lifecycles: pd.DataFrame,
    ) -> Tuple[str, List[str]]:
        rationale: List[str] = []
        if level == "low":
            rationale.append("Low temporal diversity across quadrants")
            rationale.append("Union strategy loses minimal signal")
            return "single_model", rationale
        if level == "high" and cold_start_frac > 0.30:
            rationale.append("High temporal diversity across quadrants")
            rationale.append(f"Large cold-start population ({cold_start_frac:.0%} One-time/One-shot)")
            rationale.append("Consider separate handling for entities with vs without history")
            return "consider_separate_models", rationale
        rationale.append(f"{level.capitalize()} temporal diversity across quadrants")
        rationale.append("Union windows still pragmatic for feature engineering")
        rationale.append("Model may benefit from knowing entity's engagement pattern")
        return "consider_segment_feature", rationale

    def _build_coverage_table(self, lifecycles: pd.DataFrame, selected_windows: List[str]) -> pd.DataFrame:
        duration = lifecycles["duration_days"].astype(float)
        event_count = lifecycles["event_count"].astype(float)
        rows = []
        for window in selected_windows:
            window_days = WINDOW_DAYS_MAP.get(window)
            if window_days is None:
                rows.append({"window": "all_time", "coverage_pct": 1.0, "meaningful_pct": 1.0, "zero_risk_pct": 0.0})
                continue
            has_span = duration >= window_days
            coverage = has_span.mean()
            expected_events = event_count * (window_days / duration.clip(lower=1))
            meaningful = (has_span & (expected_events >= 2)).mean()
            zero_risk = 1.0 - meaningful
            rows.append({
                "window": window,
                "coverage_pct": round(float(coverage), 4),
                "meaningful_pct": round(float(meaningful), 4),
                "zero_risk_pct": round(float(zero_risk), 4),
            })
        return pd.DataFrame(rows)
