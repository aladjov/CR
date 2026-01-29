from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from customer_retention.core.compat import DataFrame


@dataclass
class PatternAnalysisConfig:
    entity_column: str
    time_column: str
    target_column: Optional[str] = None
    aggregation_windows: List[str] = field(default_factory=list)
    velocity_window_days: int = 7
    short_momentum_window: int = 7
    long_momentum_window: int = 30
    rolling_window: int = 7
    sparkline_columns: List[str] = field(default_factory=list)
    sparkline_freq: str = "W"
    sparkline_agg: str = "mean"
    has_target: bool = False
    is_event_level: bool = True

    @classmethod
    def from_findings(cls, findings: Any, target_column: Optional[str] = None, window_override: Optional[List[str]] = None) -> "PatternAnalysisConfig":
        ts_meta = findings.time_series_metadata
        if ts_meta is None:
            raise ValueError("Findings do not contain time series metadata. Run notebook 01a first.")

        windows = window_override or ts_meta.suggested_aggregations or ["7d", "30d", "90d"]
        target_col = target_column or findings.target_column

        config = cls(
            entity_column=ts_meta.entity_column, time_column=ts_meta.time_column,
            target_column=target_col, aggregation_windows=windows,
            has_target=target_col is not None, is_event_level=True)
        config._derive_window_settings()
        return config

    def _derive_window_settings(self):
        if not self.aggregation_windows:
            return
        parsed = [self._parse_window_to_days(w) for w in self.aggregation_windows]
        window_days = sorted([d for d in parsed if d is not None])
        if not window_days:
            return
        shortest = window_days[0]
        self.velocity_window_days = shortest
        self.rolling_window = shortest
        self.short_momentum_window = shortest
        self.long_momentum_window = window_days[1] if len(window_days) >= 2 else shortest * 4

    def _parse_window_to_days(self, window: str) -> Optional[int]:
        if not window:
            return None
        w = window.lower().strip()
        multipliers = {"d": 1, "w": 7, "m": 30}
        for suffix, mult in multipliers.items():
            if w.endswith(suffix):
                try:
                    return int(w[:-1]) * mult
                except ValueError:
                    return None
        try:
            return int(w)
        except ValueError:
            return None

    def get_momentum_pairs(self) -> List[Tuple[int, int]]:
        if len(self.aggregation_windows) < 2:
            return [(self.short_momentum_window, self.long_momentum_window)]
        window_days = sorted({d for w in self.aggregation_windows if (d := self._parse_window_to_days(w))})
        pairs = [(window_days[i], window_days[i + 1]) for i in range(len(window_days) - 1)]
        return pairs if pairs else [(self.short_momentum_window, self.long_momentum_window)]

    def format_config(self) -> str:
        lines = [
            "=" * 70,
            "PATTERN ANALYSIS CONFIGURATION",
            "=" * 70,
            f"\nCore Columns:\n   Entity: {self.entity_column}\n   Time:   {self.time_column}",
            f"   Target: {self.target_column or '(none)'}",
            f"\nAggregation Windows (from findings):\n   {self.aggregation_windows}",
            f"\nDerived Settings:\n   Velocity window:  {self.velocity_window_days} days",
            f"   Rolling window:   {self.rolling_window} days\n   Momentum pairs:   {self.get_momentum_pairs()}",
        ]
        if self.sparkline_columns:
            lines.append(f"\nSparkline Config:\n   Columns: {self.sparkline_columns}")
            lines.append(f"   Frequency: {self.sparkline_freq}\n   Aggregation: {self.sparkline_agg}")
        return "\n".join(lines)

    def print_config(self):
        print(self.format_config())

    def configure_sparklines(self, df: DataFrame, columns: Optional[List[str]] = None, max_columns: int = 5):
        if columns:
            self.sparkline_columns = columns[:max_columns]
            return
        exclude = {self.entity_column, self.time_column, self.target_column} - {None}
        candidates = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude]
        self.sparkline_columns = candidates[:max_columns]


@dataclass
class PatternAnalysisResult:
    trend_detected: bool = False
    trend_direction: Optional[str] = None
    trend_strength: float = 0.0
    seasonality_detected: bool = False
    seasonality_periods: List[str] = field(default_factory=list)
    recency_effect: bool = False
    recency_correlation: float = 0.0
    cohort_effect: bool = False
    cohort_trend: Optional[str] = None
    velocity_features_recommended: List[str] = field(default_factory=list)
    momentum_features_recommended: List[str] = field(default_factory=list)

    def format_summary(self) -> str:
        lines = ["\n" + "=" * 70 + "\nPATTERN ANALYSIS SUMMARY\n" + "=" * 70]
        patterns = []
        if self.trend_detected:
            patterns.append(f"Trend: {self.trend_direction} (strength: {self.trend_strength:.2f})")
        if self.seasonality_detected:
            patterns.append(f"Seasonality: {', '.join(self.seasonality_periods)}")
        if self.recency_effect:
            patterns.append(f"Recency effect: r={self.recency_correlation:.2f}")
        if self.cohort_effect:
            patterns.append(f"Cohort effect: {self.cohort_trend}")
        if patterns:
            lines.append("\nDetected Patterns:")
            for p in patterns:
                lines.append(f"   - {p}")
        else:
            lines.append("\n   No significant patterns detected")
        if self.velocity_features_recommended:
            lines.append(f"\nRecommended velocity features: {self.velocity_features_recommended}")
        if self.momentum_features_recommended:
            lines.append(f"Recommended momentum features: {self.momentum_features_recommended}")
        return "\n".join(lines)

    def print_summary(self):
        print(self.format_summary())


def get_sparkline_frequency(time_span_days: int) -> str:
    if time_span_days <= 60:
        return "D"
    return "W" if time_span_days <= 365 else "ME"


def select_columns_by_variance(df: DataFrame, numeric_cols: List[str], max_cols: int = 6) -> List[str]:
    scores = {}
    for col in numeric_cols:
        if col not in df.columns:
            continue
        col_data = df[col].dropna()
        if len(col_data) == 0:
            continue
        std_val, mean_val = col_data.std(), abs(col_data.mean())
        if std_val == 0 or mean_val < 1e-10:
            continue
        cv = std_val / mean_val
        scores[col] = cv if not np.isnan(cv) else 0
    return sorted(scores, key=scores.get, reverse=True)[:max_cols]


def validate_not_event_level(
    df: DataFrame, entity_column: str, target_column: Optional[str]
) -> None:
    if target_column is None:
        return
    n_entities, n_rows = df[entity_column].nunique(), len(df)
    if n_entities < n_rows:
        raise ValueError(
            f"Target comparisons not allowed on event-level data. "
            f"Found {n_rows:,} rows but only {n_entities:,} entities. "
            f"Aggregate to entity level first using TimeWindowAggregator, "
            f"or use select_columns_by_variance() for column selection."
        )


def get_analysis_frequency(time_span_days: int) -> Tuple[str, str]:
    if time_span_days <= 90:
        return "D", "Daily"
    return ("W", "Weekly") if time_span_days <= 365 else ("ME", "Monthly")


@dataclass
class SparklineData:
    column: str
    weeks: List
    retained_values: List[float]
    churned_values: Optional[List[float]] = None
    has_target_split: bool = False

    @property
    def divergence_score(self) -> float:
        if not self.has_target_split or self.churned_values is None:
            return 0.0
        import numpy as np
        ret_arr = np.array([v for v in self.retained_values if v is not None and not np.isnan(v)])
        churn_arr = np.array([v for v in self.churned_values if v is not None and not np.isnan(v)])
        if len(ret_arr) == 0 or len(churn_arr) == 0:
            return 0.0
        return abs(ret_arr.mean() - churn_arr.mean()) / max(ret_arr.std(), churn_arr.std(), 0.001)


class SparklineDataBuilder:
    def __init__(self, entity_column: str, time_column: str,
                 target_column: Optional[str] = None, freq: str = "W"):
        self.entity_column = entity_column
        self.time_column = time_column
        self.target_column = target_column
        self.freq = freq

    def build(self, df: DataFrame, columns: List[str]) -> Tuple[List[SparklineData], bool]:
        import pandas as pd
        has_target = self.target_column is not None and self.target_column in df.columns
        if has_target:
            validate_not_event_level(df, self.entity_column, self.target_column)
        df_work = self._prepare_working_df(df, has_target)
        df_work['_period'] = pd.to_datetime(df_work[self.time_column]).dt.to_period(self.freq).dt.start_time
        results = [self._build_sparkline_for_column(df_work, col, has_target)
                   for col in columns if col in df_work.columns]
        return results, has_target

    def _prepare_working_df(self, df: DataFrame, has_target: bool) -> DataFrame:
        if has_target:
            entity_target = df.groupby(self.entity_column)[self.target_column].first()
            return df.merge(
                entity_target.reset_index().rename(columns={self.target_column: '_target'}),
                on=self.entity_column)
        df_work = df.copy()
        df_work['_target'] = 1
        return df_work

    def _build_sparkline_for_column(self, df_work: DataFrame, col: str, has_target: bool) -> SparklineData:
        import numpy as np
        if has_target:
            retained = df_work[df_work['_target'] == 1].groupby('_period')[col].mean()
            churned = df_work[df_work['_target'] == 0].groupby('_period')[col].mean()
            all_periods = sorted(set(retained.index) | set(churned.index))
            retained_vals = [retained.get(p, np.nan) for p in all_periods]
            churned_vals = [churned.get(p, np.nan) for p in all_periods]
        else:
            overall = df_work.groupby('_period')[col].mean()
            all_periods, retained_vals, churned_vals = sorted(overall.index), overall.tolist(), None
        return SparklineData(column=col, weeks=all_periods, retained_values=retained_vals,
                             churned_values=churned_vals, has_target_split=has_target)

    def format_summary(self, sparkline_data: List[SparklineData], has_target: bool) -> str:
        lines = ["=" * 70]
        if has_target:
            lines.append("SPARKLINE COMPARISON: Retained vs Churned Trends\n" + "=" * 70)
            lines.append("\n  Retained (target=1) | Churned (target=0)\n")
        else:
            lines.append("SPARKLINE TRENDS: Overall Patterns\n" + "=" * 70)
        for data in sparkline_data:
            if data.has_target_split:
                lines.append(f"  {data.column}: divergence={data.divergence_score:.2f}")
        return "\n".join(lines)

    def print_summary(self, sparkline_data: List[SparklineData], has_target: bool):
        print(self.format_summary(sparkline_data, has_target))


@dataclass
class FindingsValidationResult:
    valid: bool
    missing_sections: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def format_summary(self) -> str:
        lines = []
        if not self.valid:
            lines.append("MISSING REQUIRED ANALYSIS:")
            for m in self.missing_sections:
                lines.append(f"  - {m}")
        for w in self.warnings:
            lines.append(f"  Warning: {w}")
        return "\n".join(lines)

    def print_summary(self):
        print(self.format_summary())


def validate_temporal_findings(findings: Any) -> FindingsValidationResult:
    missing: List[str] = []
    warnings: List[str] = []

    if findings.time_series_metadata is None:
        missing.append("time_series_metadata (run 01a first)")
    elif not findings.time_series_metadata.suggested_aggregations:
        warnings.append("No aggregation windows defined - defaults will be used")

    pattern_meta = findings.metadata.get("temporal_patterns", {}) if findings.metadata else {}
    if not pattern_meta:
        missing.append("temporal_patterns (run 01c first)")
    else:
        for section in ["trend", "recency", "momentum"]:
            if section not in pattern_meta:
                warnings.append(f"No {section} analysis found in 01c")

    return FindingsValidationResult(
        valid=len(missing) == 0,
        missing_sections=missing,
        warnings=warnings,
    )


@dataclass
class AggregationFeatureConfig:
    trend_features: List[str] = field(default_factory=list)
    seasonality_features: List[str] = field(default_factory=list)
    cohort_features: List[str] = field(default_factory=list)
    recency_features: List[str] = field(default_factory=list)
    categorical_features: List[str] = field(default_factory=list)
    velocity_features: List[str] = field(default_factory=list)
    momentum_features: List[str] = field(default_factory=list)
    lag_features: List[str] = field(default_factory=list)
    sparkline_features: List[str] = field(default_factory=list)
    priority_features: List[str] = field(default_factory=list)
    text_pca_columns: List[str] = field(default_factory=list)
    scaling_recommendations: List[Dict[str, Any]] = field(default_factory=list)
    divergent_columns: List[str] = field(default_factory=list)
    feature_flags: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_findings(cls, findings: Any) -> "AggregationFeatureConfig":
        pattern_meta = findings.metadata.get("temporal_patterns", {}) if findings.metadata else {}

        def extract_features(section: str) -> List[str]:
            features = []
            for rec in pattern_meta.get(section, {}).get("recommendations", []):
                features.extend(rec.get("features", []))
            return features

        def extract_priority_features(section: str, priority_values: tuple = ("high",)) -> List[str]:
            features = []
            for rec in pattern_meta.get(section, {}).get("recommendations", []):
                if rec.get("priority") in priority_values or rec.get("priority") == 1:
                    features.extend(rec.get("features", []))
                    if rec.get("feature"):
                        features.append(rec["feature"])
            return features

        cohort_features = [f for rec in pattern_meta.get("cohort", {}).get("recommendations", []) if rec.get("action") != "skip_cohort_features" for f in rec.get("features", [])]
        sparkline_features = extract_features("sparkline")
        scaling_recs = [rec for rec in pattern_meta.get("sparkline", {}).get("recommendations", []) if rec.get("action") in ("robust_scale", "normalize")]
        priority_set: set = set()
        for section in ["effect_size", "predictive_power", "velocity", "momentum"]:
            priority_set.update(extract_priority_features(section))
        for rec in pattern_meta.get("effect_size", {}).get("recommendations", []):
            if rec.get("action") == "prioritize_feature" and rec.get("feature"):
                priority_set.add(rec["feature"])
        for rec in pattern_meta.get("predictive_power", {}).get("recommendations", []):
            if rec.get("action") == "include_feature" and rec.get("feature"):
                priority_set.add(rec["feature"])

        text_pca_cols = _extract_text_pca_columns(findings)

        return cls(
            trend_features=extract_features("trend"),
            seasonality_features=extract_features("seasonality"),
            cohort_features=cohort_features,
            recency_features=extract_features("recency"),
            categorical_features=extract_features("categorical"),
            velocity_features=extract_features("velocity"),
            momentum_features=extract_features("momentum"),
            lag_features=extract_features("lag"),
            sparkline_features=sparkline_features,
            text_pca_columns=text_pca_cols,
            priority_features=list(priority_set),
            scaling_recommendations=scaling_recs,
            divergent_columns=pattern_meta.get("momentum", {}).get("_divergent_columns", []),
            feature_flags=pattern_meta.get("feature_flags", {}),
        )

    def get_all_features(self) -> List[str]:
        all_feats = (
            self.trend_features + self.seasonality_features + self.cohort_features
            + self.recency_features + self.categorical_features + self.velocity_features
            + self.momentum_features + self.lag_features + self.sparkline_features
            + self.text_pca_columns
        )
        return list(dict.fromkeys(all_feats))

    def get_priority_features(self) -> List[str]:
        return self.priority_features

    def format_summary(self) -> str:
        lines = ["=" * 70, "AGGREGATION FEATURE CONFIG", "=" * 70]
        if self.trend_features:
            lines.append(f"\nTrend features: {self.trend_features}")
        if self.seasonality_features:
            lines.append(f"Seasonality features: {self.seasonality_features}")
        if self.cohort_features:
            lines.append(f"Cohort features: {self.cohort_features}")
        if self.recency_features:
            lines.append(f"Recency features: {self.recency_features}")
        if self.categorical_features:
            lines.append(f"Categorical features: {self.categorical_features}")
        if self.velocity_features:
            lines.append(f"Velocity features: {self.velocity_features}")
        if self.momentum_features:
            lines.append(f"Momentum features: {self.momentum_features}")
        if self.lag_features:
            lines.append(f"Lag features: {self.lag_features}")
        if self.sparkline_features:
            lines.append(f"Sparkline features: {self.sparkline_features}")
        if self.priority_features:
            lines.append(f"\nPriority features (from effect size/IV): {self.priority_features}")
        if self.scaling_recommendations:
            lines.append(f"Scaling recommendations: {len(self.scaling_recommendations)} features")
        if self.divergent_columns:
            lines.append(f"\nDivergent columns: {self.divergent_columns}")
        if self.text_pca_columns:
            lines.append(f"Text PCA columns: {self.text_pca_columns}")
        if self.feature_flags:
            lines.append(f"\nFeature flags: {self.feature_flags}")
        return "\n".join(lines)

    def format_recommendation_summary(self) -> str:
        sections = [
            ("trend", self.trend_features),
            ("seasonality", self.seasonality_features),
            ("recency", self.recency_features),
            ("cohort", self.cohort_features),
            ("velocity", self.velocity_features),
            ("momentum", self.momentum_features),
            ("lag", self.lag_features),
            ("sparkline", self.sparkline_features),
            ("effect_size", self.priority_features),
            ("predictive_power", self.priority_features),
            ("text_pca", self.text_pca_columns),
        ]
        lines = ["RECOMMENDATION APPLICATION SUMMARY", "=" * 50]
        lines.append(f"{'Section':<20} {'Features':>8}")
        lines.append("-" * 30)
        total = 0
        for name, features in sections:
            n = len(features)
            total += n
            lines.append(f"{name:<20} {n:>8}")
        lines.append("-" * 30)
        lines.append(f"{'Total':<20} {total:>8}")
        if self.feature_flags:
            lines.append(f"\nFeature flags: {self.feature_flags}")
        if self.scaling_recommendations:
            lines.append(f"Scaling recs: {len(self.scaling_recommendations)}")
        return "\n".join(lines)

    def print_recommendation_summary(self):
        print(self.format_recommendation_summary())

    def print_summary(self):
        print(self.format_summary())


def _extract_text_pca_columns(findings: Any) -> List[str]:
    text_processing = getattr(findings, "text_processing", None)
    if not text_processing:
        return []
    columns = []
    for meta in text_processing.values():
        cols = getattr(meta, "component_columns", None) or []
        columns.extend(cols)
    return columns


def get_duplicate_event_count(findings: Any) -> int:
    metadata = getattr(findings, "metadata", None) or {}
    issues = (metadata.get("temporal_quality") or {}).get("issues") or {}
    return issues.get("duplicate_events", 0)


def deduplicate_events(df: DataFrame, entity_column: str, time_column: str, duplicate_count: int = 0) -> Tuple[DataFrame, int]:
    if duplicate_count <= 0:
        return df, 0
    before = len(df)
    df = df.drop_duplicates(subset=[entity_column, time_column], keep="first")
    return df, before - len(df)


def create_recency_bucket_feature(df: DataFrame, recency_column: str = "days_since_last_event") -> DataFrame:
    if recency_column not in df.columns:
        return df
    edges = [0, 7, 30, 90, 180, float("inf")]
    labels = ["0-7d", "8-30d", "31-90d", "91-180d", ">180d"]
    df = df.copy()
    df["recency_bucket"] = pd.cut(df[recency_column], bins=edges, labels=labels, include_lowest=True).astype("object")
    df.loc[df[recency_column].isna(), "recency_bucket"] = np.nan
    return df


def create_momentum_ratio_features(df: DataFrame, momentum_recs: List[Dict[str, Any]]) -> DataFrame:
    df = df.copy()
    for rec in momentum_recs:
        params = rec.get("params", {})
        short_w, long_w = params.get("short_window"), params.get("long_window")
        source = rec.get("source_column", "")
        if not (short_w and long_w and source):
            continue
        short_col = f"{source}_mean_{short_w}d"
        long_col = f"{source}_mean_{long_w}d"
        if short_col not in df.columns or long_col not in df.columns:
            continue
        feature_name = f"{source}_momentum_{short_w}_{long_w}"
        df[feature_name] = df[short_col] / df[long_col].replace(0, np.nan)
        df[feature_name] = df[feature_name].fillna(1.0)
    return df
