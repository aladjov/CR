import decimal
import json
import logging
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from customer_retention.core.config.column_config import ColumnConfig, ColumnType, DatasetGranularity
from customer_retention.stages.modeling.feature_spec import LeakageExclusion

logger = logging.getLogger(__name__)

_NUMERIC_TYPES = frozenset({ColumnType.NUMERIC_CONTINUOUS, ColumnType.NUMERIC_DISCRETE})
_CATEGORICAL_TYPES = frozenset({
    ColumnType.CATEGORICAL_NOMINAL, ColumnType.CATEGORICAL_ORDINAL, ColumnType.CATEGORICAL_CYCLICAL,
})
_DATETIME_TYPES = frozenset({ColumnType.DATETIME, ColumnType.FEATURE_TIMESTAMP, ColumnType.LABEL_TIMESTAMP})


@dataclass
class ColumnClassification:
    numeric: List[str] = field(default_factory=list)
    categorical: List[str] = field(default_factory=list)
    datetime: List[str] = field(default_factory=list)
    binary: List[str] = field(default_factory=list)
    text: List[str] = field(default_factory=list)
    identifier: List[str] = field(default_factory=list)
    target: Optional[str] = None


def apply_zero_inflation_opt_in(
    findings: Dict[str, "ExplorationFindings"],
    opt_in: Dict[str, List[str]],
) -> None:
    if opt_in is None:
        raise TypeError("apply_zero_inflation_opt_in: opt_in config cannot be None")
    unknown = sorted(name for name in opt_in if name not in findings)
    if unknown:
        raise KeyError(
            f"apply_zero_inflation_opt_in: unknown dataset(s) in opt-in config: {unknown}"
        )
    for name, cols in opt_in.items():
        findings[name].zero_inflation_opt_in = list(cols)


def classify_columns(
    findings: "ExplorationFindings",
    exclude: Optional[set[str]] = None,
) -> ColumnClassification:
    cc = ColumnClassification()
    skip = exclude or set()
    for name, col in findings.columns.items():
        if name in skip:
            continue
        t = col.inferred_type
        if t in _NUMERIC_TYPES:
            cc.numeric.append(name)
        elif t in _CATEGORICAL_TYPES:
            cc.categorical.append(name)
        elif t in _DATETIME_TYPES:
            cc.datetime.append(name)
        elif t == ColumnType.BINARY:
            cc.binary.append(name)
        elif t == ColumnType.TEXT:
            cc.text.append(name)
        elif t == ColumnType.IDENTIFIER:
            cc.identifier.append(name)
        elif t == ColumnType.TARGET:
            cc.target = name
    return cc


def _parse_iso(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value))
    except (ValueError, TypeError):
        return None


def _convert_to_native(obj: Any) -> Any:
    if obj is None:
        return None
    if isinstance(obj, decimal.Decimal):
        return float(obj)
    if isinstance(obj, dict):
        return {k: _convert_to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_convert_to_native(v) for v in obj]
    if isinstance(obj, datetime):
        return obj.isoformat()
    if hasattr(obj, 'item'):
        return obj.item()
    if hasattr(obj, 'tolist'):
        return obj.tolist()
    mod = type(obj).__module__
    if mod == 'numpy':
        return obj.item() if hasattr(obj, 'item') else float(obj)
    if mod.startswith('pandas'):
        return str(obj)
    return obj


@dataclass
class TimeSeriesMetadata:
    granularity: DatasetGranularity = DatasetGranularity.UNKNOWN
    temporal_pattern: Optional[str] = None  # TIME_SERIES, EVENT_LOG, SNAPSHOT
    entity_column: Optional[str] = None
    time_column: Optional[str] = None
    avg_events_per_entity: Optional[float] = None
    time_span_days: Optional[int] = None
    unique_entities: Optional[int] = None
    suggested_aggregations: List[str] = field(default_factory=list)
    window_coverage_threshold: Optional[float] = None
    heterogeneity_level: Optional[str] = None
    eta_squared_intensity: Optional[float] = None
    eta_squared_event_count: Optional[float] = None
    temporal_segmentation_advisory: Optional[str] = None
    temporal_segmentation_recommendation: Optional[str] = None
    drift_risk_level: Optional[str] = None
    volume_drift_risk: Optional[str] = None
    population_stability: Optional[float] = None
    regime_count: Optional[int] = None
    recommended_training_start: Optional[str] = None
    def populate_from_coverage(self, windows: list, coverage_threshold: float) -> None:
        self.suggested_aggregations = windows
        self.window_coverage_threshold = coverage_threshold

    def populate_from_heterogeneity(
        self, heterogeneity_level: str, eta_squared_intensity: float,
        eta_squared_event_count: float, segmentation_advisory: str,
    ) -> None:
        self.heterogeneity_level = heterogeneity_level
        self.eta_squared_intensity = eta_squared_intensity
        self.eta_squared_event_count = eta_squared_event_count
        self.temporal_segmentation_advisory = segmentation_advisory
        self.temporal_segmentation_recommendation = (
            "include_lifecycle_quadrant" if segmentation_advisory != "single_model" else None
        )

    def populate_from_drift(
        self, risk_level: str, volume_drift_risk: str,
        population_stability: float, regime_count: int,
        recommended_training_start: Optional[str],
    ) -> None:
        self.drift_risk_level = risk_level
        self.volume_drift_risk = volume_drift_risk
        self.population_stability = population_stability
        self.regime_count = regime_count
        self.recommended_training_start = recommended_training_start

    aggregation_executed: bool = False
    aggregated_data_path: Optional[str] = None
    aggregated_findings_path: Optional[str] = None
    aggregation_windows_used: List[str] = field(default_factory=list)
    aggregation_timestamp: Optional[str] = None


@dataclass
class TextProcessingMetadata:
    column_name: str
    embedding_model: str
    embedding_dim: int
    n_components: int
    explained_variance: float
    component_columns: List[str]
    variance_threshold_used: float
    processing_approach: str = "pca"


@dataclass
class FeatureAvailabilityInfo:
    first_valid_date: Optional[str]
    last_valid_date: Optional[str]
    coverage_pct: float
    availability_type: str
    days_from_start: Optional[int]
    days_before_end: Optional[int]


@dataclass
class FeatureAvailabilityMetadata:
    data_start: str
    data_end: str
    time_span_days: int
    new_tracking: List[str]
    retired_tracking: List[str]
    partial_window: List[str]
    features: Dict[str, FeatureAvailabilityInfo] = field(default_factory=dict)


@dataclass
class ColumnFinding:
    name: str
    inferred_type: ColumnType
    confidence: float
    evidence: List[str]
    alternatives: List[ColumnType] = field(default_factory=list)
    universal_metrics: Dict[str, Any] = field(default_factory=dict)
    type_metrics: Dict[str, Any] = field(default_factory=dict)
    quality_issues: List[str] = field(default_factory=list)
    quality_score: float = 100.0
    cleaning_needed: bool = False
    cleaning_recommendations: List[str] = field(default_factory=list)
    transformation_recommendations: List[str] = field(default_factory=list)

    def to_column_config(self) -> ColumnConfig:
        return ColumnConfig(
            name=self.name,
            column_type=self.inferred_type,
            nullable=self.universal_metrics.get("null_count", 0) > 0
        )


@dataclass
class ExplorationFindings:
    source_path: str
    source_format: str
    exploration_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    row_count: int = 0
    column_count: int = 0
    memory_usage_mb: float = 0.0
    columns: Dict[str, ColumnFinding] = field(default_factory=dict)
    target_column: Optional[str] = None
    target_type: Optional[str] = None
    identifier_columns: List[str] = field(default_factory=list)
    datetime_columns: List[str] = field(default_factory=list)
    overall_quality_score: float = 100.0
    critical_issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    modeling_ready: bool = False
    blocking_issues: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    time_series_metadata: Optional[TimeSeriesMetadata] = None
    text_processing: Dict[str, TextProcessingMetadata] = field(default_factory=dict)
    feature_availability: Optional[FeatureAvailabilityMetadata] = None
    iteration_id: Optional[str] = None
    parent_iteration_id: Optional[str] = None
    # Snapshot-related fields (from temporal framework)
    snapshot_id: Optional[str] = None
    snapshot_path: Optional[str] = None
    timestamp_scenario: Optional[str] = None
    timestamp_strategy: Optional[str] = None
    datetime_ordering: List[str] = field(default_factory=list)
    datetime_derivation_sources: List[str] = field(default_factory=list)
    datetime_allow_future_columns: List[str] = field(default_factory=list)
    excluded_leaking_features: List[LeakageExclusion] = field(default_factory=list)
    zero_inflation_opt_in: List[str] = field(default_factory=list)
    field_availability_audit: Optional[Dict[str, Any]] = None
    label_timestamp_column: Optional[str] = None
    observation_window_days: int = 180

    @property
    def time_column(self) -> Optional[str]:
        if self.time_series_metadata is None:
            return None
        return self.time_series_metadata.time_column

    @property
    def is_time_series(self) -> bool:
        if self.time_series_metadata is None:
            return False
        return self.time_series_metadata.granularity == DatasetGranularity.EVENT_LEVEL

    @property
    def has_aggregated_output(self) -> bool:
        return (self.time_series_metadata is not None and
                self.time_series_metadata.aggregation_executed)

    @property
    def column_types(self) -> Dict[str, ColumnType]:
        return {name: col.inferred_type for name, col in self.columns.items()}

    @property
    def column_configs(self) -> Dict[str, ColumnConfig]:
        return {name: col.to_column_config() for name, col in self.columns.items()}

    @property
    def has_availability_issues(self) -> bool:
        if self.feature_availability is None:
            return False
        return bool(
            self.feature_availability.new_tracking
            or self.feature_availability.retired_tracking
            or self.feature_availability.partial_window
        )

    @property
    def problematic_availability_columns(self) -> List[str]:
        if self.feature_availability is None:
            return []
        return (
            self.feature_availability.new_tracking
            + self.feature_availability.retired_tracking
            + self.feature_availability.partial_window
        )

    def get_feature_availability(self, column: str) -> Optional[FeatureAvailabilityInfo]:
        if self.feature_availability is None:
            return None
        return self.feature_availability.features.get(column)

    @staticmethod
    def _normalize_enum_value(obj: Any) -> Any:
        return obj.value if hasattr(obj, 'value') else obj

    def to_dict(self) -> dict:
        result = _convert_to_native(asdict(self))
        for col_data in result.get("columns", {}).values():
            if "inferred_type" in col_data:
                col_data["inferred_type"] = self._normalize_enum_value(col_data["inferred_type"])
            if "alternatives" in col_data:
                col_data["alternatives"] = [self._normalize_enum_value(t) for t in col_data["alternatives"]]
        ts_meta = result.get("time_series_metadata")
        if ts_meta is not None and "granularity" in ts_meta:
            ts_meta["granularity"] = self._normalize_enum_value(ts_meta["granularity"])
        return result

    def to_yaml(self) -> str:
        # `safe_dump` emits tuples as lists; bare `yaml.dump` emitted
        # `!!python/tuple` tags that bare `safe_load` cannot parse. The
        # backward-compat reader (`from_yaml`) accepts both forms.
        return yaml.safe_dump(self.to_dict(), default_flow_style=False, sort_keys=False)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    def save(self, path):
        path_str = str(path)
        content = self.to_yaml() if path_str.endswith((".yaml", ".yml")) else self.to_json()
        p = path if isinstance(path, Path) else Path(path_str)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w") as f:
            f.write(content)

    @classmethod
    def _deserialize_columns(cls, data: dict) -> Dict[str, "ColumnFinding"]:
        columns = {}
        for col_name, col_data in data.get("columns", {}).items():
            if "inferred_type" in col_data:
                col_data["inferred_type"] = ColumnType(col_data["inferred_type"])
            if "alternatives" in col_data:
                col_data["alternatives"] = [ColumnType(t) for t in col_data["alternatives"]]
            columns[col_name] = ColumnFinding(**col_data)
        return columns

    @classmethod
    def _deserialize_time_series_metadata(cls, ts_meta: Optional[dict]) -> Optional["TimeSeriesMetadata"]:
        if ts_meta is None:
            return None
        if "granularity" in ts_meta:
            ts_meta["granularity"] = DatasetGranularity(ts_meta["granularity"])
        return TimeSeriesMetadata(**ts_meta)

    @classmethod
    def _deserialize_feature_availability(cls, fa_data: Optional[dict]) -> Optional["FeatureAvailabilityMetadata"]:
        if fa_data is None:
            return None
        fa_data["features"] = {
            k: FeatureAvailabilityInfo(**v)
            for k, v in fa_data.get("features", {}).items()
        }
        return FeatureAvailabilityMetadata(**fa_data)

    @classmethod
    def from_dict(cls, data: dict) -> "ExplorationFindings":
        data["columns"] = cls._deserialize_columns(data)
        data["time_series_metadata"] = cls._deserialize_time_series_metadata(data.get("time_series_metadata"))
        data["text_processing"] = {k: TextProcessingMetadata(**v) for k, v in data.get("text_processing", {}).items()}
        data["feature_availability"] = cls._deserialize_feature_availability(data.get("feature_availability"))
        data["excluded_leaking_features"] = [
            LeakageExclusion.from_dict(e) for e in data.get("excluded_leaking_features") or []
        ]
        # Forward-compat: drop fields the deployed dataclass does not declare
        # so YAMLs written by a newer framework load on an older cluster build.
        # Local-write / cluster-load skew is structural in this project (NB01..NB09
        # often run locally; NB10..NB11 on the cluster), and the engagement layer
        # used to carry `sps_compat_*` shims to handle exactly this. Log dropped
        # keys so operators can audit version drift.
        known = {f.name for f in fields(cls)}
        unknown = sorted(k for k in data.keys() if k not in known)
        if unknown:
            logger.warning(
                "ExplorationFindings.from_dict: dropping %d unknown field(s) "
                "%s — likely written by a newer framework version. Update the "
                "deployed customer_retention package to consume them.",
                len(unknown), unknown,
            )
            data = {k: v for k, v in data.items() if k in known}
        return cls(**data)

    @classmethod
    def from_yaml(cls, yaml_str: str) -> "ExplorationFindings":
        # Tolerant of `!!python/tuple` tags written by pre-fix engagements
        # (e.g. spschurn-92a9a005). The shared loader maps the tag to a
        # plain list — preserves SafeLoader's no-code-execution guarantee
        # while keeping legacy artifacts readable.
        from customer_retention.analysis.auto_explorer.exploration_manager import (
            _safe_load_tuples_ok,
        )
        return cls.from_dict(_safe_load_tuples_ok(yaml_str))

    @classmethod
    def from_json(cls, json_str: str) -> "ExplorationFindings":
        return cls.from_dict(json.loads(json_str))

    @classmethod
    def load(cls, path) -> "ExplorationFindings":
        p = path if isinstance(path, Path) else Path(str(path))
        content = p.read_text()
        path_str = str(path)
        return cls.from_yaml(content) if path_str.endswith((".yaml", ".yml")) else cls.from_json(content)

    def build_datetime_discovery_stats(self) -> dict:
        from customer_retention.core.compat.bulk_profiling import DatetimeDiscoveryCandidateStats

        stats: dict = {}
        for col_name in self.datetime_columns:
            col = self.columns.get(col_name)
            if col is None:
                continue
            tm = col.type_metrics
            um = col.universal_metrics
            null_count = um.get("null_count", 0)
            non_null = self.row_count - null_count
            coverage = non_null / self.row_count if self.row_count > 0 else 0.0
            future_count = tm.get("future_date_count", 0)
            future_fraction = future_count / non_null if non_null > 0 else 0.0
            stats[col_name] = DatetimeDiscoveryCandidateStats(
                min_date=_parse_iso(tm.get("min_date")),
                max_date=_parse_iso(tm.get("max_date")),
                coverage=coverage,
                future_fraction=future_fraction,
            )
        return stats

    _NUMERIC_INFERRED = frozenset({ColumnType.NUMERIC_CONTINUOUS, ColumnType.NUMERIC_DISCRETE})

    def reconcile_column_types(self, df) -> list[str]:
        from pandas.api.types import is_numeric_dtype
        corrected: list[str] = []
        for name, col_finding in self.columns.items():
            if name not in df.columns:
                continue
            if col_finding.inferred_type not in self._NUMERIC_INFERRED:
                continue
            if is_numeric_dtype(df[name]):
                continue
            col_finding.inferred_type = ColumnType.CATEGORICAL_NOMINAL
            col_finding.confidence = 0.7
            col_finding.evidence.append(f"Reconciled: dtype mismatch (actual={df[name].dtype})")
            corrected.append(name)
        return corrected

    TEMPORAL_METADATA_SKIP = frozenset({
        "feature_timestamp", "label_timestamp", "label_available_flag",
    })

    @classmethod
    def merge_from_datasets(
        cls,
        dataset_findings: list["ExplorationFindings"],
        row_count: int,
        column_count: int,
        source_path: str,
        renamed_columns: dict[str, str] | None = None,
        entity_key: str = "entity_id",
        as_of_column: str = "as_of_date",
    ) -> "ExplorationFindings":
        renamed_columns = renamed_columns or {}

        merged_cols: Dict[str, ColumnFinding] = {}
        target_column: Optional[str] = None
        target_type: Optional[str] = None
        all_identifier_cols: List[str] = []
        all_datetime_cols: List[str] = []

        for findings in dataset_findings:
            if target_column is None and findings.target_column:
                target_column = renamed_columns.get(findings.target_column, findings.target_column)
                target_type = findings.target_type
            all_identifier_cols.extend(findings.identifier_columns)
            all_datetime_cols.extend(findings.datetime_columns)

            for col_name, col_finding in findings.columns.items():
                if col_name in cls.TEMPORAL_METADATA_SKIP:
                    continue
                dest_name = renamed_columns.get(col_name, col_name)
                merged_cols[dest_name] = ColumnFinding(
                    name=dest_name,
                    inferred_type=col_finding.inferred_type,
                    confidence=col_finding.confidence,
                    evidence=list(col_finding.evidence),
                    alternatives=list(col_finding.alternatives),
                    universal_metrics=dict(col_finding.universal_metrics),
                    type_metrics=dict(col_finding.type_metrics),
                    quality_issues=list(col_finding.quality_issues),
                    quality_score=col_finding.quality_score,
                    cleaning_needed=col_finding.cleaning_needed,
                    cleaning_recommendations=list(col_finding.cleaning_recommendations),
                    transformation_recommendations=list(col_finding.transformation_recommendations),
                )

        merged_cols[entity_key] = ColumnFinding(
            name=entity_key,
            inferred_type=ColumnType.IDENTIFIER,
            confidence=1.0,
            evidence=["Spine column"],
        )
        merged_cols[as_of_column] = ColumnFinding(
            name=as_of_column,
            inferred_type=ColumnType.DATETIME,
            confidence=1.0,
            evidence=["Spine column"],
        )

        all_identifier_cols = [renamed_columns.get(c, c) for c in all_identifier_cols]
        all_datetime_cols = [renamed_columns.get(c, c) for c in all_datetime_cols]
        if entity_key not in all_identifier_cols:
            all_identifier_cols.append(entity_key)
        if as_of_column not in all_datetime_cols:
            all_datetime_cols.append(as_of_column)

        scores = [c.quality_score for c in merged_cols.values()]
        overall_quality = sum(scores) / len(scores) if scores else 100.0

        return cls(
            source_path=source_path,
            source_format="delta",
            row_count=row_count,
            column_count=column_count,
            columns=merged_cols,
            target_column=target_column,
            target_type=target_type,
            identifier_columns=sorted(set(all_identifier_cols)),
            datetime_columns=sorted(set(all_datetime_cols)),
            overall_quality_score=overall_quality,
            modeling_ready=True,
            blocking_issues=[],
        )
