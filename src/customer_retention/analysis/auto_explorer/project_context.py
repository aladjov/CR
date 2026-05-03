from __future__ import annotations

import datetime as _dt
import uuid
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from customer_retention.core.compat import native_pd, pd
from customer_retention.core.compat.remote_path import (
    RemotePath,
    atomic_write_text,
    is_unity_volume_path,
)
from customer_retention.core.config.column_config import DatasetGranularity


def _is_fuse_volume(path) -> bool:
    return is_unity_volume_path(path)


class PredictionObjective(str, Enum):
    IMMEDIATE_RISK = "immediate_risk"
    RENEWAL_RISK = "renewal_risk"
    DISENGAGEMENT = "disengagement"


class PredictionAnchor(str, Enum):
    NOW = "now"
    CONTRACT = "contract"
    INACTIVITY = "inactivity"


class ObjectivePriority(str, Enum):
    PRIMARY = "primary"
    SECONDARY = "secondary"
    EXPLORATORY = "exploratory"
    DISABLED = "disabled"


class TemporalPosture(str, Enum):
    STABLE = "long_memory"
    REACTIVE = "short_memory"


class CadenceInterval(str, Enum):
    DAILY = "daily"
    WEEKLY = "weekly"
    BIWEEKLY = "biweekly"
    MONTHLY = "monthly"
    BIMONTHLY = "bimonthly"


class SplitStrategy(str, Enum):
    TEMPORAL = "temporal"
    COHORT_BASED = "cohort_based"


class RawTimeColumnRole(str, Enum):
    EVENT_TIME = "event_time"
    ENTITY_UPDATE_TIME = "entity_update_time"
    INTERVAL_START_TIME = "interval_start_time"

    @property
    def should_apply_lookback(self) -> bool:
        return self != RawTimeColumnRole.INTERVAL_START_TIME


class ObjectiveAssessment(BaseModel):
    confidence: int
    suggested_anchor: PredictionAnchor
    rationale: list[str]
    feasibility: Optional[dict] = None


class ObjectiveSpec(BaseModel):
    objective: PredictionObjective
    priority: ObjectivePriority
    anchor: Optional[PredictionAnchor] = None
    parameters: dict = Field(default_factory=dict)
    assessment: Optional[ObjectiveAssessment] = None

    @property
    def effective_anchor(self) -> Optional[PredictionAnchor]:
        if self.anchor is not None:
            return self.anchor
        if self.assessment is not None:
            return self.assessment.suggested_anchor
        return None


class DatasetProvenance(BaseModel):
    landing_table: Optional[dict] = None
    bronze_table: Optional[dict] = None
    silver_table: Optional[dict] = None
    gold_table: Optional[dict] = None
    temporal_profile: Optional[dict] = None
    recommendations: dict = Field(default_factory=dict)


class DatasetValidation(BaseModel):
    temporal_quality: Optional[dict] = None
    leakage_gates: Optional[dict] = None
    join_integrity: Optional[dict] = None
    feature_timestamp_definition: Optional[str] = None
    label_timestamp_definition: Optional[str] = None


class KeyResolutionStep(BaseModel):
    bridge_dataset: str
    source_key: str
    bridge_key: str
    resolve_column: str


class DatasetRegistryEntry(BaseModel):
    name: str
    path: str
    storage_format: str = "csv"
    entity_column: Optional[str] = None
    key_resolution: list[KeyResolutionStep] = Field(default_factory=list)
    time_column: Optional[str] = None
    raw_time_column_role: Optional[RawTimeColumnRole] = None
    granularity: Optional[DatasetGranularity] = None
    row_count: Optional[int] = None
    unique_entities: Optional[int] = None
    avg_rows_per_entity: Optional[float] = None
    target_candidates: list[str] = Field(default_factory=list)
    role: Optional[str] = None
    join_keys: list[str] = Field(default_factory=list)
    join_to: Optional[str] = None
    relationship: Optional[str] = None
    detected_schema: dict[str, str] = Field(default_factory=dict)
    provenance: DatasetProvenance = Field(default_factory=DatasetProvenance)
    validation: DatasetValidation = Field(default_factory=DatasetValidation)

    @model_validator(mode="before")
    @classmethod
    def _migrate_join_key(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        legacy = data.pop("join_key", None)
        if legacy is not None and "join_keys" not in data:
            data["join_keys"] = [legacy] if isinstance(legacy, str) else legacy
        return data

    @property
    def join_key(self) -> Optional[str]:
        return self.join_keys[0] if self.join_keys else None


class MergeScaffoldEntry(BaseModel):
    left_dataset: str
    right_dataset: str
    join_keys: list[str]
    relationship: str

    @model_validator(mode="before")
    @classmethod
    def _migrate_join_key(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        legacy = data.pop("join_key", None)
        if legacy is not None and "join_keys" not in data:
            data["join_keys"] = [legacy] if isinstance(legacy, str) else legacy
        return data

    @property
    def join_key(self) -> Optional[str]:
        return self.join_keys[0] if self.join_keys else None


class ObjectiveSupportEntry(BaseModel):
    signals: dict[str, Any] = Field(default_factory=dict)
    implications: list[str] = Field(default_factory=list)
    strength: str = "weak"


class ObjectiveSupport(BaseModel):
    scope: str = "dataset"
    lifecycle_stage: str = "evidence"
    entries: dict[PredictionObjective, ObjectiveSupportEntry] = Field(default_factory=dict)


class IntentConfig(BaseModel):
    prediction_horizons: list[int] = Field(default_factory=lambda: [30, 60, 90])
    recent_window_days: int = 270
    observation_window_days: int = 270
    purge_gap_days: int = 104
    label_window_days: int = 90
    temporal_split: bool = True
    cadence_interval: CadenceInterval = CadenceInterval.WEEKLY
    split_strategy: SplitStrategy = SplitStrategy.TEMPORAL
    history_upper_limit: Optional[str] = None
    lookback_periods: Optional[int] = None
    production_full_panel_fit: bool = False

    @field_validator("lookback_periods")
    @classmethod
    def _validate_lookback_periods(cls, v):
        if v is not None and v < 1:
            raise ValueError("lookback_periods must be >= 1")
        return v


class ExplorationContract(BaseModel):
    views: list[str] = Field(default_factory=lambda: ["full_history", "recent_behavior"])
    insight_mapping_required: bool = True


class ProjectContext(BaseModel):
    project_name: str = "untitled"
    run_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    timezone: str = "UTC"
    storage_backend: str = "local"
    created_at: str = Field(default_factory=lambda: _dt.datetime.now(_dt.timezone.utc).isoformat())
    datasets: dict[str, DatasetRegistryEntry] = Field(default_factory=dict)
    original_datasets: dict[str, str] = Field(default_factory=dict)
    target_dataset: Optional[str] = None
    target_column: Optional[str] = None
    entity_column: Optional[str] = None
    join_dataset_name: Optional[str] = None
    objectives: list[ObjectiveSpec] = Field(default_factory=list)
    primary_objective: PredictionObjective = PredictionObjective.IMMEDIATE_RISK
    temporal_posture: TemporalPosture = TemporalPosture.STABLE
    merge_scaffold: list[MergeScaffoldEntry] = Field(default_factory=list)
    exploration_contract: ExplorationContract = Field(default_factory=ExplorationContract)
    objective_support: Optional[ObjectiveSupport] = None
    intent: Optional[IntentConfig] = None
    light_run: bool = False
    sample_fraction: Optional[float] = None
    sample_entity_count: Optional[int] = None
    sample_stratify_columns: Optional[list[str]] = None
    sample_filters: Optional[dict[str, str]] = None
    holdout_fraction: Optional[float] = None

    @field_validator("sample_entity_count")
    @classmethod
    def _validate_sample_entity_count(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and v < 1:
            raise ValueError("sample_entity_count must be >= 1")
        return v

    @field_validator("sample_fraction")
    @classmethod
    def _validate_sample_fraction(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and (v <= 0.0 or v > 1.0):
            raise ValueError("sample_fraction must be between 0 (exclusive) and 1 (inclusive)")
        return v

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy_posture(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        legacy_map = {"both": "long_memory", "stable": "long_memory", "reactive": "short_memory"}
        raw = data.pop("training_posture", None)
        if raw is not None and "temporal_posture" not in data:
            data["temporal_posture"] = legacy_map.get(raw, raw)
        if "temporal_posture" in data:
            data["temporal_posture"] = legacy_map.get(data["temporal_posture"], data["temporal_posture"])
        return data

    @model_validator(mode="after")
    def _validate_objectives(self) -> ProjectContext:
        if not self.objectives:
            raise ValueError("objectives must not be empty")
        objective_names = [o.objective for o in self.objectives]
        if len(objective_names) != len(set(objective_names)):
            raise ValueError("duplicate objectives found")
        primaries = [o for o in self.objectives if o.priority == ObjectivePriority.PRIMARY]
        if len(primaries) != 1:
            raise ValueError(f"exactly one PRIMARY objective required, found {len(primaries)}")
        if primaries[0].objective != self.primary_objective:
            raise ValueError("primary_objective must match the PRIMARY-priority objective")
        if self.primary_objective not in objective_names:
            raise ValueError("primary_objective must exist in objectives list")
        return self

    @property
    def primary_spec(self) -> ObjectiveSpec:
        return next(o for o in self.objectives if o.priority == ObjectivePriority.PRIMARY)

    @property
    def active_objectives(self) -> list[ObjectiveSpec]:
        return [o for o in self.objectives if o.priority != ObjectivePriority.DISABLED]

    def resolve_dataset_name(self, data_path: str | Path) -> str | None:
        path_str = str(data_path)
        for name, entry in self.datasets.items():
            if entry.path == path_str:
                return name
        query_stem = Path(path_str).stem
        for name, entry in self.datasets.items():
            if Path(entry.path).stem == query_stem:
                return name
        return None

    def save(self, path) -> None:
        p = path if isinstance(path, (Path, RemotePath)) else Path(str(path))
        p.parent.mkdir(parents=True, exist_ok=True)
        content = yaml.dump(self.model_dump(mode="json"), default_flow_style=False, sort_keys=False)
        atomic_write_text(p, content)

    @classmethod
    def load(cls, path) -> ProjectContext:
        p = path if isinstance(path, (Path, RemotePath)) else Path(str(path))
        if not p.exists():
            raise FileNotFoundError(f"Project context file not found: {p}")
        data = yaml.safe_load(p.read_text())
        return cls.model_validate(data)

    @classmethod
    def for_single_dataset(
        cls,
        path: str | Path,
        dataset_name: str,
        target_column: Optional[str] = None,
        entity_column: Optional[str] = None,
        primary_objective: PredictionObjective = PredictionObjective.IMMEDIATE_RISK,
        primary_anchor: PredictionAnchor = PredictionAnchor.NOW,
        join_dataset_name: Optional[str] = None,
        project_name: str = "untitled",
    ) -> ProjectContext:
        entry = DatasetRegistryEntry(
            name=dataset_name,
            path=str(path),
            storage_format="csv",
            entity_column=entity_column,
            target_candidates=[target_column] if target_column else [],
            role="target" if target_column else None,
        )
        spec = ObjectiveSpec(
            objective=primary_objective,
            priority=ObjectivePriority.PRIMARY,
            anchor=primary_anchor,
        )
        return cls(
            project_name=project_name,
            datasets={dataset_name: entry},
            target_dataset=dataset_name if target_column else None,
            target_column=target_column,
            entity_column=entity_column,
            join_dataset_name=join_dataset_name,
            objectives=[spec],
            primary_objective=primary_objective,
        )


_LABEL_COLS = ("label_timestamp", "label_available_flag")


def mount_target_column(
    df: pd.DataFrame,
    context: ProjectContext,
    dataset_name: str,
    target_label_df: Optional[pd.DataFrame] = None,
) -> tuple[pd.DataFrame, Optional[dict]]:
    if not context.target_dataset or not context.target_column:
        return df, None

    entry = context.datasets.get(dataset_name)
    is_target = getattr(entry, "role", None) == "target" or getattr(entry, "has_target", False)
    if entry is None or is_target or not getattr(entry, "join_key", None):
        return df, None

    target_entry = context.datasets.get(context.target_dataset)
    if target_entry is None:
        return df, None

    left_key = _resolve_left_key(entry.join_key, df)
    if left_key is None:
        return df, None

    mount_df, label_mounted = _build_mount_df(target_label_df, target_entry, context)

    drop_before_merge = [c for c in _LABEL_COLS if label_mounted and c in df.columns]
    merge_input = df.drop(columns=drop_before_merge) if drop_before_merge else df
    result = merge_input.merge(mount_df, left_on=left_key, right_on=context.entity_column, how="left")

    if entry.join_key != context.entity_column:
        extra = f"{context.entity_column}_y"
        if extra in result.columns:
            result = result.drop(columns=[extra])

    info = {
        "target_mounted_from": context.target_dataset,
        "target_mount_key": entry.join_key,
    }
    if label_mounted:
        info["label_timestamp_mounted"] = True
    return result, info


def _resolve_left_key(join_key: str, df: pd.DataFrame) -> Optional[str]:
    if join_key in df.columns:
        return join_key
    if "entity_id" in df.columns:
        return "entity_id"
    return None


def _build_mount_df(
    target_label_df: Optional[pd.DataFrame],
    target_entry: DatasetRegistryEntry,
    context: ProjectContext,
) -> tuple[pd.DataFrame, bool]:
    if target_label_df is not None:
        return _mount_df_from_label_source(target_label_df, context)
    usecols = [context.entity_column, context.target_column]
    resolved_path = str(Path(target_entry.path).resolve())
    target_df = native_pd.read_csv(resolved_path, usecols=usecols)
    return target_df.drop_duplicates(subset=[context.entity_column]), False


def _mount_df_from_label_source(
    target_label_df: pd.DataFrame,
    context: ProjectContext,
) -> tuple[pd.DataFrame, bool]:
    entity_col = "entity_id" if "entity_id" in target_label_df.columns else context.entity_column
    target_col = "target" if "target" in target_label_df.columns else context.target_column
    keep = [entity_col, target_col]
    available_label_cols = [c for c in _LABEL_COLS if c in target_label_df.columns]
    keep.extend(available_label_cols)
    mount_df = target_label_df[keep].drop_duplicates(subset=[entity_col])
    if entity_col != context.entity_column:
        mount_df = mount_df.rename(columns={entity_col: context.entity_column})
    if target_col != context.target_column:
        mount_df = mount_df.rename(columns={target_col: context.target_column})
    return mount_df, len(available_label_cols) > 0
