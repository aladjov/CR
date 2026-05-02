from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional

import yaml

logger = logging.getLogger(__name__)

SPEC_VERSION = 1

VALID_VERDICT_STATUSES: FrozenSet[str] = frozenset({
    "solid", "caution", "unstable", "overfit", "leaky",
})

HARD_BLOCK_VERDICTS: FrozenSet[str] = frozenset({"overfit", "leaky"})


@dataclass
class LeakageExclusion:
    column: str
    code: str = ""
    severity: str = "HIGH"
    rationale: str = ""

    def to_dict(self) -> dict:
        return {"column": self.column, "code": self.code,
                "severity": self.severity, "rationale": self.rationale}

    @classmethod
    def from_dict(cls, data: Any) -> LeakageExclusion:
        if isinstance(data, str):
            raise TypeError(
                "LeakageExclusion.from_dict received a bare string. "
                "Legacy flat-list shape is not supported — regenerate the run."
            )
        return cls(
            column=data["column"], code=data.get("code", ""),
            severity=data.get("severity", "HIGH"),
            rationale=data.get("rationale", ""),
        )


_STATELESS_ACTION_TO_PIPELINE_TYPE: Dict[str, str] = {
    "log_transform": "log_transform",
    "sqrt_transform": "sqrt_transform",
    "yj_transform": "yeo_johnson",
    "yeo_johnson": "yeo_johnson",
    "zero_inflation_handling": "zero_inflation_handling",
    "cap_outlier": "cap_outlier",
    "cap_then_log": "cap_then_log",
    "winsorize": "winsorize",
    "segment_aware_cap": "segment_aware_cap",
}

_FITTED_ACTION_TO_PIPELINE_TYPE: Dict[str, str] = {
    "impute": "impute_null",
    "scale": "scale",
    "encode": "encode",
    "cap_outlier": "cap_outlier",
}


def _build_transformation_step(
    pipeline_type_value: str, column: str, parameters: Dict[str, Any],
    rationale: str, source_notebook: str,
):
    from customer_retention.generators.pipeline_generator.models import (
        PipelineTransformationType,
        TransformationStep,
    )
    return TransformationStep(
        type=PipelineTransformationType(pipeline_type_value),
        column=column,
        parameters=dict(parameters or {}),
        rationale=rationale,
        source_notebook=source_notebook or None,
    )


@dataclass
class TransformStep:
    column: str
    action: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    source_notebook: str = ""

    def to_dict(self) -> dict:
        return {"column": self.column, "action": self.action,
                "parameters": dict(self.parameters),
                "source_notebook": self.source_notebook}

    @classmethod
    def from_dict(cls, data: dict) -> TransformStep:
        return cls(
            column=data["column"], action=data["action"],
            parameters=dict(data.get("parameters") or {}),
            source_notebook=data.get("source_notebook", ""),
        )

    def to_transformation_step(self):
        mapped = _STATELESS_ACTION_TO_PIPELINE_TYPE.get(self.action)
        if mapped is None:
            raise ValueError(
                f"TransformStep: unknown stateless action {self.action!r}. "
                f"Known: {sorted(_STATELESS_ACTION_TO_PIPELINE_TYPE)}"
            )
        return _build_transformation_step(
            mapped, self.column, self.parameters,
            rationale=f"feature_spec.transform_topology:{self.action}",
            source_notebook=self.source_notebook,
        )


@dataclass
class FittedTransform:
    column: str
    action: str
    method: str
    exploration_null_count: int = 0
    strategy_source: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "column": self.column, "action": self.action, "method": self.method,
            "exploration_null_count": self.exploration_null_count,
            "strategy_source": self.strategy_source,
            "parameters": dict(self.parameters),
        }

    @classmethod
    def from_dict(cls, data: dict) -> FittedTransform:
        return cls(
            column=data["column"], action=data["action"], method=data["method"],
            exploration_null_count=int(data.get("exploration_null_count", 0)),
            strategy_source=data.get("strategy_source", ""),
            parameters=dict(data.get("parameters") or {}),
        )

    def to_transformation_step(self):
        mapped = _FITTED_ACTION_TO_PIPELINE_TYPE.get(self.action)
        if mapped is None:
            raise ValueError(
                f"FittedTransform: unknown stateful action {self.action!r}. "
                f"Known: {sorted(_FITTED_ACTION_TO_PIPELINE_TYPE)}"
            )
        params = {"method": self.method, **self.parameters}
        return _build_transformation_step(
            mapped, self.column, params,
            rationale=f"feature_spec.fitted_transforms:{self.action}:{self.method}",
            source_notebook="08_baseline_experiments",
        )


@dataclass
class Verdict:
    status: str
    cv_mean: float = 0.0
    cv_std: float = 0.0
    fold_aucs: List[float] = field(default_factory=list)
    best_model: str = ""
    critical_issues: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.status not in VALID_VERDICT_STATUSES:
            raise ValueError(
                f"Verdict.status must be one of {sorted(VALID_VERDICT_STATUSES)}, got {self.status!r}"
            )

    def is_hard_block(self) -> bool:
        return self.status in HARD_BLOCK_VERDICTS

    def to_dict(self) -> dict:
        return {
            "status": self.status, "cv_mean": float(self.cv_mean),
            "cv_std": float(self.cv_std), "fold_aucs": [float(x) for x in self.fold_aucs],
            "best_model": self.best_model, "critical_issues": list(self.critical_issues),
        }

    @classmethod
    def from_dict(cls, data: dict) -> Verdict:
        return cls(
            status=data["status"],
            cv_mean=float(data.get("cv_mean", 0.0)),
            cv_std=float(data.get("cv_std", 0.0)),
            fold_aucs=[float(x) for x in data.get("fold_aucs") or []],
            best_model=data.get("best_model", ""),
            critical_issues=list(data.get("critical_issues") or []),
        )


@dataclass
class SplitSpec:
    strategy: str = "temporal_entity_aware"
    temporal_column: str = "as_of_date"
    test_size: float = 0.2
    purge_gap_days: int = 0
    random_state: int = 42
    cutoff_date: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy, "temporal_column": self.temporal_column,
            "test_size": float(self.test_size), "purge_gap_days": int(self.purge_gap_days),
            "random_state": int(self.random_state), "cutoff_date": self.cutoff_date,
        }

    @classmethod
    def from_dict(cls, data: dict) -> SplitSpec:
        return cls(
            strategy=data.get("strategy", "temporal_entity_aware"),
            temporal_column=data.get("temporal_column", "as_of_date"),
            test_size=float(data.get("test_size", 0.2)),
            purge_gap_days=int(data.get("purge_gap_days", 0)),
            random_state=int(data.get("random_state", 42)),
            cutoff_date=data.get("cutoff_date"),
        )


@dataclass
class NB05Drops:
    permanent: List[str] = field(default_factory=list)
    advisory: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"permanent": list(self.permanent), "advisory": list(self.advisory)}

    @classmethod
    def from_dict(cls, data: dict) -> NB05Drops:
        return cls(
            permanent=list(data.get("permanent") or []),
            advisory=list(data.get("advisory") or []),
        )


@dataclass
class FeatureSpec:
    spec_version: int = SPEC_VERSION
    exploration_run_id: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    target_column: str = ""
    # The exact column name on the gold/silver schema that the rendered
    # training script should resolve before applying its TARGET literal.
    # When non-empty, NB10's training renderer emits a withColumnRenamed
    # step `original_target_column -> target_column` before any filter
    # against TARGET fires. Closes the patch_target_column_resolver
    # operator-paste path for the common case where exploration applies
    # a rename and codegen needs to replay it.
    original_target_column: str = ""
    entity_column: str = "entity_id"
    timestamp_column: str = "as_of_date"
    horizon_days: int = 0
    selected_features: List[str] = field(default_factory=list)
    transform_topology: List[TransformStep] = field(default_factory=list)
    fitted_transforms: List[FittedTransform] = field(default_factory=list)
    leakage_exclusions: List[LeakageExclusion] = field(default_factory=list)
    nb05_drops: NB05Drops = field(default_factory=NB05Drops)
    split: SplitSpec = field(default_factory=SplitSpec)
    verdict: Verdict = field(default_factory=lambda: Verdict(status="solid"))
    holdout_entity_ids_path: str = ""
    holdout_entity_count: int = 0

    def validate(self) -> None:
        if not self.selected_features:
            raise ValueError("FeatureSpec: selected_features must not be empty")
        if len(self.selected_features) != len(set(self.selected_features)):
            dupes = [c for c in self.selected_features if self.selected_features.count(c) > 1]
            raise ValueError(f"FeatureSpec: selected_features must be unique; duplicates: {sorted(set(dupes))}")

        impute_columns = {ft.column for ft in self.fitted_transforms if ft.action == "impute"}
        missing_impute = [c for c in self.selected_features if c not in impute_columns]
        if missing_impute:
            raise ValueError(
                f"FeatureSpec: every selected_feature must declare an impute method "
                f"(fitted_transforms.action='impute'). Missing: {missing_impute[:10]}"
            )

        excluded_cols = {x.column for x in self.leakage_exclusions}
        bleed = [c for c in self.selected_features if c in excluded_cols]
        if bleed:
            raise ValueError(
                f"FeatureSpec: leakage-excluded columns appear in selected_features: {bleed}"
            )

    def to_dict(self) -> dict:
        return {
            "spec_version": self.spec_version,
            "exploration_run_id": self.exploration_run_id,
            "created_at": self.created_at,
            "target_column": self.target_column,
            "original_target_column": self.original_target_column,
            "entity_column": self.entity_column,
            "timestamp_column": self.timestamp_column,
            "horizon_days": int(self.horizon_days),
            "selected_features": list(self.selected_features),
            "transform_topology": [s.to_dict() for s in self.transform_topology],
            "fitted_transforms": [ft.to_dict() for ft in self.fitted_transforms],
            "leakage_exclusions": [x.to_dict() for x in self.leakage_exclusions],
            "nb05_drops": self.nb05_drops.to_dict(),
            "split": self.split.to_dict(),
            "verdict": self.verdict.to_dict(),
            "holdout_entity_ids_path": self.holdout_entity_ids_path,
            "holdout_entity_count": int(self.holdout_entity_count),
        }

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.to_dict(), default_flow_style=False, sort_keys=False)

    def save(self, path: Path) -> None:
        p = path if isinstance(path, Path) else Path(str(path))
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_yaml())
        logger.info("FeatureSpec saved to %s (%d features)", p, len(self.selected_features))

    @classmethod
    def from_dict(cls, data: dict) -> FeatureSpec:
        spec = cls(
            spec_version=int(data.get("spec_version", SPEC_VERSION)),
            exploration_run_id=data.get("exploration_run_id", ""),
            created_at=data.get("created_at", datetime.now(timezone.utc).isoformat()),
            target_column=data.get("target_column", ""),
            original_target_column=data.get("original_target_column", ""),
            entity_column=data.get("entity_column", "entity_id"),
            timestamp_column=data.get("timestamp_column", "as_of_date"),
            horizon_days=int(data.get("horizon_days", 0)),
            selected_features=list(data.get("selected_features") or []),
            transform_topology=[TransformStep.from_dict(d) for d in data.get("transform_topology") or []],
            fitted_transforms=[FittedTransform.from_dict(d) for d in data.get("fitted_transforms") or []],
            leakage_exclusions=[LeakageExclusion.from_dict(d) for d in data.get("leakage_exclusions") or []],
            nb05_drops=NB05Drops.from_dict(data.get("nb05_drops") or {}),
            split=SplitSpec.from_dict(data.get("split") or {}),
            verdict=Verdict.from_dict(data.get("verdict") or {"status": "solid"}),
            holdout_entity_ids_path=data.get("holdout_entity_ids_path", ""),
            holdout_entity_count=int(data.get("holdout_entity_count", 0)),
        )
        return spec

    @classmethod
    def load(cls, path: Path, validate: bool = True) -> FeatureSpec:
        p = path if isinstance(path, Path) else Path(str(path))
        data = yaml.safe_load(p.read_text()) or {}
        spec = cls.from_dict(data)
        if validate:
            spec.validate()
        return spec


def build_fitted_impute_transforms(
    selected_features: List[str],
    feature_kinds: Dict[str, str],
    zero_inflation_bases: List[str],
    null_counts: Optional[Dict[str, int]] = None,
) -> List[FittedTransform]:
    null_counts = null_counts or {}
    bases_set = set(zero_inflation_bases)
    transforms: List[FittedTransform] = []
    for feature in selected_features:
        kind = feature_kinds.get(feature, "numeric").lower()
        base = _strip_aggregation_suffix(feature)
        if base in bases_set and kind == "numeric":
            method, source = "zero_with_indicator", "zero_inflation_opt_in"
        elif kind == "categorical":
            method, source = "mode", "default_categorical"
        else:
            method, source = "median", "default_numeric"
        transforms.append(FittedTransform(
            column=feature, action="impute", method=method,
            exploration_null_count=int(null_counts.get(feature, 0)),
            strategy_source=source,
        ))
    return transforms


_AGG_SUFFIX_PATTERN = (
    "_count_", "_sum_", "_mean_", "_max_", "_min_", "_std_",
    "_avg_", "_nunique_", "_var_",
)


def _strip_aggregation_suffix(column: str) -> str:
    for suffix in _AGG_SUFFIX_PATTERN:
        idx = column.find(suffix)
        if idx > 0:
            return column[:idx]
    if column.endswith("_is_zero"):
        return _strip_aggregation_suffix(column[:-len("_is_zero")])
    if column.endswith("_log"):
        return _strip_aggregation_suffix(column[:-len("_log")])
    return column
