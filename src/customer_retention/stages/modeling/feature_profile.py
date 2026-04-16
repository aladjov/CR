from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

logger = logging.getLogger(__name__)

CURRENT_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class StageDecision:
    stage: str
    score: float
    score_name: str
    threshold: Optional[float]
    decision: str
    reason: Optional[str]
    rank: Optional[int]
    stage_input_count: int
    stage_output_count: int
    companion_feature: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "stage": self.stage, "score": self.score, "score_name": self.score_name,
            "threshold": self.threshold, "decision": self.decision, "reason": self.reason,
            "rank": self.rank, "stage_input_count": self.stage_input_count,
            "stage_output_count": self.stage_output_count,
            "companion_feature": self.companion_feature,
        }

    @classmethod
    def from_dict(cls, data: dict) -> StageDecision:
        return cls(
            stage=data["stage"], score=float(data["score"]),
            score_name=data["score_name"], threshold=data.get("threshold"),
            decision=data["decision"], reason=data.get("reason"),
            rank=data.get("rank"),
            stage_input_count=int(data["stage_input_count"]),
            stage_output_count=int(data["stage_output_count"]),
            companion_feature=data.get("companion_feature"),
        )


@dataclass(frozen=True)
class FeatureOrigin:
    source: Optional[str]
    base_column: Optional[str]
    family: Optional[str]
    lag_prefix: Optional[str] = None
    derivation: Optional[str] = None
    parents: Tuple[str, ...] = ()

    def to_dict(self) -> dict:
        out: Dict[str, Any] = {
            "source": self.source, "base_column": self.base_column, "family": self.family,
        }
        if self.lag_prefix is not None:
            out["lag_prefix"] = self.lag_prefix
        if self.derivation is not None:
            out["derivation"] = self.derivation
        if self.parents:
            out["parents"] = list(self.parents)
        return out

    @classmethod
    def from_dict(cls, data: dict) -> FeatureOrigin:
        return cls(
            source=data.get("source"), base_column=data.get("base_column"),
            family=data.get("family"), lag_prefix=data.get("lag_prefix"),
            derivation=data.get("derivation"),
            parents=tuple(data.get("parents", [])),
        )


@dataclass
class ColumnProfile:
    dtype: str
    non_null_count: int
    null_count: int
    origin: Optional[FeatureOrigin] = None
    selection_trace: List[StageDecision] = field(default_factory=list)
    final_score: Optional[float] = None

    def to_dict(self) -> dict:
        out: Dict[str, Any] = {
            "dtype": self.dtype, "non_null": self.non_null_count, "null_count": self.null_count,
        }
        if self.origin is not None:
            out["origin"] = self.origin.to_dict()
        if self.selection_trace:
            out["selection_trace"] = [d.to_dict() for d in self.selection_trace]
        if self.final_score is not None:
            out["final_score"] = self.final_score
        return out

    @classmethod
    def from_dict(cls, data: dict) -> ColumnProfile:
        origin = FeatureOrigin.from_dict(data["origin"]) if data.get("origin") else None
        trace = [StageDecision.from_dict(d) for d in data.get("selection_trace", [])]
        return cls(
            dtype=data["dtype"], non_null_count=data["non_null"], null_count=data["null_count"],
            origin=origin, selection_trace=trace, final_score=data.get("final_score"),
        )


@dataclass
class FeatureProfile:
    stage: str
    created_at: str
    row_count: int
    target_column: str
    features: Dict[str, ColumnProfile]
    excluded: Dict[str, str] = field(default_factory=dict)
    excluded_profiles: Dict[str, ColumnProfile] = field(default_factory=dict)
    schema_version: int = CURRENT_SCHEMA_VERSION

    @property
    def feature_count(self) -> int:
        return len(self.features)

    def to_dict(self) -> dict:
        out: Dict[str, Any] = {
            "schema_version": self.schema_version,
            "stage": self.stage, "created_at": self.created_at,
            "row_count": self.row_count, "feature_count": self.feature_count,
            "target_column": self.target_column,
            "features": {name: col.to_dict() for name, col in self.features.items()},
            "excluded": dict(self.excluded),
        }
        if self.excluded_profiles:
            out["excluded_profiles"] = {
                name: col.to_dict() for name, col in self.excluded_profiles.items()
            }
        return out

    @classmethod
    def from_dict(cls, data: dict) -> FeatureProfile:
        features = {name: ColumnProfile.from_dict(v) for name, v in data.get("features", {}).items()}
        excluded_profiles = {
            name: ColumnProfile.from_dict(v)
            for name, v in (data.get("excluded_profiles") or {}).items()
        }
        schema_version = int(data.get("schema_version", 1))
        return cls(
            stage=data["stage"], created_at=data["created_at"],
            row_count=data["row_count"], target_column=data["target_column"],
            features=features, excluded=data.get("excluded", {}) or {},
            excluded_profiles=excluded_profiles, schema_version=schema_version,
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)
        logger.info("Feature profile saved to %s (%d features)", path, self.feature_count)

    @classmethod
    def load(cls, path: Path) -> Optional[FeatureProfile]:
        if not path.exists():
            return None
        with open(path) as f:
            return cls.from_dict(yaml.safe_load(f))


class SelectionTraceRecorder:
    """In-memory collector for per-stage selection decisions.

    Selection stages emit a score dict + decision dict; the recorder
    consolidates them into :class:`StageDecision` entries keyed by feature
    name and later applies the trace to an existing :class:`FeatureProfile`
    via :meth:`apply_to_profile`.
    """

    def __init__(self) -> None:
        self._traces: Dict[str, List[StageDecision]] = {}
        self._stages: List[Dict[str, Any]] = []

    def record_stage(
        self, *, stage: str, score_name: str, scores: Dict[str, float],
        threshold: Optional[float], decisions: Dict[str, str],
        reasons: Dict[str, Optional[str]],
        stage_input_count: int, stage_output_count: int,
        companion_map: Optional[Dict[str, str]] = None,
    ) -> None:
        if not scores:
            self._stages.append({
                "stage": stage, "input": stage_input_count, "output": stage_output_count,
                "dropped": max(stage_input_count - stage_output_count, 0),
            })
            return

        ranked = self._rank_scores(scores)
        companion = companion_map or {}
        for feature, score in scores.items():
            trace = self._traces.setdefault(feature, [])
            if any(e.stage == stage for e in trace):
                raise ValueError(f"stage {stage!r} already recorded for feature {feature!r}")
            decision = decisions.get(feature, "not_evaluated")
            reason = reasons.get(feature)
            rank = ranked.get(feature)
            trace.append(StageDecision(
                stage=stage, score=float(score), score_name=score_name,
                threshold=threshold, decision=decision, reason=reason, rank=rank,
                stage_input_count=stage_input_count, stage_output_count=stage_output_count,
                companion_feature=companion.get(feature),
            ))
        dropped_count = sum(1 for d in decisions.values() if d == "dropped")
        self._stages.append({
            "stage": stage, "input": stage_input_count, "output": stage_output_count,
            "dropped": dropped_count,
        })

    def record_single(self, feature: str, decision: StageDecision) -> None:
        trace = self._traces.setdefault(feature, [])
        if any(e.stage == decision.stage for e in trace):
            raise ValueError(f"stage {decision.stage!r} already recorded for feature {feature!r}")
        trace.append(decision)

    def record_nb05_drops(
        self, drops: Dict[str, str], *,
        total_pre_nb05_features: int, total_post_nb05_features: int,
    ) -> None:
        """Record pre-selection NB05 drops as stage='nb05' entries.

        NB05 recommendations (drop_weak, drop_multicollinear, etc.) filter the
        feature pool before any of the selection-pipeline stages run. Recording
        them here gives the trace a complete end-to-end funnel — from the full
        silver feature set through NB05 → variance → correlation → L1 → rescue.
        """
        stage_name = "nb05"
        for feature, reason in drops.items():
            trace = self._traces.setdefault(feature, [])
            if any(e.stage == stage_name for e in trace):
                raise ValueError(f"stage {stage_name!r} already recorded for feature {feature!r}")
            trace.append(StageDecision(
                stage=stage_name,
                score=float("nan"),
                score_name="nb05_recommendation",
                threshold=None,
                decision="dropped",
                reason=str(reason),
                rank=None,
                stage_input_count=total_pre_nb05_features,
                stage_output_count=total_post_nb05_features,
            ))
        self._stages.append({
            "stage": stage_name,
            "input": total_pre_nb05_features,
            "output": total_post_nb05_features,
            "dropped": len(drops),
        })

    def trace_for(self, feature: str) -> List[StageDecision]:
        return list(self._traces.get(feature, []))

    def all_features(self) -> set:
        return set(self._traces.keys())

    def stage_summary(self) -> List[Dict[str, Any]]:
        return list(self._stages)

    def apply_to_profile(self, profile: FeatureProfile) -> None:
        for feature, trace in self._traces.items():
            if feature in profile.features:
                profile.features[feature].selection_trace.extend(trace)
            elif feature in profile.excluded:
                existing = profile.excluded_profiles.get(feature)
                if existing is None:
                    existing = ColumnProfile(dtype="", non_null_count=0, null_count=0)
                    profile.excluded_profiles[feature] = existing
                existing.selection_trace.extend(trace)

    @staticmethod
    def _rank_scores(scores: Dict[str, float]) -> Dict[str, Optional[int]]:
        import math as _math
        valid = [(f, s) for f, s in scores.items() if not _math.isnan(s)]
        valid.sort(key=lambda item: item[1], reverse=True)
        ranked: Dict[str, Optional[int]] = {f: None for f in scores}
        for position, (f, _) in enumerate(valid, start=1):
            ranked[f] = position
        return ranked


def build_feature_profile(
    stage: str, target_column: str, row_count: int,
    feature_stats: Dict[str, ColumnProfile],
    excluded: Optional[Dict[str, str]] = None,
    *,
    excluded_profiles: Optional[Dict[str, ColumnProfile]] = None,
) -> FeatureProfile:
    return FeatureProfile(
        stage=stage, created_at=datetime.now(timezone.utc).isoformat(),
        row_count=row_count, target_column=target_column,
        features=feature_stats, excluded=excluded or {},
        excluded_profiles=excluded_profiles or {},
    )


_DTYPE_CANONICAL = {
    "float64": "double", "float32": "float", "int32": "integer",
    "int64": "long", "int16": "short", "int8": "byte", "bool": "boolean",
}


def _canonical_dtype(dtype: str) -> str:
    return _DTYPE_CANONICAL.get(dtype, dtype)


def compare_feature_profiles(exploration: FeatureProfile, production: FeatureProfile) -> List[str]:
    discrepancies: List[str] = []
    exp_names = set(exploration.features.keys())
    prod_names = set(production.features.keys())
    exp_excluded = exploration.excluded or {}
    prod_excluded = production.excluded or {}

    missing = sorted(exp_names - prod_names)
    extra = sorted(prod_names - exp_names)
    genuinely_missing = [m for m in missing if m not in prod_excluded]
    genuinely_extra = [e for e in extra if e not in exp_excluded]
    for m in genuinely_missing:
        discrepancies.append(f"MISSING in production: {m}")
    for e in genuinely_extra:
        discrepancies.append(f"EXTRA in production: {e}")

    for name in missing:
        if name in prod_excluded:
            discrepancies.append(f"SELECTION DRIFT {name}: in exploration features but excluded in production ({prod_excluded[name]})")
    for name in extra:
        if name in exp_excluded:
            discrepancies.append(f"SELECTION DRIFT {name}: excluded in exploration ({exp_excluded[name]}) but present in production features")

    all_excluded = set(exp_excluded) | set(prod_excluded)
    for name in sorted(all_excluded):
        if name in exp_excluded and name in prod_excluded and exp_excluded[name] != prod_excluded[name]:
            discrepancies.append(f"EXCLUSION REASON {name}: exploration={exp_excluded[name]}, production={prod_excluded[name]}")

    for name in sorted(exp_names & prod_names):
        exp_col, prod_col = exploration.features[name], production.features[name]
        if _canonical_dtype(exp_col.dtype) != _canonical_dtype(prod_col.dtype):
            discrepancies.append(f"TYPE MISMATCH {name}: exploration={exp_col.dtype}, production={prod_col.dtype}")
        if exp_col.null_count == 0 and prod_col.null_count > 0:
            discrepancies.append(f"NEW NULLS {name}: production has {prod_col.null_count} nulls (exploration had 0)")
        exp_ratio = exp_col.null_count / max(exploration.row_count, 1)
        prod_ratio = prod_col.null_count / max(production.row_count, 1)
        if abs(exp_ratio - prod_ratio) > 0.1:
            discrepancies.append(f"NULL DRIFT {name}: exploration={exp_ratio:.1%}, production={prod_ratio:.1%}")

    return discrepancies
