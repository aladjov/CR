from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)


@dataclass
class ColumnProfile:
    dtype: str
    non_null_count: int
    null_count: int


@dataclass
class FeatureProfile:
    stage: str
    created_at: str
    row_count: int
    target_column: str
    features: Dict[str, ColumnProfile]
    excluded: Dict[str, str] = field(default_factory=dict)

    @property
    def feature_count(self) -> int:
        return len(self.features)

    def to_dict(self) -> dict:
        return {
            "stage": self.stage,
            "created_at": self.created_at,
            "row_count": self.row_count,
            "feature_count": self.feature_count,
            "target_column": self.target_column,
            "features": {
                name: {"dtype": col.dtype, "non_null": col.non_null_count, "null_count": col.null_count}
                for name, col in self.features.items()
            },
            "excluded": self.excluded,
        }

    @classmethod
    def from_dict(cls, data: dict) -> FeatureProfile:
        features = {
            name: ColumnProfile(dtype=v["dtype"], non_null_count=v["non_null"], null_count=v["null_count"])
            for name, v in data.get("features", {}).items()
        }
        return cls(
            stage=data["stage"], created_at=data["created_at"],
            row_count=data["row_count"], target_column=data["target_column"],
            features=features, excluded=data.get("excluded", {}),
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


def build_feature_profile(stage: str, target_column: str, row_count: int, feature_stats: Dict[str, ColumnProfile], excluded: Optional[Dict[str, str]] = None) -> FeatureProfile:
    return FeatureProfile(
        stage=stage, created_at=datetime.now(timezone.utc).isoformat(),
        row_count=row_count, target_column=target_column,
        features=feature_stats, excluded=excluded or {},
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
