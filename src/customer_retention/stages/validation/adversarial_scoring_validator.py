"""Adversarial validation between training and scoring pipelines.

Validates that the scoring pipeline produces identical features to training
for the same holdout entities, catching transformation inconsistencies.
"""
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Callable, List, Optional

import numpy as np
import pandas as pd

from customer_retention.core.utils.leakage import get_valid_feature_columns


class DriftSeverity(IntEnum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class FeatureDrift:
    feature_name: str
    max_absolute_diff: float
    mean_absolute_diff: float
    affected_entities: int
    severity: DriftSeverity
    sample_diffs: Optional[List[float]] = None


@dataclass
class AdversarialValidationResult:
    passed: bool
    entities_validated: int
    feature_drifts: List[FeatureDrift] = field(default_factory=list)
    missing_entities: int = 0
    extra_entities: int = 0

    @property
    def summary(self) -> str:
        status = "PASSED" if self.passed else "FAILED"
        lines = [
            f"Adversarial Validation: {status}",
            f"Entities validated: {self.entities_validated}",
        ]
        if self.feature_drifts:
            lines.append(f"Features with drift: {len(self.feature_drifts)}")
        if self.missing_entities:
            lines.append(f"Missing entities: {self.missing_entities}")
        return "\n".join(lines)

    def to_dataframe(self) -> pd.DataFrame:
        if not self.feature_drifts:
            return pd.DataFrame(columns=["feature_name", "severity", "max_diff", "mean_diff", "affected"])
        return pd.DataFrame([
            {
                "feature_name": d.feature_name,
                "severity": d.severity.name,
                "max_diff": d.max_absolute_diff,
                "mean_diff": d.mean_absolute_diff,
                "affected": d.affected_entities,
            }
            for d in self.feature_drifts
        ])


class AdversarialScoringValidator:
    def __init__(
        self,
        gold_features: pd.DataFrame,
        entity_column: str = "customer_id",
        target_column: str = "target",
        tolerance: float = 1e-6,
    ):
        self.gold_features = gold_features
        self.entity_column = entity_column
        self.target_column = target_column
        self.tolerance = tolerance
        self._holdout_column = f"original_{target_column}"

    def get_holdout_entity_ids(self) -> List:
        if self._holdout_column not in self.gold_features.columns:
            return []
        is_holdout = (
            self.gold_features[self.target_column].isna() &
            self.gold_features[self._holdout_column].notna()
        )
        return self.gold_features.loc[is_holdout, self.entity_column].tolist()

    def validate_features(self, recomputed_features: pd.DataFrame) -> AdversarialValidationResult:
        gold_holdout = self._get_holdout_features()
        if gold_holdout.empty:
            return AdversarialValidationResult(passed=True, entities_validated=0)
        common_entities = set(gold_holdout[self.entity_column]) & set(recomputed_features[self.entity_column])
        if not common_entities:
            return AdversarialValidationResult(
                passed=True, entities_validated=0,
                missing_entities=len(gold_holdout),
            )
        gold_mask = gold_holdout[self.entity_column].isin(common_entities)
        gold_aligned = gold_holdout[gold_mask].set_index(self.entity_column)
        recomp_mask = recomputed_features[self.entity_column].isin(common_entities)
        recomputed_aligned = recomputed_features[recomp_mask].set_index(self.entity_column)
        recomputed_aligned = recomputed_aligned.loc[gold_aligned.index]
        feature_cols = self._get_feature_columns(gold_aligned)
        drifts = []
        for col in feature_cols:
            if col not in recomputed_aligned.columns:
                continue
            drift = self._check_column_drift(gold_aligned[col], recomputed_aligned[col], col)
            if drift:
                drifts.append(drift)
        passed = len(drifts) == 0
        return AdversarialValidationResult(
            passed=passed,
            entities_validated=len(common_entities),
            feature_drifts=drifts,
            missing_entities=len(gold_holdout) - len(common_entities),
        )

    def validate_with_transform(
        self,
        silver_data: pd.DataFrame,
        transform_fn: Callable[[pd.DataFrame], pd.DataFrame],
    ) -> AdversarialValidationResult:
        holdout_ids = self.get_holdout_entity_ids()
        if not holdout_ids:
            return AdversarialValidationResult(passed=True, entities_validated=0)
        holdout_silver = silver_data[silver_data[self.entity_column].isin(holdout_ids)].copy()
        if holdout_silver.empty:
            return AdversarialValidationResult(passed=True, entities_validated=0, missing_entities=len(holdout_ids))
        recomputed = transform_fn(holdout_silver)
        return self.validate_features(recomputed)

    def _get_holdout_features(self) -> pd.DataFrame:
        if self._holdout_column not in self.gold_features.columns:
            return pd.DataFrame()
        is_holdout = (
            self.gold_features[self.target_column].isna() &
            self.gold_features[self._holdout_column].notna()
        )
        return self.gold_features[is_holdout].copy()

    def _get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        return get_valid_feature_columns(
            df,
            entity_column=self.entity_column,
            target_column=self.target_column,
            additional_exclude={self._holdout_column},
        )

    def _check_column_drift(
        self, gold_col: pd.Series, recomputed_col: pd.Series, col_name: str
    ) -> Optional[FeatureDrift]:
        if gold_col.dtype in ("object", "category") or recomputed_col.dtype in ("object", "category"):
            return self._check_categorical_drift(gold_col, recomputed_col, col_name)
        return self._check_numeric_drift(gold_col, recomputed_col, col_name)

    def _check_numeric_drift(
        self, gold_col: pd.Series, recomputed_col: pd.Series, col_name: str
    ) -> Optional[FeatureDrift]:
        gold_vals = gold_col.fillna(0).values.astype(float)
        recomputed_vals = recomputed_col.fillna(0).values.astype(float)
        diff = np.abs(gold_vals - recomputed_vals)
        affected = np.sum(diff > self.tolerance)
        if affected == 0:
            return None
        max_diff = float(np.max(diff))
        mean_diff = float(np.mean(diff[diff > self.tolerance]))
        severity = self._compute_severity(max_diff, affected, len(gold_col))
        return FeatureDrift(
            feature_name=col_name,
            max_absolute_diff=max_diff,
            mean_absolute_diff=mean_diff,
            affected_entities=int(affected),
            severity=severity,
        )

    def _check_categorical_drift(
        self, gold_col: pd.Series, recomputed_col: pd.Series, col_name: str
    ) -> Optional[FeatureDrift]:
        mismatched = gold_col.astype(str) != recomputed_col.astype(str)
        affected = mismatched.sum()
        if affected == 0:
            return None
        severity = self._compute_severity(1.0, affected, len(gold_col))
        return FeatureDrift(
            feature_name=col_name,
            max_absolute_diff=1.0,
            mean_absolute_diff=1.0,
            affected_entities=int(affected),
            severity=severity,
        )

    def _compute_severity(self, max_diff: float, affected: int, total: int) -> DriftSeverity:
        affected_pct = affected / total if total > 0 else 0
        if affected_pct > 0.5 or max_diff > 10:
            return DriftSeverity.CRITICAL
        if affected_pct > 0.2 or max_diff > 1:
            return DriftSeverity.HIGH
        if affected_pct > 0.05 or max_diff > 0.1:
            return DriftSeverity.MEDIUM
        return DriftSeverity.LOW
