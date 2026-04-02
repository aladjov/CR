"""Adversarial validation between training and scoring pipelines.

Validates that the scoring pipeline produces identical features to training
for the same holdout entities, catching transformation inconsistencies.
"""
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Callable, List, Optional

import numpy as np

from customer_retention.core.compat import _is_spark_pandas, native_pd
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
    gold_mean: Optional[float] = None
    gold_std: Optional[float] = None
    scoring_mean: Optional[float] = None
    scoring_std: Optional[float] = None
    sample_entity_values: Optional[List[dict]] = None


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
            counts: dict = {}
            for d in self.feature_drifts:
                counts[d.severity.name] = counts.get(d.severity.name, 0) + 1
            lines.append(f"  Severity: {', '.join(f'{k}={v}' for k, v in sorted(counts.items()))}")
        if self.missing_entities:
            lines.append(f"Missing entities: {self.missing_entities}")
        return "\n".join(lines)

    def to_dataframe(self) -> Any:
        if not self.feature_drifts:
            return native_pd.DataFrame(columns=[
                "feature_name", "severity", "max_diff", "mean_diff", "affected",
                "gold_mean", "gold_std", "scoring_mean", "scoring_std",
            ])
        return native_pd.DataFrame([
            {
                "feature_name": d.feature_name,
                "severity": d.severity.name,
                "max_diff": d.max_absolute_diff,
                "mean_diff": d.mean_absolute_diff,
                "affected": d.affected_entities,
                "gold_mean": d.gold_mean,
                "gold_std": d.gold_std,
                "scoring_mean": d.scoring_mean,
                "scoring_std": d.scoring_std,
            }
            for d in self.feature_drifts
        ])


class AdversarialScoringValidator:
    def __init__(
        self,
        gold_features: Any,
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

    def validate_features(self, recomputed_features: Any) -> AdversarialValidationResult:
        if _is_spark_pandas(self.gold_features):
            return self._validate_features_distributed(recomputed_features)
        return self._validate_features_local(recomputed_features)

    def _validate_features_local(self, recomputed_features: Any) -> AdversarialValidationResult:
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

    def _validate_features_distributed(self, recomputed_features: Any) -> AdversarialValidationResult:
        from customer_retention.core.compat.bulk_profiling import batch_adversarial_diffs

        n_entities, col_diffs = batch_adversarial_diffs(
            self.gold_features, recomputed_features,
            self.entity_column, self.target_column, self._holdout_column, self.tolerance,
        )
        drifts = [
            FeatureDrift(col, max_d, mean_d, cnt, self._compute_severity(max_d, cnt, n_entities))
            for col, (max_d, mean_d, cnt) in col_diffs.items()
        ]
        return AdversarialValidationResult(
            passed=len(drifts) == 0, entities_validated=n_entities, feature_drifts=drifts,
        )

    def validate_with_transform(
        self,
        silver_data: Any,
        transform_fn: Callable[[Any], Any],
    ) -> AdversarialValidationResult:
        holdout_ids = self.get_holdout_entity_ids()
        if not holdout_ids:
            return AdversarialValidationResult(passed=True, entities_validated=0)
        holdout_silver = silver_data[silver_data[self.entity_column].isin(holdout_ids)]
        if holdout_silver.empty:
            return AdversarialValidationResult(passed=True, entities_validated=0, missing_entities=len(holdout_ids))
        recomputed = transform_fn(holdout_silver)
        return self.validate_features(recomputed)

    def _get_holdout_features(self) -> Any:
        if self._holdout_column not in self.gold_features.columns:
            return native_pd.DataFrame()
        is_holdout = (
            self.gold_features[self.target_column].isna() &
            self.gold_features[self._holdout_column].notna()
        )
        return self.gold_features[is_holdout]

    def _get_feature_columns(self, df: Any) -> List[str]:
        return get_valid_feature_columns(
            df,
            entity_column=self.entity_column,
            target_column=self.target_column,
            additional_exclude={self._holdout_column},
        )

    def _check_column_drift(
        self, gold_col: Any, recomputed_col: Any, col_name: str
    ) -> Optional[FeatureDrift]:
        if gold_col.dtype in ("object", "category") or recomputed_col.dtype in ("object", "category"):
            return self._check_categorical_drift(gold_col, recomputed_col, col_name)
        return self._check_numeric_drift(gold_col, recomputed_col, col_name)

    def _check_numeric_drift(
        self, gold_col: Any, recomputed_col: Any, col_name: str
    ) -> Optional[FeatureDrift]:
        gold_vals = gold_col.fillna(0).to_numpy().astype(float)
        recomputed_vals = recomputed_col.fillna(0).to_numpy().astype(float)
        diff = np.abs(gold_vals - recomputed_vals)
        affected = np.sum(diff > self.tolerance)
        if affected == 0:
            return None
        max_diff = float(np.max(diff))
        mean_diff = float(np.mean(diff[diff > self.tolerance]))
        severity = self._compute_severity(max_diff, affected, len(gold_col))

        sample_entity_values = None
        affected_mask = diff > self.tolerance
        affected_indices = affected_mask.nonzero()[0][:5]
        if hasattr(gold_col, "index") and len(affected_indices) > 0:
            sample_entity_values = [
                {
                    "entity_id": str(gold_col.index[idx]),
                    "gold_value": float(gold_vals[idx]),
                    "scoring_value": float(recomputed_vals[idx]),
                }
                for idx in affected_indices
            ]

        return FeatureDrift(
            feature_name=col_name,
            max_absolute_diff=max_diff,
            mean_absolute_diff=mean_diff,
            affected_entities=int(affected),
            severity=severity,
            gold_mean=float(np.mean(gold_vals)),
            gold_std=float(np.std(gold_vals)),
            scoring_mean=float(np.mean(recomputed_vals)),
            scoring_std=float(np.std(recomputed_vals)),
            sample_entity_values=sample_entity_values,
        )

    def _check_categorical_drift(
        self, gold_col: Any, recomputed_col: Any, col_name: str
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
