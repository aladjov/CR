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

    def to_dataframe(self) -> Any:
        if not self.feature_drifts:
            return native_pd.DataFrame(columns=["feature_name", "severity", "max_diff", "mean_diff", "affected"])
        return native_pd.DataFrame([
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
        import pyspark.sql.functions as F  # noqa: N812
        from pyspark.sql.types import NumericType

        from customer_retention.core.compat import as_spark_df, normalize_timestamps, pandas_dtype_to_spark_schema
        from customer_retention.core.compat.detection import get_spark_session

        spark = get_spark_session()
        ek = self.entity_column
        holdout_col = self._holdout_column
        gold_holdout = as_spark_df(self.gold_features).filter(
            F.col(self.target_column).isNull() & F.col(holdout_col).isNotNull()
        )
        exclude = {ek, self.target_column, holdout_col}
        feature_cols = [
            c for c in gold_holdout.columns
            if c not in exclude and not c.startswith("original_") and c in recomputed_features.columns
        ]
        if not feature_cols:
            return AdversarialValidationResult(passed=True, entities_validated=0)

        score_pd = normalize_timestamps(recomputed_features)
        score_spark = spark.createDataFrame(score_pd, schema=pandas_dtype_to_spark_schema(score_pd))

        gold_schema = {f.name: f.dataType for f in gold_holdout.schema.fields}
        numeric_cols = [c for c in feature_cols if isinstance(gold_schema.get(c), NumericType)]
        categorical_cols = [c for c in feature_cols if c not in numeric_cols]

        gold_sel = gold_holdout.select(F.col(ek), *[F.col(c).alias(f"g_{c}") for c in feature_cols])
        score_sel = score_spark.select(F.col(ek), *[F.col(c).alias(f"s_{c}") for c in feature_cols])
        joined = gold_sel.join(F.broadcast(score_sel), ek, "inner")

        agg_exprs = [F.count("*").alias("_n")]
        for c in numeric_cols:
            diff = F.abs(F.col(f"g_{c}").cast("double") - F.col(f"s_{c}").cast("double"))
            agg_exprs.append(F.max(diff).alias(f"max_{c}"))
            agg_exprs.append(F.avg(F.when(diff > self.tolerance, diff)).alias(f"mean_{c}"))
            agg_exprs.append(F.sum(F.when(diff > self.tolerance, F.lit(1)).otherwise(F.lit(0))).alias(f"cnt_{c}"))
        for c in categorical_cols:
            mismatch = F.when(F.col(f"g_{c}").cast("string") != F.col(f"s_{c}").cast("string"), F.lit(1)).otherwise(F.lit(0))
            agg_exprs.append(F.sum(mismatch).alias(f"cat_{c}"))

        summary = joined.agg(*agg_exprs).collect()[0]
        n_entities = int(summary["_n"])

        drifts = []
        for c in numeric_cols:
            max_val = summary[f"max_{c}"]
            if max_val is None or float(max_val) <= self.tolerance:
                continue
            affected = int(summary[f"cnt_{c}"] or 0)
            drifts.append(FeatureDrift(
                feature_name=c, max_absolute_diff=float(max_val),
                mean_absolute_diff=float(summary[f"mean_{c}"] or 0),
                affected_entities=affected,
                severity=self._compute_severity(float(max_val), affected, n_entities),
            ))
        for c in categorical_cols:
            affected = int(summary[f"cat_{c}"] or 0)
            if affected == 0:
                continue
            drifts.append(FeatureDrift(
                feature_name=c, max_absolute_diff=1.0, mean_absolute_diff=1.0,
                affected_entities=affected,
                severity=self._compute_severity(1.0, affected, n_entities),
            ))

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
        return FeatureDrift(
            feature_name=col_name,
            max_absolute_diff=max_diff,
            mean_absolute_diff=mean_diff,
            affected_entities=int(affected),
            severity=severity,
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
