from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import yaml

from customer_retention.core.utils.leakage import get_valid_feature_columns


class MismatchSeverity(IntEnum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class FeatureMismatch:
    feature_name: str
    severity: MismatchSeverity
    training_mean: float
    scoring_mean: float
    max_absolute_diff: float
    mismatch_percentage: float
    training_std: Optional[float] = None
    scoring_std: Optional[float] = None
    sample_differences: Optional[List[Dict[str, Any]]] = None

    def to_dict(self) -> dict:
        return {
            "feature_name": self.feature_name,
            "severity": self.severity.name,
            "training_mean": self.training_mean,
            "scoring_mean": self.scoring_mean,
            "max_absolute_diff": self.max_absolute_diff,
            "mismatch_percentage": self.mismatch_percentage,
            "training_std": self.training_std,
            "scoring_std": self.scoring_std,
        }


@dataclass
class PredictionMismatch:
    entity_id: str
    training_prediction: int
    scoring_prediction: int
    training_proba: Optional[float] = None
    scoring_proba: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "entity_id": self.entity_id,
            "training_prediction": self.training_prediction,
            "scoring_prediction": self.scoring_prediction,
            "training_proba": self.training_proba,
            "scoring_proba": self.scoring_proba,
        }


@dataclass
class ValidationConfig:
    absolute_tolerance: float = 1e-6
    relative_tolerance: float = 1e-5
    prediction_threshold: float = 0.5
    max_sample_differences: int = 10
    severity_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "low": 0.01,
        "medium": 0.05,
        "high": 0.10,
        "critical": 0.25,
    })


@dataclass
class ValidationReport:
    passed: bool = True
    feature_mismatches: List[FeatureMismatch] = field(default_factory=list)
    prediction_mismatches: List[PredictionMismatch] = field(default_factory=list)
    features_validated: bool = False
    predictions_validated: bool = False
    missing_entities_count: int = 0
    extra_entities_count: int = 0
    total_entities_compared: int = 0
    validation_timestamp: Optional[str] = None

    def __post_init__(self):
        if self.feature_mismatches or self.prediction_mismatches:
            self.passed = False
        if self.missing_entities_count > 0:
            self.passed = False

    def summary(self) -> dict:
        high_severity = sum(1 for m in self.feature_mismatches if m.severity >= MismatchSeverity.HIGH)
        return {
            "passed": self.passed,
            "total_feature_mismatches": len(self.feature_mismatches),
            "total_prediction_mismatches": len(self.prediction_mismatches),
            "high_severity_features": high_severity,
            "missing_entities": self.missing_entities_count,
            "extra_entities": self.extra_entities_count,
            "total_compared": self.total_entities_compared,
        }

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "feature_mismatches": [m.to_dict() for m in self.feature_mismatches],
            "prediction_mismatches": [m.to_dict() for m in self.prediction_mismatches],
            "features_validated": self.features_validated,
            "predictions_validated": self.predictions_validated,
            "missing_entities_count": self.missing_entities_count,
            "extra_entities_count": self.extra_entities_count,
            "total_entities_compared": self.total_entities_compared,
            "summary": self.summary(),
        }

    def to_text(self) -> str:
        lines = ["=" * 60, "SCORING PIPELINE VALIDATION REPORT", "=" * 60]
        lines.append(f"Status: {'PASSED' if self.passed else 'FAILED'}")
        lines.append(f"Entities compared: {self.total_entities_compared}")
        if self.missing_entities_count:
            lines.append(f"Missing entities: {self.missing_entities_count}")
        if self.extra_entities_count:
            lines.append(f"Extra entities: {self.extra_entities_count}")
        if self.feature_mismatches:
            lines.append("\nFEATURE MISMATCHES:")
            lines.append("-" * 40)
            for m in self.feature_mismatches:
                lines.append(f"  {m.feature_name} [{m.severity.name}]:")
                lines.append(f"    Training mean: {m.training_mean:.6f}")
                lines.append(f"    Scoring mean:  {m.scoring_mean:.6f}")
                lines.append(f"    Max diff: {m.max_absolute_diff:.6f}")
                lines.append(f"    Mismatch %: {m.mismatch_percentage:.2f}%")
        if self.prediction_mismatches:
            lines.append("\nPREDICTION MISMATCHES:")
            lines.append("-" * 40)
            for m in self.prediction_mismatches[:10]:
                lines.append(f"  {m.entity_id}: train={m.training_prediction} vs score={m.scoring_prediction}")
            if len(self.prediction_mismatches) > 10:
                lines.append(f"  ... and {len(self.prediction_mismatches) - 10} more")
        return "\n".join(lines)

    def to_dataframe(self) -> pd.DataFrame:
        if not self.feature_mismatches:
            return pd.DataFrame(columns=["feature_name", "severity", "training_mean", "scoring_mean", "max_absolute_diff", "mismatch_percentage"])
        return pd.DataFrame([m.to_dict() for m in self.feature_mismatches])

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)


class ScoringPipelineValidator:
    def __init__(
        self, training_features: Union[pd.DataFrame, Path, str],
        scoring_features: Union[pd.DataFrame, Path, str],
        training_predictions: Optional[Union[pd.DataFrame, Path, str]] = None,
        scoring_predictions: Optional[Union[pd.DataFrame, Path, str]] = None,
        model: Optional[Any] = None, feature_columns: Optional[List[str]] = None,
        entity_column: Optional[str] = None, target_column: Optional[str] = None,
        config: Optional[ValidationConfig] = None,
    ):
        self.training_features = self._load_dataframe(training_features)
        self.scoring_features = self._load_dataframe(scoring_features)
        self.training_predictions = self._load_dataframe(training_predictions) if training_predictions is not None else None
        self.scoring_predictions = self._load_dataframe(scoring_predictions) if scoring_predictions is not None else None
        self.model = model
        self.feature_columns = feature_columns
        self.entity_column = entity_column
        self.target_column = target_column
        self.config = config or ValidationConfig()

    def _load_dataframe(self, data: Union[pd.DataFrame, Path, str]) -> pd.DataFrame:
        if isinstance(data, pd.DataFrame):
            return data
        path = Path(data)
        if path.suffix == ".parquet":
            return pd.read_parquet(path)
        if path.suffix == ".csv":
            return pd.read_csv(path)
        raise ValueError(f"Unsupported file format: {path.suffix}")

    def _get_comparable_columns(self) -> List[str]:
        train_valid = set(get_valid_feature_columns(
            self.training_features,
            entity_column=self.entity_column,
            target_column=self.target_column,
        ))
        score_valid = set(get_valid_feature_columns(
            self.scoring_features,
            entity_column=self.entity_column,
            target_column=self.target_column,
        ))
        return sorted(train_valid & score_valid)

    def _align_dataframes(self) -> tuple:
        if self.entity_column and self.entity_column in self.training_features.columns:
            train_entities = set(self.training_features[self.entity_column])
            score_entities = set(self.scoring_features[self.entity_column])
            common = train_entities & score_entities
            missing = train_entities - score_entities
            extra = score_entities - train_entities
            train_aligned = self.training_features[self.training_features[self.entity_column].isin(common)].copy()
            score_aligned = self.scoring_features[self.scoring_features[self.entity_column].isin(common)].copy()
            train_aligned = train_aligned.sort_values(self.entity_column).reset_index(drop=True)
            score_aligned = score_aligned.sort_values(self.entity_column).reset_index(drop=True)
            return train_aligned, score_aligned, len(missing), len(extra)
        return self.training_features, self.scoring_features, 0, 0

    def _classify_severity(self, mismatch_pct: float, max_diff: float = 0.0, mean_val: float = 1.0) -> MismatchSeverity:
        thresholds = self.config.severity_thresholds
        rel_diff = max_diff / abs(mean_val) if mean_val != 0 else max_diff
        if rel_diff >= thresholds["critical"]:
            return MismatchSeverity.CRITICAL
        elif rel_diff >= thresholds["high"]:
            return MismatchSeverity.HIGH
        elif rel_diff >= thresholds["medium"]:
            return MismatchSeverity.MEDIUM
        return MismatchSeverity.LOW

    def _compare_numeric_column(self, train_col: pd.Series, score_col: pd.Series, col_name: str) -> Optional[FeatureMismatch]:
        train_vals, score_vals = train_col.values.astype(float), score_col.values.astype(float)
        train_nan_mask, score_nan_mask = np.isnan(train_vals), np.isnan(score_vals)

        if not np.array_equal(train_nan_mask, score_nan_mask):
            return self._create_nan_mismatch(col_name, train_vals, score_vals, train_nan_mask, score_nan_mask)

        valid_mask = ~train_nan_mask
        if not valid_mask.any():
            return None

        train_valid, score_valid = train_vals[valid_mask], score_vals[valid_mask]
        abs_diff = np.abs(train_valid - score_valid)
        max_diff = float(np.max(abs_diff))

        if self._is_within_tolerance(max_diff, train_valid):
            return None

        mismatch_pct = np.sum(abs_diff > self.config.absolute_tolerance) / len(train_valid) * 100
        train_mean = float(np.mean(train_valid))
        return FeatureMismatch(
            feature_name=col_name, severity=self._classify_severity(mismatch_pct, max_diff, train_mean),
            training_mean=train_mean, scoring_mean=float(np.mean(score_valid)),
            max_absolute_diff=max_diff, mismatch_percentage=mismatch_pct,
            training_std=float(np.std(train_valid)) if len(train_valid) > 1 else None,
            scoring_std=float(np.std(score_valid)) if len(score_valid) > 1 else None)

    def _create_nan_mismatch(self, col_name: str, train_vals, score_vals, train_nan, score_nan) -> FeatureMismatch:
        nan_diff_count = np.sum(train_nan != score_nan)
        mismatch_pct = nan_diff_count / len(train_vals) * 100 if len(train_vals) > 0 else 0
        return FeatureMismatch(
            feature_name=col_name, severity=self._classify_severity(mismatch_pct),
            training_mean=float(np.nanmean(train_vals)), scoring_mean=float(np.nanmean(score_vals)),
            max_absolute_diff=float("inf"), mismatch_percentage=mismatch_pct)

    def _is_within_tolerance(self, max_diff: float, train_valid: np.ndarray) -> bool:
        if max_diff <= self.config.absolute_tolerance:
            return True
        train_max = np.max(np.abs(train_valid)) if len(train_valid) > 0 else 1.0
        return train_max > 0 and max_diff / train_max <= self.config.relative_tolerance

    def _compare_categorical_column(self, train_col: pd.Series, score_col: pd.Series, col_name: str) -> Optional[FeatureMismatch]:
        train_vals = train_col.astype(str).values
        score_vals = score_col.astype(str).values
        mismatches = train_vals != score_vals
        mismatch_count = np.sum(mismatches)
        if mismatch_count == 0:
            return None
        mismatch_pct = mismatch_count / len(train_vals) * 100 if len(train_vals) > 0 else 0
        return FeatureMismatch(
            feature_name=col_name,
            severity=self._classify_severity(mismatch_pct),
            training_mean=0.0,
            scoring_mean=0.0,
            max_absolute_diff=float(mismatch_count),
            mismatch_percentage=mismatch_pct,
        )

    def validate_features(self) -> ValidationReport:
        train_aligned, score_aligned, missing_count, extra_count = self._align_dataframes()
        if len(train_aligned) == 0:
            return ValidationReport(
                passed=True,
                features_validated=True,
                total_entities_compared=0,
                missing_entities_count=missing_count,
                extra_entities_count=extra_count,
            )
        comparable_cols = self._get_comparable_columns()
        feature_mismatches = []
        for col in comparable_cols:
            if col not in train_aligned.columns or col not in score_aligned.columns:
                continue
            train_col = train_aligned[col]
            score_col = score_aligned[col]
            if pd.api.types.is_numeric_dtype(train_col):
                mismatch = self._compare_numeric_column(train_col, score_col, col)
            else:
                mismatch = self._compare_categorical_column(train_col, score_col, col)
            if mismatch:
                feature_mismatches.append(mismatch)
        return ValidationReport(
            feature_mismatches=feature_mismatches,
            features_validated=True,
            total_entities_compared=len(train_aligned),
            missing_entities_count=missing_count,
            extra_entities_count=extra_count,
        )

    def validate_predictions(self) -> ValidationReport:
        if self.training_predictions is None or self.scoring_predictions is None:
            return ValidationReport(passed=True, predictions_validated=False)

        train_preds, score_preds = self._sort_predictions_by_entity(
            self.training_predictions, self.scoring_predictions)

        pred_col = "y_pred" if "y_pred" in train_preds.columns else "prediction"
        if pred_col not in train_preds.columns:
            return ValidationReport(passed=True, predictions_validated=False)

        proba_col = "y_proba" if "y_proba" in train_preds.columns else "probability"
        prediction_mismatches = self._collect_prediction_mismatches(
            train_preds, score_preds, pred_col, proba_col)

        return ValidationReport(
            prediction_mismatches=prediction_mismatches, predictions_validated=True,
            total_entities_compared=len(train_preds))

    def _sort_predictions_by_entity(self, train_preds: pd.DataFrame, score_preds: pd.DataFrame) -> tuple:
        if self.entity_column and self.entity_column in train_preds.columns:
            train_preds = train_preds.sort_values(self.entity_column).reset_index(drop=True)
            score_preds = score_preds.sort_values(self.entity_column).reset_index(drop=True)
        return train_preds, score_preds

    def _collect_prediction_mismatches(
        self, train_df: pd.DataFrame, score_df: pd.DataFrame, pred_col: str, proba_col: str,
        train_preds: Optional[np.ndarray] = None, score_preds: Optional[np.ndarray] = None,
        train_proba: Optional[np.ndarray] = None, score_proba: Optional[np.ndarray] = None,
    ) -> List[PredictionMismatch]:
        if train_preds is None:
            train_preds = train_df[pred_col].values
        if score_preds is None:
            score_preds = score_df[pred_col].values

        mismatches = []
        for idx in np.where(train_preds != score_preds)[0]:
            entity_id = str(train_df[self.entity_column].iloc[idx]) if self.entity_column else str(idx)
            t_proba = float(train_proba[idx]) if train_proba is not None else (
                float(train_df[proba_col].iloc[idx]) if proba_col in train_df.columns else None)
            s_proba = float(score_proba[idx]) if score_proba is not None else (
                float(score_df[proba_col].iloc[idx]) if proba_col in score_df.columns else None)
            mismatches.append(PredictionMismatch(
                entity_id=entity_id, training_prediction=int(train_preds[idx]),
                scoring_prediction=int(score_preds[idx]), training_proba=t_proba, scoring_proba=s_proba))
        return mismatches

    def validate(self) -> ValidationReport:
        feature_report = self.validate_features()
        if not feature_report.passed:
            return feature_report
        if self.training_predictions is not None and self.scoring_predictions is not None:
            pred_report = self.validate_predictions()
            return ValidationReport(
                passed=feature_report.passed and pred_report.passed,
                feature_mismatches=feature_report.feature_mismatches,
                prediction_mismatches=pred_report.prediction_mismatches,
                features_validated=True,
                predictions_validated=True,
                missing_entities_count=feature_report.missing_entities_count,
                extra_entities_count=feature_report.extra_entities_count,
                total_entities_compared=feature_report.total_entities_compared,
            )
        return feature_report

    def validate_with_model(self) -> ValidationReport:
        feature_report = self.validate_features()
        if self.model is None or self.feature_columns is None:
            return feature_report

        train_aligned, score_aligned, _, _ = self._align_dataframes()
        if len(train_aligned) == 0:
            return feature_report

        X_train, X_score = train_aligned[self.feature_columns].values, score_aligned[self.feature_columns].values
        train_preds, score_preds = self.model.predict(X_train), self.model.predict(X_score)
        train_proba = self.model.predict_proba(X_train)[:, 1] if hasattr(self.model, "predict_proba") else None
        score_proba = self.model.predict_proba(X_score)[:, 1] if hasattr(self.model, "predict_proba") else None

        prediction_mismatches = self._collect_prediction_mismatches(
            train_aligned, score_aligned, pred_col="", proba_col="",
            train_preds=train_preds, score_preds=score_preds,
            train_proba=train_proba, score_proba=score_proba)

        return ValidationReport(
            passed=feature_report.passed and len(prediction_mismatches) == 0,
            feature_mismatches=feature_report.feature_mismatches,
            prediction_mismatches=prediction_mismatches,
            features_validated=True, predictions_validated=True,
            missing_entities_count=feature_report.missing_entities_count,
            extra_entities_count=feature_report.extra_entities_count,
            total_entities_compared=feature_report.total_entities_compared)
