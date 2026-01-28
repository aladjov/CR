from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, List, Optional, Union

import pandas as pd

from customer_retention.core.utils.leakage import get_valid_feature_columns

from .scoring_pipeline_validator import (
    ScoringPipelineValidator,
    ValidationConfig,
    ValidationReport,
)


@dataclass
class PipelineValidationConfig:
    entity_column: str = "customer_id"
    target_column: str = "target"
    timestamp_column: str = "event_timestamp"
    validation_fraction: float = 0.2
    random_state: int = 42
    absolute_tolerance: float = 1e-6
    relative_tolerance: float = 1e-5
    save_artifacts: bool = True
    artifacts_dir: Optional[Path] = None


class PipelineValidationRunner:
    def __init__(self, config: Optional[PipelineValidationConfig] = None):
        self.config = config or PipelineValidationConfig()
        self._training_features: Optional[pd.DataFrame] = None
        self._scoring_features: Optional[pd.DataFrame] = None
        self._training_predictions: Optional[pd.DataFrame] = None
        self._model: Optional[Any] = None

    def load_training_artifacts(
        self,
        features_path: Union[str, Path],
        predictions_path: Optional[Union[str, Path]] = None,
    ) -> "PipelineValidationRunner":
        self._training_features = pd.read_parquet(features_path)
        if predictions_path:
            self._training_predictions = pd.read_parquet(predictions_path)
        return self

    def load_scoring_artifacts(
        self,
        features_path: Union[str, Path],
        predictions_path: Optional[Union[str, Path]] = None,
    ) -> "PipelineValidationRunner":
        self._scoring_features = pd.read_parquet(features_path)
        if predictions_path:
            self._scoring_predictions = pd.read_parquet(predictions_path)
        return self

    def set_model(self, model: Any) -> "PipelineValidationRunner":
        self._model = model
        return self

    def extract_validation_set(
        self,
        gold_features: pd.DataFrame,
        holdout_column: Optional[str] = None,
    ) -> tuple:
        if holdout_column and holdout_column in gold_features.columns:
            is_holdout = gold_features[holdout_column].notna()
            holdout_df = gold_features[is_holdout].copy()
            training_df = gold_features[~is_holdout].copy()
        else:
            from sklearn.model_selection import train_test_split
            training_df, holdout_df = train_test_split(
                gold_features,
                test_size=self.config.validation_fraction,
                random_state=self.config.random_state,
            )
        return training_df, holdout_df

    def validate(self) -> ValidationReport:
        if self._training_features is None or self._scoring_features is None:
            raise ValueError("Must load both training and scoring features first")
        feature_columns = self._get_feature_columns()
        validator = ScoringPipelineValidator(
            training_features=self._training_features,
            scoring_features=self._scoring_features,
            training_predictions=self._training_predictions,
            scoring_predictions=getattr(self, "_scoring_predictions", None),
            model=self._model,
            feature_columns=feature_columns,
            entity_column=self.config.entity_column,
            target_column=self.config.target_column,
            config=ValidationConfig(
                absolute_tolerance=self.config.absolute_tolerance,
                relative_tolerance=self.config.relative_tolerance,
            ),
        )
        if self._model and feature_columns:
            return validator.validate_with_model()
        return validator.validate()

    def _get_feature_columns(self) -> List[str]:
        return get_valid_feature_columns(
            self._training_features,
            entity_column=self.config.entity_column,
            target_column=self.config.target_column,
            additional_exclude={self.config.timestamp_column},
        )


def run_pipeline_validation(
    gold_features_path: Union[str, Path], entity_column: str = "customer_id",
    target_column: str = "target", holdout_column: Optional[str] = None,
    prepare_features_fn: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
    model: Optional[Any] = None, feature_columns: Optional[List[str]] = None, verbose: bool = True,
) -> ValidationReport:
    _print_header(verbose)
    gold_df = pd.read_parquet(gold_features_path)
    _log(verbose, f"Loaded {len(gold_df):,} records from {gold_features_path}")

    training_df, validation_df = _split_training_validation(
        gold_df, target_column, holdout_column, verbose)

    training_prepared, scoring_prepared = _prepare_features(
        training_df, validation_df, prepare_features_fn, verbose)

    training_predictions, scoring_predictions = _generate_predictions(
        training_prepared, scoring_prepared, model, feature_columns, entity_column, verbose)

    _log(verbose, "\nRunning validation...")
    validator = ScoringPipelineValidator(
        training_features=training_prepared, scoring_features=scoring_prepared,
        training_predictions=training_predictions, scoring_predictions=scoring_predictions,
        entity_column=entity_column, target_column=target_column)
    report = validator.validate()
    if verbose:
        print("\n" + report.to_text())
    return report


def _print_header(verbose: bool) -> None:
    if verbose:
        print("=" * 60)
        print("ADVERSARIAL PIPELINE VALIDATION")
        print("=" * 60)


def _log(verbose: bool, message: str) -> None:
    if verbose:
        print(message)


def _split_training_validation(
    gold_df: pd.DataFrame, target_column: str, holdout_column: Optional[str], verbose: bool,
) -> tuple:
    original_col = holdout_column or f"original_{target_column}"
    if original_col not in gold_df.columns:
        _log(verbose, f"Warning: Holdout column '{original_col}' not found")
        _log(verbose, "Using random split for validation")
        from sklearn.model_selection import train_test_split
        return train_test_split(gold_df, test_size=0.1, random_state=42)

    is_holdout = gold_df[target_column].isna() & gold_df[original_col].notna()
    validation_df, training_df = gold_df[is_holdout].copy(), gold_df[~is_holdout].copy()
    _log(verbose, f"Training records: {len(training_df):,}")
    _log(verbose, f"Validation (holdout) records: {len(validation_df):,}")
    return training_df, validation_df


def _prepare_features(
    training_df: pd.DataFrame, validation_df: pd.DataFrame,
    prepare_fn: Optional[Callable], verbose: bool,
) -> tuple:
    if not prepare_fn:
        return training_df, validation_df
    _log(verbose, "\nApplying feature preparation (training mode)...")
    training_prepared = prepare_fn(training_df.copy())
    _log(verbose, "Applying feature preparation (scoring mode)...")
    scoring_prepared = prepare_fn(validation_df.copy())
    return training_prepared, scoring_prepared


def _generate_predictions(
    training_df: pd.DataFrame, scoring_df: pd.DataFrame,
    model: Optional[Any], feature_columns: Optional[List[str]],
    entity_column: str, verbose: bool,
) -> tuple:
    if model is None or feature_columns is None:
        return None, None

    _log(verbose, "\nGenerating model predictions...")
    train_X, score_X = training_df[feature_columns].values, scoring_df[feature_columns].values
    train_pred, score_pred = model.predict(train_X), model.predict(score_X)

    if hasattr(model, "predict_proba"):
        train_proba, score_proba = model.predict_proba(train_X)[:, 1], model.predict_proba(score_X)[:, 1]
    else:
        train_proba, score_proba = train_pred.astype(float), score_pred.astype(float)

    return (
        _build_predictions_df(training_df, entity_column, train_pred, train_proba),
        _build_predictions_df(scoring_df, entity_column, score_pred, score_proba),
    )


def _build_predictions_df(df: pd.DataFrame, entity_column: str, y_pred, y_proba) -> pd.DataFrame:
    entity_values = df[entity_column] if entity_column in df.columns else range(len(df))
    return pd.DataFrame({entity_column: entity_values, "y_pred": y_pred, "y_proba": y_proba})


def validate_feature_transformation(
    training_df: pd.DataFrame, scoring_df: pd.DataFrame,
    transform_fn: Callable[[pd.DataFrame], pd.DataFrame],
    entity_column: str = "customer_id", verbose: bool = True,
) -> ValidationReport:
    _log(verbose, "Validating transformation consistency...")
    training_transformed = transform_fn(training_df.copy())
    scoring_transformed = transform_fn(scoring_df.copy())
    validator = ScoringPipelineValidator(
        training_features=training_transformed, scoring_features=scoring_transformed,
        entity_column=entity_column)
    report = validator.validate_features()
    if verbose:
        status = "PASSED" if report.passed else "FAILED"
        print(f"Transformation validation: {status}")
        if not report.passed:
            print(f"  Mismatched features: {len(report.feature_mismatches)}")
    return report


def compare_pipeline_outputs(
    training_output_path: Union[str, Path], scoring_output_path: Union[str, Path],
    entity_column: str = "customer_id", target_column: str = "target",
    output_report_path: Optional[Union[str, Path]] = None, verbose: bool = True,
) -> ValidationReport:
    training_df = pd.read_parquet(training_output_path)
    scoring_df = pd.read_parquet(scoring_output_path)
    _log(verbose, f"Training output: {len(training_df):,} records, {len(training_df.columns)} columns")
    _log(verbose, f"Scoring output: {len(scoring_df):,} records, {len(scoring_df.columns)} columns")

    validator = ScoringPipelineValidator(
        training_features=training_df, scoring_features=scoring_df,
        entity_column=entity_column, target_column=target_column)
    report = validator.validate()

    if verbose:
        print("\n" + report.to_text())
    if output_report_path:
        report.save(output_report_path)
        _log(verbose, f"\nReport saved to: {output_report_path}")
    return report
