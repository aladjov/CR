import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from customer_retention.analysis.auto_explorer.findings import ExplorationFindings
from customer_retention.core.config.column_config import ColumnType


@dataclass
class CleanAction:
    action_type: str
    strategy: str = ""
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TransformAction:
    action_type: str
    method: str = ""
    params: Dict[str, Any] = field(default_factory=dict)


class RecommendationParser:
    CLEANING_PATTERNS = {
        r"impute_median": ("impute", "median", {}),
        r"impute_mean": ("impute", "mean", {}),
        r"impute_mode": ("impute", "mode", {}),
        r"impute_zero": ("impute", "constant", {"fill_value": 0}),
        r"impute_constant_(.+)": ("impute", "constant", {}),
        r"cap_outliers_(\d+)": ("cap_outliers", "", {}),
        r"remove_outliers_iqr": ("remove_outliers", "iqr", {}),
        r"drop_rare_(\d+)": ("drop_rare", "", {}),
        r"drop_nulls": ("drop_nulls", "", {}),
    }

    TRANSFORM_PATTERNS = {
        r"standard_scale": ("scale", "standard", {}),
        r"minmax_scale": ("scale", "minmax", {}),
        r"robust_scale": ("scale", "robust", {}),
        r"log_transform": ("transform", "log1p", {}),
        r"sqrt_transform": ("transform", "sqrt", {}),
        r"power_transform": ("transform", "yeo_johnson", {}),
        r"onehot_encode": ("encode", "onehot", {}),
        r"label_encode": ("encode", "label", {}),
        r"ordinal_encode": ("encode", "ordinal", {}),
        r"extract_month": ("datetime_extract", "month", {}),
        r"extract_dayofweek": ("datetime_extract", "dayofweek", {}),
        r"extract_day$": ("datetime_extract", "day", {}),
        r"extract_hour": ("datetime_extract", "hour", {}),
        r"extract_year": ("datetime_extract", "year", {}),
        r"days_since": ("datetime_extract", "days_since", {}),
    }

    def parse_cleaning(self, recommendation: str) -> Optional[CleanAction]:
        for pattern, (action_type, strategy, params) in self.CLEANING_PATTERNS.items():
            match = re.match(pattern, recommendation)
            if match:
                result_params = params.copy()
                if match.groups():
                    if action_type == "cap_outliers":
                        result_params["percentile"] = int(match.group(1))
                    elif action_type == "drop_rare":
                        result_params["threshold_percent"] = int(match.group(1))
                    elif strategy == "constant" and "fill_value" not in result_params:
                        result_params["fill_value"] = match.group(1)
                return CleanAction(action_type=action_type, strategy=strategy, params=result_params)
        return None

    def parse_transform(self, recommendation: str) -> Optional[TransformAction]:
        for pattern, (action_type, method, params) in self.TRANSFORM_PATTERNS.items():
            if re.match(pattern, recommendation):
                return TransformAction(action_type=action_type, method=method, params=params.copy())
        return None


@dataclass
class MLflowConfig:
    tracking_uri: str = "./mlruns"
    experiment_name: str = "ml_pipeline"
    run_name: Optional[str] = None
    log_data_quality: bool = True
    log_transformations: bool = True
    log_feature_importance: bool = True
    nested_runs: bool = True
    model_name: Optional[str] = None


class MLflowPipelineGenerator:
    def __init__(
        self,
        mlflow_config: Optional[MLflowConfig] = None,
        output_dir: str = "./generated_pipelines",
    ):
        self.mlflow_config = mlflow_config or MLflowConfig()
        self.output_dir = output_dir
        self._parser = RecommendationParser()

    def generate_pipeline(self, findings: ExplorationFindings) -> str:
        sections = [
            self._generate_docstring(findings),
            self._generate_imports(),
            self._generate_mlflow_setup(),
        ]

        if self.mlflow_config.log_data_quality:
            sections.append(self._generate_data_quality_logging())

        sections.extend([
            self.generate_cleaning_functions(findings),
            self.generate_transform_functions(findings),
            self.generate_feature_engineering(findings),
            self.generate_model_training(findings),
            self.generate_monitoring(findings),
            self._generate_main(findings),
        ])
        return "\n\n".join(sections)

    def _generate_docstring(self, findings: ExplorationFindings) -> str:
        return f'''"""
MLflow-tracked ML Pipeline
Generated from exploration findings

Source: {findings.source_path}
Target: {findings.target_column or 'Not specified'}
Rows: {findings.row_count:,}
Features: {findings.column_count}
"""'''

    def _generate_imports(self) -> str:
        return """import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Any

import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)"""

    def _generate_mlflow_setup(self) -> str:
        return f'''
MLFLOW_TRACKING_URI = "{self.mlflow_config.tracking_uri}"
EXPERIMENT_NAME = "{self.mlflow_config.experiment_name}"


def setup_mlflow():
    """Initialize MLflow tracking."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    return mlflow.get_experiment_by_name(EXPERIMENT_NAME)'''

    def _generate_data_quality_logging(self) -> str:
        return '''
def log_data_quality_metrics(df: pd.DataFrame, prefix: str = "data"):
    """Log data quality metrics to MLflow."""
    metrics = {
        f"{prefix}_rows": len(df),
        f"{prefix}_columns": len(df.columns),
        f"{prefix}_memory_mb": df.memory_usage(deep=True).sum() / 1024 / 1024,
    }

    for col in df.columns:
        null_pct = df[col].isna().mean() * 100
        metrics[f"{prefix}_null_pct_{col}"] = null_pct

    mlflow.log_metrics(metrics)
    return metrics'''

    def generate_cleaning_functions(self, findings: ExplorationFindings) -> str:
        cleaning_steps = self._build_cleaning_steps(findings)

        code_lines = [
            "def clean_data(df: pd.DataFrame) -> pd.DataFrame:",
            '    """Apply cleaning transformations based on exploration findings."""',
            "    df = df.copy()",
            "    cleaning_stats = {}",
            "",
        ]

        if not cleaning_steps:
            code_lines.append("    # No cleaning recommendations found")
        else:
            for col_name, actions in cleaning_steps.items():
                for action in actions:
                    code_lines.extend(self._action_to_cleaning_code(col_name, action))

        code_lines.extend([
            "",
            "    mlflow.log_params({f'cleaned_{k}': v for k, v in cleaning_stats.items()})",
            "    return df",
        ])

        return "\n".join(code_lines)

    def _build_cleaning_steps(self, findings: ExplorationFindings) -> Dict[str, List[CleanAction]]:
        steps = {}
        for col_name, col_finding in findings.columns.items():
            if col_finding.inferred_type in (ColumnType.IDENTIFIER, ColumnType.TARGET):
                continue

            col_actions = []
            for rec in col_finding.cleaning_recommendations:
                action = self._parser.parse_cleaning(rec)
                if action:
                    col_actions.append(action)

            if col_actions:
                steps[col_name] = col_actions

        return steps

    def _action_to_cleaning_code(self, col_name: str, action: CleanAction) -> List[str]:
        lines = []

        if action.action_type == "impute":
            if action.strategy == "median":
                lines.extend([
                    f"    # Impute {col_name} with median",
                    f"    if df['{col_name}'].isna().any():",
                    f"        median_val = df['{col_name}'].median()",
                    f"        cleaning_stats['{col_name}_imputed'] = df['{col_name}'].isna().sum()",
                    f"        df['{col_name}'] = df['{col_name}'].fillna(median_val)",
                    "",
                ])
            elif action.strategy == "mode":
                lines.extend([
                    f"    # Impute {col_name} with mode",
                    f"    if df['{col_name}'].isna().any():",
                    f"        mode_val = df['{col_name}'].mode().iloc[0] if not df['{col_name}'].mode().empty else None",
                    "        if mode_val is not None:",
                    f"            cleaning_stats['{col_name}_imputed'] = df['{col_name}'].isna().sum()",
                    f"            df['{col_name}'] = df['{col_name}'].fillna(mode_val)",
                    "",
                ])
            elif action.strategy == "constant":
                fill_value = action.params.get("fill_value", 0)
                lines.extend([
                    f"    # Impute {col_name} with constant",
                    f"    if df['{col_name}'].isna().any():",
                    f"        cleaning_stats['{col_name}_imputed'] = df['{col_name}'].isna().sum()",
                    f"        df['{col_name}'] = df['{col_name}'].fillna({repr(fill_value)})",
                    "",
                ])

        elif action.action_type == "cap_outliers":
            percentile = action.params.get("percentile", 99)
            lines.extend([
                f"    # Cap outliers in {col_name} at {percentile}th percentile",
                f"    lower = df['{col_name}'].quantile({(100 - percentile) / 100})",
                f"    upper = df['{col_name}'].quantile({percentile / 100})",
                f"    outliers = ((df['{col_name}'] < lower) | (df['{col_name}'] > upper)).sum()",
                f"    cleaning_stats['{col_name}_outliers_capped'] = outliers",
                f"    df['{col_name}'] = df['{col_name}'].clip(lower, upper)",
                "",
            ])

        elif action.action_type == "drop_rare":
            threshold = action.params.get("threshold_percent", 5)
            lines.extend([
                f"    # Drop rare categories in {col_name} (< {threshold}%)",
                f"    value_counts = df['{col_name}'].value_counts(normalize=True)",
                f"    rare_values = value_counts[value_counts < {threshold / 100}].index",
                "    if len(rare_values) > 0:",
                f"        cleaning_stats['{col_name}_rare_dropped'] = len(rare_values)",
                f"        df.loc[df['{col_name}'].isin(rare_values), '{col_name}'] = df['{col_name}'].mode().iloc[0]",
                "",
            ])

        return lines

    def generate_transform_functions(self, findings: ExplorationFindings) -> str:
        self._get_columns_by_type(findings,
            [ColumnType.NUMERIC_CONTINUOUS, ColumnType.NUMERIC_DISCRETE])
        self._get_columns_by_type(findings,
            [ColumnType.CATEGORICAL_NOMINAL, ColumnType.CATEGORICAL_ORDINAL])

        transform_actions = self._build_transform_actions(findings)

        code_lines = [
            "def apply_transforms(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:",
            '    """Apply transformations based on exploration recommendations."""',
            "    df = df.copy()",
            "    transformers = {}",
            "",
        ]

        # Log transform for skewed columns
        log_cols = [col for col, actions in transform_actions.items()
                   if any(a.method == "log1p" for a in actions)]
        if log_cols:
            for col in log_cols:
                code_lines.extend([
                    f"    # Log transform {col} (recommended for skewness)",
                    f"    df['{col}_log'] = np.log1p(df['{col}'].clip(lower=0))",
                    f"    transformers['{col}_log_transform'] = True",
                    "",
                ])

        # Standard scaling
        scale_standard = [col for col, actions in transform_actions.items()
                        if any(a.action_type == "scale" and a.method == "standard" for a in actions)]
        if scale_standard:
            code_lines.extend([
                "    # Standard scaling",
                f"    standard_cols = {scale_standard}",
                "    if standard_cols:",
                "        scaler = StandardScaler()",
                "        df[standard_cols] = scaler.fit_transform(df[standard_cols])",
                "        transformers['standard_scaler'] = {'columns': standard_cols}",
                "",
            ])

        # MinMax scaling
        scale_minmax = [col for col, actions in transform_actions.items()
                       if any(a.action_type == "scale" and a.method == "minmax" for a in actions)]
        if scale_minmax:
            code_lines.extend([
                "    # MinMax scaling",
                f"    minmax_cols = {scale_minmax}",
                "    if minmax_cols:",
                "        minmax_scaler = MinMaxScaler()",
                "        df[minmax_cols] = minmax_scaler.fit_transform(df[minmax_cols])",
                "        transformers['minmax_scaler'] = {'columns': minmax_cols}",
                "",
            ])

        # One-hot encoding
        onehot_cols = [col for col, actions in transform_actions.items()
                      if any(a.action_type == "encode" and a.method == "onehot" for a in actions)]
        if onehot_cols:
            code_lines.extend([
                "    # One-hot encoding",
                f"    onehot_cols = {onehot_cols}",
                "    for col in onehot_cols:",
                "        dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)",
                "        df = pd.concat([df.drop(columns=[col]), dummies], axis=1)",
                "        transformers[f'{col}_onehot'] = list(dummies.columns)",
                "",
            ])

        # Label encoding
        label_cols = [col for col, actions in transform_actions.items()
                     if any(a.action_type == "encode" and a.method == "label" for a in actions)]
        if label_cols:
            code_lines.extend([
                "    # Label encoding",
                f"    label_cols = {label_cols}",
                "    label_encoders = {{}}",
                "    for col in label_cols:",
                "        le = LabelEncoder()",
                "        df[col] = le.fit_transform(df[col].astype(str))",
                "        label_encoders[col] = le",
                "    transformers['label_encoders'] = label_encoders",
                "",
            ])

        code_lines.extend([
            "    mlflow.log_params({f'transform_{k}': str(v)[:250] for k, v in transformers.items()})",
            "    return df, transformers",
        ])

        return "\n".join(code_lines)

    def _build_transform_actions(self, findings: ExplorationFindings) -> Dict[str, List[TransformAction]]:
        actions = {}
        for col_name, col_finding in findings.columns.items():
            if col_finding.inferred_type in (ColumnType.IDENTIFIER, ColumnType.TARGET):
                continue

            col_actions = []
            for rec in col_finding.transformation_recommendations:
                action = self._parser.parse_transform(rec)
                if action:
                    col_actions.append(action)

            if col_actions:
                actions[col_name] = col_actions

        return actions

    def generate_feature_engineering(self, findings: ExplorationFindings) -> str:
        datetime_cols = self._get_columns_by_type(findings, [ColumnType.DATETIME])
        transform_actions = self._build_transform_actions(findings)

        code_lines = [
            "def engineer_features(df: pd.DataFrame) -> pd.DataFrame:",
            '    """Engineer features based on exploration findings."""',
            "    df = df.copy()",
            "    new_features = []",
            "",
        ]

        # Datetime feature extraction
        for col_name in datetime_cols:
            actions = transform_actions.get(col_name, [])
            extract_types = [a.method for a in actions if a.action_type == "datetime_extract"]

            if not extract_types:
                extract_types = ["month", "dayofweek", "days_since"]

            code_lines.extend([
                f"    # Datetime features from {col_name}",
                f"    if '{col_name}' in df.columns:",
                f"        df['{col_name}'] = pd.to_datetime(df['{col_name}'], errors='coerce')",
                "",
            ])

            for ext_type in extract_types:
                if ext_type == "month":
                    code_lines.append(f"        df['{col_name}_month'] = df['{col_name}'].dt.month")
                    code_lines.append(f"        new_features.append('{col_name}_month')")
                elif ext_type == "day":
                    code_lines.append(f"        df['{col_name}_day'] = df['{col_name}'].dt.day")
                    code_lines.append(f"        new_features.append('{col_name}_day')")
                elif ext_type == "dayofweek":
                    code_lines.append(f"        df['{col_name}_dayofweek'] = df['{col_name}'].dt.dayofweek")
                    code_lines.append(f"        new_features.append('{col_name}_dayofweek')")
                elif ext_type == "hour":
                    code_lines.append(f"        df['{col_name}_hour'] = df['{col_name}'].dt.hour")
                    code_lines.append(f"        new_features.append('{col_name}_hour')")
                elif ext_type == "year":
                    code_lines.append(f"        df['{col_name}_year'] = df['{col_name}'].dt.year")
                    code_lines.append(f"        new_features.append('{col_name}_year')")
                elif ext_type == "days_since":
                    code_lines.extend([
                        f"        reference_date = df['{col_name}'].max()",
                        f"        df['{col_name}_days_since'] = (reference_date - df['{col_name}']).dt.days",
                        f"        new_features.append('{col_name}_days_since')",
                    ])

            code_lines.append("")

        code_lines.extend([
            "    if new_features:",
            "        mlflow.log_param('engineered_features', new_features)",
            "    return df",
        ])

        return "\n".join(code_lines)

    def generate_model_training(self, findings: ExplorationFindings) -> str:
        target = findings.target_column or "target"
        identifier_cols = findings.identifier_columns or []
        datetime_cols = findings.datetime_columns or []
        exclude_cols = set(identifier_cols + datetime_cols + [target])

        return f'''
def train_model(
    df: pd.DataFrame,
    target_column: str = "{target}",
    test_size: float = 0.2,
    val_size: float = 0.1,
) -> Dict[str, Any]:
    """Train model with comprehensive MLflow tracking."""

    # Exclude non-feature columns
    exclude_cols = {exclude_cols}
    feature_cols = [col for col in df.columns if col not in exclude_cols and col != target_column]

    # Handle non-numeric columns
    X = df[feature_cols].copy()
    for col in X.select_dtypes(include=['object', 'category']).columns:
        X[col] = pd.factorize(X[col])[0]
    X = X.fillna(0)

    y = df[target_column]

    # Split: train/validation/test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size/(1-test_size), random_state=42, stratify=y_temp
    )

    mlflow.log_params({{
        "train_samples": len(X_train),
        "validation_samples": len(X_val),
        "test_samples": len(X_test),
        "feature_count": len(feature_cols),
        "test_size": test_size,
        "val_size": val_size,
    }})

    # Train models
    models = {{
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "gradient_boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    }}

    results = {{}}
    best_model = None
    best_auc = 0

    for name, model in models.items():
        with mlflow.start_run(run_name=name, nested=True):
            # Train
            model.fit(X_train, y_train)

            # Validation predictions
            y_val_pred = model.predict(X_val)
            y_val_proba = model.predict_proba(X_val)[:, 1] if hasattr(model, "predict_proba") else y_val_pred

            # Test predictions
            y_test_pred = model.predict(X_test)
            y_test_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_test_pred

            # Calculate metrics
            val_metrics = {{
                "val_accuracy": accuracy_score(y_val, y_val_pred),
                "val_precision": precision_score(y_val, y_val_pred, average="weighted", zero_division=0),
                "val_recall": recall_score(y_val, y_val_pred, average="weighted", zero_division=0),
                "val_f1": f1_score(y_val, y_val_pred, average="weighted", zero_division=0),
                "val_roc_auc": roc_auc_score(y_val, y_val_proba) if len(np.unique(y_val)) > 1 else 0,
            }}

            test_metrics = {{
                "test_accuracy": accuracy_score(y_test, y_test_pred),
                "test_precision": precision_score(y_test, y_test_pred, average="weighted", zero_division=0),
                "test_recall": recall_score(y_test, y_test_pred, average="weighted", zero_division=0),
                "test_f1": f1_score(y_test, y_test_pred, average="weighted", zero_division=0),
                "test_roc_auc": roc_auc_score(y_test, y_test_proba) if len(np.unique(y_test)) > 1 else 0,
            }}

            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="roc_auc")
            cv_metrics = {{
                "cv_roc_auc_mean": cv_scores.mean(),
                "cv_roc_auc_std": cv_scores.std(),
            }}

            # Log everything
            mlflow.log_params(model.get_params())
            mlflow.log_metrics({{**val_metrics, **test_metrics, **cv_metrics}})
            mlflow.sklearn.log_model(model, f"model_{{name}}")

            results[name] = {{
                "model": model,
                "val_metrics": val_metrics,
                "test_metrics": test_metrics,
                "cv_metrics": cv_metrics,
            }}

            if val_metrics["val_roc_auc"] > best_auc:
                best_auc = val_metrics["val_roc_auc"]
                best_model = name

    mlflow.log_param("best_model", best_model)
    mlflow.log_metric("best_val_roc_auc", best_auc)

    return {{"results": results, "best_model": best_model}}'''

    def generate_monitoring(self, findings: ExplorationFindings) -> str:
        return '''
def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    """Evaluate model and log monitoring metrics."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba) if len(np.unique(y_test)) > 1 else 0,
    }

    mlflow.log_metrics({f"monitor_{k}": v for k, v in metrics.items()})

    return metrics'''

    def _generate_main(self, findings: ExplorationFindings) -> str:
        source_path = findings.source_path
        load_func = "pd.read_csv" if findings.source_format == "csv" else "pd.read_parquet"

        main_body = f'''
def main():
    """Run the complete ML pipeline with MLflow tracking."""
    setup_mlflow()

    with mlflow.start_run(run_name="full_pipeline"):
        # Load data
        print("Loading data...")
        df = {load_func}("{source_path}")'''

        if self.mlflow_config.log_data_quality:
            main_body += "\n        log_data_quality_metrics(df, prefix='raw')"

        main_body += '''

        # Clean data
        print("Cleaning data...")
        df = clean_data(df)'''

        if self.mlflow_config.log_data_quality:
            main_body += "\n        log_data_quality_metrics(df, prefix='cleaned')"

        main_body += '''

        # Apply transformations
        print("Applying transformations...")
        df, transformers = apply_transforms(df)

        # Engineer features
        print("Engineering features...")
        df = engineer_features(df)'''

        if self.mlflow_config.log_data_quality:
            main_body += "\n        log_data_quality_metrics(df, prefix='final')"

        main_body += '''

        # Train models
        print("Training models...")
        results = train_model(df)

        print(f"\\nBest model: {results['best_model']}")
        print("Pipeline complete! Check MLflow UI for results.")

        return results


if __name__ == "__main__":
    main()'''

        return main_body

    def _get_columns_by_type(
        self,
        findings: ExplorationFindings,
        col_types: List[ColumnType],
    ) -> List[str]:
        return [
            name for name, col in findings.columns.items()
            if col.inferred_type in col_types
        ]

    def generate_all(self, findings: ExplorationFindings) -> Dict[str, str]:
        return {
            "pipeline.py": self.generate_pipeline(findings),
            "requirements.txt": self._generate_requirements(),
        }

    def _generate_requirements(self) -> str:
        return """pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
mlflow>=2.10.0
scipy>=1.11.0
matplotlib>=3.7.0
seaborn>=0.12.0
"""

    def save_all(self, findings: ExplorationFindings) -> List[str]:
        files = self.generate_all(findings)
        output_path = Path(self.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        saved = []
        for filename, content in files.items():
            file_path = output_path / filename
            file_path.write_text(content)
            saved.append(filename)

        return saved
