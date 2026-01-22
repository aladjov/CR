"""Transformer persistence and consistent application for training/scoring.

Ensures that the same transformations (scaling, encoding) applied during training
are replicated exactly during scoring to prevent data leakage and prediction errors.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import json
import tempfile
import os

import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, LabelEncoder


@dataclass
class TransformerManifest:
    numeric_columns: List[str] = field(default_factory=list)
    categorical_columns: List[str] = field(default_factory=list)
    scaler_type: Optional[str] = None
    encoder_type: str = "label"
    feature_order: List[str] = field(default_factory=list)
    created_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {"numeric_columns": self.numeric_columns, "categorical_columns": self.categorical_columns,
                "scaler_type": self.scaler_type, "encoder_type": self.encoder_type,
                "feature_order": self.feature_order, "created_at": self.created_at}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TransformerManifest":
        return cls(numeric_columns=data.get("numeric_columns", []),
                   categorical_columns=data.get("categorical_columns", []),
                   scaler_type=data.get("scaler_type"), encoder_type=data.get("encoder_type", "label"),
                   feature_order=data.get("feature_order", []), created_at=data.get("created_at"))


@dataclass
class TransformerBundle:
    scaler: Optional[Any] = None
    encoders: Dict[str, LabelEncoder] = field(default_factory=dict)
    manifest: TransformerManifest = field(default_factory=TransformerManifest)

    def to_dict(self) -> Dict[str, Any]:
        return {"numeric_scaler": self.scaler, "label_encoders": self.encoders,
                "manifest": self.manifest.to_dict()}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TransformerBundle":
        return cls(scaler=data.get("numeric_scaler"), encoders=data.get("label_encoders", {}),
                   manifest=TransformerManifest.from_dict(data.get("manifest", {})))


class TransformerManager:
    """Manages transformer persistence and application for training/scoring consistency.

    Usage for Training:
        manager = TransformerManager()
        df_transformed = manager.fit_transform(df, numeric_cols, categorical_cols)
        manager.save("./output/transformers/transformers.joblib")
        manager.log_to_mlflow(run_id)

    Usage for Scoring:
        manager = TransformerManager.load_from_mlflow(run_id)
        # OR: manager = TransformerManager.load("./output/transformers/transformers.joblib")
        df_transformed = manager.transform(df)
    """

    def __init__(self, scaler_type: str = "standard"):
        """Initialize transformer manager.

        Args:
            scaler_type: Type of scaler to use ("standard", "robust", "minmax")
        """
        self._scaler_type = scaler_type
        self._bundle = TransformerBundle()
        self._is_fitted = False

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    @property
    def manifest(self) -> TransformerManifest:
        return self._bundle.manifest

    def fit_transform(self, df: pd.DataFrame,
                      numeric_columns: Optional[List[str]] = None,
                      categorical_columns: Optional[List[str]] = None,
                      exclude_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Fit transformers on training data and transform it.

        Args:
            df: Training DataFrame
            numeric_columns: Columns to scale (auto-detected if None)
            categorical_columns: Columns to encode (auto-detected if None)
            exclude_columns: Columns to exclude from transformation

        Returns:
            Transformed DataFrame
        """
        from datetime import datetime

        df = df.copy()
        exclude = set(exclude_columns or [])

        numeric_columns = self._resolve_numeric_columns(df, numeric_columns, exclude)
        categorical_columns = self._resolve_categorical_columns(df, categorical_columns, exclude)

        self._fit_numeric_scaler(df, numeric_columns)
        self._fit_categorical_encoders(df, categorical_columns)
        self._build_manifest(df, numeric_columns, categorical_columns, exclude, datetime.now().isoformat())
        self._is_fitted = True

        return df

    def _resolve_numeric_columns(self, df: pd.DataFrame, columns: Optional[List[str]], exclude: set) -> List[str]:
        if columns is None:
            columns = [c for c in df.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns
                       if c not in exclude]
        return [c for c in columns if c in df.columns and c not in exclude]

    def _resolve_categorical_columns(self, df: pd.DataFrame, columns: Optional[List[str]], exclude: set) -> List[str]:
        if columns is None:
            columns = [c for c in df.select_dtypes(include=["object", "category"]).columns if c not in exclude]
        return [c for c in columns if c in df.columns and c not in exclude]

    def _fit_numeric_scaler(self, df: pd.DataFrame, numeric_columns: List[str]) -> None:
        if numeric_columns:
            scaler = self._create_scaler()
            df[numeric_columns] = scaler.fit_transform(df[numeric_columns].fillna(0))
            self._bundle.scaler = scaler

    def _fit_categorical_encoders(self, df: pd.DataFrame, categorical_columns: List[str]) -> None:
        encoders = {}
        for col in categorical_columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
        self._bundle.encoders = encoders

    def _build_manifest(self, df: pd.DataFrame, numeric_columns: List[str],
                        categorical_columns: List[str], exclude: set, created_at: str) -> None:
        feature_order = [c for c in df.columns if c not in exclude]
        self._bundle.manifest = TransformerManifest(
            numeric_columns=numeric_columns, categorical_columns=categorical_columns,
            scaler_type=self._scaler_type, encoder_type="label",
            feature_order=feature_order, created_at=created_at)

    def transform(self, df: pd.DataFrame,
                  exclude_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Apply fitted transformers to new data (for scoring).

        Args:
            df: DataFrame to transform
            exclude_columns: Columns to exclude (e.g., entity_key, target)

        Returns:
            Transformed DataFrame with same feature order as training
        """
        if not self._is_fitted:
            raise ValueError("TransformerManager not fitted. Call fit_transform() or load().")

        df = df.copy()
        exclude = set(exclude_columns or [])
        manifest = self._bundle.manifest

        self._apply_numeric_scaling(df, manifest)
        self._apply_categorical_encoding(df, manifest)

        feature_cols = [c for c in manifest.feature_order if c not in exclude and c in df.columns]
        return df[feature_cols]

    def _apply_numeric_scaling(self, df: pd.DataFrame, manifest: TransformerManifest) -> None:
        if self._bundle.scaler is None or not manifest.numeric_columns:
            return
        present_cols = [c for c in manifest.numeric_columns if c in df.columns]
        missing_cols = [c for c in manifest.numeric_columns if c not in df.columns]

        if present_cols:
            col_indices = {col: i for i, col in enumerate(manifest.numeric_columns)}
            temp_arr = np.zeros((len(df), len(manifest.numeric_columns)))
            for col in present_cols:
                temp_arr[:, col_indices[col]] = df[col].fillna(0).values
            transformed = self._bundle.scaler.transform(temp_arr)
            for col in present_cols:
                df[col] = transformed[:, col_indices[col]]

        for col in missing_cols:
            df[col] = 0.0

    def _apply_categorical_encoding(self, df: pd.DataFrame, manifest: TransformerManifest) -> None:
        for col, encoder in self._bundle.encoders.items():
            if col in df.columns:
                df[col] = df[col].astype(str).apply(lambda x, enc=encoder: self._safe_encode(enc, x))
            elif col in manifest.categorical_columns:
                df[col] = self._safe_encode(encoder, "")

    def _safe_encode(self, encoder: LabelEncoder, value: str) -> int:
        try:
            return int(encoder.transform([value])[0])
        except ValueError:
            return 0

    def _create_scaler(self):
        scalers = {"standard": StandardScaler, "robust": RobustScaler, "minmax": MinMaxScaler}
        return scalers.get(self._scaler_type, StandardScaler)()

    def save(self, path: Union[str, Path]) -> None:
        if not self._is_fitted:
            raise ValueError("Cannot save unfitted TransformerManager")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._bundle.to_dict(), path)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "TransformerManager":
        data = joblib.load(path)
        manager = cls()
        manager._bundle = TransformerBundle.from_dict(data)
        manager._is_fitted = True
        manager._scaler_type = manager._bundle.manifest.scaler_type or "standard"
        return manager

    def log_to_mlflow(self, run_id: Optional[str] = None, artifact_path: str = "transformers") -> None:
        import mlflow

        if not self._is_fitted:
            raise ValueError("Cannot log unfitted TransformerManager")

        with tempfile.TemporaryDirectory() as tmp_dir:
            bundle_path = Path(tmp_dir) / "transformers.joblib"
            joblib.dump(self._bundle.to_dict(), bundle_path)

            manifest_path = Path(tmp_dir) / "transformer_manifest.json"
            with open(manifest_path, "w") as f:
                json.dump(self._bundle.manifest.to_dict(), f, indent=2)

            self._log_artifacts_to_mlflow(run_id, bundle_path, manifest_path, artifact_path)

    def _log_artifacts_to_mlflow(self, run_id: Optional[str], bundle_path: Path,
                                  manifest_path: Path, artifact_path: str) -> None:
        import mlflow
        if run_id:
            client = mlflow.tracking.MlflowClient()
            client.log_artifact(run_id, str(bundle_path), artifact_path)
            client.log_artifact(run_id, str(manifest_path), artifact_path)
        else:
            mlflow.log_artifact(str(bundle_path), artifact_path)
            mlflow.log_artifact(str(manifest_path), artifact_path)

    @classmethod
    def load_from_mlflow(cls, run_id: str, artifact_path: str = "transformers",
                         tracking_uri: Optional[str] = None) -> "TransformerManager":
        import mlflow
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        client = mlflow.tracking.MlflowClient()
        with tempfile.TemporaryDirectory() as tmp_dir:
            local_path = client.download_artifacts(run_id, f"{artifact_path}/transformers.joblib", tmp_dir)
            return cls.load(local_path)

    @classmethod
    def load_from_mlflow_by_experiment(cls, experiment_name: str, artifact_path: str = "transformers",
                                        tracking_uri: Optional[str] = None,
                                        run_name_filter: Optional[str] = None) -> "TransformerManager":
        import mlflow
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name(experiment_name)
        if not experiment:
            raise ValueError(f"Experiment {experiment_name} not found")

        filter_str = f'tags.mlflow.runName = "{run_name_filter}"' if run_name_filter else ""
        runs = client.search_runs(experiment_ids=[experiment.experiment_id], filter_string=filter_str,
                                  order_by=["start_time DESC"], max_results=1)
        if not runs:
            raise ValueError(f"No runs found in experiment {experiment_name}")

        return cls.load_from_mlflow(runs[0].info.run_id, artifact_path, tracking_uri)
