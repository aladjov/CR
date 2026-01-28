import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
import yaml


def _extract_transformer_params(transformer: Any) -> Dict[str, Any]:
    params = {}
    for attr in ["mean_", "scale_", "var_", "data_min_", "data_max_", "data_range_",
                 "classes_", "n_features_in_", "components_", "explained_variance_ratio_",
                 "explained_variance_", "singular_values_", "n_components_"]:
        if hasattr(transformer, attr):
            val = getattr(transformer, attr)
            if isinstance(val, np.ndarray):
                params[attr] = val.tolist()
            else:
                params[attr] = val
    return params


def _compute_params_hash(params: Dict[str, Any]) -> str:
    serialized = str(sorted(params.items()))
    return hashlib.sha256(serialized.encode()).hexdigest()[:16]


@dataclass
class FitArtifact:
    artifact_id: str
    artifact_type: str
    target_column: str
    transformer_class: str
    fit_timestamp: str
    fit_data_hash: str
    parameters: Dict[str, Any]
    file_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "artifact_type": self.artifact_type,
            "target_column": self.target_column,
            "transformer_class": self.transformer_class,
            "fit_timestamp": self.fit_timestamp,
            "fit_data_hash": self.fit_data_hash,
            "parameters": self.parameters,
            "file_path": self.file_path,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FitArtifact":
        return cls(
            artifact_id=data["artifact_id"],
            artifact_type=data["artifact_type"],
            target_column=data["target_column"],
            transformer_class=data["transformer_class"],
            fit_timestamp=data["fit_timestamp"],
            fit_data_hash=data["fit_data_hash"],
            parameters=data.get("parameters", {}),
            file_path=data.get("file_path"),
        )


@dataclass
class FitArtifactRegistry:
    artifacts_dir: Path
    _artifacts: Dict[str, FitArtifact] = field(default_factory=dict, repr=False)

    ARTIFACT_SUBDIRS = {"scaler": "scalers", "encoder": "encoders", "reducer": "reducers"}

    def __post_init__(self):
        self.artifacts_dir = Path(self.artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        for subdir in self.ARTIFACT_SUBDIRS.values():
            (self.artifacts_dir / subdir).mkdir(exist_ok=True)

    def register(self, artifact_type: str, target_column: str, transformer: Any,
                 artifact_id: Optional[str] = None, overwrite: bool = False) -> str:
        if artifact_type not in self.ARTIFACT_SUBDIRS:
            raise ValueError(f"Unknown artifact type: {artifact_type}. Must be one of {list(self.ARTIFACT_SUBDIRS.keys())}")
        generated_id = artifact_id or f"{target_column}_{artifact_type}"
        if generated_id in self._artifacts and not overwrite:
            raise ValueError(f"Artifact '{generated_id}' already exists. Use overwrite=True to replace.")
        params = _extract_transformer_params(transformer)
        subdir = self.ARTIFACT_SUBDIRS[artifact_type]
        file_path = f"{subdir}/{generated_id}.pkl"
        full_path = self.artifacts_dir / file_path
        joblib.dump(transformer, full_path)
        artifact = FitArtifact(
            artifact_id=generated_id,
            artifact_type=artifact_type,
            target_column=target_column,
            transformer_class=type(transformer).__name__,
            fit_timestamp=datetime.now().isoformat(),
            fit_data_hash=_compute_params_hash(params),
            parameters=params,
            file_path=file_path,
        )
        self._artifacts[generated_id] = artifact
        return generated_id

    def load(self, artifact_id: str) -> Any:
        if artifact_id not in self._artifacts:
            raise KeyError(f"Artifact '{artifact_id}' not found in registry")
        artifact = self._artifacts[artifact_id]
        full_path = self.artifacts_dir / artifact.file_path
        return joblib.load(full_path)

    def get_manifest(self) -> Dict[str, FitArtifact]:
        return self._artifacts.copy()

    def has_artifact(self, artifact_id: str) -> bool:
        return artifact_id in self._artifacts

    def get_artifact_info(self, artifact_id: str) -> FitArtifact:
        if artifact_id not in self._artifacts:
            raise KeyError(f"Artifact '{artifact_id}' not found")
        return self._artifacts[artifact_id]

    def save_manifest(self) -> None:
        manifest_data = {
            "version": "1.0",
            "created_at": datetime.now().isoformat(),
            "artifacts": {aid: a.to_dict() for aid, a in self._artifacts.items()},
        }
        manifest_path = self.artifacts_dir / "manifest.yaml"
        with open(manifest_path, "w") as f:
            yaml.dump(manifest_data, f, default_flow_style=False, sort_keys=False)

    @classmethod
    def load_manifest(cls, manifest_path: Path) -> "FitArtifactRegistry":
        manifest_path = Path(manifest_path)
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")
        with open(manifest_path) as f:
            data = yaml.safe_load(f)
        artifacts_dir = manifest_path.parent
        registry = cls(artifacts_dir=artifacts_dir)
        registry._artifacts = {}
        for aid, artifact_data in data.get("artifacts", {}).items():
            registry._artifacts[aid] = FitArtifact.from_dict(artifact_data)
        return registry
