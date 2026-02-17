"""Persistence for fitted transformers.

Drop-in replacement for the ``FitArtifactRegistry`` class that was
previously inlined in ``gold.py.j2`` and ``run_scoring.py.j2`` templates.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import yaml


class ArtifactStore:
    """Manages persistence of fitted transformers (scalers, encoders, etc.)."""

    def __init__(self, artifacts_dir: str | Path):
        self._dir = Path(artifacts_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._manifest: dict = {}

    def register(self, artifact_type: str, target_column: str, transformer) -> None:
        artifact_id = f"{target_column}_{artifact_type}"
        path = self._dir / f"{artifact_id}.joblib"
        joblib.dump(transformer, path)
        self._manifest[artifact_id] = {
            "type": artifact_type,
            "column": target_column,
            "path": str(path),
        }

    def save_manifest(self) -> None:
        manifest_path = self._dir / "manifest.yaml"
        with manifest_path.open("w") as f:
            yaml.dump(self._manifest, f)

    def load(self, artifact_id: str):
        if artifact_id not in self._manifest:
            raise KeyError(f"Artifact {artifact_id} not found")
        return joblib.load(self._manifest[artifact_id]["path"])

    def has(self, artifact_id: str) -> bool:
        return artifact_id in self._manifest

    @classmethod
    def from_manifest(cls, manifest_path) -> ArtifactStore:
        from pathlib import Path as _Path
        p = manifest_path if hasattr(manifest_path, "open") else _Path(str(manifest_path))
        store = cls(str(p.parent))
        with p.open("r") as f:
            store._manifest = yaml.safe_load(f) or {}
        return store
