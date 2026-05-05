from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

from customer_retention.core.compat import load_spark_table
from customer_retention.core.compat.remote_path import RemotePath, make_path

_RUN_POINTER_FILENAME = ".cr_active_run.json"


@dataclass
class RunNamespace:
    root: Union[Path, RemotePath]
    run_id: str
    snapshot_grid: Optional[Any] = field(default=None, repr=False)
    semantics: Optional[dict] = field(default=None, repr=False)
    dataset_registry: Optional[dict] = field(default=None, repr=False)
    scd_history_sources: Optional[dict] = field(default=None, repr=False)
    original_datasets: Optional[dict] = field(default=None, repr=False)

    def load_table(self, source: str) -> Any:
        return load_spark_table(source)

    @property
    def run_dir(self) -> Path:
        return self.root / "runs" / self.run_id

    @property
    def datasets_dir(self) -> Path:
        return self.run_dir / "datasets"

    @property
    def data_dir(self) -> Path:
        return self.run_dir / "data"

    @property
    def landing_dir(self) -> Path:
        return self.data_dir / "landing"

    @property
    def bronze_dir(self) -> Path:
        return self.data_dir / "bronze"

    @property
    def silver_dir(self) -> Path:
        return self.data_dir / "silver"

    @property
    def gold_dir(self) -> Path:
        return self.data_dir / "gold"

    def dataset_dir(self, name: str) -> Path:
        return self.datasets_dir / name

    def landing_table_dir(self, name: str) -> Path:
        return self.landing_dir / name

    def bronze_table_dir(self, name: str) -> Path:
        return self.bronze_dir / f"{name}_aggregated"

    def gold_table_dir(self, composite_name: str) -> Path:
        return self.gold_dir / f"gold_features_{composite_name}"

    def dataset_findings_dir(self, name: str) -> Path:
        return self.dataset_dir(name) / "findings"

    def dataset_objective_support_path(self, name: str) -> Path:
        return self.dataset_dir(name) / "objective_support.yaml"

    def dataset_docs_dir(self, name: str) -> Path:
        return self.dataset_dir(name) / "docs"

    @property
    def merged_dir(self) -> Path:
        return self.run_dir / "merged"

    @property
    def multi_dataset_findings_path(self) -> Path:
        return self.merged_dir / "multi_dataset_findings.yaml"

    @property
    def merged_findings_path(self) -> Path:
        return self.merged_dir / "silver_merged_findings.yaml"

    @property
    def merged_recommendations_path(self) -> Path:
        return self.merged_dir / "recommendations.yaml"

    @property
    def registered_functions_path(self) -> Path:
        """FW-15b — disk-backed `runtime.registry` snapshot.

        `@cr.register` writes to this file on every decoration so a fresh
        kernel (e.g. NB10 in a multi-task Databricks job) can rehydrate
        the in-memory registry via `Registry.load_from_disk()`.
        """
        return self.merged_dir / "registered_functions.json"

    @property
    def exploration_feature_profile_path(self) -> Path:
        return self.merged_dir / "exploration_feature_profile.yaml"

    @property
    def selection_config_snapshot_path(self) -> Path:
        return self.merged_dir / "selection_config_snapshot.yaml"

    @property
    def production_feature_profile_path(self) -> Path:
        return self.merged_dir / "production_feature_profile.yaml"

    @property
    def feature_spec_path(self) -> Path:
        return self.merged_dir / "feature_spec.yaml"

    @property
    def production_diagnostics_path(self) -> Path:
        return self.run_dir / "production_diagnostics.json"

    @property
    def field_availability_audit_dir(self) -> Path:
        return self.merged_dir / "field_availability_audit"

    @property
    def training_metadata_path(self) -> Path:
        return self.run_dir / "training_metadata.json"

    @property
    def bronze_metadata_path(self) -> Path:
        return self.run_dir / "bronze_metadata.json"

    @property
    def silver_metadata_path(self) -> Path:
        return self.run_dir / "silver_metadata.json"

    @property
    def gold_metadata_path(self) -> Path:
        return self.run_dir / "gold_metadata.json"

    @property
    def feature_meta_dir(self) -> Path:
        return self.run_dir / "feature_meta"

    @property
    def column_descriptions_dir(self) -> Path:
        return self.root / "column_descriptions"

    @property
    def feature_population_stats_dir(self) -> Path:
        return self.run_dir / "feature_population_stats"

    @property
    def exploration_metadata_path(self) -> Path:
        return self.run_dir / "exploration_metadata.json"

    @property
    def exploration_diagnostics_path(self) -> Path:
        return self.run_dir / "exploration_diagnostics.json"

    @property
    def cell_profiles_path(self) -> Path:
        return self.run_dir / "cell_profiles.json"

    def artifacts_dir(self, recommendations_hash: str | None) -> Path:
        return self.root / "artifacts" / (recommendations_hash or "default")

    def candidate_dir(self, name: str) -> Path:
        return self.merged_dir / name

    @property
    def session_dir(self) -> Path:
        return self.run_dir / "session" / "users"

    @property
    def diagnostics_dir(self) -> Path:
        return self.run_dir / "diagnostics"

    def diagnostic_path(self, name: str) -> Path:
        return self.diagnostics_dir / f"diag_{name}.yaml"

    def user_session_path(self, username: str) -> Path:
        return self.session_dir / f"{username}.json"

    @property
    def project_context_path(self) -> Path:
        return self.run_dir / "project_context.yaml"

    @property
    def snapshot_grid_path(self) -> Path:
        return self.run_dir / "snapshot_grid.yaml"

    @property
    def grid_votes_dir(self) -> Path:
        return self.run_dir / "grid_votes"

    @property
    def sample_entity_ids_path(self) -> Path:
        return self.run_dir / "sample_entity_ids.json"

    @property
    def holdout_entity_ids_path(self) -> Path:
        return self.run_dir / "holdout_entity_ids.json"

    @property
    def silver_merged_path(self) -> Path:
        return self.silver_dir / "silver_merged"

    def list_datasets(self) -> list[str]:
        if not self.datasets_dir.is_dir():
            return []
        return sorted(p.name for p in self.datasets_dir.iterdir() if p.is_dir())

    def list_candidates(self) -> list[str]:
        if not self.merged_dir.is_dir():
            return []
        return sorted(p.name for p in self.merged_dir.iterdir() if p.is_dir())

    def discover_all_findings(self, prefer_aggregated: bool = True) -> list[Path]:
        if not self.datasets_dir.is_dir():
            return []
        all_files: list[Path] = []
        for dataset_dir in sorted(self.datasets_dir.iterdir()):
            if not dataset_dir.is_dir():
                continue
            findings_dir = dataset_dir / "findings"
            if not findings_dir.is_dir():
                continue
            all_files.extend(sorted(findings_dir.glob("*_findings.yaml")))
        if not prefer_aggregated:
            return all_files
        return self._filter_prefer_aggregated(all_files)

    @staticmethod
    def _filter_prefer_aggregated(files: list[Path]) -> list[Path]:
        aggregated = {f for f in files if "_aggregated" in f.name}
        non_aggregated = [f for f in files if "_aggregated" not in f.name]
        if not aggregated:
            return files
        result = list(aggregated)
        aggregated_bases = set()
        for f in aggregated:
            stem = f.stem.replace("_findings", "")
            if "_aggregated" in stem:
                stem = stem.rsplit("_aggregated", 1)[0]
            aggregated_bases.add(stem)
        for f in non_aggregated:
            stem = f.stem.replace("_findings", "")
            if stem not in aggregated_bases:
                result.append(f)
        return sorted(result, key=lambda p: p.name)

    def setup(self) -> None:
        self.datasets_dir.mkdir(parents=True, exist_ok=True)
        self.merged_dir.mkdir(parents=True, exist_ok=True)
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.landing_dir.mkdir(parents=True, exist_ok=True)
        self.bronze_dir.mkdir(parents=True, exist_ok=True)
        self.silver_dir.mkdir(parents=True, exist_ok=True)
        self.gold_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def create(cls, root: Path, project_name: str) -> RunNamespace:
        run_id = f"{project_name}-{uuid.uuid4().hex[:8]}"
        ns = cls(root=root, run_id=run_id)
        ns.setup()
        return ns

    @classmethod
    def from_latest(cls, root: Optional[Path] = None) -> Optional[RunNamespace]:
        if root is None:
            from customer_retention.core.config.experiments import get_experiments_dir

            root = get_experiments_dir()
        runs_dir = root / "runs"
        if not runs_dir.is_dir():
            return None
        best_path: Optional[Path] = None
        best_mtime = -1.0
        for candidate in runs_dir.iterdir():
            if not candidate.is_dir():
                continue
            marker_mtime = cls._best_marker_mtime(candidate)
            if marker_mtime is not None and marker_mtime > best_mtime:
                best_mtime, best_path = marker_mtime, candidate
        if best_path is None:
            return None
        return cls(root=root, run_id=best_path.name)

    @staticmethod
    def _best_marker_mtime(run_dir: Path) -> Optional[float]:
        marker = run_dir / "project_context.yaml"
        return marker.stat().st_mtime if marker.exists() else None

    @classmethod
    def from_env(cls, root: Optional[Path] = None) -> Optional[RunNamespace]:
        run_id = os.environ.get("CR_RUN_ID")
        if not run_id:
            return None
        if root is None:
            from customer_retention.core.config.experiments import get_experiments_dir

            root = get_experiments_dir()
        return cls(root=root, run_id=run_id)

    @classmethod
    def sentinel_path(cls, root: Path) -> Path:
        return root / "runs" / ".active_run_id"

    @classmethod
    def from_sentinel(cls, root: Optional[Path] = None) -> Optional[RunNamespace]:
        if root is None:
            from customer_retention.core.config.experiments import get_experiments_dir

            root = get_experiments_dir()
        try:
            run_id = cls.sentinel_path(root).read_text().strip()
        except (OSError, ValueError):
            return None
        if not run_id:
            return None
        candidate = cls(root=root, run_id=run_id)
        if not candidate.run_dir.is_dir():
            return None
        return candidate

    def write_sentinel(self) -> None:
        sentinel = self.sentinel_path(self.root)
        sentinel.parent.mkdir(parents=True, exist_ok=True)
        sentinel.write_text(self.run_id)

    # ------------------------------------------------------------------
    # Project-level run pointer (.cr_active_run.json at project root)
    # ------------------------------------------------------------------

    @classmethod
    def _project_pointer_path(cls) -> Path:
        from customer_retention.core.config.experiments import _find_project_root
        return _find_project_root() / _RUN_POINTER_FILENAME

    def write_run_pointer(self) -> None:
        pointer = self._project_pointer_path()
        pointer.write_text(json.dumps({
            "experiments_root": str(self.root),
            "run_id": self.run_id,
        }, indent=2))

    @classmethod
    def from_run_pointer(cls) -> Optional[RunNamespace]:
        try:
            data = json.loads(cls._project_pointer_path().read_text())
        except (OSError, json.JSONDecodeError, ValueError):
            return None
        run_id = data.get("run_id")
        root_str = data.get("experiments_root")
        if not run_id or not root_str:
            return None
        root = make_path(root_str)
        candidate = cls(root=root, run_id=run_id)
        if not candidate.run_dir.is_dir():
            return None
        return candidate

    # ------------------------------------------------------------------
    # Unified discovery
    # ------------------------------------------------------------------

    @classmethod
    def from_env_or_latest(cls, root: Optional[Path] = None) -> Optional[RunNamespace]:
        """Resolve the active run via every file-tracked discovery tier.

        Tier order (notebook-job-safe — no env vars required after tier 1):
          1. ``CR_RUN_ID`` env var (when present, used with ``root`` if given).
          2. Explicit ``root`` is honored: try the in-root sentinel + latest
             marker. This handles the case where the caller knows the right
             experiments dir but hasn't pointed env vars at it.
          3. Project pointer (``<project_root>/.cr_active_run.json``) — the
             file NB00 writes carrying both ``run_id`` and ``experiments_root``.
             Always consulted, even when ``root`` was explicit, because the
             pointer is the only signal that survives across job tasks
             (where env vars do not propagate).
          4. Fall back to discovery without an explicit root (uses
             ``get_experiments_dir()`` which itself reads the pointer).
        """
        ns = cls.from_env(root=root)
        if ns is not None:
            return ns
        if root is not None:
            ns = cls.from_sentinel(root=root)
            if ns is not None:
                return ns
            ns = cls.from_latest(root=root)
            if ns is not None:
                return ns
        # File-tracked tier — never gated on env vars or explicit root.
        ns = cls.from_run_pointer()
        if ns is not None:
            return ns
        if root is None:
            ns = cls.from_sentinel()
            if ns is not None:
                return ns
            return cls.from_latest()
        return None
