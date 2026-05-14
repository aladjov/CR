from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from customer_retention.core.config.experiments import get_catalog, get_experiments_dir, get_schema


@dataclass
class ScoringConfig:
    pipeline_name: str
    composite_name: str
    target_column: str
    entity_key: str
    timestamp_column: str
    recommendations_hash: str
    experiments_dir: Path
    artifacts_path: Path
    mlflow_tracking_uri: str
    production_dir: Path
    catalog: str = ""
    schema: str = ""
    feast_repo_path: str = ""
    feast_feature_view: str = ""
    pipeline_dir: Path = Path()
    mlflow_run_id: str = ""
    best_model_name: str = ""
    logged_models: List[Dict[str, Any]] = field(default_factory=list)
    registered_model_name: str = ""

    @property
    def original_column(self) -> str:
        return f"original_{self.target_column}"

    @property
    def is_databricks(self) -> bool:
        return bool(self.catalog)

    @property
    def scoring_output_dir(self) -> Path:
        return self.experiments_dir / "data" / "scoring"

    @classmethod
    def from_local_config(cls, pipeline_dir: Path) -> ScoringConfig:
        pipeline_dir = Path(pipeline_dir)
        config_path = pipeline_dir / "config.py"
        if not config_path.exists():
            raise FileNotFoundError(f"No config.py found in {pipeline_dir}")
        module = _load_module_from_path("_scoring_config_gen", config_path)
        cn = getattr(module, "COMPOSITE_NAME", module.PIPELINE_NAME)
        meta = _load_training_metadata(_discover_namespace())
        return cls(
            pipeline_name=module.PIPELINE_NAME,
            composite_name=cn,
            target_column=module.TARGET_COLUMN,
            entity_key=getattr(module, "ENTITY_KEY", getattr(module, "FEAST_ENTITY_KEY", "entity_id")),
            timestamp_column=module.FEAST_TIMESTAMP_COL,
            recommendations_hash=module.RECOMMENDATIONS_HASH,
            experiments_dir=Path(module.EXPERIMENTS_DIR),
            artifacts_path=Path(module.ARTIFACTS_PATH),
            mlflow_tracking_uri=module.MLFLOW_TRACKING_URI,
            production_dir=Path(module.PRODUCTION_DIR),
            feast_repo_path=module.FEAST_REPO_PATH,
            feast_feature_view=module.FEAST_FEATURE_VIEW,
            pipeline_dir=pipeline_dir,
            mlflow_run_id=(meta or {}).get("mlflow_run_id", ""),
            best_model_name=(meta or {}).get("best_model_name", ""),
            logged_models=(meta or {}).get("logged_models", []),
            registered_model_name=(meta or {}).get("registered_model_name", ""),
        )

    @classmethod
    def from_databricks(cls) -> ScoringConfig:
        catalog = get_catalog()
        schema = get_schema()
        experiments_dir = get_experiments_dir()
        ns = _discover_namespace()
        meta = _load_training_metadata(ns) if ns else None
        if not meta:
            raise ValueError(
                "No training_metadata.json found in the run namespace. "
                "Re-run the generated training pipeline (NB10 output) before scoring."
            )
        experiment_name = meta["mlflow_experiment_name"]
        pipeline_name = meta.get("pipeline_name") or experiment_name
        target_column = meta.get("target_column", "target")
        entity_key = meta.get("entity_key", "entity_id")
        timestamp_column = meta.get("timestamp_column", "event_timestamp")
        recommendations_hash = meta.get("recommendations_hash", "")
        cn = meta.get("composite_name", experiment_name)
        artifacts_path = ns.artifacts_dir(recommendations_hash)
        return cls(
            pipeline_name=pipeline_name,
            composite_name=cn,
            target_column=target_column,
            entity_key=entity_key,
            timestamp_column=timestamp_column,
            recommendations_hash=recommendations_hash,
            experiments_dir=experiments_dir,
            artifacts_path=artifacts_path,
            mlflow_tracking_uri="databricks",
            production_dir=experiments_dir,
            catalog=catalog,
            schema=schema,
            mlflow_run_id=meta.get("mlflow_run_id", ""),
            best_model_name=meta.get("best_model_name", ""),
            logged_models=meta.get("logged_models", []),
            registered_model_name=meta.get("registered_model_name", ""),
        )


def _discover_namespace() -> Optional["RunNamespace"]:  # noqa: F821
    from customer_retention.analysis.auto_explorer.run_namespace import RunNamespace
    return RunNamespace.from_env_or_latest()


@dataclass
class ResolvedScoring:
    """Result of ``resolve_scoring_context`` — bundles every value a causal-track
    notebook needs to wire scoring/dashboard cells without hard-coding paths.

    The ``source`` field is the one-line diagnostic produced by
    ``RunNamespace.resolve`` so the notebook can echo which discovery tier
    won (operator override, run-pointer, sentinel, latest-by-mtime).
    """

    namespace: Any
    scoring_config: "ScoringConfig"
    model_uri: Optional[str]
    model_name: str
    model_version: str
    source: str


def resolve_scoring_context(
    *,
    run_id: Optional[str] = None,
    experiments_dir: Optional[Path | str] = None,
    model_uri: Optional[str] = None,
    is_databricks: Optional[bool] = None,
    mlflow_alias: str = "production",
) -> ResolvedScoring:
    """Unified resolution path for causal-track / scoring notebooks.

    Priority chain (highest first):
      1. ``run_id`` / ``experiments_dir`` overrides → pinned namespace.
      2. ``RunNamespace.from_env_or_latest`` discovery tiers
         (``CR_RUN_ID`` env, project pointer, sentinel, latest-by-mtime).

    Model URI resolution:
      • If ``model_uri`` override is passed, it wins unconditionally.
      • Else on Databricks, the registered model name is looked up from
        ``training_metadata.json`` and the ``@<mlflow_alias>`` alias is
        resolved through MLflow (defaults to ``@production``).
      • Else (local) returns ``None`` — caller falls back to local model
        artifacts via ``ScoringConfig.from_local_config``.

    The returned ``ResolvedScoring`` carries the diagnostic ``source``
    string so the notebook can print it as audit output. Raises
    ``ValueError`` only when overrides are partially supplied AND the
    pinned run does not exist — every other failure mode is reported
    via a non-None ``source`` and a ``namespace=None``.
    """
    from customer_retention.analysis.auto_explorer.run_namespace import RunNamespace
    from customer_retention.core.compat.detection import is_databricks as _is_databricks_default

    if is_databricks is None:
        is_databricks = _is_databricks_default()

    ns, source = RunNamespace.resolve(run_id=run_id, experiments_dir=experiments_dir)

    if is_databricks:
        if ns is None:
            raise ValueError(
                f"resolve_scoring_context: no run namespace resolved on Databricks "
                f"({source}). Pass `run_id` + `experiments_dir` overrides or write "
                "the run pointer via NB00 / RunNamespace.write_run_pointer()."
            )
        scoring_config = ScoringConfig.from_databricks()
        model_name = scoring_config.registered_model_name
        if model_uri:
            resolved_model_uri = model_uri
            model_version = "override"
        else:
            import mlflow

            mlflow_client = mlflow.tracking.MlflowClient()
            production_version = mlflow_client.get_model_version_by_alias(
                model_name, mlflow_alias,
            )
            model_version = production_version.version
            resolved_model_uri = f"models:/{model_name}@{mlflow_alias}"
        return ResolvedScoring(
            namespace=ns,
            scoring_config=scoring_config,
            model_uri=resolved_model_uri,
            model_name=model_name,
            model_version=model_version,
            source=source,
        )

    scoring_config = ScoringConfig.from_local_config(get_experiments_dir())
    return ResolvedScoring(
        namespace=ns,
        scoring_config=scoring_config,
        model_uri=model_uri,
        model_name=scoring_config.best_model_name or "local_model",
        model_version="local",
        source=source,
    )


def _load_training_metadata(ns) -> Optional[dict]:
    if ns is None:
        return None
    for path in (ns.training_metadata_path, ns.exploration_metadata_path):
        try:
            return json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
    return None


def _load_module_from_path(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module
