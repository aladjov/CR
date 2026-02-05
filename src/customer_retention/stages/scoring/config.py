from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path

from customer_retention.core.config.experiments import get_catalog, get_experiment_name, get_experiments_dir, get_schema

try:
    from mlflow.tracking import MlflowClient
except ImportError:  # pragma: no cover
    MlflowClient = None  # type: ignore[assignment,misc]


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
        return cls(
            pipeline_name=module.PIPELINE_NAME,
            composite_name=cn,
            target_column=module.TARGET_COLUMN,
            entity_key=module.FEAST_ENTITY_KEY,
            timestamp_column=module.FEAST_TIMESTAMP_COL,
            recommendations_hash=module.RECOMMENDATIONS_HASH,
            experiments_dir=Path(module.EXPERIMENTS_DIR),
            artifacts_path=Path(module.ARTIFACTS_PATH),
            mlflow_tracking_uri=module.MLFLOW_TRACKING_URI,
            production_dir=Path(module.PRODUCTION_DIR),
            feast_repo_path=module.FEAST_REPO_PATH,
            feast_feature_view=module.FEAST_FEATURE_VIEW,
            pipeline_dir=pipeline_dir,
        )

    @classmethod
    def from_databricks(cls) -> ScoringConfig:
        catalog = get_catalog()
        schema = get_schema()
        experiment_name = get_experiment_name()
        experiments_dir = get_experiments_dir()
        client = MlflowClient()
        experiment = client.get_experiment_by_name(experiment_name)
        if not experiment:
            raise ValueError(f"MLflow experiment '{experiment_name}' not found")
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["metrics.best_roc_auc DESC"],
            max_results=1,
        )
        if not runs:
            raise ValueError(f"No runs found in experiment '{experiment_name}'")
        run = runs[0]
        tags = run.data.tags
        params = run.data.params
        target_column = tags.get("target_column", params.get("target_column", "target"))
        entity_key = tags.get("entity_key", params.get("entity_key", "customer_id"))
        timestamp_column = tags.get("timestamp_column", params.get("timestamp_column", "event_timestamp"))
        recommendations_hash = tags.get("recommendations_hash", "")
        cn = tags.get("composite_name", experiment_name)
        artifacts_subdir = recommendations_hash or "default"
        return cls(
            pipeline_name=experiment_name,
            composite_name=cn,
            target_column=target_column,
            entity_key=entity_key,
            timestamp_column=timestamp_column,
            recommendations_hash=recommendations_hash,
            experiments_dir=experiments_dir,
            artifacts_path=experiments_dir / "artifacts" / artifacts_subdir,
            mlflow_tracking_uri="databricks",
            production_dir=experiments_dir,
            catalog=catalog,
            schema=schema,
        )


def _load_module_from_path(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module
