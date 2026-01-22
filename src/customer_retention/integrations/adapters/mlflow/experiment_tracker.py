from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

try:
    import mlflow
    from mlflow.tracking import MlflowClient
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

if TYPE_CHECKING:
    from customer_retention.analysis.auto_explorer.findings import ExplorationFindings
    from customer_retention.analysis.recommendations.pipeline import RecommendationPipeline


class ExperimentTracker:
    def __init__(self, tracking_uri: str = "./mlruns", experiment_name: str = "customer_retention"):
        if not MLFLOW_AVAILABLE:
            raise ImportError("mlflow package required. Install with: uv sync --extra ml")
        mlflow.set_tracking_uri(tracking_uri)
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self._client = MlflowClient(tracking_uri=tracking_uri)
        self._ensure_experiment()

    def log_exploration(self, findings: "ExplorationFindings", run_name: Optional[str] = None) -> str:
        with mlflow.start_run(
            run_name=run_name or f"exploration_{Path(findings.source_path).stem}",
            experiment_id=self._ensure_experiment()
        ) as run:
            self._log_exploration_params(findings)
            self._log_exploration_metrics(findings)
            self._log_column_metrics(findings)
            mlflow.log_dict(findings.to_dict(), "exploration_findings.json")
            self._set_exploration_tags(findings)
            return run.info.run_id

    def log_pipeline_execution(
        self, pipeline: "RecommendationPipeline", run_name: Optional[str] = None,
        parent_run_id: Optional[str] = None
    ) -> str:
        with mlflow.start_run(
            run_name=run_name or "recommendation_pipeline",
            experiment_id=self._ensure_experiment(),
            nested=parent_run_id is not None
        ) as run:
            self._log_pipeline_params(pipeline)
            self._log_pipeline_metrics(pipeline)
            self._log_pipeline_artifacts(pipeline)
            mlflow.set_tags({"stage": "transformation", "pipeline_fitted": str(pipeline._is_fitted)})
            return run.info.run_id

    def log_model_training(
        self, model: Any, metrics: Dict[str, float], params: Dict[str, Any],
        model_name: str = "churn_model", run_name: Optional[str] = None
    ) -> str:
        with mlflow.start_run(
            run_name=run_name or f"training_{model_name}",
            experiment_id=self._ensure_experiment()
        ) as run:
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, "model", registered_model_name=model_name)
            mlflow.set_tags({"stage": "training", "model_name": model_name})
            return run.info.run_id

    def get_best_run(self, metric: str = "overall_quality_score", ascending: bool = False) -> Optional[Dict]:
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        if experiment is None:
            return None
        runs = self._client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=[f"metrics.{metric} {'ASC' if ascending else 'DESC'}"],
            max_results=1
        )
        return runs[0].to_dictionary() if runs else None

    def list_exploration_runs(self) -> List[Dict]:
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        if experiment is None:
            return []
        runs = self._client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="tags.stage = 'exploration'"
        )
        return [r.to_dictionary() for r in runs]

    @staticmethod
    def serve_ui(host: str = "127.0.0.1", port: int = 5000, tracking_uri: str = "./mlruns"):
        import subprocess
        import sys
        subprocess.run([
            sys.executable, "-m", "mlflow", "ui",
            "--backend-store-uri", tracking_uri, "--host", host, "--port", str(port)
        ])

    def _ensure_experiment(self) -> str:
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        if experiment is None:
            return mlflow.create_experiment(self.experiment_name)
        return experiment.experiment_id

    def _log_exploration_params(self, findings: "ExplorationFindings") -> None:
        mlflow.log_params({
            "source_path": findings.source_path,
            "source_format": findings.source_format,
            "target_column": findings.target_column or "none",
        })

    def _log_exploration_metrics(self, findings: "ExplorationFindings") -> None:
        mlflow.log_metrics({
            "row_count": findings.row_count,
            "column_count": findings.column_count,
            "memory_usage_mb": findings.memory_usage_mb,
            "overall_quality_score": findings.overall_quality_score,
            "modeling_ready": 1.0 if findings.modeling_ready else 0.0,
            "critical_issues_count": len(findings.critical_issues),
            "warnings_count": len(findings.warnings),
        })

    def _log_column_metrics(self, findings: "ExplorationFindings") -> None:
        type_counts: Dict[str, int] = {}
        cleaning_needed_count = 0
        for col in findings.columns.values():
            type_name = col.inferred_type.value
            type_counts[type_name] = type_counts.get(type_name, 0) + 1
            if col.cleaning_needed:
                cleaning_needed_count += 1
        for type_name, count in type_counts.items():
            mlflow.log_metric(f"columns_{type_name}", count)
        mlflow.log_metric("columns_needing_cleaning", cleaning_needed_count)

    def _set_exploration_tags(self, findings: "ExplorationFindings") -> None:
        mlflow.set_tags({
            "stage": "exploration",
            "modeling_ready": str(findings.modeling_ready),
            "is_time_series": str(findings.is_time_series),
        })

    def _log_pipeline_params(self, pipeline: "RecommendationPipeline") -> None:
        mlflow.log_params({
            "recommendation_count": len(pipeline.recommendations),
            "is_fitted": pipeline._is_fitted,
        })

    def _log_pipeline_metrics(self, pipeline: "RecommendationPipeline") -> None:
        rec_types: Dict[str, int] = {}
        rec_categories: Dict[str, int] = {}
        for rec in pipeline.recommendations:
            rec_types[rec.recommendation_type] = rec_types.get(rec.recommendation_type, 0) + 1
            rec_categories[rec.category] = rec_categories.get(rec.category, 0) + 1
        for rec_type, count in rec_types.items():
            mlflow.log_metric(f"rec_type_{rec_type}", count)
        for category, count in rec_categories.items():
            mlflow.log_metric(f"rec_category_{category}", count)

    def _log_pipeline_artifacts(self, pipeline: "RecommendationPipeline") -> None:
        from customer_retention.analysis.recommendations.base import Platform
        mlflow.log_dict(pipeline.to_dict(), "pipeline_config.json")
        mlflow.log_text(pipeline.generate_code(), "generated_code_local.py")
        mlflow.log_text(pipeline.generate_code(Platform.DATABRICKS), "generated_code_databricks.py")
