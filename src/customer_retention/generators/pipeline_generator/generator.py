from pathlib import Path
from typing import List

from .findings_parser import FindingsParser
from .models import PipelineConfig
from .renderer import CodeRenderer


class PipelineGenerator:
    def __init__(self, findings_dir: str, output_dir: str, pipeline_name: str, experiments_dir: str = None, production_dir: str = None):
        self._findings_dir = Path(findings_dir)
        self._output_dir = Path(output_dir)
        self._pipeline_name = pipeline_name
        self._experiments_dir = experiments_dir
        self._production_dir = production_dir
        self._parser = FindingsParser(findings_dir)
        self._renderer = CodeRenderer()

    def generate(self) -> List[Path]:
        config = self._parser.parse()
        config.name = self._pipeline_name
        config.output_dir = "."
        config.experiments_dir = self._experiments_dir
        config.production_dir = self._production_dir
        self._renderer.set_docs_base(self._experiments_dir)
        generated_files = [
            self._write_run_all(config),
            self._write_config(config),
            *self._write_landing(config),
            *self._write_bronze_files(config),
            self._write_silver(config),
            self._write_gold(config),
            self._write_training(config),
            self._write_runner(config),
            self._write_workflow(config),
            *self._write_feast_repo(config),
            *self._write_validation(config),
            self._write_exploration_report(config),
        ]
        return generated_files

    def _write_run_all(self, config: PipelineConfig) -> Path:
        path = self._output_dir / "run_all.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self._renderer.render_run_all(config))
        return path

    def _write_config(self, config: PipelineConfig) -> Path:
        path = self._output_dir / "config.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self._renderer.render_config(config))
        return path

    def _write_landing(self, config: PipelineConfig) -> List[Path]:
        if not config.landing:
            return []
        landing_dir = self._output_dir / "landing"
        landing_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for landing_name, landing_config in config.landing.items():
            path = landing_dir / f"landing_{landing_name}.py"
            path.write_text(self._renderer.render_landing(landing_name, landing_config))
            paths.append(path)
        return paths

    def _write_bronze_files(self, config: PipelineConfig) -> List[Path]:
        bronze_dir = self._output_dir / "bronze"
        bronze_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for source_name, bronze_config in config.bronze.items():
            path = bronze_dir / f"bronze_{source_name}.py"
            path.write_text(self._renderer.render_bronze(source_name, bronze_config))
            paths.append(path)
        for source_name, event_config in config.bronze_event.items():
            path = bronze_dir / f"bronze_{source_name}.py"
            path.write_text(self._renderer.render_bronze_event(source_name, event_config))
            paths.append(path)
        return paths

    def _write_silver(self, config: PipelineConfig) -> Path:
        path = self._output_dir / "silver" / "silver_merge.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self._renderer.render_silver(config))
        return path

    def _write_gold(self, config: PipelineConfig) -> Path:
        path = self._output_dir / "gold" / "gold_features.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self._renderer.render_gold(config))
        return path

    def _write_training(self, config: PipelineConfig) -> Path:
        path = self._output_dir / "training" / "ml_experiment.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self._renderer.render_training(config))
        return path

    def _write_runner(self, config: PipelineConfig) -> Path:
        path = self._output_dir / "pipeline_runner.py"
        path.write_text(self._renderer.render_runner(config))
        return path

    def _write_workflow(self, config: PipelineConfig) -> Path:
        path = self._output_dir / "workflow.json"
        path.write_text(self._renderer.render_workflow(config))
        return path

    def _write_feast_repo(self, config: PipelineConfig) -> List[Path]:
        feast_dir = self._output_dir / "feature_repo"
        feast_dir.mkdir(parents=True, exist_ok=True)
        (feast_dir / "data").mkdir(parents=True, exist_ok=True)
        paths = []
        config_path = feast_dir / "feature_store.yaml"
        config_path.write_text(self._renderer.render_feast_config(config))
        paths.append(config_path)
        features_path = feast_dir / "features.py"
        features_path.write_text(self._renderer.render_feast_features(config))
        paths.append(features_path)
        return paths


    def _write_exploration_report(self, config: PipelineConfig) -> Path:
        docs_dir = self._output_dir / "docs"
        docs_dir.mkdir(parents=True, exist_ok=True)
        path = docs_dir / "exploration_report.py"
        path.write_text(self._renderer.render_exploration_report(config))
        return path

    def _write_validation(self, config: PipelineConfig) -> List[Path]:
        validation_dir = self._output_dir / "validation"
        validation_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        init_path = validation_dir / "__init__.py"
        init_path.write_text("")
        paths.append(init_path)
        validate_path = validation_dir / "validate_pipeline.py"
        validate_path.write_text(self._renderer.render_validation(config))
        paths.append(validate_path)
        run_validation_path = validation_dir / "run_validation.py"
        run_validation_path.write_text(self._renderer.render_run_validation(config))
        paths.append(run_validation_path)
        return paths
