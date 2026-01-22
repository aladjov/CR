from pathlib import Path
from typing import List
from .findings_parser import FindingsParser
from .renderer import CodeRenderer
from .models import PipelineConfig


class PipelineGenerator:
    def __init__(self, findings_dir: str, output_dir: str, pipeline_name: str):
        self._findings_dir = Path(findings_dir)
        self._output_dir = Path(output_dir)
        self._pipeline_name = pipeline_name
        self._parser = FindingsParser(findings_dir)
        self._renderer = CodeRenderer()

    def generate(self) -> List[Path]:
        config = self._parser.parse()
        config.name = self._pipeline_name
        config.output_dir = "."
        generated_files = [
            self._write_run_all(config),
            self._write_config(config),
            *self._write_bronze_files(config),
            self._write_silver(config),
            self._write_gold(config),
            self._write_training(config),
            self._write_runner(config),
            self._write_workflow(config),
            *self._write_feast_repo(config),
            self._write_scoring(config),
            self._write_dashboard(config),
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

    def _write_bronze_files(self, config: PipelineConfig) -> List[Path]:
        bronze_dir = self._output_dir / "bronze"
        bronze_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for source_name, bronze_config in config.bronze.items():
            path = bronze_dir / f"bronze_{source_name}.py"
            path.write_text(self._renderer.render_bronze(source_name, bronze_config))
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

        # Write feature_store.yaml
        config_path = feast_dir / "feature_store.yaml"
        config_path.write_text(self._renderer.render_feast_config(config))
        paths.append(config_path)

        # Write features.py
        features_path = feast_dir / "features.py"
        features_path.write_text(self._renderer.render_feast_features(config))
        paths.append(features_path)

        return paths

    def _write_scoring(self, config: PipelineConfig) -> Path:
        path = self._output_dir / "scoring" / "run_scoring.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self._renderer.render_scoring(config))
        return path

    def _write_dashboard(self, config: PipelineConfig) -> Path:
        path = self._output_dir / "scoring" / "scoring_dashboard.ipynb"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self._renderer.render_dashboard(config))
        return path
