from pathlib import Path
from typing import List

from customer_retention.core.naming import composite_name, script_name

from .databricks_renderer import DatabricksCodeRenderer
from .findings_parser import FindingsParser
from .models import PipelineConfig


class DatabricksPipelineGenerator:
    def __init__(
        self,
        findings_dir: str,
        output_dir: str,
        pipeline_name: str,
        catalog: str = "main",
        schema: str = "default",
        experiments_dir: str = None,
        namespace=None,
        intent=None,
    ):
        self._findings_dir = Path(findings_dir)
        self._output_dir = Path(output_dir)
        self._pipeline_name = pipeline_name
        self._catalog = catalog
        self._schema = schema
        self._experiments_dir = experiments_dir
        self._parser = FindingsParser(findings_dir, namespace=namespace, intent=intent)
        self._renderer = DatabricksCodeRenderer(catalog=catalog, schema=schema)

    def generate(self) -> List[Path]:
        config = self._parser.parse()
        config.name = self._pipeline_name
        config.output_dir = "."
        config.experiments_dir = self._experiments_dir
        source_names = [
            f"{s.name}_aggregated" if s.is_event_level else s.name
            for s in config.sources if not s.excluded
        ]
        config.composite_name = composite_name(source_names)
        generated_files = [
            self._write_config(config),
            *self._write_bronze_files(config),
            self._write_silver(config),
            self._write_gold(config),
            self._write_training(config),
            self._write_runner(config),
        ]
        return generated_files

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
            entity_script = script_name("bronze_entity", source_name)
            path = bronze_dir / f"{entity_script}.py"
            path.write_text(self._renderer.render_bronze(source_name, bronze_config))
            paths.append(path)
        for source_name, event_config in config.bronze_event.items():
            event_script = script_name("bronze_event", source_name)
            path = bronze_dir / f"{event_script}.py"
            path.write_text(self._renderer.render_bronze_event(source_name, event_config))
            paths.append(path)
            agg_name = f"{source_name}_aggregated"
            entity_script = script_name("bronze_entity", agg_name)
            entity_path = bronze_dir / f"{entity_script}.py"
            entity_path.write_text(
                self._renderer.render_bronze_entity(agg_name, event_config, source_name, source_name),
            )
            paths.append(entity_path)
        return paths

    def _write_silver(self, config: PipelineConfig) -> Path:
        cn = config.composite_name
        silver_script = script_name("silver", cn)
        path = self._output_dir / "silver" / f"{silver_script}.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self._renderer.render_silver(config))
        return path

    def _write_gold(self, config: PipelineConfig) -> Path:
        cn = config.composite_name
        gold_script = script_name("gold", cn)
        path = self._output_dir / "gold" / f"{gold_script}.py"
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
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self._renderer.render_runner(config))
        return path
