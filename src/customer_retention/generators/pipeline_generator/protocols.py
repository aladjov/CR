from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

from typing_extensions import Protocol, runtime_checkable

from customer_retention.core.naming import composite_name, script_name

from .findings_parser import FindingsParser
from .models import (
    BronzeEventConfig,
    BronzeLayerConfig,
    LandingLayerConfig,
    PipelineConfig,
)


@runtime_checkable
class CodeRendererProtocol(Protocol):
    def render_config(self, config: PipelineConfig) -> str: ...

    def render_landing(self, name: str, config: LandingLayerConfig) -> str: ...

    def render_bronze(self, source_name: str, bronze_config: BronzeLayerConfig) -> str: ...

    def render_bronze_event(
        self,
        source_name: str,
        config: BronzeEventConfig,
        pipeline_config: "PipelineConfig | None" = None,
    ) -> str: ...

    def render_bronze_entity(
        self, source_name: str, config: BronzeEventConfig, bronze_input_name: str, raw_source_name: str = ""
    ) -> str: ...

    def render_silver(self, config: PipelineConfig) -> str: ...

    def render_gold(self, config: PipelineConfig) -> str: ...

    def render_training(self, config: PipelineConfig) -> str: ...

    def render_runner(self, config: PipelineConfig) -> str: ...


class PipelineGeneratorBase(ABC):
    _renderer: CodeRendererProtocol
    _output_dir: Path
    _pipeline_name: str
    _findings_dir: Path
    _experiments_dir: str | None
    _parser: FindingsParser

    @abstractmethod
    def generate(self) -> List[Path]: ...

    def _build_config(self) -> PipelineConfig:
        config = self._parser.parse()
        config.name = self._pipeline_name
        config.output_dir = "."
        config.experiments_dir = self._experiments_dir
        source_names = [
            f"{s.name}_aggregated" if s.is_event_level else s.name for s in config.sources if not s.excluded
        ]
        config.composite_name = composite_name(source_names)
        return config

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
        for name, landing_config in config.landing.items():
            path = landing_dir / f"landing_{name}.py"
            path.write_text(self._renderer.render_landing(name, landing_config))
            paths.append(path)
        return paths

    def _bronze_entity_input_name(self, source_name: str, agg_name: str) -> str:
        return agg_name

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
            path.write_text(
                self._renderer.render_bronze_event(source_name, event_config, config)
            )
            paths.append(path)
            agg_name = f"{source_name}_aggregated"
            entity_script = script_name("bronze_entity", agg_name)
            entity_path = bronze_dir / f"{entity_script}.py"
            input_name = self._bronze_entity_input_name(source_name, agg_name)
            entity_path.write_text(
                self._renderer.render_bronze_entity(agg_name, event_config, input_name, source_name)
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
