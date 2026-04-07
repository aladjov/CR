from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from customer_retention.core.config.experiments import get_framework_repo_path

from .databricks_renderer import DatabricksCodeRenderer
from .findings_parser import FindingsParser
from .protocols import PipelineGeneratorBase


class DatabricksPipelineGenerator(PipelineGeneratorBase):
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
        framework_repo_path: str | None = None,
        bronze_aggregation_overrides: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
        self._findings_dir = Path(findings_dir)
        self._output_dir = Path(output_dir)
        self._pipeline_name = pipeline_name
        self._catalog = catalog
        self._schema = schema
        self._experiments_dir = experiments_dir
        self._parser = FindingsParser(
            findings_dir,
            namespace=namespace,
            intent=intent,
            bronze_aggregation_overrides=bronze_aggregation_overrides,
        )
        self._renderer = DatabricksCodeRenderer(
            catalog=catalog, schema=schema,
            framework_repo_path=framework_repo_path or get_framework_repo_path(),
        )

    def _bronze_entity_input_name(self, source_name: str, agg_name: str) -> str:
        return source_name

    def generate(self) -> List[Path]:
        config = self._build_config()
        generated_files = [
            self._write_config(config),
            *self._write_landing(config),
            *self._write_bronze_files(config),
            self._write_silver(config),
            self._write_gold(config),
            self._write_training(config),
            self._write_runner(config),
        ]
        return generated_files
