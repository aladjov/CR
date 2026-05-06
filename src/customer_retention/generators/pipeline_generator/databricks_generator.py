from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional

from customer_retention.core.config.experiments import get_framework_repo_path

from .databricks_renderer import DatabricksCodeRenderer
from .findings_parser import FindingsParser
from .override_merge import _merge_registry_bronze_overrides
from .protocols import PipelineGeneratorBase

if TYPE_CHECKING:
    from customer_retention.runtime.harvest import HarvestResult


def _auto_harvest() -> Optional["HarvestResult"]:
    """FW-15b — when the caller did not pass `harvest_result=`, hydrate the
    `runtime.registry` from disk (multi-task Databricks job: NB10 runs in a
    fresh kernel) and run the harvester. Real harvest failures propagate
    so codegen never silently produces a pipeline without user_extensions
    when the operator clearly intended one (engagement spschurn-01bb634c
    failure mode)."""
    try:
        from customer_retention.runtime.harvest import Harvester  # noqa: PLC0415
        from customer_retention.runtime.registry import registry  # noqa: PLC0415
    except ImportError:
        return None
    if not registry.get_registered():
        registry.load_from_disk()
    if not registry.get_registered():
        return None
    return Harvester().harvest()


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
        disable_user_extensions: Optional[bool] = None,
        harvest_result: Optional["HarvestResult"] = None,
        parity_mode: Optional[str] = None,
        parity_ignored_features: Optional[Iterable[str]] = None,
        raw_source_path_overrides: Optional[Dict[str, str]] = None,
        landing_lifecycle_overrides: Optional[Dict[str, Dict[str, Any]]] = None,
        landing_filter_overrides: Optional[Dict[str, str]] = None,
        landing_drop_columns_overrides: Optional[Dict[str, Iterable[str]]] = None,
        strict_datetime_parity: bool = True,
    ):
        self._findings_dir = Path(findings_dir)
        self._output_dir = Path(output_dir)
        self._pipeline_name = pipeline_name
        self._catalog = catalog
        self._schema = schema
        self._experiments_dir = experiments_dir
        merged_bronze_overrides = _merge_registry_bronze_overrides(
            namespace, bronze_aggregation_overrides
        )
        self._parser = FindingsParser(
            findings_dir,
            namespace=namespace,
            intent=intent,
            bronze_aggregation_overrides=merged_bronze_overrides,
            disable_user_extensions=disable_user_extensions,
            parity_mode=parity_mode,
            parity_ignored_features=parity_ignored_features,
            raw_source_path_overrides=raw_source_path_overrides,
            landing_lifecycle_overrides=landing_lifecycle_overrides,
            landing_filter_overrides=landing_filter_overrides,
            landing_drop_columns_overrides=landing_drop_columns_overrides,
            strict_datetime_parity=strict_datetime_parity,
        )
        self._harvest_result = (
            harvest_result if harvest_result is not None else _auto_harvest()
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
        ]
        target_derive_path = self._write_target_derive(config)
        if target_derive_path is not None:
            generated_files.append(target_derive_path)
        generated_files.append(self._write_runner(config))
        user_ext_path = self._write_user_extensions()
        if user_ext_path is not None:
            generated_files.append(user_ext_path)
        generated_files.append(self._write_generation_manifest(config, generated_files))
        return generated_files
