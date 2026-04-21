from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from customer_retention.core.naming import Manifest

from .findings_parser import FindingsParser
from .models import PipelineConfig
from .protocols import PipelineGeneratorBase
from .renderer import CodeRenderer

if TYPE_CHECKING:
    from customer_retention.runtime.harvest import HarvestResult


class PipelineGenerator(PipelineGeneratorBase):
    def __init__(
        self,
        findings_dir: str,
        output_dir: str,
        pipeline_name: str,
        experiments_dir: str = None,
        production_dir: str = None,
        namespace=None,
        intent=None,
        bronze_aggregation_overrides: Optional[Dict[str, Dict[str, Any]]] = None,
        disable_user_extensions: Optional[bool] = None,
        harvest_result: Optional["HarvestResult"] = None,
        parity_mode: Optional[str] = None,
    ):
        self._findings_dir = Path(findings_dir)
        self._output_dir = Path(output_dir)
        self._pipeline_name = pipeline_name
        self._experiments_dir = experiments_dir
        self._production_dir = production_dir
        self._namespace = namespace
        self._parser = FindingsParser(
            findings_dir,
            namespace=namespace,
            intent=intent,
            bronze_aggregation_overrides=bronze_aggregation_overrides,
            disable_user_extensions=disable_user_extensions,
            parity_mode=parity_mode,
        )
        self._renderer = CodeRenderer()
        self._harvest_result = harvest_result

    def generate(self) -> List[Path]:
        config = self._build_config()
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
            self._write_manifest(config),
        ]
        self._copy_holdout_ids()
        self._copy_feature_spec()
        user_ext_path = self._write_user_extensions()
        if user_ext_path is not None:
            generated_files.append(user_ext_path)
        generated_files.append(self._write_generation_manifest(config, generated_files))
        return generated_files

    def _copy_holdout_ids(self) -> None:
        """Copy holdout_entity_ids.json into the generated pipeline's findings dir."""
        import shutil

        src = None
        if self._namespace is not None and self._namespace.holdout_entity_ids_path.exists():
            src = self._namespace.holdout_entity_ids_path
        elif (self._findings_dir / "holdout_entity_ids.json").exists():
            src = self._findings_dir / "holdout_entity_ids.json"
        if src is None:
            return
        dst_dir = self._output_dir / "findings"
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst_dir / "holdout_entity_ids.json")

    def _copy_feature_spec(self) -> None:
        import shutil

        src = None
        if self._namespace is not None and self._namespace.feature_spec_path.exists():
            src = self._namespace.feature_spec_path
        elif (self._findings_dir / "feature_spec.yaml").exists():
            src = self._findings_dir / "feature_spec.yaml"
        if src is None:
            return
        dst_dir = self._output_dir / "findings"
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst_dir / "feature_spec.yaml")

    def _write_manifest(self, config: PipelineConfig) -> Path:
        source_names = [s.name for s in config.sources if not s.excluded]
        manifest = Manifest.from_sources(source_names, config.name, config.recommendations_hash or "")
        path = self._output_dir / "manifest.json"
        manifest.save(path)
        return path

    def _write_run_all(self, config: PipelineConfig) -> Path:
        path = self._output_dir / "run_all.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self._renderer.render_run_all(config))
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
