from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, List, Optional

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

    def render_runner(self, config: PipelineConfig, target_derive_steps: Optional[List[Any]] = None) -> str: ...


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
        self._snapshot_feature_spec_into_namespace(config)
        self._materialize_patched_feature_spec(config)
        return config

    def _snapshot_feature_spec_into_namespace(self, config: PipelineConfig) -> None:
        """Mirror the spec that was used for generation into the current run's
        namespace so NB11's parity report can resolve it without scanning
        sibling runs. No-op when there is no namespace, no spec, the spec is
        already at the destination, or a local copy already exists (idempotent,
        byte-exact, never overwrites)."""
        ns = getattr(self._parser, "_namespace", None)
        spec_source = getattr(config, "feature_spec_path", None)
        if ns is None or not spec_source:
            return
        src = Path(spec_source)
        if not src.exists():
            return
        dst = ns.feature_spec_path
        if dst == src or dst.exists():
            return
        import shutil
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    def _materialize_patched_feature_spec(self, config: PipelineConfig) -> None:
        """When `parity_ignored_features` is non-empty, materialize a stripped
        copy of the spec into ``output_dir/findings/feature_spec.yaml`` and
        repoint ``config.feature_spec_path`` at it. The runtime gate (training
        script) re-loads ``_FEATURE_SPEC_PATH`` and re-checks selected_features
        against gold columns; without this redirect, generation would pass but
        the generated training step would still raise on the same missing
        feature. The on-disk source-of-truth (namespace) is never modified.

        The path is resolved to an absolute form before being baked into the
        rendered training script's ``_FEATURE_SPEC_PATH = Path(r"...")``
        literal. NB10's typical ``output_dir = Path("../generated_pipelines")
        / "databricks" / PIPELINE_NAME`` is relative to the codegen cwd; the
        bare relative literal would resolve correctly at codegen time but
        break under the training notebook's cwd at runtime
        (`FileNotFoundError`). Resolves the §7.4 framework gap so the §2.12
        operator patch becomes redundant against fresh codegen.
        """
        ignored = getattr(self._parser, "parity_ignored_features", frozenset())
        spec = getattr(self._parser, "_feature_spec", None)
        if not ignored or spec is None:
            return
        dst_dir = self._output_dir / "findings"
        dst_dir.mkdir(parents=True, exist_ok=True)
        patched_path = (dst_dir / "feature_spec.yaml").resolve()
        spec.save(patched_path)
        config.feature_spec_path = str(patched_path)
        if config.training is not None:
            config.training.feature_spec_path = str(patched_path)

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
        # Externalise the exploration feature profile to a sibling JSON file.
        # The training template (databricks_training.py.j2 / training.py.j2)
        # discovers this file at runtime via Path(__file__).parent. Keeping
        # it out of the .py source avoids 1-5 MB inline dict literals that
        # press on the Python parser during cell-4 evaluation on shared
        # clusters and bloat the rendered file beyond human readability.
        # NaN values are preserved via allow_nan=True (Python's json.load
        # accepts NaN by default), so float('nan') round-trips correctly.
        if (
            config.training is not None
            and config.training.exploration_feature_profile is not None
        ):
            import json as _ep_json

            json_path = path.parent / "exploration_profile.json"
            json_path.write_text(
                _ep_json.dumps(
                    config.training.exploration_feature_profile,
                    indent=2,
                    allow_nan=True,
                    default=str,
                )
            )
        path.write_text(self._renderer.render_training(config))
        return path

    def _write_runner(self, config: PipelineConfig) -> Path:
        path = self._output_dir / "pipeline_runner.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        steps = self._target_derive_steps()
        path.write_text(self._renderer.render_runner(config, target_derive_steps=steps))
        return path

    def _target_derive_steps(self) -> List[Any]:
        """Return the harvest's `cross_dataset_steps` filtered to those tagged
        `expected_stage="target_derive"` and carrying at least one declared
        dataset (FW-3 imperative replay scope). Single-dataset-scope functions
        are routed through `functions_by_target` by the harvester and never
        belong in this phase."""
        harvest = getattr(self, "_harvest_result", None)
        if harvest is None:
            return []
        if getattr(self._parser, "user_extensions_disabled", False):
            return []
        return [
            rf for rf in (harvest.cross_dataset_steps or [])
            if (rf.expected_stage or rf.inferred_stage) == "target_derive"
            and (rf.datasets or [])
        ]

    def _write_target_derive(self, config: PipelineConfig) -> Optional[Path]:
        steps = self._target_derive_steps()
        if not steps:
            return None
        if not hasattr(self._renderer, "render_target_derive"):
            return None
        target_derive_dir = self._output_dir / "target_derive"
        target_derive_dir.mkdir(parents=True, exist_ok=True)
        path = target_derive_dir / "run_target_derive.py"
        path.write_text(self._renderer.render_target_derive(config, steps))
        return path

    def _write_user_extensions(self) -> Optional[Path]:
        harvest = getattr(self, "_harvest_result", None)
        if harvest is None:
            return None
        kill_switch_active = getattr(self._parser, "user_extensions_disabled", False)
        if kill_switch_active:
            return None
        from .user_extensions_emitter import write_user_extensions
        return write_user_extensions(self._output_dir, harvest)

    def _write_generation_manifest(
        self, config: PipelineConfig, generated_files: List[Path]
    ) -> Path:
        from .generation_manifest import (
            build_generation_manifest,
            write_generation_manifest,
        )
        template_versions = (
            self._renderer.template_versions()
            if hasattr(self._renderer, "template_versions")
            else {}
        )
        kill_switch_active = getattr(
            self._parser, "user_extensions_disabled", False
        )
        harvested_names = self._harvested_function_names(kill_switch_active)
        manifest = build_generation_manifest(
            config,
            generated_files,
            self._output_dir,
            template_versions=template_versions,
            kill_switch_active=kill_switch_active,
            harvested_functions=harvested_names,
        )
        return write_generation_manifest(manifest, self._output_dir)

    def _harvested_function_names(self, kill_switch_active: bool) -> List[str]:
        if kill_switch_active:
            return []
        harvest = getattr(self, "_harvest_result", None)
        if harvest is None:
            return []
        return [rf.name for rf in harvest.all_functions()]
