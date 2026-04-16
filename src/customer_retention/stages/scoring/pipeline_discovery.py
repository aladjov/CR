from __future__ import annotations

from pathlib import Path

from .config import ScoringConfig

_DEFAULT_TARGET = "databricks"


def find_generated_pipeline_dir(
    workspace_root: Path,
    scoring_config: ScoringConfig,
    target: str = _DEFAULT_TARGET,
) -> Path:
    base = Path(workspace_root) / "generated_pipelines" / target
    if not base.exists():
        raise FileNotFoundError(
            f"Generated pipelines base directory not found: {base}. "
            "Run exploration_notebooks/10_spec_generation.ipynb to produce it."
        )
    direct = _resolve_direct_pipeline_path(base, scoring_config.pipeline_name)
    if direct is not None:
        return direct
    composite = scoring_config.composite_name
    scanned = _resolve_by_composite_name(base, composite) if composite else None
    if scanned is not None:
        return scanned
    raise FileNotFoundError(
        f"Could not resolve generated pipeline directory under {base}. "
        f"Tried pipeline_name={scoring_config.pipeline_name!r} and "
        f"composite_name={composite!r}. Set PIPELINE_DIR in the configuration "
        "cell to the absolute pipeline folder, or re-run "
        "exploration_notebooks/10_spec_generation.ipynb so training metadata "
        "carries pipeline_name."
    )


def _resolve_direct_pipeline_path(base: Path, pipeline_name: str) -> Path | None:
    if not pipeline_name or "/" in pipeline_name:
        return None
    candidate = base / pipeline_name
    return candidate if candidate.exists() else None


def _resolve_by_composite_name(base: Path, composite_name: str) -> Path | None:
    target_stem = f"silver_featureset_{composite_name}"
    for child in sorted(base.iterdir()):
        if not child.is_dir():
            continue
        silver = child / "silver"
        if not silver.is_dir():
            continue
        if any(f.stem == target_stem for f in silver.iterdir()):
            return child
    return None
