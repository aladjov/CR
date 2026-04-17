"""Generation manifest for user-extensions Release A rails.

Written alongside every generated pipeline at
`<output_dir>/generation_manifest.json`. Each Phase 8 debugging session
diffs two manifests: the failing file, the template version that produced
it, and the harvest input are all named, so nobody scrolls through
generated Python hunting for the drift.

This is separate from the existing `manifest.json` (source/name/hash)
that `PipelineGenerator._write_manifest` emits.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .models import PipelineConfig

BASELINE_TAG = "user-extensions-baseline-2026-04-17"
RELEASE = "A"
MANIFEST_FILENAME = "generation_manifest.json"


def template_versions_for(templates: Dict[str, str]) -> Dict[str, str]:
    """Short content hash per template string. Detects any template drift
    automatically — no manual version bumps to remember."""
    return {
        name: "sha256:" + hashlib.sha256(tpl.encode("utf-8")).hexdigest()[:16]
        for name, tpl in templates.items()
    }


def build_generation_manifest(
    config: PipelineConfig,
    generated_files: Iterable[Path],
    output_dir: Path,
    *,
    template_versions: Dict[str, str],
    kill_switch_active: bool,
    harvested_functions: Optional[List[str]] = None,
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    landing_filters: List[Dict[str, Any]] = []
    lifecycle_enrichments: List[Dict[str, Any]] = []
    for dataset_name, lcfg in (config.landing or {}).items():
        for step in lcfg.filters:
            landing_filters.append({
                "dataset": dataset_name,
                "predicate": step.parameters.get("predicate", ""),
            })
        for step in lcfg.lifecycle_enrichments:
            lifecycle_enrichments.append({
                "dataset": dataset_name,
                "config": step.parameters.get("config", {}),
            })

    checksums: Dict[str, str] = {}
    for p in generated_files:
        if not (p.exists() and p.is_file()):
            continue
        try:
            key = p.relative_to(output_dir).as_posix()
        except ValueError:
            key = p.as_posix()
        checksums[key] = _sha256_file(p)

    ts = (now or datetime.now(timezone.utc)).strftime("%Y-%m-%dT%H:%M:%SZ")
    return {
        "generated_at": ts,
        "baseline_tag": BASELINE_TAG,
        "release": RELEASE,
        "template_versions": dict(template_versions),
        "harvested_functions": list(harvested_functions or []),
        "landing_filters": landing_filters,
        "lifecycle_enrichments": lifecycle_enrichments,
        "kill_switch_active": bool(kill_switch_active),
        "file_checksums": checksums,
    }


def write_generation_manifest(manifest: Dict[str, Any], output_dir: Path) -> Path:
    path = output_dir / MANIFEST_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return path


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return f"sha256:{h.hexdigest()}"
