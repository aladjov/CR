"""Resolve the canonical ``datasets`` mapping used by NB10 codegen and any
downstream replay machinery.

Operator escape hatch: ``DATASETS_ORIGINAL_FALLBACK`` overlays
``ProjectContext.original_datasets`` instead of being shadowed by it. The
prior framework reference cell (NB10 ``0f3cc762``) used an `if / elif`
ladder that consulted ``DATASETS_ORIGINAL_FALLBACK`` only when
``_ctx.original_datasets`` was empty. That caused operator-pasted UC
overrides for *one* dataset (e.g. an interactive ``global_temp.*`` view
left over from a prior NB00 attempt) to be silently dropped whenever NB00
had also recorded *any* original-dataset entries — and on real
engagements, NB00 always records at least one. Documented in
`docs/sps_nb10_runtime_patches_v1.md` §7.3.

The overlay strategy here is symmetric with how `replay_registered
_landing_steps` already treats `original_datasets` as the upstream of last
resort: the operator-supplied dict wins on key collision, the project
context fills any gaps. Diagnostic output names every overlaid key so the
operator can confirm the override took effect from the cell output alone.
"""
from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Tuple


def resolve_original_datasets(
    project_context: Optional[Any],
    fallback: Optional[Mapping[str, Any]],
) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """Build the canonical ``datasets`` mapping plus a per-key provenance map.

    Resolution rule:

    * Start from ``project_context.original_datasets`` (NB00 cell ``f1eb641c``
      stashes the live ``datasets`` dict here at exploration time).
    * Overlay ``fallback`` on top — an operator-supplied dict whose entries
      *take precedence* on key collision. Used to swap a stale UC view for
      a fresh upstream path without re-running NB00.

    Returns ``(datasets, provenance)`` where ``provenance[name]`` is one of
    ``"project_context"`` or ``"DATASETS_ORIGINAL_FALLBACK"``.

    Raises ``RuntimeError`` when both inputs are empty so the caller still
    surfaces the original "no datasets to resolve" failure shape.
    """
    ctx_datasets: Dict[str, Any] = {}
    if project_context is not None:
        ctx_datasets = dict(getattr(project_context, "original_datasets", None) or {})
    overrides: Dict[str, Any] = dict(fallback or {})
    if not ctx_datasets and not overrides:
        raise RuntimeError(
            "Cannot resolve original datasets: both ProjectContext"
            ".original_datasets and DATASETS_ORIGINAL_FALLBACK are empty. "
            "Re-run NB00 (cell f1eb641c stashes `_namespace.original_datasets"
            " = dict(datasets)`), or paste the NB00 § 0.2 `datasets = {...}` "
            "literal into NB10's `DATASETS_ORIGINAL_FALLBACK`."
        )
    merged: Dict[str, Any] = {**ctx_datasets, **overrides}
    provenance: Dict[str, str] = {}
    for name in merged:
        if name in overrides:
            provenance[name] = "DATASETS_ORIGINAL_FALLBACK"
        else:
            provenance[name] = "project_context"
    return merged, provenance
