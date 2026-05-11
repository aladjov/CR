"""Parity audit subsystem.

Spec: docs/wiki/Parity-Audit.md.

Reconciles the exploration manifest (NB00-NB09 invoke sites) against the
production manifest (renderer-emitted call sites) before any data flows.
Catches structurally-detectable divergences pre-execution.

Phase 1 surface: decorator + manifest + gaps. Later phases add the AST
walker, schedule parser, and runtime tracer.
"""
from __future__ import annotations

from .decorator import (
    APPLY_REGISTRY,
    ApplyOpDescriptor,
    active_dataset_hint,
    apply_context,
    apply_op,
    trace_active,
)
from .exploration_scan import DYNAMIC, scan_exploration_manifest
from .gaps import GapKind, ParityGap, diff_manifests
from .kinds import ApplyOpKind
from .manifest import (
    Manifest,
    ManifestEntry,
    SourceLocation,
    fingerprint_kwargs,
)

__all__ = [
    "APPLY_REGISTRY",
    "ApplyOpDescriptor",
    "ApplyOpKind",
    "DYNAMIC",
    "GapKind",
    "Manifest",
    "ManifestEntry",
    "ParityGap",
    "SourceLocation",
    "active_dataset_hint",
    "apply_context",
    "apply_op",
    "diff_manifests",
    "fingerprint_kwargs",
    "scan_exploration_manifest",
    "trace_active",
]
