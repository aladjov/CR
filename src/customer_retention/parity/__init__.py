"""Parity audit subsystem.

Spec: docs/wiki/Parity-Audit.md.

Reconciles the exploration manifest (NB00-NB09 invoke sites) against the
production manifest (renderer-emitted call sites) before any data flows.
Catches structurally-detectable divergences pre-execution.

Phase 1 surface: decorator + manifest + gaps. Later phases add the AST
walker, schedule parser, and runtime tracer.
"""
from __future__ import annotations

from .audit import AuditOutcome, audit_landing, audit_pipeline
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
from .production_scan import (
    AuditScope,
    infer_dataset_from_path,
    scan_generated_pipeline,
    scan_production_source,
)
from .schedule import (
    GeneratedStage,
    JobSchedule,
    ScheduledNotebook,
    parse_inner_schedule,
    parse_outer_schedule,
)
from .trace import (
    DEFAULT_TOLERANCE,
    TOLERANCE_BY_KIND,
    TraceRecord,
    audit_trace,
    clear_records,
    flush_to_yaml,
    get_records,
    load_from_yaml,
)

__all__ = [
    "APPLY_REGISTRY",
    "ApplyOpDescriptor",
    "ApplyOpKind",
    "AuditOutcome",
    "AuditScope",
    "DYNAMIC",
    "GapKind",
    "GeneratedStage",
    "JobSchedule",
    "Manifest",
    "ManifestEntry",
    "ParityGap",
    "ScheduledNotebook",
    "SourceLocation",
    "active_dataset_hint",
    "apply_context",
    "apply_op",
    "DEFAULT_TOLERANCE",
    "TOLERANCE_BY_KIND",
    "TraceRecord",
    "audit_landing",
    "audit_pipeline",
    "audit_trace",
    "clear_records",
    "flush_to_yaml",
    "get_records",
    "load_from_yaml",
    "diff_manifests",
    "fingerprint_kwargs",
    "infer_dataset_from_path",
    "parse_inner_schedule",
    "parse_outer_schedule",
    "scan_exploration_manifest",
    "scan_generated_pipeline",
    "scan_production_source",
    "trace_active",
]
