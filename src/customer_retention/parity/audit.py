"""Top-level audit orchestrators.

`audit_landing(engagement_dir, pipeline_dir)` is the T0 entry point invoked
by `-1_parity_contract.ipynb` at the start of every Databricks job: it
scans NB00–NB09 for `@apply_op` call sites, scans the generated landing
scripts, diffs the two manifests, and returns a structured `AuditOutcome`.
An empty diff lets the job proceed; a non-empty diff halts it via the
JSON exit payload Databricks interprets as a task failure.

`audit_pipeline(...)` is the T1 entry point invoked from NB10 after
exploration completes — same machinery but scoped to bronze / silver /
gold / training. `audit_trace(...)` is the T3 entry point (Phase 6).

The outcome is the only stable contract for callers: read `has_gaps`,
`format_summary()`, `format_report()`, `to_failed_json()`.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

from .exploration_scan import scan_exploration_manifest
from .gaps import ParityGap, diff_manifests
from .manifest import Manifest
from .production_scan import AuditScope, scan_generated_pipeline


@dataclass(frozen=True)
class AuditOutcome:
    gaps: Tuple[ParityGap, ...]
    exploration_manifest: Optional[Manifest] = field(default=None)
    production_manifest: Optional[Manifest] = field(default=None)
    scope: Optional[AuditScope] = field(default=None)

    @property
    def has_gaps(self) -> bool:
        return bool(self.gaps)

    @property
    def exit_code(self) -> int:
        return 1 if self.has_gaps else 0

    def format_summary(self) -> str:
        if self.has_gaps:
            return (
                f"PARITY AUDIT FAILED: {len(self.gaps)} gap"
                f"{'s' if len(self.gaps) != 1 else ''} detected"
            )
        return "PARITY AUDIT PASS: exploration and production manifests align"

    def format_report(self) -> str:
        if not self.has_gaps:
            return self.format_summary() + "\n"
        lines = [self.format_summary(), ""]
        for i, gap in enumerate(self.gaps, 1):
            lines.append(f"--- gap {i}/{len(self.gaps)} ---")
            lines.append(gap.format())
            lines.append("")
        return "\n".join(lines)

    def to_failed_json(self) -> str:
        return json.dumps({
            "status": "failed" if self.has_gaps else "passed",
            "gap_count": len(self.gaps),
            "scope": self.scope.value if self.scope else None,
            "gaps": [
                {
                    "gap_kind": gap.gap_kind.value,
                    "dataset": gap.dataset,
                    "op_kind": gap.op_kind.name if gap.op_kind else None,
                    "detail": gap.detail,
                    "exploration_location": (
                        gap.exploration_location.format()
                        if gap.exploration_location else None
                    ),
                    "production_location": (
                        gap.production_location.format()
                        if gap.production_location else None
                    ),
                }
                for gap in self.gaps
            ],
        })


def audit_landing(
    *,
    engagement_dir: Path,
    pipeline_dir: Path,
    notebook_glob: str = "*.ipynb",
) -> AuditOutcome:
    return _audit(
        engagement_dir=Path(engagement_dir),
        pipeline_dir=Path(pipeline_dir),
        scope=AuditScope.LANDING,
        notebook_glob=notebook_glob,
    )


def audit_pipeline(
    *,
    engagement_dir: Path,
    pipeline_dir: Path,
    scope: AuditScope = AuditScope.ALL,
    notebook_glob: str = "*.ipynb",
) -> AuditOutcome:
    return _audit(
        engagement_dir=Path(engagement_dir),
        pipeline_dir=Path(pipeline_dir),
        scope=scope,
        notebook_glob=notebook_glob,
    )


def _audit(
    *,
    engagement_dir: Path,
    pipeline_dir: Path,
    scope: AuditScope,
    notebook_glob: str,
) -> AuditOutcome:
    notebooks = sorted(engagement_dir.glob(notebook_glob))
    exploration = scan_exploration_manifest(notebooks)
    production = scan_generated_pipeline(pipeline_dir, scope=scope)
    gaps = diff_manifests(exploration, production)
    return AuditOutcome(
        gaps=tuple(gaps),
        exploration_manifest=exploration,
        production_manifest=production,
        scope=scope,
    )


__all__ = [
    "AuditOutcome",
    "audit_landing",
    "audit_pipeline",
]
