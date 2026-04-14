from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from customer_retention.stages.modeling.feature_spec import FeatureSpec

logger = logging.getLogger(__name__)


class ParitySeverity(str, Enum):
    INFO = "INFO"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass
class ParityFinding:
    check_id: str
    severity: ParitySeverity
    message: str
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "check_id": self.check_id,
            "severity": self.severity.value,
            "message": self.message,
            "details": dict(self.details),
        }


@dataclass
class ParityReport:
    exploration_run_id: str
    production_run_type: str
    findings: List[ParityFinding] = field(default_factory=list)

    @property
    def critical(self) -> List[ParityFinding]:
        return [f for f in self.findings if f.severity == ParitySeverity.CRITICAL]

    @property
    def passed(self) -> bool:
        return not self.critical

    def to_dict(self) -> dict:
        return {
            "exploration_run_id": self.exploration_run_id,
            "production_run_type": self.production_run_type,
            "passed": self.passed,
            "findings": [f.to_dict() for f in self.findings],
        }

    def save(self, path: Path) -> None:
        p = path if isinstance(path, Path) else Path(str(path))
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2))


def _load_json(path: Path) -> Optional[dict]:
    p = path if isinstance(path, Path) else Path(str(path))
    if not p.exists():
        return None
    return json.loads(p.read_text())


def _compare_feature_set(
    spec: FeatureSpec,
    actual_features: List[str],
    comparison_name: str,
    findings: List[ParityFinding],
) -> None:
    spec_set = set(spec.selected_features)
    actual_set = set(actual_features)
    missing = sorted(spec_set - actual_set)
    extra = sorted(actual_set - spec_set)
    if not missing and not extra:
        return
    findings.append(ParityFinding(
        check_id=f"feature_set_identical:{comparison_name}",
        severity=ParitySeverity.CRITICAL,
        message=(
            f"{comparison_name} feature set differs from spec: "
            f"{len(missing)} missing, {len(extra)} extra."
        ),
        details={
            "missing_from_actual": missing[:20],
            "extra_in_actual": extra[:20],
            "spec_count": len(spec.selected_features),
            "actual_count": len(actual_features),
        },
    ))


def _class_proportion_delta(
    exp_diag: Optional[dict],
    prod_diag: Optional[dict],
    findings: List[ParityFinding],
) -> None:
    if exp_diag is None or prod_diag is None:
        return
    exp_rate = exp_diag.get("class_proportion") or exp_diag.get("label_rate_test")
    prod_rate = prod_diag.get("label_rate_test") or prod_diag.get("class_proportion")
    if exp_rate in (None, 0) or prod_rate is None:
        return
    ratio = prod_rate / exp_rate if exp_rate else float("inf")
    # Guard against upside ratio when prod_rate > exp_rate.
    ratio = max(ratio, 1.0 / ratio) if ratio > 0 else ratio
    if ratio > 5.0:
        severity = ParitySeverity.HIGH
    elif ratio > 2.0:
        severity = ParitySeverity.MEDIUM
    else:
        return
    findings.append(ParityFinding(
        check_id="class_proportion_delta",
        severity=severity,
        message=(
            f"Label rate drift: exploration={exp_rate:.4f}, production={prod_rate:.4f} "
            f"(ratio {ratio:.1f}x)."
        ),
        details={"exploration_rate": exp_rate, "production_rate": prod_rate, "ratio": ratio},
    ))


def _cv_mean_delta(
    spec: FeatureSpec,
    prod_diag: Optional[dict],
    findings: List[ParityFinding],
) -> None:
    if prod_diag is None:
        return
    prod_cv_mean = None
    prod_cv = prod_diag.get("cv_results") or {}
    best = prod_diag.get("best_model_name")
    if best and best in prod_cv:
        prod_cv_mean = prod_cv[best].get("cv_mean")
    if prod_cv_mean is None:
        return
    spec_cv_mean = float(spec.verdict.cv_mean)
    delta = abs(spec_cv_mean - float(prod_cv_mean))
    findings.append(ParityFinding(
        check_id="cv_mean_delta",
        severity=ParitySeverity.INFO,
        message=f"CV mean: exploration={spec_cv_mean:.4f}, production={prod_cv_mean:.4f} (|Δ|={delta:.4f}).",
        details={"exploration_cv_mean": spec_cv_mean, "production_cv_mean": float(prod_cv_mean), "delta": delta},
    ))


def _verdict_consistency(
    spec: FeatureSpec,
    prod_diag: Optional[dict],
    findings: List[ParityFinding],
) -> None:
    if prod_diag is None:
        return
    prod_status = prod_diag.get("verdict", {}).get("status") if isinstance(prod_diag.get("verdict"), dict) else None
    if prod_status is None:
        return
    if prod_status != spec.verdict.status:
        findings.append(ParityFinding(
            check_id="verdict_consistency",
            severity=ParitySeverity.MEDIUM,
            message=f"Verdict mismatch: exploration={spec.verdict.status!r}, production={prod_status!r}.",
            details={"exploration_verdict": spec.verdict.status, "production_verdict": prod_status},
        ))


def compare_runs(
    spec: FeatureSpec,
    exploration_diagnostics: Optional[dict] = None,
    production_diagnostics: Optional[dict] = None,
) -> ParityReport:
    findings: List[ParityFinding] = []

    if exploration_diagnostics is not None:
        exp_features = list(exploration_diagnostics.get("feature_names") or [])
        _compare_feature_set(spec, exp_features, "exploration", findings)

    if production_diagnostics is not None:
        prod_features = list(production_diagnostics.get("feature_names") or [])
        _compare_feature_set(spec, prod_features, "production", findings)

    _class_proportion_delta(exploration_diagnostics, production_diagnostics, findings)
    _cv_mean_delta(spec, production_diagnostics, findings)
    _verdict_consistency(spec, production_diagnostics, findings)

    run_type = "production"
    if production_diagnostics is not None:
        run_type = production_diagnostics.get("run_type", "production")

    return ParityReport(
        exploration_run_id=spec.exploration_run_id,
        production_run_type=run_type,
        findings=findings,
    )


def compare_runs_from_paths(
    spec_path: Path,
    exploration_diagnostics_path: Optional[Path] = None,
    production_diagnostics_path: Optional[Path] = None,
) -> ParityReport:
    spec = FeatureSpec.load(spec_path)
    exp_diag = _load_json(exploration_diagnostics_path) if exploration_diagnostics_path else None
    prod_diag = _load_json(production_diagnostics_path) if production_diagnostics_path else None
    return compare_runs(spec, exp_diag, prod_diag)
