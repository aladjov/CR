"""Parity gaps and the diff that produces them.

A `ParityGap` is one structured divergence between exploration and production
manifests for a single dataset. `diff_manifests` computes the full set on a
pair of manifests.

The diff classifies each divergence into one of:
- PRODUCTION_ONLY: production emits a kind exploration skips
- EXPLORATION_ONLY: exploration emits a kind production skips
- KWARGS_MISMATCH: both emit but fingerprints differ (kwargs marked
  `<dynamic>` on either side suppress this comparison)
- ORDER_MISMATCH: both emit the same kinds but in different call order
- ORPHAN_*: scheduling violations (caller-supplied to the diff)
- SCHEDULE_TOPOLOGY: inner-DAG dependency violation
- RUNTIME_DRIFT: T3 row-count delta exceeds tolerance
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

from .kinds import ApplyOpKind
from .manifest import Manifest, SourceLocation

_DYNAMIC_SENTINEL = "<dynamic>"


class GapKind(str, Enum):
    PRODUCTION_ONLY = "production_only"
    EXPLORATION_ONLY = "exploration_only"
    KWARGS_MISMATCH = "kwargs_mismatch"
    ORDER_MISMATCH = "order_mismatch"
    ORPHAN_NOTEBOOK = "orphan_notebook"
    ORPHAN_REGISTRATION = "orphan_registration"
    ORPHAN_PRODUCTION = "orphan_production"
    SCHEDULE_TOPOLOGY = "schedule_topology"
    RUNTIME_DRIFT = "runtime_drift"


@dataclass(frozen=True)
class ParityGap:
    gap_kind: GapKind
    dataset: str
    op_kind: Optional[ApplyOpKind]
    exploration_location: Optional[SourceLocation]
    production_location: Optional[SourceLocation]
    detail: str

    def format(self) -> str:
        op_label = self.op_kind.name if self.op_kind is not None else "—"
        sides = []
        if self.exploration_location is not None:
            sides.append(f"exploration={self.exploration_location.format()}")
        if self.production_location is not None:
            sides.append(f"production={self.production_location.format()}")
        sides_str = " | ".join(sides) if sides else ""
        return (
            f"[{self.gap_kind.name}] dataset={self.dataset} op={op_label}\n"
            f"  {self.detail}\n"
            f"  {sides_str}"
        )


_UNKNOWN_DATASET = "<unknown>"


def diff_manifests(
    exploration: Manifest,
    production: Manifest,
) -> List[ParityGap]:
    gaps: List[ParityGap] = []
    datasets = exploration.datasets() | production.datasets()
    # Kinds exploration recorded under `<unknown>` (no apply_context block
    # wrapping the call site) act as a wildcard: they cover the same kind
    # for any production dataset, so a PRODUCTION_ONLY gap on dataset D is
    # only emitted when exploration neither has the kind for D nor for the
    # unknown bucket. Without this rule, every cell in NB01 that calls
    # `derive_extra_datetime_features` without an `apply_context(dataset=...)`
    # wrapper produces N false-positive `PRODUCTION_ONLY` gaps (one per
    # dataset). The trade-off is losing per-dataset attribution for those
    # ops, which most engagements treat as uniform across datasets anyway.
    wildcard_exp_kinds = {e.kind for e in exploration.by_dataset(_UNKNOWN_DATASET)}
    for dataset in sorted(datasets):
        if dataset == _UNKNOWN_DATASET:
            continue
        gaps.extend(_diff_for_dataset(dataset, exploration, production, wildcard_exp_kinds))
    return gaps


def _diff_for_dataset(
    dataset: str,
    exploration: Manifest,
    production: Manifest,
    wildcard_exp_kinds,
) -> List[ParityGap]:
    exp_entries = exploration.by_dataset(dataset)
    prod_entries = production.by_dataset(dataset)
    gaps: List[ParityGap] = []
    gaps.extend(_gaps_for_missing_kinds(dataset, exp_entries, prod_entries, wildcard_exp_kinds))
    gaps.extend(_gaps_for_kwargs_mismatch(dataset, exp_entries, prod_entries))
    gaps.extend(_gaps_for_order_mismatch(dataset, exp_entries, prod_entries))
    return gaps


def _gaps_for_missing_kinds(
    dataset: str,
    exp_entries: tuple,
    prod_entries: tuple,
    wildcard_exp_kinds=frozenset(),
) -> List[ParityGap]:
    dataset_exp_kinds = {e.kind for e in exp_entries}
    # Wildcards only suppress PRODUCTION_ONLY (exploration applied the op at
    # an unknown dataset hint). For EXPLORATION_ONLY we still want strict
    # dataset-scoped comparison so a real "exploration applies, production
    # skips" stays detectable.
    exp_kinds_for_prod_check = dataset_exp_kinds | set(wildcard_exp_kinds)
    prod_kinds = {e.kind for e in prod_entries}
    gaps: List[ParityGap] = []
    for kind in prod_kinds - exp_kinds_for_prod_check:
        prod_entry = next(e for e in prod_entries if e.kind is kind)
        gaps.append(
            ParityGap(
                gap_kind=GapKind.PRODUCTION_ONLY,
                dataset=dataset,
                op_kind=kind,
                exploration_location=None,
                production_location=prod_entry.source_location,
                detail=(
                    f"Production emits {kind.name} for {dataset!r}, "
                    "but exploration skips it."
                ),
            )
        )
    for kind in dataset_exp_kinds - prod_kinds:
        exp_entry = next(e for e in exp_entries if e.kind is kind)
        gaps.append(
            ParityGap(
                gap_kind=GapKind.EXPLORATION_ONLY,
                dataset=dataset,
                op_kind=kind,
                exploration_location=exp_entry.source_location,
                production_location=None,
                detail=(
                    f"Exploration applies {kind.name} to {dataset!r}, "
                    "but production omits it."
                ),
            )
        )
    return gaps


def _gaps_for_kwargs_mismatch(
    dataset: str,
    exp_entries: tuple,
    prod_entries: tuple,
) -> List[ParityGap]:
    gaps: List[ParityGap] = []
    common_kinds = {e.kind for e in exp_entries} & {e.kind for e in prod_entries}
    for kind in common_kinds:
        exp = next(e for e in exp_entries if e.kind is kind)
        prod = next(e for e in prod_entries if e.kind is kind)
        delta = _kwargs_delta(exp.kwargs_fingerprint, prod.kwargs_fingerprint)
        if delta:
            gaps.append(
                ParityGap(
                    gap_kind=GapKind.KWARGS_MISMATCH,
                    dataset=dataset,
                    op_kind=kind,
                    exploration_location=exp.source_location,
                    production_location=prod.source_location,
                    detail=(
                        f"Kwargs mismatch for {kind.name} on {dataset!r}: " + delta
                    ),
                )
            )
    return gaps


def _kwargs_delta(
    exp_kw,
    prod_kw,
) -> str:
    keys = set(exp_kw.keys()) | set(prod_kw.keys())
    diffs: List[str] = []
    for key in sorted(keys):
        exp_v = exp_kw.get(key, "<missing>")
        prod_v = prod_kw.get(key, "<missing>")
        if exp_v == _DYNAMIC_SENTINEL or prod_v == _DYNAMIC_SENTINEL:
            continue
        if exp_v != prod_v:
            diffs.append(f"{key}: exploration={exp_v!r} vs production={prod_v!r}")
    return "; ".join(diffs)


def _gaps_for_order_mismatch(
    dataset: str,
    exp_entries: tuple,
    prod_entries: tuple,
) -> List[ParityGap]:
    exp_kinds_seq = [e.kind for e in exp_entries]
    prod_kinds_seq = [e.kind for e in prod_entries]
    if exp_kinds_seq == prod_kinds_seq:
        return []
    if set(exp_kinds_seq) != set(prod_kinds_seq):
        return []
    return [
        ParityGap(
            gap_kind=GapKind.ORDER_MISMATCH,
            dataset=dataset,
            op_kind=None,
            exploration_location=exp_entries[0].source_location,
            production_location=prod_entries[0].source_location,
            detail=(
                f"Apply-op sequence differs on {dataset!r}: "
                f"exploration={[k.name for k in exp_kinds_seq]} vs "
                f"production={[k.name for k in prod_kinds_seq]}"
            ),
        )
    ]
