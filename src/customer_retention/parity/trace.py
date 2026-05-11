"""Runtime tracer for the T3 audit.

When `CR_PARITY_TRACE=1` is set, the `@apply_op` decorator routes every call
through `record_call()`, which appends a `TraceRecord` to a thread-local
buffer. `flush_to_yaml(path)` serialises the buffer; `load_from_yaml(path)`
reads it back. `audit_trace(exploration_yaml, production_yaml)` diffs the
two traces and reports row-count drift exceeding a per-kind tolerance as
`GapKind.RUNTIME_DRIFT` parity gaps.

The default tolerance is 0.5% of input rows. `LIFECYCLE_ENRICH` is widened
to 5% because lifecycle doubling varies with cancellation rate. Tolerance
is configurable via `TOLERANCE_BY_KIND` if a new kind needs to relax it.

Row counts are intentionally best-effort. On native pandas we read `len(df)`
which is O(1). On Spark DataFrames we read `.count()` only when the frame
is cached (avoiding the multi-minute scan cost on uncached production
frames); otherwise we record `None` and skip drift comparison for that
record. The audit's value comes from the records we DO have, not from
forcing counts everywhere.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple

import yaml

from .decorator import active_dataset_hint
from .gaps import GapKind, ParityGap
from .kinds import ApplyOpKind
from .manifest import SourceLocation, fingerprint_kwargs

DEFAULT_TOLERANCE = 0.005

TOLERANCE_BY_KIND: dict[ApplyOpKind, float] = {
    ApplyOpKind.LIFECYCLE_ENRICH: 0.05,
}


@dataclass(frozen=True)
class TraceRecord:
    kind: ApplyOpKind
    qualified_name: str
    dataset: str
    kwargs_fingerprint: Mapping[str, Any]
    input_rows: Optional[int]
    output_rows: Optional[int]
    call_order: int


_BUFFER: threading.local = threading.local()


def _buffer_records() -> list[TraceRecord]:
    if not hasattr(_BUFFER, "records"):
        _BUFFER.records = []
    return _BUFFER.records


def get_records() -> Tuple[TraceRecord, ...]:
    return tuple(_buffer_records())


def clear_records() -> None:
    _BUFFER.records = []


def record_call(descriptor, func, args, kwargs):
    dataset = _resolve_dataset(descriptor, args, kwargs)
    captured = _capture_kwargs(descriptor, kwargs)
    input_rows = _safe_row_count(args[0]) if args else None
    result = func(*args, **kwargs)
    output_rows = _safe_row_count(result)
    records = _buffer_records()
    records.append(
        TraceRecord(
            kind=descriptor.kind,
            qualified_name=descriptor.qualified_name,
            dataset=dataset,
            kwargs_fingerprint=fingerprint_kwargs(captured),
            input_rows=input_rows,
            output_rows=output_rows,
            call_order=len(records) + 1,
        )
    )
    return result


def flush_to_yaml(path: Path) -> None:
    path = Path(path)
    records = get_records()
    payload = {
        "version": 1,
        "records": [_record_to_dict(r) for r in records],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False))
    clear_records()


def load_from_yaml(path: Path) -> Tuple[TraceRecord, ...]:
    path = Path(path)
    if not path.exists():
        return ()
    raw = yaml.safe_load(path.read_text()) or {}
    return tuple(_record_from_dict(d) for d in raw.get("records", []))


def audit_trace(
    exploration_yaml: Path,
    production_yaml: Path,
    *,
    tolerance_by_kind: Optional[Mapping[ApplyOpKind, float]] = None,
):
    from .audit import AuditOutcome
    tolerances = dict(TOLERANCE_BY_KIND)
    if tolerance_by_kind:
        tolerances.update(tolerance_by_kind)
    exp = load_from_yaml(exploration_yaml)
    prod = load_from_yaml(production_yaml)
    gaps = _diff_traces(exp, prod, tolerances)
    return AuditOutcome(gaps=tuple(gaps))


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _resolve_dataset(descriptor, args: tuple, kwargs: dict) -> str:
    hint = active_dataset_hint()
    if hint is not None:
        return hint
    if descriptor.dataset_kwarg:
        value = kwargs.get(descriptor.dataset_kwarg)
        if isinstance(value, str):
            return value
    return "<unknown>"


def _capture_kwargs(descriptor, kwargs: dict) -> dict:
    if descriptor.capture_kwargs is None:
        return dict(kwargs)
    return {k: v for k, v in kwargs.items() if k in descriptor.capture_kwargs}


def _safe_row_count(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, tuple) and value:
        value = value[0]
    try:
        from customer_retention.core.compat import (
            _is_native_spark_df,
            _is_spark_pandas,
        )
    except ImportError:
        _is_native_spark_df = lambda _: False  # noqa: E731
        _is_spark_pandas = lambda _: False  # noqa: E731
    if _is_native_spark_df(value):
        return _spark_row_count_if_cheap(value)
    if _is_spark_pandas(value):
        return _spark_row_count_if_cheap(_to_spark(value))
    try:
        return len(value)
    except (TypeError, AttributeError):
        return None


def _spark_row_count_if_cheap(spark_df: Any) -> Optional[int]:
    """Read `.count()` only when the frame is cached; otherwise return None
    so we don't spend minutes scanning a large frame for a sanity check."""
    if spark_df is None:
        return None
    is_cached = getattr(spark_df, "is_cached", False)
    if not is_cached:
        return None
    try:
        return int(spark_df.count())
    except Exception:
        return None


def _to_spark(df: Any) -> Any:
    try:
        from customer_retention.core.compat import as_spark_df
        return as_spark_df(df)
    except Exception:
        return None


def _record_to_dict(r: TraceRecord) -> dict:
    return {
        "kind": r.kind.value,
        "qualified_name": r.qualified_name,
        "dataset": r.dataset,
        "kwargs_fingerprint": _to_yaml_safe(dict(r.kwargs_fingerprint)),
        "input_rows": r.input_rows,
        "output_rows": r.output_rows,
        "call_order": r.call_order,
    }


def _record_from_dict(d: Mapping[str, Any]) -> TraceRecord:
    return TraceRecord(
        kind=ApplyOpKind(d["kind"]),
        qualified_name=d["qualified_name"],
        dataset=d["dataset"],
        kwargs_fingerprint=dict(d.get("kwargs_fingerprint") or {}),
        input_rows=d.get("input_rows"),
        output_rows=d.get("output_rows"),
        call_order=d.get("call_order", 0),
    )


def _to_yaml_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _to_yaml_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_yaml_safe(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _diff_traces(
    exploration: tuple[TraceRecord, ...],
    production: tuple[TraceRecord, ...],
    tolerances: Mapping[ApplyOpKind, float],
) -> list[ParityGap]:
    def _key(r: TraceRecord) -> Tuple[str, ApplyOpKind]:
        return (r.dataset, r.kind)

    exp_by_key: dict = {}
    for r in exploration:
        exp_by_key.setdefault(_key(r), []).append(r)
    prod_by_key: dict = {}
    for r in production:
        prod_by_key.setdefault(_key(r), []).append(r)

    gaps: list[ParityGap] = []
    all_keys = set(exp_by_key) | set(prod_by_key)
    for dataset, kind in sorted(all_keys, key=lambda k: (k[0], k[1].value)):
        exp_recs = exp_by_key.get((dataset, kind), [])
        prod_recs = prod_by_key.get((dataset, kind), [])
        if not exp_recs:
            gaps.append(_missing_side_gap(
                kind, dataset, prod_recs, side="production_only"
            ))
            continue
        if not prod_recs:
            gaps.append(_missing_side_gap(
                kind, dataset, exp_recs, side="exploration_only"
            ))
            continue
        gap = _drift_gap_for_pair(kind, dataset, exp_recs, prod_recs, tolerances)
        if gap is not None:
            gaps.append(gap)
    return gaps


def _missing_side_gap(
    kind: ApplyOpKind,
    dataset: str,
    present_recs: list[TraceRecord],
    *,
    side: str,
) -> ParityGap:
    if side == "production_only":
        return ParityGap(
            gap_kind=GapKind.PRODUCTION_ONLY,
            dataset=dataset,
            op_kind=kind,
            exploration_location=None,
            production_location=SourceLocation(
                file=Path(present_recs[0].qualified_name),
                line=0,
            ),
            detail=(
                f"Runtime trace: production invoked {kind.name} for "
                f"{dataset!r} ({len(present_recs)} call(s)) but exploration "
                "did not."
            ),
        )
    return ParityGap(
        gap_kind=GapKind.EXPLORATION_ONLY,
        dataset=dataset,
        op_kind=kind,
        exploration_location=SourceLocation(
            file=Path(present_recs[0].qualified_name),
            line=0,
        ),
        production_location=None,
        detail=(
            f"Runtime trace: exploration invoked {kind.name} for "
            f"{dataset!r} ({len(present_recs)} call(s)) but production "
            "did not."
        ),
    )


def _drift_gap_for_pair(
    kind: ApplyOpKind,
    dataset: str,
    exp_recs: list[TraceRecord],
    prod_recs: list[TraceRecord],
    tolerances: Mapping[ApplyOpKind, float],
) -> Optional[ParityGap]:
    exp_out = _sum_outputs(exp_recs)
    prod_out = _sum_outputs(prod_recs)
    if exp_out is None or prod_out is None:
        return None
    tolerance = tolerances.get(kind, DEFAULT_TOLERANCE)
    if exp_out == 0 and prod_out == 0:
        return None
    denom = max(abs(exp_out), 1)
    delta_ratio = abs(prod_out - exp_out) / denom
    if delta_ratio <= tolerance:
        return None
    return ParityGap(
        gap_kind=GapKind.RUNTIME_DRIFT,
        dataset=dataset,
        op_kind=kind,
        exploration_location=SourceLocation(
            file=Path(exp_recs[0].qualified_name), line=0,
        ),
        production_location=SourceLocation(
            file=Path(prod_recs[0].qualified_name), line=0,
        ),
        detail=(
            f"Runtime row-count drift on {kind.name} for {dataset!r}: "
            f"exploration={exp_out} production={prod_out} "
            f"(delta={delta_ratio:.1%}, tolerance={tolerance:.1%})"
        ),
    )


def _sum_outputs(records: list[TraceRecord]) -> Optional[int]:
    counts = [r.output_rows for r in records if r.output_rows is not None]
    if not counts:
        return None
    return sum(counts)


# Sentinel for fixture helpers in tests
@dataclass
class _BufferShim:
    """Tiny shim so tests can poke records into the thread-local buffer."""
    records: list = field(default_factory=list)


__all__ = [
    "DEFAULT_TOLERANCE",
    "TOLERANCE_BY_KIND",
    "TraceRecord",
    "audit_trace",
    "clear_records",
    "flush_to_yaml",
    "get_records",
    "load_from_yaml",
    "record_call",
]
