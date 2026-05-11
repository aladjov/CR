"""Walk generated production pipeline source for `@apply_op` call sites.

Where `exploration_scan` walks `.ipynb` cells with their per-notebook import
state, `production_scan` walks the standalone `.py` files emitted by the
renderer (one per landing dataset / bronze source / etc.). The dataset hint
comes from the file name (`landing_<dataset>.py`, `bronze_event_<source>.py`,
etc.); silver/gold/training merge multiple datasets and surface under the
`<merged>` pseudo-dataset.

The walker uses the same helper-following AST machinery as exploration: a
locally-defined function in the rendered file (e.g. `derive_temporal_columns`
that internally calls `derive_extra_datetime_features`) is resolved up to a
bounded depth. This is the "walker follows helpers" design decision — the
renderer keeps abstraction-friendly helper indirection without breaking the
audit.

Two entry points:

- `scan_production_source(source, *, file_path)`: parse one rendered file.
- `scan_generated_pipeline(pipeline_dir, *, scope)`: walk an entire generated
  pipeline directory, filtered by stage scope.
"""
from __future__ import annotations

import ast
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Optional

from .decorator import APPLY_REGISTRY, ApplyOpDescriptor
from .exploration_scan import DYNAMIC
from .kinds import ApplyOpKind
from .manifest import (
    Manifest,
    ManifestEntry,
    SourceLocation,
    fingerprint_kwargs,
)

logger = logging.getLogger(__name__)


_MAX_HELPER_DEPTH = 3
_MERGED_DATASET = "<merged>"


class AuditScope(str, Enum):
    LANDING = "landing"
    TARGET_DERIVE = "target_derive"
    BRONZE = "bronze"
    SILVER = "silver"
    GOLD = "gold"
    TRAINING = "training"
    ALL = "all"


_SCOPE_DIRECTORIES = {
    AuditScope.LANDING: ("landing",),
    AuditScope.TARGET_DERIVE: ("target_derive",),
    AuditScope.BRONZE: ("bronze",),
    AuditScope.SILVER: ("silver",),
    AuditScope.GOLD: ("gold",),
    AuditScope.TRAINING: ("training",),
    AuditScope.ALL: ("landing", "target_derive", "bronze", "silver", "gold", "training"),
}


# Dataset inference from generated file naming convention. The order matters:
# more specific suffixes (e.g. `_aggregated`) must be stripped before the
# generic `bronze_entity_<x>` rule.
_NAME_PATTERNS = (
    (re.compile(r"^landing_(.+)$"), 1),
    (re.compile(r"^bronze_event_(.+)$"), 1),
    (re.compile(r"^bronze_entity_(.+?)(?:_aggregated)?$"), 1),
)

_MERGED_PREFIXES = (
    "silver_",
    "gold_",
    "training",
    "ml_experiment",
    "run_target_derive",
    "target_derive",
)


# Conventional names emitted by the renderer templates (`landing.py.j2`,
# `bronze_event.py.j2`, etc.). The walker treats a Call to one of these
# names — even when it resolves to a locally-defined helper — as semantic
# evidence of the matched apply kind. Keep this list aligned with the
# renderer's render_<stage>() outputs; growing the list requires a tagged
# fixture exercising the new template branch.
_TEMPLATE_EMITS_KIND: dict[str, ApplyOpKind] = {
    # Landing
    "derive_feature_timestamp": ApplyOpKind.FEATURE_TIMESTAMP_DERIVE,
    "derive_label_timestamp": ApplyOpKind.LABEL_TIMESTAMP_DERIVE,
    "derive_label_available_flag": ApplyOpKind.LABEL_AVAILABLE_FLAG,
    "derive_datetime_features": ApplyOpKind.DATETIME_DERIVE,
    "apply_history_window": ApplyOpKind.TEMPORAL_LOOKBACK,
    "resolve_entity_key": ApplyOpKind.KEY_RESOLUTION,
    # Bronze
    "apply_event_aggregation": ApplyOpKind.BRONZE_AGGREGATE,
    "apply_event_aggregation_per_grid_date": ApplyOpKind.BRONZE_VALUE_COUNTS,
    # Silver
    "apply_target_label_map": ApplyOpKind.SILVER_TARGET_LABEL_MAP,
    "create_holdout_mask": ApplyOpKind.SILVER_HOLDOUT_MASK,
    "apply_derived_columns": ApplyOpKind.SILVER_DERIVED_FEATURE,
    # Gold
    "apply_transformations": ApplyOpKind.GOLD_TRANSFORMATION,
    "apply_encodings": ApplyOpKind.GOLD_ENCODING,
    "apply_scalings": ApplyOpKind.GOLD_TRANSFORMATION,
    "apply_feature_selection": ApplyOpKind.GOLD_FEATURE_SPEC_GATE,
    # Training
    "_temporal_split": ApplyOpKind.TRAINING_SPLIT,
}

# Compat-layer functions called directly from generated landing code that
# we treat as apply primitives even though they're not @apply_op-decorated
# (they live in core.compat and are intentionally backend-agnostic helpers).
_COMPAT_EMITS_KIND: dict[str, ApplyOpKind] = {
    "customer_retention.core.compat.apply_sql_predicate": ApplyOpKind.LANDING_FILTER,
}


def infer_dataset_from_path(path: Path) -> Optional[str]:
    stem = path.stem
    for prefix in _MERGED_PREFIXES:
        if stem.startswith(prefix):
            return _MERGED_DATASET
    for pattern, group in _NAME_PATTERNS:
        match = pattern.match(stem)
        if match:
            return match.group(group)
    return None


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def scan_generated_pipeline(
    pipeline_dir: Path,
    *,
    scope: AuditScope = AuditScope.ALL,
) -> Manifest:
    pipeline_dir = Path(pipeline_dir)
    entries: list[ManifestEntry] = []
    call_counter = [0]
    for subdir in _SCOPE_DIRECTORIES[scope]:
        stage_dir = pipeline_dir / subdir
        if not stage_dir.exists():
            continue
        for py_file in sorted(stage_dir.glob("*.py")):
            entries.extend(
                _scan_file(py_file, call_counter=call_counter)
            )
    return Manifest(entries=tuple(entries))


def scan_production_source(
    source: str,
    *,
    file_path: Path,
    dataset: Optional[str] = None,
    call_counter: Optional[list[int]] = None,
) -> list[ManifestEntry]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        logger.warning("production_scan: SyntaxError parsing %s; skipping", file_path)
        return []
    if dataset is None:
        dataset = infer_dataset_from_path(file_path) or "<unknown>"
    if call_counter is None:
        call_counter = [0]
    walker = _ModuleWalker(
        file_path=file_path,
        dataset=dataset,
        call_counter=call_counter,
    )
    walker.walk(tree)
    return walker.entries


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _scan_file(path: Path, *, call_counter: list[int]) -> list[ManifestEntry]:
    return scan_production_source(
        path.read_text(),
        file_path=path,
        call_counter=call_counter,
    )


@dataclass
class _Frame:
    if_stack: list[ast.AST] = field(default_factory=list)
    assigns: dict[str, ast.AST] = field(default_factory=dict)


class _ModuleWalker:
    def __init__(
        self,
        *,
        file_path: Path,
        dataset: str,
        call_counter: list[int],
    ):
        self.file_path = file_path
        self.dataset = dataset
        self.call_counter = call_counter
        self.imports: dict[str, str] = {}
        self.local_functions: dict[str, ast.FunctionDef] = {}
        self.entries: list[ManifestEntry] = []

    def walk(self, tree: ast.Module) -> None:
        self._collect_top_level(tree)
        frame = _Frame()
        self._walk_body(tree.body, frame, depth=0)
        # After the top-level walk, also descend into the bodies of all
        # locally-defined functions so apply_op calls that live inside them
        # but aren't reached from a top-level call still surface. The
        # depth-bounded recursion in `_handle_call` keeps the cost in check.
        for fn in self.local_functions.values():
            self._walk_body(fn.body, _Frame(), depth=1)

    def _collect_top_level(self, tree: ast.Module) -> None:
        for node in tree.body:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    local = alias.asname or alias.name
                    self.imports[local] = alias.name
            elif isinstance(node, ast.ImportFrom) and node.module:
                for alias in node.names:
                    local = alias.asname or alias.name
                    self.imports[local] = f"{node.module}.{alias.name}"
            elif isinstance(node, ast.FunctionDef):
                self.local_functions[node.name] = node

    def _walk_body(self, body: list[ast.stmt], frame: _Frame, depth: int) -> None:
        for node in body:
            self._walk_stmt(node, frame, depth)

    def _walk_stmt(self, node: ast.stmt, frame: _Frame, depth: int) -> None:
        if isinstance(node, ast.If):
            frame.if_stack.append(node.test)
            self._walk_body(node.body, frame, depth)
            frame.if_stack.pop()
            if node.orelse:
                self._walk_body(node.orelse, frame, depth)
        elif isinstance(node, ast.With):
            self._walk_body(node.body, frame, depth)
        elif isinstance(node, ast.For):
            self._walk_body(node.body, frame, depth)
            self._walk_body(node.orelse, frame, depth)
        elif isinstance(node, ast.While):
            frame.if_stack.append(node.test)
            self._walk_body(node.body, frame, depth)
            frame.if_stack.pop()
            self._walk_body(node.orelse, frame, depth)
        elif isinstance(node, ast.Try):
            self._walk_body(node.body, frame, depth)
            for handler in node.handlers:
                self._walk_body(handler.body, frame, depth)
            self._walk_body(node.orelse, frame, depth)
            self._walk_body(node.finalbody, frame, depth)
        elif isinstance(node, ast.Assign):
            self._record_assign(node, frame)
            self._walk_expr(node.value, frame, depth)
        elif isinstance(node, ast.AugAssign):
            self._walk_expr(node.value, frame, depth)
        elif isinstance(node, ast.Expr):
            self._walk_expr(node.value, frame, depth)
        elif isinstance(node, ast.Return):
            if node.value is not None:
                self._walk_expr(node.value, frame, depth)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            pass

    def _walk_expr(self, node: ast.expr, frame: _Frame, depth: int) -> None:
        if isinstance(node, ast.Call):
            self._handle_call(node, frame, depth)
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.expr):
                self._walk_expr(child, frame, depth)

    def _record_assign(self, node: ast.Assign, frame: _Frame) -> None:
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            frame.assigns[node.targets[0].id] = node.value

    def _handle_call(self, node: ast.Call, frame: _Frame, depth: int) -> None:
        qualname = _qualname_for_callable(node.func, self.imports)
        if qualname and qualname in APPLY_REGISTRY:
            self._record_apply_op(node, frame, APPLY_REGISTRY[qualname])
            return
        if qualname and qualname in _COMPAT_EMITS_KIND:
            self._record_synthetic(
                node, frame, _COMPAT_EMITS_KIND[qualname], qualname.rsplit(".", 1)[-1],
            )
            return
        local_name = node.func.id if isinstance(node.func, ast.Name) else None
        if local_name in _TEMPLATE_EMITS_KIND:
            self._record_synthetic(
                node, frame, _TEMPLATE_EMITS_KIND[local_name], local_name,
            )
            # Continue into the helper body so kwargs and nested framework
            # calls inside the helper still surface for kwargs comparison.
            if local_name in self.local_functions and depth < _MAX_HELPER_DEPTH:
                self._walk_body(
                    self.local_functions[local_name].body, frame, depth + 1,
                )
            return
        if (
            local_name
            and local_name in self.local_functions
            and depth < _MAX_HELPER_DEPTH
        ):
            self._walk_body(
                self.local_functions[local_name].body, frame, depth + 1,
            )

    def _record_apply_op(
        self,
        node: ast.Call,
        frame: _Frame,
        descriptor: ApplyOpDescriptor,
    ) -> None:
        kwargs = self._capture_kwargs(node, frame, descriptor.capture_kwargs)
        self._emit_entry(
            node, frame, descriptor.kind,
            descriptor.qualified_name.rsplit(".", 1)[-1], kwargs,
        )

    def _record_synthetic(
        self,
        node: ast.Call,
        frame: _Frame,
        kind: ApplyOpKind,
        component: str,
    ) -> None:
        kwargs = self._capture_kwargs(node, frame, capture_kwargs=None)
        self._emit_entry(node, frame, kind, component, kwargs)

    def _emit_entry(
        self,
        node: ast.Call,
        frame: _Frame,
        kind: ApplyOpKind,
        component: str,
        kwargs: dict[str, Any],
    ) -> None:
        gate = self._capture_gate(frame)
        if gate is not None:
            kwargs = {**kwargs, "_gate": gate}
        self.call_counter[0] += 1
        self.entries.append(
            ManifestEntry(
                dataset=self.dataset,
                kind=kind,
                kwargs_fingerprint=fingerprint_kwargs(kwargs),
                call_order=self.call_counter[0],
                source_location=SourceLocation(
                    file=self.file_path,
                    line=getattr(node, "lineno", 0),
                    component=component,
                ),
            )
        )

    def _capture_kwargs(
        self,
        node: ast.Call,
        frame: _Frame,
        capture_kwargs: Optional[set[str]],
    ) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for kw in node.keywords:
            if kw.arg is None:
                continue
            if capture_kwargs and kw.arg not in capture_kwargs:
                continue
            out[kw.arg] = _evaluate(kw.value, frame.assigns)
        return out

    def _capture_gate(self, frame: _Frame) -> Optional[str]:
        return ast.unparse(frame.if_stack[-1]) if frame.if_stack else None


def _qualname_for_callable(
    func: ast.expr,
    imports: Mapping[str, str],
) -> Optional[str]:
    if isinstance(func, ast.Name):
        return imports.get(func.id)
    if isinstance(func, ast.Attribute):
        base = _qualname_for_callable(func.value, imports)
        if base is not None:
            return f"{base}.{func.attr}"
        # Special-case: `Cls(...).method` — pick up the class qualname from
        # the constructor call and append the method name.
        if isinstance(func.value, ast.Call):
            ctor = func.value.func
            ctor_qualname = _qualname_for_callable(ctor, imports)
            if ctor_qualname is not None:
                return f"{ctor_qualname}.{func.attr}"
    return None


def _evaluate(value: ast.AST, assigns: Mapping[str, ast.AST]) -> Any:
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError, TypeError):
        pass
    if isinstance(value, ast.Name) and value.id in assigns:
        try:
            return ast.literal_eval(assigns[value.id])
        except (ValueError, SyntaxError, TypeError):
            return DYNAMIC
    return DYNAMIC


__all__ = [
    "AuditScope",
    "infer_dataset_from_path",
    "scan_generated_pipeline",
    "scan_production_source",
    "_TEMPLATE_EMITS_KIND",
    "_COMPAT_EMITS_KIND",
]
