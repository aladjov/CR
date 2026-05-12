"""Walk exploration notebooks to find every `@apply_op`-decorated call site.

The walker treats a notebook as an ordered list of code cells whose
imports accumulate cell-by-cell. For each cell:

- Imports add `local_name -> qualified_name` entries to the per-notebook
  symbol table.
- Locally-defined functions are recorded so a Call to them can be resolved
  to an inline body walk (bounded depth, prevents helper-indirection from
  hiding apply primitives).
- `Call` nodes whose resolved qualname is in `APPLY_REGISTRY` produce a
  `ManifestEntry`.
- The dataset hint comes from the nearest enclosing `with apply_context(
  dataset="X"):`; if absent, falls back to a literal `dataset_kwarg=` arg;
  otherwise `<unknown>`.
- The gate condition comes from the nearest enclosing `if`; captured as
  the unparsed expression and recorded in `kwargs_fingerprint["_gate"]`.
- Kwargs are captured via literal eval, with a single-hop name-resolution
  pass for the assign-then-call pattern. Anything more complex becomes
  the `DYNAMIC` sentinel — the diff suppresses kwargs comparison for those
  entries while keeping kind comparison.
"""
from __future__ import annotations

import ast
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

from customer_retention.generators.notebook_sync.cell_types import extract_embedded_id

from .decorator import APPLY_REGISTRY, ApplyOpDescriptor
from .manifest import (
    Manifest,
    ManifestEntry,
    SourceLocation,
    fingerprint_kwargs,
)

logger = logging.getLogger(__name__)

DYNAMIC = "<dynamic>"

_MAX_HELPER_DEPTH = 3


def scan_exploration_manifest(
    notebook_paths: Iterable[Path],
    *,
    framework_modules: Optional[Iterable[str]] = None,
) -> Manifest:
    if framework_modules:
        import importlib
        for mod in framework_modules:
            importlib.import_module(mod)
    entries: list[ManifestEntry] = []
    call_counter = [0]
    for path in notebook_paths:
        scanner = _NotebookScanner(Path(path), call_counter)
        entries.extend(scanner.scan())
    return Manifest(entries=tuple(entries))


# ---------------------------------------------------------------------------
# Notebook scanner
# ---------------------------------------------------------------------------


@dataclass
class _Cell:
    cell_id: Optional[str]
    source: str
    lineno_offset: int  # for file-wide line numbers (unused, kept for future)


@dataclass
class _NotebookScanner:
    path: Path
    call_counter: list[int]
    imports: dict[str, str] = field(default_factory=dict)
    local_functions: dict[str, ast.FunctionDef] = field(default_factory=dict)
    entries: list[ManifestEntry] = field(default_factory=list)

    def scan(self) -> list[ManifestEntry]:
        cells = _read_code_cells(self.path)
        for cell in cells:
            self._scan_cell(cell)
        return self.entries

    def _scan_cell(self, cell: _Cell) -> None:
        try:
            tree = ast.parse(cell.source)
        except SyntaxError:
            logger.warning(
                "exploration_scan: skipping cell %s in %s due to SyntaxError",
                cell.cell_id, self.path.name,
            )
            return
        _CellWalker(self, cell, tree).walk()


# ---------------------------------------------------------------------------
# Per-cell AST walker
# ---------------------------------------------------------------------------


@dataclass
class _Frame:
    dataset_stack: list[str] = field(default_factory=list)
    if_stack: list[ast.AST] = field(default_factory=list)
    assigns: dict[str, ast.AST] = field(default_factory=dict)


class _CellWalker:
    def __init__(self, scanner: _NotebookScanner, cell: _Cell, tree: ast.Module):
        self.scanner = scanner
        self.cell = cell
        self.tree = tree

    def walk(self) -> None:
        self._collect_imports_and_functions(self.tree)
        frame = _Frame()
        self._walk_body(self.tree.body, frame, depth=0)

    def _collect_imports_and_functions(self, tree: ast.Module) -> None:
        for node in tree.body:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    local = alias.asname or alias.name
                    self.scanner.imports[local] = alias.name
            elif isinstance(node, ast.ImportFrom) and node.module:
                for alias in node.names:
                    local = alias.asname or alias.name
                    self.scanner.imports[local] = f"{node.module}.{alias.name}"
            elif isinstance(node, ast.FunctionDef):
                self.scanner.local_functions[node.name] = node

    def _walk_body(self, body: list[ast.stmt], frame: _Frame, depth: int) -> None:
        for node in body:
            self._walk_stmt(node, frame, depth)

    def _walk_stmt(self, node: ast.stmt, frame: _Frame, depth: int) -> None:
        if isinstance(node, ast.With):
            self._walk_with(node, frame, depth)
        elif isinstance(node, ast.If):
            frame.if_stack.append(node.test)
            self._walk_body(node.body, frame, depth)
            frame.if_stack.pop()
            if node.orelse:
                self._walk_body(node.orelse, frame, depth)
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
            # Local definitions don't fire apply_op calls until invoked
            pass

    def _walk_with(self, node: ast.With, frame: _Frame, depth: int) -> None:
        pushed = False
        for item in node.items:
            dataset = _extract_apply_context_dataset(item, self.scanner.imports)
            if dataset is not None:
                frame.dataset_stack.append(dataset)
                pushed = True
                break
        try:
            self._walk_body(node.body, frame, depth)
        finally:
            if pushed:
                frame.dataset_stack.pop()

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
        qualname = self._resolve_call_qualname(node)
        descriptor = _lookup_in_registry(qualname)
        if descriptor is not None:
            self._record_apply_op(node, frame, descriptor)
            return
        # Local helper fallback — walk into the helper body if known
        local_name = _call_root_name(node)
        if local_name and local_name in self.scanner.local_functions and depth < _MAX_HELPER_DEPTH:
            self._walk_body(
                self.scanner.local_functions[local_name].body, frame, depth + 1
            )

    def _resolve_call_qualname(self, node: ast.Call) -> Optional[str]:
        return _qualname_for_callable(node.func, self.scanner.imports)

    def _record_apply_op(
        self,
        node: ast.Call,
        frame: _Frame,
        descriptor: ApplyOpDescriptor,
    ) -> None:
        kwargs = self._capture_kwargs(node, frame, descriptor)
        gate = self._capture_gate(frame)
        if gate is not None:
            kwargs = {**kwargs, "_gate": gate}
        dataset = self._resolve_dataset(node, frame, descriptor, kwargs)
        self.scanner.call_counter[0] += 1
        self.scanner.entries.append(
            ManifestEntry(
                dataset=dataset,
                kind=descriptor.kind,
                kwargs_fingerprint=fingerprint_kwargs(kwargs),
                call_order=self.scanner.call_counter[0],
                source_location=SourceLocation(
                    file=self.scanner.path,
                    line=getattr(node, "lineno", 0),
                    cell_id=self.cell.cell_id,
                ),
            )
        )

    def _capture_kwargs(
        self,
        node: ast.Call,
        frame: _Frame,
        descriptor: ApplyOpDescriptor,
    ) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for kw in node.keywords:
            if kw.arg is None:
                continue
            if descriptor.capture_kwargs and kw.arg not in descriptor.capture_kwargs:
                continue
            out[kw.arg] = _evaluate(kw.value, frame.assigns)
        return out

    def _capture_gate(self, frame: _Frame) -> Optional[str]:
        if not frame.if_stack:
            return None
        return ast.unparse(frame.if_stack[-1])

    def _resolve_dataset(
        self,
        node: ast.Call,
        frame: _Frame,
        descriptor: ApplyOpDescriptor,
        kwargs: Mapping[str, Any],
    ) -> str:
        if frame.dataset_stack:
            return frame.dataset_stack[-1]
        if descriptor.dataset_kwarg:
            value = kwargs.get(descriptor.dataset_kwarg)
            if isinstance(value, str):
                return value
        return "<unknown>"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_code_cells(path: Path) -> list[_Cell]:
    nb = json.loads(path.read_text())
    out: list[_Cell] = []
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = cell.get("source", [])
        if isinstance(source, list):
            source_lines = source
            source_text = "".join(source)
        else:
            source_lines = source.splitlines(keepends=True)
            source_text = source
        cell_id = extract_embedded_id(source_lines) or cell.get("id")
        out.append(_Cell(cell_id=cell_id, source=source_text, lineno_offset=0))
    return out


def _lookup_in_registry(qualname: Optional[str]) -> Optional[ApplyOpDescriptor]:
    """Resolve a notebook-side qualname to an `ApplyOpDescriptor`.

    Strategy:

    1. Exact match against `APPLY_REGISTRY` keys.
    2. Re-export match: when the notebook imports `derive_X` from a parent
       package (e.g. `customer_retention.stages.profiling`) but the
       framework decorates it inside a submodule (`...profiling.time_window_aggregator`),
       the resolved qualname is `.profiling.derive_X` while the registered
       key is `.profiling.time_window_aggregator.derive_X`. We match when
       the registered prefix is a *deeper* path that shares the same
       package prefix as the resolved qualname.

    Returns `None` if no unique match is found (ambiguous matches are also
    None — better silence than false-positive).
    """
    if not qualname:
        return None
    descriptor = APPLY_REGISTRY.get(qualname)
    if descriptor is not None:
        return descriptor
    last_segment = qualname.rsplit(".", 1)[-1]
    notebook_prefix = qualname.rsplit(".", 1)[0] if "." in qualname else ""
    candidates: list[ApplyOpDescriptor] = []
    for key, desc in APPLY_REGISTRY.items():
        if not key.endswith("." + last_segment):
            continue
        registered_prefix = key.rsplit(".", 1)[0]
        if registered_prefix == notebook_prefix or registered_prefix.startswith(notebook_prefix + "."):
            candidates.append(desc)
    if len(candidates) == 1:
        return candidates[0]
    return None


def _qualname_for_callable(
    func: ast.expr,
    imports: Mapping[str, str],
) -> Optional[str]:
    if isinstance(func, ast.Name):
        return imports.get(func.id)
    if isinstance(func, ast.Attribute):
        base = _qualname_for_callable(func.value, imports)
        if base is None:
            return None
        return f"{base}.{func.attr}"
    return None


def _call_root_name(node: ast.Call) -> Optional[str]:
    return node.func.id if isinstance(node.func, ast.Name) else None


def _extract_apply_context_dataset(
    item: ast.withitem,
    imports: Mapping[str, str],
) -> Optional[str]:
    call = item.context_expr
    if not isinstance(call, ast.Call):
        return None
    qualname = _qualname_for_callable(call.func, imports)
    if qualname != "customer_retention.parity.apply_context":
        return None
    for kw in call.keywords:
        if kw.arg == "dataset" and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
            return kw.value.value
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
