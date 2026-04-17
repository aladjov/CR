"""Cardinal-rule validation for @cr.register functions.

Declarative and imperative lanes must not mix. A `registry.add_*` call
inside a `@cr.register` function body is a bug: the registry call runs
only at exploration time, but the function is harvested into production
— so the registration is silently dropped. This validator catches the
collision at NB10 codegen time, with a clear fix: split into two cells.

The AST walk respects local shadowing: if the function rebinds
`registry` as an argument or local variable, the `registry.add_*` call
resolves to the local name, not the singleton — and is therefore safe.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Set, Tuple

if TYPE_CHECKING:
    from .registry import RegisteredFunction


@dataclass
class ValidationError:
    function_name: str
    reason: str
    notebook_path: Optional[str] = None
    cell_id: Optional[str] = None

    def __str__(self) -> str:
        loc = ""
        if self.notebook_path:
            loc = f" ({self.notebook_path}"
            if self.cell_id:
                loc += f":{self.cell_id}"
            loc += ")"
        return f"@cr.register function '{self.function_name}'{loc}: {self.reason}"


def validate_registered_function(rf: "RegisteredFunction") -> Optional[ValidationError]:
    try:
        tree = ast.parse(rf.source)
    except SyntaxError as exc:
        return ValidationError(
            function_name=rf.name,
            reason=f"source failed to parse: {exc.msg} (line {exc.lineno})",
            notebook_path=str(rf.notebook_path) if rf.notebook_path else None,
            cell_id=rf.cell_id,
        )
    violations = _find_registry_add_calls(tree)
    if not violations:
        return None
    attr, lineno = violations[0]
    return ValidationError(
        function_name=rf.name,
        reason=_cardinal_rule_message(attr, lineno),
        notebook_path=str(rf.notebook_path) if rf.notebook_path else None,
        cell_id=rf.cell_id,
    )


def _cardinal_rule_message(attr: str, lineno: int) -> str:
    return (
        f"calls 'registry.{attr}' on line {lineno}.\n\n"
        "This mixes declarative and imperative lanes. Declarative registry "
        "calls run only at exploration time; harvesting them into production "
        "will either crash or silently lose the declaration.\n\n"
        "Fix: split into two cells.\n"
        f"  - Cell A: registry.{attr}(...) calls only (no @cr.register).\n"
        "  - Cell B: imperative function with @cr.register (no registry.* calls inside).\n\n"
        "See docs/user_extensions.md § 3."
    )


def _find_registry_add_calls(tree: ast.AST) -> List[Tuple[str, int]]:
    violations: List[Tuple[str, int]] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if "registry" in _local_names(node):
            continue
        for sub in ast.walk(node):
            if not isinstance(sub, ast.Call):
                continue
            if not isinstance(sub.func, ast.Attribute):
                continue
            val = sub.func.value
            if not isinstance(val, ast.Name) or val.id != "registry":
                continue
            if sub.func.attr.startswith("add_"):
                violations.append((sub.func.attr, sub.lineno))
    return violations


def _local_names(fn: ast.AST) -> Set[str]:
    names: Set[str] = set()
    args = getattr(fn, "args", None)
    if args is not None:
        for arg in list(args.args) + list(args.kwonlyargs) + list(args.posonlyargs):
            names.add(arg.arg)
        if args.vararg is not None:
            names.add(args.vararg.arg)
        if args.kwarg is not None:
            names.add(args.kwarg.arg)
    for node in ast.walk(fn):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                _collect_target_names(target, names)
        elif isinstance(node, ast.AnnAssign):
            _collect_target_names(node.target, names)
        elif isinstance(node, ast.AugAssign):
            _collect_target_names(node.target, names)
        elif isinstance(node, ast.For):
            _collect_target_names(node.target, names)
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            for item in node.items:
                if item.optional_vars is not None:
                    _collect_target_names(item.optional_vars, names)
    return names


def _collect_target_names(target: ast.AST, out: Set[str]) -> None:
    if isinstance(target, ast.Name):
        out.add(target.id)
    elif isinstance(target, (ast.Tuple, ast.List)):
        for elt in target.elts:
            _collect_target_names(elt, out)
    elif isinstance(target, ast.Starred):
        _collect_target_names(target.value, out)
