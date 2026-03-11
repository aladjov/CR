import ast
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class AssignmentKind(str, Enum):
    DICT = "dict"
    LIST = "list"
    SCALAR = "scalar"
    COMPLEX = "complex"


@dataclass
class ParsedAssignment:
    name: str
    kind: AssignmentKind
    raw_source: str
    literal_value: Any | None
    line_start: int
    line_end: int


@dataclass
class ParsedConfigCell:
    tag_line: str
    preamble: str
    assignments: dict[str, ParsedAssignment] = field(default_factory=dict)
    trailing: str = ""
    parse_succeeded: bool = True


def _classify_node(node: ast.expr) -> tuple[AssignmentKind, Any | None]:
    try:
        value = ast.literal_eval(node)
    except (ValueError, TypeError):
        return AssignmentKind.COMPLEX, None
    if isinstance(value, dict):
        return AssignmentKind.DICT, value
    if isinstance(value, (list, tuple)):
        return AssignmentKind.LIST, value
    return AssignmentKind.SCALAR, value


def _is_user_assignment(name: str) -> bool:
    return bool(name) and not name.startswith("_")


def _extract_assign_name(node: ast.stmt) -> Optional[str]:
    if isinstance(node, ast.Assign) and len(node.targets) == 1:
        target = node.targets[0]
        if isinstance(target, ast.Name):
            return target.id
    return None


def parse_config_cell(source: str) -> ParsedConfigCell:
    lines = source.split("\n")
    tag_line = lines[0].rstrip("\n") if lines else ""
    body = "\n".join(lines[1:]) if len(lines) > 1 else ""

    try:
        tree = ast.parse(body)
    except SyntaxError:
        return ParsedConfigCell(tag_line=tag_line, preamble=body, parse_succeeded=False)

    body_lines = body.split("\n")
    assignments: dict[str, ParsedAssignment] = {}
    assignment_ranges: list[tuple[int, int, str]] = []

    for node in ast.iter_child_nodes(tree):
        name = _extract_assign_name(node)
        if not name or not _is_user_assignment(name):
            continue
        kind, literal_value = _classify_node(node.value)
        line_start = node.lineno
        line_end = node.end_lineno or node.lineno
        assignment_ranges.append((line_start, line_end, name))
        assignments[name] = ParsedAssignment(
            name=name, kind=kind, raw_source="",
            literal_value=literal_value,
            line_start=line_start, line_end=line_end,
        )

    assignment_ranges.sort(key=lambda r: r[0])
    _attach_raw_sources(body_lines, assignment_ranges, assignments)
    preamble, trailing = _extract_preamble_trailing(body_lines, assignment_ranges)

    return ParsedConfigCell(
        tag_line=tag_line, preamble=preamble,
        assignments=assignments, trailing=trailing,
    )


def _attach_raw_sources(
    body_lines: list[str],
    ranges: list[tuple[int, int, str]],
    assignments: dict[str, ParsedAssignment],
) -> None:
    for i, (start, end, name) in enumerate(ranges):
        comment_start = (ranges[i - 1][1] + 1) if i > 0 else 1
        preamble_boundary = _find_preamble_boundary(body_lines, ranges)
        if comment_start <= preamble_boundary:
            comment_start = preamble_boundary + 1
        raw_lines = body_lines[comment_start - 1:end]
        assignments[name].raw_source = "\n".join(raw_lines)


def _find_preamble_boundary(body_lines: list[str], ranges: list[tuple[int, int, str]]) -> int:
    if not ranges:
        return len(body_lines)
    first_assign_line = ranges[0][0]
    boundary = first_assign_line - 1
    while boundary > 0 and body_lines[boundary - 1].strip().startswith("#"):
        boundary -= 1
    return boundary


def _extract_preamble_trailing(
    body_lines: list[str], ranges: list[tuple[int, int, str]],
) -> tuple[str, str]:
    if not ranges:
        return "\n".join(body_lines), ""
    preamble_end = _find_preamble_boundary(body_lines, ranges)
    preamble = "\n".join(body_lines[:preamble_end])
    last_end = ranges[-1][1]
    trailing = "\n".join(body_lines[last_end:])
    return preamble, trailing
