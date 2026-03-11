from __future__ import annotations

from .config_parser import AssignmentKind, ParsedAssignment, ParsedConfigCell, parse_config_cell
from .conflict import wrap_conflict
from .merge_report import MergeAction


def merge_config_cell(
    base_source: str, theirs_source: str, ours_source: str,
    theirs_label: str = "theirs", ours_label: str = "ours",
) -> tuple[str, MergeAction, list[str]]:
    base_parsed = parse_config_cell(base_source)
    theirs_parsed = parse_config_cell(theirs_source)
    ours_parsed = parse_config_cell(ours_source)

    if not (base_parsed.parse_succeeded and theirs_parsed.parse_succeeded and ours_parsed.parse_succeeded):
        t_body = theirs_source.split("\n", 1)[1] if "\n" in theirs_source else ""
        o_body = ours_source.split("\n", 1)[1] if "\n" in ours_source else ""
        conflict_body = wrap_conflict(t_body, o_body, theirs_label, ours_label)
        merged = f"{base_parsed.tag_line}\n{conflict_body}"
        return merged, MergeAction.CONFLICT, ["parse failure — textual conflict"]

    all_names = _ordered_names(base_parsed, theirs_parsed, ours_parsed)
    merged_lines: list[str] = []
    conflicts: list[str] = []
    has_any_conflict = False

    for name in all_names:
        b = base_parsed.assignments.get(name)
        t = theirs_parsed.assignments.get(name)
        o = ours_parsed.assignments.get(name)
        result_source, is_conflict = _merge_assignment(b, t, o, theirs_label, ours_label)
        merged_lines.append(result_source)
        if is_conflict:
            has_any_conflict = True
            conflicts.append(name)

    tag = base_parsed.tag_line
    preamble = base_parsed.preamble
    trailing = base_parsed.trailing
    body_parts = [p for p in [preamble, "\n".join(merged_lines), trailing] if p.strip()]
    merged_source = tag + "\n" + "\n".join(body_parts)

    if has_any_conflict:
        return merged_source, MergeAction.CONFLICT, conflicts
    return merged_source, MergeAction.AUTO_MERGED, []


def _ordered_names(base: ParsedConfigCell, theirs: ParsedConfigCell, ours: ParsedConfigCell) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for source in (base, theirs, ours):
        for name in source.assignments:
            if name not in seen:
                seen.add(name)
                result.append(name)
    return result


def _merge_assignment(
    base: ParsedAssignment | None, theirs: ParsedAssignment | None, ours: ParsedAssignment | None,
    theirs_label: str, ours_label: str,
) -> tuple[str, bool]:
    if theirs is None and ours is None:
        return base.raw_source if base else "", False
    if base is None and theirs and ours is None:
        return theirs.raw_source, False
    if base is None and ours and theirs is None:
        return ours.raw_source, False
    if base is None and theirs and ours:
        if theirs.raw_source == ours.raw_source:
            return theirs.raw_source, False
        return _merge_by_kind(theirs, theirs, ours, theirs_label, ours_label)
    if theirs is None and base and ours:
        return ours.raw_source, False
    if ours is None and base and theirs:
        return theirs.raw_source, False

    t_raw, o_raw, b_raw = theirs.raw_source, ours.raw_source, base.raw_source
    if b_raw == t_raw:
        return o_raw, False
    if b_raw == o_raw:
        return t_raw, False
    if t_raw == o_raw:
        return t_raw, False
    return _merge_by_kind(base, theirs, ours, theirs_label, ours_label)


def _merge_by_kind(
    base: ParsedAssignment, theirs: ParsedAssignment, ours: ParsedAssignment,
    theirs_label: str, ours_label: str,
) -> tuple[str, bool]:
    if theirs.kind == AssignmentKind.DICT and ours.kind == AssignmentKind.DICT:
        return _merge_dict(base, theirs, ours, theirs_label, ours_label)
    if theirs.kind == AssignmentKind.LIST and ours.kind == AssignmentKind.LIST:
        return _merge_list(base, theirs, ours)
    if theirs.kind == AssignmentKind.SCALAR and ours.kind == AssignmentKind.SCALAR:
        if theirs.literal_value == ours.literal_value:
            return theirs.raw_source, False
        return wrap_conflict(theirs.raw_source, ours.raw_source, theirs_label, ours_label), True
    return wrap_conflict(theirs.raw_source, ours.raw_source, theirs_label, ours_label), True


def _merge_dict(
    base: ParsedAssignment, theirs: ParsedAssignment, ours: ParsedAssignment,
    theirs_label: str, ours_label: str,
) -> tuple[str, bool]:
    t_val = theirs.literal_value or {}
    o_val = ours.literal_value or {}
    all_keys = list(dict.fromkeys(list(t_val) + list(o_val)))
    merged: dict = {}

    for key in all_keys:
        in_t, in_o = key in t_val, key in o_val
        if in_t and in_o:
            if t_val[key] == o_val[key]:
                merged[key] = t_val[key]
            else:
                conflict_src = wrap_conflict(theirs.raw_source, ours.raw_source, theirs_label, ours_label)
                return conflict_src, True
        elif in_t:
            merged[key] = t_val[key]
        else:
            merged[key] = o_val[key]

    return f"{theirs.name} = {repr(merged)}", False


def _merge_list(
    base: ParsedAssignment, theirs: ParsedAssignment, ours: ParsedAssignment,
) -> tuple[str, bool]:
    t_val = list(theirs.literal_value or [])
    o_val = list(ours.literal_value or [])
    seen: set = set()
    merged: list = []
    for item in t_val + o_val:
        key = repr(item)
        if key not in seen:
            seen.add(key)
            merged.append(item)
    return f"{theirs.name} = {repr(merged)}", False
