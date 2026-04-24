"""Parse ``docs/sps_table_descriptions.md`` into ``ColumnDescriptionRow`` seeds.

The source file mixes two column formats:

- ``NAME (type): description``           (most tables)
- ``NAME: type — description``           (subscription table, em-dash)

Table FQNs appear either as a header line ending with ``:`` or inline in
prose (``The table <fqn> contains ...``). The parser recognizes both,
skips unrelated content (the dataset dict at the bottom of the file, blank
lines, the title), and emits one ``ColumnDescriptionRow`` per recognized
column line with ``source='imported_from_md'``.

``unit``, ``polarity``, ``pii_class``, ``value_examples`` stay NULL — those
fields require human judgment and are filled in via a subsequent review
pass. ``business_name`` is a deterministic title-cased fallback so the
dashboard renders something readable before any manual curation.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, List, Optional

if TYPE_CHECKING:  # pragma: no cover
    from customer_retention.stages.causal.column_descriptions_writer import ColumnDescriptionRow

_FQN_HEADER_RE = re.compile(r"^([a-z][\w]*\.[a-z][\w]*\.[a-z][\w]*):\s*$")
_FQN_IN_PROSE_RE = re.compile(r"\b([a-z][\w]*\.[a-z][\w]*\.[a-z][\w]*)\b")
_COLUMN_PAREN_RE = re.compile(r"^([A-Za-z][\w]*)\s*\(([^)]+)\):\s*(.+)$")
_COLUMN_DASH_RE = re.compile(r"^([A-Za-z][\w]*):\s*([\w(),\s]+?)\s+[—\-]\s+(.+)$")


def parse_table_descriptions_md(path: Path | str) -> List["ColumnDescriptionRow"]:
    """Parse the SPS tables markdown into column-description seed rows."""
    return list(_iter_rows(Path(path).read_text().splitlines()))


def _iter_rows(lines: Iterable[str]) -> Iterable["ColumnDescriptionRow"]:
    current_fqn: Optional[str] = None
    for raw in lines:
        line = raw.strip()
        if not line or _is_dataset_dict_line(line):
            continue
        fqn = _detect_fqn(line)
        if fqn is not None:
            current_fqn = fqn
            continue
        if current_fqn is None:
            continue
        parsed = _parse_column_line(line)
        if parsed is None:
            continue
        from customer_retention.stages.causal.column_descriptions_writer import (
            ColumnDescriptionRow,
        )

        column_name, _dtype, description = parsed
        catalog, schema, table = current_fqn.split(".")
        yield ColumnDescriptionRow(
            table=table,
            column_name=column_name,
            catalog=catalog,
            schema=schema,
            business_name=_auto_business_name(column_name),
            business_definition=description.strip(),
            source="imported_from_md",
        )


def _detect_fqn(line: str) -> Optional[str]:
    header = _FQN_HEADER_RE.match(line)
    if header:
        return header.group(1)
    if line.lower().startswith("the table "):
        inline = _FQN_IN_PROSE_RE.search(line)
        if inline:
            return inline.group(1)
    return None


def _parse_column_line(line: str) -> Optional[tuple[str, str, str]]:
    paren = _COLUMN_PAREN_RE.match(line)
    if paren:
        return paren.group(1), paren.group(2).strip(), paren.group(3).strip()
    dash = _COLUMN_DASH_RE.match(line)
    if dash:
        return dash.group(1), dash.group(2).strip(), dash.group(3).strip()
    return None


def _is_dataset_dict_line(line: str) -> bool:
    """Skip lines that belong to the dataset-config dict at the bottom of the file."""
    return (
        line.startswith("#")
        or line.startswith('"')
        or line.startswith("datasets")
        or line in ("{", "}")
    )


def _auto_business_name(column_name: str) -> str:
    if "_" in column_name:
        return " ".join(word.capitalize() for word in column_name.split("_"))
    return column_name
