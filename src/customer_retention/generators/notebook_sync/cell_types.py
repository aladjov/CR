import re
from enum import Enum
from typing import List, NamedTuple, Optional


class CellSyncType(str, Enum):
    CONFIG = "config"
    USER_CODE = "user_code"
    CODE = "code"
    CODE_SYSTEM = "code_system"
    DOC = "doc"


_TAG_PREFIX = "# @cr:"
_DOC_TAG_PREFIX = "[//]: # (cr:"
_VALID_TYPE_VALUES = {t.value for t in CellSyncType}
_TAG_PATTERN = re.compile(
    r"^# @cr:(\w+)"
    r"(?:\s+name='([^']*)')?"
    r"(?:\s+id=(\S+))?"
    r"$"
)
_DOC_TAG_PATTERN = re.compile(
    r"^\[//\]: # \(cr:(\w+)"
    r"(?:\s+name='([^']*)')?"
    r"(?:\s+id=(\S+))?"
    r"\)$"
)


class TagInfo(NamedTuple):
    cell_type: CellSyncType
    name: Optional[str]
    embedded_id: Optional[str]


def _parse_code_tag_line(line: str) -> Optional[TagInfo]:
    match = _TAG_PATTERN.match(line.rstrip("\n"))
    if not match:
        return None
    type_value = match.group(1)
    if type_value not in _VALID_TYPE_VALUES:
        return None
    return TagInfo(
        cell_type=CellSyncType(type_value),
        name=match.group(2) or None,
        embedded_id=match.group(3) or None,
    )


def _parse_doc_tag_line(line: str) -> Optional[TagInfo]:
    match = _DOC_TAG_PATTERN.match(line.rstrip("\n"))
    if not match:
        return None
    type_value = match.group(1)
    if type_value not in _VALID_TYPE_VALUES:
        return None
    return TagInfo(
        cell_type=CellSyncType(type_value),
        name=match.group(2) or None,
        embedded_id=match.group(3) or None,
    )


def _parse_tag_line(line: str) -> Optional[TagInfo]:
    return _parse_code_tag_line(line) or _parse_doc_tag_line(line)


def detect_cell_sync_type(source_lines: List[str]) -> CellSyncType:
    if not source_lines:
        return CellSyncType.CODE
    info = _parse_tag_line(source_lines[0])
    return info.cell_type if info else CellSyncType.CODE


def extract_embedded_id(source_lines: List[str]) -> Optional[str]:
    if not source_lines:
        return None
    info = _parse_tag_line(source_lines[0])
    return info.embedded_id if info else None


def extract_tag_name(source_lines: List[str]) -> Optional[str]:
    if not source_lines:
        return None
    info = _parse_tag_line(source_lines[0])
    return info.name if info else None


def has_magic_comment(source_lines: List[str]) -> bool:
    if not source_lines:
        return False
    return _parse_tag_line(source_lines[0]) is not None


def strip_magic_comment(source_lines: List[str]) -> List[str]:
    if has_magic_comment(source_lines):
        return source_lines[1:]
    return source_lines


def prepend_magic_comment(source_lines: List[str], cell_type: CellSyncType) -> List[str]:
    stripped = strip_magic_comment(source_lines)
    if cell_type == CellSyncType.DOC:
        return [f"{_DOC_TAG_PREFIX}{cell_type.value})\n", *stripped]
    return [f"{_TAG_PREFIX}{cell_type.value}\n", *stripped]
