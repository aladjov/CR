from enum import Enum
from typing import List


class CellSyncType(str, Enum):
    CONFIG = "config"
    USER_CODE = "user_code"
    CODE = "code"


_TAG_PREFIX = "# @cr:"
_VALID_TAGS = {f"{_TAG_PREFIX}{t.value}" for t in CellSyncType}


def detect_cell_sync_type(source_lines: List[str]) -> CellSyncType:
    if not source_lines:
        return CellSyncType.CODE
    tag = source_lines[0].rstrip("\n")
    if tag in _VALID_TAGS:
        return CellSyncType(tag[len(_TAG_PREFIX):])
    return CellSyncType.CODE


def has_magic_comment(source_lines: List[str]) -> bool:
    if not source_lines:
        return False
    return source_lines[0].rstrip("\n") in _VALID_TAGS


def strip_magic_comment(source_lines: List[str]) -> List[str]:
    if has_magic_comment(source_lines):
        return source_lines[1:]
    return source_lines


def prepend_magic_comment(source_lines: List[str], cell_type: CellSyncType) -> List[str]:
    stripped = strip_magic_comment(source_lines)
    return [f"{_TAG_PREFIX}{cell_type.value}\n", *stripped]
