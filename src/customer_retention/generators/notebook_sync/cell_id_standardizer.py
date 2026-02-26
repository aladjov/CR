import copy
import re
from typing import List, Tuple

import nbformat

_FILENAME_PATTERN = re.compile(r"^(-?\d+)((?:[a-z](?:_[a-z])?)?)_")


def notebook_prefix_from_filename(filename: str) -> str:
    stem = filename.replace(".ipynb", "")
    m = _FILENAME_PATTERN.match(stem)
    if not m:
        raise ValueError(f"Cannot extract prefix from filename: {filename}")
    number = m.group(1).replace("-", "m")
    suffix = m.group(2).replace("_", "")
    return f"nb{number}{suffix}"


def standardize_cell_ids(
    notebook: nbformat.NotebookNode, filename: str
) -> nbformat.NotebookNode:
    prefix = notebook_prefix_from_filename(filename)
    result = copy.deepcopy(notebook)
    for seq, cell in enumerate(result.cells, start=1):
        cell.id = f"{prefix}-{seq:03d}"
    return result


def _ensure_source_lines(cell) -> List[str]:
    src = cell.source
    if isinstance(src, str):
        return src.splitlines(keepends=True)
    return list(src)


def split_blended_cell(
    cell: nbformat.NotebookNode,
    split_after_line: int,
    config_id: str,
    code_id: str,
) -> Tuple[nbformat.NotebookNode, nbformat.NotebookNode]:
    lines = _ensure_source_lines(cell)

    config_cell = nbformat.v4.new_code_cell(source="".join(lines[:split_after_line]))
    config_cell.id = config_id
    config_cell.source = lines[:split_after_line]
    config_cell.metadata = copy.deepcopy(cell.metadata)
    config_cell.outputs = []
    config_cell.execution_count = None

    code_cell = nbformat.v4.new_code_cell(source="".join(lines[split_after_line:]))
    code_cell.id = code_id
    code_cell.source = lines[split_after_line:]
    code_cell.metadata = copy.deepcopy(cell.metadata)
    code_cell.outputs = list(cell.outputs) if cell.outputs else []
    code_cell.execution_count = cell.execution_count

    return config_cell, code_cell
