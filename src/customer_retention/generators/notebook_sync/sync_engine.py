import copy
import uuid
from typing import Dict, List, Optional, Tuple

import nbformat

from .cell_types import CellSyncType, detect_cell_sync_type, extract_embedded_id, extract_tag_name
from .sync_report import CellSyncEntry, SyncAction, SyncReport

_SYSTEM_CELL_ID = "cr-syspath"
_SYSTEM_CELL_NAME = "framework_path"
_SYSTEM_CELL_TEMPLATE = (
    "# @cr:code_system name='framework_path' id={cell_id}\n"
    "import sys\n"
    "\n"
    'FRAMEWORK_REPO_ROOT = "{repo_path}"\n'
    "_src = f\"{{FRAMEWORK_REPO_ROOT}}/src\"\n"
    "if _src not in sys.path:\n"
    "    sys.path.insert(0, _src)\n"
)


def _source_lines(cell) -> List[str]:
    src = cell.source
    if isinstance(src, str):
        return src.splitlines(keepends=True)
    return list(src)


def _sources_equal(a, b) -> bool:
    return "".join(_source_lines(a)) == "".join(_source_lines(b))


def _is_placeholder_id(value: str) -> bool:
    return value.startswith("<") or value.startswith("{")


def apply_embedded_ids(nb: nbformat.NotebookNode) -> None:
    for cell in nb.get("cells", []):
        embedded = extract_embedded_id(_source_lines(cell))
        if embedded and not _is_placeholder_id(embedded):
            cell["id"] = embedded
        elif "id" not in cell or (embedded and _is_placeholder_id(embedded)):
            cell["id"] = str(uuid.uuid4())[:8]


class NotebookSyncEngine:

    def sync(
        self, repo_nb: nbformat.NotebookNode, user_nb: nbformat.NotebookNode
    ) -> Tuple[nbformat.NotebookNode, SyncReport]:
        apply_embedded_ids(repo_nb)
        apply_embedded_ids(user_nb)
        repo_map: Dict[str, nbformat.NotebookNode] = {c.id: c for c in repo_nb.cells}
        user_map: Dict[str, nbformat.NotebookNode] = {c.id: c for c in user_nb.cells}
        repo_ids = set(repo_map)

        user_positions = self._build_user_positions(user_nb.cells, repo_ids)

        entries: List[CellSyncEntry] = []
        merged_cells: List[nbformat.NotebookNode] = []

        for repo_cell in repo_nb.cells:
            cid = repo_cell.id
            if cid in user_map:
                cell, entry = self._merge_matched_cell(repo_cell, user_map[cid])
            else:
                cell, entry = self._add_new_cell(repo_cell)
            merged_cells.append(cell)
            entries.append(entry)
            self._insert_user_only_cells_after(cid, user_positions, merged_cells, entries)

        self._insert_remaining_user_cells(user_positions, merged_cells, entries)
        self._record_removals(user_map, repo_ids, user_positions, entries)

        merged = copy.deepcopy(repo_nb)
        merged.cells = merged_cells
        return merged, SyncReport(entries)

    def _merge_matched_cell(
        self, repo_cell, user_cell
    ) -> Tuple[nbformat.NotebookNode, CellSyncEntry]:
        cid = repo_cell.id
        if repo_cell.cell_type == "markdown":
            return self._take_repo_markdown(repo_cell, user_cell)
        cell_type = detect_cell_sync_type(_source_lines(user_cell))
        if cell_type in (CellSyncType.CONFIG, CellSyncType.USER_CODE, CellSyncType.CODE_SYSTEM):
            return self._preserve_user_cell(repo_cell, user_cell, cid)
        return self._overwrite_code_cell(repo_cell, user_cell, cid)

    def _take_repo_markdown(self, repo_cell, user_cell):
        cell = copy.deepcopy(repo_cell)
        if _sources_equal(repo_cell, user_cell):
            return cell, CellSyncEntry(repo_cell.id, SyncAction.UNCHANGED, "markdown unchanged")
        return cell, CellSyncEntry(repo_cell.id, SyncAction.UPDATED, "markdown updated from repo")

    def _preserve_user_cell(self, repo_cell, user_cell, cid):
        cell = copy.deepcopy(user_cell)
        if _sources_equal(repo_cell, user_cell):
            return cell, CellSyncEntry(cid, SyncAction.UNCHANGED, "config/user_code unchanged")
        return cell, CellSyncEntry(cid, SyncAction.PRESERVED, "user config/code preserved")

    def _overwrite_code_cell(self, repo_cell, user_cell, cid):
        cell = copy.deepcopy(repo_cell)
        cell.outputs = []
        cell.execution_count = None
        if _sources_equal(repo_cell, user_cell):
            return cell, CellSyncEntry(cid, SyncAction.UNCHANGED, "code unchanged")
        return cell, CellSyncEntry(cid, SyncAction.UPDATED, "code updated from repo")

    def _add_new_cell(self, repo_cell):
        cell = copy.deepcopy(repo_cell)
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None
        return cell, CellSyncEntry(cell.id, SyncAction.ADDED, "new cell from repo")

    def _build_user_positions(self, user_cells, repo_ids):
        positions: Dict[str, List[nbformat.NotebookNode]] = {}
        last_anchor = None
        for cell in user_cells:
            if cell.id in repo_ids:
                last_anchor = cell.id
            elif self._should_keep_user_cell(cell):
                positions.setdefault(last_anchor, []).append(cell)
        return positions

    def _should_keep_user_cell(self, cell) -> bool:
        if cell.cell_type == "markdown":
            return True
        cell_type = detect_cell_sync_type(_source_lines(cell))
        return cell_type in (CellSyncType.CONFIG, CellSyncType.USER_CODE, CellSyncType.CODE_SYSTEM)

    def _insert_user_only_cells_after(self, anchor_id, user_positions, merged_cells, entries):
        for cell in user_positions.pop(anchor_id, []):
            merged_cells.append(copy.deepcopy(cell))
            entries.append(CellSyncEntry(cell.id, SyncAction.USER_ADDED_KEPT, "user-added cell kept"))

    def _insert_remaining_user_cells(self, user_positions, merged_cells, entries):
        for anchor, cells in list(user_positions.items()):
            for cell in cells:
                merged_cells.append(copy.deepcopy(cell))
                entries.append(CellSyncEntry(cell.id, SyncAction.USER_ADDED_KEPT, "user-added cell kept"))
            del user_positions[anchor]

    @staticmethod
    def _build_source_preview(cell, max_lines: int = 5) -> str:
        lines = _source_lines(cell)
        preview = lines[:max_lines]
        return "".join(preview).rstrip("\n")

    def _record_removals(self, user_map, repo_ids, user_positions, entries):
        for cid, cell in user_map.items():
            if cid in repo_ids:
                continue
            if not self._should_keep_user_cell(cell):
                preview = self._build_source_preview(cell)
                entries.append(CellSyncEntry(
                    cid, SyncAction.REMOVED, "orphaned cell removed",
                    source_preview=preview,
                ))

    @staticmethod
    def build_system_cell(
        framework_repo_path: str, cell_id: str = _SYSTEM_CELL_ID,
    ) -> nbformat.NotebookNode:
        source = _SYSTEM_CELL_TEMPLATE.format(repo_path=framework_repo_path, cell_id=cell_id)
        cell = nbformat.v4.new_code_cell(source=source)
        cell.id = cell_id
        return cell

    @staticmethod
    def ensure_system_cell(
        nb: nbformat.NotebookNode,
        framework_repo_path: Optional[str],
    ) -> bool:
        cells = nb.get("cells", [])
        if not framework_repo_path:
            return NotebookSyncEngine._remove_system_cells(cells)
        for i, cell in enumerate(cells):
            cell_type = detect_cell_sync_type(_source_lines(cell))
            if cell_type == CellSyncType.CODE_SYSTEM:
                if i == 0:
                    return False
                cells.insert(0, cells.pop(i))
                return True
        system_cell = NotebookSyncEngine.build_system_cell(framework_repo_path)
        nb.cells.insert(0, system_cell)
        return True

    @staticmethod
    def _is_framework_path_cell(cell: nbformat.NotebookNode) -> bool:
        lines = _source_lines(cell)
        return (detect_cell_sync_type(lines) == CellSyncType.CODE_SYSTEM
                and extract_tag_name(lines) == _SYSTEM_CELL_NAME)

    @staticmethod
    def _remove_system_cells(cells: List[nbformat.NotebookNode]) -> bool:
        removed = [c for c in cells if NotebookSyncEngine._is_framework_path_cell(c)]
        for c in removed:
            cells.remove(c)
        return len(removed) > 0
