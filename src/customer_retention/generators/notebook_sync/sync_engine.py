import copy
import uuid
from typing import Dict, List, Tuple

import nbformat

from .cell_types import CellSyncType, detect_cell_sync_type, extract_embedded_id
from .sync_report import CellSyncEntry, SyncAction, SyncReport


def _source_lines(cell) -> List[str]:
    src = cell.source
    if isinstance(src, str):
        return src.splitlines(keepends=True)
    return list(src)


def _sources_equal(a, b) -> bool:
    return "".join(_source_lines(a)) == "".join(_source_lines(b))


def apply_embedded_ids(nb: nbformat.NotebookNode) -> None:
    for cell in nb.get("cells", []):
        embedded = extract_embedded_id(_source_lines(cell))
        if embedded:
            cell["id"] = embedded
        elif "id" not in cell:
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
        if cell_type in (CellSyncType.CONFIG, CellSyncType.USER_CODE):
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
        return cell_type in (CellSyncType.CONFIG, CellSyncType.USER_CODE)

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
