from __future__ import annotations

import copy
from typing import Dict, List, Tuple

import nbformat

from customer_retention.generators.notebook_sync.cell_types import CellSyncType, detect_cell_sync_type
from customer_retention.generators.notebook_sync.sync_engine import _source_lines, apply_embedded_ids

from .config_merger import merge_config_cell
from .conflict import wrap_conflict
from .merge_report import CellMergeEntry, MergeAction, MergeReport


class NotebookMergeEngine:

    def __init__(self, theirs_label: str = "theirs", ours_label: str = "ours"):
        self.theirs_label = theirs_label
        self.ours_label = ours_label

    def merge(
        self, base_nb: nbformat.NotebookNode,
        theirs_nb: nbformat.NotebookNode, ours_nb: nbformat.NotebookNode,
    ) -> Tuple[nbformat.NotebookNode, MergeReport]:
        apply_embedded_ids(base_nb)
        apply_embedded_ids(theirs_nb)
        apply_embedded_ids(ours_nb)

        base_map: Dict[str, nbformat.NotebookNode] = {c.id: c for c in base_nb.cells}
        theirs_map: Dict[str, nbformat.NotebookNode] = {c.id: c for c in theirs_nb.cells}
        ours_map: Dict[str, nbformat.NotebookNode] = {c.id: c for c in ours_nb.cells}
        base_ids = set(base_map)

        theirs_extra = self._build_extra_positions(theirs_nb.cells, base_ids)
        ours_extra = self._build_extra_positions(ours_nb.cells, base_ids)

        entries: List[CellMergeEntry] = []
        merged_cells: List[nbformat.NotebookNode] = []

        for base_cell in base_nb.cells:
            cid = base_cell.id
            cell, entry = self._merge_cell(base_cell, theirs_map.get(cid), ours_map.get(cid))
            merged_cells.append(cell)
            entries.append(entry)
            self._insert_extra_after(cid, theirs_extra, ours_extra, merged_cells, entries)

        self._insert_remaining_extra(theirs_extra, ours_extra, merged_cells, entries)
        self._handle_non_base_cells(theirs_map, ours_map, base_ids, theirs_extra, ours_extra, merged_cells, entries)

        result = copy.deepcopy(base_nb)
        result.cells = merged_cells
        return result, MergeReport(entries)

    def _merge_cell(
        self, base_cell, theirs_cell, ours_cell,
    ) -> Tuple[nbformat.NotebookNode, CellMergeEntry]:
        cid = base_cell.id
        if base_cell.cell_type == "markdown":
            return self._take_base(base_cell, cid)
        cell_type = detect_cell_sync_type(_source_lines(base_cell))
        if cell_type == CellSyncType.CODE or cell_type == CellSyncType.DOC:
            return self._take_base(base_cell, cid)
        t_cell = theirs_cell or base_cell
        o_cell = ours_cell or base_cell
        return self._three_way_merge(base_cell, t_cell, o_cell, cid, cell_type)

    def _take_base(self, base_cell, cid):
        cell = copy.deepcopy(base_cell)
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None
        return cell, CellMergeEntry(cid, MergeAction.FROM_BASE, "framework cell from base")

    def _three_way_merge(self, base_cell, theirs_cell, ours_cell, cid, cell_type):
        b_src = "".join(_source_lines(base_cell))
        t_src = "".join(_source_lines(theirs_cell))
        o_src = "".join(_source_lines(ours_cell))

        if b_src == t_src and b_src == o_src:
            return copy.deepcopy(base_cell), CellMergeEntry(cid, MergeAction.UNCHANGED, "unchanged")
        if b_src == t_src:
            return copy.deepcopy(ours_cell), CellMergeEntry(cid, MergeAction.TOOK_OURS, "ours changed")
        if b_src == o_src:
            return copy.deepcopy(theirs_cell), CellMergeEntry(cid, MergeAction.TOOK_THEIRS, "theirs changed")
        if t_src == o_src:
            return copy.deepcopy(theirs_cell), CellMergeEntry(cid, MergeAction.BOTH_SAME, "both changed identically")

        if cell_type == CellSyncType.CONFIG:
            return self._structural_merge(base_cell, b_src, t_src, o_src, cid)
        return self._textual_conflict(base_cell, t_src, o_src, cid)

    def _structural_merge(self, base_cell, b_src, t_src, o_src, cid):
        merged_source, action, conflicts = merge_config_cell(
            b_src, t_src, o_src, self.theirs_label, self.ours_label,
        )
        cell = copy.deepcopy(base_cell)
        cell.source = merged_source
        cell.outputs = []
        cell.execution_count = None
        msg = f"auto-merged ({', '.join(conflicts)})" if conflicts else "auto-merged"
        return cell, CellMergeEntry(cid, action, msg)

    def _textual_conflict(self, base_cell, t_src, o_src, cid):
        tag_line = t_src.split("\n", 1)[0]
        t_body = t_src.split("\n", 1)[1] if "\n" in t_src else ""
        o_body = o_src.split("\n", 1)[1] if "\n" in o_src else ""
        conflict = wrap_conflict(t_body, o_body, self.theirs_label, self.ours_label)
        cell = copy.deepcopy(base_cell)
        cell.source = f"{tag_line}\n{conflict}"
        cell.outputs = []
        cell.execution_count = None
        return cell, CellMergeEntry(cid, MergeAction.CONFLICT, "both changed differently")

    def _build_extra_positions(self, cells, base_ids):
        positions: Dict[str | None, List[nbformat.NotebookNode]] = {}
        last_anchor = None
        for cell in cells:
            if cell.id in base_ids:
                last_anchor = cell.id
            elif self._should_keep(cell):
                positions.setdefault(last_anchor, []).append(cell)
        return positions

    def _should_keep(self, cell) -> bool:
        if cell.cell_type == "markdown":
            return True
        cell_type = detect_cell_sync_type(_source_lines(cell))
        return cell_type in (CellSyncType.CONFIG, CellSyncType.USER_CODE)

    def _insert_extra_after(self, anchor_id, theirs_extra, ours_extra, merged_cells, entries):
        seen_ids: set[str] = set()
        for cell in theirs_extra.pop(anchor_id, []):
            merged_cells.append(copy.deepcopy(cell))
            entries.append(CellMergeEntry(cell.id, MergeAction.ADDED_THEIRS, "added by theirs"))
            seen_ids.add(cell.id)
        for cell in ours_extra.pop(anchor_id, []):
            if cell.id not in seen_ids:
                merged_cells.append(copy.deepcopy(cell))
                entries.append(CellMergeEntry(cell.id, MergeAction.ADDED_OURS, "added by ours"))

    def _insert_remaining_extra(self, theirs_extra, ours_extra, merged_cells, entries):
        seen_ids = {c.id for c in merged_cells}
        for cells in list(theirs_extra.values()):
            for cell in cells:
                if cell.id not in seen_ids:
                    merged_cells.append(copy.deepcopy(cell))
                    entries.append(CellMergeEntry(cell.id, MergeAction.ADDED_THEIRS, "added by theirs"))
                    seen_ids.add(cell.id)
        theirs_extra.clear()
        for cells in list(ours_extra.values()):
            for cell in cells:
                if cell.id not in seen_ids:
                    merged_cells.append(copy.deepcopy(cell))
                    entries.append(CellMergeEntry(cell.id, MergeAction.ADDED_OURS, "added by ours"))
                    seen_ids.add(cell.id)
        ours_extra.clear()

    def _handle_non_base_cells(self, theirs_map, ours_map, base_ids, theirs_extra, ours_extra, merged_cells, entries):
        existing_ids = {c.id for c in merged_cells}
        for cid, cell in theirs_map.items():
            if cid not in base_ids and cid not in existing_ids and self._should_keep(cell):
                merged_cells.append(copy.deepcopy(cell))
                entries.append(CellMergeEntry(cid, MergeAction.ADDED_THEIRS, "added by theirs"))
                existing_ids.add(cid)
        for cid, cell in ours_map.items():
            if cid not in base_ids and cid not in existing_ids and self._should_keep(cell):
                merged_cells.append(copy.deepcopy(cell))
                entries.append(CellMergeEntry(cid, MergeAction.ADDED_OURS, "added by ours"))
                existing_ids.add(cid)
