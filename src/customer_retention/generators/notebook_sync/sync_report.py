from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import List


class SyncAction(str, Enum):
    UNCHANGED = "unchanged"
    UPDATED = "updated"
    PRESERVED = "preserved"
    ADDED = "added"
    REMOVED = "removed"
    USER_ADDED_KEPT = "user_added_kept"


@dataclass
class CellSyncEntry:
    cell_id: str
    action: SyncAction
    message: str
    source_preview: str | None = None


@dataclass
class SyncReport:
    entries: List[CellSyncEntry] = field(default_factory=list)

    @property
    def counts(self) -> Counter:
        result = Counter({action: 0 for action in SyncAction})
        result.update(e.action for e in self.entries)
        return result

    @property
    def has_changes(self) -> bool:
        return any(e.action != SyncAction.UNCHANGED for e in self.entries)

    @property
    def removed_entries(self) -> List[CellSyncEntry]:
        return [e for e in self.entries if e.action == SyncAction.REMOVED]

    @property
    def has_removals(self) -> bool:
        return any(e.action == SyncAction.REMOVED for e in self.entries)

    def format_summary(self) -> str:
        c = self.counts
        parts = [f"{c[a]} {a.value}" for a in SyncAction]
        return f"Sync: {', '.join(parts)}"

    def format_details(self) -> str:
        if not self.entries:
            return "No cells processed."
        lines = [self.format_summary(), ""]
        for e in self.entries:
            lines.append(f"  [{e.action.value:>16}] {e.cell_id}: {e.message}")
            if e.source_preview:
                for preview_line in e.source_preview.split("\n"):
                    lines.append(f"                      | {preview_line}")
        return "\n".join(lines)

    def format_removal_warning(self) -> str:
        removed = self.removed_entries
        if not removed:
            return ""
        lines = [f"{len(removed)} cell(s) will be REMOVED:", ""]
        for e in removed:
            lines.append(f"  Cell {e.cell_id}:")
            if e.source_preview:
                for preview_line in e.source_preview.split("\n"):
                    lines.append(f"    | {preview_line}")
            lines.append("")
        return "\n".join(lines)
