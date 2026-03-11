from collections import Counter
from dataclasses import dataclass, field
from enum import Enum


class MergeAction(str, Enum):
    UNCHANGED = "unchanged"
    TOOK_THEIRS = "took_theirs"
    TOOK_OURS = "took_ours"
    BOTH_SAME = "both_same"
    AUTO_MERGED = "auto_merged"
    CONFLICT = "conflict"
    FROM_BASE = "from_base"
    ADDED_THEIRS = "added_theirs"
    ADDED_OURS = "added_ours"


_NO_CHANGE_ACTIONS = frozenset({MergeAction.UNCHANGED, MergeAction.FROM_BASE})


@dataclass
class CellMergeEntry:
    cell_id: str
    action: MergeAction
    message: str


@dataclass
class MergeReport:
    entries: list[CellMergeEntry] = field(default_factory=list)

    @property
    def counts(self) -> Counter:
        result = Counter({action: 0 for action in MergeAction})
        result.update(e.action for e in self.entries)
        return result

    @property
    def has_conflicts(self) -> bool:
        return any(e.action == MergeAction.CONFLICT for e in self.entries)

    @property
    def conflict_count(self) -> int:
        return sum(1 for e in self.entries if e.action == MergeAction.CONFLICT)

    @property
    def conflict_entries(self) -> list[CellMergeEntry]:
        return [e for e in self.entries if e.action == MergeAction.CONFLICT]

    @property
    def has_changes(self) -> bool:
        return any(e.action not in _NO_CHANGE_ACTIONS for e in self.entries)

    def format_summary(self) -> str:
        c = self.counts
        parts = [f"{c[a]} {a.value}" for a in MergeAction]
        return f"Merge: {', '.join(parts)}"

    def format_details(self) -> str:
        if not self.entries:
            return "No cells processed."
        lines = [self.format_summary(), ""]
        for e in self.entries:
            lines.append(f"  [{e.action.value:>16}] {e.cell_id}: {e.message}")
        return "\n".join(lines)
