from customer_retention.generators.notebook_sync.sync_report import (
    CellSyncEntry,
    SyncAction,
    SyncReport,
)


class TestSyncAction:

    def test_all_actions_exist(self):
        assert SyncAction.UNCHANGED
        assert SyncAction.UPDATED
        assert SyncAction.PRESERVED
        assert SyncAction.ADDED
        assert SyncAction.REMOVED
        assert SyncAction.USER_ADDED_KEPT


class TestCellSyncEntry:

    def test_creation(self):
        entry = CellSyncEntry(cell_id="nb00-001", action=SyncAction.UPDATED, message="code updated")
        assert entry.cell_id == "nb00-001"
        assert entry.action == SyncAction.UPDATED
        assert entry.message == "code updated"


class TestSyncReport:

    def _sample_entries(self):
        return [
            CellSyncEntry("c1", SyncAction.UNCHANGED, "no change"),
            CellSyncEntry("c2", SyncAction.UPDATED, "code updated"),
            CellSyncEntry("c3", SyncAction.PRESERVED, "config kept"),
            CellSyncEntry("c4", SyncAction.ADDED, "new cell"),
            CellSyncEntry("c5", SyncAction.REMOVED, "orphaned"),
            CellSyncEntry("c6", SyncAction.USER_ADDED_KEPT, "user cell"),
            CellSyncEntry("c7", SyncAction.UNCHANGED, "no change 2"),
        ]

    def test_counts(self):
        report = SyncReport(self._sample_entries())
        counts = report.counts
        assert counts[SyncAction.UNCHANGED] == 2
        assert counts[SyncAction.UPDATED] == 1
        assert counts[SyncAction.PRESERVED] == 1
        assert counts[SyncAction.ADDED] == 1
        assert counts[SyncAction.REMOVED] == 1
        assert counts[SyncAction.USER_ADDED_KEPT] == 1

    def test_counts_empty_report(self):
        report = SyncReport([])
        for action in SyncAction:
            assert report.counts[action] == 0

    def test_format_summary_includes_counts(self):
        report = SyncReport(self._sample_entries())
        summary = report.format_summary()
        assert "2 unchanged" in summary
        assert "1 updated" in summary
        assert "1 preserved" in summary
        assert "1 added" in summary
        assert "1 removed" in summary

    def test_format_details_includes_all_entries(self):
        report = SyncReport(self._sample_entries())
        details = report.format_details()
        assert "c1" in details
        assert "c2" in details
        assert "c5" in details

    def test_format_summary_empty(self):
        report = SyncReport([])
        summary = report.format_summary()
        assert "0 unchanged" in summary

    def test_has_changes(self):
        report = SyncReport(self._sample_entries())
        assert report.has_changes is True

    def test_has_changes_false_when_all_unchanged(self):
        entries = [
            CellSyncEntry("c1", SyncAction.UNCHANGED, ""),
            CellSyncEntry("c2", SyncAction.UNCHANGED, ""),
        ]
        report = SyncReport(entries)
        assert report.has_changes is False

    def test_has_changes_empty(self):
        report = SyncReport([])
        assert report.has_changes is False

    def test_source_preview_defaults_to_none(self):
        entry = CellSyncEntry("c1", SyncAction.REMOVED, "orphaned")
        assert entry.source_preview is None

    def test_source_preview_can_be_set(self):
        entry = CellSyncEntry("c1", SyncAction.REMOVED, "orphaned", source_preview="import os\nprint('hi')")
        assert entry.source_preview == "import os\nprint('hi')"

    def test_format_details_includes_source_preview(self):
        entries = [
            CellSyncEntry("c1", SyncAction.REMOVED, "orphaned", source_preview="import os\nprint('hi')"),
        ]
        report = SyncReport(entries)
        details = report.format_details()
        assert "import os" in details
        assert "print('hi')" in details

    def test_removed_entries(self):
        entries = self._sample_entries()
        report = SyncReport(entries)
        removed = report.removed_entries
        assert len(removed) == 1
        assert removed[0].cell_id == "c5"

    def test_has_removals(self):
        entries = self._sample_entries()
        report = SyncReport(entries)
        assert report.has_removals is True

    def test_has_removals_false_when_none_removed(self):
        entries = [
            CellSyncEntry("c1", SyncAction.UNCHANGED, "no change"),
            CellSyncEntry("c2", SyncAction.UPDATED, "code updated"),
        ]
        report = SyncReport(entries)
        assert report.has_removals is False

    def test_format_removal_warning(self):
        entries = [
            CellSyncEntry("c1", SyncAction.REMOVED, "orphaned", source_preview="import os\nprint('hi')"),
            CellSyncEntry("c2", SyncAction.REMOVED, "orphaned", source_preview="x = 1"),
        ]
        report = SyncReport(entries)
        warning = report.format_removal_warning()
        assert "c1" in warning
        assert "c2" in warning
        assert "import os" in warning
        assert "x = 1" in warning

    def test_format_removal_warning_empty_when_no_removals(self):
        entries = [CellSyncEntry("c1", SyncAction.UNCHANGED, "no change")]
        report = SyncReport(entries)
        assert report.format_removal_warning() == ""
