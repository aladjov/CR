from customer_retention.generators.notebook_merge.merge_report import (
    CellMergeEntry,
    MergeAction,
    MergeReport,
)


class TestMergeAction:

    def test_all_actions_are_strings(self):
        for action in MergeAction:
            assert isinstance(action.value, str)

    def test_conflict_action_exists(self):
        assert MergeAction.CONFLICT == "conflict"


class TestMergeReport:

    def test_empty_report(self):
        report = MergeReport([])
        assert report.has_conflicts is False
        assert report.conflict_count == 0
        assert "0 conflict" in report.format_summary()

    def test_counts(self):
        entries = [
            CellMergeEntry("c1", MergeAction.UNCHANGED, "ok"),
            CellMergeEntry("c2", MergeAction.TOOK_THEIRS, "took A"),
            CellMergeEntry("c3", MergeAction.CONFLICT, "both changed"),
        ]
        report = MergeReport(entries)
        assert report.counts[MergeAction.UNCHANGED] == 1
        assert report.counts[MergeAction.TOOK_THEIRS] == 1
        assert report.counts[MergeAction.CONFLICT] == 1
        assert report.counts[MergeAction.TOOK_OURS] == 0

    def test_has_conflicts_true(self):
        report = MergeReport([CellMergeEntry("c1", MergeAction.CONFLICT, "x")])
        assert report.has_conflicts is True

    def test_has_conflicts_false(self):
        report = MergeReport([
            CellMergeEntry("c1", MergeAction.UNCHANGED, "ok"),
            CellMergeEntry("c2", MergeAction.AUTO_MERGED, "merged"),
        ])
        assert report.has_conflicts is False

    def test_conflict_count(self):
        report = MergeReport([
            CellMergeEntry("c1", MergeAction.CONFLICT, "x"),
            CellMergeEntry("c2", MergeAction.CONFLICT, "y"),
            CellMergeEntry("c3", MergeAction.UNCHANGED, "ok"),
        ])
        assert report.conflict_count == 2

    def test_conflict_entries(self):
        e1 = CellMergeEntry("c1", MergeAction.CONFLICT, "x")
        e2 = CellMergeEntry("c2", MergeAction.UNCHANGED, "ok")
        report = MergeReport([e1, e2])
        assert report.conflict_entries == [e1]

    def test_format_summary_includes_all_actions(self):
        entries = [
            CellMergeEntry("c1", MergeAction.UNCHANGED, "ok"),
            CellMergeEntry("c2", MergeAction.TOOK_THEIRS, "took"),
            CellMergeEntry("c3", MergeAction.AUTO_MERGED, "merged"),
        ]
        summary = MergeReport(entries).format_summary()
        assert "1 unchanged" in summary
        assert "1 took_theirs" in summary
        assert "1 auto_merged" in summary
        assert "0 conflict" in summary

    def test_format_details_lists_cells(self):
        entries = [
            CellMergeEntry("c1", MergeAction.UNCHANGED, "no change"),
            CellMergeEntry("c2", MergeAction.CONFLICT, "both differ"),
        ]
        details = MergeReport(entries).format_details()
        assert "c1" in details
        assert "c2" in details
        assert "both differ" in details

    def test_format_details_empty_report(self):
        assert "No cells" in MergeReport([]).format_details()

    def test_has_changes_false_when_all_unchanged_or_from_base(self):
        report = MergeReport([
            CellMergeEntry("c1", MergeAction.UNCHANGED, "ok"),
            CellMergeEntry("c2", MergeAction.FROM_BASE, "framework"),
        ])
        assert report.has_changes is False

    def test_has_changes_true_when_merge_occurred(self):
        report = MergeReport([
            CellMergeEntry("c1", MergeAction.UNCHANGED, "ok"),
            CellMergeEntry("c2", MergeAction.TOOK_THEIRS, "took"),
        ])
        assert report.has_changes is True
