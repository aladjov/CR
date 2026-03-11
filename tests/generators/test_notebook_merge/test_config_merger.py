from customer_retention.generators.notebook_merge.config_merger import merge_config_cell
from customer_retention.generators.notebook_merge.conflict import has_conflict_markers
from customer_retention.generators.notebook_merge.merge_report import MergeAction

TAG = "# @cr:config name='t' id=aabb1122"


def _cell(body: str) -> str:
    return f"{TAG}\n{body}"


class TestDictMergeDisjointKeys:

    def test_different_keys_union(self):
        base = _cell('DROP_COLUMNS = {}\n')
        theirs = _cell('DROP_COLUMNS = {"emails": ["body"]}\n')
        ours = _cell('DROP_COLUMNS = {"transactions": []}\n')
        merged_source, action, _ = merge_config_cell(base, theirs, ours)
        assert action == MergeAction.AUTO_MERGED
        assert "emails" in merged_source
        assert "transactions" in merged_source

    def test_one_side_adds_keys(self):
        base = _cell('DROP_COLUMNS = {"emails": ["body"]}\n')
        theirs = _cell('DROP_COLUMNS = {"emails": ["body"], "transactions": []}\n')
        ours = _cell('DROP_COLUMNS = {"emails": ["body"]}\n')
        merged_source, action, _ = merge_config_cell(base, theirs, ours)
        assert action == MergeAction.AUTO_MERGED
        assert "transactions" in merged_source


class TestDictMergeSameKey:

    def test_same_key_same_value(self):
        base = _cell('D = {}\n')
        theirs = _cell('D = {"a": 1}\n')
        ours = _cell('D = {"a": 1}\n')
        _, action, _ = merge_config_cell(base, theirs, ours)
        assert action == MergeAction.AUTO_MERGED

    def test_same_key_different_value_conflict(self):
        base = _cell('D = {}\n')
        theirs = _cell('D = {"a": 1}\n')
        ours = _cell('D = {"a": 2}\n')
        merged_source, action, conflicts = merge_config_cell(base, theirs, ours)
        assert action == MergeAction.CONFLICT
        assert len(conflicts) > 0
        assert has_conflict_markers(merged_source)


class TestListMerge:

    def test_union_dedup(self):
        base = _cell('EXCLUDE = []\n')
        theirs = _cell('EXCLUDE = ["a", "b"]\n')
        ours = _cell('EXCLUDE = ["b", "c"]\n')
        merged_source, action, _ = merge_config_cell(base, theirs, ours)
        assert action == MergeAction.AUTO_MERGED
        assert "'a'" in merged_source
        assert "'b'" in merged_source
        assert "'c'" in merged_source

    def test_identical_lists(self):
        base = _cell('X = []\n')
        theirs = _cell('X = ["a"]\n')
        ours = _cell('X = ["a"]\n')
        _, action, _ = merge_config_cell(base, theirs, ours)
        assert action == MergeAction.AUTO_MERGED


class TestScalarMerge:

    def test_both_changed_same_value(self):
        base = _cell('X = 90\n')
        theirs = _cell('X = 120\n')
        ours = _cell('X = 120\n')
        _, action, _ = merge_config_cell(base, theirs, ours)
        assert action in (MergeAction.AUTO_MERGED, MergeAction.BOTH_SAME)

    def test_both_changed_different_value_conflict(self):
        base = _cell('X = 90\n')
        theirs = _cell('X = 120\n')
        ours = _cell('X = 60\n')
        merged_source, action, conflicts = merge_config_cell(base, theirs, ours)
        assert action == MergeAction.CONFLICT
        assert has_conflict_markers(merged_source)


class TestComplexAssignment:

    def test_complex_same_both_sides(self):
        base = _cell('POSTURE = TemporalPosture.STABLE\n')
        theirs = _cell('POSTURE = TemporalPosture.VOLATILE\n')
        ours = _cell('POSTURE = TemporalPosture.VOLATILE\n')
        _, action, _ = merge_config_cell(base, theirs, ours)
        assert action != MergeAction.CONFLICT

    def test_complex_different_conflict(self):
        base = _cell('POSTURE = TemporalPosture.STABLE\n')
        theirs = _cell('POSTURE = TemporalPosture.VOLATILE\n')
        ours = _cell('POSTURE = TemporalPosture.MIXED\n')
        _, action, _ = merge_config_cell(base, theirs, ours)
        assert action == MergeAction.CONFLICT


class TestMixedAssignments:

    def test_multiple_assignments_partial_conflict(self):
        base = _cell('X = 90\nY = []\nZ = {}\n')
        theirs = _cell('X = 120\nY = ["a"]\nZ = {"k1": 1}\n')
        ours = _cell('X = 60\nY = ["b"]\nZ = {"k2": 2}\n')
        merged_source, action, conflicts = merge_config_cell(base, theirs, ours)
        assert action == MergeAction.CONFLICT
        assert len(conflicts) == 1
        assert "k1" in merged_source
        assert "k2" in merged_source

    def test_all_auto_resolved(self):
        base = _cell('X = 90\nY = []\nZ = {}\n')
        theirs = _cell('X = 90\nY = ["a"]\nZ = {"k1": 1}\n')
        ours = _cell('X = 90\nY = ["b"]\nZ = {"k2": 2}\n')
        merged_source, action, conflicts = merge_config_cell(base, theirs, ours)
        assert action == MergeAction.AUTO_MERGED
        assert len(conflicts) == 0

    def test_new_assignment_in_theirs(self):
        base = _cell('X = 1\n')
        theirs = _cell('X = 1\nNEW_VAR = "hello"\n')
        ours = _cell('X = 1\n')
        merged_source, action, _ = merge_config_cell(base, theirs, ours)
        assert "NEW_VAR" in merged_source

    def test_new_assignment_in_both_same_value(self):
        base = _cell('X = 1\n')
        theirs = _cell('X = 1\nNEW_VAR = 42\n')
        ours = _cell('X = 1\nNEW_VAR = 42\n')
        merged_source, action, _ = merge_config_cell(base, theirs, ours)
        assert "NEW_VAR" in merged_source
        assert action != MergeAction.CONFLICT


class TestPreamblePreserved:

    def test_preamble_from_base(self):
        base = _cell('from foo import Bar\n\nX = 1\n')
        theirs = _cell('from foo import Bar\n\nX = 2\n')
        ours = _cell('from foo import Bar\n\nX = 2\n')
        merged_source, _, _ = merge_config_cell(base, theirs, ours)
        assert "from foo import Bar" in merged_source

    def test_tag_line_preserved(self):
        base = _cell('X = 1\n')
        theirs = _cell('X = 2\n')
        ours = _cell('X = 2\n')
        merged_source, _, _ = merge_config_cell(base, theirs, ours)
        assert TAG in merged_source


class TestParseFailureFallback:

    def test_unparseable_cell_textual_conflict(self):
        base = _cell('X = 1\n')
        theirs = _cell('X = {\n')
        ours = _cell('X = 2\n')
        merged_source, action, _ = merge_config_cell(base, theirs, ours)
        assert action == MergeAction.CONFLICT
        assert has_conflict_markers(merged_source)
