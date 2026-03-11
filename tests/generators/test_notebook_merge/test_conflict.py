from customer_retention.generators.notebook_merge.conflict import (
    has_conflict_markers,
    wrap_conflict,
)


class TestWrapConflict:

    def test_basic_markers(self):
        result = wrap_conflict("X = 1\n", "X = 2\n")
        assert "# <<<< theirs" in result
        assert "X = 1" in result
        assert "# ====" in result
        assert "X = 2" in result
        assert "# >>>> ours" in result

    def test_custom_labels(self):
        result = wrap_conflict("X = 1\n", "X = 2\n", theirs_label="alice", ours_label="bob")
        assert "# <<<< alice" in result
        assert "# >>>> bob" in result

    def test_multiline_sources(self):
        theirs = "X = {\n    'a': 1,\n}\n"
        ours = "X = {\n    'b': 2,\n}\n"
        result = wrap_conflict(theirs, ours)
        assert "'a': 1" in result
        assert "'b': 2" in result

    def test_empty_theirs(self):
        result = wrap_conflict("", "X = 2\n")
        assert "# <<<< theirs" in result
        assert "# >>>> ours" in result

    def test_trailing_newlines_normalized(self):
        result = wrap_conflict("X = 1", "X = 2")
        lines = result.split("\n")
        assert lines[0] == "# <<<< theirs"
        assert lines[-1] == "# >>>> ours"


class TestHasConflictMarkers:

    def test_detects_markers(self):
        source = "# <<<< theirs\nX = 1\n# ====\nX = 2\n# >>>> ours"
        assert has_conflict_markers(source) is True

    def test_no_markers(self):
        assert has_conflict_markers("X = 1\nY = 2\n") is False

    def test_partial_markers_not_detected(self):
        assert has_conflict_markers("# <<<< theirs\nX = 1\n") is False

    def test_custom_labels_detected(self):
        source = "# <<<< alice\nX = 1\n# ====\nX = 2\n# >>>> bob"
        assert has_conflict_markers(source) is True
