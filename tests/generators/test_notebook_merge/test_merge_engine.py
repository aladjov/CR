import nbformat

from customer_retention.generators.notebook_merge.conflict import has_conflict_markers
from customer_retention.generators.notebook_merge.merge_engine import NotebookMergeEngine
from customer_retention.generators.notebook_merge.merge_report import MergeAction


def _make_nb(cells):
    nb = nbformat.v4.new_notebook()
    nb.cells = cells
    return nb


def _code_cell(cell_id, source_lines):
    cell = nbformat.v4.new_code_cell(source="".join(source_lines))
    cell.id = cell_id
    cell.source = source_lines
    return cell


def _md_cell(cell_id, source_lines):
    cell = nbformat.v4.new_markdown_cell(source="".join(source_lines))
    cell.id = cell_id
    cell.source = source_lines
    return cell


class TestThreeWayIdentical:

    def setup_method(self):
        self.engine = NotebookMergeEngine()

    def test_identical_notebooks_no_changes(self):
        cells = [
            _code_cell("c1", ["# @cr:code\n", "import os\n"]),
            _code_cell("c2", ["# @cr:config\n", "X = 1\n"]),
        ]
        base = _make_nb(list(cells))
        theirs = _make_nb(list(cells))
        ours = _make_nb(list(cells))
        merged, report = self.engine.merge(base, theirs, ours)
        assert len(merged.cells) == 2
        assert report.has_conflicts is False
        assert report.has_changes is False


class TestCodeCellsFromBase:

    def setup_method(self):
        self.engine = NotebookMergeEngine()

    def test_code_cell_always_from_base(self):
        base = _make_nb([_code_cell("c1", ["# @cr:code\n", "import os\n", "import sys\n"])])
        theirs = _make_nb([_code_cell("c1", ["# @cr:code\n", "import os\n"])])
        ours = _make_nb([_code_cell("c1", ["# @cr:code\n", "import json\n"])])
        merged, report = self.engine.merge(base, theirs, ours)
        assert "import sys" in "".join(merged.cells[0].source)
        assert report.counts[MergeAction.FROM_BASE] == 1

    def test_markdown_cell_from_base(self):
        base = _make_nb([_md_cell("m1", ["# Title v3\n"])])
        theirs = _make_nb([_md_cell("m1", ["# Title v1\n"])])
        ours = _make_nb([_md_cell("m1", ["# Title v2\n"])])
        merged, _ = self.engine.merge(base, theirs, ours)
        assert "Title v3" in "".join(merged.cells[0].source)


class TestConfigCellThreeWay:

    def setup_method(self):
        self.engine = NotebookMergeEngine()

    def test_theirs_changed_only(self):
        base = _make_nb([_code_cell("c1", ["# @cr:config\n", "X = 1\n"])])
        theirs = _make_nb([_code_cell("c1", ["# @cr:config\n", "X = 42\n"])])
        ours = _make_nb([_code_cell("c1", ["# @cr:config\n", "X = 1\n"])])
        merged, report = self.engine.merge(base, theirs, ours)
        assert "X = 42" in "".join(merged.cells[0].source)
        assert report.counts[MergeAction.TOOK_THEIRS] == 1

    def test_ours_changed_only(self):
        base = _make_nb([_code_cell("c1", ["# @cr:config\n", "X = 1\n"])])
        theirs = _make_nb([_code_cell("c1", ["# @cr:config\n", "X = 1\n"])])
        ours = _make_nb([_code_cell("c1", ["# @cr:config\n", "X = 99\n"])])
        merged, report = self.engine.merge(base, theirs, ours)
        assert "X = 99" in "".join(merged.cells[0].source)
        assert report.counts[MergeAction.TOOK_OURS] == 1

    def test_both_changed_same_way(self):
        base = _make_nb([_code_cell("c1", ["# @cr:config\n", "X = 1\n"])])
        theirs = _make_nb([_code_cell("c1", ["# @cr:config\n", "X = 42\n"])])
        ours = _make_nb([_code_cell("c1", ["# @cr:config\n", "X = 42\n"])])
        merged, report = self.engine.merge(base, theirs, ours)
        assert "X = 42" in "".join(merged.cells[0].source)
        assert report.counts[MergeAction.BOTH_SAME] == 1

    def test_both_changed_differently_conflict(self):
        base = _make_nb([_code_cell("c1", ["# @cr:config\n", "X = 1\n"])])
        theirs = _make_nb([_code_cell("c1", ["# @cr:config\n", "X = 42\n"])])
        ours = _make_nb([_code_cell("c1", ["# @cr:config\n", "X = 99\n"])])
        merged, report = self.engine.merge(base, theirs, ours)
        assert report.has_conflicts is True
        assert has_conflict_markers("".join(merged.cells[0].source))

    def test_dict_config_auto_merged(self):
        base = _make_nb([_code_cell("c1", ["# @cr:config\n", "D = {}\n"])])
        theirs = _make_nb([_code_cell("c1", ["# @cr:config\n", 'D = {"a": 1}\n'])])
        ours = _make_nb([_code_cell("c1", ["# @cr:config\n", 'D = {"b": 2}\n'])])
        merged, report = self.engine.merge(base, theirs, ours)
        src = "".join(merged.cells[0].source)
        assert "a" in src
        assert "b" in src
        assert report.counts[MergeAction.AUTO_MERGED] == 1


class TestUserCodeCellThreeWay:

    def setup_method(self):
        self.engine = NotebookMergeEngine()

    def test_user_code_theirs_only(self):
        base = _make_nb([_code_cell("c1", ["# @cr:user_code\n", "def f(): pass\n"])])
        theirs = _make_nb([_code_cell("c1", ["# @cr:user_code\n", "def f(): return 1\n"])])
        ours = _make_nb([_code_cell("c1", ["# @cr:user_code\n", "def f(): pass\n"])])
        merged, report = self.engine.merge(base, theirs, ours)
        assert "return 1" in "".join(merged.cells[0].source)
        assert report.counts[MergeAction.TOOK_THEIRS] == 1

    def test_user_code_conflict(self):
        base = _make_nb([_code_cell("c1", ["# @cr:user_code\n", "def f(): pass\n"])])
        theirs = _make_nb([_code_cell("c1", ["# @cr:user_code\n", "def f(): return 1\n"])])
        ours = _make_nb([_code_cell("c1", ["# @cr:user_code\n", "def f(): return 2\n"])])
        merged, report = self.engine.merge(base, theirs, ours)
        assert report.has_conflicts is True


class TestUserAddedCells:

    def setup_method(self):
        self.engine = NotebookMergeEngine()

    def test_cell_added_by_theirs(self):
        base = _make_nb([_code_cell("c1", ["# @cr:code\n", "a = 1\n"])])
        theirs = _make_nb([
            _code_cell("c1", ["# @cr:code\n", "a = 1\n"]),
            _code_cell("t-new", ["# @cr:config\n", "NEW = True\n"]),
        ])
        ours = _make_nb([_code_cell("c1", ["# @cr:code\n", "a = 1\n"])])
        merged, report = self.engine.merge(base, theirs, ours)
        ids = [c.id for c in merged.cells]
        assert "t-new" in ids
        assert report.counts[MergeAction.ADDED_THEIRS] == 1

    def test_cell_added_by_ours(self):
        base = _make_nb([_code_cell("c1", ["# @cr:code\n", "a = 1\n"])])
        theirs = _make_nb([_code_cell("c1", ["# @cr:code\n", "a = 1\n"])])
        ours = _make_nb([
            _code_cell("c1", ["# @cr:code\n", "a = 1\n"]),
            _code_cell("o-new", ["# @cr:user_code\n", "custom()\n"]),
        ])
        merged, report = self.engine.merge(base, theirs, ours)
        ids = [c.id for c in merged.cells]
        assert "o-new" in ids
        assert report.counts[MergeAction.ADDED_OURS] == 1

    def test_both_added_same_cell(self):
        base = _make_nb([_code_cell("c1", ["# @cr:code\n", "a = 1\n"])])
        theirs = _make_nb([
            _code_cell("c1", ["# @cr:code\n", "a = 1\n"]),
            _code_cell("new", ["# @cr:config\n", "V = 1\n"]),
        ])
        ours = _make_nb([
            _code_cell("c1", ["# @cr:code\n", "a = 1\n"]),
            _code_cell("new", ["# @cr:config\n", "V = 1\n"]),
        ])
        merged, _ = self.engine.merge(base, theirs, ours)
        ids = [c.id for c in merged.cells]
        assert ids.count("new") == 1

    def test_user_added_markdown_kept(self):
        base = _make_nb([_code_cell("c1", ["# @cr:code\n", "a = 1\n"])])
        theirs = _make_nb([
            _code_cell("c1", ["# @cr:code\n", "a = 1\n"]),
            _md_cell("t-note", ["## Alice's notes\n"]),
        ])
        ours = _make_nb([_code_cell("c1", ["# @cr:code\n", "a = 1\n"])])
        merged, _ = self.engine.merge(base, theirs, ours)
        ids = [c.id for c in merged.cells]
        assert "t-note" in ids

    def test_theirs_cells_before_ours_at_same_anchor(self):
        base = _make_nb([
            _code_cell("c1", ["# @cr:code\n", "a = 1\n"]),
            _code_cell("c2", ["# @cr:code\n", "b = 2\n"]),
        ])
        theirs = _make_nb([
            _code_cell("c1", ["# @cr:code\n", "a = 1\n"]),
            _code_cell("t-x", ["# @cr:config\n", "T = 1\n"]),
            _code_cell("c2", ["# @cr:code\n", "b = 2\n"]),
        ])
        ours = _make_nb([
            _code_cell("c1", ["# @cr:code\n", "a = 1\n"]),
            _code_cell("o-x", ["# @cr:config\n", "O = 1\n"]),
            _code_cell("c2", ["# @cr:code\n", "b = 2\n"]),
        ])
        merged, _ = self.engine.merge(base, theirs, ours)
        ids = [c.id for c in merged.cells]
        assert ids.index("t-x") < ids.index("o-x")
        assert ids.index("o-x") < ids.index("c2")


class TestCellMissingFromOneSide:

    def setup_method(self):
        self.engine = NotebookMergeEngine()

    def test_config_cell_missing_from_ours_kept(self):
        base = _make_nb([
            _code_cell("c1", ["# @cr:code\n", "a = 1\n"]),
            _code_cell("c2", ["# @cr:config\n", "X = 1\n"]),
        ])
        theirs = _make_nb([
            _code_cell("c1", ["# @cr:code\n", "a = 1\n"]),
            _code_cell("c2", ["# @cr:config\n", "X = 42\n"]),
        ])
        ours = _make_nb([
            _code_cell("c1", ["# @cr:code\n", "a = 1\n"]),
        ])
        merged, _ = self.engine.merge(base, theirs, ours)
        ids = [c.id for c in merged.cells]
        assert "c2" in ids

    def test_code_cell_always_included_from_base(self):
        base = _make_nb([
            _code_cell("c1", ["# @cr:code\n", "a = 1\n"]),
            _code_cell("c2", ["# @cr:code\n", "b = 2\n"]),
        ])
        theirs = _make_nb([_code_cell("c1", ["# @cr:code\n", "a = 1\n"])])
        ours = _make_nb([_code_cell("c1", ["# @cr:code\n", "a = 1\n"])])
        merged, _ = self.engine.merge(base, theirs, ours)
        assert len(merged.cells) == 2


class TestOrdering:

    def setup_method(self):
        self.engine = NotebookMergeEngine()

    def test_base_ordering_preserved(self):
        base = _make_nb([
            _code_cell("c1", ["# @cr:code\n", "a = 1\n"]),
            _md_cell("m1", ["# Section\n"]),
            _code_cell("c2", ["# @cr:config\n", "B = 2\n"]),
        ])
        theirs = _make_nb([
            _code_cell("c2", ["# @cr:config\n", "B = 99\n"]),
            _md_cell("m1", ["# Section\n"]),
            _code_cell("c1", ["# @cr:code\n", "a = 1\n"]),
        ])
        ours = _make_nb([
            _md_cell("m1", ["# Section\n"]),
            _code_cell("c1", ["# @cr:code\n", "a = 1\n"]),
            _code_cell("c2", ["# @cr:config\n", "B = 2\n"]),
        ])
        merged, _ = self.engine.merge(base, theirs, ours)
        assert [c.id for c in merged.cells] == ["c1", "m1", "c2"]


class TestEmbeddedIds:

    def setup_method(self):
        self.engine = NotebookMergeEngine()

    def test_embedded_ids_match_across_notebooks(self):
        base = _make_nb([
            _code_cell("c1", ["# @cr:config name='settings' id=a1b2c3d4\n", "X = 1\n"]),
        ])
        theirs = _make_nb([
            _code_cell("rand-aaa", ["# @cr:config name='settings' id=a1b2c3d4\n", "X = 42\n"]),
        ])
        ours = _make_nb([
            _code_cell("rand-bbb", ["# @cr:config name='settings' id=a1b2c3d4\n", "X = 1\n"]),
        ])
        merged, report = self.engine.merge(base, theirs, ours)
        assert "X = 42" in "".join(merged.cells[0].source)
        assert report.counts[MergeAction.TOOK_THEIRS] == 1


class TestCustomLabels:

    def test_labels_in_conflict_markers(self):
        engine = NotebookMergeEngine(theirs_label="alice", ours_label="bob")
        base = _make_nb([_code_cell("c1", ["# @cr:config\n", "X = 1\n"])])
        theirs = _make_nb([_code_cell("c1", ["# @cr:config\n", "X = 42\n"])])
        ours = _make_nb([_code_cell("c1", ["# @cr:config\n", "X = 99\n"])])
        merged, _ = engine.merge(base, theirs, ours)
        src = "".join(merged.cells[0].source)
        assert "alice" in src
        assert "bob" in src


class TestMixedScenario:

    def setup_method(self):
        self.engine = NotebookMergeEngine()

    def test_full_notebook_merge(self):
        engine = NotebookMergeEngine()
        base = _make_nb([
            _md_cell("m1", ["# Notebook\n"]),
            _code_cell("c1", ["# @cr:config\n", "X = 1\n"]),
            _code_cell("c2", ["# @cr:code\n", "import os\n"]),
            _code_cell("c3", ["# @cr:user_code\n", "def target(): pass\n"]),
            _code_cell("c4", ["# @cr:code\n", "run()\n"]),
        ])
        theirs = _make_nb([
            _md_cell("m1", ["# Notebook\n"]),
            _code_cell("c1", ["# @cr:config\n", "X = 42\n"]),
            _code_cell("c2", ["# @cr:code\n", "import os\n"]),
            _code_cell("c3", ["# @cr:user_code\n", "def target(): pass\n"]),
            _code_cell("c4", ["# @cr:code\n", "run()\n"]),
        ])
        ours = _make_nb([
            _md_cell("m1", ["# Notebook\n"]),
            _code_cell("c1", ["# @cr:config\n", "X = 1\n"]),
            _code_cell("c2", ["# @cr:code\n", "import os\n"]),
            _code_cell("c3", ["# @cr:user_code\n", "def target():\n", "    return True\n"]),
            _code_cell("c4", ["# @cr:code\n", "run()\n"]),
        ])
        merged, report = self.engine.merge(base, theirs, ours)
        assert len(merged.cells) == 5
        assert "X = 42" in "".join(merged.cells[1].source)
        assert "return True" in "".join(merged.cells[3].source)
        assert report.counts[MergeAction.TOOK_THEIRS] == 1
        assert report.counts[MergeAction.TOOK_OURS] == 1
        assert report.has_conflicts is False


class TestEmptyNotebooks:

    def setup_method(self):
        self.engine = NotebookMergeEngine()

    def test_empty_base(self):
        base = _make_nb([])
        theirs = _make_nb([_code_cell("c1", ["# @cr:config\n", "X = 1\n"])])
        ours = _make_nb([])
        merged, _ = self.engine.merge(base, theirs, ours)
        assert any(c.id == "c1" for c in merged.cells)

    def test_all_empty(self):
        merged, report = self.engine.merge(_make_nb([]), _make_nb([]), _make_nb([]))
        assert len(merged.cells) == 0
        assert report.has_changes is False
