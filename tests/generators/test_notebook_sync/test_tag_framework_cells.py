from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

from scripts.notebooks.tag_framework_cells import tag_untagged_code_cells


def _make_nb(cells):
    nb = new_notebook()
    nb.cells = cells
    return nb


def _code_cell(cell_id, source):
    c = new_code_cell(source=source)
    c.id = cell_id
    return c


def _md_cell(cell_id, source):
    c = new_markdown_cell(source=source)
    c.id = cell_id
    return c


class TestTagUntaggedCodeCells:

    def test_tags_untagged_code_cell(self):
        nb = _make_nb([_code_cell("c1", "import os\n")])
        count = tag_untagged_code_cells(nb)
        assert count == 1
        assert nb.cells[0].source.startswith("# @cr:code\n")

    def test_preserves_existing_config_tag(self):
        nb = _make_nb([_code_cell("c1", "# @cr:config\nX = 1\n")])
        count = tag_untagged_code_cells(nb)
        assert count == 0
        assert nb.cells[0].source == "# @cr:config\nX = 1\n"

    def test_preserves_existing_user_code_tag(self):
        nb = _make_nb([_code_cell("c1", "# @cr:user_code\ndef my_fn(): pass\n")])
        count = tag_untagged_code_cells(nb)
        assert count == 0
        assert nb.cells[0].source == "# @cr:user_code\ndef my_fn(): pass\n"

    def test_preserves_existing_code_tag(self):
        nb = _make_nb([_code_cell("c1", "# @cr:code\nimport os\n")])
        count = tag_untagged_code_cells(nb)
        assert count == 0
        assert nb.cells[0].source == "# @cr:code\nimport os\n"

    def test_skips_markdown_cells(self):
        nb = _make_nb([_md_cell("m1", "# Title\n")])
        count = tag_untagged_code_cells(nb)
        assert count == 0
        assert nb.cells[0].source == "# Title\n"

    def test_empty_code_cell(self):
        nb = _make_nb([_code_cell("c1", "")])
        count = tag_untagged_code_cells(nb)
        assert count == 1
        assert nb.cells[0].source == "# @cr:code\n"

    def test_idempotent(self):
        nb = _make_nb([_code_cell("c1", "import os\n")])
        tag_untagged_code_cells(nb)
        first_source = nb.cells[0].source
        count = tag_untagged_code_cells(nb)
        assert count == 0
        assert nb.cells[0].source == first_source

    def test_source_as_list(self):
        cell = new_code_cell(source="import os\n")
        cell.id = "c1"
        cell.source = ["import os\n"]
        nb = _make_nb([cell])
        count = tag_untagged_code_cells(nb)
        assert count == 1
        src = nb.cells[0].source
        if isinstance(src, list):
            assert src[0] == "# @cr:code\n"
        else:
            assert src.startswith("# @cr:code\n")

    def test_returns_count(self):
        nb = _make_nb([
            _code_cell("c1", "import os\n"),
            _code_cell("c2", "# @cr:config\nX = 1\n"),
            _code_cell("c3", "import sys\n"),
            _md_cell("m1", "# Title\n"),
        ])
        count = tag_untagged_code_cells(nb)
        assert count == 2

    def test_mixed_cells(self):
        nb = _make_nb([
            _md_cell("m1", "# Intro\n"),
            _code_cell("c1", "# @cr:config\nX = 1\n"),
            _code_cell("c2", "import os\n"),
            _code_cell("c3", "# @cr:user_code\ndef target(): pass\n"),
            _code_cell("c4", "print('hello')\n"),
            _md_cell("m2", "## Section 2\n"),
        ])
        count = tag_untagged_code_cells(nb)
        assert count == 2
        assert nb.cells[0].source == "# Intro\n"
        assert nb.cells[1].source.startswith("# @cr:config\n")
        assert nb.cells[2].source.startswith("# @cr:code\n")
        assert nb.cells[3].source.startswith("# @cr:user_code\n")
        assert nb.cells[4].source.startswith("# @cr:code\n")
        assert nb.cells[5].source == "## Section 2\n"
