import nbformat

from customer_retention.generators.notebook_sync.sync_engine import NotebookSyncEngine
from customer_retention.generators.notebook_sync.sync_report import SyncAction


def _make_nb(cells):
    nb = nbformat.v4.new_notebook()
    nb.cells = cells
    return nb


def _code_cell(cell_id, source_lines, outputs=None, execution_count=None):
    cell = nbformat.v4.new_code_cell(source="".join(source_lines))
    cell.id = cell_id
    cell.source = source_lines
    if outputs:
        cell.outputs = outputs
    if execution_count is not None:
        cell.execution_count = execution_count
    return cell


def _md_cell(cell_id, source_lines):
    cell = nbformat.v4.new_markdown_cell(source="".join(source_lines))
    cell.id = cell_id
    cell.source = source_lines
    return cell


class TestNotebookSyncEngine:

    def setup_method(self):
        self.engine = NotebookSyncEngine()

    def test_identical_notebooks_no_changes(self):
        cells = [
            _code_cell("c1", ["# @cr:code\n", "import os\n"]),
            _code_cell("c2", ["# @cr:config\n", "X = 1\n"]),
        ]
        repo = _make_nb(list(cells))
        user = _make_nb(list(cells))
        merged, report = self.engine.sync(repo, user)
        assert len(merged.cells) == 2
        assert report.has_changes is False

    def test_code_cell_overwritten_from_repo(self):
        repo = _make_nb([_code_cell("c1", ["# @cr:code\n", "import os\n", "import sys\n"])])
        user = _make_nb([_code_cell("c1", ["# @cr:code\n", "import os\n"])])
        merged, report = self.engine.sync(repo, user)
        assert "".join(merged.cells[0].source) == "# @cr:code\nimport os\nimport sys\n"
        assert report.counts[SyncAction.UPDATED] == 1

    def test_untagged_code_cell_overwritten(self):
        repo = _make_nb([_code_cell("c1", ["import os\n", "import sys\n"])])
        user = _make_nb([_code_cell("c1", ["import os\n"])])
        merged, report = self.engine.sync(repo, user)
        assert "".join(merged.cells[0].source) == "import os\nimport sys\n"

    def test_config_cell_preserved(self):
        repo = _make_nb([_code_cell("c1", ["# @cr:config\n", "X = 1\n"])])
        user = _make_nb([_code_cell("c1", ["# @cr:config\n", "X = 42\n"])])
        merged, report = self.engine.sync(repo, user)
        assert "".join(merged.cells[0].source) == "# @cr:config\nX = 42\n"
        assert report.counts[SyncAction.PRESERVED] == 1

    def test_user_code_cell_preserved(self):
        repo = _make_nb([_code_cell("c1", ["# @cr:user_code\n", "def target(): pass\n"])])
        user = _make_nb([_code_cell("c1", ["# @cr:user_code\n", "def target():\n", "    return col > 5\n"])])
        merged, report = self.engine.sync(repo, user)
        assert "return col > 5" in "".join(merged.cells[0].source)
        assert report.counts[SyncAction.PRESERVED] == 1

    def test_config_cell_outputs_preserved(self):
        outputs = [{"output_type": "stream", "name": "stdout", "text": ["hello\n"]}]
        repo = _make_nb([_code_cell("c1", ["# @cr:config\n", "X = 1\n"])])
        user = _make_nb([_code_cell("c1", ["# @cr:config\n", "X = 42\n"], outputs=outputs, execution_count=5)])
        merged, report = self.engine.sync(repo, user)
        assert merged.cells[0].outputs == outputs
        assert merged.cells[0].execution_count == 5

    def test_code_cell_outputs_stripped(self):
        outputs = [{"output_type": "stream", "name": "stdout", "text": ["hello\n"]}]
        repo = _make_nb([_code_cell("c1", ["# @cr:code\n", "print('hi')\n"])])
        user = _make_nb([_code_cell("c1", ["# @cr:code\n", "print('hi')\n"], outputs=outputs, execution_count=3)])
        merged, report = self.engine.sync(repo, user)
        assert merged.cells[0].outputs == []
        assert merged.cells[0].execution_count is None

    def test_new_cell_added_from_repo(self):
        repo = _make_nb([
            _code_cell("c1", ["# @cr:code\n", "import os\n"]),
            _code_cell("c2", ["# @cr:code\n", "import sys\n"]),
        ])
        user = _make_nb([_code_cell("c1", ["# @cr:code\n", "import os\n"])])
        merged, report = self.engine.sync(repo, user)
        assert len(merged.cells) == 2
        assert merged.cells[1].id == "c2"
        assert report.counts[SyncAction.ADDED] == 1

    def test_removed_untagged_cell_dropped(self):
        repo = _make_nb([_code_cell("c1", ["# @cr:code\n", "import os\n"])])
        user = _make_nb([
            _code_cell("c1", ["# @cr:code\n", "import os\n"]),
            _code_cell("c2", ["print('orphan')\n"]),
        ])
        merged, report = self.engine.sync(repo, user)
        assert len(merged.cells) == 1
        assert report.counts[SyncAction.REMOVED] == 1

    def test_user_added_config_cell_kept(self):
        repo = _make_nb([_code_cell("c1", ["# @cr:code\n", "import os\n"])])
        user = _make_nb([
            _code_cell("c1", ["# @cr:code\n", "import os\n"]),
            _code_cell("user-custom", ["# @cr:config\n", "MY_VAR = True\n"]),
        ])
        merged, report = self.engine.sync(repo, user)
        assert len(merged.cells) == 2
        assert any(c.id == "user-custom" for c in merged.cells)
        assert report.counts[SyncAction.USER_ADDED_KEPT] == 1

    def test_user_added_user_code_cell_kept(self):
        repo = _make_nb([_code_cell("c1", ["# @cr:code\n", "import os\n"])])
        user = _make_nb([
            _code_cell("c1", ["# @cr:code\n", "import os\n"]),
            _code_cell("user-logic", ["# @cr:user_code\n", "custom_logic()\n"]),
        ])
        merged, report = self.engine.sync(repo, user)
        assert any(c.id == "user-logic" for c in merged.cells)

    def test_markdown_overwritten_from_repo(self):
        repo = _make_nb([_md_cell("m1", ["# Title v2\n"])])
        user = _make_nb([_md_cell("m1", ["# Title v1\n"])])
        merged, report = self.engine.sync(repo, user)
        assert "".join(merged.cells[0].source) == "# Title v2\n"

    def test_user_added_markdown_kept(self):
        repo = _make_nb([_md_cell("m1", ["# Title\n"])])
        user = _make_nb([
            _md_cell("m1", ["# Title\n"]),
            _md_cell("user-note", ["## My Notes\n"]),
        ])
        merged, report = self.engine.sync(repo, user)
        assert len(merged.cells) == 2
        assert any(c.id == "user-note" for c in merged.cells)

    def test_repo_ordering_preserved(self):
        repo = _make_nb([
            _code_cell("c1", ["# @cr:code\n", "a = 1\n"]),
            _md_cell("m1", ["# Section\n"]),
            _code_cell("c2", ["# @cr:config\n", "B = 2\n"]),
        ])
        user = _make_nb([
            _code_cell("c2", ["# @cr:config\n", "B = 99\n"]),
            _md_cell("m1", ["# Section\n"]),
            _code_cell("c1", ["# @cr:code\n", "a = 1\n"]),
        ])
        merged, report = self.engine.sync(repo, user)
        assert [c.id for c in merged.cells] == ["c1", "m1", "c2"]

    def test_user_only_cells_anchored_after_preceding_repo_cell(self):
        repo = _make_nb([
            _code_cell("c1", ["# @cr:code\n", "a = 1\n"]),
            _code_cell("c3", ["# @cr:code\n", "c = 3\n"]),
        ])
        user = _make_nb([
            _code_cell("c1", ["# @cr:code\n", "a = 1\n"]),
            _code_cell("user-x", ["# @cr:config\n", "X = 1\n"]),
            _code_cell("c3", ["# @cr:code\n", "c = 3\n"]),
        ])
        merged, report = self.engine.sync(repo, user)
        ids = [c.id for c in merged.cells]
        assert ids.index("user-x") > ids.index("c1")
        assert ids.index("user-x") < ids.index("c3")

    def test_mixed_scenario(self):
        repo = _make_nb([
            _md_cell("m1", ["# Notebook\n"]),
            _code_cell("c1", ["# @cr:config\n", "X = 1\n"]),
            _code_cell("c2", ["# @cr:code\n", "import os\n"]),
            _code_cell("c3", ["# @cr:user_code\n", "def target(): pass\n"]),
            _code_cell("c4", ["# @cr:code\n", "run()\n"]),
        ])
        user = _make_nb([
            _md_cell("m1", ["# Old Title\n"]),
            _code_cell("c1", ["# @cr:config\n", "X = 42\n"]),
            _code_cell("c2", ["# @cr:code\n", "import os\n"], outputs=[{"output_type": "stream", "name": "stdout", "text": ["ok"]}]),
            _code_cell("c3", ["# @cr:user_code\n", "def target():\n", "    return True\n"]),
        ])
        merged, report = self.engine.sync(repo, user)
        assert len(merged.cells) == 5
        assert "".join(merged.cells[0].source) == "# Notebook\n"
        assert "X = 42" in "".join(merged.cells[1].source)
        assert merged.cells[2].outputs == []
        assert "return True" in "".join(merged.cells[3].source)
        assert merged.cells[4].id == "c4"
        assert report.counts[SyncAction.ADDED] == 1

    def test_empty_repo_notebook(self):
        repo = _make_nb([])
        user = _make_nb([_code_cell("c1", ["# @cr:config\n", "X = 1\n"])])
        merged, report = self.engine.sync(repo, user)
        assert len(merged.cells) == 1
        assert report.counts[SyncAction.USER_ADDED_KEPT] == 1

    def test_empty_user_notebook(self):
        repo = _make_nb([
            _code_cell("c1", ["# @cr:code\n", "import os\n"]),
        ])
        user = _make_nb([])
        merged, report = self.engine.sync(repo, user)
        assert len(merged.cells) == 1
        assert report.counts[SyncAction.ADDED] == 1

    def test_user_added_config_cell_with_databricks_init_preserved(self):
        repo = _make_nb([
            _code_cell("c1", ["# @cr:code\n", "import os\n"]),
            _code_cell("c2", ["# @cr:code\n", "run()\n"]),
        ])
        user = _make_nb([
            _code_cell("c1", ["# @cr:code\n", "import os\n"]),
            _code_cell("user-dbx-init", [
                "# @cr:config\n",
                "from customer_retention.integrations.databricks_init import databricks_init\n",
                "\n",
                "result = databricks_init(\n",
                '    catalog="prod_catalog",\n',
                '    schema="churn_model",\n',
                '    workspace_path="Users/me@company.com/cr",\n',
                '    model_name="customer_retention",\n',
                ")\n",
            ]),
            _code_cell("c2", ["# @cr:code\n", "run()\n"]),
        ])
        merged, report = self.engine.sync(repo, user)
        ids = [c.id for c in merged.cells]
        assert "user-dbx-init" in ids
        init_cell = next(c for c in merged.cells if c.id == "user-dbx-init")
        assert "databricks_init(" in "".join(init_cell.source)
        assert 'catalog="prod_catalog"' in "".join(init_cell.source)
        assert report.counts[SyncAction.USER_ADDED_KEPT] == 1

    def test_user_added_untagged_databricks_init_dropped(self):
        repo = _make_nb([
            _code_cell("c1", ["# @cr:code\n", "import os\n"]),
        ])
        user = _make_nb([
            _code_cell("c1", ["# @cr:code\n", "import os\n"]),
            _code_cell("user-dbx-init", [
                "from customer_retention.integrations.databricks_init import databricks_init\n",
                "result = databricks_init(catalog='prod')\n",
            ]),
        ])
        merged, report = self.engine.sync(repo, user)
        ids = [c.id for c in merged.cells]
        assert "user-dbx-init" not in ids
        assert report.counts[SyncAction.REMOVED] == 1

    def test_removed_cell_has_source_preview(self):
        repo = _make_nb([_code_cell("c1", ["# @cr:code\n", "import os\n"])])
        user = _make_nb([
            _code_cell("c1", ["# @cr:code\n", "import os\n"]),
            _code_cell("c2", ["print('orphan')\n", "x = 1\n", "y = 2\n"]),
        ])
        merged, report = self.engine.sync(repo, user)
        removed = [e for e in report.entries if e.action == SyncAction.REMOVED]
        assert len(removed) == 1
        assert removed[0].source_preview is not None
        assert "print('orphan')" in removed[0].source_preview

    def test_removed_cell_preview_truncated_to_5_lines(self):
        long_source = [f"line_{i}\n" for i in range(10)]
        repo = _make_nb([_code_cell("c1", ["# @cr:code\n", "import os\n"])])
        user = _make_nb([
            _code_cell("c1", ["# @cr:code\n", "import os\n"]),
            _code_cell("c2", long_source),
        ])
        merged, report = self.engine.sync(repo, user)
        removed = [e for e in report.entries if e.action == SyncAction.REMOVED]
        assert removed[0].source_preview is not None
        preview_lines = removed[0].source_preview.strip().split("\n")
        assert len(preview_lines) <= 5

    def test_non_removed_entries_have_no_preview(self):
        repo = _make_nb([_code_cell("c1", ["# @cr:code\n", "import os\n", "import sys\n"])])
        user = _make_nb([_code_cell("c1", ["# @cr:code\n", "import os\n"])])
        merged, report = self.engine.sync(repo, user)
        for entry in report.entries:
            assert entry.source_preview is None

    def test_source_as_string_handled(self):
        repo = _make_nb([_code_cell("c1", ["# @cr:code\n", "a = 1\n"])])
        user_cell = _code_cell("c1", ["# @cr:code\n", "a = 1\n"])
        user_cell.source = "# @cr:code\na = 1\n"
        user = _make_nb([user_cell])
        merged, report = self.engine.sync(repo, user)
        assert len(merged.cells) == 1

    def test_embedded_ids_match_when_nbformat_ids_missing(self):
        repo = _make_nb([
            _code_cell("c1", ["# @cr:config name='settings' id=a1b2c3d4\n", "X = 1\n"]),
            _code_cell("c2", ["# @cr:code name='run' id=e5f6a7b8\n", "new_code()\n"]),
        ])
        user_cell_1 = _code_cell("random-aaa", ["# @cr:config name='settings' id=a1b2c3d4\n", "X = 42\n"])
        user_cell_2 = _code_cell("random-bbb", ["# @cr:code name='run' id=e5f6a7b8\n", "old_code()\n"])
        user = _make_nb([user_cell_1, user_cell_2])
        merged, report = self.engine.sync(repo, user)
        assert len(merged.cells) == 2
        assert "X = 42" in "".join(merged.cells[0].source)
        assert "new_code()" in "".join(merged.cells[1].source)
        assert report.counts[SyncAction.PRESERVED] == 1
        assert report.counts[SyncAction.UPDATED] == 1

    def test_embedded_ids_match_when_user_has_no_nbformat_ids(self):
        repo = _make_nb([
            _code_cell("c1", ["# @cr:code name='run' id=a1b2c3d4\n", "run()\n"]),
        ])
        user_cell = nbformat.v4.new_code_cell(source="# @cr:code name='run' id=a1b2c3d4\nold()\n")
        if "id" in user_cell:
            del user_cell["id"]
        user = _make_nb([user_cell])
        merged, report = self.engine.sync(repo, user)
        assert len(merged.cells) == 1
        assert "run()" in "".join(merged.cells[0].source)
        assert report.counts[SyncAction.UPDATED] == 1

    def test_user_added_cells_preserved_with_embedded_ids(self):
        repo = _make_nb([
            _code_cell("c1", ["# @cr:code name='run' id=a1b2c3d4\n", "run()\n"]),
            _code_cell("c2", ["# @cr:code name='done' id=e5f6a7b8\n", "done()\n"]),
        ])
        user = _make_nb([
            _code_cell("random-a", ["# @cr:code name='run' id=a1b2c3d4\n", "run()\n"]),
            _code_cell("user-custom", ["# @cr:user_code\n", "my_analysis()\n"]),
            _code_cell("random-b", ["# @cr:code name='done' id=e5f6a7b8\n", "done()\n"]),
        ])
        merged, report = self.engine.sync(repo, user)
        assert len(merged.cells) == 3
        ids = [c.id for c in merged.cells]
        assert ids.index("user-custom") > ids.index("a1b2c3d4")
        assert ids.index("user-custom") < ids.index("e5f6a7b8")


class TestDocTaggedMarkdownSync:

    def setup_method(self):
        self.engine = NotebookSyncEngine()

    def test_doc_tagged_markdown_matched_by_embedded_id(self):
        repo = _make_nb([
            _md_cell("a1b2c3d4", ["[//]: # (cr:doc name='intro' id=a1b2c3d4)\n", "# Introduction\n"]),
        ])
        user = _make_nb([
            _md_cell("different-id", ["[//]: # (cr:doc name='intro' id=a1b2c3d4)\n", "# Old Intro\n"]),
        ])
        merged, report = self.engine.sync(repo, user)
        assert len(merged.cells) == 1
        assert "# Introduction\n" in merged.cells[0].source

    def test_doc_tagged_markdown_not_in_user_added(self):
        repo = _make_nb([
            _md_cell("a1b2c3d4", ["[//]: # (cr:doc name='intro' id=a1b2c3d4)\n", "# Intro\n"]),
            _md_cell("b2c3d4e5", ["[//]: # (cr:doc name='setup' id=b2c3d4e5)\n", "# Setup\n"]),
        ])
        user = _make_nb([
            _md_cell("a1b2c3d4", ["[//]: # (cr:doc name='intro' id=a1b2c3d4)\n", "# Intro\n"]),
        ])
        merged, report = self.engine.sync(repo, user)
        assert len(merged.cells) == 2
        assert report.counts[SyncAction.ADDED] == 1

    def test_user_doc_tagged_markdown_not_in_repo_kept(self):
        repo = _make_nb([
            _md_cell("a1b2c3d4", ["[//]: # (cr:doc name='intro' id=a1b2c3d4)\n", "# Intro\n"]),
        ])
        user = _make_nb([
            _md_cell("a1b2c3d4", ["[//]: # (cr:doc name='intro' id=a1b2c3d4)\n", "# Intro\n"]),
            _md_cell("user-note", ["## My Notes\n"]),
        ])
        merged, report = self.engine.sync(repo, user)
        assert len(merged.cells) == 2
        assert any(c.id == "user-note" for c in merged.cells)

    def test_multiple_syncs_no_duplicate_sections(self):
        repo = _make_nb([
            _md_cell("a1b2c3d4", ["[//]: # (cr:doc name='intro' id=a1b2c3d4)\n", "# Intro\n"]),
            _code_cell("c1c1c1c1", ["# @cr:code name='run' id=c1c1c1c1\n", "run()\n"]),
        ])
        merged1, _ = self.engine.sync(repo, _make_nb(list(repo.cells)))
        merged2, _ = self.engine.sync(repo, merged1)
        merged3, report = self.engine.sync(repo, merged2)
        assert len(merged3.cells) == 2
        assert report.has_changes is False

    def test_untagged_user_markdown_still_kept(self):
        repo = _make_nb([
            _md_cell("a1b2c3d4", ["[//]: # (cr:doc name='intro' id=a1b2c3d4)\n", "# Intro\n"]),
        ])
        user = _make_nb([
            _md_cell("a1b2c3d4", ["[//]: # (cr:doc name='intro' id=a1b2c3d4)\n", "# Intro\n"]),
            _md_cell("user-plain", ["Some plain markdown\n"]),
        ])
        merged, report = self.engine.sync(repo, user)
        assert len(merged.cells) == 2
        assert any(c.id == "user-plain" for c in merged.cells)
        assert report.counts[SyncAction.USER_ADDED_KEPT] == 1

    def test_mixed_doc_tagged_markdown_and_code(self):
        repo = _make_nb([
            _md_cell("m1m1m1m1", ["[//]: # (cr:doc name='title' id=m1m1m1m1)\n", "# Notebook\n"]),
            _code_cell("c1c1c1c1", ["# @cr:config name='settings' id=c1c1c1c1\n", "X = 1\n"]),
            _md_cell("m2m2m2m2", ["[//]: # (cr:doc name='analysis' id=m2m2m2m2)\n", "## Analysis\n"]),
            _code_cell("c2c2c2c2", ["# @cr:code name='run' id=c2c2c2c2\n", "run()\n"]),
        ])
        user = _make_nb([
            _md_cell("rand-m1", ["[//]: # (cr:doc name='title' id=m1m1m1m1)\n", "# Old Title\n"]),
            _code_cell("rand-c1", ["# @cr:config name='settings' id=c1c1c1c1\n", "X = 42\n"]),
            _md_cell("rand-m2", ["[//]: # (cr:doc name='analysis' id=m2m2m2m2)\n", "## Analysis\n"]),
            _code_cell("rand-c2", ["# @cr:code name='run' id=c2c2c2c2\n", "old()\n"]),
        ])
        merged, report = self.engine.sync(repo, user)
        assert len(merged.cells) == 4
        assert "# Notebook\n" in merged.cells[0].source
        assert "X = 42" in "".join(merged.cells[1].source)
        assert "## Analysis\n" in merged.cells[2].source
        assert "run()" in "".join(merged.cells[3].source)
        assert report.counts[SyncAction.PRESERVED] == 1
        assert report.counts[SyncAction.UPDATED] == 2
