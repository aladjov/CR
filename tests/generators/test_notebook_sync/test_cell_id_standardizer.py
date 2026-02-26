import nbformat

from customer_retention.generators.notebook_sync.cell_id_standardizer import (
    notebook_prefix_from_filename,
    split_blended_cell,
    standardize_cell_ids,
)


class TestNotebookPrefixFromFilename:

    def test_simple_numbered(self):
        assert notebook_prefix_from_filename("00_start_here.ipynb") == "nb00"

    def test_data_discovery(self):
        assert notebook_prefix_from_filename("01_data_discovery.ipynb") == "nb01"

    def test_suffix_letter(self):
        assert notebook_prefix_from_filename("01a_temporal_deep_dive.ipynb") == "nb01a"

    def test_double_suffix(self):
        assert notebook_prefix_from_filename("01a_a_temporal_text_deep_dive.ipynb") == "nb01aa"

    def test_negative_number(self):
        assert notebook_prefix_from_filename("-1_sample_datasets.ipynb") == "nbm1"

    def test_two_digit(self):
        assert notebook_prefix_from_filename("10_spec_generation.ipynb") == "nb10"

    def test_twelve(self):
        assert notebook_prefix_from_filename("12_view_documentation.ipynb") == "nb12"

    def test_no_extension(self):
        assert notebook_prefix_from_filename("05_relationship_analysis") == "nb05"

    def test_letter_suffixes(self):
        assert notebook_prefix_from_filename("01b_temporal_quality.ipynb") == "nb01b"
        assert notebook_prefix_from_filename("01c_temporal_patterns.ipynb") == "nb01c"
        assert notebook_prefix_from_filename("01d_event_aggregation.ipynb") == "nb01d"

    def test_04a(self):
        assert notebook_prefix_from_filename("04a_text_columns_deep_dive.ipynb") == "nb04a"


class TestStandardizeCellIds:

    def _make_nb(self, cell_ids):
        nb = nbformat.v4.new_notebook()
        for cid in cell_ids:
            cell = nbformat.v4.new_code_cell(source="x = 1")
            cell.id = cid
            nb.cells.append(cell)
        return nb

    def test_sequential_ids(self):
        nb = self._make_nb(["abc", "def", "ghi"])
        result = standardize_cell_ids(nb, "00_start_here.ipynb")
        assert [c.id for c in result.cells] == ["nb00-001", "nb00-002", "nb00-003"]

    def test_preserves_source(self):
        nb = nbformat.v4.new_notebook()
        cell = nbformat.v4.new_code_cell(source="hello = 1")
        cell.id = "old-id"
        nb.cells.append(cell)
        result = standardize_cell_ids(nb, "01_data_discovery.ipynb")
        assert result.cells[0].source == "hello = 1"

    def test_preserves_markdown(self):
        nb = nbformat.v4.new_notebook()
        md = nbformat.v4.new_markdown_cell(source="# Title")
        md.id = "old-md"
        nb.cells.append(md)
        result = standardize_cell_ids(nb, "02_source_integrity.ipynb")
        assert result.cells[0].id == "nb02-001"
        assert result.cells[0].source == "# Title"

    def test_uniqueness(self):
        nb = self._make_nb([f"c{i}" for i in range(50)])
        result = standardize_cell_ids(nb, "05_relationship_analysis.ipynb")
        ids = [c.id for c in result.cells]
        assert len(ids) == len(set(ids))

    def test_mixed_cell_types(self):
        nb = nbformat.v4.new_notebook()
        nb.cells.append(nbformat.v4.new_markdown_cell("# Title"))
        nb.cells[-1].id = "m1"
        nb.cells.append(nbformat.v4.new_code_cell("x = 1"))
        nb.cells[-1].id = "c1"
        nb.cells.append(nbformat.v4.new_markdown_cell("## Section"))
        nb.cells[-1].id = "m2"
        result = standardize_cell_ids(nb, "03_dataset_merge.ipynb")
        assert [c.id for c in result.cells] == ["nb03-001", "nb03-002", "nb03-003"]

    def test_preserves_outputs_and_execution_count(self):
        nb = nbformat.v4.new_notebook()
        cell = nbformat.v4.new_code_cell(source="print(1)")
        cell.id = "old"
        cell.outputs = [{"output_type": "stream", "name": "stdout", "text": ["1\n"]}]
        cell.execution_count = 5
        nb.cells.append(cell)
        result = standardize_cell_ids(nb, "00_start_here.ipynb")
        assert result.cells[0].outputs == cell.outputs
        assert result.cells[0].execution_count == 5


class TestSplitBlendedCell:

    def _make_code_cell(self, source_lines, cell_id="test-cell"):
        cell = nbformat.v4.new_code_cell(source="".join(source_lines))
        cell.id = cell_id
        cell.source = source_lines
        return cell

    def test_basic_split(self):
        lines = [
            "PROJECT_NAME = 'test'\n",
            "LIGHT_RUN = False\n",
            "\n",
            "from customer_retention import something\n",
            "do_stuff()\n",
        ]
        cell = self._make_code_cell(lines)
        config, code = split_blended_cell(cell, 2, "cfg-001", "code-001")
        assert config.id == "cfg-001"
        assert code.id == "code-001"
        assert config.source == ["PROJECT_NAME = 'test'\n", "LIGHT_RUN = False\n"]
        assert code.source == ["\n", "from customer_retention import something\n", "do_stuff()\n"]

    def test_split_preserves_outputs_on_code_cell(self):
        lines = ["X = 1\n", "print(X)\n"]
        cell = self._make_code_cell(lines)
        cell.outputs = [{"output_type": "stream", "name": "stdout", "text": ["1\n"]}]
        cell.execution_count = 3
        config, code = split_blended_cell(cell, 1, "cfg", "code")
        assert config.outputs == []
        assert config.execution_count is None
        assert code.outputs == cell.outputs
        assert code.execution_count == 3

    def test_split_at_boundary(self):
        lines = ["X = 1\n"]
        cell = self._make_code_cell(lines)
        config, code = split_blended_cell(cell, 1, "cfg", "code")
        assert config.source == ["X = 1\n"]
        assert code.source == []

    def test_split_preserves_metadata(self):
        lines = ["X = 1\n", "import os\n"]
        cell = self._make_code_cell(lines)
        cell.metadata["tags"] = ["important"]
        config, code = split_blended_cell(cell, 1, "cfg", "code")
        assert config.metadata.get("tags") == ["important"]

    def test_source_as_string_split(self):
        cell = nbformat.v4.new_code_cell(source="X = 1\nimport os\n")
        cell.id = "test"
        config, code = split_blended_cell(cell, 1, "cfg", "code")
        assert "".join(config.source) == "X = 1\n"
        assert "".join(code.source) == "import os\n"
