

class TestCellBuilder:
    def test_markdown_cell(self):
        from customer_retention.generators.notebook_generator.cell_builder import CellBuilder
        cell = CellBuilder.markdown("Test content")
        assert cell.cell_type == "markdown"
        assert cell.source == "Test content"

    def test_code_cell(self):
        from customer_retention.generators.notebook_generator.cell_builder import CellBuilder
        cell = CellBuilder.code("print('hello')")
        assert cell.cell_type == "code"
        assert cell.source == "print('hello')"

    def test_code_cell_with_metadata(self):
        from customer_retention.generators.notebook_generator.cell_builder import CellBuilder
        cell = CellBuilder.code("x = 1", metadata={"tags": ["skip"]})
        assert cell.metadata.get("tags") == ["skip"]

    def test_header_cell_level_1(self):
        from customer_retention.generators.notebook_generator.cell_builder import CellBuilder
        cell = CellBuilder.header("Title")
        assert cell.cell_type == "markdown"
        assert cell.source == "# Title"

    def test_header_cell_level_2(self):
        from customer_retention.generators.notebook_generator.cell_builder import CellBuilder
        cell = CellBuilder.header("Section", level=2)
        assert cell.source == "## Section"

    def test_section_cell_with_description(self):
        from customer_retention.generators.notebook_generator.cell_builder import CellBuilder
        cell = CellBuilder.section("Data Loading", "Load data from source")
        assert "## Data Loading" in cell.source
        assert "Load data from source" in cell.source

    def test_section_cell_without_description(self):
        from customer_retention.generators.notebook_generator.cell_builder import CellBuilder
        cell = CellBuilder.section("Data Loading")
        assert cell.source == "## Data Loading"

    def test_databricks_separator(self):
        from customer_retention.generators.notebook_generator.cell_builder import CellBuilder
        sep = CellBuilder.databricks_separator()
        assert "COMMAND ----------" in sep

    def test_create_notebook_empty(self):
        from customer_retention.generators.notebook_generator.cell_builder import CellBuilder
        nb = CellBuilder.create_notebook([])
        assert nb.nbformat == 4
        assert len(nb.cells) == 0

    def test_create_notebook_with_cells(self):
        from customer_retention.generators.notebook_generator.cell_builder import CellBuilder
        cells = [
            CellBuilder.header("Test Notebook"),
            CellBuilder.code("x = 1"),
        ]
        nb = CellBuilder.create_notebook(cells)
        assert len(nb.cells) == 2

    def test_imports_cell(self):
        from customer_retention.generators.notebook_generator.cell_builder import CellBuilder
        imports = ["pandas as pd", "numpy as np"]
        cell = CellBuilder.imports_cell(imports)
        assert "import pandas as pd" in cell.source
        assert "import numpy as np" in cell.source

    def test_from_imports_cell(self):
        from customer_retention.generators.notebook_generator.cell_builder import CellBuilder
        from_imports = {"customer_retention.stages.cleaning": ["MissingValueHandler", "OutlierHandler"]}
        cell = CellBuilder.from_imports_cell(from_imports)
        assert "from customer_retention.stages.cleaning import MissingValueHandler, OutlierHandler" in cell.source
