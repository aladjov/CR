from typing import Dict, List, Optional
import nbformat


DATABRICKS_SEPARATOR = "\n# COMMAND ----------\n"


class CellBuilder:
    @staticmethod
    def markdown(content: str) -> nbformat.NotebookNode:
        return nbformat.v4.new_markdown_cell(content)

    @staticmethod
    def code(source: str, metadata: Optional[Dict] = None) -> nbformat.NotebookNode:
        cell = nbformat.v4.new_code_cell(source)
        if metadata:
            cell.metadata.update(metadata)
        return cell

    @staticmethod
    def header(title: str, level: int = 1) -> nbformat.NotebookNode:
        return CellBuilder.markdown(f"{'#' * level} {title}")

    @staticmethod
    def section(title: str, description: str = "") -> nbformat.NotebookNode:
        content = f"## {title}"
        if description:
            content += f"\n\n{description}"
        return CellBuilder.markdown(content)

    @staticmethod
    def databricks_separator() -> str:
        return DATABRICKS_SEPARATOR

    @staticmethod
    def create_notebook(cells: List[nbformat.NotebookNode]) -> nbformat.NotebookNode:
        nb = nbformat.v4.new_notebook()
        nb.cells = cells
        return nb

    @staticmethod
    def imports_cell(imports: List[str]) -> nbformat.NotebookNode:
        lines = [f"import {imp}" for imp in imports]
        return CellBuilder.code("\n".join(lines))

    @staticmethod
    def from_imports_cell(from_imports: Dict[str, List[str]]) -> nbformat.NotebookNode:
        lines = [f"from {module} import {', '.join(names)}" for module, names in from_imports.items()]
        return CellBuilder.code("\n".join(lines))
