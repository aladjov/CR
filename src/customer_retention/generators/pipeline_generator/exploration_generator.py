from __future__ import annotations

from pathlib import Path

from customer_retention.analysis.auto_explorer.project_context import ProjectContext

from .databricks_renderer import DatabricksCodeRenderer


class DatabricksExplorationGenerator:
    def __init__(
        self,
        project_context_path: str,
        output_dir: str,
        notebooks_base_path: str = "./exploration_notebooks",
        findings_base_path: str = "./experiments/findings",
        catalog: str = "main",
        schema: str = "default",
    ):
        self._context_path = Path(project_context_path)
        self._output_dir = Path(output_dir)
        self._notebooks_base_path = notebooks_base_path
        self._findings_base_path = findings_base_path
        self._renderer = DatabricksCodeRenderer(catalog=catalog, schema=schema)

    def generate(self) -> Path:
        context = ProjectContext.load(self._context_path)
        code = self._renderer.render_exploration_runner(
            project_name=context.project_name,
            datasets=context.datasets,
            notebooks_base_path=self._notebooks_base_path,
            findings_base_path=self._findings_base_path,
        )
        output_path = self._output_dir / "exploration_runner.py"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(code)
        return output_path

    def generate_workflow(self) -> Path:
        context = ProjectContext.load(self._context_path)
        yaml_content = self._renderer.render_for_each_workflow(
            project_name=context.project_name,
            notebooks_base_path=self._notebooks_base_path,
        )
        output_path = self._output_dir / "databricks_bundle.yaml"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(yaml_content)
        return output_path
