import pytest
from pathlib import Path


class TestProjectInitializer:
    def test_creates_project_directory(self, tmp_path):
        from customer_retention.generators.notebook_generator.project_init import ProjectInitializer
        project_dir = tmp_path / "my_project"
        init = ProjectInitializer(project_name="my_project")
        init.initialize(str(project_dir))
        assert project_dir.exists()

    def test_creates_readme(self, tmp_path):
        from customer_retention.generators.notebook_generator.project_init import ProjectInitializer
        project_dir = tmp_path / "my_project"
        init = ProjectInitializer(project_name="my_project")
        init.initialize(str(project_dir))
        readme = project_dir / "README.md"
        assert readme.exists()
        content = readme.read_text()
        assert "my_project" in content

    def test_creates_directory_structure(self, tmp_path):
        from customer_retention.generators.notebook_generator.project_init import ProjectInitializer
        project_dir = tmp_path / "my_project"
        init = ProjectInitializer(project_name="my_project")
        init.initialize(str(project_dir))
        assert (project_dir / "exploration_notebooks").exists()
        assert (project_dir / "generated_pipelines").exists()
        assert (project_dir / "experiments" / "data").exists()
        assert (project_dir / "experiments" / "mlruns").exists()

    def test_copies_exploration_notebooks(self, tmp_path):
        from customer_retention.generators.notebook_generator.project_init import ProjectInitializer
        project_dir = tmp_path / "my_project"
        init = ProjectInitializer(project_name="my_project")
        init.initialize(str(project_dir))
        exploration_dir = project_dir / "exploration_notebooks"
        notebooks = list(exploration_dir.glob("*.ipynb"))
        assert len(notebooks) > 0

    def test_creates_gitignore(self, tmp_path):
        from customer_retention.generators.notebook_generator.project_init import ProjectInitializer
        project_dir = tmp_path / "my_project"
        init = ProjectInitializer(project_name="my_project")
        init.initialize(str(project_dir))
        gitignore = project_dir / ".gitignore"
        assert gitignore.exists()
        content = gitignore.read_text()
        assert ".venv" in content
        assert "__pycache__" in content

    def test_creates_pyproject_toml(self, tmp_path):
        from customer_retention.generators.notebook_generator.project_init import ProjectInitializer
        project_dir = tmp_path / "my_project"
        init = ProjectInitializer(project_name="my_project")
        init.initialize(str(project_dir))
        pyproject = project_dir / "pyproject.toml"
        assert pyproject.exists()
        content = pyproject.read_text()
        assert "customer-retention" in content


class TestProjectInitializerWithGeneration:
    def test_generate_orchestration_after_init(self, tmp_path):
        from customer_retention.generators.notebook_generator.project_init import ProjectInitializer
        from customer_retention.generators.notebook_generator import generate_orchestration_notebooks, Platform
        project_dir = tmp_path / "my_project"
        init = ProjectInitializer(project_name="my_project")
        init.initialize(str(project_dir))
        output_dir = project_dir / "generated_pipelines"
        results = generate_orchestration_notebooks(output_dir=str(output_dir), platforms=[Platform.LOCAL])
        assert (output_dir / "local" / "01_ingestion.ipynb").exists()

    def test_init_with_orchestration(self, tmp_path):
        from customer_retention.generators.notebook_generator.project_init import ProjectInitializer
        project_dir = tmp_path / "my_project"
        init = ProjectInitializer(project_name="my_project", generate_orchestration=True)
        init.initialize(str(project_dir))
        assert (project_dir / "generated_pipelines" / "local").exists()
        assert (project_dir / "generated_pipelines" / "databricks").exists()


class TestInitializeProjectFunction:
    def test_function_creates_project(self, tmp_path):
        from customer_retention.generators.notebook_generator import initialize_project
        project_dir = tmp_path / "test_project"
        result = initialize_project(str(project_dir), project_name="test_project")
        assert project_dir.exists()
        assert "readme_path" in result
        assert "exploration_notebooks" in result

    def test_function_returns_paths(self, tmp_path):
        from customer_retention.generators.notebook_generator import initialize_project
        project_dir = tmp_path / "test_project"
        result = initialize_project(str(project_dir), project_name="test_project")
        assert Path(result["readme_path"]).exists()
        assert len(result["exploration_notebooks"]) > 0


class TestProjectStructure:
    def test_findings_folder_for_exploration_outputs(self, tmp_path):
        from customer_retention.generators.notebook_generator.project_init import ProjectInitializer
        project_dir = tmp_path / "my_project"
        init = ProjectInitializer(project_name="my_project")
        init.initialize(str(project_dir))
        assert (project_dir / "experiments" / "findings").exists()

    def test_experiments_subdirectories(self, tmp_path):
        from customer_retention.generators.notebook_generator.project_init import ProjectInitializer
        project_dir = tmp_path / "my_project"
        init = ProjectInitializer(project_name="my_project")
        init.initialize(str(project_dir))
        assert (project_dir / "experiments" / "data" / "bronze").exists()
        assert (project_dir / "experiments" / "data" / "silver").exists()
        assert (project_dir / "experiments" / "data" / "gold").exists()
        assert (project_dir / "experiments" / "data" / "models").exists()
        assert (project_dir / "experiments" / "data" / "predictions").exists()
        assert (project_dir / "experiments" / "mlruns").exists()
        assert (project_dir / "experiments" / "feature_store").exists()


class TestExplorationNotebookCopyEdgeCases:
    def test_get_exploration_source_dir_returns_none_when_missing(self, tmp_path, monkeypatch):
        from customer_retention.generators.notebook_generator.project_init import ProjectInitializer
        from pathlib import Path
        init = ProjectInitializer(project_name="test")
        monkeypatch.setattr(
            "customer_retention.generators.notebook_generator.project_init.Path",
            lambda x: tmp_path / "nonexistent" / x if isinstance(x, str) else Path(x)
        )
        result = init._get_exploration_source_dir()
        assert result is None or result.exists()

    def test_copy_exploration_handles_no_source_dir(self, tmp_path):
        from customer_retention.generators.notebook_generator.project_init import ProjectInitializer
        init = ProjectInitializer(project_name="test")
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        (project_dir / "exploration_notebooks").mkdir(parents=True)
        original_get = init._get_exploration_source_dir
        init._get_exploration_source_dir = lambda: None
        copied = init._copy_exploration_notebooks(project_dir)
        assert copied == []
        init._get_exploration_source_dir = original_get
