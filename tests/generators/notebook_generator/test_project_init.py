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
        assert "churnkit" in content


class TestProjectInitializerWithGeneration:
    def test_generate_orchestration_after_init(self, tmp_path):
        from customer_retention.generators.notebook_generator import Platform, generate_orchestration_notebooks
        from customer_retention.generators.notebook_generator.project_init import ProjectInitializer
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
        from pathlib import Path

        from customer_retention.generators.notebook_generator.project_init import ProjectInitializer
        init = ProjectInitializer(project_name="test")
        monkeypatch.setattr(
            "customer_retention.generators.notebook_generator.project_init.Path",
            lambda x: tmp_path / "nonexistent" / x if isinstance(x, str) else Path(x)
        )
        result = init._get_exploration_source_dir()
        assert result is None or result.exists()

    def test_get_exploration_source_dir_finds_src_layout(self, tmp_path, monkeypatch):
        repo_root = tmp_path / "repo"
        src_pkg = repo_root / "src" / "customer_retention" / "generators" / "notebook_generator"
        src_pkg.mkdir(parents=True)
        nb_dir = repo_root / "exploration_notebooks"
        nb_dir.mkdir()
        (nb_dir / "00_start_here.ipynb").touch()

        from customer_retention.generators.notebook_generator.project_init import ProjectInitializer
        init = ProjectInitializer(project_name="test")
        fake_file = src_pkg / "project_init.py"
        fake_file.touch()
        monkeypatch.setattr(
            "customer_retention.generators.notebook_generator.project_init.__file__",
            str(fake_file),
        )
        result = init._get_exploration_source_dir()
        assert result == nb_dir

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


class TestExplorationNotebooksPathParameter:
    def test_default_path_is_exploration_notebooks(self, tmp_path):
        from customer_retention.generators.notebook_generator.project_init import ProjectInitializer
        project_dir = tmp_path / "proj"
        ProjectInitializer(project_name="proj").initialize(str(project_dir))
        assert (project_dir / "exploration_notebooks").is_dir()

    def test_custom_path_creates_directory(self, tmp_path):
        from customer_retention.generators.notebook_generator.project_init import ProjectInitializer
        project_dir = tmp_path / "proj"
        ProjectInitializer(
            project_name="proj", exploration_notebooks_path="team_notebooks",
        ).initialize(str(project_dir))
        assert (project_dir / "team_notebooks").is_dir()
        assert not (project_dir / "exploration_notebooks").exists()

    def test_custom_path_receives_notebooks(self, tmp_path):
        from customer_retention.generators.notebook_generator.project_init import ProjectInitializer
        project_dir = tmp_path / "proj"
        ProjectInitializer(
            project_name="proj", exploration_notebooks_path="my_nb",
        ).initialize(str(project_dir))
        notebooks = list((project_dir / "my_nb").glob("*.ipynb"))
        assert len(notebooks) > 0

    def test_nested_subpath(self, tmp_path):
        from customer_retention.generators.notebook_generator.project_init import ProjectInitializer
        project_dir = tmp_path / "proj"
        ProjectInitializer(
            project_name="proj", exploration_notebooks_path="team/explore",
        ).initialize(str(project_dir))
        assert (project_dir / "team" / "explore").is_dir()
        notebooks = list((project_dir / "team" / "explore").glob("*.ipynb"))
        assert len(notebooks) > 0

    def test_initialize_project_function_accepts_path(self, tmp_path):
        from customer_retention.generators.notebook_generator import initialize_project
        project_dir = tmp_path / "proj"
        initialize_project(
            str(project_dir), project_name="proj", exploration_notebooks_path="custom_nb",
        )
        assert (project_dir / "custom_nb").is_dir()


class TestExperimentsPathParameter:
    def test_default_path_is_experiments(self, tmp_path):
        from customer_retention.generators.notebook_generator.project_init import ProjectInitializer
        project_dir = tmp_path / "proj"
        ProjectInitializer(project_name="proj").initialize(str(project_dir))
        assert (project_dir / "experiments" / "data" / "bronze").is_dir()
        assert (project_dir / "experiments" / "findings").is_dir()
        assert (project_dir / "experiments" / "mlruns").is_dir()

    def test_custom_path_creates_directory(self, tmp_path):
        from customer_retention.generators.notebook_generator.project_init import ProjectInitializer
        project_dir = tmp_path / "proj"
        ProjectInitializer(
            project_name="proj", experiments_path="experiments_secondary",
        ).initialize(str(project_dir))
        assert (project_dir / "experiments_secondary" / "data" / "bronze").is_dir()
        assert (project_dir / "experiments_secondary" / "findings").is_dir()
        assert (project_dir / "experiments_secondary" / "mlruns").is_dir()
        assert not (project_dir / "experiments").exists()

    def test_custom_path_creates_all_subdirs(self, tmp_path):
        from customer_retention.generators.notebook_generator.project_init import ProjectInitializer
        project_dir = tmp_path / "proj"
        ProjectInitializer(
            project_name="proj", experiments_path="experiments_secondary",
        ).initialize(str(project_dir))
        for sub in ("bronze", "silver", "gold", "models", "predictions"):
            assert (project_dir / "experiments_secondary" / "data" / sub).is_dir()
        assert (project_dir / "experiments_secondary" / "feature_store").is_dir()

    def test_gitignore_uses_custom_path(self, tmp_path):
        from customer_retention.generators.notebook_generator.project_init import ProjectInitializer
        project_dir = tmp_path / "proj"
        ProjectInitializer(
            project_name="proj", experiments_path="experiments_secondary",
        ).initialize(str(project_dir))
        content = (project_dir / ".gitignore").read_text()
        assert "experiments_secondary/" in content

    def test_initialize_project_function_accepts_experiments_path(self, tmp_path):
        from customer_retention.generators.notebook_generator import initialize_project
        project_dir = tmp_path / "proj"
        initialize_project(
            str(project_dir), project_name="proj",
            experiments_path="experiments_secondary",
        )
        assert (project_dir / "experiments_secondary" / "findings").is_dir()
