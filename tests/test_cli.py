"""Tests for the churnkit CLI entry point."""

from pathlib import Path
from unittest.mock import patch


class TestCLIInitProject:
    """Tests for the churnkit-init CLI command."""

    def test_default_args_creates_project(self, tmp_path):
        """CLI with --output creates a valid project."""
        from customer_retention.cli import init_project

        with patch("sys.argv", ["churnkit-init", "--output", str(tmp_path)]):
            ret = init_project()

        assert ret == 0
        assert (tmp_path / "README.md").exists()
        assert (tmp_path / ".gitignore").exists()
        assert (tmp_path / "pyproject.toml").exists()
        assert (tmp_path / "exploration_notebooks").is_dir()
        assert (tmp_path / "experiments" / "data" / "bronze").is_dir()

    def test_name_appears_in_readme(self, tmp_path):
        """--name flag is used in the generated README."""
        from customer_retention.cli import init_project

        with patch("sys.argv", [
            "churnkit-init", "--output", str(tmp_path),
            "--name", "My Churn Project",
        ]):
            ret = init_project()

        assert ret == 0
        readme = (tmp_path / "README.md").read_text()
        assert "My Churn Project" in readme

    def test_name_defaults_to_directory_name(self, tmp_path):
        """When --name is omitted, the output directory name is used."""
        from customer_retention.cli import init_project

        project_dir = tmp_path / "acme_churn"
        with patch("sys.argv", ["churnkit-init", "--output", str(project_dir)]):
            ret = init_project()

        assert ret == 0
        readme = (project_dir / "README.md").read_text()
        assert "acme_churn" in readme

    def test_name_appears_in_pyproject(self, tmp_path):
        """--name is propagated to the generated pyproject.toml."""
        from customer_retention.cli import init_project

        with patch("sys.argv", [
            "churnkit-init", "--output", str(tmp_path),
            "--name", "retail_churn",
        ]):
            init_project()

        pyproject = (tmp_path / "pyproject.toml").read_text()
        assert 'name = "retail_churn"' in pyproject

    def test_platform_local(self, tmp_path):
        """--platform local passes only LOCAL to the initializer."""
        from customer_retention.cli import init_project
        from customer_retention.generators.notebook_generator import Platform

        calls = []

        def capture_init(self, output_dir):
            calls.append({"platforms": self.platforms, "project_name": self.project_name})
            return {}

        with patch(
            "customer_retention.generators.notebook_generator.project_init.ProjectInitializer.initialize",
            capture_init,
        ):
            with patch("sys.argv", [
                "churnkit-init", "--output", str(tmp_path),
                "--platform", "local",
            ]):
                ret = init_project()

        assert ret == 0
        assert len(calls) == 1
        assert calls[0]["platforms"] == [Platform.LOCAL]

    def test_platform_databricks(self, tmp_path):
        """--platform databricks passes only DATABRICKS to the initializer."""
        from customer_retention.cli import init_project
        from customer_retention.generators.notebook_generator import Platform

        calls = []

        def capture_init(self, output_dir):
            calls.append({"platforms": self.platforms})
            return {}

        with patch(
            "customer_retention.generators.notebook_generator.project_init.ProjectInitializer.initialize",
            capture_init,
        ):
            with patch("sys.argv", [
                "churnkit-init", "--output", str(tmp_path),
                "--platform", "databricks",
            ]):
                ret = init_project()

        assert ret == 0
        assert calls[0]["platforms"] == [Platform.DATABRICKS]

    def test_platform_both(self, tmp_path):
        """--platform both (the default) passes both platforms."""
        from customer_retention.cli import init_project
        from customer_retention.generators.notebook_generator import Platform

        calls = []

        def capture_init(self, output_dir):
            calls.append({"platforms": self.platforms})
            return {}

        with patch(
            "customer_retention.generators.notebook_generator.project_init.ProjectInitializer.initialize",
            capture_init,
        ):
            with patch("sys.argv", ["churnkit-init", "--output", str(tmp_path)]):
                ret = init_project()

        assert ret == 0
        assert calls[0]["platforms"] == [Platform.LOCAL, Platform.DATABRICKS]

    def test_returns_1_on_error(self, tmp_path):
        """CLI returns exit code 1 when initialization fails."""
        from customer_retention.cli import init_project

        with patch(
            "customer_retention.generators.notebook_generator.project_init.ProjectInitializer.initialize",
            side_effect=RuntimeError("boom"),
        ):
            with patch("sys.argv", ["churnkit-init", "--output", str(tmp_path)]):
                ret = init_project()

        assert ret == 1

    def test_short_flags(self, tmp_path):
        """-o and -n short flags work."""
        from customer_retention.cli import init_project

        with patch("sys.argv", [
            "churnkit-init", "-o", str(tmp_path), "-n", "ShortFlags",
        ]):
            ret = init_project()

        assert ret == 0
        assert "ShortFlags" in (tmp_path / "README.md").read_text()

    def test_exploration_notebooks_copied(self, tmp_path):
        """CLI-created project includes exploration notebooks."""
        from customer_retention.cli import init_project

        with patch("sys.argv", ["churnkit-init", "--output", str(tmp_path)]):
            init_project()

        notebooks = list((tmp_path / "exploration_notebooks").glob("*.ipynb"))
        assert len(notebooks) > 0

    def test_full_directory_structure(self, tmp_path):
        """CLI creates the complete expected directory tree."""
        from customer_retention.cli import init_project

        with patch("sys.argv", ["churnkit-init", "--output", str(tmp_path)]):
            init_project()

        expected_dirs = [
            "exploration_notebooks",
            "generated_pipelines/local",
            "generated_pipelines/databricks",
            "experiments/findings",
            "experiments/data/bronze",
            "experiments/data/silver",
            "experiments/data/gold",
            "experiments/data/models",
            "experiments/data/predictions",
            "experiments/mlruns",
            "experiments/feature_store",
        ]
        for d in expected_dirs:
            assert (tmp_path / d).is_dir(), f"Missing directory: {d}"


class TestCLIParameterCombinations:
    """Test every meaningful combination of CLI parameters."""

    def test_output_only(self, tmp_path):
        """--output alone: name defaults to dir name, platform defaults to both."""
        from customer_retention.cli import init_project

        project_dir = tmp_path / "my_project"
        with patch("sys.argv", ["churnkit-init", "--output", str(project_dir)]):
            ret = init_project()

        assert ret == 0
        assert "my_project" in (project_dir / "README.md").read_text()
        assert 'name = "my_project"' in (project_dir / "pyproject.toml").read_text()

    def test_output_and_name(self, tmp_path):
        """--output + --name: name overrides directory name."""
        from customer_retention.cli import init_project

        project_dir = tmp_path / "dir_name"
        with patch("sys.argv", [
            "churnkit-init", "--output", str(project_dir), "--name", "custom_name",
        ]):
            ret = init_project()

        assert ret == 0
        readme = (project_dir / "README.md").read_text()
        assert "custom_name" in readme
        assert "dir_name" not in readme

    def test_output_and_platform_local(self, tmp_path):
        """--output + --platform local: end-to-end with local only."""
        from customer_retention.cli import init_project

        with patch("sys.argv", [
            "churnkit-init", "--output", str(tmp_path), "--platform", "local",
        ]):
            ret = init_project()

        assert ret == 0
        assert (tmp_path / "README.md").exists()

    def test_output_and_platform_databricks(self, tmp_path):
        """--output + --platform databricks: end-to-end with databricks only."""
        from customer_retention.cli import init_project

        with patch("sys.argv", [
            "churnkit-init", "--output", str(tmp_path), "--platform", "databricks",
        ]):
            ret = init_project()

        assert ret == 0
        assert (tmp_path / "README.md").exists()

    def test_all_three_flags(self, tmp_path):
        """--output + --name + --platform: all flags together."""
        from customer_retention.cli import init_project

        with patch("sys.argv", [
            "churnkit-init",
            "--output", str(tmp_path),
            "--name", "Full Combo",
            "--platform", "local",
        ]):
            ret = init_project()

        assert ret == 0
        assert "Full Combo" in (tmp_path / "README.md").read_text()
        assert 'name = "Full Combo"' in (tmp_path / "pyproject.toml").read_text()

    def test_all_three_flags_short(self, tmp_path):
        """Short flags: -o, -n, --platform."""
        from customer_retention.cli import init_project

        with patch("sys.argv", [
            "churnkit-init", "-o", str(tmp_path), "-n", "Short",
            "--platform", "databricks",
        ]):
            ret = init_project()

        assert ret == 0
        assert "Short" in (tmp_path / "README.md").read_text()

    def test_nested_output_dir_created(self, tmp_path):
        """Output dir with non-existent parents is created automatically."""
        from customer_retention.cli import init_project

        deep_dir = tmp_path / "a" / "b" / "c" / "project"
        with patch("sys.argv", ["churnkit-init", "--output", str(deep_dir)]):
            ret = init_project()

        assert ret == 0
        assert deep_dir.is_dir()
        assert (deep_dir / "README.md").exists()

    def test_existing_output_dir_not_clobbered(self, tmp_path):
        """Running twice on the same dir doesn't fail."""
        from customer_retention.cli import init_project

        for _ in range(2):
            with patch("sys.argv", ["churnkit-init", "--output", str(tmp_path)]):
                ret = init_project()
            assert ret == 0

    def test_name_with_spaces(self, tmp_path):
        """Project name with spaces is handled."""
        from customer_retention.cli import init_project

        with patch("sys.argv", [
            "churnkit-init", "--output", str(tmp_path),
            "--name", "My Big Project",
        ]):
            ret = init_project()

        assert ret == 0
        assert "My Big Project" in (tmp_path / "README.md").read_text()
        assert 'name = "My Big Project"' in (tmp_path / "pyproject.toml").read_text()

    def test_name_with_special_characters(self, tmp_path):
        """Project name with hyphens and underscores."""
        from customer_retention.cli import init_project

        with patch("sys.argv", [
            "churnkit-init", "--output", str(tmp_path),
            "--name", "my-churn_project_v2",
        ]):
            ret = init_project()

        assert ret == 0
        assert "my-churn_project_v2" in (tmp_path / "README.md").read_text()


class TestGeneratedReadmeCodeExample:
    """Verify the code example in the generated README actually works."""

    def test_generated_readme_import_statement(self, tmp_path):
        """The import in the generated README must resolve."""
        from customer_retention.generators.notebook_generator import (
            Platform,
            generate_orchestration_notebooks,
        )
        # If this import fails, the README's example is broken.
        assert callable(generate_orchestration_notebooks)
        assert hasattr(Platform, "LOCAL")
        assert hasattr(Platform, "DATABRICKS")

    def test_generated_readme_code_runs_without_findings(self, tmp_path):
        """The README example works when findings_path points to a missing file.

        generate_orchestration_notebooks with a non-existent findings_path should
        still produce notebooks (findings are optional and default to None-based
        generation).
        """
        from customer_retention.generators.notebook_generator import (
            Platform,
            generate_orchestration_notebooks,
        )

        results = generate_orchestration_notebooks(
            output_dir=str(tmp_path),
            platforms=[Platform.LOCAL, Platform.DATABRICKS],
        )

        assert Platform.LOCAL in results
        assert Platform.DATABRICKS in results
        assert len(results[Platform.LOCAL]) > 0
        assert len(results[Platform.DATABRICKS]) > 0

    def test_generated_readme_code_local_only(self, tmp_path):
        """README example with only LOCAL platform."""
        from customer_retention.generators.notebook_generator import (
            Platform,
            generate_orchestration_notebooks,
        )

        results = generate_orchestration_notebooks(
            output_dir=str(tmp_path),
            platforms=[Platform.LOCAL],
        )

        assert Platform.LOCAL in results
        assert Platform.DATABRICKS not in results
        for path in results[Platform.LOCAL]:
            assert Path(path).exists()

    def test_generated_readme_code_databricks_only(self, tmp_path):
        """README example with only DATABRICKS platform."""
        from customer_retention.generators.notebook_generator import (
            Platform,
            generate_orchestration_notebooks,
        )

        results = generate_orchestration_notebooks(
            output_dir=str(tmp_path),
            platforms=[Platform.DATABRICKS],
        )

        assert Platform.DATABRICKS in results
        assert Platform.LOCAL not in results
        for path in results[Platform.DATABRICKS]:
            assert Path(path).exists()

    def test_generated_readme_output_dir_matches(self, tmp_path):
        """The generated notebooks go into the expected subdirectories."""
        from customer_retention.generators.notebook_generator import (
            Platform,
            generate_orchestration_notebooks,
        )

        output_dir = tmp_path / "generated_pipelines"
        results = generate_orchestration_notebooks(
            output_dir=str(output_dir),
            platforms=[Platform.LOCAL, Platform.DATABRICKS],
        )

        # README says output goes into platform subdirs
        assert (output_dir / "local").is_dir()
        assert (output_dir / "databricks").is_dir()
        for path in results[Platform.LOCAL]:
            assert "local" in path
        for path in results[Platform.DATABRICKS]:
            assert "databricks" in path


class TestGeneratedProjectFiles:
    """Verify the generated project files are internally consistent."""

    def test_pyproject_requires_python_matches_churnkit(self, tmp_path):
        """Generated pyproject.toml requires >=3.10, matching churnkit itself."""
        from customer_retention.cli import init_project

        with patch("sys.argv", ["churnkit-init", "--output", str(tmp_path)]):
            init_project()

        pyproject = (tmp_path / "pyproject.toml").read_text()
        assert 'requires-python = ">=3.10"' in pyproject

    def test_pyproject_lists_churnkit_dependency(self, tmp_path):
        """Generated pyproject.toml includes churnkit as a dependency."""
        from customer_retention.cli import init_project

        with patch("sys.argv", ["churnkit-init", "--output", str(tmp_path)]):
            init_project()

        pyproject = (tmp_path / "pyproject.toml").read_text()
        assert '"churnkit"' in pyproject

    def test_gitignore_excludes_experiments(self, tmp_path):
        """Generated .gitignore excludes the experiments/ directory."""
        from customer_retention.cli import init_project

        with patch("sys.argv", ["churnkit-init", "--output", str(tmp_path)]):
            init_project()

        gitignore = (tmp_path / ".gitignore").read_text()
        assert "experiments/" in gitignore

    def test_readme_structure_section_matches_directories(self, tmp_path):
        """README documents the directories that actually get created."""
        from customer_retention.cli import init_project

        with patch("sys.argv", ["churnkit-init", "--output", str(tmp_path)]):
            init_project()

        readme = (tmp_path / "README.md").read_text()
        # Every directory mentioned in the README must exist
        assert "exploration_notebooks/" in readme
        assert (tmp_path / "exploration_notebooks").is_dir()
        assert "generated_pipelines/" in readme
        assert (tmp_path / "generated_pipelines").is_dir()
        assert "experiments/" in readme
        assert (tmp_path / "experiments").is_dir()
        assert "findings/" in readme
        assert (tmp_path / "experiments" / "findings").is_dir()
        assert "data/" in readme
        assert (tmp_path / "experiments" / "data").is_dir()
        assert "mlruns/" in readme
        assert (tmp_path / "experiments" / "mlruns").is_dir()
        assert "feature_store/" in readme
        assert (tmp_path / "experiments" / "feature_store").is_dir()

    def test_readme_usage_example_has_valid_import(self, tmp_path):
        """The Python import in the generated README is resolvable."""
        from customer_retention.cli import init_project

        with patch("sys.argv", ["churnkit-init", "--output", str(tmp_path)]):
            init_project()

        readme = (tmp_path / "README.md").read_text()
        assert "from customer_retention.generators.notebook_generator import" in readme
        assert "generate_orchestration_notebooks" in readme
        assert "Platform" in readme

    def test_readme_getting_started_references_existing_paths(self, tmp_path):
        """Getting Started steps reference paths that exist in the project."""
        from customer_retention.cli import init_project

        with patch("sys.argv", ["churnkit-init", "--output", str(tmp_path)]):
            init_project()

        readme = (tmp_path / "README.md").read_text()
        # "Place your data in experiments/data/"
        assert "experiments/data/" in readme
        assert (tmp_path / "experiments" / "data").is_dir()


class TestMainReadmeQuickStart:
    """Verify the main repo README Quick Start section works end-to-end."""

    def test_churnkit_init_output_flag(self, tmp_path):
        """README step 2: churnkit-init --output ./my_project"""
        from customer_retention.cli import init_project

        project_dir = tmp_path / "my_project"
        with patch("sys.argv", ["churnkit-init", "--output", str(project_dir)]):
            ret = init_project()

        assert ret == 0
        assert project_dir.is_dir()

    def test_exploration_notebook_01_exists_after_init(self, tmp_path):
        """README step 3: exploration_notebooks/01_data_discovery.ipynb must exist."""
        from customer_retention.cli import init_project

        with patch("sys.argv", ["churnkit-init", "--output", str(tmp_path)]):
            init_project()

        nb_path = tmp_path / "exploration_notebooks" / "01_data_discovery.ipynb"
        assert nb_path.exists(), "01_data_discovery.ipynb not copied to project"

    def test_exploration_notebook_01_has_data_path(self, tmp_path):
        """README step 3: the notebook should contain a DATA_PATH variable."""
        import json

        from customer_retention.cli import init_project

        with patch("sys.argv", ["churnkit-init", "--output", str(tmp_path)]):
            init_project()

        nb_path = tmp_path / "exploration_notebooks" / "01_data_discovery.ipynb"
        nb = json.loads(nb_path.read_text())
        all_source = "\n".join(
            "".join(cell["source"])
            for cell in nb["cells"]
            if cell["cell_type"] == "code"
        )
        assert "DATA_PATH" in all_source

    def test_ml_extra_exists(self):
        """README step 1: pip install 'churnkit[ml]' requires the ml extra to exist."""
        import importlib.metadata

        # Check that [ml] extra is defined in package metadata
        requires = importlib.metadata.requires("churnkit") or []
        ml_extras = [r for r in requires if "extra == 'ml'" in r]
        assert len(ml_extras) > 0, "No [ml] optional dependencies found in package metadata"
