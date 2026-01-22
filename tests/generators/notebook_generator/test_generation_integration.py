import nbformat


class TestFullNotebookGenerationWorkflow:
    def test_generate_validate_and_report_local_notebooks(self, tmp_path):
        from customer_retention.generators.notebook_generator import Platform, generate_and_validate_notebooks
        results = generate_and_validate_notebooks(
            output_dir=str(tmp_path),
            platforms=[Platform.LOCAL]
        )
        assert Platform.LOCAL in results
        result = results[Platform.LOCAL]
        assert result.all_valid
        assert len(result.notebook_paths) == 10  # Holdout stage disabled by default
        assert result.validation_report.total_notebooks == 10
        assert result.validation_report.all_passed
        report_path = tmp_path / "local" / "VALIDATION_REPORT.md"
        assert report_path.exists()
        report_content = report_path.read_text()
        assert "PASSED" in report_content
        assert "01_ingestion" in report_content

    def test_generate_validate_and_report_databricks_notebooks(self, tmp_path):
        from customer_retention.generators.notebook_generator import Platform, generate_and_validate_notebooks
        results = generate_and_validate_notebooks(
            output_dir=str(tmp_path),
            platforms=[Platform.DATABRICKS]
        )
        assert Platform.DATABRICKS in results
        assert results[Platform.DATABRICKS].all_valid
        assert (tmp_path / "databricks" / "VALIDATION_REPORT.md").exists()

    def test_generate_both_platforms_full_workflow(self, tmp_path):
        from customer_retention.generators.notebook_generator import Platform, generate_and_validate_notebooks
        results = generate_and_validate_notebooks(output_dir=str(tmp_path))
        assert Platform.LOCAL in results
        assert Platform.DATABRICKS in results
        assert results[Platform.LOCAL].all_valid
        assert results[Platform.DATABRICKS].all_valid


class TestFullScriptGenerationWorkflow:
    def test_generate_and_validate_local_scripts(self, tmp_path):
        from customer_retention.generators.notebook_generator import Platform, generate_orchestration_scripts
        from customer_retention.generators.notebook_generator.runner import ScriptRunner
        results = generate_orchestration_scripts(
            output_dir=str(tmp_path),
            platforms=[Platform.LOCAL]
        )
        assert Platform.LOCAL in results
        assert len(results[Platform.LOCAL]) == 10  # Holdout stage disabled by default
        runner = ScriptRunner()
        report = runner.validate_sequence(str(tmp_path / "local"), "local")
        assert report.all_passed
        assert report.total_notebooks == 10

    def test_generate_and_validate_databricks_scripts(self, tmp_path):
        from customer_retention.generators.notebook_generator import Platform, generate_orchestration_scripts
        from customer_retention.generators.notebook_generator.runner import ScriptRunner
        results = generate_orchestration_scripts(
            output_dir=str(tmp_path),
            platforms=[Platform.DATABRICKS]
        )
        assert Platform.DATABRICKS in results
        runner = ScriptRunner()
        report = runner.validate_sequence(str(tmp_path / "databricks"), "databricks")
        assert report.all_passed


class TestFullProjectInitWorkflow:
    def test_initialize_and_generate_orchestration(self, tmp_path):
        from customer_retention.generators.notebook_generator import (
            Platform,
            generate_orchestration_notebooks,
            initialize_project,
        )
        project_dir = tmp_path / "my_project"
        result = initialize_project(str(project_dir), project_name="my_project")
        assert project_dir.exists()
        assert (project_dir / "README.md").exists()
        assert (project_dir / "generated_pipelines").exists()
        orchestration_dir = project_dir / "generated_pipelines"
        generate_orchestration_notebooks(
            output_dir=str(orchestration_dir),
            platforms=[Platform.LOCAL]
        )
        assert (orchestration_dir / "local" / "01_ingestion.ipynb").exists()

    def test_initialize_with_auto_orchestration(self, tmp_path):
        from customer_retention.generators.notebook_generator.project_init import ProjectInitializer
        project_dir = tmp_path / "auto_project"
        init = ProjectInitializer(
            project_name="auto_project",
            generate_orchestration=True
        )
        init.initialize(str(project_dir))
        assert (project_dir / "generated_pipelines" / "local").exists()
        assert (project_dir / "generated_pipelines" / "databricks").exists()


class TestNotebookContentIntegrity:
    def test_local_notebooks_have_framework_imports(self, tmp_path):
        from customer_retention.generators.notebook_generator import Platform, generate_orchestration_notebooks
        generate_orchestration_notebooks(
            output_dir=str(tmp_path),
            platforms=[Platform.LOCAL]
        )
        nb_path = tmp_path / "local" / "01_ingestion.ipynb"
        with open(nb_path, "r") as f:
            nb = nbformat.read(f, as_version=4)
        all_code = "\n".join(cell.source for cell in nb.cells if cell.cell_type == "code")
        assert "customer_retention" in all_code
        assert "from customer_retention" in all_code

    def test_databricks_notebooks_have_no_framework_imports(self, tmp_path):
        from customer_retention.generators.notebook_generator import Platform, generate_orchestration_notebooks
        generate_orchestration_notebooks(
            output_dir=str(tmp_path),
            platforms=[Platform.DATABRICKS]
        )
        nb_path = tmp_path / "databricks" / "01_ingestion.ipynb"
        with open(nb_path, "r") as f:
            nb = nbformat.read(f, as_version=4)
        all_code = "\n".join(cell.source for cell in nb.cells if cell.cell_type == "code")
        assert "from customer_retention" not in all_code
        assert "spark" in all_code

    def test_all_notebooks_have_valid_python_syntax(self, tmp_path):
        from customer_retention.generators.notebook_generator import generate_orchestration_notebooks
        generate_orchestration_notebooks(output_dir=str(tmp_path))
        for platform in ["local", "databricks"]:
            platform_dir = tmp_path / platform
            for nb_path in platform_dir.glob("*.ipynb"):
                with open(nb_path, "r") as f:
                    nb = nbformat.read(f, as_version=4)
                all_code = "\n".join(
                    cell.source for cell in nb.cells if cell.cell_type == "code"
                )
                compile(all_code, f"<{nb_path.name}>", "exec")


class TestScriptContentIntegrity:
    def test_local_scripts_have_framework_imports(self, tmp_path):
        from customer_retention.generators.notebook_generator import Platform, generate_orchestration_scripts
        generate_orchestration_scripts(
            output_dir=str(tmp_path),
            platforms=[Platform.LOCAL]
        )
        script_path = tmp_path / "local" / "01_ingestion.py"
        content = script_path.read_text()
        assert "customer_retention" in content

    def test_databricks_scripts_have_no_framework_imports(self, tmp_path):
        from customer_retention.generators.notebook_generator import Platform, generate_orchestration_scripts
        generate_orchestration_scripts(
            output_dir=str(tmp_path),
            platforms=[Platform.DATABRICKS]
        )
        script_path = tmp_path / "databricks" / "01_ingestion.py"
        content = script_path.read_text()
        assert "from customer_retention" not in content
        assert "spark" in content

    def test_all_scripts_have_main_block(self, tmp_path):
        from customer_retention.generators.notebook_generator import generate_orchestration_scripts
        generate_orchestration_scripts(output_dir=str(tmp_path))
        for platform in ["local", "databricks"]:
            platform_dir = tmp_path / platform
            for script_path in platform_dir.glob("*.py"):
                content = script_path.read_text()
                assert 'if __name__ == "__main__"' in content

    def test_all_scripts_have_docstrings(self, tmp_path):
        from customer_retention.generators.notebook_generator import generate_orchestration_scripts
        generate_orchestration_scripts(output_dir=str(tmp_path))
        for platform in ["local", "databricks"]:
            platform_dir = tmp_path / platform
            for script_path in platform_dir.glob("*.py"):
                content = script_path.read_text()
                assert '"""' in content


class TestGenerationWithCustomConfig:
    def test_notebooks_with_custom_config(self, tmp_path):
        from customer_retention.generators.notebook_generator import (
            FeatureStoreConfig,
            MLflowConfig,
            NotebookConfig,
            Platform,
            generate_orchestration_notebooks,
        )
        config = NotebookConfig(
            project_name="custom_project",
            mlflow=MLflowConfig(
                tracking_uri="http://localhost:5000",
                experiment_name="custom_experiment"
            ),
            feature_store=FeatureStoreConfig(
                catalog="prod",
                schema="ml_features"
            )
        )
        results = generate_orchestration_notebooks(
            output_dir=str(tmp_path),
            platforms=[Platform.DATABRICKS],
            config=config
        )
        nb_path = tmp_path / "databricks" / "01_ingestion.ipynb"
        with open(nb_path, "r") as f:
            nb = nbformat.read(f, as_version=4)
        all_code = "\n".join(cell.source for cell in nb.cells if cell.cell_type == "code")
        assert "prod" in all_code
        assert "ml_features" in all_code


class TestValidationReportContent:
    def test_report_contains_all_stage_names(self, tmp_path):
        from customer_retention.generators.notebook_generator import Platform, generate_and_validate_notebooks
        results = generate_and_validate_notebooks(
            output_dir=str(tmp_path),
            platforms=[Platform.LOCAL]
        )
        report = results[Platform.LOCAL].validation_report
        names = [r.notebook_name for r in report.results]
        expected_stages = [
            "01_ingestion", "02_profiling", "03_cleaning", "04_transformation",
            "05_feature_engineering", "06_feature_selection", "07_model_training",
            "08_deployment", "09_monitoring", "10_batch_inference"
        ]
        for stage in expected_stages:
            assert stage in names

    def test_markdown_report_is_well_formatted(self, tmp_path):
        from customer_retention.generators.notebook_generator import Platform, generate_and_validate_notebooks
        results = generate_and_validate_notebooks(
            output_dir=str(tmp_path),
            platforms=[Platform.LOCAL]
        )
        report_path = tmp_path / "local" / "VALIDATION_REPORT.md"
        content = report_path.read_text()
        assert "# Notebook Validation Report" in content
        assert "## Summary" in content
        assert "## Results" in content
        assert "| Notebook | Status | Duration | Error |" in content
        assert "**Total Notebooks:** 10" in content
