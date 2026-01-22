import pytest
from pathlib import Path
from datetime import datetime


class TestNotebookValidationResult:
    def test_success_result(self):
        from customer_retention.generators.notebook_generator.runner import NotebookValidationResult
        result = NotebookValidationResult(notebook_name="01_ingestion", success=True, duration_seconds=1.5)
        assert result.success
        assert result.duration_seconds == 1.5
        assert result.error is None

    def test_failure_result_with_error(self):
        from customer_retention.generators.notebook_generator.runner import NotebookValidationResult
        result = NotebookValidationResult(notebook_name="02_profiling", success=False, duration_seconds=0.5, error="NameError: x not defined")
        assert not result.success
        assert "NameError" in result.error


class TestValidationReport:
    def test_all_passed(self):
        from customer_retention.generators.notebook_generator.runner import ValidationReport, NotebookValidationResult
        results = [
            NotebookValidationResult("01_ingestion", True, 1.0),
            NotebookValidationResult("02_profiling", True, 2.0),
        ]
        report = ValidationReport(results=results, platform="local")
        assert report.all_passed
        assert report.total_notebooks == 2
        assert report.passed_count == 2
        assert report.failed_count == 0

    def test_some_failed(self):
        from customer_retention.generators.notebook_generator.runner import ValidationReport, NotebookValidationResult
        results = [
            NotebookValidationResult("01_ingestion", True, 1.0),
            NotebookValidationResult("02_profiling", False, 0.5, error="SyntaxError"),
        ]
        report = ValidationReport(results=results, platform="local")
        assert not report.all_passed
        assert report.failed_count == 1

    def test_total_duration(self):
        from customer_retention.generators.notebook_generator.runner import ValidationReport, NotebookValidationResult
        results = [
            NotebookValidationResult("01", True, 1.5),
            NotebookValidationResult("02", True, 2.5),
        ]
        report = ValidationReport(results=results, platform="local")
        assert report.total_duration_seconds == 4.0

    def test_to_markdown(self):
        from customer_retention.generators.notebook_generator.runner import ValidationReport, NotebookValidationResult
        results = [NotebookValidationResult("01_ingestion", True, 1.0)]
        report = ValidationReport(results=results, platform="local")
        md = report.to_markdown()
        assert "01_ingestion" in md
        assert "local" in md.lower()


class TestNotebookRunner:
    def test_validate_syntax_success(self):
        from customer_retention.generators.notebook_generator.runner import NotebookRunner
        runner = NotebookRunner()
        code = "x = 1\ny = x + 2"
        assert runner.validate_syntax(code)

    def test_validate_syntax_failure(self):
        from customer_retention.generators.notebook_generator.runner import NotebookRunner
        runner = NotebookRunner()
        code = "x = 1\nif x"
        assert not runner.validate_syntax(code)

    def test_extract_code_from_notebook(self, tmp_path):
        from customer_retention.generators.notebook_generator.runner import NotebookRunner
        from customer_retention.generators.notebook_generator import LocalNotebookGenerator, NotebookConfig, NotebookStage
        import nbformat

        generator = LocalNotebookGenerator(NotebookConfig(), None)
        nb = generator.generate_stage(NotebookStage.INGESTION)
        nb_path = tmp_path / "test.ipynb"
        with open(nb_path, "w") as f:
            nbformat.write(nb, f)

        runner = NotebookRunner()
        code = runner.extract_code(str(nb_path))
        assert len(code) > 0
        assert "customer_retention" in code

    def test_validate_notebook_syntax(self, tmp_path):
        from customer_retention.generators.notebook_generator.runner import NotebookRunner
        from customer_retention.generators.notebook_generator import LocalNotebookGenerator, NotebookConfig, NotebookStage
        import nbformat

        generator = LocalNotebookGenerator(NotebookConfig(), None)
        nb = generator.generate_stage(NotebookStage.INGESTION)
        nb_path = tmp_path / "test.ipynb"
        with open(nb_path, "w") as f:
            nbformat.write(nb, f)

        runner = NotebookRunner()
        result = runner.validate_notebook(str(nb_path))
        assert result.success

    def test_validate_sequence_all_pass(self, tmp_path):
        from customer_retention.generators.notebook_generator.runner import NotebookRunner
        from customer_retention.generators.notebook_generator import LocalNotebookGenerator, NotebookConfig
        import nbformat

        generator = LocalNotebookGenerator(NotebookConfig(), None)
        notebooks = generator.generate_all()
        for stage, nb in notebooks.items():
            nb_path = tmp_path / f"{stage.value}.ipynb"
            with open(nb_path, "w") as f:
                nbformat.write(nb, f)

        runner = NotebookRunner()
        report = runner.validate_sequence(str(tmp_path), platform="local")
        assert report.total_notebooks == 10  # Holdout stage disabled by default
        assert report.all_passed


class TestNotebookRunnerExecution:
    def test_dry_run_mode(self, tmp_path):
        from customer_retention.generators.notebook_generator.runner import NotebookRunner
        from customer_retention.generators.notebook_generator import LocalNotebookGenerator, NotebookConfig
        import nbformat

        generator = LocalNotebookGenerator(NotebookConfig(), None)
        notebooks = generator.generate_all()
        for stage, nb in notebooks.items():
            nb_path = tmp_path / f"{stage.value}.ipynb"
            with open(nb_path, "w") as f:
                nbformat.write(nb, f)

        runner = NotebookRunner(dry_run=True)
        report = runner.validate_sequence(str(tmp_path), platform="local")
        assert report.all_passed

    def test_stops_on_first_failure_when_configured(self, tmp_path):
        from customer_retention.generators.notebook_generator.runner import NotebookRunner
        import nbformat

        nb1 = nbformat.v4.new_notebook()
        nb1.cells = [nbformat.v4.new_code_cell("x = 1")]
        with open(tmp_path / "01_first.ipynb", "w") as f:
            nbformat.write(nb1, f)

        nb2 = nbformat.v4.new_notebook()
        nb2.cells = [nbformat.v4.new_code_cell("if x")]
        with open(tmp_path / "02_second.ipynb", "w") as f:
            nbformat.write(nb2, f)

        nb3 = nbformat.v4.new_notebook()
        nb3.cells = [nbformat.v4.new_code_cell("y = 2")]
        with open(tmp_path / "03_third.ipynb", "w") as f:
            nbformat.write(nb3, f)

        runner = NotebookRunner(stop_on_failure=True)
        report = runner.validate_sequence(str(tmp_path), platform="test")
        assert report.failed_count == 1
        assert report.total_notebooks == 2


class TestIntegrationWithGeneration:
    def test_generate_and_validate_local(self, tmp_path):
        from customer_retention.generators.notebook_generator import generate_orchestration_notebooks, Platform
        from customer_retention.generators.notebook_generator.runner import NotebookRunner

        results = generate_orchestration_notebooks(output_dir=str(tmp_path), platforms=[Platform.LOCAL])
        runner = NotebookRunner(dry_run=True)
        report = runner.validate_sequence(str(tmp_path / "local"), platform="local")
        assert report.all_passed
        assert report.total_notebooks == 10  # Holdout stage disabled by default

    def test_generate_and_validate_databricks(self, tmp_path):
        from customer_retention.generators.notebook_generator import generate_orchestration_notebooks, Platform
        from customer_retention.generators.notebook_generator.runner import NotebookRunner

        results = generate_orchestration_notebooks(output_dir=str(tmp_path), platforms=[Platform.DATABRICKS])
        runner = NotebookRunner(dry_run=True)
        report = runner.validate_sequence(str(tmp_path / "databricks"), platform="databricks")
        assert report.all_passed
        assert report.total_notebooks == 10  # Holdout stage disabled by default


class TestGenerationResult:
    def test_generation_result_all_valid(self, tmp_path):
        from customer_retention.generators.notebook_generator import generate_and_validate_notebooks, Platform
        results = generate_and_validate_notebooks(output_dir=str(tmp_path), platforms=[Platform.LOCAL])
        assert Platform.LOCAL in results
        assert results[Platform.LOCAL].all_valid
        assert len(results[Platform.LOCAL].notebook_paths) == 10  # Holdout stage disabled by default

    def test_creates_validation_report_file(self, tmp_path):
        from customer_retention.generators.notebook_generator import generate_and_validate_notebooks, Platform
        generate_and_validate_notebooks(output_dir=str(tmp_path), platforms=[Platform.LOCAL])
        report_path = tmp_path / "local" / "VALIDATION_REPORT.md"
        assert report_path.exists()
        content = report_path.read_text()
        assert "Notebook Validation Report" in content
        assert "PASSED" in content

    def test_both_platforms_generate_reports(self, tmp_path):
        from customer_retention.generators.notebook_generator import generate_and_validate_notebooks, Platform
        results = generate_and_validate_notebooks(output_dir=str(tmp_path))
        assert Platform.LOCAL in results
        assert Platform.DATABRICKS in results
        assert (tmp_path / "local" / "VALIDATION_REPORT.md").exists()
        assert (tmp_path / "databricks" / "VALIDATION_REPORT.md").exists()

    def test_report_contains_all_notebooks(self, tmp_path):
        from customer_retention.generators.notebook_generator import generate_and_validate_notebooks, Platform
        results = generate_and_validate_notebooks(output_dir=str(tmp_path), platforms=[Platform.LOCAL])
        report = results[Platform.LOCAL].validation_report
        assert report.total_notebooks == 10  # Holdout stage disabled by default
        notebook_names = [r.notebook_name for r in report.results]
        assert "01_ingestion" in notebook_names
        assert "10_batch_inference" in notebook_names
