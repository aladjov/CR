import pytest
from pathlib import Path
import nbformat


class TestNotebookRunnerExceptionHandling:
    def test_validate_notebook_with_invalid_file(self, tmp_path):
        from customer_retention.generators.notebook_generator.runner import NotebookRunner
        runner = NotebookRunner()
        result = runner.validate_notebook(str(tmp_path / "nonexistent.ipynb"))
        assert not result.success
        assert "No such file" in result.error or "FileNotFoundError" in result.error

    def test_validate_notebook_with_corrupt_json(self, tmp_path):
        from customer_retention.generators.notebook_generator.runner import NotebookRunner
        runner = NotebookRunner()
        corrupt_file = tmp_path / "corrupt.ipynb"
        corrupt_file.write_text("{invalid json content")
        result = runner.validate_notebook(str(corrupt_file))
        assert not result.success
        assert result.error is not None


class TestValidateGeneratedNotebooksFunction:
    def test_validates_with_default_platforms(self, tmp_path):
        from customer_retention.generators.notebook_generator.runner import validate_generated_notebooks
        from customer_retention.generators.notebook_generator import generate_orchestration_notebooks, Platform
        generate_orchestration_notebooks(output_dir=str(tmp_path), platforms=[Platform.LOCAL])
        reports = validate_generated_notebooks(str(tmp_path))
        assert "local" in reports
        assert reports["local"].all_passed

    def test_validates_with_custom_platforms(self, tmp_path):
        from customer_retention.generators.notebook_generator.runner import validate_generated_notebooks
        from customer_retention.generators.notebook_generator import generate_orchestration_notebooks, Platform
        generate_orchestration_notebooks(output_dir=str(tmp_path), platforms=[Platform.DATABRICKS])
        reports = validate_generated_notebooks(str(tmp_path), platforms=["databricks"])
        assert "databricks" in reports

    def test_skips_nonexistent_platform_dirs(self, tmp_path):
        from customer_retention.generators.notebook_generator.runner import validate_generated_notebooks
        reports = validate_generated_notebooks(str(tmp_path), platforms=["local", "databricks"])
        assert "local" not in reports
        assert "databricks" not in reports


class TestScriptRunnerSyntaxFailure:
    def test_validate_syntax_returns_false_for_invalid_code(self):
        from customer_retention.generators.notebook_generator.runner import ScriptRunner
        runner = ScriptRunner()
        invalid_code = "if x\n  print('missing colon')"
        assert not runner.validate_syntax(invalid_code)

    def test_validate_script_with_syntax_error(self, tmp_path):
        from customer_retention.generators.notebook_generator.runner import ScriptRunner
        runner = ScriptRunner()
        script_file = tmp_path / "invalid.py"
        script_file.write_text("if x\n  print('syntax error')")
        result = runner.validate_script(str(script_file))
        assert not result.success
        assert "Syntax validation failed" in result.error

    def test_validate_script_with_exception(self, tmp_path):
        from customer_retention.generators.notebook_generator.runner import ScriptRunner
        runner = ScriptRunner()
        result = runner.validate_script(str(tmp_path / "nonexistent.py"))
        assert not result.success
        assert result.error is not None

    def test_validate_sequence_stop_on_failure(self, tmp_path):
        from customer_retention.generators.notebook_generator.runner import ScriptRunner
        (tmp_path / "01_first.py").write_text("x = 1")
        (tmp_path / "02_second.py").write_text("if x\n  invalid")
        (tmp_path / "03_third.py").write_text("y = 2")
        runner = ScriptRunner(stop_on_failure=True)
        report = runner.validate_sequence(str(tmp_path), platform="test")
        assert report.failed_count == 1
        assert report.total_notebooks == 2


class TestValidationReportLongError:
    def test_to_markdown_truncates_long_error(self):
        from customer_retention.generators.notebook_generator.runner import ValidationReport, NotebookValidationResult
        long_error = "A" * 100
        results = [NotebookValidationResult("test", False, 1.0, error=long_error)]
        report = ValidationReport(results=results, platform="local")
        md = report.to_markdown()
        assert "..." in md
        assert len(long_error) > 50
