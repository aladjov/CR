
import pytest


class TestOutputFormat:
    def test_output_format_enum(self):
        from customer_retention.generators.notebook_generator.config import OutputFormat
        assert OutputFormat.NOTEBOOK.value == "notebook"
        assert OutputFormat.SCRIPT.value == "script"

    def test_config_default_format_is_notebook(self):
        from customer_retention.generators.notebook_generator.config import NotebookConfig, OutputFormat
        config = NotebookConfig()
        assert config.output_format == OutputFormat.NOTEBOOK


class TestScriptGenerator:
    def test_generates_python_files(self, tmp_path):
        from customer_retention.generators.notebook_generator.config import NotebookConfig
        from customer_retention.generators.notebook_generator.script_generator import LocalScriptGenerator
        generator = LocalScriptGenerator(NotebookConfig(), None)
        paths = generator.save_all(str(tmp_path))
        assert len(paths) == 10  # Holdout stage disabled by default
        assert all(p.endswith(".py") for p in paths)

    def test_script_has_main_block(self, tmp_path):
        from customer_retention.generators.notebook_generator.base import NotebookStage
        from customer_retention.generators.notebook_generator.config import NotebookConfig
        from customer_retention.generators.notebook_generator.script_generator import LocalScriptGenerator
        generator = LocalScriptGenerator(NotebookConfig(), None)
        code = generator.generate_stage_code(NotebookStage.INGESTION)
        assert 'if __name__ == "__main__"' in code

    def test_script_has_docstring(self, tmp_path):
        from customer_retention.generators.notebook_generator.base import NotebookStage
        from customer_retention.generators.notebook_generator.config import NotebookConfig
        from customer_retention.generators.notebook_generator.script_generator import LocalScriptGenerator
        generator = LocalScriptGenerator(NotebookConfig(), None)
        code = generator.generate_stage_code(NotebookStage.INGESTION)
        assert '"""' in code

    def test_local_script_has_framework_imports(self):
        from customer_retention.generators.notebook_generator.base import NotebookStage
        from customer_retention.generators.notebook_generator.config import NotebookConfig
        from customer_retention.generators.notebook_generator.script_generator import LocalScriptGenerator
        generator = LocalScriptGenerator(NotebookConfig(), None)
        code = generator.generate_stage_code(NotebookStage.INGESTION)
        assert "customer_retention" in code

    def test_databricks_script_no_framework(self):
        from customer_retention.generators.notebook_generator.base import NotebookStage
        from customer_retention.generators.notebook_generator.config import NotebookConfig
        from customer_retention.generators.notebook_generator.script_generator import DatabricksScriptGenerator
        generator = DatabricksScriptGenerator(NotebookConfig(), None)
        code = generator.generate_stage_code(NotebookStage.INGESTION)
        assert "from customer_retention" not in code
        assert "spark" in code


class TestScriptCodeCompiles:
    @pytest.mark.parametrize("stage", [
        "INGESTION", "PROFILING", "CLEANING", "TRANSFORMATION",
        "FEATURE_ENGINEERING", "FEATURE_SELECTION", "MODEL_TRAINING",
        "DEPLOYMENT", "MONITORING", "BATCH_INFERENCE"
    ])
    def test_local_script_compiles(self, stage):
        from customer_retention.generators.notebook_generator.base import NotebookStage
        from customer_retention.generators.notebook_generator.config import NotebookConfig
        from customer_retention.generators.notebook_generator.script_generator import LocalScriptGenerator
        generator = LocalScriptGenerator(NotebookConfig(), None)
        code = generator.generate_stage_code(NotebookStage[stage])
        compile(code, f"<{stage}>", "exec")

    @pytest.mark.parametrize("stage", [
        "INGESTION", "PROFILING", "CLEANING", "TRANSFORMATION",
        "FEATURE_ENGINEERING", "FEATURE_SELECTION", "MODEL_TRAINING",
        "DEPLOYMENT", "MONITORING", "BATCH_INFERENCE"
    ])
    def test_databricks_script_compiles(self, stage):
        from customer_retention.generators.notebook_generator.base import NotebookStage
        from customer_retention.generators.notebook_generator.config import NotebookConfig
        from customer_retention.generators.notebook_generator.script_generator import DatabricksScriptGenerator
        generator = DatabricksScriptGenerator(NotebookConfig(), None)
        code = generator.generate_stage_code(NotebookStage[stage])
        compile(code, f"<{stage}>", "exec")


class TestGenerateScriptsFunction:
    def test_generate_scripts_local(self, tmp_path):
        from customer_retention.generators.notebook_generator import Platform, generate_orchestration_scripts
        results = generate_orchestration_scripts(output_dir=str(tmp_path), platforms=[Platform.LOCAL])
        assert Platform.LOCAL in results
        assert len(results[Platform.LOCAL]) == 10  # Holdout stage disabled by default

    def test_generate_scripts_both_platforms(self, tmp_path):
        from customer_retention.generators.notebook_generator import generate_orchestration_scripts
        results = generate_orchestration_scripts(output_dir=str(tmp_path))
        assert (tmp_path / "local").exists()
        assert (tmp_path / "databricks").exists()

    def test_scripts_directory_structure(self, tmp_path):
        from customer_retention.generators.notebook_generator import Platform, generate_orchestration_scripts
        generate_orchestration_scripts(output_dir=str(tmp_path), platforms=[Platform.LOCAL])
        assert (tmp_path / "local" / "01_ingestion.py").exists()
        assert (tmp_path / "local" / "10_batch_inference.py").exists()


class TestScriptRunnerValidation:
    def test_validate_scripts(self, tmp_path):
        from customer_retention.generators.notebook_generator import Platform, generate_orchestration_scripts
        from customer_retention.generators.notebook_generator.runner import ScriptRunner
        generate_orchestration_scripts(output_dir=str(tmp_path), platforms=[Platform.LOCAL])
        runner = ScriptRunner()
        report = runner.validate_sequence(str(tmp_path / "local"), platform="local")
        assert report.all_passed
        assert report.total_notebooks == 10  # Holdout stage disabled by default
