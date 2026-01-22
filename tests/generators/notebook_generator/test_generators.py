import pytest
import nbformat


class TestLocalNotebookGenerator:
    def test_generates_all_stages(self):
        from customer_retention.generators.notebook_generator import LocalNotebookGenerator, NotebookConfig, NotebookStage
        generator = LocalNotebookGenerator(NotebookConfig(), None)
        notebooks = generator.generate_all()
        assert len(notebooks) == 10  # Holdout stage disabled by default
        for stage in generator.available_stages:
            assert stage in notebooks

    def test_local_notebook_has_framework_imports(self):
        from customer_retention.generators.notebook_generator import LocalNotebookGenerator, NotebookConfig, NotebookStage
        generator = LocalNotebookGenerator(NotebookConfig(), None)
        nb = generator.generate_stage(NotebookStage.INGESTION)
        code = "".join(cell.source for cell in nb.cells if cell.cell_type == "code")
        assert "customer_retention" in code

    def test_local_notebook_valid_format(self):
        from customer_retention.generators.notebook_generator import LocalNotebookGenerator, NotebookConfig, NotebookStage
        generator = LocalNotebookGenerator(NotebookConfig(), None)
        nb = generator.generate_stage(NotebookStage.CLEANING)
        assert nb.nbformat == 4
        assert len(nb.cells) > 0

    def test_save_creates_files(self, tmp_path):
        from customer_retention.generators.notebook_generator import LocalNotebookGenerator, NotebookConfig
        generator = LocalNotebookGenerator(NotebookConfig(), None)
        paths = generator.save_all(str(tmp_path))
        assert len(paths) == 10  # Holdout stage disabled by default
        for path in paths:
            assert path.endswith(".ipynb")


class TestDatabricksNotebookGenerator:
    def test_generates_all_stages(self):
        from customer_retention.generators.notebook_generator import DatabricksNotebookGenerator, NotebookConfig, NotebookStage
        generator = DatabricksNotebookGenerator(NotebookConfig(), None)
        notebooks = generator.generate_all()
        assert len(notebooks) == 10  # Holdout stage disabled by default

    def test_databricks_notebook_no_framework_imports(self):
        from customer_retention.generators.notebook_generator import DatabricksNotebookGenerator, NotebookConfig, NotebookStage
        generator = DatabricksNotebookGenerator(NotebookConfig(), None)
        nb = generator.generate_stage(NotebookStage.INGESTION)
        code = "".join(cell.source for cell in nb.cells if cell.cell_type == "code")
        assert "from customer_retention" not in code

    def test_databricks_uses_spark(self):
        from customer_retention.generators.notebook_generator import DatabricksNotebookGenerator, NotebookConfig, NotebookStage
        generator = DatabricksNotebookGenerator(NotebookConfig(), None)
        nb = generator.generate_stage(NotebookStage.INGESTION)
        code = "".join(cell.source for cell in nb.cells if cell.cell_type == "code")
        assert "spark" in code

    def test_databricks_uses_catalog_schema(self):
        from customer_retention.generators.notebook_generator import (
            DatabricksNotebookGenerator, NotebookConfig, NotebookStage, FeatureStoreConfig
        )
        config = NotebookConfig(feature_store=FeatureStoreConfig(catalog="test_cat", schema="test_schema"))
        generator = DatabricksNotebookGenerator(config, None)
        nb = generator.generate_stage(NotebookStage.INGESTION)
        code = "".join(cell.source for cell in nb.cells if cell.cell_type == "code")
        assert "test_cat" in code
        assert "test_schema" in code


class TestGenerateOrchestrationNotebooks:
    def test_generates_for_both_platforms(self, tmp_path):
        from customer_retention.generators.notebook_generator import generate_orchestration_notebooks, Platform
        results = generate_orchestration_notebooks(output_dir=str(tmp_path))
        assert Platform.LOCAL in results
        assert Platform.DATABRICKS in results
        assert len(results[Platform.LOCAL]) == 10  # Holdout stage disabled by default
        assert len(results[Platform.DATABRICKS]) == 10  # Holdout stage disabled by default

    def test_generates_local_only(self, tmp_path):
        from customer_retention.generators.notebook_generator import generate_orchestration_notebooks, Platform
        results = generate_orchestration_notebooks(output_dir=str(tmp_path), platforms=[Platform.LOCAL])
        assert Platform.LOCAL in results
        assert Platform.DATABRICKS not in results

    def test_creates_platform_subdirectories(self, tmp_path):
        from customer_retention.generators.notebook_generator import generate_orchestration_notebooks, Platform
        generate_orchestration_notebooks(output_dir=str(tmp_path))
        assert (tmp_path / "local").exists()
        assert (tmp_path / "databricks").exists()


class TestNotebookCodeCompiles:
    @pytest.mark.parametrize("stage", [
        "INGESTION", "PROFILING", "CLEANING", "TRANSFORMATION",
        "FEATURE_ENGINEERING", "FEATURE_SELECTION", "MODEL_TRAINING",
        "DEPLOYMENT", "MONITORING", "BATCH_INFERENCE"
    ])
    def test_local_code_compiles(self, stage):
        from customer_retention.generators.notebook_generator import LocalNotebookGenerator, NotebookConfig, NotebookStage
        generator = LocalNotebookGenerator(NotebookConfig(), None)
        nb = generator.generate_stage(NotebookStage[stage])
        code = "\n".join(cell.source for cell in nb.cells if cell.cell_type == "code")
        compile(code, f"<{stage}>", "exec")

    @pytest.mark.parametrize("stage", [
        "INGESTION", "PROFILING", "CLEANING", "TRANSFORMATION",
        "FEATURE_ENGINEERING", "FEATURE_SELECTION", "MODEL_TRAINING",
        "DEPLOYMENT", "MONITORING", "BATCH_INFERENCE"
    ])
    def test_databricks_code_compiles(self, stage):
        from customer_retention.generators.notebook_generator import DatabricksNotebookGenerator, NotebookConfig, NotebookStage
        generator = DatabricksNotebookGenerator(NotebookConfig(), None)
        nb = generator.generate_stage(NotebookStage[stage])
        code = "\n".join(cell.source for cell in nb.cells if cell.cell_type == "code")
        compile(code, f"<{stage}>", "exec")
