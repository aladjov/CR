from abc import ABC

import pytest


class TestNotebookStage:
    def test_all_eleven_stages_defined(self):
        from customer_retention.generators.notebook_generator.base import NotebookStage
        assert len(NotebookStage) == 11

    def test_stage_values(self):
        from customer_retention.generators.notebook_generator.base import NotebookStage
        assert NotebookStage.INGESTION.value == "01_ingestion"
        assert NotebookStage.PROFILING.value == "02_profiling"
        assert NotebookStage.CLEANING.value == "03_cleaning"
        assert NotebookStage.TRANSFORMATION.value == "04_transformation"
        assert NotebookStage.FEATURE_ENGINEERING.value == "05_feature_engineering"
        assert NotebookStage.FEATURE_SELECTION.value == "06_feature_selection"
        assert NotebookStage.MODEL_TRAINING.value == "07_model_training"
        assert NotebookStage.DEPLOYMENT.value == "08_deployment"
        assert NotebookStage.MONITORING.value == "09_monitoring"
        assert NotebookStage.BATCH_INFERENCE.value == "10_batch_inference"

    def test_stage_ordering(self):
        from customer_retention.generators.notebook_generator.base import NotebookStage
        stages = list(NotebookStage)
        assert stages[0] == NotebookStage.INGESTION
        assert stages[-1] == NotebookStage.FEATURE_STORE


class TestNotebookGenerator:
    def test_is_abstract(self):
        from customer_retention.generators.notebook_generator.base import NotebookGenerator
        assert issubclass(NotebookGenerator, ABC)

    def test_cannot_instantiate_directly(self):
        from customer_retention.generators.notebook_generator.base import NotebookGenerator
        from customer_retention.generators.notebook_generator.config import NotebookConfig
        with pytest.raises(TypeError):
            NotebookGenerator(NotebookConfig(), None)

    def test_generate_stage_is_abstract(self):
        import nbformat

        from customer_retention.generators.notebook_generator.base import NotebookGenerator
        from customer_retention.generators.notebook_generator.config import NotebookConfig

        class ConcreteGenerator(NotebookGenerator):
            def generate_stage(self, stage):
                return nbformat.v4.new_notebook()

        generator = ConcreteGenerator(NotebookConfig(), None)
        assert generator is not None

    def test_generate_all_returns_dict(self):
        import nbformat

        from customer_retention.generators.notebook_generator.base import NotebookGenerator, NotebookStage
        from customer_retention.generators.notebook_generator.config import NotebookConfig

        class ConcreteGenerator(NotebookGenerator):
            def generate_stage(self, stage):
                return nbformat.v4.new_notebook()

        generator = ConcreteGenerator(NotebookConfig(), None)
        result = generator.generate_all()
        assert isinstance(result, dict)
        assert len(result) == 11
        assert all(stage in result for stage in NotebookStage)

    def test_save_all_creates_files(self, tmp_path):
        import nbformat

        from customer_retention.generators.notebook_generator.base import NotebookGenerator
        from customer_retention.generators.notebook_generator.config import NotebookConfig

        class ConcreteGenerator(NotebookGenerator):
            def generate_stage(self, stage):
                nb = nbformat.v4.new_notebook()
                nb.cells.append(nbformat.v4.new_code_cell("print('test')"))
                return nb

        generator = ConcreteGenerator(NotebookConfig(), None)
        paths = generator.save_all(str(tmp_path))
        assert len(paths) == 11
        assert all(p.endswith(".ipynb") for p in paths)
        for path in paths:
            assert (tmp_path / path.split("/")[-1]).exists()
