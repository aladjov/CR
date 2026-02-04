"""Tests for the s11_feature_store notebook stage generator."""

import pytest

from customer_retention.generators.notebook_generator.base import NotebookStage
from customer_retention.generators.notebook_generator.config import (
    FeatureStoreConfig,
    NotebookConfig,
    Platform,
)
from customer_retention.generators.notebook_generator.stages.s11_feature_store import (
    FeatureStoreStage,
)


class TestFeatureStoreStage:
    @pytest.fixture
    def stage(self):
        config = NotebookConfig()
        return FeatureStoreStage(config, findings=None)

    @pytest.fixture
    def stage_with_custom_config(self):
        config = NotebookConfig(
            feature_store=FeatureStoreConfig(catalog="my_catalog", schema="my_schema")
        )
        return FeatureStoreStage(config, findings=None)

    # --- properties ---
    def test_stage_property(self, stage):
        assert stage.stage == NotebookStage.FEATURE_STORE

    def test_title_property(self, stage):
        assert "11" in stage.title
        assert "Feature Store" in stage.title

    def test_description_property(self, stage):
        desc = stage.description
        assert "feature store" in desc.lower()
        assert len(desc) > 0

    # --- generate_local_cells ---
    def test_generate_local_cells_returns_list(self, stage):
        cells = stage.generate_local_cells()
        assert isinstance(cells, list)
        assert len(cells) > 0

    def test_generate_local_cells_contains_code_cells(self, stage):
        cells = stage.generate_local_cells()
        code_cells = [c for c in cells if c.cell_type == "code"]
        assert len(code_cells) > 0

    def test_generate_local_cells_contains_markdown_cells(self, stage):
        cells = stage.generate_local_cells()
        md_cells = [c for c in cells if c.cell_type == "markdown"]
        assert len(md_cells) > 0

    def test_generate_local_cells_code_compiles(self, stage):
        cells = stage.generate_local_cells()
        code = "\n".join(c.source for c in cells if c.cell_type == "code")
        compile(code, "<feature_store_local>", "exec")

    def test_generate_local_cells_has_feature_store_imports(self, stage):
        cells = stage.generate_local_cells()
        code = "\n".join(c.source for c in cells if c.cell_type == "code")
        assert "get_feature_store_manager" in code
        assert "FeatureRegistry" in code

    def test_generate_local_cells_uses_auto_detection(self, stage):
        cells = stage.generate_local_cells()
        code = "\n".join(c.source for c in cells if c.cell_type == "code")
        assert "is_databricks" in code
        assert 'backend="feast"' not in code

    def test_generate_local_cells_no_direct_feast_import(self, stage):
        cells = stage.generate_local_cells()
        code = "\n".join(c.source for c in cells if c.cell_type == "code")
        assert "from feast import" not in code

    def test_generate_local_cells_has_sections(self, stage):
        cells = stage.generate_local_cells()
        md_text = " ".join(c.source for c in cells if c.cell_type == "markdown")
        assert "Setup" in md_text
        assert "Summary" in md_text

    # --- generate_databricks_cells ---
    def test_generate_databricks_cells_returns_list(self, stage_with_custom_config):
        cells = stage_with_custom_config.generate_databricks_cells()
        assert isinstance(cells, list)
        assert len(cells) > 0

    def test_generate_databricks_cells_uses_catalog_schema(self, stage_with_custom_config):
        cells = stage_with_custom_config.generate_databricks_cells()
        code = "\n".join(c.source for c in cells if c.cell_type == "code")
        assert "my_catalog" in code
        assert "my_schema" in code

    def test_generate_databricks_cells_code_compiles(self, stage_with_custom_config):
        cells = stage_with_custom_config.generate_databricks_cells()
        code = "\n".join(c.source for c in cells if c.cell_type == "code")
        compile(code, "<feature_store_databricks>", "exec")

    def test_generate_databricks_cells_has_spark(self, stage_with_custom_config):
        cells = stage_with_custom_config.generate_databricks_cells()
        code = "\n".join(c.source for c in cells if c.cell_type == "code")
        assert "spark" in code

    def test_generate_databricks_cells_has_sections(self, stage_with_custom_config):
        cells = stage_with_custom_config.generate_databricks_cells()
        md_text = " ".join(c.source for c in cells if c.cell_type == "markdown")
        assert "Setup" in md_text
        assert "Summary" in md_text

    # --- generate (platform dispatch) ---
    def test_generate_local(self, stage):
        cells = stage.generate(Platform.LOCAL)
        assert isinstance(cells, list)
        assert len(cells) > 0

    def test_generate_databricks(self, stage_with_custom_config):
        cells = stage_with_custom_config.generate(Platform.DATABRICKS)
        assert isinstance(cells, list)
        assert len(cells) > 0

    # --- header_cells ---
    def test_header_cells_include_title(self, stage):
        header = stage.header_cells()
        assert len(header) >= 1
        assert "Feature Store" in header[0].source
