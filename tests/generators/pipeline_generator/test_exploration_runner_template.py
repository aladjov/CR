import ast
from types import SimpleNamespace

import pytest

from customer_retention.generators.pipeline_generator.databricks_renderer import DatabricksCodeRenderer


@pytest.fixture
def renderer():
    return DatabricksCodeRenderer(catalog="ml_catalog", schema="retention")


@pytest.fixture
def sample_datasets():
    return {
        "profiles": SimpleNamespace(
            path="/data/profiles.csv",
            has_target=True,
            role="target",
        ),
        "transactions": SimpleNamespace(
            path="/data/transactions.parquet",
            has_target=False,
            role="feature",
        ),
        "tickets": SimpleNamespace(
            path="/data/tickets.csv",
            has_target=False,
            role="feature",
        ),
    }


class TestExplorationRunnerTemplate:
    def test_renders_valid_python(self, renderer, sample_datasets):
        code = renderer.render_exploration_runner(
            "test_project", sample_datasets, "./notebooks", "./findings",
        )
        ast.parse(code)

    def test_contains_all_datasets(self, renderer, sample_datasets):
        code = renderer.render_exploration_runner(
            "test_project", sample_datasets, "./notebooks", "./findings",
        )
        assert '"profiles"' in code
        assert '"transactions"' in code
        assert '"tickets"' in code

    def test_contains_dataset_paths(self, renderer, sample_datasets):
        code = renderer.render_exploration_runner(
            "test_project", sample_datasets, "./notebooks", "./findings",
        )
        assert "/data/profiles.csv" in code
        assert "/data/transactions.parquet" in code

    def test_uses_dbutils_notebook_run(self, renderer, sample_datasets):
        code = renderer.render_exploration_runner(
            "test_project", sample_datasets, "./notebooks", "./findings",
        )
        assert "dbutils.notebook.run" in code

    def test_has_per_dataset_loop(self, renderer, sample_datasets):
        code = renderer.render_exploration_runner(
            "test_project", sample_datasets, "./notebooks", "./findings",
        )
        assert "for ds_name in dataset_order" in code
        assert "CR_DATASET_ID" in code

    def test_passes_data_path_to_nb01(self, renderer, sample_datasets):
        code = renderer.render_exploration_runner(
            "test_project", sample_datasets, "./notebooks", "./findings",
        )
        assert '"data_path"' in code
        assert '"dataset_name"' in code

    def test_has_global_section(self, renderer, sample_datasets):
        code = renderer.render_exploration_runner(
            "test_project", sample_datasets, "./notebooks", "./findings",
        )
        assert "Global analysis" in code
        assert "03_dataset_merge" in code

    def test_uses_skip_logic_import(self, renderer, sample_datasets):
        code = renderer.render_exploration_runner(
            "test_project", sample_datasets, "./notebooks", "./findings",
        )
        assert "detect_skip_set_for_dataset" in code
        assert "detect_global_skip_set" in code

    def test_target_dataset_identified(self, renderer, sample_datasets):
        code = renderer.render_exploration_runner(
            "test_project", sample_datasets, "./notebooks", "./findings",
        )
        assert 'TARGET_DATASET = "profiles"' in code

    def test_custom_paths(self, renderer, sample_datasets):
        code = renderer.render_exploration_runner(
            "test_project", sample_datasets, "/Workspace/notebooks", "/dbfs/findings",
        )
        assert "/Workspace/notebooks" in code
        assert "/dbfs/findings" in code

    def test_project_name_in_header(self, renderer, sample_datasets):
        code = renderer.render_exploration_runner(
            "my_project", sample_datasets, "./notebooks", "./findings",
        )
        assert "my_project" in code

    def test_summary_section(self, renderer, sample_datasets):
        code = renderer.render_exploration_runner(
            "test_project", sample_datasets, "./notebooks", "./findings",
        )
        assert "Results:" in code
        assert "Total time:" in code

    def test_critical_notebook_stop(self, renderer, sample_datasets):
        code = renderer.render_exploration_runner(
            "test_project", sample_datasets, "./notebooks", "./findings",
        )
        assert "03_dataset_merge" in code
        assert "Critical notebook failed" in code

    def test_single_dataset(self, renderer):
        datasets = {
            "profiles": SimpleNamespace(path="/data/profiles.csv", has_target=True, role="target"),
        }
        code = renderer.render_exploration_runner(
            "single_ds", datasets, "./notebooks", "./findings",
        )
        ast.parse(code)
        assert '"profiles"' in code
        assert 'TARGET_DATASET = "profiles"' in code

    def test_no_explicit_target_uses_first(self, renderer):
        datasets = {
            "alpha": SimpleNamespace(path="/data/alpha.csv", has_target=False, role="feature"),
            "beta": SimpleNamespace(path="/data/beta.csv", has_target=False, role="feature"),
        }
        code = renderer.render_exploration_runner(
            "no_target", datasets, "./notebooks", "./findings",
        )
        ast.parse(code)
        assert 'TARGET_DATASET = "alpha"' in code

    def test_setup_notebook_runs_first(self, renderer, sample_datasets):
        code = renderer.render_exploration_runner(
            "test_project", sample_datasets, "./notebooks", "./findings",
        )
        setup_pos = code.index('run_notebook("00_start_here")')
        loop_pos = code.index("for ds_name in dataset_order")
        assert setup_pos < loop_pos
