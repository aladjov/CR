import ast
from types import SimpleNamespace

import pytest
import yaml

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


class TestForEachWorkflowTemplate:
    def test_renders_valid_yaml(self, renderer):
        output = renderer.render_for_each_workflow(
            "test_project", "/Workspace/notebooks",
        )
        data = yaml.safe_load(output)
        assert "resources" in data
        assert "jobs" in data["resources"]

    def test_job_name_from_project(self, renderer):
        output = renderer.render_for_each_workflow(
            "retention_v2", "/Workspace/notebooks",
        )
        data = yaml.safe_load(output)
        job = data["resources"]["jobs"]["retention_v2_exploration"]
        assert job["name"] == "retention_v2 Exploration"

    def test_setup_task_present(self, renderer):
        output = renderer.render_for_each_workflow(
            "test_project", "/Workspace/nb",
        )
        data = yaml.safe_load(output)
        tasks = data["resources"]["jobs"]["test_project_exploration"]["tasks"]
        setup = [t for t in tasks if t["task_key"] == "00_Setup"]
        assert len(setup) == 1
        assert "/Workspace/nb/00_start_here" in setup[0]["notebook_task"]["notebook_path"]

    def test_per_dataset_notebooks_are_for_each_tasks(self, renderer):
        output = renderer.render_for_each_workflow(
            "test_project", "/Workspace/nb",
        )
        data = yaml.safe_load(output)
        tasks = data["resources"]["jobs"]["test_project_exploration"]["tasks"]
        for_each_tasks = [t for t in tasks if "for_each_task" in t]
        task_keys = [t["task_key"] for t in for_each_tasks]
        assert "01_Data_Discovery" in task_keys
        assert "01a_Temporal_Deep_Dive" in task_keys
        assert "01d_Event_Aggregation" in task_keys
        assert "02_Source_Integrity" in task_keys

    def test_for_each_tasks_have_experiments_dir(self, renderer):
        output = renderer.render_for_each_workflow(
            "test_project", "/Workspace/nb",
        )
        data = yaml.safe_load(output)
        tasks = data["resources"]["jobs"]["test_project_exploration"]["tasks"]
        for task in tasks:
            if "for_each_task" not in task:
                continue
            params = task["for_each_task"]["task"]["notebook_task"]["base_parameters"]
            assert "experiments_dir" in params, (
                f"{task['task_key']} missing experiments_dir"
            )
            assert "run_id" in params
            assert "dataset_id" in params

    def test_global_notebooks_have_run_id_and_experiments_dir(self, renderer):
        output = renderer.render_for_each_workflow(
            "test_project", "/Workspace/nb",
        )
        data = yaml.safe_load(output)
        tasks = data["resources"]["jobs"]["test_project_exploration"]["tasks"]
        global_keys = {
            "03_Dataset_Merge", "04_Column_Deep_Dive", "05_Relationship_Analysis",
            "08_Baseline_Experiments", "10_Spec_Generation",
        }
        for task in tasks:
            if task["task_key"] not in global_keys:
                continue
            params = task["notebook_task"]["base_parameters"]
            assert "run_id" in params, f"{task['task_key']} missing run_id"
            assert "experiments_dir" in params, (
                f"{task['task_key']} missing experiments_dir"
            )

    def test_for_each_concurrency_is_one(self, renderer):
        output = renderer.render_for_each_workflow(
            "test_project", "/Workspace/nb",
        )
        data = yaml.safe_load(output)
        tasks = data["resources"]["jobs"]["test_project_exploration"]["tasks"]
        for task in tasks:
            if "for_each_task" not in task:
                continue
            assert task["for_each_task"]["concurrency"] == 1

    def test_notebooks_base_path_used_everywhere(self, renderer):
        output = renderer.render_for_each_workflow(
            "test_project", "/Workspace/custom/path",
        )
        data = yaml.safe_load(output)
        tasks = data["resources"]["jobs"]["test_project_exploration"]["tasks"]
        for task in tasks:
            if "notebook_task" in task:
                assert task["notebook_task"]["notebook_path"].startswith(
                    "/Workspace/custom/path/"
                )
            if "for_each_task" in task:
                inner_path = task["for_each_task"]["task"]["notebook_task"]["notebook_path"]
                assert inner_path.startswith("/Workspace/custom/path/")

    def test_dependency_chain_per_dataset(self, renderer):
        output = renderer.render_for_each_workflow(
            "test_project", "/Workspace/nb",
        )
        data = yaml.safe_load(output)
        tasks = data["resources"]["jobs"]["test_project_exploration"]["tasks"]
        task_map = {t["task_key"]: t for t in tasks}
        assert task_map["01_Data_Discovery"]["depends_on"] == [
            {"task_key": "00_Setup"}
        ]
        assert task_map["01a_Temporal_Deep_Dive"]["depends_on"] == [
            {"task_key": "01_Data_Discovery"}
        ]
        assert task_map["01d_Event_Aggregation"]["depends_on"] == [
            {"task_key": "01c_Temporal_Patterns"}
        ]

    def test_global_depends_on_last_per_dataset(self, renderer):
        output = renderer.render_for_each_workflow(
            "test_project", "/Workspace/nb",
        )
        data = yaml.safe_load(output)
        tasks = data["resources"]["jobs"]["test_project_exploration"]["tasks"]
        task_map = {t["task_key"]: t for t in tasks}
        assert task_map["03_Dataset_Merge"]["depends_on"] == [
            {"task_key": "02_Source_Integrity"}
        ]

    def test_queue_enabled(self, renderer):
        output = renderer.render_for_each_workflow(
            "test_project", "/Workspace/nb",
        )
        data = yaml.safe_load(output)
        job = data["resources"]["jobs"]["test_project_exploration"]
        assert job["queue"]["enabled"] is True

    def test_all_expected_tasks_present(self, renderer):
        output = renderer.render_for_each_workflow(
            "test_project", "/Workspace/nb",
        )
        data = yaml.safe_load(output)
        tasks = data["resources"]["jobs"]["test_project_exploration"]["tasks"]
        keys = {t["task_key"] for t in tasks}
        expected = {
            "00_Setup",
            "01_Data_Discovery", "01a_Temporal_Deep_Dive",
            "01a_a_Temporal_Text_Deep_Dive", "01b_Temporal_Quality",
            "01c_Temporal_Patterns", "01d_Event_Aggregation",
            "02_Source_Integrity",
            "03_Dataset_Merge", "04_Column_Deep_Dive",
            "04a_Text_Columns_Deep_Dive", "05_Relationship_Analysis",
            "06_Feature_Opportunities", "07_Modeling_Readiness",
            "08_Baseline_Experiments", "09_Business_Alignment",
            "10_Spec_Generation", "11_Scoring_Validation",
            "12_View_Documentation",
        }
        assert keys == expected

    def test_databricks_expressions_literal(self, renderer):
        """Databricks {{tasks...}} expressions must survive Jinja2 rendering."""
        output = renderer.render_for_each_workflow(
            "test_project", "/Workspace/nb",
        )
        assert "{{tasks.00_Setup.values.dataset_names}}" in output
        assert "{{tasks.00_Setup.values.run_id}}" in output
        assert "{{tasks.00_Setup.values.experiments_dir}}" in output
        assert "{{input}}" in output
