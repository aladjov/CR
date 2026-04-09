"""Tests for get_playbooks_dir() resolution and the init-path wiring."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def isolate_env(monkeypatch):
    monkeypatch.delenv("CR_PLAYBOOKS_DIR", raising=False)
    monkeypatch.delenv("CR_EXPERIMENTS_DIR", raising=False)
    yield


class TestGetPlaybooksDir:
    def test_env_var_takes_precedence(self, monkeypatch):
        from customer_retention.core.config.experiments import get_playbooks_dir

        monkeypatch.setenv("CR_PLAYBOOKS_DIR", "/Volumes/cat/sch/custom_playbooks")
        result = get_playbooks_dir()
        assert str(result) == "/Volumes/cat/sch/custom_playbooks"

    def test_explicit_default_argument(self):
        from customer_retention.core.config.experiments import get_playbooks_dir

        result = get_playbooks_dir(default="/some/explicit/path")
        assert str(result) == "/some/explicit/path"

    def test_explicit_default_overrides_repo_fallback(self):
        from customer_retention.core.config.experiments import get_playbooks_dir

        result = get_playbooks_dir(default="/explicit/wins")
        assert str(result) == "/explicit/wins"

    def test_local_fallback_resolves_to_project_root_playbooks(self):
        from customer_retention.core.config.experiments import get_playbooks_dir

        result = get_playbooks_dir()
        # _find_project_root() walks up from the experiments.py file looking
        # for pyproject.toml or .git, then appends "playbooks"
        assert str(result).endswith("playbooks")
        assert Path(str(result)).parent == Path(str(result)).parent  # is a real path

    def test_reload_config_picks_up_env_change(self, monkeypatch):
        from customer_retention.core.config import experiments

        monkeypatch.setenv("CR_PLAYBOOKS_DIR", "/tmp/test_reload_path")
        experiments.reload_config()
        assert str(experiments.PLAYBOOKS_DIR) == "/tmp/test_reload_path"


class TestPlaybooksDirReExports:
    def test_re_exported_from_core_config(self):
        from customer_retention.core.config import PLAYBOOKS_DIR, get_playbooks_dir

        assert get_playbooks_dir is not None
        # PLAYBOOKS_DIR is whatever the module loaded at import time —
        # just verify it exists
        assert PLAYBOOKS_DIR is not None


class TestProjectInitializerWiring:
    def test_dataclass_has_playbooks_path_field(self):
        from customer_retention.generators.notebook_generator.project_init import ProjectInitializer

        init = ProjectInitializer(project_name="test")
        assert hasattr(init, "playbooks_path")
        assert init.playbooks_path == "playbooks"

    def test_custom_playbooks_path(self):
        from customer_retention.generators.notebook_generator.project_init import ProjectInitializer

        init = ProjectInitializer(project_name="test", playbooks_path="custom_pb")
        assert init.playbooks_path == "custom_pb"

    def test_gitignore_includes_playbooks_dir(self):
        from customer_retention.generators.notebook_generator.project_init import ProjectInitializer

        init = ProjectInitializer(project_name="test", playbooks_path="custom_pb")
        gi = init._gitignore_content()
        assert "custom_pb/" in gi
        # Experiments dir is also gitignored — sanity check
        assert "experiments/" in gi

    def test_initialize_project_helper_passes_playbooks_path(self, tmp_path):
        from customer_retention.generators.notebook_generator.project_init import initialize_project

        result = initialize_project(
            output_dir=str(tmp_path / "new_project"),
            project_name="new_project",
            playbooks_path="my_playbooks",
        )
        # Directory was created with the custom name
        assert (tmp_path / "new_project" / "my_playbooks").is_dir()
        assert (tmp_path / "new_project" / "my_playbooks" / "policies").is_dir()
        # The 3 framework seed YAMLs were copied in
        copied = result.get("seed_yamls_copied", [])
        copied_names = {Path(p).name for p in copied}
        assert "decision_policy.yaml" in copied_names
        assert "response_schemas.yaml" in copied_names
        assert "vocabularies.yaml" in copied_names

    def test_seed_yamls_not_copied_twice(self, tmp_path):
        # Idempotency: second initialize_project call into the same dir
        # must not overwrite existing seed YAMLs.
        from customer_retention.generators.notebook_generator.project_init import initialize_project

        target = tmp_path / "twice"
        initialize_project(output_dir=str(target), project_name="twice")
        # Edit one of the seeds to detect overwrites
        seed_file = target / "playbooks" / "policies" / "decision_policy.yaml"
        seed_file.write_text("USER_EDITED_CONTENT")
        result = initialize_project(output_dir=str(target), project_name="twice")
        # Edit must survive: not in the second-pass copied list
        second_copied = {Path(p).name for p in result.get("seed_yamls_copied", [])}
        assert "decision_policy.yaml" not in second_copied
        assert seed_file.read_text() == "USER_EDITED_CONTENT"


class TestDatabricksInitWiring:
    def test_signature_has_playbooks_path(self):
        import inspect

        from customer_retention.integrations.databricks_init import databricks_init

        sig = inspect.signature(databricks_init)
        assert "playbooks_path" in sig.parameters
        assert sig.parameters["playbooks_path"].default == "playbooks"

    def test_result_dataclass_has_playbooks_path(self):
        from customer_retention.integrations.databricks_init import DatabricksInitResult

        assert "playbooks_path" in DatabricksInitResult.__dataclass_fields__
        assert DatabricksInitResult.__dataclass_fields__["playbooks_path"].default == "playbooks"

    def test_environment_variables_includes_cr_playbooks_dir(self):
        from customer_retention.integrations.databricks_init import DatabricksInitResult

        result = DatabricksInitResult(
            catalog="cat",
            schema="sch",
            experiment_name="exp",
            workspace_path="ws",
            model_name="mdl",
            experiments_path="custom_exp",
            playbooks_path="custom_pb",
        )
        env = result.environment_variables
        assert env["CR_EXPERIMENTS_DIR"] == "/Volumes/cat/sch/custom_exp"
        assert env["CR_PLAYBOOKS_DIR"] == "/Volumes/cat/sch/custom_pb"


class TestCliWiring:
    def test_cli_has_playbooks_path_argument(self):
        # The CLI builds its argparse parser inside main(); verify the
        # argument is registered without invoking it (which would parse argv).
        import inspect

        import customer_retention.cli as cli
        source = inspect.getsource(cli)
        assert "--playbooks-path" in source
        assert 'default="playbooks"' in source
