import getpass
import os
from unittest.mock import patch

import pytest

from customer_retention.analysis.auto_explorer.run_namespace import RunNamespace
from customer_retention.analysis.auto_explorer.session import (
    SessionState,
    get_current_username,
    initialize_run,
    load_notebook_findings,
    mark_notebook,
    resolve_active_dataset,
    resolve_findings_path,
    resolve_target_column,
    sanitize_username,
    set_active_dataset,
)


@pytest.fixture
def namespace(tmp_path):
    ns = RunNamespace.create(root=tmp_path, project_name="test")
    return ns


class TestSessionStateSaveLoad:
    def test_save_and_load_roundtrip(self, tmp_path):
        path = tmp_path / "session.json"
        state = SessionState(active_dataset="customers", active_run_id="run-abc", last_notebook="01_data_discovery")
        state.save(path)
        loaded = SessionState.load(path)
        assert loaded.active_dataset == "customers"
        assert loaded.active_run_id == "run-abc"
        assert loaded.last_notebook == "01_data_discovery"

    def test_save_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "nested" / "deep" / "session.json"
        state = SessionState(active_dataset="ds", active_run_id="run-1")
        state.save(path)
        assert path.exists()

    def test_load_returns_none_for_missing_file(self, tmp_path):
        path = tmp_path / "nonexistent.json"
        assert SessionState.load(path) is None

    def test_save_with_none_active_dataset(self, tmp_path):
        path = tmp_path / "session.json"
        state = SessionState(active_dataset=None, active_run_id="run-abc")
        state.save(path)
        loaded = SessionState.load(path)
        assert loaded.active_dataset is None

    def test_load_handles_corrupt_json(self, tmp_path):
        path = tmp_path / "session.json"
        path.write_text("not valid json{{{")
        assert SessionState.load(path) is None

    def test_load_handles_unexpected_exception(self, tmp_path):
        path = tmp_path / "session.json"
        path.write_text('{"active_run_id": "run-1"}')
        with patch("json.loads", side_effect=RuntimeError("unexpected")):
            assert SessionState.load(path) is None


class TestGetCurrentUsername:
    def test_local_returns_system_user(self, monkeypatch):
        monkeypatch.delenv("CR_USERNAME", raising=False)
        monkeypatch.delenv("DATABRICKS_RUNTIME_VERSION", raising=False)
        username = get_current_username()
        assert username == getpass.getuser()

    def test_cr_username_env_var_overrides(self, monkeypatch):
        monkeypatch.setenv("CR_USERNAME", "alice")
        username = get_current_username()
        assert username == "alice"

    def test_databricks_username_env_var(self, monkeypatch):
        monkeypatch.delenv("CR_USERNAME", raising=False)
        monkeypatch.setenv("DATABRICKS_RUNTIME_VERSION", "17.3")
        monkeypatch.setenv("DATABRICKS_USERNAME", "bob@company.com")
        username = get_current_username()
        assert username == "bob_company_com"


class TestSanitizeUsername:
    def test_plain_name_unchanged(self):
        assert sanitize_username("alice") == "alice"

    def test_email_sanitized(self):
        assert sanitize_username("alice@company.com") == "alice_company_com"

    def test_preserves_hyphens_underscores(self):
        assert sanitize_username("alice-bob_charlie") == "alice-bob_charlie"

    def test_strips_whitespace(self):
        assert sanitize_username("  alice  ") == "alice"

    def test_multiple_dots_and_ats(self):
        assert sanitize_username("first.last@sub.company.com") == "first_last_sub_company_com"


class TestGetCurrentUsernameDatabricksJob:
    def test_uses_spark_context_when_no_env_vars(self, monkeypatch):
        monkeypatch.delenv("CR_USERNAME", raising=False)
        monkeypatch.delenv("DATABRICKS_USERNAME", raising=False)
        monkeypatch.setenv("DATABRICKS_RUNTIME_VERSION", "14.3")
        with patch(
            "customer_retention.core.compat.detection.get_databricks_username",
            return_value="job_user@corp.com",
        ):
            assert get_current_username() == "job_user_corp_com"

    def test_databricks_username_env_still_preferred(self, monkeypatch):
        monkeypatch.delenv("CR_USERNAME", raising=False)
        monkeypatch.setenv("DATABRICKS_RUNTIME_VERSION", "14.3")
        monkeypatch.setenv("DATABRICKS_USERNAME", "env_user@corp.com")
        assert get_current_username() == "env_user_corp_com"

    def test_falls_back_to_getpass_when_no_context(self, monkeypatch):
        monkeypatch.delenv("CR_USERNAME", raising=False)
        monkeypatch.delenv("DATABRICKS_USERNAME", raising=False)
        monkeypatch.setenv("DATABRICKS_RUNTIME_VERSION", "14.3")
        with patch(
            "customer_retention.core.compat.detection.get_databricks_username",
            return_value=None,
        ):
            assert get_current_username() == getpass.getuser()

    def test_cr_username_not_sanitized(self, monkeypatch):
        monkeypatch.setenv("CR_USERNAME", "alice")
        assert get_current_username() == "alice"


class TestResolveActiveDataset:
    def test_env_var_takes_priority(self, namespace, monkeypatch):
        monkeypatch.setenv("CR_DATASET_ID", "from_env")
        result = resolve_active_dataset(namespace)
        assert result == "from_env"

    def test_session_state_second_priority(self, namespace, monkeypatch):
        monkeypatch.delenv("CR_DATASET_ID", raising=False)
        monkeypatch.setenv("CR_USERNAME", "testuser")
        state = SessionState(active_dataset="from_session", active_run_id=namespace.run_id)
        state.save(namespace.user_session_path("testuser"))
        result = resolve_active_dataset(namespace, username="testuser")
        assert result == "from_session"

    def test_first_dataset_third_priority(self, namespace, monkeypatch):
        monkeypatch.delenv("CR_DATASET_ID", raising=False)
        (namespace.datasets_dir / "alpha_dataset").mkdir()
        (namespace.datasets_dir / "beta_dataset").mkdir()
        result = resolve_active_dataset(namespace, username="nobody")
        assert result == "alpha_dataset"

    def test_returns_none_when_no_datasets(self, namespace, monkeypatch):
        monkeypatch.delenv("CR_DATASET_ID", raising=False)
        result = resolve_active_dataset(namespace, username="nobody")
        assert result is None

    def test_uses_get_current_username_when_none(self, namespace, monkeypatch):
        monkeypatch.delenv("CR_DATASET_ID", raising=False)
        monkeypatch.setenv("CR_USERNAME", "autouser")
        state = SessionState(active_dataset="auto_ds", active_run_id=namespace.run_id)
        state.save(namespace.user_session_path("autouser"))
        result = resolve_active_dataset(namespace)
        assert result == "auto_ds"


class TestSetActiveDataset:
    def test_creates_new_session(self, namespace, monkeypatch):
        monkeypatch.setenv("CR_USERNAME", "alice")
        set_active_dataset(namespace, "customers", username="alice")
        state = SessionState.load(namespace.user_session_path("alice"))
        assert state.active_dataset == "customers"

    def test_updates_existing_session(self, namespace, monkeypatch):
        monkeypatch.setenv("CR_USERNAME", "alice")
        set_active_dataset(namespace, "customers", username="alice")
        set_active_dataset(namespace, "transactions", username="alice")
        state = SessionState.load(namespace.user_session_path("alice"))
        assert state.active_dataset == "transactions"

    def test_uses_get_current_username_when_none(self, namespace, monkeypatch):
        monkeypatch.setenv("CR_USERNAME", "bob")
        set_active_dataset(namespace, "emails")
        state = SessionState.load(namespace.user_session_path("bob"))
        assert state.active_dataset == "emails"


class TestResolveFindingsPath:
    def test_returns_aggregated_when_preferred(self, namespace):
        ds_name = "customers"
        findings_dir = namespace.dataset_findings_dir(ds_name)
        findings_dir.mkdir(parents=True)
        (findings_dir / "customers_findings.yaml").touch()
        (findings_dir / "customers_aggregated_findings.yaml").touch()
        result = resolve_findings_path(namespace, ds_name, prefer_aggregated=True)
        assert result is not None
        assert "aggregated" in result.name

    def test_returns_regular_when_no_aggregated(self, namespace):
        ds_name = "customers"
        findings_dir = namespace.dataset_findings_dir(ds_name)
        findings_dir.mkdir(parents=True)
        (findings_dir / "customers_findings.yaml").touch()
        result = resolve_findings_path(namespace, ds_name, prefer_aggregated=True)
        assert result is not None
        assert result.name == "customers_findings.yaml"

    def test_returns_non_aggregated_when_not_preferred(self, namespace):
        ds_name = "customers"
        findings_dir = namespace.dataset_findings_dir(ds_name)
        findings_dir.mkdir(parents=True)
        (findings_dir / "customers_findings.yaml").touch()
        (findings_dir / "customers_aggregated_findings.yaml").touch()
        result = resolve_findings_path(namespace, ds_name, prefer_aggregated=False)
        assert result is not None
        assert "aggregated" not in result.name

    def test_returns_none_when_no_findings(self, namespace):
        result = resolve_findings_path(namespace, "nonexistent")
        assert result is None

    def test_returns_none_when_dir_exists_but_empty(self, namespace):
        ds_name = "customers"
        findings_dir = namespace.dataset_findings_dir(ds_name)
        findings_dir.mkdir(parents=True)
        result = resolve_findings_path(namespace, ds_name)
        assert result is None


class TestMarkNotebook:
    def test_creates_session_with_notebook(self, namespace, monkeypatch):
        monkeypatch.setenv("CR_USERNAME", "alice")
        mark_notebook(namespace, "01_data_discovery.ipynb", username="alice")
        state = SessionState.load(namespace.user_session_path("alice"))
        assert state is not None
        assert state.last_notebook == "01_data_discovery.ipynb"

    def test_updates_existing_session(self, namespace, monkeypatch):
        monkeypatch.setenv("CR_USERNAME", "alice")
        set_active_dataset(namespace, "customers", username="alice")
        mark_notebook(namespace, "04_column_deep_dive.ipynb", username="alice")
        state = SessionState.load(namespace.user_session_path("alice"))
        assert state.last_notebook == "04_column_deep_dive.ipynb"
        assert state.active_dataset == "customers"

    def test_uses_default_username(self, namespace, monkeypatch):
        monkeypatch.setenv("CR_USERNAME", "bob")
        mark_notebook(namespace, "02_source_integrity.ipynb")
        state = SessionState.load(namespace.user_session_path("bob"))
        assert state is not None
        assert state.last_notebook == "02_source_integrity.ipynb"

    def test_preserves_active_run_id(self, namespace, monkeypatch):
        monkeypatch.setenv("CR_USERNAME", "alice")
        set_active_dataset(namespace, "customers", username="alice")
        original_state = SessionState.load(namespace.user_session_path("alice"))
        original_run_id = original_state.active_run_id
        mark_notebook(namespace, "05_relationship_analysis.ipynb", username="alice")
        state = SessionState.load(namespace.user_session_path("alice"))
        assert state.active_run_id == original_run_id


class TestInitializeRun:
    def test_creates_namespace_directories(self, tmp_path):
        ns = initialize_run(root=tmp_path, project_name="myproj")
        assert ns.datasets_dir.is_dir()
        assert ns.merged_dir.is_dir()
        assert ns.session_dir.is_dir()

    def test_sets_cr_run_id_env_var(self, tmp_path, monkeypatch):
        monkeypatch.delenv("CR_RUN_ID", raising=False)
        ns = initialize_run(root=tmp_path, project_name="myproj")
        assert os.environ["CR_RUN_ID"] == ns.run_id

    def test_creates_session_state_for_user(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CR_USERNAME", "alice")
        ns = initialize_run(root=tmp_path, project_name="myproj", username="alice")
        assert ns.user_session_path("alice").exists()

    def test_session_state_has_run_id(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CR_USERNAME", "alice")
        ns = initialize_run(root=tmp_path, project_name="myproj", username="alice")
        state = SessionState.load(ns.user_session_path("alice"))
        assert state.active_run_id == ns.run_id

    def test_run_id_contains_project_name(self, tmp_path):
        ns = initialize_run(root=tmp_path, project_name="retention")
        assert ns.run_id.startswith("retention-")

    def test_uses_explicit_username(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CR_USERNAME", "ignored")
        ns = initialize_run(root=tmp_path, project_name="proj", username="bob")
        assert ns.user_session_path("bob").exists()
        assert not ns.user_session_path("ignored").exists()

    def test_default_username(self, tmp_path, monkeypatch):
        monkeypatch.delenv("CR_USERNAME", raising=False)
        monkeypatch.delenv("DATABRICKS_RUNTIME_VERSION", raising=False)
        ns = initialize_run(root=tmp_path, project_name="proj")
        expected_user = getpass.getuser()
        assert ns.user_session_path(expected_user).exists()


class TestResolveTargetColumn:
    @staticmethod
    def _make_context(**overrides):
        from customer_retention.analysis.auto_explorer.project_context import (
            ObjectiveAssessment,
            ObjectivePriority,
            ObjectiveSpec,
            PredictionAnchor,
            PredictionObjective,
            ProjectContext,
        )

        defaults = dict(
            project_name="test",
            objectives=[
                ObjectiveSpec(
                    objective=PredictionObjective.IMMEDIATE_RISK,
                    priority=ObjectivePriority.PRIMARY,
                    anchor=PredictionAnchor.NOW,
                    assessment=ObjectiveAssessment(confidence=90, suggested_anchor=PredictionAnchor.NOW, rationale=["test"]),
                ),
            ],
        )
        defaults.update(overrides)
        return ProjectContext(**defaults)

    def test_prefers_project_context(self, namespace):
        ctx = self._make_context(target_column="churned")
        ctx.save(namespace.project_context_path)

        class _Findings:
            target_column = "other"

        result = resolve_target_column(namespace, _Findings())
        assert result == "churned"

    def test_falls_back_to_findings(self, namespace):
        class _Findings:
            target_column = "detected"

        result = resolve_target_column(namespace, _Findings())
        assert result == "detected"

    def test_no_namespace(self):
        class _Findings:
            target_column = "from_findings"

        result = resolve_target_column(None, _Findings())
        assert result == "from_findings"

    def test_project_context_no_target(self, namespace):
        ctx = self._make_context(target_column=None)
        ctx.save(namespace.project_context_path)

        class _Findings:
            target_column = "detected"

        result = resolve_target_column(namespace, _Findings())
        assert result == "detected"


class TestLoadNotebookFindings:
    def _make_namespace_with_findings(self, tmp_path, dataset_name="customers", monkeypatch=None):
        ns = RunNamespace.create(root=tmp_path, project_name="test")
        findings_dir = ns.dataset_findings_dir(dataset_name)
        findings_dir.mkdir(parents=True)
        findings_file = findings_dir / f"{dataset_name}_findings.yaml"
        findings_file.write_text(f"source: {dataset_name}\n")
        (ns.datasets_dir / dataset_name).mkdir(exist_ok=True)
        if monkeypatch:
            monkeypatch.setenv("CR_RUN_ID", ns.run_id)
            monkeypatch.setenv("CR_EXPERIMENTS_DIR", str(tmp_path))
            monkeypatch.setenv("CR_USERNAME", "testuser")
        return ns

    def test_load_notebook_findings_from_namespace(self, tmp_path, monkeypatch):
        ns = self._make_namespace_with_findings(tmp_path, monkeypatch=monkeypatch)
        path, namespace, ds_name = load_notebook_findings("04_column_deep_dive.ipynb", root=tmp_path)
        assert path is not None
        assert "customers_findings.yaml" in path
        assert namespace is not None
        assert namespace.run_id == ns.run_id
        assert ds_name == "customers"

    def test_load_notebook_findings_fallback_to_findings_dir(self, tmp_path, monkeypatch):
        monkeypatch.delenv("CR_RUN_ID", raising=False)
        monkeypatch.delenv("CR_EXPERIMENTS_DIR", raising=False)
        monkeypatch.setenv("CR_USERNAME", "testuser")
        findings_dir = tmp_path / "findings"
        findings_dir.mkdir()
        (findings_dir / "customers_findings.yaml").write_text("source: customers\n")
        path, namespace, ds_name = load_notebook_findings(
            "04_column_deep_dive.ipynb", root=tmp_path, findings_dir=findings_dir
        )
        assert path is not None
        assert "customers_findings.yaml" in path
        assert namespace is None
        assert ds_name is None

    def test_load_notebook_findings_marks_notebook(self, tmp_path, monkeypatch):
        ns = self._make_namespace_with_findings(tmp_path, monkeypatch=monkeypatch)
        _path, _ns, _ds = load_notebook_findings("04_column_deep_dive.ipynb", root=tmp_path)
        state = SessionState.load(ns.user_session_path("testuser"))
        assert state is not None
        assert state.last_notebook == "04_column_deep_dive.ipynb"

    def test_load_notebook_findings_exclude_aggregated(self, tmp_path, monkeypatch):
        ns = self._make_namespace_with_findings(tmp_path, dataset_name="orders", monkeypatch=monkeypatch)
        findings_dir = ns.dataset_findings_dir("orders")
        (findings_dir / "orders_aggregated_findings.yaml").write_text("source: orders_agg\n")
        path, namespace, ds_name = load_notebook_findings(
            "01d_event_aggregation.ipynb", root=tmp_path, exclude_aggregated=True
        )
        assert path is not None
        assert "_aggregated" not in path
        assert ds_name == "orders"

    def test_load_notebook_findings_raises_when_nothing_found(self, tmp_path, monkeypatch):
        monkeypatch.delenv("CR_RUN_ID", raising=False)
        monkeypatch.delenv("CR_EXPERIMENTS_DIR", raising=False)
        monkeypatch.setenv("CR_USERNAME", "testuser")
        findings_dir = tmp_path / "findings"
        findings_dir.mkdir()
        with pytest.raises(FileNotFoundError):
            _path, _ns, _ds = load_notebook_findings(
                "04_column_deep_dive.ipynb", root=tmp_path, findings_dir=findings_dir
            )

    def test_load_notebook_findings_prefers_aggregated(self, tmp_path, monkeypatch):
        ns = self._make_namespace_with_findings(tmp_path, dataset_name="orders", monkeypatch=monkeypatch)
        findings_dir = ns.dataset_findings_dir("orders")
        (findings_dir / "orders_aggregated_findings.yaml").write_text("source: orders_agg\n")
        path, _, ds_name = load_notebook_findings("04_column_deep_dive.ipynb", root=tmp_path)
        assert "_aggregated" in path
        assert ds_name == "orders"

    def test_load_notebook_findings_returns_dataset_name(self, tmp_path, monkeypatch):
        self._make_namespace_with_findings(tmp_path, dataset_name="transactions", monkeypatch=monkeypatch)
        _, _, ds_name = load_notebook_findings("01a_temporal_deep_dive.ipynb", root=tmp_path)
        assert ds_name == "transactions"

    def test_load_notebook_findings_prefer_merged_returns_merged_path(self, tmp_path, monkeypatch):
        ns = self._make_namespace_with_findings(tmp_path, monkeypatch=monkeypatch)
        ns.merged_dir.mkdir(parents=True, exist_ok=True)
        ns.merged_findings_path.write_text("source: silver_merged\n")
        path, namespace, ds_name = load_notebook_findings(
            "06_feature_opportunities.ipynb", prefer_merged=True, root=tmp_path
        )
        assert path == str(ns.merged_findings_path)
        assert namespace is not None
        assert ds_name is None

    def test_load_notebook_findings_prefer_merged_falls_through(self, tmp_path, monkeypatch):
        ns = self._make_namespace_with_findings(tmp_path, monkeypatch=monkeypatch)
        path, namespace, ds_name = load_notebook_findings(
            "06_feature_opportunities.ipynb", prefer_merged=True, root=tmp_path
        )
        assert "customers_findings.yaml" in path
        assert ds_name == "customers"

    def test_load_notebook_findings_prefer_merged_false_ignores_merged(self, tmp_path, monkeypatch):
        ns = self._make_namespace_with_findings(tmp_path, monkeypatch=monkeypatch)
        ns.merged_dir.mkdir(parents=True, exist_ok=True)
        ns.merged_findings_path.write_text("source: silver_merged\n")
        path, namespace, ds_name = load_notebook_findings(
            "06_feature_opportunities.ipynb", prefer_merged=False, root=tmp_path
        )
        assert "customers_findings.yaml" in path
        assert ds_name == "customers"

    def test_load_notebook_findings_merged_returns_none_dataset_name(self, tmp_path, monkeypatch):
        ns = self._make_namespace_with_findings(tmp_path, monkeypatch=monkeypatch)
        ns.merged_dir.mkdir(parents=True, exist_ok=True)
        ns.merged_findings_path.write_text("source: silver_merged\n")
        _, _, ds_name = load_notebook_findings(
            "07_modeling_readiness.ipynb", prefer_merged=True, root=tmp_path
        )
        assert ds_name is None
