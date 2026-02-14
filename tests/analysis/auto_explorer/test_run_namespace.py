

from customer_retention.analysis.auto_explorer.run_namespace import RunNamespace


class TestRunNamespacePathConstruction:
    def test_run_dir(self, tmp_path):
        ns = RunNamespace(root=tmp_path, run_id="myproject-abcd1234")
        assert ns.run_dir == tmp_path / "runs" / "myproject-abcd1234"

    def test_datasets_dir(self, tmp_path):
        ns = RunNamespace(root=tmp_path, run_id="proj-abc")
        assert ns.datasets_dir == tmp_path / "runs" / "proj-abc" / "datasets"

    def test_dataset_dir(self, tmp_path):
        ns = RunNamespace(root=tmp_path, run_id="proj-abc")
        assert ns.dataset_dir("customers") == tmp_path / "runs" / "proj-abc" / "datasets" / "customers"

    def test_dataset_findings_dir(self, tmp_path):
        ns = RunNamespace(root=tmp_path, run_id="proj-abc")
        expected = tmp_path / "runs" / "proj-abc" / "datasets" / "customers" / "findings"
        assert ns.dataset_findings_dir("customers") == expected

    def test_dataset_objective_support_path(self, tmp_path):
        ns = RunNamespace(root=tmp_path, run_id="proj-abc")
        expected = tmp_path / "runs" / "proj-abc" / "datasets" / "customers" / "objective_support.yaml"
        assert ns.dataset_objective_support_path("customers") == expected

    def test_dataset_docs_dir(self, tmp_path):
        ns = RunNamespace(root=tmp_path, run_id="proj-abc")
        expected = tmp_path / "runs" / "proj-abc" / "datasets" / "customers" / "docs"
        assert ns.dataset_docs_dir("customers") == expected

    def test_data_dir(self, tmp_path):
        ns = RunNamespace(root=tmp_path, run_id="proj-abc")
        assert ns.data_dir == tmp_path / "runs" / "proj-abc" / "data"

    def test_landing_dir(self, tmp_path):
        ns = RunNamespace(root=tmp_path, run_id="proj-abc")
        assert ns.landing_dir == tmp_path / "runs" / "proj-abc" / "data" / "landing"

    def test_bronze_dir(self, tmp_path):
        ns = RunNamespace(root=tmp_path, run_id="proj-abc")
        assert ns.bronze_dir == tmp_path / "runs" / "proj-abc" / "data" / "bronze"

    def test_silver_dir(self, tmp_path):
        ns = RunNamespace(root=tmp_path, run_id="proj-abc")
        assert ns.silver_dir == tmp_path / "runs" / "proj-abc" / "data" / "silver"

    def test_gold_dir(self, tmp_path):
        ns = RunNamespace(root=tmp_path, run_id="proj-abc")
        assert ns.gold_dir == tmp_path / "runs" / "proj-abc" / "data" / "gold"

    def test_landing_table_dir(self, tmp_path):
        ns = RunNamespace(root=tmp_path, run_id="proj-abc")
        expected = tmp_path / "runs" / "proj-abc" / "data" / "landing" / "customers"
        assert ns.landing_table_dir("customers") == expected

    def test_bronze_table_dir(self, tmp_path):
        ns = RunNamespace(root=tmp_path, run_id="proj-abc")
        expected = tmp_path / "runs" / "proj-abc" / "data" / "bronze" / "events_aggregated"
        assert ns.bronze_table_dir("events") == expected

    def test_gold_table_dir(self, tmp_path):
        ns = RunNamespace(root=tmp_path, run_id="proj-abc")
        expected = tmp_path / "runs" / "proj-abc" / "data" / "gold" / "gold_features_cust_emai__abc1234"
        assert ns.gold_table_dir("cust_emai__abc1234") == expected

    def test_merged_dir(self, tmp_path):
        ns = RunNamespace(root=tmp_path, run_id="proj-abc")
        assert ns.merged_dir == tmp_path / "runs" / "proj-abc" / "merged"

    def test_candidate_dir(self, tmp_path):
        ns = RunNamespace(root=tmp_path, run_id="proj-abc")
        expected = tmp_path / "runs" / "proj-abc" / "merged" / "candidate_v1"
        assert ns.candidate_dir("candidate_v1") == expected

    def test_session_dir(self, tmp_path):
        ns = RunNamespace(root=tmp_path, run_id="proj-abc")
        expected = tmp_path / "runs" / "proj-abc" / "session" / "users"
        assert ns.session_dir == expected

    def test_user_session_path(self, tmp_path):
        ns = RunNamespace(root=tmp_path, run_id="proj-abc")
        expected = tmp_path / "runs" / "proj-abc" / "session" / "users" / "alice.json"
        assert ns.user_session_path("alice") == expected

    def test_project_context_path(self, tmp_path):
        ns = RunNamespace(root=tmp_path, run_id="proj-abc")
        expected = tmp_path / "runs" / "proj-abc" / "project_context.yaml"
        assert ns.project_context_path == expected

    def test_snapshot_grid_path(self, tmp_path):
        ns = RunNamespace(root=tmp_path, run_id="proj-abc")
        expected = tmp_path / "runs" / "proj-abc" / "snapshot_grid.yaml"
        assert ns.snapshot_grid_path == expected

    def test_silver_merged_path(self, tmp_path):
        ns = RunNamespace(root=tmp_path, run_id="proj-abc")
        expected = tmp_path / "runs" / "proj-abc" / "data" / "silver" / "silver_merged"
        assert ns.silver_merged_path == expected


class TestListDatasets:
    def test_list_datasets_empty(self, tmp_path):
        ns = RunNamespace(root=tmp_path, run_id="proj-abc")
        ns.setup()
        assert ns.list_datasets() == []

    def test_list_datasets_with_populated_dirs(self, tmp_path):
        ns = RunNamespace(root=tmp_path, run_id="proj-abc")
        ns.setup()
        (ns.datasets_dir / "customers").mkdir()
        (ns.datasets_dir / "transactions").mkdir()
        result = ns.list_datasets()
        assert sorted(result) == ["customers", "transactions"]

    def test_list_datasets_ignores_files(self, tmp_path):
        ns = RunNamespace(root=tmp_path, run_id="proj-abc")
        ns.setup()
        (ns.datasets_dir / "customers").mkdir()
        (ns.datasets_dir / "stray_file.txt").touch()
        assert ns.list_datasets() == ["customers"]


class TestListCandidates:
    def test_list_candidates_empty(self, tmp_path):
        ns = RunNamespace(root=tmp_path, run_id="proj-abc")
        ns.setup()
        assert ns.list_candidates() == []

    def test_list_candidates_with_populated_dirs(self, tmp_path):
        ns = RunNamespace(root=tmp_path, run_id="proj-abc")
        ns.setup()
        (ns.merged_dir / "v1").mkdir()
        (ns.merged_dir / "v2").mkdir()
        result = ns.list_candidates()
        assert sorted(result) == ["v1", "v2"]


class TestSetup:
    def test_setup_creates_directories(self, tmp_path):
        ns = RunNamespace(root=tmp_path, run_id="proj-abc")
        ns.setup()
        assert ns.run_dir.is_dir()
        assert ns.datasets_dir.is_dir()
        assert ns.merged_dir.is_dir()
        assert ns.session_dir.is_dir()
        assert ns.landing_dir.is_dir()
        assert ns.bronze_dir.is_dir()
        assert ns.silver_dir.is_dir()
        assert ns.gold_dir.is_dir()

    def test_setup_is_idempotent(self, tmp_path):
        ns = RunNamespace(root=tmp_path, run_id="proj-abc")
        ns.setup()
        ns.setup()
        assert ns.run_dir.is_dir()


class TestCreateFactory:
    def test_create_generates_run_id_with_project_name(self, tmp_path):
        ns = RunNamespace.create(root=tmp_path, project_name="retention")
        assert ns.run_id.startswith("retention-")
        assert len(ns.run_id) == len("retention-") + 8

    def test_create_sets_up_directories(self, tmp_path):
        ns = RunNamespace.create(root=tmp_path, project_name="myproject")
        assert ns.run_dir.is_dir()
        assert ns.datasets_dir.is_dir()

    def test_create_unique_run_ids(self, tmp_path):
        ns1 = RunNamespace.create(root=tmp_path, project_name="proj")
        ns2 = RunNamespace.create(root=tmp_path, project_name="proj")
        assert ns1.run_id != ns2.run_id


class TestFromEnv:
    def test_from_env_reads_run_id(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CR_RUN_ID", "test-run-12345678")
        monkeypatch.setenv("CR_EXPERIMENTS_DIR", str(tmp_path))
        ns = RunNamespace.from_env()
        assert ns.run_id == "test-run-12345678"
        assert ns.root == tmp_path

    def test_from_env_returns_none_without_run_id(self, monkeypatch):
        monkeypatch.delenv("CR_RUN_ID", raising=False)
        assert RunNamespace.from_env() is None

    def test_from_env_uses_custom_root(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CR_RUN_ID", "my-run-abc12345")
        ns = RunNamespace.from_env(root=tmp_path)
        assert ns.root == tmp_path
        assert ns.run_id == "my-run-abc12345"


class TestMergedFindingsPath:
    def test_merged_findings_path(self, tmp_path):
        ns = RunNamespace(root=tmp_path, run_id="proj-abc")
        expected = tmp_path / "runs" / "proj-abc" / "merged" / "silver_merged_findings.yaml"
        assert ns.merged_findings_path == expected


class TestMultiDatasetFindingsPath:
    def test_returns_merged_multi_dataset_findings(self, tmp_path):
        ns = RunNamespace(root=tmp_path, run_id="proj-abc")
        expected = tmp_path / "runs" / "proj-abc" / "merged" / "multi_dataset_findings.yaml"
        assert ns.multi_dataset_findings_path == expected


class TestMergedRecommendationsPath:
    def test_returns_merged_recommendations(self, tmp_path):
        ns = RunNamespace(root=tmp_path, run_id="proj-abc")
        expected = tmp_path / "runs" / "proj-abc" / "merged" / "recommendations.yaml"
        assert ns.merged_recommendations_path == expected


class TestDiscoverAllFindings:
    def test_empty_datasets_returns_empty(self, tmp_path):
        ns = RunNamespace(root=tmp_path, run_id="proj-abc")
        ns.setup()
        assert ns.discover_all_findings() == []

    def test_single_dataset_single_findings(self, tmp_path):
        ns = RunNamespace(root=tmp_path, run_id="proj-abc")
        ns.setup()
        findings_dir = ns.dataset_findings_dir("customers")
        findings_dir.mkdir(parents=True)
        (findings_dir / "customers_findings.yaml").touch()
        result = ns.discover_all_findings()
        assert len(result) == 1
        assert result[0].name == "customers_findings.yaml"

    def test_multiple_datasets_with_findings(self, tmp_path):
        ns = RunNamespace(root=tmp_path, run_id="proj-abc")
        ns.setup()
        for name in ["customers", "orders"]:
            fd = ns.dataset_findings_dir(name)
            fd.mkdir(parents=True)
            (fd / f"{name}_findings.yaml").touch()
        result = ns.discover_all_findings()
        assert len(result) == 2

    def test_prefer_aggregated_true(self, tmp_path):
        ns = RunNamespace(root=tmp_path, run_id="proj-abc")
        ns.setup()
        fd = ns.dataset_findings_dir("orders")
        fd.mkdir(parents=True)
        (fd / "orders_findings.yaml").touch()
        (fd / "orders_aggregated_findings.yaml").touch()
        result = ns.discover_all_findings(prefer_aggregated=True)
        assert len(result) == 1
        assert "aggregated" in result[0].name

    def test_prefer_aggregated_false(self, tmp_path):
        ns = RunNamespace(root=tmp_path, run_id="proj-abc")
        ns.setup()
        fd = ns.dataset_findings_dir("orders")
        fd.mkdir(parents=True)
        (fd / "orders_findings.yaml").touch()
        (fd / "orders_aggregated_findings.yaml").touch()
        result = ns.discover_all_findings(prefer_aggregated=False)
        assert len(result) == 2

class TestFromLatest:
    def _make_run_with_context(self, root, run_id, mtime_offset=0):
        ns = RunNamespace(root=root, run_id=run_id)
        ns.setup()
        ctx_path = ns.project_context_path
        ctx_path.write_text(f"run_id: {run_id}\n")
        import os
        import time
        base = time.time() + mtime_offset
        os.utime(ctx_path, (base, base))
        return ns

    def test_returns_none_when_no_runs_dir(self, tmp_path):
        result = RunNamespace.from_latest(root=tmp_path)
        assert result is None

    def test_returns_none_when_no_project_context(self, tmp_path):
        runs_dir = tmp_path / "runs" / "some-run"
        runs_dir.mkdir(parents=True)
        result = RunNamespace.from_latest(root=tmp_path)
        assert result is None

    def test_returns_single_run(self, tmp_path):
        self._make_run_with_context(tmp_path, "proj-abc12345")
        result = RunNamespace.from_latest(root=tmp_path)
        assert result is not None
        assert result.run_id == "proj-abc12345"
        assert result.root == tmp_path

    def test_returns_latest_by_mtime(self, tmp_path):
        self._make_run_with_context(tmp_path, "proj-older", mtime_offset=-100)
        self._make_run_with_context(tmp_path, "proj-newer", mtime_offset=0)
        result = RunNamespace.from_latest(root=tmp_path)
        assert result.run_id == "proj-newer"

    def test_uses_mtime_not_alphabetical(self, tmp_path):
        self._make_run_with_context(tmp_path, "zzz-run", mtime_offset=-100)
        self._make_run_with_context(tmp_path, "aaa-run", mtime_offset=0)
        result = RunNamespace.from_latest(root=tmp_path)
        assert result.run_id == "aaa-run"

    def test_ignores_dirs_without_project_context(self, tmp_path):
        (tmp_path / "runs" / "no-context-run").mkdir(parents=True)
        self._make_run_with_context(tmp_path, "has-context")
        result = RunNamespace.from_latest(root=tmp_path)
        assert result.run_id == "has-context"

    def test_ignores_files_in_runs_dir(self, tmp_path):
        self._make_run_with_context(tmp_path, "real-run")
        (tmp_path / "runs" / "stray_file.txt").touch()
        result = RunNamespace.from_latest(root=tmp_path)
        assert result.run_id == "real-run"

    def test_finds_run_with_datasets_dir_only(self, tmp_path):
        ns = RunNamespace(root=tmp_path, run_id="datasets-only")
        ns.setup()
        result = RunNamespace.from_latest(root=tmp_path)
        assert result is not None
        assert result.run_id == "datasets-only"

    def test_prefers_project_context_over_datasets_only(self, tmp_path):
        import time

        ns_datasets = RunNamespace(root=tmp_path, run_id="datasets-only")
        ns_datasets.setup()

        time.sleep(0.05)

        self._make_run_with_context(tmp_path, "has-context", mtime_offset=0)

        result = RunNamespace.from_latest(root=tmp_path)
        assert result.run_id == "has-context"

    def test_latest_datasets_only_wins_over_older_context(self, tmp_path):
        self._make_run_with_context(tmp_path, "old-context", mtime_offset=-100)

        ns_datasets = RunNamespace(root=tmp_path, run_id="new-datasets")
        ns_datasets.setup()
        import os
        import time

        base = time.time()
        os.utime(ns_datasets.datasets_dir, (base, base))

        result = RunNamespace.from_latest(root=tmp_path)
        assert result.run_id == "new-datasets"


    def test_mixed_aggregated_and_non_aggregated(self, tmp_path):
        ns = RunNamespace(root=tmp_path, run_id="proj-abc")
        ns.setup()
        fd_orders = ns.dataset_findings_dir("orders")
        fd_orders.mkdir(parents=True)
        (fd_orders / "orders_findings.yaml").touch()
        (fd_orders / "orders_aggregated_findings.yaml").touch()
        fd_cust = ns.dataset_findings_dir("customers")
        fd_cust.mkdir(parents=True)
        (fd_cust / "customers_findings.yaml").touch()
        result = ns.discover_all_findings(prefer_aggregated=True)
        assert len(result) == 2
        names = {r.name for r in result}
        assert "orders_aggregated_findings.yaml" in names
        assert "customers_findings.yaml" in names
