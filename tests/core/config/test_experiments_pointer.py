"""Tests for the project-pointer tier in ``get_experiments_dir`` + ``from_env_or_latest``.

Cycle 013 D3 surfaced the failure mode this fix resolves: notebook-job
tasks can't rely on env vars, so file-tracked discovery via
``.cr_active_run.json`` must work even when an explicit (wrong) root is
passed in.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture
def isolated_env(monkeypatch):
    """Strip discovery-related env vars so each test starts from a clean slate."""
    for var in (
        "CR_EXPERIMENTS_DIR",
        "CR_RUN_ID",
        "CR_WORKSPACE_PATH",
        "DATABRICKS_RUNTIME_VERSION",
    ):
        monkeypatch.delenv(var, raising=False)
    yield monkeypatch


class TestGetExperimentsDirReadsPointer:
    def test_pointer_overrides_project_root_fallback(self, tmp_path, isolated_env):
        from customer_retention.core.config import experiments as cfg

        target = tmp_path / "custom_experiments"
        target.mkdir()
        pointer = tmp_path / ".cr_active_run.json"
        pointer.write_text(json.dumps({
            "experiments_root": str(target),
            "run_id": "run-fixture",
        }))

        with patch.object(cfg, "_find_project_root", return_value=tmp_path):
            resolved = cfg.get_experiments_dir()
        assert str(resolved) == str(target)

    def test_missing_pointer_falls_back_to_project_root(self, tmp_path, isolated_env):
        from customer_retention.core.config import experiments as cfg

        with patch.object(cfg, "_find_project_root", return_value=tmp_path):
            resolved = cfg.get_experiments_dir()
        assert str(resolved) == str(tmp_path / "experiments")

    def test_explicit_default_wins_over_pointer(self, tmp_path, isolated_env):
        from customer_retention.core.config import experiments as cfg

        pointer = tmp_path / ".cr_active_run.json"
        pointer.write_text(json.dumps({
            "experiments_root": str(tmp_path / "from_pointer"),
            "run_id": "run-fixture",
        }))
        with patch.object(cfg, "_find_project_root", return_value=tmp_path):
            resolved = cfg.get_experiments_dir(default=str(tmp_path / "explicit"))
        assert str(resolved) == str(tmp_path / "explicit")

    def test_env_var_wins_over_pointer(self, tmp_path, isolated_env):
        from customer_retention.core.config import experiments as cfg

        pointer = tmp_path / ".cr_active_run.json"
        pointer.write_text(json.dumps({
            "experiments_root": str(tmp_path / "from_pointer"),
            "run_id": "run-fixture",
        }))
        os.environ["CR_EXPERIMENTS_DIR"] = str(tmp_path / "from_env")
        try:
            with patch.object(cfg, "_find_project_root", return_value=tmp_path):
                resolved = cfg.get_experiments_dir()
        finally:
            os.environ.pop("CR_EXPERIMENTS_DIR", None)
        assert str(resolved) == str(tmp_path / "from_env")


class TestFromEnvOrLatestConsultsPointerWithExplicitRoot:
    def test_pointer_consulted_when_explicit_root_yields_nothing(self, tmp_path, isolated_env):
        """The Cycle-013 fix: explicit root that has no runs/ should fall through to the pointer."""
        from customer_retention.analysis.auto_explorer.run_namespace import RunNamespace

        # The "wrong" explicit root has no runs/ subdir
        wrong_root = tmp_path / "wrong"
        wrong_root.mkdir()

        # The "right" pointer-resolved root carries an actual run
        right_root = tmp_path / "right"
        (right_root / "runs" / "run-real").mkdir(parents=True)
        (right_root / "runs" / "run-real" / "project_context.yaml").write_text("dummy: true")

        # Project pointer at framework root carries the right info
        framework_root = tmp_path / "framework"
        framework_root.mkdir()
        (framework_root / ".cr_active_run.json").write_text(json.dumps({
            "experiments_root": str(right_root),
            "run_id": "run-real",
        }))

        with patch(
            "customer_retention.core.config.experiments._find_project_root",
            return_value=framework_root,
        ):
            ns = RunNamespace.from_env_or_latest(root=Path(wrong_root))

        assert ns is not None, "pointer-tier fallback should have resolved the run"
        assert ns.run_id == "run-real"
        assert str(ns.root) == str(right_root)
