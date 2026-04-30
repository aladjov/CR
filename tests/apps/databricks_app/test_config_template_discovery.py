"""``load_config`` is a thin env-var reader since the UC-table template
migration. Volume-based template discovery has been removed -- the App
now fetches the per-dataset HTML body via the SQL warehouse from
``v_dashboard_template_active`` instead, so there is no path string to
resolve at config-load time.

This file used to exercise the file-scan fallback for
``CR_PROFILE_TEMPLATE_PATH``; those tests are obsolete and have been
replaced with the small contract that remains: env vars wire through
to the ``AppConfig`` dataclass.
"""
from __future__ import annotations

import pytest

from src import config as cfgmod  # noqa: E402


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch):
    monkeypatch.delenv("CR_CATALOG", raising=False)
    monkeypatch.delenv("CR_SCHEMA", raising=False)
    monkeypatch.delenv("CR_WAREHOUSE_ID", raising=False)
    monkeypatch.delenv("CR_PROFILE_TEMPLATE_PATH", raising=False)


def test_load_config_uses_default_catalog_schema_when_env_unset():
    cfg = cfgmod.load_config()
    assert cfg.catalog == "churnkit"
    assert cfg.schema == "analysis"
    assert cfg.warehouse_id == ""
    assert cfg.fqn_prefix == "churnkit.analysis"


def test_load_config_picks_up_env_overrides(monkeypatch):
    monkeypatch.setenv("CR_CATALOG", "alt_cat")
    monkeypatch.setenv("CR_SCHEMA", "alt_sch")
    monkeypatch.setenv("CR_WAREHOUSE_ID", "wh-1234")
    cfg = cfgmod.load_config()
    assert cfg.catalog == "alt_cat"
    assert cfg.schema == "alt_sch"
    assert cfg.warehouse_id == "wh-1234"
    assert cfg.fqn_prefix == "alt_cat.alt_sch"


def test_app_config_has_no_profile_template_path_field():
    # The volume path is gone -- the dataclass should not expose it any more
    # so callers don't accidentally re-introduce volume-FUSE access paths.
    cfg = cfgmod.load_config()
    assert not hasattr(cfg, "profile_template_path")
