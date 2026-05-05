"""FW-15a (Lane-2 tracker) + FW-15b (disk-backed registry) tests."""
from __future__ import annotations

import json

import pytest

from customer_retention.runtime.registry import (
    RegisteredFunction,
    Registry,
)
from customer_retention.runtime.registry import (
    registry as live_registry,
)


@pytest.fixture
def isolated_persist(tmp_path, monkeypatch):
    persist = tmp_path / "registered_functions.json"
    monkeypatch.setenv("CR_RUNTIME_REGISTRY_PATH", str(persist))
    yield persist


class TestLane2Tracker:
    def test_initial_lane2_status_is_false(self):
        r = Registry()
        rf = RegisteredFunction(
            name="derive_churn_target", source="def f(): pass", scope="datasets",
            datasets=["account"], primary="account", expected_stage="target_derive",
        )
        r.register(rf)
        assert not r.is_lane2_executed("derive_churn_target")
        assert r.lane2_status() == {}

    def test_mark_lane2_executed_flips_flag(self):
        r = Registry()
        r.register(RegisteredFunction(name="f", source="", scope="dataset", dataset="a"))
        r.mark_lane2_executed("f")
        assert r.is_lane2_executed("f")
        assert r.lane2_status() == {"f": True}

    def test_clear_resets_lane2_status(self):
        r = Registry()
        r.register(RegisteredFunction(name="f", source="", scope="dataset", dataset="a"))
        r.mark_lane2_executed("f")
        r.clear()
        assert r.lane2_status() == {}

    def test_unrelated_function_unaffected(self):
        r = Registry()
        r.register(RegisteredFunction(name="f", source="", scope="dataset", dataset="a"))
        r.register(RegisteredFunction(name="g", source="", scope="dataset", dataset="b"))
        r.mark_lane2_executed("f")
        assert r.is_lane2_executed("f")
        assert not r.is_lane2_executed("g")


class TestDiskPersistence:
    def test_register_writes_to_disk(self, isolated_persist):
        live_registry.clear()
        live_registry.register(RegisteredFunction(
            name="derive_churn_target", source="def f(): pass", scope="datasets",
            datasets=["account", "contract"], primary="account",
            expected_stage="target_derive",
        ))
        assert isolated_persist.exists()
        payload = json.loads(isolated_persist.read_text())
        assert payload["version"] == 1
        assert payload["records"][0]["name"] == "derive_churn_target"
        assert payload["records"][0]["expected_stage"] == "target_derive"
        live_registry.clear()

    def test_load_from_disk_round_trips(self, isolated_persist):
        live_registry.clear()
        rf = RegisteredFunction(
            name="filter_request_events", source="def f(): pass",
            scope="dataset", dataset="request",
        )
        live_registry.register(rf)
        live_registry.clear()  # simulate fresh kernel
        loaded = live_registry.load_from_disk(isolated_persist)
        assert loaded == 1
        names = [r.name for r in live_registry.get_registered()]
        assert "filter_request_events" in names
        live_registry.clear()

    def test_load_handles_missing_file(self, tmp_path):
        r = Registry()
        n = r.load_from_disk(tmp_path / "nope.json")
        assert n == 0
        assert r.get_registered() == []

    def test_load_corrupted_json_raises_with_actionable_message(self, tmp_path):
        """A corrupt persistence file must NOT silently zero-rehydrate —
        that would let codegen produce a pipeline with no
        user_extensions.py and silently break exploration↔production
        parity (engagement spschurn-01bb634c failure mode)."""
        bad = tmp_path / "bad.json"
        bad.write_text("not json{")
        r = Registry()
        with pytest.raises(RuntimeError, match="cannot parse"):
            r.load_from_disk(bad)

    def test_load_non_object_payload_raises(self, tmp_path):
        bad = tmp_path / "list.json"
        bad.write_text(json.dumps([1, 2, 3]))
        r = Registry()
        with pytest.raises(RuntimeError, match="not a JSON object"):
            r.load_from_disk(bad)

    def test_load_missing_records_key_raises(self, tmp_path):
        bad = tmp_path / "no_records.json"
        bad.write_text(json.dumps({"version": 1}))
        r = Registry()
        with pytest.raises(RuntimeError, match="missing the 'records' list"):
            r.load_from_disk(bad)

    def test_persistence_path_can_be_overridden_via_env(self, tmp_path, monkeypatch):
        explicit = tmp_path / "elsewhere.json"
        monkeypatch.setenv("CR_RUNTIME_REGISTRY_PATH", str(explicit))
        live_registry.clear()
        live_registry.register(RegisteredFunction(
            name="x", source="", scope="dataset", dataset="a",
        ))
        assert explicit.exists()
        live_registry.clear()

    def test_persist_failure_propagates_not_silent(self, tmp_path, monkeypatch):
        """A failed disk write must raise — silent failure would let a
        fresh NB10 kernel rehydrate stale data and break parity (this
        was the spschurn-01bb634c failure mode at the codegen layer)."""
        # Point persistence at a path whose parent IS a regular file so
        # `mkdir(parents=True, exist_ok=True)` raises FileExistsError /
        # NotADirectoryError (platform-dependent).
        blocker = tmp_path / "blocker"
        blocker.write_text("not a directory")
        explicit = blocker / "subdir" / "registered_functions.json"
        monkeypatch.setenv("CR_RUNTIME_REGISTRY_PATH", str(explicit))
        live_registry.clear()
        with pytest.raises((FileExistsError, NotADirectoryError, OSError)):
            live_registry.register(RegisteredFunction(
                name="x", source="", scope="dataset", dataset="a",
            ))
        live_registry.clear()


class TestAutoHarvestFallback:
    def test_databricks_generator_auto_harvest_no_registry(self, monkeypatch, tmp_path):
        """When neither in-memory nor disk has registered functions,
        `_auto_harvest()` returns None and the generator continues with
        `_harvest_result=None` (no `user_extensions.py` written)."""
        monkeypatch.setenv("CR_RUNTIME_REGISTRY_PATH", str(tmp_path / "empty.json"))
        from customer_retention.generators.pipeline_generator.databricks_generator import (
            _auto_harvest,
        )
        live_registry.clear()
        assert _auto_harvest() is None

    def test_databricks_generator_auto_harvest_with_registry(self, monkeypatch, tmp_path):
        """When the in-memory registry holds a function, `_auto_harvest()`
        invokes the harvester and returns a non-empty `HarvestResult`."""
        monkeypatch.setenv("CR_RUNTIME_REGISTRY_PATH", str(tmp_path / "reg.json"))
        from customer_retention.generators.pipeline_generator.databricks_generator import (
            _auto_harvest,
        )
        live_registry.clear()
        live_registry.register(RegisteredFunction(
            name="enrich_contract_lifecycle", source="def f(): pass",
            scope="dataset", dataset="contract", expected_stage="landing_post",
        ))
        result = _auto_harvest()
        assert result is not None
        assert any(
            rf.name == "enrich_contract_lifecycle"
            for rf in result.all_functions()
        )
        live_registry.clear()
