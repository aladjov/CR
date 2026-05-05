"""Tests for cr.persist_dataset / cr.register_session_view."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from customer_retention.runtime import cr
from customer_retention.runtime.persistence import (
    persist_dataset,
    register_session_view,
)


@dataclass
class StubNamespace:
    """Minimal duck-typed namespace pointing at a tmp landing dir."""
    root: Path

    def landing_table_dir(self, name: str) -> Path:
        return self.root / "landing" / name


@pytest.fixture
def stub_namespace(tmp_path):
    return StubNamespace(root=tmp_path)


@pytest.fixture
def patched_save(monkeypatch):
    """Replace save_active_dataset with a lightweight test double that
    just creates the target directory; the real save_active_dataset
    requires a Delta engine which is overkill for unit tests."""
    calls = []

    def _stub_save(namespace, name, df, **kw):
        path = namespace.landing_table_dir(name)
        path.mkdir(parents=True, exist_ok=True)
        # Touch a marker file so the directory passes `.is_dir()` checks.
        (path / "_marker").write_text("ok")
        calls.append({"namespace": namespace, "name": name, "df": df})
        return path

    monkeypatch.setattr(
        "customer_retention.analysis.auto_explorer.active_dataset_store.save_active_dataset",
        _stub_save,
    )
    return calls


class TestPersistDataset:
    def test_writes_to_landing_dir_and_returns_path_handle(
        self, stub_namespace, patched_save
    ):
        df = object()
        handle = persist_dataset(
            stub_namespace, "account", df, purpose="derive_churn_target",
        )
        landing = stub_namespace.landing_table_dir("account")
        assert handle == str(landing)
        assert landing.is_dir()
        assert len(patched_save) == 1
        assert patched_save[0]["name"] == "account"
        assert patched_save[0]["df"] is df

    def test_returned_handle_points_at_existing_artifact(
        self, stub_namespace, patched_save
    ):
        handle = persist_dataset(
            stub_namespace, "case", object(), purpose="filter_cancellations",
        )
        # Datasets contract: handle must point at a currently-existing
        # artifact. After persist_dataset returns, the path must exist.
        assert Path(handle).is_dir()

    def test_purpose_required(self, stub_namespace, patched_save):
        with pytest.raises(ValueError, match="purpose"):
            persist_dataset(stub_namespace, "account", object(), purpose="")
        with pytest.raises(ValueError, match="purpose"):
            persist_dataset(stub_namespace, "account", object(), purpose="   ")

    def test_handle_loadable_signature(self, stub_namespace, patched_save):
        # The returned string must be the same path that
        # `load_active_dataset` would compute from `(namespace, name)`.
        # We don't actually load (no Delta engine in unit tests) — we
        # check the path equality which is the load contract.
        handle = persist_dataset(
            stub_namespace, "contract", object(), purpose="lifecycle",
        )
        assert handle == str(stub_namespace.landing_table_dir("contract"))

    def test_exposed_on_cr_namespace(self):
        assert hasattr(cr, "persist_dataset")
        assert cr.persist_dataset is persist_dataset


class TestRegisterSessionView:
    def test_purpose_required(self):
        with pytest.raises(ValueError, match="purpose"):
            register_session_view(object(), "v", purpose="")

    def test_exposed_on_cr_namespace(self):
        assert hasattr(cr, "register_session_view")
        assert cr.register_session_view is register_session_view
