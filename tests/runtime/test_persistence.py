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


@pytest.fixture
def no_spark(monkeypatch):
    """Force the no-Spark fallback path so unit tests exercise the
    Delta-path-string return value (the temp-view branch needs a real
    SparkSession with Delta support, which is integration-test scope)."""
    monkeypatch.setattr(
        "customer_retention.core.compat.detection.get_spark_session",
        lambda: None,
    )


class TestPersistDataset:
    def test_writes_to_landing_dir_and_returns_path_handle_when_no_spark(
        self, stub_namespace, patched_save, no_spark
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

    def test_returned_handle_points_at_existing_artifact_no_spark(
        self, stub_namespace, patched_save, no_spark
    ):
        handle = persist_dataset(
            stub_namespace, "case", object(), purpose="filter_cancellations",
        )
        # Datasets contract: handle must point at a currently-existing
        # artifact. After persist_dataset returns, the path must exist.
        assert Path(handle).is_dir()

    def test_purpose_required(self, stub_namespace, patched_save, no_spark):
        with pytest.raises(ValueError, match="purpose"):
            persist_dataset(stub_namespace, "account", object(), purpose="")
        with pytest.raises(ValueError, match="purpose"):
            persist_dataset(stub_namespace, "account", object(), purpose="   ")

    def test_handle_loadable_signature_no_spark(
        self, stub_namespace, patched_save, no_spark
    ):
        # When no Spark session is available the helper falls back to
        # returning the Delta path string. NB01's
        # `landing_table_dir.is_dir()` check then picks it up.
        handle = persist_dataset(
            stub_namespace, "contract", object(), purpose="lifecycle",
        )
        assert handle == str(stub_namespace.landing_table_dir("contract"))

    def test_returns_global_temp_view_handle_when_spark_available(
        self, stub_namespace, patched_save, monkeypatch
    ):
        """When Spark is available, the return value is a queryable
        ``global_temp.<view>`` handle that downstream cells can pass to
        any reader. The Delta is the durable backing; the view is the
        in-session face."""

        class _StubReader:
            def __init__(self, fmt: str): self.fmt = fmt
            def load(self, path: str): return _StubFrame(path)

        class _StubFrame:
            def __init__(self, path: str): self.path = path

            # Spark API name; ruff N802 lowercase rule must be ignored
            # locally — this method overrides Spark's signature exactly.
            def createOrReplaceGlobalTempView(self, name: str):  # noqa: N802
                stub_spark.views[name] = self.path

        class _StubSpark:
            def __init__(self): self.views = {}
            @property
            def read(self):
                # mimics spark.read.format("delta").load(path)
                class _Format:
                    def format(self, fmt): return _StubReader(fmt)
                return _Format()

        stub_spark = _StubSpark()
        monkeypatch.setattr(
            "customer_retention.core.compat.detection.get_spark_session",
            lambda: stub_spark,
        )

        handle = persist_dataset(
            stub_namespace, "account", object(), purpose="derive_churn_target",
        )
        assert handle == "global_temp.cr_landing_account"
        assert "cr_landing_account" in stub_spark.views

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
