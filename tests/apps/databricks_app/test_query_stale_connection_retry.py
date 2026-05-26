"""Tests for ``_query``'s stale-connection retry path.

These tests pin the contract that EVERY first-attempt failure inside
``_query`` triggers a connection rebuild + one retry, regardless of
the exception message. The previous implementation gated the retry on
a string-match heuristic against ``_STALE_CONNECTION_HINTS`` which
only covered Thrift-level "session closed" / "broken pipe" errors --
after multi-day idle the L1 portfolio failed with token-expiry /
warehouse-cold errors whose messages weren't in the hint set, so the
recovery never kicked in.

We exercise three categories:

1. The historical "session closed" message still matches the hint set
   AND the retry kicks in (no regression on the original path).
2. Failure messages that historically did NOT match the hints (token
   expiry, warehouse cold, network) still trigger the retry now.
3. Persistent failure (both attempts raise) propagates the LAST
   exception cleanly rather than silently swallowing.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pyarrow as pa
import pytest

_APP_DIR = Path(__file__).resolve().parents[3] / "apps" / "databricks_app"


def _install_streamlit_stub():
    """Install a stub of the streamlit module so importing data.py works
    outside the streamlit runtime. Other test files in this suite also
    drop barebones stubs into sys.modules via setdefault -- since we
    need ``cache_resource`` and ``cache_data`` callable decorators, we
    UNCONDITIONALLY overwrite whatever stub got there first."""
    st = types.ModuleType("streamlit")

    class _Cache:
        def __init__(self):
            self._cache = {}

        def __call__(self, *args, **kwargs):
            # Support @st.cache_resource(show_spinner=False) and
            # @st.cache_data(ttl=..., show_spinner=False) decorators.
            if args and callable(args[0]):
                return self._wrap(args[0])
            return self._wrap

        def _wrap(self, fn):
            def wrapper(*a, **kw):
                key = (a, tuple(sorted(kw.items())))
                if key not in self._cache:
                    self._cache[key] = fn(*a, **kw)
                return self._cache[key]

            wrapper.clear = lambda: self._cache.clear()
            return wrapper

    st.cache_resource = _Cache()
    st.cache_data = _Cache()
    st.warning = lambda *a, **k: None
    st.error = lambda *a, **k: None
    # Force-install: other test files in this suite use ``setdefault``
    # with a barebones types.ModuleType("streamlit") (no ``cache_resource``).
    # That partial stub leaks across pytest collection order and breaks
    # ``data.py``'s ``@st.cache_resource(show_spinner=False)`` decorator
    # at module-import time. Overwriting wins the race deterministically.
    sys.modules["streamlit"] = st
    return st


def _install_all_stubs():
    """Idempotent install of every external-module stub data.py needs.
    Called on import AND from the ``data_mod`` fixture so prior tests
    that installed partial stubs (via ``sys.modules.setdefault``) don't
    leak through and break us when test ordering puts them first."""
    _install_streamlit_stub()

    # Databricks SDK + SQL connector. Force-set; other tests use
    # setdefault with bare ModuleType.
    _databricks = types.ModuleType("databricks")
    _databricks_sdk = types.ModuleType("databricks.sdk")
    _databricks_sdk_core = types.ModuleType("databricks.sdk.core")
    _databricks_sdk_core.Config = MagicMock
    _databricks_sdk.core = _databricks_sdk_core
    _databricks.sdk = _databricks_sdk
    sys.modules["databricks"] = _databricks
    sys.modules["databricks.sdk"] = _databricks_sdk
    sys.modules["databricks.sdk.core"] = _databricks_sdk_core

    _databricks_sql = types.ModuleType("databricks.sql")
    _databricks_sql.connect = MagicMock
    sys.modules["databricks.sql"] = _databricks_sql

    # Transitive: data.py imports things its sibling modules drag in.
    for _name in ("plotly", "plotly.express", "plotly.graph_objects"):
        sys.modules.setdefault(_name, types.ModuleType(_name))


_install_all_stubs()


def _load_data_module():
    """Import the data module fresh against the stubs above.

    ``data.py`` uses ``from . import diagnostics`` and ``from .config
    import AppConfig, load_config``. We can't rely on the surrounding
    package being on sys.path (it imports streamlit at module load
    time, which the stubs above shim) so we prebind a synthetic package
    whose ``diagnostics`` and ``config`` submodules expose the names
    ``data.py`` needs.
    """
    import importlib.util

    diagnostics_stub = types.ModuleType("diagnostics")
    diagnostics_stub.record = lambda *a, **k: None
    diagnostics_stub.is_enabled = lambda: False

    config_stub = types.ModuleType("config")

    class _AppConfig:
        warehouse_id = "wh_test"
        fqn_prefix = "c.s"

    config_stub.AppConfig = _AppConfig
    config_stub.load_config = lambda: _AppConfig()

    pkg = types.ModuleType("data_under_test_pkg")
    pkg.diagnostics = diagnostics_stub
    pkg.config = config_stub
    sys.modules["data_under_test_pkg"] = pkg
    sys.modules["data_under_test_pkg.diagnostics"] = diagnostics_stub
    sys.modules["data_under_test_pkg.config"] = config_stub

    spec = importlib.util.spec_from_file_location(
        "data_under_test_pkg.data", _APP_DIR / "src" / "data.py",
    )
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = "data_under_test_pkg"
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def data_mod():
    # Re-install stubs in case another test file's import-time
    # ``sys.modules.setdefault("streamlit", ...)`` snuck in between
    # module-import time and now (xdist worker ordering is not stable
    # across runs).
    _install_all_stubs()
    return _load_data_module()


class _StubCursor:
    """Cursor that fails on the first ``execute`` and succeeds on the
    second. Used to assert ``_query`` rebuilds the connection and
    retries."""

    def __init__(self, first_exc: Exception, ok_df: pd.DataFrame):
        self._first_exc = first_exc
        self._ok_df = ok_df
        self.execute_calls = 0
        self.closed = False

    def execute(self, sql_text, *args, **kwargs):
        self.execute_calls += 1
        if self.execute_calls == 1:
            raise self._first_exc

    def fetchall_arrow(self):
        return pa.Table.from_pandas(self._ok_df)

    def close(self):
        self.closed = True


class _StubConn:
    def __init__(self, cursor: _StubCursor):
        self._cursor = cursor

    def cursor(self):
        return self._cursor


def _stub_cfg():
    cfg = MagicMock()
    cfg.warehouse_id = "wh_1"
    return cfg


def _patch_connect(data_mod, cursor: _StubCursor):
    """Replace ``_shared_warehouse_connection`` with a stub that returns a
    connection wrapping ``cursor``. We use the same cursor instance
    across the rebuild so we can count execute calls AND assert that
    the cache was cleared in between."""
    rebuild_calls = []

    def fake_shared(_warehouse_id):
        rebuild_calls.append(_warehouse_id)
        return _StubConn(cursor)

    # cache_resource decorator wrapped the original; replace with our
    # fake which itself acts as the decorator-resolved callable.
    fake_shared.clear = lambda: rebuild_calls.append("__clear__")
    data_mod._shared_warehouse_connection = fake_shared
    return rebuild_calls


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestThriftStaleMessageRetries:
    """The original hint-set path. The first failure carries a classic
    'session closed' message; recovery must kick in and the rebuilt
    connection must succeed."""

    def test_session_closed_first_attempt_recovers(self, data_mod):
        cur = _StubCursor(
            first_exc=Exception("session was closed"),
            ok_df=pd.DataFrame({"n": [42]}),
        )
        rebuild_calls = _patch_connect(data_mod, cur)
        df = data_mod._query(_stub_cfg(), "SELECT 1")
        assert df["n"].iloc[0] == 42
        assert cur.execute_calls == 2, "_query must retry after the first failure"
        assert "__clear__" in rebuild_calls, (
            "_query must call _shared_warehouse_connection.clear() between attempts"
        )


class TestNonStaleHintFailuresAlsoRetry:
    """Regression guard for the 3-day-idle outage. Failures whose message
    does NOT match the historical hint set must STILL trigger a retry,
    not propagate immediately."""

    @pytest.mark.parametrize(
        "exc_msg",
        [
            # Token / OAuth expiry.
            "401 Unauthorized",
            "invalid_grant: token has expired",
            "could not be authenticated",
            # Warehouse cold-start / stopped.
            "warehouse is not running",
            "warehouse is starting; please retry",
            "no warehouses available",
            # HTTP / network failures.
            "connection refused",
            "name or service not known",
            "Request timed out",
            "Service Unavailable (503)",
            # Completely unfamiliar message.
            "internal driver error: something exploded in the JVM",
        ],
    )
    def test_first_failure_retries_regardless_of_message(self, data_mod, exc_msg):
        cur = _StubCursor(
            first_exc=RuntimeError(exc_msg),
            ok_df=pd.DataFrame({"hit": [1]}),
        )
        _patch_connect(data_mod, cur)
        df = data_mod._query(_stub_cfg(), "SELECT 1")
        assert df["hit"].iloc[0] == 1, (
            f"_query did not recover from {exc_msg!r} -- after 3-day idle, "
            "ANY first-attempt failure must trigger rebuild + retry."
        )
        assert cur.execute_calls == 2


class TestPersistentFailurePropagates:
    """If BOTH attempts fail the exception must surface (we don't want to
    swallow real errors). The caller's try/except handles UX."""

    def test_persistent_syntax_error_propagates_after_retry(self, data_mod):
        class _AlwaysFailCursor:
            def __init__(self):
                self.execute_calls = 0
                self.closed = False

            def execute(self, *a, **k):
                self.execute_calls += 1
                raise SyntaxError("table 'does_not_exist' not found")

            def fetchall_arrow(self):  # pragma: no cover -- unreached
                raise AssertionError("should not be called when execute fails")

            def close(self):
                self.closed = True

        cur = _AlwaysFailCursor()
        _patch_connect(data_mod, cur)

        with pytest.raises(SyntaxError, match="does_not_exist"):
            data_mod._query(_stub_cfg(), "SELECT * FROM bogus")
        assert cur.execute_calls == 2, (
            "even a 'real' error gets one retry attempt (the cost of the blanket "
            "retry policy); the failure on the retry surfaces normally"
        )


class TestStaleHintListCoversCommonLongIdleErrors:
    """The hint set is no longer a retry GATE, but it still labels
    diagnostic events. Pin that the new long-idle messages we expect
    in production actually match the heuristic so operators can spot
    them in the Diagnostics tab."""

    @pytest.mark.parametrize(
        "exc_msg",
        [
            "401 Unauthorized",
            "invalid_grant",
            "token has expired",
            "warehouse is not running",
            "connection refused",
            "name resolution failure",
            "request timed out",
            "Service Unavailable",
            "Bad Gateway",
        ],
    )
    def test_long_idle_messages_match_stale_hint(self, data_mod, exc_msg):
        assert data_mod._looks_like_stale_connection(Exception(exc_msg)), (
            f"hint set should LABEL {exc_msg!r} as stale for diagnostic clarity, "
            "even though the retry no longer depends on a match"
        )
