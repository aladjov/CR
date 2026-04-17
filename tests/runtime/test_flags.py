from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from customer_retention.runtime.flags import is_user_extensions_disabled


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    monkeypatch.delenv("CR_DISABLE_USER_EXTENSIONS", raising=False)


@pytest.fixture
def no_spark(monkeypatch):
    """Disable spark.conf lookup so tests that want the env-var/default
    fall-through are not intercepted by a host Spark session."""
    monkeypatch.setattr(
        "customer_retention.runtime.flags._read_spark_conf", lambda: None
    )


class TestExplicitWins:
    def test_explicit_true_returns_true(self, no_spark):
        assert is_user_extensions_disabled(True) is True

    def test_explicit_false_returns_false(self, no_spark, monkeypatch):
        monkeypatch.setenv("CR_DISABLE_USER_EXTENSIONS", "1")
        assert is_user_extensions_disabled(False) is False

    def test_explicit_true_beats_env_false(self, no_spark, monkeypatch):
        monkeypatch.setenv("CR_DISABLE_USER_EXTENSIONS", "0")
        assert is_user_extensions_disabled(True) is True


class TestSparkConfFallback:
    def test_spark_conf_true_wins_over_env(self, monkeypatch):
        monkeypatch.setenv("CR_DISABLE_USER_EXTENSIONS", "0")
        monkeypatch.setattr(
            "customer_retention.runtime.flags._read_spark_conf", lambda: "true"
        )
        assert is_user_extensions_disabled() is True

    def test_spark_conf_false_wins_over_env_true(self, monkeypatch):
        monkeypatch.setenv("CR_DISABLE_USER_EXTENSIONS", "1")
        monkeypatch.setattr(
            "customer_retention.runtime.flags._read_spark_conf", lambda: "false"
        )
        assert is_user_extensions_disabled() is False

    def test_spark_conf_missing_falls_through_to_env(self, monkeypatch):
        monkeypatch.setenv("CR_DISABLE_USER_EXTENSIONS", "1")
        monkeypatch.setattr(
            "customer_retention.runtime.flags._read_spark_conf", lambda: None
        )
        assert is_user_extensions_disabled() is True


class TestEnvFallback:
    @pytest.mark.parametrize("raw,expected", [
        ("1", True), ("true", True), ("TRUE", True),
        ("yes", True), ("on", True),
        ("0", False), ("false", False), ("no", False), ("", False),
    ])
    def test_env_coercion(self, no_spark, monkeypatch, raw, expected):
        monkeypatch.setenv("CR_DISABLE_USER_EXTENSIONS", raw)
        assert is_user_extensions_disabled() is expected


class TestDefault:
    def test_all_absent_returns_false(self, no_spark):
        assert is_user_extensions_disabled() is False


class TestSparkConfReadRobustness:
    def test_pyspark_import_failure_returns_none(self, monkeypatch):
        import customer_retention.runtime.flags as flags_mod
        real = flags_mod._read_spark_conf
        # Force the internal reader to hit the ImportError path by
        # temporarily stubbing SparkSession import
        with patch.dict("sys.modules", {"pyspark.sql": None}):
            assert real() is None

    def test_no_active_session_returns_none(self, monkeypatch):
        fake_spark_sql = MagicMock()
        fake_spark_sql.SparkSession.getActiveSession.return_value = None
        monkeypatch.setitem(__import__("sys").modules, "pyspark.sql", fake_spark_sql)
        import customer_retention.runtime.flags as flags_mod
        assert flags_mod._read_spark_conf() is None
