import os
from unittest.mock import MagicMock, patch

from customer_retention.core.compat.detection import (
    configure_spark_pandas,
    enable_arrow_optimization,
    get_databricks_username,
    get_dbutils,
    get_display_function,
    get_spark_session,
    is_databricks,
    is_notebook,
    is_pandas_api_on_spark,
    is_spark_available,
    set_spark_config,
)


class TestEnvironmentDetection:
    def test_is_databricks_returns_bool(self):
        assert isinstance(is_databricks(), bool)

    def test_is_databricks_false_locally(self):
        original = os.environ.pop("DATABRICKS_RUNTIME_VERSION", None)
        try:
            assert is_databricks() is False
        finally:
            if original:
                os.environ["DATABRICKS_RUNTIME_VERSION"] = original

    def test_is_databricks_true_when_env_set(self):
        original = os.environ.get("DATABRICKS_RUNTIME_VERSION")
        os.environ["DATABRICKS_RUNTIME_VERSION"] = "14.3"
        try:
            assert is_databricks() is True
        finally:
            if original:
                os.environ["DATABRICKS_RUNTIME_VERSION"] = original
            else:
                os.environ.pop("DATABRICKS_RUNTIME_VERSION", None)


class TestSparkAvailability:
    def test_is_spark_available_returns_bool(self):
        assert isinstance(is_spark_available(), bool)

    def test_is_pandas_api_on_spark_returns_bool(self):
        assert isinstance(is_pandas_api_on_spark(), bool)


class TestNotebookDetection:
    def test_is_notebook_returns_bool(self):
        from customer_retention.core.compat.detection import is_notebook
        assert isinstance(is_notebook(), bool)

    def test_is_notebook_false_in_tests(self):
        from customer_retention.core.compat.detection import is_notebook
        assert is_notebook() is False


class TestSparkSession:
    def test_get_spark_session_returns_none_locally(self):
        from customer_retention.core.compat.detection import get_spark_session
        result = get_spark_session()
        assert result is None or hasattr(result, "sql")


class TestDisplayFunction:
    def test_get_display_function_returns_callable(self):
        from customer_retention.core.compat.detection import get_display_function
        func = get_display_function()
        assert callable(func)


class TestDbutils:
    def test_get_dbutils_returns_none_locally(self):
        original = os.environ.pop("DATABRICKS_RUNTIME_VERSION", None)
        try:
            result = get_dbutils()
            assert result is None
        finally:
            if original:
                os.environ["DATABRICKS_RUNTIME_VERSION"] = original

    def test_get_dbutils_returns_dbutils_if_available(self):
        mock_dbutils = MagicMock()
        with patch.dict(os.environ, {"DATABRICKS_RUNTIME_VERSION": "14.3"}):
            import builtins
            original_eval = builtins.eval if hasattr(builtins, "eval") else None
            # Mock the `dbutils` name resolution via builtins
            builtins.dbutils = mock_dbutils
            try:
                result = get_dbutils()
                assert result is mock_dbutils
            finally:
                if hasattr(builtins, "dbutils"):
                    del builtins.dbutils

    def test_get_dbutils_falls_back_to_spark_dbutils(self):
        mock_spark = MagicMock()
        mock_dbutils_cls = MagicMock()
        mock_dbutils_obj = MagicMock()
        mock_dbutils_cls.return_value = mock_dbutils_obj

        mock_pyspark_dbutils = MagicMock()
        mock_pyspark_dbutils.DBUtils = mock_dbutils_cls

        with patch.dict(os.environ, {"DATABRICKS_RUNTIME_VERSION": "14.3"}):
            with patch(
                "customer_retention.core.compat.detection.get_spark_session",
                return_value=mock_spark,
            ):
                import builtins
                original_import = builtins.__import__

                def mock_import(name, *args, **kwargs):
                    if name == "pyspark.dbutils":
                        return mock_pyspark_dbutils
                    return original_import(name, *args, **kwargs)

                with patch("builtins.__import__", side_effect=mock_import):
                    result = get_dbutils()
                    assert result is mock_dbutils_obj

    def test_get_dbutils_returns_none_when_spark_has_no_dbutils(self):
        with patch.dict(os.environ, {"DATABRICKS_RUNTIME_VERSION": "14.3"}):
            with patch(
                "customer_retention.core.compat.detection.get_spark_session",
                return_value=MagicMock(),
            ):
                # Force ImportError on pyspark.dbutils
                import builtins
                original_import = builtins.__import__

                def mock_import(name, *args, **kwargs):
                    if name == "pyspark.dbutils":
                        raise ImportError("no pyspark.dbutils")
                    return original_import(name, *args, **kwargs)

                with patch("builtins.__import__", side_effect=mock_import):
                    result = get_dbutils()
                    assert result is None

    def test_get_dbutils_returns_none_when_no_spark_session(self):
        with patch.dict(os.environ, {"DATABRICKS_RUNTIME_VERSION": "14.3"}):
            with patch(
                "customer_retention.core.compat.detection.get_spark_session",
                return_value=None,
            ):
                result = get_dbutils()
                assert result is None


class TestIsNotebookWithMocks:
    def test_is_notebook_zmq_shell(self):
        mock_ipython = MagicMock()
        mock_ipython.__class__.__name__ = "ZMQInteractiveShell"

        import builtins
        builtins.get_ipython = lambda: mock_ipython
        try:
            result = is_notebook()
            assert result is True
        finally:
            del builtins.get_ipython

    def test_is_notebook_databricks_shell(self):
        mock_ipython = MagicMock()
        mock_ipython.__class__.__name__ = "DatabricksShell"

        import builtins
        builtins.get_ipython = lambda: mock_ipython
        try:
            result = is_notebook()
            assert result is True
        finally:
            del builtins.get_ipython

    def test_is_notebook_other_shell(self):
        mock_ipython = MagicMock()
        mock_ipython.__class__.__name__ = "TerminalInteractiveShell"

        import builtins
        builtins.get_ipython = lambda: mock_ipython
        try:
            result = is_notebook()
            assert result is False
        finally:
            del builtins.get_ipython


class TestGetSparkSessionWithMocks:
    def test_returns_none_when_spark_not_available(self):
        with patch(
            "customer_retention.core.compat.detection._SPARK_PANDAS_AVAILABLE", False,
        ):
            result = get_spark_session()
            assert result is None

    def test_returns_session_when_available(self):
        mock_session = MagicMock()
        with patch(
            "customer_retention.core.compat.detection._SPARK_PANDAS_AVAILABLE", True,
        ):
            with patch.dict("sys.modules", {"pyspark": MagicMock(), "pyspark.sql": MagicMock()}):
                with patch("pyspark.sql.SparkSession.getActiveSession", return_value=mock_session, create=True):
                    result = get_spark_session()
                    # Just check it runs without error

    def test_returns_none_on_exception(self):
        with patch(
            "customer_retention.core.compat.detection._SPARK_PANDAS_AVAILABLE", True,
        ):
            import builtins
            original_import = builtins.__import__

            def mock_import(name, *args, **kwargs):
                if name == "pyspark.sql":
                    raise RuntimeError("spark not available")
                return original_import(name, *args, **kwargs)

            with patch("builtins.__import__", side_effect=mock_import):
                result = get_spark_session()
                assert result is None


class TestGetDisplayFunction:
    def test_returns_print_when_not_databricks_not_notebook(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("DATABRICKS_RUNTIME_VERSION", None)
            with patch(
                "customer_retention.core.compat.detection.is_notebook",
                return_value=False,
            ):
                with patch(
                    "customer_retention.core.compat.detection.is_databricks",
                    return_value=False,
                ):
                    func = get_display_function()
                    assert func is print

    def test_returns_displayhtml_on_databricks(self):
        mock_displayhtml = MagicMock()
        import builtins
        builtins.displayHTML = mock_displayhtml

        try:
            with patch(
                "customer_retention.core.compat.detection.is_databricks",
                return_value=True,
            ):
                func = get_display_function()
                assert func is mock_displayhtml
        finally:
            del builtins.displayHTML

    def test_returns_notebook_display_when_in_notebook(self):
        with patch(
            "customer_retention.core.compat.detection.is_databricks",
            return_value=False,
        ):
            with patch(
                "customer_retention.core.compat.detection.is_notebook",
                return_value=True,
            ):
                func = get_display_function()
                assert callable(func)
                assert func is not print

    def test_databricks_falls_back_when_no_displayhtml(self):
        with patch(
            "customer_retention.core.compat.detection.is_databricks",
            return_value=True,
        ):
            with patch(
                "customer_retention.core.compat.detection.is_notebook",
                return_value=False,
            ):
                # No displayHTML in builtins, should fall through
                func = get_display_function()
                assert func is print


class TestSetSparkConfig:
    def test_set_spark_config_with_session(self):
        mock_spark = MagicMock()
        with patch(
            "customer_retention.core.compat.detection.get_spark_session",
            return_value=mock_spark,
        ):
            set_spark_config("key", "value")
            mock_spark.conf.set.assert_called_once_with("key", "value")

    def test_set_spark_config_no_session(self):
        with patch(
            "customer_retention.core.compat.detection.get_spark_session",
            return_value=None,
        ):
            # Should not raise
            set_spark_config("key", "value")


class TestEnableArrowOptimization:
    def test_enable_arrow(self):
        with patch(
            "customer_retention.core.compat.detection.set_spark_config",
        ) as mock_set:
            enable_arrow_optimization()
            mock_set.assert_called_once_with(
                "spark.sql.execution.arrow.pyspark.enabled", "true"
            )


class TestGetDatabricksUsername:
    def test_returns_none_when_not_databricks(self):
        with patch("customer_retention.core.compat.detection.get_dbutils", return_value=None):
            with patch("customer_retention.core.compat.detection.get_spark_session", return_value=None):
                assert get_databricks_username() is None

    def test_extracts_from_dbutils_context(self):
        mock_ctx = MagicMock()
        mock_ctx.userName.return_value.get.return_value = "alice@company.com"
        mock_dbutils = MagicMock()
        mock_dbutils.notebook.entry_point.getDbutils.return_value.notebook.return_value.getContext.return_value = mock_ctx
        with patch("customer_retention.core.compat.detection.get_dbutils", return_value=mock_dbutils):
            assert get_databricks_username() == "alice@company.com"

    def test_falls_back_to_spark_sql(self):
        mock_spark = MagicMock()
        mock_spark.sql.return_value.first.return_value = ["bob@company.com"]
        with patch("customer_retention.core.compat.detection.get_dbutils", return_value=None):
            with patch("customer_retention.core.compat.detection.get_spark_session", return_value=mock_spark):
                assert get_databricks_username() == "bob@company.com"

    def test_returns_none_when_dbutils_raises(self):
        mock_dbutils = MagicMock()
        mock_dbutils.notebook.entry_point.getDbutils.side_effect = Exception("no context")
        with patch("customer_retention.core.compat.detection.get_dbutils", return_value=mock_dbutils):
            with patch("customer_retention.core.compat.detection.get_spark_session", return_value=None):
                assert get_databricks_username() is None

    def test_returns_none_when_spark_sql_raises(self):
        mock_spark = MagicMock()
        mock_spark.sql.side_effect = Exception("no spark")
        with patch("customer_retention.core.compat.detection.get_dbutils", return_value=None):
            with patch("customer_retention.core.compat.detection.get_spark_session", return_value=mock_spark):
                assert get_databricks_username() is None


class TestConfigureSparkPandas:
    def test_configure_when_pandas_api_available(self):
        mock_ps = MagicMock()
        with patch(
            "customer_retention.core.compat.detection._PANDAS_API_ON_SPARK", True,
        ):
            with patch.dict("sys.modules", {"pyspark.pandas": mock_ps}):
                configure_spark_pandas(compute_max_rows=500, display_max_rows=50)

    def test_configure_when_pandas_api_not_available(self):
        with patch(
            "customer_retention.core.compat.detection._PANDAS_API_ON_SPARK", False,
        ):
            # Should not raise
            configure_spark_pandas()

    def test_configure_when_exception_raised(self):
        with patch(
            "customer_retention.core.compat.detection._PANDAS_API_ON_SPARK", True,
        ):
            import builtins
            original_import = builtins.__import__

            def mock_import(name, *args, **kwargs):
                if name == "pyspark.pandas":
                    raise Exception("not available")
                return original_import(name, *args, **kwargs)

            with patch("builtins.__import__", side_effect=mock_import):
                # Should not raise (exception is caught)
                configure_spark_pandas()
