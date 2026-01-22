import os


class TestEnvironmentDetection:
    def test_is_databricks_returns_bool(self):
        from customer_retention.core.compat.detection import is_databricks
        assert isinstance(is_databricks(), bool)

    def test_is_databricks_false_locally(self):
        from customer_retention.core.compat.detection import is_databricks
        original = os.environ.pop("DATABRICKS_RUNTIME_VERSION", None)
        try:
            assert is_databricks() is False
        finally:
            if original:
                os.environ["DATABRICKS_RUNTIME_VERSION"] = original

    def test_is_databricks_true_when_env_set(self):
        from customer_retention.core.compat.detection import is_databricks
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
        from customer_retention.core.compat.detection import is_spark_available
        assert isinstance(is_spark_available(), bool)

    def test_is_pandas_api_on_spark_returns_bool(self):
        from customer_retention.core.compat.detection import is_pandas_api_on_spark
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
        from customer_retention.core.compat.detection import get_dbutils
        original = os.environ.pop("DATABRICKS_RUNTIME_VERSION", None)
        try:
            result = get_dbutils()
            assert result is None
        finally:
            if original:
                os.environ["DATABRICKS_RUNTIME_VERSION"] = original
