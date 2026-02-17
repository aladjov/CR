from __future__ import annotations

import os
from typing import Any, Callable, Optional

_SPARK_PANDAS_AVAILABLE = False
_PANDAS_API_ON_SPARK = False

_PYSPARK_IMPORTABLE = False
try:
    import pyspark.pandas as ps  # noqa: F401
    _PYSPARK_IMPORTABLE = True
except Exception:
    pass

if not _PYSPARK_IMPORTABLE:
    try:
        import databricks.koalas as ps  # noqa: F401
        _PYSPARK_IMPORTABLE = True
    except Exception:
        pass

if _PYSPARK_IMPORTABLE:
    _on_databricks = bool(os.environ.get("DATABRICKS_RUNTIME_VERSION"))
    _has_active_session = False
    if not _on_databricks:
        try:
            from pyspark.sql import SparkSession
            _has_active_session = SparkSession.getActiveSession() is not None
        except Exception:
            pass
    if _on_databricks or _has_active_session:
        _SPARK_PANDAS_AVAILABLE = True
        _PANDAS_API_ON_SPARK = True


def is_spark_available() -> bool:
    return _SPARK_PANDAS_AVAILABLE


def is_pandas_api_on_spark() -> bool:
    return _PANDAS_API_ON_SPARK


def is_databricks() -> bool:
    return bool(os.environ.get("DATABRICKS_RUNTIME_VERSION"))


def is_notebook() -> bool:
    try:
        shell = get_ipython().__class__.__name__  # type: ignore[name-defined]
        return shell in ("ZMQInteractiveShell", "DatabricksShell", "Shell")
    except NameError:
        return False


def get_spark_session() -> Optional[Any]:
    if not _SPARK_PANDAS_AVAILABLE:
        return None
    try:
        from pyspark.sql import SparkSession
        return SparkSession.getActiveSession()
    except Exception:
        return None


def get_display_function() -> Callable[[str], None]:
    if is_databricks():
        try:
            return displayHTML  # type: ignore[name-defined]
        except NameError:
            pass
    if is_notebook():
        from IPython.display import HTML, display
        return lambda html: display(HTML(html))
    return print


def get_dbutils() -> Optional[Any]:
    if not is_databricks():
        return None
    try:
        return dbutils  # type: ignore[name-defined]
    except NameError:
        spark = get_spark_session()
        if spark:
            try:
                from pyspark.dbutils import DBUtils
                return DBUtils(spark)
            except ImportError:
                pass
    return None


def get_databricks_username() -> Optional[str]:
    dbu = get_dbutils()
    if dbu:
        try:
            ctx = dbu.notebook.entry_point.getDbutils().notebook().getContext()
            return ctx.userName().get()
        except Exception:
            pass
    spark = get_spark_session()
    if spark:
        try:
            return spark.sql("SELECT current_user()").first()[0]
        except Exception:
            pass
    return None


def set_spark_config(key: str, value: Any) -> None:
    spark = get_spark_session()
    if spark:
        spark.conf.set(key, value)


def enable_arrow_optimization() -> None:
    set_spark_config("spark.sql.execution.arrow.pyspark.enabled", "true")


def configure_spark_pandas(compute_max_rows: int = 1000, display_max_rows: int = 100) -> None:
    if _PANDAS_API_ON_SPARK:
        try:
            import pyspark.pandas as ps
            ps.set_option("compute.max_rows", compute_max_rows)
            ps.set_option("display.max_rows", display_max_rows)
        except Exception:
            pass
