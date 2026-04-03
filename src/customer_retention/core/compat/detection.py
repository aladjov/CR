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

_BUNDLE_VAR_TO_ENV = {
    "catalog": "CR_CATALOG",
    "schema": "CR_SCHEMA",
}


def _load_bundle_variables() -> None:
    import yaml  # noqa: PLC0415 — deferred import, only used for CR_SPARK_REMOTE

    _bundle_file = os.path.join(os.getcwd(), "databricks.yml")
    if not os.path.isfile(_bundle_file):
        return
    try:
        with open(_bundle_file) as f:
            bundle = yaml.safe_load(f)
    except Exception:
        return
    variables = bundle.get("variables", {})
    target_name = os.environ.get("DATABRICKS_BUNDLE_TARGET", "dev")
    target_vars = bundle.get("targets", {}).get(target_name, {}).get("variables", {})
    for var_name, env_name in _BUNDLE_VAR_TO_ENV.items():
        value = target_vars.get(var_name)
        if not value or "${" in str(value):
            value = variables.get(var_name, {}).get("default")
        if value:
            os.environ.setdefault(env_name, str(value))


if _PYSPARK_IMPORTABLE:
    _on_databricks = bool(os.environ.get("DATABRICKS_RUNTIME_VERSION"))
    _has_active_session = False

    if not _on_databricks and os.environ.get("CR_SPARK_REMOTE"):
        try:
            _env_file = os.path.join(os.getcwd(), ".databricks", ".databricks.env")
            if os.path.isfile(_env_file):
                with open(_env_file) as _f:
                    for _line in _f:
                        _line = _line.strip()
                        if _line and not _line.startswith("#") and "=" in _line:
                            _k, _, _v = _line.partition("=")
                            os.environ.setdefault(_k.strip(), _v.strip())
            _load_bundle_variables()
            from databricks.connect import DatabricksSession
            DatabricksSession.builder.getOrCreate()
        except Exception as _exc:
            import warnings
            warnings.warn(
                f"CR_SPARK_REMOTE is set but DatabricksSession failed: {_exc}",
                stacklevel=1,
            )

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


def is_remote_spark() -> bool:
    if bool(os.environ.get("DATABRICKS_RUNTIME_VERSION")):
        return False
    return bool(os.environ.get("CR_SPARK_REMOTE"))


def use_distributed_processing() -> bool:
    return is_databricks() or is_remote_spark()


def connect_remote_spark() -> Any:
    global _SPARK_PANDAS_AVAILABLE, _PANDAS_API_ON_SPARK
    if _SPARK_PANDAS_AVAILABLE:
        return get_spark_session()
    if not _PYSPARK_IMPORTABLE:
        raise ImportError("pyspark is not installed")
    from databricks.connect import DatabricksSession
    spark = DatabricksSession.builder.getOrCreate()
    _SPARK_PANDAS_AVAILABLE = True
    _PANDAS_API_ON_SPARK = True
    from customer_retention.core.compat import _activate_spark_pandas
    _activate_spark_pandas()
    return spark


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
    if not (is_databricks() or is_remote_spark() or bool(os.environ.get("CR_SPARK_REMOTE"))):
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


def get_default_parallelism() -> int:
    """Return Spark default parallelism, safe for shared clusters.

    Uses ``spark.conf.get`` with a fallback (RuntimeConfig API — works on
    Unity Catalog shared clusters where ``sparkContext`` is blocked).
    Falls back to ``sparkContext`` for local mode where the conf may
    not be explicitly set.  Returns 0 if neither is available.

    All downstream callers (``_ensure_parallelism``, ``_repartition_for_training``)
    use only Spark SQL / DataFrame APIs (``repartition``, ``cache``, ``inputFiles``)
    which are Spark Connect safe since 3.4.0.
    """
    spark = get_spark_session()
    if not spark:
        return 0
    # Two-arg form returns the default without raising if key is absent
    val = spark.conf.get("spark.default.parallelism", None)
    if val is not None:
        try:
            return int(val)
        except (TypeError, ValueError):
            pass
    # Fallback for local mode — sparkContext is available there
    try:
        return int(spark.sparkContext.defaultParallelism)
    except Exception:
        return 0


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
