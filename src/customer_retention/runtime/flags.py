from __future__ import annotations

import os
from typing import Optional

_SPARK_CONF_KEY = "spark.cr.disable_user_extensions"
_ENV_VAR = "CR_DISABLE_USER_EXTENSIONS"


def is_user_extensions_disabled(explicit: Optional[bool] = None) -> bool:
    """Resolve the user-extensions kill switch.

    Order, top wins: explicit kwarg → spark.conf → os.environ → False.
    Env vars are unreliable on Databricks (cluster env vars need a restart;
    notebook-scope env vars don't persist across tasks), so spark.conf is
    the production-ops path.
    """
    if explicit is not None:
        return bool(explicit)
    spark_val = _read_spark_conf()
    if spark_val is not None:
        return _coerce_bool(spark_val)
    env_val = os.environ.get(_ENV_VAR)
    if env_val is not None:
        return _coerce_bool(env_val)
    return False


def _read_spark_conf() -> Optional[str]:
    try:
        from pyspark.sql import SparkSession
    except ImportError:
        return None
    try:
        spark = SparkSession.getActiveSession()
    except (ImportError, AttributeError):
        return None
    if spark is None:
        return None
    try:
        return spark.conf.get(_SPARK_CONF_KEY, None)
    except (AttributeError, RuntimeError):
        return None


def _coerce_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    s = str(value).strip().lower()
    return s in ("1", "true", "yes", "on")
