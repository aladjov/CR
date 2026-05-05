"""Durable Lane-2 rebind helper.

Contract: ``datasets[name]`` is *always* a string handle to a currently
materializable artifact — a Delta path, a UC table name, or a Spark
``global_temp.*`` view registered in the active session. It is never a
path or table name that does not yet exist.

When user_code mutates a frame and wants the rest of the run (and the
next exploration notebook) to see the new state, the cell must call
``cr.persist_dataset(namespace, name, df)`` and assign the return value
to ``datasets[name]``. The helper:

1. Writes ``df`` as a Delta table at ``namespace.landing_table_dir(name)``.
2. Returns a string identifier — the Delta path on local runs, or the UC
   table name on Databricks (``catalog.schema.<name>``) — that points at
   the just-persisted artifact.

This replaces ``register_temp_view`` in cross-notebook user_code:
``register_temp_view`` is session-scoped, so a fresh kernel in NB01 sees
no view and silently falls back to upstream — silently dropping the
operator's enrichment. ``persist_dataset`` survives kernel boundaries
because the Delta is on disk.

For *intra-notebook* rebinds where the cost of writing a Delta is too
high (e.g. a fast filter applied to a 100M-row frame just to display a
sample), ``cr.register_session_view(df, name)`` is the lighter helper —
it returns ``global_temp.<name>`` and the contract holds for the rest of
the SparkSession only. It MUST be replaced by ``persist_dataset`` before
the notebook hands off, otherwise NB01's load falls through.
"""
from __future__ import annotations

from typing import Any


def persist_dataset(namespace: Any, dataset_name: str, df: Any, *, purpose: str) -> str:
    """Write ``df`` to ``namespace.landing_table_dir(dataset_name)`` as
    Delta and return a queryable string handle that the rest of the
    notebook can pass to any ``_load_source`` / ``load_spark_table``
    reader.

    ``purpose`` is a required, non-empty audit string documenting why
    the caller is persisting (e.g. ``"derive_churn_target"``,
    ``"augment_parents_with_scd_history"``). Code review can scan call
    sites and ask "is this the right rebind point?". This mirrors the
    audit contract on ``register_temp_view``.

    Two-layer return value:

    1. **Delta on disk** — written to ``namespace.landing_table_dir``.
       Survives kernel restart so NB01's ``landing_table_dir(name).is_dir()``
       check picks it up automatically.
    2. **Spark global temp view** — registered against the Delta in the
       current SparkSession so subsequent NB00 cells reading
       ``datasets[name]`` (via ``_load_source`` or ``load_spark_table``)
       see a pyspark.pandas DataFrame, not a stray native-pandas read of
       the Delta directory. The view name is what we return.

    The temp view is the queryable face; the Delta is the durable
    backing. ``datasets[name]`` always points at a *currently
    materializable* artifact: in this kernel's session it is a live temp
    view; in the next kernel the landing Delta on disk is the truth.
    """
    if not isinstance(purpose, str) or not purpose.strip():
        raise ValueError(
            "persist_dataset: 'purpose' kwarg is required and must be a "
            "non-empty audit string. This protects against accidental "
            "writes during exploratory work — every persistence call "
            "should answer 'why am I rebinding datasets[...] across "
            "the notebook boundary?'"
        )
    from customer_retention.analysis.auto_explorer.active_dataset_store import (
        save_active_dataset,
    )
    save_active_dataset(namespace, dataset_name, df)
    return _register_landing_temp_view(namespace, dataset_name, purpose)


def _register_landing_temp_view(namespace: Any, dataset_name: str, purpose: str) -> str:
    """Read the just-persisted Delta back as a Spark frame and register
    it as ``global_temp.cr_landing_<dataset_name>``. Returns the view's
    qualified name. If no Spark session is available (pure-pandas local
    test environment), fall back to returning the Delta path string —
    callers running without Spark won't be reading via temp views
    anyway.
    """
    from customer_retention.core.compat.detection import get_spark_session
    landing_path = namespace.landing_table_dir(dataset_name)
    spark = get_spark_session()
    if spark is None:
        return str(landing_path)
    spark_df = spark.read.format("delta").load(str(landing_path))
    view_name = f"cr_landing_{dataset_name}"
    spark_df.createOrReplaceGlobalTempView(view_name)
    return f"global_temp.{view_name}"


def register_session_view(spark_df: Any, view_name: str, *, purpose: str) -> str:
    """Thin wrapper around ``register_temp_view`` for explicit
    intra-notebook rebinds.

    Returned handle is ``global_temp.<view_name>``. Use this only when:

    1. The frame is too large to persist as a Delta on every cell run.
    2. The rebind does NOT need to outlive the current SparkSession.

    For cross-notebook user_code rebinds (the common case) call
    ``persist_dataset`` instead — its Delta survives kernel restarts.
    """
    from customer_retention.core.compat import register_temp_view
    return register_temp_view(spark_df, view_name, purpose=purpose)
