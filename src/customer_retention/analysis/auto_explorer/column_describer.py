"""Generate column descriptions via Databricks Foundation Model APIs.

This module is Databricks-only.  On non-Databricks environments every public
function returns empty results so callers never need to guard.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Optional

from customer_retention.core.compat.detection import is_databricks

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Endpoint auto-detection
# ---------------------------------------------------------------------------

# Preferred Foundation Model endpoints, in priority order.
# First available chat endpoint wins.
_PREFERRED_ENDPOINTS = [
    "databricks-claude-sonnet-4",
    "databricks-meta-llama-3-3-70b-instruct",
    "databricks-meta-llama-3-1-70b-instruct",
    "databricks-meta-llama-3-1-405b-instruct",
    "databricks-dbrx-instruct",
]


def resolve_endpoint(endpoint: Optional[str] = None) -> str:
    """Return an explicit endpoint name, or auto-detect one.

    When *endpoint* is ``None``, queries the workspace for available
    Foundation Model chat endpoints and returns the first match from
    :data:`_PREFERRED_ENDPOINTS`.  Falls back to any ``databricks-*``
    chat endpoint if none of the preferred ones exist.

    Raises :class:`RuntimeError` if no suitable endpoint is found.
    """
    if endpoint is not None:
        return endpoint

    try:
        import mlflow.deployments

        client = mlflow.deployments.get_deploy_client("databricks")
        all_eps = client.list_endpoints()
    except Exception:
        logger.warning("Could not list serving endpoints", exc_info=True)
        raise RuntimeError(
            "Cannot auto-detect LLM endpoint. "
            "Set LLM_ENDPOINT explicitly in the config cell."
        )

    ready_chat: dict[str, bool] = {}
    for ep in all_eps:
        name = ep.get("name", "")
        task = ep.get("task")
        state = ep.get("state", {})
        if task == "llm/v1/chat" and state.get("ready") == "READY":
            ready_chat[name] = True

    # Try preferred list first.
    for pref in _PREFERRED_ENDPOINTS:
        if pref in ready_chat:
            logger.info("Auto-detected LLM endpoint: %s", pref)
            return pref

    # Fall back to any databricks-* chat endpoint.
    for name in sorted(ready_chat):
        if name.startswith("databricks-"):
            logger.info("Auto-detected LLM endpoint (fallback): %s", name)
            return name

    raise RuntimeError(
        "No Foundation Model chat endpoints available on this workspace. "
        "Set LLM_ENDPOINT explicitly in the config cell."
    )


# ---------------------------------------------------------------------------
# Unity Catalog metadata helpers
# ---------------------------------------------------------------------------

def _parse_table_fqn(table_fqn: str) -> tuple[str, str, str]:
    """Split *catalog.schema.table* into its three parts."""
    parts = table_fqn.split(".")
    if len(parts) != 3:
        raise ValueError(
            f"Expected 'catalog.schema.table' but got {table_fqn!r}"
        )
    return parts[0], parts[1], parts[2]


def fetch_uc_column_comments(
    table_fqn: str,
    *,
    spark: Any = None,
) -> dict[str, str]:
    """Return ``{column_name: comment}`` from Unity Catalog metadata.

    Only columns that already have a non-empty comment are included.
    Returns an empty dict when not on Databricks or when the query fails.
    """
    if not is_databricks():
        return {}

    try:
        if spark is None:
            from customer_retention.core.compat.detection import get_spark_session

            spark = get_spark_session()

        catalog, schema, table = _parse_table_fqn(table_fqn)
        rows = spark.sql(
            "SELECT column_name, comment "
            "FROM system.information_schema.columns "
            "WHERE table_catalog = %s "
            "AND table_schema = %s "
            "AND table_name = %s "
            "AND comment IS NOT NULL "
            "AND comment != ''",
            args=[catalog, schema, table],
        ).collect()
        return {row["column_name"]: row["comment"] for row in rows}
    except Exception:
        logger.warning(
            "Could not fetch UC column comments for %s", table_fqn, exc_info=True
        )
        return {}


# ---------------------------------------------------------------------------
# LLM description generation
# ---------------------------------------------------------------------------

_PROMPT_TEMPLATE = """\
You are a data documentation assistant.  Given the fully-qualified Unity \
Catalog table name and its columns, generate a short (one sentence) business \
description for every column.

Table: {table_fqn}

Columns:
{column_list}

Reply with ONLY the column descriptions, one per line, using exactly this \
format (no extra text, no blank lines):
COLUMN_NAME: description text here"""


def _build_prompt(table_fqn: str, columns: list[str]) -> str:
    column_list = "\n".join(f"- {c}" for c in columns)
    return _PROMPT_TEMPLATE.format(table_fqn=table_fqn, column_list=column_list)


def _parse_llm_response(
    text: str,
    columns: list[str],
) -> dict[str, str]:
    """Parse ``COLUMN_NAME: description`` lines from the LLM response.

    Only returns entries whose column name actually appears in *columns*.
    """
    valid = set(columns)
    result: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        match = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*:\s*(.+)$", line)
        if match:
            col_name = match.group(1)
            desc = match.group(2).strip()
            if col_name in valid and desc:
                result[col_name] = desc
    return result


def generate_column_descriptions(
    table_fqn: str,
    columns: list[str],
    *,
    endpoint: Optional[str] = None,
    existing_comments: Optional[dict[str, str]] = None,
) -> dict[str, str]:
    """Call a Databricks serving endpoint to produce column descriptions.

    Parameters
    ----------
    table_fqn:
        Fully-qualified Unity Catalog table name, e.g.
        ``catalog.schema.table``.
    columns:
        Column names to describe.
    endpoint:
        Databricks Foundation Model serving endpoint name.  ``None``
        triggers auto-detection of the best available chat endpoint.
    existing_comments:
        Pre-fetched Unity Catalog comments.  Columns that already have a
        comment are *not* sent to the LLM — their existing description is
        kept as-is.

    Returns
    -------
    dict mapping column name to description string.  Empty when not on
    Databricks or when the LLM call fails.
    """
    if not is_databricks():
        logger.info("Not on Databricks — skipping LLM column descriptions.")
        return {}

    existing_comments = existing_comments or {}

    # Columns that still need a description.
    need_desc = [c for c in columns if c not in existing_comments]

    # Start with whatever UC already knows.
    descriptions: dict[str, str] = dict(existing_comments)

    if not need_desc:
        return descriptions

    try:
        resolved = resolve_endpoint(endpoint)
        import mlflow.deployments  # available on Databricks ML Runtime

        client = mlflow.deployments.get_deploy_client("databricks")
        response = client.predict(
            endpoint=resolved,
            inputs={
                "messages": [
                    {"role": "user", "content": _build_prompt(table_fqn, need_desc)},
                ],
                "max_tokens": 4096,
                "temperature": 0.1,
            },
        )
        text = response["choices"][0]["message"]["content"]
        llm_descs = _parse_llm_response(text, need_desc)
        descriptions.update(llm_descs)
    except Exception:
        logger.warning(
            "LLM column description generation failed for %s",
            table_fqn,
            exc_info=True,
        )

    return descriptions


# ---------------------------------------------------------------------------
# High-level convenience function
# ---------------------------------------------------------------------------

def describe_datasets(
    dataset_paths: dict[str, str],
    loaded_frames: dict[str, Any],
    *,
    endpoint: Optional[str] = None,
) -> dict[str, dict[str, str]]:
    """Generate column descriptions for every UC-table dataset.

    File-path datasets are silently skipped (no LLM call).
    When *endpoint* is ``None``, the best available chat endpoint is
    auto-detected.

    Returns
    -------
    ``{dataset_name: {column_name: description}}`` — ready for
    :func:`generate_schema_report`.
    """
    from customer_retention.analysis.auto_explorer.dataset_fingerprinter import (
        is_table_name,
    )
    from customer_retention.core.compat.ops import ops

    all_descriptions: dict[str, dict[str, str]] = {}

    for name, source_path in dataset_paths.items():
        if not is_table_name(source_path):
            continue

        df = loaded_frames.get(name)
        if df is None:
            continue

        columns = list(ops.get_dtype_info(df).keys())
        uc_comments = fetch_uc_column_comments(source_path)
        descs = generate_column_descriptions(
            source_path,
            columns,
            endpoint=endpoint,
            existing_comments=uc_comments,
        )
        if descs:
            all_descriptions[name] = descs

    return all_descriptions


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_descriptions(
    columns: list[str],
    descriptions: dict[str, str],
) -> list[str]:
    """Return column names that are missing a description."""
    return [c for c in columns if c not in descriptions]
