from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from customer_retention.core.compat.detection import is_databricks
from customer_retention.core.compat.ops import ops

logger = logging.getLogger(__name__)

_PANDAS_TO_SPARK: dict[str, str] = {
    "object": "string",
    "category": "string",
    "int8": "tinyint",
    "Int8": "tinyint",
    "int16": "smallint",
    "Int16": "smallint",
    "int32": "int",
    "Int32": "int",
    "int64": "bigint",
    "Int64": "bigint",
    "float32": "float",
    "Float32": "float",
    "float64": "double",
    "Float64": "double",
    "bool": "boolean",
    "boolean": "boolean",
    "long": "bigint",
    "short": "smallint",
}


def normalize_dtype_label(dtype_str: str) -> str:
    if dtype_str in _PANDAS_TO_SPARK:
        return _PANDAS_TO_SPARK[dtype_str]
    if dtype_str.startswith("datetime64") or dtype_str in ("timestamp", "timestamp_ntz"):
        return "timestamp"
    return dtype_str


def generate_schema_report(
    loaded_frames: dict[str, Any],
    dataset_paths: dict[str, str],
    *,
    column_descriptions: Optional[dict[str, dict[str, str]]] = None,
) -> str:
    column_descriptions = column_descriptions or {}
    lines: list[str] = ["Dataset Schema Report", "=" * 50, ""]
    for name, source_path in dataset_paths.items():
        df = loaded_frames.get(name)
        if df is None:
            continue
        dtype_info = ops.get_dtype_info(df)
        descs = column_descriptions.get(name, {})
        lines.append(f"{source_path}:")
        lines.append("")
        for col, raw_dtype in dtype_info.items():
            label = normalize_dtype_label(raw_dtype)
            desc = descs.get(col)
            entry = f"{col} ({label}): {desc}" if desc else f"{col} ({label})"
            lines.append(entry)
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# High-level builder (hides LLM + validation from notebook cells)
# ---------------------------------------------------------------------------

@dataclass
class SchemaReportResult:
    """Returned by :func:`build_schema_report`."""

    text: str
    warnings: list[str] = field(default_factory=list)
    column_count: int = 0
    dataset_count: int = 0
    llm_endpoint_used: Optional[str] = None


def build_schema_report(
    loaded_frames: dict[str, Any],
    dataset_paths: dict[str, str],
    *,
    llm_descriptions: bool = False,
    llm_endpoint: Optional[str] = None,
) -> SchemaReportResult:
    """Generate a schema report, optionally enriched with LLM descriptions.

    When *llm_descriptions* is ``True`` **and** the code is running on
    Databricks, each Unity-Catalog-table dataset is sent to a Foundation
    Model serving endpoint to produce column descriptions.  File-path
    datasets are silently skipped.

    Returns a :class:`SchemaReportResult` with the report text, any
    validation warnings (columns that the LLM failed to describe), and
    summary counts.
    """
    col_descriptions: dict[str, dict[str, str]] = {}
    warnings: list[str] = []
    resolved_endpoint: Optional[str] = None

    if llm_descriptions:
        if is_databricks():
            from customer_retention.analysis.auto_explorer.column_describer import (
                describe_datasets,
                resolve_endpoint,
                validate_descriptions,
            )

            try:
                resolved_endpoint = resolve_endpoint(llm_endpoint)
            except RuntimeError as exc:
                warnings.append(
                    f"**LLM Descriptions:** {exc} "
                    "Set `LLM_ENDPOINT` in the config cell."
                )

            if resolved_endpoint is not None:
                col_descriptions = describe_datasets(
                    dataset_paths,
                    loaded_frames,
                    endpoint=resolved_endpoint,
                )

                for name in dataset_paths:
                    df = loaded_frames.get(name)
                    if df is None:
                        continue
                    all_cols = list(ops.get_dtype_info(df).keys())
                    missing = validate_descriptions(
                        all_cols, col_descriptions.get(name, {})
                    )
                    if missing:
                        preview = "`, `".join(missing[:10])
                        msg = (
                            f"**Warning:** {len(missing)}/{len(all_cols)} "
                            f"columns in **{name}** have no description: "
                            f"`{preview}`"
                        )
                        if len(missing) > 10:
                            msg += f" ... and {len(missing) - 10} more"
                        warnings.append(msg)
                        logger.warning(
                            "%d/%d columns in %s have no LLM description",
                            len(missing),
                            len(all_cols),
                            name,
                        )
        else:
            warnings.append(
                "**LLM Descriptions:** skipped (not running on Databricks)."
            )

    report_text = generate_schema_report(
        loaded_frames, dataset_paths, column_descriptions=col_descriptions
    )

    col_count = sum(1 for line in report_text.splitlines() if "(" in line)
    ds_count = len(loaded_frames)

    return SchemaReportResult(
        text=report_text,
        warnings=warnings,
        column_count=col_count,
        dataset_count=ds_count,
        llm_endpoint_used=resolved_endpoint,
    )
