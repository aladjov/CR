from __future__ import annotations

from typing import Any, Optional

from customer_retention.core.compat.ops import ops

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
