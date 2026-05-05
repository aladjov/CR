"""Target-column dtype validation + reusable encoding helpers (FW-12).

The framework treats classification targets as numeric 0/1. Pre-FW-12, a
multi-class string target (engagement repro: ``PARTNER_CLASSIFICATION``
carrying values like ``'Reseller'``, ``'Distributor'``) propagated from raw
data → silver merge → NB05's relationship analysis cell, where the cell's
``try_cast(... AS double)`` nulled every row and aborted the run. Operators
recovered with an in-cell ``TARGET_LABEL_MAP``, but the mapping did not
flow back upstream — silver/gold/training kept seeing the un-encoded
string column on every subsequent run.

This module provides a single classifier + a single encoding application
helper, used by:

* NB01 ``validate_target_dtype`` cell — runs at exploration profile time,
  raises with a paste-ready ``TARGET_LABEL_MAP`` template on multi-class
  string targets, auto-encodes binary string targets (sorted-order 0/1).
* Silver merge template — applies the registered mapping during target
  write so gold sees a numeric column.
* NB05 ``load_findings`` cell — reads the registered mapping first, falls
  back to the in-cell escape hatch only when no mapping is registered.
* ``FindingsParser._enforce_spec_schema_parity`` — codegen-time assertion
  that the target dtype is numeric or a label map is registered.

The classifier and encoder are deliberately framework-agnostic: they
accept pandas, pyspark.pandas, and PySpark DataFrames, dispatching to the
distributed implementation when the input is Spark-backed. Distinct-value
collection is bounded by ``distinct_cap`` so a high-cardinality column
cannot OOM the driver.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Cap on observed distinct values surfaced in diagnostic output. Matches
# the limit NB05's load cell uses when printing the value distribution.
DEFAULT_DISTINCT_CAP = 20

# Numeric kinds — anything that can be `try_cast(... AS double)` cleanly.
_NUMERIC_DTYPE_PREFIXES = (
    "int", "long", "double", "float", "decimal",
    "bigint", "smallint", "tinyint", "byte", "short",
    "number",  # pandas numeric
)

_BOOL_DTYPE_PREFIXES = ("bool",)

# Classification kinds returned by ``classify_target_dtype``. The kind
# determines what NB01 / silver / NB05 / codegen each do next.
KIND_NUMERIC = "numeric"            # already cast-friendly; no encoding needed
KIND_BOOLEAN = "boolean"            # bool column; trivially castable
KIND_BINARY_STRING = "binary_string"        # exactly 2 non-null distinct values
KIND_MULTI_CLASS_STRING = "multi_class_string"  # >2 distinct values
KIND_SINGLE_VALUE = "single_value"  # 1 distinct value — degenerate target
KIND_MISSING = "missing"            # target column absent / all null
KIND_DEGENERATE = "degenerate"      # FW-13 — too few non-null rows to model
KIND_UNKNOWN = "unknown"             # could not classify (treat as failure)

# FW-13 — minimum non-null rows the target must carry to be considered
# modelable. Caller can override via `min_non_null=` on classify/validate.
# Threshold is `max(MIN_NON_NULL_FOR_TARGET, MIN_NON_NULL_FRACTION * total)`.
MIN_NON_NULL_FOR_TARGET = 100
MIN_NON_NULL_FRACTION = 0.005  # 0.5% of total rows


@dataclass
class TargetClassification:
    """Result of ``classify_target_dtype``.

    ``kind`` drives downstream behavior:

    * ``numeric`` / ``boolean`` — target is fine as-is, no encoding needed.
    * ``binary_string`` — auto-encode using ``proposed_mapping`` (sorted
      ascending → 0, descending → 1).
    * ``multi_class_string`` — operator decision required: pick a different
      target, register a ``TARGET_LABEL_MAP``, or run ``@cr.register
      (target_derive)``. Caller raises in strict mode.
    * ``single_value`` — degenerate; raise. A target with no variance
      cannot be modeled.
    * ``missing`` — target column not on the DataFrame; raise.

    ``distinct_values`` is capped at ``distinct_cap`` for diagnostic
    output. ``distinct_count`` is the true distinct count (may exceed
    ``len(distinct_values)`` when capped).
    """
    kind: str
    column: str
    dtype: str
    distinct_values: List[Any] = field(default_factory=list)
    distinct_count: Optional[int] = None
    total_rows: Optional[int] = None
    null_rows: int = 0
    proposed_mapping: Optional[Dict[Any, int]] = None
    recommendation: str = ""

    @property
    def needs_encoding(self) -> bool:
        return self.kind in (KIND_BINARY_STRING, KIND_MULTI_CLASS_STRING)

    @property
    def is_actionable(self) -> bool:
        """True when the caller can proceed with no operator action.

        Numeric, boolean, and binary-string-with-auto-mapping all qualify.
        """
        return self.kind in (KIND_NUMERIC, KIND_BOOLEAN) or (
            self.kind == KIND_BINARY_STRING
            and self.proposed_mapping is not None
        )


def _dtype_kind(dtype_str: str) -> str:
    s = (dtype_str or "").strip().lower()
    if any(s.startswith(p) for p in _NUMERIC_DTYPE_PREFIXES):
        return "numeric"
    if any(s.startswith(p) for p in _BOOL_DTYPE_PREFIXES):
        return "boolean"
    if "object" in s or "string" in s:
        return "string"
    return "unknown"


def auto_binary_encoding(distinct_values: List[Any]) -> Optional[Dict[Any, int]]:
    """Sorted-order ``{value: 0/1}`` map when there are exactly 2 distinct values.

    Returns ``None`` for any other count (caller resolves multi-class via a
    registered ``TARGET_LABEL_MAP`` instead). The sort order is deterministic
    so re-running classification on the same data always picks the same
    encoding — important for reproducibility across exploration → silver →
    training.
    """
    non_null = [v for v in distinct_values if v is not None]
    if len(non_null) != 2:
        return None
    try:
        a, b = sorted(non_null, key=lambda v: (str(type(v).__name__), str(v)))
    except TypeError:
        return None
    return {a: 0, b: 1}


def _classify_pandas_like(df, target_column: str, distinct_cap: int, *, min_non_null=None) -> TargetClassification:
    if target_column not in df.columns:
        return TargetClassification(
            kind=KIND_MISSING, column=target_column, dtype="absent",
            recommendation=(
                f"Target column {target_column!r} not on DataFrame "
                f"(columns: {sorted(df.columns)[:10]})."
            ),
        )
    series = df[target_column]
    dtype_str = str(getattr(series, "dtype", ""))
    kind_hint = _dtype_kind(dtype_str)
    total_rows = int(len(series))
    null_rows = int(series.isna().sum())
    if kind_hint in ("numeric", "boolean"):
        kind = KIND_NUMERIC if kind_hint == "numeric" else KIND_BOOLEAN
        return TargetClassification(
            kind=kind, column=target_column, dtype=dtype_str,
            total_rows=total_rows, null_rows=null_rows,
            recommendation="target dtype is numeric — no encoding required",
        )
    # String / object path: collect distinct values bounded by `distinct_cap`.
    try:
        non_null = series.dropna()
    except Exception:  # pragma: no cover — fail-open
        return TargetClassification(
            kind=KIND_UNKNOWN, column=target_column, dtype=dtype_str,
            total_rows=total_rows, null_rows=null_rows,
            recommendation=f"could not extract non-null values from dtype={dtype_str!r}",
        )
    distinct_count_total = int(non_null.nunique())
    distinct_values: List[Any] = []
    if distinct_count_total > 0:
        # `value_counts(dropna=True)` is bounded by distinct_count so this
        # is safe even on a high-cardinality column. We trim to `distinct_cap`
        # for the surfaced list.
        vc = non_null.value_counts(dropna=True)
        distinct_values = [v for v in vc.index.tolist()[:distinct_cap]]
    return _classify_from_distinct_values(
        target_column=target_column, dtype_str=dtype_str,
        total_rows=total_rows, null_rows=null_rows,
        distinct_values=distinct_values,
        distinct_count_total=distinct_count_total,
        min_non_null=min_non_null,
    )


def _classify_spark_like(df, target_column: str, distinct_cap: int, *, min_non_null=None) -> TargetClassification:
    """Distributed classifier path for PySpark / pyspark.pandas inputs."""
    from customer_retention.core.compat import as_spark_df  # noqa: PLC0415

    sp = as_spark_df(df)
    if target_column not in sp.columns:
        return TargetClassification(
            kind=KIND_MISSING, column=target_column, dtype="absent",
            recommendation=(
                f"Target column {target_column!r} not on DataFrame "
                f"(columns: {sorted(sp.columns)[:10]})."
            ),
        )
    from pyspark.sql import functions as F  # noqa: PLC0415,N812

    schema_field = next((f for f in sp.schema.fields if f.name == target_column), None)
    dtype_str = str(schema_field.dataType.simpleString()) if schema_field is not None else "?"
    kind_hint = _dtype_kind(dtype_str)
    total_diag = sp.agg(
        F.count(F.lit(1)).alias("rows"),
        F.sum(F.when(F.col(target_column).isNull(), 1).otherwise(0)).alias("nulls"),
    ).head()
    total_rows = int(total_diag["rows"] or 0)
    null_rows = int(total_diag["nulls"] or 0)
    if kind_hint in ("numeric", "boolean"):
        kind = KIND_NUMERIC if kind_hint == "numeric" else KIND_BOOLEAN
        return TargetClassification(
            kind=kind, column=target_column, dtype=dtype_str,
            total_rows=total_rows, null_rows=null_rows,
            recommendation="target dtype is numeric — no encoding required",
        )
    # Distinct value collection capped at distinct_cap + 1 so we can detect
    # when the true distinct count exceeds the cap without collecting it all.
    cap_plus_one_rows = (
        sp.where(F.col(target_column).isNotNull())
          .select(target_column).distinct()
          .limit(distinct_cap + 1).collect()
    )
    capped_values = [r[target_column] for r in cap_plus_one_rows]
    if len(capped_values) > distinct_cap:
        # True distinct count is > cap; surface the bound.
        distinct_count_total = int(
            sp.where(F.col(target_column).isNotNull())
              .agg(F.countDistinct(F.col(target_column)).alias("d")).head()["d"]
        )
        distinct_values = sorted(capped_values[:distinct_cap], key=lambda v: str(v))
    else:
        distinct_count_total = len(capped_values)
        distinct_values = sorted(capped_values, key=lambda v: str(v))
    return _classify_from_distinct_values(
        target_column=target_column, dtype_str=dtype_str,
        total_rows=total_rows, null_rows=null_rows,
        distinct_values=distinct_values,
        distinct_count_total=distinct_count_total,
        min_non_null=min_non_null,
    )


def _classify_from_distinct_values(
    *, target_column, dtype_str, total_rows, null_rows,
    distinct_values, distinct_count_total, min_non_null=None,
) -> TargetClassification:
    """Shared classification logic given a distinct-value list bounded by cap."""
    if distinct_count_total == 0:
        return TargetClassification(
            kind=KIND_MISSING, column=target_column, dtype=dtype_str,
            total_rows=total_rows, null_rows=null_rows,
            recommendation="target column is entirely NULL",
        )
    # FW-13 — degenerate-target rejection BEFORE the binary-string fast path.
    # `PARTNER_CLASSIFICATION` had `total_rows=7000`, `null_rows=6998`,
    # `distinct_count=2` — without this guard FW-12 cheerfully auto-encoded
    # `{Customer Referral: 0, Reseller: 1}` and propagated a 99.97%-NULL
    # column through silver / gold / training.
    non_null = max(0, (total_rows or 0) - (null_rows or 0))
    threshold = (
        min_non_null
        if min_non_null is not None
        else max(MIN_NON_NULL_FOR_TARGET, int(MIN_NON_NULL_FRACTION * (total_rows or 0)))
    )
    if total_rows and non_null < threshold:
        pct = (100.0 * (null_rows or 0) / total_rows) if total_rows else 0.0
        return TargetClassification(
            kind=KIND_DEGENERATE, column=target_column, dtype=dtype_str,
            total_rows=total_rows, null_rows=null_rows,
            distinct_values=distinct_values, distinct_count=distinct_count_total,
            recommendation=(
                f"target {target_column!r} has only {non_null} non-null rows "
                f"out of {total_rows} ({pct:.2f}% missing) — below the modelable "
                f"threshold of {threshold} non-null rows. Pick a different "
                f"target column or fix the upstream derivation that should "
                f"populate this column."
            ),
        )
    if distinct_count_total == 1:
        return TargetClassification(
            kind=KIND_SINGLE_VALUE, column=target_column, dtype=dtype_str,
            total_rows=total_rows, null_rows=null_rows,
            distinct_values=distinct_values, distinct_count=1,
            recommendation=(
                "target has only one distinct value — cannot be modeled. "
                "Pick a different target column."
            ),
        )
    if distinct_count_total == 2:
        proposed = auto_binary_encoding(distinct_values)
        return TargetClassification(
            kind=KIND_BINARY_STRING, column=target_column, dtype=dtype_str,
            total_rows=total_rows, null_rows=null_rows,
            distinct_values=distinct_values, distinct_count=2,
            proposed_mapping=proposed,
            recommendation=(
                f"binary string target — auto-encoded as "
                f"{proposed!r} (sorted ascending → 0, descending → 1). "
                f"Override via `registry.set_target_label_map(...)` if "
                f"the polarity is wrong."
            ),
        )
    return TargetClassification(
        kind=KIND_MULTI_CLASS_STRING, column=target_column, dtype=dtype_str,
        total_rows=total_rows, null_rows=null_rows,
        distinct_values=distinct_values, distinct_count=distinct_count_total,
        recommendation=(
            f"multi-class string target ({distinct_count_total} distinct values). "
            f"Either pick a different target, register a TARGET_LABEL_MAP "
            f"via `registry.set_target_label_map(target, mapping, ...)`, "
            f"or define a `@cr.register(target_derive)` cell that derives "
            f"a binary target."
        ),
    )


def classify_target_dtype(
    df: Any,
    target_column: str,
    *,
    distinct_cap: int = DEFAULT_DISTINCT_CAP,
    min_non_null: Optional[int] = None,
) -> TargetClassification:
    """Inspect ``target_column`` on ``df`` and return its classification.

    Dispatches to a Spark-distributed path when ``df`` is a pyspark.pandas
    DataFrame or PySpark DataFrame, else uses pandas. Bounded by
    ``distinct_cap`` to avoid OOM on high-cardinality columns. ``min_non_null``
    overrides the FW-13 degenerate-target threshold.
    """
    from customer_retention.core.compat import _is_spark_pandas  # noqa: PLC0415

    has_to_spark = hasattr(df, "to_spark")
    is_pyspark = type(df).__module__.startswith("pyspark.sql")
    if _is_spark_pandas(df) or is_pyspark or has_to_spark:
        try:
            return _classify_spark_like(df, target_column, distinct_cap, min_non_null=min_non_null)
        except Exception as e:  # pragma: no cover — fall back gracefully
            logger.debug("Spark classifier failed (%s); using pandas path", e)
    return _classify_pandas_like(df, target_column, distinct_cap, min_non_null=min_non_null)


def render_label_map_template(distinct_values: List[Any], indent: str = "    ") -> str:
    """Format a paste-ready ``TARGET_LABEL_MAP`` template for the operator.

    Each surfaced distinct value gets a ``<value>: 0,  # …`` line so the
    operator only has to flip ``0`` → ``1`` for the churn-positive labels.
    Used in NB01's raise message and in NB05's escape-hatch comment.
    """
    if not distinct_values:
        return f"{indent}# (no distinct values observed)"
    lines = []
    for v in distinct_values:
        repr_v = repr(v)
        lines.append(f"{indent}{repr_v}: 0,  # set to 1 if this value is the positive (churn) class")
    return "\n".join(lines)


def apply_target_encoding(df: Any, target_column: str, mapping: Dict[Any, int]) -> Any:
    """Apply ``mapping`` to ``df[target_column]`` returning the same df type.

    PySpark / pyspark.pandas → CASE WHEN (one branch per mapped value).
    Pandas → ``map(...).astype("Int64")``. Unmapped values become NULL in
    both paths; the null-label filter at training time drops them.

    The function never mutates the input DataFrame; it returns a new one.
    """
    if not mapping:
        return df
    from customer_retention.core.compat import _is_spark_pandas  # noqa: PLC0415

    has_to_spark = hasattr(df, "to_spark")
    is_pyspark = type(df).__module__.startswith("pyspark.sql")
    if _is_spark_pandas(df) or is_pyspark or has_to_spark:
        from pyspark.sql import functions as F  # noqa: PLC0415,N812

        from customer_retention.core.compat import as_spark_df  # noqa: PLC0415

        sp = as_spark_df(df)
        case = F
        for value, label in mapping.items():
            case = case.when(F.col(target_column) == F.lit(value), F.lit(int(label)))
        sp_encoded = sp.withColumn(
            target_column, case.otherwise(F.lit(None).cast("double")),
        )
        if _is_spark_pandas(df) or has_to_spark:
            from customer_retention.core.compat.spark_backend import _as_pandas_api  # noqa: PLC0415
            return _as_pandas_api(sp_encoded)
        return sp_encoded
    # Pandas path.
    encoded = df.copy()
    encoded[target_column] = (
        encoded[target_column].map(mapping).astype("Int64")
    )
    return encoded


def validate_target_or_raise(
    df: Any,
    target_column: str,
    *,
    label_map: Optional[Dict[Any, int]] = None,
    distinct_cap: int = DEFAULT_DISTINCT_CAP,
    min_non_null: Optional[int] = None,
    strict: bool = True,
) -> TargetClassification:
    """Single entry point used by NB01 and codegen.

    Returns the ``TargetClassification`` on success.

    Raises ``ValueError`` (with a paste-ready ``TARGET_LABEL_MAP`` template
    in the message) when ``strict`` and:

    * the column is multi-class string with no ``label_map`` provided, or
    * the column has only one distinct value, or
    * the column is missing / entirely NULL.

    When ``label_map`` is supplied, the function additionally checks that
    every observed distinct value has a mapping entry. Unmapped values
    raise so the operator catches typos / case mismatches before the
    encoded column ships through silver / gold / training.
    """
    classification = classify_target_dtype(
        df, target_column, distinct_cap=distinct_cap, min_non_null=min_non_null,
    )
    if classification.kind in (KIND_NUMERIC, KIND_BOOLEAN):
        return classification
    if not strict:
        return classification
    if classification.kind == KIND_MISSING:
        raise ValueError(
            f"Target column {target_column!r} is missing / entirely NULL on "
            f"the input DataFrame. {classification.recommendation}"
        )
    if classification.kind == KIND_DEGENERATE:
        raise ValueError(
            f"Target column {target_column!r} is degenerate: "
            f"{classification.recommendation}"
        )
    if classification.kind == KIND_SINGLE_VALUE:
        raise ValueError(
            f"Target column {target_column!r} has only one distinct value "
            f"({classification.distinct_values!r}) — cannot be modeled. "
            f"Pick a different target column."
        )
    if classification.kind == KIND_BINARY_STRING:
        # Auto-encoding suffices unless operator passed a label_map that
        # disagrees with the observed values.
        if label_map is None:
            return classification
        observed = set(classification.distinct_values)
        unmapped = sorted(observed - set(label_map.keys()), key=str)
        if unmapped:
            raise ValueError(
                f"Registered TARGET_LABEL_MAP for {target_column!r} is "
                f"missing entries for observed distinct values: {unmapped!r}. "
                f"Add them to the map or remove unused entries."
            )
        return classification
    if classification.kind == KIND_MULTI_CLASS_STRING:
        if label_map is None:
            template = render_label_map_template(classification.distinct_values)
            raise ValueError(
                f"Target column {target_column!r} is a multi-class string "
                f"target ({classification.distinct_count} distinct values, "
                f"top-{len(classification.distinct_values)}: "
                f"{classification.distinct_values!r}). Pick one of:\n"
                f"  (a) Choose a different target column.\n"
                f"  (b) Register a label map in this notebook:\n"
                f"        registry.set_target_label_map({target_column!r}, {{\n"
                f"{template}\n"
                f"        }}, rationale=\"churn-positive class definition\", "
                f"source_notebook=\"01_data_discovery.ipynb\")\n"
                f"  (c) Define a `@cr.register(target_derive)` cell that "
                f"derives a binary target column.\n"
                f"(framework gap: docs/sps_nb10_runtime_patches_v1.md §FW-12)"
            )
        observed = set(classification.distinct_values)
        unmapped = sorted(observed - set(label_map.keys()), key=str)
        if unmapped:
            raise ValueError(
                f"Registered TARGET_LABEL_MAP for {target_column!r} is "
                f"missing entries for observed values: {unmapped!r}. "
                f"Add them or check for case/whitespace mismatches against "
                f"the surfaced distinct list."
            )
        return classification
    # KIND_UNKNOWN
    raise ValueError(
        f"Target column {target_column!r} dtype {classification.dtype!r} "
        f"could not be classified. {classification.recommendation}"
    )
