from typing import Any, List, Optional

from jinja2 import Environment

from .models import (
    BronzeEventConfig,
    BronzeLayerConfig,
    PipelineConfig,
    PipelineTransformationType,
    TransformationStep,
)
from .renderer import (
    DEFAULT_NOTEBOOK_MAP,
    SECTION_MAP,
    InlineLoader,
    _notebook_title,
    _sorted_landing_names,
    group_steps,
    provenance_key,
    render_python_literal,
)


def render_spark_step_call(step: TransformationStep) -> str:
    t, col, p = step.type, step.column, step.parameters
    registry = _SPARK_REGISTRY.get(t)
    if registry is not None:
        return registry(col, p)
    raise ValueError(f"Unknown transformation type: {t}")


def _impute_null(col, p):
    # Guard against recommendations that target a column not in this stage's
    # input — same rationale as the per-column transforms below.
    value = p.get("value", 0)
    if isinstance(value, str):
        return f'(df.fillna("{value}", subset=["{col}"]) if "{col}" in df.columns else df)'
    return f'(df.fillna({value}, subset=["{col}"]) if "{col}" in df.columns else df)'


def _cap_outlier(col, p):
    lower = p.get("lower", 0)
    upper = p.get("upper", 1000000)
    # Recommendations registered against exploration's silver-merged frame can
    # target columns that don't exist in this stage's input (e.g. silver-derived
    # `days_since_*`/`days_until_*` recency features mistakenly attributed to
    # bronze). Guard the withColumn so the rec degrades to a no-op at runtime
    # instead of failing the stage with UNRESOLVED_COLUMN.
    return (
        f'(df.withColumn("{col}", '
        f'F.when(F.col("{col}") < {lower}, {lower})'
        f'.when(F.col("{col}") > {upper}, {upper})'
        f'.otherwise(F.col("{col}"))) '
        f'if "{col}" in df.columns else df)'
    )


def _drop_column(col, _p):
    return f'df.drop("{col}")'


def _winsorize(col, p):
    lower = p.get("lower_bound", 0)
    upper = p.get("upper_bound", 1000000)
    # See `_cap_outlier` — same column-existence guard.
    return (
        f'(df.withColumn("{col}", '
        f'F.when(F.col("{col}") < {lower}, {lower})'
        f'.when(F.col("{col}") > {upper}, {upper})'
        f'.otherwise(F.col("{col}"))) '
        f'if "{col}" in df.columns else df)'
    )


def _log_transform(col, _p):
    return (
        f'(df.withColumn("{col}", F.log1p(F.col("{col}"))) '
        f'if "{col}" in df.columns else df)'
    )


def _sqrt_transform(col, _p):
    return (
        f'(df.withColumn("{col}", F.sqrt(F.abs(F.col("{col}")))) '
        f'if "{col}" in df.columns else df)'
    )


def _encode_one_hot(col, _p):
    return f'_encode_one_hot(df, "{col}")'


def _scale_standard(col, _p):
    return f'_scale_standard(df, "{col}")'


def _feature_select(col, _p):
    return f'df.drop("{col}")'


def _derived_ratio(col, p):
    num = p.get("numerator", "")
    den = p.get("denominator", "")
    return (
        f'(df.withColumn("{col}", '
        f'F.col("{num}") / F.when(F.col("{den}") != 0, F.col("{den}")).otherwise(F.lit(None))) '
        f'if all(c in df.columns for c in ["{num}", "{den}"]) else df)'
    )


def _derived_interaction(col, p):
    features = p.get("features", [])
    col_a = features[0] if len(features) > 0 else p.get("col_a", "")
    col_b = features[1] if len(features) > 1 else p.get("col_b", "")
    return (
        f'(df.withColumn("{col}", F.col("{col_a}") * F.col("{col_b}")) '
        f'if all(c in df.columns for c in ["{col_a}", "{col_b}"]) else df)'
    )


def _derived_composite(col, p):
    columns = p.get("columns", [])
    if not columns:
        raise ValueError(f"Composite derived column '{col}' requires non-empty 'columns' parameter")
    expr_parts = " + ".join(f'F.col("{c}")' for c in columns)
    cols_repr = "[" + ", ".join(f'"{c}"' for c in columns) + "]"
    return (
        f'(df.withColumn("{col}", ({expr_parts}) / {len(columns)}) '
        f'if all(c in df.columns for c in {cols_repr}) else df)'
    )


def _segment_aware_cap(col, p):
    n_segments = p.get("n_segments", 2)
    return (
        f'(_segment_aware_cap(df, "{col}", n_segments={n_segments}) '
        f'if "{col}" in df.columns else df)'
    )


def _zero_inflation_handling(col, _p):
    return (
        f'(df.withColumn("{col}_is_zero", F.when(F.col("{col}") == 0, 1).otherwise(0))'
        f'.withColumn("{col}_log", F.when(F.col("{col}") > 0, F.log1p(F.col("{col}"))).otherwise(0)) '
        f'if "{col}" in df.columns else df)'
    )


def _cap_then_log(col, _p):
    return f'(_cap_then_log(df, "{col}") if "{col}" in df.columns else df)'


def _type_cast(col, p):
    dtype = p.get("dtype", "double")
    spark_type = {"float": "double", "int": "int", "string": "string"}.get(dtype, dtype)
    return (
        f'(df.withColumn("{col}", F.col("{col}").cast("{spark_type}")) '
        f'if "{col}" in df.columns else df)'
    )


def _yeo_johnson(col, _p):
    return (
        f'(df.withColumn("{col}", F.log1p(F.abs(F.col("{col}")))) '
        f'if "{col}" in df.columns else df)'
    )


def _dispatch_encode(col, p):
    method = p.get("method", "one_hot")
    if method in ("one_hot", "onehot"):
        return _encode_one_hot(col, p)
    return f'_label_encode(df, "{col}")'


def _dispatch_scale(col, p):
    method = p.get("method", "standard")
    if method == "minmax":
        return f'_scale_minmax(df, "{col}")'
    return _scale_standard(col, p)


def _filter_step(col, p):
    condition = p.get("condition", "non_negative")
    if condition == "non_negative":
        expr = f'df.filter(F.col("{col}") >= 0)'
    elif condition == "range":
        min_val = p.get("min_value", 0)
        max_val = p.get("max_value", 1000000)
        expr = f'df.filter(F.col("{col}").between({min_val}, {max_val}))'
    elif condition == "valid_values":
        valid = p.get("valid_values", [])
        expr = f'df.filter(F.col("{col}").isin({valid}))'
    else:
        expr = f'df.filter(F.col("{col}").isNotNull())'
    return f'({expr} if "{col}" in df.columns else df)'


def _resolve_derived_sources(p, col, *, prefix=None, suffix=None, infix=None):
    """Resolve source-column name(s) for a silver-derived rec.

    Prefers explicit ``source_columns`` (set by FW-2's hydration when
    ``add_silver_derived(..., source_columns=[...])`` was called). Falls
    back to the deterministic column-name pattern emitted by
    `stages/profiling/temporal_analyzer.py`:
      - prefix=`days_since_`     -> `days_since_<src>`     -> [<src>]
      - prefix=`tenure_from_`    -> `tenure_from_<src>`    -> [<src>]
      - suffix=`_month_sin_cos`  -> `<src>_month_sin_cos`  -> [<src>]
      - suffix=`_is_weekend`     -> `<src>_is_weekend`     -> [<src>]
      - infix=`_and_`            -> `days_between_<a>_and_<b>` -> [<a>, <b>]
    Returns possibly-empty list; caller emits a no-op rendering when empty.
    """
    explicit = p.get("source_columns") or []
    if explicit:
        return list(explicit)
    if prefix and col.startswith(prefix):
        return [col[len(prefix):]]
    if suffix and col.endswith(suffix):
        return [col[:-len(suffix)]]
    if infix and infix in col:
        rest = col[len("days_between_"):] if col.startswith("days_between_") else col
        a, b = rest.split(infix, 1)
        return [a, b]
    return []


def _derived_recency(col, p):
    sources = _resolve_derived_sources(p, col, prefix="days_since_")
    if not sources:
        return f'df  # silver_derived recency {col!r}: source unresolvable'
    src = sources[0]
    # `try_to_date` returns NULL on non-date inputs (e.g. a `cohort_quarter`
    # int cast to STRING). With strict `to_date`, such recs would crash the
    # stage with CAST_INVALID_INPUT at write time.
    return (
        f'(df.withColumn("{col}", '
        f'F.when(F.col("{src}").isNotNull() & F.col("as_of_date").isNotNull(), '
        f'F.datediff(F.col("as_of_date"), F.try_to_date(F.col("{src}"))).cast("double"))'
        f'.otherwise(F.lit(None))) '
        f'if "{src}" in df.columns else df)'
    )


def _derived_duration(col, p):
    sources = _resolve_derived_sources(p, col, infix="_and_")
    if len(sources) < 2:
        return f'df  # silver_derived duration {col!r}: source unresolvable'
    a, b = sources[0], sources[1]
    # `try_to_date` for both endpoints — see `_derived_recency`.
    return (
        f'(df.withColumn("{col}", '
        f'F.when(F.col("{a}").isNotNull() & F.col("{b}").isNotNull(), '
        f'F.datediff(F.try_to_date(F.col("{b}")), F.try_to_date(F.col("{a}"))).cast("double"))'
        f'.otherwise(F.lit(None))) '
        f'if all(c in df.columns for c in ["{a}", "{b}"]) else df)'
    )


def _derived_cyclical(col, p):
    """Emits two columns: ``<src>_month_sin`` and ``<src>_month_cos``.

    The recommendation's ``target_column`` is the umbrella ``<src>_month_sin_cos``
    name; downstream feature selection lists the same umbrella name and the
    silver_derived parity gate greps for it. The actual columns persisted
    to silver carry the suffix variants, so this handler emits BOTH and the
    umbrella name is recorded in the apply_derived_columns provenance comment
    above the call (see ``spark_provenance_block``).
    """
    sources = _resolve_derived_sources(p, col, suffix="_month_sin_cos")
    if not sources:
        return f'df  # silver_derived cyclical {col!r}: source unresolvable'
    src = sources[0]
    # `try_to_date` so a non-date STRING (e.g. cohort_quarter cast to "1")
    # yields NULL month -> NULL cyclical, instead of crashing the stage.
    return (
        f'(df.withColumn("{src}_month_sin", '
        f'F.when(F.col("{src}").isNotNull(), '
        f'F.sin(2 * 3.141592653589793 * F.month(F.try_to_date(F.col("{src}"))) / 12)).otherwise(F.lit(None)))'
        f'.withColumn("{src}_month_cos", '
        f'F.when(F.col("{src}").isNotNull(), '
        f'F.cos(2 * 3.141592653589793 * F.month(F.try_to_date(F.col("{src}"))) / 12)).otherwise(F.lit(None))) '
        f'if "{src}" in df.columns else df)'
    )


def _derived_tenure(col, p):
    sources = _resolve_derived_sources(p, col, prefix="tenure_from_")
    if not sources:
        return f'df  # silver_derived tenure {col!r}: source unresolvable'
    src = sources[0]
    # `try_to_date` — see `_derived_recency`.
    return (
        f'(df.withColumn("{col}", '
        f'F.when(F.col("{src}").isNotNull() & F.col("as_of_date").isNotNull(), '
        f'(F.datediff(F.col("as_of_date"), F.try_to_date(F.col("{src}"))) / 365.25).cast("double"))'
        f'.otherwise(F.lit(None))) '
        f'if "{src}" in df.columns else df)'
    )


def _derived_extraction(col, p):
    """``<src>_is_weekend`` — produced by the upstream landing template's
    ``derive_datetime_features`` for every ``datetime_derivation_sources``
    entry. The runtime check ``"<col>" in df.columns`` makes silver a
    passthrough when landing already emitted it; otherwise silver re-derives
    from the raw source so the column is always present at silver_merged.
    """
    sources = _resolve_derived_sources(p, col, suffix="_is_weekend")
    if not sources:
        return f'df  # silver_derived extraction {col!r}: source unresolvable'
    src = sources[0]
    # Three-way fall-through: passthrough if `{col}` is already present
    # (landing emitted it), derive from `{src}` when present, otherwise no-op
    # so a missing source from a phantom rec doesn't crash the stage.
    return (
        f'(df if "{col}" in df.columns else '
        f'(df.withColumn("{col}", '
        f'F.when(F.col("{src}").isNotNull(), '
        f'F.when(F.dayofweek(F.try_to_date(F.col("{src}"))).isin(1, 7), 1.0).otherwise(0.0))'
        f'.otherwise(F.lit(None))) '
        f'if "{src}" in df.columns else df))'
    )


def _dispatch_derived(col, p):
    action = p.get("action", "ratio")
    if action == "ratio":
        return _derived_ratio(col, p)
    if action == "interaction":
        return _derived_interaction(col, p)
    if action == "composite":
        return _derived_composite(col, p)
    if action == "recency":
        return _derived_recency(col, p)
    if action == "duration":
        return _derived_duration(col, p)
    if action == "cyclical":
        return _derived_cyclical(col, p)
    if action == "tenure":
        return _derived_tenure(col, p)
    if action == "extraction":
        return _derived_extraction(col, p)
    return _derived_ratio(col, p)


_SPARK_REGISTRY = {
    PipelineTransformationType.IMPUTE_NULL: _impute_null,
    PipelineTransformationType.CAP_OUTLIER: _cap_outlier,
    PipelineTransformationType.TYPE_CAST: _type_cast,
    PipelineTransformationType.DROP_COLUMN: _drop_column,
    PipelineTransformationType.WINSORIZE: _winsorize,
    PipelineTransformationType.SEGMENT_AWARE_CAP: _segment_aware_cap,
    PipelineTransformationType.LOG_TRANSFORM: _log_transform,
    PipelineTransformationType.SQRT_TRANSFORM: _sqrt_transform,
    PipelineTransformationType.YEO_JOHNSON: _yeo_johnson,
    PipelineTransformationType.ZERO_INFLATION_HANDLING: _zero_inflation_handling,
    PipelineTransformationType.CAP_THEN_LOG: _cap_then_log,
    PipelineTransformationType.ENCODE: _dispatch_encode,
    PipelineTransformationType.SCALE: _dispatch_scale,
    PipelineTransformationType.FEATURE_SELECT: _feature_select,
    PipelineTransformationType.DERIVED_COLUMN: _dispatch_derived,
    PipelineTransformationType.FILTER: _filter_step,
}


def spark_provenance_block(steps) -> str:
    seen = set()
    entries = []
    for step in steps:
        key = provenance_key(step)
        if not key or key in seen:
            continue
        seen.add(key)
        notebook = step.source_notebook or DEFAULT_NOTEBOOK_MAP.get(step.type)
        if not notebook:
            continue
        title = _notebook_title(notebook)
        section = SECTION_MAP.get(step.type, "")
        if section:
            entries.append(f"{title} > {section}")
        else:
            entries.append(title)
    if not entries:
        return ""
    body = "\n    ".join(f"Source: {e}" for e in entries)
    return f'    """\n    {body}\n    """'


DATABRICKS_TEMPLATES = {
    "databricks_config.py.j2": """# Databricks notebook source
# MAGIC %md
# MAGIC # Pipeline Configuration: {{ config.name }}

# COMMAND ----------

PIPELINE_NAME = "{{ config.name }}"
COMPOSITE_NAME = "{{ config.composite_name or config.name }}"
TARGET_COLUMN = "{{ config.target_column }}"
# FW-12: codegen-time target_label_map ({value: 0|1}) for silver merge to
# apply during the target write. Empty dict means no encoding (numeric
# target). Sourced from `RecommendationRegistry.gold.target_label_map`.
TARGET_LABEL_MAP = {{ config.gold.target_label_map | tojson if config.gold and config.gold.target_label_map else '{}' }}
TIMESTAMP_COLUMN = "event_timestamp"
FIT_MODE = {{ 'True' if config.fit_mode else 'False' }}
RECOMMENDATIONS_HASH = {{ '"%s"' % config.recommendations_hash if config.recommendations_hash else 'None' }}
ENTITY_KEY = "{{ config.feast.entity_key if config.feast else 'entity_id' }}"
{% if config.silver.holdout_entity_ids %}
HOLDOUT_ENTITY_IDS = {{ config.silver.holdout_entity_ids }}
{% else %}
HOLDOUT_ENTITY_IDS = None
{% endif %}

CATALOG = "{{ catalog }}"
SCHEMA = "{{ schema }}"

def table_name(name: str) -> str:
    return f"{CATALOG}.{SCHEMA}.{name}"

def bronze_table(source_name: str) -> str:
    return table_name(f"bronze_entity_{source_name}")

def silver_table() -> str:
    return table_name(f"silver_featureset_{COMPOSITE_NAME}")

def gold_table() -> str:
    return table_name(f"gold_features_{COMPOSITE_NAME}")

def landing_table(source_name: str) -> str:
    return table_name(f"landing_{source_name}")

_UC_TABLE_FILE_SUFFIXES = (".csv", ".parquet", ".json", ".orc", ".avro", ".delta", ".txt")

def _is_uc_table_reference(path: str) -> bool:
    if not path:
        return False
    if "/" in path or "\\\\" in path or ":" in path:
        return False
    if "." not in path:
        return False
    return not path.lower().endswith(_UC_TABLE_FILE_SUFFIXES)

def read_raw_source(path: str, fmt: str):
    if isinstance(path, str) and path.startswith("global_temp."):
        raise RuntimeError(
            f"read_raw_source: refusing to read session-scoped temp view {path!r}. "
            "Spark global_temp views vanish at exploration-kernel exit, so this "
            "generated landing notebook would fail on a fresh dbutils.notebook.run "
            "session. Root cause: NB00 build_dataset_registry recorded a "
            "post-user_code-mutation path. Fix: ensure cell f1eb641c stashed "
            "_namespace.original_datasets before any register_temp_view call, "
            "and re-run NB00->NB10. If the upstream needs filtering, declare "
            "registry.add_landing_filter(dataset=..., predicate=...) so the "
            "generated landing replays the filter from the persistent source."
        )
    if _is_uc_table_reference(path):
        return spark.read.table(path)
    if fmt == "csv":
        return spark.read.option("header", "true").option("inferSchema", "true").csv(path)
    return spark.read.format(fmt).load(path)

SOURCES = {
{% for source in config.sources %}
    "{{ source.name }}": {
        "path": "{{ source.raw_source_path or source.path }}",
        "format": "{{ source.format }}",
        "entity_key": "{{ source.entity_key }}",
{% if source.time_column %}
        "time_column": "{{ source.time_column }}",
{% endif %}
        "is_event_level": {{ source.is_event_level }},
    },
{% endfor %}
}

RAW_SOURCES = {
{% for name, landing in config.landing.items() %}
    "{{ name }}": {
        "path": "{{ landing.raw_source_path }}",
        "format": "{{ landing.raw_source_format }}",
        "entity_key": "{{ landing.entity_column }}",
        "time_column": "{{ landing.time_column }}",
    },
{% endfor %}
}
""",
    "databricks_bronze.py.j2": """# Databricks notebook source
# MAGIC %md
# MAGIC # Bronze: {{ source }} (entity)

# COMMAND ----------

from pyspark.sql import functions as F

# COMMAND ----------

# MAGIC %run ../config

# COMMAND ----------

SOURCE_NAME = "{{ source }}"

# FW-19: entity-level bronze reads from the landing UC table, not the raw
# upstream. Landing applies datetime derivation, lifecycle enrichment, and
# filters; bronze recs (null-handling, outlier-handling) target the post-
# landing schema, so reading raw here causes [UNRESOLVED_COLUMN] crashes
# whenever a bronze step references a derived column. Mirrors the contract
# `bronze_event` already uses.
def load_source():
    return spark.table(landing_table(SOURCE_NAME))

{% set groups = group_steps(config.transformations) %}
def apply_transformations(df):
{%- for func_name, steps in groups %}
    df = {{ func_name }}(df)
{%- endfor %}
    return df

{% for func_name, steps in groups %}
def {{ func_name }}(df):
{%- set _prov = spark_provenance_block(steps) %}
{%- if _prov %}
{{ _prov }}
{%- endif %}
{%- for t in steps %}
    # {{ t.rationale }}
    df = {{ render_spark_step_call(t) }}
{%- endfor %}
    return df
{% endfor %}

{%- if config.lifecycle %}
{%- if config.lifecycle.include_recency_bucket %}

def add_recency_tenure(df, raw_df):
    \"\"\"Source: Data Discovery > Recency Analysis\"\"\"
    entity_col = "{{ config.entity_column or config.source.entity_key }}"
    time_col = "{{ config.time_column or config.source.time_column }}"
    reference_date = raw_df.agg(F.max(time_col)).collect()[0][0]
    entity_stats = raw_df.groupBy(entity_col).agg(
        F.min(time_col).alias("first_seen"),
        F.max(time_col).alias("last_seen"),
    )
    entity_stats = entity_stats.withColumn(
        "days_since_last", F.datediff(F.lit(reference_date), F.col("last_seen"))
    ).withColumn(
        "days_since_first", F.datediff(F.lit(reference_date), F.col("first_seen"))
    )
    df = df.join(entity_stats.select(entity_col, "days_since_last", "days_since_first"), on=entity_col, how="left")
    return df

def add_recency_buckets(df):
    \"\"\"Source: Data Discovery > Recency Analysis\"\"\"
{%- set edges = config.lifecycle.recency_bucket_edges %}
{%- set labels = config.lifecycle.recency_bucket_labels %}
    df = df.withColumn("recency_bucket", F.when(F.col("days_since_last") <= {{ edges[1] }}, "{{ labels[0] }}")
{%- for i in range(2, edges | length) %}
        .when(F.col("days_since_last") <= {{ edges[i] }}, "{{ labels[i - 1] }}")
{%- endfor %}
        .otherwise("{{ labels[-1] }}"))
    return df
{%- endif %}
{%- if config.lifecycle.include_lifecycle_quadrant %}

def add_lifecycle_quadrant(df):
    \"\"\"Source: Data Discovery > Lifecycle Segmentation.

    Parity contract with NB01d `classify_lifecycle_quadrants`: tenure threshold
    is the median of `duration_days = days_since_first - days_since_last` (the
    entity's active span), intensity is `event_count / max(duration_days, 1)`
    (events-per-day rate). Quadrant names must match `LIFECYCLE_LABELS` from
    `time_series_profiler.py`. Using raw `days_since_first` and raw
    `event_count_*` as was done previously produced different medians, shifted
    quadrant boundaries, and silently populated only 2-of-4 categories — the
    FeatureSpec parity gate then rejected the gold table.
    \"\"\"
    if "days_since_first" not in df.columns or "days_since_last" not in df.columns:
        return df
    event_count_cols = sorted(c for c in df.columns if c.startswith("event_count_"))
    if not event_count_cols:
        return df
    event_count_col = (
        "event_count_all_time" if "event_count_all_time" in event_count_cols
        else event_count_cols[-1]
    )
    df = df.withColumn(
        "_lifecycle_duration_days",
        (F.col("days_since_first") - F.col("days_since_last")).cast("double"),
    ).withColumn(
        "_lifecycle_intensity",
        F.col(event_count_col).cast("double")
        / F.greatest(F.col("_lifecycle_duration_days"), F.lit(1.0)),
    )
    _quantile_result = df.approxQuantile(
        ["_lifecycle_duration_days", "_lifecycle_intensity"], [0.5], 0.01,
    )
    tenure_med = _quantile_result[0][0] if _quantile_result[0] else 0.0
    intensity_med = _quantile_result[1][0] if _quantile_result[1] else 0.0
    _long = F.col("_lifecycle_duration_days") >= F.lit(tenure_med)
    _high = F.col("_lifecycle_intensity") >= F.lit(intensity_med)
    df = df.withColumn("lifecycle_quadrant",
        F.when(_long & _high, "steady_loyal_lifecycle")
        .when(_long & ~_high, "occasional_loyal_lifecycle")
        .when(~_long & _high, "intense_brief_lifecycle")
        .otherwise("one_shot_lifecycle"))
    df = df.drop("_lifecycle_duration_days", "_lifecycle_intensity")
    return df
{%- endif %}
{%- if config.lifecycle.include_cyclical_features %}

def add_cyclical_features(df, raw_df):
    \"\"\"Source: Data Discovery > Cyclical Patterns\"\"\"
    entity_col = "{{ config.entity_column or config.source.entity_key }}"
    time_col = "{{ config.time_column or config.source.time_column }}"
    mean_dow = raw_df.groupBy(entity_col).agg(
        F.mean(F.dayofweek(F.col(time_col)).cast("double")).alias("mean_dow")
    )
    df = df.join(mean_dow, on=entity_col, how="left")
    df = df.withColumn("dow_sin", F.sin(2 * 3.141592653589793 * F.col("mean_dow") / 7))
    df = df.withColumn("dow_cos", F.cos(2 * 3.141592653589793 * F.col("mean_dow") / 7))
    df = df.drop("mean_dow")
    return df
{%- endif %}
{%- if config.lifecycle.include_month_cyclical %}

def add_month_quarter_cyclical(df, raw_df):
    \"\"\"Source: Data Discovery > Seasonal Patterns\"\"\"
    entity_col = "{{ config.entity_column or config.source.entity_key }}"
    time_col = "{{ config.time_column or config.source.time_column }}"
    mean_month = raw_df.groupBy(entity_col).agg(
        F.mean(F.month(F.col(time_col)).cast("double")).alias("mean_month")
    )
    df = df.join(mean_month, on=entity_col, how="left")
    df = df.withColumn("month_sin", F.sin(2 * 3.141592653589793 * F.col("mean_month") / 12))
    df = df.withColumn("month_cos", F.cos(2 * 3.141592653589793 * F.col("mean_month") / 12))
{%- if config.lifecycle.include_quarter_cyclical %}
    mean_quarter = raw_df.groupBy(entity_col).agg(
        F.mean(F.quarter(F.col(time_col)).cast("double")).alias("mean_quarter")
    )
    df = df.join(mean_quarter, on=entity_col, how="left")
    df = df.withColumn("quarter_sin", F.sin(2 * 3.141592653589793 * F.col("mean_quarter") / 4))
    df = df.withColumn("quarter_cos", F.cos(2 * 3.141592653589793 * F.col("mean_quarter") / 4))
    df = df.drop("mean_month", "mean_quarter")
{%- else %}
    df = df.drop("mean_month")
{%- endif %}
    return df
{%- endif %}
{%- if config.lifecycle.include_trend_features %}

def add_trend_features(df):
    \"\"\"Source: Data Discovery > Trend Analysis\"\"\"
    import numpy as np
    pdf = df.toPandas()
    window_cols = sorted([c for c in pdf.columns if c.startswith("event_count_") and c != "event_count_all_time"])
    all_time_col = "event_count_all_time" if "event_count_all_time" in pdf.columns else None
    if window_cols and all_time_col:
        pdf["recent_vs_overall_ratio"] = pdf[window_cols[0]] / pdf[all_time_col].replace(0, float("nan"))
    if len(window_cols) >= 2:
        x = np.arange(len(window_cols), dtype=float)
        window_values = pdf[window_cols].values
        slopes = np.array([np.polyfit(x, row, 1)[0] if not np.any(np.isnan(row)) else 0.0 for row in window_values])
        pdf["entity_trend_slope"] = slopes
    df = spark.createDataFrame(pdf)
    return df
{%- endif %}
{%- if config.lifecycle.include_cohort_features %}

def add_cohort_features(df, raw_df):
    \"\"\"Source: Data Discovery > Cohort Analysis\"\"\"
    entity_col = "{{ config.entity_column or config.source.entity_key }}"
    time_col = "{{ config.time_column or config.source.time_column }}"
    first_event = raw_df.groupBy(entity_col).agg(F.min(time_col).alias("first_event"))
    first_event = first_event.withColumn("cohort_year", F.year("first_event"))
    first_event = first_event.withColumn("cohort_quarter", F.quarter("first_event"))
    df = df.join(first_event.select(entity_col, "cohort_year", "cohort_quarter"), on=entity_col, how="left")
    return df
{%- endif %}
{%- if config.lifecycle.momentum_pairs %}

def add_momentum_ratios(df):
    \"\"\"Source: Data Discovery > Engagement Momentum\"\"\"
{%- for pair in config.lifecycle.momentum_pairs %}
    short_col = "event_count_{{ pair.short_window }}"
    long_col = "event_count_{{ pair.long_window }}"
    if short_col in df.columns and long_col in df.columns:
        df = df.withColumn("momentum_{{ pair.short_window }}_{{ pair.long_window }}", F.col(short_col) / F.when(F.col(long_col) != 0, F.col(long_col)).otherwise(F.lit(None)))
{%- endfor %}
    return df
{%- endif %}

def enrich_lifecycle(df):
    \"\"\"Source: Data Discovery > Entity Lifecycle Enrichment\"\"\"
    raw_table = bronze_table("{{ source }}")
    raw_df = spark.table(raw_table)
{%- if config.lifecycle.include_recency_bucket %}
    df = add_recency_tenure(df, raw_df)
    df = add_recency_buckets(df)
{%- endif %}
{%- if config.lifecycle.include_lifecycle_quadrant %}
    df = add_lifecycle_quadrant(df)
{%- endif %}
{%- if config.lifecycle.include_cyclical_features %}
    df = add_cyclical_features(df, raw_df)
{%- endif %}
{%- if config.lifecycle.include_month_cyclical %}
    df = add_month_quarter_cyclical(df, raw_df)
{%- endif %}
{%- if config.lifecycle.include_trend_features %}
    df = add_trend_features(df)
{%- endif %}
{%- if config.lifecycle.include_cohort_features %}
    df = add_cohort_features(df, raw_df)
{%- endif %}
{%- if config.lifecycle.momentum_pairs %}
    df = add_momentum_ratios(df)
{%- endif %}
    return df
{%- endif %}
{%- if config.text_features %}

def compute_text_features_entity(df):
    from customer_retention.stages.profiling.text_processor import TextColumnProcessor, TextProcessingConfig
    pdf = df.toPandas()
{% for tf in config.text_features %}
    if "{{ tf.column }}" in pdf.columns:
        processor = TextColumnProcessor(TextProcessingConfig(embedding_model="{{ tf.embedding_model }}"), registry=None)
        if FIT_MODE:
            pdf, result = processor.process_column(pdf, "{{ tf.column }}", fit=True)
        else:
            pdf, result = processor.process_column(pdf, "{{ tf.column }}", fit=False)
{% endfor %}
    return spark.createDataFrame(pdf)
{% endif %}

# COMMAND ----------

def run_bronze():
    df = load_source()
    df = apply_transformations(df)
{%- if config.lifecycle %}
    df = enrich_lifecycle(df)
{%- endif %}
{%- if config.text_features %}
    df = compute_text_features_entity(df)
{%- endif %}
    output_table = bronze_table(SOURCE_NAME)
    df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(output_table)
    return df

import json as _bronze_json
import traceback as _bronze_tb
try:
    result = run_bronze()
    _summary = f"{result.count():,} rows, {len(result.columns)} columns"
    # Narrow-projection display (see silver template for full rationale): rendering
    # wide DataFrames through Databricks' display widget serialises the entire
    # logical plan to protobuf, which trips Catalyst attribute-id resolution at
    # wide schemas. Project to entity/key cols + row-limit; the table is already
    # committed to Delta above.
    _disp = [c for c in ("entity_id", "ACCOUNT_ID", "as_of_date") if c in result.columns] or list(result.columns[:10])
    display(result.select(*_disp).limit(20))
except Exception as _bronze_exc:
    print("=" * 70, flush=True)
    print(f"[bronze] FATAL: {type(_bronze_exc).__name__}: {_bronze_exc}", flush=True)
    print("=" * 70, flush=True)
    print(_bronze_tb.format_exc(), flush=True)
    dbutils.notebook.exit(_bronze_json.dumps({
        "status": "FAILED",
        "stage": "bronze",
        "error_type": type(_bronze_exc).__name__,
        "error_message": str(_bronze_exc)[:2000],
    }))
    raise
dbutils.notebook.exit(_summary)
""",
    "databricks_bronze_event.py.j2": """# Databricks notebook source
# MAGIC %md
# MAGIC # Bronze Event: {{ source }}

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import NumericType

from customer_retention.core.naming import sanitize_column_token

# COMMAND ----------

# MAGIC %run ../config

# COMMAND ----------

SOURCE_NAME = "{{ source }}"
ENTITY_COLUMN = "{{ config.entity_column }}"
TIME_COLUMN = "{{ config.time_column }}"

def load_source():
    return spark.table(landing_table(SOURCE_NAME))

{% set pre_groups = group_steps(config.pre_shaping) %}
def apply_pre_shaping(df):
{%- for func_name, steps in pre_groups %}
    df = {{ func_name }}(df)
{%- endfor %}
    return df

{% for func_name, steps in pre_groups %}
def {{ func_name }}(df):
{%- set _prov = spark_provenance_block(steps) %}
{%- if _prov %}
{{ _prov }}
{%- endif %}
{%- for t in steps %}
    # {{ t.rationale }}
    df = {{ render_spark_step_call(t) }}
{%- endfor %}
    return df
{% endfor %}

{%- if config.deduplicate %}

def deduplicate(df):
    \"\"\"Source: Source Integrity > Duplicate Detection\"\"\"
{%- if config.deduplicate is not true and config.deduplicate.strategy is defined and config.deduplicate.strategy == "keep_most_complete" %}
    _all_cols = [f.name for f in df.schema.fields if f.name not in (ENTITY_COLUMN, TIME_COLUMN)]
    _null_expr = sum(F.when(F.col(c).isNull(), 1).otherwise(0) for c in _all_cols) if _all_cols else F.lit(0)
    df = df.withColumn("_null_count", _null_expr)
{%- if config.deduplicate.conflict_columns %}
    _partition_cols = {{ config.deduplicate.conflict_columns }}
{%- else %}
    _partition_cols = [ENTITY_COLUMN, TIME_COLUMN]
{%- endif %}
    window = Window.partitionBy(*_partition_cols).orderBy(F.col("_null_count").asc(), F.monotonically_increasing_id())
    df = df.withColumn("_row_num", F.row_number().over(window))
    df = df.filter(F.col("_row_num") == 1).drop("_row_num", "_null_count")
{% elif config.deduplicate is not true and config.deduplicate.conflict_columns is defined and config.deduplicate.conflict_columns %}
    window = Window.partitionBy(*{{ config.deduplicate.conflict_columns }}).orderBy(F.monotonically_increasing_id())
    df = df.withColumn("_row_num", F.row_number().over(window))
    df = df.filter(F.col("_row_num") == 1).drop("_row_num")
{% else %}
    window = Window.partitionBy(ENTITY_COLUMN, TIME_COLUMN).orderBy(F.monotonically_increasing_id())
    df = df.withColumn("_row_num", F.row_number().over(window))
    df = df.filter(F.col("_row_num") == 1).drop("_row_num")
{% endif %}
    return df
{% endif %}

{%- if config.datetime_derivation %}

DATETIME_DERIVATION_SOURCES = {{ config.datetime_derivation.source_columns }}
MASK_FUTURE_COLUMNS = {{ config.datetime_derivation.mask_future_columns }}

def derive_datetime_features(df):
    \"\"\"Source: Data Discovery > Datetime Feature Derivation\"\"\"
    ref_col = "{{ config.datetime_derivation.reference_column }}"
    mask_set = set(MASK_FUTURE_COLUMNS)
    for col in DATETIME_DERIVATION_SOURCES:
        if col not in [f.name for f in df.schema.fields]:
            continue
        ts_col = F.to_timestamp(F.col(col))
        ref_ts = F.to_timestamp(F.col(ref_col))
        delta_hours = (F.unix_timestamp(ts_col) - F.unix_timestamp(ref_ts)) / 3600.0
        hour_val = F.hour(ts_col).cast("double")
        dow_val = (F.dayofweek(ts_col) - 1).cast("double")
        is_weekend_val = F.when(F.dayofweek(ts_col).isin(1, 7), 1.0).otherwise(0.0)
        if col in mask_set:
            future_mask = ts_col > ref_ts
            df = df.withColumn(f"{col}_delta_hours", F.when(future_mask, None).otherwise(delta_hours))
            df = df.withColumn(f"{col}_hour", F.when(future_mask, None).otherwise(hour_val))
            df = df.withColumn(f"{col}_dow", F.when(future_mask, None).otherwise(dow_val))
            df = df.withColumn(f"{col}_is_weekend", F.when(future_mask, None).otherwise(is_weekend_val))
        else:
            df = df.withColumn(f"{col}_delta_hours", delta_hours)
            df = df.withColumn(f"{col}_hour", hour_val)
            df = df.withColumn(f"{col}_dow", dow_val)
            df = df.withColumn(f"{col}_is_weekend", is_weekend_val)
    return df
{%- endif %}
{%- if config.aggregation %}
def _window_to_days(window_str):
    if window_str.endswith("d"):
        return int(window_str[:-1])
    if window_str.endswith("h"):
        return max(1, int(window_str[:-1]) // 24)
    return int(window_str)

CATEGORICAL_COLUMNS = {{ config.aggregation.categorical_columns }}
BINARY_COLUMNS = {{ config.aggregation.binary_columns }}
COLUMN_BLOCKED_FUNCS = {{ config.aggregation.column_blocked_funcs }}
CATEGORICAL_VALUE_COUNTS = {{ config.aggregation.categorical_value_counts }}
{%- if config.per_grid_date_mode %}
GRID_DATES = {{ grid_dates }}
VALUE_COUNTS_COLUMNS = {{ config.value_counts_columns | list }}
AGGREGATION_WINDOWS = {{ config.aggregation.windows }}


def _spark_cumulative_at(spine_df, running_df, anchor_col):
    # Backward-asof "cumulative count at anchor" via a single Window pass.
    #
    # spine_df schema:   (ENTITY_COLUMN, anchor_col)
    # running_df schema: (ENTITY_COLUMN, _event_ts, _running)
    #
    # Both sides are unioned into one stream tagged with _is_anchor. Within
    # each entity, rows are ordered by (_ts asc, _is_anchor asc) so events
    # at the same instant as an anchor sort BEFORE the anchor — backward-asof
    # semantics. F.max("_running") OVER (UNBOUNDED PRECEDING, CURRENT ROW)
    # then propagates the latest cumulative count into each anchor row in a
    # single sort+scan, with no per-entity Cartesian materialization.
    spine_marker = spine_df.select(
        F.col(ENTITY_COLUMN),
        F.col(anchor_col).alias("_ts"),
        F.lit(None).cast("long").alias("_running"),
        F.lit(True).alias("_is_anchor"),
    )
    running_marker = running_df.select(
        F.col(ENTITY_COLUMN),
        F.col("_event_ts").alias("_ts"),
        F.col("_running"),
        F.lit(False).alias("_is_anchor"),
    )
    unified = spine_marker.unionByName(running_marker)
    w = (
        Window.partitionBy(ENTITY_COLUMN)
        .orderBy(F.col("_ts").asc(), F.col("_is_anchor").asc())
        .rowsBetween(Window.unboundedPreceding, Window.currentRow)
    )
    return (
        unified.withColumn("_cum_raw", F.max("_running").over(w))
        .filter(F.col("_is_anchor"))
        .select(
            F.col(ENTITY_COLUMN),
            F.col("_ts").alias(anchor_col),
            F.coalesce(F.col("_cum_raw"), F.lit(0)).cast("long").alias("_cum"),
        )
    )


def apply_event_aggregation_per_grid_date(df):
    # Per-grid-date count aggregation using Spark Window functions.
    # For each (entity, value-of-VALUE_COUNTS_COLUMN, window) we compute the
    # cumulative event count at each grid date and at (grid_date - window).
    # The window count is the difference. Output has one row per
    # (entity, as_of_date) which downstream temporal_merger handles via equi-join.
    df = df.withColumn(TIME_COLUMN, F.to_timestamp(F.col(TIME_COLUMN)))

    grid_rows = [(d,) for d in GRID_DATES]
    grid_df = spark.createDataFrame(grid_rows, ["as_of_date"]).withColumn(
        "as_of_date", F.to_timestamp("as_of_date")
    )
    entities = df.select(ENTITY_COLUMN).distinct()
    spine = entities.crossJoin(F.broadcast(grid_df))

    output = spine
    for vc_col in VALUE_COUNTS_COLUMNS:
        if vc_col not in [f.name for f in df.schema.fields]:
            continue
        distinct_values = sorted(
            r[0]
            for r in df.filter(F.col(vc_col).isNotNull())
            .select(vc_col)
            .distinct()
            .collect()
        )
        for value in distinct_values:
            sub = df.filter(F.col(vc_col) == value).select(
                ENTITY_COLUMN, F.col(TIME_COLUMN).alias("_event_ts")
            )
            w = Window.partitionBy(ENTITY_COLUMN).orderBy(F.col("_event_ts").cast("long"))
            running = sub.withColumn("_running", F.row_number().over(w))

            cum_at_G = _spark_cumulative_at(spine, running, "as_of_date")

            for window_str in AGGREGATION_WINDOWS:
                col_name = f"{vc_col}_{sanitize_column_token(value)}_count_{window_str}"
                if window_str == "all_time":
                    contribution = cum_at_G.withColumnRenamed("_cum", col_name)
                else:
                    days = _window_to_days(window_str)
                    spine_shifted = spine.withColumn(
                        "as_of_date_shifted",
                        F.expr(f"as_of_date - INTERVAL {days} DAYS"),
                    )
                    cum_at_minus = _spark_cumulative_at(
                        spine_shifted, running, "as_of_date_shifted"
                    )
                    cum_g_renamed = cum_at_G.withColumnRenamed("_cum", "_cum_g")
                    cum_m_renamed = cum_at_minus.select(
                        ENTITY_COLUMN, "as_of_date_shifted", F.col("_cum").alias("_cum_m")
                    )
                    # Re-attach the unshifted as_of_date by joining the shifted
                    # spine back. We can simply align on row order via join.
                    pair = (
                        cum_g_renamed.join(
                            spine.select(
                                ENTITY_COLUMN,
                                "as_of_date",
                                F.expr(
                                    f"as_of_date - INTERVAL {days} DAYS"
                                ).alias("as_of_date_shifted"),
                            ),
                            on=[ENTITY_COLUMN, "as_of_date"],
                            how="inner",
                        )
                        .join(
                            cum_m_renamed,
                            on=[ENTITY_COLUMN, "as_of_date_shifted"],
                            how="left",
                        )
                        .withColumn("_cum_m", F.coalesce(F.col("_cum_m"), F.lit(0)))
                        .withColumn(col_name, F.col("_cum_g") - F.col("_cum_m"))
                        .select(ENTITY_COLUMN, "as_of_date", col_name)
                    )
                    contribution = pair

                output = output.join(
                    contribution, on=[ENTITY_COLUMN, "as_of_date"], how="left"
                ).withColumn(
                    col_name, F.coalesce(F.col(col_name), F.lit(0)).cast("long")
                )

    return output
{%- endif %}

def _get_numeric_columns(df, value_columns):
    numeric_cols = set()
    for field in df.schema.fields:
        if isinstance(field.dataType, NumericType):
            numeric_cols.add(field.name)
    return [c for c in value_columns if c in numeric_cols]

def apply_event_aggregation(df):
    \"\"\"Source: Event Aggregation > Time-Window Analysis\"\"\"
    # Narrow BINARY_COLUMNS to numerically-coercible Spark dtypes. F.mean / F.sum
    # / F.max over a STRING column raise [CAST_INVALID_INPUT] under UC ANSI mode
    # (where spark.conf.set("spark.sql.ansi.enabled","false") is admin-locked
    # and silently no-ops). The renderer's existence guard checks names, not
    # dtypes, so a binary-by-semantics column stored as STRING (e.g. "Yes"/"No")
    # would otherwise reach the aggregator and fail.
    global BINARY_COLUMNS
    _numeric_dtype_prefixes = (
        "int", "bigint", "tinyint", "smallint",
        "double", "float", "long", "short", "byte",
        "boolean", "decimal",
    )
    _df_dtypes = dict(df.dtypes)
    def _is_numeric_dtype(_col):
        _t = (_df_dtypes.get(_col) or "").lower()
        return any(_t.startswith(_p) for _p in _numeric_dtype_prefixes)
    BINARY_COLUMNS = [c for c in BINARY_COLUMNS if _is_numeric_dtype(c)]
    reference_date = df.agg(F.max(TIME_COLUMN)).collect()[0][0]
    numeric_columns = _get_numeric_columns(df, {{ config.aggregation.value_columns }})
    _schema_cols = {f.name for f in df.schema.fields}
    for _cvc_col in CATEGORICAL_VALUE_COUNTS:
        if _cvc_col not in _schema_cols:
            raise ValueError(
                f"categorical_value_counts declares column {_cvc_col!r} but it is not "
                f"present in the dataframe; fix upstream config or enrichment"
            )
    results = []
{% for window in config.aggregation.windows %}
{%- if window == "all_time" %}
    window_df = df
{%- else %}
    window_df = df.filter(
        F.col(TIME_COLUMN) >= F.date_sub(F.lit(reference_date), _window_to_days("{{ window }}"))
    )
{%- endif %}
    agg_exprs = [F.count("*").alias("event_count_{{ window }}")]
    for col in numeric_columns:
        _blocked = COLUMN_BLOCKED_FUNCS.get(col, [])
{%- for agg_func in config.aggregation.agg_funcs %}
        if "{{ agg_func }}" not in _blocked:
            agg_exprs.append(F.{{ agg_func }}(col).alias(f"{col}_{{ agg_func }}_{{ window }}"))
{%- endfor %}
    for col in CATEGORICAL_COLUMNS:
        if col in [f.name for f in window_df.schema.fields]:
            _blocked = COLUMN_BLOCKED_FUNCS.get(col, [])
            if "nunique" not in _blocked:
                agg_exprs.append(F.countDistinct(col).alias(f"{col}_nunique_{{ window }}"))
            if "mode" not in _blocked:
                agg_exprs.append(F.first(col).alias(f"{col}_mode_{{ window }}"))
    for col in BINARY_COLUMNS:
        if col in [f.name for f in window_df.schema.fields]:
            _blocked = COLUMN_BLOCKED_FUNCS.get(col, [])
            if "rate" not in _blocked:
                agg_exprs.append(F.mean(col).alias(f"{col}_rate_{{ window }}"))
            if "count" not in _blocked:
                agg_exprs.append(F.sum(col).alias(f"{col}_count_{{ window }}"))
            if "any" not in _blocked:
                agg_exprs.append(F.max(col).alias(f"{col}_any_{{ window }}"))
    for _cvc_col, _cvc_values in CATEGORICAL_VALUE_COUNTS.items():
        for _cvc_value in _cvc_values:
            agg_exprs.append(
                F.sum(F.when(F.col(_cvc_col) == _cvc_value, 1).otherwise(0))
                 .alias(f"{_cvc_col}_{sanitize_column_token(_cvc_value)}_count_{{ window }}")
            )
    window_agg = window_df.groupBy(ENTITY_COLUMN).agg(*agg_exprs)
    results.append(window_agg)
{% endfor %}
    merged = results[0]
    for r in results[1:]:
        merged = merged.join(r, on=ENTITY_COLUMN, how="outer")
    _fill_cols = [c for c in merged.columns if any(c.endswith(s) for s in ("_count", "_sum", "_rate")) or c.startswith("event_count_")]
    if _fill_cols:
        merged = merged.fillna(0, subset=_fill_cols)
    if "feature_timestamp" in [f.name for f in df.schema.fields]:
        ts_agg = df.groupBy(ENTITY_COLUMN).agg(F.max("feature_timestamp").alias("feature_timestamp"))
        merged = merged.join(ts_agg, on=ENTITY_COLUMN, how="left")
    if TARGET_COLUMN in [f.name for f in df.schema.fields]:
        target_agg = df.groupBy(ENTITY_COLUMN).agg(F.first(TARGET_COLUMN, ignorenulls=True).alias(TARGET_COLUMN))
        merged = merged.join(target_agg, on=ENTITY_COLUMN, how="left")
    return merged, reference_date
{% endif %}

{%- if config.temporal_features and config.temporal_features.has_renderable_content() %}

def compute_temporal_features(agg_df, raw_df):
    \"\"\"Source: Temporal Deep Dive > Lag Features\"\"\"
    from customer_retention.stages.profiling.spark_temporal_feature_engineer import SparkTemporalFeatureEngineer
    from customer_retention.stages.profiling.temporal_feature_engineer import TemporalAggregationConfig
    value_cols = {{ config.temporal_features.lag_columns or (config.aggregation.value_columns if config.aggregation else []) }}
    # Narrow value_cols to columns that (a) exist on raw_df AND (b) carry a
    # numeric Spark dtype that can be cast to DOUBLE. The temporal engineer
    # casts each entry to DOUBLE for windowed aggregation; STRING flags like
    # "Yes"/"No" otherwise raise [CAST_INVALID_INPUT] under UC ANSI mode.
    _numeric_dtypes = (
        "double", "float", "int", "bigint", "smallint", "tinyint",
        "long", "short", "byte", "integer",
    )
    _raw_dtypes = dict(raw_df.dtypes)
    value_cols = [
        _c for _c in value_cols
        if _c in raw_df.columns
        and (
            _raw_dtypes.get(_c, "string") in _numeric_dtypes
            or _raw_dtypes.get(_c, "string").startswith("decimal")
        )
    ]
    eng_config = TemporalAggregationConfig(
        lag_window_days={{ config.temporal_features.lag_window_days }},
        num_lags={{ config.temporal_features.num_lags }},
        lag_aggregations={{ config.temporal_features.lag_agg_funcs }},
        compute_velocity={{ 'velocity' in config.temporal_features.feature_groups }},
        compute_acceleration={{ 'acceleration' in config.temporal_features.feature_groups }},
        compute_lifecycle={{ 'lifecycle' in config.temporal_features.feature_groups }},
        compute_recency={{ 'recency' in config.temporal_features.feature_groups }},
        compute_regularity={{ 'regularity' in config.temporal_features.feature_groups }},
        compute_cohort={{ 'cohort_comparison' in config.temporal_features.feature_groups }},
    )
    engineer = SparkTemporalFeatureEngineer(eng_config)
    result = engineer.compute(raw_df, ENTITY_COLUMN, TIME_COLUMN, value_cols)
    temporal_df = result.features_df
    if hasattr(temporal_df, "to_spark"):
        temporal_df = temporal_df.to_spark()
    merge_cols = [c for c in temporal_df.columns if c != ENTITY_COLUMN]
    return agg_df.join(temporal_df.select(ENTITY_COLUMN, *merge_cols), on=ENTITY_COLUMN, how="left")
{% endif %}

{%- if config.text_features %}

def compute_text_features(agg_df, raw_df):
    \"\"\"Source: Temporal Text Deep Dive > Text Embeddings\"\"\"
    from customer_retention.stages.profiling.text_processor import TextColumnProcessor, TextProcessingConfig
    agg_pdf = agg_df.toPandas() if hasattr(agg_df, "toPandas") else agg_df
    raw_pdf = raw_df.toPandas() if hasattr(raw_df, "toPandas") else raw_df
{% for tf in config.text_features %}
    if "{{ tf.column }}" in raw_pdf.columns:
        processor = TextColumnProcessor(TextProcessingConfig(embedding_model="{{ tf.embedding_model }}"), registry=None)
        text_data = raw_pdf.groupby(ENTITY_COLUMN)["{{ tf.column }}"].first().reset_index()
        if FIT_MODE:
            text_data, result = processor.process_column(text_data, "{{ tf.column }}", fit=True)
        else:
            text_data, result = processor.process_column(text_data, "{{ tf.column }}", fit=False)
        component_cols = result.component_columns
        agg_pdf = agg_pdf.merge(text_data[[ENTITY_COLUMN] + component_cols], on=ENTITY_COLUMN, how="left")
{% endfor %}
    return spark.createDataFrame(agg_pdf)
{% endif %}

# COMMAND ----------

def run_bronze_event():
    raw_df = load_source()
    df = apply_pre_shaping(raw_df)
{%- if config.deduplicate %}
    df = deduplicate(df)
{%- endif %}
{%- if config.datetime_derivation %}
    df = derive_datetime_features(df)
{%- endif %}
{%- if config.aggregation %}
{%- if config.per_grid_date_mode %}
    agg_df = apply_event_aggregation_per_grid_date(df)
    reference_date = None
{%- else %}
    agg_df, reference_date = apply_event_aggregation(df)
{%- endif %}
{%- if config.temporal_features and config.temporal_features.has_renderable_content() %}
    # Belt-and-suspenders: any unforeseen failure inside compute_temporal_features
    # (empty filtered value_cols, schema drift, etc.) must not block the producer.
    # The aggregation rollups in agg_df are already populated; we keep saveAsTable
    # unconditional so bronze_entity_*_aggregated consumers can still read the
    # producer table. Downstream missing temporal columns are handled at silver.
    try:
        agg_df = compute_temporal_features(agg_df, raw_df)
    except Exception as _temporal_err:
        print(
            "[bronze_event] compute_temporal_features failed for", SOURCE_NAME,
            "->", str(_temporal_err)[:200],
            "; writing without temporal features",
        )
{%- endif %}
{%- if config.text_features %}
    agg_df = compute_text_features(agg_df, raw_df)
{%- endif %}
    output_table = bronze_table("{{ source }}_events")
    agg_df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(output_table)
    from delta.tables import DeltaTable
    _z_cols = [c for c in [ENTITY_COLUMN, "as_of_date"] if c in [f.name for f in agg_df.schema.fields]]
    if _z_cols:
        DeltaTable.forName(spark, output_table).optimize().executeZOrderBy(_z_cols)
    else:
        DeltaTable.forName(spark, output_table).optimize().executeCompaction()
    return agg_df
{%- else %}
    output_table = bronze_table("{{ source }}_events")
    df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(output_table)
    from delta.tables import DeltaTable
    _z_cols = [c for c in [ENTITY_COLUMN, TIME_COLUMN] if c in [f.name for f in df.schema.fields]]
    if _z_cols:
        DeltaTable.forName(spark, output_table).optimize().executeZOrderBy(_z_cols)
    else:
        DeltaTable.forName(spark, output_table).optimize().executeCompaction()
    return df
{%- endif %}

import json as _bronze_json
import traceback as _bronze_tb
try:
    result = run_bronze_event()
    _summary = f"{result.count():,} rows, {len(result.columns)} columns"
    # Narrow-projection display (see silver template for full rationale).
    _disp = [c for c in ("entity_id", "ACCOUNT_ID", "as_of_date", "feature_timestamp") if c in result.columns] or list(result.columns[:10])
    display(result.select(*_disp).limit(20))
except Exception as _bronze_exc:
    print("=" * 70, flush=True)
    print(f"[bronze_event] FATAL: {type(_bronze_exc).__name__}: {_bronze_exc}", flush=True)
    print("=" * 70, flush=True)
    print(_bronze_tb.format_exc(), flush=True)
    dbutils.notebook.exit(_bronze_json.dumps({
        "status": "FAILED",
        "stage": "bronze_event",
        "error_type": type(_bronze_exc).__name__,
        "error_message": str(_bronze_exc)[:2000],
    }))
    raise
dbutils.notebook.exit(_summary)
""",
    "databricks_bronze_entity.py.j2": """# Databricks notebook source
# MAGIC %md
# MAGIC # Bronze Entity: {{ source }} (aggregated)

# COMMAND ----------

from pyspark.sql import functions as F

# COMMAND ----------

# MAGIC %run ../config

# COMMAND ----------

SOURCE_NAME = "{{ source }}"
ENTITY_COLUMN = "{{ config.entity_column }}"

def load_aggregated():
    return spark.table(bronze_table("{{ bronze_input_name }}_events"))

{% set post_groups = group_steps(config.post_shaping) %}
{% if config.post_shaping %}
def apply_post_shaping(df):
    # Each per-column group is shallow on its own. A `localCheckpoint`
    # between groups is unreliable on Spark Connect so we don't add one
    # here; bronze typically stays well under the column count where
    # the wide-schema attribute-resolver bug surfaces.
{%- for func_name, steps in post_groups %}
    df = {{ func_name }}(df)
{%- endfor %}
    return df

{% for func_name, steps in post_groups %}
def {{ func_name }}(df):
{%- set _prov = spark_provenance_block(steps) %}
{%- if _prov %}
{{ _prov }}
{%- endif %}
{%- for t in steps %}
    # {{ t.rationale }}
    df = {{ render_spark_step_call(t) }}
{%- endfor %}
    return df
{% endfor %}
{% endif %}

{%- if config.lifecycle %}
{%- if config.lifecycle.include_recency_bucket %}

def add_recency_tenure(df):
    \"\"\"Source: Data Discovery > Recency Analysis\"\"\"
    raw_df = spark.table(landing_table("{{ raw_source }}"))
    time_col = "{{ config.time_column }}"
    reference_date = raw_df.agg(F.max(time_col)).collect()[0][0]
    entity_stats = raw_df.groupBy(ENTITY_COLUMN).agg(
        F.min(time_col).alias("first_seen"),
        F.max(time_col).alias("last_seen"),
    )
    entity_stats = entity_stats.withColumn(
        "days_since_last", F.datediff(F.lit(reference_date), F.col("last_seen"))
    ).withColumn(
        "days_since_first", F.datediff(F.lit(reference_date), F.col("first_seen"))
    )
    df = df.join(entity_stats.select(ENTITY_COLUMN, "days_since_last", "days_since_first"), on=ENTITY_COLUMN, how="left")
    return df

def add_recency_buckets(df):
    \"\"\"Source: Data Discovery > Recency Analysis\"\"\"
{%- set edges = config.lifecycle.recency_bucket_edges %}
{%- set labels = config.lifecycle.recency_bucket_labels %}
    df = df.withColumn("recency_bucket", F.when(F.col("days_since_last") <= {{ edges[1] }}, "{{ labels[0] }}")
{%- for i in range(2, edges | length) %}
        .when(F.col("days_since_last") <= {{ edges[i] }}, "{{ labels[i - 1] }}")
{%- endfor %}
        .otherwise("{{ labels[-1] }}"))
    return df
{%- endif %}
{%- if config.lifecycle.include_lifecycle_quadrant %}

def add_lifecycle_quadrant(df):
    \"\"\"Source: Data Discovery > Lifecycle Segmentation.

    Parity contract with NB01d `classify_lifecycle_quadrants`: tenure threshold
    is the median of `duration_days = days_since_first - days_since_last` (the
    entity's active span), intensity is `event_count / max(duration_days, 1)`
    (events-per-day rate). Prefers `event_count_all_time` as the count source,
    falling back to the widest windowed count. Uses a batched `approxQuantile`
    (one Spark job, both medians) so the step stays cheap on large gold tables.
    \"\"\"
    if "days_since_first" not in df.columns or "days_since_last" not in df.columns:
        return df
    event_count_cols = sorted(c for c in df.columns if c.startswith("event_count_"))
    if not event_count_cols:
        return df
    event_count_col = (
        "event_count_all_time" if "event_count_all_time" in event_count_cols
        else event_count_cols[-1]
    )
    df = df.withColumn(
        "_lifecycle_duration_days",
        (F.col("days_since_first") - F.col("days_since_last")).cast("double"),
    ).withColumn(
        "_lifecycle_intensity",
        F.col(event_count_col).cast("double")
        / F.greatest(F.col("_lifecycle_duration_days"), F.lit(1.0)),
    )
    _quantile_result = df.approxQuantile(
        ["_lifecycle_duration_days", "_lifecycle_intensity"], [0.5], 0.01,
    )
    tenure_med = _quantile_result[0][0] if _quantile_result[0] else 0.0
    intensity_med = _quantile_result[1][0] if _quantile_result[1] else 0.0
    _long = F.col("_lifecycle_duration_days") >= F.lit(tenure_med)
    _high = F.col("_lifecycle_intensity") >= F.lit(intensity_med)
    df = df.withColumn("lifecycle_quadrant",
        F.when(_long & _high, "steady_loyal_lifecycle")
        .when(_long & ~_high, "occasional_loyal_lifecycle")
        .when(~_long & _high, "intense_brief_lifecycle")
        .otherwise("one_shot_lifecycle"))
    df = df.drop("_lifecycle_duration_days", "_lifecycle_intensity")
    return df
{%- endif %}
{%- if config.lifecycle.include_cyclical_features %}

def add_cyclical_features(df):
    \"\"\"Source: Data Discovery > Cyclical Patterns\"\"\"
    raw_df = spark.table(landing_table("{{ raw_source }}"))
    time_col = "{{ config.time_column }}"
    mean_dow = raw_df.groupBy(ENTITY_COLUMN).agg(
        F.mean(F.dayofweek(F.col(time_col)).cast("double")).alias("mean_dow")
    )
    df = df.join(mean_dow, on=ENTITY_COLUMN, how="left")
    df = df.withColumn("dow_sin", F.sin(2 * 3.141592653589793 * F.col("mean_dow") / 7))
    df = df.withColumn("dow_cos", F.cos(2 * 3.141592653589793 * F.col("mean_dow") / 7))
    df = df.drop("mean_dow")
    return df
{%- endif %}
{%- if config.lifecycle.include_month_cyclical %}

def add_month_quarter_cyclical(df):
    \"\"\"Source: Data Discovery > Seasonal Patterns\"\"\"
    raw_df = spark.table(landing_table("{{ raw_source }}"))
    time_col = "{{ config.time_column }}"
    mean_month = raw_df.groupBy(ENTITY_COLUMN).agg(
        F.mean(F.month(F.col(time_col)).cast("double")).alias("mean_month")
    )
    df = df.join(mean_month, on=ENTITY_COLUMN, how="left")
    df = df.withColumn("month_sin", F.sin(2 * 3.141592653589793 * F.col("mean_month") / 12))
    df = df.withColumn("month_cos", F.cos(2 * 3.141592653589793 * F.col("mean_month") / 12))
{%- if config.lifecycle.include_quarter_cyclical %}
    mean_quarter = raw_df.groupBy(ENTITY_COLUMN).agg(
        F.mean(F.quarter(F.col(time_col)).cast("double")).alias("mean_quarter")
    )
    df = df.join(mean_quarter, on=ENTITY_COLUMN, how="left")
    df = df.withColumn("quarter_sin", F.sin(2 * 3.141592653589793 * F.col("mean_quarter") / 4))
    df = df.withColumn("quarter_cos", F.cos(2 * 3.141592653589793 * F.col("mean_quarter") / 4))
    df = df.drop("mean_month", "mean_quarter")
{%- else %}
    df = df.drop("mean_month")
{%- endif %}
    return df
{%- endif %}
{%- if config.lifecycle.include_trend_features %}

def add_trend_features(df):
    \"\"\"Source: Data Discovery > Trend Analysis\"\"\"
    import numpy as np
    pdf = df.toPandas()
    window_cols = sorted([c for c in pdf.columns if c.startswith("event_count_") and c != "event_count_all_time"])
    all_time_col = "event_count_all_time" if "event_count_all_time" in pdf.columns else None
    if window_cols and all_time_col:
        pdf["recent_vs_overall_ratio"] = pdf[window_cols[0]] / pdf[all_time_col].replace(0, float("nan"))
    if len(window_cols) >= 2:
        x = np.arange(len(window_cols), dtype=float)
        window_values = pdf[window_cols].values
        slopes = np.array([np.polyfit(x, row, 1)[0] if not np.any(np.isnan(row)) else 0.0 for row in window_values])
        pdf["entity_trend_slope"] = slopes
    df = spark.createDataFrame(pdf)
    return df
{%- endif %}
{%- if config.lifecycle.include_cohort_features %}

def add_cohort_features(df):
    \"\"\"Source: Data Discovery > Cohort Analysis\"\"\"
    raw_df = spark.table(landing_table("{{ raw_source }}"))
    time_col = "{{ config.time_column }}"
    first_event = raw_df.groupBy(ENTITY_COLUMN).agg(F.min(time_col).alias("first_event"))
    first_event = first_event.withColumn("cohort_year", F.year("first_event"))
    first_event = first_event.withColumn("cohort_quarter", F.quarter("first_event"))
    df = df.join(first_event.select(ENTITY_COLUMN, "cohort_year", "cohort_quarter"), on=ENTITY_COLUMN, how="left")
    return df
{%- endif %}
{%- if config.lifecycle.momentum_pairs %}

def add_momentum_ratios(df):
    \"\"\"Source: Data Discovery > Engagement Momentum\"\"\"
{%- for pair in config.lifecycle.momentum_pairs %}
    short_col = "event_count_{{ pair.short_window }}"
    long_col = "event_count_{{ pair.long_window }}"
    if short_col in df.columns and long_col in df.columns:
        df = df.withColumn("momentum_{{ pair.short_window }}_{{ pair.long_window }}", F.col(short_col) / F.when(F.col(long_col) != 0, F.col(long_col)).otherwise(F.lit(None)))
{%- endfor %}
    return df
{%- endif %}

def enrich_lifecycle(df):
    \"\"\"Source: Data Discovery > Entity Lifecycle Enrichment\"\"\"
{%- if config.lifecycle.include_recency_bucket %}
    df = add_recency_tenure(df)
    df = add_recency_buckets(df)
{%- endif %}
{%- if config.lifecycle.include_lifecycle_quadrant %}
    df = add_lifecycle_quadrant(df)
{%- endif %}
{%- if config.lifecycle.include_cyclical_features %}
    df = add_cyclical_features(df)
{%- endif %}
{%- if config.lifecycle.include_month_cyclical %}
    df = add_month_quarter_cyclical(df)
{%- endif %}
{%- if config.lifecycle.include_trend_features %}
    df = add_trend_features(df)
{%- endif %}
{%- if config.lifecycle.include_cohort_features %}
    df = add_cohort_features(df)
{%- endif %}
{%- if config.lifecycle.momentum_pairs %}
    df = add_momentum_ratios(df)
{%- endif %}
    return df
{%- endif %}

# COMMAND ----------

def run_bronze_entity():
    df = load_aggregated()
{%- if config.post_shaping %}
    df = apply_post_shaping(df)
{%- endif %}
{%- if config.lifecycle %}
    df = enrich_lifecycle(df)
{%- endif %}
    output_table = bronze_table(SOURCE_NAME)
    df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(output_table)
    from delta.tables import DeltaTable
    _z_cols = [c for c in [ENTITY_COLUMN] if c in [f.name for f in df.schema.fields]]
    if _z_cols:
        DeltaTable.forName(spark, output_table).optimize().executeZOrderBy(_z_cols)
    else:
        DeltaTable.forName(spark, output_table).optimize().executeCompaction()
    return df

import json as _bronze_json
import traceback as _bronze_tb
try:
    result = run_bronze_entity()
    _summary = f"{result.count():,} rows, {len(result.columns)} columns"
    # Narrow-projection display (see silver template for full rationale).
    _disp = [c for c in ("entity_id", "ACCOUNT_ID", "as_of_date") if c in result.columns] or list(result.columns[:10])
    display(result.select(*_disp).limit(20))
except Exception as _bronze_exc:
    print("=" * 70, flush=True)
    print(f"[bronze_entity] FATAL: {type(_bronze_exc).__name__}: {_bronze_exc}", flush=True)
    print("=" * 70, flush=True)
    print(_bronze_tb.format_exc(), flush=True)
    dbutils.notebook.exit(_bronze_json.dumps({
        "status": "FAILED",
        "stage": "bronze_entity",
        "error_type": type(_bronze_exc).__name__,
        "error_message": str(_bronze_exc)[:2000],
    }))
    raise
dbutils.notebook.exit(_summary)
""",
    "databricks_silver.py.j2": """# Databricks notebook source
# MAGIC %md
# MAGIC # Silver: Feature Set {{ config.composite_name or config.name }}

# COMMAND ----------

from pyspark.sql import functions as F
{% if config.silver.grid_dates %}
from customer_retention.stages.temporal.spark_temporal_merger import SparkTemporalMerger
from customer_retention.stages.temporal.temporal_merger import MergeConfig, DatasetMergeInput
from customer_retention.core.config.column_config import DatasetGranularity
{% endif %}

# COMMAND ----------

# MAGIC %run ../config

# COMMAND ----------

{% if config.silver.grid_dates %}
{% set has_key_resolution = config.silver.merge_sources | selectattr('key_resolution_steps') | list | length > 0 %}
GRID_DATES = {{ config.silver.grid_dates }}

MERGE_SOURCE_META = [
{% for src in config.silver.merge_sources %}
    {"name": "{{ src.name }}", "granularity": "{{ src.granularity }}"{{ ', "feature_timestamp_column": "' + src.feature_timestamp_column + '"' if src.feature_timestamp_column else '' }}, "key_resolution_steps": [{% for kr in src.key_resolution_steps %}{"bridge_dataset": "{{ kr.bridge_dataset }}", "source_key": "{{ kr.source_key }}", "bridge_key": "{{ kr.bridge_key }}", "resolve_column": "{{ kr.resolve_column }}"}{{ ", " if not loop.last else "" }}{% endfor %}]},
{% endfor %}
]
{% endif %}

def _bronze_output_name(name):
    source = SOURCES[name]
    if source.get("is_event_level"):
        return f"{name}_aggregated"
    return name

def load_bronze_outputs():
    outputs = {}
    for name, source in SOURCES.items():
        tbl = bronze_table(_bronze_output_name(name))
        outputs[name] = spark.table(tbl)
    return outputs

{% if config.silver.grid_dates %}
# Event-source configs carry the event-grain ID (e.g. OPPORTUNITY_ID) in
# entity_key, but the bronze aggregation step rolls up to the entity grain
# and the event-grain column is dropped. We trust the configured key when
# it survives in the base bronze output; otherwise fall back to a small
# set of conventional entity columns. Raise a clear error when none match
# so the failure mode is unambiguous instead of an opaque UNRESOLVED_COLUMN.
def _resolve_silver_entity_key(configured_key, bronze_outputs, base_source):
    base_cols = bronze_outputs[base_source].columns
    base_cols_lower = {c.lower(): c for c in base_cols}
    if configured_key in base_cols:
        return configured_key
    if configured_key.lower() in base_cols_lower:
        return base_cols_lower[configured_key.lower()]
    for candidate in ("entity_id", "ACCOUNT_ID", "account_id", "CUSTOMER_ID", "customer_id", "USER_ID", "user_id"):
        if candidate in base_cols or candidate.lower() in base_cols_lower:
            return base_cols_lower.get(candidate.lower(), candidate)
    raise RuntimeError(
        f"silver merge: configured entity_key '{configured_key}' is not in "
        f"base source '{base_source}' (cols={base_cols}). The bronze "
        f"aggregation likely dropped the event-grain column; set "
        f"config.silver.entity_key to the entity-grain key (e.g. ACCOUNT_ID)."
    )


def merge_sources(bronze_outputs):
    _configured_entity_key = "{{ config.silver.entity_key or config.sources[0].entity_key }}"
    base_source = "{{ config.sources[0].name }}"
    raw_entity_key = _resolve_silver_entity_key(_configured_entity_key, bronze_outputs, base_source)
    if raw_entity_key != _configured_entity_key:
        print(f"  silver merge: configured entity_key '{_configured_entity_key}' missing from "
              f"'{base_source}' bronze output -- falling back to '{raw_entity_key}'")
    entity_ids = bronze_outputs[base_source].select(raw_entity_key).distinct()
{% if has_key_resolution %}
    for meta in MERGE_SOURCE_META:
        kr_steps = meta.get("key_resolution_steps", [])
        if not kr_steps:
            continue
        df = bronze_outputs[meta["name"]]
        for step in kr_steps:
            # Codegen guard (closes GR12 / patch 14): when the bronze
            # aggregator already pre-resolved the source_key upstream
            # (rolling up to ACCOUNT_ID grain), the join column is no
            # longer present on the bronze output. Skip the now-redundant
            # join — the resolve_column is already on `df` and the
            # downstream merger picks it up unchanged.
            if step["source_key"] not in df.columns:
                print(
                    f"  silver merge: skipping kr_step for source={meta['name']!r}, "
                    f"source_key={step['source_key']!r} not in bronze output "
                    f"(already pre-resolved upstream — resolve_column={step['resolve_column']!r})"
                )
                continue
            bridge_subset = bronze_outputs[step["bridge_dataset"]].select(
                step["bridge_key"], step["resolve_column"]
            ).dropDuplicates([step["bridge_key"]])
            df = df.join(
                bridge_subset,
                df[step["source_key"]] == bridge_subset[step["bridge_key"]],
                "inner",
            )
            if step["source_key"] != step["bridge_key"]:
                df = df.drop(bridge_subset[step["bridge_key"]])
        bronze_outputs[meta["name"]] = df
{% endif %}
    if raw_entity_key != "entity_id":
        for name, df in bronze_outputs.items():
            if raw_entity_key in df.columns:
                bronze_outputs[name] = df.withColumnRenamed(raw_entity_key, "entity_id")
    merger = SparkTemporalMerger(MergeConfig(
        entity_key="entity_id",
        scratch_namespace=f"{CATALOG}.{SCHEMA}",
    ))
    spine = merger.build_spine(entity_ids, GRID_DATES)
    inputs = []
    for meta in MERGE_SOURCE_META:
        name = meta["name"]
        if name not in bronze_outputs:
            continue
        granularity = DatasetGranularity(meta["granularity"])
        inputs.append(DatasetMergeInput(
            name=name,
            df=bronze_outputs[name],
            granularity=granularity,
            feature_timestamp_column=meta.get("feature_timestamp_column"),
        ))
    try:
        merged, _report = merger.merge_all(spine, inputs)
        if hasattr(merged, "to_spark"):
            merged = merged.to_spark()
    finally:
        _dropped = merger.cleanup_scratch_tables(spark)
        if _dropped:
            print(f"  silver scratch cleanup: dropped {_dropped} _silver_merge_ckpt_* table(s)")
    print(f"  merge complete: {len(merged.columns)} columns, datasets={_report.datasets_merged}")
    print(f"  spine: {_report.spine_rows:,} rows = {_report.spine_entities:,} entities x {_report.spine_dates} dates ({_report.spine_stats_seconds:.1f}s)")
    print(f"  checkpoints: {_report.checkpoint_count} ({_report.checkpoint_seconds:.1f}s), validation: {_report.validation_seconds:.1f}s, merge_total: {_report.merge_total_seconds:.1f}s")
    print(f"  per-source merge timings:")
    for _ds_name in _report.datasets_merged:
        _ds_secs = _report.seconds_per_dataset.get(_ds_name, 0.0)
        _ds_cols = _report.columns_per_dataset.get(_ds_name, 0)
        print(f"    {_ds_name}: {_ds_secs:.1f}s (+{_ds_cols} cols)")
    return merged, _report
{% else %}
def merge_sources(bronze_outputs):
    raw_entity_key = "{{ config.silver.entity_key or config.sources[0].entity_key }}"
    base_source = "{{ config.sources[0].name }}"
    merged = bronze_outputs[base_source]
{% for join in config.silver.joins %}
{% if join.left_keys | length == 1 %}
    merged = merged.join(
        bronze_outputs["{{ join.right_source }}"],
        merged["{{ join.left_keys[0] }}"] == bronze_outputs["{{ join.right_source }}"]["{{ join.right_keys[0] }}"],
        "{{ join.how }}",
    ).drop(bronze_outputs["{{ join.right_source }}"]["{{ join.right_keys[0] }}"])
{% else %}
    _join_cond = {% for i in range(join.left_keys | length) %}{% if not loop.first %} & {% endif %}(merged["{{ join.left_keys[i] }}"] == bronze_outputs["{{ join.right_source }}"]["{{ join.right_keys[i] }}"]){% endfor %}

    _drop_cols = [{% for k in join.right_keys %}bronze_outputs["{{ join.right_source }}"]["{{ k }}"]{{ ", " if not loop.last else "" }}{% endfor %}]
    _joined = merged.join(bronze_outputs["{{ join.right_source }}"], _join_cond, "{{ join.how }}")
    merged = _joined{% for k in join.right_keys %}.drop(bronze_outputs["{{ join.right_source }}"]["{{ k }}"]){% endfor %}

{% endif %}
{% endfor %}
    if raw_entity_key != "entity_id" and raw_entity_key in merged.columns:
        merged = merged.withColumnRenamed(raw_entity_key, "entity_id")
    return merged, None
{% endif %}

{% set derived_groups = group_steps(config.silver.derived_columns) %}
{% if config.silver.derived_columns %}
def apply_derived_columns(df):
    # Plan barrier between groups is the silver write's staging round-trip
    # (in `run_silver` below) — `localCheckpoint` is unreliable on Spark
    # Connect at this column count and just adds plan markers Catalyst has
    # to walk later.
{%- for func_name, steps in derived_groups %}
    df = {{ func_name }}(df)
{%- endfor %}
    return df

{% for func_name, steps in derived_groups %}
def {{ func_name }}(df):
{%- set _prov = spark_provenance_block(steps) %}
{%- if _prov %}
{{ _prov }}
{%- endif %}
{%- for t in steps %}
    # {{ t.rationale }}
    df = {{ render_spark_step_call(t) }}
{%- endfor %}
    return df
{% endfor %}
{% endif %}

# COMMAND ----------

def apply_target_label_map(df):
    # FW-12: encode a categorical target via the registered TARGET_LABEL_MAP.
    # No-op when the map is empty (numeric target = standard path) or when
    # the target column already evaluates as numeric. When the map is set,
    # applies a Spark CASE WHEN that maps each value to 0/1; unmapped values
    # become NULL and the null-label filter at training time drops them.
    # Mirrors `customer_retention.stages.profiling.target_validator
    # .apply_target_encoding` so exploration and codegen stay in lock-step.
    if not TARGET_LABEL_MAP:
        return df
    if TARGET_COLUMN not in [f.name for f in df.schema.fields]:
        return df
    _dtype = df.schema[TARGET_COLUMN].dataType.simpleString()
    if _dtype.startswith(("int", "long", "double", "float", "decimal", "bigint",
                          "smallint", "tinyint", "byte", "short")):
        # Already numeric — do not double-encode.
        return df
    _case = F
    for _val, _lbl in TARGET_LABEL_MAP.items():
        _case = _case.when(F.col(TARGET_COLUMN) == F.lit(_val), F.lit(int(_lbl)))
    print(f"  silver target encoding: applying TARGET_LABEL_MAP "
          f"({len(TARGET_LABEL_MAP)} entries) to {TARGET_COLUMN!r}")
    return df.withColumn(TARGET_COLUMN, _case.otherwise(F.lit(None).cast("double")))

# COMMAND ----------

def create_holdout_mask(df, holdout_fraction=0.1, random_state=42):
    original_col = f"original_{TARGET_COLUMN}"
    if original_col in df.columns:
        return df
    if TARGET_COLUMN not in [f.name for f in df.schema.fields]:
        return df
    if HOLDOUT_ENTITY_IDS is not None:
        from pyspark.sql.types import StringType, StructField, StructType
        spark = df.sparkSession
        holdout_ids = spark.createDataFrame(
            [(str(eid),) for eid in HOLDOUT_ENTITY_IDS],
            StructType([StructField("entity_id", StringType(), True)]),
        )
        holdout_ids = holdout_ids.withColumn(
            "entity_id", F.col("entity_id").cast(df.schema["entity_id"].dataType)
        )
        print(f"  Using {holdout_ids.count():,} pre-computed holdout entity IDs")
    else:
        frac = min(1.0, max(0.0, holdout_fraction))
        holdout_ids = df.select("entity_id").distinct().sample(
            withReplacement=False, fraction=frac, seed=random_state
        )
    df = df.join(holdout_ids, on="entity_id", how="left_semi").withColumn(
        original_col, F.col(TARGET_COLUMN)
    ).withColumn(
        TARGET_COLUMN, F.lit(None).cast(df.schema[TARGET_COLUMN].dataType)
    ).unionByName(
        df.join(holdout_ids, on="entity_id", how="left_anti").withColumn(
            original_col, F.lit(None).cast(df.schema[TARGET_COLUMN].dataType)
        )
    )
    return df

# COMMAND ----------

def run_silver():
    import time as _time
    _timings = {}
    _t0 = _time.monotonic()
    bronze_outputs = load_bronze_outputs()
    _timings["load_bronze"] = round(_time.monotonic() - _t0, 2)
    print(f"  load_bronze: {_timings['load_bronze']:.1f}s")
    _t1 = _time.monotonic()
    merged, _report = merge_sources(bronze_outputs)
    _timings["merge_sources"] = round(_time.monotonic() - _t1, 2)
    print(f"  merge_sources: {_timings['merge_sources']:.1f}s ({len(merged.columns)} cols)")
{% if config.silver.derived_columns %}
    _t2 = _time.monotonic()
    merged = apply_derived_columns(merged)
    _timings["apply_derived"] = round(_time.monotonic() - _t2, 2)
    print(f"  apply_derived: {_timings['apply_derived']:.1f}s")
{% endif %}
    _t_tlm = _time.monotonic()
    merged = apply_target_label_map(merged)
    _timings["target_label_map"] = round(_time.monotonic() - _t_tlm, 2)
    if _timings["target_label_map"] > 0.05:
        print(f"  target_label_map: {_timings['target_label_map']:.1f}s")
    _t3 = _time.monotonic()
    merged = create_holdout_mask(merged)
    _timings["holdout_mask"] = round(_time.monotonic() - _t3, 2)
    print(f"  holdout_mask: {_timings['holdout_mask']:.1f}s")
    _t4 = _time.monotonic()
    # Wide-schema barrier (~2620 cols): the cumulative AttributeReferences
    # from temporal merge + apply_derived_columns + holdout-mask can trip
    # Catalyst's adaptive-plan attribute-index resolver on the write path
    # with `IllegalArgumentException: Cannot find column index for attribute
    # X#NNNN`. Disable AQE for the write and stage through a temp Delta
    # table so the final write sees a fresh, shallow plan with stable
    # attribute IDs. Same class of bug the display narrow-projection
    # workaround below documents.
    output_table = silver_table()
    _staging_table = output_table + "__staging"
    _aqe_prev = spark.conf.get("spark.sql.adaptive.enabled", "true")
    spark.conf.set("spark.sql.adaptive.enabled", "false")
    try:
        (merged.write.format("delta")
            .mode("overwrite")
            .option("overwriteSchema", "true")
            .saveAsTable(_staging_table))
        merged = spark.table(_staging_table)
        merged.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(output_table)
    finally:
        spark.conf.set("spark.sql.adaptive.enabled", _aqe_prev)
        spark.sql(f"DROP TABLE IF EXISTS {_staging_table}")
    _timings["delta_write"] = round(_time.monotonic() - _t4, 2)
    print(f"  delta_write (staging round-trip): {_timings['delta_write']:.1f}s")
    # Z-Order skipped by default. On wide silver tables it adds 30-90 min for
    # a marginal read-time speedup that doesn't materially help downstream.
    if OPTIMIZE_AFTER_WRITE:
        _t5 = _time.monotonic()
        from delta.tables import DeltaTable
        _z_cols = [c for c in ["entity_id", "as_of_date"] if c in [f.name for f in merged.schema.fields]]
        if _z_cols:
            DeltaTable.forName(spark, output_table).optimize().executeZOrderBy(_z_cols)
        else:
            DeltaTable.forName(spark, output_table).optimize().executeCompaction()
        _timings["optimize"] = round(_time.monotonic() - _t5, 2)
        print(f"  optimize: {_timings['optimize']:.1f}s")
    else:
        _timings["optimize"] = 0.0
        print(f"  optimize: SKIPPED (OPTIMIZE_AFTER_WRITE=False)")
    _timings["total"] = round(_time.monotonic() - _t0, 2)
    print(f"  total: {_timings['total']:.1f}s")
    return output_table, _timings, _report

# Set to True to re-enable post-write OPTIMIZE+ZORDER on the silver table.
OPTIMIZE_AFTER_WRITE = False

import json
import traceback as _silver_tb
try:
    _output_table, _silver_timings, _silver_report = run_silver()
    _result = spark.table(_output_table)
    _row_count = _result.count()
    _col_count = len(_result.columns)
except Exception as _silver_exc:
    print("=" * 70, flush=True)
    print(f"[silver] FATAL: {type(_silver_exc).__name__}: {_silver_exc}", flush=True)
    print("=" * 70, flush=True)
    print(_silver_tb.format_exc(), flush=True)
    dbutils.notebook.exit(json.dumps({
        "status": "FAILED",
        "stage": "silver",
        "error_type": type(_silver_exc).__name__,
        "error_message": str(_silver_exc)[:2000],
    }))
    raise

_silver_results = {
    "status": "OK",
    "rows": _row_count,
    "columns": _col_count,
    "elapsed_seconds": _silver_timings,
    "merge_breakdown": (
        {
            "spine_rows": _silver_report.spine_rows,
            "spine_entities": _silver_report.spine_entities,
            "spine_dates": _silver_report.spine_dates,
            "spine_stats_seconds": round(_silver_report.spine_stats_seconds, 2),
            "checkpoint_count": _silver_report.checkpoint_count,
            "checkpoint_seconds": round(_silver_report.checkpoint_seconds, 2),
            "validation_seconds": round(_silver_report.validation_seconds, 2),
            "merge_total_seconds": round(_silver_report.merge_total_seconds, 2),
            "datasets_merged": list(_silver_report.datasets_merged),
            "columns_per_dataset": dict(_silver_report.columns_per_dataset),
            "seconds_per_dataset": {k: round(v, 2) for k, v in _silver_report.seconds_per_dataset.items()},
            "total_columns": _silver_report.total_columns,
        }
        if _silver_report is not None else None
    ),
}

from pathlib import Path
from customer_retention.analysis.auto_explorer.run_namespace import RunNamespace
try:
    _exp_dir = dbutils.widgets.get("experiments_dir")
    _run_id = dbutils.widgets.get("run_id")
    _NAMESPACE = RunNamespace(root=Path(_exp_dir), run_id=_run_id) if _exp_dir and _run_id else None
except Exception:
    _NAMESPACE = RunNamespace.from_env_or_latest()
if _NAMESPACE is not None:
    _silver_meta = dict(_silver_results)
    _silver_meta["column_list"] = _result.columns
    _NAMESPACE.silver_metadata_path.parent.mkdir(parents=True, exist_ok=True)
    _NAMESPACE.silver_metadata_path.write_text(json.dumps(_silver_meta, default=str))

print("\\n" + "=" * 60)
print("SILVER RESULTS")
print("=" * 60)
print(json.dumps(_silver_results, indent=2, default=str))

# Narrow-projection display: the silver Delta table is already committed (row
# count + OPTIMIZE done above); rendering the full 2620-col DataFrame through
# Databricks' display widget serialises the entire logical plan to protobuf,
# which trips Catalyst attribute-id resolution at very wide schemas (same class
# of bug as the merge-time XX000). Display the spine columns only — operators
# inspect the full table via spark.table(...) when they need it.
_display_cols = [c for c in ("entity_id", "as_of_date", TARGET_COLUMN if "TARGET_COLUMN" in dir() else None) if c and c in _result.columns]
if _display_cols:
    display(_result.select(*_display_cols).limit(20))
else:
    display(_result.limit(20))
dbutils.notebook.exit(json.dumps(_silver_results, default=str))
""",
    "databricks_gold.py.j2": r"""# Databricks notebook source
# MAGIC %md
# MAGIC # Gold: Features {{ config.composite_name or config.name }}

# COMMAND ----------

from pyspark.sql import functions as F

from customer_retention.core.naming import sanitize_column_token

# COMMAND ----------

# MAGIC %run ../config

# COMMAND ----------

def _encode_one_hot(df, col, max_categories=100):
    if col not in df.columns:
        # Fail-safe: the encoder column was dropped upstream (feature_selection,
        # leakage exclusion, silver merge prefix collision, etc.). Raising here
        # would block gold's saveAsTable 6-7h after the silent drop happened.
        # Instead, log and skip — gold ships without the affected one-hot
        # columns. Training's PARITY_IGNORED_FEATURES is the operator escape
        # hatch for the missing `<col>_<category>` columns.
        print(
            f"[GOLD] _encode_one_hot: skipping {col!r} - not in df.columns; "
            f"will be absent from gold"
        )
        return df
    categories = [row[col] for row in df.select(col).distinct().collect() if row[col] is not None]
    if len(categories) > max_categories:
        print(f"WARNING: column '{col}' has {len(categories)} categories (>{max_categories}), using label encoding instead")
        return _label_encode(df, col)
    # Dedup raw categories whose sanitized safe_names collide (whitespace,
    # punctuation, case variants in Salesforce string columns). Group all
    # raw values that map to the same safe_name into a single `isin`
    # indicator so we lose no information and emit one column per token.
    safe_to_raw = {}
    for cat in sorted(categories):
        safe_name = f"{col}_{sanitize_column_token(cat)}"
        key = safe_name.lower()
        safe_to_raw.setdefault(key, (safe_name, []))[1].append(cat)
    for safe_name, group in safe_to_raw.values():
        if len(group) == 1:
            df = df.withColumn(safe_name, F.when(F.col(col) == group[0], 1).otherwise(0))
        else:
            df = df.withColumn(safe_name, F.when(F.col(col).isin(group), 1).otherwise(0))
    df = df.drop(col)
    return df

def _label_encode(df, col):
    if col not in df.columns:
        # Same fail-safe as _encode_one_hot: warn + return df unchanged so gold
        # completes; PARITY_IGNORED_FEATURES handles the downstream miss.
        print(
            f"[GOLD] _label_encode: skipping {col!r} - not in df.columns; "
            f"will be absent from gold"
        )
        return df
    # Spark-SQL broadcast-join label encoding. Coding_Practices.md:112 forbids
    # pyspark.ml estimators for data prep on shared clusters — the indexer fit
    # path overflows the Spark Connect ML cache (10 GB cap) on multi-million-row
    # silver outputs. Collect distinct non-null values, build a small
    # (value, index) DataFrame, broadcast-join, coalesce nulls/unknowns to
    # len(values).
    _vals = sorted({
        row[0] for row in
        df.select(col).where(F.col(col).isNotNull()).distinct().collect()
    })
    if not _vals:
        print(f"[GOLD] _label_encode: {col!r} has no non-null values; dropping")
        return df.drop(col)
    _n = len(_vals)
    _col_dtype = df.schema[col].dataType.simpleString()
    _lookup = df.sparkSession.createDataFrame(
        [(_v, _i) for _i, _v in enumerate(_vals)],
        schema=f"{col} {_col_dtype}, __idx int",
    )
    _tmp = f"__{col}_idx"
    df = (
        df.join(F.broadcast(_lookup), on=col, how="left")
          .withColumn(_tmp, F.coalesce(F.col("__idx"), F.lit(_n)).cast("int"))
          .drop("__idx")
          .drop(col)
          .withColumnRenamed(_tmp, col)
    )
    return df

# Chained `withColumn` builds one Project node per call; with hundreds of
# columns, Catalyst optimization and Spark Connect plan serialization fail with
# opaque SparkConnectGrpcException (status=UNKNOWN). The batch helpers below
# fold N transforms into ONE `select(...)` (one Project node) and chunk at
# `_BATCH_CHUNK_SIZE` so a single select doesn't itself become unwieldy.
_BATCH_CHUNK_SIZE = 500

def _apply_batched_select(df, expr_by_col):
    # Apply a per-column expression map via chunked single-select calls.
    # `expr_by_col` is a dict[col -> Column]. For every column present in
    # `df.columns`, emits the mapped expression aliased back to the same name;
    # other columns pass through unchanged. Chunked at `_BATCH_CHUNK_SIZE` to
    # keep each select width bounded.
    targets = [c for c in df.columns if c in expr_by_col]
    if not targets:
        return df
    for chunk_start in range(0, len(targets), _BATCH_CHUNK_SIZE):
        chunk = set(targets[chunk_start:chunk_start + _BATCH_CHUNK_SIZE])
        df = df.select(*[
            expr_by_col[c].alias(c) if c in chunk else F.col(c)
            for c in df.columns
        ])
    return df

def _batch_scale_standard(df, cols):
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return df
    exprs = []
    for c in cols:
        exprs.extend([F.mean(c).alias(f"{c}__mean"), F.stddev(c).alias(f"{c}__std")])
    stats = df.agg(*exprs).collect()[0]
    expr_by_col = {}
    for c in cols:
        mean_val = stats[f"{c}__mean"] or 0
        std_val = stats[f"{c}__std"] or 1
        if std_val == 0:
            std_val = 1
        expr_by_col[c] = (F.col(c) - mean_val) / std_val
    return _apply_batched_select(df, expr_by_col)

def _batch_scale_minmax(df, cols):
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return df
    exprs = []
    for c in cols:
        exprs.extend([F.min(c).alias(f"{c}__min"), F.max(c).alias(f"{c}__max")])
    stats = df.agg(*exprs).collect()[0]
    expr_by_col = {}
    for c in cols:
        min_val = stats[f"{c}__min"] or 0
        max_val = stats[f"{c}__max"] or 1
        range_val = max_val - min_val
        if range_val == 0:
            range_val = 1
        expr_by_col[c] = (F.col(c) - min_val) / range_val
    return _apply_batched_select(df, expr_by_col)

def _batch_segment_aware_cap(df, cols):
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return df
    quantile_map = dict(zip(cols, df.approxQuantile(cols, [0.25, 0.75], 0.01)))
    expr_by_col = {}
    for c in cols:
        qs = quantile_map[c]
        if len(qs) == 2:
            q1, q3 = qs
            iqr = q3 - q1
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            expr_by_col[c] = (
                F.when(F.col(c) < lower, lower)
                .when(F.col(c) > upper, upper)
                .otherwise(F.col(c))
            )
    return _apply_batched_select(df, expr_by_col)

def _batch_cap_then_log(df, cols):
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return df
    quantile_map = dict(zip(cols, df.approxQuantile(cols, [0.99], 0.01)))
    expr_by_col = {}
    for c in cols:
        qs = quantile_map[c]
        if qs:
            expr_by_col[c] = F.log1p(
                F.greatest(F.least(F.col(c), F.lit(qs[0])), F.lit(0)).cast("double")
            )
    return _apply_batched_select(df, expr_by_col)

def _batch_log_transform(df, cols):
    # log1p applied to each column in one chunked `select` (keeps in-place).
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return df
    return _apply_batched_select(df, {c: F.log1p(F.col(c)) for c in cols})

def _batch_sqrt_transform(df, cols):
    # sqrt(abs(col)) applied to each column in one chunked `select`.
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return df
    return _apply_batched_select(df, {c: F.sqrt(F.abs(F.col(c))) for c in cols})

def _batch_yeo_johnson(df, cols):
    # Codegen Yeo-Johnson approximation: log1p(abs(col)). Single chunked select.
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return df
    return _apply_batched_select(df, {c: F.log1p(F.abs(F.col(c))) for c in cols})

def _batch_zero_inflation(df, cols):
    # For each col, emits two NEW columns `<col>_is_zero` and `<col>_log` in
    # a single chunked select. Keeps the original column intact.
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return df
    new_exprs = []
    for c in cols:
        new_exprs.append(F.when(F.col(c) == 0, 1).otherwise(0).alias(f"{c}_is_zero"))
        new_exprs.append(
            F.when(F.col(c) > 0, F.log1p(F.col(c))).otherwise(0).alias(f"{c}_log")
        )
    for chunk_start in range(0, len(new_exprs), _BATCH_CHUNK_SIZE):
        chunk = new_exprs[chunk_start:chunk_start + _BATCH_CHUNK_SIZE]
        df = df.select(*[F.col(c) for c in df.columns], *chunk)
    return df

def _batch_one_hot_encode(df, cols, max_categories=100):
    # Bulk one-hot encode multiple columns.
    # Collects distinct values for ALL listed columns in ONE Spark job via
    # `collect_set` aggregation, then materialises the indicator columns via a
    # single chunked `select`. Falls back to per-column `_label_encode` for
    # columns whose cardinality exceeds `max_categories`.
    # With 24 categorical columns this collapses 24 separate `distinct().collect()`
    # Spark jobs into 1, and replaces the chained `withColumn` per category with
    # a single Project node per chunk.
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return df
    # Narrow projection for the distinct collection — running agg+collect on
    # the wide df (~3000 cols) trips Catalyst's adaptive-plan attribute-index
    # resolver during execution. `df.select(*cols)` gives the agg a tiny
    # schema to plan against; the resulting collect is cheap.
    distinct_exprs = [F.collect_set(F.col(c)).alias(c) for c in cols]
    row = df.select(*cols).agg(*distinct_exprs).collect()[0]
    one_hot_plan = {}
    label_encode_cols = []
    for c in cols:
        cats = [v for v in (row[c] or []) if v is not None]
        if not cats:
            continue
        if len(cats) > max_categories:
            print(
                f"WARNING: column '{c}' has {len(cats)} categories "
                f"(>{max_categories}), using label encoding instead"
            )
            label_encode_cols.append(c)
        else:
            one_hot_plan[c] = sorted(str(v) for v in cats)
    if one_hot_plan:
        new_exprs = []
        for c, cats in one_hot_plan.items():
            # Dedup by sanitized safe_name (case-insensitive). `collect_set`
            # returns raw distinct values, but `sanitize_column_token`
            # collapses whitespace-padded / punctuation-only / case-different
            # variants to the same token. Without this, two raw values
            # (`"1"` and `" 1"`) emit two encoded columns with the same
            # alias and `saveAsTable` rejects them as case-insensitive
            # duplicates with COLUMN_ALREADY_EXISTS. We coalesce all raw
            # values mapping to the same safe_name into a single
            # `isin`-based indicator so no information is dropped.
            safe_to_raw = {}
            for cat in cats:
                key = f"{c}_{sanitize_column_token(cat)}".lower()
                safe_to_raw.setdefault(key, (f"{c}_{sanitize_column_token(cat)}", []))[1].append(cat)
            for safe_name, group in safe_to_raw.values():
                if len(group) == 1:
                    expr = F.when(F.col(c) == group[0], 1).otherwise(0)
                else:
                    expr = F.when(F.col(c).isin(group), 1).otherwise(0)
                new_exprs.append(expr.alias(safe_name))
        for chunk_start in range(0, len(new_exprs), _BATCH_CHUNK_SIZE):
            chunk = new_exprs[chunk_start:chunk_start + _BATCH_CHUNK_SIZE]
            df = df.select(*[F.col(c) for c in df.columns], *chunk)
        df = df.drop(*list(one_hot_plan.keys()))
    for c in label_encode_cols:
        df = _label_encode(df, c)
    return df

# COMMAND ----------

def apply_encodings(df):
{%- set _prov = spark_provenance_block(config.gold.encodings) %}
{%- if _prov %}
{{ _prov }}
{%- endif %}
{%- set enc_ns = namespace(one_hot=[], label=[]) %}
{%- for step in config.gold.encodings %}
{%- if step.parameters.get('method') in ('one_hot', 'onehot') %}{% set enc_ns.one_hot = enc_ns.one_hot + [step.column] %}{% else %}{% set enc_ns.label = enc_ns.label + [step.column] %}{% endif %}
{%- endfor %}
{%- if enc_ns.one_hot %}
    df = _batch_one_hot_encode(df, [{% for c in enc_ns.one_hot %}"{{ c }}"{{ ", " if not loop.last }}{% endfor %}])
{%- endif %}
{%- for c in enc_ns.label %}
    df = _label_encode(df, "{{ c }}")
{%- endfor %}
    return df

def apply_scalings(df):
{%- set _prov = spark_provenance_block(config.gold.scalings) %}
{%- if _prov %}
{{ _prov }}
{%- endif %}
{%- if config.gold.scalings %}
{%- set ns = namespace(standard=[], minmax=[]) %}
{%- for step in config.gold.scalings %}
{%- if step.parameters.get('method') == 'minmax' %}{% set ns.minmax = ns.minmax + [step.column] %}{% else %}{% set ns.standard = ns.standard + [step.column] %}{% endif %}
{%- endfor %}
{%- if ns.standard %}
    df = _batch_scale_standard(df, [{% for c in ns.standard %}"{{ c }}"{{ ", " if not loop.last }}{% endfor %}])
{%- endif %}
{%- if ns.minmax %}
    df = _batch_scale_minmax(df, [{% for c in ns.minmax %}"{{ c }}"{{ ", " if not loop.last }}{% endfor %}])
{%- endif %}
{%- endif %}
    return df

{% set transform_groups = group_steps(config.gold.transformations) %}
def apply_transformations(df):
    # Each `apply_*_transforms` group below collapses N per-column
    # withColumn into ONE chunked single-`select`, so the per-group plan
    # is shallow. The hard plan barrier between transformations and the
    # next stage (encoding) is the staging round-trip in `run_gold` —
    # `localCheckpoint` here is unreliable on Spark Connect at this
    # column count (often returns a logical-plan reference instead of
    # forcing materialisation), so we don't rely on it.
{%- for func_name, steps in transform_groups %}
    df = {{ func_name }}(df)
{%- endfor %}
    return df

{% for func_name, steps in transform_groups %}
def {{ func_name }}(df):
{%- set _prov = spark_provenance_block(steps) %}
{%- if _prov %}
{{ _prov }}
{%- endif %}
{%- if func_name == "apply_cap_then_log_transforms" %}
    df = _batch_cap_then_log(df, [{% for t in steps %}"{{ t.column }}"{{ ", " if not loop.last }}{% endfor %}])
{%- elif func_name == "cap_segment_aware_outliers" %}
    df = _batch_segment_aware_cap(df, [{% for t in steps %}"{{ t.column }}"{{ ", " if not loop.last }}{% endfor %}])
{%- elif func_name == "apply_log_transforms" %}
    df = _batch_log_transform(df, [{% for t in steps %}"{{ t.column }}"{{ ", " if not loop.last }}{% endfor %}])
{%- elif func_name == "apply_sqrt_transforms" %}
    df = _batch_sqrt_transform(df, [{% for t in steps %}"{{ t.column }}"{{ ", " if not loop.last }}{% endfor %}])
{%- elif func_name == "apply_power_transforms" %}
    df = _batch_yeo_johnson(df, [{% for t in steps %}"{{ t.column }}"{{ ", " if not loop.last }}{% endfor %}])
{%- elif func_name == "handle_zero_inflation" %}
    df = _batch_zero_inflation(df, [{% for t in steps %}"{{ t.column }}"{{ ", " if not loop.last }}{% endfor %}])
{%- else %}
{%- for t in steps %}
    # {{ t.rationale }}
    df = {{ render_spark_step_call(t) }}
{%- endfor %}
{%- endif %}
    return df
{% endfor %}

def apply_feature_selection(df):
{%- if config.gold.feature_exclusion_prefixes %}
    from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser
    _prefix_drops = FindingsParser.find_leakage_excluded_columns(df.columns, {{ config.gold.feature_exclusion_prefixes }})
    if _prefix_drops:
        df = df.drop(*_prefix_drops)
        print(f"  Dropped {len(_prefix_drops)} leakage-excluded columns")
{%- endif %}
{%- if config.gold.feature_selections %}
    drop_cols = {{ config.gold.feature_selections }}
    df = df.drop(*[c for c in drop_cols if c in df.columns])
{%- endif %}
    return df

# COMMAND ----------

def _cast_timestamp_ntz_to_timestamp(df):
    from pyspark.sql.types import TimestampNTZType, TimestampType
    ntz_names = {f.name for f in df.schema.fields if isinstance(f.dataType, TimestampNTZType)}
    if not ntz_names:
        return df
    cols = [
        F.col(f"`{f.name}`").cast(TimestampType()).alias(f.name)
        if f.name in ntz_names
        else F.col(f"`{f.name}`")
        for f in df.schema.fields
    ]
    return df.select(cols)

def _register_feature_table(table_name, df):
    has_ts = TIMESTAMP_COLUMN in [f.name for f in df.schema.fields]
    pk = ["entity_id", TIMESTAMP_COLUMN] if has_ts else ["entity_id"]
    for col in pk:
        spark.sql(f"ALTER TABLE {table_name} ALTER COLUMN `{col}` SET NOT NULL")
    constraint_name = table_name.replace(".", "_") + "_pk"
    pk_clause = ", ".join(
        f"`{c}` TIMESERIES" if c == TIMESTAMP_COLUMN and has_ts else f"`{c}`" for c in pk
    )
    try:
        spark.sql(f"ALTER TABLE {table_name} ADD CONSTRAINT {constraint_name} PRIMARY KEY ({pk_clause})")
        print(f"[GOLD] Registered feature table: {table_name} PK=({pk_clause})")
    except Exception as e:
        if "already exists" not in str(e).lower():
            raise
        print(f"[GOLD] Feature table {table_name} already registered")

def _stage_and_reload(df, table_name):
    # Force a hard plan barrier by writing to Delta and reading back. Used
    # between major gold stages because `localCheckpoint(eager=True)` is
    # unreliable on Spark Connect at ~3000 columns — it often returns a
    # logical-plan reference that still carries the original ExprIds
    # rather than forcing executor-side materialisation, so the deep
    # plan accumulates and Catalyst's column-index resolver crashes
    # later. A Delta round-trip is unambiguous: the returned df's plan
    # is a single read-from-table with stable, fresh attribute IDs.
    (df.write.format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(table_name))
    return spark.table(table_name)

def run_gold():
    import time as _time
    _t0 = _time.monotonic()
    output_table = gold_table()
    _stg_xforms = output_table + "__stg_xforms"
    _stg_enc = output_table + "__stg_enc"
    _stg_final = output_table + "__stg_final"
    # Disable AQE for the *entire* gold pipeline. AQE rewrites plans at
    # runtime and reassigns ExprIds — at ~3000 columns this trips the
    # column-index resolver with `IllegalArgumentException: Cannot find
    # column index for attribute X#NNNN`. Setting it before silver read
    # ensures every select-derived plan is built without AQE rewrites,
    # not just the final write. Restored in `finally` below.
    _aqe_prev = spark.conf.get("spark.sql.adaptive.enabled", "true")
    spark.conf.set("spark.sql.adaptive.enabled", "false")
    try:
        df = spark.table(silver_table())
        print(f"  load_silver: {_time.monotonic() - _t0:.1f}s")
        _t1 = _time.monotonic()
        df = apply_transformations(df)
        # Hard plan barrier before the wide-schema agg+collect inside
        # `_batch_one_hot_encode` — running that against the deep
        # post-transform plan trips the same column-index resolver bug
        # before encoding even begins. Staging here gives the agg a
        # shallow input plan.
        df = _stage_and_reload(df, _stg_xforms)
        print(f"  transformations: {_time.monotonic() - _t1:.1f}s")
        _t2 = _time.monotonic()
        df = apply_feature_selection(df)
        print(f"  feature_selection: {_time.monotonic() - _t2:.1f}s")
        _t3 = _time.monotonic()
        df = apply_encodings(df)
        # Stage between encoding fan-out and scaling so the scaling agg
        # plans against a fresh, post-encoding schema.
        df = _stage_and_reload(df, _stg_enc)
        print(f"  encodings: {_time.monotonic() - _t3:.1f}s")
        _t4 = _time.monotonic()
        df = apply_scalings(df)
        print(f"  scalings: {_time.monotonic() - _t4:.1f}s")
        if "as_of_date" in df.columns:
            df = df.withColumnRenamed("as_of_date", TIMESTAMP_COLUMN)
        elif "feature_timestamp" in df.columns:
            df = df.withColumnRenamed("feature_timestamp", TIMESTAMP_COLUMN)
        df = _cast_timestamp_ntz_to_timestamp(df)
        from pyspark.sql.types import DoubleType, LongType, IntegerType, ShortType
        _NUMERIC_TYPES = (DoubleType, LongType, IntegerType, ShortType)
        _f32_exprs = [
            F.col(c).cast("float").alias(c) if isinstance(df.schema[c].dataType, _NUMERIC_TYPES)
            else F.col(c)
            for c in df.columns
        ]
        df = df.select(*_f32_exprs)
        # Final stage barrier so the output_table write plans against a
        # shallow read (the rename + ntz cast + f32 cast otherwise stack
        # back up before write).
        _t5 = _time.monotonic()
        df = _stage_and_reload(df, _stg_final)
        df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(output_table)
        print(f"  delta_write: {_time.monotonic() - _t5:.1f}s")
    finally:
        spark.conf.set("spark.sql.adaptive.enabled", _aqe_prev)
        for _t in (_stg_xforms, _stg_enc, _stg_final):
            try:
                spark.sql(f"DROP TABLE IF EXISTS {_t}")
            except Exception:
                pass
    del df
    from delta.tables import DeltaTable
    saved = spark.table(output_table)
    _t6 = _time.monotonic()
    _z_cols = [c for c in ["entity_id", TIMESTAMP_COLUMN] if c in saved.columns]
    if _z_cols:
        DeltaTable.forName(spark, output_table).optimize().executeZOrderBy(_z_cols)
    else:
        DeltaTable.forName(spark, output_table).optimize().executeCompaction()
    print(f"  optimize: {_time.monotonic() - _t6:.1f}s")
    _register_feature_table(output_table, saved)
    print(f"  gold_total: {_time.monotonic() - _t0:.1f}s")
    return saved

import json as _gold_json
import traceback as _gold_tb
try:
    result = run_gold()
    _row_count = result.count()
    _col_count = len(result.columns)
    _summary = f"{_row_count:,} rows, {_col_count} columns"
except Exception as _gold_exc:
    print("=" * 70, flush=True)
    print(f"[gold] FATAL: {type(_gold_exc).__name__}: {_gold_exc}", flush=True)
    print("=" * 70, flush=True)
    print(_gold_tb.format_exc(), flush=True)
    dbutils.notebook.exit(_gold_json.dumps({
        "status": "FAILED",
        "stage": "gold",
        "error_type": type(_gold_exc).__name__,
        "error_message": str(_gold_exc)[:2000],
    }))
    raise

from pathlib import Path
from customer_retention.analysis.auto_explorer.run_namespace import RunNamespace
try:
    _exp_dir = dbutils.widgets.get("experiments_dir")
    _run_id = dbutils.widgets.get("run_id")
    _NAMESPACE = RunNamespace(root=Path(_exp_dir), run_id=_run_id) if _exp_dir and _run_id else None
except Exception:
    _NAMESPACE = RunNamespace.from_env_or_latest()
if _NAMESPACE is not None:
    import json
    _meta_cols = {TARGET_COLUMN, TIMESTAMP_COLUMN, "entity_id", "as_of_date", "feature_timestamp"}
    _gold_meta = {
        "rows": _row_count, "columns": _col_count,
        "feature_count": len([c for c in result.columns if c not in _meta_cols]),
        "feature_version": f"v1.0.0_{RECOMMENDATIONS_HASH}" if RECOMMENDATIONS_HASH else "v1.0.0",
    }
    _NAMESPACE.gold_metadata_path.parent.mkdir(parents=True, exist_ok=True)
    _NAMESPACE.gold_metadata_path.write_text(json.dumps(_gold_meta))
    try:
        from customer_retention.stages.causal.interpretation import (
            FeatureLineage as _FeatureLineage,
            build_feature_meta_rows as _build_feature_meta_rows,
            load_column_descriptions_sidecar as _load_column_descriptions_sidecar,
            parse_aggregation_feature_name as _parse_aggregation_feature_name,
            write_feature_meta_sidecar as _write_feature_meta_sidecar,
        )
        # Source table for feature_meta. The pipeline's primary source is
        # the first entry of config.sources; fall back to composite_name
        # for purely derived pipelines. Without this, source_table lands
        # NULL in feature_meta and cascades NULL into v_feature_provenance.
        _PRIMARY_BRONZE_TABLE = bronze_table("{{ config.sources[0].name if config.sources else (config.composite_name or config.name) }}")
        _fm_lineages = []
        for _col in result.columns:
            if _col in _meta_cols:
                continue
            _lineage = _parse_aggregation_feature_name(_col, source_table=_PRIMARY_BRONZE_TABLE)
            if _lineage is None:
                # Defensive fallback — always emit a non-empty source_columns
                # so compile_predicate_prose has a column to look up in
                # column_descriptions even for unrecognised feature patterns.
                _lineage = _FeatureLineage(
                    feature_name=_col,
                    source_columns=[_col],
                    source_table=_PRIMARY_BRONZE_TABLE,
                    aggregation_kind="passthrough",
                )
            _fm_lineages.append(_lineage)
        _fm_rows = _build_feature_meta_rows(
            composite_name=COMPOSITE_NAME,
            lineages=_fm_lineages,
            column_descriptions=_load_column_descriptions_sidecar(_NAMESPACE),
        )
        _write_feature_meta_sidecar(_NAMESPACE, COMPOSITE_NAME, _fm_rows)
    except Exception as _fm_exc:
        print(f"feature_meta sidecar aborted: {type(_fm_exc).__name__}: {_fm_exc}")

# Narrow-projection display: gold is the widest stage (silver ~2620 cols +
# one-hot expansion + label encoding). Rendering the full plan through
# Databricks' display widget serialises the entire logical plan to protobuf,
# which trips Catalyst attribute-id resolution at this width. The Delta table
# is already committed above; operators inspect the full schema via
# spark.table(...) when needed.
_disp = [c for c in ("entity_id", "ACCOUNT_ID", "as_of_date", TARGET_COLUMN if "TARGET_COLUMN" in dir() else None) if c and c in result.columns] or list(result.columns[:10])
display(result.select(*_disp).limit(20))
dbutils.notebook.exit(_summary)
""",
    "databricks_training.py.j2": """# Databricks notebook source
# MAGIC %md
# MAGIC # Training: {{ config.name }}

# COMMAND ----------
{% set best_model_type = config.training.best_model_type if config.training else None %}
{% set production_cv_folds = config.training.production_cv_folds if config.training else None %}
{% set production_full_panel_fit = config.training.production_full_panel_fit if config.training else False %}
{% set feature_spec_path = config.feature_spec_path %}
import json
import logging
import os
import tempfile
import warnings
import csv
from pathlib import Path
import pyspark
import mlflow
import mlflow.spark
from mlflow.models import infer_signature
from pyspark.ml.classification import (
{% if best_model_type is none or best_model_type == "logistic_regression" %}    LogisticRegression,
{% endif %}{% if best_model_type is none or best_model_type == "random_forest" %}    RandomForestClassifier,
{% endif %}{% if best_model_type is none or best_model_type == "xgboost" %}    GBTClassifier,
{% endif %})
from pyspark.ml import PipelineModel
from pyspark.ml.feature import SQLTransformer, VectorAssembler
from pyspark.ml.functions import vector_to_array
from pyspark.ml.linalg import VectorUDT
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, DoubleType
from customer_retention.stages.modeling.feature_profile import FeatureProfile, ColumnProfile, build_feature_profile, compare_feature_profiles
from customer_retention.stages.modeling.shap_attribution import compute_shap_attribution, log_attribution
{% if feature_spec_path %}
from customer_retention.stages.modeling.feature_spec import FeatureSpec
{% endif %}
from customer_retention.analysis.auto_explorer.run_namespace import RunNamespace
from customer_retention.core.compat.timing import log_timing
from customer_retention.core.config.column_config import GOLD_METADATA_COLUMNS
from customer_retention.core.config.experiments import get_mlflow_dfs_tmpdir
{% if config.training and config.training.imbalance_strategy == "smote" %}
from imblearn.over_sampling import SMOTE
{% endif %}

# COMMAND ----------

# MAGIC %run ../config

# COMMAND ----------

logger = logging.getLogger("training")

TARGET = TARGET_COLUMN
_DFS_TMPDIR = get_mlflow_dfs_tmpdir()
_PIP_REQS = [f"pyspark=={pyspark.__version__}"]
_NUMERIC_TYPES = ("double", "float", "integer", "long", "short", "boolean", "byte", "decimal")
_EXCLUDE_COLS = {TARGET, TIMESTAMP_COLUMN, ENTITY_KEY} | GOLD_METADATA_COLUMNS
_vector_schema = StructType([
    StructField("features", VectorUDT(), True),
    StructField("label", DoubleType(), True),
])
{% if config.training and config.training.exploration_feature_profile %}
# Exploration feature profile is loaded from a sibling JSON file written
# next to this script by the renderer (PipelineGeneratorBase._write_training).
# This avoids embedding a 1-5 MB Python dict literal in the source — the
# inline form was a JVM-heap-pressure risk on shared clusters during cell-4
# parse and made the rendered file unreadable to humans.
_EXPLORATION_PROFILE = None
_ep_candidates = []
try:
    _ep_candidates.append(Path(__file__).parent / "exploration_profile.json")
except NameError:
    pass
_ep_candidates.append(Path.cwd() / "training" / "exploration_profile.json")
_ep_candidates.append(Path.cwd() / "exploration_profile.json")
for _ep_candidate in _ep_candidates:
    if _ep_candidate.exists():
        with open(_ep_candidate) as _ep_f:
            _EXPLORATION_PROFILE = json.load(_ep_f)
        break
{% else %}
_EXPLORATION_PROFILE = None
{% endif %}
try:
    _exp_dir = dbutils.widgets.get("experiments_dir")
    _run_id = dbutils.widgets.get("run_id")
    _NAMESPACE = RunNamespace(root=Path(_exp_dir), run_id=_run_id) if _exp_dir and _run_id else None
except Exception:
    _NAMESPACE = RunNamespace.from_env_or_latest()

{% if feature_spec_path %}
_FEATURE_SPEC_PATH = Path(r"{{ feature_spec_path }}")
PRODUCTION_TEST_SIZE = {{ config.training.production_internal_split_test_size if config.training else 0.1 }}
PRODUCTION_RANDOM_STATE = 42
{% else %}
_FEATURE_SPEC_PATH = None
{% endif %}

def _assert_rows(count, stage):
    if count == 0:
        raise ValueError(f"[TRAINING] {stage}: 0 rows remaining — cannot proceed")
    return count

{% if feature_spec_path %}
def _apply_feature_spec_gate(df, spec):
    if spec.verdict.is_hard_block() and not os.environ.get("CR_OVERRIDE_UNSTABLE_SPEC"):
        raise RuntimeError(
            f"FeatureSpec verdict={spec.verdict.status!r}; refusing to train. "
            "Set CR_OVERRIDE_UNSTABLE_SPEC=1 to override."
        )
    if spec.verdict.status == "unstable":
        warnings.warn(
            f"FeatureSpec verdict=unstable (cv_std={spec.verdict.cv_std:.3f}); proceeding.",
            stacklevel=1,
        )
    _leakage_excluded = {e.column for e in spec.leakage_exclusions}
    _leakage_drops = [c for c in df.columns if c in _leakage_excluded]
    if _leakage_drops:
        df = df.drop(*_leakage_drops)
    _missing = [c for c in spec.selected_features if c not in df.columns]
    if _missing:
        # Operator escape hatch: when an upstream phantom-rec or column-rename
        # left a spec'd feature unproduced, raising here blocks training
        # entirely. The orchestrator passes `auto_drop_missing_features` via
        # `_ns_params` (same channel as `experiments_dir`/`run_id`); when it
        # resolves truthy the missing features are warned and dropped so
        # training proceeds. Operator should then add them to
        # `PARITY_IGNORED_FEATURES` post-hoc once the cause is identified.
        try:
            _auto_drop = bool(dbutils.widgets.get("auto_drop_missing_features"))
        except Exception:
            _auto_drop = False
        if _auto_drop:
            warnings.warn(
                f"[TRAINING] auto-dropping {len(_missing)} missing features from spec "
                f"(auto_drop_missing_features=1): {_missing[:20]}",
                stacklevel=1,
            )
            _missing_set = set(_missing)
            spec.selected_features = [
                c for c in spec.selected_features if c not in _missing_set
            ]
        else:
            raise RuntimeError(
                f"FeatureSpec parity violation: gold missing {len(_missing)} declared "
                f"features: {_missing[:10]}. Bronze/silver/gold derivation is out of sync "
                "with exploration — regenerate gold or re-run NB08. "
                "Pass auto_drop_missing_features=1 via the runner's _ns_params to drop "
                "missing features and continue."
            )
    keep = list(spec.selected_features)
    for meta_col in (TARGET, TIMESTAMP_COLUMN, ENTITY_KEY, spec.target_column,
                     spec.entity_column, spec.timestamp_column):
        if meta_col and meta_col in df.columns and meta_col not in keep:
            keep.append(meta_col)
    for c in df.columns:
        if c.startswith("original_") and c not in keep:
            keep.append(c)
    return df.select(*keep), _leakage_drops
{% endif %}

def load_training_data():
    return spark.table(gold_table())

_NAFILLED_SUFFIX = "__nafilled"


def _build_null_filler(feature_cols):
    # SQLTransformer emits a copy of every feature column with NULL and NaN
    # coerced to 0.0, under a `<col>__nafilled` alias. We use SELECT * so the
    # filler never references columns that might be absent at scoring time:
    #   - At training, SELECT * passes entity_id / event_timestamp / label
    #     through untouched and adds the *__nafilled* copies alongside.
    #   - At fe.score_batch time, FE feeds only the feature columns; SELECT *
    #     passes those through and the same *__nafilled* projection runs.
    # The VectorAssembler then reads inputCols=[<col>__nafilled, ...] so the
    # vector is built from the null-safe copies in both worlds.
    #
    # pyspark.ml.Imputer is ruled out by Coding_Practices.md past ~100 cols
    # (serialized model exceeds the 1 GB ceiling). SQLTransformer is a pure
    # Transformer with no .fit() step so it scales to any column count.
    projections = ["*"]
    projections.extend(
        f"COALESCE(nanvl(CAST(`{c}` AS DOUBLE), 0.0), 0.0) AS `{c}{_NAFILLED_SUFFIX}`"
        for c in feature_cols
    )
    return SQLTransformer(
        statement="SELECT " + ", ".join(projections) + " FROM __THIS__"
    )


def prepare_features(df):
    exclude_prefixes = ["original_"]
    feature_cols = [
        c for c in df.columns
        if c not in _EXCLUDE_COLS and not any(c.startswith(p) for p in exclude_prefixes)
        and df.schema[c].dataType.typeName() in _NUMERIC_TYPES
    ]
    if not feature_cols:
        col_types = {c: df.schema[c].dataType.typeName() for c in df.columns}
        raise ValueError(f"[TRAINING] No numeric feature columns found. Column types: {col_types}")
    filler = _build_null_filler(feature_cols)
    filled = filler.transform(df)
    filled_cols = [f"{c}{_NAFILLED_SUFFIX}" for c in feature_cols]
    assembler = VectorAssembler(
        inputCols=filled_cols, outputCol="features", handleInvalid="keep"
    )
    keep = ["features", F.col(TARGET).alias("label")]
    if TIMESTAMP_COLUMN in df.columns:
        keep.append(TIMESTAMP_COLUMN)
    assembled = assembler.transform(filled).select(*keep)
    return assembled, feature_cols, filler, assembler

def _temporal_split(assembled, test_size):
    import datetime

    cutoff_ts = assembled.select(
        F.percentile_approx(F.unix_timestamp(F.col(TIMESTAMP_COLUMN)), 1.0 - test_size).alias("cutoff")
    ).collect()[0]["cutoff"]
    cutoff_date = datetime.datetime.fromtimestamp(cutoff_ts)

    # Entity-grouped split — fully distributed via deterministic hash.
    # No .collect() — scales to 400K+ entities without driver OOM.
    test_pct = int(test_size * 100)
    bucket = F.abs(F.hash(F.col(ENTITY_KEY), F.lit(42))) % F.lit(100)
    test_df = assembled.filter(bucket < F.lit(test_pct))
    train_df = assembled.filter(bucket >= F.lit(test_pct))
{% if config.training and config.training.purge_gap_days %}
    purge_gap_days = {{ config.training.purge_gap_days }}
    train_df = train_df.filter(F.col(TIMESTAMP_COLUMN) < F.date_sub(F.lit(cutoff_date), purge_gap_days))
{% endif %}
    return train_df, test_df, cutoff_date

def _extract_feature_importance(fitted, feature_cols):
    if hasattr(fitted, "featureImportances"):
        importances = fitted.featureImportances.toArray()
    elif hasattr(fitted, "coefficients"):
        importances = [abs(v) for v in fitted.coefficients.toArray()]
    else:
        return None
    return sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True)

def _log_feature_importance(fitted, feature_cols):
    pairs = _extract_feature_importance(fitted, feature_cols)
    if pairs is None:
        return
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        writer = csv.writer(f)
        writer.writerow(["feature", "importance"])
        writer.writerows(pairs)
        tmp_path = f.name
    mlflow.log_artifact(tmp_path, "feature_importance")

def _log_training_progress(fitted, name):
    if hasattr(fitted, "summary") and hasattr(fitted.summary, "objectiveHistory"):
        for step, loss in enumerate(fitted.summary.objectiveHistory):
            mlflow.log_metric(f"{name}_loss", loss, step=step)

def _evaluate_model(predictions):
    roc_eval = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC")
    pr_eval = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderPR")
    acc_eval = MulticlassClassificationEvaluator(labelCol="label", metricName="accuracy")
    f1_eval = MulticlassClassificationEvaluator(labelCol="label", metricName="f1")
    prec_eval = MulticlassClassificationEvaluator(labelCol="label", metricName="weightedPrecision")
    rec_eval = MulticlassClassificationEvaluator(labelCol="label", metricName="weightedRecall")
    return {
        "roc_auc": roc_eval.evaluate(predictions),
        "pr_auc": pr_eval.evaluate(predictions),
        "accuracy": acc_eval.evaluate(predictions),
        "f1": f1_eval.evaluate(predictions),
        "weighted_precision": prec_eval.evaluate(predictions),
        "weighted_recall": rec_eval.evaluate(predictions),
    }

def _mlflow_evaluate_predictions(predictions):
    _MAX_EVAL_ROWS = 100_000
    eval_pdf = predictions.select(
        F.col("label"),
        vector_to_array(F.col("probability"))[1].alias("prob_1"),
        F.col("prediction"),
    ).limit(_MAX_EVAL_ROWS).toPandas()
    mlflow.evaluate(
        data=eval_pdf,
        model_type="classifier",
        targets="label",
        predictions="prediction",
        extra_metrics=None,
    )

def _log_best_model(model, df, feature_cols, test_df, filler, assembler):
    _registered_name = f"{CATALOG}.{SCHEMA}.model_{COMPOSITE_NAME}"
    _logged_pipeline = PipelineModel(stages=[filler, assembler, model])
    try:
        from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup
        fe = FeatureEngineeringClient()
        ts_key = TIMESTAMP_COLUMN if TIMESTAMP_COLUMN in df.columns else None
        entity_cols = ["entity_id"] + ([TIMESTAMP_COLUMN] if ts_key else [])
        lookups = [FeatureLookup(
            table_name=gold_table(), feature_names=feature_cols,
            lookup_key="entity_id", timestamp_lookup_key=ts_key,
        )]
        training_set = fe.create_training_set(
            df=df.select(*entity_cols, TARGET), feature_lookups=lookups,
            label=TARGET, exclude_columns=entity_cols,
        )
        fe.log_model(
            model=_logged_pipeline, artifact_path="best_model", flavor=mlflow.spark,
            training_set=training_set,
            registered_model_name=_registered_name,
        )
        print(f"[TRAINING] Model registered: {_registered_name}")
    except ImportError:
        _raw_sample = df.select(*feature_cols, F.col(TARGET).alias("label")).limit(5)
        _sig = infer_signature(_raw_sample, _logged_pipeline.transform(_raw_sample))
        mlflow.spark.log_model(
            _logged_pipeline, "best_model", dfs_tmpdir=_DFS_TMPDIR,
            pip_requirements=_PIP_REQS, signature=_sig,
            registered_model_name=_registered_name,
        )
        print(f"[TRAINING] Model registered (fallback path): {_registered_name}")
    return _registered_name


def _promote_to_production(registered_name, parent_run_id):
    from mlflow.tracking import MlflowClient as _MlflowClient
    _client = _MlflowClient()
    _versions = _client.search_model_versions(f"name='{registered_name}'")
    if not _versions:
        print(f"[TRAINING] WARNING: no model versions found for {registered_name}")
        return None
    _matching = [v for v in _versions if v.run_id == parent_run_id]
    _candidates = _matching or _versions
    _latest = max(_candidates, key=lambda v: int(v.version))
    if not _matching:
        print(f"[TRAINING] No version matched run_id={parent_run_id}; using latest v{_latest.version}")
    _client.set_registered_model_alias(registered_name, "production", _latest.version)
    print(f"[TRAINING] Alias @production -> {registered_name} v{_latest.version}")
    return _latest.version

def _resolve_target_column(df_columns, requested):
    # Resolve TARGET_COLUMN against the actual gold schema. The configured
    # `requested` literal can drift from gold reality when landing's
    # original_target_column rename was unset, or a case mismatch
    # ("Churned" vs "churned") leaks through, or silver's prefix collision
    # moved the target under `<ds>__<col>`. Priority: exact > case-insensitive
    # > original_<target> prefix > substring > fuzzy. Raises with a label-
    # shaped diagnostic when no candidate is unambiguous.
    import difflib

    if requested in df_columns:
        return requested
    _lower = {c.lower(): c for c in df_columns}
    if requested.lower() in _lower:
        resolved = _lower[requested.lower()]
        print(f"[TRAINING] target_column resolver: case-insensitive {requested!r} -> {resolved!r}")
        return resolved
    _ranked, _seen = [], set()
    def _add(_name, _why):
        if _name and _name not in _seen and _name in df_columns:
            _ranked.append((_name, _why)); _seen.add(_name)
    _orig = f"original_{requested}"
    _add(_orig if _orig in df_columns else _lower.get(_orig.lower()), "original_prefix")
    for _c in df_columns:
        if requested.lower() in _c.lower():
            _add(_c, "substring")
    for _c in difflib.get_close_matches(requested, list(df_columns), n=5, cutoff=0.6):
        _add(_c, "fuzzy")
    if len(_ranked) == 1:
        resolved, why = _ranked[0]
        print(f"[TRAINING] target_column resolver: {requested!r} -> {resolved!r} ({why})")
        return resolved
    _label_shaped = [c for c in df_columns if any(t in c.lower() for t in
                     ("churn", "label", "target", "flag", "is_", "has_", "_y"))]
    if not _ranked:
        raise RuntimeError(
            f"[TRAINING] target_column {requested!r} not found in gold and no "
            f"unambiguous candidate exists. Label-shaped cols ({len(_label_shaped)}): "
            f"{_label_shaped[:30]}. First 30 cols: {list(df_columns)[:30]}. "
            "Fix the layer that dropped the target (typically landing's "
            "original_target_column rename or silver's prefix collision)."
        )
    raise RuntimeError(
        f"[TRAINING] target_column {requested!r} ambiguous in gold — "
        f"candidates: {[c for c, _ in _ranked]}. Pin TARGET_COLUMN in "
        "config.py to disambiguate, or regenerate findings with the "
        "correct target_column."
    )


def train_and_evaluate():
    global TARGET, _EXCLUDE_COLS
    _results = {"models": {}, "feature_profile": {}}
    with log_timing("load_gold_table", logger):
        df = load_training_data()
    raw_count = _assert_rows(df.count(), "gold_table")
    # Runtime target_column resolution: gold's actual schema is the
    # source-of-truth. The configured TARGET literal can diverge when
    # landing's original_target_column rename was unset, when case mismatch
    # leaks through, or when silver's prefix collision moved the column.
    # Rebind TARGET (and refresh _EXCLUDE_COLS so the resolved name is
    # excluded from feature_cols too) before any df.filter touches it.
    TARGET = _resolve_target_column(df.columns, TARGET_COLUMN)
    _EXCLUDE_COLS = {TARGET, TIMESTAMP_COLUMN, ENTITY_KEY} | GOLD_METADATA_COLUMNS
    col_types = {}
    for field in df.schema.fields:
        col_types.setdefault(field.dataType.typeName(), []).append(field.name)
    type_summary = {t: len(cols) for t, cols in sorted(col_types.items())}
    print(f"[TRAINING] Gold table: {raw_count:,} rows, {len(df.columns)} columns")
    print(f"[TRAINING] Column types: {type_summary}")
    _results["gold_data"] = {"rows": raw_count, "columns": len(df.columns), "column_types": type_summary}

{% if config.training and config.training.recommended_training_start %}
    if TIMESTAMP_COLUMN in df.columns:
        df = df.filter(F.col(TIMESTAMP_COLUMN) >= F.lit("{{ config.training.recommended_training_start }}"))
        after_start = df.count()
        print(f"[TRAINING] After training_start filter: {after_start:,} rows (dropped {raw_count - after_start:,})")
        _assert_rows(after_start, "after training_start filter")
{% endif %}
{% if config.training and config.training.filter_future_dates %}
    if TIMESTAMP_COLUMN in df.columns:
        df = df.filter(F.col(TIMESTAMP_COLUMN) <= F.current_timestamp())
{% endif %}
    df = df.filter(F.col(TARGET).isNotNull())
    filtered_count = _assert_rows(df.count(), "after_null_label_filter")
    print(f"[TRAINING] After null-label filter: {filtered_count:,} rows")

{% if feature_spec_path %}
    with log_timing("feature_spec_gate", logger):
        _SPEC = FeatureSpec.load(_FEATURE_SPEC_PATH)
        if _NAMESPACE is not None:
            _runtime_spec_path = _NAMESPACE.feature_spec_path
            if _runtime_spec_path != _FEATURE_SPEC_PATH and not _runtime_spec_path.exists():
                import shutil as _spec_shutil
                _runtime_spec_path.parent.mkdir(parents=True, exist_ok=True)
                _spec_shutil.copy2(_FEATURE_SPEC_PATH, _runtime_spec_path)
                print(f"[TRAINING] FeatureSpec snapshotted to {_runtime_spec_path} for parity report")
        df, _spec_leakage_drops = _apply_feature_spec_gate(df, _SPEC)
        print(
            f"[TRAINING] FeatureSpec applied: {len(_SPEC.selected_features)} features, "
            f"verdict={_SPEC.verdict.status}, leakage_drops={len(_spec_leakage_drops)}"
        )
{% endif %}

    with log_timing("prepare_features", logger):
        assembled, feature_cols, _filler, _assembler = prepare_features(df)
    assembled_count = _assert_rows(assembled.count(), "after_assembly")
    print(f"[TRAINING] Assembled: {assembled_count:,} rows, {len(feature_cols)} features")
    _results["feature_count"] = len(feature_cols)
    _results["filtered_rows"] = filtered_count

    with log_timing("feature_profile", logger):
        null_exprs = [F.sum(F.when(F.col(c).isNull(), 1).otherwise(0)).alias(c) for c in feature_cols]
        null_row = df.select(null_exprs).collect()[0]
        feature_stats = {}
        excluded_cols = {}
        for c in df.columns:
            if c in _EXCLUDE_COLS:
                excluded_cols[c] = "metadata"
            elif any(c.startswith(p) for p in ["original_"]):
                excluded_cols[c] = "original_prefix"
{%- if config.gold.feature_selections %}
        for _fs_col in {{ config.gold.feature_selections }}:
            excluded_cols[_fs_col] = "feature_selection"
{%- endif %}
{% if feature_spec_path %}
        for _c in _spec_leakage_drops:
            excluded_cols[_c] = "leakage_exclusion"
{% else %}
        if _NAMESPACE is not None:
            _rec_path = _NAMESPACE.merged_recommendations_path
            if _rec_path.exists():
                import yaml as _rec_yaml
                from customer_retention.analysis.auto_explorer.layered_recommendations import RecommendationRegistry
                with _rec_path.open() as _rf:
                    _drop_recs = RecommendationRegistry.from_dict(_rec_yaml.safe_load(_rf))
                _runtime_drops = set()
                for _rec in getattr(getattr(_drop_recs, 'gold', None), 'feature_selection', []):
                    if _rec.action in ('drop_multicollinear', 'drop_weak', 'drop_l1_zero', 'drop_chi_squared', 'drop_gbdt_importance', 'drop_rescue_consensus', 'drop_availability', 'drop_zero_variance'):
                        excluded_cols[_rec.target_column] = _rec.action
                    if _rec.action in ('drop_l1_zero', 'drop_chi_squared', 'drop_gbdt_importance', 'drop_rescue_consensus', 'drop_zero_variance'):
                        _runtime_drops.add(_rec.target_column)
                _actual_runtime = [c for c in _runtime_drops if c in feature_cols]
                if _actual_runtime:
                    df = df.drop(*_actual_runtime)
                    feature_cols = [c for c in feature_cols if c not in _runtime_drops]
                    print(f"[TRAINING] Runtime L1/variance drops: {len(_actual_runtime)} features")
{% endif %}
        for c in feature_cols:
            null_count = int(null_row[c])
            feature_stats[c] = ColumnProfile(dtype=df.schema[c].dataType.typeName(), non_null_count=filtered_count - null_count, null_count=null_count)
        prod_profile = build_feature_profile("production", TARGET, filtered_count, feature_stats, excluded_cols)
        print(f"[TRAINING] Production profile: {prod_profile.feature_count} features, {filtered_count:,} rows")
        if _NAMESPACE is not None:
            prod_profile.save(_NAMESPACE.production_feature_profile_path)
            print(f"[TRAINING] Production profile saved to {_NAMESPACE.production_feature_profile_path}")
        _results["feature_profile"]["production_features"] = prod_profile.feature_count
        _results["feature_profile"]["production_rows"] = filtered_count
        _results["feature_profile"]["excluded_details"] = excluded_cols
        if _EXPLORATION_PROFILE is not None:
            _exp_feats = _EXPLORATION_PROFILE.get("features", {})
            _exp_excl = _EXPLORATION_PROFILE.get("excluded", {})
            _prod_feature_set = set(feature_cols)
            for _pc in _prod_feature_set:
                if _pc not in _exp_feats and _pc not in _exp_excl:
                    _exp_feats[_pc] = {"dtype": "float32", "non_null": _EXPLORATION_PROFILE.get("row_count", 0), "null_count": 0}
            for _sc in [c for c in list(_exp_feats) if c not in _prod_feature_set and c not in _exp_excl]:
                _exp_excl[_sc] = "not_in_pipeline"
                del _exp_feats[_sc]
            _EXPLORATION_PROFILE["features"] = _exp_feats
            _EXPLORATION_PROFILE["excluded"] = _exp_excl
            _EXPLORATION_PROFILE["feature_count"] = len(_exp_feats)
            exp_profile = FeatureProfile.from_dict(_EXPLORATION_PROFILE)
            discrepancies = compare_feature_profiles(exp_profile, prod_profile)
            _results["feature_profile"]["exploration_features"] = exp_profile.feature_count
            _results["feature_profile"]["discrepancies"] = discrepancies
            if discrepancies:
                print(f"[TRAINING] WARNING: {len(discrepancies)} feature discrepancies vs exploration:")
                for d in discrepancies:
                    print(f"[TRAINING]   {d}")
            else:
                print("[TRAINING] Feature profile matches exploration")
        else:
            print("[TRAINING] WARNING: No exploration feature profile available for comparison")
        if _NAMESPACE is not None and '_drop_recs' in dir():
            _gold_fs = getattr(getattr(_drop_recs, 'gold', None), 'feature_selection', [])
            _prioritized_rationales = {r.target_column: r.rationale for r in _gold_fs if r.action == 'prioritize'}
            _correlated_map = {}
            for _r in _gold_fs:
                if _r.action == 'drop_multicollinear' and _r.parameters:
                    _correlated_map.setdefault(_r.target_column, []).append(
                        (_r.parameters.get('correlated_with', ''), _r.parameters.get('correlation', 0)))
            _dropped_strong = {c: _prioritized_rationales[c] for c in _prioritized_rationales if c in excluded_cols}
            if _dropped_strong:
                print(f"[TRAINING] NOTE: {len(_dropped_strong)} strong predictors dropped (signal preserved via correlated survivors):")
                for _c in sorted(_dropped_strong):
                    _survivors = [f"{p} (r={r:.2f})" for p, r in _correlated_map.get(_c, []) if p not in excluded_cols]
                    _surv_str = ", ".join(_survivors[:3]) if _survivors else "none found"
                    print(f"[TRAINING]   {_c}: {_dropped_strong[_c]} -> survived by: {_surv_str}")

{% if production_full_panel_fit %}
    # Full-panel fit: skip temporal split. Both `train_df` and `test_df`
    # bind to the full assembled frame so per-model evaluation reports
    # train metrics (no holdout, no CV). Triggered by
    # `TrainingConfig.production_full_panel_fit=True`.
    train_df = assembled
    test_df = assembled
    cutoff_date = None
    train_count = _assert_rows(train_df.count(), "full_panel_fit")
    test_count = train_count
    print(f"[TRAINING] Full-panel fit: train={train_count:,} rows (no holdout, no CV)")
    split_info = {"mode": "full_panel_fit", "train_count": train_count, "test_count": 0}
    _results["split"] = split_info
{% else %}
    with log_timing("temporal_split", logger):
        train_df, test_df, cutoff_date = _temporal_split(assembled, {% if feature_spec_path %}PRODUCTION_TEST_SIZE{% else %}{{ config.training.test_size if config.training else 0.2 }}{% endif %})
    train_count = _assert_rows(train_df.count(), "train_set_after_split")
    test_count = _assert_rows(test_df.count(), "test_set_after_split")
    print(f"[TRAINING] Split: train={train_count:,}, test={test_count:,}")
    split_info = {"cutoff_date": str(cutoff_date), "train_count": train_count, "test_count": test_count}
    print(f"[TRAINING] Split info: {split_info}")
    _results["split"] = split_info
{% endif %}

    label_dist = {float(row["label"]): row["count"] for row in train_df.groupBy("label").count().collect()}
    print(f"[TRAINING] Label distribution: {label_dist}")
    _results["label_distribution"] = label_dist
    if len(label_dist) < 2:
        raise ValueError(f"[TRAINING] Only {len(label_dist)} class(es) — Need at least 2 for binary classification")

    mlflow.autolog(disable=True)
    mlflow.end_run()
    mlflow.set_experiment(f"/Shared/training_{COMPOSITE_NAME}")

{% if config.training and config.training.imbalance_strategy == "class_weight" %}
    label_counts = train_df.groupBy("label").count().collect()
    total = sum(row["count"] for row in label_counts)
    n_classes = len(label_counts)
    weight_map = {row["label"]: total / (n_classes * row["count"]) for row in label_counts}
    from pyspark.sql.functions import udf
    from pyspark.sql.types import DoubleType
    weight_udf = udf(lambda label: float(weight_map.get(label, 1.0)), DoubleType())
    train_df = train_df.withColumn("class_weight", weight_udf(F.col("label")))
{% endif %}
{% if config.training and config.training.imbalance_strategy == "smote" %}
    train_pdf = train_df.toPandas()
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(
        train_pdf["features"].apply(lambda v: v.toArray()).tolist(),
        train_pdf["label"],
    )
    import pandas as pd
    from pyspark.ml.linalg import Vectors
    resampled_pdf = pd.DataFrame({"features": [Vectors.dense(x) for x in X_resampled], "label": y_resampled})
    train_df = spark.createDataFrame(resampled_pdf, schema=_vector_schema)
{% endif %}

{% set weight_param = ', weightCol="class_weight"' if config.training and config.training.imbalance_strategy == "class_weight" else '' %}
    models = {}
{% if best_model_type is none or best_model_type == "logistic_regression" %}
    models["LogisticRegression"] = LogisticRegression(maxIter=100, featuresCol="features", labelCol="label"{{ weight_param }})
{% endif %}
{% if best_model_type is none or best_model_type == "random_forest" %}
    models["RandomForest"] = RandomForestClassifier(numTrees=100, featuresCol="features", labelCol="label"{{ weight_param }})
{% endif %}
{% if best_model_type is none or best_model_type == "xgboost" %}
    models["GBTClassifier"] = GBTClassifier(maxIter=50, featuresCol="features", labelCol="label"{{ weight_param }})
{% endif %}

    best_model_name = None
    best_auc = -1.0
    best_model = None
    best_metrics = {}
    _logged_models = []
    _registered_model_name = ""
    _registered_model_version = None

    _experiment_name = f"/Shared/training_{COMPOSITE_NAME}"
    with mlflow.start_run(run_name=f"training_{COMPOSITE_NAME}") as _parent_run:
        mlflow.set_tag("composite_name", COMPOSITE_NAME)
        mlflow.set_tag("pipeline_name", PIPELINE_NAME)
        mlflow.set_tag("target_column", TARGET)
        mlflow.set_tag("entity_key", "entity_id")
        mlflow.set_tag("timestamp_column", TIMESTAMP_COLUMN)
        if RECOMMENDATIONS_HASH:
            mlflow.set_tag("recommendations_hash", RECOMMENDATIONS_HASH)
{% if production_full_panel_fit %}
        mlflow.set_tag("training_mode", "full_panel_fit")
{% else %}
        mlflow.set_tag("training_mode", "temporal_split")
{% endif %}
        mlflow.log_params({"train_samples": train_count, "test_samples": test_count, "n_features": len(feature_cols)})

        for name, model in models.items():
            with mlflow.start_run(run_name=name, nested=True) as _nested_run:
                with log_timing(f"fit_{name}", logger, train_rows=train_count):
                    fitted = model.fit(train_df)
                _log_training_progress(fitted, name)
                mlflow.log_param("model_type", name)
                mlflow.log_param("num_features", len(feature_cols))
                predictions = fitted.transform(test_df)
                _sig = infer_signature(test_df, predictions)
                _artifact_path = f"model_{name}"
                _log_info = mlflow.spark.log_model(fitted, _artifact_path, dfs_tmpdir=_DFS_TMPDIR, pip_requirements=_PIP_REQS, signature=_sig)
                _logged_models.append({
                    "artifact_path": _artifact_path,
                    "model_uri": _log_info.model_uri,
                    "flavor": "spark",
                    "run_id": _nested_run.info.run_id,
                    "display_name": name,
                    "wrapper_meta_artifact_path": None,
                })
                metrics = _evaluate_model(predictions)
                print(f"[TRAINING] {name}: AUC={metrics['roc_auc']:.4f}, PR-AUC={metrics['pr_auc']:.4f}, F1={metrics['f1']:.4f}")
                _results["models"][name] = metrics
                mlflow.log_metrics(metrics)
                _log_feature_importance(fitted, feature_cols)
                try:
                    _mlflow_evaluate_predictions(predictions)
                except Exception as _eval_err:
                    print(f"[TRAINING] WARNING: MLflow evaluate failed for {name}: {_eval_err}")
                if metrics["roc_auc"] > best_auc:
                    best_auc = metrics["roc_auc"]
                    best_model_name = name
                    best_model = fitted
                    best_metrics = metrics

        if best_model_name is not None:
            mlflow.set_tag("best_model", best_model_name)
            mlflow.log_metric("best_roc_auc", best_auc)
            mlflow.log_metrics({f"best_{k}": v for k, v in best_metrics.items()})
        with tempfile.TemporaryDirectory() as _tmp_dir:
            _features_path = str(Path(_tmp_dir) / "features.json")
            with open(_features_path, "w") as f:
                json.dump({"feature_columns": feature_cols, "count": len(feature_cols)}, f)
            mlflow.log_artifact(_features_path)
        if best_model is not None:
            _registered_model_name = _log_best_model(best_model, df, feature_cols, test_df, _filler, _assembler)
            _registered_model_version = _promote_to_production(_registered_model_name, _parent_run.info.run_id)
            with log_timing("shap_attribution", logger):
                _attribution = compute_shap_attribution(
                    pipeline_model=PipelineModel(stages=[_filler, _assembler, best_model]),
                    df=df,
                    feature_columns=feature_cols,
                )
                log_attribution(_attribution)
            print(f"[TRAINING] SHAP attribution logged: {len(_attribution.importances)} features, sample_size={_attribution.sample_size}")

    _results["best_model"] = best_model_name
    _results["best_roc_auc"] = best_auc
    _results["registered_model_name"] = _registered_model_name
    _results["registered_model_version"] = _registered_model_version

    if _NAMESPACE is not None:
        _training_meta = {
            "pipeline_name": PIPELINE_NAME,
            "mlflow_experiment_name": _experiment_name,
            "mlflow_run_id": _parent_run.info.run_id,
            "composite_name": COMPOSITE_NAME,
            "target_column": TARGET,
            "entity_key": "entity_id",
            "timestamp_column": TIMESTAMP_COLUMN,
            "recommendations_hash": RECOMMENDATIONS_HASH or "",
            "best_model_name": best_model_name,
            "best_roc_auc": best_auc,
            "feature_columns": feature_cols,
            "logged_models": _logged_models,
            "registered_model_name": _registered_model_name,
            "registered_model_version": _registered_model_version,
        }
        _NAMESPACE.training_metadata_path.parent.mkdir(parents=True, exist_ok=True)
        _NAMESPACE.training_metadata_path.write_text(json.dumps(_training_meta))
        print(f"[TRAINING] Metadata saved to {_NAMESPACE.training_metadata_path}")

{% if feature_spec_path %}
    if _NAMESPACE is not None:
        _train_label_rate = train_df.filter(F.col("label") == 1).count() / max(train_df.count(), 1)
        _test_label_rate = test_df.filter(F.col("label") == 1).count() / max(test_df.count(), 1)
        _prod_diag = {
            "run_type": "production",
            "exploration_run_id": _SPEC.exploration_run_id,
            "feature_spec_source_path": str(_FEATURE_SPEC_PATH),
            "feature_count": len(feature_cols),
            "feature_names": list(feature_cols),
            "split": {**_results.get("split", {}), "train": train_count, "test": test_count},
            "test_metrics": {n: v for n, v in _results["models"].items()},
            "label_distribution": _results.get("label_distribution", {}),
            "label_rate_train": float(_train_label_rate),
            "label_rate_test": float(_test_label_rate),
            "best_model_name": best_model_name,
            "best_roc_auc": best_auc,
        }
        _NAMESPACE.production_diagnostics_path.write_text(json.dumps(_prod_diag, default=str))
        print(f"[TRAINING] Production diagnostics saved to {_NAMESPACE.production_diagnostics_path}")
{% endif %}

    return _results

# COMMAND ----------

import traceback as _training_tb
try:
    _training_results = train_and_evaluate()
    if isinstance(_training_results, dict):
        _training_results.setdefault("status", "OK")
    print("\\n" + "=" * 60)
    print("TRAINING RESULTS")
    print("=" * 60)
    print(json.dumps(_training_results, indent=2, default=str))
except Exception as _training_exc:
    print("=" * 70, flush=True)
    print(f"[training] FATAL: {type(_training_exc).__name__}: {_training_exc}", flush=True)
    print("=" * 70, flush=True)
    print(_training_tb.format_exc(), flush=True)
    dbutils.notebook.exit(json.dumps({
        "status": "FAILED",
        "stage": "training",
        "error_type": type(_training_exc).__name__,
        "error_message": str(_training_exc)[:2000],
    }))
    raise
dbutils.notebook.exit(json.dumps(_training_results, default=str))
""",
    "databricks_landing.py.j2": """# Databricks notebook source
# MAGIC %md
# MAGIC # Landing: {{ name }}

# COMMAND ----------

from pyspark.sql import functions as F

# COMMAND ----------

# MAGIC %run ../config

# COMMAND ----------

# Disable Delta's advisory zorder stats-collection check. The check fires
# when a zorder target column falls outside the default 32-column stats
# window; without this, executeZOrderBy aborts AFTER a successful write
# with [DELTA_ZORDERING_ON_COLUMN_WITHOUT_STATS]. Advisory only — data
# correctness and skipping behavior are unaffected.
spark.conf.set("spark.databricks.delta.optimize.zorder.checkStatsCollection.enabled", "false")

# COMMAND ----------

SOURCE_NAME = "{{ name }}"
ENTITY_COLUMN = "{{ config.entity_column }}"
TIME_COLUMN = "{{ config.time_column }}"

def load_source():
    source_config = RAW_SOURCES[SOURCE_NAME]
    return read_raw_source(source_config["path"], source_config["format"])

def derive_feature_timestamp(df):
    \"\"\"Source: Data Discovery > Timestamp Detection\"\"\"
{%- if config.timestamp_coalesce %}
{%- set cols = config.timestamp_coalesce.datetime_columns_ordered %}
    df = df.withColumn("feature_timestamp", F.coalesce(
{%- for col in cols %}
        F.to_timestamp(F.col("{{ col }}")){{ "," if not loop.last else "" }}
{%- endfor %}
    ))
{%- else %}
    if TIME_COLUMN in [f.name for f in df.schema.fields]:
        df = df.withColumn("feature_timestamp", F.to_timestamp(F.col(TIME_COLUMN)))
    elif "feature_timestamp" not in [f.name for f in df.schema.fields]:
        raise ValueError(f"Time column '{TIME_COLUMN}' not found. Available: {df.columns}")
{%- endif %}
    return df

def derive_label_timestamp(df):
    \"\"\"Source: Intent Contract > Label Horizon\"\"\"
{%- if config.label_timestamp %}
{%- set lt = config.label_timestamp %}
{%- if lt.label_column %}
    label_ts = F.to_timestamp(F.col("{{ lt.label_column }}"))
    fallback_ts = F.expr(f"feature_timestamp + INTERVAL {{ lt.fallback_window_days }} DAYS")
    df = df.withColumn("label_timestamp", F.coalesce(label_ts, fallback_ts))
{%- else %}
    df = df.withColumn("label_timestamp", F.expr(f"feature_timestamp + INTERVAL {{ lt.fallback_window_days }} DAYS"))
{%- endif %}
{%- else %}
    df = df.withColumn("label_timestamp", F.expr("feature_timestamp + INTERVAL 180 DAYS"))
{%- endif %}
    return df

def derive_label_available_flag(df):
    if TARGET_COLUMN in [f.name for f in df.schema.fields]:
        df = df.withColumn("label_available_flag", F.col(TARGET_COLUMN).isNotNull())
    else:
        df = df.withColumn("label_available_flag", F.lit(False))
    return df

{%- if config.datetime_derivation %}

DATETIME_DERIVATION_SOURCES = {{ config.datetime_derivation.source_columns }}
MASK_FUTURE_COLUMNS = {{ config.datetime_derivation.mask_future_columns }}

def derive_datetime_features(df):
    \"\"\"Source: Data Discovery > Datetime Feature Derivation\"\"\"
    ref_col = "{{ config.datetime_derivation.reference_column }}"
    mask_set = set(MASK_FUTURE_COLUMNS)
    for col in DATETIME_DERIVATION_SOURCES:
        if col not in [f.name for f in df.schema.fields]:
            continue
        ts_col = F.to_timestamp(F.col(col))
        ref_ts = F.to_timestamp(F.col(ref_col))
        delta_hours = (F.unix_timestamp(ts_col) - F.unix_timestamp(ref_ts)) / 3600.0
        hour_val = F.hour(ts_col).cast("double")
        dow_val = (F.dayofweek(ts_col) - 1).cast("double")
        is_weekend_val = F.when(F.dayofweek(ts_col).isin(1, 7), 1.0).otherwise(0.0)
        if col in mask_set:
            future_mask = ts_col > ref_ts
            df = df.withColumn(f"{col}_delta_hours", F.when(future_mask, None).otherwise(delta_hours))
            df = df.withColumn(f"{col}_hour", F.when(future_mask, None).otherwise(hour_val))
            df = df.withColumn(f"{col}_dow", F.when(future_mask, None).otherwise(dow_val))
            df = df.withColumn(f"{col}_is_weekend", F.when(future_mask, None).otherwise(is_weekend_val))
        else:
            df = df.withColumn(f"{col}_delta_hours", delta_hours)
            df = df.withColumn(f"{col}_hour", hour_val)
            df = df.withColumn(f"{col}_dow", dow_val)
            df = df.withColumn(f"{col}_is_weekend", is_weekend_val)
    return df
{%- endif %}
{%- if config.history_window %}

def apply_history_window(df):
    \"\"\"Source: Temporal Deep Dive > History Window\"\"\"
{%- if config.history_window.upper_limit %}
    upper = F.lit("{{ config.history_window.upper_limit }}").cast("timestamp")
{%- else %}
    upper = df.agg(F.max("feature_timestamp")).collect()[0][0]
{%- endif %}
{%- if config.history_window.lookback_periods %}
    lookback_days = {{ config.history_window.lookback_periods }} * {{ config.history_window.cadence_days }}
    lower = F.date_sub(F.lit(upper), lookback_days)
    df = df.filter(F.col("feature_timestamp").isNull() | (F.col("feature_timestamp") >= lower))
{%- endif %}
{%- if config.history_window.upper_limit %}
    df = df.filter(F.col("feature_timestamp").isNull() | (F.col("feature_timestamp") <= upper))
{%- endif %}
    return df
{%- endif %}

{%- if config.key_resolution_steps %}

# COMMAND ----------

def resolve_entity_key(df):
    \"\"\"Source: Intent Contract > Key Resolution\"\"\"
{%- for step in config.key_resolution_steps %}
    _bridge = spark.read.format("delta").table(landing_table("{{ step.bridge_dataset }}"))
    # Alias the bridge key as `__cr_bridge_<KEY>` so the inner join never produces
    # two physical columns named identically when ``source_key == bridge_key``.
    # UC's case-insensitive duplicate detection rejects such schemas at
    # saveAsTable; aliasing + unconditional drop sidesteps the collision.
    _bridge = _bridge.select(F.col("{{ step.bridge_key }}").alias("__cr_bridge_{{ step.bridge_key }}"), "{{ step.resolve_column }}").dropDuplicates(["__cr_bridge_{{ step.bridge_key }}"])
    df = df.join(_bridge, df["{{ step.source_key }}"] == _bridge["__cr_bridge_{{ step.bridge_key }}"], "inner")
    df = df.drop("__cr_bridge_{{ step.bridge_key }}")
{%- endfor %}
    return df
{%- endif %}

# COMMAND ----------

def run_landing():
    df = load_source()
{%- if config.drop_columns %}
    # Case-EXACT drops (NB10 LANDING_DROP_COLUMNS_OVERRIDES). `df.drop(name)` is
    # case-INSENSITIVE under default `spark.sql.caseSensitive=false`; using
    # `df.select(*[c for c in df.columns if c not in <set>])` lets a Snowflake
    # quoted-identifier `"opportunity_id"` be dropped without also dropping
    # the case-folded `OPPORTUNITY_ID`.
    _drop_set = set({{ config.drop_columns | python_repr }})
    df = df.select(*[c for c in df.columns if c not in _drop_set])
{%- endif %}
{%- if config.raw_time_column %}
    df = df.withColumnRenamed("{{ config.raw_time_column }}", TIME_COLUMN)
{%- endif %}
{%- if config.original_target_column %}
    df = df.withColumnRenamed("{{ config.original_target_column }}", TARGET_COLUMN)
{%- endif %}
{%- if config.filters %}
{%- set _all_sibling_views = [] %}
{%- for step in config.filters %}
{%- for v in (step.parameters.sibling_views or []) %}
{%- if v not in _all_sibling_views %}{% set _ = _all_sibling_views.append(v) %}{% endif %}
{%- endfor %}
{%- endfor %}
{%- if _all_sibling_views %}
    # Cohort-scope sample_filter siblings: the operator's NB00 predicate
    # references these datasets by their bare names (e.g. "from contract"),
    # which only resolves in Spark SQL when each name is a registered temp
    # view. Mirrors `analysis.auto_explorer.sampling._expose_frames_as_views`
    # so the same predicate string runs unchanged in production.
    for _v in {{ _all_sibling_views | python_repr }}:
        spark.read.format("delta").table(landing_table(_v)).createOrReplaceTempView(_v)
{%- endif %}
{%- for step in config.filters %}
    df = df.filter({{ step.parameters.predicate | python_repr }})
{%- endfor %}
{%- endif %}
{%- if config.lifecycle_enrichments %}
    from customer_retention.stages.lifecycle.config import LifecycleEnrichmentConfig
    from customer_retention.stages.lifecycle.enrich import enrich_lifecycle_dataset
{%- for step in config.lifecycle_enrichments %}
    df = enrich_lifecycle_dataset(df, LifecycleEnrichmentConfig.from_dict({{ step.parameters.config | python_repr }}))
{%- endfor %}
{%- endif %}
{%- if config.key_resolution_steps %}
    df = resolve_entity_key(df)
{%- endif %}
    df = derive_feature_timestamp(df)
    df = derive_label_timestamp(df)
    df = derive_label_available_flag(df)
{%- if config.datetime_derivation %}
    df = derive_datetime_features(df)
{%- endif %}
{%- if config.history_window %}
    df = apply_history_window(df)
{%- endif %}
    output_table = landing_table(SOURCE_NAME)
    df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(output_table)
    from delta.tables import DeltaTable
    _z_cols = [c for c in [ENTITY_COLUMN, TIME_COLUMN] if c in [f.name for f in df.schema.fields]]
    if _z_cols:
        DeltaTable.forName(spark, output_table).optimize().executeZOrderBy(_z_cols)
    else:
        DeltaTable.forName(spark, output_table).optimize().executeCompaction()
    return df

import json as _landing_json
import traceback as _landing_tb
try:
    result = run_landing()
    _summary = f"{result.count():,} rows, {len(result.columns)} columns"
    # Row-limited display (landing is narrow but uniform pattern across stages).
    display(result.limit(20))
except Exception as _landing_exc:
    print("=" * 70, flush=True)
    print(f"[landing] FATAL: {type(_landing_exc).__name__}: {_landing_exc}", flush=True)
    print("=" * 70, flush=True)
    print(_landing_tb.format_exc(), flush=True)
    dbutils.notebook.exit(_landing_json.dumps({
        "status": "FAILED",
        "stage": "landing",
        "error_type": type(_landing_exc).__name__,
        "error_message": str(_landing_exc)[:2000],
    }))
    raise
dbutils.notebook.exit(_summary)
""",
    "databricks_target_derive.py.j2": """# Databricks notebook source
# MAGIC %md
# MAGIC # Target Derivation: {{ config.name }}
# MAGIC
# MAGIC FW-3 imperative-lane replay. Reads the per-dataset `landing_*` UC
# MAGIC tables that each registered function declares as inputs, calls the
# MAGIC function via `user_extensions`, and writes the per-dataset outputs
# MAGIC back to `landing_<dataset>` (overwrite mode). Bronze always reads
# MAGIC `landing_<dataset>`, so the post-derive enrichment propagates
# MAGIC automatically without any operator-side path rewiring.

# COMMAND ----------

# MAGIC %run ../config

# COMMAND ----------

import sys
from pathlib import Path
from customer_retention.analysis.auto_explorer.run_namespace import RunNamespace

# Resolve the pipeline root (one level above this `target_derive/` script)
# and make the sibling `user_extensions.py` importable. `__file__` is set
# under `dbutils.notebook.run`; the bare-CWD fallback covers exotic exec
# contexts and unit-test invocations.
try:
    _PIPELINE_DIR = Path(__file__).resolve().parent.parent
except NameError:
    _PIPELINE_DIR = Path.cwd().parent
if str(_PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_DIR))

try:
    _exp_dir = dbutils.widgets.get("experiments_dir")
    _run_id = dbutils.widgets.get("run_id")
    _NAMESPACE = RunNamespace(root=Path(_exp_dir), run_id=_run_id) if _exp_dir and _run_id else None
except Exception:
    _NAMESPACE = RunNamespace.from_env_or_latest()

# COMMAND ----------

{% for step in steps %}
# ── target_derive: {{ step.name }} ─────────────────────────────────────
print("[target_derive] running {{ step.name }} on datasets={{ step.datasets }}")
from user_extensions import {{ step.name }} as _fn_{{ loop.index }}

_frames_{{ loop.index }} = {
{%- for ds in step.datasets %}
    "{{ ds }}": spark.table(f"{CATALOG}.{SCHEMA}.landing_{{ ds }}"),
{%- endfor %}
}
_result_{{ loop.index }} = _fn_{{ loop.index }}(spark, _frames_{{ loop.index }}, _NAMESPACE)

if _result_{{ loop.index }} is None:
    print("[target_derive] {{ step.name }}: returned None — no outputs to write")
else:
    for _ds_name, _df in _result_{{ loop.index }}.items():
        _table = f"{CATALOG}.{SCHEMA}.landing_{_ds_name}"
        (_df.write.format("delta")
            .mode("overwrite")
            .option("overwriteSchema", "true")
            .saveAsTable(_table))
        print(f"[target_derive] wrote {_table}: {_df.count():,} rows, {len(_df.columns)} columns")

# COMMAND ----------

{% endfor %}
print("[target_derive] complete: {{ steps|length }} step(s)")
""",
    "databricks_runner.py.j2": """# Databricks notebook source
# MAGIC %md
# MAGIC # Pipeline Runner: {{ config.name }}

# COMMAND ----------

# MAGIC %md
# MAGIC ## Execution Order
# MAGIC 1. Bronze: per-source entity and event notebooks
# MAGIC 2. Silver: feature set merge
# MAGIC 3. Gold: feature engineering
# MAGIC 4. Training: ML experiment

# COMMAND ----------

import json
import os
import time

os.environ["CR_BATCH_EXECUTION"] = "1"

# COMMAND ----------

# MAGIC %run ./config

# COMMAND ----------

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")

# COMMAND ----------

from pathlib import Path
from customer_retention.analysis.auto_explorer.run_namespace import RunNamespace
try:
    _exp_dir = dbutils.widgets.get("experiments_dir")
    _run_id = dbutils.widgets.get("run_id")
    _NAMESPACE = RunNamespace(root=Path(_exp_dir), run_id=_run_id) if _exp_dir and _run_id else None
except Exception:
    _NAMESPACE = RunNamespace.from_env_or_latest()
_ns_params = {"experiments_dir": str(_NAMESPACE.root), "run_id": _NAMESPACE.run_id} if _NAMESPACE else {}

# COMMAND ----------

_log = []
_profile = []

def _spark_job_id():
    try: return spark._jsc.sc().dagScheduler().nextJobId().get()
    except Exception: return -1

def run_notebook(path, timeout=86400):
    sj_before = _spark_job_id()
    start = time.time()
    failure_payload = None
    try:
        result = dbutils.notebook.run(path, timeout, _ns_params)
        elapsed = time.time() - start
        sj_after = _spark_job_id()
        _profile.append({"notebook": path, "elapsed": round(elapsed, 3),
                         "spark_jobs": (sj_after - sj_before) if sj_before >= 0 and sj_after >= 0 else None,
                         "status": "completed"})
        # Generated notebooks structured-exit on top-level exceptions with
        # {"status":"FAILED",...}; surface that as a real RuntimeError so the
        # pipeline halts at the failed stage instead of cascading.
        if result:
            try:
                _parsed = json.loads(result)
            except (ValueError, TypeError):
                _parsed = None
            if isinstance(_parsed, dict) and _parsed.get("status") == "FAILED":
                failure_payload = _parsed
    except Exception as exc:
        elapsed = time.time() - start
        sj_after = _spark_job_id()
        _profile.append({"notebook": path, "elapsed": round(elapsed, 3),
                         "spark_jobs": (sj_after - sj_before) if sj_before >= 0 and sj_after >= 0 else None,
                         "status": "failed"})
        line = f"{path}: FAILED ({elapsed:.1f}s) -> {type(exc).__name__}: {exc}"
        print(line)
        _log.append(line)
        raise
    line = f"{path}: {result} ({elapsed:.1f}s)"
    print(line)
    _log.append(line)
    if failure_payload is not None:
        raise RuntimeError(
            f"{path} failed: {failure_payload.get('error_type', '?')}: "
            f"{failure_payload.get('error_message', '')} "
            f"(see spawned notebook output for full traceback)"
        )
    return result

# COMMAND ----------
{% if config.landing %}

# MAGIC %md
# MAGIC ## Landing Layer

# COMMAND ----------

{% for name in sorted_landing_names(config.landing) %}
run_notebook("landing/landing_{{ name }}")
{% endfor %}

# COMMAND ----------
{% endif %}
{% if target_derive_steps %}

# MAGIC %md
# MAGIC ## Target-Derive Phase (FW-3 imperative replay)
# MAGIC
# MAGIC Runs registered cross-dataset functions whose `expected_stage` is
# MAGIC `target_derive` between landing and bronze. Each function reads its
# MAGIC declared `landing_<dataset>` UC tables, mutates them, and writes the
# MAGIC enriched output back to `landing_<dataset>` (overwrite mode). Bronze
# MAGIC consumes the post-derive view transparently.

# COMMAND ----------

run_notebook("target_derive/run_target_derive")

# COMMAND ----------
{% endif %}

# MAGIC %md
# MAGIC ## Bronze Layer

# COMMAND ----------

_bronze_results = {}
{% for source_name in config.bronze %}
_bronze_results["{{ source_name }}"] = run_notebook("bronze/bronze_entity_{{ source_name }}")
{% endfor %}
{% for source_name in config.bronze_event %}
_bronze_results["event_{{ source_name }}"] = run_notebook("bronze/bronze_event_{{ source_name }}")
_bronze_results["{{ source_name }}_aggregated"] = run_notebook("bronze/bronze_entity_{{ source_name }}_aggregated")
{% endfor %}

if _NAMESPACE is not None:
    import json
    _NAMESPACE.bronze_metadata_path.parent.mkdir(parents=True, exist_ok=True)
    _NAMESPACE.bronze_metadata_path.write_text(json.dumps({"sources": _bronze_results, "total_sources": len(_bronze_results)}))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Silver Layer

# COMMAND ----------

run_notebook("silver/silver_featureset_{{ config.composite_name or config.name }}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Gold Layer

# COMMAND ----------

run_notebook("gold/gold_features_{{ config.composite_name or config.name }}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Training

# COMMAND ----------

run_notebook("training/ml_experiment")

# COMMAND ----------

if _NAMESPACE is not None and _profile:
    _profiles_json = {
        "version": 1, "environment": "databricks",
        "generated_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "notebooks": {
            e["notebook"]: {"total_elapsed": e["elapsed"], "cells": [
                {"cell_name": e["notebook"], "cell_id": e["notebook"],
                 "elapsed_sec": e["elapsed"], "status": e["status"],
                 "spark_jobs": e["spark_jobs"], "start_time": None,
                 "end_time": None, "peak_memory_mb": None}
            ]} for e in _profile
        },
    }
    _NAMESPACE.cell_profiles_path.parent.mkdir(parents=True, exist_ok=True)
    _NAMESPACE.cell_profiles_path.write_text(json.dumps(_profiles_json, indent=2))

dbutils.notebook.exit("\\n".join(_log))
""",
    "databricks_for_each_workflow.yaml.j2": """# Databricks Asset Bundle - Exploration Workflow (for_each_task pattern)
#
# Changes from basic sequential runner:
# - Per-dataset notebooks run via for_each_task (parallel dataset processing)
# - experiments_dir passed to ALL downstream tasks via base_parameters
#   (REQUIRED: without this, notebooks cannot locate the experiments directory
#    in multi-task jobs where env vars do not propagate between tasks)
# - run_id passed to ALL downstream tasks via base_parameters
#   (supplements sentinel-based discovery with explicit parameter passing)
# - concurrency: 1 ensures per-dataset notebooks run sequentially per dataset,
#   preventing race conditions on shared files. Per-dataset vote files
#   (grid_votes/) provide additional safety for the snapshot grid.
#
# Add cluster configuration (existing_cluster_id or job_cluster_key) to each
# task, or set a default at the job level.
resources:
  jobs:
    {{ project_name }}_exploration:
      name: "{{ project_name }} Exploration"
      tasks:
        - task_key: 00_Setup
          notebook_task:
            notebook_path: {{ notebooks_base_path }}/00_start_here
            source: WORKSPACE
{% for nb in per_dataset_notebooks %}
        - task_key: {{ nb.task_key }}
          depends_on:
            - task_key: {{ nb.depends_on }}
          for_each_task:
            inputs: "{% raw %}{{tasks.00_Setup.values.dataset_names}}{% endraw %}"
            concurrency: 1
            task:
              task_key: {{ nb.inner_task_key }}
              notebook_task:
                notebook_path: {{ notebooks_base_path }}/{{ nb.notebook_name }}
                base_parameters:
                  dataset_id: "{% raw %}{{input}}{% endraw %}"
                  run_id: "{% raw %}{{tasks.00_Setup.values.run_id}}{% endraw %}"
                  experiments_dir: "{% raw %}{{tasks.00_Setup.values.experiments_dir}}{% endraw %}"
                source: WORKSPACE
{% endfor %}
{% for nb in global_notebooks %}
        - task_key: {{ nb.task_key }}
          depends_on:
            - task_key: {{ nb.depends_on }}
          notebook_task:
            notebook_path: {{ notebooks_base_path }}/{{ nb.notebook_name }}
            base_parameters:
              run_id: "{% raw %}{{tasks.00_Setup.values.run_id}}{% endraw %}"
              experiments_dir: "{% raw %}{{tasks.00_Setup.values.experiments_dir}}{% endraw %}"
            source: WORKSPACE
{% endfor %}
      queue:
        enabled: true
""",
    "databricks_exploration_runner.py.j2": """# Databricks notebook source
# MAGIC %md
# MAGIC # Exploration Runner: {{ project_name }}
# MAGIC
# MAGIC Automated orchestration of exploration notebooks across all registered datasets.
# MAGIC Mirrors the local `run_exploration.py` but uses `dbutils.notebook.run()`.

# COMMAND ----------

import os
import time

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

NOTEBOOKS_BASE_PATH = "{{ notebooks_base_path }}"
FINDINGS_BASE_PATH = "{{ findings_base_path }}"

DATASETS = {
{% for ds in datasets %}
    "{{ ds.name }}": {
        "path": "{{ ds.path }}",
        "role": "{{ ds.role }}",
    },
{% endfor %}
}

TARGET_DATASET = "{{ target_dataset }}"

PER_DATASET_NOTEBOOKS = [
    "01_data_discovery",
    "01a_temporal_deep_dive",
    "01a_a_temporal_text_deep_dive",
    "01b_temporal_quality",
    "01c_temporal_patterns",
    "01d_event_aggregation",
    "02_source_integrity",
    "04a_text_columns_deep_dive",
]

GLOBAL_NOTEBOOKS = [
    "03_dataset_merge",
    "04_column_deep_dive",
    "05_relationship_analysis",
    "06_feature_opportunities",
    "07_modeling_readiness",
    "08_baseline_experiments",
    "09_business_alignment",
    "10_spec_generation",
]

NOTEBOOK_TIMEOUT = 28800  # 8h ceiling per spawned notebook

# COMMAND ----------

# MAGIC %md
# MAGIC ## Helpers

# COMMAND ----------

def run_notebook(name, timeout=NOTEBOOK_TIMEOUT, params=None):
    path = f"{NOTEBOOKS_BASE_PATH}/{name}"
    print(f"Running: {path}")
    start = time.time()
    try:
        dbutils.notebook.run(path, timeout, params or {})
        elapsed = time.time() - start
        print(f"  OK ({elapsed:.0f}s)")
        return True, elapsed
    except Exception as exc:
        elapsed = time.time() - start
        print(f"  FAILED ({elapsed:.0f}s): {exc}")
        return False, elapsed

# COMMAND ----------

# MAGIC %md
# MAGIC ## Skip Logic

# COMMAND ----------

from customer_retention.analysis.auto_explorer.skip_logic import (
    detect_skip_set_for_dataset,
    detect_global_skip_set,
)
from pathlib import Path

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup

# COMMAND ----------

run_notebook("00_start_here")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Per-Dataset Processing

# COMMAND ----------

results = {}
findings_dir = Path(FINDINGS_BASE_PATH)

dataset_order = sorted(DATASETS.keys(), key=lambda n: (0 if n == TARGET_DATASET else 1, n))

for ds_name in dataset_order:
    ds_info = DATASETS[ds_name]
    os.environ["CR_DATASET_ID"] = ds_name

    print(f"\\n{'=' * 60}")
    print(f"Dataset: {ds_name} ({ds_info['role']})")
    print(f"{'=' * 60}")

    ok, elapsed = run_notebook(
        "01_data_discovery",
        params={"data_path": ds_info["path"], "dataset_name": ds_name},
    )
    results[f"01_data_discovery:{ds_name}"] = ("OK" if ok else "FAILED", elapsed)

    if not ok:
        print(f"  Skipping remaining per-dataset notebooks for {ds_name}")
        continue

    skip_set, skip_reasons = detect_skip_set_for_dataset(findings_dir, ds_name)

    for nb in PER_DATASET_NOTEBOOKS[1:]:
        if nb in skip_set:
            print(f"  [{nb}] SKIPPED - {skip_reasons[nb]}")
            results[f"{nb}:{ds_name}"] = ("SKIPPED", 0)
            continue
        ok, elapsed = run_notebook(nb)
        results[f"{nb}:{ds_name}"] = ("OK" if ok else "FAILED", elapsed)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Global Analysis

# COMMAND ----------

os.environ["CR_DATASET_ID"] = TARGET_DATASET

print(f"\\n{'=' * 60}")
print("Global analysis (all datasets processed)")
print(f"{'=' * 60}")

try:
    from customer_retention.analysis.auto_explorer import ProjectContext
    _ctx = ProjectContext.load(str(findings_dir / "project_context.yaml"))
    global_skip, global_reasons = detect_global_skip_set(findings_dir, _ctx)
except (FileNotFoundError, KeyError, ValueError, TypeError) as _skip_exc:
    print(f"  Skip detection unavailable ({_skip_exc}), running all notebooks")
    global_skip, global_reasons = set(), {}

for nb in GLOBAL_NOTEBOOKS:
    if nb in global_skip:
        print(f"  [{nb}] SKIPPED - {global_reasons[nb]}")
        results[nb] = ("SKIPPED", 0)
        continue
    ok, elapsed = run_notebook(nb)
    results[nb] = ("OK" if ok else "FAILED", elapsed)
    if not ok and nb == "03_dataset_merge":
        print("  Critical notebook failed, stopping global phase")
        break

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary

# COMMAND ----------

ok_count = sum(1 for s, _ in results.values() if s == "OK")
fail_count = sum(1 for s, _ in results.values() if s == "FAILED")
skip_count = sum(1 for s, _ in results.values() if s == "SKIPPED")
total_time = sum(e for _, e in results.values())

print(f"\\n{'=' * 60}")
print(f"Results: {ok_count} OK, {fail_count} FAILED, {skip_count} SKIPPED")
print(f"Total time: {total_time:.0f}s")
print(f"{'=' * 60}")

if fail_count:
    print("\\nFailed notebooks:")
    for key, (status, elapsed) in results.items():
        if status == "FAILED":
            print(f"  - {key} ({elapsed:.0f}s)")
""",
}


class DatabricksCodeRenderer:
    _TEMPLATE_MAP = {
        "config": "databricks_config.py.j2",
        "landing": "databricks_landing.py.j2",
        "bronze": "databricks_bronze.py.j2",
        "bronze_event": "databricks_bronze_event.py.j2",
        "bronze_entity": "databricks_bronze_entity.py.j2",
        "target_derive": "databricks_target_derive.py.j2",
        "silver": "databricks_silver.py.j2",
        "gold": "databricks_gold.py.j2",
        "training": "databricks_training.py.j2",
        "runner": "databricks_runner.py.j2",
        "exploration_runner": "databricks_exploration_runner.py.j2",
        "for_each_workflow": "databricks_for_each_workflow.yaml.j2",
    }

    def __init__(self, catalog: str = "main", schema: str = "default", framework_repo_path: str | None = None):
        self._catalog = catalog
        self._schema = schema
        self._framework_repo_path = framework_repo_path
        self._env = Environment(loader=InlineLoader(DATABRICKS_TEMPLATES))
        self._env.globals["render_spark_step_call"] = render_spark_step_call
        self._env.globals["group_steps"] = group_steps
        self._env.globals["spark_provenance_block"] = spark_provenance_block
        self._env.globals["sorted_landing_names"] = _sorted_landing_names
        self._env.filters["python_repr"] = repr
        self._env.filters["py_source"] = render_python_literal

    _NOTEBOOK_HEADER = "# Databricks notebook source\n"

    def _inject_sys_path(self, rendered: str) -> str:
        if not self._framework_repo_path:
            return rendered
        if not rendered.startswith(self._NOTEBOOK_HEADER):
            return rendered
        sys_path_block = (
            self._NOTEBOOK_HEADER +
            "import sys\n"
            "\n"
            f'FRAMEWORK_REPO_ROOT = "{self._framework_repo_path}"\n'
            '_src = f"{FRAMEWORK_REPO_ROOT}/src"\n'
            "if _src not in sys.path:\n"
            "    sys.path.insert(0, _src)\n"
            "\n# COMMAND ----------\n\n"
        )
        return rendered.replace(self._NOTEBOOK_HEADER, sys_path_block, 1)

    def _render(self, template_key: str, **context) -> str:
        rendered = self._env.get_template(self._TEMPLATE_MAP[template_key]).render(**context)
        return self._inject_sys_path(rendered)

    def template_versions(self):
        from .generation_manifest import template_versions_for
        return template_versions_for(DATABRICKS_TEMPLATES)

    def render_config(self, config: PipelineConfig) -> str:
        return self._render("config", config=config, catalog=self._catalog, schema=self._schema)

    def render_landing(self, name: str, config) -> str:
        return self._render("landing", name=name, config=config)

    def render_bronze(self, source_name: str, bronze_config: BronzeLayerConfig) -> str:
        return self._render("bronze", source=source_name, config=bronze_config)

    def render_bronze_event(
        self,
        source_name: str,
        config: BronzeEventConfig,
        pipeline_config: "PipelineConfig | None" = None,
    ) -> str:
        grid_dates = []
        if pipeline_config is not None and pipeline_config.silver is not None:
            grid_dates = list(pipeline_config.silver.grid_dates or [])
        return self._render(
            "bronze_event", source=source_name, config=config, grid_dates=grid_dates
        )

    def render_bronze_entity(
        self, source_name: str, config: BronzeEventConfig, bronze_input_name: str, raw_source_name: str = ""
    ) -> str:
        return self._render(
            "bronze_entity",
            source=source_name,
            config=config,
            bronze_input_name=bronze_input_name,
            raw_source=raw_source_name or source_name,
        )

    def render_silver(self, config: PipelineConfig) -> str:
        return self._render("silver", config=config)

    def render_gold(self, config: PipelineConfig) -> str:
        return self._render("gold", config=config)

    def render_training(self, config: PipelineConfig) -> str:
        return self._render("training", config=config)

    def render_runner(self, config: PipelineConfig, target_derive_steps: Optional[List[Any]] = None) -> str:
        return self._render(
            "runner",
            config=config,
            target_derive_steps=list(target_derive_steps or []),
        )

    def render_target_derive(self, config: PipelineConfig, steps: List[Any]) -> str:
        return self._render("target_derive", config=config, steps=steps)

    def render_exploration_runner(
        self, project_name: str, datasets, notebooks_base_path: str, findings_base_path: str,
        framework_repo_path: str | None = None,
    ) -> str:
        ds_list = []
        target_dataset = ""
        for name, entry in datasets.items():
            role = (
                "target"
                if getattr(entry, "has_target", False) or getattr(entry, "role", None) == "target"
                else "feature"
            )
            ds_list.append({"name": name, "path": entry.path, "role": role})
            if role == "target":
                target_dataset = name
        if not target_dataset and ds_list:
            target_dataset = ds_list[0]["name"]
        saved = self._framework_repo_path
        self._framework_repo_path = framework_repo_path or saved
        rendered = self._render(
            "exploration_runner",
            project_name=project_name,
            datasets=ds_list,
            target_dataset=target_dataset,
            notebooks_base_path=notebooks_base_path,
            findings_base_path=findings_base_path,
        )
        self._framework_repo_path = saved
        return rendered

    _PER_DATASET_NOTEBOOKS = [
        ("01_Data_Discovery", "01_data_discovery", "run_01", "00_Setup"),
        ("01a_Temporal_Deep_Dive", "01a_temporal_deep_dive", "run_01a", "01_Data_Discovery"),
        ("01a_a_Temporal_Text_Deep_Dive", "01a_a_temporal_text_deep_dive", "run_01a_a", "01a_Temporal_Deep_Dive"),
        ("01b_Temporal_Quality", "01b_temporal_quality", "run_01b", "01a_a_Temporal_Text_Deep_Dive"),
        ("01c_Temporal_Patterns", "01c_temporal_patterns", "run_01c", "01b_Temporal_Quality"),
        ("01d_Event_Aggregation", "01d_event_aggregation", "run_01d", "01c_Temporal_Patterns"),
        ("02_Source_Integrity", "02_source_integrity", "run_02", "01d_Event_Aggregation"),
    ]

    _GLOBAL_NOTEBOOKS = [
        ("03_Dataset_Merge", "03_dataset_merge", "02_Source_Integrity"),
        ("04_Column_Deep_Dive", "04_column_deep_dive", "03_Dataset_Merge"),
        ("04a_Text_Columns_Deep_Dive", "04a_text_columns_deep_dive", "04_Column_Deep_Dive"),
        ("05_Relationship_Analysis", "05_relationship_analysis", "04a_Text_Columns_Deep_Dive"),
        ("06_Feature_Opportunities", "06_feature_opportunities", "05_Relationship_Analysis"),
        ("07_Modeling_Readiness", "07_modeling_readiness", "06_Feature_Opportunities"),
        ("08_Baseline_Experiments", "08_baseline_experiments", "07_Modeling_Readiness"),
        ("09_Business_Alignment", "09_business_alignment", "08_Baseline_Experiments"),
        ("10_Spec_Generation", "10_spec_generation", "09_Business_Alignment"),
        ("11_Scoring_Validation", "11_scoring_validation", "10_Spec_Generation"),
        ("12_View_Documentation", "12_view_documentation", "11_Scoring_Validation"),
    ]

    # Causal-track notebooks live at a different repo root (`causal_notebooks/`,
    # sibling to `exploration_notebooks/`). Phase 3 ships a dedicated workflow
    # task group with its own `notebooks_base_path` so they aren't mixed in
    # with the exploration DAG above.

    def render_for_each_workflow(self, project_name: str, notebooks_base_path: str) -> str:
        per_dataset = [
            {
                "task_key": task_key,
                "notebook_name": nb_name,
                "inner_task_key": inner_key,
                "depends_on": depends_on,
            }
            for task_key, nb_name, inner_key, depends_on in self._PER_DATASET_NOTEBOOKS
        ]
        global_nbs = [
            {
                "task_key": task_key,
                "notebook_name": nb_name,
                "depends_on": depends_on,
            }
            for task_key, nb_name, depends_on in self._GLOBAL_NOTEBOOKS
        ]
        return self._render(
            "for_each_workflow",
            project_name=project_name,
            notebooks_base_path=notebooks_base_path,
            per_dataset_notebooks=per_dataset,
            global_notebooks=global_nbs,
        )
