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
)


def render_spark_step_call(step: TransformationStep) -> str:
    t, col, p = step.type, step.column, step.parameters
    registry = _SPARK_REGISTRY.get(t)
    if registry is not None:
        return registry(col, p)
    raise ValueError(f"Unknown transformation type: {t}")


def _impute_null(col, p):
    value = p.get("value", 0)
    if isinstance(value, str):
        return f'df.fillna("{value}", subset=["{col}"])'
    return f'df.fillna({value}, subset=["{col}"])'


def _cap_outlier(col, p):
    lower = p.get("lower", 0)
    upper = p.get("upper", 1000000)
    return (
        f'df.withColumn("{col}", '
        f'F.when(F.col("{col}") < {lower}, {lower})'
        f'.when(F.col("{col}") > {upper}, {upper})'
        f'.otherwise(F.col("{col}")))'
    )


def _drop_column(col, _p):
    return f'df.drop("{col}")'


def _winsorize(col, p):
    lower = p.get("lower_bound", 0)
    upper = p.get("upper_bound", 1000000)
    return (
        f'df.withColumn("{col}", '
        f'F.when(F.col("{col}") < {lower}, {lower})'
        f'.when(F.col("{col}") > {upper}, {upper})'
        f'.otherwise(F.col("{col}")))'
    )


def _log_transform(col, _p):
    return f'df.withColumn("{col}", F.log1p(F.col("{col}")))'


def _sqrt_transform(col, _p):
    return f'df.withColumn("{col}", F.sqrt(F.abs(F.col("{col}"))))'


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
        f'df.withColumn("{col}", '
        f'F.col("{num}") / F.when(F.col("{den}") != 0, F.col("{den}")).otherwise(F.lit(None)))'
    )


def _derived_interaction(col, p):
    features = p.get("features", [])
    col_a = features[0] if len(features) > 0 else p.get("col_a", "")
    col_b = features[1] if len(features) > 1 else p.get("col_b", "")
    return f'df.withColumn("{col}", F.col("{col_a}") * F.col("{col_b}"))'


def _derived_composite(col, p):
    columns = p.get("columns", [])
    if not columns:
        raise ValueError(f"Composite derived column '{col}' requires non-empty 'columns' parameter")
    expr_parts = " + ".join(f'F.col("{c}")' for c in columns)
    return f'df.withColumn("{col}", ({expr_parts}) / {len(columns)})'


def _segment_aware_cap(col, p):
    n_segments = p.get("n_segments", 2)
    return f'_segment_aware_cap(df, "{col}", n_segments={n_segments})'


def _zero_inflation_handling(col, _p):
    return (
        f'df.withColumn("{col}_is_zero", F.when(F.col("{col}") == 0, 1).otherwise(0))'
        f'.withColumn("{col}_log", F.when(F.col("{col}") > 0, F.log1p(F.col("{col}"))).otherwise(0))'
    )


def _cap_then_log(col, _p):
    return f'_cap_then_log(df, "{col}")'


def _type_cast(col, p):
    dtype = p.get("dtype", "double")
    spark_type = {"float": "double", "int": "int", "string": "string"}.get(dtype, dtype)
    return f'df.withColumn("{col}", F.col("{col}").cast("{spark_type}"))'


def _yeo_johnson(col, _p):
    return f'df.withColumn("{col}", F.log1p(F.abs(F.col("{col}"))))'


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
        return f'df.filter(F.col("{col}") >= 0)'
    if condition == "range":
        min_val = p.get("min_value", 0)
        max_val = p.get("max_value", 1000000)
        return f'df.filter(F.col("{col}").between({min_val}, {max_val}))'
    if condition == "valid_values":
        valid = p.get("valid_values", [])
        return f'df.filter(F.col("{col}").isin({valid}))'
    return f'df.filter(F.col("{col}").isNotNull())'


def _dispatch_derived(col, p):
    action = p.get("action", "ratio")
    if action == "ratio":
        return _derived_ratio(col, p)
    if action == "interaction":
        return _derived_interaction(col, p)
    if action == "composite":
        return _derived_composite(col, p)
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

def load_source():
    source_config = SOURCES[SOURCE_NAME]
    path = source_config["path"]
    fmt = source_config["format"]
    if fmt == "csv":
        return spark.read.option("header", "true").option("inferSchema", "true").csv(path)
    return spark.read.format(fmt).load(path)

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
    \"\"\"Source: Data Discovery > Lifecycle Segmentation\"\"\"
    if "days_since_first" not in df.columns:
        return df
    intensity_cols = [c for c in df.columns if c.startswith("event_count_")]
    if not intensity_cols:
        return df
    tenure_med = df.approxQuantile("days_since_first", [0.5], 0.01)[0]
    intensity_med = df.approxQuantile(intensity_cols[0], [0.5], 0.01)[0]
    df = df.withColumn("lifecycle_quadrant",
        F.when((F.col("days_since_first") >= tenure_med) & (F.col(intensity_cols[0]) >= intensity_med), "steady_loyal_lifecycle")
        .when((F.col("days_since_first") >= tenure_med) & (F.col(intensity_cols[0]) < intensity_med), "occasional_loyal_lifecycle")
        .when((F.col("days_since_first") < tenure_med) & (F.col(intensity_cols[0]) >= intensity_med), "intense_brief_lifecycle")
        .otherwise("one_shot_lifecycle"))
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

result = run_bronze()
_summary = f"{result.count():,} rows, {len(result.columns)} columns"
display(result)
dbutils.notebook.exit(_summary)
""",
    "databricks_bronze_event.py.j2": """# Databricks notebook source
# MAGIC %md
# MAGIC # Bronze Event: {{ source }}

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import NumericType

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

def _get_numeric_columns(df, value_columns):
    numeric_cols = set()
    for field in df.schema.fields:
        if isinstance(field.dataType, NumericType):
            numeric_cols.add(field.name)
    return [c for c in value_columns if c in numeric_cols]

def apply_event_aggregation(df):
    \"\"\"Source: Event Aggregation > Time-Window Analysis\"\"\"
    reference_date = df.agg(F.max(TIME_COLUMN)).collect()[0][0]
    numeric_columns = _get_numeric_columns(df, {{ config.aggregation.value_columns }})
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
{%- for agg_func in config.aggregation.agg_funcs if agg_func != "count" %}
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

{%- if config.temporal_features %}

def compute_temporal_features(agg_df, raw_df):
    \"\"\"Source: Temporal Deep Dive > Lag Features\"\"\"
    from customer_retention.stages.profiling.spark_temporal_feature_engineer import SparkTemporalFeatureEngineer
    from customer_retention.stages.profiling.temporal_feature_engineer import TemporalAggregationConfig
    value_cols = {{ config.temporal_features.lag_columns or (config.aggregation.value_columns if config.aggregation else []) }}
    eng_config = TemporalAggregationConfig(
        lag_window_days={{ config.temporal_features.lag_window_days }},
        num_lags={{ config.temporal_features.num_lags }},
        lag_aggregations={{ config.temporal_features.lag_agg_funcs }},
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
    agg_df, reference_date = apply_event_aggregation(df)
{%- if config.temporal_features %}
    agg_df = compute_temporal_features(agg_df, raw_df)
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

result = run_bronze_event()
_summary = f"{result.count():,} rows, {len(result.columns)} columns"
display(result)
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
    \"\"\"Source: Data Discovery > Lifecycle Segmentation\"\"\"
    if "days_since_first" not in df.columns:
        return df
    intensity_cols = [c for c in df.columns if c.startswith("event_count_")]
    if not intensity_cols:
        return df
    tenure_med = df.approxQuantile("days_since_first", [0.5], 0.01)[0]
    intensity_med = df.approxQuantile(intensity_cols[0], [0.5], 0.01)[0]
    df = df.withColumn("lifecycle_quadrant",
        F.when((F.col("days_since_first") >= tenure_med) & (F.col(intensity_cols[0]) >= intensity_med), "steady_loyal_lifecycle")
        .when((F.col("days_since_first") >= tenure_med) & (F.col(intensity_cols[0]) < intensity_med), "occasional_loyal_lifecycle")
        .when((F.col("days_since_first") < tenure_med) & (F.col(intensity_cols[0]) >= intensity_med), "intense_brief_lifecycle")
        .otherwise("one_shot_lifecycle"))
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

result = run_bronze_entity()
_summary = f"{result.count():,} rows, {len(result.columns)} columns"
display(result)
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
def merge_sources(bronze_outputs):
    raw_entity_key = "{{ config.silver.entity_key or config.sources[0].entity_key }}"
    base_source = "{{ config.sources[0].name }}"
    entity_ids = bronze_outputs[base_source].select(raw_entity_key).distinct()
{% if has_key_resolution %}
    for meta in MERGE_SOURCE_META:
        kr_steps = meta.get("key_resolution_steps", [])
        if not kr_steps:
            continue
        df = bronze_outputs[meta["name"]]
        for step in kr_steps:
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
    merger = SparkTemporalMerger(MergeConfig(entity_key="entity_id"))
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
    merged, _report = merger.merge_all(spine, inputs)
    if hasattr(merged, "to_spark"):
        merged = merged.to_spark()
    print(f"  merge complete: {len(merged.columns)} columns, datasets={_report.datasets_merged}")
    return merged
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
    return merged
{% endif %}

{% set derived_groups = group_steps(config.silver.derived_columns) %}
{% if config.silver.derived_columns %}
def apply_derived_columns(df):
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
    _t0 = _time.monotonic()
    bronze_outputs = load_bronze_outputs()
    print(f"  load_bronze: {_time.monotonic() - _t0:.1f}s")
    _t1 = _time.monotonic()
    merged = merge_sources(bronze_outputs)
    print(f"  merge_sources: {_time.monotonic() - _t1:.1f}s ({len(merged.columns)} cols)")
{% if config.silver.derived_columns %}
    _t2 = _time.monotonic()
    merged = apply_derived_columns(merged)
    print(f"  apply_derived: {_time.monotonic() - _t2:.1f}s")
{% endif %}
    _t3 = _time.monotonic()
    merged = create_holdout_mask(merged)
    print(f"  holdout_mask: {_time.monotonic() - _t3:.1f}s")
    _t4 = _time.monotonic()
    output_table = silver_table()
    merged.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(output_table)
    print(f"  delta_write: {_time.monotonic() - _t4:.1f}s")
    _t5 = _time.monotonic()
    from delta.tables import DeltaTable
    _z_cols = [c for c in ["entity_id", "as_of_date"] if c in [f.name for f in merged.schema.fields]]
    if _z_cols:
        DeltaTable.forName(spark, output_table).optimize().executeZOrderBy(_z_cols)
    else:
        DeltaTable.forName(spark, output_table).optimize().executeCompaction()
    print(f"  optimize: {_time.monotonic() - _t5:.1f}s")
    print(f"  total: {_time.monotonic() - _t0:.1f}s")
    return output_table

_output_table = run_silver()
_result = spark.table(_output_table)
_row_count = _result.count()
_col_count = len(_result.columns)
_summary = f"{_row_count:,} rows, {_col_count} columns"

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
    _silver_meta = {"rows": _row_count, "columns": _col_count, "column_list": _result.columns}
    _NAMESPACE.silver_metadata_path.parent.mkdir(parents=True, exist_ok=True)
    _NAMESPACE.silver_metadata_path.write_text(json.dumps(_silver_meta))

display(_result)
dbutils.notebook.exit(_summary)
""",
    "databricks_gold.py.j2": r"""# Databricks notebook source
# MAGIC %md
# MAGIC # Gold: Features {{ config.composite_name or config.name }}

# COMMAND ----------

from pyspark.sql import functions as F

# COMMAND ----------

# MAGIC %run ../config

# COMMAND ----------

def _encode_one_hot(df, col, max_categories=100):
    if col not in df.columns:
        print(f"WARNING: column '{col}' not in DataFrame, skipping one-hot encoding")
        return df
    categories = [row[col] for row in df.select(col).distinct().collect() if row[col] is not None]
    if len(categories) > max_categories:
        print(f"WARNING: column '{col}' has {len(categories)} categories (>{max_categories}), using label encoding instead")
        return _label_encode(df, col)
    for cat in sorted(categories):
        safe_name = f"{col}_{cat}".replace(" ", "_").replace("-", "_")
        df = df.withColumn(safe_name, F.when(F.col(col) == cat, 1).otherwise(0))
    df = df.drop(col)
    return df

def _label_encode(df, col):
    if col not in df.columns:
        print(f"WARNING: column '{col}' not in DataFrame, skipping label encoding")
        return df
    from pyspark.ml.feature import StringIndexer
    indexer = StringIndexer(inputCol=col, outputCol=f"{col}_encoded", handleInvalid="keep")
    df = indexer.fit(df).transform(df)
    df = df.drop(col)
    return df

def _batch_scale_standard(df, cols):
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return df
    exprs = []
    for c in cols:
        exprs.extend([F.mean(c).alias(f"{c}__mean"), F.stddev(c).alias(f"{c}__std")])
    stats = df.agg(*exprs).collect()[0]
    for c in cols:
        mean_val = stats[f"{c}__mean"] or 0
        std_val = stats[f"{c}__std"] or 1
        if std_val == 0:
            std_val = 1
        df = df.withColumn(c, (F.col(c) - mean_val) / std_val)
    return df

def _batch_scale_minmax(df, cols):
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return df
    exprs = []
    for c in cols:
        exprs.extend([F.min(c).alias(f"{c}__min"), F.max(c).alias(f"{c}__max")])
    stats = df.agg(*exprs).collect()[0]
    for c in cols:
        min_val = stats[f"{c}__min"] or 0
        max_val = stats[f"{c}__max"] or 1
        range_val = max_val - min_val
        if range_val == 0:
            range_val = 1
        df = df.withColumn(c, (F.col(c) - min_val) / range_val)
    return df

def _batch_segment_aware_cap(df, cols):
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return df
    quantile_map = dict(zip(cols, df.approxQuantile(cols, [0.25, 0.75], 0.01)))
    for c in cols:
        qs = quantile_map[c]
        if len(qs) == 2:
            q1, q3 = qs
            iqr = q3 - q1
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            df = df.withColumn(c,
                F.when(F.col(c) < lower, lower)
                .when(F.col(c) > upper, upper)
                .otherwise(F.col(c)))
    return df

def _batch_cap_then_log(df, cols):
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return df
    quantile_map = dict(zip(cols, df.approxQuantile(cols, [0.99], 0.01)))
    for c in cols:
        qs = quantile_map[c]
        if qs:
            df = df.withColumn(c, F.log1p(F.greatest(F.least(F.col(c), F.lit(qs[0])), F.lit(0)).cast("double")))
    return df

# COMMAND ----------

def apply_encodings(df):
{%- set _prov = spark_provenance_block(config.gold.encodings) %}
{%- if _prov %}
{{ _prov }}
{%- endif %}
{%- for step in config.gold.encodings %}
    # {{ step.rationale }}
    df = {{ render_spark_step_call(step) }}
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
    for field in df.schema.fields:
        if isinstance(field.dataType, TimestampNTZType):
            df = df.withColumn(field.name, F.col(field.name).cast(TimestampType()))
    return df

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

def run_gold():
    import time as _time
    _t0 = _time.monotonic()
    df = spark.table(silver_table())
    print(f"  load_silver: {_time.monotonic() - _t0:.1f}s")
    _t1 = _time.monotonic()
    df = apply_transformations(df)
    print(f"  transformations: {_time.monotonic() - _t1:.1f}s")
    _t2 = _time.monotonic()
    df = apply_feature_selection(df)
    print(f"  feature_selection: {_time.monotonic() - _t2:.1f}s")
    _t3 = _time.monotonic()
    df = apply_encodings(df)
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
    output_table = gold_table()
    _t5 = _time.monotonic()
    df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(output_table)
    print(f"  delta_write: {_time.monotonic() - _t5:.1f}s")
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

result = run_gold()
_row_count = result.count()
_col_count = len(result.columns)
_summary = f"{_row_count:,} rows, {_col_count} columns"

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

display(result.limit(1000))
dbutils.notebook.exit(_summary)
""",
    "databricks_training.py.j2": """# Databricks notebook source
# MAGIC %md
# MAGIC # Training: {{ config.name }}

# COMMAND ----------
{% set best_model_type = config.training.best_model_type if config.training else None %}
{% set production_cv_folds = config.training.production_cv_folds if config.training else None %}
import json
import logging
import tempfile
import csv
from pathlib import Path
import mlflow
import mlflow.spark
from pyspark.ml.classification import (
{% if best_model_type is none or best_model_type == "logistic_regression" %}    LogisticRegression,
{% endif %}{% if best_model_type is none or best_model_type == "random_forest" %}    RandomForestClassifier,
{% endif %}{% if best_model_type is none or best_model_type == "xgboost" %}    GBTClassifier,
{% endif %})
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.functions import vector_to_array
from pyspark.ml.linalg import VectorUDT
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, DoubleType
from customer_retention.stages.modeling.feature_profile import FeatureProfile, ColumnProfile, build_feature_profile, compare_feature_profiles
from customer_retention.analysis.auto_explorer.run_namespace import RunNamespace
from customer_retention.core.compat.timing import log_timing
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
_NUMERIC_TYPES = ("double", "float", "integer", "long", "short", "boolean", "byte", "decimal")
_EXCLUDE_COLS = {TARGET, TIMESTAMP_COLUMN, ENTITY_KEY, "as_of_date", "feature_timestamp", "label_timestamp", "label_available_flag"}
_vector_schema = StructType([
    StructField("features", VectorUDT(), True),
    StructField("label", DoubleType(), True),
])
{% if config.training and config.training.exploration_feature_profile %}
_EXPLORATION_PROFILE = {{ config.training.exploration_feature_profile }}
{% else %}
_EXPLORATION_PROFILE = None
{% endif %}
try:
    _exp_dir = dbutils.widgets.get("experiments_dir")
    _run_id = dbutils.widgets.get("run_id")
    _NAMESPACE = RunNamespace(root=Path(_exp_dir), run_id=_run_id) if _exp_dir and _run_id else None
except Exception:
    _NAMESPACE = RunNamespace.from_env_or_latest()

def _assert_rows(count, stage):
    if count == 0:
        raise ValueError(f"[TRAINING] {stage}: 0 rows remaining — cannot proceed")
    return count

def load_training_data():
    return spark.table(gold_table())

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
    df = df.fillna(0, subset=feature_cols)
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features", handleInvalid="error")
    keep = ["features", F.col(TARGET).alias("label")]
    if TIMESTAMP_COLUMN in df.columns:
        keep.append(TIMESTAMP_COLUMN)
    assembled = assembler.transform(df).select(*keep)
    return assembled, feature_cols

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
    eval_pdf = predictions.select(
        F.col("label"),
        vector_to_array(F.col("probability"))[1].alias("prob_1"),
        F.col("prediction"),
    ).toPandas()
    mlflow.evaluate(
        data=eval_pdf,
        model_type="classifier",
        targets="label",
        predictions="prediction",
        extra_metrics=None,
    )

def _log_best_model(model, df, feature_cols):
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
            model=model, artifact_path="best_model", flavor=mlflow.spark,
            training_set=training_set,
            registered_model_name=f"{CATALOG}.{SCHEMA}.model_{COMPOSITE_NAME}",
        )
        print(f"[TRAINING] Model registered: {CATALOG}.{SCHEMA}.model_{COMPOSITE_NAME}")
    except ImportError:
        mlflow.spark.log_model(model, "best_model", dfs_tmpdir=_DFS_TMPDIR)

def train_and_evaluate():
    _results = {"models": {}, "feature_profile": {}}
    with log_timing("load_gold_table", logger):
        df = load_training_data()
    raw_count = _assert_rows(df.count(), "gold_table")
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

    with log_timing("prepare_features", logger):
        assembled, feature_cols = prepare_features(df)
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
        if _NAMESPACE is not None:
            _rec_path = _NAMESPACE.merged_recommendations_path
            if _rec_path.exists():
                import yaml as _rec_yaml
                from customer_retention.analysis.auto_explorer.layered_recommendations import RecommendationRegistry
                with _rec_path.open() as _rf:
                    _drop_recs = RecommendationRegistry.from_dict(_rec_yaml.safe_load(_rf))
                _runtime_drops = set()
                for _rec in getattr(getattr(_drop_recs, 'gold', None), 'feature_selection', []):
                    if _rec.action in ('drop_multicollinear', 'drop_weak', 'drop_l1_zero', 'drop_availability', 'drop_zero_variance'):
                        excluded_cols[_rec.target_column] = _rec.action
                    if _rec.action in ('drop_l1_zero', 'drop_zero_variance'):
                        _runtime_drops.add(_rec.target_column)
                _actual_runtime = [c for c in _runtime_drops if c in feature_cols]
                if _actual_runtime:
                    df = df.drop(*_actual_runtime)
                    feature_cols = [c for c in feature_cols if c not in _runtime_drops]
                    print(f"[TRAINING] Runtime L1/variance drops: {len(_actual_runtime)} features")
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

    with log_timing("temporal_split", logger):
        train_df, test_df, cutoff_date = _temporal_split(assembled, {{ config.training.test_size if config.training else 0.2 }})
    train_count = _assert_rows(train_df.count(), "train_set_after_split")
    test_count = _assert_rows(test_df.count(), "test_set_after_split")
    print(f"[TRAINING] Split: train={train_count:,}, test={test_count:,}")
    split_info = {"cutoff_date": str(cutoff_date), "train_count": train_count, "test_count": test_count}
    print(f"[TRAINING] Split info: {split_info}")
    _results["split"] = split_info

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

    _experiment_name = f"/Shared/training_{COMPOSITE_NAME}"
    with mlflow.start_run(run_name=f"training_{COMPOSITE_NAME}") as _parent_run:
        mlflow.set_tag("composite_name", COMPOSITE_NAME)
        mlflow.set_tag("pipeline_name", PIPELINE_NAME)
        mlflow.set_tag("target_column", TARGET)
        mlflow.set_tag("entity_key", "entity_id")
        mlflow.set_tag("timestamp_column", TIMESTAMP_COLUMN)
        if RECOMMENDATIONS_HASH:
            mlflow.set_tag("recommendations_hash", RECOMMENDATIONS_HASH)
        mlflow.log_params({"train_samples": train_count, "test_samples": test_count, "n_features": len(feature_cols)})

        for name, model in models.items():
            with mlflow.start_run(run_name=name, nested=True):
                with log_timing(f"fit_{name}", logger, train_rows=train_count):
                    fitted = model.fit(train_df)
                _log_training_progress(fitted, name)
                predictions = fitted.transform(test_df)
                metrics = _evaluate_model(predictions)
                print(f"[TRAINING] {name}: AUC={metrics['roc_auc']:.4f}, PR-AUC={metrics['pr_auc']:.4f}, F1={metrics['f1']:.4f}")
                _results["models"][name] = metrics
                mlflow.log_param("model_type", name)
                mlflow.log_param("num_features", len(feature_cols))
                mlflow.spark.log_model(fitted, f"model_{name}", dfs_tmpdir=_DFS_TMPDIR)
                mlflow.log_metrics(metrics)
                _log_feature_importance(fitted, feature_cols)
                _mlflow_evaluate_predictions(predictions)
                if metrics["roc_auc"] > best_auc:
                    best_auc = metrics["roc_auc"]
                    best_model_name = name
                    best_model = fitted
                    best_metrics = metrics

        mlflow.set_tag("best_model", best_model_name)
        mlflow.log_metric("best_roc_auc", best_auc)
        mlflow.log_metrics({f"best_{k}": v for k, v in best_metrics.items()})
        with tempfile.TemporaryDirectory() as _tmp_dir:
            _features_path = str(Path(_tmp_dir) / "features.json")
            with open(_features_path, "w") as f:
                json.dump({"feature_columns": feature_cols, "count": len(feature_cols)}, f)
            mlflow.log_artifact(_features_path)
        _log_best_model(best_model, df, feature_cols)

    _results["best_model"] = best_model_name
    _results["best_roc_auc"] = best_auc

    if _NAMESPACE is not None:
        _training_meta = {
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
        }
        _NAMESPACE.training_metadata_path.parent.mkdir(parents=True, exist_ok=True)
        _NAMESPACE.training_metadata_path.write_text(json.dumps(_training_meta))
        print(f"[TRAINING] Metadata saved to {_NAMESPACE.training_metadata_path}")

    return _results

# COMMAND ----------

_training_results = train_and_evaluate()
print("\\n" + "=" * 60)
print("TRAINING RESULTS")
print("=" * 60)
print(json.dumps(_training_results, indent=2, default=str))
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

SOURCE_NAME = "{{ name }}"
ENTITY_COLUMN = "{{ config.entity_column }}"
TIME_COLUMN = "{{ config.time_column }}"

def load_source():
    source_config = RAW_SOURCES[SOURCE_NAME]
    path = source_config["path"]
    fmt = source_config["format"]
    if fmt == "csv":
        return spark.read.option("header", "true").option("inferSchema", "true").csv(path)
    return spark.read.format(fmt).load(path)

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
    _bridge = _bridge.select("{{ step.bridge_key }}", "{{ step.resolve_column }}").dropDuplicates(["{{ step.bridge_key }}"])
    df = df.join(_bridge, df["{{ step.source_key }}"] == _bridge["{{ step.bridge_key }}"], "inner")
{%- if step.source_key != step.bridge_key %}
    df = df.drop("{{ step.bridge_key }}")
{%- endif %}
{%- endfor %}
    return df
{%- endif %}

# COMMAND ----------

def run_landing():
    df = load_source()
{%- if config.raw_time_column %}
    df = df.withColumnRenamed("{{ config.raw_time_column }}", TIME_COLUMN)
{%- endif %}
{%- if config.original_target_column %}
    df = df.withColumnRenamed("{{ config.original_target_column }}", TARGET_COLUMN)
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

result = run_landing()
_summary = f"{result.count():,} rows, {len(result.columns)} columns"
display(result)
dbutils.notebook.exit(_summary)
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

def run_notebook(path, timeout=3600):
    sj_before = _spark_job_id()
    start = time.time()
    try:
        result = dbutils.notebook.run(path, timeout, _ns_params)
        elapsed = time.time() - start
        sj_after = _spark_job_id()
        _profile.append({"notebook": path, "elapsed": round(elapsed, 3),
                         "spark_jobs": (sj_after - sj_before) if sj_before >= 0 and sj_after >= 0 else None,
                         "status": "completed"})
    except Exception as exc:
        elapsed = time.time() - start
        sj_after = _spark_job_id()
        _profile.append({"notebook": path, "elapsed": round(elapsed, 3),
                         "spark_jobs": (sj_after - sj_before) if sj_before >= 0 and sj_after >= 0 else None,
                         "status": "failed"})
        result = f"FAILED: {exc}"
    line = f"{path}: {result} ({elapsed:.1f}s)"
    print(line)
    _log.append(line)
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

NOTEBOOK_TIMEOUT = 1800

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

    def render_config(self, config: PipelineConfig) -> str:
        return self._render("config", config=config, catalog=self._catalog, schema=self._schema)

    def render_landing(self, name: str, config) -> str:
        return self._render("landing", name=name, config=config)

    def render_bronze(self, source_name: str, bronze_config: BronzeLayerConfig) -> str:
        return self._render("bronze", source=source_name, config=bronze_config)

    def render_bronze_event(self, source_name: str, config: BronzeEventConfig) -> str:
        return self._render("bronze_event", source=source_name, config=config)

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

    def render_runner(self, config: PipelineConfig) -> str:
        return self._render("runner", config=config)

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
