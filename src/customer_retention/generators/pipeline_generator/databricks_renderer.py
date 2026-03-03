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

{% if config.lifecycle %}
{% if config.lifecycle.include_recency_bucket %}
def add_recency_tenure(df, raw_df):
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
{% set edges = config.lifecycle.recency_bucket_edges %}
{% set labels = config.lifecycle.recency_bucket_labels %}
    df = df.withColumn("recency_bucket", F.when(F.col("days_since_last") <= {{ edges[1] }}, "{{ labels[0] }}")
{% for i in range(2, edges | length) %}
        .when(F.col("days_since_last") <= {{ edges[i] }}, "{{ labels[i - 1] }}")
{% endfor %}
        .otherwise("{{ labels[-1] }}"))
    return df
{% endif %}
{% if config.lifecycle.include_lifecycle_quadrant %}
def add_lifecycle_quadrant(df):
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
{% endif %}
{% if config.lifecycle.include_cyclical_features %}
def add_cyclical_features(df, raw_df):
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
{% endif %}
{% if config.lifecycle.include_month_cyclical %}
def add_month_quarter_cyclical(df, raw_df):
    entity_col = "{{ config.entity_column or config.source.entity_key }}"
    time_col = "{{ config.time_column or config.source.time_column }}"
    mean_month = raw_df.groupBy(entity_col).agg(
        F.mean(F.month(F.col(time_col)).cast("double")).alias("mean_month")
    )
    df = df.join(mean_month, on=entity_col, how="left")
    df = df.withColumn("month_sin", F.sin(2 * 3.141592653589793 * F.col("mean_month") / 12))
    df = df.withColumn("month_cos", F.cos(2 * 3.141592653589793 * F.col("mean_month") / 12))
{% if config.lifecycle.include_quarter_cyclical %}
    mean_quarter = raw_df.groupBy(entity_col).agg(
        F.mean(F.quarter(F.col(time_col)).cast("double")).alias("mean_quarter")
    )
    df = df.join(mean_quarter, on=entity_col, how="left")
    df = df.withColumn("quarter_sin", F.sin(2 * 3.141592653589793 * F.col("mean_quarter") / 4))
    df = df.withColumn("quarter_cos", F.cos(2 * 3.141592653589793 * F.col("mean_quarter") / 4))
    df = df.drop("mean_month", "mean_quarter")
{% else %}
    df = df.drop("mean_month")
{% endif %}
    return df
{% endif %}
{% if config.lifecycle.include_trend_features %}
def add_trend_features(df):
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
{% endif %}
{% if config.lifecycle.include_cohort_features %}
def add_cohort_features(df, raw_df):
    entity_col = "{{ config.entity_column or config.source.entity_key }}"
    time_col = "{{ config.time_column or config.source.time_column }}"
    first_event = raw_df.groupBy(entity_col).agg(F.min(time_col).alias("first_event"))
    first_event = first_event.withColumn("cohort_year", F.year("first_event"))
    first_event = first_event.withColumn("cohort_quarter", F.quarter("first_event"))
    df = df.join(first_event.select(entity_col, "cohort_year", "cohort_quarter"), on=entity_col, how="left")
    return df
{% endif %}
{% if config.lifecycle.momentum_pairs %}
def add_momentum_ratios(df):
{% for pair in config.lifecycle.momentum_pairs %}
    short_col = "event_count_{{ pair.short_window }}"
    long_col = "event_count_{{ pair.long_window }}"
    if short_col in df.columns and long_col in df.columns:
        df = df.withColumn("momentum_{{ pair.short_window }}_{{ pair.long_window }}", F.col(short_col) / F.when(F.col(long_col) != 0, F.col(long_col)).otherwise(F.lit(None)))
{% endfor %}
    return df
{% endif %}
def enrich_lifecycle(df):
    raw_table = bronze_table("{{ source }}")
    raw_df = spark.table(raw_table)
{% if config.lifecycle.include_recency_bucket %}
    df = add_recency_tenure(df, raw_df)
    df = add_recency_buckets(df)
{% endif %}
{% if config.lifecycle.include_lifecycle_quadrant %}
    df = add_lifecycle_quadrant(df)
{% endif %}
{% if config.lifecycle.include_cyclical_features %}
    df = add_cyclical_features(df, raw_df)
{% endif %}
{% if config.lifecycle.include_month_cyclical %}
    df = add_month_quarter_cyclical(df, raw_df)
{% endif %}
{% if config.lifecycle.include_trend_features %}
    df = add_trend_features(df)
{% endif %}
{% if config.lifecycle.include_cohort_features %}
    df = add_cohort_features(df, raw_df)
{% endif %}
{% if config.lifecycle.momentum_pairs %}
    df = add_momentum_ratios(df)
{% endif %}
    return df
{% endif %}

{% if config.text_features %}
def compute_text_features_entity(df):
    from customer_retention.stages.profiling.text_processor import TextColumnProcessor, TextProcessingConfig
    pdf = df.toPandas()
{% for tf in config.text_features %}
    if "{{ tf.column }}" in pdf.columns:
        processor = TextColumnProcessor(TextProcessingConfig(embedding_model="{{ tf.embedding_model }}"), registry=None)
        pdf, result = processor.process_column(pdf, "{{ tf.column }}", fit=True)
{% endfor %}
    return spark.createDataFrame(pdf)
{% endif %}

# COMMAND ----------

def run_bronze():
    df = load_source()
    df = apply_transformations(df)
{% if config.lifecycle %}
    df = enrich_lifecycle(df)
{% endif %}
{% if config.text_features %}
    df = compute_text_features_entity(df)
{% endif %}
    output_table = bronze_table(SOURCE_NAME)
    df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(output_table)
    return df

result = run_bronze()
display(result)
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

{% if config.deduplicate %}
def deduplicate(df):
{% if config.deduplicate is not true and config.deduplicate.strategy is defined and config.deduplicate.strategy == "keep_most_complete" %}
    _all_cols = [f.name for f in df.schema.fields if f.name not in (ENTITY_COLUMN, TIME_COLUMN)]
    _null_expr = sum(F.when(F.col(c).isNull(), 1).otherwise(0) for c in _all_cols) if _all_cols else F.lit(0)
    df = df.withColumn("_null_count", _null_expr)
{% if config.deduplicate.conflict_columns %}
    _partition_cols = {{ config.deduplicate.conflict_columns }}
{% else %}
    _partition_cols = [ENTITY_COLUMN, TIME_COLUMN]
{% endif %}
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

{% if config.datetime_derivation %}
DATETIME_DERIVATION_SOURCES = {{ config.datetime_derivation.source_columns }}
MASK_FUTURE_COLUMNS = {{ config.datetime_derivation.mask_future_columns }}

def derive_datetime_features(df):
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
{% endif %}

{% if config.aggregation %}
def _window_to_days(window_str):
    if window_str.endswith("d"):
        return int(window_str[:-1])
    if window_str.endswith("h"):
        return max(1, int(window_str[:-1]) // 24)
    return int(window_str)

CATEGORICAL_COLUMNS = {{ config.aggregation.categorical_columns }}

def _get_numeric_columns(df, value_columns):
    numeric_cols = set()
    for field in df.schema.fields:
        if isinstance(field.dataType, NumericType):
            numeric_cols.add(field.name)
    return [c for c in value_columns if c in numeric_cols]

def apply_event_aggregation(df):
    reference_date = df.agg(F.max(TIME_COLUMN)).collect()[0][0]
    numeric_columns = _get_numeric_columns(df, {{ config.aggregation.value_columns }})
    results = []
{% for window in config.aggregation.windows %}
{% if window == "all_time" %}
    window_df = df
{% else %}
    window_df = df.filter(
        F.col(TIME_COLUMN) >= F.date_sub(F.lit(reference_date), _window_to_days("{{ window }}"))
    )
{% endif %}
    agg_exprs = [F.count("*").alias("event_count_{{ window }}")]
    for col in numeric_columns:
{% for agg_func in config.aggregation.agg_funcs %}
{% if agg_func != "count" %}
        agg_exprs.append(F.{{ agg_func }}(col).alias(f"{col}_{{ agg_func }}_{{ window }}"))
{% endif %}
{% endfor %}
    for col in CATEGORICAL_COLUMNS:
        if col in [f.name for f in window_df.schema.fields]:
            agg_exprs.append(F.countDistinct(col).alias(f"{col}_nunique_{{ window }}"))
            agg_exprs.append(F.first(col).alias(f"{col}_mode_{{ window }}"))
    window_agg = window_df.groupBy(ENTITY_COLUMN).agg(*agg_exprs)
    results.append(window_agg)
{% endfor %}
    merged = results[0]
    for r in results[1:]:
        merged = merged.join(r, on=ENTITY_COLUMN, how="outer")
    if "feature_timestamp" in [f.name for f in df.schema.fields]:
        ts_agg = df.groupBy(ENTITY_COLUMN).agg(F.max("feature_timestamp").alias("feature_timestamp"))
        merged = merged.join(ts_agg, on=ENTITY_COLUMN, how="left")
    return merged, reference_date
{% endif %}

{% if config.temporal_features %}
def compute_temporal_features(agg_df, raw_df):
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
    merge_cols = [c for c in temporal_df.columns if c != ENTITY_COLUMN]
    agg_pdf = agg_df.toPandas() if hasattr(agg_df, "toPandas") else agg_df
    merged = agg_pdf.merge(temporal_df[[ENTITY_COLUMN] + merge_cols], on=ENTITY_COLUMN, how="left")
    return spark.createDataFrame(merged)
{% endif %}

{% if config.text_features %}
def compute_text_features(agg_df, raw_df):
    from customer_retention.stages.profiling.text_processor import TextColumnProcessor, TextProcessingConfig
    agg_pdf = agg_df.toPandas() if hasattr(agg_df, "toPandas") else agg_df
    raw_pdf = raw_df.toPandas() if hasattr(raw_df, "toPandas") else raw_df
{% for tf in config.text_features %}
    if "{{ tf.column }}" in raw_pdf.columns:
        processor = TextColumnProcessor(TextProcessingConfig(embedding_model="{{ tf.embedding_model }}"), registry=None)
        text_data = raw_pdf.groupby(ENTITY_COLUMN)["{{ tf.column }}"].first().reset_index()
        text_data, result = processor.process_column(text_data, "{{ tf.column }}", fit=True)
        component_cols = result.component_columns
        agg_pdf = agg_pdf.merge(text_data[[ENTITY_COLUMN] + component_cols], on=ENTITY_COLUMN, how="left")
{% endfor %}
    return spark.createDataFrame(agg_pdf)
{% endif %}

# COMMAND ----------

def run_bronze_event():
    raw_df = load_source()
    df = apply_pre_shaping(raw_df)
{% if config.deduplicate %}
    df = deduplicate(df)
{% endif %}
{% if config.datetime_derivation %}
    df = derive_datetime_features(df)
{% endif %}
{% if config.aggregation %}
    agg_df, reference_date = apply_event_aggregation(df)
{% if config.temporal_features %}
    agg_df = compute_temporal_features(agg_df, raw_df)
{% endif %}
{% if config.text_features %}
    agg_df = compute_text_features(agg_df, raw_df)
{% endif %}
    output_table = bronze_table("{{ source }}_events")
    agg_df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(output_table)
    from delta.tables import DeltaTable
    _z_cols = [c for c in [ENTITY_COLUMN, "as_of_date"] if c in [f.name for f in agg_df.schema.fields]]
    if _z_cols:
        DeltaTable.forName(spark, output_table).optimize().executeZOrderBy(_z_cols)
    else:
        DeltaTable.forName(spark, output_table).optimize().executeCompaction()
    return agg_df
{% else %}
    output_table = bronze_table("{{ source }}_events")
    df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(output_table)
    from delta.tables import DeltaTable
    _z_cols = [c for c in [ENTITY_COLUMN, TIME_COLUMN] if c in [f.name for f in df.schema.fields]]
    if _z_cols:
        DeltaTable.forName(spark, output_table).optimize().executeZOrderBy(_z_cols)
    else:
        DeltaTable.forName(spark, output_table).optimize().executeCompaction()
    return df
{% endif %}

result = run_bronze_event()
display(result)
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

{% if config.lifecycle %}
{% if config.lifecycle.include_recency_bucket %}
def add_recency_tenure(df):
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
{% set edges = config.lifecycle.recency_bucket_edges %}
{% set labels = config.lifecycle.recency_bucket_labels %}
    df = df.withColumn("recency_bucket", F.when(F.col("days_since_last") <= {{ edges[1] }}, "{{ labels[0] }}")
{% for i in range(2, edges | length) %}
        .when(F.col("days_since_last") <= {{ edges[i] }}, "{{ labels[i - 1] }}")
{% endfor %}
        .otherwise("{{ labels[-1] }}"))
    return df
{% endif %}
{% if config.lifecycle.include_lifecycle_quadrant %}
def add_lifecycle_quadrant(df):
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
{% endif %}
{% if config.lifecycle.include_month_cyclical %}
def add_month_quarter_cyclical(df):
    raw_df = spark.table(landing_table("{{ raw_source }}"))
    time_col = "{{ config.time_column }}"
    mean_month = raw_df.groupBy(ENTITY_COLUMN).agg(
        F.mean(F.month(F.col(time_col)).cast("double")).alias("mean_month")
    )
    df = df.join(mean_month, on=ENTITY_COLUMN, how="left")
    df = df.withColumn("month_sin", F.sin(2 * 3.141592653589793 * F.col("mean_month") / 12))
    df = df.withColumn("month_cos", F.cos(2 * 3.141592653589793 * F.col("mean_month") / 12))
{% if config.lifecycle.include_quarter_cyclical %}
    mean_quarter = raw_df.groupBy(ENTITY_COLUMN).agg(
        F.mean(F.quarter(F.col(time_col)).cast("double")).alias("mean_quarter")
    )
    df = df.join(mean_quarter, on=ENTITY_COLUMN, how="left")
    df = df.withColumn("quarter_sin", F.sin(2 * 3.141592653589793 * F.col("mean_quarter") / 4))
    df = df.withColumn("quarter_cos", F.cos(2 * 3.141592653589793 * F.col("mean_quarter") / 4))
    df = df.drop("mean_month", "mean_quarter")
{% else %}
    df = df.drop("mean_month")
{% endif %}
    return df
{% endif %}
{% if config.lifecycle.include_trend_features %}
def add_trend_features(df):
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
{% endif %}
{% if config.lifecycle.include_cohort_features %}
def add_cohort_features(df):
    raw_df = spark.table(landing_table("{{ raw_source }}"))
    time_col = "{{ config.time_column }}"
    first_event = raw_df.groupBy(ENTITY_COLUMN).agg(F.min(time_col).alias("first_event"))
    first_event = first_event.withColumn("cohort_year", F.year("first_event"))
    first_event = first_event.withColumn("cohort_quarter", F.quarter("first_event"))
    df = df.join(first_event.select(ENTITY_COLUMN, "cohort_year", "cohort_quarter"), on=ENTITY_COLUMN, how="left")
    return df
{% endif %}
{% if config.lifecycle.momentum_pairs %}
def add_momentum_ratios(df):
{% for pair in config.lifecycle.momentum_pairs %}
    short_col = "event_count_{{ pair.short_window }}"
    long_col = "event_count_{{ pair.long_window }}"
    if short_col in df.columns and long_col in df.columns:
        df = df.withColumn("momentum_{{ pair.short_window }}_{{ pair.long_window }}", F.col(short_col) / F.when(F.col(long_col) != 0, F.col(long_col)).otherwise(F.lit(None)))
{% endfor %}
    return df
{% endif %}
def enrich_lifecycle(df):
{% if config.lifecycle.include_recency_bucket %}
    df = add_recency_tenure(df)
    df = add_recency_buckets(df)
{% endif %}
{% if config.lifecycle.include_lifecycle_quadrant %}
    df = add_lifecycle_quadrant(df)
{% endif %}
{% if config.lifecycle.include_month_cyclical %}
    df = add_month_quarter_cyclical(df)
{% endif %}
{% if config.lifecycle.include_trend_features %}
    df = add_trend_features(df)
{% endif %}
{% if config.lifecycle.include_cohort_features %}
    df = add_cohort_features(df)
{% endif %}
{% if config.lifecycle.momentum_pairs %}
    df = add_momentum_ratios(df)
{% endif %}
    return df
{% endif %}

# COMMAND ----------

def run_bronze_entity():
    df = load_aggregated()
{% if config.post_shaping %}
    df = apply_post_shaping(df)
{% endif %}
{% if config.lifecycle %}
    df = enrich_lifecycle(df)
{% endif %}
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
display(result)
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
    entity_ids = bronze_outputs[base_source].select(raw_entity_key).distinct().toPandas()[raw_entity_key]
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
    return merged
{% else %}
def merge_sources(bronze_outputs):
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

def run_silver():
    bronze_outputs = load_bronze_outputs()
    merged = merge_sources(bronze_outputs)
{% if config.silver.derived_columns %}
    merged = apply_derived_columns(merged)
{% endif %}
    output_table = silver_table()
    merged.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(output_table)
    from delta.tables import DeltaTable
    _z_cols = [c for c in ["entity_id", "as_of_date"] if c in [f.name for f in merged.schema.fields]]
    if _z_cols:
        DeltaTable.forName(spark, output_table).optimize().executeZOrderBy(_z_cols)
    else:
        DeltaTable.forName(spark, output_table).optimize().executeCompaction()
    return merged

result = run_silver()
display(result)
""",
    "databricks_gold.py.j2": """# Databricks notebook source
# MAGIC %md
# MAGIC # Gold: Features {{ config.composite_name or config.name }}

# COMMAND ----------

from pyspark.sql import functions as F

# COMMAND ----------

# MAGIC %run ../config

# COMMAND ----------

def _encode_one_hot(df, col):
    if col not in df.columns:
        print(f"WARNING: column '{col}' not in DataFrame, skipping one-hot encoding")
        return df
    categories = [row[col] for row in df.select(col).distinct().collect() if row[col] is not None]
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

def _scale_standard(df, col):
    stats = df.agg(F.mean(col).alias("mean_val"), F.stddev(col).alias("std_val")).collect()[0]
    mean_val = stats["mean_val"] or 0
    std_val = stats["std_val"] or 1
    if std_val == 0:
        std_val = 1
    df = df.withColumn(col, (F.col(col) - mean_val) / std_val)
    return df

def _scale_minmax(df, col):
    stats = df.agg(F.min(col).alias("min_val"), F.max(col).alias("max_val")).collect()[0]
    min_val = stats["min_val"] or 0
    max_val = stats["max_val"] or 1
    range_val = max_val - min_val
    if range_val == 0:
        range_val = 1
    df = df.withColumn(col, (F.col(col) - min_val) / range_val)
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

def _segment_aware_cap(df, col, n_segments=2):
    quantiles = df.approxQuantile(col, [0.25, 0.75], 0.01)
    if len(quantiles) == 2:
        q1, q3 = quantiles
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        df = df.withColumn(col,
            F.when(F.col(col) < lower, lower)
            .when(F.col(col) > upper, upper)
            .otherwise(F.col(col)))
    return df

def _cap_then_log(df, col):
    quantiles = df.approxQuantile(col, [0.99], 0.01)
    if not quantiles:
        return df
    q99 = quantiles[0]
    return df.withColumn(col, F.log1p(F.greatest(F.least(F.col(col), F.lit(q99)), F.lit(0)).cast("double")))

# COMMAND ----------

def apply_encodings(df):
{%- set _prov = spark_provenance_block(config.gold.encodings) %}
{%- if _prov %}
{{ _prov }}
{%- endif %}
{% if config.gold.encodings %}
{% for step in config.gold.encodings %}
    # {{ step.rationale }}
    df = {{ render_spark_step_call(step) }}
{% endfor %}
{% endif %}
    return df

def apply_scalings(df):
{%- set _prov = spark_provenance_block(config.gold.scalings) %}
{%- if _prov %}
{{ _prov }}
{%- endif %}
{% if config.gold.scalings %}
{% set ns = namespace(standard=[], minmax=[]) %}
{% for step in config.gold.scalings %}
{% if step.parameters.get('method') == 'minmax' %}{% set ns.minmax = ns.minmax + [step.column] %}{% else %}{% set ns.standard = ns.standard + [step.column] %}{% endif %}
{% endfor %}
{% if ns.standard %}
    df = _batch_scale_standard(df, [{% for c in ns.standard %}"{{ c }}"{{ ", " if not loop.last }}{% endfor %}])
{% endif %}
{% if ns.minmax %}
    df = _batch_scale_minmax(df, [{% for c in ns.minmax %}"{{ c }}"{{ ", " if not loop.last }}{% endfor %}])
{% endif %}
{% endif %}
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
{%- for t in steps %}
    # {{ t.rationale }}
    df = {{ render_spark_step_call(t) }}
{%- endfor %}
    return df
{% endfor %}

def apply_feature_selection(df):
{% if config.gold.feature_selections %}
    drop_cols = {{ config.gold.feature_selections }}
    df = df.drop(*[c for c in drop_cols if c in df.columns])
{% endif %}
    return df

# COMMAND ----------

def run_gold():
    df = spark.table(silver_table())
    df = apply_transformations(df)
    df = apply_encodings(df)
    df = apply_scalings(df)
    df = apply_feature_selection(df)
    if "as_of_date" in df.columns:
        df = df.withColumnRenamed("as_of_date", TIMESTAMP_COLUMN)
    elif "feature_timestamp" in df.columns:
        df = df.withColumnRenamed("feature_timestamp", TIMESTAMP_COLUMN)
    output_table = gold_table()
    df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(output_table)
    from delta.tables import DeltaTable
    _z_cols = [c for c in ["entity_id", TIMESTAMP_COLUMN] if c in [f.name for f in df.schema.fields]]
    if _z_cols:
        DeltaTable.forName(spark, output_table).optimize().executeZOrderBy(_z_cols)
    else:
        DeltaTable.forName(spark, output_table).optimize().executeCompaction()
    return df

result = run_gold()
display(result)
""",
    "databricks_training.py.j2": """# Databricks notebook source
# MAGIC %md
# MAGIC # Training: {{ config.name }}

# COMMAND ----------

import mlflow
import mlflow.spark
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, GBTClassifier
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.sql import functions as F
from customer_retention.stages.modeling.data_splitter import DataSplitter, SplitStrategy
{% if config.training and config.training.imbalance_strategy == "smote" %}
from imblearn.over_sampling import SMOTE
{% endif %}

# COMMAND ----------

# MAGIC %run ../config

# COMMAND ----------

TARGET = TARGET_COLUMN

def load_training_data():
    return spark.table(gold_table())

def prepare_features(df):
    exclude_prefixes = ["original_"]
    exclude_cols = {TARGET, "as_of_date"}
    feature_cols = [
        c for c in df.columns
        if c not in exclude_cols and not any(c.startswith(p) for p in exclude_prefixes)
        and df.schema[c].dataType.typeName() in ("double", "float", "integer", "long", "short")
    ]
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features", handleInvalid="skip")
    assembled = assembler.transform(df).select("features", F.col(TARGET).alias("label"))
    return assembled, feature_cols

def train_and_evaluate():
    df = load_training_data()
{% if config.training and config.training.recommended_training_start %}
    if TIMESTAMP_COLUMN in df.columns:
        df = df.filter(F.col(TIMESTAMP_COLUMN) >= F.lit("{{ config.training.recommended_training_start }}"))
{% endif %}
{% if config.training and config.training.filter_future_dates %}
    if TIMESTAMP_COLUMN in df.columns:
        df = df.filter(F.col(TIMESTAMP_COLUMN) <= F.current_timestamp())
{% endif %}
    assembled, feature_cols = prepare_features(df)
    pdf = assembled.toPandas()
    splitter = DataSplitter(
        target_column=TARGET_COLUMN,
        strategy=SplitStrategy.TEMPORAL,
        temporal_column=TIMESTAMP_COLUMN,
{% if config.training and config.training.purge_gap_days %}
        purge_gap_days={{ config.training.purge_gap_days }},
{% endif %}
        test_size={{ config.training.test_size if config.training else 0.2 }},
    )
    splits = splitter.split(pdf)
    train_pdf, test_pdf = splits.X_train, splits.X_test
    train_df = spark.createDataFrame(train_pdf)
    test_df = spark.createDataFrame(test_pdf)

    mlflow.set_experiment(f"/Shared/{PIPELINE_NAME}")
    mlflow.set_tag("composite_name", COMPOSITE_NAME)

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
    train_df = spark.createDataFrame(resampled_pdf)
{% endif %}

{% set weight_param = ', weightCol="class_weight"' if config.training and config.training.imbalance_strategy == "class_weight" else '' %}
    models = {
        "LogisticRegression": LogisticRegression(maxIter=100, featuresCol="features", labelCol="label"{{ weight_param }}),
        "RandomForest": RandomForestClassifier(numTrees=100, featuresCol="features", labelCol="label"{{ weight_param }}),
        "GBTClassifier": GBTClassifier(maxIter=50, featuresCol="features", labelCol="label"{{ weight_param }}),
    }

    best_model_name = None
    best_auc = -1.0
    best_model = None
    binary_eval = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC")
    multi_eval = MulticlassClassificationEvaluator(labelCol="label", metricName="f1")

    for name, model in models.items():
        with mlflow.start_run(run_name=name, nested=True):
            fitted = model.fit(train_df)
            predictions = fitted.transform(test_df)
            auc = binary_eval.evaluate(predictions)
            f1 = multi_eval.evaluate(predictions)
            mlflow.log_param("model_type", name)
            mlflow.log_param("num_features", len(feature_cols))
            mlflow.spark.log_model(fitted, f"model_{name}")
            mlflow.log_metric("auc", auc)
            mlflow.log_metric("f1", f1)
            if auc > best_auc:
                best_auc = auc
                best_model_name = name
                best_model = fitted

    with mlflow.start_run(run_name="best_model"):
        mlflow.log_param("best_model", best_model_name)
        mlflow.log_metric("best_auc", best_auc)
        mlflow.spark.log_model(best_model, "best_model")

    return best_model_name, best_auc

# COMMAND ----------

best_name, best_auc = train_and_evaluate()
print(f"Best model: {best_name} with AUC: {best_auc:.4f}")
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
{% if config.timestamp_coalesce %}
{% set cols = config.timestamp_coalesce.datetime_columns_ordered %}
    df = df.withColumn("feature_timestamp", F.coalesce(
{% for col in cols %}
        F.to_timestamp(F.col("{{ col }}")){{ "," if not loop.last else "" }}
{% endfor %}
    ))
{% else %}
    if TIME_COLUMN in [f.name for f in df.schema.fields]:
        df = df.withColumn("feature_timestamp", F.to_timestamp(F.col(TIME_COLUMN)))
    elif "feature_timestamp" not in [f.name for f in df.schema.fields]:
        raise ValueError(f"Time column '{TIME_COLUMN}' not found. Available: {df.columns}")
{% endif %}
    return df

def derive_label_timestamp(df):
{% if config.label_timestamp %}
{% set lt = config.label_timestamp %}
{% if lt.label_column %}
    label_ts = F.to_timestamp(F.col("{{ lt.label_column }}"))
    fallback_ts = F.expr(f"feature_timestamp + INTERVAL {{ lt.fallback_window_days }} DAYS")
    df = df.withColumn("label_timestamp", F.coalesce(label_ts, fallback_ts))
{% else %}
    df = df.withColumn("label_timestamp", F.expr(f"feature_timestamp + INTERVAL {{ lt.fallback_window_days }} DAYS"))
{% endif %}
{% else %}
    df = df.withColumn("label_timestamp", F.expr("feature_timestamp + INTERVAL 180 DAYS"))
{% endif %}
    return df

def derive_label_available_flag(df):
    if TARGET_COLUMN in [f.name for f in df.schema.fields]:
        df = df.withColumn("label_available_flag", F.col(TARGET_COLUMN).isNotNull())
    else:
        df = df.withColumn("label_available_flag", F.lit(False))
    return df

{% if config.datetime_derivation %}
DATETIME_DERIVATION_SOURCES = {{ config.datetime_derivation.source_columns }}
MASK_FUTURE_COLUMNS = {{ config.datetime_derivation.mask_future_columns }}

def derive_datetime_features(df):
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
{% endif %}

{% if config.history_window %}
def apply_history_window(df):
{% if config.history_window.upper_limit %}
    upper = F.lit("{{ config.history_window.upper_limit }}").cast("timestamp")
{% else %}
    upper = df.agg(F.max("feature_timestamp")).collect()[0][0]
{% endif %}
{% if config.history_window.lookback_periods %}
    lookback_days = {{ config.history_window.lookback_periods }} * {{ config.history_window.cadence_days }}
    lower = F.date_sub(F.lit(upper), lookback_days)
    df = df.filter(F.col("feature_timestamp").isNull() | (F.col("feature_timestamp") >= lower))
{% endif %}
{% if config.history_window.upper_limit %}
    df = df.filter(F.col("feature_timestamp").isNull() | (F.col("feature_timestamp") <= upper))
{% endif %}
    return df
{% endif %}

# COMMAND ----------

def run_landing():
    df = load_source()
{% if config.raw_time_column %}
    df = df.withColumnRenamed("{{ config.raw_time_column }}", TIME_COLUMN)
{% endif %}
{% if config.original_target_column %}
    df = df.withColumnRenamed("{{ config.original_target_column }}", TARGET_COLUMN)
{% endif %}
    df = derive_feature_timestamp(df)
    df = derive_label_timestamp(df)
    df = derive_label_available_flag(df)
{% if config.datetime_derivation %}
    df = derive_datetime_features(df)
{% endif %}
{% if config.history_window %}
    df = apply_history_window(df)
{% endif %}
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
display(result)
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

import time

# COMMAND ----------

# MAGIC %run ./config

# COMMAND ----------

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")

# COMMAND ----------

def run_notebook(path, timeout=3600):
    print(f"Running: {path}")
    start = time.time()
    result = dbutils.notebook.run(path, timeout)
    elapsed = time.time() - start
    print(f"Completed: {path} in {elapsed:.1f}s")
    return result

# COMMAND ----------
{% if config.landing %}

# MAGIC %md
# MAGIC ## Landing Layer

# COMMAND ----------

{% for name in config.landing %}
run_notebook("landing/landing_{{ name }}")
{% endfor %}

# COMMAND ----------
{% endif %}

# MAGIC %md
# MAGIC ## Bronze Layer

# COMMAND ----------

{% for source_name in config.bronze %}
run_notebook("bronze/bronze_entity_{{ source_name }}")
{% endfor %}
{% for source_name in config.bronze_event %}
run_notebook("bronze/bronze_event_{{ source_name }}")
run_notebook("bronze/bronze_entity_{{ source_name }}_aggregated")
{% endfor %}

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
except Exception:
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

    def __init__(self, catalog: str = "main", schema: str = "default"):
        self._catalog = catalog
        self._schema = schema
        self._env = Environment(loader=InlineLoader(DATABRICKS_TEMPLATES))
        self._env.globals["render_spark_step_call"] = render_spark_step_call
        self._env.globals["group_steps"] = group_steps
        self._env.globals["spark_provenance_block"] = spark_provenance_block

    def _render(self, template_key: str, **context) -> str:
        return self._env.get_template(self._TEMPLATE_MAP[template_key]).render(**context)

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
        self, project_name: str, datasets, notebooks_base_path: str, findings_base_path: str
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
        return self._render(
            "exploration_runner",
            project_name=project_name,
            datasets=ds_list,
            target_dataset=target_dataset,
            notebooks_base_path=notebooks_base_path,
            findings_base_path=findings_base_path,
        )

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
