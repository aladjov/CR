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
        f'df.withColumn("{col}", F.col("{num}") / F.when(F.col("{den}") != 0, F.col("{den}")).otherwise(F.lit(None)))'
    )


def _derived_interaction(col, p):
    features = p.get("features", [])
    col_a = features[0] if len(features) > 0 else p.get("col_a", "")
    col_b = features[1] if len(features) > 1 else p.get("col_b", "")
    return f'df.withColumn("{col}", F.col("{col_a}") * F.col("{col_b}"))'


def _derived_composite(col, p):
    columns = p.get("columns", [])
    if not columns:
        return f'df.withColumn("{col}", F.lit(None))'
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
    return f'df.withColumn("{col}", F.log1p(F.least(F.col("{col}"), F.lit(F.col("{col}").cast("double")))))'


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
    df = df.withColumn("recency_bucket", F.when(F.col("days_since_last") <= 7, "0-7d")
        .when(F.col("days_since_last") <= 30, "7-30d")
        .when(F.col("days_since_last") <= 90, "30-90d")
        .when(F.col("days_since_last") <= 180, "90-180d")
        .when(F.col("days_since_last") <= 365, "180-365d")
        .otherwise("365d+"))
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
        F.when((F.col("days_since_first") >= tenure_med) & (F.col(intensity_cols[0]) >= intensity_med), "loyal")
        .when((F.col("days_since_first") >= tenure_med) & (F.col(intensity_cols[0]) < intensity_med), "at_risk")
        .when((F.col("days_since_first") < tenure_med) & (F.col(intensity_cols[0]) >= intensity_med), "new_active")
        .otherwise("new_inactive"))
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
    return df
{% endif %}

# COMMAND ----------

def run_bronze():
    df = load_source()
    df = apply_transformations(df)
{% if config.lifecycle %}
    df = enrich_lifecycle(df)
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

# COMMAND ----------

# MAGIC %run ../config

# COMMAND ----------

SOURCE_NAME = "{{ source }}"
ENTITY_COLUMN = "{{ config.entity_column }}"
TIME_COLUMN = "{{ config.time_column }}"

def load_source():
    source_config = SOURCES[SOURCE_NAME]
    path = source_config["path"]
    fmt = source_config["format"]
    if fmt == "csv":
        return spark.read.option("header", "true").option("inferSchema", "true").csv(path)
    return spark.read.format(fmt).load(path)

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
    window = Window.partitionBy(ENTITY_COLUMN, TIME_COLUMN).orderBy(F.monotonically_increasing_id())
    df = df.withColumn("_row_num", F.row_number().over(window))
    df = df.filter(F.col("_row_num") == 1).drop("_row_num")
    return df
{% endif %}

{% if config.aggregation %}
def _window_to_days(window_str):
    if window_str.endswith("d"):
        return int(window_str[:-1])
    if window_str.endswith("h"):
        return max(1, int(window_str[:-1]) // 24)
    return int(window_str)

def apply_event_aggregation(df):
    reference_date = df.agg(F.max(TIME_COLUMN)).collect()[0][0]
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
{% for val_col in config.aggregation.value_columns %}
{% for agg_func in config.aggregation.agg_funcs %}
{% if agg_func != "count" %}
    agg_exprs.append(F.{{ agg_func }}("{{ val_col }}").alias("{{ val_col }}_{{ agg_func }}_{{ window }}"))
{% endif %}
{% endfor %}
{% endfor %}
    window_agg = window_df.groupBy(ENTITY_COLUMN).agg(*agg_exprs)
    results.append(window_agg)
{% endfor %}
    merged = results[0]
    for r in results[1:]:
        merged = merged.join(r, on=ENTITY_COLUMN, how="outer")
    return merged, reference_date
{% endif %}

# COMMAND ----------

def run_bronze_event():
    df = load_source()
{% if config.raw_time_column %}
    df = df.withColumnRenamed("{{ config.raw_time_column }}", TIME_COLUMN)
{% endif %}
    df = apply_pre_shaping(df)
{% if config.deduplicate %}
    df = deduplicate(df)
{% endif %}
{% if config.aggregation %}
    agg_df, reference_date = apply_event_aggregation(df)
    output_table = bronze_table("{{ source }}_events")
    agg_df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(output_table)
    return agg_df
{% else %}
    output_table = bronze_table("{{ source }}_events")
    df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(output_table)
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
    raw_df = spark.table(bronze_table("{{ raw_source }}_events"))
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
    df = df.withColumn("recency_bucket", F.when(F.col("days_since_last") <= 7, "0-7d")
        .when(F.col("days_since_last") <= 30, "7-30d")
        .when(F.col("days_since_last") <= 90, "30-90d")
        .when(F.col("days_since_last") <= 180, "90-180d")
        .when(F.col("days_since_last") <= 365, "180-365d")
        .otherwise("365d+"))
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
        F.when((F.col("days_since_first") >= tenure_med) & (F.col(intensity_cols[0]) >= intensity_med), "loyal")
        .when((F.col("days_since_first") >= tenure_med) & (F.col(intensity_cols[0]) < intensity_med), "at_risk")
        .when((F.col("days_since_first") < tenure_med) & (F.col(intensity_cols[0]) >= intensity_med), "new_active")
        .otherwise("new_inactive"))
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
    return df

result = run_bronze_entity()
display(result)
""",
    "databricks_silver.py.j2": """# Databricks notebook source
# MAGIC %md
# MAGIC # Silver: Feature Set {{ config.composite_name or config.name }}

# COMMAND ----------

from pyspark.sql import functions as F

# COMMAND ----------

# MAGIC %run ../config

# COMMAND ----------

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

def merge_sources(bronze_outputs):
    base_source = "{{ config.sources[0].name }}"
    merged = bronze_outputs[base_source]
{% for join in config.silver.joins %}
    merged = merged.join(
        bronze_outputs["{{ join.right_source }}"],
        merged["{{ join.left_key }}"] == bronze_outputs["{{ join.right_source }}"]["{{ join.right_key }}"],
        "{{ join.how }}",
    ).drop(bronze_outputs["{{ join.right_source }}"]["{{ join.right_key }}"])
{% endfor %}
    return merged

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
    categories = [row[col] for row in df.select(col).distinct().collect() if row[col] is not None]
    for cat in sorted(categories):
        safe_name = f"{col}_{cat}".replace(" ", "_").replace("-", "_")
        df = df.withColumn(safe_name, F.when(F.col(col) == cat, 1).otherwise(0))
    df = df.drop(col)
    return df

def _label_encode(df, col):
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
{% for step in config.gold.scalings %}
    # {{ step.rationale }}
    df = {{ render_spark_step_call(step) }}
{% endfor %}
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
    output_table = gold_table()
    df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(output_table)
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

# COMMAND ----------

# MAGIC %run ../config

# COMMAND ----------

TARGET = TARGET_COLUMN

def load_training_data():
    return spark.table(gold_table())

def prepare_features(df):
    exclude_prefixes = ["original_"]
    feature_cols = [
        c for c in df.columns
        if c != TARGET and not any(c.startswith(p) for p in exclude_prefixes)
        and df.schema[c].dataType.typeName() in ("double", "float", "integer", "long", "short")
    ]
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features", handleInvalid="skip")
    assembled = assembler.transform(df).select("features", F.col(TARGET).alias("label"))
    return assembled, feature_cols

def train_and_evaluate():
    df = load_training_data()
    assembled, feature_cols = prepare_features(df)
    train_df, test_df = assembled.randomSplit([0.8, 0.2], seed=42)

    mlflow.set_experiment(f"/Shared/{PIPELINE_NAME}")
    mlflow.set_tag("composite_name", COMPOSITE_NAME)

    models = {
        "LogisticRegression": LogisticRegression(maxIter=100, featuresCol="features", labelCol="label"),
        "RandomForest": RandomForestClassifier(numTrees=100, featuresCol="features", labelCol="label"),
        "GBTClassifier": GBTClassifier(maxIter=50, featuresCol="features", labelCol="label"),
    }

    best_model_name = None
    best_auc = 0.0
    best_model = None
    binary_eval = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC")
    multi_eval = MulticlassClassificationEvaluator(labelCol="label", metricName="f1")

    for name, model in models.items():
        with mlflow.start_run(run_name=name, nested=True):
            fitted = model.fit(train_df)
            predictions = fitted.transform(test_df)
            auc = binary_eval.evaluate(predictions)
            f1 = multi_eval.evaluate(predictions)
            mlflow.log_metric("auc", auc)
            mlflow.log_metric("f1", f1)
            mlflow.log_param("model_type", name)
            mlflow.log_param("num_features", len(feature_cols))
            mlflow.spark.log_model(fitted, f"model_{name}")
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

def run_notebook(path, timeout=3600):
    print(f"Running: {path}")
    start = time.time()
    result = dbutils.notebook.run(path, timeout)
    elapsed = time.time() - start
    print(f"Completed: {path} in {elapsed:.1f}s")
    return result

# COMMAND ----------

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
}


class DatabricksCodeRenderer:
    _TEMPLATE_MAP = {
        "config": "databricks_config.py.j2",
        "bronze": "databricks_bronze.py.j2",
        "bronze_event": "databricks_bronze_event.py.j2",
        "bronze_entity": "databricks_bronze_entity.py.j2",
        "silver": "databricks_silver.py.j2",
        "gold": "databricks_gold.py.j2",
        "training": "databricks_training.py.j2",
        "runner": "databricks_runner.py.j2",
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
