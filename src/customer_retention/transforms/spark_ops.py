"""Native Spark implementations of transform ops.

Mirrors ops.py but operates on native pyspark.sql.DataFrame via
``withColumn`` — never touches pyspark.pandas, so ``_psseries``
corruption is impossible.  Used by ``TransformExecutor._apply_all_distributed``.
"""

from __future__ import annotations

from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql import functions as F  # noqa: N812


def spark_impute_null(df: SparkDataFrame, column: str, *, value=0) -> SparkDataFrame:
    if column not in df.columns:
        return df
    if value == "median":
        med = df.agg(F.percentile_approx(F.col(column), 0.5).alias("v")).head()["v"]
        return df.fillna({column: float(med) if med is not None else 0})
    return df.fillna({column: value})


def spark_cap_outlier(df: SparkDataFrame, column: str, *, lower: float = 0, upper: float = 1_000_000) -> SparkDataFrame:
    if column not in df.columns:
        return df
    return df.withColumn(column, F.greatest(F.least(F.col(column), F.lit(upper)), F.lit(lower)))


def spark_type_cast(df: SparkDataFrame, column: str, *, dtype: str = "float") -> SparkDataFrame:
    if column not in df.columns:
        return df
    spark_type = {"float": "double", "int": "int", "string": "string"}.get(dtype, dtype)
    return df.withColumn(column, F.col(column).cast(spark_type))


def spark_drop_column(df: SparkDataFrame, column: str) -> SparkDataFrame:
    return df.drop(column) if column in df.columns else df


def spark_winsorize(df: SparkDataFrame, column: str, *, lower_bound: float = 0, upper_bound: float = 1_000_000) -> SparkDataFrame:
    if column not in df.columns:
        return df
    return df.withColumn(column, F.greatest(F.least(F.col(column), F.lit(upper_bound)), F.lit(lower_bound)))


def spark_segment_aware_cap(df: SparkDataFrame, column: str, *, n_segments: int = 2) -> SparkDataFrame:
    if column not in df.columns:
        return df
    quantiles = df.approxQuantile(column, [0.25, 0.75], 0.01)
    if len(quantiles) == 2:
        q1, q3 = quantiles
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        return df.withColumn(column, F.greatest(F.least(F.col(column), F.lit(upper)), F.lit(lower)))
    return df


def spark_log_transform(df: SparkDataFrame, column: str) -> SparkDataFrame:
    if column not in df.columns:
        return df
    return df.withColumn(column, F.log1p(F.greatest(F.col(column), F.lit(0)).cast("double")))


def spark_sqrt_transform(df: SparkDataFrame, column: str) -> SparkDataFrame:
    if column not in df.columns:
        return df
    return df.withColumn(column, F.sqrt(F.abs(F.col(column))))


def spark_zero_inflation(df: SparkDataFrame, column: str) -> SparkDataFrame:
    if column not in df.columns:
        return df
    return (
        df.withColumn(f"{column}_is_zero", F.when(F.col(column) == 0, 1).otherwise(0))
          .withColumn(column, F.when(F.col(column) > 0, F.log1p(F.col(column).cast("double"))).otherwise(F.lit(0.0)))
    )


def spark_cap_then_log(df: SparkDataFrame, column: str, *, q99: float | None = None) -> SparkDataFrame:
    if column not in df.columns:
        return df
    if q99 is None:
        quantiles = df.approxQuantile(column, [0.99], 0.01)
        q99 = quantiles[0] if quantiles else None
    if q99 is None:
        return df
    return df.withColumn(column, F.log1p(F.greatest(F.least(F.col(column), F.lit(q99)), F.lit(0)).cast("double")))


def spark_one_hot_encode(df: SparkDataFrame, column: str) -> SparkDataFrame:
    if column not in df.columns:
        return df
    categories = [row[column] for row in df.select(column).distinct().collect() if row[column] is not None]
    for cat in sorted(str(c) for c in categories):
        safe_name = f"{column}_{cat}".replace(" ", "_").replace("-", "_")
        df = df.withColumn(safe_name, F.when(F.col(column) == cat, 1).otherwise(0))
    return df.drop(column)


def spark_feature_select(df: SparkDataFrame, column: str) -> SparkDataFrame:
    return df.drop(column) if column in df.columns else df


def spark_derived_ratio(df: SparkDataFrame, column: str, *, numerator: str, denominator: str) -> SparkDataFrame:
    if numerator not in df.columns or denominator not in df.columns:
        return df
    return df.withColumn(column, F.col(numerator) / F.when(F.col(denominator) == 0, F.lit(None)).otherwise(F.col(denominator)))


def spark_derived_interaction(df: SparkDataFrame, column: str, *, col_a: str, col_b: str) -> SparkDataFrame:
    if col_a not in df.columns or col_b not in df.columns:
        return df
    return df.withColumn(column, F.col(col_a) * F.col(col_b))


def spark_derived_composite(df: SparkDataFrame, column: str, *, columns: list[str]) -> SparkDataFrame:
    valid = [c for c in columns if c in df.columns]
    if not valid:
        return df
    avg_expr = sum(F.coalesce(F.col(c), F.lit(0.0)) for c in valid) / len(valid)
    return df.withColumn(column, avg_expr)
