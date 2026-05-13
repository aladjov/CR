"""Native Spark implementations of transform ops.

Mirrors ops.py but operates on native pyspark.sql.DataFrame via
``withColumn`` — never touches pyspark.pandas, so ``_psseries``
corruption is impossible.  Used by ``TransformExecutor._apply_all_distributed``.
"""

from __future__ import annotations

from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql import functions as F  # noqa: N812

from customer_retention.core.naming import sanitize_column_token


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
          .withColumn(f"{column}_log", F.when(F.col(column) > 0, F.log1p(F.col(column).cast("double"))).otherwise(F.lit(0.0)))
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
    # Dedup categories whose sanitized safe_names collide (Salesforce
    # whitespace padding, punctuation variants, case differences). Group
    # raw values that map to the same safe_name and encode as a single
    # `isin`-based indicator so two raw values ("1" and " 1") don't emit
    # two columns aliased identically and trip Delta's case-insensitive
    # duplicate detection at saveAsTable.
    safe_to_raw = {}
    for cat in sorted(str(c) for c in categories):
        safe_name = f"{column}_{sanitize_column_token(cat)}"
        key = safe_name.lower()
        safe_to_raw.setdefault(key, (safe_name, []))[1].append(cat)
    for safe_name, group in safe_to_raw.values():
        if len(group) == 1:
            df = df.withColumn(safe_name, F.when(F.col(column) == group[0], 1).otherwise(0))
        else:
            df = df.withColumn(safe_name, F.when(F.col(column).isin(group), 1).otherwise(0))
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


def spark_standard_scale(df: SparkDataFrame, column: str, *, mean: float, scale: float) -> SparkDataFrame:
    if column not in df.columns:
        return df
    return df.withColumn(column, (F.col(column) - mean) / scale)


def spark_minmax_scale(df: SparkDataFrame, column: str, *, scale: float, offset: float) -> SparkDataFrame:
    if column not in df.columns:
        return df
    return df.withColumn(column, F.col(column) * scale + offset)


def spark_yeo_johnson(df: SparkDataFrame, column: str, *, lmbda: float, std_mean: float | None = None, std_scale: float | None = None, standardize: bool = True) -> SparkDataFrame:
    if column not in df.columns:
        return df
    c = F.coalesce(F.col(column), F.lit(0.0))
    if abs(lmbda) < 1e-12:
        pos_expr = F.log1p(c)
    else:
        pos_expr = (F.pow(c + 1, lmbda) - 1) / lmbda
    if abs(lmbda - 2) < 1e-12:
        neg_expr = -F.log1p(-c)
    else:
        neg_expr = -(F.pow(-c + 1, 2 - lmbda) - 1) / (2 - lmbda)
    yj = F.when(c >= 0, pos_expr).otherwise(neg_expr)
    if standardize and std_mean is not None and std_scale is not None:
        yj = (yj - std_mean) / std_scale
    return df.withColumn(column, yj)


# ── Batch ops (single select instead of N withColumn) ─────────────


def spark_batch_log_transform(df: SparkDataFrame, columns: list[str]) -> SparkDataFrame:
    valid = set(columns) & set(df.columns)
    if not valid:
        return df
    return df.select(*[
        F.log1p(F.greatest(F.col(c), F.lit(0)).cast("double")).alias(c) if c in valid else F.col(c)
        for c in df.columns
    ])


def spark_batch_sqrt_transform(df: SparkDataFrame, columns: list[str]) -> SparkDataFrame:
    valid = set(columns) & set(df.columns)
    if not valid:
        return df
    return df.select(*[
        F.sqrt(F.abs(F.col(c))).alias(c) if c in valid else F.col(c)
        for c in df.columns
    ])


def spark_batch_zero_inflation(df: SparkDataFrame, columns: list[str]) -> SparkDataFrame:
    valid = set(columns) & set(df.columns)
    if not valid:
        return df
    existing = [F.col(c) for c in df.columns]
    log_cols = [
        F.when(F.col(c) > 0, F.log1p(F.col(c).cast("double"))).otherwise(F.lit(0.0)).alias(f"{c}_log")
        for c in columns if c in valid
    ]
    flags = [F.when(F.col(c) == 0, 1).otherwise(0).alias(f"{c}_is_zero") for c in columns if c in valid]
    return df.select(*existing, *log_cols, *flags)


def spark_batch_cap_then_log(df: SparkDataFrame, columns_q99: list[tuple[str, float | None]]) -> SparkDataFrame:
    valid_map = {c: q for c, q in columns_q99 if c in df.columns and q is not None}
    if not valid_map:
        return df
    return df.select(*[
        F.log1p(F.greatest(F.least(F.col(c), F.lit(valid_map[c])), F.lit(0)).cast("double")).alias(c)
        if c in valid_map else F.col(c)
        for c in df.columns
    ])


def _spark_yj_expr(col_name: str, lmbda: float, std_mean: float | None, std_scale: float | None, standardize: bool):
    c = F.coalesce(F.col(col_name), F.lit(0.0))
    pos_expr = F.log1p(c) if abs(lmbda) < 1e-12 else (F.pow(c + 1, lmbda) - 1) / lmbda
    neg_expr = -F.log1p(-c) if abs(lmbda - 2) < 1e-12 else -(F.pow(-c + 1, 2 - lmbda) - 1) / (2 - lmbda)
    yj = F.when(c >= 0, pos_expr).otherwise(neg_expr)
    if standardize and std_mean is not None and std_scale is not None:
        yj = (yj - std_mean) / std_scale
    return yj.alias(col_name)


def spark_batch_yeo_johnson(df: SparkDataFrame, params_list: list[tuple[str, float, float | None, float | None, bool]]) -> SparkDataFrame:
    param_map = {col: (lmbda, mean, scale, std) for col, lmbda, mean, scale, std in params_list if col in df.columns}
    if not param_map:
        return df
    return df.select(*[
        _spark_yj_expr(c, *param_map[c]) if c in param_map else F.col(c)
        for c in df.columns
    ])


def spark_batch_standard_scale(df: SparkDataFrame, items: list[tuple[str, float, float]]) -> SparkDataFrame:
    # Collapse N standard-scale steps into one select(*). The per-column
    # equivalent is `df.withColumn(c, (col(c)-mean)/scale)` chained N
    # times, which at ~3000 SCALE steps produces a plan deep enough that
    # `localCheckpoint(eager=True)` over Spark Connect SIGSEGVs inside
    # plan canonicalisation. One projection has constant plan depth.
    by_col = {col: (mean, scale) for col, mean, scale in items if col in df.columns}
    if not by_col:
        return df
    return df.select(*[
        ((F.col(c) - by_col[c][0]) / by_col[c][1]).alias(c) if c in by_col else F.col(c)
        for c in df.columns
    ])


def spark_batch_minmax_scale(df: SparkDataFrame, items: list[tuple[str, float, float]]) -> SparkDataFrame:
    by_col = {col: (scale, offset) for col, scale, offset in items if col in df.columns}
    if not by_col:
        return df
    return df.select(*[
        (F.col(c) * by_col[c][0] + by_col[c][1]).alias(c) if c in by_col else F.col(c)
        for c in df.columns
    ])


def spark_batch_one_hot_encode(df: SparkDataFrame, columns: list[str], chunk_size: int = 500) -> SparkDataFrame:
    # Bit-identical to applying spark_one_hot_encode per column in order,
    # but collapses N `distinct().collect()` Spark jobs into one
    # `collect_set` aggregation and N×K `withColumn` calls into a single
    # chunked select. At 25 categorical columns each scanning 200K+ rows
    # this turns ~10 minutes of sequential per-column jobs into seconds.
    valid = [c for c in columns if c in df.columns]
    if not valid:
        return df
    # Narrow projection so the agg plan stays shallow on wide post-transform
    # frames where Catalyst's adaptive resolver is otherwise expensive.
    distinct_exprs = [F.collect_set(F.col(c)).alias(c) for c in valid]
    row = df.select(*valid).agg(*distinct_exprs).collect()[0]
    new_exprs = []
    drop_cols = []
    for c in valid:
        cats = [v for v in (row[c] or []) if v is not None]
        if not cats:
            continue
        # Dedup by sanitized safe_name (case-insensitive). Two raw values
        # whose tokens collapse to the same token are coalesced via isin
        # so saveAsTable doesn't reject the duplicate alias.
        safe_to_raw: dict[str, tuple[str, list]] = {}
        for cat in sorted(str(v) for v in cats):
            safe_name = f"{c}_{sanitize_column_token(cat)}"
            key = safe_name.lower()
            safe_to_raw.setdefault(key, (safe_name, []))[1].append(cat)
        for safe_name, group in safe_to_raw.values():
            if len(group) == 1:
                expr = F.when(F.col(c) == group[0], 1).otherwise(0)
            else:
                expr = F.when(F.col(c).isin(group), 1).otherwise(0)
            new_exprs.append(expr.alias(safe_name))
        drop_cols.append(c)
    if not new_exprs:
        return df
    base_cols = [F.col(c) for c in df.columns]
    for start in range(0, len(new_exprs), chunk_size):
        chunk = new_exprs[start:start + chunk_size]
        df = df.select(*base_cols, *chunk)
        base_cols = [F.col(c) for c in df.columns]
    if drop_cols:
        df = df.drop(*drop_cols)
    return df
