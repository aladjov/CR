"""Per-slice top SHAP drivers writer.

Computes per-row linear SHAP for the top-K entities of every
``(playbook_id, archetype_id, risk_tier)`` slice in the latest
``eligibility_snapshot`` run, reduces each row to its top-N drivers by
``|shap_contribution|``, and merges the result into ``top_shap_drivers``.

Why per-slice: the dashboard surfaces the most actionable accounts in each
slice, not a uniform 500-row tail. Running SHAP only over those rows keeps
the cost bounded by ``|slices| * K`` (typically a few thousand) while
hydrating the same rows the CSM will actually click into.

Distributed-by-construction:

- Slice ranking is one ``Window`` over ``decisioned`` columns already on
  ``eligibility_snapshot`` — no shuffle of raw features.
- Per-entity dedup is one ``distinct``; gold-features lookup is one
  windowed ``row_number`` keeping the latest ``as_of_date`` per entity.
- ``compute_shap_distributed`` is one ``select`` of ``F.lit``-bound
  attribution constants — full executor parallelism, no UDFs.
- Driver-array reduction is a single column expression (``F.array(struct...)``
  → ``F.array_sort`` → ``F.slice``) — no shuffle, fan-out bounded by the
  attribution feature count.
- The Delta merge uses :func:`delta_writer.merge_dataframe_into` so the
  source DataFrame is never collected to the driver.

The single ``count()`` action at the end is the only Spark job needed
purely for reporting; everything else feeds the downstream MERGE.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, List, Optional

from customer_retention.stages.modeling.shap_attribution import (
    ShapAttribution,
    load_attribution_from_model_uri,
)

from .delta_writer import merge_dataframe_into
from .schemas import top_shap_drivers_schema
from .shap_runner import compute_shap_distributed

logger = logging.getLogger(__name__)

if TYPE_CHECKING:  # pragma: no cover
    from pyspark.sql import Column, DataFrame, SparkSession


_DEFAULT_PER_SLICE_K: int = 20
_DEFAULT_TOP_DRIVERS_PER_ROW: int = 5
_FEATURE_BATCH: int = 100
_MERGE_KEYS = ("model_name", "model_version", "entity_id")


@dataclass
class TopDriversConfig:
    """All inputs the per-slice SHAP writer needs."""

    spark: "SparkSession"
    snapshot_table_fqn: str
    gold_features_fqn: str
    top_shap_drivers_fqn: str
    model_name: str
    model_version: str
    scoring_run_id: str
    as_of_date: datetime
    model_uri: Optional[str] = None
    attribution: Optional[ShapAttribution] = None
    entity_id_column: str = "entity_id"
    per_slice_k: int = _DEFAULT_PER_SLICE_K
    top_drivers_per_row: int = _DEFAULT_TOP_DRIVERS_PER_ROW
    probability_column: str = "churn_probability"
    value_at_risk_column: str = "value_at_risk"
    # Timestamp column on the gold table used to pick the latest row per
    # entity. ``None`` (the default) auto-detects: tries ``as_of_date``,
    # then ``event_timestamp``, then assumes the table is already
    # entity-grain (one row per entity) and skips the latest-row window
    # entirely. Set explicitly to override autodetection.
    gold_timestamp_column: Optional[str] = None


@dataclass
class TopDriversResult:
    """Counts emitted by one writer pass."""

    rows_scored: int
    drivers_per_row: int
    per_slice_k: int
    feature_count: int
    target_table_fqn: str

    def summary(self) -> str:
        return (
            f"top_shap_drivers: scored {self.rows_scored:,} entities "
            f"({self.drivers_per_row} drivers/row, K={self.per_slice_k}/slice, "
            f"{self.feature_count} attribution features) → {self.target_table_fqn}"
        )


def compute_and_write_top_shap_drivers(config: TopDriversConfig) -> TopDriversResult:
    """End-to-end driver. See module docstring for the distributed plan."""
    if config.per_slice_k <= 0:
        raise ValueError("per_slice_k must be positive")
    if config.top_drivers_per_row <= 0:
        raise ValueError("top_drivers_per_row must be positive")

    attribution = _resolve_attribution(config)
    if not attribution.feature_columns:
        raise ValueError("attribution has no feature_columns; nothing to score")

    entity_df = _select_top_per_slice(config)
    feature_df = _join_latest_gold_features(config, entity_df, attribution.feature_columns)

    shap_df = _compute_shap_in_batches(feature_df, attribution, config.entity_id_column)
    enriched = feature_df.join(shap_df, on=config.entity_id_column, how="inner")

    top_drivers_df = _emit_top_drivers_array(
        enriched, attribution.feature_columns, config.top_drivers_per_row,
    )
    target_df = _shape_target_rows(top_drivers_df, config)

    rows_scored = int(target_df.count())
    if rows_scored == 0:
        logger.warning(
            "no rows produced for scoring_run_id=%s; skipping top_shap_drivers merge",
            config.scoring_run_id,
        )
        return TopDriversResult(
            rows_scored=0,
            drivers_per_row=config.top_drivers_per_row,
            per_slice_k=config.per_slice_k,
            feature_count=len(attribution.feature_columns),
            target_table_fqn=config.top_shap_drivers_fqn,
        )

    merge_dataframe_into(
        config.spark,
        target_df,
        top_shap_drivers_schema(),
        config.top_shap_drivers_fqn,
        _MERGE_KEYS,
    )
    return TopDriversResult(
        rows_scored=rows_scored,
        drivers_per_row=config.top_drivers_per_row,
        per_slice_k=config.per_slice_k,
        feature_count=len(attribution.feature_columns),
        target_table_fqn=config.top_shap_drivers_fqn,
    )


def _resolve_attribution(config: TopDriversConfig) -> ShapAttribution:
    if config.attribution is not None:
        return config.attribution
    if not config.model_uri:
        raise ValueError(
            "TopDriversConfig requires either attribution or model_uri; got neither"
        )
    return load_attribution_from_model_uri(config.model_uri)


def _select_top_per_slice(config: TopDriversConfig) -> "DataFrame":
    """Window per ``(playbook, archetype, risk_tier)``, keep ranks ≤ K.

    Two non-obvious correctness invariants:

    (1) **PRIMARY-only slicing.** ``eligibility_snapshot`` carries one row
        per ``(entity × policy)``, so each entity has many rows and shows
        up in many ``(playbook, archetype, risk_tier)`` slices. If we
        windowed over the unfiltered snapshot, the top-K per snapshot
        slice would heavily overlap on the same high-``expected_loss``
        customers, ``.distinct()`` would collapse the cache to a tiny
        global set (~28 entities for ``per_slice_k=20`` in practice), and
        the dashboard's "In Scope" filter (which lives on
        ``v_account_primary_recommendation`` = primary-only) would see at
        most a handful per slice. We anchor on
        ``policy_rank_among_eligible = 1`` so each entity appears in
        exactly its PRIMARY slice; ``per_slice_k`` then caps that
        primary slice and the dashboard's slice and cache slice align
        1:1.

    (2) **Deterministic tiebreaker.** Low-risk slices have
        ``churn_probability ≈ 0`` and frequently NULL ``value_at_risk``,
        so ``expected_loss`` ties at 0 for thousands of entities; the
        bare ``churn_probability DESC`` secondary doesn't break those
        ties. Add ``value_at_risk DESC NULLS LAST`` (rewards populated
        VAR over NULL VAR at the same probability) and ``entity_id ASC``
        as the final-ditch deterministic key so the cache returns the
        same K entities run-over-run.
    """
    from pyspark.sql import Window
    from pyspark.sql import functions as F  # noqa: N812

    snapshot = config.spark.table(config.snapshot_table_fqn).filter(
        (F.col("scoring_run_id") == F.lit(config.scoring_run_id))
        & (F.col("policy_rank_among_eligible") == F.lit(1))
    )
    expected_loss = F.coalesce(F.col(config.probability_column), F.lit(0.0)) * F.coalesce(
        F.col(config.value_at_risk_column), F.lit(0.0)
    )
    slice_window = Window.partitionBy(
        F.col("playbook_id"), F.col("archetype_id"), F.col("risk_tier"),
    ).orderBy(
        expected_loss.desc(),
        F.col(config.probability_column).desc(),
        F.col(config.value_at_risk_column).desc_nulls_last(),
        F.col(config.entity_id_column).asc(),
    )
    ranked = snapshot.withColumn("__cr_slice_rank", F.row_number().over(slice_window))
    return ranked.filter(F.col("__cr_slice_rank") <= F.lit(int(config.per_slice_k))).select(
        F.col(config.entity_id_column)
    ).distinct()


_GOLD_TIMESTAMP_CANDIDATES: tuple[str, ...] = ("as_of_date", "event_timestamp")


def _resolve_gold_timestamp_column(
    config: TopDriversConfig, available_columns: set[str]
) -> Optional[str]:
    """Return the column name to order by when picking the latest gold row.

    - When ``config.gold_timestamp_column`` is set explicitly, require it.
      Failing fast prevents silently falling back to entity-grain on a
      typo.
    - Otherwise auto-detect among well-known names (``as_of_date``,
      ``event_timestamp``).
    - Returns ``None`` when neither name is present, signalling that the
      gold table is already entity-grain (one row per entity, no temporal
      dedupe needed).
    """
    explicit = config.gold_timestamp_column
    if explicit is not None:
        if explicit not in available_columns:
            raise ValueError(
                f"gold_timestamp_column={explicit!r} not present in "
                f"{config.gold_features_fqn}"
            )
        return explicit
    for candidate in _GOLD_TIMESTAMP_CANDIDATES:
        if candidate in available_columns:
            return candidate
    return None


def _join_latest_gold_features(
    config: TopDriversConfig,
    entity_df: "DataFrame",
    feature_columns: List[str],
) -> "DataFrame":
    """Inner-join to the most recent gold row per entity. Single windowed
    select bounded by ``|entities| * |features|`` — no driver collect.

    When the gold table has no temporal column (e.g. when the upstream
    pipeline pre-aggregates to one row per entity), the windowed dedupe is
    skipped and the gold table is used as-is. Multiple rows per entity in
    that case will fan out the inner join, so a defensive ``dropDuplicates``
    on the entity key is applied before the join.
    """
    from pyspark.sql import Window
    from pyspark.sql import functions as F  # noqa: N812

    gold = config.spark.table(config.gold_features_fqn)
    available = set(gold.columns)
    missing = [c for c in feature_columns if c not in available]
    if missing:
        raise ValueError(
            f"attribution features missing in {config.gold_features_fqn}: "
            f"{missing[:10]} (+{max(len(missing) - 10, 0)} more)"
        )
    timestamp_col = _resolve_gold_timestamp_column(config, available)
    if timestamp_col is not None:
        latest_window = Window.partitionBy(F.col(config.entity_id_column)).orderBy(
            F.col(timestamp_col).desc()
        )
        latest = (
            gold.withColumn("__cr_gold_rank", F.row_number().over(latest_window))
            .filter(F.col("__cr_gold_rank") == F.lit(1))
            .drop("__cr_gold_rank")
        )
    else:
        # Entity-grain gold: one row per entity already. ``dropDuplicates``
        # is a guard against accidental duplicates that would otherwise
        # fan out the inner join.
        logger.info(
            "gold table %s has no temporal column (looked for %s); treating as entity-grain",
            config.gold_features_fqn,
            ", ".join(_GOLD_TIMESTAMP_CANDIDATES),
        )
        latest = gold.dropDuplicates([config.entity_id_column])
    projection = [F.col(config.entity_id_column)] + [
        F.col(c).cast("double").alias(c) for c in feature_columns
    ]
    return entity_df.join(latest.select(*projection), on=config.entity_id_column, how="inner")


def _compute_shap_in_batches(
    feature_df: "DataFrame",
    attribution: ShapAttribution,
    join_key: str,
) -> "DataFrame":
    """Batch the attribution dict into groups of ``_FEATURE_BATCH`` features
    so Catalyst plans stay small even when training emitted hundreds of
    features. Each batch is one ``compute_shap_distributed`` call returning
    ``join_key + shap_<feat>``; we join the batches on ``join_key``.

    Below the batch threshold, falls back to a single ``compute_shap_distributed``
    call so the simple-case plan is unchanged.
    """
    feature_columns = list(attribution.feature_columns)
    if len(feature_columns) <= _FEATURE_BATCH:
        return _shap_for_feature_subset(feature_df, attribution, feature_columns, join_key)

    accum: Optional["DataFrame"] = None
    for start in range(0, len(feature_columns), _FEATURE_BATCH):
        batch = feature_columns[start : start + _FEATURE_BATCH]
        part = _shap_for_feature_subset(feature_df, attribution, batch, join_key)
        accum = part if accum is None else accum.join(part, on=join_key, how="inner")
    return accum  # type: ignore[return-value]


def _shap_for_feature_subset(
    feature_df: "DataFrame",
    attribution: ShapAttribution,
    feature_subset: List[str],
    join_key: str,
) -> "DataFrame":
    sub_attribution = ShapAttribution(
        importances={c: attribution.importances.get(c, 0.0) for c in feature_subset},
        background_means={c: attribution.background_means.get(c, 0.0) for c in feature_subset},
        feature_columns=feature_subset,
        sample_size=attribution.sample_size,
    )
    result = compute_shap_distributed(feature_df, sub_attribution, join_key=join_key)
    if result.shap_df is None:
        raise RuntimeError("compute_shap_distributed returned an empty result")
    return result.shap_df.select(join_key, *result.shap_columns)


def _emit_top_drivers_array(
    enriched: "DataFrame",
    feature_columns: List[str],
    top_drivers_per_row: int,
) -> "DataFrame":
    """Build per-row ``array<struct<feature,value,shap_contribution,direction>>``
    sorted by ``|shap_contribution|`` desc and sliced to ``top_drivers_per_row``.

    All arithmetic is column-level Spark expressions — single distributed
    select, no shuffle, fan-out bounded by the feature count.
    """
    from pyspark.sql import functions as F  # noqa: N812

    drivers = F.array(*[_driver_struct(name) for name in feature_columns])
    sorted_drivers = F.sort_array(drivers, asc=False)
    return enriched.select(
        F.col(enriched.columns[0]).alias(enriched.columns[0]),
        F.slice(sorted_drivers, 1, int(top_drivers_per_row)).alias("__cr_top_drivers"),
    )


def _driver_struct(feature_name: str) -> "Column":
    """Struct ordered ``(rank_key, feature, value, shap_contribution, direction)``
    so ``F.sort_array`` (which compares element-wise) ranks by ``|shap|`` desc.

    The rank key sits in the first struct field so ``sort_array(asc=False)``
    uses it as the primary key without needing a comparator lambda — the
    whole array operation stays inside Spark's optimized expression
    evaluator (no Python lambda, no UDF)."""
    from pyspark.sql import functions as F  # noqa: N812

    shap_col = F.col(f"shap_{feature_name}")
    direction = F.when(shap_col >= F.lit(0.0), F.lit("positive")).otherwise(F.lit("negative"))
    return F.struct(
        F.abs(shap_col).alias("__rank_key"),
        F.lit(feature_name).alias("feature"),
        F.col(feature_name).cast("double").alias("value"),
        shap_col.alias("shap_contribution"),
        direction.alias("direction"),
    )


def _shape_target_rows(
    drivers_df: "DataFrame",
    config: TopDriversConfig,
) -> "DataFrame":
    """Project to the ``top_shap_drivers`` schema, dropping the rank key
    field that lives only on the in-flight struct."""
    from pyspark.sql import functions as F  # noqa: N812

    cleaned = F.transform(
        F.col("__cr_top_drivers"),
        lambda d: F.struct(
            d["feature"].alias("feature"),
            d["value"].alias("value"),
            d["shap_contribution"].alias("shap_contribution"),
            d["direction"].alias("direction"),
        ),
    )
    return drivers_df.select(
        F.lit(config.model_name).alias("model_name"),
        F.lit(config.model_version).alias("model_version"),
        F.col(config.entity_id_column).alias("entity_id"),
        F.lit(config.as_of_date.replace(tzinfo=timezone.utc) if config.as_of_date.tzinfo is None
              else config.as_of_date).cast("timestamp").alias("as_of_date"),
        cleaned.alias("top_drivers"),
    )
