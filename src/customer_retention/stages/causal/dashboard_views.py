"""Loader + publisher for the six causal-track dashboard SQL views.

The view DDLs live as a single ``sql/dashboard_views.sql`` file shipped
inside the package. This module reads the file, substitutes the
``{catalog}`` / ``{schema}`` placeholders, splits it into individual
``CREATE OR REPLACE VIEW`` statements, and submits each one to Spark.

Keeping the SQL in a static file (rather than concatenated string
literals in Python) makes the views diffable, lintable with standard SQL
tooling, and reviewable in a dedicated ``.sql`` file by SQL-fluent
operators.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from importlib.resources import files
from typing import TYPE_CHECKING, List, Optional, Sequence

logger = logging.getLogger(__name__)

if TYPE_CHECKING:  # pragma: no cover
    from pathlib import Path  # noqa: F401 -- referenced in annotations

    from pyspark.sql import SparkSession


_VIEW_FILE_NAME = "dashboard_views.sql"

# Sentinel-delimited block in the SQL file that is only emitted when a concrete
# composite_name is supplied. Keep the markers in sync with the .sql file.
_DEVIATION_BLOCK_OPEN = "-- @cr:deviation-block:open"
_DEVIATION_BLOCK_CLOSE = "-- @cr:deviation-block:close"

# Sentinel-delimited block for v_feature_provenance — gated on the optional
# feature_meta + column_descriptions tables existing in UC. Older causal-track
# builds did not produce these tables; the view's CREATE OR REPLACE would fail
# at validation time on those clusters and take the whole publish call down.
_PROVENANCE_BLOCK_OPEN = "-- @cr:provenance-block:open"
_PROVENANCE_BLOCK_CLOSE = "-- @cr:provenance-block:close"

DASHBOARD_VIEW_NAMES: tuple[str, ...] = (
    "v_ranked_at_risk_customers",
    "v_archetype_overview",
    "v_playbook_eligibility_rules",
    "v_holdout_assignments",
    "v_capacity_utilization",
    "v_run_anchor_history",
    "v_account_primary_recommendation",
    "v_portfolio_risk_matrix",
    "v_playbook_archetype_rollup",
    "v_eligible_all_playbooks",
    "v_account_explanation",
    "v_run_context",
)
DASHBOARD_DEVIATION_VIEW_NAMES: tuple[str, ...] = (
    "v_account_feature_deviation",
    "v_account_feature_deviation_topn",
)
DASHBOARD_PROVENANCE_VIEW_NAMES: tuple[str, ...] = (
    "v_feature_provenance",
)
DASHBOARD_TEMPLATE_VIEW_NAMES: tuple[str, ...] = (
    "v_dashboard_template_active",
)
DASHBOARD_TEMPLATE_TABLE_NAMES: tuple[str, ...] = (
    "dashboard_template_overrides",
)


def load_dashboard_view_sql() -> str:
    """Return the raw SQL text shipped at ``stages/causal/sql/dashboard_views.sql``."""
    resource = files("customer_retention.stages.causal.sql") / _VIEW_FILE_NAME
    return resource.read_text(encoding="utf-8")


def _strip_deviation_block(sql_text: str) -> str:
    """Drop the ``@cr:deviation-block`` section so the rendered SQL parses.

    The deviation views reference ``gold_features_{composite_name}`` which has
    no inert default; when no composite name is supplied the block is excised
    rather than left with an unbound placeholder.
    """
    pattern = re.compile(
        re.escape(_DEVIATION_BLOCK_OPEN) + r".*?" + re.escape(_DEVIATION_BLOCK_CLOSE),
        re.DOTALL,
    )
    return pattern.sub("", sql_text)


def _strip_deviation_markers(sql_text: str) -> str:
    """Drop the sentinel-marker lines themselves so they don't survive as
    standalone comment-only chunks after ``split_view_statements``.
    """
    return "\n".join(
        line for line in sql_text.splitlines()
        if line.strip() not in (_DEVIATION_BLOCK_OPEN, _DEVIATION_BLOCK_CLOSE)
    )


def _strip_provenance_block(sql_text: str) -> str:
    """Drop the ``@cr:provenance-block`` section so the rendered SQL parses
    on clusters that don't yet have ``feature_meta`` / ``column_descriptions``.
    """
    pattern = re.compile(
        re.escape(_PROVENANCE_BLOCK_OPEN) + r".*?" + re.escape(_PROVENANCE_BLOCK_CLOSE),
        re.DOTALL,
    )
    return pattern.sub("", sql_text)


def _strip_provenance_markers(sql_text: str) -> str:
    return "\n".join(
        line for line in sql_text.splitlines()
        if line.strip() not in (_PROVENANCE_BLOCK_OPEN, _PROVENANCE_BLOCK_CLOSE)
    )


_GOLD_DEDUP_ORDER_PREFERENCE: tuple[str, ...] = (
    "as_of_date",
    "inference_point_in_time",
    "scoring_run_id",
)


def _resolve_gold_dedup_order_by(gold_columns: Optional[Sequence[str]]) -> str:
    """Choose the ORDER BY expression for the deviation view's gold dedupe.

    The deviation view's ``gold_latest`` CTE uses ``ROW_NUMBER() OVER
    (PARTITION BY entity_id ORDER BY <expr>)`` to pick one row per entity.
    Different gold schemas have different timestamp columns:

      • ``as_of_date`` — most pipelines after the temporal merge
      • ``inference_point_in_time`` — scoring-time stamp on some flavors
      • ``scoring_run_id`` — last-resort lexicographic ordering

    When none of these are present, fall back to ``1`` — ``ORDER BY 1``
    sorts by the first selected column, which is deterministic per
    schema and keeps Spark from rejecting the DDL. The pick may be
    arbitrary in this fallback path, but every entity still ends up
    with exactly one row, which is what the topN ranking needs.
    """
    if not gold_columns:
        return "1"
    available = set(gold_columns)
    for col in _GOLD_DEDUP_ORDER_PREFERENCE:
        if col in available:
            return f"`{col}` DESC"
    return "1"


def render_dashboard_view_sql(
    catalog: str,
    schema: str,
    *,
    composite_name: Optional[str] = None,
    include_provenance: bool = True,
    gold_struct_cols: Optional[List[str]] = None,
    gold_columns: Optional[Sequence[str]] = None,
) -> str:
    """Substitute ``{catalog}`` / ``{schema}`` (and optionally ``{composite_name}``)
    into the raw SQL template.

    When ``composite_name`` is omitted, the deviation views (which reference
    ``gold_features_{composite_name}``) are stripped from the output so the
    remaining DDL stays parseable. ``include_provenance=False`` strips the
    ``v_feature_provenance`` block — used by the publisher when the optional
    ``feature_meta`` / ``column_descriptions`` tables aren't materialized yet
    so the view's CREATE OR REPLACE wouldn't validate.

    ``gold_struct_cols`` is the comma-separated list of NUMERIC columns from
    ``gold_features_{composite_name}`` that get unpivoted into the deviation
    view's ``gold_long`` CTE. Required when the deviation block is included
    (i.e. ``composite_name`` is supplied). The publisher introspects the
    gold table's schema and passes the filtered list; tests pass an explicit
    list. Without this, ``STRUCT(*)`` would include non-double columns
    (entity_id, as_of_date, scoring_run_id) and ``FROM_JSON`` would return
    NULL for the entire map — silently zeroing the deviation view.

    ``gold_columns`` is the FULL column list of ``gold_features_{composite_name}``
    (not just the numeric ones). Used to pick the dedup ORDER BY in the
    ``gold_latest`` CTE — different pipelines have different timestamp
    columns (``as_of_date`` vs. ``inference_point_in_time`` vs. neither).
    Without this, the SQL hardcodes ``as_of_date`` and fails to publish
    on schemas that don't carry it.
    """
    text = load_dashboard_view_sql()
    if composite_name:
        text = _strip_deviation_markers(text).replace("{composite_name}", composite_name)
        # ``gold_struct_cols`` being empty silently substitutes ``STRUCT(1)``
        # which makes the deviation view return zero rows for every entity
        # (the JOIN ``feature_name = "col1"`` matches nothing in
        # ``feature_population_stats``). Warn loudly when callers omit it
        # despite asking for the deviation block -- this is exactly the
        # regression that took ``refresh_dashboard_view_materializations``
        # offline for the SPS engagement until we caught it via probe.
        if not gold_struct_cols:
            logger.warning(
                "render_dashboard_view_sql: composite_name=%r supplied without "
                "gold_struct_cols. The deviation view body will substitute "
                "STRUCT(1) and produce zero rows. Callers should introspect "
                "gold_features_%s and pass gold_struct_cols (numeric columns) "
                "+ gold_columns (all columns).",
                composite_name, composite_name,
            )
        struct_args = ", ".join(f"`{c}`" for c in (gold_struct_cols or [])) or "1"
        text = text.replace("{gold_struct_cols}", struct_args)
        text = text.replace(
            "{gold_dedup_order_by}",
            _resolve_gold_dedup_order_by(gold_columns),
        )
    else:
        text = _strip_deviation_block(text)
    if include_provenance:
        text = _strip_provenance_markers(text)
    else:
        text = _strip_provenance_block(text)
    return text.replace("{catalog}", catalog).replace("{schema}", schema)


_GOLD_NUMERIC_SPARK_TYPES = (
    "DoubleType", "FloatType", "IntegerType", "LongType", "ShortType", "ByteType",
    "DecimalType",
)
_GOLD_EXCLUDED_COLS = frozenset({
    "entity_id", "as_of_date", "scoring_run_id", "model_name", "model_version",
    "inference_point_in_time",
})


def _gold_numeric_columns(spark: "SparkSession", gold_fqn: str) -> List[str]:
    """Return the numeric (double-castable) columns of ``gold_features_<CN>``.

    Used by ``publish_dashboard_views`` to substitute into the deviation
    view's ``gold_long`` CTE so ``STRUCT(...)`` only carries columns that
    can be JSON-parsed as DOUBLE. Excludes well-known metadata columns
    (entity_id, as_of_date, scoring_run_id, ...) even when they happen to
    be numeric — they're metadata, not features.

    Returns an empty list when the table doesn't exist or the introspection
    fails; the caller treats that as "publish without the deviation block".
    """
    try:
        struct_type = spark.table(gold_fqn).schema
    except Exception as exc:  # noqa: BLE001 — best-effort introspection
        logger.warning("could not introspect %s schema for numeric cols: %s", gold_fqn, exc)
        return []
    out: List[str] = []
    for field in struct_type:
        name = field.name
        if name in _GOLD_EXCLUDED_COLS:
            continue
        type_name = type(field.dataType).__name__
        if type_name in _GOLD_NUMERIC_SPARK_TYPES:
            out.append(name)
    return out


def _gold_all_columns(spark: "SparkSession", gold_fqn: str) -> List[str]:
    """Return ``gold_features_<CN>``'s full column list.

    Used by ``render_dashboard_view_sql`` to choose the ORDER BY for the
    deviation view's ``gold_latest`` CTE. Different pipelines emit gold
    with different timestamp columns (``as_of_date`` vs.
    ``inference_point_in_time``) — or none — so the SQL must be
    parameterized rather than hardcoded.
    """
    try:
        struct_type = spark.table(gold_fqn).schema
    except Exception as exc:  # noqa: BLE001 — best-effort introspection
        logger.warning("could not introspect %s schema for all cols: %s", gold_fqn, exc)
        return []
    return [field.name for field in struct_type]


_VIEW_NAME_RE = re.compile(
    r"\bCREATE\s+(?:OR\s+REPLACE\s+)?VIEW\s+(?:IF\s+NOT\s+EXISTS\s+)?"
    r"(`?[\w.]+`?(?:\.`?[\w.]+`?){0,2})",
    re.IGNORECASE,
)


def _extract_view_fqn(stmt: str) -> Optional[str]:
    """Return the fully-qualified view name from a CREATE [OR REPLACE] VIEW
    statement, or ``None`` if the statement isn't a view creation. Used by
    ``publish_dashboard_views`` to issue ``DROP VIEW IF EXISTS`` before the
    CREATE so the stored view schema is reset on every publish.
    """
    match = _VIEW_NAME_RE.search(stmt)
    return match.group(1) if match else None


def split_view_statements(sql_text: str) -> List[str]:
    """Split a multi-statement SQL string on semicolons.

    A statement-ending ``;`` only counts when it sits outside ``--`` line
    comments and outside single-quoted string literals.  This lets
    operator-supplied SQL include English prose like
    ``-- NB03 materialises and the scoring path consumes; reading from it``
    inside CTE comments without the splitter cutting the statement in two
    and handing Spark a truncated DDL (which surfaces as ``[PARSE_SYNTAX_ERROR]
    Syntax error at or near end of input``).

    Comments (``--`` lines) are kept on the preceding statement so view
    headers stay readable in error messages. Empty trailing statements
    after the final semicolon are dropped.
    """
    statements: List[str] = []
    buf: List[str] = []
    in_line_comment = False
    in_single_quote = False
    in_backtick = False
    i = 0
    n = len(sql_text)
    while i < n:
        ch = sql_text[i]
        nxt = sql_text[i + 1] if i + 1 < n else ""
        if in_line_comment:
            buf.append(ch)
            if ch == "\n":
                in_line_comment = False
            i += 1
            continue
        if in_single_quote:
            buf.append(ch)
            # Handle escaped quote ``''`` (SQL-standard).
            if ch == "'" and nxt == "'":
                buf.append(nxt)
                i += 2
                continue
            if ch == "'":
                in_single_quote = False
            i += 1
            continue
        if in_backtick:
            buf.append(ch)
            if ch == "`":
                in_backtick = False
            i += 1
            continue
        if ch == "-" and nxt == "-":
            in_line_comment = True
            buf.append(ch)
            i += 1
            continue
        if ch == "'":
            in_single_quote = True
            buf.append(ch)
            i += 1
            continue
        if ch == "`":
            in_backtick = True
            buf.append(ch)
            i += 1
            continue
        if ch == ";":
            stripped = "".join(buf).strip()
            if stripped:
                statements.append(stripped)
            buf = []
            i += 1
            continue
        buf.append(ch)
        i += 1
    tail = "".join(buf).strip()
    if tail:
        statements.append(tail)
    return statements


_DEVIATION_PREREQ_TABLES: tuple[str, ...] = ("feature_population_stats",)
_POPULATION_STATS_SIDECAR_FILENAME = "feature_population_stats.json"


def _try_materialize_population_stats_from_sidecar(
    spark: "SparkSession",
    catalog: str,
    schema: str,
    experiments_dir: "Optional[Path | str]" = None,
) -> bool:
    """Best-effort: materialize the population-stats JSON sidecar into UC.

    The framework writes population stats to a JSON sidecar in the run
    namespace; the deviation views read a UC Delta table. When the table
    is missing but the sidecar exists, materializing it on the fly lets
    ``publish_dashboard_views`` continue without operator intervention.

    ``experiments_dir`` is the run-namespace root. On Databricks
    multi-task jobs env vars don't propagate between notebook tasks, so
    ``RunNamespace.from_env_or_latest()`` returns None and the sidecar
    is never located. Passing an explicit root unblocks the in-root
    sentinel / latest-marker discovery tiers that don't require env vars.

    Returns ``True`` when the table now exists (whether already-present
    or just-materialized), ``False`` when neither the table nor a
    discoverable sidecar is available.
    """
    fqn = f"{catalog}.{schema}.feature_population_stats"
    if spark.catalog.tableExists(fqn):
        return True
    try:
        from customer_retention.analysis.auto_explorer.run_namespace import RunNamespace
    except ImportError:
        return False
    root_path: "Optional[Path]" = None
    if experiments_dir is not None:
        from pathlib import Path as _Path
        root_path = _Path(str(experiments_dir))
    try:
        ns = RunNamespace.from_env_or_latest(root=root_path)
    except Exception:  # noqa: BLE001 -- best-effort, never fail the publish
        ns = None
    if ns is None:
        return False
    sidecar = ns.feature_population_stats_dir / _POPULATION_STATS_SIDECAR_FILENAME
    if not sidecar.exists():
        return False
    try:
        from .population_stats import materialize_population_stats_from_sidecar

        materialize_population_stats_from_sidecar(spark, sidecar, fqn)
    except Exception as exc:  # noqa: BLE001 -- log + degrade
        logger.warning(
            "auto-materialization of %s from sidecar %s failed: %s. "
            "Deviation views will be skipped this run.",
            fqn, sidecar, exc,
        )
        return spark.catalog.tableExists(fqn)
    return spark.catalog.tableExists(fqn)


_PROVENANCE_PREREQ_TABLES: tuple[str, ...] = (
    "feature_meta",
    "column_descriptions",
)


def _try_materialize_feature_meta_from_sidecar(
    spark: "SparkSession",
    catalog: str,
    schema: str,
    experiments_dir: "Optional[Path | str]" = None,
) -> bool:
    """Best-effort: materialize the ``feature_meta`` JSON sidecar into UC.

    Mirrors ``_try_materialize_population_stats_from_sidecar``. The
    framework writes feature lineage as a JSON sidecar in the run
    namespace at gold-materialization time; the provenance view reads a
    UC Delta table. When the table is missing but the sidecar exists,
    materializing it on the fly lets ``publish_dashboard_views`` continue
    without operator intervention. ``experiments_dir`` is the run-
    namespace root (see the population-stats helper for why this is load-
    bearing on Databricks). Returns ``True`` when the table now exists,
    ``False`` when neither table nor sidecar is reachable.
    """
    fqn = f"{catalog}.{schema}.feature_meta"
    if spark.catalog.tableExists(fqn):
        return True
    try:
        from customer_retention.analysis.auto_explorer.run_namespace import RunNamespace
        from customer_retention.stages.causal.feature_meta_writer import (
            FeatureMetaConfig,
            write_feature_meta,
        )
        from customer_retention.stages.causal.interpretation.sidecars import (
            load_feature_meta_sidecar,
        )
    except ImportError:
        return False
    root_path: "Optional[Path]" = None
    if experiments_dir is not None:
        from pathlib import Path as _Path
        root_path = _Path(str(experiments_dir))
    try:
        ns = RunNamespace.from_env_or_latest(root=root_path)
    except Exception:  # noqa: BLE001 -- best-effort, never fail the publish
        ns = None
    if ns is None:
        return False
    sidecar = load_feature_meta_sidecar(ns) or {}
    if not sidecar:
        return False
    try:
        rows = list(sidecar.values())
        write_feature_meta(
            FeatureMetaConfig(
                spark=spark, table_fqn=fqn, run_id=ns.run_id, rows=rows,
            )
        )
    except Exception as exc:  # noqa: BLE001 -- log + degrade
        logger.warning(
            "auto-materialization of %s from sidecar at %s failed: %s. "
            "v_feature_provenance will be skipped this run.",
            fqn, ns.feature_meta_dir, exc,
        )
        return spark.catalog.tableExists(fqn)
    return spark.catalog.tableExists(fqn)


def _synthetic_placeholder_column_descriptions():
    """Synthetic ``ColumnDescriptionRow``s for placeholder source-column tokens.

    ``parse_aggregation_feature_name`` emits synthetic source-column names
    like ``event`` (for ``event_count_*``) and ``event_gap`` (for
    ``inter_event_gap_*``). Those tokens are not real columns on any
    bronze/landing table, so the LEFT JOIN in ``v_feature_provenance``
    that produces ``source_column_defs[].business_definition`` returns
    NULL for every event-derived feature unless we seed a row for each
    placeholder. Operators see these phrases in the dashboard's Feature
    dictionary panel under each SHAP driver.
    """
    from customer_retention.stages.causal.column_descriptions_writer import (
        ColumnDescriptionRow,
    )
    return [
        ColumnDescriptionRow(
            table="__synthetic__",
            column_name="event",
            business_name="Engagement event",
            business_definition=(
                "A single engagement record from the source landing/bronze "
                "table — one row per event such as an email send, open, "
                "click, unsubscribe, or bounce. ``event_count_*`` features "
                "count rows of this kind within a rolling time window."
            ),
            source="framework_synthetic",
        ),
        ColumnDescriptionRow(
            table="__synthetic__",
            column_name="event_gap",
            business_name="Inter-event gap",
            business_definition=(
                "Time delta in days between two consecutive engagement "
                "events for the same entity. ``inter_event_gap_*`` features "
                "aggregate this gap series with min / max / mean."
            ),
            source="framework_synthetic",
        ),
    ]


def _seed_synthetic_column_descriptions(
    spark: "SparkSession", catalog: str, schema: str
) -> None:
    """Upsert the synthetic placeholder rows. Idempotent — safe on re-run."""
    fqn = f"{catalog}.{schema}.column_descriptions"
    try:
        from customer_retention.stages.causal.column_descriptions_writer import (
            ColumnDescriptionsConfig,
            write_column_descriptions,
        )
        write_column_descriptions(
            ColumnDescriptionsConfig(
                spark=spark, table_fqn=fqn,
                rows=_synthetic_placeholder_column_descriptions(),
            )
        )
    except Exception as exc:  # noqa: BLE001 — best-effort; never fail the publish
        logger.warning(
            "could not seed synthetic placeholder column_descriptions into %s: %s",
            fqn, exc,
        )


def _try_materialize_column_descriptions_from_sidecar(
    spark: "SparkSession",
    catalog: str,
    schema: str,
    experiments_dir: "Optional[Path | str]" = None,
) -> bool:
    """Best-effort: materialize the ``column_descriptions`` JSON sidecar into UC.

    Always seeds the synthetic placeholder rows for ``event`` / ``event_gap``
    too — those are required so the dashboard's Feature dictionary panel
    shows a business definition for event-derived features (which use those
    tokens as their source_column placeholders). ``experiments_dir`` is the
    run-namespace root (see population-stats helper docstring).
    """
    fqn = f"{catalog}.{schema}.column_descriptions"
    if spark.catalog.tableExists(fqn):
        # Existing table — still seed synthetic placeholders since they may
        # not have been present when the table was first written.
        _seed_synthetic_column_descriptions(spark, catalog, schema)
        return True
    try:
        from customer_retention.analysis.auto_explorer.run_namespace import RunNamespace
        from customer_retention.stages.causal.column_descriptions_writer import (
            ColumnDescriptionsConfig,
            write_column_descriptions,
        )
        from customer_retention.stages.causal.interpretation.sidecars import (
            load_column_descriptions_sidecar,
        )
    except ImportError:
        return False
    root_path: "Optional[Path]" = None
    if experiments_dir is not None:
        from pathlib import Path as _Path
        root_path = _Path(str(experiments_dir))
    try:
        ns = RunNamespace.from_env_or_latest(root=root_path)
    except Exception:  # noqa: BLE001
        ns = None
    sidecar = load_column_descriptions_sidecar(ns) if ns is not None else {}
    rows = list((sidecar or {}).values())
    rows.extend(_synthetic_placeholder_column_descriptions())
    try:
        write_column_descriptions(
            ColumnDescriptionsConfig(spark=spark, table_fqn=fqn, rows=rows)
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "auto-materialization of %s from sidecar failed: %s. "
            "v_feature_provenance will be skipped this run.",
            fqn, exc,
        )
        return spark.catalog.tableExists(fqn)
    return spark.catalog.tableExists(fqn)


def _provenance_prerequisites_present(
    spark: "SparkSession",
    catalog: str,
    schema: str,
    experiments_dir: "Optional[Path | str]" = None,
) -> tuple[bool, list[str]]:
    """Return ``(ok, missing)`` for tables ``v_feature_provenance`` reads.

    Tries to materialize each prereq table from its JSON sidecar in the
    run namespace before failing — the framework's gold-materialization
    step writes the lineage as JSON, but operators that skip exploration
    (causal-track-only reruns) wouldn't otherwise have these tables in UC.
    Older causal-track builds with neither table nor sidecar fall back to
    skipping the provenance view so the rest of the dashboard publishes.
    """
    if not _try_materialize_feature_meta_from_sidecar(
        spark, catalog, schema, experiments_dir=experiments_dir,
    ):
        return False, [f"{catalog}.{schema}.feature_meta"]
    if not _try_materialize_column_descriptions_from_sidecar(
        spark, catalog, schema, experiments_dir=experiments_dir,
    ):
        return False, [f"{catalog}.{schema}.column_descriptions"]
    return True, []


def _deviation_prerequisites_present(
    spark: "SparkSession",
    catalog: str,
    schema: str,
    composite_name: str,
    experiments_dir: "Optional[Path | str]" = None,
) -> tuple[bool, list[str]]:
    """Return ``(ok, missing)`` for the tables the deviation views read.

    The deviation views reference both ``feature_population_stats`` and
    ``gold_features_<composite_name>``. On clusters where population stats
    live as a JSON sidecar (the framework's run-namespace layout) rather
    than a UC Delta table, the publisher first tries to materialize the
    sidecar to UC; if that succeeds the table is considered present. If
    the sidecar is unreachable too, the deviation block is skipped so
    the rest of the dashboard publishes successfully instead of failing
    the whole call with ``[TABLE_OR_VIEW_NOT_FOUND]`` at DDL parse time.
    """
    missing: list[str] = []
    if not _try_materialize_population_stats_from_sidecar(
        spark, catalog, schema, experiments_dir=experiments_dir,
    ):
        missing.append(f"{catalog}.{schema}.feature_population_stats")
    gold_fqn = f"{catalog}.{schema}.gold_features_{composite_name}"
    if not spark.catalog.tableExists(gold_fqn):
        missing.append(gold_fqn)
    return (not missing), missing


@dataclass(frozen=True)
class MaterializedViewSpec:
    """A view whose body gets materialized as a Delta table at publish time.

    After ``publish_dashboard_views`` submits all view DDLs, each spec
    triggers a three-step materialization pass:

      1. ``CREATE OR REPLACE TABLE <table_fqn> USING DELTA AS SELECT *
         FROM <view_fqn>`` — snapshot the view's result set into a table.
      2. ``OPTIMIZE <table_fqn> ZORDER BY (<zorder_col>)`` — index the
         table on the per-click filter column so point lookups stay fast.
      3. ``CREATE OR REPLACE VIEW <view_fqn> AS SELECT * FROM
         <table_fqn>`` — re-point the view at the materialized table.

    Step 3 preserves the public view name + every column the original
    view emitted, so app code, downstream notebooks, and tests that read
    through the view name keep working with identical semantics. The
    only observable change is read latency on a per-entity ``WHERE
    entity_id = :eid`` lookup.

    ``refresh_dependents`` lists views downstream of the materialized
    one (they reference ``view_name`` by name in their SELECT/FROM).
    After step 3 we re-execute each dependent's CREATE OR REPLACE VIEW
    statement so its stored schema metadata realigns with the now
    table-backed source. Without this refresh, Spark fails reads
    through the dependent with ``[DATATYPE_MISMATCH.CAST_WITHOUT_SUGGESTION]``
    because the dependent's metadata captured the pre-materialization
    schema (e.g. ``DOUBLE NOT NULL`` on COALESCE'd columns inside an
    ``ARRAY<STRUCT<...>>``) while the table-backed source emits the
    relaxed type (Delta does not preserve NOT NULL by default).
    """

    view_name: str
    table_name: str
    zorder_col: str
    requires_composite: bool
    refresh_dependents: tuple[str, ...] = ()


# Hot-path views whose bodies are re-executed on every L1-L4 page click.
# Materializing each into a Delta table indexed on ``entity_id`` collapses
# per-click execution from a multi-CTE / explode-and-filter scan into a
# point lookup that the SQL warehouse can serve in milliseconds.
#
# Order matters: ``v_account_primary_recommendation`` is the upstream
# source for ``v_eligible_all_playbooks``, ``v_portfolio_risk_matrix``,
# and ``v_playbook_archetype_rollup`` (none of which are materialized —
# they're aggregations small enough to stay live) so it must be
# materialized first to feed those downstream views.
_MATERIALIZED_VIEW_SPECS: tuple[MaterializedViewSpec, ...] = (
    MaterializedViewSpec(
        view_name="v_account_primary_recommendation",
        table_name="dashboard_account_primary_recommendation",
        zorder_col="entity_id",
        requires_composite=False,
        # v_portfolio_risk_matrix / v_playbook_archetype_rollup /
        # v_eligible_all_playbooks all source from
        # v_account_primary_recommendation. Their stored schemas capture
        # the NOT NULL nullability of the COALESCE-derived columns
        # (``expected_loss``, ``alternates[].expected_loss``) at the
        # original publish. The materialization relaxes those columns
        # to nullable (Delta default) -- without a follow-up refresh
        # any read through these dependents throws
        # CAST_WITHOUT_SUGGESTION.
        refresh_dependents=(
            "v_portfolio_risk_matrix",
            "v_playbook_archetype_rollup",
            "v_eligible_all_playbooks",
        ),
    ),
    MaterializedViewSpec(
        view_name="v_account_explanation",
        table_name="dashboard_account_explanation",
        zorder_col="entity_id",
        requires_composite=False,
    ),
    MaterializedViewSpec(
        view_name="v_account_feature_deviation",
        table_name="dashboard_account_feature_deviation",
        zorder_col="entity_id",
        requires_composite=True,
        # v_account_feature_deviation_topn ranks rows of
        # v_account_feature_deviation. Same metadata-refresh problem as
        # above: the topn view's schema was recorded against the
        # original (CTE-derived) source body; once the source becomes a
        # Delta table the topn's metadata is stale.
        refresh_dependents=("v_account_feature_deviation_topn",),
    ),
    MaterializedViewSpec(
        view_name="v_account_feature_deviation_topn",
        table_name="dashboard_account_feature_deviation_topn",
        zorder_col="entity_id",
        requires_composite=True,
    ),
)


def materialize_view_as_table(
    spark: "SparkSession",
    catalog: str,
    schema: str,
    spec: MaterializedViewSpec,
) -> bool:
    """Run the CTAS + OPTIMIZE + re-point sequence for one spec.

    Returns ``True`` when the view ends up reading from the materialized
    table (i.e. CTAS + re-point succeeded), ``False`` when CTAS or the
    final view re-point failed and the view was left in its pre-call
    state. A failed OPTIMIZE step does NOT flip the return to ``False`` —
    the table still serves point lookups faster than the original view
    body, just without Z-ORDER clustering. Every failure is logged so
    operators can investigate without the publish call itself crashing.
    """
    view_fqn = f"{catalog}.{schema}.{spec.view_name}"
    table_fqn = f"{catalog}.{schema}.{spec.table_name}"
    try:
        spark.sql(
            f"CREATE OR REPLACE TABLE {table_fqn} USING DELTA "
            f"AS SELECT * FROM {view_fqn}"
        )
    except Exception as exc:  # noqa: BLE001 — best-effort, never fail publish
        logger.warning(
            "materialization CTAS of %s -> %s failed: %s. "
            "View left as-is (slower per-click read path).",
            view_fqn, table_fqn, exc,
        )
        return False
    try:
        spark.sql(f"OPTIMIZE {table_fqn} ZORDER BY (`{spec.zorder_col}`)")
    except Exception as exc:  # noqa: BLE001 — non-Databricks Delta may
        # not support OPTIMIZE / ZORDER. The table is still a useful
        # indexable target compared to re-running the view body each
        # click, so we log and continue rather than rolling back.
        logger.warning(
            "OPTIMIZE ZORDER on %s failed: %s. "
            "Table is still queryable; clustering will be unindexed.",
            table_fqn, exc,
        )
    try:
        spark.sql(
            f"CREATE OR REPLACE VIEW {view_fqn} AS SELECT * FROM {table_fqn}"
        )
    except Exception as exc:  # noqa: BLE001 — the table exists but the
        # view still points at its original body. Per-click reads stay
        # on the old slow path; operators see this in the warning log.
        logger.warning(
            "re-pointing %s at %s failed: %s. "
            "View still resolves via its original body.",
            view_fqn, table_fqn, exc,
        )
        return False
    logger.info(
        "materialized %s -> %s (Z-ORDER by %s)",
        view_fqn, table_fqn, spec.zorder_col,
    )
    return True


def _find_view_ddl(statements: Sequence[str], view_fqn: str) -> Optional[str]:
    """Return the ``CREATE OR REPLACE VIEW <view_fqn>`` statement from a list.

    ``statements`` is the rendered SQL split into per-statement strings
    (the output of ``split_view_statements``). Each statement keeps its
    leading ``--`` comment header (split_view_statements preserves
    comments on the following statement so error messages stay readable),
    so a bare ``startswith`` on the whole statement wouldn't match --
    iterate line-by-line and skip ``--`` comments. Returns ``None`` when
    no matching statement is found (e.g. the view wasn't part of the
    published set on this run, or the block was stripped because its
    prerequisites were missing).
    """
    needle = f"CREATE OR REPLACE VIEW {view_fqn}"
    for stmt in statements:
        for line in stmt.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("--"):
                continue
            if stripped.startswith(needle):
                return stmt
            # First executable line of the statement; if it didn't match
            # the needle there's no point scanning the rest of the body.
            break
    return None


def _refresh_dependent_views(
    spark: "SparkSession",
    catalog: str,
    schema: str,
    statements: Sequence[str],
    dependent_view_names: Sequence[str],
) -> None:
    """Re-publish each dependent view to refresh its stored schema metadata.

    Run after a materialization that changes the dependent's upstream
    source from a CTE-derived view body to a Delta-backed view alias.
    Without this refresh, Spark resolves the dependent's stored
    metadata (captured at original publish time) against the now
    table-backed source and errors out with
    ``[DATATYPE_MISMATCH.CAST_WITHOUT_SUGGESTION]`` whenever the
    nullability differs (which happens reliably for COALESCE-derived
    columns: NOT NULL before materialization, NULL-allowed after).

    Each failure is logged but never propagates — the materialization
    pass is best-effort and the publish call should not crash on a
    single dependent's refresh.
    """
    for name in dependent_view_names:
        fqn = f"{catalog}.{schema}.{name}"
        ddl = _find_view_ddl(statements, fqn)
        if ddl is None:
            logger.warning(
                "dependent view %s not found in rendered SQL; skipping refresh. "
                "Reads through it may fail with CAST_WITHOUT_SUGGESTION until "
                "the next publish.",
                fqn,
            )
            continue
        try:
            spark.sql(ddl)
            logger.info("refreshed dependent view %s after materialization", fqn)
        except Exception as exc:  # noqa: BLE001 — best-effort
            logger.warning(
                "refresh of dependent view %s failed: %s. "
                "Reads through it may fail with CAST_WITHOUT_SUGGESTION until "
                "the next publish.",
                fqn, exc,
            )


def _materialize_hot_views(
    spark: "SparkSession",
    catalog: str,
    schema: str,
    *,
    include_deviation: bool,
    statements: Sequence[str],
) -> List[str]:
    """Materialize every applicable spec; return the list of view names rewired.

    ``include_deviation`` gates the deviation-block specs (which require
    a composite_name + the upstream gold table). Specs whose individual
    materialization fails are silently skipped (with a warning log); the
    return value lets callers / tests observe which ones succeeded.

    ``statements`` is the rendered SQL (output of
    ``split_view_statements``) passed through so each spec's
    ``refresh_dependents`` can be re-published from the same source as
    the initial publish -- guaranteeing the refreshed view body matches
    what was originally validated.

    Each spec's OWN view DDL is also re-published before its CTAS pass.
    After a prior materialization, the view body is the trivial
    ``SELECT * FROM <table_name>`` pass-through; re-running the CTAS as-is
    would read from the stale table (via that pass-through view) and
    write the same stale rows back, never picking up upstream changes to
    ``eligibility_snapshot`` (e.g. project-side ``value_at_risk``
    backfills run between publishes). Re-publishing the original CTE body
    immediately before the CTAS forces the source to be the live
    upstream tables. Idempotent on first publish (the view is already at
    its original body).
    """
    rewired: List[str] = []
    for spec in _MATERIALIZED_VIEW_SPECS:
        if spec.requires_composite and not include_deviation:
            continue
        view_fqn = f"{catalog}.{schema}.{spec.view_name}"
        original_ddl = _find_view_ddl(statements, view_fqn)
        if original_ddl is not None:
            try:
                spark.sql(original_ddl)
            except Exception as exc:  # noqa: BLE001 -- best-effort, fall
                # through to the CTAS even when re-publish fails so the
                # operator at least sees the warning surface in logs.
                logger.warning(
                    "could not re-publish %s before its CTAS; the refresh "
                    "may read stale rows from the previously-materialized "
                    "table: %s", view_fqn, exc,
                )
        if materialize_view_as_table(spark, catalog, schema, spec):
            rewired.append(spec.view_name)
            _refresh_dependent_views(
                spark, catalog, schema, statements, spec.refresh_dependents,
            )
    return rewired


def refresh_dashboard_view_materializations(
    spark: "SparkSession",
    catalog: str,
    schema: str,
    *,
    composite_name: Optional[str] = None,
) -> List[str]:
    """Re-run only the CTAS + OPTIMIZE pass for the materialized hot-path views.

    ``publish_dashboard_views`` captures a CTAS snapshot of
    ``v_account_primary_recommendation`` / ``v_account_explanation`` /
    ``v_account_feature_deviation`` / ``_topn`` into Delta tables that
    the L1-L4 views then read from. Any data change to
    ``eligibility_snapshot`` AFTER the publish (custom backfills,
    out-of-band updates) is therefore invisible to the dashboard until
    the next full publish.

    This helper re-runs ONLY the materialization pass -- no view DDLs are
    re-published, no prerequisite checks fire, no provenance / deviation
    block gating logic re-evaluates. Useful when:

    - A deployment patches ``eligibility_snapshot`` after the framework
      c05 publish (e.g. the SPS override that backfills
      ``value_at_risk`` from ``contract_arr`` / opportunity bookings).
    - An operator re-runs a single column's recomputation without
      wanting the full publish cycle.

    The dependent-view refresh step from the publisher is still needed
    because the dependent views' stored schema metadata captures the
    nullability of the COALESCE-derived columns at the *previous*
    materialization. We re-render the SQL once just to extract those
    dependent CREATE OR REPLACE VIEW statements -- the catalog isn't
    re-published; we only execute the dependents whose source we just
    re-CTAS'd.

    When ``composite_name`` is supplied, the function introspects
    ``gold_features_<composite_name>`` to recover the numeric column
    list + full column list that the deviation view body needs
    substituted into ``STRUCT(...)`` and ``ORDER BY``. Without this the
    re-published deviation view body collapses to ``STRUCT(1)`` and the
    materialization captures zero rows -- a silent regression that
    masquerades as missing data (this is what shipped accidentally and
    broke the L4 "Vs. training population" panel for SPS).

    Returns the list of view names whose materialized table was
    refreshed. ``composite_name`` gates the deviation-block specs the
    same way ``publish_dashboard_views`` does -- supply it when the
    refresh should also re-snapshot the per-feature deviation tables.
    """
    gold_numeric_cols: List[str] = []
    gold_all_cols: List[str] = []
    if composite_name:
        gold_fqn = f"{catalog}.{schema}.gold_features_{composite_name}"
        if spark.catalog.tableExists(gold_fqn):
            gold_numeric_cols = _gold_numeric_columns(spark, gold_fqn)
            gold_all_cols = _gold_all_columns(spark, gold_fqn)
        else:
            logger.warning(
                "refresh: gold table %s does not exist; the deviation view body "
                "will not pick up STRUCT() columns and will return zero rows. "
                "Re-run gold materialization first.", gold_fqn,
            )
    rendered = render_dashboard_view_sql(
        catalog,
        schema,
        composite_name=composite_name,
        include_provenance=True,
        gold_struct_cols=gold_numeric_cols,
        gold_columns=gold_all_cols,
    )
    statements = split_view_statements(rendered)
    rewired = _materialize_hot_views(
        spark, catalog, schema,
        include_deviation=composite_name is not None,
        statements=statements,
    )
    if rewired:
        logger.info(
            "refreshed materializations for: %s", ", ".join(rewired),
        )
    return rewired


def publish_dashboard_views(
    spark: "SparkSession",
    catalog: str,
    schema: str,
    *,
    composite_name: Optional[str] = None,
    experiments_dir: "Optional[Path | str]" = None,
) -> List[str]:
    """Execute every dashboard view DDL in order.

    Returns the list of statements that were submitted (one per view).
    Each statement is a ``CREATE OR REPLACE VIEW`` so re-running is a
    no-op when the underlying schema is unchanged.

    Before submitting, the ``run_context`` Delta table is ensured via
    idempotent ``CREATE TABLE IF NOT EXISTS`` — Spark validates the
    ``v_run_context`` view body against its referenced table at creation
    time, so this makes publish order-independent from the per-run
    ``write_run_context`` write.

    When ``composite_name`` is supplied AND the deviation prerequisites
    (``feature_population_stats`` UC table + ``gold_features_<cn>`` UC
    table) are present, the deviation views (``v_account_feature_deviation``
    / ``_topn``) are also published. When the prerequisites are missing
    (e.g. population stats live as a JSON sidecar on Volume rather than a
    UC table), the deviation block is skipped with a logged warning so
    the rest of the dashboard publishes successfully.

    ``experiments_dir`` is the run-namespace root used by the
    sidecar-to-UC auto-materialization helpers (population_stats,
    feature_meta, column_descriptions). On Databricks multi-task jobs
    env vars do not propagate between notebook tasks, so the framework's
    default ``RunNamespace.from_env_or_latest()`` lookup returns None
    and the sidecars are never located -- the deviation block and the
    provenance view both end up skipped despite the sidecars existing.
    Pass the explicit experiments-dir from the caller's resolved
    ``RunNamespace.root`` and the auto-materialization works without
    relying on env vars.
    """
    from .run_context_writer import ensure_run_context_table

    ensure_run_context_table(spark, f"{catalog}.{schema}.run_context")

    effective_composite = composite_name
    gold_numeric_cols: List[str] = []
    gold_all_cols: List[str] = []
    if composite_name:
        ok, missing = _deviation_prerequisites_present(
            spark, catalog, schema, composite_name,
            experiments_dir=experiments_dir,
        )
        if not ok:
            logger.warning(
                "deviation views skipped: missing prerequisite table(s) %s; "
                "publishing the rest of the dashboard. Populate the missing "
                "table(s) (or pass composite_name=None) to silence this notice.",
                ", ".join(missing),
            )
            effective_composite = None
        else:
            gold_fqn = f"{catalog}.{schema}.gold_features_{composite_name}"
            gold_numeric_cols = _gold_numeric_columns(spark, gold_fqn)
            gold_all_cols = _gold_all_columns(spark, gold_fqn)
            if not gold_numeric_cols:
                logger.warning(
                    "deviation views skipped: %s has zero numeric columns "
                    "after metadata filter -- the gold_long CTE would unpivot "
                    "an empty struct. Re-run the gold materialisation step "
                    "and republish.",
                    gold_fqn,
                )
                effective_composite = None
            else:
                logger.info(
                    "deviation views will project %d numeric columns from %s "
                    "into the gold_long CTE",
                    len(gold_numeric_cols), gold_fqn,
                )

    include_provenance, prov_missing = _provenance_prerequisites_present(
        spark, catalog, schema, experiments_dir=experiments_dir,
    )
    if not include_provenance:
        logger.warning(
            "v_feature_provenance skipped: missing prerequisite table(s) %s; "
            "publishing the rest of the dashboard. Re-run gold materialization "
            "to populate feature_meta / column_descriptions and republish.",
            ", ".join(prov_missing),
        )

    rendered = render_dashboard_view_sql(
        catalog,
        schema,
        composite_name=effective_composite,
        include_provenance=include_provenance,
        gold_struct_cols=gold_numeric_cols,
        gold_columns=gold_all_cols,
    )
    statements = split_view_statements(rendered)
    # Drop each view before re-creating it so schema migrations actually
    # take effect. `CREATE OR REPLACE VIEW` is documented as schema-safe,
    # but on Unity Catalog the stored view schema is not always updated
    # when the new body's column nullability differs from the stored one
    # (e.g. an array-of-struct field flipping ``DOUBLE NOT NULL`` ↔
    # ``DOUBLE`` between publishes). The stale stored schema then trips
    # ``DATATYPE_MISMATCH.CAST_WITHOUT_SUGGESTION`` at query time inside
    # any downstream view that selects the column via ``base.*``. A
    # ``DROP VIEW IF EXISTS`` per statement removes the stored schema
    # before the CREATE writes the new one, and the publish loop is
    # ordered so base views are recreated before dependents — Spark
    # views don't cascade-invalidate, the next CREATE just compiles
    # against whichever base view is current.
    submitted: List[str] = []
    for stmt in statements:
        view_fqn = _extract_view_fqn(stmt)
        if view_fqn is not None:
            spark.sql(f"DROP VIEW IF EXISTS {view_fqn}")
        spark.sql(stmt)
        submitted.append(stmt)
        logger.info("published dashboard view (%d chars)", len(stmt))
    if effective_composite is None and composite_name is None:
        logger.info(
            "deviation views skipped (no composite_name supplied; pass composite_name= to enable)"
        )

    # Materialize the four hot-path views whose bodies the app re-executes
    # on every L1->L4 click. After this pass each rewired view's body is
    # ``SELECT * FROM <table>`` (Z-ORDERED on entity_id) so a per-entity
    # WHERE collapses to an indexed point lookup. View names and emitted
    # columns are unchanged -- downstream readers (the Streamlit app,
    # other views that source from these) keep working with identical
    # semantics.
    rewired = _materialize_hot_views(
        spark, catalog, schema,
        include_deviation=effective_composite is not None,
        statements=statements,
    )
    if rewired:
        logger.info(
            "hot-path views materialized as Delta tables: %s",
            ", ".join(rewired),
        )

    return submitted
