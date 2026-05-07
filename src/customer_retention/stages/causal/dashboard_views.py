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
from importlib.resources import files
from typing import TYPE_CHECKING, List, Optional, Sequence

logger = logging.getLogger(__name__)

if TYPE_CHECKING:  # pragma: no cover
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
    spark: "SparkSession", catalog: str, schema: str
) -> bool:
    """Best-effort: materialize the population-stats JSON sidecar into UC.

    The framework writes population stats to a JSON sidecar in the run
    namespace; the deviation views read a UC Delta table. When the table
    is missing but the sidecar exists, materializing it on the fly lets
    ``publish_dashboard_views`` continue without operator intervention.

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
    try:
        ns = RunNamespace.from_env_or_latest()
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
    spark: "SparkSession", catalog: str, schema: str
) -> bool:
    """Best-effort: materialize the ``feature_meta`` JSON sidecar into UC.

    Mirrors ``_try_materialize_population_stats_from_sidecar``. The
    framework writes feature lineage as a JSON sidecar in the run
    namespace at gold-materialization time; the provenance view reads a
    UC Delta table. When the table is missing but the sidecar exists,
    materializing it on the fly lets ``publish_dashboard_views`` continue
    without operator intervention. Returns ``True`` when the table now
    exists, ``False`` when neither table nor sidecar is reachable.
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
    try:
        ns = RunNamespace.from_env_or_latest()
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
    spark: "SparkSession", catalog: str, schema: str
) -> bool:
    """Best-effort: materialize the ``column_descriptions`` JSON sidecar into UC.

    Always seeds the synthetic placeholder rows for ``event`` / ``event_gap``
    too — those are required so the dashboard's Feature dictionary panel
    shows a business definition for event-derived features (which use those
    tokens as their source_column placeholders).
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
    try:
        ns = RunNamespace.from_env_or_latest()
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
    spark: "SparkSession", catalog: str, schema: str
) -> tuple[bool, list[str]]:
    """Return ``(ok, missing)`` for tables ``v_feature_provenance`` reads.

    Tries to materialize each prereq table from its JSON sidecar in the
    run namespace before failing — the framework's gold-materialization
    step writes the lineage as JSON, but operators that skip exploration
    (causal-track-only reruns) wouldn't otherwise have these tables in UC.
    Older causal-track builds with neither table nor sidecar fall back to
    skipping the provenance view so the rest of the dashboard publishes.
    """
    if not _try_materialize_feature_meta_from_sidecar(spark, catalog, schema):
        return False, [f"{catalog}.{schema}.feature_meta"]
    if not _try_materialize_column_descriptions_from_sidecar(spark, catalog, schema):
        return False, [f"{catalog}.{schema}.column_descriptions"]
    return True, []


def _deviation_prerequisites_present(
    spark: "SparkSession", catalog: str, schema: str, composite_name: str
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
    if not _try_materialize_population_stats_from_sidecar(spark, catalog, schema):
        missing.append(f"{catalog}.{schema}.feature_population_stats")
    gold_fqn = f"{catalog}.{schema}.gold_features_{composite_name}"
    if not spark.catalog.tableExists(gold_fqn):
        missing.append(gold_fqn)
    return (not missing), missing


def publish_dashboard_views(
    spark: "SparkSession",
    catalog: str,
    schema: str,
    *,
    composite_name: Optional[str] = None,
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
    """
    from .run_context_writer import ensure_run_context_table

    ensure_run_context_table(spark, f"{catalog}.{schema}.run_context")

    effective_composite = composite_name
    gold_numeric_cols: List[str] = []
    gold_all_cols: List[str] = []
    if composite_name:
        ok, missing = _deviation_prerequisites_present(spark, catalog, schema, composite_name)
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

    include_provenance, prov_missing = _provenance_prerequisites_present(spark, catalog, schema)
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
    submitted: List[str] = []
    for stmt in statements:
        spark.sql(stmt)
        submitted.append(stmt)
        logger.info("published dashboard view (%d chars)", len(stmt))
    if effective_composite is None and composite_name is None:
        logger.info(
            "deviation views skipped (no composite_name supplied; pass composite_name= to enable)"
        )
    return submitted
