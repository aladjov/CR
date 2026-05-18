"""L4 — Customer profile card rendered via Handlebars HTML template.

The profile is one HTML template body. Default ships with the app; CSMs
override per-dataset by re-running the framework's
``apply_profile_override(...)`` cell, which appends a row to
``{catalog}.{schema}.dashboard_template_overrides`` (exposed by
``v_dashboard_template_active``). The app reads the active row through
the SQL warehouse — no Volume FUSE access is involved.

The template declares any additional tables to join against the selected
entity in its YAML frontmatter; values are merged into the template
context so ``{{account.mrr}}`` etc. work without Python changes.

When rendering fails for any reason the panel falls back to a non-styled
dump of every column on v_account_explanation so the CSM still sees the
data.
"""
from __future__ import annotations

import time
from typing import Any, Optional

import pandas as pd
import streamlit as st

from . import data, diagnostics, state
from .template import DataSource, bundle_css, load_template_from_text, render_html


def _clean(v: Any) -> Any:
    """Coerce a Spark/pandas/Arrow value into a Handlebars-safe Python object.

    The dashboard reads via ``cur.fetchall_arrow().to_pandas()``, which lands
    Spark ``ARRAY<STRUCT<..>>`` columns as ``numpy.ndarray`` of dicts. Pybars
    evaluates ``{{#if account_top_shap_features}}`` by calling ``bool(arr)``,
    and a numpy ndarray with >1 elements raises
    ``ValueError: The truth value of an array with more than one element is
    ambiguous`` — which kills the whole render.

    Recursively coerce:
    - ``numpy.ndarray`` -> ``list`` (each item cleaned)
    - ``list`` / ``tuple`` -> ``list`` (each item cleaned)
    - ``dict`` -> ``dict`` (each value cleaned) so nested struct fields work
    - scalar NaN / NaT / ``pd.NA`` -> ``None``
    - anything else -> as-is
    """
    if v is None:
        return None
    # numpy ndarray FIRST -- bool(ndarray) is the failure mode.
    try:
        import numpy as _np  # noqa: PLC0415 - cheap, pandas already imports numpy
    except ImportError:
        _np = None
    if _np is not None and isinstance(v, _np.ndarray):
        return [_clean(x) for x in v.tolist()]
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    if isinstance(v, dict):
        return {k: _clean(val) for k, val in v.items()}
    # Scalar NaN/NaT detection -- pd.isna on arrays returns an array, so we
    # only call it after the array/list/dict branches above. Falls back to
    # the value unchanged on TypeError (non-NA-aware types like custom
    # objects) so we never swallow legitimate values.
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    return v


def _row_to_context(row: pd.Series) -> dict[str, Any]:
    return {k: _clean(v) for k, v in row.items()}


def _rows_to_context(df: pd.DataFrame) -> list[dict[str, Any]]:
    return [{k: _clean(v) for k, v in row.items()} for _, row in df.iterrows()]


def _build_provenance_index(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Index ``v_feature_provenance`` rows by ``feature_name`` for O(1) lookup.

    Empty input returns an empty dict so callers don't need to special-case
    the missing-view path (older causal-track builds without the view).
    Source-column definitions are filled in best-effort: when
    ``column_descriptions`` has no row for a given source column, we still
    surface a one-line description synthesized from the source table name
    so the dictionary block isn't blank.
    """
    if df is None or df.empty or "feature_name" not in df.columns:
        return {}
    out: dict[str, dict[str, Any]] = {}
    for _, row in df.iterrows():
        if row.get("feature_name") is None:
            continue
        ctx = _row_to_context(row)
        ctx["source_column_defs"] = _ensure_source_column_defs(ctx)
        out[str(row["feature_name"])] = ctx
    return out


def _ensure_source_column_defs(ctx: dict[str, Any]) -> list[dict[str, Any]]:
    """Guarantee at least one row per source column in the dictionary block.

    ``v_feature_provenance.source_column_defs`` only emits rows for columns
    that exist in ``column_descriptions``. When that table is empty (the
    common case for environments that haven't run the bootstrap NB cell),
    the dictionary block ends up listing source columns with no
    explanation. Synthesize a fallback description from the source table
    so the block always shows something useful.
    """
    existing = ctx.get("source_column_defs") or []
    have = {str((d or {}).get("column_name") or "") for d in existing if d}
    have.discard("")
    src_table = ctx.get("source_table") or ""
    src_columns = ctx.get("source_columns") or []
    out = list(existing)
    if not isinstance(src_columns, (list, tuple)):
        src_columns = [src_columns]
    for col in src_columns:
        col_str = str(col or "").strip()
        if not col_str or col_str in have:
            continue
        out.append({
            "column_name": col_str,
            "business_name": None,
            "business_definition": _synthesize_source_definition(col_str, src_table),
            "unit": None,
        })
        have.add(col_str)
    return out


def _synthesize_source_definition(column_name: str, source_table: Any) -> str:
    """One-line fallback description when column_descriptions has no row."""
    table = str(source_table or "").strip()
    table_short = table.rsplit(".", 1)[-1] if table else ""
    if table_short.startswith("bronze_event_"):
        return f"A row in {table_short} represents one {column_name!s} event recorded for the customer."
    if table_short.startswith("bronze_entity_"):
        return f"A column from {table_short}, the entity-grain bronze table for this dataset."
    if table_short.startswith("silver_") or table_short.startswith("gold_"):
        return f"A derived column from {table_short}."
    if table_short:
        return f"A column from {table_short}."
    return f"Source column {column_name}."


def _build_deviation_index(rows: Any) -> dict[str, dict[str, Any]]:
    """Index ``feature_deviation`` rows by ``feature_name`` so SHAP rows
    can pick up the customer's raw value + population quantile bounds.

    Empty input returns ``{}`` so callers don't have to special-case the
    missing-data path (older runs that never wrote ``feature_population_stats``).
    """
    if not rows:
        return {}
    index: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            row = _clean(row)
        if not isinstance(row, dict):
            continue
        name = row.get("feature_name")
        if name is None:
            continue
        index[str(name)] = row
    return index


def _enrich_shap_with_meta(
    shap_rows: Any,
    provenance: dict[str, dict[str, Any]],
    deviation: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Attach ``meta`` (provenance) and ``raw_value`` + population
    quantiles (from ``feature_deviation``) to each SHAP row, keyed by
    ``feature``.

    Always returns a list of cleaned dicts. Rows whose feature has no
    matching provenance / deviation entry get ``None`` for the missing
    fields so the template's ``{{#if this.meta}}`` and quantile-band
    helper short-circuit cleanly.
    """
    if not shap_rows:
        return []
    deviation = deviation or {}
    cleaned: list[dict[str, Any]] = []
    for row in shap_rows:
        if not isinstance(row, dict):
            row = _clean(row)
        if not isinstance(row, dict):
            continue
        feature = row.get("feature")
        key = str(feature) if feature is not None else None
        meta = provenance.get(key) if key else None
        dev = deviation.get(key) if key else None
        new_row = dict(row)
        new_row["meta"] = meta
        if dev:
            new_row["raw_value"] = dev.get("feature_value")
            for q in ("q05", "q25", "q50", "q75", "q95"):
                new_row[q] = dev.get(q)
            new_row["z"] = dev.get("z")
        else:
            new_row["raw_value"] = None
            new_row["q05"] = new_row["q25"] = new_row["q50"] = None
            new_row["q75"] = new_row["q95"] = None
            new_row["z"] = None
        cleaned.append(new_row)
    return cleaned


@st.cache_data(ttl=60, show_spinner=False)
def _fetch_data_source_cached(
    source: str, join_key: str, entity_id: str,
    order_by: Optional[str], limit: int,
) -> pd.DataFrame:
    """Cached, connection-shared backing for ``_fetch_data_source``.

    Splitting the cacheable bit out of the orchestrator means a repeat
    visit to the same account hits the in-process cache instead of
    re-querying the warehouse. The shaping to dict / list-of-dicts is
    done by the caller -- a pandas DataFrame is the cleanest hashable
    return type for st.cache_data.
    """
    return data.fetch_template_data_source(
        source, join_key, entity_id, order_by=order_by, limit=limit,
    )


def _fetch_data_source(ds: DataSource, entity_id: str):
    """Fetch rows of a data source joined on entity_id.

    Routes through ``data.fetch_template_data_source`` so the query
    borrows the shared session-scoped SQL connection (see ``data.py``'s
    ``_shared_warehouse_connection``) instead of opening a fresh
    Databricks SQL handshake per render. Returns an empty dict (or list,
    when ``ds.as_list``) when no matching row exists so ``{{#if x}}`` /
    ``{{#each x}}`` treat it as falsy/empty.
    """
    default_limit = 50 if ds.as_list else 1
    limit = int(ds.limit or default_limit)
    df = _fetch_data_source_cached(ds.source, ds.join_key, entity_id, ds.order_by, limit)
    if ds.as_list:
        return _rows_to_context(df)
    if df.empty:
        return {}
    return {k: _clean(v) for k, v in df.iloc[0].items()}


def render(entity_id: str | None = None) -> None:
    """Render the L4 customer profile.

    When ``entity_id`` is provided the panel renders for that entity;
    otherwise it falls back to ``selected_entity`` (the drill-down pick).
    The parameterized form is what lets the Search tab share the same
    template+enrichment pipeline without coupling its state to the
    L1->L4 cascade.
    """
    entity = entity_id if entity_id is not None else state.get("selected_entity")
    if not entity:
        return

    _render_t0 = time.perf_counter()
    diagnostics.record("L4_begin", entity=entity)

    detail_df = data.account_explanation(entity)
    if detail_df.empty:
        diagnostics.record(
            "L4_end",
            entity=entity,
            elapsed_ms=int((time.perf_counter() - _render_t0) * 1000),
            outcome="no_detail",
        )
        st.warning(f"No detail row found for entity **{entity}**.")
        return

    # Per-dataset HTML body: read from the UC ``v_dashboard_template_active``
    # view via the SQL warehouse (the same auth path every other dashboard
    # query uses). On miss, ``load_template_from_text`` reads the bundled
    # default. No volume access — that route was unreliable from Apps even
    # with READ_VOLUME granted.
    html_text = data.load_template_html_from_uc()
    template = load_template_from_text(html_text)

    # Build the full template context: flat account_explanation columns
    # + one nested dict per declared data source.
    context: dict[str, Any] = _row_to_context(detail_df.iloc[0])

    # Enrich SHAP rows with per-feature provenance so the template can render
    # the "Feature dictionary" table beneath the bar chart. The provenance
    # view is loaded once (cached) and looked up by feature_name. The flat
    # ``feature_dictionary`` list is the SAME data deduped by feature so the
    # dictionary table iterates without repeating identical rows when a
    # feature appears more than once in the SHAP set.
    #
    # The same SHAP rows also pick up the customer's raw (unscaled) feature
    # value plus population quantiles from ``feature_deviation`` (already
    # loaded as a template data source, so the lookup is in-memory). This
    # is what powers the "very low / typical / very high" band pill that
    # replaces the cryptic scaled value the panel used to show.
    try:
        provenance_df = data.feature_provenance()
    except Exception:
        provenance_df = pd.DataFrame()
    provenance_index = _build_provenance_index(provenance_df)
    deviation_index = _build_deviation_index(context.get("feature_deviation"))
    enriched_shap = _enrich_shap_with_meta(
        context.get("account_top_shap_features"),
        provenance_index,
        deviation=deviation_index,
    )
    context["account_top_shap_features"] = enriched_shap
    seen: set[str] = set()
    feature_dictionary: list[dict[str, Any]] = []
    for row in enriched_shap:
        feature = row.get("feature")
        if feature is None or feature in seen or not row.get("meta"):
            continue
        seen.add(feature)
        feature_dictionary.append({"feature": feature, **row["meta"]})
    context["feature_dictionary"] = feature_dictionary

    for ds in template.data_sources:
        try:
            context[ds.name] = _fetch_data_source(ds, entity)
        except Exception:
            # Empty placeholder keeps {{#if ds}} / {{#each ds}} falsy/empty
            # so the template still renders the rest of the panel cleanly.
            context[ds.name] = [] if ds.as_list else {}

    # Render the template. On any render error, show a pivoted fallback so the
    # CSM still sees the raw fields.
    #
    # CSS is injected via ``st.markdown(<style>..., unsafe_allow_html=True)``
    # so the rules attach at page scope (matches the ``app.py`` theme.css
    # path). The HTML BODY is then rendered via ``st.html()`` which preserves
    # the full markup as-is (including ``<dt>``/``<dd>``, ``<details>``,
    # ``<nav>``, etc.). Combining them into a single ``st.markdown`` doesn't
    # work because Streamlit's markdown parser strips/escapes most of the
    # element types we use -- ``<dt>``/``<dd>`` get rendered as literal text,
    # which is what produced the "raw HTML dump" failure mode.
    try:
        st.markdown(bundle_css(template), unsafe_allow_html=True)
        st.html(render_html(template, context))
    except Exception as exc:
        st.error(f"Template render failed ({exc}). Showing raw data instead.")
        _render_pivoted_fallback(detail_df.iloc[0])

    diagnostics.record(
        "L4_end",
        entity=entity,
        elapsed_ms=int((time.perf_counter() - _render_t0) * 1000),
        data_sources=len(template.data_sources),
    )


def _render_pivoted_fallback(row: pd.Series) -> None:
    df = pd.DataFrame({"field": row.index, "value": row.values})
    df = df[df["value"].notna()]
    st.dataframe(df, hide_index=True, use_container_width=True)
