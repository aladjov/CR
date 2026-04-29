"""L4 — Customer profile card rendered via Handlebars HTML template.

The profile is one HTML template (default ships with the app; CSMs override via
`CR_PROFILE_TEMPLATE_PATH`). The template declares any additional tables to join
against the selected entity in its YAML frontmatter; values are merged into the
template context so `{{account.mrr}}` etc. work without Python changes.

When rendering fails for any reason the panel falls back to a non-styled dump
of every column on v_account_explanation so the CSM still sees the data.
"""
from __future__ import annotations

import os
from typing import Any

import pandas as pd
import streamlit as st
from databricks import sql
from databricks.sdk.core import Config

from . import data, state
from .config import load_config
from .template import DataSource, bundle_css, load_template, render_html


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
    """
    if df is None or df.empty or "feature_name" not in df.columns:
        return {}
    return {
        str(row["feature_name"]): _row_to_context(row)
        for _, row in df.iterrows()
        if row.get("feature_name") is not None
    }


def _enrich_shap_with_meta(
    shap_rows: Any,
    provenance: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Attach a ``meta`` dict to each SHAP row keyed by ``feature``.

    Always returns a list of cleaned dicts. Rows whose feature has no meta
    entry get ``meta = None`` so the template's ``{{#if this.meta}}`` short-
    circuits cleanly.
    """
    if not shap_rows:
        return []
    cleaned: list[dict[str, Any]] = []
    for row in shap_rows:
        if not isinstance(row, dict):
            row = _clean(row)
        if not isinstance(row, dict):
            continue
        feature = row.get("feature")
        meta = provenance.get(str(feature)) if feature is not None else None
        new_row = dict(row)
        new_row["meta"] = meta
        cleaned.append(new_row)
    return cleaned


def _fetch_data_source(ds: DataSource, entity_id: str):
    """Fetch rows of a data source joined on entity_id.

    Returns an empty dict (or list, when ``ds.as_list``) when no matching row
    exists so ``{{#if x}}`` / ``{{#each x}}`` treat it as falsy/empty.
    """
    cfg = load_config()
    sdk_cfg = Config()
    order_clause = f"ORDER BY {ds.order_by}" if ds.order_by else ""
    default_limit = 50 if ds.as_list else 1
    limit_clause = f"LIMIT {int(ds.limit or default_limit)}"
    fqn = f"{cfg.fqn_prefix}.{ds.source}"

    conn = sql.connect(
        server_hostname=sdk_cfg.host.replace("https://", "").rstrip("/"),
        http_path=f"/sql/1.0/warehouses/{cfg.warehouse_id}",
        credentials_provider=lambda: sdk_cfg.authenticate,
    )
    try:
        cur = conn.cursor()
        cur.execute(
            f"SELECT * FROM {fqn} WHERE `{ds.join_key}` = :eid {order_clause} {limit_clause}",
            {"eid": entity_id},
        )
        df = cur.fetchall_arrow().to_pandas()
        if ds.as_list:
            return _rows_to_context(df)
        if df.empty:
            return {}
        return {k: _clean(v) for k, v in df.iloc[0].items()}
    finally:
        conn.close()


def render() -> None:
    entity = state.get("selected_entity")
    if not entity:
        return

    detail_df = data.account_explanation(entity)
    if detail_df.empty:
        st.warning(f"No detail row found for entity **{entity}**.")
        return

    cfg = load_config()
    requested_path = cfg.profile_template_path or None
    template = load_template(requested_path)

    # Build the full template context: flat account_explanation columns
    # + one nested dict per declared data source.
    context: dict[str, Any] = _row_to_context(detail_df.iloc[0])

    # Enrich SHAP rows with per-feature provenance so the template can render
    # the "Feature dictionary" table beneath the bar chart. The provenance
    # view is loaded once (cached) and looked up by feature_name. The flat
    # ``feature_dictionary`` list is the SAME data deduped by feature so the
    # dictionary table iterates without repeating identical rows when a
    # feature appears more than once in the SHAP set.
    try:
        provenance_df = data.feature_provenance()
    except Exception:
        provenance_df = pd.DataFrame()
    provenance_index = _build_provenance_index(provenance_df)
    enriched_shap = _enrich_shap_with_meta(
        context.get("account_top_shap_features"), provenance_index
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

    data_source_diagnostics: list[dict[str, Any]] = []
    for ds in template.data_sources:
        ds_diag = {
            "name": ds.name,
            "source": ds.source,
            "join_key": ds.join_key,
            "as_list": ds.as_list,
            "status": "ok",
            "error": "",
            "row_count": 0,
        }
        try:
            value = _fetch_data_source(ds, entity)
            context[ds.name] = value
            if ds.as_list:
                ds_diag["row_count"] = len(value) if value else 0
                ds_diag["status"] = "ok" if value else "empty"
            else:
                ds_diag["row_count"] = 1 if value else 0
                ds_diag["status"] = "ok" if value else "empty"
        except Exception as exc:
            ds_diag["status"] = "error"
            ds_diag["error"] = f"{type(exc).__name__}: {exc}"
            context[ds.name] = [] if ds.as_list else {}
        data_source_diagnostics.append(ds_diag)

    # Inject the load diagnostic into the template context so the empty-state
    # panel can render it (operators sometimes can't reach the App's Logs tab).
    context["_template_diagnostic"] = {
        **template.diagnostic,
        "catalog": cfg.catalog,
        "schema": cfg.schema,
        "data_sources": data_source_diagnostics,
        # Show the raw env var separately so operators can tell the difference
        # between "env var was set" and "auto-discovery picked it up".
        "env_var_set": bool(os.environ.get("CR_PROFILE_TEMPLATE_PATH", "").strip()),
    }

    # Render the template. On any render error, show a pivoted fallback so the
    # CSM still sees the raw fields.
    try:
        html = bundle_css(template) + render_html(template, context)
        st.html(html)
    except Exception as exc:
        st.error(f"Template render failed ({exc}). Showing raw data instead.")
        _render_pivoted_fallback(detail_df.iloc[0])


def _render_pivoted_fallback(row: pd.Series) -> None:
    df = pd.DataFrame({"field": row.index, "value": row.values})
    df = df[df["value"].notna()]
    st.dataframe(df, hide_index=True, use_container_width=True)
