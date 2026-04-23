"""L4 — Customer profile card rendered via Handlebars HTML template.

The profile is one HTML template (default ships with the app; CSMs override via
`CR_PROFILE_TEMPLATE_PATH`). The template declares any additional tables to join
against the selected entity in its YAML frontmatter; values are merged into the
template context so `{{account.mrr}}` etc. work without Python changes.

When rendering fails for any reason the panel falls back to a non-styled dump
of every column on v_account_explanation so the CSM still sees the data.
"""
from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st
from databricks import sql
from databricks.sdk.core import Config

from . import data, state
from .config import load_config
from .template import DataSource, bundle_css, load_template, render_html


def _clean(v: Any) -> Any:
    """Convert pandas NaN to None so `{{#if x}}` handles missing values correctly."""
    if isinstance(v, float) and pd.isna(v):
        return None
    return v


def _row_to_context(row: pd.Series) -> dict[str, Any]:
    return {k: _clean(v) for k, v in row.items()}


def _fetch_data_source(ds: DataSource, entity_id: str) -> dict[str, Any]:
    """Fetch the first row of a data source joined on entity_id.

    Returns an empty dict when the row is missing so `{{#if account}}` treats it
    as falsy in the template.
    """
    cfg = load_config()
    sdk_cfg = Config()
    order_clause = f"ORDER BY {ds.order_by}" if ds.order_by else ""
    limit_clause = f"LIMIT {int(ds.limit or 1)}"
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
    template = load_template(cfg.profile_template_path or None)

    # Build the full template context: flat account_explanation columns
    # + one nested dict per declared data source.
    context: dict[str, Any] = _row_to_context(detail_df.iloc[0])
    for ds in template.data_sources:
        try:
            context[ds.name] = _fetch_data_source(ds, entity)
        except Exception as exc:
            st.warning(f"Template data source `{ds.name}` failed — leaving empty. ({exc})")
            context[ds.name] = {}

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
